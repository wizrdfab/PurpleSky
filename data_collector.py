"""
Copyright (C) 2026 Fabián Zúñiga Franck

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import logging
import time
import json
import os
import threading
import queue
import signal
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient
from binance.um_futures import UMFutures
from binance.error import ClientError

# --- Configuration ---
SNAPSHOT_INTERVAL = 3600  # Refresh snapshot every 1 hour (3600s) to self-heal
WATCHDOG_TIMEOUT = 60     # Reconnect if no data for 60 seconds
FLUSH_INTERVAL = 5        # Flush file buffers every 5 seconds
BUFFER_SIZE = 100         # Or flush after 100 lines

# --- Logging Setup ---
# Log to file AND console
log_file = "data_collector.log"

# Root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# File Handler
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

# Console Handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

# Specific Logger for this script
logger = logging.getLogger("DataCollector")

# Silence noisy libraries if needed (optional)
# logging.getLogger("urllib3").setLevel(logging.WARNING)

class FileManager:
    """
    Manages open file handles to avoid opening/closing on every message.
    Handles daily rotation and buffering.
    """
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.files = {} # Key: (symbol, data_type, date_str) -> file_handle
        self.buffers = {} # Key: file_handle -> list of strings
        self.lock = threading.Lock()
        self.last_flush = time.time()

    def write(self, symbol, dtype, data, timestamp_s):
        """
        timestamp_s: float (unix timestamp)
        """
        # Determine Date from Timestamp
        dt = datetime.fromtimestamp(timestamp_s, tz=timezone.utc)
        date_str = dt.strftime("%Y-%m-%d")
        
        key = (symbol, dtype)
        
        with self.lock:
            # Check if we have an open file for this (symbol, dtype)
            # We need to check if the DATE matches the current open file's date.
            # To simplify, we track the current open file's date in a separate dict or just key by date too.
            # Keying by date implies we might have multiple days open at crossing, which is fine.
            
            full_key = (symbol, dtype, date_str)
            
            if full_key not in self.files:
                # Close *previous* day's file for this symbol/type if exists to save resources?
                # For simplicity, we can just iterate and close old dates, or rely on OS to handle a few open files.
                # Better: Close any file for (symbol, dtype) that isn't `date_str`.
                self._close_old_files(symbol, dtype, date_str)
                self._open_file(symbol, dtype, date_str)
            
            f = self.files[full_key]
            
            # Format Data
            line = ""
            if dtype == 'trade':
                # CSV Format
                # timestamp,symbol,side,size,price,...
                line = f"{data['timestamp']},{data['symbol']},{data['side']},{data['size']},{data['price']},{data['tickDirection']},{data['trdMatchID']},{data['grossValue']},{data['homeNotional']},{data['foreignNotional']},{data['RPI']}\n"
            elif dtype == 'orderbook':
                # JSONL Format
                line = json.dumps(data) + "\n"
                
            f.write(line)
            
            # Auto-Flush logic handled in periodic loop or by buffer size?
            # Python's file object is already buffered. We just ensure we flush periodically.
            
    def _open_file(self, symbol, dtype, date_str):
        base_dir = self.data_dir / f"{symbol}_Binance"
        
        if dtype == 'trade':
            path = base_dir / "Trade"
            fname = path / f"{symbol}{date_str}.csv"
            exists = fname.exists()
            f = open(fname, 'a', buffering=8192, encoding='utf-8') # 8kb buffer
            if not exists:
                f.write("timestamp,symbol,side,size,price,tickDirection,trdMatchID,grossValue,homeNotional,foreignNotional,RPI\n")
        else:
            path = base_dir / "Orderbook"
            fname = path / f"{date_str}_{symbol}_ob200.data"
            f = open(fname, 'a', buffering=8192, encoding='utf-8')

        self.files[(symbol, dtype, date_str)] = f
        logger.info(f"Opened file: {fname}")

    def _close_old_files(self, symbol, dtype, current_date_str):
        # Find keys that match symbol/dtype but NOT current_date_str
        to_close = []
        for k in self.files:
            s, t, d = k
            if s == symbol and t == dtype and d != current_date_str:
                to_close.append(k)
        
        for k in to_close:
            logger.info(f"Rotated/Closed file for {k}")
            self.files[k].close()
            del self.files[k]

    def flush_all(self):
        with self.lock:
            for f in self.files.values():
                try:
                    f.flush()
                except Exception as e:
                    logger.error(f"Flush error: {e}")

    def close_all(self):
        with self.lock:
            for f in self.files.values():
                f.close()
            self.files.clear()


class RobustDataCollector:
    def __init__(self, symbols, data_dir="data"):
        self.symbols = [s.upper() for s in symbols]
        self.data_dir = Path(data_dir)
        self.rest_client = UMFutures()
        self.ws_client = None
        self.file_manager = FileManager(self.data_dir)
        self.write_queue = queue.Queue(maxsize=50000) # Max 50k items buffer
        self.snapshot_queue = queue.Queue(maxsize=100) # Queue for REST requests
        
        self.running = False
        self.writer_thread = None
        self.snapshot_thread = None
        self.monitor_thread = None
        
        self.last_msg_time = time.time()
        self.last_snapshot_time = {} # symbol -> timestamp
        
        # Ensure Dirs
        for s in self.symbols:
            (self.data_dir / f"{s}_Binance" / "Trade").mkdir(parents=True, exist_ok=True)
            (self.data_dir / f"{s}_Binance" / "Orderbook").mkdir(parents=True, exist_ok=True)

    def start(self):
        self.running = True
        logger.info(f"Starting Robust Collector for {len(self.symbols)} symbols...")
        
        # Writer Thread
        self.writer_thread = threading.Thread(target=self.writer_loop, daemon=True)
        self.writer_thread.start()
        
        # Snapshot Worker Thread
        self.snapshot_thread = threading.Thread(target=self.snapshot_worker_loop, daemon=True)
        self.snapshot_thread.start()
        
        # Connect
        self._connect_and_subscribe()
        
        # Monitor Loop (Main Thread)
        try:
            while self.running:
                now = time.time()
                
                # 1. Watchdog
                if now - self.last_msg_time > WATCHDOG_TIMEOUT:
                    logger.warning(f"WATCHDOG: No data for {WATCHDOG_TIMEOUT}s. Reconnecting...")
                    self._reconnect()
                    self.last_msg_time = now # Reset to avoid loop
                
                # 2. Periodic Snapshots (Self-Healing)
                for symbol in self.symbols:
                    last_snap = self.last_snapshot_time.get(symbol, 0)
                    if now - last_snap > SNAPSHOT_INTERVAL:
                        self._fetch_snapshot(symbol)
                        self.last_snapshot_time[symbol] = now
                
                # 3. Flush Files
                self.file_manager.flush_all()
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.stop()
        except Exception as e:
            logger.critical(f"CRITICAL ERROR in Main Loop: {e}", exc_info=True)
            self.stop()

    def stop(self):
        logger.info("Stopping...")
        self.running = False
        if self.ws_client:
            self.ws_client.stop()
        if self.writer_thread:
            self.writer_thread.join(timeout=5)
        if self.snapshot_thread:
            self.snapshot_thread.join(timeout=5)
        self.file_manager.close_all()
        logger.info("Stopped.")

    def _connect_and_subscribe(self):
        try:
            if self.ws_client:
                self.ws_client.stop()
                
            self.ws_client = UMFuturesWebsocketClient(on_message=self.on_ws_message, on_error=self.on_ws_error)
            
            for symbol in self.symbols:
                # Subscribe
                self.ws_client.agg_trade(symbol=symbol)
                self.ws_client.diff_book_depth(symbol=symbol, speed=100)
                time.sleep(0.2) # Rate limit protection
                
                # Initial Snapshot
                if symbol not in self.last_snapshot_time: # Only if never fetched
                    self._fetch_snapshot(symbol)
                    self.last_snapshot_time[symbol] = time.time()
            
            self.last_msg_time = time.time()
            logger.info("Connected and Subscribed.")
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            time.sleep(5) # Backoff

    def _reconnect(self):
        logger.info("Reconnecting...")
        try:
            self.ws_client.stop()
        except:
            pass
        time.sleep(2)
        self._connect_and_subscribe()

    def _fetch_snapshot(self, symbol):
        try:
            self.snapshot_queue.put(symbol, block=False)
        except queue.Full:
            logger.warning(f"Snapshot queue full! Skipping snapshot for {symbol}")

    def snapshot_worker_loop(self):
        while self.running:
            try:
                symbol = self.snapshot_queue.get(timeout=1)
                try:
                    # Limit 500 for better parity with "ob200"
                    depth = self.rest_client.depth(symbol=symbol, limit=500)
                    snapshot_obj = self._transform_snapshot(symbol, depth)
                    self.write_queue.put(('orderbook', symbol, snapshot_obj))
                    logger.info(f"Snapshot refreshed for {symbol}")
                    time.sleep(0.1) # Small delay to be gentle on API limits
                except Exception as e:
                    logger.error(f"Snapshot failed for {symbol}: {e}")
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Snapshot Worker Error: {e}")

    def on_ws_message(self, _, message):
        self.last_msg_time = time.time()
        try:
            if isinstance(message, str):
                msg = json.loads(message)
            else:
                msg = message

            event_type = msg.get('e')
            
            if event_type == 'aggTrade':
                self._handle_trade(msg)
            elif event_type == 'depthUpdate':
                self._handle_depth(msg)
                
        except Exception as e:
            logger.error(f"Parse error: {e}")

    def on_ws_error(self, _, error):
        logger.error(f"WebSocket Error: {error}")

    def _handle_trade(self, msg):
        symbol = msg['s']
        price = float(msg['p'])
        size = float(msg['q'])
        timestamp_ms = msg['T']
        is_maker = msg['m']
        side = "Sell" if is_maker else "Buy"
        
        row = {
            'timestamp': timestamp_ms / 1000.0,
            'symbol': symbol,
            'side': side,
            'size': size,
            'price': price,
            'tickDirection': 'ZeroPlusTick', 
            'trdMatchID': msg['a'],
            'grossValue': price * size,
            'homeNotional': size,
            'foreignNotional': price * size,
            'RPI': 0
        }
        try:
            self.write_queue.put(('trade', symbol, row), block=False)
        except queue.Full:
            logger.warning("Queue FULL! Dropping trade data.")

    def _handle_depth(self, msg):
        symbol = msg['s']
        out = {
            "topic": f"orderbook.200.{symbol}",
            "type": "delta",
            "ts": msg['T'],
            "data": {"s": symbol, "b": msg.get('b', []), "a": msg.get('a', [])},
            "u": msg.get('u'),
            "seq": msg.get('U'),
            "cts": msg.get('E')
        }
        try:
            self.write_queue.put(('orderbook', symbol, out), block=False)
        except queue.Full:
            logger.warning("Queue FULL! Dropping depth data.")

    def _transform_snapshot(self, symbol, depth_data):
        ts = depth_data.get('T', int(time.time() * 1000))
        return {
            "topic": f"orderbook.200.{symbol}",
            "type": "snapshot",
            "ts": ts,
            "data": {"s": symbol, "b": depth_data.get('bids', []), "a": depth_data.get('asks', [])},
            "u": depth_data.get('lastUpdateId'),
            "seq": depth_data.get('lastUpdateId'),
            "cts": depth_data.get('E', ts)
        }

    def writer_loop(self):
        while self.running or not self.write_queue.empty():
            try:
                item = self.write_queue.get(timeout=1)
                dtype, symbol, data = item
                
                # Use Timestamp from data to determine file date
                if dtype == 'trade':
                    ts = data['timestamp']
                else:
                    ts = data['ts'] / 1000.0
                
                self.file_manager.write(symbol, dtype, data, ts)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Writer Error: {e}")

if __name__ == "__main__":
    # Auto-detect symbols from existing folder structure if not provided
    DEFAULT_SYMBOLS = []
    try:
        data_path = Path("data")
        if data_path.exists():
            for d in data_path.iterdir():
                if d.is_dir() and not d.name.endswith("_Binance") and "USDT" in d.name:
                    DEFAULT_SYMBOLS.append(d.name)
    except Exception:
        pass
    
    if not DEFAULT_SYMBOLS:
        DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]

    symbols = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_SYMBOLS
    
    print(f"Starting ROBUST Data Collector for: {symbols}")
    print(f"Logs: {log_file}")
    
    collector = RobustDataCollector(symbols)
    collector.start()