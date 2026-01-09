import time
import logging
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("data_collector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DataCollector")

class BinanceDataCollector:
    def __init__(self, symbol: str, data_dir: str = "data"):
        self.symbol = symbol.upper()
        self.base_dir = Path(data_dir) / self.symbol
        self.trade_dir = self.base_dir / "trades"
        self.ob_dir = self.base_dir / "orderbook"
        
        self.trade_dir.mkdir(parents=True, exist_ok=True)
        self.ob_dir.mkdir(parents=True, exist_ok=True)
        
        self.ws_client = UMFuturesWebsocketClient(on_message=self.on_message)
        
        self.trade_buffer = []
        self.ob_buffer = []
        self.buffer_lock = threading.Lock()
        
        self.last_flush = time.time()
        self.flush_interval = 5.0 # Flush every 5 seconds
        self.running = False

    def start(self):
        logger.info(f"Starting Data Collector for {self.symbol}...")
        
        # Subscribe to Agg Trades and Depth (Level 2)
        # @aggTrade: Aggregate trades
        # @depth20@100ms: Top 20 bids/asks, 100ms update speed
        self.ws_client.agg_trade(symbol=self.symbol.lower())
        self.ws_client.partial_book_depth(
            symbol=self.symbol.lower(), 
            level=20, 
            speed=100
        )
        
        self.running = True
        self.flush_thread = threading.Thread(target=self._flush_loop)
        self.flush_thread.daemon = True
        self.flush_thread.start()
        
        logger.info("Websocket connected and subscribed.")

    def on_message(self, _, msg):
        try:
            data = json.loads(msg)
            event_type = data.get('e')
            
            if event_type == 'aggTrade':
                self._handle_trade(data)
            elif event_type == 'depthUpdate':
                # partial_book_depth returns 'depthUpdate' event?
                # Actually for partial depth, the event type is often just the payload or specific event.
                # Let's check the payload structure for partial depth.
                # Standard partial depth stream payload doesn't always have 'e'.
                # It usually looks like: { "e": "depthUpdate", ... } is for Diff Depth.
                # Partial depth usually: { "lastUpdateId": ..., "bids": [], "asks": [] }
                self._handle_depth(data)
            else:
                # Check for Partial Depth (no event type, just bids/asks)
                if 'b' in data and 'a' in data:
                    self._handle_depth(data)
                    
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def _handle_trade(self, data):
        # Map Binance AggTrade to CSV format
        # data_loader expects: timestamp, price, size, side
        # Binance: T (time), p (price), q (qty), m (isBuyerMaker)
        
        ts = int(data['T']) / 1000.0 # Convert ms to seconds
        price = data['p']
        size = data['q']
        is_buyer_maker = data['m']
        
        # If Buyer is Maker, then Aggressor was Seller -> Side = Sell
        side = "Sell" if is_buyer_maker else "Buy"
        
        row = f"{ts},{price},{size},{side}"
        
        with self.buffer_lock:
            self.trade_buffer.append(row)

    def _handle_depth(self, data):
        # Map Binance Depth to JSONL format
        # data_loader expects: {"type": "snapshot", "ts": ..., "data": {"b": [...], "a": [...]}}
        # Binance Partial: { "e": "depthUpdate", "E": 123, "T": 123, "b": [], "a": [] } OR just { "b": [], "a": [] }
        
        # Use 'E' (Event Time) or 'T' (Transaction Time) or current time
        ts = data.get('E') or data.get('T') or int(time.time() * 1000)
        
        bids = data.get('b', [])
        asks = data.get('a', [])
        
        # Format for data_loader
        # It expects "snapshot" type for full replacements
        payload = {
            "type": "snapshot",
            "ts": ts,
            "data": {
                "b": bids,
                "a": asks
            }
        }
        
        with self.buffer_lock:
            self.ob_buffer.append(json.dumps(payload))

    def _flush_loop(self):
        while self.running:
            time.sleep(self.flush_interval)
            self._flush()

    def _flush(self):
        with self.buffer_lock:
            t_buf = self.trade_buffer[:]
            o_buf = self.ob_buffer[:]
            self.trade_buffer.clear()
            self.ob_buffer.clear()
            
        if not t_buf and not o_buf:
            return

        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y%m%d")
        
        # Write Trades
        if t_buf:
            file_path = self.trade_dir / f"trades_{date_str}.csv"
            # specific format: timestamp,price,size,side
            # Add header if new file? data_loader checks has_side by reading first row?
            # data_loader: preview = pd.read_csv(f, nrows=1) -> check columns
            # We should add header if file doesn't exist.
            
            new_file = not file_path.exists()
            try:
                with open(file_path, "a") as f:
                    if new_file:
                        f.write("timestamp,price,size,side\n")
                    for row in t_buf:
                        f.write(row + "\n")
                logger.info(f"Flushed {len(t_buf)} trades.")
            except Exception as e:
                logger.error(f"Failed to write trades: {e}")

        # Write Orderbook
        if o_buf:
            file_path = self.ob_dir / f"orderbook_{date_str}.jsonl"
            try:
                with open(file_path, "a") as f:
                    for row in o_buf:
                        f.write(row + "\n")
                logger.info(f"Flushed {len(o_buf)} OB snapshots.")
            except Exception as e:
                logger.error(f"Failed to write OB: {e}")

    def stop(self):
        self.running = False
        self.ws_client.stop()
        self._flush()
        logger.info("Collector stopped.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="FARTCOINUSDT", help="Symbol to collect")
    args = parser.parse_args()
    
    collector = BinanceDataCollector(args.symbol)
    try:
        collector.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        collector.stop()
