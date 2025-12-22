#!/usr/bin/env python3
"""
Robust trade collector for Bybit.

Features:
- Auto-reconnect on WebSocket disconnection
- Handles permission denied and other file errors gracefully
- Service-ready with proper signal handling (SIGTERM, SIGINT)
- Exponential backoff on errors
- Buffers data during file write failures

Usage:
    python trade_collector.py --symbol ZECUSDT --out data/zec_live_trades.csv --testnet

As a systemd service, create /etc/systemd/system/sofia-collector.service:

    [Unit]
    Description=Sofia Trade Collector
    After=network.target

    [Service]
    Type=simple
    User=your_user
    WorkingDirectory=/home/your_user/sofia
    ExecStart=/home/your_user/sofia/venv/bin/python trade_collector.py --symbol ZECUSDT --out data/zec_live_trades.csv
    Restart=always
    RestartSec=10

    [Install]
    WantedBy=multi-user.target

Then: sudo systemctl daemon-reload && sudo systemctl enable sofia-collector && sudo systemctl start sofia-collector
"""
import argparse
import csv
import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class TradeCollector:
    INITIAL_RETRY_DELAY = 1.0
    MAX_RETRY_DELAY = 300.0
    FILE_RETRY_DELAY = 5.0
    MAX_FILE_RETRIES = 5

    def __init__(self, symbol: str, out_file: Path, testnet: bool = False,
                 flush_interval: float = 1.0, header_mode: str = "auto"):
        self.symbol = symbol
        self.out_file = out_file
        self.testnet = testnet
        self.flush_interval = flush_interval
        self.header_mode = header_mode
        self.ws = None
        self.buffer = []
        self.lock = threading.Lock()
        self.running = False
        self.total_collected = 0
        self.total_flushed = 0
        self.failed_writes = 0
        self._ensure_directory()
        self._init_csv()

    def _ensure_directory(self):
        for attempt in range(self.MAX_FILE_RETRIES):
            try:
                self.out_file.parent.mkdir(parents=True, exist_ok=True)
                return
            except PermissionError as e:
                logger.warning(f"Permission denied creating directory (attempt {attempt + 1}): {e}")
                if attempt < self.MAX_FILE_RETRIES - 1:
                    time.sleep(self.FILE_RETRY_DELAY)
            except Exception as e:
                logger.error(f"Error creating directory: {e}")
                if attempt < self.MAX_FILE_RETRIES - 1:
                    time.sleep(self.FILE_RETRY_DELAY)

    def _init_csv(self):
        write_header = self._should_write_header()
        if write_header:
            self._write_with_retry([], write_header=True)
        elif (not self.out_file.exists()) or self.out_file.stat().st_size == 0:
            logger.info("Starting collector with headerless CSV output.")

    def _should_write_header(self) -> bool:
        try:
            file_has_rows = self.out_file.exists() and self.out_file.stat().st_size > 0
        except (PermissionError, OSError):
            return False
        if file_has_rows:
            return False
        if self.header_mode == "with-header":
            return True
        if self.header_mode == "no-header":
            return False
        inferred = self._infer_existing_header_mode()
        return True if inferred is None else inferred

    def _infer_existing_header_mode(self) -> Optional[bool]:
        detected = self._detect_header_for_file(self.out_file)
        if detected is not None:
            return detected
        try:
            candidates = sorted(self.out_file.parent.glob("*.csv"))
        except Exception:
            candidates = []
        for candidate in candidates:
            if candidate == self.out_file:
                continue
            detected = self._detect_header_for_file(candidate)
            if detected is not None:
                return detected
        return None

    def _detect_header_for_file(self, path: Path) -> Optional[bool]:
        try:
            if (not path.exists()) or path.stat().st_size == 0:
                return None
            with path.open("rb") as f:
                chunk = f.read(1024)
                first_line = chunk.split(b'\n')[0].decode("utf-8", errors="ignore").strip()
        except Exception:
            return None
        if not first_line:
            return None
        lower = first_line.lower()
        if any(token in lower for token in ("timestamp", "price", "size", "side")):
            return True
        first_token = first_line.split(",")[0].strip()
        try:
            float(first_token)
            return False
        except ValueError:
            return True

    def _write_with_retry(self, rows: list, write_header: bool = False, mode: str = "a") -> bool:
        if write_header:
            mode = "w"
            rows = [["timestamp", "symbol", "side", "size", "price", "tickDirection",
                    "trdMatchID", "grossValue", "homeNotional", "foreignNotional", "RPI"]]
        for attempt in range(self.MAX_FILE_RETRIES):
            try:
                self.out_file.parent.mkdir(parents=True, exist_ok=True)
                with self.out_file.open(mode, newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(rows)
                return True
            except (PermissionError, OSError) as e:
                logger.warning(f"File write error (attempt {attempt + 1}/{self.MAX_FILE_RETRIES}): {e}")
                if attempt < self.MAX_FILE_RETRIES - 1:
                    time.sleep(self.FILE_RETRY_DELAY)
            except Exception as e:
                logger.error(f"Unexpected error writing: {e}")
                if attempt < self.MAX_FILE_RETRIES - 1:
                    time.sleep(self.FILE_RETRY_DELAY)
        self.failed_writes += 1
        return False

    def _on_trade(self, message: dict):
        if "data" not in message:
            return
        rows = []
        for t in message["data"]:
            try:
                rows.append([
                    t.get("T") / 1000.0, t.get("s"), t.get("S"),
                    float(t.get("v")), float(t.get("p")),
                    t.get("L") or "", t.get("i") or "",
                    t.get("grossValue") or "", t.get("homeNotional") or "",
                    t.get("foreignNotional") or "", t.get("RPI") or ""
                ])
            except Exception:
                continue
        if rows:
            with self.lock:
                self.buffer.extend(rows)
                self.total_collected += len(rows)

    def _flusher(self):
        failed_rows = []
        while self.running:
            time.sleep(self.flush_interval)
            with self.lock:
                if not self.buffer and not failed_rows:
                    continue
                rows = failed_rows + self.buffer
                self.buffer = []
                failed_rows = []
            if self._write_with_retry(rows):
                self.total_flushed += len(rows)
                logger.info(f"Flushed {len(rows)} rows (collected={self.total_collected}, flushed={self.total_flushed})")
            else:
                failed_rows = rows
                logger.warning(f"Write failed, buffering {len(rows)} rows for retry")

    def _connect_websocket(self) -> bool:
        from pybit.unified_trading import WebSocket
        retry_delay = self.INITIAL_RETRY_DELAY
        attempt = 0
        while self.running:
            attempt += 1
            try:
                logger.info(f"Connecting to WebSocket (attempt {attempt})...")
                self.ws = WebSocket(testnet=self.testnet, channel_type="linear")
                self.ws.trade_stream(symbol=self.symbol, callback=self._on_trade)
                logger.info(f"Connected! Collecting trades for {self.symbol} -> {self.out_file}")
                return True
            except Exception as e:
                logger.error(f"WebSocket connection failed: {e}")
                if self.running:
                    logger.info(f"Retrying in {retry_delay:.1f} seconds...")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, self.MAX_RETRY_DELAY)
        return False

    def start(self):
        self.running = True
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        flush_thread = threading.Thread(target=self._flusher, daemon=True)
        flush_thread.start()
        logger.info(f"Trade collector starting for {self.symbol}")
        while self.running:
            if not self._connect_websocket():
                break
            while self.running:
                try:
                    time.sleep(5)
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    break
            if self.running:
                logger.warning("Connection lost, reconnecting...")
                time.sleep(self.INITIAL_RETRY_DELAY)
        self._final_flush()
        logger.info(f"Collector stopped. collected={self.total_collected}, flushed={self.total_flushed}")

    def _signal_handler(self, signum, frame):
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}, shutting down...")
        self.stop()

    def _final_flush(self):
        with self.lock:
            rows = self.buffer
            self.buffer = []
        if rows:
            if self._write_with_retry(rows):
                self.total_flushed += len(rows)
                logger.info(f"Final flush: {len(rows)} rows")
            else:
                logger.error(f"CRITICAL: Failed to flush {len(rows)} rows on shutdown!")

    def stop(self):
        if not self.running:
            return
        self.running = False
        logger.info("Stopping collector...")


def main():
    parser = argparse.ArgumentParser(description="Bybit trade collector to CSV")
    parser.add_argument("--symbol", required=True, help="Symbol, e.g., ZECUSDT")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--testnet", action="store_true", help="Use testnet")
    parser.add_argument("--flush-interval", type=float, default=1.0, help="Seconds between disk flushes")
    parser.add_argument("--header-mode", choices=["auto", "with-header", "no-header"], default="auto",
                        help="CSV header mode")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    args = parser.parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    logger.info(f"Starting trade collector for {args.symbol}")
    logger.info(f"Output file: {args.out}")
    logger.info(f"Testnet: {args.testnet}")
    collector = TradeCollector(args.symbol, Path(args.out), testnet=args.testnet,
                               flush_interval=args.flush_interval, header_mode=args.header_mode)
    try:
        collector.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
