#!/usr/bin/env python3
"""
Lightweight trade collector for Bybit.

Subscribes to public trades via WebSocket and appends them to a CSV so
the live paper trader can bootstrap history on restart.

Columns: timestamp (sec since epoch), price, size, side

Usage:
    python trade_collector.py --symbol ZECUSDT --out data/zec_live_trades.csv --testnet
"""
import argparse
import csv
import threading
from pathlib import Path
from time import sleep
from datetime import datetime
from typing import Optional
from pybit.unified_trading import WebSocket


class TradeCollector:
    def __init__(
        self,
        symbol: str,
        out_file: Path,
        testnet: bool = False,
        flush_interval: float = 1.0,
        header_mode: str = "auto",
    ):
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

        # Ensure parent directory exists
        self.out_file.parent.mkdir(parents=True, exist_ok=True)

        # Write header if file doesn't exist or is empty (auto-detect existing format).
        write_header = self._should_write_header()
        if write_header:
            with self.out_file.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "symbol",
                    "side",
                    "size",
                    "price",
                    "tickDirection",
                    "trdMatchID",
                    "grossValue",
                    "homeNotional",
                    "foreignNotional",
                    "RPI",
                ])
        elif (not self.out_file.exists()) or self.out_file.stat().st_size == 0:
            print("Starting collector with headerless CSV output for compatibility.")

    def _should_write_header(self) -> bool:
        file_has_rows = self.out_file.exists() and self.out_file.stat().st_size > 0
        if file_has_rows:
            return False

        if self.header_mode == "with-header":
            return True
        if self.header_mode == "no-header":
            return False

        inferred = self._infer_existing_header_mode()
        if inferred is None:
            return True
        return inferred

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
            # Only read first 1KB to detect header (avoid reading huge files)
            with path.open("rb") as f:
                chunk = f.read(1024)
                first_line = chunk.split(b'\n')[0].decode("utf-8", errors="ignore").lstrip("\ufeff").strip()
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

    def _on_trade(self, message: dict):
        if "data" not in message:
            return
        rows = []
        for t in message["data"]:
            ts = t.get("T")
            price = t.get("p")
            size = t.get("v")
            side = t.get("S")
            tick_dir = t.get("L") or t.get("tickDirection") or ""
            trade_id = t.get("i") or t.get("trdMatchID") or ""
            gross_value = t.get("grossValue") or ""
            home_notional = t.get("homeNotional") or ""
            foreign_notional = t.get("foreignNotional") or ""
            rpi = t.get("RPI") or ""
            try:
                rows.append([
                    ts / 1000.0,
                    t.get("s"),
                    side,
                    float(size),
                    float(price),
                    tick_dir,
                    trade_id,
                    gross_value,
                    home_notional,
                    foreign_notional,
                    rpi,
                ])
            except Exception:
                continue
        if rows:
            with self.lock:
                self.buffer.extend(rows)
                self.total_collected += len(rows)

    def _flusher(self):
        while self.running:
            sleep(self.flush_interval)
            with self.lock:
                if not self.buffer:
                    continue
                rows = self.buffer
                self.buffer = []
            with self.out_file.open("a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            self.total_flushed += len(rows)
            print(f"[{datetime.utcnow().isoformat()}] Flushed {len(rows)} rows (total_collected={self.total_collected}, total_flushed={self.total_flushed})")

    def start(self):
        self.running = True
        # Start flusher thread
        flush_thread = threading.Thread(target=self._flusher, daemon=True)
        flush_thread.start()

        self.ws = WebSocket(testnet=self.testnet, channel_type="linear")
        self.ws.trade_stream(symbol=self.symbol, callback=self._on_trade)
        print(f"[{datetime.utcnow().isoformat()}] Collecting trades for {self.symbol} -> {self.out_file}")

        try:
            while True:
                sleep(1)
        except KeyboardInterrupt:
            self.stop()
            # Force a final flush before exit
            with self.lock:
                rows = self.buffer
                self.buffer = []
            if rows:
                with self.out_file.open("a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(rows)
                self.total_flushed += len(rows)
                print(f"[{datetime.utcnow().isoformat()}] Final flush {len(rows)} rows (total_collected={self.total_collected}, total_flushed={self.total_flushed})")
            print("Collector stopped.")

    def stop(self):
        self.running = False
        print("Stopping collector...")


def main():
    parser = argparse.ArgumentParser(description="Bybit trade collector to CSV")
    parser.add_argument("--symbol", required=True, help="Symbol, e.g., ZECUSDT")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--testnet", action="store_true", help="Use testnet")
    parser.add_argument("--flush-interval", type=float, default=1.0, help="Seconds between disk flushes")
    parser.add_argument(
        "--header-mode",
        choices=["auto", "with-header", "no-header"],
        default="auto",
        help="CSV header mode (auto uses existing data format when possible)",
    )
    args = parser.parse_args()

    collector = TradeCollector(
        args.symbol,
        Path(args.out),
        testnet=args.testnet,
        flush_interval=args.flush_interval,
        header_mode=args.header_mode,
    )
    collector.start()


if __name__ == "__main__":
    main()
