
"""
Simulated exchange backed by trade CSVs and orderbook .data files.
"""

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd

from config import CONF


def resolve_symbol_dir(data_dir: Path, symbol: str) -> Path:
    if (data_dir / "Trade").exists():
        return data_dir
    candidate = data_dir / symbol
    if (candidate / "Trade").exists():
        return candidate
    raise FileNotFoundError(f"Trade folder not found under {data_dir}")


def list_sorted_files(path: Path, pattern: str) -> List[Path]:
    return sorted(path.glob(pattern))


def extract_date_from_name(path: Path) -> Optional[date]:
    match = re.search(r"\d{4}-\d{2}-\d{2}", path.name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(0), "%Y-%m-%d").date()
    except Exception:
        return None


def filter_files_from_ts(files: List[Path], start_ts: float) -> List[Path]:
    if not files:
        return files
    start_date = datetime.utcfromtimestamp(start_ts).date()
    filtered: List[Path] = []
    for path in files:
        file_date = extract_date_from_name(path)
        if file_date is None or file_date >= start_date:
            filtered.append(path)
    return filtered if filtered else files


def decimal_step_from_str(value_str: str) -> float:
    try:
        dec = Decimal(value_str)
    except (InvalidOperation, TypeError):
        return 0.0
    exponent = -dec.as_tuple().exponent
    if exponent <= 0:
        return 1.0
    return float(Decimal(1).scaleb(-exponent))


def bar_time_from_ts(ts_sec: float, tf_seconds: int) -> int:
    return int(ts_sec // tf_seconds) * tf_seconds


@dataclass
class TradeRow:
    timestamp: float
    price: float
    size: float
    side: str
    trade_id: str
    price_str: str
    size_str: str


class TradeStream:
    def __init__(self, trade_files: List[Path]):
        self.trade_files = trade_files
        self.file_idx = 0
        self.file_handle = None
        self.reader = None
        self.header_idx: Dict[str, int] = {}
        self.buffered: Optional[TradeRow] = None
        self.done = False
        self.row_counter = 0

    def _open_next_file(self) -> None:
        if self.file_handle:
            self.file_handle.close()
        if self.file_idx >= len(self.trade_files):
            self.done = True
            self.file_handle = None
            self.reader = None
            return
        path = self.trade_files[self.file_idx]
        self.file_handle = open(path, "r", newline="")
        self.reader = csv.reader(self.file_handle)
        header = next(self.reader, None)
        if not header:
            self.file_idx += 1
            self._open_next_file()
            return
        self.header_idx = {name: idx for idx, name in enumerate(header)}
        self.file_idx += 1

    def _read_next_row(self) -> Optional[TradeRow]:
        while True:
            if self.reader is None:
                self._open_next_file()
                if self.reader is None:
                    return None
            row = next(self.reader, None)
            if row is None:
                self._open_next_file()
                continue
            try:
                ts_str = row[self.header_idx["timestamp"]]
                price_str = row[self.header_idx["price"]]
                size_str = row[self.header_idx["size"]]
                side = row[self.header_idx["side"]]
            except Exception:
                continue
            trade_id = ""
            if "trdMatchID" in self.header_idx:
                trade_id = row[self.header_idx["trdMatchID"]]
            elif "id" in self.header_idx:
                trade_id = row[self.header_idx["id"]]
            else:
                self.row_counter += 1
                trade_id = f"row-{self.row_counter}"
            return TradeRow(
                timestamp=float(ts_str),
                price=float(price_str),
                size=float(size_str),
                side=side,
                trade_id=trade_id,
                price_str=price_str,
                size_str=size_str,
            )

    def peek(self) -> Optional[TradeRow]:
        if self.buffered is None and not self.done:
            self.buffered = self._read_next_row()
        return self.buffered

    def next(self) -> Optional[TradeRow]:
        if self.buffered is not None:
            row = self.buffered
            self.buffered = None
            return row
        return self._read_next_row()

    def skip_to_time(self, ts_sec: float) -> None:
        if self.done:
            return
        while True:
            row = self.peek()
            if row is None:
                return
            if row.timestamp >= ts_sec:
                return
            self.buffered = None
            _ = self._read_next_row()


class OrderbookStream:
    def __init__(self, ob_files: List[Path]):
        self.ob_files = ob_files
        self.file_idx = 0
        self.file_handle = None
        self.buffered: Optional[Dict] = None
        self.done = False
        self.book_bids: Dict[float, float] = {}
        self.book_asks: Dict[float, float] = {}
        self.ready = False

    def _open_next_file(self) -> None:
        if self.file_handle:
            self.file_handle.close()
        if self.file_idx >= len(self.ob_files):
            self.done = True
            self.file_handle = None
            return
        path = self.ob_files[self.file_idx]
        self.file_handle = open(path, "r")
        self.file_idx += 1

    def _parse_levels(self, levels: List) -> List[Tuple[float, float]]:
        parsed: List[Tuple[float, float]] = []
        for level in levels or []:
            if not level or len(level) < 2:
                continue
            try:
                price = float(level[0])
                size = float(level[1])
            except Exception:
                continue
            parsed.append((price, size))
        return parsed

    def _apply_snapshot(self, bids: List, asks: List) -> None:
        bid_levels = self._parse_levels(bids)
        ask_levels = self._parse_levels(asks)
        self.book_bids = {price: size for price, size in bid_levels if size > 0}
        self.book_asks = {price: size for price, size in ask_levels if size > 0}
        self.ready = True

    def _apply_delta(self, bids: List, asks: List) -> None:
        for price, size in self._parse_levels(bids):
            if size <= 0:
                self.book_bids.pop(price, None)
            else:
                self.book_bids[price] = size
        for price, size in self._parse_levels(asks):
            if size <= 0:
                self.book_asks.pop(price, None)
            else:
                self.book_asks[price] = size

    def _build_book(self, ts: int) -> Optional[Dict]:
        if not self.book_bids or not self.book_asks:
            return None
        bids = sorted(self.book_bids.items(), key=lambda x: x[0], reverse=True)
        asks = sorted(self.book_asks.items(), key=lambda x: x[0])
        return {"ts": int(ts), "b": [[p, s] for p, s in bids], "a": [[p, s] for p, s in asks]}

    def _read_next_snapshot(self) -> Optional[Dict]:
        while True:
            if self.file_handle is None:
                self._open_next_file()
                if self.file_handle is None:
                    return None
            line = self.file_handle.readline()
            if not line:
                self._open_next_file()
                continue
            try:
                raw = json.loads(line)
            except Exception:
                continue
            data = raw.get("data", {})
            if not data:
                continue
            ts = raw.get("ts")
            if ts is None:
                continue
            msg_type = raw.get("type")
            bids = data.get("b") or []
            asks = data.get("a") or []
            if msg_type == "snapshot":
                if bids and asks:
                    self._apply_snapshot(bids, asks)
                else:
                    continue
            elif msg_type == "delta":
                if not self.ready:
                    continue
                if bids or asks:
                    self._apply_delta(bids, asks)
                else:
                    continue
            else:
                continue
            book = self._build_book(ts)
            if book:
                return book

    def peek(self) -> Optional[Dict]:
        if self.buffered is None and not self.done:
            self.buffered = self._read_next_snapshot()
        return self.buffered

    def next(self) -> Optional[Dict]:
        if self.buffered is not None:
            snap = self.buffered
            self.buffered = None
            return snap
        return self._read_next_snapshot()

    def skip_to_time(self, ts_ms: int) -> None:
        if self.done:
            return
        while True:
            snap = self.peek()
            if snap is None:
                return
            if snap["ts"] >= ts_ms:
                return
            self.buffered = None
            _ = self._read_next_snapshot()

    def advance_to(self, ts_ms: int) -> Optional[Dict]:
        last = None
        while True:
            snap = self.peek()
            if snap is None:
                break
            if snap["ts"] > ts_ms:
                break
            last = self.next()
        return last


@dataclass
class SimOrder:
    order_id: str
    side: str
    price: float
    qty: float
    created_ts: float


class SimulatedExchange:
    def __init__(
        self,
        symbol: str,
        data_dir: Path,
        clock,
        start_time: float,
        trade_enable_time: float,
        initial_equity: float,
        tf_seconds: int,
        start_bar_index: int,
        test_start_index: int,
        total_bars: int,
        verbose: bool = True,
        status_every_bars: int = 0,
    ):
        self.symbol = symbol
        self.clock = clock
        self.trade_enable_time = trade_enable_time
        self.trading_enabled = False
        self.done = False
        self.tf_seconds = tf_seconds
        self.start_bar_index = start_bar_index
        self.test_start_index = test_start_index
        self.total_bars = total_bars
        self.verbose = verbose
        self.status_every_bars = status_every_bars
        self.bar_count = 0
        self.last_bar_time: Optional[int] = None
        self.test_start_announced = False
        self.trade_stream_finished_logged = False
        self.orderbook_finished_logged = False

        symbol_dir = resolve_symbol_dir(data_dir, symbol)
        trade_dir = symbol_dir / "Trade"
        ob_dir = symbol_dir / "Orderbook"

        trade_files = list_sorted_files(trade_dir, "*.csv")
        ob_files = list_sorted_files(ob_dir, "*.data")
        if not trade_files:
            raise FileNotFoundError(f"No trade CSV files under {trade_dir}")
        if not ob_files:
            raise FileNotFoundError(f"No orderbook files under {ob_dir}")

        orig_trade_count = len(trade_files)
        orig_ob_count = len(ob_files)
        trade_files = filter_files_from_ts(trade_files, start_time)
        ob_files = filter_files_from_ts(ob_files, start_time)

        self.trade_stream = TradeStream(trade_files)
        self.orderbook_stream = OrderbookStream(ob_files)

        if self.verbose and (len(trade_files) != orig_trade_count or len(ob_files) != orig_ob_count):
            start_label = self._format_time(start_time)
            self._log(
                "Filtered files from "
                f"{start_label}: trades {len(trade_files)}/{orig_trade_count}, "
                f"orderbook {len(ob_files)}/{orig_ob_count}"
            )

        if self.verbose:
            self._log(f"Skipping trades to {self._format_time(start_time)}")
        self.trade_stream.skip_to_time(start_time)
        if self.verbose:
            self._log(f"Skipping orderbook to {self._format_time(start_time)} (may take a while)")
        self.orderbook_stream.skip_to_time(int(start_time * 1000))

        self.clock.set(start_time)

        self.maker_fee = CONF.strategy.maker_fee
        self.taker_fee = CONF.strategy.taker_fee

        self.min_qty, self.qty_step, self.tick_size = self._infer_instrument_info()
        self.cash_balance = float(initial_equity)
        self.cum_realized = 0.0
        self.last_trade_price: Optional[float] = None
        self.orders: Dict[str, SimOrder] = {}
        self.position: Dict[str, float] = {}
        self.order_seq = 0
        self.last_ob_returned_ts: Optional[int] = None
        self.session = self._Session(self)

        self._log(
            "Initialized. "
            f"min_qty={self.min_qty} qty_step={self.qty_step} tick_size={self.tick_size} "
            f"status_every_bars={self.status_every_bars}"
        )

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[SIM] {message}")

    def _format_time(self, ts_sec: float) -> str:
        try:
            return datetime.utcfromtimestamp(ts_sec).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return str(ts_sec)

    def _print_status(self, global_idx: int, bar_time: int) -> None:
        equity = self.cash_balance
        if self.position:
            equity += self.position.get("unrealized_pnl", 0.0)
        pos_side = self.position.get("side", "Flat") if self.position else "Flat"
        pos_size = self.position.get("size", 0.0) if self.position else 0.0
        overall_pct = 0.0
        if self.total_bars > 0:
            overall_pct = (100.0 * (global_idx + 1)) / self.total_bars
        test_total = max(self.total_bars - self.test_start_index, 1)
        test_pct = 0.0
        if global_idx >= self.test_start_index:
            test_idx = global_idx - self.test_start_index
            test_pct = (100.0 * (test_idx + 1)) / test_total
        self._log(
            "Status "
            f"bar={global_idx} overall={overall_pct:.1f}% test={test_pct:.1f}% "
            f"time={self._format_time(bar_time)} pos={pos_side} {pos_size:.4f} "
            f"equity={equity:.2f} orders={len(self.orders)}"
        )

    def _on_new_bar(self, bar_time: int) -> None:
        if bar_time == self.last_bar_time:
            return
        self.last_bar_time = bar_time
        self.bar_count += 1
        global_idx = self.start_bar_index + self.bar_count - 1
        if not self.test_start_announced and global_idx >= self.test_start_index:
            self.test_start_announced = True
            self._log(f"Test window start at bar {global_idx} time {self._format_time(bar_time)}")
        if self.status_every_bars and self.bar_count % self.status_every_bars == 0:
            self._print_status(global_idx, bar_time)

    def _infer_instrument_info(self) -> Tuple[float, float, float]:
        trade = self.trade_stream.peek()
        snap = self.orderbook_stream.peek()
        price_str = None
        size_str = None
        if trade:
            price_str = trade.price_str
            size_str = trade.size_str
        if not price_str and snap:
            price_str = snap["b"][0][0]
        if not size_str and snap:
            size_str = snap["b"][0][1]
        tick_size = decimal_step_from_str(price_str or "0")
        qty_step = decimal_step_from_str(size_str or "0")
        min_qty = qty_step
        return min_qty, qty_step, tick_size

    def _update_trading_enabled(self) -> None:
        if not self.trading_enabled and self.clock.time() >= self.trade_enable_time:
            self.trading_enabled = True
            self._log(f"Trading enabled at {self._format_time(self.clock.time())}")

    def _update_unrealized(self, price: float) -> None:
        if not self.position:
            return
        entry = self.position.get("entry_price", 0.0)
        size = self.position.get("size", 0.0)
        side = self.position.get("side", "Buy")
        direction = 1.0 if side == "Buy" else -1.0
        self.position["mark_price"] = price
        self.position["unrealized_pnl"] = (price - entry) * size * direction

    def _close_position(self, price: float, fee_rate: float, reason: str) -> None:
        if not self.position:
            return
        entry = self.position.get("entry_price", 0.0)
        size = self.position.get("size", 0.0)
        side = self.position.get("side", "Buy")
        direction = 1.0 if side == "Buy" else -1.0
        pnl = (price - entry) * size * direction
        fee = price * size * fee_rate
        pnl_net = pnl - fee
        self.cash_balance += pnl_net
        self.cum_realized += pnl_net
        self.position = {}
        self._log(
            f"Position closed ({reason}) price={price:.6f} pnl_net={pnl_net:.2f} equity={self.cash_balance:.2f}"
        )

    def _fill_order(self, order: SimOrder) -> None:
        fee = order.price * order.qty * self.maker_fee
        self.cash_balance -= fee
        self.position = {
            "size": order.qty,
            "side": order.side,
            "entry_price": order.price,
            "mark_price": order.price,
            "unrealized_pnl": 0.0,
            "tp": None,
            "sl": None,
        }
        self._log(
            f"Filled order {order.order_id} {order.side} price={order.price:.6f} qty={order.qty:.6f} "
            f"fee={fee:.4f} equity={self.cash_balance:.2f}"
        )

    def _check_tp_sl(self, price: float) -> None:
        if not self.position:
            return
        tp = self.position.get("tp")
        sl = self.position.get("sl")
        side = self.position.get("side", "Buy")
        if side == "Buy":
            if tp is not None and price >= tp:
                self._close_position(tp, self.maker_fee, "tp")
                return
            if sl is not None and price <= sl:
                self._close_position(sl, self.taker_fee, "sl")
                return
        else:
            if tp is not None and price <= tp:
                self._close_position(tp, self.maker_fee, "tp")
                return
            if sl is not None and price >= sl:
                self._close_position(sl, self.taker_fee, "sl")
                return

    def _process_trade(self, trade: TradeRow) -> None:
        bar_time = bar_time_from_ts(trade.timestamp, self.tf_seconds)
        self._on_new_bar(bar_time)
        price = trade.price
        if price <= 0:
            return
        self.last_trade_price = price
        if self.position:
            self._update_unrealized(price)
            self._check_tp_sl(price)
            if self.position:
                self._update_unrealized(price)
            return
        if not self.trading_enabled:
            return
        for order_id, order in list(self.orders.items()):
            if order.side == "Buy" and price <= order.price:
                self.orders.pop(order_id, None)
                self._fill_order(order)
                return
            if order.side == "Sell" and price >= order.price:
                self.orders.pop(order_id, None)
                self._fill_order(order)
                return

    def fetch_recent_trades(self, limit: int = 1000) -> pd.DataFrame:
        trades: List[Dict] = []
        for _ in range(limit):
            row = self.trade_stream.next()
            if row is None:
                break
            trades.append({
                "timestamp": row.timestamp,
                "price": row.price,
                "size": row.size,
                "side": row.side,
                "id": row.trade_id,
            })
            self._process_trade(row)
        if trades:
            self.clock.set(trades[-1]["timestamp"])
            self._update_trading_enabled()
        else:
            self.done = self.trade_stream.done
            if self.done and not self.trade_stream_finished_logged:
                self.trade_stream_finished_logged = True
                self._log(f"Trade stream finished at {self._format_time(self.clock.time())}")
        return pd.DataFrame(trades)

    def fetch_orderbook(self, limit: int = 50) -> Dict:
        if self.orderbook_stream.done:
            if not self.orderbook_finished_logged:
                self.orderbook_finished_logged = True
                self._log(f"Orderbook stream finished at {self._format_time(self.clock.time())}")
            return {}
        snap = self.orderbook_stream.advance_to(int(self.clock.time() * 1000))
        if not snap:
            return {}
        if self.last_ob_returned_ts == snap["ts"]:
            return {}
        self.last_ob_returned_ts = snap["ts"]
        return {"ts": snap["ts"], "b": snap["b"], "a": snap["a"]}

    def fetch_kline(self, interval: str = "15", limit: int = 200) -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_instrument_info(self) -> Dict:
        return {"min_qty": self.min_qty, "qty_step": self.qty_step, "tick_size": self.tick_size}

    def get_wallet_balance(self) -> Dict:
        equity = self.cash_balance
        if self.position:
            equity += self.position.get("unrealized_pnl", 0.0)
        return {"equity": equity, "available": equity, "total_balance": equity}

    def place_limit_order(self, side: str, price: float, qty: float, reduce_only: bool = False) -> Dict:
        if not self.trading_enabled:
            return {}
        if self.position:
            return {}
        if qty <= 0:
            return {}
        self.order_seq += 1
        order_id = f"sim-{self.order_seq}"
        self.orders[order_id] = SimOrder(
            order_id=order_id,
            side=side,
            price=price,
            qty=qty,
            created_ts=self.clock.time(),
        )
        self._log(f"Order placed {order_id} {side} price={price:.6f} qty={qty:.6f}")
        return {"result": {"orderId": order_id}}

    def cancel_all_orders(self) -> None:
        count = len(self.orders)
        self.orders = {}
        if count:
            self._log(f"Canceled all orders ({count})")

    def market_close(self, side: str, qty: float) -> None:
        if not self.position:
            return
        price = self.last_trade_price if self.last_trade_price else self.position.get("entry_price", 0.0)
        self._close_position(price, self.taker_fee, "market_close")

    def place_tp_sl(self, side: str, qty: float, tp: float, sl: float) -> None:
        if not self.position:
            return
        self.position["tp"] = tp
        self.position["sl"] = sl
        self._log(f"TP/SL set tp={tp:.6f} sl={sl:.6f}")

    def should_stop(self) -> bool:
        return self.done

    class _Session:
        def __init__(self, exchange: "SimulatedExchange"):
            self.exchange = exchange

        def get_open_orders(self, category: str = "linear", symbol: str = "") -> Dict:
            items = []
            for order in self.exchange.orders.values():
                items.append({
                    "orderId": order.order_id,
                    "side": order.side,
                    "price": order.price,
                    "qty": order.qty,
                })
            return {"result": {"list": items}}

        def cancel_order(self, category: str = "linear", symbol: str = "", orderId: str = "") -> Dict:
            removed = self.exchange.orders.pop(orderId, None)
            if removed:
                self.exchange._log(f"Order canceled {orderId}")
            return {"result": {"orderId": orderId}}

        def get_positions(self, category: str = "linear", symbol: str = "") -> Dict:
            if not self.exchange.position:
                return {"result": {"list": []}}
            pos = self.exchange.position
            size = pos.get("size", 0.0)
            side = pos.get("side", "Buy")
            entry = pos.get("entry_price", 0.0)
            mark = pos.get("mark_price", entry)
            unreal = pos.get("unrealized_pnl", 0.0)
            return {
                "result": {
                    "list": [
                        {
                            "symbol": self.exchange.symbol,
                            "size": size,
                            "side": side,
                            "avgPrice": entry,
                            "markPrice": mark,
                            "unrealisedPnl": unreal,
                            "cumRealisedPnl": self.exchange.cum_realized,
                        }
                    ]
                }
            }

