
"""
Live Trading V2 - Robust execution engine.
"""

import argparse
import json
import logging
import math
import os
import time
import traceback
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from config import CONF, GlobalConfig
from feature_engine import FeatureEngine

try:
    from exchange_client import ExchangeClient
except Exception as exc:
    ExchangeClient = None
    EXCHANGE_IMPORT_ERROR = exc

STATE_VERSION = 1
DEFAULT_WINDOW_BARS = 500
DEFAULT_TRADE_POLL_SEC = 1.0
DEFAULT_OB_POLL_SEC = 1.0
DEFAULT_RECONCILE_SEC = 30.0
DEFAULT_HEARTBEAT_SEC = 30.0
DEFAULT_INSTR_REFRESH_SEC = 300.0
DEFAULT_DATA_LAG_WARN_SEC = 15.0
DEFAULT_DATA_LAG_ERROR_SEC = 60.0
DEFAULT_DRIFT_WINDOW = 200
DEFAULT_DRIFT_Z = 3.0
DEFAULT_MAX_LEVERAGE = 5.0

BAR_COLUMNS = [
    "bar_time",
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "vol_buy",
    "vol_sell",
    "vol_delta",
    "dollar_val",
    "total_val",
    "vwap",
    "taker_buy_ratio",
    "buy_vol",
    "sell_vol",
    "ob_spread_mean",
    "ob_micro_dev_mean",
    "ob_micro_dev_std",
    "ob_micro_dev_last",
    "ob_imbalance_mean",
    "ob_imbalance_last",
    "ob_bid_depth_mean",
    "ob_ask_depth_mean",
]


# ---------------------------
# Utility helpers
# ---------------------------

def utc_now_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def timeframe_to_seconds(tf: str) -> int:
    mapping = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "1h": 3600,
        "4h": 14400,
    }
    if tf not in mapping:
        raise ValueError(f"Unsupported timeframe: {tf}")
    return mapping[tf]


def bar_time_from_ts(ts_sec: float, tf_seconds: int) -> int:
    return int(ts_sec // tf_seconds) * tf_seconds


def round_down(value: float, step: float) -> float:
    if step <= 0:
        return value
    return math.floor(value / step) * step


def atomic_write_json(path: Path, data: Dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    tmp_path.replace(path)


# ---------------------------
# Logging
# ---------------------------

def setup_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("LiveTradingV2")
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    if not logger.handlers:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        file_handler = logging.FileHandler(log_dir / "live_trading_v2.log")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(log_level)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger


# ---------------------------
# State management
# ---------------------------

class StateStore:
    def __init__(self, path: Path, symbol: str, logger: logging.Logger):
        self.path = path
        self.symbol = symbol
        self.logger = logger
        self.state = self._default_state()

    def _default_state(self) -> Dict:
        return {
            "version": STATE_VERSION,
            "symbol": self.symbol,
            "created_at": utc_now_str(),
            "last_trade_id": None,
            "last_trade_ts": 0.0,
            "last_bar_time": None,
            "last_processed_bar_time": None,
            "active_orders": {},
            "position": {},
            "daily": {
                "date": datetime.utcnow().strftime("%Y-%m-%d"),
                "realized_pnl": 0.0,
                "equity_start": None,
                "last_cum_realized": None,
            },
            "metrics": {
                "api_errors": 0,
                "last_error": None,
                "last_ok_ts": None,
            },
        }

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            with open(self.path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self.state.update(data)
        except Exception as exc:
            self.logger.error(f"Failed to load state: {exc}")

    def save(self) -> None:
        try:
            atomic_write_json(self.path, self.state)
        except Exception as exc:
            self.logger.error(f"Failed to save state: {exc}")

    def reset_daily_if_needed(self) -> None:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        if self.state["daily"]["date"] != today:
            self.state["daily"] = {
                "date": today,
                "realized_pnl": 0.0,
                "equity_start": None,
                "last_cum_realized": None,
            }
            self.save()


# ---------------------------
# Instrument info
# ---------------------------

@dataclass
class InstrumentInfo:
    min_qty: float = 0.0
    qty_step: float = 0.0
    tick_size: float = 0.0


# ---------------------------
# Tracking helpers
# ---------------------------

class LatencyTracker:
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self.avg_ms: Optional[float] = None
        self.max_ms: float = 0.0

    def update(self, latency_ms: float) -> None:
        if self.avg_ms is None:
            self.avg_ms = latency_ms
        else:
            self.avg_ms = (self.alpha * latency_ms) + (1.0 - self.alpha) * self.avg_ms
        self.max_ms = max(self.max_ms, latency_ms)


class RollingWindow:
    def __init__(self, maxlen: int):
        self.data: Deque[float] = deque(maxlen=maxlen)

    def add(self, value: float) -> None:
        if value is None:
            return
        if isinstance(value, float) and math.isnan(value):
            return
        self.data.append(float(value))

    def stats(self) -> Tuple[Optional[float], Optional[float]]:
        if len(self.data) < 2:
            return None, None
        arr = np.array(self.data, dtype=float)
        return float(np.mean(arr)), float(np.std(arr))


class DriftMonitor:
    def __init__(self, keys: List[str], window: int, z_threshold: float):
        self.keys = keys
        self.window = window
        self.z_threshold = z_threshold
        self.windows = {key: RollingWindow(window) for key in keys}

    def update(self, row: pd.Series, extras: Dict[str, float]) -> List[str]:
        alerts = []
        for key in self.keys:
            if key in row:
                value = row.get(key)
            else:
                value = extras.get(key)
            window = self.windows.get(key)
            if window is None:
                continue
            mean, std = window.stats()
            window.add(value)
            if mean is None or std is None or std == 0:
                continue
            z = (float(value) - mean) / std
            if abs(z) >= self.z_threshold:
                alerts.append(f"drift:{key} z={z:.2f}")
        return alerts


# ---------------------------
# Market data aggregation
# ---------------------------

class TradeBarBuilder:
    def __init__(self, tf_seconds: int):
        self.tf_seconds = tf_seconds
        self.current_bar_time: Optional[int] = None
        self.last_price: Optional[float] = None
        self._reset()

    def _reset(self) -> None:
        self.open = None
        self.high = None
        self.low = None
        self.close = None
        self.volume = 0.0
        self.vol_buy = 0.0
        self.vol_sell = 0.0
        self.trade_count = 0
        self.dollar_val = 0.0

    def _start_bar(self, bar_time: int, price: float) -> None:
        self.current_bar_time = bar_time
        self._reset()
        self.open = price
        self.high = price
        self.low = price
        self.close = price

    def _update_bar(self, price: float, size: float, side: str) -> None:
        self.high = max(self.high, price) if self.high is not None else price
        self.low = min(self.low, price) if self.low is not None else price
        self.close = price
        self.volume += size
        self.dollar_val += price * size
        if side == "Buy":
            self.vol_buy += size
        elif side == "Sell":
            self.vol_sell += size
        self.trade_count += 1
        self.last_price = price

    def _synthetic_bar(self, bar_time: int) -> Dict:
        price = self.last_price if self.last_price is not None else 0.0
        return {
            "bar_time": bar_time,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 0.0,
            "vol_buy": 0.0,
            "vol_sell": 0.0,
            "trade_count": 0,
            "dollar_val": 0.0,
        }

    def _finalize_bar(self, force_synthetic: bool = False) -> Optional[Dict]:
        if self.current_bar_time is None:
            return None
        if self.trade_count == 0 and force_synthetic:
            bar = self._synthetic_bar(self.current_bar_time)
        elif self.trade_count == 0:
            return None
        else:
            bar = {
                "bar_time": self.current_bar_time,
                "open": self.open,
                "high": self.high,
                "low": self.low,
                "close": self.close,
                "volume": self.volume,
                "vol_buy": self.vol_buy,
                "vol_sell": self.vol_sell,
                "trade_count": self.trade_count,
                "dollar_val": self.dollar_val,
            }
        bar["vol_delta"] = bar["vol_buy"] - bar["vol_sell"]
        bar["total_val"] = bar["dollar_val"]
        bar["buy_vol"] = bar["vol_buy"]
        bar["sell_vol"] = bar["vol_sell"]
        if bar["volume"] > 0:
            bar["vwap"] = bar["dollar_val"] / bar["volume"]
            bar["taker_buy_ratio"] = bar["vol_buy"] / bar["volume"]
        else:
            bar["vwap"] = bar["close"]
            bar["taker_buy_ratio"] = 0.5
        return bar

    def add_trade(self, ts_sec: float, price: float, size: float, side: str) -> List[Dict]:
        completed: List[Dict] = []
        bar_time = bar_time_from_ts(ts_sec, self.tf_seconds)

        if self.current_bar_time is None:
            self._start_bar(bar_time, price)
        elif bar_time < self.current_bar_time:
            return completed
        elif bar_time > self.current_bar_time:
            completed.append(self._finalize_bar(force_synthetic=True))
            gap = int((bar_time - self.current_bar_time) / self.tf_seconds) - 1
            if gap > 0:
                for i in range(gap):
                    gap_time = self.current_bar_time + self.tf_seconds * (i + 1)
                    completed.append(self._synthetic_bar(gap_time))
            self._start_bar(bar_time, price)

        self._update_bar(price, size, side)
        return [c for c in completed if c is not None]

    def force_close(self, now_ts: float) -> List[Dict]:
        completed: List[Dict] = []
        if self.current_bar_time is None:
            return completed
        next_close = self.current_bar_time + self.tf_seconds
        if now_ts < next_close:
            return completed

        completed.append(self._finalize_bar(force_synthetic=True))
        gap = int((now_ts - self.current_bar_time) / self.tf_seconds) - 1
        if gap > 0:
            for i in range(gap):
                gap_time = self.current_bar_time + self.tf_seconds * (i + 1)
                completed.append(self._synthetic_bar(gap_time))
        self.current_bar_time = None
        return [c for c in completed if c is not None]


class OrderbookAggregator:
    def __init__(self, tf_seconds: int, depth_levels: int, logger: logging.Logger):
        self.tf_seconds = tf_seconds
        self.depth_levels = depth_levels
        self.logger = logger
        self.buckets: Dict[int, Dict] = {}
        self.last_snapshot_ts: Optional[int] = None

    def ingest(self, snapshot: Dict) -> None:
        if not snapshot:
            return
        bids = snapshot.get("b") or []
        asks = snapshot.get("a") or []
        if not bids or not asks:
            return
        ts_ms = safe_int(snapshot.get("ts"), default=int(time.time() * 1000))
        self.last_snapshot_ts = ts_ms
        bar_time = bar_time_from_ts(ts_ms / 1000.0, self.tf_seconds)

        try:
            bb_price = safe_float(bids[0][0])
            bb_size = safe_float(bids[0][1])
            ba_price = safe_float(asks[0][0])
            ba_size = safe_float(asks[0][1])
        except Exception:
            return

        bid_depth = sum(safe_float(b[1]) for b in bids[: self.depth_levels])
        ask_depth = sum(safe_float(a[1]) for a in asks[: self.depth_levels])

        spread = ba_price - bb_price
        mid = (ba_price + bb_price) / 2.0
        micro = (ba_price * bb_size + bb_price * ba_size) / (bb_size + ba_size + 1e-9)
        micro_dev = micro - mid
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-9)

        bucket = self.buckets.get(bar_time)
        if bucket is None:
            bucket = {
                "count": 0,
                "spread_sum": 0.0,
                "micro_sum": 0.0,
                "micro_sq_sum": 0.0,
                "micro_last": 0.0,
                "imb_sum": 0.0,
                "imb_last": 0.0,
                "bid_depth_sum": 0.0,
                "ask_depth_sum": 0.0,
            }
            self.buckets[bar_time] = bucket

        bucket["count"] += 1
        bucket["spread_sum"] += spread
        bucket["micro_sum"] += micro_dev
        bucket["micro_sq_sum"] += micro_dev * micro_dev
        bucket["micro_last"] = micro_dev
        bucket["imb_sum"] += imbalance
        bucket["imb_last"] = imbalance
        bucket["bid_depth_sum"] += bid_depth
        bucket["ask_depth_sum"] += ask_depth

    def finalize(self, bar_time: int) -> Optional[Dict]:
        bucket = self.buckets.pop(bar_time, None)
        if not bucket or bucket["count"] == 0:
            return None
        count = bucket["count"]
        micro_mean = bucket["micro_sum"] / count
        micro_var = (bucket["micro_sq_sum"] / count) - (micro_mean * micro_mean)
        micro_std = math.sqrt(max(0.0, micro_var))

        return {
            "ob_spread_mean": bucket["spread_sum"] / count,
            "ob_micro_dev_mean": micro_mean,
            "ob_micro_dev_std": micro_std,
            "ob_micro_dev_last": bucket["micro_last"],
            "ob_imbalance_mean": bucket["imb_sum"] / count,
            "ob_imbalance_last": bucket["imb_last"],
            "ob_bid_depth_mean": bucket["bid_depth_sum"] / count,
            "ob_ask_depth_mean": bucket["ask_depth_sum"] / count,
        }

    def prune_before(self, bar_time: int) -> None:
        stale = [bt for bt in self.buckets.keys() if bt < bar_time]
        for bt in stale:
            self.buckets.pop(bt, None)


class BarWindow:
    def __init__(self, tf_seconds: int, depth_levels: int, window_size: int, logger: logging.Logger):
        self.tf_seconds = tf_seconds
        self.depth_levels = depth_levels
        self.window_size = window_size
        self.logger = logger
        self.trade_builder = TradeBarBuilder(tf_seconds)
        self.ob_agg = OrderbookAggregator(tf_seconds, depth_levels, logger)
        self.bars = pd.DataFrame(columns=BAR_COLUMNS)
        self.last_ob: Dict[str, float] = {k: 0.0 for k in BAR_COLUMNS if k.startswith("ob_")}

    def bootstrap_from_klines(self, klines: pd.DataFrame) -> None:
        if klines.empty:
            return
        df = klines.copy()
        if "timestamp" not in df.columns:
            return
        df = df.sort_values("timestamp")
        if len(df) > 1:
            df = df.iloc[:-1]

        rows = []
        for _, row in df.iterrows():
            ts = safe_float(row["timestamp"])
            bar_time = bar_time_from_ts(ts, self.tf_seconds)
            volume = safe_float(row.get("volume"))
            close = safe_float(row.get("close"))
            open_p = safe_float(row.get("open"))
            high = safe_float(row.get("high"))
            low = safe_float(row.get("low"))
            turnover = safe_float(row.get("turnover"))
            dollar_val = turnover if turnover > 0 else close * volume
            trade_count = 1 if volume > 0 else 0
            vol_buy = volume / 2.0
            vol_sell = volume / 2.0

            bar = {
                "bar_time": bar_time,
                "datetime": pd.to_datetime(bar_time, unit="s"),
                "open": open_p,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "trade_count": trade_count,
                "vol_buy": vol_buy,
                "vol_sell": vol_sell,
                "vol_delta": vol_buy - vol_sell,
                "dollar_val": dollar_val,
                "total_val": dollar_val,
                "vwap": (dollar_val / volume) if volume > 0 else close,
                "taker_buy_ratio": 0.5,
                "buy_vol": vol_buy,
                "sell_vol": vol_sell,
            }
            for key in self.last_ob:
                bar[key] = 0.0
            rows.append(bar)

        if rows:
            self.bars = pd.DataFrame(rows)
            self.bars = self.bars.tail(self.window_size).reset_index(drop=True)
            last_row = self.bars.iloc[-1]
            self.trade_builder.last_price = safe_float(last_row.get("close"))

    def ingest_orderbook(self, snapshot: Dict) -> None:
        self.ob_agg.ingest(snapshot)

    def ingest_trades(self, trades: List[Dict], now_ts: float) -> List[int]:
        closed_times: List[int] = []
        for trade in trades:
            ts = trade["timestamp"]
            price = trade["price"]
            size = trade["size"]
            side = trade["side"]
            completed = self.trade_builder.add_trade(ts, price, size, side)
            for bar in completed:
                self._append_bar(bar)
                closed_times.append(bar["bar_time"])

        forced = self.trade_builder.force_close(now_ts)
        for bar in forced:
            self._append_bar(bar)
            closed_times.append(bar["bar_time"])

        return closed_times

    def _append_bar(self, bar: Dict) -> None:
        ob = self.ob_agg.finalize(bar["bar_time"])
        if ob:
            self.last_ob.update(ob)
        for key, val in self.last_ob.items():
            bar.setdefault(key, val)

        bar["datetime"] = pd.to_datetime(bar["bar_time"], unit="s")
        row = {col: bar.get(col, 0.0) for col in BAR_COLUMNS}

        self.bars = pd.concat([self.bars, pd.DataFrame([row])], ignore_index=True)
        if len(self.bars) > self.window_size:
            self.bars = self.bars.iloc[-self.window_size :].reset_index(drop=True)

    def latest_bar_time(self) -> Optional[int]:
        if self.bars.empty:
            return None
        return int(self.bars.iloc[-1]["bar_time"])

    def latest_close(self) -> Optional[float]:
        if self.bars.empty:
            return None
        return safe_float(self.bars.iloc[-1]["close"])


# ---------------------------
# Exchange wrapper
# ---------------------------

class SafeExchange:
    def __init__(self, exchange: ExchangeClient, logger: logging.Logger):
        self.exchange = exchange
        self.logger = logger
        self.latency = LatencyTracker()

    def _call(self, name: str, func, *args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            latency_ms = (time.time() - start) * 1000.0
            self.latency.update(latency_ms)
            return result
        except Exception as exc:
            latency_ms = (time.time() - start) * 1000.0
            self.latency.update(latency_ms)
            self.logger.error(f"API error in {name}: {exc}")
            return None

    def fetch_recent_trades(self, limit: int) -> pd.DataFrame:
        return self._call("fetch_recent_trades", self.exchange.fetch_recent_trades, limit)

    def fetch_orderbook(self, limit: int) -> Dict:
        return self._call("fetch_orderbook", self.exchange.fetch_orderbook, limit)

    def fetch_kline(self, interval: str, limit: int) -> pd.DataFrame:
        return self._call("fetch_kline", self.exchange.fetch_kline, interval, limit)

    def get_wallet_balance(self) -> Dict:
        return self._call("get_wallet_balance", self.exchange.get_wallet_balance)

    def get_open_orders(self) -> List[Dict]:
        def _call_open():
            return self.exchange.session.get_open_orders(category="linear", symbol=self.exchange.symbol)
        resp = self._call("get_open_orders", _call_open)
        if not resp:
            return []
        return resp.get("result", {}).get("list", [])

    def get_position_details(self) -> Dict:
        def _call_positions():
            return self.exchange.session.get_positions(category="linear", symbol=self.exchange.symbol)
        resp = self._call("get_positions", _call_positions)
        if not resp:
            return {}
        positions = resp.get("result", {}).get("list", [])
        for pos in positions:
            if pos.get("symbol") != self.exchange.symbol:
                continue
            size = safe_float(pos.get("size"))
            side = pos.get("side")
            avg_price = safe_float(pos.get("avgPrice"))
            mark_price = safe_float(pos.get("markPrice"))
            unreal_pnl = safe_float(pos.get("unrealisedPnl"))
            cum_realized = safe_float(pos.get("cumRealisedPnl"))
            return {
                "size": size,
                "side": side,
                "avg_price": avg_price,
                "mark_price": mark_price,
                "unrealized_pnl": unreal_pnl,
                "cum_realized_pnl": cum_realized,
            }
        return {}

    def get_instrument_info(self) -> Dict:
        return self._call("fetch_instrument_info", self.exchange.fetch_instrument_info)

    def place_limit_order(self, side: str, price: float, qty: float) -> Dict:
        return self._call("place_limit_order", self.exchange.place_limit_order, side, price, qty)

    def cancel_order(self, order_id: str) -> bool:
        def _cancel():
            return self.exchange.session.cancel_order(category="linear", symbol=self.exchange.symbol, orderId=order_id)
        resp = self._call("cancel_order", _cancel)
        return bool(resp)

    def cancel_all_orders(self) -> None:
        self._call("cancel_all_orders", self.exchange.cancel_all_orders)

    def market_close(self, side: str, qty: float) -> None:
        self._call("market_close", self.exchange.market_close, side, qty)

    def set_tp_sl(self, side: str, qty: float, tp: float, sl: float) -> None:
        self._call("place_tp_sl", self.exchange.place_tp_sl, side, qty, tp, sl)


# ---------------------------
# Live trading engine
# ---------------------------

class LiveTradingV2:
    def __init__(self, args: argparse.Namespace):
        self.logger = setup_logger(args.log_level)
        self.config = CONF
        self.config.data.data_dir = Path(args.data_dir)
        self.config.data.symbol = args.symbol

        self.tf_seconds = timeframe_to_seconds(self.config.features.base_timeframe)
        self.window_size = args.window
        self.poll_trades_sec = args.trade_poll_sec
        self.poll_ob_sec = args.ob_poll_sec
        self.reconcile_sec = args.reconcile_sec
        self.heartbeat_sec = args.heartbeat_sec
        self.instr_refresh_sec = args.instr_refresh_sec
        self.data_lag_warn = args.data_lag_warn_sec
        self.data_lag_error = args.data_lag_error_sec
        self.max_leverage = args.max_leverage

        self.dry_run = args.dry_run
        self.trade_enabled = True

        self.state = StateStore(Path(f"bot_state_{self.config.data.symbol}_v2.json"), self.config.data.symbol, self.logger)
        self.state.load()
        self.state.reset_daily_if_needed()

        self.exchange = self._init_exchange()
        self.api = SafeExchange(self.exchange, self.logger)
        self.instrument = InstrumentInfo()

        self.feature_engine = FeatureEngine(self.config.features)
        self.model_long, self.model_short, self.model_features = self._load_model(args.model_dir)
        self._apply_model_params(args.model_dir)

        self.bar_window = BarWindow(self.tf_seconds, self.config.data.ob_levels, self.window_size, self.logger)
        self._bootstrap_bars()

        self.drift_monitor = DriftMonitor(args.drift_keys, args.drift_window, args.drift_z)

        self.last_trade_ids: Deque[str] = deque(maxlen=500)

        self._refresh_instrument_info(force=True)
        self._reconcile("startup")
        self._log_manifest(args.model_dir)

    def _init_exchange(self) -> ExchangeClient:
        if ExchangeClient is None:
            raise RuntimeError(f"exchange_client import failed: {EXCHANGE_IMPORT_ERROR}")

        key = os.getenv("BYBIT_API_KEY")
        secret = os.getenv("BYBIT_API_SECRET")
        if not key or not secret:
            self.logger.warning("API keys missing. Switching to dry run mode.")
            self.dry_run = True
            key = "dummy"
            secret = "dummy"

        return ExchangeClient(key, secret, self.config.data.symbol)

    def _load_model(self, model_dir: str) -> Tuple[object, object, List[str]]:
        path = Path(model_dir)
        if not path.exists():
            raise FileNotFoundError(f"Model path not found: {path}")
        model_long = joblib.load(path / "model_long.pkl")
        model_short = joblib.load(path / "model_short.pkl")
        features = joblib.load(path / "features.pkl")
        if not isinstance(features, list) or not features:
            raise ValueError("features.pkl must contain a non-empty list")
        return model_long, model_short, features

    def _apply_model_params(self, model_dir: str) -> None:
        path = Path(model_dir) / "params.json"
        if not path.exists():
            self.logger.warning("params.json not found. Using config defaults.")
            return
        try:
            with open(path, "r") as f:
                params = json.load(f)
        except Exception as exc:
            self.logger.error(f"Failed to load params.json: {exc}")
            return

        self.config.strategy.base_limit_offset_atr = safe_float(params.get("limit_offset_atr"), self.config.strategy.base_limit_offset_atr)
        self.config.strategy.take_profit_atr = safe_float(params.get("take_profit_atr"), self.config.strategy.take_profit_atr)
        self.config.strategy.stop_loss_atr = safe_float(params.get("stop_loss_atr"), self.config.strategy.stop_loss_atr)
        self.config.model.model_threshold = safe_float(params.get("model_threshold"), self.config.model.model_threshold)

    def _bootstrap_bars(self) -> None:
        interval_map = {60: "1", 300: "5", 900: "15", 3600: "60", 14400: "240"}
        interval = interval_map.get(self.tf_seconds, "15")
        klines = self.api.fetch_kline(interval=interval, limit=self.window_size)
        if isinstance(klines, pd.DataFrame) and not klines.empty:
            self.bar_window.bootstrap_from_klines(klines)
            self.logger.info(f"Bootstrapped {len(self.bar_window.bars)} bars from klines.")
        else:
            self.logger.warning("No klines available for bootstrap.")

    def _refresh_instrument_info(self, force: bool = False) -> None:
        info = self.api.get_instrument_info()
        if not info:
            if force:
                self.logger.warning("Instrument info unavailable.")
            return
        self.instrument = InstrumentInfo(
            min_qty=safe_float(info.get("min_qty")),
            qty_step=safe_float(info.get("qty_step")),
            tick_size=safe_float(info.get("tick_size")),
        )

    def _log_manifest(self, model_dir: str) -> None:
        self.logger.info("=" * 60)
        self.logger.info("LIVE TRADING V2 MANIFEST")
        self.logger.info("=" * 60)
        self.logger.info(f"Symbol: {self.config.data.symbol}")
        self.logger.info(f"Timeframe: {self.config.features.base_timeframe}")
        self.logger.info(f"Model Dir: {model_dir}")
        self.logger.info(f"Feature Count: {len(self.model_features)}")
        self.logger.info(f"Threshold: {self.config.model.model_threshold:.3f}")
        self.logger.info(f"Limit Offset ATR: {self.config.strategy.base_limit_offset_atr:.3f}")
        self.logger.info(f"TP ATR: {self.config.strategy.take_profit_atr:.3f}")
        self.logger.info(f"SL ATR: {self.config.strategy.stop_loss_atr:.3f}")
        self.logger.info(f"Order Timeout Bars: {self.config.strategy.time_limit_bars}")
        self.logger.info(f"Max Holding Bars: {self.config.strategy.max_holding_bars}")
        self.logger.info(f"Risk Per Trade: {self.config.strategy.risk_per_trade:.3f}")
        self.logger.info(f"Dry Run: {self.dry_run}")
        self.logger.info(f"Min Qty: {self.instrument.min_qty} | Qty Step: {self.instrument.qty_step} | Tick Size: {self.instrument.tick_size}")
        self.logger.info("=" * 60)

    def _filter_new_trades(self, trades_df: pd.DataFrame) -> List[Dict]:
        if trades_df is None or trades_df.empty:
            return []
        trades_df = trades_df.sort_values("timestamp")
        new_trades = []
        last_ts = safe_float(self.state.state.get("last_trade_ts"), 0.0)
        for _, row in trades_df.iterrows():
            ts = safe_float(row.get("timestamp"))
            trade_id = str(row.get("id")) if "id" in row else None
            if ts < last_ts:
                continue
            if trade_id and trade_id in self.last_trade_ids:
                continue
            new_trades.append({
                "timestamp": ts,
                "price": safe_float(row.get("price")),
                "size": safe_float(row.get("size")),
                "side": row.get("side") or "",
                "id": trade_id,
            })
        if new_trades:
            max_ts = max(t["timestamp"] for t in new_trades)
            self.state.state["last_trade_ts"] = max_ts
            for trade in new_trades:
                if trade["id"]:
                    self.last_trade_ids.append(trade["id"])
        return new_trades

    def _reconcile(self, reason: str) -> None:
        if self.dry_run:
            return
        open_orders = self.api.get_open_orders()
        position = self.api.get_position_details()
        state_orders = self.state.state.get("active_orders", {})

        active_ids = set()
        for o in open_orders:
            oid = o.get("orderId")
            if not oid:
                continue
            active_ids.add(oid)
            if oid not in state_orders:
                state_orders[oid] = {
                    "side": o.get("side"),
                    "price": safe_float(o.get("price")),
                    "qty": safe_float(o.get("qty")),
                    "created_bar_time": self.state.state.get("last_bar_time"),
                    "created_ts": utc_now_str(),
                    "external": True,
                }
                self.logger.warning(f"External order detected: {oid}")

        missing = [oid for oid in state_orders.keys() if oid not in active_ids]
        for oid in missing:
            state_orders.pop(oid, None)

        self.state.state["active_orders"] = state_orders

        pos_size = safe_float(position.get("size")) if position else 0.0
        if pos_size == 0 and self.state.state.get("position"):
            self.logger.info("Position closed on exchange.")
            self.state.state["position"] = {}
        elif pos_size != 0:
            side = position.get("side")
            entry_price = safe_float(position.get("avg_price"))
            mark_price = safe_float(position.get("mark_price"))
            unreal_pnl = safe_float(position.get("unrealized_pnl"))
            cum_realized = safe_float(position.get("cum_realized_pnl"))
            pos_state = self.state.state.get("position", {})
            if not pos_state:
                self.logger.warning("External position detected. Taking ownership.")
                pos_state = {
                    "side": side,
                    "size": pos_size,
                    "entry_price": entry_price,
                    "entry_bar_time": self.state.state.get("last_bar_time"),
                    "created_ts": utc_now_str(),
                    "tp_sl_set": False,
                }
            pos_state.update({
                "side": side,
                "size": pos_size,
                "entry_price": entry_price,
                "mark_price": mark_price,
                "unrealized_pnl": unreal_pnl,
                "cum_realized_pnl": cum_realized,
            })
            self.state.state["position"] = pos_state

        daily = self.state.state.get("daily", {})
        last_cum = daily.get("last_cum_realized")
        if position:
            cum_realized = safe_float(position.get("cum_realized_pnl"), None)
            if cum_realized is not None:
                if last_cum is None:
                    daily["last_cum_realized"] = cum_realized
                else:
                    delta = cum_realized - last_cum
                    if abs(delta) > 0:
                        daily["realized_pnl"] = safe_float(daily.get("realized_pnl")) + delta
                        daily["last_cum_realized"] = cum_realized
        self.state.state["daily"] = daily

        self.state.save()
        self.logger.info(f"Reconcile complete ({reason}). Open orders: {len(state_orders)}")

    def _ensure_tp_sl(self, latest_row: pd.Series) -> None:
        if self.dry_run:
            return
        pos = self.state.state.get("position", {})
        if not pos:
            return
        if pos.get("tp_sl_set"):
            return
        size = safe_float(pos.get("size"))
        entry_price = safe_float(pos.get("entry_price"))
        if size <= 0 or entry_price <= 0:
            return
        side = pos.get("side")
        atr = safe_float(latest_row.get("atr"))
        if atr <= 0:
            return
        tp_atr = self.config.strategy.take_profit_atr
        sl_atr = self.config.strategy.stop_loss_atr

        if side == "Buy":
            tp = entry_price + (atr * tp_atr)
            sl = entry_price - (atr * sl_atr)
        else:
            tp = entry_price - (atr * tp_atr)
            sl = entry_price + (atr * sl_atr)

        self.api.set_tp_sl(side, size, tp, sl)
        pos["tp_sl_set"] = True
        self.state.state["position"] = pos
        self.state.save()
        self.logger.info(f"TP/SL set for position: TP={tp:.6f} SL={sl:.6f}")

    def _expire_orders(self, bar_time: int) -> None:
        orders = self.state.state.get("active_orders", {})
        if not orders:
            return
        expired = []
        for oid, info in orders.items():
            created_bar = info.get("created_bar_time")
            if created_bar is None:
                continue
            age_bars = int((bar_time - created_bar) / self.tf_seconds)
            if age_bars > self.config.strategy.time_limit_bars:
                expired.append(oid)
        if not expired:
            return
        for oid in expired:
            if not self.dry_run:
                self.api.cancel_order(oid)
            orders.pop(oid, None)
            self.logger.info(f"Order expired and canceled: {oid}")
        self.state.state["active_orders"] = orders
        self.state.save()

    def _cancel_active_orders(self, reason: str) -> None:
        orders = self.state.state.get("active_orders", {})
        if not orders:
            return
        for oid in list(orders.keys()):
            if not self.dry_run:
                self.api.cancel_order(oid)
            orders.pop(oid, None)
        self.state.state["active_orders"] = orders
        self.state.save()
        self.logger.info(f"Active orders canceled ({reason}).")

    def _check_position_timeout(self, bar_time: int) -> None:
        pos = self.state.state.get("position", {})
        if not pos:
            return
        entry_bar = pos.get("entry_bar_time")
        if entry_bar is None:
            return
        age_bars = int((bar_time - entry_bar) / self.tf_seconds)
        if age_bars < self.config.strategy.max_holding_bars:
            return
        size = safe_float(pos.get("size"))
        if size <= 0:
            return
        side = pos.get("side") or "Buy"
        self.logger.warning("Position timeout reached. Closing at market.")
        if not self.dry_run:
            self.api.market_close(side, abs(size))
        self.state.state["position"] = {}
        self.state.save()

    def _place_order(self, side: str, latest_row: pd.Series, equity: float) -> Optional[str]:
        atr = safe_float(latest_row.get("atr"))
        close = safe_float(latest_row.get("close"))
        if atr <= 0 or close <= 0:
            return None

        offset = self.config.strategy.base_limit_offset_atr
        if side == "Buy":
            price = close - (atr * offset)
        else:
            price = close + (atr * offset)

        tick = self.instrument.tick_size
        price = round_down(price, tick) if tick > 0 else price
        if price <= 0:
            return None

        stop_dist = atr * self.config.strategy.stop_loss_atr
        risk_dollars = equity * self.config.strategy.risk_per_trade
        qty = risk_dollars / stop_dist if stop_dist > 0 else 0.0

        max_qty = (equity * self.max_leverage) / price if price > 0 else qty
        qty = min(qty, max_qty)

        qty = round_down(qty, self.instrument.qty_step)
        qty = round(qty, 5)
	
        if qty <= 0:
            self.logger.warning(f"Qty size for entry <=0. Skipping order.")
            return None
        if qty < 5 / price:
            qty = 5 / price
            self.logger.warning(f"Qty < min order value for the instrument. Setting Qty to the minimum value.")		
        
        self.logger.info(f"Placing {side} limit: price={price:.6f} qty={qty:.6f}")
        if self.dry_run:
            return "dry_run"

        resp = self.api.place_limit_order(side, price, qty)
        if not resp:
            return None
        order_id = resp.get("result", {}).get("orderId")
        if order_id:
            self.state.state["active_orders"][order_id] = {
                "side": side,
                "price": price,
                "qty": qty,
                "created_bar_time": self.state.state.get("last_bar_time"),
                "created_ts": utc_now_str(),
                "external": False,
            }
            self.state.save()
        return order_id

    def _check_feature_coverage(self, df_feat: pd.DataFrame) -> Tuple[bool, List[str]]:
        missing = [f for f in self.model_features if f not in df_feat.columns]
        if missing:
            return False, missing
        last_row = df_feat[self.model_features].iloc[-1]
        if last_row.isnull().any():
            missing_nan = last_row[last_row.isnull()].index.tolist()
            return False, missing_nan
        finite_mask = np.isfinite(last_row.astype(float))
        if not finite_mask.all():
            bad = last_row.index[~finite_mask].tolist()
            return False, bad
        return True, []

    def _compute_predictions(self, df_feat: pd.DataFrame) -> Tuple[float, float]:
        X = df_feat[self.model_features]
        pred_long = float(self.model_long.predict(X)[-1])
        pred_short = float(self.model_short.predict(X)[-1])
        return pred_long, pred_short

    def _process_bar(self, bar_time: int) -> None:
        if self.state.state.get("last_processed_bar_time") == bar_time:
            return
        self.state.state["last_processed_bar_time"] = bar_time
        self.state.state["last_bar_time"] = bar_time

        bars_df = self.bar_window.bars.copy()
        if bars_df.empty:
            return
        df_feat = self.feature_engine.calculate_features(bars_df)

        ok, missing = self._check_feature_coverage(df_feat)
        if not ok:
            self.trade_enabled = False
            self.logger.error(f"Feature mismatch or NaN. Missing: {missing}")
            self.state.save()
            return
        self.trade_enabled = True

        latest = df_feat.iloc[-1]
        pred_long, pred_short = self._compute_predictions(df_feat)
        threshold = self.config.model.model_threshold

        equity = 10000.0
        if not self.dry_run:
            bal = self.api.get_wallet_balance()
            if isinstance(bal, dict):
                equity = safe_float(bal.get("equity"), equity)
                if self.state.state["daily"]["equity_start"] is None and equity > 0:
                    self.state.state["daily"]["equity_start"] = equity

        spread = safe_float(latest.get("ob_spread_mean"))
        close = safe_float(latest.get("close"))
        spread_pct = (spread / close) if close > 0 else 0.0

        self.logger.info(
            f"BAR {pd.to_datetime(bar_time, unit='s')} | close={close:.6f} atr={safe_float(latest.get('atr')):.6f} "
            f"spread={spread_pct:.4%} predL={pred_long:.3f} predS={pred_short:.3f}"
        )

        drift_alerts = self.drift_monitor.update(latest, {"pred_long": pred_long, "pred_short": pred_short})
        if drift_alerts:
            self.logger.warning("Drift alerts: " + ", ".join(drift_alerts))

        if spread_pct > self.config.live.max_spread_pct:
            self.logger.warning(f"Spread too wide ({spread_pct:.4%}). Skipping orders.")
            self.state.save()
            return

        self._expire_orders(bar_time)
        self._check_position_timeout(bar_time)

        pos = self.state.state.get("position", {})
        daily_pnl = safe_float(self.state.state.get("daily", {}).get("realized_pnl"))
        drawdown_limit = -equity * self.config.live.max_daily_drawdown_pct
        if daily_pnl < drawdown_limit:
            self.logger.critical("Daily drawdown limit exceeded. Trading paused.")
            if pos and not self.dry_run:
                size = safe_float(pos.get("size"))
                side = pos.get("side") or "Buy"
                if size > 0:
                    self.api.market_close(side, abs(size))
            self._cancel_active_orders("drawdown")
            self.trade_enabled = False
            self.state.save()
            return

        if pos:
            if self.state.state.get("active_orders"):
                self._cancel_active_orders("position_open")
            self._ensure_tp_sl(latest)
            self.state.save()
            return

        last_trade_ts = safe_float(self.state.state.get("last_trade_ts"), 0.0)
        if last_trade_ts > 0 and (time.time() - last_trade_ts) > self.data_lag_error:
            self.logger.error("Trade data lag exceeds error threshold. Skipping orders.")
            self.state.save()
            return

        if not self.trade_enabled:
            self.state.save()
            return

        orders = self.state.state.get("active_orders", {})
        has_buy = any(info.get("side") == "Buy" for info in orders.values())
        has_sell = any(info.get("side") == "Sell" for info in orders.values())

        if pred_long > threshold and not has_buy:
            self._place_order("Buy", latest, equity)
        if pred_short > threshold and not has_sell:
            self._place_order("Sell", latest, equity)

        self.state.save()

    def _log_heartbeat(self) -> None:
        last_bar = self.state.state.get("last_bar_time")
        last_trade_ts = safe_float(self.state.state.get("last_trade_ts"))
        now = time.time()
        lag_trade = now - last_trade_ts if last_trade_ts > 0 else None
        lag_bar = now - last_bar if last_bar else None

        pos = self.state.state.get("position", {})
        pos_size = safe_float(pos.get("size")) if pos else 0.0
        pos_side = pos.get("side") if pos else "Flat"
        unreal = safe_float(pos.get("unrealized_pnl")) if pos else 0.0
        daily_pnl = safe_float(self.state.state.get("daily", {}).get("realized_pnl"))

        equity = None
        if not self.dry_run:
            bal = self.api.get_wallet_balance()
            if isinstance(bal, dict):
                equity = safe_float(bal.get("equity"))

        latency_avg = self.api.latency.avg_ms or 0.0
        latency_max = self.api.latency.max_ms

        self.logger.info(
            "HEARTBEAT | "
            f"pos={pos_side} {pos_size:.4f} | "
            f"open_orders={len(self.state.state.get('active_orders', {}))} | "
            f"equity={equity if equity is not None else 'n/a'} | "
            f"daily_pnl={daily_pnl:.2f} | unreal={unreal:.2f} | "
            f"lag_trade={lag_trade:.1f}s | lag_bar={lag_bar:.1f}s | "
            f"latency_avg={latency_avg:.1f}ms max={latency_max:.1f}ms"
        )

        if lag_trade is not None and lag_trade > self.data_lag_warn:
            self.logger.warning(f"Trade data lag: {lag_trade:.1f}s")
        if lag_trade is not None and lag_trade > self.data_lag_error:
            self.logger.error("Trade data lag exceeds error threshold.")

        if pos and equity is not None:
            entry_price = safe_float(pos.get("entry_price"))
            close = self.bar_window.latest_close()
            if entry_price > 0 and close is not None:
                side = 1.0 if pos_side == "Buy" else -1.0
                local_unreal = (close - entry_price) * pos_size * side
                diff = abs(local_unreal - unreal)
                if diff > max(1.0, equity * 0.001):
                    self.logger.warning(
                        f"PnL mismatch. local={local_unreal:.2f} exch={unreal:.2f} diff={diff:.2f}"
                    )

    def run(self) -> None:
        last_trade_poll = 0.0
        last_ob_poll = 0.0
        last_reconcile = 0.0
        last_heartbeat = 0.0
        last_instr_refresh = 0.0
        error_count = 0

        while True:
            try:
                now = time.time()

                if now - last_instr_refresh >= self.instr_refresh_sec:
                    self._refresh_instrument_info()
                    last_instr_refresh = now

                if now - last_ob_poll >= self.poll_ob_sec:
                    ob = self.api.fetch_orderbook(limit=self.config.data.ob_levels)
                    if isinstance(ob, dict) and ob:
                        self.bar_window.ingest_orderbook(ob)
                    last_ob_poll = now

                if now - last_trade_poll >= self.poll_trades_sec:
                    trades_df = self.api.fetch_recent_trades(limit=1000)
                    if isinstance(trades_df, pd.DataFrame) and not trades_df.empty:
                        new_trades = self._filter_new_trades(trades_df)
                        closed_times = self.bar_window.ingest_trades(new_trades, now)
                        for bar_time in closed_times:
                            self._process_bar(bar_time)
                    last_trade_poll = now

                if now - last_reconcile >= self.reconcile_sec:
                    self._reconcile("interval")
                    last_reconcile = now

                if now - last_heartbeat >= self.heartbeat_sec:
                    self._log_heartbeat()
                    last_heartbeat = now

                error_count = 0
                time.sleep(0.1)

            except KeyboardInterrupt:
                self.logger.info("Shutdown requested.")
                break
            except Exception as exc:
                error_count += 1
                self.logger.error(f"Runtime error: {exc}")
                self.logger.debug(traceback.format_exc())
                if error_count >= self.config.live.max_api_errors:
                    self.logger.critical("Max error threshold exceeded. Shutting down.")
                    break
                backoff = min(5 * (2 ** (error_count - 1)), 60)
                self.logger.info(f"Backing off for {backoff}s")
                time.sleep(backoff)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live Trading V2")
    parser.add_argument("--model-dir", type=str, default=f"models_v2/{CONF.data.symbol}/rank_1")
    parser.add_argument("--symbol", type=str, default=CONF.data.symbol)
    parser.add_argument("--data-dir", type=str, default=str(CONF.data.data_dir))
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW_BARS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--trade-poll-sec", type=float, default=DEFAULT_TRADE_POLL_SEC)
    parser.add_argument("--ob-poll-sec", type=float, default=DEFAULT_OB_POLL_SEC)
    parser.add_argument("--reconcile-sec", type=float, default=DEFAULT_RECONCILE_SEC)
    parser.add_argument("--heartbeat-sec", type=float, default=DEFAULT_HEARTBEAT_SEC)
    parser.add_argument("--instr-refresh-sec", type=float, default=DEFAULT_INSTR_REFRESH_SEC)
    parser.add_argument("--data-lag-warn-sec", type=float, default=DEFAULT_DATA_LAG_WARN_SEC)
    parser.add_argument("--data-lag-error-sec", type=float, default=DEFAULT_DATA_LAG_ERROR_SEC)
    parser.add_argument("--max-leverage", type=float, default=DEFAULT_MAX_LEVERAGE)
    parser.add_argument("--drift-window", type=int, default=DEFAULT_DRIFT_WINDOW)
    parser.add_argument("--drift-z", type=float, default=DEFAULT_DRIFT_Z)
    parser.add_argument(
        "--drift-keys",
        type=str,
        default="atr,rsi,vol_z,ob_spread_bps,ob_imbalance_mean,taker_buy_ratio,pred_long,pred_short",
        help="Comma-separated drift metrics to track",
    )
    args = parser.parse_args()
    args.drift_keys = [k.strip() for k in args.drift_keys.split(",") if k.strip()]
    return args


if __name__ == "__main__":
    cli_args = parse_args()
    trader = LiveTradingV2(cli_args)
    trader.run()
