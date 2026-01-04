
"""
Live Trading V2 - Robust execution engine.
"""

import argparse
import json
import logging
import math
import os
import time
import threading
import traceback
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
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

try:
    from pybit.unified_trading import WebSocket as BybitWebSocket
except Exception as exc:
    BybitWebSocket = None
    WS_IMPORT_ERROR = exc

STATE_VERSION = 1
DEFAULT_WINDOW_BARS = 10000
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
DEFAULT_MIN_OB_DENSITY_PCT = 50.0
DEFAULT_MIN_TRADE_BARS = 24
DEFAULT_WS_TRADE_QUEUE_MAX = 5000
DEFAULT_WS_OB_QUEUE_MAX = 200
DEFAULT_WS_PRIVATE_QUEUE_MAX = 1000
DEFAULT_WS_PRIVATE_STALE_SEC = 120.0
DEFAULT_WS_PRIVATE_REST_SEC = 300.0
DEFAULT_FEATURE_Z_WINDOW = 200
DEFAULT_FEATURE_Z_THRESHOLD = 3.0
DEFAULT_MODEL_DIR = f"models_v9/{CONF.data.symbol}/{CONF.data.symbol}/rank_1"
DEFAULT_BOOTSTRAP_KLINES_DAYS = 0.0

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
    "ob_bid_slope_mean",
    "ob_ask_slope_mean",
    "ob_bid_integrity_mean",
    "ob_ask_integrity_mean",
]


# ---------------------------
# Utility helpers
# ---------------------------

def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def safe_log_message(value: object) -> str:
    text = str(value)
    try:
        text.encode("ascii")
        return text
    except UnicodeEncodeError:
        return text.encode("ascii", "backslashreplace").decode("ascii")


def safe_int(value, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def safe_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


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


def round_up(value: float, step: float) -> float:
    if step <= 0:
        return value
    return math.ceil(value / step) * step


def atomic_write_json(path: Path, data: Dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    tmp_path.replace(path)


def append_jsonl(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        json.dump(payload, f, default=str)
        f.write("\n")


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
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "realized_pnl": 0.0,
                "equity_start": None,
                "last_cum_realized": None,
            },
            "metrics": {
                "api_errors": 0,
                "last_error": None,
                "last_ok_ts": None,
                "last_closed_pnl_time": 0,
                "last_exec_time": 0,
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
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
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
    min_notional: float = 0.0


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

    def reset(self) -> None:
        self.avg_ms = None
        self.max_ms = 0.0


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
# Health & sanity checks
# ---------------------------

class HealthMonitor:
    def __init__(self, window: int = 20):
        self.window = window
        self.outcomes: List[int] = []
        self.confidences: List[float] = []

    def record_trade(self, pnl: float) -> None:
        self.outcomes.append(1 if pnl > 0 else 0)
        if len(self.outcomes) > self.window:
            self.outcomes.pop(0)

    def record_prediction(self, pred_val: float) -> None:
        self.confidences.append(pred_val)
        if len(self.confidences) > 12:
            self.confidences.pop(0)

    def check_sentiment(self) -> str:
        if not self.confidences:
            return "CALIBRATING"
        avg_conf = sum(self.confidences) / len(self.confidences)
        state = "NEUTRAL"
        if avg_conf > 0.75:
            state = "PINNED BULLISH"
        elif avg_conf < 0.25:
            state = "PINNED BEARISH"
        return f"{state} | 1H Avg: {avg_conf:.3f}"

    def check_regime(self, df: pd.DataFrame) -> str:
        if len(df) < 50:
            return "CALIBRATING"
        col = "ob_bid_slope_mean"
        if col not in df.columns:
            return "DATA MISSING"
        valid = df[df[col] > 0]
        if len(valid) < 24:
            return "COLLECTING BASELINE"
        baseline_mean = valid[col].tail(288).mean()
        baseline_std = valid[col].tail(288).std()
        current_mean = valid[col].tail(12).mean()
        if baseline_std == 0:
            return "STABLE (Zero Vol)"
        z_drift = (current_mean - baseline_mean) / (baseline_std + 1e-9)
        status = "STABLE"
        if abs(z_drift) > 3.0:
            status = "CRITICAL DRIFT"
        elif abs(z_drift) > 2.0:
            status = "WARNING"
        return f"{status} | Drift Z: {z_drift:.2f}"

    def check_health(self) -> str:
        if len(self.outcomes) < 5:
            return "CALIBRATING"
        wr = sum(self.outcomes) / len(self.outcomes)
        status = "HEALTHY"
        if wr < 0.40:
            status = "CRITICAL"
        elif wr < 0.50:
            status = "WARNING"
        return f"{status} | Rolling WR: {wr:.1%}"


class SanityCheck:
    def check(self, row: pd.Series) -> List[str]:
        warnings: List[str] = []
        z_cols = [c for c in row.index if c.endswith("_z")]
        for c in z_cols:
            val = safe_float(row.get(c))
            if abs(val) > 10.0:
                warnings.append(f"EXTREME Z-SCORE: {c}={val:.2f}")

        vwap_cols = [c for c in row.index if "vwap" in c and "dist" in c]
        for c in vwap_cols:
            val = safe_float(row.get(c))
            if abs(val) > 20.0:
                warnings.append(f"VWAP DISLOCATION: {c}={val:.2f}")

        if "taker_buy_ratio" in row.index and safe_float(row.get("taker_buy_ratio")) == 0.5:
            warnings.append("SYNTHETIC DATA: taker_buy_ratio=0.5")

        if "ob_bid_elasticity" in row.index and safe_float(row.get("ob_bid_elasticity")) == 0.0:
            warnings.append("MISSING ORDERBOOK: ob_bid_elasticity=0.0")

        return warnings


class FeatureBaseline:
    def __init__(self, keys: List[str], window: int):
        self.keys = keys
        self.windows = {key: RollingWindow(window) for key in keys}

    def update(self, row: pd.Series) -> Dict[str, float]:
        zscores: Dict[str, float] = {}
        for key in self.keys:
            val = safe_float(row.get(key))
            window = self.windows.get(key)
            if window is None:
                continue
            mean, std = window.stats()
            window.add(val)
            if mean is None or std is None or std == 0:
                continue
            zscores[key] = (val - mean) / std
        return zscores


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

        bids_slice = bids[: self.depth_levels]
        asks_slice = asks[: self.depth_levels]
        bid_depth = sum(safe_float(b[1]) for b in bids_slice)
        ask_depth = sum(safe_float(a[1]) for a in asks_slice)

        spread = ba_price - bb_price
        mid = (ba_price + bb_price) / 2.0
        micro = (ba_price * bb_size + bb_price * ba_size) / (bb_size + ba_size + 1e-9)
        micro_dev = micro - mid
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-9)
        bid_slope = 0.0
        if bid_depth > 0 and len(bids_slice) > 1:
            deepest_bid = safe_float(bids_slice[-1][0])
            bid_slope = (bb_price - deepest_bid) / bid_depth
        ask_slope = 0.0
        if ask_depth > 0 and len(asks_slice) > 1:
            deepest_ask = safe_float(asks_slice[-1][0])
            ask_slope = (deepest_ask - ba_price) / ask_depth
        bid_integrity = bb_size / bid_depth if bid_depth > 0 else 0.0
        ask_integrity = ba_size / ask_depth if ask_depth > 0 else 0.0

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
                "bid_slope_sum": 0.0,
                "ask_slope_sum": 0.0,
                "bid_integrity_sum": 0.0,
                "ask_integrity_sum": 0.0,
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
        bucket["bid_slope_sum"] += bid_slope
        bucket["ask_slope_sum"] += ask_slope
        bucket["bid_integrity_sum"] += bid_integrity
        bucket["ask_integrity_sum"] += ask_integrity

    def finalize(self, bar_time: int) -> Optional[Dict]:
        bucket = self.buckets.pop(bar_time, None)
        if not bucket or bucket["count"] == 0:
            return None
        count = bucket["count"]
        micro_mean = bucket["micro_sum"] / count
        if count > 1:
            micro_var = (bucket["micro_sq_sum"] - (count * micro_mean * micro_mean)) / (count - 1)
        else:
            micro_var = 0.0
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
            "ob_bid_slope_mean": bucket["bid_slope_sum"] / count,
            "ob_ask_slope_mean": bucket["ask_slope_sum"] / count,
            "ob_bid_integrity_mean": bucket["bid_integrity_sum"] / count,
            "ob_ask_integrity_mean": bucket["ask_integrity_sum"] / count,
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

    def load_bars(self, bars_df: pd.DataFrame) -> None:
        if bars_df is None or bars_df.empty:
            return
        df = bars_df.copy().tail(self.window_size).reset_index(drop=True)
        self.bars = df
        if not self.bars.empty:
            last_row = self.bars.iloc[-1]
            self.trade_builder.last_price = safe_float(last_row.get("close"))
            for key in self.last_ob:
                self.last_ob[key] = safe_float(last_row.get(key))

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
            status_code = getattr(exc, "status_code", None)
            message = safe_log_message(exc)
            if name == "cancel_order" and status_code == 110001:
                self.logger.info(f"Cancel order not found: {message}")
                return None
            self.logger.error(f"API error in {name}: {message}")
            return None

    def place_limit_order(
        self,
        side: str,
        price: float,
        qty: float,
        tp: float = 0.0,
        sl: float = 0.0,
        reduce_only: bool = False,
    ) -> Dict:
        return self._call(
            "place_limit_order",
            self.exchange.place_limit_order,
            side, price, qty, tp, sl, reduce_only
        )


    def fetch_recent_trades(self, limit: int) -> pd.DataFrame:
        return self._call("fetch_recent_trades", self.exchange.fetch_recent_trades, limit)

    def fetch_orderbook(self, limit: int) -> Dict:
        return self._call("fetch_orderbook", self.exchange.fetch_orderbook, limit)

    def fetch_kline(self, interval: str, limit: int, **kwargs) -> pd.DataFrame:
        return self._call("fetch_kline", self.exchange.fetch_kline, interval, limit, **kwargs)

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
            avg_price = safe_float(pos.get("avgPrice") or pos.get("entryPrice"))
            mark_price = safe_float(pos.get("markPrice"))
            unreal_pnl = safe_float(pos.get("unrealisedPnl"))
            cum_realized = safe_float(pos.get("cumRealisedPnl"))
            stop_loss = safe_float(pos.get("stopLoss"))
            take_profit = safe_float(pos.get("takeProfit"))
            tpsl_mode = pos.get("tpslMode") or pos.get("tpSlMode")
            created_time = safe_int(pos.get("createdTime"))
            updated_time = safe_int(pos.get("updatedTime"))
            return {
                "size": size,
                "side": side,
                "avg_price": avg_price,
                "mark_price": mark_price,
                "unrealized_pnl": unreal_pnl,
                "cum_realized_pnl": cum_realized,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "tpsl_mode": tpsl_mode,
                "created_time": created_time,
                "updated_time": updated_time,
            }
        return {}

    def get_instrument_info(self) -> Dict:
        return self._call("fetch_instrument_info", self.exchange.fetch_instrument_info)

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

    def fetch_closed_pnl(self, limit: int = 50) -> List[Dict]:
        return self._call("fetch_closed_pnl", self.exchange.fetch_closed_pnl, limit)


# ---------------------------
# Live trading engine
# ---------------------------

class LiveTradingV2:
    def __init__(self, args: argparse.Namespace):
        self.logger = setup_logger(args.log_level)
        self.config = CONF
        self.config.data.data_dir = Path(args.data_dir)
        self.config.data.symbol = args.symbol
        self.config.data.ob_levels = args.ob_levels
        self.testnet = args.testnet
        default_model_dir = f"models_v9/{self.config.data.symbol}/{self.config.data.symbol}/rank_1"
        if args.model_dir == DEFAULT_MODEL_DIR and args.model_dir != default_model_dir:
            self.logger.warning(f"Model dir default adjusted to {default_model_dir} for symbol {self.config.data.symbol}.")
            args.model_dir = default_model_dir
        self.bootstrap_klines_days = args.bootstrap_klines_days

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
        self.min_ob_density_pct = args.min_ob_density_pct
        self.min_trade_bars = args.min_trade_bars
        self.log_features = args.log_features
        self.log_feature_z = args.log_feature_z
        self.feature_z_threshold = args.feature_z_threshold
        self.feature_z_window = args.feature_z_window
        self.exchange_leverage = args.exchange_leverage
        self.use_ws_trades = args.use_ws_trades
        self.use_ws_ob = args.use_ws_ob
        self.use_ws_private = args.use_ws_private
        self.ws_private_topics = [
            t.strip().lower() for t in args.ws_private_topics.split(",") if t.strip()
        ]
        self.ws_trade_queue_max = args.ws_trade_queue_max
        self.ws_ob_queue_max = args.ws_ob_queue_max
        self.ws_private_queue_max = args.ws_private_queue_max
        self.ws_private_stale_sec = args.ws_private_stale_sec
        self.ws_private_rest_sec = args.ws_private_rest_sec
        self.log_open_orders_raw = args.log_open_orders_raw
        self.open_orders_log_path = Path(args.open_orders_log_path) if args.open_orders_log_path else None
        if self.open_orders_log_path is None:
            self.open_orders_log_path = Path(f"open_orders_{self.config.data.symbol}.jsonl")
        self.last_open_orders_sig: Optional[Tuple] = None
        self.keys_file = Path(args.keys_file).expanduser() if args.keys_file else None
        self.keys_profile = args.keys_profile

        self.dry_run = args.dry_run
        self.trade_enabled = True
        self.continuous_trade_bars = 0
        self.last_trade_bar_time: Optional[int] = None
        self.ws_trade_last_ts = 0.0
        self.ws_ob_last_ts = 0.0
        self.ws_trade_queue: Deque[Dict] = deque(maxlen=self.ws_trade_queue_max)
        self.ws_ob_queue: Deque[Dict] = deque(maxlen=self.ws_ob_queue_max)
        self.ws_lock = threading.Lock()
        self.ws = None
        self.ws_ob_book = {"b": {}, "a": {}}
        self.ws_ob_initialized = False
        self.ws_private = None
        self.ws_private_lock = threading.Lock()
        self.ws_private_order_queue: Deque[Dict] = deque(maxlen=self.ws_private_queue_max)
        self.ws_private_exec_queue: Deque[Dict] = deque(maxlen=self.ws_private_queue_max)
        self.ws_private_position_queue: Deque[Dict] = deque(maxlen=self.ws_private_queue_max)
        self.ws_private_wallet_queue: Deque[Dict] = deque(maxlen=self.ws_private_queue_max)
        self.ws_private_last_ts = 0.0
        self.ws_private_last_rest = 0.0
        self.ws_private_equity: Optional[float] = None
        self.ws_trade_last_ingest_ts = 0.0
        self.ws_ob_last_ingest_ts = 0.0
        self.api_key: Optional[str] = None
        self.api_secret: Optional[str] = None

        self.state = StateStore(Path(f"bot_state_{self.config.data.symbol}_v2.json"), self.config.data.symbol, self.logger)
        self.state.load()
        self.state.reset_daily_if_needed()

        self.exchange = self._init_exchange()
        if not self.dry_run:
            if not self.exchange.startup_check():
                raise RuntimeError("Startup checks failed.")
            if self.exchange_leverage:
                self.exchange.set_leverage(self.exchange_leverage)
        self.api = SafeExchange(self.exchange, self.logger)
        self.instrument = InstrumentInfo()

        self.feature_engine = FeatureEngine(self.config.features)
        self.model_dir = self._resolve_model_dir(args.model_dir, self.config.data.symbol)
        if Path(args.model_dir) != self.model_dir:
            self.logger.warning(f"Resolved model dir: {self.model_dir} (from {args.model_dir}).")
        self.model_long, self.model_short, self.model_features = self._load_model(self.model_dir)
        self._apply_model_params(self.model_dir)
        self.min_feature_bars = self._min_window_for_features(self.model_features)
        if self.window_size < self.min_feature_bars:
            self.logger.warning(
                f"Window size {self.window_size} < required {self.min_feature_bars}; expanding window."
            )
            self.window_size = self.min_feature_bars
        self.feature_baseline = FeatureBaseline(self.model_features, args.feature_z_window)

        self.bar_window = BarWindow(self.tf_seconds, self.config.data.ob_levels, self.window_size, self.logger)
        self.history_path = Path(f"data_history_{self.config.data.symbol}.csv")
        self._bootstrap_bars()
        self._load_history()

        self.drift_monitor = DriftMonitor(args.drift_keys, args.drift_window, args.drift_z)
        self.health = HealthMonitor()
        self.sanity = SanityCheck()

        self.last_trade_ids: Deque[str] = deque(maxlen=500)

        self._refresh_instrument_info(force=True)
        self._init_websockets()
        self._reconcile("startup")
        self._log_manifest(str(self.model_dir))

    def _init_exchange(self) -> ExchangeClient:
        if ExchangeClient is None:
            raise RuntimeError(f"exchange_client import failed: {EXCHANGE_IMPORT_ERROR}")

        key = None
        secret = None
        file_key, file_secret = self._load_keys_file()
        if file_key and file_secret:
            key = file_key
            secret = file_secret
        else:
            key = os.getenv("BYBIT_API_KEY")
            secret = os.getenv("BYBIT_API_SECRET")
        if not key or not secret:
            self.logger.warning("API keys missing. Switching to dry run mode.")
            self.dry_run = True
            key = "dummy"
            secret = "dummy"

        self.api_key = key
        self.api_secret = secret
        return ExchangeClient(key, secret, self.config.data.symbol, testnet=self.testnet)

    def _load_keys_file(self) -> Tuple[Optional[str], Optional[str]]:
        if not self.keys_file:
            return None, None
        path = self.keys_file
        if not path.exists():
            self.logger.warning(f"Keys file not found: {path}")
            return None, None
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as exc:
            self.logger.error(f"Failed to read keys file {path}: {exc}")
            return None, None

        profile = self.keys_profile or "default"
        entry = None
        if isinstance(data, dict):
            if isinstance(data.get("profiles"), dict):
                data = data["profiles"]
            entry = data.get(profile)
            if entry is None and profile != "default":
                entry = data.get("default")
                if entry is not None:
                    self.logger.warning(f"Keys profile '{profile}' not found; using 'default'.")
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get("name") == profile:
                    entry = item
                    break
            if entry is None and profile != "default":
                for item in data:
                    if isinstance(item, dict) and item.get("name") == "default":
                        entry = item
                        self.logger.warning(f"Keys profile '{profile}' not found; using 'default'.")
                        break
            if entry is None and len(data) == 1 and isinstance(data[0], dict):
                entry = data[0]
        else:
            self.logger.error(f"Keys file {path} has unsupported format.")
            return None, None

        if not isinstance(entry, dict):
            self.logger.warning(f"Keys profile '{profile}' missing in {path}.")
            return None, None

        key = entry.get("api_key") or entry.get("key")
        secret = entry.get("api_secret") or entry.get("secret")
        if not key or not secret:
            self.logger.warning(f"Keys profile '{profile}' missing api_key/api_secret in {path}.")
            return None, None

        self.logger.info(f"Loaded API keys from {path} profile '{profile}'.")
        return str(key), str(secret)

    def _resolve_model_dir(self, model_dir: str, symbol: str) -> Path:
        path = Path(model_dir)
        candidates = [path]
        if path.is_dir():
            if not (path / "model_long.pkl").exists():
                candidates.extend([
                    path / "rank_1",
                    path / symbol / "rank_1",
                    path / symbol / symbol / "rank_1",
                ])
        else:
            candidates.extend([
                path / symbol / "rank_1",
                path / symbol / symbol / "rank_1",
            ])

        seen = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            if (candidate / "model_long.pkl").exists() and (candidate / "features.pkl").exists():
                return candidate
        return path

    def _load_model(self, model_dir: Path) -> Tuple[object, object, List[str]]:
        path = Path(model_dir)
        if not path.exists():
            raise FileNotFoundError(f"Model path not found: {path}")
        model_long = joblib.load(path / "model_long.pkl")
        model_short = joblib.load(path / "model_short.pkl")
        features = joblib.load(path / "features.pkl")
        if not isinstance(features, list) or not features:
            raise ValueError("features.pkl must contain a non-empty list")
        return model_long, model_short, features

    def _apply_model_params(self, model_dir: Path) -> None:
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
        self._validate_model_metadata(params)

    def _validate_model_metadata(self, params: Dict) -> None:
        meta = params.get("meta") if isinstance(params.get("meta"), dict) else {}
        model_symbol = params.get("symbol") or meta.get("symbol")
        model_timeframe = params.get("timeframe") or meta.get("timeframe")
        model_ob_levels = params.get("ob_levels") or meta.get("ob_levels")
        missing_meta = []
        if not model_symbol:
            missing_meta.append("symbol")
        if not model_timeframe:
            missing_meta.append("timeframe")
        if not model_ob_levels:
            missing_meta.append("ob_levels")
        if missing_meta:
            self.logger.warning(
                "params.json missing metadata (" + ", ".join(missing_meta) + "). Parity checks limited."
            )

        if model_symbol and model_symbol != self.config.data.symbol:
            msg = (
                f"Model symbol '{model_symbol}' does not match live symbol '{self.config.data.symbol}'."
            )
            self.logger.error(msg)
            raise RuntimeError(msg)
        if model_timeframe and model_timeframe != self.config.features.base_timeframe:
            self.logger.warning(
                f"Model timeframe '{model_timeframe}' does not match live timeframe "
                f"'{self.config.features.base_timeframe}'. Using model timeframe."
            )
            self.config.features.base_timeframe = model_timeframe
            self.tf_seconds = timeframe_to_seconds(self.config.features.base_timeframe)
        if model_ob_levels:
            try:
                model_ob_levels = int(float(model_ob_levels))
            except Exception:
                model_ob_levels = None
        if model_ob_levels and model_ob_levels != self.config.data.ob_levels:
            self.logger.warning(
                f"Model ob_levels {model_ob_levels} does not match live ob_levels "
                f"{self.config.data.ob_levels}. Using model ob_levels."
            )
            self.config.data.ob_levels = model_ob_levels

    def _min_window_for_features(self, features: List[str]) -> int:
        required = 60
        feature_set = set(features)
        if feature_set.intersection({"taker_buy_z", "vol_z", "spread_z", "atr_z"}):
            required = max(required, 24)
        if "vwap_4h_dist" in feature_set:
            required = max(required, 48)
        if feature_set.intersection({"vwap_24h_dist", "vol_intraday", "atr_regime"}):
            required = max(required, 288)
        if feature_set.intersection({"vol_macro", "atr_macro"}):
            required = max(required, 288)
        return required

    def _bootstrap_bars(self) -> None:
        if self.history_path.exists() and self.history_path.stat().st_size > 0:
            self.logger.info("History file present; skipping bootstrap.")
            return

        bootstrapped = False
        if self.bootstrap_klines_days and self.bootstrap_klines_days > 0:
            bootstrapped = self._bootstrap_from_klines(self.bootstrap_klines_days)
        if not bootstrapped:
            self._bootstrap_from_trades()

    def _bootstrap_from_klines(self, days: float) -> bool:
        interval_map = {60: "1", 300: "5", 900: "15", 3600: "60", 14400: "240"}
        interval = interval_map.get(self.tf_seconds, "15")
        target_bars = int((days * 86400) / self.tf_seconds)
        if target_bars <= 0:
            return False
        limit = min(200, target_bars)
        batches: List[pd.DataFrame] = []
        total_rows = 0
        current_end: Optional[int] = None
        max_iters = max(1, int(target_bars / limit) + 5)

        for _ in range(max_iters):
            kwargs = {}
            if current_end is not None:
                kwargs["end"] = int(current_end)
            klines = self.api.fetch_kline(interval=interval, limit=limit, **kwargs)
            if not isinstance(klines, pd.DataFrame) or klines.empty:
                break
            batches.append(klines)
            total_rows += len(klines)
            oldest_ts = safe_float(klines["timestamp"].min())
            if oldest_ts <= 0:
                break
            next_end = int(oldest_ts * 1000) - 1
            if current_end is not None and next_end >= current_end:
                break
            current_end = next_end
            if total_rows >= target_bars:
                break
            time.sleep(0.1)

        if not batches:
            self.logger.warning("No klines available for bootstrap.")
            return False

        full = pd.concat(batches).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        full = full.tail(target_bars)
        full = full[full["close"] > 0].copy()
        if full.empty:
            self.logger.warning("Kline bootstrap returned empty data.")
            return False

        self.bar_window.bootstrap_from_klines(full)
        last_bar = self.bar_window.latest_bar_time()
        if last_bar is not None:
            self.state.state["last_trade_ts"] = max(
                safe_float(self.state.state.get("last_trade_ts"), 0.0),
                float(last_bar),
            )
        self.logger.info(
            f"Bootstrapped {len(self.bar_window.bars)} bars from klines ({days:.1f} days)."
        )
        return True

    def _bootstrap_from_trades(self) -> bool:
        trades_df = self.api.fetch_recent_trades(limit=1000)
        if not isinstance(trades_df, pd.DataFrame) or trades_df.empty:
            self.logger.warning("No recent trades available for bootstrap.")
            return False

        trades_df = trades_df.sort_values("timestamp")
        trades: List[Dict] = []
        for _, row in trades_df.iterrows():
            ts = safe_float(row.get("timestamp"))
            price = safe_float(row.get("price"))
            size = safe_float(row.get("size"))
            side = row.get("side") or ""
            if ts <= 0 or price <= 0 or size <= 0:
                continue
            trades.append({
                "timestamp": ts,
                "price": price,
                "size": size,
                "side": side,
            })

        if not trades:
            self.logger.warning("No valid trades found for bootstrap.")
            return False

        last_ts = trades[-1]["timestamp"]
        self.bar_window.ingest_trades(trades, last_ts)
        self.state.state["last_trade_ts"] = max(
            safe_float(self.state.state.get("last_trade_ts"), 0.0),
            last_ts,
        )
        self.logger.info(
            f"Bootstrapped {len(self.bar_window.bars)} bars from recent trades ({len(trades)} trades)."
        )
        return True

    def _load_history(self) -> None:
        if not self.history_path.exists():
            return
        try:
            df = pd.read_csv(self.history_path)
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            else:
                df = pd.read_csv(self.history_path, index_col=0, parse_dates=True).reset_index()
                df.rename(columns={df.columns[0]: "datetime"}, inplace=True)
            df.dropna(subset=["datetime"], inplace=True)
            if "bar_time" not in df.columns:
                df["bar_time"] = (df["datetime"].astype("int64") // 1_000_000_000).astype(int)
                df["bar_time"] = df["bar_time"].apply(lambda ts: bar_time_from_ts(ts, self.tf_seconds))

            for col in BAR_COLUMNS:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[BAR_COLUMNS]
            df = df.drop_duplicates(subset=["bar_time"], keep="last").sort_values("bar_time")

            base = self.bar_window.bars.copy()
            if not base.empty:
                for col in BAR_COLUMNS:
                    if col not in base.columns:
                        base[col] = 0.0
                merged = df.set_index("bar_time").combine_first(base.set_index("bar_time"))
                merged = merged.reset_index().sort_values("bar_time")
                merged["datetime"] = pd.to_datetime(merged["bar_time"], unit="s")
                merged = merged[BAR_COLUMNS]
            else:
                merged = df

            self.bar_window.load_bars(merged)
            last_bar = self.bar_window.latest_bar_time()
            if last_bar is not None:
                self.state.state["last_bar_time"] = last_bar
                last_processed = self.state.state.get("last_processed_bar_time")
                if not last_processed or last_processed < last_bar:
                    self.state.state["last_processed_bar_time"] = last_bar
                last_trade_ts = safe_float(self.state.state.get("last_trade_ts"), 0.0)
                if last_trade_ts < last_bar:
                    self.state.state["last_trade_ts"] = float(last_bar)
            self.logger.info(f"Loaded {len(self.bar_window.bars)} bars from history.")
        except Exception as exc:
            self.logger.error(f"Failed to load history: {exc}")

    def _save_history(self) -> None:
        if self.bar_window.bars.empty:
            return
        try:
            save_df = self.bar_window.bars.tail(self.window_size).copy()
            save_df.to_csv(self.history_path, index=False)
        except Exception as exc:
            self.logger.error(f"Failed to save history: {exc}")

    def _data_health(self) -> str:
        bars = self.bar_window.bars
        if bars.empty:
            return "Data: EMPTY"
        trade_bars = bars[bars["trade_count"] > 0]
        n = len(trade_bars)
        macro_target = max(self.min_feature_bars, 1)
        macro_pct = min(100.0, (n / macro_target) * 100)
        window = min(24, n)
        if window == 0:
            ob_density = 0.0
        else:
            recent = trade_bars.tail(window)
            ob_density = (recent["ob_spread_mean"] > 0).sum() / window * 100
        return f"Bars: {n} | Macro: {macro_pct:.0f}% | OB-Density(2h): {ob_density:.0f}% | TradeCont: {self.continuous_trade_bars}"

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
            min_notional=safe_float(info.get("min_notional")),
        )

    def _select_ws_ob_depth(self, levels: int) -> int:
        allowed = [1, 50, 200, 500]
        for depth in allowed:
            if depth >= levels:
                if depth != levels:
                    self.logger.warning(f"WS depth {depth} used for ob_levels={levels}.")
                return depth
        max_depth = allowed[-1]
        self.logger.warning(f"WS depth {max_depth} used for ob_levels={levels} (max supported).")
        return max_depth

    def _init_websockets(self) -> None:
        if self.use_ws_trades or self.use_ws_ob:
            if BybitWebSocket is None:
                self.logger.warning(f"WebSocket import failed: {WS_IMPORT_ERROR}")
                self.use_ws_trades = False
                self.use_ws_ob = False
            else:
                try:
                    self.ws = BybitWebSocket(testnet=self.testnet, channel_type="linear")
                    if self.use_ws_trades:
                        self.ws.trade_stream(self.config.data.symbol, self._on_ws_trade)
                    if self.use_ws_ob:
                        depth = self._select_ws_ob_depth(self.config.data.ob_levels)
                        self.ws.orderbook_stream(depth, self.config.data.symbol, self._on_ws_orderbook)
                    self.logger.info(f"WebSocket active: trades={self.use_ws_trades} orderbook={self.use_ws_ob}")
                except Exception as exc:
                    self.logger.error(f"Failed to start WebSocket: {exc}")
                    self.use_ws_trades = False
                    self.use_ws_ob = False
                    self.ws = None
        self._init_private_websocket()

    def _init_private_websocket(self) -> None:
        if not self.use_ws_private:
            return
        if BybitWebSocket is None:
            self.logger.warning(f"WebSocket import failed: {WS_IMPORT_ERROR}")
            self.use_ws_private = False
            return
        if not self.api_key or not self.api_secret or self.api_key == "dummy":
            self.logger.warning("Private WS disabled: API keys missing.")
            self.use_ws_private = False
            return
        try:
            self.ws_private = BybitWebSocket(
                testnet=self.testnet,
                channel_type="private",
                api_key=self.api_key,
                api_secret=self.api_secret,
            )
            topics = set(self.ws_private_topics)
            if "order" in topics:
                self.ws_private.order_stream(self._on_ws_private_order)
            if "execution" in topics:
                self.ws_private.execution_stream(self._on_ws_private_execution)
            if "position" in topics:
                self.ws_private.position_stream(self._on_ws_private_position)
            if "wallet" in topics:
                self.ws_private.wallet_stream(self._on_ws_private_wallet)
            self.logger.info(f"Private WebSocket active: {','.join(sorted(topics))}")
        except Exception as exc:
            self.logger.error(f"Failed to start private WebSocket: {exc}")
            self.use_ws_private = False
            self.ws_private = None

    def _on_ws_trade(self, message: Dict) -> None:
        data = message.get("data", [])
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            return
        for trade in data:
            ts_raw = safe_float(
                trade.get("T")
                or trade.get("ts")
                or trade.get("time")
                or trade.get("timestamp")
            )
            if ts_raw <= 0:
                continue
            ts_sec = ts_raw / 1000.0 if ts_raw > 1e10 else ts_raw
            price = safe_float(trade.get("p") or trade.get("price"))
            size = safe_float(trade.get("v") or trade.get("size"))
            side = trade.get("S") or trade.get("side") or ""
            trade_id = str(
                trade.get("i")
                or trade.get("execId")
                or trade.get("tradeId")
                or trade.get("id")
                or ""
            )
            if price <= 0 or size <= 0:
                continue
            entry = {
                "timestamp": ts_sec,
                "price": price,
                "size": size,
                "side": side,
                "id": trade_id,
            }
        with self.ws_lock:
            self.ws_trade_queue.append(entry)
        self.ws_trade_last_ts = max(self.ws_trade_last_ts, ts_sec)
        self.ws_trade_last_ingest_ts = time.time()

    def _on_ws_orderbook(self, message: Dict) -> None:
        data = message.get("data")
        if not isinstance(data, dict):
            return
        msg_type = str(message.get("type") or "").lower()
        ts_ms = safe_int(data.get("ts") or message.get("ts") or message.get("cts"))
        bids = data.get("b") or []
        asks = data.get("a") or []
        if not bids and not asks:
            return

        if msg_type in ("", "snapshot"):
            book_bids: Dict[float, float] = {}
            book_asks: Dict[float, float] = {}
            for price_raw, size_raw in bids:
                price = safe_float(price_raw)
                size = safe_float(size_raw)
                if price > 0 and size > 0:
                    book_bids[price] = size
            for price_raw, size_raw in asks:
                price = safe_float(price_raw)
                size = safe_float(size_raw)
                if price > 0 and size > 0:
                    book_asks[price] = size
            self.ws_ob_book = {"b": book_bids, "a": book_asks}
            self.ws_ob_initialized = True
        elif msg_type == "delta":
            if not self.ws_ob_initialized:
                return
            book_bids = self.ws_ob_book.get("b", {})
            book_asks = self.ws_ob_book.get("a", {})
            for price_raw, size_raw in bids:
                price = safe_float(price_raw)
                size = safe_float(size_raw)
                if price <= 0:
                    continue
                if size <= 0:
                    book_bids.pop(price, None)
                else:
                    book_bids[price] = size
            for price_raw, size_raw in asks:
                price = safe_float(price_raw)
                size = safe_float(size_raw)
                if price <= 0:
                    continue
                if size <= 0:
                    book_asks.pop(price, None)
                else:
                    book_asks[price] = size
            self.ws_ob_book = {"b": book_bids, "a": book_asks}
        else:
            return

        book_bids = self.ws_ob_book.get("b", {})
        book_asks = self.ws_ob_book.get("a", {})
        max_levels = max(self.config.data.ob_levels, 1)
        bids_sorted = sorted(book_bids.items(), key=lambda x: -x[0])[:max_levels]
        asks_sorted = sorted(book_asks.items(), key=lambda x: x[0])[:max_levels]
        snapshot = {
            "b": [[p, s] for p, s in bids_sorted],
            "a": [[p, s] for p, s in asks_sorted],
            "ts": ts_ms,
        }

        with self.ws_lock:
            self.ws_ob_queue.append(snapshot)
        if ts_ms > 0:
            self.ws_ob_last_ts = ts_ms / 1000.0
            self.ws_ob_last_ingest_ts = time.time()

    def _drain_ws_trades(self) -> List[Dict]:
        with self.ws_lock:
            if not self.ws_trade_queue:
                return []
            trades = list(self.ws_trade_queue)
            self.ws_trade_queue.clear()

        trades.sort(key=lambda t: t.get("timestamp", 0))
        new_trades: List[Dict] = []
        last_ts = safe_float(self.state.state.get("last_trade_ts"), 0.0)
        for trade in trades:
            ts = safe_float(trade.get("timestamp"))
            trade_id = trade.get("id") or None
            if ts < last_ts:
                continue
            if trade_id and trade_id in self.last_trade_ids:
                continue
            new_trades.append(trade)
            if trade_id:
                self.last_trade_ids.append(trade_id)

        if new_trades:
            max_ts = max(t["timestamp"] for t in new_trades)
            self.state.state["last_trade_ts"] = max_ts
        return new_trades

    def _drain_ws_orderbook(self) -> Optional[Dict]:
        with self.ws_lock:
            if not self.ws_ob_queue:
                return None
            snapshot = self.ws_ob_queue[-1]
            self.ws_ob_queue.clear()
        return snapshot

    def _on_ws_private_order(self, message: Dict) -> None:
        with self.ws_private_lock:
            self.ws_private_order_queue.append(message)
        self.ws_private_last_ts = time.time()

    def _on_ws_private_execution(self, message: Dict) -> None:
        with self.ws_private_lock:
            self.ws_private_exec_queue.append(message)
        self.ws_private_last_ts = time.time()

    def _on_ws_private_position(self, message: Dict) -> None:
        with self.ws_private_lock:
            self.ws_private_position_queue.append(message)
        self.ws_private_last_ts = time.time()

    def _on_ws_private_wallet(self, message: Dict) -> None:
        with self.ws_private_lock:
            # Currently unused, but keep for future diagnostics.
            self.ws_private_wallet_queue.append(message)
        self.ws_private_last_ts = time.time()

    def _drain_ws_private(self) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
        with self.ws_private_lock:
            if not (
                self.ws_private_order_queue
                or self.ws_private_exec_queue
                or self.ws_private_position_queue
                or self.ws_private_wallet_queue
            ):
                return [], [], [], []
            orders = list(self.ws_private_order_queue)
            executions = list(self.ws_private_exec_queue)
            positions = list(self.ws_private_position_queue)
            wallets = list(self.ws_private_wallet_queue)
            self.ws_private_order_queue.clear()
            self.ws_private_exec_queue.clear()
            self.ws_private_position_queue.clear()
            self.ws_private_wallet_queue.clear()
        return orders, executions, positions, wallets

    def _iter_ws_data(self, message: Dict) -> List[Dict]:
        data = message.get("data", [])
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
        return []

    def _apply_ws_order_update(self, order: Dict, state_orders: Dict) -> bool:
        if order.get("symbol") != self.config.data.symbol:
            return False
        if self._is_protective_order(order):
            return False
        oid = order.get("orderId")
        if not oid:
            return False
        status = str(order.get("orderStatus") or "").lower()
        leaves_qty = safe_float(order.get("leavesQty"))
        create_type = str(order.get("createType") or "").lower()
        closed_statuses = {"cancelled", "filled", "deactivated", "rejected"}
        active_statuses = {"new", "partiallyfilled", "partially_filled", "untriggered", "triggered"}

        if status in closed_statuses or (leaves_qty == 0 and status):
            if oid in state_orders:
                state_orders.pop(oid, None)
                return True
            return False

        if status in active_statuses or leaves_qty > 0:
            created_ms = safe_int(order.get("createdTime"))
            created_bar = self.state.state.get("last_bar_time")
            if created_ms > 0:
                created_bar = bar_time_from_ts(created_ms / 1000.0, self.tf_seconds)
            price = safe_float(order.get("price"))
            qty = safe_float(order.get("qty"))
            tp = safe_float(order.get("takeProfit"))
            sl = safe_float(order.get("stopLoss"))
            existing = state_orders.get(oid, {})
            if tp <= 0:
                tp = safe_float(existing.get("tp"))
            if sl <= 0:
                sl = safe_float(existing.get("sl"))
            external = existing.get("external")
            if external is None:
                external = create_type not in {"createbyuser", ""}
            state_orders[oid] = {
                "side": order.get("side"),
                "price": price,
                "qty": qty,
                "tp": tp,
                "sl": sl,
                "created_bar_time": created_bar,
                "created_ts": utc_now_str(),
                "external": external,
            }
            return True

        return False

    def _apply_ws_position_update(self, pos: Dict) -> bool:
        if pos.get("symbol") != self.config.data.symbol:
            return False
        size = safe_float(pos.get("size"))
        if size <= 0:
            if self.state.state.get("position"):
                self.state.state["position"] = {}
                return True
            return False

        side = pos.get("side")
        entry_price = safe_float(pos.get("entryPrice") or pos.get("avgPrice"))
        mark_price = safe_float(pos.get("markPrice"))
        unreal_pnl = safe_float(pos.get("unrealisedPnl"))
        cum_realized = safe_float(pos.get("cumRealisedPnl"))
        stop_loss = safe_float(pos.get("stopLoss"))
        take_profit = safe_float(pos.get("takeProfit"))
        tpsl_mode = pos.get("tpslMode") or pos.get("tpSlMode")
        created_time = safe_int(pos.get("createdTime"))

        pos_state = self.state.state.get("position", {})
        if not pos_state:
            entry_bar_time = self.state.state.get("last_bar_time")
            if created_time > 0:
                entry_bar_time = bar_time_from_ts(created_time / 1000.0, self.tf_seconds)
            pos_state = {
                "side": side,
                "size": size,
                "entry_price": entry_price,
                "entry_bar_time": entry_bar_time,
                "created_ts": utc_now_str(),
                "tp_sl_set": False,
                "entry_ts_ms": created_time if created_time > 0 else int(time.time() * 1000),
            }

        pos_state.update({
            "side": side,
            "size": size,
            "entry_price": entry_price,
            "mark_price": mark_price,
            "unrealized_pnl": unreal_pnl,
            "cum_realized_pnl": cum_realized,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "tpsl_mode": tpsl_mode,
            "entry_ts_ms": pos_state.get("entry_ts_ms"),
        })
        if stop_loss > 0 or take_profit > 0:
            pos_state["tp_sl_set"] = True
        self.state.state["position"] = pos_state
        return True

    def _apply_ws_execution_update(self, execution: Dict) -> bool:
        if execution.get("symbol") != self.config.data.symbol:
            return False
        exec_time = safe_int(execution.get("execTime"))
        if exec_time <= 0:
            return False
        metrics = self.state.state.get("metrics", {})
        metrics["last_exec_time"] = max(exec_time, safe_int(metrics.get("last_exec_time")))
        self.state.state["metrics"] = metrics
        return True

    def _apply_ws_wallet_update(self, wallet: Dict) -> bool:
        total_equity = safe_float(wallet.get("totalEquity"))
        if total_equity <= 0 and isinstance(wallet.get("coin"), list) and wallet["coin"]:
            total_equity = safe_float(wallet["coin"][0].get("equity"))
        if total_equity <= 0:
            return False
        self.ws_private_equity = total_equity
        return True

    def _process_ws_private(self) -> None:
        if not self.use_ws_private:
            return
        orders, executions, positions, _wallets = self._drain_ws_private()
        if not (orders or executions or positions):
            return
        state_orders = self.state.state.get("active_orders", {})
        updated = False
        for msg in orders:
            for item in self._iter_ws_data(msg):
                updated |= self._apply_ws_order_update(item, state_orders)
        for msg in positions:
            for item in self._iter_ws_data(msg):
                updated |= self._apply_ws_position_update(item)
        for msg in executions:
            for item in self._iter_ws_data(msg):
                updated |= self._apply_ws_execution_update(item)
        for msg in _wallets:
            for item in self._iter_ws_data(msg):
                if self._apply_ws_wallet_update(item):
                    updated = True
        if updated:
            self.state.state["active_orders"] = state_orders

    def _log_manifest(self, model_dir: str) -> None:
        self.logger.info("=" * 60)
        self.logger.info("LIVE TRADING V2 MANIFEST")
        self.logger.info("=" * 60)
        self.logger.info(f"Symbol: {self.config.data.symbol}")
        self.logger.info(f"Timeframe: {self.config.features.base_timeframe}")
        self.logger.info(f"Model Dir: {model_dir}")
        self.logger.info(f"Testnet: {self.testnet}")
        self.logger.info(f"Feature Count: {len(self.model_features)}")
        self.logger.info(f"Window Bars: {self.window_size} | Min Feature Bars: {self.min_feature_bars}")
        self.logger.info(f"Threshold: {self.config.model.model_threshold:.3f}")
        self.logger.info(f"Limit Offset ATR: {self.config.strategy.base_limit_offset_atr:.3f}")
        self.logger.info(f"TP ATR: {self.config.strategy.take_profit_atr:.3f}")
        self.logger.info(f"SL ATR: {self.config.strategy.stop_loss_atr:.3f}")
        self.logger.info(f"Order Timeout Bars: {self.config.strategy.time_limit_bars}")
        self.logger.info(f"Max Holding Bars: {self.config.strategy.max_holding_bars}")
        self.logger.info(f"Risk Per Trade: {self.config.strategy.risk_per_trade:.3f}")
        self.logger.info(f"OB Levels: {self.config.data.ob_levels} | Min OB Density: {self.min_ob_density_pct:.0f}% | Trade Warmup: {self.min_trade_bars}")
        self.logger.info(f"Bootstrap Klines Days: {self.bootstrap_klines_days}")
        self.logger.info(f"Dry Run: {self.dry_run}")
        self.logger.info(f"Exchange Leverage: {self.exchange_leverage if self.exchange_leverage else 'unchanged'}")
        if self.keys_file:
            self.logger.info(f"Keys File: {self.keys_file} | Profile: {self.keys_profile}")
        else:
            self.logger.info("Keys File: env")
        if self.log_open_orders_raw:
            self.logger.info(f"Open Orders Raw Log: {self.open_orders_log_path}")
        self.logger.info(f"WS Trades: {self.use_ws_trades} | WS OB: {self.use_ws_ob}")
        self.logger.info(f"WS Private: {self.use_ws_private} | Topics: {','.join(self.ws_private_topics) or 'none'}")
        if self.use_ws_private:
            self.logger.info(
                f"WS Private Stale: {self.ws_private_stale_sec:.0f}s | REST Fallback: {self.ws_private_rest_sec:.0f}s"
            )
        self.logger.info(
            f"Feature Z Window: {self.feature_z_window} | Z Threshold: {self.feature_z_threshold:.2f} | Log Z: {self.log_feature_z}"
        )
        self.logger.info(
            f"Min Qty: {self.instrument.min_qty} | Qty Step: {self.instrument.qty_step} | "
            f"Tick Size: {self.instrument.tick_size} | Min Notional: {self.instrument.min_notional}"
        )
        self.logger.info(f"History File: {self.history_path}")
        self.logger.info("=" * 60)

    def _filter_new_trades(self, trades_df: pd.DataFrame) -> List[Dict]:
        if trades_df is None or trades_df.empty:
            return []
        trades_df = trades_df.sort_values("timestamp")
        new_trades = []
        last_ts = safe_float(self.state.state.get("last_trade_ts"), 0.0)
        for _, row in trades_df.iterrows():
            ts = safe_float(row.get("timestamp"))
            trade_id = row.get("id") if "id" in row else None
            if trade_id is None or trade_id == "" or (isinstance(trade_id, float) and pd.isna(trade_id)):
                trade_id = row.get("execId") if "execId" in row else None
            if trade_id is not None and trade_id != "":
                trade_id = str(trade_id)
            else:
                trade_id = None
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
        now = time.time()
        state_orders = self.state.state.get("active_orders", {})

        if self.use_ws_private and self.ws_private is not None:
            self._process_ws_private()
            ws_fresh = self.ws_private_last_ts > 0 and (now - self.ws_private_last_ts) <= self.ws_private_stale_sec
            rest_due = self.ws_private_rest_sec > 0 and (now - self.ws_private_last_rest) >= self.ws_private_rest_sec
            if ws_fresh and not rest_due:
                self._update_closed_pnl()
                self.state.save()
                self.logger.info(f"Reconcile complete ({reason}|ws). Open orders: {len(state_orders)}")
                return

        open_orders = self.api.get_open_orders()
        position = self.api.get_position_details()
        state_orders = self.state.state.get("active_orders", {})

        if self.log_open_orders_raw and isinstance(open_orders, list):
            sig = self._open_orders_signature(open_orders)
            if sig != self.last_open_orders_sig:
                payload = {
                    "ts": utc_now_str(),
                    "symbol": self.config.data.symbol,
                    "reason": reason,
                    "open_orders": open_orders,
                }
                append_jsonl(self.open_orders_log_path, payload)
                self.last_open_orders_sig = sig

        active_ids = set()
        protective_count = 0
        for o in open_orders:
            oid = o.get("orderId")
            if not oid:
                continue
            if self._is_protective_order(o):
                protective_count += 1
                continue
            active_ids.add(oid)
            if oid not in state_orders:
                created_ms = safe_int(o.get("createdTime"))
                created_bar = self.state.state.get("last_bar_time")
                if created_ms > 0:
                    created_bar = bar_time_from_ts(created_ms / 1000.0, self.tf_seconds)
                state_orders[oid] = {
                    "side": o.get("side"),
                    "price": safe_float(o.get("price")),
                    "qty": safe_float(o.get("qty")),
                    "created_bar_time": created_bar,
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
            stop_loss = safe_float(position.get("stop_loss"))
            take_profit = safe_float(position.get("take_profit"))
            tpsl_mode = position.get("tpsl_mode")
            created_time = safe_int(position.get("created_time"))
            pos_state = self.state.state.get("position", {})
            if not pos_state:
                self.logger.warning("External position detected. Taking ownership.")
                entry_bar_time = self.state.state.get("last_bar_time")
                if created_time > 0:
                    entry_bar_time = bar_time_from_ts(created_time / 1000.0, self.tf_seconds)
                pos_state = {
                    "side": side,
                    "size": pos_size,
                    "entry_price": entry_price,
                    "entry_bar_time": entry_bar_time,
                    "created_ts": utc_now_str(),
                    "tp_sl_set": False,
                    "entry_ts_ms": created_time if created_time > 0 else int(time.time() * 1000),
                }
            pos_state.update({
                "side": side,
                "size": pos_size,
                "entry_price": entry_price,
                "mark_price": mark_price,
                "unrealized_pnl": unreal_pnl,
                "cum_realized_pnl": cum_realized,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "tpsl_mode": tpsl_mode,
                "entry_ts_ms": pos_state.get("entry_ts_ms"),
            })
            if stop_loss > 0 or take_profit > 0:
                pos_state["tp_sl_set"] = True
            self.state.state["position"] = pos_state
        self._update_closed_pnl()

        self.state.save()
        if protective_count:
            self.logger.info(
                f"Reconcile complete ({reason}). Open orders: {len(state_orders)} | Protective: {protective_count}"
            )
        else:
            self.logger.info(f"Reconcile complete ({reason}). Open orders: {len(state_orders)}")
        if self.use_ws_private and self.ws_private is not None:
            self.ws_private_last_rest = now

    def _update_closed_pnl(self) -> None:
        records = self.api.fetch_closed_pnl(limit=50)
        if not records:
            return
        metrics = self.state.state.get("metrics", {})
        last_ts = safe_int(metrics.get("last_closed_pnl_time"), 0)
        new_records = [
            r for r in records
            if safe_int(r.get("createdTime")) > last_ts
        ]
        if not new_records:
            return
        new_records.sort(key=lambda r: safe_int(r.get("createdTime")))
        total_pnl = 0.0
        max_ts = last_ts
        for r in new_records:
            pnl = safe_float(r.get("closedPnl"))
            total_pnl += pnl
            max_ts = max(max_ts, safe_int(r.get("createdTime")))
            self.health.record_trade(pnl)

        daily = self.state.state.get("daily", {})
        daily["realized_pnl"] = safe_float(daily.get("realized_pnl")) + total_pnl
        self.state.state["daily"] = daily
        metrics["last_closed_pnl_time"] = max_ts
        self.state.state["metrics"] = metrics
        self.logger.info(f"Closed PnL update: {total_pnl:.4f} ({len(new_records)} fills)")

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

    def _is_protective_order(self, order: Dict) -> bool:
        order_type = str(order.get("orderType") or "").lower()
        order_filter = str(order.get("orderFilter") or "").lower()
        stop_order_type = str(order.get("stopOrderType") or "").lower()
        create_type = str(order.get("createType") or "").lower()

        if safe_bool(order.get("reduceOnly")) or safe_bool(order.get("closeOnTrigger")):
            return True
        if "tpsl" in order_filter:
            return True
        if order_filter in {"stoporder", "tpslorder"}:
            return True

        protective_types = {
            "takeprofit",
            "stoploss",
            "takeprofitlimit",
            "stoplosslimit",
            "takeprofitmarket",
            "stoplossmarket",
        }
        if order_type in protective_types:
            return True

        if "stop" in create_type or "takeprofit" in create_type or "trailing" in create_type:
            return True

        protective_stop_types = {"takeprofit", "stoploss", "trailingstop"}
        return stop_order_type in protective_stop_types

    def _open_orders_signature(self, orders: List[Dict]) -> Tuple:
        sig = []
        for o in orders:
            sig.append(
                (
                    str(o.get("orderId") or ""),
                    str(o.get("orderStatus") or ""),
                    str(o.get("orderType") or ""),
                    str(o.get("stopOrderType") or ""),
                    str(o.get("createType") or ""),
                    safe_bool(o.get("reduceOnly")),
                    safe_bool(o.get("closeOnTrigger")),
                    str(o.get("takeProfit") or ""),
                    str(o.get("stopLoss") or ""),
                    str(o.get("triggerPrice") or ""),
                )
            )
        return tuple(sorted(sig))

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

        qty_step = self.instrument.qty_step
        min_qty = self.instrument.min_qty
        min_notional = self.instrument.min_notional

        min_qty_req = 0.0
        if min_notional > 0 and price > 0:
            min_qty_req = max(min_qty_req, min_notional / price)
        if min_qty > 0:
            min_qty_req = max(min_qty_req, min_qty)

        qty = max(qty, min_qty_req)

        if max_qty > 0 and qty > max_qty:
            if max_qty >= min_qty_req:
                qty = max_qty
            else:
                self.logger.warning("Min size exceeds leverage cap; using min size.")
                qty = min_qty_req

        if qty_step > 0:
            qty = round_down(qty, qty_step)
            if qty < min_qty_req:
                qty = round_up(min_qty_req, qty_step)
            if max_qty > 0 and qty > max_qty:
                if max_qty >= min_qty_req:
                    qty = round_down(max_qty, qty_step)
                    if qty < min_qty_req:
                        qty = round_up(min_qty_req, qty_step)
                else:
                    self.logger.warning("Min size exceeds leverage cap after rounding; using min size.")
                    qty = round_up(min_qty_req, qty_step)

        # Final guard
        if qty <= 0:
            self.logger.warning("Qty quantized to 0; skipping order.")
            return None      	
        
        self.logger.info(f"Placing {side} limit: price={price:.6f} qty={qty:.6f}")
        if self.dry_run:
            return "dry_run"

        # Attaching TP and SL to the limit position right away
        tp_atr = self.config.strategy.take_profit_atr
        sl_atr = self.config.strategy.stop_loss_atr
        tick = self.instrument.tick_size

        def round_to_tick(x: float, tick: float, mode: str) -> float:
            if tick <= 0:
                return x
            if mode == "up":
                return math.ceil(x / tick) * tick
            return math.floor(x / tick) * tick

        if side == "Buy":
            tp = price + (atr * tp_atr)
            sl = price - (atr * sl_atr)
            # Widen slightly to reduce rejection risk from min-distance rules.
            tp = round_to_tick(tp, tick, "up")    # further away (higher)
            sl = round_to_tick(sl, tick, "down")  # further away (lower)
        else:
            tp = price - (atr * tp_atr)
            sl = price + (atr * sl_atr)
            tp = round_to_tick(tp, tick, "down")  # further away (lower)
            sl = round_to_tick(sl, tick, "up")    # further away (higher)

        resp = self.api.place_limit_order(side, price, qty, tp=tp, sl=sl, reduce_only=False)
        
        if resp is None:
            # Transport/exception path (ExchangeClient caught exception and returned None)
            self.logger.error("place_limit_order returned None (exception / transport failure).")
            return None
        
        ret_code = resp.get("retCode")
        if ret_code != 0:
            self.logger.error(f"Bybit rejected order: retCode={ret_code} retMsg={resp.get('retMsg')}")
            return None

        order_id = (resp.get("result") or {}).get("orderId")
        if not order_id:
            self.logger.error(f"Bybit response missing orderId: {resp}")
            return None

        self.state.state["active_orders"][order_id] = {
            "side": side,
            "price": price,
            "qty": qty,
            "tp": tp,
            "sl": sl,
            "created_bar_time": self.state.state.get("last_bar_time"),
            "created_ts": utc_now_str(),
            "external": False,
        }
        self.state.save()

        return order_id

    def _log_feature_vector(self, row: pd.Series) -> None:
        if not self.log_features:
            return
        msg = f"\n[FEATURES] Input Vector ({len(self.model_features)} features):\n"
        for i in range(0, len(self.model_features), 3):
            chunk = self.model_features[i:i + 3]
            line = " | ".join([f"{name}: {safe_float(row.get(name)):.6f}" for name in chunk])
            msg += f"  {line}\n"
        self.logger.info(msg)

    def _log_feature_zscores(self, zscores: Dict[str, float]) -> None:
        if not zscores:
            return
        if self.log_feature_z:
            msg = f"\n[FEATURE-Z] Rolling Z-Scores ({len(zscores)} features):\n"
            for i in range(0, len(self.model_features), 3):
                chunk = self.model_features[i:i + 3]
                line_parts = []
                for name in chunk:
                    if name in zscores:
                        line_parts.append(f"{name}: {zscores[name]:.2f}")
                    else:
                        line_parts.append(f"{name}: n/a")
                msg += f"  {' | '.join(line_parts)}\n"
            self.logger.info(msg)
            return

        alerts = {k: v for k, v in zscores.items() if abs(v) >= self.feature_z_threshold}
        if not alerts:
            return
        parts = [f"{k}={v:.2f}" for k, v in alerts.items()]
        self.logger.warning("[FEATURE-Z] " + ", ".join(parts))

    def _check_feature_coverage(self, df_feat: pd.DataFrame, bar_time: Optional[int] = None) -> Tuple[bool, List[str]]:
        missing = [f for f in self.model_features if f not in df_feat.columns]
        if missing:
            return False, missing
        if bar_time is not None and "bar_time" in df_feat.columns:
            row_match = df_feat[df_feat["bar_time"] == bar_time]
            if row_match.empty:
                return False, ["bar_time_missing"]
            last_row = row_match[self.model_features].iloc[-1]
        else:
            last_row = df_feat[self.model_features].iloc[-1]
        if last_row.isnull().any():
            missing_nan = last_row[last_row.isnull()].index.tolist()
            return False, missing_nan
        finite_mask = np.isfinite(last_row.astype(float))
        if not finite_mask.all():
            bad = last_row.index[~finite_mask].tolist()
            return False, bad
        return True, []

    def _compute_predictions(self, row: pd.Series) -> Tuple[float, float]:
        X = row[self.model_features].values.reshape(1, -1)
        pred_long = float(self.model_long.predict(X)[0])
        pred_short = float(self.model_short.predict(X)[0])
        return pred_long, pred_short

    def _process_bar(self, bar_time: int) -> None:
        if self.state.state.get("last_processed_bar_time") == bar_time:
            return
        self.state.state["last_processed_bar_time"] = bar_time
        self.state.state["last_bar_time"] = bar_time
        try:
            bars_df = self.bar_window.bars.copy()
            if bars_df.empty:
                return
            if len(bars_df) < self.min_feature_bars:
                self.logger.warning(f"Warmup: {len(bars_df)} bars < {self.min_feature_bars}.")
                return

            bar_rows = bars_df[bars_df["bar_time"] == bar_time]
            if bar_rows.empty:
                self.logger.warning(f"Bar time {bar_time} missing in window.")
                return
            raw_row = bar_rows.iloc[-1]
            trade_count = safe_int(raw_row.get("trade_count"))
            if trade_count > 0:
                if self.last_trade_bar_time is None or (bar_time - self.last_trade_bar_time) != self.tf_seconds:
                    self.continuous_trade_bars = 1
                else:
                    self.continuous_trade_bars += 1
                self.last_trade_bar_time = bar_time
            else:
                self.continuous_trade_bars = 0

            if trade_count == 0:
                self.logger.info(f"Bar {pd.to_datetime(bar_time, unit='s')} has no trades. Skipping.")
                return

            bars_df = bars_df[bars_df["trade_count"] > 0].copy()
            if bars_df.empty:
                self.logger.warning("No trade bars available after filtering.")
                return
            df_feat = self.feature_engine.calculate_features(bars_df)

            ok, missing = self._check_feature_coverage(df_feat, bar_time=bar_time)
            if not ok:
                self.trade_enabled = False
                self.logger.error(f"Feature mismatch or NaN. Missing: {missing}")
                return
            self.trade_enabled = True

            row_match = df_feat[df_feat["bar_time"] == bar_time]
            if row_match.empty:
                self.logger.warning(f"Feature row missing for bar_time={bar_time}.")
                return
            latest = row_match.iloc[-1]

            recent = bars_df.tail(min(24, len(bars_df)))
            ob_density = (recent["ob_spread_mean"] > 0).sum() / max(len(recent), 1) * 100
            if ob_density < self.min_ob_density_pct:
                self.logger.warning(f"OB density {ob_density:.0f}% below threshold {self.min_ob_density_pct:.0f}%.")
                return

            if self.continuous_trade_bars < self.min_trade_bars:
                self.logger.info(
                    f"Trade warmup: {self.continuous_trade_bars}/{self.min_trade_bars} continuous bars."
                )
                return

            self._log_feature_vector(latest)
            for warning in self.sanity.check(latest):
                self.logger.warning(f"[DATA INTEGRITY] {warning}")
            zscores = self.feature_baseline.update(latest)
            self._log_feature_zscores(zscores)

            pred_long, pred_short = self._compute_predictions(latest)
            threshold = self.config.model.model_threshold
            self.health.record_prediction(max(pred_long, pred_short))

            equity = 10000.0
            if not self.dry_run:
                if self.use_ws_private and self.ws_private_equity:
                    equity = self.ws_private_equity
                else:
                    bal = self.api.get_wallet_balance()
                    if isinstance(bal, dict):
                        equity = safe_float(bal.get("equity"), equity)
                if self.state.state["daily"]["equity_start"] is None and equity > 0:
                    self.state.state["daily"]["equity_start"] = equity

            spread = safe_float(latest.get("ob_spread_mean"))
            close = safe_float(latest.get("close"))
            spread_pct = (spread / close) if close > 0 else 0.0

            regime = self.health.check_regime(df_feat)
            sentiment = self.health.check_sentiment()
            self.logger.info(
                f"BAR {pd.to_datetime(bar_time, unit='s')} | close={close:.6f} atr={safe_float(latest.get('atr')):.6f} "
                f"spread={spread_pct:.4%} predL={pred_long:.3f} predS={pred_short:.3f} | "
                f"{self._data_health()} | Regime: {regime} | Sentiment: {sentiment} | Health: {self.health.check_health()}"
            )

            drift_alerts = self.drift_monitor.update(latest, {"pred_long": pred_long, "pred_short": pred_short})
            if drift_alerts:
                self.logger.warning("Drift alerts: " + ", ".join(drift_alerts))

            if spread_pct > self.config.live.max_spread_pct:
                self.logger.warning(f"Spread too wide ({spread_pct:.4%}). Skipping orders.")
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
                return

            if pos:
                if self.state.state.get("active_orders"):
                    self._cancel_active_orders("position_open")
                self._ensure_tp_sl(latest)
                return

            last_trade_ts = safe_float(self.state.state.get("last_trade_ts"), 0.0)
            if last_trade_ts > 0 and (time.time() - last_trade_ts) > self.data_lag_error:
                self.logger.error("Trade data lag exceeds error threshold. Skipping orders.")
                return

            if not self.trade_enabled:
                return

            orders = self.state.state.get("active_orders", {})
            has_buy = any(info.get("side") == "Buy" for info in orders.values())
            has_sell = any(info.get("side") == "Sell" for info in orders.values())

            if pred_long > threshold and not has_buy:
                self._place_order("Buy", latest, equity)
            if pred_short > threshold and not has_sell:
                self._place_order("Sell", latest, equity)
        finally:
            self.state.save()
            self._save_history()

    def _log_heartbeat(self) -> None:
        last_bar = self.state.state.get("last_bar_time")
        last_trade_ts = safe_float(self.state.state.get("last_trade_ts"))
        now = time.time()
        lag_trade = now - last_trade_ts if last_trade_ts > 0 else None
        lag_bar = now - last_bar if last_bar else None
        ob_ts = self.bar_window.ob_agg.last_snapshot_ts
        ob_lag = (time.time() * 1000 - ob_ts) / 1000.0 if ob_ts else None
        lag_trade_str = f"{lag_trade:.1f}s" if lag_trade is not None else "n/a"
        lag_bar_str = f"{lag_bar:.1f}s" if lag_bar is not None else "n/a"
        ob_lag_str = f"{ob_lag:.1f}s" if ob_lag is not None else "n/a"

        pos = self.state.state.get("position", {})
        pos_size = safe_float(pos.get("size")) if pos else 0.0
        pos_side = pos.get("side") if pos else "Flat"
        unreal = safe_float(pos.get("unrealized_pnl")) if pos else 0.0
        daily_pnl = safe_float(self.state.state.get("daily", {}).get("realized_pnl"))

        equity = None
        if not self.dry_run:
            if self.use_ws_private and self.ws_private_equity:
                equity = self.ws_private_equity
            else:
                bal = self.api.get_wallet_balance()
                if isinstance(bal, dict):
                    equity = safe_float(bal.get("equity"))

        latency_avg = self.api.latency.avg_ms or 0.0
        latency_max = self.api.latency.max_ms
        ws_trade_latency = None
        ws_ob_latency = None
        if self.use_ws_trades and self.ws_trade_last_ts > 0 and self.ws_trade_last_ingest_ts > 0:
            ws_trade_latency = (self.ws_trade_last_ingest_ts - self.ws_trade_last_ts) * 1000.0
            if ws_trade_latency < 0 or ws_trade_latency > 60000:
                ws_trade_latency = None
        if self.use_ws_ob and self.ws_ob_last_ts > 0 and self.ws_ob_last_ingest_ts > 0:
            ws_ob_latency = (self.ws_ob_last_ingest_ts - self.ws_ob_last_ts) * 1000.0
            if ws_ob_latency < 0 or ws_ob_latency > 60000:
                ws_ob_latency = None

        data_health = self._data_health()
        health = self.health.check_health()
        sentiment = self.health.check_sentiment()

        self.logger.info(
            "HEARTBEAT | "
            f"pos={pos_side} {pos_size:.4f} | "
            f"open_orders={len(self.state.state.get('active_orders', {}))} | "
            f"equity={equity if equity is not None else 'n/a'} | "
            f"daily_pnl={daily_pnl:.2f} | unreal={unreal:.2f} | "
            f"lag_trade={lag_trade_str} | lag_bar={lag_bar_str} | ob_lag={ob_lag_str} | "
            f"rest_latency={latency_avg:.1f}ms max={latency_max:.1f}ms | "
            f"{data_health} | Health: {health} | Sentiment: {sentiment}"
        )
        if ws_trade_latency is not None or ws_ob_latency is not None:
            ws_parts = []
            if ws_trade_latency is not None:
                ws_parts.append(f"trade={ws_trade_latency:.1f}ms")
            if ws_ob_latency is not None:
                ws_parts.append(f"ob={ws_ob_latency:.1f}ms")
            if ws_parts:
                self.logger.info("WS_LATENCY | " + " | ".join(ws_parts))

        if lag_trade is not None and lag_trade > self.data_lag_warn:
            self.logger.warning(f"Trade data lag: {lag_trade:.1f}s")
        if lag_trade is not None and lag_trade > self.data_lag_error:
            self.logger.error("Trade data lag exceeds error threshold.")
        if ob_lag is not None and ob_lag > self.data_lag_warn:
            self.logger.warning(f"Orderbook data lag: {ob_lag:.1f}s")

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
        last_time_sync = 0.0
        error_count = 0

        while True:
            try:
                now = time.time()

                if not self.dry_run and now - last_time_sync >= 3600.0:
                    drift = self.exchange.check_time_sync()
                    self.logger.info(f"Periodic time sync: drift={drift:.1f}ms")
                    last_time_sync = now

                if self.use_ws_private:
                    self._process_ws_private()

                if now - last_instr_refresh >= self.instr_refresh_sec:
                    self._refresh_instrument_info()
                    last_instr_refresh = now

                if now - last_ob_poll >= self.poll_ob_sec:
                    if self.use_ws_ob:
                        snapshot = self._drain_ws_orderbook()
                        if snapshot:
                            self.bar_window.ingest_orderbook(snapshot)
                        elif self.ws_ob_last_ts > 0 and (now - self.ws_ob_last_ts) > self.data_lag_warn:
                            self.logger.warning("WS orderbook stream lagging.")
                    else:
                        ob = self.api.fetch_orderbook(limit=self.config.data.ob_levels)
                        if isinstance(ob, dict) and ob:
                            self.bar_window.ingest_orderbook(ob)
                    last_ob_poll = now

                if now - last_trade_poll >= self.poll_trades_sec:
                    new_trades: List[Dict] = []
                    if self.use_ws_trades:
                        new_trades = self._drain_ws_trades()
                    else:
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
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--symbol", type=str, default=CONF.data.symbol)
    parser.add_argument("--data-dir", type=str, default=str(CONF.data.data_dir))
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW_BARS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--log-features", action="store_true", help="Log model feature vector each bar")
    parser.add_argument("--log-feature-z", action="store_true", help="Log per-feature z-scores each bar")
    parser.add_argument("--keys-file", type=str, default="", help="Path to JSON file with API keys")
    parser.add_argument("--keys-profile", type=str, default="default", help="Profile name inside keys file")
    parser.add_argument("--log-open-orders-raw", action="store_true", help="Log raw open orders payload when it changes")
    parser.add_argument("--open-orders-log-path", type=str, default="", help="Path for raw open orders JSONL log")
    parser.add_argument("--trade-poll-sec", type=float, default=DEFAULT_TRADE_POLL_SEC)
    parser.add_argument("--ob-poll-sec", type=float, default=DEFAULT_OB_POLL_SEC)
    parser.add_argument("--reconcile-sec", type=float, default=DEFAULT_RECONCILE_SEC)
    parser.add_argument("--heartbeat-sec", type=float, default=DEFAULT_HEARTBEAT_SEC)
    parser.add_argument("--instr-refresh-sec", type=float, default=DEFAULT_INSTR_REFRESH_SEC)
    parser.add_argument("--data-lag-warn-sec", type=float, default=DEFAULT_DATA_LAG_WARN_SEC)
    parser.add_argument("--data-lag-error-sec", type=float, default=DEFAULT_DATA_LAG_ERROR_SEC)
    parser.add_argument("--ob-levels", type=int, default=CONF.data.ob_levels)
    parser.add_argument("--min-ob-density-pct", type=float, default=DEFAULT_MIN_OB_DENSITY_PCT)
    parser.add_argument("--min-trade-bars", type=int, default=DEFAULT_MIN_TRADE_BARS)
    parser.add_argument("--max-leverage", type=float, default=DEFAULT_MAX_LEVERAGE)
    parser.add_argument("--exchange-leverage", type=int, default=0, help="Set exchange leverage (0 disables)")
    parser.add_argument("--testnet", action="store_true", help="Use Bybit testnet endpoints")
    parser.add_argument("--use-ws-trades", action="store_true", help="Use WebSocket trade stream")
    parser.add_argument("--use-ws-ob", action="store_true", help="Use WebSocket orderbook stream")
    parser.add_argument("--use-ws-private", action="store_true", help="Use private WebSocket streams")
    parser.add_argument(
        "--ws-private-topics",
        type=str,
        default="order,execution,position",
        help="Comma-separated private WS topics",
    )
    parser.add_argument("--ws-trade-queue-max", type=int, default=DEFAULT_WS_TRADE_QUEUE_MAX)
    parser.add_argument("--ws-ob-queue-max", type=int, default=DEFAULT_WS_OB_QUEUE_MAX)
    parser.add_argument("--ws-private-queue-max", type=int, default=DEFAULT_WS_PRIVATE_QUEUE_MAX)
    parser.add_argument("--ws-private-stale-sec", type=float, default=DEFAULT_WS_PRIVATE_STALE_SEC)
    parser.add_argument("--ws-private-rest-sec", type=float, default=DEFAULT_WS_PRIVATE_REST_SEC)
    parser.add_argument("--feature-z-window", type=int, default=DEFAULT_FEATURE_Z_WINDOW)
    parser.add_argument("--feature-z-threshold", type=float, default=DEFAULT_FEATURE_Z_THRESHOLD)
    parser.add_argument("--bootstrap-klines-days", type=float, default=DEFAULT_BOOTSTRAP_KLINES_DAYS)
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
