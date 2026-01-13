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

import argparse
import json
import logging
import math
import os
import re
import time
import threading
import traceback
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Set, Tuple

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
DEFAULT_MIN_TRADE_BARS = 0
DEFAULT_WS_TRADE_QUEUE_MAX = 5000
DEFAULT_WS_OB_QUEUE_MAX = 200
DEFAULT_WS_PRIVATE_QUEUE_MAX = 1000
DEFAULT_WS_PRIVATE_STALE_SEC = 120.0
DEFAULT_WS_PRIVATE_REST_SEC = 300.0
DEFAULT_FEATURE_Z_WINDOW = 200
DEFAULT_FEATURE_Z_THRESHOLD = 3.0
DEFAULT_ORDERLINK_PREFIX = "LTV2"
DEFAULT_POSTONLY_RETRY_MAX = 2
DEFAULT_REST_CONFIRM_ATTEMPTS = 3
DEFAULT_REST_CONFIRM_SLEEP_SEC = 0.2
DEFAULT_MODEL_DIR = "models"
DEFAULT_BOOTSTRAP_KLINES_DAYS = 30.0
DEFAULT_METRICS_LOG_PATH = ""
DEFAULT_SIGNAL_LOG_PATH = ""
DEFAULT_CONTINUITY_BARS = 24
DEFAULT_MAX_LAST_BAR_AGE_SEC = 0.0
DEFAULT_FAST_BOOTSTRAP = False
DEFAULT_POSITION_MODE = "hedge"
DEFAULT_TP_MAKER = False
DEFAULT_TP_MAKER_FALLBACK_SEC = 5.0
DEFAULT_BROKER_ID = ""

ERRCODE_RE = re.compile(r"ErrCode:\s*(\d+)")

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


def parse_err_code(message: str) -> Optional[int]:
    if not message:
        return None
    match = ERRCODE_RE.search(message)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def safe_int(value, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def normalize_ts_ms(value: object) -> int:
    ts = safe_int(value)
    if ts <= 0:
        return 0
    if ts < 1_000_000_000_000:
        return ts * 1000
    if ts > 1_000_000_000_000_000:
        return int(ts / 1000)
    return ts


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


def step_decimals(step: float) -> int:
    if step <= 0:
        return 0
    try:
        dec = Decimal(str(step)).normalize()
        return max(-dec.as_tuple().exponent, 0)
    except Exception:
        return 0


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
            "positions": {
                "long": {},
                "short": {},
            },
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
            "entry_intent": {
                "Buy": {},
                "Sell": {},
            },
            "virtual_positions": [],
            "internal_execs": [],
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
    max_qty: float = 0.0
    qty_step: float = 0.0
    tick_size: float = 0.0
    min_notional: float = 0.0
    max_notional: float = 0.0


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
        last_bar_time = self.current_bar_time
        if gap > 0:
            for i in range(gap):
                gap_time = self.current_bar_time + self.tf_seconds * (i + 1)
                completed.append(self._synthetic_bar(gap_time))
                last_bar_time = gap_time
        next_bar_time = last_bar_time + self.tf_seconds if last_bar_time is not None else None
        if next_bar_time is not None and now_ts >= next_bar_time:
            seed_price = self.last_price if self.last_price is not None else 0.0
            self._start_bar(next_bar_time, seed_price)
        else:
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

    def ingest_trades(
        self,
        trades: List[Dict],
        now_ts: float,
        on_close: Optional[Callable[[int], None]] = None,
    ) -> List[int]:
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
                if on_close:
                    on_close(bar["bar_time"])

        forced = self.trade_builder.force_close(now_ts)
        for bar in forced:
            self._append_bar(bar)
            closed_times.append(bar["bar_time"])
            if on_close:
                on_close(bar["bar_time"])

        return closed_times

    def _append_bar(self, bar: Dict) -> None:
        bar_time = safe_int(bar.get("bar_time"))
        if bar_time <= 0:
            return
        bar["bar_time"] = bar_time
        ob = self.ob_agg.finalize(bar_time)
        if ob:
            self.last_ob.update(ob)
        for key, val in self.last_ob.items():
            bar.setdefault(key, val)

        bar["datetime"] = pd.to_datetime(bar_time, unit="s")
        row = {col: bar.get(col, 0.0) for col in BAR_COLUMNS}

        if self.bars.empty:
            self.bars = pd.DataFrame([row])
            return

        last_bar_time = safe_int(self.bars.iloc[-1]["bar_time"])
        existing = self.bars["bar_time"] == bar_time
        if existing.any():
            self.bars.loc[existing, :] = row
        else:
            self.bars = pd.concat([self.bars, pd.DataFrame([row])], ignore_index=True)

        if existing.any() or bar_time < last_bar_time:
            self.bars = (
                self.bars.drop_duplicates(subset=["bar_time"], keep="last")
                .sort_values("bar_time")
                .reset_index(drop=True)
            )
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
        self.error_count = 0
        self.last_error: Optional[str] = None
        self.last_error_ts: float = 0.0

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
            if status_code is None:
                status_code = parse_err_code(message)
            if name == "cancel_order" and status_code == 110001:
                self.logger.info(f"Cancel order not found: {message}")
                return None
            self.error_count += 1
            self.last_error = message
            self.last_error_ts = time.time()
            self.logger.error(f"API error in {name}: {message}")
            return None

    def record_error(self, message: str, status_code: Optional[int] = None) -> None:
        if status_code is not None:
            message = f"{message} (ErrCode: {status_code})"
        self.error_count += 1
        self.last_error = safe_log_message(message)
        self.last_error_ts = time.time()

    def place_limit_order(
        self,
        side: str,
        price: float,
        qty: float,
        tp: float = 0.0,
        sl: float = 0.0,
        reduce_only: bool = False,
        position_idx: Optional[int] = None,
        order_link_id: Optional[str] = None,
    ) -> Dict:
        return self._call(
            "place_limit_order",
            self.exchange.place_limit_order,
            side, price, qty, tp, sl, reduce_only, position_idx, order_link_id
        )

    def place_market_order(
        self,
        side: str,
        qty: float,
        reduce_only: bool = False,
        position_idx: Optional[int] = None,
        order_link_id: Optional[str] = None,
    ) -> Dict:
        return self._call(
            "place_market_order",
            self.exchange.place_market_order,
            side, qty, reduce_only, position_idx, order_link_id,
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

    def get_open_order(self, order_id: Optional[str] = None, order_link_id: Optional[str] = None) -> Optional[Dict]:
        if not order_id and not order_link_id:
            return None

        def _call_open():
            params = {"category": "linear", "symbol": self.exchange.symbol}
            if order_id:
                params["orderId"] = order_id
            if order_link_id:
                params["orderLinkId"] = order_link_id
            return self.exchange.session.get_open_orders(**params)

        resp = self._call("get_open_order", _call_open)
        if not resp:
            return None
        orders = resp.get("result", {}).get("list", [])
        return orders[0] if orders else None

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
            created_time = normalize_ts_ms(pos.get("createdTime"))
            updated_time = normalize_ts_ms(pos.get("updatedTime"))
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

    def get_positions_details(self) -> List[Dict]:
        def _call_positions():
            return self.exchange.session.get_positions(category="linear", symbol=self.exchange.symbol)
        resp = self._call("get_positions", _call_positions)
        if not resp:
            return []
        positions = resp.get("result", {}).get("list", [])
        results: List[Dict] = []
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
            created_time = normalize_ts_ms(pos.get("createdTime"))
            updated_time = normalize_ts_ms(pos.get("updatedTime"))
            position_idx = safe_int(pos.get("positionIdx"))
            results.append({
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
                "position_idx": position_idx,
            })
        return results

    def get_instrument_info(self) -> Dict:
        return self._call("fetch_instrument_info", self.exchange.fetch_instrument_info)

    def cancel_order(self, order_id: Optional[str] = None, order_link_id: Optional[str] = None) -> bool:
        def _cancel():
            params = {"category": "linear", "symbol": self.exchange.symbol}
            if order_id:
                params["orderId"] = order_id
            if order_link_id:
                params["orderLinkId"] = order_link_id
            if getattr(self.exchange, "broker_id", ""):
                params["brokerId"] = self.exchange.broker_id
            return self.exchange.session.cancel_order(**params)
        resp = self._call("cancel_order", _cancel)
        return bool(resp)

    def cancel_all_orders(self) -> None:
        self._call("cancel_all_orders", self.exchange.cancel_all_orders)

    def market_close(self, side: str, qty: float, position_idx: Optional[int] = None) -> None:
        self._call("market_close", self.exchange.market_close, side, qty, position_idx)

    def set_tp_sl(self, side: str, qty: float, tp: float, sl: float, position_idx: Optional[int] = None) -> None:
        self._call("place_tp_sl", self.exchange.place_tp_sl, side, qty, tp, sl, position_idx)

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
        self.position_mode = (args.position_mode or DEFAULT_POSITION_MODE).lower()
        self.hedge_mode = self.position_mode == "hedge"
        self.bootstrap_klines_days = args.bootstrap_klines_days

        self.tf_seconds = timeframe_to_seconds(self.config.features.base_timeframe)
        self.window_size = args.window
        self.history_writer = args.history_writer
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
        self.continuity_bars = max(DEFAULT_CONTINUITY_BARS, args.continuity_bars)
        self.max_last_bar_age_sec = args.max_last_bar_age_sec
        if self.max_last_bar_age_sec <= 0:
            self.max_last_bar_age_sec = float(self.tf_seconds)
        self.fast_bootstrap = args.fast_bootstrap
        self.tp_maker = args.tp_maker
        self.tp_maker_fallback_sec = max(0.0, args.tp_maker_fallback_sec)
        self.log_features = args.log_features
        self.log_feature_z = args.log_feature_z
        self.feature_z_threshold = args.feature_z_threshold
        self.feature_z_window = args.feature_z_window
        self.orderlink_prefix = DEFAULT_ORDERLINK_PREFIX
        self.postonly_retry_max = DEFAULT_POSTONLY_RETRY_MAX
        self.rest_confirm_attempts = DEFAULT_REST_CONFIRM_ATTEMPTS
        self.rest_confirm_sleep_sec = DEFAULT_REST_CONFIRM_SLEEP_SEC
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
        self.last_open_orders_sig: Optional[Tuple] = None
        self.log_positions_raw = args.log_positions_raw
        self.positions_log_path = Path(args.positions_log_path) if args.positions_log_path else None
        self.positions_raw_logged = False
        self.keys_file = Path(args.keys_file).expanduser() if args.keys_file else None
        self.keys_profile = args.keys_profile
        self.broker_id = (args.broker_id or "").strip()

        self.dry_run = args.dry_run
        self.signal_only = args.signal_only
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
        self.ws_restart_cooldown_sec = max(self.data_lag_error, 30.0)
        self.ws_last_restart_ts = 0.0
        self.api_key: Optional[str] = None
        self.api_secret: Optional[str] = None
        self.start_time = time.time()
        self.start_time_utc = utc_now_str()
        self.runtime_error_count = 0
        self.last_runtime_error: Optional[str] = None
        self.last_runtime_error_ts: float = 0.0
        self.last_reconcile_info: Dict[str, object] = {}
        self.last_drift_alerts: List[str] = []
        self.last_drift_time: Optional[str] = None
        self.last_regime: Optional[str] = None
        self.last_sentiment: Optional[str] = None
        self.last_health: Optional[str] = None
        self.last_prediction: Dict[str, object] = {}
        self.last_direction: Dict[str, object] = {}
        self.direction_gate_signals = False
        self.direction_threshold_set = False
        self.aggressive_threshold_set = False
        self.direction_threshold_in_params = False
        self.aggressive_threshold_in_params = False
        self.last_signal: Dict[str, object] = {}
        self.last_feature_vector: Optional[Dict[str, object]] = None
        self.last_feature_row: Optional[pd.Series] = None

        self.state = StateStore(Path(f"bot_state_{self.config.data.symbol}_v2.json"), self.config.data.symbol, self.logger)
        self.state.load()
        self.state.reset_daily_if_needed()
        self._init_position_state()
        self._reset_entry_intents_on_start()
        self._ensure_virtual_positions_state()
        self._prune_internal_execs()

        self.exchange = self._init_exchange()
        if not self.dry_run:
            if not self.exchange.startup_check(self.position_mode):
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
        self.dir_model_long, self.dir_model_short = self._load_direction_models(self.model_dir)
        if (self.direction_threshold_set or self.aggressive_threshold_set) and (
            self.dir_model_long is None or self.dir_model_short is None
        ):
            missing = []
            if self.dir_model_long is None:
                missing.append("dir_model_long.pkl")
            if self.dir_model_short is None:
                missing.append("dir_model_short.pkl")
            msg = (
                "Directional thresholds are set but direction models are missing: "
                + ", ".join(missing)
            )
            self.logger.error(msg)
            raise RuntimeError(msg)
        self.min_feature_bars = self._min_window_for_features(self.model_features)
        if self.window_size < self.min_feature_bars:
            self.logger.warning(
                f"Window size {self.window_size} < required {self.min_feature_bars}; expanding window."
            )
            self.window_size = self.min_feature_bars
        self.feature_baseline = FeatureBaseline(self.model_features, args.feature_z_window)

        self.bar_window = BarWindow(self.tf_seconds, self.config.data.ob_levels, self.window_size, self.logger)
        self.metrics_key = self._resolve_log_key(args.metrics_log_path, self.config.data.symbol, self.model_dir)
        log_dir = self._resolve_log_dir(args.metrics_log_path)
        if args.metrics_log_path:
            self.metrics_path = Path(args.metrics_log_path)
        else:
            self.metrics_path = Path(f"live_metrics_{self.metrics_key}.jsonl")
        if args.signal_log_path:
            self.signal_log_path = Path(args.signal_log_path)
        else:
            self.signal_log_path = log_dir / f"signals_{self.metrics_key}.jsonl"
        if self.open_orders_log_path is None:
            self.open_orders_log_path = Path(f"open_orders_{self.metrics_key}.jsonl")
        if self.positions_log_path is None:
            self.positions_log_path = Path(f"positions_raw_{self.metrics_key}.jsonl")
        self.history_path = log_dir / f"data_history_{self.config.data.symbol}.csv"
        self._load_history()
        self._fill_history_gaps()
        self._bootstrap_bars()
        self._fill_history_gaps()
        self._fast_bootstrap_fill_gaps()
        self._seed_trade_builder_from_history()

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
        return ExchangeClient(
            key,
            secret,
            self.config.data.symbol,
            testnet=self.testnet,
            broker_id=self.broker_id,
        )

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

    def _model_log_key(self, model_dir: Optional[Path], symbol: str) -> str:
        if not model_dir:
            return symbol
        try:
            path = Path(model_dir)
        except Exception:
            return symbol
        name = path.name
        if not name:
            return symbol
        if name.lower().startswith("rank"):
            parent = path.parent
            if parent and parent.name:
                return parent.name
        return name

    def _resolve_log_key(
        self, metrics_log_path: Optional[str], symbol: str, model_dir: Optional[Path] = None
    ) -> str:
        fallback = self._model_log_key(model_dir, symbol)
        if not metrics_log_path:
            return fallback
        try:
            path = Path(metrics_log_path)
        except Exception:
            return fallback
        stem = path.stem
        if stem.startswith("live_metrics_"):
            stem = stem.replace("live_metrics_", "", 1)
        return stem or fallback

    def _resolve_log_dir(self, metrics_log_path: Optional[str]) -> Path:
        if not metrics_log_path:
            return Path(".")
        try:
            return Path(metrics_log_path).expanduser().parent
        except Exception:
            return Path(".")

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

    def _load_direction_models(self, model_dir: Path) -> Tuple[Optional[object], Optional[object]]:
        path = Path(model_dir)
        long_path = path / "dir_model_long.pkl"
        short_path = path / "dir_model_short.pkl"
        dir_long = None
        dir_short = None
        if long_path.exists():
            try:
                dir_long = joblib.load(long_path)
            except Exception as exc:
                self.logger.error(f"Failed to load dir_model_long.pkl: {exc}")
        if short_path.exists():
            try:
                dir_short = joblib.load(short_path)
            except Exception as exc:
                self.logger.error(f"Failed to load dir_model_short.pkl: {exc}")
        if dir_long or dir_short:
            self.logger.info("Direction models loaded.")
        else:
            self.logger.info("Direction models not found; using pass-through.")
        return dir_long, dir_short

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

        self.direction_threshold_in_params = "direction_threshold" in params
        self.aggressive_threshold_in_params = "aggressive_threshold" in params
        self.direction_threshold_set = self.direction_threshold_in_params
        self.aggressive_threshold_set = self.aggressive_threshold_in_params

        self.config.strategy.base_limit_offset_atr = safe_float(params.get("limit_offset_atr"), self.config.strategy.base_limit_offset_atr)
        self.config.strategy.take_profit_atr = safe_float(params.get("take_profit_atr"), self.config.strategy.take_profit_atr)
        self.config.strategy.stop_loss_atr = safe_float(params.get("stop_loss_atr"), self.config.strategy.stop_loss_atr)
        self.config.model.model_threshold = safe_float(params.get("model_threshold"), self.config.model.model_threshold)
        if self.direction_threshold_set:
            self.config.model.direction_threshold = safe_float(
                params.get("direction_threshold"),
                self.config.model.direction_threshold,
            )
        if self.aggressive_threshold_set:
            self.config.model.aggressive_threshold = safe_float(
                params.get("aggressive_threshold"),
                self.config.model.aggressive_threshold,
            )
        self.direction_gate_signals = self.direction_threshold_set and self.aggressive_threshold_set
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
        ready, reasons = self._history_ready()
        if ready:
            self.logger.info("History meets requirements; bootstrap not required.")
            return

        detail = ", ".join(reasons) if reasons else "unknown"
        self.logger.warning(f"History insufficient ({detail}); bootstrapping.")

        existing = self._normalize_bars(self.bar_window.bars.copy())
        required_bars = max(self.min_feature_bars, self.continuity_bars, 1)
        required_days = (required_bars * self.tf_seconds) / 86400.0
        target_days = max(required_days, self.bootstrap_klines_days or 0.0)

        bootstrapped = False
        if target_days > 0:
            bootstrapped = self._bootstrap_from_klines(target_days)
        if not bootstrapped:
            bootstrapped = self._bootstrap_from_trades()

        merged = existing
        if bootstrapped:
            merged = self._merge_history(existing, self.bar_window.bars.copy())
        if merged is None or merged.empty:
            seed = self._get_seed_price()
            if seed > 0 and self._bootstrap_synthetic(required_bars, seed):
                return
            self.logger.warning("Bootstrap failed; no usable data sources.")
            return

        seed = self._get_seed_price()
        merged = self._pad_history_to_min_bars(merged, required_bars, seed)
        if merged is None or merged.empty:
            self.logger.warning("Bootstrap failed; merged history unusable.")
            return
        self.bar_window.load_bars(merged)
        self._save_history()

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
        if self.bar_window.bars.empty:
            self.logger.warning("Trade bootstrap produced no bars; falling back.")
            return False
        self.state.state["last_trade_ts"] = max(
            safe_float(self.state.state.get("last_trade_ts"), 0.0),
            last_ts,
        )
        self.logger.info(
            f"Bootstrapped {len(self.bar_window.bars)} bars from recent trades ({len(trades)} trades)."
        )
        return True

    def _fill_history_gaps(self) -> None:
        bars_df = self._normalize_bars(self.bar_window.bars.copy())
        if bars_df is None or bars_df.empty:
            return

        start_bar = safe_int(bars_df["bar_time"].min())
        last_bar = safe_int(bars_df["bar_time"].max())
        end_bar = bar_time_from_ts(time.time() - self.tf_seconds, self.tf_seconds)
        if end_bar < last_bar:
            end_bar = last_bar
        if start_bar <= 0 or end_bar <= 0 or end_bar <= start_bar:
            return

        expected_times = list(range(start_bar, end_bar + self.tf_seconds, self.tf_seconds))
        existing_times = set(bars_df["bar_time"].astype(int).tolist())
        missing = [bt for bt in expected_times if bt not in existing_times]
        if not missing:
            self.logger.info("Gap fill: no missing bars detected.")
            return

        self.logger.info(
            f"Gap fill: {len(missing)} missing bars between {start_bar} and {end_bar}."
        )

        interval_map = {60: "1", 300: "5", 900: "15", 3600: "60", 14400: "240"}
        interval = interval_map.get(self.tf_seconds, "15")
        limit = 200
        start_ms = int(start_bar * 1000)
        end_ms = int((end_bar + self.tf_seconds) * 1000)
        target_bars = len(expected_times)
        max_iters = max(1, int(target_bars / limit) + 10)
        batches: List[pd.DataFrame] = []
        current_end = end_ms

        for _ in range(max_iters):
            klines = self.api.fetch_kline(interval=interval, limit=limit, end=current_end)
            if not isinstance(klines, pd.DataFrame) or klines.empty:
                break
            batches.append(klines)
            oldest_ts = safe_float(klines["timestamp"].min())
            if oldest_ts <= 0:
                break
            oldest_ms = int(oldest_ts * 1000)
            if oldest_ms <= start_ms:
                break
            current_end = oldest_ms - 1
            time.sleep(0.1)

        kline_map: Dict[int, Dict[str, object]] = {}
        if batches:
            full = pd.concat(batches).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
            full = full[(full["timestamp"] >= start_bar) & (full["timestamp"] <= end_bar)]
            for _, row in full.iterrows():
                ts = safe_float(row.get("timestamp"))
                if ts <= 0:
                    continue
                bar_time = bar_time_from_ts(ts, self.tf_seconds)
                if bar_time not in missing:
                    continue
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
                for key in self.bar_window.last_ob:
                    bar[key] = 0.0
                kline_map[bar_time] = bar

        fill_rows = []
        filled_klines = 0
        filled_synth = 0
        missing_left = 0
        last_close = None
        seed_price = self._get_seed_price()
        existing_lookup = bars_df.set_index("bar_time")

        for bar_time in expected_times:
            if bar_time in existing_times:
                try:
                    last_close = safe_float(existing_lookup.loc[bar_time, "close"])
                except Exception:
                    last_close = last_close
                if (last_close is None or last_close <= 0) and seed_price > 0:
                    last_close = seed_price
                continue
            bar = kline_map.get(bar_time)
            if bar:
                fill_rows.append(bar)
                filled_klines += 1
                last_close = safe_float(bar.get("close"))
                continue
            if last_close is not None and last_close > 0:
                volume = 0.0
                bar = {
                    "bar_time": bar_time,
                    "datetime": pd.to_datetime(bar_time, unit="s"),
                    "open": last_close,
                    "high": last_close,
                    "low": last_close,
                    "close": last_close,
                    "volume": volume,
                    "trade_count": 0,
                    "vol_buy": 0.0,
                    "vol_sell": 0.0,
                    "vol_delta": 0.0,
                    "dollar_val": 0.0,
                    "total_val": 0.0,
                    "vwap": last_close,
                    "taker_buy_ratio": 0.5,
                    "buy_vol": 0.0,
                    "sell_vol": 0.0,
                }
                for key in self.bar_window.last_ob:
                    bar[key] = 0.0
                fill_rows.append(bar)
                filled_synth += 1
            elif seed_price > 0:
                bar = self._synthetic_bar_row(bar_time, seed_price)
                fill_rows.append(bar)
                filled_synth += 1
                last_close = seed_price
            else:
                missing_left += 1

        if not fill_rows:
            if missing_left:
                self.logger.warning(
                    f"Gap fill: unable to fill {missing_left} bars (no reference data)."
                )
            return

        fill_df = pd.DataFrame(fill_rows)
        merged = bars_df.set_index("bar_time").combine_first(fill_df.set_index("bar_time"))
        merged = merged.reset_index().sort_values("bar_time")
        merged = self._normalize_bars(merged)
        if merged is None or merged.empty:
            self.logger.warning("Gap fill: normalized bars empty after merge.")
            return
        self.bar_window.load_bars(merged)
        self._save_history()

        last_bar_filled = self.bar_window.latest_bar_time()
        if last_bar_filled is not None:
            self.state.state["last_bar_time"] = last_bar_filled
            last_processed = self.state.state.get("last_processed_bar_time")
            if not last_processed or last_processed < last_bar_filled:
                self.state.state["last_processed_bar_time"] = last_bar_filled
            last_trade_ts = safe_float(self.state.state.get("last_trade_ts"), 0.0)
            if last_trade_ts < last_bar_filled:
                self.state.state["last_trade_ts"] = float(last_bar_filled)

        self.logger.info(
            f"Gap fill complete: klines={filled_klines} synthetic={filled_synth} "
            f"remaining={missing_left}."
        )

    def _fast_bootstrap_fill_gaps(self) -> None:
        if not self.fast_bootstrap:
            return
        bars_df = self.bar_window.bars.copy()
        if bars_df.empty:
            seed = self._get_seed_price()
            if seed > 0 and self._bootstrap_synthetic(self.continuity_bars, seed):
                self.logger.info("Fast bootstrap: created synthetic continuity window.")
            else:
                self.logger.warning("Fast bootstrap: no bars loaded and no seed price available.")
            return
        bars_df = bars_df.drop_duplicates(subset=["bar_time"]).sort_values("bar_time")
        end_bar = int(bars_df["bar_time"].max())
        start_bar = end_bar - (self.continuity_bars - 1) * self.tf_seconds
        expected_times = set(range(start_bar, end_bar + 1, self.tf_seconds))
        existing_times = set(bars_df["bar_time"].astype(int).tolist())
        missing = sorted(expected_times - existing_times)
        if not missing:
            self.logger.info("Fast bootstrap: no missing bars in continuity window.")
            return

        interval_map = {60: "1", 300: "5", 900: "15", 3600: "60", 14400: "240"}
        interval = interval_map.get(self.tf_seconds, "15")
        limit = min(200, len(expected_times) + 5)
        end_ms = int((end_bar + self.tf_seconds) * 1000)
        klines = self.api.fetch_kline(interval=interval, limit=limit, end=end_ms)
        rows = []
        row_times: set = set()
        if isinstance(klines, pd.DataFrame) and not klines.empty:
            for _, row in klines.iterrows():
                ts = safe_float(row.get("timestamp"))
                if ts <= 0:
                    continue
                bar_time = bar_time_from_ts(ts, self.tf_seconds)
                if bar_time not in missing:
                    continue
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
                for key in self.bar_window.last_ob:
                    bar[key] = 0.0
                rows.append(bar)
                row_times.add(bar_time)
        else:
            self.logger.warning("Fast bootstrap: kline fetch empty; using synthetic bars.")

        missing_left = [bt for bt in missing if bt not in row_times]
        if missing_left:
            seed = safe_float(bars_df.iloc[-1].get("close"))
            if seed <= 0:
                seed = self._get_seed_price()
            if seed > 0:
                for bar_time in missing_left:
                    rows.append(self._synthetic_bar_row(bar_time, seed))
            else:
                self.logger.warning("Fast bootstrap: no price seed for synthetic fill.")

        if not rows:
            self.logger.warning("Fast bootstrap: no bars to fill after kline + synthetic.")
            return

        fill_df = pd.DataFrame(rows)
        merged = bars_df.set_index("bar_time").combine_first(fill_df.set_index("bar_time"))
        merged = merged.reset_index().sort_values("bar_time")
        merged = self._normalize_bars(merged)
        if merged is None or merged.empty:
            self.logger.warning("Fast bootstrap: normalized bars empty after gap fill.")
            return
        self.bar_window.load_bars(merged)
        self._save_history()
        last_bar_filled = self.bar_window.latest_bar_time()
        if last_bar_filled is not None:
            self.state.state["last_bar_time"] = last_bar_filled
            last_processed = self.state.state.get("last_processed_bar_time")
            if not last_processed or last_processed < last_bar_filled:
                self.state.state["last_processed_bar_time"] = last_bar_filled
            last_trade_ts = safe_float(self.state.state.get("last_trade_ts"), 0.0)
            if last_trade_ts < last_bar_filled:
                self.state.state["last_trade_ts"] = float(last_bar_filled)
        self.logger.info(
            f"Fast bootstrap filled {len(rows)} missing bars in last {self.continuity_bars}."
        )

    def _normalize_bars(self, bars_df: pd.DataFrame) -> pd.DataFrame:
        if bars_df is None or bars_df.empty:
            return bars_df
        df = bars_df.copy()
        for col in BAR_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0
        df["bar_time"] = pd.to_numeric(df["bar_time"], errors="coerce").fillna(0).astype(int)
        before = len(df)
        df = df[df["bar_time"] > 0]
        if len(df) < before:
            self.logger.warning(f"Normalized bars: dropped {before - len(df)} invalid bar_time rows.")
        df = df.drop_duplicates(subset=["bar_time"], keep="last").sort_values("bar_time").reset_index(drop=True)

        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df.loc[df["close"] <= 0, "close"] = np.nan
        df["close"] = df["close"].ffill().bfill()
        if df["close"].isna().any():
            df["close"] = df["close"].fillna(0.0)
            self.logger.warning("Normalized bars: close missing; filled remaining rows with 0.0.")

        for col in ("open", "high", "low"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] <= 0, col] = np.nan
            df[col] = df[col].fillna(df["close"])
        df["high"] = df[["high", "open", "close"]].max(axis=1)
        df["low"] = df[["low", "open", "close"]].min(axis=1)

        for col in ("volume", "vol_buy", "vol_sell", "dollar_val", "total_val", "buy_vol", "sell_vol"):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        df["trade_count"] = pd.to_numeric(df["trade_count"], errors="coerce").fillna(0).astype(int)
        df.loc[(df["trade_count"] <= 0) & (df["volume"] > 0), "trade_count"] = 1

        df["vol_delta"] = df["vol_buy"] - df["vol_sell"]
        df["dollar_val"] = df["dollar_val"].where(df["dollar_val"] > 0, df["close"] * df["volume"])
        df["total_val"] = df["total_val"].where(df["total_val"] > 0, df["dollar_val"])
        df["vwap"] = np.where(df["volume"] > 0, df["dollar_val"] / df["volume"], df["close"])
        df["taker_buy_ratio"] = np.where(df["volume"] > 0, df["vol_buy"] / df["volume"], 0.5)
        df["buy_vol"] = df["vol_buy"]
        df["sell_vol"] = df["vol_sell"]

        ob_cols = [c for c in BAR_COLUMNS if c.startswith("ob_")]
        if ob_cols:
            for col in ob_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df[ob_cols] = df[ob_cols].ffill().fillna(0.0)

        df["datetime"] = pd.to_datetime(df["bar_time"], unit="s", errors="coerce")
        return df[BAR_COLUMNS]

    def _get_seed_price(self) -> float:
        last_close = self.bar_window.latest_close()
        if last_close and last_close > 0:
            return last_close
        last_trade = safe_float(self.bar_window.trade_builder.last_price)
        if last_trade > 0:
            return last_trade
        try:
            trades_df = self.api.fetch_recent_trades(limit=20)
            if isinstance(trades_df, pd.DataFrame) and not trades_df.empty:
                price = safe_float(trades_df.iloc[-1].get("price"))
                if price > 0:
                    return price
            elif isinstance(trades_df, list) and trades_df:
                price = safe_float(trades_df[-1].get("price"))
                if price > 0:
                    return price
        except Exception:
            pass
        try:
            ob = self.api.fetch_orderbook(limit=1)
            if isinstance(ob, dict):
                bids = ob.get("b") or []
                asks = ob.get("a") or []
                if bids and asks:
                    bid = safe_float(bids[0][0])
                    ask = safe_float(asks[0][0])
                    if bid > 0 and ask > 0:
                        return (bid + ask) * 0.5
        except Exception:
            pass
        return 0.0

    def _synthetic_bar_row(self, bar_time: int, price: float) -> Dict[str, object]:
        price = safe_float(price)
        bar = {
            "bar_time": bar_time,
            "datetime": pd.to_datetime(bar_time, unit="s"),
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 0.0,
            "trade_count": 0,
            "vol_buy": 0.0,
            "vol_sell": 0.0,
            "vol_delta": 0.0,
            "dollar_val": 0.0,
            "total_val": 0.0,
            "vwap": price,
            "taker_buy_ratio": 0.5,
            "buy_vol": 0.0,
            "sell_vol": 0.0,
        }
        for key in self.bar_window.last_ob:
            bar[key] = 0.0
        return bar

    def _merge_history(self, primary: pd.DataFrame, secondary: pd.DataFrame) -> pd.DataFrame:
        if primary is None or primary.empty:
            return self._normalize_bars(secondary)
        if secondary is None or secondary.empty:
            return self._normalize_bars(primary)
        merged = primary.set_index("bar_time").combine_first(secondary.set_index("bar_time"))
        merged = merged.reset_index().sort_values("bar_time")
        return self._normalize_bars(merged)

    def _history_gap_count(self, bars_df: pd.DataFrame) -> int:
        if bars_df is None or len(bars_df) < 2 or self.tf_seconds <= 0:
            return 0
        times = bars_df["bar_time"].astype(int).to_numpy()
        diffs = np.diff(times)
        gaps = diffs[diffs > self.tf_seconds]
        if gaps.size == 0:
            return 0
        missing = int(np.sum((gaps / self.tf_seconds) - 1))
        return max(0, missing)

    def _history_ready(self) -> Tuple[bool, List[str]]:
        bars_df = self._normalize_bars(self.bar_window.bars.copy())
        if bars_df is None or bars_df.empty:
            return False, ["no bars"]
        self.bar_window.load_bars(bars_df)

        reasons: List[str] = []
        if len(bars_df) < self.min_feature_bars:
            reasons.append(f"bars<{self.min_feature_bars}")
        gap_count = self._history_gap_count(bars_df)
        if gap_count:
            reasons.append(f"gaps={gap_count}")
        last_bar = safe_int(bars_df.iloc[-1]["bar_time"])
        bar_close = last_bar + self.tf_seconds
        age_sec = time.time() - bar_close
        if age_sec < 0:
            age_sec = 0.0
        if age_sec > self.max_last_bar_age_sec:
            reasons.append(f"last_bar_age>{self.max_last_bar_age_sec:.1f}s")
        if (bars_df["close"] <= 0).any():
            reasons.append("close<=0")
        return len(reasons) == 0, reasons

    def _pad_history_to_min_bars(self, bars_df: pd.DataFrame, min_bars: int, seed_price: float) -> pd.DataFrame:
        if bars_df is None or bars_df.empty or min_bars <= 0:
            return bars_df
        missing = max(0, min_bars - len(bars_df))
        if missing == 0:
            return bars_df
        first_bar = safe_int(bars_df.iloc[0]["bar_time"])
        price = safe_float(bars_df.iloc[0].get("close"))
        if price <= 0:
            price = seed_price
        if price <= 0:
            return bars_df

        rows = []
        for i in range(missing):
            bar_time = first_bar - self.tf_seconds * (missing - i)
            rows.append(self._synthetic_bar_row(bar_time, price))
        padded = pd.concat([pd.DataFrame(rows), bars_df], ignore_index=True)
        return self._normalize_bars(padded)

    def _bootstrap_synthetic(self, min_bars: int, seed_price: float) -> bool:
        if min_bars <= 0 or seed_price <= 0:
            return False
        end_bar = bar_time_from_ts(time.time() - self.tf_seconds, self.tf_seconds)
        if end_bar <= 0:
            return False
        start_bar = end_bar - (min_bars - 1) * self.tf_seconds
        rows = []
        for bar_time in range(start_bar, end_bar + self.tf_seconds, self.tf_seconds):
            rows.append(self._synthetic_bar_row(bar_time, seed_price))
        df = self._normalize_bars(pd.DataFrame(rows))
        if df is None or df.empty:
            return False
        self.bar_window.load_bars(df)
        self._save_history()
        self.logger.warning(f"Synthetic bootstrap created {len(df)} bars using price seed {seed_price:.6f}.")
        return True

    def _seed_trade_builder_from_history(self) -> None:
        if self.bar_window.trade_builder.current_bar_time is not None:
            return
        last_bar = self.bar_window.latest_bar_time()
        last_close = self.bar_window.latest_close()
        if last_bar is None or not last_close or last_close <= 0:
            return
        next_bar = last_bar + self.tf_seconds
        self.bar_window.trade_builder._start_bar(next_bar, last_close)

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

            df = self._normalize_bars(df)
            if df is None or df.empty:
                self.logger.warning("History load: no usable bars after normalization.")
                return

            base = self.bar_window.bars.copy()
            if not base.empty:
                base = self._normalize_bars(base)
                if base is None or base.empty:
                    merged = df
                else:
                    merged = df.set_index("bar_time").combine_first(base.set_index("bar_time"))
                    merged = merged.reset_index().sort_values("bar_time")
                    merged = self._normalize_bars(merged)
            else:
                merged = df

            if merged is None or merged.empty:
                self.logger.warning("History load: normalized bars empty after merge.")
                return
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

    def _latest_history_bar_time(self) -> int:
        path = self.history_path
        try:
            if not path.exists() or path.stat().st_size <= 0:
                return 0
        except Exception:
            return 0
        try:
            size = path.stat().st_size
        except Exception:
            return 0
        block = min(size, 64 * 1024)
        try:
            with open(path, "rb") as f:
                if size > block:
                    f.seek(-block, os.SEEK_END)
                data = f.read(block)
        except Exception:
            return 0
        lines = data.splitlines()
        if size > block and lines:
            lines = lines[1:]
        for raw in reversed(lines):
            try:
                text = raw.decode("utf-8", errors="ignore").strip()
            except Exception:
                continue
            if not text or text.startswith("bar_time"):
                continue
            parts = text.split(",", 1)
            if not parts:
                continue
            try:
                return int(float(parts[0]))
            except Exception:
                continue
        return 0

    def _save_history(self) -> None:
        if not self.history_writer:
            return
        if self.bar_window.bars.empty:
            return
        try:
            latest_bar = self.bar_window.latest_bar_time()
            if latest_bar is None:
                return
            disk_bar = self._latest_history_bar_time()
            if disk_bar >= latest_bar:
                return
            save_df = self.bar_window.bars.tail(self.window_size).copy()
            save_df = save_df.drop_duplicates(subset=["bar_time"], keep="last").sort_values("bar_time")
            save_df.to_csv(self.history_path, index=False)
        except Exception as exc:
            self.logger.error(f"Failed to save history: {exc}")

    def _init_position_state(self) -> None:
        positions = self.state.state.get("positions")
        if not isinstance(positions, dict):
            positions = {}
        positions.setdefault("long", {})
        positions.setdefault("short", {})
        if self.hedge_mode:
            long_pos = positions.get("long", {})
            short_pos = positions.get("short", {})
            if not long_pos and not short_pos:
                pos = self.state.state.get("position", {})
                side = pos.get("side")
                size = safe_float(pos.get("size"))
                if pos and size > 0 and side in {"Buy", "Sell"}:
                    key = "long" if side == "Buy" else "short"
                    positions[key] = pos
        self.state.state["positions"] = positions
        self.state.save()

    def _get_positions(self) -> Tuple[Dict, Dict]:
        if not self.hedge_mode:
            pos = self.state.state.get("position", {})
            return pos, {}
        positions = self.state.state.get("positions", {})
        if not isinstance(positions, dict):
            return {}, {}
        return positions.get("long", {}), positions.get("short", {})

    def _entry_intents(self) -> Dict[str, Dict]:
        intents = self.state.state.get("entry_intent")
        if not isinstance(intents, dict):
            intents = {}
        intents.setdefault("Buy", {})
        intents.setdefault("Sell", {})
        return intents

    def _virtual_positions(self) -> List[Dict]:
        positions = self.state.state.get("virtual_positions")
        if not isinstance(positions, list):
            positions = []
        self.state.state["virtual_positions"] = positions
        return positions

    def _virtual_positions_count(self, include_external: bool = False) -> int:
        count = 0
        for pos in self._virtual_positions():
            if safe_float(pos.get("size")) <= 0:
                continue
            if pos.get("external") and not include_external:
                continue
            count += 1
        return count

    def _virtual_positions_size(self, side: str, include_external: bool = True) -> float:
        total = 0.0
        for pos in self._virtual_positions():
            if pos.get("side") != side:
                continue
            if safe_float(pos.get("size")) <= 0:
                continue
            if pos.get("external") and not include_external:
                continue
            total += safe_float(pos.get("size"))
        return total

    def _virtual_position_for_order(self, order_link_id: Optional[str]) -> Optional[Dict]:
        if not order_link_id:
            return None
        for pos in self._virtual_positions():
            if pos.get("order_link_id") == order_link_id:
                return pos
        return None

    def _entry_order_kind(self, order_link_id: Optional[str]) -> Optional[str]:
        if not order_link_id:
            return None
        text = str(order_link_id)
        if not text.startswith(self.orderlink_prefix):
            return None
        idx = len(self.orderlink_prefix)
        if idx >= len(text):
            return None
        return text[idx].upper()

    def _is_entry_order_link(self, order_link_id: Optional[str]) -> bool:
        return self._entry_order_kind(order_link_id) in {"E", "A"}

    def _add_or_update_virtual_position(
        self,
        side: str,
        size: float,
        entry_price: float,
        atr: float,
        tp_mult: float = 1.0,
        entry_bar_time: Optional[int] = None,
        entry_ts_ms: Optional[int] = None,
        order_link_id: Optional[str] = None,
        order_id: Optional[str] = None,
        external: bool = False,
        origin: str = "execution",
    ) -> Optional[Dict]:
        if side not in {"Buy", "Sell"}:
            return None
        if size <= 0 or entry_price <= 0:
            return None
        positions = self._virtual_positions()
        existing = self._virtual_position_for_order(order_link_id) if order_link_id else None
        if existing:
            old_size = safe_float(existing.get("size"))
            new_size = old_size + size
            if new_size <= 0:
                return existing
            weighted = (safe_float(existing.get("entry_price")) * old_size + entry_price * size) / new_size
            existing["size"] = new_size
            existing["entry_price"] = weighted
            if atr <= 0:
                atr = safe_float(existing.get("atr_at_entry"))
            if atr > 0:
                existing["atr_at_entry"] = atr
            existing["tp_mult"] = max(safe_float(existing.get("tp_mult"), 1.0), tp_mult)
            tp, sl = self._compute_tp_sl_targets(side, weighted, safe_float(existing.get("atr_at_entry")), tp_mult=existing["tp_mult"])
            existing["tp"] = tp
            existing["sl"] = sl
            existing["order_id"] = order_id or existing.get("order_id")
            existing_origin = existing.get("origin")
            existing_bar = safe_int(existing.get("entry_bar_time"))
            existing_ts = safe_int(existing.get("entry_ts_ms"))
            if entry_bar_time:
                if existing_origin != "execution":
                    existing["entry_bar_time"] = entry_bar_time
                elif existing_bar <= 0 or entry_bar_time < existing_bar:
                    existing["entry_bar_time"] = entry_bar_time
            if entry_ts_ms:
                if existing_origin != "execution":
                    existing["entry_ts_ms"] = entry_ts_ms
                elif existing_ts <= 0 or entry_ts_ms < existing_ts:
                    existing["entry_ts_ms"] = entry_ts_ms
            if origin == "execution" and existing_origin != "execution":
                existing["origin"] = "execution"
            return existing

        if atr <= 0 and self.last_feature_row is not None:
            atr = safe_float(self.last_feature_row.get("atr"))
        tp, sl = self._compute_tp_sl_targets(side, entry_price, atr, tp_mult=tp_mult)
        position_idx = None
        if self.hedge_mode:
            position_idx = 1 if side == "Buy" else 2
        payload = {
            "id": uuid.uuid4().hex[:12],
            "side": side,
            "size": size,
            "entry_price": entry_price,
            "tp": tp,
            "sl": sl,
            "atr_at_entry": atr,
            "tp_mult": tp_mult,
            "entry_bar_time": entry_bar_time,
            "entry_ts_ms": entry_ts_ms,
            "order_link_id": order_link_id,
            "order_id": order_id,
            "position_idx": position_idx,
            "external": external,
            "origin": origin,
            "created_ts": utc_now_str(),
        }
        positions.append(payload)
        return payload

    def _reconcile_virtual_positions_side(
        self,
        side: str,
        actual_size: float,
        entry_price: float,
        entry_bar_time: Optional[int],
        external: bool,
    ) -> None:
        if side not in {"Buy", "Sell"}:
            return
        positions = self._virtual_positions()
        side_positions = [p for p in positions if p.get("side") == side and safe_float(p.get("size")) > 0]
        current_size = sum(safe_float(p.get("size")) for p in side_positions)
        tol = 1e-8
        if actual_size <= tol:
            if side_positions:
                self.state.state["virtual_positions"] = [p for p in positions if p.get("side") != side]
            return
        if current_size <= tol:
            self._add_or_update_virtual_position(
                side,
                actual_size,
                entry_price,
                safe_float(self.last_feature_row.get("atr")) if self.last_feature_row is not None else 0.0,
                entry_bar_time=entry_bar_time,
                entry_ts_ms=None,
                external=external,
                origin="reconcile",
            )
            return
        if actual_size > current_size + tol:
            delta = actual_size - current_size
            self._add_or_update_virtual_position(
                side,
                delta,
                entry_price,
                safe_float(self.last_feature_row.get("atr")) if self.last_feature_row is not None else 0.0,
                entry_bar_time=entry_bar_time,
                entry_ts_ms=None,
                external=external,
                origin="reconcile",
            )
            return
        if actual_size < current_size - tol:
            remaining = actual_size
            kept = []
            for pos in side_positions:
                size = safe_float(pos.get("size"))
                if remaining <= tol:
                    continue
                if size <= remaining + tol:
                    kept.append(pos)
                    remaining -= size
                else:
                    pos["size"] = remaining
                    kept.append(pos)
                    remaining = 0.0
            self.state.state["virtual_positions"] = [
                p for p in positions if p.get("side") != side
            ] + kept

    def _register_entry_intent_fill(
        self,
        side: str,
        intent: Optional[Dict],
        fallback_bar_time: Optional[int] = None,
    ) -> None:
        if side not in {"Buy", "Sell"}:
            return
        if not isinstance(intent, dict) or not intent:
            return
        order_link_id = intent.get("order_link_id")
        if order_link_id and self._virtual_position_for_order(order_link_id):
            return
        qty = safe_float(intent.get("qty"))
        price = safe_float(intent.get("price"))
        atr = safe_float(intent.get("atr_at_entry"))
        tp_mult = safe_float(intent.get("tp_mult"), 1.0)
        entry_bar_time = safe_int(intent.get("created_bar_time")) or safe_int(fallback_bar_time)
        entry_ts_ms = safe_int(intent.get("entry_ts_ms"))
        self._add_or_update_virtual_position(
            side,
            qty,
            price,
            atr,
            tp_mult=tp_mult,
            entry_bar_time=entry_bar_time or None,
            entry_ts_ms=entry_ts_ms or None,
            order_link_id=order_link_id,
            order_id=intent.get("order_id"),
            external=False,
            origin="intent",
        )

    def _track_entry_execution(self, execution: Dict) -> None:
        order_link_id = execution.get("orderLinkId") or execution.get("order_link_id")
        if not self._is_entry_order_link(order_link_id):
            return
        side = execution.get("side")
        qty = safe_float(execution.get("execQty") or execution.get("orderQty") or execution.get("qty"))
        price = safe_float(execution.get("execPrice") or execution.get("price"))
        if qty <= 0 or price <= 0 or side not in {"Buy", "Sell"}:
            return
        intent = self._entry_intents().get(side, {})
        if intent and order_link_id and intent.get("order_link_id") not in {None, order_link_id}:
            intent = {}
        atr = safe_float(intent.get("atr_at_entry"))
        tp_mult = safe_float(intent.get("tp_mult"), 1.0)
        exec_time = normalize_ts_ms(execution.get("execTime") or execution.get("exec_time"))
        entry_bar_time = None
        if exec_time > 0:
            entry_bar_time = bar_time_from_ts(exec_time / 1000.0, self.tf_seconds)
        if entry_bar_time is None:
            entry_bar_time = safe_int(self.state.state.get("last_bar_time"))
        self._add_or_update_virtual_position(
            side,
            qty,
            price,
            atr,
            tp_mult=tp_mult,
            entry_bar_time=entry_bar_time or None,
            entry_ts_ms=exec_time or None,
            order_link_id=order_link_id,
            order_id=execution.get("orderId") or execution.get("order_id"),
            external=False,
            origin="execution",
        )

    def _manage_virtual_positions(self, bar_time: int, bar_high: float, bar_low: float, bar_close: float) -> None:
        if self.signal_only:
            return
        positions = self._virtual_positions()
        if not positions:
            return
        closed = []
        for pos in positions:
            if pos.get("external"):
                continue
            side = pos.get("side")
            size = safe_float(pos.get("size"))
            if side not in {"Buy", "Sell"} or size <= 0:
                continue
            entry_bar = safe_int(pos.get("entry_bar_time"))
            entry_price = safe_float(pos.get("entry_price"))
            tp = safe_float(pos.get("tp"))
            sl = safe_float(pos.get("sl"))
            atr = safe_float(pos.get("atr_at_entry"))
            if (tp <= 0 or sl <= 0) and entry_price > 0:
                if atr <= 0 and self.last_feature_row is not None:
                    atr = safe_float(self.last_feature_row.get("atr"))
                if atr > 0:
                    tp_calc, sl_calc = self._compute_tp_sl_targets(
                        side,
                        entry_price,
                        atr,
                        tp_mult=safe_float(pos.get("tp_mult"), 1.0),
                    )
                    if tp <= 0:
                        pos["tp"] = tp_calc
                        tp = tp_calc
                    if sl <= 0:
                        pos["sl"] = sl_calc
                        sl = sl_calc
            exit_price = None
            reason = None
            if entry_bar > 0 and bar_time >= entry_bar:
                age_bars = int((bar_time - entry_bar) / self.tf_seconds)
                if age_bars >= self.config.strategy.max_holding_bars:
                    exit_price = bar_close
                    reason = "time"
            if exit_price is None and bar_high > 0 and bar_low > 0:
                if side == "Buy":
                    if sl > 0 and bar_low <= sl:
                        exit_price = sl
                        reason = "sl"
                    elif tp > 0 and bar_high >= tp:
                        exit_price = tp
                        reason = "tp"
                else:
                    if sl > 0 and bar_high >= sl:
                        exit_price = sl
                        reason = "sl"
                    elif tp > 0 and bar_low <= tp:
                        exit_price = tp
                        reason = "tp"
            if exit_price is not None:
                closed.append((pos, exit_price, reason))

        if not closed:
            return
        for pos, exit_price, reason in closed:
            qty = safe_float(pos.get("size"))
            if qty <= 0:
                continue
            position_idx = pos.get("position_idx") if self.hedge_mode else None
            self.logger.info(
                f"Virtual exit {reason}: {pos.get('side')} qty={qty:.6f} price={exit_price:.6f}"
            )
            if not self.dry_run:
                self.api.market_close(pos.get("side"), abs(qty), position_idx=position_idx)
        closed_ids = {pos.get("id") for pos, _exit, _reason in closed if pos.get("id")}
        if closed_ids:
            self.state.state["virtual_positions"] = [
                pos for pos in positions if pos.get("id") not in closed_ids
            ]

    def _resolve_entry_ts_ms(
        self,
        created_time: int,
        internal_exec: Optional[Dict],
        intent: Optional[Dict],
    ) -> Tuple[int, str]:
        created_ms = normalize_ts_ms(created_time)
        entry_ts_ms = created_ms
        entry_source = "created_time"

        exec_ms = 0
        if internal_exec:
            exec_ms = normalize_ts_ms(internal_exec.get("exec_time"))
        if exec_ms > 0 and (created_ms <= 0 or exec_ms >= created_ms):
            entry_ts_ms = exec_ms
            entry_source = "execution"
        elif intent:
            entry_source = "entry_intent"

        if entry_ts_ms <= 0:
            entry_ts_ms = int(time.time() * 1000)
        return entry_ts_ms, entry_source

    def _is_internal_order_link_id(self, order_link_id: Optional[str]) -> bool:
        if not order_link_id:
            return False
        return str(order_link_id).startswith(self.orderlink_prefix)

    def _reset_entry_intents_on_start(self) -> None:
        intents = {"Buy": {}, "Sell": {}}
        orders = self.state.state.get("active_orders", {})
        if isinstance(orders, dict):
            for oid, info in orders.items():
                side = info.get("side")
                if side not in {"Buy", "Sell"}:
                    continue
                if not self._is_internal_order_link_id(info.get("order_link_id")):
                    continue
                intents[side] = {
                    "order_id": oid,
                    "order_link_id": info.get("order_link_id"),
                    "price": info.get("price"),
                    "qty": info.get("qty"),
                    "tp": info.get("tp"),
                    "sl": info.get("sl"),
                    "atr_at_entry": info.get("atr_at_entry"),
                    "created_bar_time": info.get("created_bar_time"),
                    "created_ts": info.get("created_ts"),
                }
        self.state.state["entry_intent"] = intents

    def _ensure_virtual_positions_state(self) -> None:
        positions = self._virtual_positions()
        if positions:
            return
        if self.hedge_mode:
            state_positions = self.state.state.get("positions", {})
            if isinstance(state_positions, dict):
                for key, side in (("long", "Buy"), ("short", "Sell")):
                    pos = state_positions.get(key, {})
                    size = safe_float(pos.get("size"))
                    if size <= 0:
                        continue
                    entry_price = safe_float(pos.get("entry_price") or pos.get("avg_price"))
                    entry_bar = safe_int(pos.get("entry_bar_time"))
                    entry_ts_ms = safe_int(pos.get("entry_ts_ms"))
                    self._add_or_update_virtual_position(
                        side,
                        size,
                        entry_price,
                        0.0,
                        entry_bar_time=entry_bar or None,
                        entry_ts_ms=entry_ts_ms or None,
                        external=bool(pos.get("external")),
                        origin="bootstrap",
                    )
        else:
            pos = self.state.state.get("position", {})
            size = safe_float(pos.get("size"))
            side = pos.get("side")
            if size > 0 and side in {"Buy", "Sell"}:
                entry_price = safe_float(pos.get("entry_price") or pos.get("avg_price"))
                entry_bar = safe_int(pos.get("entry_bar_time"))
                entry_ts_ms = safe_int(pos.get("entry_ts_ms"))
                self._add_or_update_virtual_position(
                    side,
                    size,
                    entry_price,
                    0.0,
                    entry_bar_time=entry_bar or None,
                    entry_ts_ms=entry_ts_ms or None,
                    external=bool(pos.get("external")),
                    origin="bootstrap",
                )

    def _prune_internal_execs(self, now: Optional[float] = None) -> None:
        execs = self.state.state.get("internal_execs", [])
        if not isinstance(execs, list) or not execs:
            self.state.state["internal_execs"] = []
            return
        now = now or time.time()
        max_age_sec = max(
            float(self.tf_seconds) * float(self.config.strategy.max_holding_bars),
            float(self.tf_seconds) * 6.0,
        )
        kept = []
        for item in execs:
            exec_time = safe_float(item.get("exec_time"))
            if exec_time <= 0:
                continue
            age = now - (exec_time / 1000.0)
            if age <= max_age_sec:
                kept.append(item)
        if len(kept) > 200:
            kept = kept[-200:]
        self.state.state["internal_execs"] = kept

    def _record_internal_exec(self, execution: Dict) -> None:
        execs = self.state.state.get("internal_execs", [])
        if not isinstance(execs, list):
            execs = []
        execs.append(execution)
        self.state.state["internal_execs"] = execs
        self._track_entry_execution(execution)
        self._prune_internal_execs()

    def _find_recent_internal_exec(self, side: Optional[str], position_idx: int = 0) -> Optional[Dict]:
        execs = self.state.state.get("internal_execs", [])
        if not isinstance(execs, list) or not execs:
            return None
        now = time.time()
        max_age_sec = max(
            float(self.tf_seconds) * float(self.config.strategy.max_holding_bars),
            float(self.tf_seconds) * 6.0,
        )
        best = None
        for item in execs:
            exec_time = safe_float(item.get("exec_time"))
            if exec_time <= 0:
                continue
            if now - (exec_time / 1000.0) > max_age_sec:
                continue
            if side and item.get("side") and item.get("side") != side:
                continue
            if self.hedge_mode and position_idx:
                exec_idx = safe_int(item.get("position_idx"))
                if exec_idx and exec_idx != position_idx:
                    continue
            if best is None or exec_time > safe_float(best.get("exec_time")):
                best = item
        return best

    def _clear_entry_intent(
        self,
        side: str,
        order_id: Optional[str] = None,
        order_link_id: Optional[str] = None,
        force: bool = False,
    ) -> None:
        if not side:
            return
        intents = self._entry_intents()
        intent = intents.get(side, {})
        if not intent:
            return
        if order_id and intent.get("order_id") != order_id:
            if not order_link_id or intent.get("order_link_id") != order_link_id:
                return
        if order_link_id and intent.get("order_link_id") != order_link_id:
            return
        if intent.get("filled") and not force:
            return
        intents[side] = {}
        self.state.state["entry_intent"] = intents

    def _new_order_link_id(self, kind: str, side: str) -> str:
        kind_code = (kind or "X")[0].upper()
        side_code = "B" if side == "Buy" else "S"
        suffix = uuid.uuid4().hex[:16]
        return f"{self.orderlink_prefix}{kind_code}{side_code}{suffix}"

    def _is_post_only_reject(self, ret_code: Optional[int], ret_msg: str) -> bool:
        msg = (ret_msg or "").lower()
        if ret_code in {110017}:
            return True
        if "postonly" in msg or "post only" in msg:
            return True
        if "maker" in msg and "taker" in msg:
            return True
        if "immediate" in msg and "execute" in msg:
            return True
        return False

    def _is_duplicate_order_link(self, ret_code: Optional[int], ret_msg: str) -> bool:
        msg = (ret_msg or "").lower()
        if ret_code in {110041}:
            return True
        if "orderlink" in msg and ("duplicate" in msg or "exist" in msg):
            return True
        return False

    def _get_state_order(self, order_id: Optional[str], order_link_id: Optional[str]) -> Optional[Dict]:
        orders = self.state.state.get("active_orders", {})
        if order_id and order_id in orders:
            return orders.get(order_id)
        if order_link_id:
            for info in orders.values():
                if info.get("order_link_id") == order_link_id:
                    return info
        return None

    def _confirm_order_resting(
        self,
        order_id: Optional[str],
        order_link_id: Optional[str],
    ) -> Optional[Dict]:
        for attempt in range(self.rest_confirm_attempts):
            if self.use_ws_private:
                self._process_ws_private()
                state_order = self._get_state_order(order_id, order_link_id)
                if state_order:
                    return state_order
                if attempt < self.rest_confirm_attempts - 1:
                    time.sleep(self.rest_confirm_sleep_sec)
                    continue
            if order_id or order_link_id:
                order = self.api.get_open_order(order_id=order_id, order_link_id=order_link_id)
                if order:
                    return order
            if attempt < self.rest_confirm_attempts - 1:
                time.sleep(self.rest_confirm_sleep_sec)
        return None

    def _tp_order_needs_resize(self, pos: Dict, size: float) -> bool:
        tp_qty = safe_float(pos.get("tp_order_qty"))
        if tp_qty <= 0:
            return False
        desired = abs(size)
        if desired <= 0:
            return False
        qty_step = self.instrument.qty_step
        tolerance = max(qty_step, desired * 0.01)
        return abs(tp_qty - desired) >= tolerance

    def _position_size(self, pos: Dict) -> float:
        return safe_float(pos.get("size")) if pos else 0.0

    def _is_external_position(self, pos: Dict) -> bool:
        return bool(pos.get("external")) if pos else False

    def _position_key(self, side: Optional[str], position_idx: int = 0) -> Optional[str]:
        if position_idx == 1:
            return "long"
        if position_idx == 2:
            return "short"
        if side == "Buy":
            return "long"
        if side == "Sell":
            return "short"
        return None

    def _entry_price_within_tolerance(self, entry_price: float, ref_price: float, atr: float) -> bool:
        if entry_price <= 0 or ref_price <= 0:
            return True
        tol = max(entry_price * 0.002, atr * 0.5)
        tick = self.instrument.tick_size
        if tick > 0:
            tol = max(tol, tick * 5)
        return abs(entry_price - ref_price) <= tol

    def _position_matches_missing_order(
        self,
        side: Optional[str],
        order_info: Dict,
        position: Dict,
        positions: List[Dict],
    ) -> bool:
        if not side or order_info.get("external"):
            return False
        if self.hedge_mode:
            order_idx = safe_int(order_info.get("position_idx"))
            for pos in positions:
                if safe_float(pos.get("size")) <= 0:
                    continue
                pos_side = pos.get("side")
                if pos_side and pos_side != side:
                    continue
                if order_idx:
                    pos_idx = safe_int(pos.get("position_idx"))
                    if pos_idx and pos_idx != order_idx:
                        continue
                entry_price = safe_float(pos.get("avg_price") or pos.get("entry_price"))
                order_price = safe_float(order_info.get("price"))
                atr = safe_float(order_info.get("atr_at_entry"))
                if self._entry_price_within_tolerance(entry_price, order_price, atr):
                    return True
            return False

        if not position or safe_float(position.get("size")) <= 0:
            return False
        pos_side = position.get("side")
        if pos_side and pos_side != side:
            return False
        entry_price = safe_float(position.get("avg_price") or position.get("entry_price"))
        order_price = safe_float(order_info.get("price"))
        atr = safe_float(order_info.get("atr_at_entry"))
        return self._entry_price_within_tolerance(entry_price, order_price, atr)

    def _build_position_uid(
        self,
        payload: Dict,
        side: Optional[str],
        position_idx: int,
        entry_ts_ms: int,
    ) -> Tuple[Optional[str], str]:
        position_id = payload.get("positionId") or payload.get("positionID") or payload.get("position_id")
        if position_id:
            return f"id:{position_id}", "position_id"
        created_ms = normalize_ts_ms(payload.get("createdTime") or payload.get("created_time"))
        if created_ms > 0:
            return f"ct:{created_ms}|idx:{position_idx}|side:{side}", "created_time"
        if entry_ts_ms > 0:
            return f"et:{entry_ts_ms}|idx:{position_idx}|side:{side}", "entry_ts"
        return None, "unknown"

    def _update_position_identity(
        self,
        pos_state: Dict,
        payload: Dict,
        side: Optional[str],
        position_idx: int,
    ) -> None:
        entry_ts_ms = safe_int(pos_state.get("entry_ts_ms"))
        uid, source = self._build_position_uid(payload, side, position_idx, entry_ts_ms)
        if uid:
            pos_state["position_uid"] = uid
            pos_state["position_uid_source"] = source
        elif not pos_state.get("position_uid"):
            pos_state["position_uid_source"] = source
        origin_source = pos_state.get("origin_source")
        pos_state["entry_verified"] = source in {"position_id", "created_time"} or origin_source in {
            "execution",
            "entry_intent",
        }

    def _maybe_refresh_position_entry(self, pos_state: Dict, created_time: int) -> None:
        created_ms = normalize_ts_ms(created_time)
        if created_ms <= 0:
            return
        existing_ms = safe_int(pos_state.get("entry_ts_ms"))
        if existing_ms <= 0 or created_ms > existing_ms:
            pos_state["entry_ts_ms"] = created_ms
            pos_state["entry_bar_time"] = bar_time_from_ts(created_ms / 1000.0, self.tf_seconds)
            pos_state["origin_source"] = "created_time"
        elif not pos_state.get("entry_bar_time"):
            pos_state["entry_bar_time"] = bar_time_from_ts(existing_ms / 1000.0, self.tf_seconds)

    def _data_health(self) -> str:
        bars = self.bar_window.bars
        if bars.empty:
            return "Data: EMPTY"
        trade_bars = bars[bars["trade_count"] > 0]
        n_trade = len(trade_bars)
        total = len(bars)
        macro_target = max(self.min_feature_bars, 1)
        macro_pct = min(100.0, (total / macro_target) * 100)
        window = min(24, total)
        if window == 0:
            ob_density = 0.0
        else:
            recent = bars.tail(window)
            ob_density = (recent["ob_spread_mean"] > 0).sum() / window * 100
        return (
            f"Bars: {n_trade}/{total} | Macro: {macro_pct:.0f}% | "
            f"OB-Density(2h): {ob_density:.0f}% | TradeCont: {self.continuous_trade_bars}"
        )

    def _history_continuity_ready(self, bars_df: pd.DataFrame) -> Tuple[bool, str]:
        if bars_df.empty:
            return False, "no bars loaded"
        need = max(1, self.continuity_bars)
        if len(bars_df) < need:
            return False, f"need {need} bars, have {len(bars_df)}"
        recent = bars_df.tail(need)
        times = recent["bar_time"].astype(int).to_numpy()
        if len(times) < need:
            return False, f"need {need} bars, have {len(times)}"
        diffs = np.diff(times)
        if len(diffs) > 0 and not np.all(diffs == self.tf_seconds):
            return False, f"non-continuous bars in last {need}"
        last_bar = int(times[-1])
        bar_close = last_bar + self.tf_seconds
        age_sec = time.time() - bar_close
        if age_sec < 0:
            age_sec = 0.0
        if age_sec > self.max_last_bar_age_sec:
            return False, f"last bar close age {age_sec:.1f}s > {self.max_last_bar_age_sec:.1f}s"
        end_bar = bar_time_from_ts(time.time() - self.tf_seconds, self.tf_seconds)
        if end_bar > last_bar:
            missing = int((end_bar - last_bar) / self.tf_seconds)
            return False, f"missing {missing} bars to {end_bar}"
        return True, "ok"

    def _refresh_instrument_info(self, force: bool = False) -> None:
        info = self.api.get_instrument_info()
        if not info:
            if force:
                self.logger.warning("Instrument info unavailable.")
            return
        self.instrument = InstrumentInfo(
            min_qty=safe_float(info.get("min_qty")),
            max_qty=safe_float(info.get("max_qty")),
            qty_step=safe_float(info.get("qty_step")),
            tick_size=safe_float(info.get("tick_size")),
            min_notional=safe_float(info.get("min_notional")),
            max_notional=safe_float(info.get("max_notional")),
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

    def _restart_websockets(self, reason: str) -> None:
        if not (self.use_ws_trades or self.use_ws_ob or self.use_ws_private):
            return
        now = time.time()
        if now - self.ws_last_restart_ts < self.ws_restart_cooldown_sec:
            return
        self.ws_last_restart_ts = now
        self.logger.warning(f"Restarting WebSocket streams ({reason}).")
        if self.ws is not None:
            try:
                if hasattr(self.ws, "exit"):
                    self.ws.exit()
            except Exception as exc:
                self.logger.warning(f"WS exit failed: {exc}")
        if self.ws_private is not None:
            try:
                if hasattr(self.ws_private, "exit"):
                    self.ws_private.exit()
            except Exception as exc:
                self.logger.warning(f"WS private exit failed: {exc}")
        self.ws = None
        self.ws_private = None
        self.ws_trade_queue.clear()
        self.ws_ob_queue.clear()
        self.ws_ob_book = {"b": {}, "a": {}}
        self.ws_ob_initialized = False
        self.ws_trade_last_ts = 0.0
        self.ws_ob_last_ts = 0.0
        self.ws_trade_last_ingest_ts = 0.0
        self.ws_ob_last_ingest_ts = 0.0
        self._init_websockets()

    def _on_ws_trade(self, message: Dict) -> None:
        data = message.get("data", [])
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            return
        entries: List[Dict] = []
        latest_ts = self.ws_trade_last_ts
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
            entries.append(entry)
            latest_ts = max(latest_ts, ts_sec)
        if not entries:
            return
        with self.ws_lock:
            self.ws_trade_queue.extend(entries)
        self.ws_trade_last_ts = latest_ts
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
            return self._apply_ws_tp_order_update(order)
        oid = order.get("orderId")
        if not oid:
            return False
        order_link_id = order.get("orderLinkId")
        status = str(order.get("orderStatus") or "").lower()
        leaves_qty = safe_float(order.get("leavesQty"))
        create_type = str(order.get("createType") or "").lower()
        closed_statuses = {"cancelled", "filled", "deactivated", "rejected"}
        active_statuses = {"new", "partiallyfilled", "partially_filled", "untriggered", "triggered"}

        if status in closed_statuses or (leaves_qty == 0 and status):
            if oid in state_orders:
                side = order.get("side")
                if side:
                    if status == "filled":
                        intents = self._entry_intents()
                        intent = intents.get(side, {})
                        if intent and (
                            intent.get("order_id") == oid
                            or (order_link_id and intent.get("order_link_id") == order_link_id)
                        ):
                            intent["filled"] = True
                            intents[side] = intent
                            self.state.state["entry_intent"] = intents
                            created_bar = safe_int(intent.get("created_bar_time"))
                            current_bar = safe_int(self.state.state.get("last_bar_time"))
                            if created_bar and current_bar:
                                filled_bar = max(created_bar, current_bar)
                            else:
                                filled_bar = created_bar or current_bar
                            self._register_entry_intent_fill(side, intent, filled_bar or None)
                    else:
                        self._clear_entry_intent(side, oid, order_link_id, force=True)
                state_orders.pop(oid, None)
                return True
            return False

        if status in active_statuses or leaves_qty > 0:
            created_ms = normalize_ts_ms(order.get("createdTime"))
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
                if self._is_internal_order_link_id(order_link_id):
                    external = False
                else:
                    external = create_type not in {"createbyuser", ""}
            state_orders[oid] = {
                "side": order.get("side"),
                "price": price,
                "qty": qty,
                "tp": tp,
                "sl": sl,
                "atr_at_entry": safe_float(existing.get("atr_at_entry")),
                "created_bar_time": created_bar,
                "created_ts": utc_now_str(),
                "external": external,
                "order_link_id": order_link_id or existing.get("order_link_id"),
            }
            return True

        return False

    def _apply_ws_tp_order_update(self, order: Dict) -> bool:
        if order.get("symbol") != self.config.data.symbol:
            return False
        oid = order.get("orderId")
        order_link_id = order.get("orderLinkId")
        if not oid and not order_link_id:
            return False
        status = str(order.get("orderStatus") or "").lower()
        leaves_qty = safe_float(order.get("leavesQty"))
        closed_statuses = {"cancelled", "filled", "deactivated", "rejected"}
        is_closed = status in closed_statuses or (leaves_qty == 0 and status)
        price = safe_float(order.get("price"))
        qty = leaves_qty if leaves_qty > 0 else safe_float(order.get("qty"))

        def _matches(pos: Dict) -> bool:
            tp_oid = pos.get("tp_order_id")
            tp_link = pos.get("tp_order_link_id")
            if tp_oid and oid and tp_oid == oid:
                return True
            if tp_link and order_link_id and tp_link == order_link_id:
                return True
            return False

        updated = False
        if self.hedge_mode:
            positions = self.state.state.get("positions", {})
            if not isinstance(positions, dict):
                return False
            for key in ("long", "short"):
                pos = positions.get(key, {})
                if not pos or not _matches(pos):
                    continue
                if is_closed:
                    pos["tp_order_id"] = None
                    pos["tp_order_link_id"] = None
                    pos["tp_order_price"] = None
                    pos["tp_order_qty"] = None
                    pos["tp_order_ts"] = None
                    pos["tp_trigger_seen_ts"] = None
                else:
                    pos["tp_order_id"] = oid or pos.get("tp_order_id")
                    pos["tp_order_link_id"] = order_link_id or pos.get("tp_order_link_id")
                    if price > 0:
                        pos["tp_order_price"] = price
                    if qty > 0:
                        pos["tp_order_qty"] = qty
                    pos["tp_order_ts"] = pos.get("tp_order_ts") or time.time()
                positions[key] = pos
                updated = True
            if updated:
                self.state.state["positions"] = positions
            return updated

        pos = self.state.state.get("position", {})
        if pos and _matches(pos):
            if is_closed:
                pos["tp_order_id"] = None
                pos["tp_order_link_id"] = None
                pos["tp_order_price"] = None
                pos["tp_order_qty"] = None
                pos["tp_order_ts"] = None
                pos["tp_trigger_seen_ts"] = None
            else:
                pos["tp_order_id"] = oid or pos.get("tp_order_id")
                pos["tp_order_link_id"] = order_link_id or pos.get("tp_order_link_id")
                if price > 0:
                    pos["tp_order_price"] = price
                if qty > 0:
                    pos["tp_order_qty"] = qty
                pos["tp_order_ts"] = pos.get("tp_order_ts") or time.time()
            self.state.state["position"] = pos
            return True
        return False

    def _apply_ws_position_update(self, pos: Dict) -> bool:
        if pos.get("symbol") != self.config.data.symbol:
            return False
        size = safe_float(pos.get("size"))
        side = pos.get("side")
        position_idx = safe_int(pos.get("positionIdx"))
        if self.hedge_mode:
            key = self._position_key(side, position_idx)
            if not key:
                return False
            positions = self.state.state.get("positions", {})
            pos_state = positions.get(key, {})
            if size <= 0:
                if pos_state:
                    positions[key] = {}
                    self.state.state["positions"] = positions
                    if side:
                        self._clear_entry_intent(side, force=True)
                        self._reconcile_virtual_positions_side(side, 0.0, 0.0, None, external=False)
                    return True
                return False
        else:
            if size <= 0:
                if self.state.state.get("position"):
                    self.state.state["position"] = {}
                    if side:
                        self._clear_entry_intent(side, force=True)
                        self._reconcile_virtual_positions_side(side, 0.0, 0.0, None, external=False)
                    return True
                return False

        entry_price = safe_float(pos.get("entryPrice") or pos.get("avgPrice"))
        mark_price = safe_float(pos.get("markPrice"))
        unreal_pnl = safe_float(pos.get("unrealisedPnl"))
        cum_realized = safe_float(pos.get("cumRealisedPnl"))
        stop_loss = safe_float(pos.get("stopLoss"))
        take_profit = safe_float(pos.get("takeProfit"))
        tpsl_mode = pos.get("tpslMode") or pos.get("tpSlMode")
        created_time = safe_int(pos.get("createdTime"))

        if self.hedge_mode:
            if not pos_state:
                intent = self._entry_intents().get(side, {})
                internal_exec = self._find_recent_internal_exec(side, position_idx)
                external = not bool(intent or internal_exec)
                entry_ts_ms, entry_source = self._resolve_entry_ts_ms(created_time, internal_exec, intent)
                entry_bar_time = bar_time_from_ts(entry_ts_ms / 1000.0, self.tf_seconds)
                pos_state = {
                    "side": side,
                    "size": size,
                    "entry_price": entry_price,
                    "entry_bar_time": entry_bar_time,
                    "created_ts": utc_now_str(),
                    "tp_sl_set": False,
                    "entry_ts_ms": entry_ts_ms,
                    "position_idx": position_idx,
                    "external": external,
                    "origin": "external" if external else "internal",
                    "origin_source": entry_source,
                }
                if external:
                    self.logger.warning("External position detected. Pausing new entries.")

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
                "position_idx": position_idx,
                "external": pos_state.get("external", False),
            })
            self._maybe_refresh_position_entry(pos_state, created_time)
            self._update_position_identity(pos_state, pos, side, position_idx)
            if stop_loss > 0 or take_profit > 0:
                pos_state["tp_sl_set"] = True
            positions[key] = pos_state
            self.state.state["positions"] = positions
            self._reconcile_virtual_positions_side(
                side,
                size,
                entry_price,
                safe_int(pos_state.get("entry_bar_time")) or None,
                bool(pos_state.get("external")),
            )
            intents = self._entry_intents()
            intent = intents.get(side, {})
            if intent:
                intent["filled"] = True
                intents[side] = intent
                self.state.state["entry_intent"] = intents
                filled_bar = safe_int(pos_state.get("entry_bar_time"))
                self._register_entry_intent_fill(side, intent, filled_bar or None)
            if self.tp_maker:
                if not pos_state.get("external"):
                    seeded = self._seed_tp_sl_from_order(pos_state, side)
                    latest_row = None if seeded else self.last_feature_row
                    self._ensure_tp_sl(latest_row, pos_state, side, position_idx=position_idx, state_key=key)
                    if pos_state.get("tp_target") or pos_state.get("sl_target"):
                        self._clear_entry_intent(side)
            return True

        pos_state = self.state.state.get("position", {})
        if not pos_state:
            intent = self._entry_intents().get(side, {})
            internal_exec = self._find_recent_internal_exec(side)
            external = not bool(intent or internal_exec)
            entry_ts_ms, entry_source = self._resolve_entry_ts_ms(created_time, internal_exec, intent)
            entry_bar_time = bar_time_from_ts(entry_ts_ms / 1000.0, self.tf_seconds)
            pos_state = {
                "side": side,
                "size": size,
                "entry_price": entry_price,
                "entry_bar_time": entry_bar_time,
                "created_ts": utc_now_str(),
                "tp_sl_set": False,
                "entry_ts_ms": entry_ts_ms,
                "external": external,
                "origin": "external" if external else "internal",
                "origin_source": entry_source,
            }
            if external:
                self.logger.warning("External position detected. Pausing new entries.")

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
            "external": pos_state.get("external", False),
        })
        self._maybe_refresh_position_entry(pos_state, created_time)
        self._update_position_identity(pos_state, pos, side, position_idx)
        if stop_loss > 0 or take_profit > 0:
            pos_state["tp_sl_set"] = True
        self.state.state["position"] = pos_state
        self._reconcile_virtual_positions_side(
            side,
            size,
            entry_price,
            safe_int(pos_state.get("entry_bar_time")) or None,
            bool(pos_state.get("external")),
        )
        intents = self._entry_intents()
        intent = intents.get(side, {})
        if intent:
            intent["filled"] = True
            intents[side] = intent
            self.state.state["entry_intent"] = intents
            filled_bar = safe_int(pos_state.get("entry_bar_time"))
            self._register_entry_intent_fill(side, intent, filled_bar or None)
        if self.tp_maker:
            if not pos_state.get("external"):
                seeded = self._seed_tp_sl_from_order(pos_state, side)
                latest_row = None if seeded else self.last_feature_row
                self._ensure_tp_sl(latest_row, pos_state, side)
                if pos_state.get("tp_target") or pos_state.get("sl_target"):
                    self._clear_entry_intent(side)
        return True

    def _apply_ws_execution_update(self, execution: Dict) -> bool:
        if execution.get("symbol") != self.config.data.symbol:
            return False
        exec_time = normalize_ts_ms(execution.get("execTime"))
        if exec_time <= 0:
            return False
        metrics = self.state.state.get("metrics", {})
        metrics["last_exec_time"] = max(exec_time, safe_int(metrics.get("last_exec_time")))
        self.state.state["metrics"] = metrics
        order_link_id = execution.get("orderLinkId")
        order_id = execution.get("orderId")
        side = execution.get("side")
        position_idx = safe_int(execution.get("positionIdx"))
        qty = safe_float(execution.get("execQty") or execution.get("orderQty") or execution.get("qty"))
        price = safe_float(execution.get("execPrice") or execution.get("price"))
        if self._is_internal_order_link_id(order_link_id) or self._get_state_order(order_id, order_link_id):
            self._record_internal_exec({
                "exec_time": exec_time,
                "side": side,
                "qty": qty,
                "price": price,
                "order_id": order_id,
                "order_link_id": order_link_id,
                "position_idx": position_idx,
            })
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
        self.logger.info(f"Position Mode: {self.position_mode}")
        self.logger.info(f"Feature Count: {len(self.model_features)}")
        self.logger.info(f"Window Bars: {self.window_size} | Min Feature Bars: {self.min_feature_bars}")
        self.logger.info(f"Threshold: {self.config.model.model_threshold:.3f}")
        dir_thresh_text = f"{self.config.model.direction_threshold:.3f}" if self.direction_threshold_set else "n/a"
        agg_thresh_text = f"{self.config.model.aggressive_threshold:.3f}" if self.aggressive_threshold_set else "n/a"
        self.logger.info(
            f"Direction Models: {bool(self.dir_model_long or self.dir_model_short)} | "
            f"Dir Threshold: {dir_thresh_text} | Aggressive: {agg_thresh_text} | "
            f"Dir Gate: {self.direction_gate_signals}"
        )
        self.logger.info(f"Limit Offset ATR: {self.config.strategy.base_limit_offset_atr:.3f}")
        self.logger.info(f"TP ATR: {self.config.strategy.take_profit_atr:.3f}")
        self.logger.info(f"SL ATR: {self.config.strategy.stop_loss_atr:.3f}")
        self.logger.info(
            f"TP Maker: {self.tp_maker} | TP Fallback: {self.tp_maker_fallback_sec:.1f}s"
        )
        self.logger.info(f"Order Timeout Bars: {self.config.strategy.time_limit_bars}")
        self.logger.info(f"Max Holding Bars: {self.config.strategy.max_holding_bars}")
        self.logger.info(f"Risk Per Trade: {self.config.strategy.risk_per_trade:.3f}")
        self.logger.info(
            f"OB Levels: {self.config.data.ob_levels} | Min OB Density: {self.min_ob_density_pct:.0f}% | "
            f"Trade Continuity Target: {self.min_trade_bars} | History Continuity Bars: {self.continuity_bars} | "
            f"Max Last Bar Age: {self.max_last_bar_age_sec:.0f}s"
        )
        self.logger.info(f"Bootstrap Klines Days: {self.bootstrap_klines_days}")
        self.logger.info(f"Fast Bootstrap: {self.fast_bootstrap}")
        self.logger.info(f"Dry Run: {self.dry_run}")
        self.logger.info(f"Signal Only: {self.signal_only}")
        self.logger.info(f"Exchange Leverage: {self.exchange_leverage if self.exchange_leverage else 'unchanged'}")
        if self.keys_file:
            self.logger.info(f"Keys File: {self.keys_file} | Profile: {self.keys_profile}")
        else:
            self.logger.info("Keys File: env")
        if self.log_open_orders_raw:
            self.logger.info(f"Open Orders Raw Log: {self.open_orders_log_path}")
        if self.log_positions_raw:
            self.logger.info(f"Positions Raw Log: {self.positions_log_path}")
        self.logger.info(f"Metrics Log: {self.metrics_path}")
        self.logger.info(f"Signals Log: {self.signal_log_path}")
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
        self.logger.info(f"History Writer: {self.history_writer}")
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

    def _log_positions_raw(self, reason: str) -> None:
        if not self.log_positions_raw or self.positions_raw_logged:
            return
        try:
            def _call_positions():
                return self.exchange.session.get_positions(
                    category="linear",
                    symbol=self.config.data.symbol,
                )
            resp = self.api._call("get_positions_raw", _call_positions)
            payload = {
                "ts": utc_now_str(),
                "symbol": self.config.data.symbol,
                "reason": reason,
                "response": resp,
            }
            append_jsonl(self.positions_log_path, payload)
            self.positions_raw_logged = True
            self.logger.info(f"Positions raw payload logged: {self.positions_log_path}")
        except Exception as exc:
            self.logger.error(f"Failed to log positions raw payload: {exc}")

    def _reconcile(self, reason: str) -> None:
        if self.dry_run:
            return
        now = time.time()
        state_orders = self.state.state.get("active_orders", {})
        self._log_positions_raw(reason)

        if self.use_ws_private and self.ws_private is not None:
            self._process_ws_private()
            ws_fresh = self.ws_private_last_ts > 0 and (now - self.ws_private_last_ts) <= self.ws_private_stale_sec
            rest_due = self.ws_private_rest_sec > 0 and (now - self.ws_private_last_rest) >= self.ws_private_rest_sec
            rest_needed = False
            if ws_fresh and not rest_due:
                if self.hedge_mode:
                    positions_state = self.state.state.get("positions", {})
                    if isinstance(positions_state, dict):
                        for key in ("long", "short"):
                            pos_state = positions_state.get(key, {})
                            if pos_state and safe_float(pos_state.get("size")) > 0:
                                if not safe_bool(pos_state.get("entry_verified")):
                                    rest_needed = True
                                    break
                else:
                    pos_state = self.state.state.get("position", {})
                    if pos_state and safe_float(pos_state.get("size")) > 0:
                        if not safe_bool(pos_state.get("entry_verified")):
                            rest_needed = True
            if ws_fresh and not rest_due and not rest_needed:
                self._update_closed_pnl()
                self.state.save()
                self.last_reconcile_info = {
                    "ts": utc_now_str(),
                    "reason": reason,
                    "source": "ws",
                    "open_orders": len(state_orders),
                }
                self.logger.info(f"Reconcile complete ({reason}|ws). Open orders: {len(state_orders)}")
                return

        open_orders = self.api.get_open_orders()
        position = {}
        positions = []
        if self.hedge_mode:
            positions = self.api.get_positions_details()
        else:
            position = self.api.get_position_details()
        state_orders = self.state.state.get("active_orders", {})
        if isinstance(open_orders, list):
            self._sync_tp_orders(open_orders)

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
                created_ms = normalize_ts_ms(o.get("createdTime"))
                created_bar = self.state.state.get("last_bar_time")
                if created_ms > 0:
                    created_bar = bar_time_from_ts(created_ms / 1000.0, self.tf_seconds)
                order_link_id = o.get("orderLinkId")
                external = not self._is_internal_order_link_id(order_link_id)
                state_orders[oid] = {
                    "side": o.get("side"),
                    "price": safe_float(o.get("price")),
                    "qty": safe_float(o.get("qty")),
                    "created_bar_time": created_bar,
                    "created_ts": utc_now_str(),
                    "external": external,
                    "order_link_id": order_link_id,
                }
                if external:
                    self.logger.warning(f"External order detected: {oid}")

        missing = [oid for oid in state_orders.keys() if oid not in active_ids]
        for oid in missing:
            info = state_orders.get(oid, {})
            side = info.get("side")
            order_link_id = info.get("order_link_id")
            keep_intent = False
            if side:
                keep_intent = self._position_matches_missing_order(side, info, position, positions)
            state_orders.pop(oid, None)
            if side and keep_intent:
                intents = self._entry_intents()
                intent = intents.get(side, {})
                if (
                    not intent
                    or (intent.get("order_id") != oid and (not order_link_id or intent.get("order_link_id") != order_link_id))
                ):
                    intent = {
                        "order_id": oid,
                        "order_link_id": order_link_id,
                        "price": info.get("price"),
                        "qty": info.get("qty"),
                        "tp": info.get("tp"),
                        "sl": info.get("sl"),
                        "atr_at_entry": info.get("atr_at_entry"),
                        "created_bar_time": info.get("created_bar_time"),
                        "created_ts": info.get("created_ts"),
                        "position_idx": info.get("position_idx"),
                    }
                intent["filled"] = True
                intents[side] = intent
                self.state.state["entry_intent"] = intents
                created_bar = safe_int(intent.get("created_bar_time"))
                current_bar = safe_int(self.state.state.get("last_bar_time"))
                if created_bar and current_bar:
                    filled_bar = max(created_bar, current_bar)
                else:
                    filled_bar = created_bar or current_bar
                self._register_entry_intent_fill(side, intent, filled_bar or None)
            elif side:
                self._clear_entry_intent(side, oid, order_link_id)

        self.state.state["active_orders"] = state_orders

        if self.hedge_mode:
            positions_state = self.state.state.get("positions", {})
            if not isinstance(positions_state, dict):
                positions_state = {"long": {}, "short": {}}
            seen_keys = set()
            for pos in positions:
                pos_size = safe_float(pos.get("size"))
                side = pos.get("side")
                key = self._position_key(side, safe_int(pos.get("position_idx")))
                if not key or pos_size == 0:
                    continue
                seen_keys.add(key)
                entry_price = safe_float(pos.get("avg_price"))
                mark_price = safe_float(pos.get("mark_price"))
                unreal_pnl = safe_float(pos.get("unrealized_pnl"))
                cum_realized = safe_float(pos.get("cum_realized_pnl"))
                stop_loss = safe_float(pos.get("stop_loss"))
                take_profit = safe_float(pos.get("take_profit"))
                tpsl_mode = pos.get("tpsl_mode")
                created_time = safe_int(pos.get("created_time"))
                pos_state = positions_state.get(key, {})
                if not pos_state:
                    intent = self._entry_intents().get(side, {})
                    internal_exec = self._find_recent_internal_exec(side, safe_int(pos.get("position_idx")))
                    external = not bool(intent or internal_exec)
                    entry_ts_ms, entry_source = self._resolve_entry_ts_ms(created_time, internal_exec, intent)
                    entry_bar_time = bar_time_from_ts(entry_ts_ms / 1000.0, self.tf_seconds)
                    if external:
                        self.logger.warning("External position detected. Pausing new entries.")
                    pos_state = {
                        "side": side,
                        "size": pos_size,
                        "entry_price": entry_price,
                        "entry_bar_time": entry_bar_time,
                        "created_ts": utc_now_str(),
                        "tp_sl_set": False,
                        "entry_ts_ms": entry_ts_ms,
                        "position_idx": safe_int(pos.get("position_idx")),
                        "external": external,
                        "origin": "external" if external else "internal",
                        "origin_source": entry_source,
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
                    "position_idx": safe_int(pos.get("position_idx")),
                    "external": pos_state.get("external", False),
                })
                self._maybe_refresh_position_entry(pos_state, created_time)
                self._update_position_identity(pos_state, pos, side, safe_int(pos.get("position_idx")))
                if stop_loss > 0 or take_profit > 0:
                    pos_state["tp_sl_set"] = True
                positions_state[key] = pos_state
                self._reconcile_virtual_positions_side(
                    side,
                    pos_size,
                    entry_price,
                    safe_int(pos_state.get("entry_bar_time")) or None,
                    bool(pos_state.get("external")),
                )
                intents = self._entry_intents()
                intent = intents.get(side, {})
                if intent:
                    intent["filled"] = True
                    intents[side] = intent
                    self.state.state["entry_intent"] = intents
                    filled_bar = safe_int(pos_state.get("entry_bar_time"))
                    self._register_entry_intent_fill(side, intent, filled_bar or None)
                if self.tp_maker and not pos_state.get("external"):
                    seeded = self._seed_tp_sl_from_order(pos_state, side)
                    latest_row = None if seeded else self.last_feature_row
                    self._ensure_tp_sl(latest_row, pos_state, side, position_idx=safe_int(pos.get("position_idx")), state_key=key)
                    if pos_state.get("tp_target") or pos_state.get("sl_target"):
                        self._clear_entry_intent(side)

            for key in ("long", "short"):
                if key not in seen_keys and positions_state.get(key):
                    side = "Buy" if key == "long" else "Sell"
                    positions_state[key] = {}
                    self._clear_entry_intent(side, force=True)
                    self._reconcile_virtual_positions_side(side, 0.0, 0.0, None, external=False)
            self.state.state["positions"] = positions_state
        else:
            pos_size = safe_float(position.get("size")) if position else 0.0
            if pos_size == 0 and self.state.state.get("position"):
                self.logger.info("Position closed on exchange.")
                closed_side = self.state.state.get("position", {}).get("side")
                self.state.state["position"] = {}
                if closed_side:
                    self._clear_entry_intent(closed_side, force=True)
                    self._reconcile_virtual_positions_side(closed_side, 0.0, 0.0, None, external=False)
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
                    intent = self._entry_intents().get(side, {})
                    internal_exec = self._find_recent_internal_exec(side)
                    external = not bool(intent or internal_exec)
                    entry_ts_ms, entry_source = self._resolve_entry_ts_ms(created_time, internal_exec, intent)
                    entry_bar_time = bar_time_from_ts(entry_ts_ms / 1000.0, self.tf_seconds)
                    if external:
                        self.logger.warning("External position detected. Pausing new entries.")
                    pos_state = {
                        "side": side,
                        "size": pos_size,
                        "entry_price": entry_price,
                        "entry_bar_time": entry_bar_time,
                        "created_ts": utc_now_str(),
                        "tp_sl_set": False,
                        "entry_ts_ms": entry_ts_ms,
                        "external": external,
                        "origin": "external" if external else "internal",
                        "origin_source": entry_source,
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
                    "external": pos_state.get("external", False),
                })
                self._maybe_refresh_position_entry(pos_state, created_time)
                self._update_position_identity(pos_state, position, side, 0)
                if stop_loss > 0 or take_profit > 0:
                    pos_state["tp_sl_set"] = True
                self.state.state["position"] = pos_state
                self._reconcile_virtual_positions_side(
                    side,
                    pos_size,
                    entry_price,
                    safe_int(pos_state.get("entry_bar_time")) or None,
                    bool(pos_state.get("external")),
                )
                intents = self._entry_intents()
                intent = intents.get(side, {})
                if intent:
                    intent["filled"] = True
                    intents[side] = intent
                    self.state.state["entry_intent"] = intents
                    filled_bar = safe_int(pos_state.get("entry_bar_time"))
                    self._register_entry_intent_fill(side, intent, filled_bar or None)
                if self.tp_maker and not pos_state.get("external"):
                    seeded = self._seed_tp_sl_from_order(pos_state, side)
                    latest_row = None if seeded else self.last_feature_row
                    self._ensure_tp_sl(latest_row, pos_state, side)
                    if pos_state.get("tp_target") or pos_state.get("sl_target"):
                        self._clear_entry_intent(side)
        self._update_closed_pnl()

        self.state.save()
        self.last_reconcile_info = {
            "ts": utc_now_str(),
            "reason": reason,
            "source": "rest",
            "open_orders": len(state_orders),
            "protective_orders": protective_count,
        }
        if protective_count:
            self.logger.info(
                f"Reconcile complete ({reason}). Open orders: {len(state_orders)} | Protective: {protective_count}"
            )
        else:
            self.logger.info(f"Reconcile complete ({reason}). Open orders: {len(state_orders)}")
        if self.use_ws_private and self.ws_private is not None:
            self.ws_private_last_rest = now

    def _update_closed_pnl(self) -> None:
        if self.signal_only:
            return
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

    def _compute_tp_sl_targets(
        self,
        side: str,
        entry_price: float,
        atr: float,
        tp_mult: float = 1.0,
    ) -> Tuple[float, float]:
        if entry_price <= 0 or atr <= 0:
            return 0.0, 0.0
        tp_atr = self.config.strategy.take_profit_atr * max(tp_mult, 0.0)
        sl_atr = self.config.strategy.stop_loss_atr
        tick = self.instrument.tick_size

        if side == "Buy":
            tp = entry_price + (atr * tp_atr)
            sl = entry_price - (atr * sl_atr)
            if tick > 0:
                tp = math.ceil(tp / tick) * tick
                sl = math.floor(sl / tick) * tick
        else:
            tp = entry_price - (atr * tp_atr)
            sl = entry_price + (atr * sl_atr)
            if tick > 0:
                tp = math.floor(tp / tick) * tick
                sl = math.ceil(sl / tick) * tick
        return tp, sl

    def _match_active_order_for_side(self, side: str, entry_price: float) -> Optional[Dict]:
        orders = self.state.state.get("active_orders", {})
        if not orders:
            return None
        candidates = []
        for info in orders.values():
            if info.get("side") != side:
                continue
            candidates.append(info)
        if not candidates:
            return None
        if entry_price > 0:
            return min(candidates, key=lambda info: abs(safe_float(info.get("price")) - entry_price))
        return max(candidates, key=lambda info: safe_int(info.get("created_bar_time"), 0))

    def _seed_tp_sl_from_order(self, pos: Dict, side: str) -> bool:
        entry_price = safe_float(pos.get("entry_price"))
        intents = self._entry_intents()
        intent = intents.get(side, {})
        if intent:
            intent_price = safe_float(intent.get("price"))
            atr = safe_float(intent.get("atr_at_entry"))
            if not self._entry_price_within_tolerance(entry_price, intent_price, atr):
                intent = {}
        if intent:
            atr = safe_float(intent.get("atr_at_entry"))
            tp = safe_float(intent.get("tp"))
            sl = safe_float(intent.get("sl"))
            tp_mult = safe_float(intent.get("tp_mult"), 1.0)
            if tp_mult <= 0:
                tp_mult = 1.0
            if atr > 0 and entry_price > 0:
                tp, sl = self._compute_tp_sl_targets(side, entry_price, atr, tp_mult=tp_mult)
            if tp > 0:
                pos["tp_target"] = tp
            if sl > 0:
                pos["sl_target"] = sl
            pos["tp_seed_source"] = "entry_intent"
            return tp > 0 and sl > 0
        info = self._match_active_order_for_side(side, entry_price)
        if not info:
            return False
        atr = safe_float(info.get("atr_at_entry"))
        if atr > 0 and entry_price > 0:
            tp, sl = self._compute_tp_sl_targets(side, entry_price, atr)
        else:
            tp = safe_float(info.get("tp"))
            sl = safe_float(info.get("sl"))
        if tp > 0:
            pos["tp_target"] = tp
        if sl > 0:
            pos["sl_target"] = sl
        pos["tp_seed_source"] = "active_order"
        return tp > 0 and sl > 0

    def _place_tp_limit_order(
        self,
        side: str,
        size: float,
        tp_price: float,
        position_idx: Optional[int] = None,
        order_link_id: Optional[str] = None,
    ) -> Optional[str]:
        tick = self.instrument.tick_size
        if tick > 0:
            if side == "Buy":
                tp_price = round_up(tp_price, tick)
            else:
                tp_price = round_down(tp_price, tick)
        if tp_price <= 0:
            return None
        qty = abs(size)
        if qty <= 0:
            return None
        order_side = "Sell" if side == "Buy" else "Buy"
        resp = self.api.place_limit_order(
            order_side,
            tp_price,
            qty,
            tp=0.0,
            sl=0.0,
            reduce_only=True,
            position_idx=position_idx,
            order_link_id=order_link_id,
        )
        if resp is None:
            if order_link_id:
                order = self.api.get_open_order(order_link_id=order_link_id)
                if order and order.get("orderId"):
                    return order.get("orderId")
            self.logger.error("TP limit placement failed (transport error).")
            return None
        ret_code = resp.get("retCode")
        if ret_code != 0:
            ret_msg = resp.get("retMsg")
            if self._is_duplicate_order_link(ret_code, ret_msg) and order_link_id:
                order = self.api.get_open_order(order_link_id=order_link_id)
                if order and order.get("orderId"):
                    return order.get("orderId")
            if ret_code == 110017:
                self.logger.info("TP limit rejected: position already closed; clearing local position state.")
                self._clear_position_state(side, position_idx)
                return None
            self.logger.error(f"TP limit rejected: retCode={ret_code} retMsg={ret_msg}")
            self.api.record_error(f"TP limit rejected: {ret_msg}", status_code=ret_code)
            return None
        order_id = (resp.get("result") or {}).get("orderId")
        if not order_id:
            if order_link_id:
                order = self.api.get_open_order(order_link_id=order_link_id)
                if order and order.get("orderId"):
                    return order.get("orderId")
            self.logger.error(f"TP limit response missing orderId: {resp}")
            return None
        self.logger.info(f"TP limit placed: {order_side} {qty:.6f} @ {tp_price:.6f}")
        return order_id

    def _clear_position_state(self, side: Optional[str], position_idx: Optional[int]) -> None:
        if self.hedge_mode:
            key = self._position_key(side, safe_int(position_idx))
            if not key:
                return
            positions = self.state.state.get("positions", {})
            if positions.get(key):
                positions[key] = {}
                self.state.state["positions"] = positions
                if side:
                    self._clear_entry_intent(side, force=True)
                self.state.save()
            return

        pos = self.state.state.get("position", {})
        if not pos:
            return
        if side and pos.get("side") != side:
            return
        self.state.state["position"] = {}
        if side:
            self._clear_entry_intent(side, force=True)
        self.state.save()

    def _sync_tp_orders(self, open_orders: List[Dict]) -> None:
        if not open_orders:
            return
        by_id = {o.get("orderId"): o for o in open_orders if o.get("orderId")}
        by_link = {o.get("orderLinkId"): o for o in open_orders if o.get("orderLinkId")}

        def _sync_pos(pos: Dict) -> bool:
            if not pos:
                return False
            tp_oid = pos.get("tp_order_id")
            tp_link = pos.get("tp_order_link_id")
            order = None
            if tp_oid and tp_oid in by_id:
                order = by_id.get(tp_oid)
            elif tp_link and tp_link in by_link:
                order = by_link.get(tp_link)
            if order:
                pos["tp_order_id"] = order.get("orderId") or tp_oid
                pos["tp_order_link_id"] = order.get("orderLinkId") or tp_link
                price = safe_float(order.get("price"))
                if price > 0:
                    pos["tp_order_price"] = price
                leaves = safe_float(order.get("leavesQty"))
                qty = leaves if leaves > 0 else safe_float(order.get("qty"))
                if qty > 0:
                    pos["tp_order_qty"] = qty
                pos["tp_order_ts"] = pos.get("tp_order_ts") or time.time()
                pos["tp_trigger_seen_ts"] = None
                return True
            if tp_oid or tp_link:
                pos["tp_order_id"] = None
                pos["tp_order_link_id"] = None
                pos["tp_order_ts"] = None
                pos["tp_order_price"] = None
                pos["tp_order_qty"] = None
                pos["tp_trigger_seen_ts"] = None
                return True
            return False

        updated = False
        if self.hedge_mode:
            positions = self.state.state.get("positions", {})
            if not isinstance(positions, dict):
                return
            for key in ("long", "short"):
                pos = positions.get(key, {})
                if _sync_pos(pos):
                    positions[key] = pos
                    updated = True
            if updated:
                self.state.state["positions"] = positions
            return

        pos = self.state.state.get("position", {})
        if _sync_pos(pos):
            self.state.state["position"] = pos

    def _latest_tp_reference_price(self, pos: Dict) -> float:
        last_trade = safe_float(self.bar_window.trade_builder.last_price)
        if last_trade > 0:
            return last_trade
        mark_price = safe_float(pos.get("mark_price"))
        if mark_price > 0:
            return mark_price
        return safe_float(self.bar_window.latest_close())

    def _check_tp_limit_fallback(self, now: Optional[float] = None) -> None:
        if not self.tp_maker or self.dry_run or self.signal_only:
            return
        if self._virtual_positions():
            return
        if self.tp_maker_fallback_sec <= 0:
            return
        now = now or time.time()
        updated = False
        if self.hedge_mode:
            positions = self.state.state.get("positions", {})
            if not isinstance(positions, dict):
                return
            for key in ("long", "short"):
                pos = positions.get(key, {})
                if self._check_tp_fallback_for_position(pos, now):
                    positions[key] = pos
                    updated = True
            if updated:
                self.state.state["positions"] = positions
                self.state.save()
            return

        pos = self.state.state.get("position", {})
        if self._check_tp_fallback_for_position(pos, now):
            self.state.state["position"] = pos
            self.state.save()

    def _check_tp_fallback_for_position(self, pos: Dict, now: float) -> bool:
        if not pos:
            return False
        if pos.get("tp_mode") != "maker":
            return False
        tp_order_id = pos.get("tp_order_id")
        tp_price = safe_float(pos.get("tp_order_price") or pos.get("tp_target"))
        if not tp_order_id or tp_price <= 0:
            return False
        size = safe_float(pos.get("size"))
        if size <= 0:
            pos["tp_order_id"] = None
            pos["tp_order_link_id"] = None
            pos["tp_order_ts"] = None
            pos["tp_order_price"] = None
            pos["tp_order_qty"] = None
            pos["tp_trigger_seen_ts"] = None
            return True
        side = pos.get("side")
        if side not in {"Buy", "Sell"}:
            return False
        price = self._latest_tp_reference_price(pos)
        if price <= 0:
            return False
        crossed = price >= tp_price if side == "Buy" else price <= tp_price
        trigger_ts = pos.get("tp_trigger_seen_ts")
        if crossed:
            if not trigger_ts:
                pos["tp_trigger_seen_ts"] = now
                return True
            if (now - trigger_ts) >= self.tp_maker_fallback_sec:
                position_idx = safe_int(pos.get("position_idx")) if self.hedge_mode else None
                self.logger.warning(
                    f"TP limit fallback triggered for {side} (price={price:.6f} tp={tp_price:.6f})."
                )
                if not self.dry_run:
                    if tp_order_id:
                        self.api.cancel_order(tp_order_id, pos.get("tp_order_link_id"))
                    self.api.market_close(side, abs(size), position_idx=position_idx)
                pos["tp_order_id"] = None
                pos["tp_order_link_id"] = None
                pos["tp_order_ts"] = None
                pos["tp_order_price"] = None
                pos["tp_order_qty"] = None
                pos["tp_trigger_seen_ts"] = None
                pos["tp_mode"] = "market"
                return True
        else:
            if trigger_ts is not None:
                pos["tp_trigger_seen_ts"] = None
                return True
        return False

    def _ensure_tp_sl(
        self,
        latest_row: Optional[pd.Series],
        pos: Optional[Dict] = None,
        side: Optional[str] = None,
        position_idx: Optional[int] = None,
        state_key: Optional[str] = None,
    ) -> None:
        if self._virtual_positions():
            return
        if self.dry_run:
            return
        if pos is None:
            pos = self.state.state.get("position", {})
        if not pos:
            return
        size = safe_float(pos.get("size"))
        entry_price = safe_float(pos.get("entry_price"))
        if size <= 0 or entry_price <= 0:
            return
        side = side or pos.get("side")
        atr = 0.0
        if latest_row is not None:
            atr = safe_float(latest_row.get("atr"))
        tp_target = safe_float(pos.get("tp_target"))
        sl_target = safe_float(pos.get("sl_target"))
        if tp_target <= 0:
            tp_target = safe_float(pos.get("take_profit"))
        if sl_target <= 0:
            sl_target = safe_float(pos.get("stop_loss"))
        if tp_target <= 0 or sl_target <= 0:
            if atr <= 0:
                return
            tp_target, sl_target = self._compute_tp_sl_targets(side, entry_price, atr)
        if tp_target > 0:
            pos["tp_target"] = tp_target
        if sl_target > 0:
            pos["sl_target"] = sl_target
        if pos.get("tp_sl_set"):
            if not self.tp_maker:
                return
            if pos.get("tp_mode") == "market":
                return
            tp_order_id = pos.get("tp_order_id")
            if tp_order_id:
                if not self._tp_order_needs_resize(pos, size):
                    return
                self.logger.info("TP limit size mismatch; replacing order.")
                if not self.dry_run:
                    self.api.cancel_order(tp_order_id, pos.get("tp_order_link_id"))
                new_link_id = self._new_order_link_id("T", side)
                new_tp_id = self._place_tp_limit_order(
                    side,
                    size,
                    tp_target,
                    position_idx=position_idx,
                    order_link_id=new_link_id,
                )
                if new_tp_id:
                    pos["tp_order_id"] = new_tp_id
                    pos["tp_order_link_id"] = new_link_id
                    pos["tp_order_ts"] = time.time()
                    pos["tp_order_price"] = tp_target
                    pos["tp_order_qty"] = abs(size)
                    pos["tp_trigger_seen_ts"] = None
                    pos["tp_mode"] = "maker"
                    pos["tp_sl_set"] = True
                    self.logger.info(
                        f"TP limit resized: {side} {abs(size):.6f} @ {tp_target:.6f}"
                    )
                else:
                    pos["tp_order_id"] = None
                    pos["tp_order_link_id"] = None
                    pos["tp_order_qty"] = None
                    self.logger.warning("TP limit resize failed; keeping existing SL.")
                if self.hedge_mode and state_key:
                    positions = self.state.state.get("positions", {})
                    positions[state_key] = pos
                    self.state.state["positions"] = positions
                else:
                    self.state.state["position"] = pos
                self.state.save()
                return

        if self.tp_maker:
            self.api.set_tp_sl(side, size, tp=0.0, sl=sl_target, position_idx=position_idx)
            tp_order_id = pos.get("tp_order_id")
            tp_link_id = pos.get("tp_order_link_id")
            if not tp_order_id:
                tp_link_id = self._new_order_link_id("T", side)
                tp_order_id = self._place_tp_limit_order(
                    side,
                    size,
                    tp_target,
                    position_idx=position_idx,
                    order_link_id=tp_link_id,
                )
            if tp_order_id:
                pos["tp_order_id"] = tp_order_id
                if tp_link_id and not pos.get("tp_order_link_id"):
                    pos["tp_order_link_id"] = tp_link_id
                pos["tp_order_ts"] = pos.get("tp_order_ts") or time.time()
                pos["tp_order_price"] = tp_target
                pos["tp_order_qty"] = abs(size)
                pos["tp_trigger_seen_ts"] = None
                pos["tp_mode"] = "maker"
                pos["tp_sl_set"] = True
                self.logger.info(
                    f"TP/SL set (maker TP) for position: TP={tp_target:.6f} SL={sl_target:.6f}"
                )
            else:
                self.logger.warning("TP limit placement failed; falling back to market TP.")
                self.api.set_tp_sl(side, size, tp=tp_target, sl=sl_target, position_idx=position_idx)
                pos["tp_order_id"] = None
                pos["tp_order_link_id"] = None
                pos["tp_order_qty"] = None
                pos["tp_mode"] = "market"
                pos["tp_sl_set"] = True
                self.logger.info(
                    f"TP/SL set (market TP) for position: TP={tp_target:.6f} SL={sl_target:.6f}"
                )
        else:
            self.api.set_tp_sl(side, size, tp_target, sl_target, position_idx=position_idx)
            pos["tp_sl_set"] = True
            pos["tp_mode"] = "market"
            pos["tp_order_id"] = None
            pos["tp_order_link_id"] = None
            pos["tp_order_qty"] = None
            self.logger.info(f"TP/SL set for position: TP={tp_target:.6f} SL={sl_target:.6f}")
        if self.hedge_mode and state_key:
            positions = self.state.state.get("positions", {})
            positions[state_key] = pos
            self.state.state["positions"] = positions
        else:
            self.state.state["position"] = pos
        self.state.save()

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
                self.api.cancel_order(oid, orders.get(oid, {}).get("order_link_id"))
            side = orders.get(oid, {}).get("side")
            order_link_id = orders.get(oid, {}).get("order_link_id")
            orders.pop(oid, None)
            if side:
                self._clear_entry_intent(side, oid, order_link_id, force=True)
            self.logger.info(f"Order expired and canceled: {oid}")
        self.state.state["active_orders"] = orders
        self.state.save()

    def _cancel_active_orders(self, reason: str) -> None:
        orders = self.state.state.get("active_orders", {})
        if not orders:
            return
        for oid in list(orders.keys()):
            if not self.dry_run:
                self.api.cancel_order(oid, orders.get(oid, {}).get("order_link_id"))
            side = orders.get(oid, {}).get("side")
            order_link_id = orders.get(oid, {}).get("order_link_id")
            orders.pop(oid, None)
            if side:
                self._clear_entry_intent(side, oid, order_link_id, force=True)
        self.state.state["active_orders"] = orders
        self.state.save()
        self.logger.info(f"Active orders canceled ({reason}).")

    def _cancel_orders_for_sides(self, sides: List[str], reason: str) -> None:
        if not sides:
            return
        orders = self.state.state.get("active_orders", {})
        if not orders:
            return
        sides_set = {s for s in sides if s}
        canceled = []
        for oid, info in list(orders.items()):
            if info.get("side") in sides_set:
                if not self.dry_run:
                    self.api.cancel_order(oid, info.get("order_link_id"))
                orders.pop(oid, None)
                self._clear_entry_intent(info.get("side"), oid, info.get("order_link_id"), force=True)
                canceled.append(oid)
        if canceled:
            self.state.state["active_orders"] = orders
            self.state.save()
            self.logger.info(f"Active orders canceled ({reason}): {len(canceled)}")

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
                    str(o.get("orderLinkId") or ""),
                    safe_bool(o.get("reduceOnly")),
                    safe_bool(o.get("closeOnTrigger")),
                    str(o.get("takeProfit") or ""),
                    str(o.get("stopLoss") or ""),
                    str(o.get("triggerPrice") or ""),
                )
            )
        return tuple(sorted(sig))

    def _check_position_timeout(self, bar_time: int) -> None:
        if self.hedge_mode:
            positions = self.state.state.get("positions", {})
            if not isinstance(positions, dict):
                return
            updated = False
            for key in ("long", "short"):
                pos = positions.get(key, {})
                if not pos:
                    continue
                entry_bar = pos.get("entry_bar_time")
                if entry_bar is None:
                    continue
                age_bars = int((bar_time - entry_bar) / self.tf_seconds)
                if age_bars < self.config.strategy.max_holding_bars:
                    continue
                size = safe_float(pos.get("size"))
                if size <= 0:
                    continue
                side = pos.get("side") or ("Buy" if key == "long" else "Sell")
                position_idx = safe_int(pos.get("position_idx"))
                self.logger.warning("Position timeout reached. Closing at market.")
                if not self.dry_run:
                    self.api.market_close(side, abs(size), position_idx=position_idx)
                positions[key] = {}
                if side:
                    self._clear_entry_intent(side, force=True)
                updated = True
            if updated:
                self.state.state["positions"] = positions
                self.state.save()
            return

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
        if side:
            self._clear_entry_intent(side, force=True)
        self.state.save()

    def _calc_entry_price(self, side: str, close: float, atr: float) -> float:
        offset = self.config.strategy.base_limit_offset_atr
        price = close - (atr * offset) if side == "Buy" else close + (atr * offset)
        tick = self.instrument.tick_size
        if tick > 0:
            price = round_down(price, tick) if side == "Buy" else round_up(price, tick)
        return price

    def _signal_entry_price(self, side: str, close: float, atr: float) -> float:
        return self._calc_entry_price(side, close, atr)

    def _build_signal(
        self,
        side: str,
        pred: float,
        latest_row: pd.Series,
        bar_time: int,
        threshold: float,
        ts: str,
        ts_ms: int,
    ) -> Optional[Dict[str, object]]:
        atr = safe_float(latest_row.get("atr"))
        close = safe_float(latest_row.get("close"))
        if atr <= 0 or close <= 0:
            return None
        entry_price = self._signal_entry_price(side, close, atr)
        if entry_price <= 0:
            return None
        tp, sl = self._compute_tp_sl_targets(side, entry_price, atr)
        tick_size = safe_float(getattr(self.instrument, "tick_size", 0.0))
        if tick_size > 0:
            price_decimals = step_decimals(tick_size)
            entry_price = round(entry_price, price_decimals)
            tp = round(tp, price_decimals) if tp else 0.0
            sl = round(sl, price_decimals) if sl else 0.0
        return {
            "id": f"{bar_time}-{side}-{uuid.uuid4().hex[:8]}",
            "ts": ts,
            "ts_ms": ts_ms,
            "bar_time": bar_time,
            "symbol": self.config.data.symbol,
            "side": side,
            "pred": round(float(pred), 6),
            "threshold": round(float(threshold), 6),
            "entry_price": entry_price,
            "target": entry_price,
            "tp": tp,
            "sl": sl,
            "tp_atr": round(safe_float(self.config.strategy.take_profit_atr), 6),
            "sl_atr": round(safe_float(self.config.strategy.stop_loss_atr), 6),
            "tick_size": round(tick_size, 8) if tick_size else 0.0,
            "atr": round(atr, 6),
            "close": round(close, 6),
            "timeframe": self.config.features.base_timeframe,
            "mode": "signal_only" if self.signal_only else "live",
        }

    def _record_signals(self, signals: List[Dict[str, object]]) -> None:
        if not signals:
            return
        try:
            for signal in signals:
                append_jsonl(self.signal_log_path, signal)
        except Exception as exc:
            self.logger.error(f"Failed to write signal log: {exc}")

    def _update_signals(
        self,
        bar_time: int,
        latest_row: pd.Series,
        pred_long: float,
        pred_short: float,
        threshold: float,
        dir_pred_long: float,
        dir_pred_short: float,
        dir_threshold: Optional[float],
        gate_signals: bool,
    ) -> None:
        ts = utc_now_str()
        ts_ms = int(time.time() * 1000)
        signals: List[Dict[str, object]] = []
        long_ok = True
        short_ok = True
        if gate_signals and dir_threshold is not None:
            long_ok = dir_pred_long > dir_threshold
            short_ok = dir_pred_short > dir_threshold
        if pred_long > threshold and long_ok:
            signal = self._build_signal("Buy", pred_long, latest_row, bar_time, threshold, ts, ts_ms)
            if signal:
                signals.append(signal)
        if pred_short > threshold and short_ok:
            signal = self._build_signal("Sell", pred_short, latest_row, bar_time, threshold, ts, ts_ms)
            if signal:
                signals.append(signal)
        if signals:
            self.last_signal = {
                "ts": ts,
                "ts_ms": ts_ms,
                "bar_time": bar_time,
                "signals": signals,
            }
            self._record_signals(signals)
        else:
            self.last_signal = {}

    def _calc_order_qty(self, price: float, atr: float, equity: float) -> float:
        if price <= 0 or atr <= 0 or equity <= 0:
            return 0.0
        stop_dist = atr * self.config.strategy.stop_loss_atr
        risk_dollars = equity * self.config.strategy.risk_per_trade
        qty = risk_dollars / stop_dist if stop_dist > 0 else 0.0

        max_qty_leverage = (equity * self.max_leverage) / price if price > 0 else 0.0
        max_qty_allowed = max_qty_leverage if max_qty_leverage > 0 else 0.0
        if self.instrument.max_qty > 0:
            max_qty_allowed = (
                min(max_qty_allowed, self.instrument.max_qty) if max_qty_allowed > 0 else self.instrument.max_qty
            )
        if self.instrument.max_notional > 0 and price > 0:
            max_notional_qty = self.instrument.max_notional / price
            max_qty_allowed = (
                min(max_qty_allowed, max_notional_qty) if max_qty_allowed > 0 else max_notional_qty
            )
        if max_qty_allowed > 0:
            qty = min(qty, max_qty_allowed)

        qty_step = self.instrument.qty_step
        min_qty = self.instrument.min_qty
        min_notional = self.instrument.min_notional

        min_qty_req = 0.0
        if min_notional > 0 and price > 0:
            min_qty_req = max(min_qty_req, min_notional / price)
        if min_qty > 0:
            min_qty_req = max(min_qty_req, min_qty)

        qty = max(qty, min_qty_req)

        if max_qty_allowed > 0 and min_qty_req > max_qty_allowed:
            self.logger.warning("Min size exceeds max allowed; order blocked.")
            return 0.0

        if max_qty_allowed > 0 and qty > max_qty_allowed:
            if max_qty_allowed >= min_qty_req:
                qty = max_qty_allowed
            else:
                self.logger.warning("Min size exceeds max cap; using min size.")
                qty = min_qty_req

        if qty_step > 0:
            qty = round_down(qty, qty_step)
            if qty < min_qty_req:
                qty = round_up(min_qty_req, qty_step)
            if max_qty_allowed > 0 and qty > max_qty_allowed:
                if max_qty_allowed >= min_qty_req:
                    qty = round_down(max_qty_allowed, qty_step)
                    if qty < min_qty_req:
                        qty = round_up(min_qty_req, qty_step)
                else:
                    self.logger.warning("Min size exceeds max cap after rounding; using min size.")
                    qty = round_up(min_qty_req, qty_step)

        if qty <= 0:
            self.logger.warning("Qty quantized to 0; order blocked.")
            return 0.0
        return qty

    def _place_order(self, side: str, latest_row: pd.Series, equity: float) -> Optional[str]:
        atr = safe_float(latest_row.get("atr"))
        close = safe_float(latest_row.get("close"))
        if atr <= 0 or close <= 0:
            return None

        price = self._calc_entry_price(side, close, atr)
        if price <= 0:
            return None

        qty = self._calc_order_qty(price, atr, equity)
        if qty <= 0:
            return None
        
        self.logger.info(f"Placing {side} limit: price={price:.6f} qty={qty:.6f}")
        if self.dry_run:
            return "dry_run"

        order_link_id = self._new_order_link_id("E", side)
        position_idx = 1 if self.hedge_mode and side == "Buy" else 2 if self.hedge_mode else None
        tp, sl = self._compute_tp_sl_targets(side, price, atr)
        order_tp = tp if not self.tp_maker else 0.0
        order_id: Optional[str] = None
        order_info: Optional[Dict] = None

        for attempt in range(self.postonly_retry_max + 1):
            resp = self.api.place_limit_order(
                side,
                price,
                qty,
                tp=order_tp,
                sl=sl,
                reduce_only=False,
                position_idx=position_idx,
                order_link_id=order_link_id,
            )
            if resp is None:
                order_info = self.api.get_open_order(order_link_id=order_link_id)
                if order_info and order_info.get("orderId"):
                    order_id = order_info.get("orderId")
                    break
                self.logger.error("place_limit_order returned None (exception / transport failure).")
                return None

            ret_code = resp.get("retCode")
            ret_msg = resp.get("retMsg")
            if ret_code == 0:
                order_id = (resp.get("result") or {}).get("orderId")
                if not order_id:
                    order_info = self.api.get_open_order(order_link_id=order_link_id)
                    if order_info and order_info.get("orderId"):
                        order_id = order_info.get("orderId")
                        break
                    self.logger.error(f"Bybit response missing orderId: {resp}")
                    return None
                break

            if self._is_duplicate_order_link(ret_code, ret_msg):
                order_info = self.api.get_open_order(order_link_id=order_link_id)
                if order_info and order_info.get("orderId"):
                    order_id = order_info.get("orderId")
                    break
                self.logger.error(f"Duplicate orderLinkId but open order not found: {ret_msg}")
                return None

            if self._is_post_only_reject(ret_code, ret_msg) and attempt < self.postonly_retry_max:
                if tick <= 0:
                    self.logger.error("PostOnly reject but tick size is unknown; aborting.")
                    return None
                if side == "Buy":
                    price = round_down(price - tick, tick)
                else:
                    price = round_up(price + tick, tick)
                if price <= 0:
                    self.logger.error("PostOnly retry produced invalid price; aborting.")
                    return None
                tp, sl = self._compute_tp_sl_targets(side, price, atr)
                order_tp = tp if not self.tp_maker else 0.0
                self.logger.warning(f"PostOnly reject; nudging price to {price:.6f}.")
                continue

            self.logger.error(f"Bybit rejected order: retCode={ret_code} retMsg={ret_msg}")
            self.api.record_error(f"Order rejected: {ret_msg}", status_code=ret_code)
            return None

        if not order_id:
            self.logger.error("Failed to obtain orderId after placement attempts.")
            return None

        confirmed = self._confirm_order_resting(order_id, order_link_id)
        if confirmed:
            info_price = safe_float(confirmed.get("price"))
            if info_price > 0:
                price = info_price
            info_qty = safe_float(confirmed.get("leavesQty"))
            if info_qty <= 0:
                info_qty = safe_float(confirmed.get("qty"))
            if info_qty > 0:
                qty = info_qty
        else:
            self.logger.warning("Order not confirmed resting; tracking anyway.")

        self.state.state["active_orders"][order_id] = {
            "side": side,
            "price": price,
            "qty": qty,
            "tp": tp,
            "sl": sl,
            "atr_at_entry": atr,
            "created_bar_time": self.state.state.get("last_bar_time"),
            "created_ts": utc_now_str(),
            "external": False,
            "position_idx": position_idx,
            "order_link_id": order_link_id,
        }
        intents = self._entry_intents()
        intents[side] = {
            "order_id": order_id,
            "order_link_id": order_link_id,
            "price": price,
            "qty": qty,
            "tp": tp,
            "sl": sl,
            "atr_at_entry": atr,
            "created_bar_time": self.state.state.get("last_bar_time"),
            "created_ts": utc_now_str(),
        }
        self.state.state["entry_intent"] = intents
        self.state.save()

        return order_id

    def _place_market_order(
        self,
        side: str,
        latest_row: pd.Series,
        equity: float,
        tp_mult: float = 1.0,
    ) -> Optional[str]:
        atr = safe_float(latest_row.get("atr"))
        close = safe_float(latest_row.get("close"))
        if atr <= 0 or close <= 0:
            return None

        price = close
        qty = self._calc_order_qty(price, atr, equity)
        if qty <= 0:
            return None

        tp, sl = self._compute_tp_sl_targets(side, price, atr, tp_mult=tp_mult)
        self.logger.info(f"Aggressive {side} market: price={price:.6f} qty={qty:.6f} tp={tp:.6f} sl={sl:.6f}")
        if self.dry_run:
            return "dry_run"

        order_link_id = self._new_order_link_id("A", side)
        position_idx = 1 if self.hedge_mode and side == "Buy" else 2 if self.hedge_mode else None
        resp = self.api.place_market_order(
            side,
            qty,
            reduce_only=False,
            position_idx=position_idx,
            order_link_id=order_link_id,
        )
        if resp is None:
            self.logger.error("place_market_order returned None (exception / transport failure).")
            return None

        ret_code = resp.get("retCode")
        ret_msg = resp.get("retMsg")
        if ret_code != 0:
            self.logger.error(f"Bybit rejected market order: retCode={ret_code} retMsg={ret_msg}")
            self.api.record_error(f"Market order rejected: {ret_msg}", status_code=ret_code)
            return None

        order_id = (resp.get("result") or {}).get("orderId")
        intents = self._entry_intents()
        intents[side] = {
            "order_id": order_id,
            "order_link_id": order_link_id,
            "price": price,
            "qty": qty,
            "tp": tp,
            "sl": sl,
            "atr_at_entry": atr,
            "tp_mult": tp_mult,
            "aggressive": True,
            "created_bar_time": self.state.state.get("last_bar_time"),
            "created_ts": utc_now_str(),
        }
        self.state.state["entry_intent"] = intents
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

    def _compute_direction_predictions(self, row: pd.Series) -> Tuple[float, float, str]:
        if self.dir_model_long is None and self.dir_model_short is None:
            return 1.0, 1.0, "pass"
        X = row[self.model_features].values.reshape(1, -1)
        source = "model"
        pred_long = 1.0
        pred_short = 1.0
        if self.dir_model_long is not None:
            try:
                pred_long = float(self.dir_model_long.predict(X)[0])
                if not np.isfinite(pred_long):
                    pred_long = 0.0
            except Exception as exc:
                self.logger.error(f"Direction long prediction failed: {exc}")
                pred_long = 0.0
                source = "error"
        else:
            if source != "error":
                source = "partial"
        if self.dir_model_short is not None:
            try:
                pred_short = float(self.dir_model_short.predict(X)[0])
                if not np.isfinite(pred_short):
                    pred_short = 0.0
            except Exception as exc:
                self.logger.error(f"Direction short prediction failed: {exc}")
                pred_short = 0.0
                source = "error"
        else:
            if source != "error":
                source = "partial"
        return pred_long, pred_short, source

    def _process_bar(self, bar_time: int) -> None:
        if self.state.state.get("last_processed_bar_time") == bar_time:
            return
        self.state.state["last_processed_bar_time"] = bar_time
        self.state.state["last_bar_time"] = bar_time
        try:
            bars_df = self.bar_window.bars.copy()
            if bars_df.empty:
                return
            if bars_df["bar_time"].duplicated().any():
                bars_df = (
                    bars_df.drop_duplicates(subset=["bar_time"], keep="last")
                    .sort_values("bar_time")
                    .tail(self.window_size)
                    .reset_index(drop=True)
                )
                self.bar_window.load_bars(bars_df)
            bars_all = bars_df
            if len(bars_df) < self.min_feature_bars:
                self.logger.warning(f"Warmup: {len(bars_df)} bars < {self.min_feature_bars}; padding history.")
                seed = self._get_seed_price()
                padded = self._pad_history_to_min_bars(bars_df, self.min_feature_bars, seed)
                if padded is None or padded.empty or len(padded) < self.min_feature_bars:
                    return
                self.bar_window.load_bars(padded)
                bars_df = padded
                bars_all = bars_df

            bar_rows = bars_df[bars_df["bar_time"] == bar_time]
            if bar_rows.empty:
                seed = self._get_seed_price()
                if seed > 0:
                    synth_df = pd.DataFrame([self._synthetic_bar_row(bar_time, seed)])
                    merged = pd.concat([bars_df, synth_df], ignore_index=True)
                    merged = self._normalize_bars(merged)
                    if merged is not None and not merged.empty:
                        self.bar_window.load_bars(merged)
                        bars_df = merged
                        bars_all = bars_df
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
                self.logger.info(f"Bar {pd.to_datetime(bar_time, unit='s')} has no trades.")

            bars_df = bars_all.copy()
            if bars_df.empty:
                self.logger.warning("No bars available after loading window.")
                return
            ready, reason = self._history_continuity_ready(bars_all)
            if not ready:
                self.logger.info(f"History continuity warmup: {reason}. Attempting recovery.")
                self._fill_history_gaps()
                bars_df = self.bar_window.bars.copy()
                bars_all = bars_df
                ready, reason = self._history_continuity_ready(bars_all)
                if not ready:
                    self.logger.info(f"History continuity still not ready: {reason}.")
                    return
            df_feat = self.feature_engine.calculate_features(bars_df)
            df_feat = df_feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)

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
            self.last_feature_vector = self._serialize_feature_vector(latest)
            self.last_feature_row = latest.copy()

            recent = bars_df.tail(min(24, len(bars_df)))
            ob_density = (recent["ob_spread_mean"] > 0).sum() / max(len(recent), 1) * 100
            ob_ok = ob_density >= self.min_ob_density_pct
            if not ob_ok:
                self.logger.warning(f"OB density {ob_density:.0f}% below threshold {self.min_ob_density_pct:.0f}%.")

            if self.min_trade_bars > 0 and self.continuous_trade_bars < self.min_trade_bars:
                self.logger.info(
                    f"Trade continuity below target: {self.continuous_trade_bars}/{self.min_trade_bars} bars."
                )

            self._log_feature_vector(latest)
            for warning in self.sanity.check(latest):
                self.logger.warning(f"[DATA INTEGRITY] {warning}")
            zscores = self.feature_baseline.update(latest)
            self._log_feature_zscores(zscores)

            pred_long, pred_short = self._compute_predictions(latest)
            threshold = self.config.model.model_threshold
            dir_pred_long, dir_pred_short, dir_source = self._compute_direction_predictions(latest)
            dir_threshold = self.config.model.direction_threshold if self.direction_threshold_set else None
            aggressive_threshold = self.config.model.aggressive_threshold if self.aggressive_threshold_set else None
            gate_signals = self.direction_gate_signals and dir_threshold is not None
            self.health.record_prediction(max(pred_long, pred_short))
            self.last_prediction = {
                "bar_time": bar_time,
                "pred_long": pred_long,
                "pred_short": pred_short,
                "threshold": threshold,
            }
            self.last_direction = {
                "bar_time": bar_time,
                "pred_long": dir_pred_long,
                "pred_short": dir_pred_short,
                "threshold": dir_threshold,
                "aggressive_threshold": aggressive_threshold,
                "source": dir_source,
            }
            self._update_signals(
                bar_time,
                latest,
                pred_long,
                pred_short,
                threshold,
                dir_pred_long,
                dir_pred_short,
                dir_threshold,
                gate_signals,
            )

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
                f"spread={spread_pct:.4%} predL={pred_long:.3f} predS={pred_short:.3f} "
                f"dirL={dir_pred_long:.3f} dirS={dir_pred_short:.3f} | "
                f"{self._data_health()} | Regime: {regime} | Sentiment: {sentiment} | Health: {self.health.check_health()}"
            )

            drift_alerts = self.drift_monitor.update(latest, {"pred_long": pred_long, "pred_short": pred_short})
            self.last_drift_alerts = drift_alerts
            self.last_drift_time = utc_now_str()
            if drift_alerts:
                self.logger.warning("Drift alerts: " + ", ".join(drift_alerts))

            if self.signal_only:
                self.last_regime = regime
                self.last_sentiment = sentiment
                self.last_health = self.health.check_health()
                return

            bar_high = safe_float(latest.get("high"))
            bar_low = safe_float(latest.get("low"))
            bar_close = close
            self._manage_virtual_positions(bar_time, bar_high, bar_low, bar_close)

            if trade_count == 0:
                self.logger.info("No trades in bar; orders deferred.")
                return
            if not ob_ok:
                return

            if spread_pct > self.config.live.max_spread_pct:
                self.logger.warning(f"Spread too wide ({spread_pct:.4%}). Orders paused.")
                return

            self._expire_orders(bar_time)
            self._check_position_timeout(bar_time)

            long_pos, short_pos = self._get_positions()
            external_long = self._is_external_position(long_pos)
            external_short = self._is_external_position(short_pos)
            external_present = external_long or external_short
            long_size = self._position_size(long_pos) if not external_long else 0.0
            short_size = self._position_size(short_pos) if not external_short else 0.0
            has_long = long_size > 0
            has_short = short_size > 0
            has_position = has_long or has_short
            max_positions = safe_int(self.config.strategy.max_positions, 1)
            if max_positions <= 0:
                max_positions = 1
            open_positions = self._virtual_positions_count()
            daily_pnl = safe_float(self.state.state.get("daily", {}).get("realized_pnl"))
            drawdown_limit = -equity * self.config.live.max_daily_drawdown_pct
            if daily_pnl < drawdown_limit:
                self.logger.critical("Daily drawdown limit exceeded. Trading paused.")
                if has_position and not self.dry_run:
                    if self.hedge_mode:
                        if has_long:
                            idx = safe_int(long_pos.get("position_idx")) or 1
                            self.api.market_close("Buy", abs(long_size), position_idx=idx)
                        if has_short:
                            idx = safe_int(short_pos.get("position_idx")) or 2
                            self.api.market_close("Sell", abs(short_size), position_idx=idx)
                    else:
                        size = safe_float(long_pos.get("size"))
                        side = long_pos.get("side") or "Buy"
                        if size > 0:
                            self.api.market_close(side, abs(size))
                self._cancel_active_orders("drawdown")
                self.trade_enabled = False
                return

            if external_present and self.state.state.get("active_orders"):
                self._cancel_active_orders("external_position")

            if has_position:
                if self.hedge_mode:
                    if has_long:
                        idx = safe_int(long_pos.get("position_idx")) or 1
                        self._ensure_tp_sl(latest, long_pos, "Buy", position_idx=idx, state_key="long")
                    if has_short:
                        idx = safe_int(short_pos.get("position_idx")) or 2
                        self._ensure_tp_sl(latest, short_pos, "Sell", position_idx=idx, state_key="short")
                else:
                    self._ensure_tp_sl(latest)
                if open_positions >= max_positions:
                    return

            if open_positions >= max_positions:
                return

            if external_present:
                self.logger.warning("External position present. Pausing new entries.")
                return

            last_trade_ts = safe_float(self.state.state.get("last_trade_ts"), 0.0)
            if last_trade_ts > 0 and (time.time() - last_trade_ts) > self.data_lag_error:
                self.logger.error("Trade data lag exceeds error threshold. Orders paused.")
                return

            if not self.trade_enabled:
                return

            orders = self.state.state.get("active_orders", {})
            has_buy = any(info.get("side") == "Buy" for info in orders.values())
            has_sell = any(info.get("side") == "Sell" for info in orders.values())

            aggressive_long = False
            aggressive_short = False
            if self.aggressive_threshold_set and aggressive_threshold is not None and dir_source != "pass":
                aggressive_long = dir_pred_long > aggressive_threshold
                aggressive_short = dir_pred_short > aggressive_threshold

            if aggressive_long or aggressive_short:
                if not self.hedge_mode and aggressive_long and aggressive_short:
                    if dir_pred_long >= dir_pred_short:
                        aggressive_short = False
                    else:
                        aggressive_long = False
                sides = []
                if aggressive_long:
                    sides.append("Buy")
                if aggressive_short:
                    sides.append("Sell")
                if sides:
                    self._cancel_orders_for_sides(sides, "aggressive_override")
                if aggressive_long:
                    self._place_market_order("Buy", latest, equity, tp_mult=10.0)
                if aggressive_short:
                    self._place_market_order("Sell", latest, equity, tp_mult=10.0)
                self.last_regime = regime
                self.last_sentiment = sentiment
                self.last_health = self.health.check_health()
                return

            long_ok = True
            short_ok = True
            if gate_signals and dir_threshold is not None:
                long_ok = dir_pred_long > dir_threshold
                short_ok = dir_pred_short > dir_threshold

            if pred_long > threshold and long_ok and not has_buy:
                self._place_order("Buy", latest, equity)
            if pred_short > threshold and short_ok and not has_sell:
                self._place_order("Sell", latest, equity)
            self.last_regime = regime
            self.last_sentiment = sentiment
            self.last_health = self.health.check_health()
        finally:
            self.state.save()
            self._save_history()

    def _data_health_metrics(self) -> Dict[str, float]:
        bars = self.bar_window.bars
        if bars.empty:
            return {
                "bars": 0,
                "bars_total": 0,
                "bars_zero_trade": 0,
                "macro_pct": 0.0,
                "ob_density_pct": 0.0,
                "trade_cont": float(self.continuous_trade_bars),
            }
        trade_bars = bars[bars["trade_count"] > 0]
        n_trade = len(trade_bars)
        total = len(bars)
        macro_target = max(self.min_feature_bars, 1)
        macro_pct = min(100.0, (total / macro_target) * 100) if macro_target else 0.0
        window = min(24, total)
        if window == 0:
            ob_density = 0.0
        else:
            recent = bars.tail(window)
            ob_density = (recent["ob_spread_mean"] > 0).sum() / window * 100
        return {
            "bars": int(n_trade),
            "bars_total": int(total),
            "bars_zero_trade": int(total - n_trade),
            "macro_pct": float(macro_pct),
            "ob_density_pct": float(ob_density),
            "trade_cont": float(self.continuous_trade_bars),
        }

    def _serialize_feature_vector(self, row: pd.Series) -> Dict[str, object]:
        names = list(self.model_features)
        values: List[Optional[float]] = []
        for name in names:
            raw = row.get(name)
            try:
                val = float(raw)
            except Exception:
                val = None
            if val is None or math.isnan(val) or math.isinf(val):
                values.append(None)
            else:
                values.append(round(val, 6))
        return {"names": names, "values": values}

    def _write_metrics(self, payload: Dict[str, object]) -> None:
        try:
            append_jsonl(self.metrics_path, payload)
        except Exception as exc:
            self.logger.error(f"Failed to write metrics log: {exc}")

    def _log_heartbeat(self) -> None:
        last_bar = self.state.state.get("last_bar_time")
        last_trade_ts = safe_float(self.state.state.get("last_trade_ts"))
        last_close = self.bar_window.latest_close()
        now = time.time()
        lag_trade = now - last_trade_ts if last_trade_ts > 0 else None
        lag_bar = now - last_bar if last_bar else None
        ob_ts = self.bar_window.ob_agg.last_snapshot_ts
        ob_lag = (time.time() * 1000 - ob_ts) / 1000.0 if ob_ts else None
        lag_trade_str = f"{lag_trade:.1f}s" if lag_trade is not None else "n/a"
        lag_bar_str = f"{lag_bar:.1f}s" if lag_bar is not None else "n/a"
        ob_lag_str = f"{ob_lag:.1f}s" if ob_lag is not None else "n/a"

        long_pos, short_pos = self._get_positions()
        long_size = self._position_size(long_pos)
        short_size = self._position_size(short_pos)
        has_long = long_size > 0
        has_short = short_size > 0
        if self.hedge_mode:
            if has_long and has_short:
                pos_label = f"Long {long_size:.4f} / Short {short_size:.4f}"
            elif has_long:
                pos_label = f"Long {long_size:.4f}"
            elif has_short:
                pos_label = f"Short {short_size:.4f}"
            else:
                pos_label = "Flat"
            pos_side = "Hedge" if has_long and has_short else ("Buy" if has_long else "Sell" if has_short else "Flat")
            pos_size = long_size - short_size
            unreal = safe_float(long_pos.get("unrealized_pnl")) + safe_float(short_pos.get("unrealized_pnl"))
        else:
            pos = self.state.state.get("position", {})
            pos_size = safe_float(pos.get("size")) if pos else 0.0
            pos_side = pos.get("side") if pos else "Flat"
            pos_label = f"{pos_side} {pos_size:.4f}"
            unreal = safe_float(pos.get("unrealized_pnl")) if pos else 0.0
        daily_pnl = safe_float(self.state.state.get("daily", {}).get("realized_pnl"))
        if self.signal_only:
            daily_pnl = 0.0

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
            f"pos={pos_label} | "
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

        if equity is not None:
            if not self.hedge_mode:
                pos = self.state.state.get("position", {})
                if pos:
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
            else:
                close = self.bar_window.latest_close()
                if close is not None and (has_long ^ has_short):
                    pos = long_pos if has_long else short_pos
                    entry_price = safe_float(pos.get("entry_price"))
                    size = self._position_size(pos)
                    if entry_price > 0 and size > 0:
                        side = 1.0 if has_long else -1.0
                        local_unreal = (close - entry_price) * size * side
                        diff = abs(local_unreal - unreal)
                        if diff > max(1.0, equity * 0.001):
                            self.logger.warning(
                                f"PnL mismatch. local={local_unreal:.2f} exch={unreal:.2f} diff={diff:.2f}"
                            )

        pos_ref = {}
        if self.hedge_mode:
            if has_long and not has_short:
                pos_ref = long_pos
            elif has_short and not has_long:
                pos_ref = short_pos
        else:
            pos_ref = self.state.state.get("position", {})

        position_payload = {
            "side": pos_side,
            "size": pos_size,
            "entry_price": safe_float(pos_ref.get("entry_price")) if pos_ref else 0.0,
            "mark_price": safe_float(pos_ref.get("mark_price")) if pos_ref else 0.0,
            "unreal_pnl": unreal,
            "stop_loss": safe_float(pos_ref.get("stop_loss")) if pos_ref else 0.0,
            "take_profit": safe_float(pos_ref.get("take_profit")) if pos_ref else 0.0,
            "tpsl_mode": pos_ref.get("tpsl_mode") if pos_ref else None,
            "external": bool(pos_ref.get("external")) if pos_ref else False,
        }
        if self.hedge_mode and has_long and has_short:
            position_payload["size"] = long_size + short_size
        positions_payload = None
        if self.hedge_mode:
            positions_payload = {
                "long": {
                    "side": long_pos.get("side"),
                    "size": long_size,
                    "entry_price": safe_float(long_pos.get("entry_price")),
                    "mark_price": safe_float(long_pos.get("mark_price")),
                    "unreal_pnl": safe_float(long_pos.get("unrealized_pnl")),
                    "stop_loss": safe_float(long_pos.get("stop_loss")),
                "take_profit": safe_float(long_pos.get("take_profit")),
                "tpsl_mode": long_pos.get("tpsl_mode"),
                "position_idx": safe_int(long_pos.get("position_idx")),
                "external": bool(long_pos.get("external")),
            },
            "short": {
                "side": short_pos.get("side"),
                "size": short_size,
                "entry_price": safe_float(short_pos.get("entry_price")),
                "mark_price": safe_float(short_pos.get("mark_price")),
                "unreal_pnl": safe_float(short_pos.get("unrealized_pnl")),
                "stop_loss": safe_float(short_pos.get("stop_loss")),
                "take_profit": safe_float(short_pos.get("take_profit")),
                "tpsl_mode": short_pos.get("tpsl_mode"),
                "position_idx": safe_int(short_pos.get("position_idx")),
                "external": bool(short_pos.get("external")),
            },
        }

        metrics = {
            "ts": utc_now_str(),
            "symbol": self.config.data.symbol,
            "position_mode": self.position_mode,
            "model": {
                "dir": str(self.model_dir),
                "keys_profile": self.keys_profile,
            },
            "startup_time": self.start_time_utc,
            "uptime_sec": round(now - self.start_time, 2),
            "trade_enabled": self.trade_enabled,
            "dry_run": self.dry_run,
            "signal_only": self.signal_only,
            "prediction": self.last_prediction,
            "direction": self.last_direction,
            "signal": self.last_signal,
            "feature_vector": self.last_feature_vector,
            "position": position_payload,
            "positions": positions_payload,
            "orders": {
                "open_orders": len(self.state.state.get("active_orders", {})),
                "last_reconcile": self.last_reconcile_info,
            },
            "market": {
                "last_close": last_close,
                "last_bar_time": last_bar,
            },
            "equity": equity,
            "daily_pnl": daily_pnl,
            "latency": {
                "rest_avg_ms": round(latency_avg, 2),
                "rest_max_ms": round(latency_max, 2),
                "ws_trade_ms": round(ws_trade_latency, 2) if ws_trade_latency is not None else None,
                "ws_ob_ms": round(ws_ob_latency, 2) if ws_ob_latency is not None else None,
                "lag_trade_sec": round(lag_trade, 2) if lag_trade is not None else None,
                "lag_bar_sec": round(lag_bar, 2) if lag_bar is not None else None,
                "ob_lag_sec": round(ob_lag, 2) if ob_lag is not None else None,
            },
            "data_health": self._data_health_metrics(),
            "health": {
                "status": health,
                "sentiment": sentiment,
                "regime": self.last_regime,
                "last_prediction": self.last_prediction,
            },
            "drift": {
                "alerts": self.last_drift_alerts,
                "last_alert_time": self.last_drift_time,
            },
            "errors": {
                "runtime_count": self.runtime_error_count,
                "last_runtime_error": self.last_runtime_error,
                "last_runtime_error_time": self.last_runtime_error_ts,
                "api_count": self.api.error_count,
                "last_api_error": self.api.last_error,
                "last_api_error_time": self.api.last_error_ts,
            },
        }
        self._write_metrics(metrics)

    def run(self) -> None:
        last_trade_poll = 0.0
        last_ob_poll = 0.0
        last_reconcile = 0.0
        last_heartbeat = 0.0
        last_instr_refresh = 0.0
        last_time_sync = 0.0
        last_tp_fallback_check = 0.0
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
                        else:
                            ob_lag = (now - self.ws_ob_last_ts) if self.ws_ob_last_ts > 0 else None
                            if ob_lag is not None and ob_lag > self.data_lag_error:
                                self.logger.warning("WS orderbook stale; fetching REST snapshot.")
                                ob = self.api.fetch_orderbook(limit=self.config.data.ob_levels)
                                if isinstance(ob, dict) and ob:
                                    self.bar_window.ingest_orderbook(ob)
                                self._restart_websockets("orderbook_lag")
                            elif ob_lag is not None and ob_lag > self.data_lag_warn:
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
                        trade_lag = (now - self.ws_trade_last_ts) if self.ws_trade_last_ts > 0 else None
                        if (not new_trades) and trade_lag is not None and trade_lag > self.data_lag_error:
                            self.logger.warning("WS trade stream stale; fetching REST trades.")
                            trades_df = self.api.fetch_recent_trades(limit=1000)
                            if isinstance(trades_df, pd.DataFrame) and not trades_df.empty:
                                new_trades = self._filter_new_trades(trades_df)
                            self._restart_websockets("trade_lag")
                    else:
                        trades_df = self.api.fetch_recent_trades(limit=1000)
                        if isinstance(trades_df, pd.DataFrame) and not trades_df.empty:
                            new_trades = self._filter_new_trades(trades_df)
                    self.bar_window.ingest_trades(new_trades, now, on_close=self._process_bar)
                    last_trade_poll = now

                if self.tp_maker and (now - last_tp_fallback_check) >= 1.0:
                    self._check_tp_limit_fallback(now=now)
                    last_tp_fallback_check = now

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
                self.runtime_error_count += 1
                self.last_runtime_error = safe_log_message(exc)
                self.last_runtime_error_ts = time.time()
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
    parser.add_argument("--signal-only", action="store_true", help="Emit model signals only (no orders)")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--log-features", action="store_true", help="Log model feature vector each bar")
    parser.add_argument("--log-feature-z", action="store_true", help="Log per-feature z-scores each bar")
    parser.add_argument("--keys-file", type=str, default="", help="Path to JSON file with API keys")
    parser.add_argument("--keys-profile", type=str, default="default", help="Profile name inside keys file")
    parser.add_argument(
        "--broker-id",
        type=str,
        default=os.getenv("BYBIT_BROKER_ID", DEFAULT_BROKER_ID),
        help="Optional Bybit brokerId for order/cancel requests",
    )
    parser.add_argument("--log-open-orders-raw", action="store_true", help="Log raw open orders payload when it changes")
    parser.add_argument("--open-orders-log-path", type=str, default="", help="Path for raw open orders JSONL log")
    parser.add_argument("--log-positions-raw", action="store_true", help="Log raw positions payload once")
    parser.add_argument("--positions-log-path", type=str, default="", help="Path for raw positions JSONL log")
    parser.add_argument("--trade-poll-sec", type=float, default=DEFAULT_TRADE_POLL_SEC)
    parser.add_argument("--ob-poll-sec", type=float, default=DEFAULT_OB_POLL_SEC)
    parser.add_argument("--reconcile-sec", type=float, default=DEFAULT_RECONCILE_SEC)
    parser.add_argument("--heartbeat-sec", type=float, default=DEFAULT_HEARTBEAT_SEC)
    parser.add_argument("--instr-refresh-sec", type=float, default=DEFAULT_INSTR_REFRESH_SEC)
    parser.add_argument("--data-lag-warn-sec", type=float, default=DEFAULT_DATA_LAG_WARN_SEC)
    parser.add_argument("--data-lag-error-sec", type=float, default=DEFAULT_DATA_LAG_ERROR_SEC)
    parser.add_argument("--ob-levels", type=int, default=50)
    parser.add_argument("--min-ob-density-pct", type=float, default=DEFAULT_MIN_OB_DENSITY_PCT)
    parser.add_argument("--min-trade-bars", type=int, default=DEFAULT_MIN_TRADE_BARS)
    parser.add_argument("--continuity-bars", type=int, default=DEFAULT_CONTINUITY_BARS)
    parser.add_argument("--max-last-bar-age-sec", type=float, default=DEFAULT_MAX_LAST_BAR_AGE_SEC)
    parser.add_argument("--max-leverage", type=float, default=DEFAULT_MAX_LEVERAGE)
    parser.add_argument("--exchange-leverage", type=int, default=0, help="Set exchange leverage (0 disables)")
    parser.add_argument("--testnet", action="store_true", help="Use Bybit testnet endpoints")
    parser.add_argument(
        "--position-mode",
        type=str,
        default=DEFAULT_POSITION_MODE,
        choices=["oneway", "hedge"],
        help="Position mode: oneway or hedge",
    )
    parser.add_argument(
        "--tp-maker",
        action="store_true",
        default=DEFAULT_TP_MAKER,
        help="Use maker TP limit with fallback to market",
    )
    parser.add_argument(
        "--tp-maker-fallback-sec",
        type=float,
        default=DEFAULT_TP_MAKER_FALLBACK_SEC,
        help="Seconds after TP breach before market fallback",
    )
    parser.add_argument("--use-ws-trades", action="store_true", default=True, help="Use WebSocket trade stream")
    parser.add_argument("--use-ws-ob", action="store_true", default=True, help="Use WebSocket orderbook stream")
    parser.add_argument("--use-ws-private", action="store_true", default=True, help="Use private WebSocket streams")
    parser.add_argument(
        "--ws-private-topics",
        type=str,
        default="order,execution,position,wallet",
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
    parser.add_argument(
        "--history-writer",
        dest="history_writer",
        action="store_true",
        help="Write shared history CSV (default).",
    )
    parser.add_argument(
        "--no-history-writer",
        dest="history_writer",
        action="store_false",
        help="Disable history CSV writes for this process.",
    )
    parser.set_defaults(history_writer=True)
    parser.add_argument(
        "--fast-bootstrap",
        action="store_true",
        default=DEFAULT_FAST_BOOTSTRAP,
        help="Bypass continuity gate and fill missing recent bars from klines",
    )
    parser.add_argument("--metrics-log-path", type=str, default=DEFAULT_METRICS_LOG_PATH)
    parser.add_argument("--signal-log-path", type=str, default=DEFAULT_SIGNAL_LOG_PATH)
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
    if hasattr(args, "position_mode") and args.position_mode:
        args.position_mode = args.position_mode.lower()
    return args


if __name__ == "__main__":
    cli_args = parse_args()
    trader = LiveTradingV2(cli_args)
    trader.run()
