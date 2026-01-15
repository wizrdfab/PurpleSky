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
import csv
import json
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

try:
    from pybit.unified_trading import HTTP as BybitHTTP
except Exception:
    BybitHTTP = None

try:
    import joblib
except Exception:
    joblib = None

try:
    from translations import get_language, get_dashboard_translations, detect_system_language, set_language
except ImportError:
    # Fallback if translations module not available
    def get_language():
        return "en"
    def get_dashboard_translations():
        return {}
    def detect_system_language():
        return "en"
    def set_language(lang):
        pass


def tail_jsonl(path: Path, limit: int = 200, max_bytes: int = 512 * 1024):
    if not path.exists() or path.stat().st_size == 0:
        return []
    size = path.stat().st_size
    block = min(size, max_bytes)
    with open(path, "rb") as f:
        f.seek(-block, os.SEEK_END)
        data = f.read(block)
    lines = data.splitlines()
    if size > block and lines:
        lines = lines[1:]
    records = []
    for line in lines[-limit:]:
        try:
            records.append(json.loads(line.decode("utf-8")))
        except Exception:
            continue
    return records


def tail_csv_dicts(path: Path, limit: int = 500, max_bytes: int = 2 * 1024 * 1024):
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            header_line = f.readline().strip()
    except Exception:
        return []
    if not header_line:
        return []
    try:
        header = next(csv.reader([header_line]))
    except Exception:
        return []
    if not header:
        return []
    size = path.stat().st_size
    block = min(size, max_bytes)
    try:
        with open(path, "rb") as f:
            f.seek(-block, os.SEEK_END)
            data = f.read(block)
    except Exception:
        return []
    lines = data.splitlines()
    if size > block and lines:
        lines = lines[1:]
    rows: List[str] = []
    for line in lines:
        try:
            text = line.decode("utf-8", errors="ignore").strip()
        except Exception:
            continue
        if not text:
            continue
        if text == header_line:
            continue
        rows.append(text)
    if not rows:
        return []
    rows = rows[-limit:]
    parsed = []
    try:
        reader = csv.reader(rows)
        for row in reader:
            if not row or len(row) < len(header):
                continue
            parsed.append({k: v for k, v in zip(header, row)})
    except Exception:
        return []
    return parsed


def read_latest(path: Path):
    records = tail_jsonl(path, limit=1, max_bytes=64 * 1024)
    return records[-1] if records else {}


def safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def maybe_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def safe_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def latest_bar_time_from_csv(path: Path) -> int:
    rows = tail_csv_dicts(path, limit=3, max_bytes=64 * 1024)
    latest = 0
    for row in rows:
        bar_time = safe_int(row.get("bar_time"), 0)
        if bar_time > latest:
            latest = bar_time
    return latest


def safe_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


DEFAULT_DIRECTION_THRESHOLD = 0.5
DEFAULT_AGGRESSIVE_THRESHOLD = 0.8
MODEL_META_CACHE = {}
MODEL_META_LOCK = threading.Lock()


def parse_ts(ts: str):
    if not ts:
        return None
    try:
        return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    except Exception:
        return None


def day_key_from_ms(ts_ms: int) -> Optional[str]:
    if not ts_ms:
        return None
    try:
        return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        return None


def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")


def load_key_profiles(path: Path) -> Dict[str, Dict[str, str]]:
    if not path or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict):
        return {}
    return profiles


def extract_wallet_balance(resp: dict) -> Dict[str, float]:
    if not isinstance(resp, dict):
        return {}
    result = resp.get("result", {})
    acct_list = result.get("list", [])
    if not acct_list:
        return {}
    acct = acct_list[0]
    total_equity = safe_float(acct.get("totalEquity"))
    if total_equity <= 0:
        total_equity = safe_float(acct.get("totalWalletBalance"))
    total_available = safe_float(acct.get("totalAvailableBalance"))
    coins = acct.get("coin") if isinstance(acct.get("coin"), list) else []
    if coins:
        if total_equity <= 0:
            usd_values = [safe_float(c.get("usdValue")) for c in coins]
            usd_values = [v for v in usd_values if v > 0]
            if usd_values:
                total_equity = sum(usd_values)
        if total_equity <= 0 and len(coins) == 1:
            total_equity = safe_float(coins[0].get("equity")) or safe_float(coins[0].get("walletBalance"))
        if total_equity <= 0:
            stable_sum = 0.0
            for coin in coins:
                if str(coin.get("coin")).upper() in {"USDT", "USDC"}:
                    stable_sum += safe_float(coin.get("equity")) or safe_float(coin.get("walletBalance"))
            if stable_sum > 0:
                total_equity = stable_sum
        if total_available <= 0 and len(coins) == 1:
            total_available = safe_float(coins[0].get("availableToWithdraw")) or safe_float(coins[0].get("availableBalance"))
        if total_available <= 0:
            stable_avail = 0.0
            for coin in coins:
                if str(coin.get("coin")).upper() in {"USDT", "USDC"}:
                    stable_avail += safe_float(coin.get("availableToWithdraw")) or safe_float(coin.get("availableBalance"))
            if stable_avail > 0:
                total_available = stable_avail
    return {
        "equity": total_equity,
        "available": total_available,
    }


def extract_asset_balance(resp: dict) -> Dict[str, float]:
    if not isinstance(resp, dict):
        return {}
    result = resp.get("result", {})
    balances = result.get("balance") or result.get("list") or []
    if isinstance(balances, dict):
        balances = [balances]
    if not isinstance(balances, list):
        return {}
    total_equity = 0.0
    total_available = 0.0
    usd_values = []
    stable_total = 0.0
    stable_available = 0.0
    for coin in balances:
        usd_val = safe_float(coin.get("usdValue"))
        if usd_val > 0:
            usd_values.append(usd_val)
        coin_name = str(coin.get("coin") or "").upper()
        wallet = safe_float(coin.get("walletBalance"))
        available = safe_float(coin.get("transferBalance")) or safe_float(coin.get("availableBalance")) or safe_float(coin.get("availableToWithdraw"))
        if coin_name in {"USDT", "USDC"}:
            if wallet > 0:
                stable_total += wallet
            if available > 0:
                stable_available += available
    if usd_values:
        total_equity = sum(usd_values)
    elif stable_total > 0:
        total_equity = stable_total
    elif len(balances) == 1:
        total_equity = safe_float(balances[0].get("walletBalance"))

    if stable_available > 0:
        total_available = stable_available
    elif len(balances) == 1:
        total_available = safe_float(balances[0].get("transferBalance")) or safe_float(balances[0].get("availableBalance")) or safe_float(balances[0].get("availableToWithdraw"))

    return {
        "equity": total_equity,
        "available": total_available,
    }


def feature_map(latest: dict):
    vec = latest.get("feature_vector") or {}
    names = vec.get("names") or []
    values = vec.get("values") or []
    mapping = {}
    for i, name in enumerate(names):
        if i >= len(values):
            break
        mapping[name] = values[i]
    return mapping


def extract_keys_profile(latest: dict) -> Optional[str]:
    if not isinstance(latest, dict):
        return None
    profile = latest.get("keys_profile")
    if profile:
        return str(profile)
    model = latest.get("model") or {}
    profile = model.get("keys_profile")
    if profile:
        return str(profile)
    return None


def resolve_latest_symbol(latest: dict, fallback: str) -> str:
    if isinstance(latest, dict):
        symbol = latest.get("symbol")
        if symbol:
            return str(symbol)
    return fallback


def count_entries_for_symbol(file_map: dict, symbol: str) -> int:
    if not file_map or not symbol:
        return 0
    total = 0
    for key, path in file_map.items():
        latest = read_latest(path)
        candidate_symbol = resolve_latest_symbol(latest, key)
        if candidate_symbol == symbol:
            total += 1
    return total


def primary_entry_for_symbol(file_map: dict, symbol: str) -> Optional[str]:
    if not file_map or not symbol:
        return None
    candidates = []
    for key, path in file_map.items():
        latest = read_latest(path)
        candidate_symbol = resolve_latest_symbol(latest, key)
        if candidate_symbol == symbol:
            candidates.append(key)
    if not candidates:
        return None
    if symbol in candidates:
        return symbol
    no_suffix = [c for c in candidates if "_" not in c]
    if no_suffix:
        return min(no_suffix, key=len)
    return min(candidates, key=len)


def apply_dash_meta(latest: dict, dash_key: str) -> dict:
    if not isinstance(latest, dict):
        return latest
    symbol = latest.get("symbol") or dash_key
    latest["symbol"] = symbol
    latest["dash_key"] = dash_key
    return latest


def dash_key_from_metrics_path(metrics_path: Path) -> Optional[str]:
    if not metrics_path:
        return None
    stem = metrics_path.stem
    if stem.startswith("live_metrics_"):
        key = stem.replace("live_metrics_", "", 1)
    else:
        key = stem
    return key or None


def model_cache_key(model_dir: Path) -> str:
    try:
        return str(model_dir.resolve())
    except Exception:
        return str(model_dir)


def resolve_model_dir(model_dir_value, metrics_path: Optional[Path]) -> Optional[Path]:
    if not model_dir_value:
        return None
    try:
        model_dir = Path(str(model_dir_value))
    except Exception:
        return None
    if model_dir.is_absolute():
        return model_dir
    candidates = []
    if metrics_path:
        candidates.append(metrics_path.parent / model_dir)
    candidates.append(Path.cwd() / model_dir)
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate
        except Exception:
            continue
    return model_dir


def read_model_params(model_dir: Optional[Path]) -> dict:
    if not model_dir:
        return {}
    params_path = model_dir / "params.json"
    if not params_path.exists():
        return {}
    try:
        payload = json.loads(params_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def model_label_from_params(params: dict) -> Optional[str]:
    if not isinstance(params, dict):
        return None
    keys = (
        "dashboard_name",
        "display_name",
        "tab_name",
        "model_name",
        "name",
        "label",
        "alias",
    )
    for key in keys:
        val = params.get(key)
        if isinstance(val, str):
            label = val.strip()
            if label:
                return label
    return None


def derive_model_label(model_dir: Optional[Path]) -> str:
    if not model_dir:
        return ""
    try:
        cleaned = model_dir.as_posix()
    except Exception:
        cleaned = str(model_dir)
    parts = [p for p in cleaned.replace("\\", "/").split("/") if p]
    if not parts:
        return ""
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return parts[-1]


def load_model_meta(model_dir: Optional[Path]) -> dict:
    if not model_dir:
        return {}
    params_path = model_dir / "params.json"
    features_path = model_dir / "features.pkl"
    dir_long_path = model_dir / "dir_model_long.pkl"
    dir_short_path = model_dir / "dir_model_short.pkl"
    mtimes = {
        "params": params_path.stat().st_mtime if params_path.exists() else None,
        "features": features_path.stat().st_mtime if features_path.exists() else None,
        "dir_long": dir_long_path.stat().st_mtime if dir_long_path.exists() else None,
        "dir_short": dir_short_path.stat().st_mtime if dir_short_path.exists() else None,
    }
    key = model_cache_key(model_dir)
    with MODEL_META_LOCK:
        cached = MODEL_META_CACHE.get(key)
        if cached and cached.get("mtimes") == mtimes:
            return cached

    params = read_model_params(model_dir)
    label = model_label_from_params(params)
    features = None
    if joblib and features_path.exists():
        try:
            features = joblib.load(features_path)
        except Exception:
            features = None
    if isinstance(features, tuple):
        features = list(features)
    if not isinstance(features, list):
        features = None

    dir_model_long = None
    dir_model_short = None
    if joblib and dir_long_path.exists():
        try:
            dir_model_long = joblib.load(dir_long_path)
        except Exception:
            dir_model_long = None
    if joblib and dir_short_path.exists():
        try:
            dir_model_short = joblib.load(dir_short_path)
        except Exception:
            dir_model_short = None

    meta = {
        "params": params,
        "label": label,
        "features": features,
        "dir_model_long": dir_model_long,
        "dir_model_short": dir_model_short,
        "model_dir": str(model_dir),
        "mtimes": mtimes,
    }
    with MODEL_META_LOCK:
        MODEL_META_CACHE[key] = meta
    return meta


def build_feature_row(latest: dict, features: Optional[List[str]]) -> Optional[List[float]]:
    if not latest or not features:
        return None
    vec = latest.get("feature_vector") or {}
    names = vec.get("names") or []
    values = vec.get("values") or []
    if not names or not values:
        return None
    mapping = {}
    for i, name in enumerate(names):
        if i >= len(values):
            break
        mapping[name] = values[i]
    row = []
    for feature in features:
        val = mapping.get(feature)
        try:
            row.append(float(val))
        except Exception:
            row.append(0.0)
    return row


def normalize_bar_time(value) -> Optional[int]:
    ts = safe_int(value, 0)
    if ts <= 0:
        return None
    if ts > 1_000_000_000_000:
        ts = int(ts / 1000)
    return ts


def resolve_direction_info(latest: dict, meta: dict) -> Optional[dict]:
    if not isinstance(latest, dict):
        return None
    direction = latest.get("direction")
    direction_data = direction if isinstance(direction, dict) else {}
    threshold_present = isinstance(direction, dict) and "threshold" in direction
    aggressive_present = isinstance(direction, dict) and "aggressive_threshold" in direction
    pred_long = maybe_float(direction_data.get("pred_long"))
    pred_short = maybe_float(direction_data.get("pred_short"))
    source = direction_data.get("source") if isinstance(direction, dict) else None

    pred = latest.get("prediction") or {}
    if pred_long is None and pred_short is None:
        if "pred_dir_long" in pred or "pred_dir_short" in pred:
            pred_long = maybe_float(pred.get("pred_dir_long"))
            pred_short = maybe_float(pred.get("pred_dir_short"))
            source = source or "prediction"

    last_pred = (latest.get("health") or {}).get("last_prediction") or {}
    if pred_long is None and pred_short is None:
        if "pred_dir_long" in last_pred or "pred_dir_short" in last_pred:
            pred_long = maybe_float(last_pred.get("pred_dir_long"))
            pred_short = maybe_float(last_pred.get("pred_dir_short"))
            source = source or "health"

    params = meta.get("params") if isinstance(meta, dict) else {}
    param_has_direction = isinstance(params, dict) and (
        "direction_threshold" in params or "aggressive_threshold" in params
    )
    has_models = bool(meta.get("dir_model_long") or meta.get("dir_model_short")) if isinstance(meta, dict) else False

    if (pred_long is None or pred_short is None) and has_models:
        row = build_feature_row(latest, meta.get("features"))
        if row:
            if pred_long is None and meta.get("dir_model_long"):
                try:
                    pred_long = float(meta["dir_model_long"].predict([row])[0])
                    source = source or "computed"
                except Exception:
                    pred_long = pred_long
            if pred_short is None and meta.get("dir_model_short"):
                try:
                    pred_short = float(meta["dir_model_short"].predict([row])[0])
                    source = source or "computed"
                except Exception:
                    pred_short = pred_short

    if (
        pred_long is None
        and pred_short is None
        and not direction
        and not param_has_direction
        and not has_models
    ):
        return None

    threshold = maybe_float(direction_data.get("threshold")) if threshold_present else None
    aggressive = maybe_float(direction_data.get("aggressive_threshold")) if aggressive_present else None
    if threshold is None and not threshold_present and isinstance(params, dict):
        threshold = maybe_float(params.get("direction_threshold"))
    if aggressive is None and not aggressive_present and isinstance(params, dict):
        aggressive = maybe_float(params.get("aggressive_threshold"))
    if threshold is None and not threshold_present and (
        param_has_direction or has_models or pred_long is not None or pred_short is not None
    ):
        threshold = DEFAULT_DIRECTION_THRESHOLD
    if aggressive is None and not aggressive_present and (
        param_has_direction or has_models or pred_long is not None or pred_short is not None
    ):
        aggressive = DEFAULT_AGGRESSIVE_THRESHOLD

    bar_time = normalize_bar_time(direction.get("bar_time") if isinstance(direction, dict) else None)
    if bar_time is None:
        bar_time = normalize_bar_time(pred.get("bar_time"))
    if bar_time is None:
        bar_time = normalize_bar_time(last_pred.get("bar_time"))
    if bar_time is None:
        bar_time = normalize_bar_time((latest.get("market") or {}).get("last_bar_time"))
    if bar_time is None:
        bar_time = normalize_bar_time(latest.get("bar_time"))
    if bar_time is None:
        ts = parse_ts(latest.get("ts"))
        if ts:
            bar_time = int(ts)

    payload = dict(direction_data) if isinstance(direction, dict) else {}
    payload.update({
        "pred_long": pred_long,
        "pred_short": pred_short,
        "threshold": threshold,
        "aggressive_threshold": aggressive,
        "bar_time": bar_time,
        "source": source or payload.get("source"),
    })
    return payload


def apply_model_meta(latest: dict, metrics_path: Path) -> Tuple[dict, dict]:
    if not isinstance(latest, dict):
        return latest, {}
    model_dir = resolve_model_dir((latest.get("model") or {}).get("dir"), metrics_path)
    meta = load_model_meta(model_dir) if model_dir else {}
    label = meta.get("label") if isinstance(meta, dict) else None
    if label:
        model = latest.get("model") or {}
        if not model.get("label"):
            model["label"] = label
        latest["model"] = model
    return latest, meta


class TradeStatsState:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.file_pos = 0
        self.buffer = b""
        self.last_daily_pnl = None
        self.last_ts = None
        self.last_startup = None
        self.trade_count = 0
        self.sum_trades = 0.0
        self.sum_wins = 0.0
        self.sum_losses = 0.0
        self.win_count = 0
        self.loss_count = 0
        self.neg_sq_sum = 0.0
        self.equity_peak = None
        self.max_drawdown = 0.0

    def update_from_path(self, metrics_path: Path) -> None:
        if not metrics_path.exists():
            return
        size = metrics_path.stat().st_size
        if size < self.file_pos:
            self.reset()
        if size == self.file_pos and not self.buffer:
            return
        with open(metrics_path, "rb") as f:
            f.seek(self.file_pos)
            data = f.read()
            self.file_pos = f.tell()
        if not data and not self.buffer:
            return
        data = self.buffer + data
        lines = data.splitlines()
        if data and not data.endswith(b"\n"):
            self.buffer = lines[-1] if lines else data
            lines = lines[:-1]
        else:
            self.buffer = b""
        for line in lines:
            try:
                item = json.loads(line.decode("utf-8"))
            except Exception:
                continue
            self._process_item(item)

    def _process_item(self, item: dict) -> None:
        ts = parse_ts(item.get("ts"))
        daily = safe_float(item.get("daily_pnl"))
        startup = item.get("startup_time")

        if self.last_startup is None and startup:
            self.last_startup = startup

        if startup and self.last_startup and startup != self.last_startup:
            self.last_startup = startup
            self.last_daily_pnl = daily
            self.last_ts = ts
        else:
            diff = None
            if self.last_daily_pnl is not None:
                diff = daily - self.last_daily_pnl
                if abs(diff) <= 1e-9:
                    diff = None
                elif self.last_ts is not None and ts is not None:
                    prev_date = datetime.fromtimestamp(self.last_ts, tz=timezone.utc).date()
                    curr_date = datetime.fromtimestamp(ts, tz=timezone.utc).date()
                    if curr_date != prev_date and daily <= 0.0001:
                        diff = None
            if diff is not None:
                self.trade_count += 1
                self.sum_trades += diff
                if diff > 0:
                    self.sum_wins += diff
                    self.win_count += 1
                else:
                    self.sum_losses += diff
                    self.loss_count += 1
                    self.neg_sq_sum += diff * diff
            self.last_daily_pnl = daily
            self.last_ts = ts
            if startup:
                self.last_startup = startup

        equity = safe_float(item.get("equity"))
        if equity > 0:
            if self.equity_peak is None or equity > self.equity_peak:
                self.equity_peak = equity
            if self.equity_peak:
                dd = (equity - self.equity_peak) / self.equity_peak
                if dd < self.max_drawdown:
                    self.max_drawdown = dd

    def summary(self) -> Dict[str, Optional[float]]:
        max_dd_pct = None
        if self.equity_peak is not None:
            max_dd_pct = abs(self.max_drawdown) * 100.0
        if self.trade_count == 0:
            return {
                "trades": 0,
                "win_rate": None,
                "sortino": None,
                "profit_factor": None,
                "avg_trade": None,
                "avg_win": None,
                "avg_loss": None,
                "max_drawdown_pct": max_dd_pct,
            }

        win_rate = self.win_count / self.trade_count if self.trade_count else None
        avg_trade = self.sum_trades / self.trade_count if self.trade_count else None
        avg_win = self.sum_wins / self.win_count if self.win_count else None
        avg_loss = self.sum_losses / self.loss_count if self.loss_count else None
        profit_factor = None
        if self.sum_wins > 0 and self.sum_losses < 0:
            profit_factor = self.sum_wins / abs(self.sum_losses)

        if self.loss_count:
            downside = (self.neg_sq_sum / self.loss_count) ** 0.5
            sortino = (avg_trade or 0.0) / downside if downside > 0 else 0
        else:
            sortino = 10.0

        return {
            "trades": self.trade_count,
            "win_rate": win_rate,
            "sortino": sortino,
            "profit_factor": profit_factor,
            "avg_trade": avg_trade,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "max_drawdown_pct": max_dd_pct,
        }


class ClosedPnlState:
    def __init__(self, profile: Optional[str] = None) -> None:
        self.profile = profile
        self.reset(keep_profile=True)

    def reset(self, keep_profile: bool = True) -> None:
        profile = self.profile if keep_profile else None
        self.profile = profile
        self.backfill_done = False
        self.last_time = 0
        self.last_ids: set = set()
        self.trade_count = 0
        self.sum_trades = 0.0
        self.sum_wins = 0.0
        self.sum_losses = 0.0
        self.win_count = 0
        self.loss_count = 0
        self.neg_sq_sum = 0.0
        self.cum_pnl = 0.0
        self.peak_pnl = 0.0
        self.max_drawdown = 0.0
        self.day_pnls: Dict[str, float] = {}

    def to_dict(self) -> Dict[str, object]:
        return {
            "profile": self.profile,
            "backfill_done": self.backfill_done,
            "last_time": self.last_time,
            "last_ids": list(self.last_ids),
            "trade_count": self.trade_count,
            "sum_trades": self.sum_trades,
            "sum_wins": self.sum_wins,
            "sum_losses": self.sum_losses,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "neg_sq_sum": self.neg_sq_sum,
            "cum_pnl": self.cum_pnl,
            "peak_pnl": self.peak_pnl,
            "max_drawdown": self.max_drawdown,
            "day_pnls": self.day_pnls,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "ClosedPnlState":
        state = cls(profile=payload.get("profile"))
        state.backfill_done = bool(payload.get("backfill_done"))
        state.last_time = safe_int(payload.get("last_time"), 0)
        state.last_ids = set(payload.get("last_ids") or [])
        state.trade_count = safe_int(payload.get("trade_count"), 0)
        state.sum_trades = safe_float(payload.get("sum_trades"))
        state.sum_wins = safe_float(payload.get("sum_wins"))
        state.sum_losses = safe_float(payload.get("sum_losses"))
        state.win_count = safe_int(payload.get("win_count"), 0)
        state.loss_count = safe_int(payload.get("loss_count"), 0)
        state.neg_sq_sum = safe_float(payload.get("neg_sq_sum"))
        state.cum_pnl = safe_float(payload.get("cum_pnl"))
        state.peak_pnl = safe_float(payload.get("peak_pnl"))
        state.max_drawdown = safe_float(payload.get("max_drawdown"))
        day_pnls = payload.get("day_pnls") or {}
        if isinstance(day_pnls, dict):
            state.day_pnls = {str(k): safe_float(v) for k, v in day_pnls.items()}
        return state

    def _record_id(self, record: Dict[str, object]) -> Optional[str]:
        for key in ("orderId", "execId", "orderLinkId", "tradeId"):
            val = record.get(key)
            if val:
                return str(val)
        return None

    def _apply_pnl(self, pnl: float) -> None:
        self.trade_count += 1
        self.sum_trades += pnl
        if pnl > 0:
            self.sum_wins += pnl
            self.win_count += 1
        elif pnl < 0:
            self.sum_losses += pnl
            self.loss_count += 1
            self.neg_sq_sum += pnl * pnl
        self.cum_pnl += pnl
        if self.cum_pnl > self.peak_pnl:
            self.peak_pnl = self.cum_pnl
        drawdown = self.cum_pnl - self.peak_pnl
        if drawdown < self.max_drawdown:
            self.max_drawdown = drawdown

    def ingest_record(self, record: Dict[str, object], allow_old: bool = False) -> bool:
        created = safe_int(record.get("createdTime"), 0)
        record_id = self._record_id(record)
        if not allow_old:
            if created < self.last_time:
                return False
            if created == self.last_time and record_id and record_id in self.last_ids:
                return False
        pnl = safe_float(record.get("closedPnl"))
        self._apply_pnl(pnl)
        if created > self.last_time:
            self.last_time = created
            self.last_ids = set()
        if created == self.last_time and record_id:
            self.last_ids.add(record_id)
        day_key = day_key_from_ms(created)
        if day_key:
            self.day_pnls[day_key] = self.day_pnls.get(day_key, 0.0) + pnl
        return True

    def prune_days(self, cutoff_day: str) -> None:
        if not self.day_pnls or not cutoff_day:
            return
        self.day_pnls = {day: val for day, val in self.day_pnls.items() if day >= cutoff_day}

    def summary(self) -> Dict[str, Optional[float]]:
        max_dd_pct = None
        if self.peak_pnl > 0:
            max_dd_pct = abs(self.max_drawdown) / self.peak_pnl * 100.0
        if self.trade_count == 0:
            return {
                "trades": 0,
                "win_rate": None,
                "sortino": None,
                "profit_factor": None,
                "avg_trade": None,
                "avg_win": None,
                "avg_loss": None,
                "max_drawdown_pct": max_dd_pct,
            }

        win_rate = self.win_count / self.trade_count if self.trade_count else None
        avg_trade = self.sum_trades / self.trade_count if self.trade_count else None
        avg_win = self.sum_wins / self.win_count if self.win_count else None
        avg_loss = self.sum_losses / self.loss_count if self.loss_count else None
        profit_factor = None
        if self.sum_wins > 0 and self.sum_losses < 0:
            profit_factor = self.sum_wins / abs(self.sum_losses)

        if self.loss_count:
            downside = (self.neg_sq_sum / self.loss_count) ** 0.5
            sortino = (avg_trade or 0.0) / downside if downside > 0 else 0
        else:
            sortino = 10.0

        return {
            "trades": self.trade_count,
            "win_rate": win_rate,
            "sortino": sortino,
            "profit_factor": profit_factor,
            "avg_trade": avg_trade,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "max_drawdown_pct": max_dd_pct,
        }


class ClosedPnlMonitor:
    def __init__(
        self,
        keys_path: Path,
        metrics_dir: Path,
        poll_sec: float = 60.0,
        testnet: bool = False,
        symbols_filter: Optional[set] = None,
        page_limit: int = 50,
        backfill_sleep: float = 0.2,
        reset_cutoff_ms: Optional[int] = None,
    ):
        self.keys_path = keys_path
        self.metrics_dir = metrics_dir
        self.poll_sec = poll_sec
        self.testnet = testnet
        self.symbols_filter = symbols_filter
        self.page_limit = max(10, min(200, page_limit))
        self.backfill_sleep = max(0.05, backfill_sleep)
        self.cache_path = metrics_dir / "closed_pnl_stats.json"
        if reset_cutoff_ms is None:
            self.cutoff_ms = 1767484800000
        else:
            self.cutoff_ms = reset_cutoff_ms
        self.cutoff_day = day_key_from_ms(self.cutoff_ms) or "1970-01-01"
        self.states: Dict[str, ClosedPnlState] = {}
        self.stats: Dict[str, Dict[str, Optional[float]]] = {}
        self.sessions: Dict[str, object] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        if reset_cutoff_ms is None:
            self._load_cache()

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:
            return
        symbols = payload.get("symbols")
        if not isinstance(symbols, dict):
            return
        for symbol, entry in symbols.items():
            if not isinstance(entry, dict):
                continue
            if "day_pnls" not in entry:
                continue
            state = ClosedPnlState.from_dict(entry)
            if state.last_time and state.last_time < self.cutoff_ms:
                continue
            state.prune_days(self.cutoff_day)
            self.states[symbol] = state
            self.stats[symbol] = state.summary()

    def _save_cache(self) -> None:
        payload = {
            "version": 1,
            "updated_at": utc_now_str(),
            "symbols": {sym: state.to_dict() for sym, state in self.states.items()},
        }
        try:
            self.cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            return

    def _refresh_sessions(self) -> Dict[str, Dict[str, str]]:
        profiles = load_key_profiles(self.keys_path)
        if not profiles:
            return {}
        for name, creds in profiles.items():
            if name in self.sessions:
                continue
            key = creds.get("api_key")
            secret = creds.get("api_secret")
            if not key or not secret or BybitHTTP is None:
                continue
            try:
                self.sessions[name] = BybitHTTP(
                    testnet=self.testnet,
                    api_key=key,
                    api_secret=secret,
                )
            except Exception:
                continue
        return profiles

    def _fetch_closed_pnl_page(self, session, symbol: str, cursor: Optional[str] = None):
        params = {"category": "linear", "symbol": symbol, "limit": self.page_limit}
        if cursor:
            params["cursor"] = cursor
        try:
            resp = session.get_closed_pnl(**params)
        except Exception:
            return None, None
        if not isinstance(resp, dict) or resp.get("retCode") not in (None, 0):
            return None, None
        result = resp.get("result") or {}
        records = result.get("list") or []
        next_cursor = result.get("nextPageCursor") or result.get("cursor")
        return records, next_cursor

    def _discover_symbol_profiles(self) -> Dict[str, str]:
        file_map = discover_metrics_files(self.metrics_dir, self.symbols_filter)
        mapping: Dict[str, str] = {}
        for symbol, path in file_map.items():
            latest = read_latest(path)
            profile = extract_keys_profile(latest)
            if profile:
                mapping[symbol] = profile
        return mapping

    def _backfill_symbol(self, symbol: str, session, state: ClosedPnlState) -> None:
        cursor = None
        while not self._stop.is_set():
            records, cursor = self._fetch_closed_pnl_page(session, symbol, cursor)
            if records is None:
                break
            if not records:
                break
            for record in records:
                if safe_int(record.get("createdTime"), 0) < self.cutoff_ms:
                    continue
                state.ingest_record(record, allow_old=True)
            if not cursor:
                break
            time.sleep(self.backfill_sleep)
        state.backfill_done = True

    def _update_symbol(self, symbol: str, session, state: ClosedPnlState) -> None:
        prev_last_time = state.last_time
        prev_last_ids = set(state.last_ids)
        cursor = None
        while not self._stop.is_set():
            records, cursor = self._fetch_closed_pnl_page(session, symbol, cursor)
            if records is None:
                break
            if not records:
                break
            oldest = None
            for record in records:
                created = safe_int(record.get("createdTime"), 0)
                if created < self.cutoff_ms:
                    continue
                if oldest is None or created < oldest:
                    oldest = created
                record_id = state._record_id(record)
                if created > prev_last_time or (created == prev_last_time and record_id and record_id not in prev_last_ids):
                    state.ingest_record(record, allow_old=False)
            if not cursor:
                break
            if oldest is None or oldest <= prev_last_time:
                break
            time.sleep(self.backfill_sleep)

    def start(self) -> None:
        if BybitHTTP is None:
            return
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def get_stats(self, symbol: str) -> Dict[str, Optional[float]]:
        return self.stats.get(symbol, {})

    def get_daily_pnl(self, symbol: str, day_key: Optional[str] = None) -> Optional[float]:
        state = self.states.get(symbol)
        if not state:
            return None
        day_key = day_key or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if day_key in state.day_pnls:
            return state.day_pnls.get(day_key)
        if state.backfill_done:
            return 0.0
        return None

    def _run(self) -> None:
        while not self._stop.is_set():
            self._refresh_sessions()
            symbol_profiles = self._discover_symbol_profiles()
            for symbol, profile in symbol_profiles.items():
                if self._stop.is_set():
                    break
                session = self.sessions.get(profile)
                if not session:
                    continue
                state = self.states.get(symbol)
                if state is None or state.profile != profile:
                    state = ClosedPnlState(profile=profile)
                    self.states[symbol] = state
                if not state.backfill_done:
                    self._backfill_symbol(symbol, session, state)
                else:
                    self._update_symbol(symbol, session, state)
                self.stats[symbol] = state.summary()
            self._save_cache()
            self._stop.wait(self.poll_sec)


_TRADE_STATS_CACHE: Dict[str, TradeStatsState] = {}
CLOSED_PNL_MONITOR: Optional[ClosedPnlMonitor] = None


def seed_trade_stats_cache(file_map: Dict[str, Path]) -> None:
    for path in file_map.values():
        if not isinstance(path, Path):
            try:
                path = Path(path)
            except Exception:
                continue
        if not path.exists():
            continue
        state = TradeStatsState()
        try:
            state.file_pos = path.stat().st_size
        except Exception:
            continue
        latest = read_latest(path)
        if isinstance(latest, dict):
            daily_val = latest.get("daily_pnl")
            if daily_val is not None and daily_val != "":
                state.last_daily_pnl = safe_float(daily_val)
            ts_val = parse_ts(latest.get("ts"))
            if ts_val is not None:
                state.last_ts = ts_val
            startup = latest.get("startup_time")
            if startup:
                state.last_startup = startup
            equity = safe_float(latest.get("equity"))
            if equity > 0:
                state.equity_peak = equity
        _TRADE_STATS_CACHE[str(path)] = state


def compute_trade_stats(metrics_path: Path, limit: int = 400):
    _ = limit
    key = str(metrics_path)
    state = _TRADE_STATS_CACHE.get(key)
    if state is None:
        state = TradeStatsState()
        _TRADE_STATS_CACHE[key] = state
    state.update_from_path(metrics_path)
    metrics_summary = state.summary()

    if CLOSED_PNL_MONITOR:
        symbol = metrics_path.stem.replace("live_metrics_", "", 1)
        if symbol == metrics_path.stem:
            latest = read_latest(metrics_path)
            symbol = latest.get("symbol") or symbol
        closed_stats = CLOSED_PNL_MONITOR.get_stats(symbol)
        if closed_stats:
            combined = dict(closed_stats)
            if metrics_summary.get("max_drawdown_pct") is not None:
                combined["max_drawdown_pct"] = metrics_summary.get("max_drawdown_pct")
            return combined
    return metrics_summary


def exchange_daily_pnl(symbol: str) -> Optional[float]:
    if not CLOSED_PNL_MONITOR:
        return None
    return CLOSED_PNL_MONITOR.get_daily_pnl(symbol)


def compute_balance_stats(history: List[dict], flow_threshold_pct: float = 2.0) -> Dict[str, float]:
    if not history or len(history) < 2:
        return {}
    enriched = []
    for item in history:
        ts = parse_ts(item.get("ts"))
        equity = safe_float(item.get("total_equity"))
        if equity > 0 and ts is not None:
            enriched.append((ts, equity))
    if len(enriched) < 2:
        return {}
    enriched.sort(key=lambda x: x[0])
    times = [x[0] for x in enriched]
    equities = [x[1] for x in enriched]

    adjusted = []
    flow = 0.0
    flow_events = 0
    for i, eq in enumerate(equities):
        if i == 0:
            adjusted.append(eq)
            continue
        prev = equities[i - 1]
        delta = eq - prev
        threshold = max(1.0, prev * (flow_threshold_pct / 100.0))
        if abs(delta) > threshold:
            flow += delta
            flow_events += 1
        adjusted.append(eq - flow)

    returns = []
    for i in range(1, len(adjusted)):
        prev = adjusted[i - 1]
        if prev <= 0:
            continue
        returns.append((adjusted[i] - prev) / prev)

    stability_pct = None
    volatility_pct = None
    if returns:
        stability_pct = (sum(1 for r in returns if r >= 0) / len(returns)) * 100.0
    if len(returns) >= 2:
        mean_ret = sum(returns) / len(returns)
        var = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
        volatility_pct = (var ** 0.5) * 100.0

    max_dd = 0.0
    peak = adjusted[0]
    for val in adjusted:
        peak = max(peak, val)
        if peak > 0:
            dd = (val - peak) / peak
            if dd < max_dd:
                max_dd = dd
    max_drawdown_pct = abs(max_dd) * 100.0 if len(adjusted) > 1 else None

    slope_per_day = None
    smoothness_r2 = None
    if len(times) >= 2:
        t0 = times[0]
        xs = [(t - t0) / 86400.0 for t in times]
        x_mean = sum(xs) / len(xs)
        y_mean = sum(adjusted) / len(adjusted)
        var_x = sum((x - x_mean) ** 2 for x in xs)
        if var_x > 0:
            cov_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, adjusted))
            slope = cov_xy / var_x
            slope_per_day = slope
            ss_tot = sum((y - y_mean) ** 2 for y in adjusted)
            if ss_tot > 0:
                ss_res = sum((y - (slope * x + (y_mean - slope * x_mean))) ** 2 for x, y in zip(xs, adjusted))
                smoothness_r2 = max(0.0, 1.0 - (ss_res / ss_tot))

    total_return_pct = None
    if adjusted[0] > 0:
        total_return_pct = ((adjusted[-1] / adjusted[0]) - 1.0) * 100.0
    raw_total_return_pct = None
    if equities[0] > 0:
        raw_total_return_pct = ((equities[-1] / equities[0]) - 1.0) * 100.0

    return {
        "total_return_pct": total_return_pct,
        "raw_total_return_pct": raw_total_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "volatility_pct": volatility_pct,
        "stability_pct": stability_pct,
        "smoothness_r2": smoothness_r2,
        "slope_per_day": slope_per_day,
        "points": len(adjusted),
        "flow_events": flow_events,
        "flow_threshold_pct": flow_threshold_pct,
    }


class BalanceMonitor:
    def __init__(
        self,
        keys_path: Path,
        metrics_dir: Path,
        poll_sec: float = 30.0,
        history_limit: int = 10000,
        testnet: bool = False,
        flow_threshold_pct: float = 2.0,
        reset_history: bool = False,
    ):
        self.keys_path = keys_path
        self.metrics_dir = metrics_dir
        self.poll_sec = poll_sec
        self.history_limit = history_limit
        self.testnet = testnet
        self.flow_threshold_pct = flow_threshold_pct
        self.reset_history = reset_history
        self.history_path = metrics_dir / "balance_history.jsonl"
        self.history = deque(maxlen=history_limit)
        self._history_loaded = False
        self.latest: Dict[str, object] = {}
        self.sessions: Dict[str, object] = {}
        self.member_ids: Dict[str, str] = {}
        self.wallet_perms: Dict[str, bool] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _load_history(self) -> None:
        if self._history_loaded:
            return
        for item in tail_jsonl(self.history_path, limit=self.history_limit):
            self.history.append(item)
        self._history_loaded = True

    def _sync_latest_from_history(self) -> None:
        if not self.history:
            return
        last = self.history[-1]
        if not isinstance(last, dict):
            return
        profiles = last.get("profiles")
        if not isinstance(profiles, list):
            profiles = []
        total_equity = safe_float(last.get("total_equity"))
        total_available = safe_float(last.get("total_available"))
        total_unified = safe_float(last.get("total_unified"))
        total_funding = safe_float(last.get("total_funding"))
        profile_count = len(profiles) if profiles else safe_int(last.get("profile_count"), 0)
        self.latest = {
            "ts": last.get("ts"),
            "total_equity": total_equity,
            "total_available": total_available,
            "total_unified": total_unified,
            "total_funding": total_funding,
            "profile_count": profile_count,
            "profiles": profiles,
            "stats": compute_balance_stats(list(self.history), flow_threshold_pct=self.flow_threshold_pct),
            "funding_error": last.get("funding_error"),
        }

    def _refresh_sessions(self) -> Dict[str, Dict[str, str]]:
        profiles = load_key_profiles(self.keys_path)
        if not profiles:
            return {}
        for name, creds in profiles.items():
            if name in self.sessions:
                continue
            key = creds.get("api_key")
            secret = creds.get("api_secret")
            if not key or not secret or BybitHTTP is None:
                continue
            try:
                self.sessions[name] = BybitHTTP(
                    testnet=self.testnet,
                    api_key=key,
                    api_secret=secret,
                )
            except Exception:
                continue
        return profiles

    def _fetch_balance(self, session, account_types: List[str], coin: Optional[str] = None) -> Dict[str, float]:
        for account_type in account_types:
            try:
                params = {"accountType": account_type}
                if coin:
                    params["coin"] = coin
                resp = session.get_wallet_balance(**params)
            except Exception:
                continue
            if isinstance(resp, dict) and resp.get("retCode") not in (None, 0):
                continue
            bal = extract_wallet_balance(resp)
            if safe_float(bal.get("equity")) > 0 or safe_float(bal.get("available")) > 0:
                return bal
        return {}

    def _fetch_asset_balance(
        self,
        session,
        account_type: str,
        coin: Optional[str] = None,
        member_id: Optional[str] = None,
    ) -> Dict[str, float]:
        try:
            if coin:
                params = {"accountType": account_type, "coin": coin}
                if member_id:
                    params["memberId"] = member_id
                resp = session.get_coin_balance(**params)
            else:
                params = {"accountType": account_type}
                if member_id:
                    params["memberId"] = member_id
                resp = session.get_coins_balance(**params)
        except Exception:
            return {}
        if isinstance(resp, dict) and resp.get("retCode") not in (None, 0):
            return {}
        bal = extract_asset_balance(resp)
        if safe_float(bal.get("equity")) > 0 or safe_float(bal.get("available")) > 0:
            return bal
        return {}

    def _fetch_asset_balance_with_error(
        self,
        session,
        account_type: str,
        coin: Optional[str] = None,
        member_id: Optional[str] = None,
    ) -> Tuple[Dict[str, float], Optional[int], Optional[str]]:
        try:
            if coin:
                params = {"accountType": account_type, "coin": coin}
                if member_id:
                    params["memberId"] = member_id
                resp = session.get_coin_balance(**params)
            else:
                params = {"accountType": account_type}
                if member_id:
                    params["memberId"] = member_id
                resp = session.get_coins_balance(**params)
        except Exception as exc:
            return {}, None, str(exc)
        if isinstance(resp, dict) and resp.get("retCode") not in (None, 0):
            return {}, safe_int(resp.get("retCode")), str(resp.get("retMsg") or "")
        bal = extract_asset_balance(resp)
        return bal, None, None

    def _get_member_id(self, session, name: str) -> Optional[str]:
        cached = self.member_ids.get(name)
        if cached:
            return cached
        try:
            resp = session.get_api_key_information()
        except Exception:
            return None
        if not isinstance(resp, dict) or resp.get("retCode") not in (None, 0):
            return None
        result = resp.get("result", {}) if isinstance(resp, dict) else {}
        perms = result.get("permissions") if isinstance(result, dict) else {}
        wallet_perm = None
        if isinstance(perms, dict):
            wallet_list = perms.get("Wallet")
            wallet_perm = bool(wallet_list)
        if wallet_perm is not None:
            self.wallet_perms[name] = wallet_perm
        user_id = result.get("userID") or result.get("uid") or result.get("userId")
        if user_id is None:
            return None
        member_id = str(user_id)
        self.member_ids[name] = member_id
        return member_id

    def start(self) -> None:
        if self.reset_history:
            self.history.clear()
            self._history_loaded = True
        else:
            self._load_history()
        self._sync_latest_from_history()
        if not self.keys_path or not self.keys_path.exists() or BybitHTTP is None:
            return
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _poll_once(self) -> None:
        profiles = self._refresh_sessions()
        if not profiles:
            return
        totals = {
            "equity": 0.0,
            "available": 0.0,
            "unified": 0.0,
            "funding": 0.0,
        }
        profile_payloads = []
        funding_errors = []
        for name, creds in profiles.items():
            session = self.sessions.get(name)
            if session is None:
                continue
            member_id = self._get_member_id(session, name)
            unified_bal = self._fetch_balance(session, ["UNIFIED"], coin=None)
            if not unified_bal:
                unified_bal = self._fetch_balance(session, ["UNIFIED"], coin="USDT")
            fund_bal = {}
            funding_error = None
            if self.wallet_perms.get(name, True):
                fund_bal, fund_code, fund_msg = self._fetch_asset_balance_with_error(
                    session, "FUND", coin=None, member_id=member_id
                )
                if not fund_bal and fund_code:
                    funding_error = f"retCode={fund_code} {fund_msg}".strip()
                    if fund_code == 10005:
                        self.wallet_perms[name] = False
                if not fund_bal and not funding_error:
                    fund_bal, fund_code, fund_msg = self._fetch_asset_balance_with_error(
                        session, "FUND", coin="USDT", member_id=member_id
                    )
                    if not fund_bal and fund_code:
                        funding_error = f"retCode={fund_code} {fund_msg}".strip()
                        if fund_code == 10005:
                            self.wallet_perms[name] = False
            else:
                funding_error = "wallet permission missing"
            if not fund_bal:
                fund_bal = self._fetch_balance(session, ["FUNDING"], coin=None)
            if not fund_bal:
                fund_bal = self._fetch_balance(session, ["FUNDING"], coin="USDT")
            if funding_error:
                funding_errors.append(f"{name}: {funding_error}")

            unified_eq = safe_float(unified_bal.get("equity"))
            unified_avail = safe_float(unified_bal.get("available"))
            funding_eq = safe_float(fund_bal.get("equity"))
            funding_avail = safe_float(fund_bal.get("available"))
            total_equity = unified_eq + funding_eq
            total_available = unified_avail + funding_avail

            if total_equity <= 0:
                continue
            totals["equity"] += total_equity
            totals["available"] += total_available
            totals["unified"] += unified_eq
            totals["funding"] += funding_eq
            profile_payloads.append({
                "name": name,
                "unified_equity": unified_eq,
                "funding_equity": funding_eq,
                "total_equity": total_equity,
                "available": total_available,
                "funding_error": funding_error,
            })

        if not profile_payloads:
            return
        now_str = utc_now_str()
        record = {
            "ts": now_str,
            "total_equity": totals["equity"],
            "total_available": totals["available"],
            "total_unified": totals["unified"],
            "total_funding": totals["funding"],
            "profiles": [
                {"name": p["name"], "total_equity": p["total_equity"]}
                for p in profile_payloads
            ],
        }
        self.history.append(record)
        append_jsonl(self.history_path, record)

        self.latest = {
            "ts": now_str,
            "total_equity": totals["equity"],
            "total_available": totals["available"],
            "total_unified": totals["unified"],
            "total_funding": totals["funding"],
            "profile_count": len(profile_payloads),
            "profiles": profile_payloads,
            "stats": compute_balance_stats(list(self.history), flow_threshold_pct=self.flow_threshold_pct),
            "funding_error": "; ".join(funding_errors) if funding_errors else None,
        }

    def get_history(self, limit: int = 200):
        if limit <= 0:
            return []
        if not self.history:
            self._load_history()
            self._sync_latest_from_history()
        return list(self.history)[-limit:]

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._poll_once()
            except Exception:
                pass
            time.sleep(self.poll_sec)


def is_protective_order(order: dict) -> bool:
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


def read_open_orders_latest(metrics_path: Path, symbol: str):
    key = dash_key_from_metrics_path(metrics_path)
    key_path = metrics_path.with_name(f"open_orders_{key}.jsonl") if key else None
    if key_path and key_path.exists() and key_path.stat().st_size > 0:
        orders_path = key_path
    else:
        orders_path = metrics_path.with_name(f"open_orders_{symbol}.jsonl")
    latest = read_latest(orders_path)
    orders = latest.get("open_orders") if isinstance(latest, dict) else None
    if not isinstance(orders, list):
        return []
    resting = []
    for order in orders:
        if is_protective_order(order):
            continue
        resting.append({
            "order_id": order.get("orderId"),
            "side": order.get("side"),
            "price": safe_float(order.get("price")),
            "qty": safe_float(order.get("qty")),
            "status": order.get("orderStatus"),
            "reduce_only": safe_bool(order.get("reduceOnly")),
            "position_idx": order.get("positionIdx"),
        })
    return resting


def read_protective_orders_latest(metrics_path: Path, symbol: str):
    key = dash_key_from_metrics_path(metrics_path)
    key_path = metrics_path.with_name(f"open_orders_{key}.jsonl") if key else None
    if key_path and key_path.exists() and key_path.stat().st_size > 0:
        orders_path = key_path
    else:
        orders_path = metrics_path.with_name(f"open_orders_{symbol}.jsonl")
    latest = read_latest(orders_path)
    orders = latest.get("open_orders") if isinstance(latest, dict) else None
    if not isinstance(orders, list):
        return []
    protective = []
    for order in orders:
        if not is_protective_order(order):
            continue
        trigger_price = safe_float(
            order.get("triggerPrice")
            or order.get("trigger_price")
            or order.get("stopPrice")
            or order.get("stop_price")
        )
        protective.append({
            "order_id": order.get("orderId"),
            "side": order.get("side"),
            "price": safe_float(order.get("price")),
            "trigger_price": trigger_price,
            "qty": safe_float(order.get("qty")),
            "status": order.get("orderStatus"),
            "reduce_only": safe_bool(order.get("reduceOnly")),
            "close_on_trigger": safe_bool(order.get("closeOnTrigger")),
            "order_type": order.get("orderType"),
            "order_filter": order.get("orderFilter"),
            "stop_order_type": order.get("stopOrderType"),
            "create_type": order.get("createType"),
            "position_idx": order.get("positionIdx"),
        })
    return protective


def extract_price(latest: dict):
    pos = latest.get("position") or {}
    price = safe_float(pos.get("mark_price"))
    if price > 0:
        return price
    positions = latest.get("positions") or {}
    if isinstance(positions, dict):
        for key in ("long", "short"):
            p = safe_float((positions.get(key) or {}).get("mark_price"))
            if p > 0:
                return p
    market = latest.get("market") or {}
    market_close = safe_float(market.get("last_close"))
    if market_close > 0:
        return market_close
    last_close = safe_float(latest.get("last_close"))
    if last_close > 0:
        return last_close
    fmap = feature_map(latest)
    for key in ("close", "mark_price", "last_price"):
        val = fmap.get(key)
        if val is not None:
            try:
                val_f = float(val)
            except Exception:
                continue
            if val_f > 0:
                return val_f
    return None


def augment_latest(latest: dict, symbol: str, metrics_path: Path):
    if not isinstance(latest, dict):
        return latest
    latest, meta = apply_model_meta(latest, metrics_path)
    direction = resolve_direction_info(latest, meta)
    if direction:
        latest["direction"] = direction
    exch_daily = exchange_daily_pnl(symbol)
    if exch_daily is not None:
        latest["daily_pnl"] = exch_daily
    price = extract_price(latest)
    resting = read_open_orders_latest(metrics_path, symbol)
    protective = read_protective_orders_latest(metrics_path, symbol)
    latest["dashboard"] = {
        "price": price,
        "resting_orders": resting,
        "resting_count": len(resting),
        "protective_orders": protective,
        "protective_count": len(protective),
        "trade_stats": compute_trade_stats(metrics_path),
    }
    return latest


def discover_metrics_files(metrics_dir: Path, symbols=None):
    mapping = {}
    if not metrics_dir.exists():
        return mapping
    for path in sorted(metrics_dir.glob("live_metrics_*.jsonl")):
        key = path.stem.replace("live_metrics_", "", 1)
        if not key:
            continue
        latest = read_latest(path)
        symbol = resolve_latest_symbol(latest, key)
        if symbols and symbol not in symbols and key not in symbols:
            continue
        if symbol not in mapping:
            mapping[symbol] = path
    return mapping


def discover_metrics_entries(metrics_dir: Path, symbols=None):
    mapping = {}
    if not metrics_dir.exists():
        return mapping
    for path in sorted(metrics_dir.glob("live_metrics_*.jsonl")):
        key = path.stem.replace("live_metrics_", "", 1)
        if not key:
            continue
        if symbols:
            latest = read_latest(path)
            symbol = resolve_latest_symbol(latest, key)
            if symbol not in symbols and key not in symbols:
                continue
        mapping[key] = path
    return mapping


def summary_from_files(file_map):
    items = []
    for key in sorted(file_map.keys()):
        path = file_map[key]
        latest = read_latest(path)
        if not latest:
            latest = {"symbol": None, "ts": None}
        latest = apply_dash_meta(latest, key)
        symbol = latest.get("symbol") or key
        latest = augment_latest(latest, symbol, path)
        items.append(latest)
    return items


def format_model_display(latest: dict, entry_key: str, metrics_path: Path) -> Tuple[str, str]:
    if not isinstance(latest, dict):
        return entry_key or "", entry_key or ""
    latest, _meta = apply_model_meta(latest, metrics_path)
    symbol = resolve_latest_symbol(latest, entry_key)
    model = latest.get("model") or {}
    label = model.get("label") or model.get("keys_profile")
    if not label:
        model_dir = resolve_model_dir(model.get("dir"), metrics_path)
        label = derive_model_label(model_dir) if model_dir else None
    tags = []
    if label and label != symbol:
        tags.append(str(label))
    if entry_key and entry_key != symbol and entry_key not in tags:
        tags.append(str(entry_key))
    display = f"{symbol} ({' / '.join(tags)})" if tags else symbol
    return display, symbol


EXTRA_SCRIPT = """
<script>
(function() {
  if (window.__dashExtrasInjected) return;
  window.__dashExtrasInjected = true;
  const DT = (key) => (window.DASH_TRANS && window.DASH_TRANS[key])
    || key.replace("dash_", "").replace(/_/g, " ");

  function fmt(v, d) {
    if (v === null || v === undefined) return "--";
    const num = Number(v);
    if (Number.isNaN(num)) return "--";
    return num.toFixed(d);
  }

  function ensureStyles() {
    if (document.getElementById("dashIcebergStyles")) return;
    const style = document.createElement("style");
    style.id = "dashIcebergStyles";
    style.textContent = `
      .iceberg-controls { display: flex; flex-wrap: wrap; gap: 6px; justify-content: flex-end; }
      .mini-btn {
        background: #1f2937;
        border: 1px solid #334155;
        color: #e5e7eb;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 11px;
        cursor: pointer;
      }
      .mini-btn:hover { border-color: #60a5fa; }
      .chart-header { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; }
      .chart-sub { color: var(--muted); font-size: 12px; }
      .toggle-btn.donate-btn.active {
        border-color: rgba(168,85,247,0.7);
        box-shadow: 0 0 0 2px rgba(168,85,247,0.2);
      }
      .modal {
        position: fixed;
        inset: 0;
        display: none;
        align-items: center;
        justify-content: center;
        z-index: 50;
      }
      .modal.open { display: flex; }
      .modal-backdrop {
        position: absolute;
        inset: 0;
        background: rgba(2, 6, 23, 0.72);
        backdrop-filter: blur(2px);
      }
      .modal-card {
        position: relative;
        z-index: 1;
        background: #0b1220;
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 12px;
        width: min(520px, calc(100% - 32px));
        padding: 16px 18px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.45);
      }
      .modal-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 8px;
      }
      .modal-head h3 { margin: 0; font-size: 16px; }
      .modal-body .row { margin-top: 10px; }
      .modal-body .row span:last-child { text-align: right; }
      .icon-btn {
        background: transparent;
        border: 1px solid rgba(148, 163, 184, 0.4);
        color: var(--text);
        width: 28px;
        height: 28px;
        border-radius: 8px;
        cursor: pointer;
        font-size: 16px;
        line-height: 1;
      }
      .icon-btn:hover { border-color: #60a5fa; }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 12px; }
    `;
    document.head.appendChild(style);
  }

  ensureStyles();

  function ensureCard() {
    if (document.getElementById("dashMarketCard")) return;
    const grid = document.querySelector("#tradersSection .grid");
    if (!grid) return;
    const card = document.createElement("div");
    card.className = "card";
    card.id = "dashMarketCard";
    card.innerHTML = `
      <h3>${DT("dash_market")}</h3>
      <div class="value" id="dashPrice">--</div>
      <div class="row"><span>${DT("dash_position_mode")}</span><span id="dashPosMode">--</span></div>
      <div class="row"><span>${DT("dash_resting_orders")}</span><span id="dashOrderCount">--</span></div>
      <div class="row"><span>${DT("dash_limits")}</span><span id="dashOrderList" style="white-space: pre-line; text-align: right;">--</span></div>
    `;
    grid.appendChild(card);
  }

  function ensurePerfCard() {
    if (document.getElementById("dashPerfCard")) return;
    const grid = document.querySelector("#tradersSection .grid");
    if (!grid) return;
    const card = document.createElement("div");
    card.className = "card";
    card.id = "dashPerfCard";
    card.innerHTML = `
      <h3>${DT("dash_performance")}</h3>
      <div class="value" id="dashWinRate">--</div>
      <div class="row"><span>${DT("dash_trades")}</span><span id="dashTradeCount">--</span></div>
      <div class="row"><span>${DT("dash_sortino")}</span><span id="dashSortino">--</span></div>
      <div class="row"><span>${DT("dash_profit_factor")}</span><span id="dashProfitFactor">--</span></div>
      <div class="row"><span>${DT("dash_avg_trade")}</span><span id="dashAvgTrade">--</span></div>
      <div class="row"><span>${DT("dash_max_dd")}</span><span id="dashMaxDD">--</span></div>
    `;
    grid.appendChild(card);
  }

  function ensureIcebergCard() {
    if (document.getElementById("dashIcebergCard")) return;
    const grid = document.querySelector("#tradersSection .grid");
    if (!grid) return;
    const card = document.createElement("div");
    card.className = "card";
    card.id = "dashIcebergCard";
    card.innerHTML = `
      <h3>${DT("dash_iceberg")}</h3>
      <div class="value" id="dashIcebergStatus">--</div>
      <div class="row"><span>${DT("dash_entry_buy")}</span><span id="dashIcebergBuy" style="white-space: pre-line; text-align: right;">--</span></div>
      <div class="row"><span>${DT("dash_entry_sell")}</span><span id="dashIcebergSell" style="white-space: pre-line; text-align: right;">--</span></div>
      <div class="row"><span>${DT("dash_tp_long")}</span><span id="dashIcebergTpLong" style="white-space: pre-line; text-align: right;">--</span></div>
      <div class="row"><span>${DT("dash_tp_short")}</span><span id="dashIcebergTpShort" style="white-space: pre-line; text-align: right;">--</span></div>
      <div class="row"><span>L1</span><span id="dashIcebergL1">--</span></div>
      <div class="row"><span>${DT("dash_controls")}</span>
        <span id="dashIcebergControls" class="iceberg-controls">
          <button class="mini-btn" id="icebergCancelBuy">${DT("dash_cancel_buy")}</button>
          <button class="mini-btn" id="icebergCancelSell">${DT("dash_cancel_sell")}</button>
          <button class="mini-btn" id="icebergCancelTpLong">${DT("dash_cancel_tp_long")}</button>
          <button class="mini-btn" id="icebergCancelTpShort">${DT("dash_cancel_tp_short")}</button>
        </span>
      </div>
    `;
    grid.appendChild(card);
  }

  function ensureDonateUI() {
    const actions = document.querySelector("header .header-actions");
    if (!actions) return;
    let button = document.getElementById("donateToggle");
    if (!button) {
      button = document.createElement("button");
      button.className = "toggle-btn active donate-btn";
      button.id = "donateToggle";
      button.type = "button";
      const sound = document.getElementById("soundToggle");
      if (sound && sound.parentElement === actions) {
        actions.insertBefore(button, sound);
      } else {
        actions.prepend(button);
      }
    }
      button.textContent = DT("dash_donate_referral");
      button.classList.add("donate-btn");

    let modal = document.getElementById("donateModal");
    if (!modal) {
      modal = document.createElement("div");
      modal.id = "donateModal";
      modal.className = "modal";
      modal.setAttribute("aria-hidden", "true");
      modal.innerHTML = `
        <div class="modal-backdrop" data-close="true"></div>
        <div class="modal-card" role="dialog" aria-modal="true" aria-labelledby="donateTitle">
          <div class="modal-head">
            <h3 id="donateTitle">${DT("dash_donate_referral")}</h3>
            <button class="icon-btn" id="donateClose" type="button" aria-label="${DT("dash_close")}">×</button>
          </div>
          <div class="modal-body">
            <div class="sub" id="donateNote">${DT("dash_donate_note")}</div>
            <div class="row"><span>${DT("dash_bitcoin")}</span><span class="mono">1PucNiXsUCzfrMqUGCPfwgdyE3BL8Xnrrp</span></div>
            <div class="row"><span>${DT("dash_ethereum")}</span><span class="mono">0x58ef00f47d6e94dfc486a2ed9b3dd3cfaf3c9714</span></div>
            <div class="row"><span>${DT("dash_referral_link")}</span><span class="mono">https://www.bybit.com/invite?ref=14VP14Z</span></div>
          </div>
        </div>
      `;
      document.body.appendChild(modal);
    } else {
      const title = modal.querySelector("#donateTitle");
      if (title) title.textContent = DT("dash_donate_referral");
      const note = modal.querySelector("#donateNote");
      if (note) note.textContent = DT("dash_donate_note");
      const closeBtn = modal.querySelector("#donateClose");
      if (closeBtn) closeBtn.setAttribute("aria-label", DT("dash_close"));
    }

    const openModal = () => {
      if (!modal) return;
      modal.classList.add("open");
      modal.setAttribute("aria-hidden", "false");
    };
    const closeModal = () => {
      if (!modal) return;
      modal.classList.remove("open");
      modal.setAttribute("aria-hidden", "true");
    };

    if (!button.dataset.bound) {
      button.addEventListener("click", openModal);
      button.dataset.bound = "1";
    }
    if (modal && !modal.dataset.bound) {
      const closeBtn = modal.querySelector("#donateClose");
      const backdrop = modal.querySelector("[data-close]");
      if (closeBtn) closeBtn.addEventListener("click", closeModal);
      if (backdrop) backdrop.addEventListener("click", closeModal);
      document.addEventListener("keydown", (event) => {
        if (event.key === "Escape") closeModal();
      });
      modal.dataset.bound = "1";
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", ensureDonateUI);
  } else {
    ensureDonateUI();
  }

  let chart = null;
  let chartInitialized = false;
  let chartTimer = null;
  let chartFetchInFlight = false;
  let chartFullCandles = [];
  let chartIntervalSec = 60;
  let chartLastKey = null;
  let chartPricePrecision = 6;
  let chartLibLoading = false;
  let chartLibFallback = false;
  const MIN_PRICE_PRECISION = 5;
  let chartNeedsFit = true;
  let chartUserZoom = false;
  let chartZoomState = null;
  let chartOverlay = { lines: [], markers: [], livePrice: null };
  let chartDirectionHistory = [];
  let chartDirectionKey = null;
  let chartDirectionLastFetch = 0;
  let chartDirectionFetchInFlight = false;
  const chartStateCache = {};
  const CHART_LEGEND_KEY = "dash_chart_legend_v1";
  const defaultChartLegend = { pred: true, dir: true, gate: true, orders: true };
  let chartLegend = Object.assign({}, defaultChartLegend, loadChartLegend());
  let chartLegendBound = false;

  function loadChartLegend() {
    try {
      const raw = localStorage.getItem(CHART_LEGEND_KEY);
      if (!raw) return {};
      const data = JSON.parse(raw);
      return data && typeof data === "object" ? data : {};
    } catch (err) {
      return {};
    }
  }

  function saveChartLegend() {
    try {
      localStorage.setItem(CHART_LEGEND_KEY, JSON.stringify(chartLegend || {}));
    } catch (err) {
      return;
    }
  }

  function isLegendEnabled(key) {
    if (!key) return true;
    return chartLegend[key] !== false;
  }

  function syncChartLegendButtons() {
    const wrap = document.getElementById("chartLegend");
    if (!wrap) return;
    wrap.querySelectorAll("[data-legend]").forEach(btn => {
      const key = (btn.dataset.legend || "").toLowerCase();
      if (!key) return;
      btn.classList.toggle("active", isLegendEnabled(key));
    });
  }

  function ensureChartSection() {
    if (document.getElementById("icebergChart")) return;
    const chartSection = document.getElementById("chartSection");
    if (chartSection) return;
    const detailWrap = document.querySelector("#tradersSection .detail-wrap");
    if (!detailWrap) return;
    const wrap = document.createElement("div");
    wrap.id = "icebergChartWrap";
    wrap.innerHTML = `
      <div class="chart-header">
        <h3>${DT("dash_iceberg_chart")}</h3>
        <div class="chart-sub" id="icebergChartMeta">--</div>
      </div>
      <div id="icebergChart"></div>
    `;
    detailWrap.appendChild(wrap);
  }

  function markChartInteracted() {
    chartUserZoom = true;
    chartNeedsFit = false;
    if (chart && chart.getOption) {
      const dz = chart.getOption().dataZoom || [];
      if (dz.length) {
        chartZoomState = {
          start: dz[0].start,
          end: dz[0].end,
          startValue: dz[0].startValue,
          endValue: dz[0].endValue,
        };
      }
    }
  }

  function loadChartLib(callback) {
    if (window.echarts) {
      callback();
      return;
    }
    if (chartLibLoading) {
      setTimeout(() => loadChartLib(callback), 200);
      return;
    }
    chartLibLoading = true;
    const script = document.createElement("script");
    script.src = "/static/echarts.min.js";
    script.onerror = () => {
      if (!chartLibFallback) {
        chartLibFallback = true;
        chartLibLoading = false;
        const fallback = document.createElement("script");
        fallback.src = "https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js";
        fallback.onerror = () => {
          chartLibLoading = false;
          const meta = document.getElementById("icebergChartMeta");
          if (meta) meta.textContent = DT("dash_chart_failed");
        };
        fallback.onload = () => {
          chartLibLoading = false;
          callback();
        };
        document.head.appendChild(fallback);
        return;
      }
      chartLibLoading = false;
      const meta = document.getElementById("icebergChartMeta");
      if (meta) meta.textContent = DT("dash_chart_failed");
    };
    script.onload = () => {
      chartLibLoading = false;
      callback();
    };
    document.head.appendChild(script);
  }

  function initChart() {
    if (chartInitialized || !window.echarts) return;
    ensureChartSection();
    const container = document.getElementById("icebergChart");
    if (!container) return;
    if (container.clientWidth <= 0 || container.clientHeight <= 0) return;
    chart = window.echarts.init(container, null, { renderer: "canvas", useDirtyRect: true });
    chartInitialized = true;
    chartPricePrecision = MIN_PRICE_PRECISION;
    chart.setOption(baseChartOption(), { notMerge: true, lazyUpdate: true });
    chart.on("datazoom", () => {
      const dz = chart.getOption().dataZoom || [];
      if (dz.length) {
        chartZoomState = {
          start: dz[0].start,
          end: dz[0].end,
          startValue: dz[0].startValue,
          endValue: dz[0].endValue,
        };
      }
      chartUserZoom = true;
      chartNeedsFit = false;
    });
    if (chart.getZr) {
      const zr = chart.getZr();
      zr.on("mousedown", markChartInteracted);
      zr.on("mousewheel", markChartInteracted);
      zr.on("touchstart", markChartInteracted);
    }
  }

  function syncChartRangeButtons() {
    const rangeWrap = document.getElementById("chartRange");
    if (!rangeWrap) return;
    const target = (chartRangeKey || "").toLowerCase();
    rangeWrap.querySelectorAll(".range-btn").forEach(btn => {
      const key = (btn.dataset.range || "").toLowerCase();
      btn.classList.toggle("active", key === target);
    });
  }

  function stashChartState(key) {
    if (!key) return;
    chartStateCache[key] = {
      fullCandles: chartFullCandles,
      intervalSec: chartIntervalSec,
      pricePrecision: chartPricePrecision,
      zoomState: chartZoomState,
      userZoom: chartUserZoom,
      rangeSec: chartRangeSec,
      rangeLabel: chartRangeLabel,
      rangeKey: chartRangeKey,
    };
  }

  function restoreChartState(key) {
    const cached = chartStateCache[key];
    if (!cached) return false;
    chartFullCandles = Array.isArray(cached.fullCandles) ? cached.fullCandles : [];
    chartIntervalSec = Number(cached.intervalSec) || chartIntervalSec;
    chartPricePrecision = Number(cached.pricePrecision) || chartPricePrecision;
    chartZoomState = cached.zoomState || null;
    chartUserZoom = Boolean(cached.userZoom);
    chartRangeSec = cached.rangeSec ?? chartRangeSec;
    chartRangeLabel = cached.rangeLabel || chartRangeLabel;
    chartRangeKey = cached.rangeKey || chartRangeKey;
    chartNeedsFit = !chartUserZoom;
    syncChartRangeButtons();
    return chartFullCandles.length > 0;
  }

  function parseTs(ts) {
    if (ts === null || ts === undefined || ts === "") return null;
    if (typeof ts === "number") {
      const asNum = Number(ts);
      if (!Number.isFinite(asNum)) return null;
      return asNum > 1e12 ? Math.floor(asNum) : Math.floor(asNum * 1000);
    }
    if (typeof ts === "string") {
      const cleaned = ts.trim();
      if (!cleaned) return null;
      const numeric = Number(cleaned);
      if (Number.isFinite(numeric)) {
        return numeric > 1e12 ? Math.floor(numeric) : Math.floor(numeric * 1000);
      }
      const iso = cleaned.replace(" ", "T");
      const parsed = Date.parse(iso.endsWith("Z") ? iso : `${iso}Z`);
      if (!Number.isFinite(parsed)) return null;
      return parsed;
    }
    return null;
  }

  function inferPrecision(value) {
    if (!Number.isFinite(value) || value <= 0) return 6;
    const raw = value.toString();
    if (raw.includes("e")) return 6;
    const parts = raw.split(".");
    const decimals = parts.length > 1 ? parts[1].length : 0;
    return Math.min(8, Math.max(MIN_PRICE_PRECISION, decimals));
  }

  function applyPricePrecision(value) {
    const precision = inferPrecision(value);
    if (precision === chartPricePrecision) return;
    chartPricePrecision = precision;
  }

  function fmtPrice(value) {
    if (value === null || value === undefined) return "--";
    const num = Number(value);
    if (!Number.isFinite(num)) return "--";
    const precision = Math.max(MIN_PRICE_PRECISION, chartPricePrecision || 0);
    return num.toFixed(precision);
  }

  function fmtQty(value) {
    if (value === null || value === undefined) return "--";
    const num = Number(value);
    if (!Number.isFinite(num)) return "--";
    const abs = Math.abs(num);
    const precision = abs >= 1 ? 4 : 6;
    return num.toFixed(precision);
  }

  function stripQuoteSymbol(symbol) {
    if (!symbol) return "";
    const upper = String(symbol).toUpperCase();
    if (upper.endsWith("USDT") || upper.endsWith("USDC")) {
      return String(symbol).slice(0, -4);
    }
    return String(symbol);
  }

  function modelFolderSuffix(modelDir) {
    if (!modelDir) return "";
    const cleaned = String(modelDir).replace(/\\/g, "/");
    const parts = cleaned.split("/").filter(Boolean);
    if (!parts.length) return "";
    let folder = parts[parts.length - 1];
    if (folder.toLowerCase().startsWith("rank_") && parts.length >= 2) {
      folder = parts[parts.length - 2];
    }
    const idx = folder.indexOf("_");
    if (idx < 0 || idx >= folder.length - 1) return "";
    return folder.slice(idx + 1);
  }

  function shortModelLabel(data) {
    if (!data) return "";
    const symbol = data.symbol || data.dash_key || "";
    const base = stripQuoteSymbol(symbol);
    const suffix = modelFolderSuffix(data.model?.dir);
    if (base && suffix) return `${base} ${suffix}`;
    return base || suffix || "";
  }

  function setChartMeta(text) {
    const meta = document.getElementById("icebergChartMeta");
    if (meta) meta.textContent = text;
  }

  function chartTooltipFormatter(params) {
    if (!Array.isArray(params)) return "";
    let time = null;
    let candle = null;
    let lineValue = null;
    let volume = null;
    params.forEach(p => {
      if (p.seriesId === "primary") {
        const data = p.data || [];
        if (p.seriesType === "candlestick") {
          time = data[0];
          candle = { open: data[1], close: data[2], low: data[3], high: data[4] };
        } else {
          time = data[0];
          lineValue = Array.isArray(data) ? data[1] : p.value;
        }
      } else if (p.seriesId === "volume") {
        const data = p.data || [];
        if (Array.isArray(data)) {
          volume = data[1];
        } else if (Array.isArray(p.value)) {
          volume = p.value[1];
        } else {
          volume = p.value;
        }
      }
    });
    const timeText = time ? new Date(time).toLocaleString() : "--";
    const lines = [timeText];
    if (candle) {
      lines.push(`O ${fmtPrice(candle.open)} H ${fmtPrice(candle.high)} L ${fmtPrice(candle.low)} C ${fmtPrice(candle.close)}`);
    } else if (lineValue !== null && lineValue !== undefined) {
      lines.push(`${DT("dash_price")} ${fmtPrice(lineValue)}`);
    }
    if (volume !== null && volume !== undefined && volume !== 0) {
      lines.push(`${DT("dash_volume")} ${fmtQty(volume)}`);
    }
    return lines.join("<br/>");
  }

  function baseChartOption() {
    const axisLine = { lineStyle: { color: "#1f2937" } };
    const splitLine = { show: true, lineStyle: { color: "#1f2937" } };
    return {
      backgroundColor: "transparent",
      animation: false,
      axisPointer: { link: [{ xAxisIndex: [0, 1] }] },
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "cross", label: { color: "#e5e7eb" } },
        backgroundColor: "rgba(15, 23, 42, 0.92)",
        borderColor: "#1f2937",
        textStyle: { color: "#e5e7eb", fontFamily: "Space Grotesk, sans-serif", fontSize: 12 },
        formatter: chartTooltipFormatter,
      },
      grid: [
        { left: 56, right: 56, top: 20, height: "62%" },
        { left: 56, right: 56, top: "70%", height: "20%" },
      ],
      xAxis: [
        { type: "time", boundaryGap: true, axisLine, axisLabel: { color: "#94a3b8" }, splitLine },
        { type: "time", gridIndex: 1, boundaryGap: true, axisLine, axisLabel: { show: false }, axisTick: { show: false }, splitLine: { show: false } },
      ],
      yAxis: [
        { type: "value", scale: true, axisLine, axisLabel: { color: "#94a3b8", formatter: (val) => fmtPrice(val) }, splitLine },
        { type: "value", gridIndex: 1, scale: true, axisLine, axisLabel: { color: "#64748b", fontSize: 10 }, splitLine: { show: false } },
      ],
      dataZoom: buildDataZoom([]),
      series: [],
    };
  }

  function buildDataZoom(filtered) {
    const zoom = {};
    const shouldAuto = !chartUserZoom || chartNeedsFit;
    if (chartUserZoom && chartZoomState) {
      if (chartZoomState.start !== undefined && chartZoomState.end !== undefined) {
        zoom.start = chartZoomState.start;
        zoom.end = chartZoomState.end;
      } else if (chartZoomState.startValue !== undefined && chartZoomState.endValue !== undefined) {
        zoom.startValue = chartZoomState.startValue;
        zoom.endValue = chartZoomState.endValue;
      }
    } else if (shouldAuto && filtered && filtered.length) {
      zoom.startValue = filtered[0].time;
      zoom.endValue = filtered[filtered.length - 1].time;
    }
    const inside = {
      type: "inside",
      xAxisIndex: [0, 1],
      filterMode: "none",
      zoomOnMouseWheel: true,
      moveOnMouseWheel: true,
      moveOnMouseMove: true,
    };
    const slider = {
      type: "slider",
      xAxisIndex: [0, 1],
      bottom: 6,
      height: 18,
      borderColor: "#1f2937",
      fillerColor: "rgba(56, 189, 248, 0.18)",
      backgroundColor: "rgba(15, 23, 42, 0.6)",
      handleStyle: { color: "#38bdf8", borderColor: "#38bdf8" },
      textStyle: { color: "#94a3b8" },
    };
    return [Object.assign({}, inside, zoom), Object.assign({}, slider, zoom)];
  }

  function applyChartOption(series, filtered) {
    if (!chart) return;
    chart.setOption({ series, dataZoom: buildDataZoom(filtered || []) }, { lazyUpdate: true, replaceMerge: ["series"] });
    chartNeedsFit = false;
    applyChartOverlay();
  }

  function renderCandles(filtered) {
    const candleData = (filtered || []).map(c => [c.time, c.open, c.close, c.low, c.high]);
    const volumeData = (filtered || []).map(c => ({
      value: [c.time, c.volume || 0],
      itemStyle: { color: c.close >= c.open ? "rgba(34, 197, 94, 0.45)" : "rgba(239, 68, 68, 0.45)" },
    }));
    const series = [
      {
        id: "primary",
        name: DT("dash_price"),
        type: "candlestick",
        data: candleData,
        encode: { x: 0, y: [1, 2, 3, 4] },
        itemStyle: { color: "#22c55e", color0: "#ef4444", borderColor: "#22c55e", borderColor0: "#ef4444" },
      },
      {
        id: "volume",
        name: DT("dash_volume"),
        type: "bar",
        xAxisIndex: 1,
        yAxisIndex: 1,
        barWidth: "60%",
        data: volumeData,
        encode: { x: 0, y: 1 },
      },
    ];
    applyChartOption(series, filtered);
  }

  function renderLine(points) {
    const lineData = (points || []).map(p => [p.time, p.value]);
    const series = [
      {
        id: "primary",
        name: DT("dash_price"),
        type: "line",
        data: lineData,
        encode: { x: 0, y: 1 },
        showSymbol: false,
        lineStyle: { color: "#38bdf8", width: 1.4 },
        emphasis: { focus: "series" },
      },
      {
        id: "volume",
        name: DT("dash_volume"),
        type: "bar",
        xAxisIndex: 1,
        yAxisIndex: 1,
        barWidth: "60%",
        data: [],
        encode: { x: 0, y: 1 },
      },
    ];
    applyChartOption(series, points);
  }

  function applyChartOverlay() {
    if (!chartInitialized || !chart) return;
    const seriesList = chart.getOption().series || [];
    if (!seriesList.some(s => s.id === "primary")) return;
    const lines = [];
    (chartOverlay.lines || []).forEach(item => {
      if (item.group && !isLegendEnabled(item.group)) return;
      const price = Number(item.price || 0);
      if (!Number.isFinite(price) || price <= 0) return;
      lines.push({
        yAxis: price,
        name: item.title || "",
        lineStyle: { color: item.color || "#94a3b8", width: 1, type: "dashed" },
        label: { show: true, formatter: "{b}", color: item.color || "#e5e7eb", fontSize: 10 },
      });
    });
    if (chartOverlay.livePrice && Number.isFinite(chartOverlay.livePrice)) {
      const lastPrice = Number(chartOverlay.livePrice);
      lines.push({
        yAxis: lastPrice,
        name: `${DT("dash_last")} ${fmtPrice(lastPrice)}`,
        lineStyle: { color: "#38bdf8", width: 1, type: "solid" },
        label: { show: true, formatter: "{b}", color: "#38bdf8", fontSize: 10 },
      });
    }
    const markers = (chartOverlay.markers || [])
      .filter(m => Number.isFinite(m.time) && Number.isFinite(m.price))
      .filter(m => !m.group || isLegendEnabled(m.group))
      .map(m => ({
        coord: [m.time, m.price],
        name: m.title || "",
        symbol: m.symbol || "pin",
        symbolSize: m.symbolSize || 32,
        itemStyle: { color: m.color || "#38bdf8" },
        label: { show: true, color: m.labelColor || "#0b1220", fontSize: 9, formatter: "{b}" },
      }));
    chart.setOption({
      series: [
        {
          id: "primary",
          markLine: { symbol: ["none", "none"], label: { show: true, formatter: "{b}" }, data: lines },
          markPoint: { symbol: "pin", symbolSize: 32, label: { show: true, formatter: "{b}" }, data: markers },
        },
      ],
    }, { lazyUpdate: true });
  }

  function normalizeCandles(rows) {
    if (!Array.isArray(rows)) return [];
    const out = [];
    rows.forEach(row => {
      const rawTime = row.time || row.t || row.timestamp || 0;
      const timeNum = Number(rawTime);
      const time = Number.isFinite(timeNum) ? (timeNum > 1e12 ? Math.floor(timeNum) : Math.floor(timeNum * 1000)) : 0;
      const open = Number(row.open || 0);
      const high = Number(row.high || 0);
      const low = Number(row.low || 0);
      const close = Number(row.close || 0);
      if (!time || !open || !high || !low || !close) return;
      const volume = Number(row.volume || 0);
      out.push({
        time: Math.floor(time),
        open,
        high,
        low,
        close,
        volume: Number.isFinite(volume) ? volume : 0,
      });
    });
    out.sort((a, b) => a.time - b.time);
    return out;
  }

  function computeCandleLimit() {
    if (!chartRangeSec) return 20000;
    const interval = chartIntervalSec > 0 ? chartIntervalSec : 60;
    const desired = Math.ceil(chartRangeSec / interval) + 120;
    return Math.min(20000, Math.max(300, desired));
  }

  function computeDirectionLimit() {
    const sampleSec = window.__dashMetricsIntervalSec || 2;
    const targetRange = chartRangeSec || (24 * 60 * 60);
    const desired = Math.ceil(targetRange / sampleSec) + 200;
    return Math.min(10000, Math.max(300, desired));
  }

  function filterCandles(candles) {
    if (!chartRangeSec || !candles.length) return candles;
    const last = candles[candles.length - 1];
    const cutoff = last.time - chartRangeSec * 1000;
    return candles.filter(c => c.time >= cutoff);
  }

  function setCandlesData(candles) {
    chartFullCandles = candles || [];
    const filtered = filterCandles(chartFullCandles);
    if (!filtered.length) {
      renderCandles([]);
      setChartMeta(DT("dash_no_chart_data"));
      return;
    }
    renderCandles(filtered);
    const last = filtered[filtered.length - 1];
    applyPricePrecision(last.close);
    const interval = chartIntervalSec ? `${chartIntervalSec}s` : "--";
    const lastTime = new Date(last.time).toLocaleTimeString();
    const label = chartRangeLabel || DT("dash_all");
    setChartMeta(`${DT("dash_range")} ${label} | ${DT("dash_bars")} ${filtered.length} | ${DT("dash_last")} ${fmtPrice(last.close)} | ${lastTime} | ${DT("dash_interval")} ${interval}`);
  }

  function metricPrice(m) {
    const market = m.market || {};
    const pos = m.position || {};
    const l1 = (m.iceberg && m.iceberg.l1) ? m.iceberg.l1 : {};
    const bid = Number(l1.bid || 0);
    const ask = Number(l1.ask || 0);
    const mid = (bid > 0 && ask > 0) ? (bid + ask) * 0.5 : 0;
    const price = Number(market.last_close || mid || pos.mark_price || pos.entry_price || 0);
    return Number.isFinite(price) ? price : 0;
  }

  async function fetchMetricHistory(key) {
    if (!chartInitialized) return;
    const sampleSec = window.__dashMetricsIntervalSec || 2;
    const targetRange = chartRangeSec || (24 * 60 * 60);
    const desired = Math.ceil(targetRange / sampleSec) + 200;
    const limit = Math.min(10000, Math.max(300, desired));
    const resp = await fetch(`/api/metrics?key=${encodeURIComponent(key)}&limit=${limit}`);
    if (!resp.ok) return;
    const data = await resp.json();
    if (!Array.isArray(data)) return;
    const points = data.map(m => {
      const time = parseTs(m.ts);
      const value = metricPrice(m);
      if (!time || !value) return null;
      return { time, value };
    }).filter(Boolean);
    if (!points.length) {
      setChartMeta(DT("dash_no_chart_data"));
      return;
    }
    const now = Date.now();
    const cutoff = chartRangeSec ? (now - chartRangeSec * 1000) : 0;
    const filtered = cutoff > 0 ? points.filter(p => p.time >= cutoff) : points;
    if (!filtered.length) {
      setChartMeta(DT("dash_no_data_range"));
      return;
    }
    chartFullCandles = [];
    renderLine(filtered);
    const last = filtered[filtered.length - 1];
    applyPricePrecision(last.value);
    const label = chartRangeLabel || DT("dash_all");
    setChartMeta(`${DT("dash_range")} ${label} | ${DT("dash_points")} ${filtered.length} | ${DT("dash_last")} ${fmtPrice(last.value)} | ${DT("dash_metrics_feed")}`);
  }

  async function fetchDirectionHistory(force = false) {
    if (!chartInitialized) return;
    const key = window.__dashKey || window.__dashSymbol;
    if (!key || chartDirectionFetchInFlight) return;
    const now = Date.now();
    if (!force && chartDirectionKey === key && (now - chartDirectionLastFetch) < 8000) {
      return;
    }
    chartDirectionFetchInFlight = true;
    chartDirectionLastFetch = now;
    const currentKey = key;
    try {
      const limit = computeDirectionLimit();
      const resp = await fetch(`/api/metrics?key=${encodeURIComponent(key)}&limit=${limit}`);
      if (!resp.ok) return;
      const data = await resp.json();
      if (!Array.isArray(data)) return;
      if (currentKey !== (window.__dashKey || window.__dashSymbol)) return;
      chartDirectionHistory = data;
      chartDirectionKey = currentKey;
    } catch (_err) {
      return;
    } finally {
      chartDirectionFetchInFlight = false;
    }
  }

  async function fetchChartHistory(force = false) {
    if (!chartInitialized) return;
    const key = window.__dashKey || window.__dashSymbol;
    if (!key || chartFetchInFlight) return;
    if (force) {
      chartNeedsFit = true;
      chartUserZoom = false;
      chartZoomState = null;
    }
    chartFetchInFlight = true;
    const currentKey = key;
    try {
      const limit = computeCandleLimit();
      const resp = await fetch(`/api/candles?key=${encodeURIComponent(key)}&limit=${limit}`);
      if (!resp.ok) {
        await fetchMetricHistory(key);
        return;
      }
      const payload = await resp.json();
      if (currentKey !== (window.__dashKey || window.__dashSymbol)) return;
      if (payload && payload.interval_sec) {
        const interval = Number(payload.interval_sec);
        if (Number.isFinite(interval) && interval > 0) {
          chartIntervalSec = interval;
        }
      }
      const candles = normalizeCandles(payload ? payload.candles : []);
      if (!candles.length) {
        await fetchMetricHistory(key);
        return;
      }
      setCandlesData(candles);
    } catch (_err) {
      await fetchMetricHistory(key);
    } finally {
      chartFetchInFlight = false;
    }
  }

  function updateLivePriceLine(price) {
    if (!price || !chartInitialized) return;
    applyPricePrecision(price);
    if (chartOverlay.livePrice === price) return;
    chartOverlay.livePrice = price;
    applyChartOverlay();
  }

  function resolveMarkerTime(ts) {
    const time = parseTs(ts);
    if (!time) return null;
    if (!chartFullCandles.length) return time;
    for (let i = chartFullCandles.length - 1; i >= 0; i--) {
      if (chartFullCandles[i].time <= time) return chartFullCandles[i].time;
    }
    return chartFullCandles[0].time;
  }

  function resolveMarkerPrice(time, fallback) {
    const base = Number(fallback || 0);
    if (!chartFullCandles.length) return base > 0 ? base : 0;
    for (let i = chartFullCandles.length - 1; i >= 0; i--) {
      const candle = chartFullCandles[i];
      if (candle.time <= time) {
        return Number(candle.close || candle.open || base || 0);
      }
    }
    const first = chartFullCandles[0];
    return Number(first?.close || first?.open || base || 0);
  }

  function numOrNaN(value) {
    if (value === null || value === undefined || value === "") return NaN;
    const num = Number(value);
    return Number.isFinite(num) ? num : NaN;
  }

  function strengthRatio(value, threshold) {
    if (!Number.isFinite(value) || !Number.isFinite(threshold)) return 0;
    if (value <= threshold) return 0;
    const span = Math.max(0.0001, 1 - threshold);
    return Math.max(0, Math.min(1, (value - threshold) / span));
  }

  function updateChartMarkers(data) {
    const markers = [];
    const ice = data.iceberg || {};
    const tp = data.iceberg_tp || {};
    const pushMarker = (time, price, color, title, symbol, symbolSize, labelColor, group) => {
      const t = Number(time || 0);
      const p = Number(price || 0);
      if (!Number.isFinite(t) || t <= 0 || !Number.isFinite(p) || p <= 0) return;
      markers.push({ time: t, price: p, color, title, symbol, symbolSize, labelColor, group });
    };
    if (ice.buy && ice.buy.active) {
      const time = resolveMarkerTime(ice.buy.last_post_ts || ice.buy.created_ts);
      const price = Number(ice.buy.last_price || ice.buy.target_price || 0);
      const info = ice.buy.last_post_ts ? icebergClipLabel(ice.buy) : icebergEntryLabel(ice.buy);
      const label = ice.buy.last_post_ts ? DT("dash_buy_clip") : DT("dash_entry_buy");
      pushMarker(time, price, "#22c55e", lineTitle(label, info), undefined, undefined, undefined, "orders");
    }
    if (ice.sell && ice.sell.active) {
      const time = resolveMarkerTime(ice.sell.last_post_ts || ice.sell.created_ts);
      const price = Number(ice.sell.last_price || ice.sell.target_price || 0);
      const info = ice.sell.last_post_ts ? icebergClipLabel(ice.sell) : icebergEntryLabel(ice.sell);
      const label = ice.sell.last_post_ts ? DT("dash_sell_clip") : DT("dash_entry_sell");
      pushMarker(time, price, "#f97316", lineTitle(label, info), undefined, undefined, undefined, "orders");
    }
    if (tp.long && (tp.long.active || tp.long.paused)) {
      const time = resolveMarkerTime(tp.long.tp_order_ts || tp.long.created_ts);
      const price = Number(tp.long.tp_order_price || tp.long.tp_target || 0);
      pushMarker(time, price, "#60a5fa", lineTitle(DT("dash_tp_long"), tpInfoLabel(tp.long)), undefined, undefined, undefined, "orders");
    }
    if (tp.short && (tp.short.active || tp.short.paused)) {
      const time = resolveMarkerTime(tp.short.tp_order_ts || tp.short.created_ts);
      const price = Number(tp.short.tp_order_price || tp.short.tp_target || 0);
      pushMarker(time, price, "#f59e0b", lineTitle(DT("dash_tp_short"), tpInfoLabel(tp.short)), undefined, undefined, undefined, "orders");
    }
    const prediction = data.prediction || data.health?.last_prediction || {};
    const predThreshold = numOrNaN(prediction.threshold);
    const predTime = resolveMarkerTime(prediction.bar_time || data.market?.last_bar_time || data.ts);
    const predLong = numOrNaN(prediction.pred_long);
    const predShort = numOrNaN(prediction.pred_short);

    const direction = data.direction || {};
    const dirThreshold = numOrNaN(direction.threshold);
    const dirAggressive = numOrNaN(direction.aggressive_threshold);
    const dirTime = resolveMarkerTime(direction.bar_time || prediction.bar_time || data.market?.last_bar_time || data.ts);
    const dirLong = numOrNaN(direction.pred_long);
    const dirShort = numOrNaN(direction.pred_short);

    const cuePositions = (price) => {
      const baseOffset = price > 0 ? price * 0.0012 : 0;
      const step = price > 0 ? price * 0.0009 : 0;
      return {
        long: {
          pred: price + baseOffset,
          dir: price + baseOffset + step,
          gate: price + baseOffset + step * 2,
        },
        short: {
          pred: price - baseOffset,
          dir: price - baseOffset - step,
          gate: price - baseOffset - step * 2,
        },
      };
    };

    const historyKey = window.__dashKey || window.__dashSymbol;
    const metricsHistory = (chartDirectionKey === historyKey && Array.isArray(chartDirectionHistory))
      ? chartDirectionHistory
      : [];
    const useHistory = metricsHistory.length > 0;
    const barKeyFrom = (value) => {
      if (value === null || value === undefined || value === "") return null;
      const num = Number(value);
      if (Number.isFinite(num) && num > 0) {
        return num > 1e12 ? Math.floor(num / 1000) : Math.floor(num);
      }
      const parsed = parseTs(value);
      return parsed ? Math.floor(parsed / 1000) : null;
    };

    if (useHistory) {
      const byBar = new Map();
      metricsHistory.forEach(item => {
        const pred = item.prediction || item.health?.last_prediction || {};
        const dir = item.direction || {};
        if (!dir && !pred) return;
        const barKey = barKeyFrom(dir.bar_time || pred.bar_time || item.market?.last_bar_time || item.bar_time || item.ts);
        if (!barKey) return;
        const tsMs = item.ts ? (parseTs(String(item.ts)) || 0) : 0;
        const existing = byBar.get(barKey);
        if (!existing || (tsMs && tsMs >= (existing.tsMs || 0))) {
          byBar.set(barKey, {
            barKey,
            tsMs,
            predLong: numOrNaN(pred.pred_long),
            predShort: numOrNaN(pred.pred_short),
            predThreshold: numOrNaN(pred.threshold),
            dirLong: numOrNaN(dir.pred_long),
            dirShort: numOrNaN(dir.pred_short),
            dirThreshold: numOrNaN(dir.threshold),
            dirAggressive: numOrNaN(dir.aggressive_threshold),
          });
        }
      });

      byBar.forEach(entry => {
        const time = resolveMarkerTime(entry.barKey);
        if (!time) return;
        const basePrice = resolveMarkerPrice(time, resolveLivePrice(data));
        if (basePrice <= 0) return;
        const pos = cuePositions(basePrice);
        if (Number.isFinite(entry.predThreshold)) {
          if (Number.isFinite(entry.predLong) && entry.predLong >= entry.predThreshold) {
            const strength = strengthRatio(entry.predLong, entry.predThreshold);
            const size = 14 + Math.round(strength * 12);
            const label = `PL ${entry.predLong.toFixed(2)}`;
            pushMarker(time, pos.long.pred, "#22c55e", label, "triangle", size, "#0b1220", "pred");
          }
          if (Number.isFinite(entry.predShort) && entry.predShort >= entry.predThreshold) {
            const strength = strengthRatio(entry.predShort, entry.predThreshold);
            const size = 14 + Math.round(strength * 12);
            const label = `PS ${entry.predShort.toFixed(2)}`;
            pushMarker(time, pos.short.pred, "#ef4444", label, "triangle", size, "#0b1220", "pred");
          }
        }
        if (Number.isFinite(entry.dirThreshold) && Number.isFinite(entry.dirAggressive)) {
          if (Number.isFinite(entry.dirLong) && entry.dirLong >= entry.dirThreshold) {
            const strength = strengthRatio(entry.dirLong, entry.dirThreshold);
            const aggressive = Number.isFinite(entry.dirAggressive) && entry.dirLong >= entry.dirAggressive;
            const size = 16 + Math.round(strength * 14) + (aggressive ? 10 : 0);
            const label = `${aggressive ? "AGG " : ""}DL ${entry.dirLong.toFixed(2)}`;
            const color = aggressive ? "#0ea5e9" : "#38bdf8";
            pushMarker(time, pos.long.dir, color, label, aggressive ? "diamond" : "circle", size, "#0b1220", "dir");
          }
          if (Number.isFinite(entry.dirShort) && entry.dirShort >= entry.dirThreshold) {
            const strength = strengthRatio(entry.dirShort, entry.dirThreshold);
            const aggressive = Number.isFinite(entry.dirAggressive) && entry.dirShort >= entry.dirAggressive;
            const size = 16 + Math.round(strength * 14) + (aggressive ? 10 : 0);
            const label = `${aggressive ? "AGG " : ""}DS ${entry.dirShort.toFixed(2)}`;
            const color = aggressive ? "#ec4899" : "#f472b6";
            pushMarker(time, pos.short.dir, color, label, aggressive ? "diamond" : "circle", size, "#0b1220", "dir");
          }
        }
        const gateEnabled = Number.isFinite(entry.dirThreshold) && Number.isFinite(entry.dirAggressive);
        if (gateEnabled && Number.isFinite(entry.predThreshold)) {
          if (
            Number.isFinite(entry.predLong)
            && Number.isFinite(entry.dirLong)
            && entry.predLong >= entry.predThreshold
            && entry.dirLong >= entry.dirThreshold
          ) {
            const strength = Math.min(
              strengthRatio(entry.predLong, entry.predThreshold),
              strengthRatio(entry.dirLong, entry.dirThreshold)
            );
            const size = 18 + Math.round(strength * 12);
            const label = `GL ${entry.predLong.toFixed(2)}`;
            pushMarker(time, pos.long.gate, "#facc15", label, "pin", size, "#0b1220", "gate");
          }
          if (
            Number.isFinite(entry.predShort)
            && Number.isFinite(entry.dirShort)
            && entry.predShort >= entry.predThreshold
            && entry.dirShort >= entry.dirThreshold
          ) {
            const strength = Math.min(
              strengthRatio(entry.predShort, entry.predThreshold),
              strengthRatio(entry.dirShort, entry.dirThreshold)
            );
            const size = 18 + Math.round(strength * 12);
            const label = `GS ${entry.predShort.toFixed(2)}`;
            pushMarker(time, pos.short.gate, "#f97316", label, "pin", size, "#0b1220", "gate");
          }
        }
      });
    } else {
      const gateEnabled = Number.isFinite(dirThreshold) && Number.isFinite(dirAggressive);
      if (predTime && Number.isFinite(predThreshold)) {
        const basePrice = resolveMarkerPrice(predTime, resolveLivePrice(data));
        if (basePrice > 0) {
          const pos = cuePositions(basePrice);
          if (Number.isFinite(predLong) && predLong >= predThreshold) {
            const strength = strengthRatio(predLong, predThreshold);
            const size = 14 + Math.round(strength * 12);
            const label = `PL ${predLong.toFixed(2)}`;
            pushMarker(predTime, pos.long.pred, "#22c55e", label, "triangle", size, "#0b1220", "pred");
          }
          if (Number.isFinite(predShort) && predShort >= predThreshold) {
            const strength = strengthRatio(predShort, predThreshold);
            const size = 14 + Math.round(strength * 12);
            const label = `PS ${predShort.toFixed(2)}`;
            pushMarker(predTime, pos.short.pred, "#ef4444", label, "triangle", size, "#0b1220", "pred");
          }
        }
      }

      if (dirTime && Number.isFinite(dirThreshold) && Number.isFinite(dirAggressive)) {
        const basePrice = resolveMarkerPrice(dirTime, resolveLivePrice(data));
        if (basePrice > 0) {
          const pos = cuePositions(basePrice);
          if (Number.isFinite(dirLong) && dirLong >= dirThreshold) {
            const strength = strengthRatio(dirLong, dirThreshold);
            const aggressive = Number.isFinite(dirAggressive) && dirLong >= dirAggressive;
            const size = 16 + Math.round(strength * 14) + (aggressive ? 10 : 0);
            const label = `${aggressive ? "AGG " : ""}DL ${dirLong.toFixed(2)}`;
            const color = aggressive ? "#0ea5e9" : "#38bdf8";
            pushMarker(dirTime, pos.long.dir, color, label, aggressive ? "diamond" : "circle", size, "#0b1220", "dir");
          }
          if (Number.isFinite(dirShort) && dirShort >= dirThreshold) {
            const strength = strengthRatio(dirShort, dirThreshold);
            const aggressive = Number.isFinite(dirAggressive) && dirShort >= dirAggressive;
            const size = 16 + Math.round(strength * 14) + (aggressive ? 10 : 0);
            const label = `${aggressive ? "AGG " : ""}DS ${dirShort.toFixed(2)}`;
            const color = aggressive ? "#ec4899" : "#f472b6";
            pushMarker(dirTime, pos.short.dir, color, label, aggressive ? "diamond" : "circle", size, "#0b1220", "dir");
          }
        }
      }

      if (predTime && Number.isFinite(predThreshold) && gateEnabled) {
        const basePrice = resolveMarkerPrice(predTime, resolveLivePrice(data));
        if (basePrice > 0) {
          const pos = cuePositions(basePrice);
          if (Number.isFinite(predLong) && Number.isFinite(dirLong) && predLong >= predThreshold && dirLong >= dirThreshold) {
            const strength = Math.min(strengthRatio(predLong, predThreshold), strengthRatio(dirLong, dirThreshold));
            const size = 18 + Math.round(strength * 12);
            const label = `GL ${predLong.toFixed(2)}`;
            pushMarker(predTime, pos.long.gate, "#facc15", label, "pin", size, "#0b1220", "gate");
          }
          if (Number.isFinite(predShort) && Number.isFinite(dirShort) && predShort >= predThreshold && dirShort >= dirThreshold) {
            const strength = Math.min(strengthRatio(predShort, predThreshold), strengthRatio(dirShort, dirThreshold));
            const size = 18 + Math.round(strength * 12);
            const label = `GS ${predShort.toFixed(2)}`;
            pushMarker(predTime, pos.short.gate, "#f97316", label, "pin", size, "#0b1220", "gate");
          }
        }
      }
    }
    chartOverlay.markers = markers;
    applyChartOverlay();
  }

  function updateChartLines(data) {
    const levels = [];
    const ice = data.iceberg || {};
    if (ice.buy && ice.buy.active) {
      levels.push({ price: ice.buy.target_price, color: "#22c55e", title: priceTitle(DT("dash_entry_buy"), ice.buy.target_price, icebergEntryLabel(ice.buy)) });
      if (ice.buy.tp) levels.push({ price: ice.buy.tp, color: "#38bdf8", title: priceTitle(DT("dash_entry_tp"), ice.buy.tp, "") });
      if (ice.buy.sl) levels.push({ price: ice.buy.sl, color: "#ef4444", title: priceTitle(DT("dash_entry_sl"), ice.buy.sl, "") });
      if (ice.buy.order_id && ice.buy.last_price) {
        levels.push({ price: ice.buy.last_price, color: "#16a34a", title: priceTitle(DT("dash_buy_clip"), ice.buy.last_price, icebergClipLabel(ice.buy)) });
      }
    }
    if (ice.sell && ice.sell.active) {
      levels.push({ price: ice.sell.target_price, color: "#f97316", title: priceTitle(DT("dash_entry_sell"), ice.sell.target_price, icebergEntryLabel(ice.sell)) });
      if (ice.sell.tp) levels.push({ price: ice.sell.tp, color: "#38bdf8", title: priceTitle(DT("dash_entry_tp"), ice.sell.tp, "") });
      if (ice.sell.sl) levels.push({ price: ice.sell.sl, color: "#ef4444", title: priceTitle(DT("dash_entry_sl"), ice.sell.sl, "") });
      if (ice.sell.order_id && ice.sell.last_price) {
        levels.push({ price: ice.sell.last_price, color: "#ea580c", title: priceTitle(DT("dash_sell_clip"), ice.sell.last_price, icebergClipLabel(ice.sell)) });
      }
    }
    const tp = data.iceberg_tp || {};
    if (tp.long && (tp.long.active || tp.long.paused)) {
      if (tp.long.tp_target) levels.push({ price: tp.long.tp_target, color: "#60a5fa", title: priceTitle(DT("dash_tp_long"), tp.long.tp_target, tpInfoLabel(tp.long)) });
      if (tp.long.sl_target) levels.push({ price: tp.long.sl_target, color: "#ef4444", title: priceTitle(DT("dash_sl_long"), tp.long.sl_target, "") });
      if (tp.long.tp_order_price) levels.push({ price: tp.long.tp_order_price, color: "#0ea5e9", title: priceTitle(DT("dash_tp_clip"), tp.long.tp_order_price, tpInfoLabel(tp.long)) });
    }
    if (tp.short && (tp.short.active || tp.short.paused)) {
      if (tp.short.tp_target) levels.push({ price: tp.short.tp_target, color: "#f59e0b", title: priceTitle(DT("dash_tp_short"), tp.short.tp_target, tpInfoLabel(tp.short)) });
      if (tp.short.sl_target) levels.push({ price: tp.short.sl_target, color: "#ef4444", title: priceTitle(DT("dash_sl_short"), tp.short.sl_target, "") });
      if (tp.short.tp_order_price) levels.push({ price: tp.short.tp_order_price, color: "#d97706", title: priceTitle(DT("dash_tp_clip"), tp.short.tp_order_price, tpInfoLabel(tp.short)) });
    }
    chartOverlay.lines = levels.filter(level => Number(level.price) > 0).map(level => {
      if (level.group) return level;
      return Object.assign({ group: "orders" }, level);
    });
    applyChartOverlay();
  }

  let controlsBound = false;
  let chartControlsBound = false;
  let chartRangeSec = 15 * 60;
  let chartRangeLabel = "15M";
  let chartRangeKey = "15m";
  async function sendCommand(scope, side) {
    const key = window.__dashKey;
    const symbol = window.__dashSymbol;
    if (!symbol && !key) return;
    const payload = {
      ts: Date.now() / 1000,
      action: "cancel_iceberg",
      scope,
      side,
    };
    if (symbol) payload.symbol = symbol;
    if (key) payload.key = key;
    try {
      await fetch("/api/command", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    } catch (err) {
      console.error("command failed", err);
    }
  }

  function bindIcebergControls() {
    if (controlsBound) return;
    const buyBtn = document.getElementById("icebergCancelBuy");
    const sellBtn = document.getElementById("icebergCancelSell");
    const tpLongBtn = document.getElementById("icebergCancelTpLong");
    const tpShortBtn = document.getElementById("icebergCancelTpShort");
    if (!buyBtn || !sellBtn || !tpLongBtn || !tpShortBtn) return;
    buyBtn.addEventListener("click", () => sendCommand("entry", "buy"));
    sellBtn.addEventListener("click", () => sendCommand("entry", "sell"));
    tpLongBtn.addEventListener("click", () => sendCommand("tp", "long"));
    tpShortBtn.addEventListener("click", () => sendCommand("tp", "short"));
    controlsBound = true;
  }

  function bindChartRangeControls() {
    if (chartControlsBound) return;
    const rangeWrap = document.getElementById("chartRange");
    if (!rangeWrap) return;
    const mapping = {
      "15m": { label: "15M", sec: 15 * 60 },
      "30m": { label: "30M", sec: 30 * 60 },
      "1h": { label: "1H", sec: 60 * 60 },
      "4h": { label: "4H", sec: 4 * 60 * 60 },
      "6h": { label: "6H", sec: 6 * 60 * 60 },
      "24h": { label: "24H", sec: 24 * 60 * 60 },
      "1d": { label: "1D", sec: 24 * 60 * 60 },
      "all": { label: DT("dash_all"), sec: 0 },
    };
    rangeWrap.querySelectorAll(".range-btn").forEach(btn => {
        btn.addEventListener("click", () => {
          rangeWrap.querySelectorAll(".range-btn").forEach(b => b.classList.remove("active"));
          btn.classList.add("active");
          const key = (btn.dataset.range || "").toLowerCase();
          const meta = mapping[key];
          if (meta) {
            chartNeedsFit = true;
            chartUserZoom = false;
            chartZoomState = null;
            chartRangeSec = meta.sec;
            chartRangeLabel = meta.label;
            chartRangeKey = key;
            if (chartFullCandles.length) {
              setCandlesData(chartFullCandles);
            } else {
              fetchChartHistory(true);
            }
        }
      });
    });
    chartControlsBound = true;
  }

  function bindChartLegendControls() {
    const wrap = document.getElementById("chartLegend");
    if (!wrap) return;
    if (!chartLegendBound) {
      wrap.querySelectorAll(".toggle-btn").forEach(btn => {
        btn.addEventListener("click", () => {
          const key = (btn.dataset.legend || "").toLowerCase();
          if (!key) return;
          const next = !isLegendEnabled(key);
          chartLegend[key] = next;
          btn.classList.toggle("active", next);
          saveChartLegend();
          applyChartOverlay();
        });
      });
      chartLegendBound = true;
    }
    syncChartLegendButtons();
  }

  function formatOrders(list) {
    if (!Array.isArray(list) || !list.length) return "--";
    return list.slice(0, 4).map(item => {
      const side = item.side || "?";
      const price = fmtPrice(item.price);
      const qty = fmtQty(item.qty);
      return `${side} ${price} x${qty}`;
    }).join("\\n");
  }

  function formatIcebergSide(ice) {
    if (!ice || !ice.active) return "--";
    const target = fmtPrice(ice.target_price);
    const tp = fmtPrice(ice.tp);
    const sl = fmtPrice(ice.sl);
    const rem = fmtQty(ice.remaining_qty);
    const total = fmtQty(ice.total_qty);
    const clip = fmtQty(ice.clip_qty);
    const state = ice.order_id ? DT("dash_posted") : DT("dash_waiting_state");
    const ready = (ice.activation_ready === null || ice.activation_ready === undefined)
      ? "--"
      : (ice.activation_ready ? DT("dash_ready") : DT("dash_waiting_state"));
    return `${DT("dash_target_short")} ${target} ${DT("dash_tp")} ${tp} ${DT("dash_sl")} ${sl}\\n${DT("dash_remaining_short")} ${rem}/${total} ${DT("dash_clip")} ${clip} ${state} ${ready}`;
  }

  function formatTpIceberg(tp) {
    if (!tp || (!tp.active && !tp.paused)) return "--";
    const target = fmtPrice(tp.tp_target);
    const sl = fmtPrice(tp.sl_target);
    const size = fmtQty(tp.size);
    const clipQty = fmtQty(tp.tp_order_qty);
    const clipPrice = fmtPrice(tp.tp_order_price);
    const clip = (tp.tp_order_qty && tp.tp_order_price) ? `${clipQty} @ ${clipPrice}` : "--";
    const state = tp.paused ? DT("dash_paused") : (tp.tp_order_id ? DT("dash_posted") : DT("dash_waiting_state"));
    const ready = (tp.activation_ready === null || tp.activation_ready === undefined)
      ? "--"
      : (tp.activation_ready ? DT("dash_ready") : DT("dash_waiting_state"));
    return `${DT("dash_tp")} ${target} ${DT("dash_sl")} ${sl}\\n${DT("dash_size")} ${size} ${DT("dash_clip")} ${clip} ${state} ${ready}`;
  }

  function icebergStateLabel(ice) {
    if (!ice || !ice.active) return "";
    if (ice.order_id) return DT("dash_posted");
    if (ice.activation_ready === true) return DT("dash_ready");
    if (ice.activation_ready === false) return DT("dash_waiting_state");
    return DT("dash_waiting_state");
  }

  function icebergFillLabel(ice) {
    if (!ice) return "";
    const total = Number(ice.total_qty || 0);
    if (!total) return "";
    const filled = Number(ice.filled_qty || 0);
    return `${fmtQty(filled)}/${fmtQty(total)}`;
  }

  function icebergEntryLabel(ice) {
    if (!ice || !ice.active) return "";
    const parts = [];
    const total = Number(ice.total_qty || 0);
    if (total > 0) parts.push(`${DT("dash_size")} ${fmtQty(total)}`);
    const fill = icebergFillLabel(ice);
    if (fill) parts.push(`${DT("dash_fill")} ${fill}`);
    const state = icebergStateLabel(ice);
    if (state) parts.push(state);
    return parts.join(" ");
  }

  function icebergClipLabel(ice) {
    if (!ice || !ice.active) return "";
    const parts = [];
    const clip = Number(ice.clip_qty || 0);
    if (clip > 0) parts.push(`${DT("dash_clip")} ${fmtQty(clip)}`);
    const state = icebergStateLabel(ice);
    if (state) parts.push(state);
    return parts.join(" ");
  }

  function tpStateLabel(tp) {
    if (!tp) return "";
    if (tp.paused) return DT("dash_paused");
    if (tp.tp_order_id) return DT("dash_posted");
    if (tp.activation_ready === true) return DT("dash_ready");
    if (tp.activation_ready === false) return DT("dash_waiting_state");
    return DT("dash_waiting_state");
  }

  function tpInfoLabel(tp) {
    if (!tp || (!tp.active && !tp.paused)) return "";
    const parts = [];
    const size = Number(tp.size || 0);
    if (size > 0) parts.push(`${DT("dash_size")} ${fmtQty(size)}`);
    const clip = Number(tp.tp_order_qty || 0);
    if (clip > 0) parts.push(`${DT("dash_clip")} ${fmtQty(clip)}`);
    const state = tpStateLabel(tp);
    if (state) parts.push(state);
    return parts.join(" ");
  }

  function lineTitle(base, info) {
    if (!info) return base;
    return `${base} ${info}`;
  }

  function priceTitle(base, price, info) {
    const tagged = fmtPrice(price);
    const text = tagged === "--" ? base : `${base} ${tagged}`;
    return lineTitle(text, info);
  }

  function resolveLivePrice(data) {
    if (!data) return 0;
    const dash = data.dashboard || {};
    const direct = Number(dash.price || 0);
    if (direct > 0) return direct;
    const ice = data.iceberg || {};
    const l1 = ice.l1 || {};
    const bid = Number(l1.bid || 0);
    const ask = Number(l1.ask || 0);
    if (bid > 0 && ask > 0) return (bid + ask) * 0.5;
    return 0;
  }

  function updateExtra(data) {
    if (!data) return;
    ensureCard();
    ensurePerfCard();
    ensureIcebergCard();
    ensureDonateUI();
    ensureChartSection();
    bindIcebergControls();
    bindChartRangeControls();
    bindChartLegendControls();
    const dashKey = data.dash_key || data.symbol;
    if (dashKey) {
      window.__dashKey = dashKey;
    }
    if (data.symbol) {
      window.__dashSymbol = data.symbol;
    }
    if (data.metrics_interval_sec) {
      window.__dashMetricsIntervalSec = Number(data.metrics_interval_sec) || window.__dashMetricsIntervalSec;
    }
    const chartTitle = document.getElementById("chartTitle");
    if (chartTitle && (data.symbol || dashKey)) {
      const base = data.symbol || dashKey;
      const modelLabel = shortModelLabel(data);
      const tags = [];
      if (dashKey && data.symbol && dashKey !== data.symbol) tags.push(dashKey);
      if (modelLabel && modelLabel !== data.symbol && modelLabel !== dashKey) tags.push(modelLabel);
      const tagText = tags.length ? ` (${tags.join(" / ")})` : "";
      chartTitle.textContent = `${base}${tagText} ${DT("dash_chart")}`;
    }
    const chartSub = document.getElementById("chartSub");
    if (chartSub) {
      const modelLabel = shortModelLabel(data);
      const tags = [];
      if (dashKey && data.symbol && dashKey !== data.symbol) tags.push(dashKey);
      if (modelLabel && modelLabel !== data.symbol && modelLabel !== dashKey) tags.push(modelLabel);
      const tag = tags.length ? ` | ${tags.join(" / ")}` : "";
      chartSub.textContent = data.ts ? `${DT("dash_updated")} ${data.ts}${tag}` : DT("dash_waiting");
    }
    const chartStatus = document.getElementById("chartStatus");
    if (chartStatus) chartStatus.textContent = data.health?.status || DT("dash_live");
    const dash = data.dashboard || {};
    const price = dash.price;
    const priceEl = document.getElementById("dashPrice");
    if (priceEl) priceEl.textContent = price ? fmtPrice(price) : "--";
    const modeEl = document.getElementById("dashPosMode");
    if (modeEl) modeEl.textContent = data.position_mode || "--";
    const countEl = document.getElementById("dashOrderCount");
    if (countEl) countEl.textContent = dash.resting_count ?? (dash.resting_orders ? dash.resting_orders.length : "--");
    const listEl = document.getElementById("dashOrderList");
    if (listEl) listEl.textContent = formatOrders(dash.resting_orders);

    const stats = dash.trade_stats || {};
    const winRate = stats.win_rate;
    const winRateText = (winRate === null || winRate === undefined) ? "--" : `${(winRate * 100).toFixed(1)}%`;
    const winRateEl = document.getElementById("dashWinRate");
    if (winRateEl) winRateEl.textContent = winRateText;
    const tradeCountEl = document.getElementById("dashTradeCount");
    if (tradeCountEl) tradeCountEl.textContent = stats.trades ?? "--";
    const sortinoEl = document.getElementById("dashSortino");
    if (sortinoEl) sortinoEl.textContent = (stats.sortino === null || stats.sortino === undefined) ? "--" : fmt(stats.sortino, 2);
    const pfEl = document.getElementById("dashProfitFactor");
    if (pfEl) pfEl.textContent = (stats.profit_factor === null || stats.profit_factor === undefined) ? "--" : fmt(stats.profit_factor, 2);
    const avgTradeEl = document.getElementById("dashAvgTrade");
    if (avgTradeEl) avgTradeEl.textContent = (stats.avg_trade === null || stats.avg_trade === undefined) ? "--" : fmt(stats.avg_trade, 4);
    const maxDdEl = document.getElementById("dashMaxDD");
    if (maxDdEl) maxDdEl.textContent = (stats.max_drawdown_pct === null || stats.max_drawdown_pct === undefined) ? "--" : `${fmt(stats.max_drawdown_pct, 2)}%`;

    const ice = data.iceberg || {};
    const buy = ice.buy || {};
    const sell = ice.sell || {};
    const tp = data.iceberg_tp || {};
    const tpLong = tp.long || {};
    const tpShort = tp.short || {};
    const statusParts = [];
    if (buy.active) statusParts.push(DT("dash_entry_buy"));
    if (sell.active) statusParts.push(DT("dash_entry_sell"));
    if (tpLong.active) {
      statusParts.push(DT("dash_tp_long"));
    } else if (tpLong.paused) {
      statusParts.push(`${DT("dash_tp_long")} (${DT("dash_paused")})`);
    }
    if (tpShort.active) {
      statusParts.push(DT("dash_tp_short"));
    } else if (tpShort.paused) {
      statusParts.push(`${DT("dash_tp_short")} (${DT("dash_paused")})`);
    }
    const statusText = statusParts.length ? statusParts.join(" & ") : DT("dash_idle");
    const statusEl = document.getElementById("dashIcebergStatus");
    if (statusEl) statusEl.textContent = statusText;
    const buyEl = document.getElementById("dashIcebergBuy");
    if (buyEl) buyEl.textContent = formatIcebergSide(buy);
    const sellEl = document.getElementById("dashIcebergSell");
    if (sellEl) sellEl.textContent = formatIcebergSide(sell);
    const tpLongEl = document.getElementById("dashIcebergTpLong");
    if (tpLongEl) tpLongEl.textContent = formatTpIceberg(tpLong);
    const tpShortEl = document.getElementById("dashIcebergTpShort");
    if (tpShortEl) tpShortEl.textContent = formatTpIceberg(tpShort);
    const l1 = ice.l1 || {};
    const l1Text = (l1.bid && l1.ask)
      ? `${fmtPrice(l1.bid)} / ${fmtPrice(l1.ask)} (${fmt(l1.age_sec, 1)}s | ${DT("dash_exchange_short")} ${fmt(l1.exchange_age_sec, 1)}s | ${l1.source || DT("dash_na")})`
      : "--";
    const l1El = document.getElementById("dashIcebergL1");
    if (l1El) l1El.textContent = l1Text;

    loadChartLib(() => {
      initChart();
      const key = window.__dashKey || window.__dashSymbol;
      const keyChanged = key && key !== chartLastKey;
      let restored = false;
      if (keyChanged) {
        if (chartLastKey) {
          stashChartState(chartLastKey);
        }
        chartLastKey = key;
        chartOverlay = { lines: [], markers: [], livePrice: null };
        restored = restoreChartState(key);
        if (!restored) {
          chartFullCandles = [];
          chartNeedsFit = true;
          chartUserZoom = false;
          chartZoomState = null;
          chartRangeSec = 15 * 60;
          chartRangeLabel = "15M";
          chartRangeKey = "15m";
          syncChartRangeButtons();
        }
        if (chart) {
          chart.clear();
          chart.setOption(baseChartOption(), { notMerge: true, lazyUpdate: true });
        }
        if (restored && chartFullCandles.length) {
          setCandlesData(chartFullCandles);
        }
      }
      updateChartLines(data);
      updateChartMarkers(data);
      updateLivePriceLine(resolveLivePrice(data));
      if (!chartTimer) {
        fetchChartHistory(true);
        fetchDirectionHistory(true);
        chartTimer = setInterval(() => {
          fetchChartHistory();
          fetchDirectionHistory();
        }, 10000);
      } else if (keyChanged) {
        if (!restored || !chartFullCandles.length) {
          fetchChartHistory(true);
        }
        fetchDirectionHistory(true);
      }
    });
  }

  window.refreshIcebergChart = function() {
    loadChartLib(() => {
      initChart();
      fetchChartHistory(!chartFullCandles.length);
      fetchDirectionHistory(true);
    });
  };

  function hook() {
    if (typeof updateDetail === "function") {
      const base = updateDetail;
      window.updateDetail = function(data) {
        base(data);
        updateExtra(data);
      };
      return;
    }
    if (typeof updateLatest === "function") {
      const base = updateLatest;
      window.updateLatest = function(data) {
        base(data);
        updateExtra(data);
      };
      return;
    }
    setTimeout(hook, 250);
  }

  hook();
})();
</script>
"""


def inject_dashboard_html(html: str, lang: str = None) -> str:
    if "dashMarketCard" in html and "dashPerfCard" in html and "dashIcebergCard" in html:
        return html
    marker = "</body>"
    if marker in html:
        return html.replace(marker, EXTRA_SCRIPT + "\n" + marker)
    return html + EXTRA_SCRIPT


def inject_translations(html: str) -> str:
    """Inject translation strings into the dashboard HTML."""
    translations = get_dashboard_translations()
    if not translations:
        return html

    # Create JavaScript translation object
    import json
    trans_json = json.dumps(translations, ensure_ascii=False)
    trans_script = f"""<script>
window.DASH_TRANS = {trans_json};
</script>
"""

    # Inject before the first <script> tag or at end of <head>
    if "<script>" in html:
        return html.replace("<script>", trans_script + "<script>", 1)
    elif "</head>" in html:
        return html.replace("</head>", trans_script + "</head>")
    return html


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title id="pageTitle">Live Trading Dashboard</title>
  <style>
    :root {
      --bg1: #0f172a;
      --bg2: #111827;
      --panel: #0b1220;
      --accent: #22c55e;
      --accent2: #f59e0b;
      --danger: #ef4444;
      --muted: #94a3b8;
      --text: #e5e7eb;
      --glow: rgba(34, 197, 94, 0.2);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Space Grotesk", "Bahnschrift", "Trebuchet MS", sans-serif;
      color: var(--text);
      background: radial-gradient(1200px 600px at 10% 10%, #0b1430 0%, transparent 60%),
                  radial-gradient(1200px 600px at 90% 20%, #1b2a3a 0%, transparent 60%),
                  linear-gradient(160deg, var(--bg1), var(--bg2));
      min-height: 100vh;
    }
    header {
      padding: 28px 32px 10px;
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      gap: 16px;
    }
    h1 {
      margin: 0;
      font-size: 28px;
      letter-spacing: 0.5px;
    }
    .sub {
      color: var(--muted);
      font-size: 13px;
      margin-top: 6px;
    }
    .pill {
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.08);
      font-size: 12px;
      color: var(--muted);
    }
    .grid {
      padding: 16px 32px 40px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
    }
    .card {
      background: linear-gradient(180deg, rgba(11,18,32,0.95), rgba(11,18,32,0.85));
      border: 1px solid rgba(148,163,184,0.15);
      border-radius: 16px;
      padding: 16px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.25);
      animation: fadeUp 0.5s ease forwards;
      opacity: 0;
    }
    .card h3 {
      margin: 0 0 10px;
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
    }
    .value {
      font-size: 24px;
      font-weight: 600;
    }
    .row {
      display: flex;
      justify-content: space-between;
      font-size: 13px;
      padding: 4px 0;
      border-bottom: 1px dashed rgba(148,163,184,0.1);
    }
    .row:last-child { border-bottom: none; }
    .ok { color: var(--accent); }
    .warn { color: var(--accent2); }
    .bad { color: var(--danger); }
    .muted { color: var(--muted); }
    canvas {
      width: 100%;
      height: 80px;
      margin-top: 8px;
    }
    @keyframes fadeUp {
      from { transform: translateY(10px); opacity: 0; }
      to { transform: translateY(0); opacity: 1; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1 id="symbol">Live Trading Dashboard</h1>
      <div class="sub" id="subline">Waiting for data...</div>
    </div>
    <div class="pill" id="statusPill">Offline</div>
  </header>
  <section class="grid">
    <div class="card" style="animation-delay: 0.05s">
      <h3>Equity & PnL</h3>
      <div class="value" id="equityVal">--</div>
      <div class="row"><span>Daily PnL</span><span id="dailyPnl">--</span></div>
      <div class="row"><span>Unrealized</span><span id="unrealPnl">--</span></div>
      <canvas id="equityChart"></canvas>
    </div>
    <div class="card" style="animation-delay: 0.08s">
      <h3>Account Balance</h3>
      <div class="value" id="aggBalance">--</div>
      <div class="row"><span>Available</span><span id="aggAvailable">--</span></div>
      <div class="row"><span>Profiles</span><span id="aggProfiles">--</span></div>
      <div class="row"><span>Total Return</span><span id="aggReturn">--</span></div>
      <div class="row"><span>Max DD</span><span id="aggMaxDD">--</span></div>
      <div class="row"><span>Volatility</span><span id="aggVolatility">--</span></div>
      <div class="row"><span>Stability</span><span id="aggStability">--</span></div>
      <div class="row"><span>Smoothness</span><span id="aggSmoothness">--</span></div>
      <div class="row"><span>Trend / Day</span><span id="aggTrend">--</span></div>
      <canvas id="balanceChart"></canvas>
    </div>
    <div class="card" style="animation-delay: 0.1s">
      <h3>Position</h3>
      <div class="value" id="posSide">--</div>
      <div class="row"><span>Size</span><span id="posSize">--</span></div>
      <div class="row"><span>Entry</span><span id="posEntry">--</span></div>
      <div class="row"><span>Mark</span><span id="posMark">--</span></div>
      <div class="row"><span>TP / SL</span><span id="posTPSL">--</span></div>
    </div>
    <div class="card" style="animation-delay: 0.15s">
      <h3>Health</h3>
      <div class="value" id="healthVal">--</div>
      <div class="row"><span>Sentiment</span><span id="sentimentVal">--</span></div>
      <div class="row"><span>Regime</span><span id="regimeVal">--</span></div>
      <div class="row"><span>Trade Enabled</span><span id="tradeEnabled">--</span></div>
      <div class="row"><span>Drift Alerts</span><span id="driftAlerts">--</span></div>
    </div>
    <div class="card" style="animation-delay: 0.2s">
      <h3>Data Quality</h3>
      <div class="value" id="barsVal">--</div>
      <div class="row"><span>Macro %</span><span id="macroPct">--</span></div>
      <div class="row"><span>OB Density</span><span id="obDensity">--</span></div>
      <div class="row"><span>Trade Continuity</span><span id="tradeCont">--</span></div>
      <div class="row"><span>Lag Trade / Bar</span><span id="lagTradeBar">--</span></div>
    </div>
    <div class="card" style="animation-delay: 0.25s">
      <h3>Latency</h3>
      <div class="value" id="restLatency">--</div>
      <div class="row"><span>WS Trade</span><span id="wsTrade">--</span></div>
      <div class="row"><span>WS OB</span><span id="wsOb">--</span></div>
      <div class="row"><span>OB Lag</span><span id="obLag">--</span></div>
      <canvas id="latencyChart"></canvas>
    </div>
    <div class="card" style="animation-delay: 0.3s">
      <h3>Orders</h3>
      <div class="value" id="openOrders">--</div>
      <div class="row"><span>Last Reconcile</span><span id="lastReconcile">--</span></div>
      <div class="row"><span>Source</span><span id="reconcileSource">--</span></div>
      <div class="row"><span>Protective</span><span id="protectiveCount">--</span></div>
    </div>
    <div class="card" style="animation-delay: 0.35s">
      <h3>Errors</h3>
      <div class="value" id="errorCount">--</div>
      <div class="row"><span>Last Runtime</span><span id="lastRuntime">--</span></div>
      <div class="row"><span>Last API</span><span id="lastApi">--</span></div>
    </div>
  </section>
  <script>
    const fmt = (v, d=2) => (v === null || v === undefined) ? "--" : Number(v).toFixed(d);
    const fmtPct = (v) => (v === null || v === undefined) ? "--" : `${Number(v).toFixed(1)}%`;
    const fmtSec = (v) => (v === null || v === undefined) ? "--" : `${Number(v).toFixed(1)}s`;
    const fmtMs = (v) => (v === null || v === undefined) ? "--" : `${Number(v).toFixed(1)}ms`;

    // Translation helper
    const T = (key) => (window.DASH_TRANS && window.DASH_TRANS[key]) || key.replace("dash_", "").replace(/_/g, " ");
    const formatT = (key, vars = {}) => {
      let text = T(key);
      Object.entries(vars).forEach(([k, v]) => {
        text = text.replace(`{${k}}`, v);
      });
      return text;
    };
    function applyTranslations() {
      const trans = window.DASH_TRANS || {};
      if (!Object.keys(trans).length) return;

      const title = document.getElementById("pageTitle");
      if (title) title.textContent = T("dash_title");
      document.title = T("dash_title");
      const headerTitle = document.querySelector("header h1");
      if (headerTitle) headerTitle.textContent = T("dash_title");
      const symbol = document.getElementById("symbol");
      if (symbol && !symbol.dataset.updated) symbol.textContent = T("dash_title");
      const subline = document.getElementById("subline");
      if (subline && !subline.dataset.updated) subline.textContent = T("dash_waiting");
      const pill = document.getElementById("statusPill");
      if (pill && !pill.dataset.updated) pill.textContent = T("dash_offline");
      const accountSub = document.getElementById("accountSub");
      if (accountSub && !accountSub.dataset.updated) accountSub.textContent = T("dash_waiting_balance");
      const balanceNote = document.getElementById("balanceNote");
      if (balanceNote && !balanceNote.dataset.updated) balanceNote.textContent = T("dash_flow_note");
      const detailSub = document.getElementById("detailSub");
      if (detailSub && !detailSub.dataset.updated) detailSub.textContent = T("dash_no_data_loaded");
      const donateClose = document.getElementById("donateClose");
      if (donateClose) donateClose.setAttribute("aria-label", T("dash_close"));

      const textMap = {
        "Account Balance": "dash_account_balance",
        "Traders": "dash_traders",
        "Total Balance": "dash_total_balance",
        "Unified Account": "dash_unified_account",
        "Funding Account": "dash_funding_account",
        "Performance": "dash_performance",
        "Balance Curve (Raw + Flow-Adjusted)": "dash_balance_curve",
        "Curve Diagnostics": "dash_curve_diagnostics",
        "Profiles": "dash_profiles",
        "Equity and PnL": "dash_equity_pnl",
        "Equity & PnL": "dash_equity_pnl",
        "Position": "dash_position",
        "Health": "dash_health",
        "Model": "dash_model",
        "Signal": "dash_signal",
        "Data Quality": "dash_data_quality",
        "Latency": "dash_latency",
        "Orders": "dash_orders",
        "Errors": "dash_errors",
        "Features": "dash_features",
        "Signal Stream": "dash_signal_stream",
        "Available": "dash_available",
        "Share": "dash_share",
        "Updated": "dash_updated",
        "Flow Events": "dash_flow_events",
        "Max DD": "dash_max_dd",
        "Volatility": "dash_volatility",
        "Stability": "dash_stability",
        "Smoothness": "dash_smoothness",
        "Trend / Day": "dash_trend_day",
        "Raw Return": "dash_raw_return",
        "Points": "dash_points",
        "Daily PnL": "dash_daily_pnl",
        "Unrealized": "dash_unrealized",
        "Size": "dash_size",
        "Entry": "dash_entry",
        "Mark": "dash_mark",
        "TP / SL": "dash_tp_sl",
        "Sentiment": "dash_sentiment",
        "Regime": "dash_regime",
        "Trade Enabled": "dash_trade_enabled",
        "Drift Alerts": "dash_drift_alerts",
        "Pred Long": "dash_pred_long",
        "Pred Short": "dash_pred_short",
        "Dir Long": "dash_dir_long",
        "Dir Short": "dash_dir_short",
        "Threshold": "dash_threshold",
        "Pred Bar": "dash_pred_bar",
        "Dir Threshold": "dash_dir_threshold",
        "Aggressive": "dash_aggressive",
        "Dir Bar": "dash_dir_bar",
        "Model Path": "dash_model_path",
        "Keys Profile": "dash_keys_profile",
        "Side": "dash_side",
        "Entry Price": "dash_entry_price",
        "TP Price": "dash_tp_price",
        "SL Price": "dash_sl_price",
        "Time": "dash_time",
        "Macro %": "dash_macro_pct",
        "OB Density": "dash_ob_density",
        "Trade Continuity": "dash_trade_continuity",
        "Lag Trade / Bar": "dash_lag_trade_bar",
        "WS Trade": "dash_ws_trade",
        "WS OB": "dash_ws_ob",
        "OB Lag": "dash_ob_lag",
        "Last Reconcile": "dash_last_reconcile",
        "Source": "dash_source",
        "Protective": "dash_protective",
        "Last Runtime": "dash_last_runtime",
        "Last API": "dash_last_api",
        "Active Signals": "dash_active_signals",
        "Signals": "dash_signals",
        "Direction": "dash_direction",
        "Dir": "dash_direction_short",
        "Agg": "dash_aggressive_short",
        "Levels": "dash_levels",
        "TP/SL": "dash_tp_sl",
        "Sound Off": "dash_sound_off",
        "Sound On": "dash_sound_on",
        "All": "dash_all",
        "Split Scale": "dash_split_scale",
        "Shared Scale": "dash_shared_scale",
        "Raw Curve": "dash_raw_curve",
        "Flow-Adjusted": "dash_flow_adjusted",
        "Raw": "dash_raw",
        "Flow-adjusted": "dash_flow_adjusted",
        "Save": "dash_save",
        "Reset": "dash_reset",
        "Signal Mode": "dash_signal_mode",
        "Automated Mode": "dash_automated_mode",
        "Show All": "dash_show_all",
        "No data loaded": "dash_no_data_loaded",
        "Flow-adjusted removes large deposits/withdrawals.": "dash_flow_note",
        "Donate & Referral link": "dash_donate_referral",
        "Contributions support me directly and help improve this tool.": "dash_donate_note",
        "Bitcoin (BTC)": "dash_bitcoin",
        "Ethereum (ETH)": "dash_ethereum",
        "Bybit referral link": "dash_referral_link",
      };

      const apply = (el) => {
        const key = textMap[el.textContent.trim()];
        if (key) el.textContent = T(key);
      };

      document.querySelectorAll(
        "h2, h3, .row > span:first-child, .nav-btn, .signal-strip__label, .toggle-btn, .signal-btn, .range-btn"
      ).forEach(apply);
    }

    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", applyTranslations);
    } else {
      applyTranslations();
    }

    function sparkline(canvas, values, color) {
      const ctx = canvas.getContext("2d");
      const w = canvas.width = canvas.clientWidth;
      const h = canvas.height = canvas.clientHeight;
      ctx.clearRect(0, 0, w, h);
      if (!values.length) return;
      const min = Math.min(...values);
      const max = Math.max(...values);
      const span = (max - min) || 1;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      values.forEach((v, i) => {
        const x = (i / (values.length - 1)) * w;
        const y = h - ((v - min) / span) * h;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    function updateLatest(data) {
      if (!data || !data.symbol) return;
      document.getElementById("symbol").textContent = `${T("dash_title")} - ${data.symbol}`;
      document.getElementById("symbol").dataset.updated = "1";
      document.getElementById("subline").textContent = `${T("dash_startup")}: ${data.startup_time} - ${T("dash_uptime")}: ${Math.round(data.uptime_sec)}s`;
      document.getElementById("subline").dataset.updated = "1";
      document.getElementById("statusPill").textContent = data.trade_enabled ? T("dash_trading_enabled") : T("dash_trading_paused");
      document.getElementById("statusPill").dataset.updated = "1";
      document.getElementById("statusPill").className = "pill " + (data.trade_enabled ? "ok" : "warn");

      document.getElementById("equityVal").textContent = data.equity ? fmt(data.equity, 4) : "--";
      document.getElementById("dailyPnl").textContent = fmt(data.daily_pnl, 4);
      document.getElementById("unrealPnl").textContent = fmt(data.position.unreal_pnl, 4);

      document.getElementById("posSide").textContent = data.position.side || "--";
      document.getElementById("posSize").textContent = fmt(data.position.size, 4);
      document.getElementById("posEntry").textContent = fmt(data.position.entry_price, 6);
      document.getElementById("posMark").textContent = fmt(data.position.mark_price, 6);
      const tp = fmt(data.position.take_profit, 6);
      const sl = fmt(data.position.stop_loss, 6);
      document.getElementById("posTPSL").textContent = `${tp} / ${sl}`;

      document.getElementById("healthVal").textContent = data.health.status || "--";
      document.getElementById("sentimentVal").textContent = data.health.sentiment || "--";
      document.getElementById("regimeVal").textContent = data.health.regime || "--";
      document.getElementById("tradeEnabled").textContent = data.trade_enabled ? T("dash_yes") : T("dash_no");
      const drift = data.drift.alerts && data.drift.alerts.length ? data.drift.alerts.join(", ") : T("dash_none");
      document.getElementById("driftAlerts").textContent = drift;

      document.getElementById("barsVal").textContent = data.data_health.bars || 0;
      document.getElementById("macroPct").textContent = fmtPct(data.data_health.macro_pct);
      document.getElementById("obDensity").textContent = fmtPct(data.data_health.ob_density_pct);
      document.getElementById("tradeCont").textContent = data.data_health.trade_cont ?? "--";
      document.getElementById("lagTradeBar").textContent = `${fmtSec(data.latency.lag_trade_sec)} / ${fmtSec(data.latency.lag_bar_sec)}`;

      document.getElementById("restLatency").textContent = fmtMs(data.latency.rest_avg_ms);
      document.getElementById("wsTrade").textContent = fmtMs(data.latency.ws_trade_ms);
      document.getElementById("wsOb").textContent = fmtMs(data.latency.ws_ob_ms);
      document.getElementById("obLag").textContent = fmtSec(data.latency.ob_lag_sec);

      document.getElementById("openOrders").textContent = data.orders.open_orders ?? 0;
      const rec = data.orders.last_reconcile || {};
      document.getElementById("lastReconcile").textContent = rec.ts || "--";
      document.getElementById("reconcileSource").textContent = rec.source || "--";
      document.getElementById("protectiveCount").textContent = rec.protective_orders ?? "--";

      const errors = data.errors || {};
      document.getElementById("errorCount").textContent = `${errors.runtime_count ?? 0} / ${errors.api_count ?? 0}`;
      document.getElementById("lastRuntime").textContent = errors.last_runtime_error || "--";
      document.getElementById("lastApi").textContent = errors.last_api_error || "--";
    }

    function updateBalances(data) {
      if (!data) return;
      const stats = data.stats || {};
      document.getElementById("aggBalance").textContent = data.total_equity ? fmt(data.total_equity, 4) : "--";
      document.getElementById("aggAvailable").textContent = data.total_available ? fmt(data.total_available, 4) : "--";
      document.getElementById("aggProfiles").textContent = data.profile_count ?? "--";
      document.getElementById("aggReturn").textContent = stats.total_return_pct === null || stats.total_return_pct === undefined
        ? "--"
        : `${fmt(stats.total_return_pct, 2)}%`;
      document.getElementById("aggMaxDD").textContent = stats.max_drawdown_pct === null || stats.max_drawdown_pct === undefined
        ? "--"
        : `${fmt(stats.max_drawdown_pct, 2)}%`;
      document.getElementById("aggVolatility").textContent = stats.volatility_pct === null || stats.volatility_pct === undefined
        ? "--"
        : `${fmt(stats.volatility_pct, 2)}%`;
      document.getElementById("aggStability").textContent = stats.stability_pct === null || stats.stability_pct === undefined
        ? "--"
        : `${fmt(stats.stability_pct, 1)}%`;
      document.getElementById("aggSmoothness").textContent = stats.smoothness_r2 === null || stats.smoothness_r2 === undefined
        ? "--"
        : fmt(stats.smoothness_r2, 2);
      document.getElementById("aggTrend").textContent = stats.slope_per_day === null || stats.slope_per_day === undefined
        ? "--"
        : fmt(stats.slope_per_day, 4);
    }

    async function fetchBalances() {
      const resp = await fetch("/api/balances");
      if (!resp.ok) return;
      const data = await resp.json();
      updateBalances(data);
    }

    async function fetchBalanceHistory() {
      const resp = await fetch("/api/balance_history?limit=200");
      if (!resp.ok) return;
      const data = await resp.json();
      const balances = data.map(x => Number(x.total_equity || 0));
      sparkline(document.getElementById("balanceChart"), balances, "#38bdf8");
    }

    async function fetchLatest() {
      const resp = await fetch("/api/latest");
      if (!resp.ok) return;
      const data = await resp.json();
      updateLatest(data);
    }

    async function fetchHistory() {
      const resp = await fetch("/api/metrics?limit=200");
      if (!resp.ok) return;
      const data = await resp.json();
      const equity = data.map(x => Number(x.equity || 0));
      const latency = data.map(x => Number(x.latency?.rest_avg_ms || 0));
      sparkline(document.getElementById("equityChart"), equity, "#22c55e");
      sparkline(document.getElementById("latencyChart"), latency, "#f59e0b");
    }

    fetchLatest();
    fetchHistory();
    fetchBalances();
    fetchBalanceHistory();
    setInterval(fetchLatest, 2000);
    setInterval(fetchHistory, 10000);
    setInterval(fetchBalances, 5000);
    setInterval(fetchBalanceHistory, 15000);
  </script>
</body>
</html>
"""

def resource_path(name: str) -> Path:
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", "")
        if base:
            candidate = Path(base) / name
            if candidate.exists():
                return candidate
    return Path(__file__).with_name(name)


HTML_TEMPLATE_PATH = resource_path("live_dashboard.html")
if HTML_TEMPLATE_PATH.exists():
    try:
        HTML_PAGE = HTML_TEMPLATE_PATH.read_text(encoding="utf-8")
    except Exception:
        pass
HTML_PAGE = inject_dashboard_html(HTML_PAGE)

# Cache for translated HTML per language
_HTML_CACHE = {}

def get_translated_html() -> str:
    """Get HTML with translations for current language."""
    lang = get_language()
    if lang not in _HTML_CACHE:
        translations = get_dashboard_translations()
        print(f"[DEBUG] Language: {lang}, translations count: {len(translations)}")
        if translations:
            print(f"[DEBUG] Sample keys: {list(translations.keys())[:5]}")
        translated = inject_translations(HTML_PAGE)
        has_trans = "DASH_TRANS" in translated
        print(f"[DEBUG] DASH_TRANS injected: {has_trans}")
        _HTML_CACHE[lang] = translated
    return _HTML_CACHE[lang]


class MetricsHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    metrics_dir: Path = Path(".")
    symbols_filter = None
    fixed_files = None
    history_limit: int = 200
    balance_monitor = None
    static_dir: Path = resource_path("static")

    def _send(self, status, payload, content_type="application/json"):
        try:
            if isinstance(payload, (dict, list)):
                data = json.dumps(payload, default=str).encode("utf-8")
            elif isinstance(payload, str):
                data = payload.encode("utf-8")
            else:
                data = payload if isinstance(payload, (bytes, bytearray)) else b""
        except Exception:
            data = b""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(data)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _resolve_files(self):
        if self.fixed_files:
            return self.fixed_files
        return discover_metrics_entries(self.metrics_dir, self.symbols_filter)

    def _choose_entry(self, file_map, key: Optional[str] = None, symbol: Optional[str] = None):
        if key and key in file_map:
            return key
        if symbol:
            if symbol in file_map:
                return symbol
            for candidate in sorted(file_map.keys()):
                latest = read_latest(file_map[candidate])
                candidate_symbol = resolve_latest_symbol(latest, candidate)
                if candidate_symbol == symbol:
                    return candidate
        keys = sorted(file_map.keys())
        return keys[0] if keys else None

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/command":
            return self._send(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except Exception:
            length = 0
        body = self.rfile.read(length) if length > 0 else b""
        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except Exception:
            return self._send(400, {"error": "invalid json"})
        file_map = self._resolve_files()
        key = payload.get("key")
        symbol = payload.get("symbol")
        entry_key = self._choose_entry(file_map, key=key, symbol=symbol)
        if not entry_key:
            return self._send(400, {"error": "unknown symbol"})
        latest = read_latest(file_map[entry_key])
        symbol = resolve_latest_symbol(latest, entry_key)
        payload["ts"] = payload.get("ts") or time.time()
        payload["symbol"] = symbol
        cmd_path = self.metrics_dir / f"commands_{symbol}.jsonl"
        append_jsonl(cmd_path, payload)
        return self._send(200, {"ok": True})

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self._send(200, get_translated_html(), "text/html; charset=utf-8")

        if parsed.path.startswith("/static/"):
            rel_path = parsed.path.replace("/static/", "", 1)
            base = self.static_dir.resolve()
            target = (self.static_dir / rel_path).resolve()
            try:
                target.relative_to(base)
            except ValueError:
                return self._send(403, {"error": "forbidden"})
            if not target.exists() or not target.is_file():
                return self._send(404, {"error": "not found"})
            content_type = "application/octet-stream"
            if target.suffix == ".js":
                content_type = "application/javascript"
            elif target.suffix == ".css":
                content_type = "text/css"
            elif target.suffix in {".svg", ".png"}:
                content_type = "image/" + target.suffix.lstrip(".")
            return self._send(200, target.read_bytes(), content_type)

        file_map = self._resolve_files()
        if parsed.path == "/api/symbols":
            return self._send(200, sorted(file_map.keys()))

        if parsed.path == "/api/summary":
            return self._send(200, summary_from_files(file_map))

        if parsed.path == "/api/latest":
            qs = parse_qs(parsed.query)
            key = qs.get("key", [""])[0] or None
            symbol = qs.get("symbol", [""])[0] or None
            entry_key = self._choose_entry(file_map, key=key, symbol=symbol)
            if not entry_key:
                return self._send(200, {})
            path = file_map[entry_key]
            latest = read_latest(path)
            if not latest:
                latest = {"symbol": None, "ts": None}
            latest = apply_dash_meta(latest, entry_key)
            symbol = latest.get("symbol") or entry_key
            latest = augment_latest(latest, symbol, path)
            return self._send(200, latest)

        if parsed.path == "/api/metrics":
            qs = parse_qs(parsed.query)
            key = qs.get("key", [""])[0] or None
            symbol = qs.get("symbol", [""])[0] or None
            entry_key = self._choose_entry(file_map, key=key, symbol=symbol)
            if not entry_key:
                return self._send(200, [])
            limit = int(qs.get("limit", [self.history_limit])[0])
            max_bytes = max(512 * 1024, limit * 1024)
            max_bytes = min(max_bytes, 16 * 1024 * 1024)
            return self._send(200, tail_jsonl(file_map[entry_key], limit=limit, max_bytes=max_bytes))

        if parsed.path == "/api/signals":
            qs = parse_qs(parsed.query)
            key = qs.get("key", [""])[0] or None
            symbol = qs.get("symbol", [""])[0] or None
            entry_key = self._choose_entry(file_map, key=key, symbol=symbol)
            if not entry_key:
                return self._send(200, [])
            limit = int(qs.get("limit", [40])[0])
            limit = max(1, min(200, limit))
            max_bytes = max(256 * 1024, limit * 1024)
            max_bytes = min(max_bytes, 4 * 1024 * 1024)
            latest = read_latest(file_map[entry_key])
            symbol = resolve_latest_symbol(latest, entry_key)
            key_signals = self.metrics_dir / f"signals_{entry_key}.jsonl"
            if key_signals.exists() and key_signals.stat().st_size > 0:
                signals_path = key_signals
            else:
                symbol_signals = self.metrics_dir / f"signals_{symbol}.jsonl"
                if symbol_signals.exists() and symbol_signals.stat().st_size > 0:
                    primary = primary_entry_for_symbol(file_map, symbol)
                    if entry_key == symbol or count_entries_for_symbol(file_map, symbol) <= 1 or entry_key == primary:
                        signals_path = symbol_signals
                    else:
                        return self._send(200, [])
                else:
                    return self._send(200, [])
            signals = tail_jsonl(signals_path, limit=limit, max_bytes=max_bytes)
            if signals:
                has_keys = any(s.get("entry_key") for s in signals if isinstance(s, dict))
                if has_keys:
                    signals = [
                        s for s in signals
                        if isinstance(s, dict) and (s.get("entry_key") == entry_key)
                    ]
            return self._send(200, signals)

        if parsed.path == "/api/candles":
            qs = parse_qs(parsed.query)
            key = qs.get("key", [""])[0] or None
            symbol = qs.get("symbol", [""])[0] or None
            entry_key = self._choose_entry(file_map, key=key, symbol=symbol)
            if not entry_key:
                return self._send(200, {"candles": [], "interval_sec": None})
            limit = int(qs.get("limit", [2000])[0])
            limit = max(100, min(20000, limit))
            max_bytes = max(512 * 1024, limit * 256)
            max_bytes = min(max_bytes, 16 * 1024 * 1024)
            latest = read_latest(file_map[entry_key])
            symbol = resolve_latest_symbol(latest, entry_key)
            key_candles = self.metrics_dir / f"data_history_{entry_key}.csv"
            symbol_candles = self.metrics_dir / f"data_history_{symbol}.csv"
            key_exists = key_candles.exists() and key_candles.stat().st_size > 0
            symbol_exists = symbol_candles.exists() and symbol_candles.stat().st_size > 0
            if key_exists and symbol_exists:
                key_last = latest_bar_time_from_csv(key_candles)
                symbol_last = latest_bar_time_from_csv(symbol_candles)
                if key_last and symbol_last:
                    candles_path = symbol_candles if symbol_last >= key_last else key_candles
                elif symbol_last:
                    candles_path = symbol_candles
                elif key_last:
                    candles_path = key_candles
                else:
                    try:
                        key_mtime = key_candles.stat().st_mtime
                        symbol_mtime = symbol_candles.stat().st_mtime
                    except Exception:
                        key_mtime = 0.0
                        symbol_mtime = 0.0
                    candles_path = symbol_candles if symbol_mtime >= key_mtime else key_candles
            elif key_exists:
                candles_path = key_candles
            else:
                candles_path = symbol_candles
            rows = tail_csv_dicts(candles_path, limit=limit, max_bytes=max_bytes)
            candles = []
            for row in rows:
                bar_time = safe_int(row.get("bar_time"), 0)
                if bar_time <= 0:
                    continue
                o = safe_float(row.get("open"))
                h = safe_float(row.get("high"))
                l = safe_float(row.get("low"))
                c = safe_float(row.get("close"))
                if o <= 0 or h <= 0 or l <= 0 or c <= 0:
                    continue
                v = safe_float(row.get("volume"))
                candles.append({
                    "time": bar_time,
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": v,
                })
            interval_sec = None
            if len(candles) >= 2:
                diffs = []
                for idx in range(max(1, len(candles) - 5), len(candles)):
                    diff = candles[idx]["time"] - candles[idx - 1]["time"]
                    if diff > 0:
                        diffs.append(diff)
                if diffs:
                    interval_sec = int(sorted(diffs)[len(diffs) // 2])
            return self._send(200, {"candles": candles, "interval_sec": interval_sec})

        if parsed.path == "/api/balances":
            if not self.balance_monitor:
                return self._send(200, {})
            return self._send(200, self.balance_monitor.latest or {})

        if parsed.path == "/api/balance_history":
            if not self.balance_monitor:
                return self._send(200, [])
            qs = parse_qs(parsed.query)
            limit = int(qs.get("limit", [self.history_limit])[0])
            return self._send(200, self.balance_monitor.get_history(limit))

        return self._send(404, {"error": "not found"})


class DiscordNotifier:
    def __init__(
        self,
        webhook_url: str,
        metrics_dir: Path,
        poll_sec: float = 5.0,
        offline_sec: float = 120.0,
        cooldown_sec: float = 60.0,
        drawdown_pct: float = 10.0,
        fixed_files=None,
        symbols_filter=None,
    ):
        self.webhook_url = webhook_url
        self.metrics_dir = metrics_dir
        self.poll_sec = poll_sec
        self.offline_sec = offline_sec
        self.cooldown_sec = cooldown_sec
        self.drawdown_pct = drawdown_pct
        self.fixed_files = fixed_files
        self.symbols_filter = symbols_filter
        self.last_state = {}
        self.last_sent = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        if self.webhook_url:
            self._thread.start()

    def stop(self):
        self._stop.set()

    def _send(self, content: str) -> None:
        if not self.webhook_url:
            return
        payload = json.dumps({"content": content}).encode("utf-8")
        req = Request(
            self.webhook_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "live-dashboard/1.0",
            },
        )
        try:
            with urlopen(req, timeout=10) as resp:
                resp.read()
        except HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            print(f"Discord notify failed: HTTP {exc.code} {exc.reason} {detail}")
        except URLError as exc:
            print(f"Discord notify failed: {exc}")
        except Exception as exc:
            print(f"Discord notify error: {exc}")

    def _cooldown_ok(self, entry_key: str, event: str) -> bool:
        key = f"{entry_key}:{event}"
        now = time.time()
        last = self.last_sent.get(key, 0.0)
        if now - last < self.cooldown_sec:
            return False
        self.last_sent[key] = now
        return True

    def _notify(self, entry_key: str, event: str, message: str) -> None:
        if not self._cooldown_ok(entry_key, event):
            return
        self._send(message)

    def _is_order_error(self, text: str) -> bool:
        if not text:
            return False
        lower = text.lower()
        keywords = [
            "place",
            "create",
            "reject",
            "insufficient",
            "order",
            "cancel",
            "close",
            "set_trading_stop",
            "stoploss",
            "takeprofit",
            "position",
        ]
        return any(k in lower for k in keywords)

    def _check_entry(self, entry_key: str, latest: dict, metrics_path: Path) -> None:
        state = self.last_state.setdefault(entry_key, {})
        now = time.time()
        ts = parse_ts(latest.get("ts")) if latest else None
        online = ts is not None and (now - ts) <= self.offline_sec
        display, symbol = format_model_display(latest or {}, entry_key, metrics_path)
        if state.get("online", True) and not online:
            age = now - ts if ts else None
            age_text = f"{age:.1f}s" if age is not None else "unknown"
            self._notify(entry_key, "offline", f"{display} offline (last update {age_text} ago)")
        state["online"] = online

        if not latest or not online:
            return

        startup = latest.get("startup_time")
        if startup and startup != state.get("startup_time"):
            self._notify(entry_key, "started", f"{display} started (startup {startup})")
            state["startup_time"] = startup

        errors = latest.get("errors") or {}
        runtime_count = int(errors.get("runtime_count") or 0)
        api_count = int(errors.get("api_count") or 0)
        if runtime_count > state.get("runtime_count", 0) or api_count > state.get("api_count", 0):
            err_text = errors.get("last_runtime_error") or errors.get("last_api_error") or "unknown error"
            self._notify(entry_key, "error", f"{display} error: {err_text}")
            if self._is_order_error(err_text):
                self._notify(entry_key, "order_error", f"{display} order error: {err_text}")
        state["runtime_count"] = runtime_count
        state["api_count"] = api_count

        equity = safe_float(latest.get("equity"), 0.0)
        daily_pnl = safe_float(latest.get("daily_pnl"), 0.0)
        exch_daily = exchange_daily_pnl(symbol)
        if exch_daily is not None:
            daily_pnl = exch_daily
        drawdown = equity > 0 and daily_pnl <= -(self.drawdown_pct / 100.0) * equity
        if drawdown and not state.get("drawdown_alerted"):
            pct = (daily_pnl / equity) * 100.0 if equity else 0.0
            self._notify(
                entry_key,
                "drawdown",
                f"{display} drawdown alert: {daily_pnl:.2f} ({pct:.1f}%)",
            )
            state["drawdown_alerted"] = True
        elif not drawdown:
            state["drawdown_alerted"] = False

    def _run(self) -> None:
        while not self._stop.is_set():
            if self.fixed_files:
                file_map = self.fixed_files
            else:
                file_map = discover_metrics_entries(self.metrics_dir, self.symbols_filter)
            for entry_key, path in file_map.items():
                latest = read_latest(path)
                if latest and not latest.get("symbol"):
                    latest["symbol"] = resolve_latest_symbol(latest, entry_key)
                self._check_entry(entry_key, latest or {}, path)
            time.sleep(self.poll_sec)


class SignalNotifier:
    def __init__(
        self,
        webhook_url: str,
        telegram_token: str,
        telegram_chat_id: str,
        metrics_dir: Path,
        poll_sec: float = 5.0,
        min_pred: float = 0.9,
        fixed_files=None,
        symbols_filter=None,
        max_seen: int = 200,
        include_aggressive: bool = True,
    ):
        self.webhook_url = webhook_url
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.metrics_dir = metrics_dir
        self.poll_sec = poll_sec
        self.min_pred = max(0.0, float(min_pred))
        self.fixed_files = fixed_files
        self.symbols_filter = symbols_filter
        self.max_seen = max(20, int(max_seen))
        self.include_aggressive = include_aggressive
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._bootstrapped = False
        self._seen = {}
        self._seen_set = {}
        self._seen_aggressive = {}

    def start(self):
        if self.webhook_url or (self.telegram_token and self.telegram_chat_id):
            self._thread.start()

    def stop(self):
        self._stop.set()

    def _send_discord(self, content: str) -> None:
        if not self.webhook_url:
            return
        payload = json.dumps({"content": content}).encode("utf-8")
        req = Request(
            self.webhook_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "live-dashboard/1.0",
            },
        )
        try:
            with urlopen(req, timeout=10) as resp:
                resp.read()
        except HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            print(f"Signal notify (discord) failed: HTTP {exc.code} {exc.reason} {detail}")
        except URLError as exc:
            print(f"Signal notify (discord) failed: {exc}")
        except Exception as exc:
            print(f"Signal notify (discord) error: {exc}")

    def _send_telegram(self, content: str) -> None:
        if not self.telegram_token or not self.telegram_chat_id:
            return
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        payload = json.dumps({"chat_id": self.telegram_chat_id, "text": content}).encode("utf-8")
        req = Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "live-dashboard/1.0",
            },
        )
        try:
            with urlopen(req, timeout=10) as resp:
                resp.read()
        except HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            print(f"Signal notify (telegram) failed: HTTP {exc.code} {exc.reason} {detail}")
        except URLError as exc:
            print(f"Signal notify (telegram) failed: {exc}")
        except Exception as exc:
            print(f"Signal notify (telegram) error: {exc}")

    def _send(self, content: str) -> None:
        self._send_discord(content)
        self._send_telegram(content)

    def _tick_decimals(self, tick: float) -> int:
        if tick <= 0:
            return 6
        text = f"{tick}".lower()
        if "e-" in text:
            parts = text.split("e-")
            if len(parts) == 2:
                try:
                    return int(parts[1])
                except Exception:
                    return 6
        if "." in text:
            return len(text.split(".", 1)[1])
        return 0

    def _fmt_price(self, value, tick) -> str:
        val = safe_float(value, 0.0)
        if val <= 0:
            return "n/a"
        decimals = self._tick_decimals(safe_float(tick, 0.0))
        return f"{val:.{decimals}f}"

    def _signal_key(self, signal: dict) -> Optional[str]:
        key = signal.get("id")
        if key:
            return str(key)
        bar_time = signal.get("bar_time")
        side = signal.get("side")
        if bar_time and side:
            return f"{bar_time}-{side}"
        return None

    def _remember_signal(self, entry_key: str, signal_id: str) -> None:
        history = self._seen.setdefault(entry_key, deque(maxlen=self.max_seen))
        seen = self._seen_set.setdefault(entry_key, set())
        if signal_id in seen:
            return
        if len(history) >= self.max_seen:
            oldest = history.popleft()
            if oldest in seen:
                seen.remove(oldest)
        history.append(signal_id)
        seen.add(signal_id)

    def _seen_signal(self, entry_key: str, signal_id: str) -> bool:
        return signal_id in self._seen_set.get(entry_key, set())

    def _remember_aggressive(self, entry_key: str, side: str, bar_time: int) -> None:
        if entry_key not in self._seen_aggressive:
            self._seen_aggressive[entry_key] = {}
        self._seen_aggressive[entry_key][side] = bar_time

    def _seen_aggressive_side(self, entry_key: str, side: str) -> Optional[int]:
        return self._seen_aggressive.get(entry_key, {}).get(side)

    def _has_threshold(self, latest: dict, meta: dict, key: str, param_key: Optional[str] = None) -> bool:
        direction = latest.get("direction")
        if isinstance(direction, dict) and key in direction and direction.get(key) is not None:
            return True
        params = meta.get("params") if isinstance(meta, dict) else None
        lookup = param_key or key
        if isinstance(params, dict) and lookup in params and params.get(lookup) is not None:
            return True
        return False

    def _format_signal_message(self, display: str, signal: dict, threshold_override: Optional[float] = None) -> str:
        side_raw = str(signal.get("side") or "").lower()
        side = "LONG" if side_raw == "buy" else "SHORT" if side_raw == "sell" else side_raw.upper() or "SIGNAL"
        pred = safe_float(signal.get("pred"), 0.0)
        threshold = safe_float(signal.get("threshold"), self.min_pred)
        if threshold_override is not None:
            threshold = threshold_override
        tick = safe_float(signal.get("tick_size"), 0.0)
        entry = self._fmt_price(signal.get("entry_price") or signal.get("target"), tick)
        tp = self._fmt_price(signal.get("tp"), tick)
        sl = self._fmt_price(signal.get("sl"), tick)
        ts = signal.get("ts") or ""
        timeframe = signal.get("timeframe") or ""
        mode = signal.get("mode") or ""
        lines = [
            f"{display} {side} signal {pred:.3f} (>= {threshold:.2f})",
            f"Entry {entry} | TP {tp} | SL {sl}",
        ]
        if ts:
            lines.append(f"Time: {ts}")
        if timeframe:
            lines.append(f"TF: {timeframe}")
        if mode:
            lines.append(f"Mode: {mode}")
        return "\n".join(lines)

    def _format_aggressive_message(
        self,
        display: str,
        side: str,
        pred: float,
        aggressive: float,
        threshold: Optional[float],
        bar_time: Optional[int],
        ts: Optional[str],
    ) -> str:
        side_label = "LONG" if side == "long" else "SHORT"
        lines = [f"{display} DIR {side_label} {pred:.3f} (>= {aggressive:.2f})"]
        if threshold is not None:
            lines.append(f"Dir Threshold {threshold:.2f} | Bar {bar_time or 'n/a'}")
        else:
            lines.append(f"Bar {bar_time or 'n/a'}")
        if ts:
            lines.append(f"Time: {ts}")
        return "\n".join(lines)

    def _direction_bar_time(self, direction: dict, latest: dict) -> Optional[int]:
        bar_time = normalize_bar_time(direction.get("bar_time")) if isinstance(direction, dict) else None
        if bar_time is None:
            bar_time = normalize_bar_time((latest.get("market") or {}).get("last_bar_time"))
        if bar_time is None:
            bar_time = normalize_bar_time(latest.get("bar_time"))
        if bar_time is None:
            ts = parse_ts(latest.get("ts")) if latest else None
            if ts:
                bar_time = int(ts)
        return bar_time

    def _prime_seen(self, entry_key: str, signals_path: Path) -> None:
        signals = tail_jsonl(signals_path, limit=self.max_seen, max_bytes=256 * 1024)
        for signal in signals:
            signal_id = self._signal_key(signal)
            if signal_id:
                self._remember_signal(entry_key, signal_id)

    def _run(self) -> None:
        while not self._stop.is_set():
            if self.fixed_files:
                file_map = self.fixed_files
            else:
                file_map = discover_metrics_entries(self.metrics_dir, self.symbols_filter)
            entries = list(file_map.items())
            if not self._bootstrapped:
                for entry_key, path in entries:
                    latest = read_latest(path)
                    if not latest:
                        continue
                    latest = apply_dash_meta(latest, entry_key)
                    latest, meta = apply_model_meta(latest, path)
                    symbol = resolve_latest_symbol(latest, entry_key)
                    key_signals = self.metrics_dir / f"signals_{entry_key}.jsonl"
                    if key_signals.exists() and key_signals.stat().st_size > 0:
                        signals_path = key_signals
                    else:
                        symbol_signals = self.metrics_dir / f"signals_{symbol}.jsonl"
                        if symbol_signals.exists() and symbol_signals.stat().st_size > 0:
                            primary = primary_entry_for_symbol(file_map, symbol)
                            if entry_key == symbol or count_entries_for_symbol(file_map, symbol) <= 1 or entry_key == primary:
                                signals_path = symbol_signals
                            else:
                                continue
                        else:
                            continue
                    self._prime_seen(entry_key, signals_path)
                    if self.include_aggressive:
                        has_aggressive = self._has_threshold(
                            latest, meta, "aggressive_threshold", "aggressive_threshold"
                        )
                        if has_aggressive:
                            direction = resolve_direction_info(latest, meta)
                            aggressive = maybe_float(direction.get("aggressive_threshold")) if direction else None
                            source = direction.get("source") if direction else None
                            bar_time = self._direction_bar_time(direction or {}, latest)
                            if aggressive is not None and source != "pass" and bar_time:
                                pred_long = maybe_float(direction.get("pred_long")) if direction else None
                                pred_short = maybe_float(direction.get("pred_short")) if direction else None
                                if pred_long is not None and pred_long >= aggressive:
                                    self._remember_aggressive(entry_key, "long", bar_time)
                                if pred_short is not None and pred_short >= aggressive:
                                    self._remember_aggressive(entry_key, "short", bar_time)
                self._bootstrapped = True
                time.sleep(self.poll_sec)
                continue
            for entry_key, path in entries:
                latest = read_latest(path)
                if not latest:
                    continue
                latest = apply_dash_meta(latest, entry_key)
                latest, meta = apply_model_meta(latest, path)
                display, symbol = format_model_display(latest, entry_key, path)
                has_direction = self._has_threshold(latest, meta, "threshold", "direction_threshold")
                has_aggressive = self._has_threshold(
                    latest, meta, "aggressive_threshold", "aggressive_threshold"
                )
                is_new = has_direction and has_aggressive
                direction = resolve_direction_info(latest, meta) if (has_direction or has_aggressive) else None
                if self.include_aggressive and has_aggressive and direction:
                    aggressive = maybe_float(direction.get("aggressive_threshold"))
                    source = direction.get("source")
                    bar_time = self._direction_bar_time(direction, latest)
                    if aggressive is not None and source != "pass" and bar_time:
                        pred_long = maybe_float(direction.get("pred_long"))
                        if pred_long is not None and pred_long >= aggressive:
                            last_seen = self._seen_aggressive_side(entry_key, "long")
                            if last_seen is None or bar_time > last_seen:
                                message = self._format_aggressive_message(
                                    display,
                                    "long",
                                    pred_long,
                                    aggressive,
                                    maybe_float(direction.get("threshold")),
                                    bar_time,
                                    latest.get("ts"),
                                )
                                self._send(message)
                                self._remember_aggressive(entry_key, "long", bar_time)
                        pred_short = maybe_float(direction.get("pred_short"))
                        if pred_short is not None and pred_short >= aggressive:
                            last_seen = self._seen_aggressive_side(entry_key, "short")
                            if last_seen is None or bar_time > last_seen:
                                message = self._format_aggressive_message(
                                    display,
                                    "short",
                                    pred_short,
                                    aggressive,
                                    maybe_float(direction.get("threshold")),
                                    bar_time,
                                    latest.get("ts"),
                                )
                                self._send(message)
                                self._remember_aggressive(entry_key, "short", bar_time)
                key_signals = self.metrics_dir / f"signals_{entry_key}.jsonl"
                if key_signals.exists() and key_signals.stat().st_size > 0:
                    signals_path = key_signals
                else:
                    symbol_signals = self.metrics_dir / f"signals_{symbol}.jsonl"
                    if symbol_signals.exists() and symbol_signals.stat().st_size > 0:
                        primary = primary_entry_for_symbol(file_map, symbol)
                        if entry_key == symbol or count_entries_for_symbol(file_map, symbol) <= 1 or entry_key == primary:
                            signals_path = symbol_signals
                        else:
                            continue
                    else:
                        continue
                signals = tail_jsonl(signals_path, limit=40, max_bytes=256 * 1024)
                if not signals:
                    continue
                has_keys = any(s.get("entry_key") for s in signals if isinstance(s, dict))
                if has_keys:
                    signals = [
                        s for s in signals
                        if isinstance(s, dict) and (s.get("entry_key") == entry_key)
                    ]
                    if not signals:
                        continue
                sorted_signals = sorted(
                    signals,
                    key=lambda s: safe_int(s.get("ts_ms") or 0) or safe_int(s.get("bar_time") or 0),
                )
                for signal in sorted_signals:
                    pred = safe_float(signal.get("pred"), 0.0)
                    if is_new:
                        threshold = maybe_float(signal.get("threshold"))
                        min_pred = threshold if threshold is not None else self.min_pred
                    else:
                        min_pred = self.min_pred
                    if pred < min_pred:
                        continue
                    signal_id = self._signal_key(signal)
                    if not signal_id or self._seen_signal(entry_key, signal_id):
                        continue
                    message = self._format_signal_message(display, signal, min_pred)
                    self._send(message)
                    self._remember_signal(entry_key, signal_id)
            time.sleep(self.poll_sec)


class PredictionNotifier:
    def __init__(
        self,
        webhook_url: str,
        telegram_token: str,
        telegram_chat_id: str,
        metrics_dir: Path,
        poll_sec: float = 5.0,
        min_pred: float = 0.9,
        fixed_files=None,
        symbols_filter=None,
    ):
        self.webhook_url = webhook_url
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.metrics_dir = metrics_dir
        self.poll_sec = poll_sec
        self.min_pred = max(0.0, float(min_pred))
        self.fixed_files = fixed_files
        self.symbols_filter = symbols_filter
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._bootstrapped = False
        self._seen = {}

    def start(self):
        if self.webhook_url or (self.telegram_token and self.telegram_chat_id):
            self._thread.start()

    def stop(self):
        self._stop.set()

    def _send_discord(self, content: str) -> None:
        if not self.webhook_url:
            return
        payload = json.dumps({"content": content}).encode("utf-8")
        req = Request(
            self.webhook_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "live-dashboard/1.0",
            },
        )
        try:
            with urlopen(req, timeout=10) as resp:
                resp.read()
        except HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            print(f"Prediction notify (discord) failed: HTTP {exc.code} {exc.reason} {detail}")
        except URLError as exc:
            print(f"Prediction notify (discord) failed: {exc}")
        except Exception as exc:
            print(f"Prediction notify (discord) error: {exc}")

    def _send_telegram(self, content: str) -> None:
        if not self.telegram_token or not self.telegram_chat_id:
            return
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        payload = json.dumps({"chat_id": self.telegram_chat_id, "text": content}).encode("utf-8")
        req = Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "live-dashboard/1.0",
            },
        )
        try:
            with urlopen(req, timeout=10) as resp:
                resp.read()
        except HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            print(f"Prediction notify (telegram) failed: HTTP {exc.code} {exc.reason} {detail}")
        except URLError as exc:
            print(f"Prediction notify (telegram) failed: {exc}")
        except Exception as exc:
            print(f"Prediction notify (telegram) error: {exc}")

    def _send(self, content: str) -> None:
        self._send_discord(content)
        self._send_telegram(content)

    def _seen_key(self, entry_key: str, side: str) -> Optional[int]:
        return self._seen.get(entry_key, {}).get(side)

    def _remember(self, entry_key: str, side: str, bar_time: int) -> None:
        if entry_key not in self._seen:
            self._seen[entry_key] = {}
        self._seen[entry_key][side] = bar_time

    def _prediction_bar_time(self, prediction: dict, latest: dict) -> Optional[int]:
        bar_time = normalize_bar_time(prediction.get("bar_time"))
        if bar_time is None:
            bar_time = normalize_bar_time((latest.get("market") or {}).get("last_bar_time"))
        if bar_time is None:
            bar_time = normalize_bar_time(latest.get("bar_time"))
        if bar_time is None:
            ts = parse_ts(latest.get("ts")) if latest else None
            if ts:
                bar_time = int(ts)
        return bar_time

    def _format_prediction_message(
        self,
        display: str,
        side: str,
        pred: float,
        threshold: float,
        bar_time: Optional[int],
        ts: Optional[str],
    ) -> str:
        side_label = "LONG" if side == "long" else "SHORT"
        lines = [f"{display} PRED {side_label} {pred:.3f} (>= {threshold:.2f})"]
        lines.append(f"Bar {bar_time or 'n/a'}")
        if ts:
            lines.append(f"Time: {ts}")
        return "\n".join(lines)

    def _run(self) -> None:
        while not self._stop.is_set():
            if self.fixed_files:
                file_map = self.fixed_files
            else:
                file_map = discover_metrics_entries(self.metrics_dir, self.symbols_filter)
            entries = list(file_map.items())
            if not self._bootstrapped:
                for entry_key, path in entries:
                    latest = read_latest(path)
                    if not latest:
                        continue
                    latest = apply_dash_meta(latest, entry_key)
                    prediction = latest.get("prediction") or (latest.get("health") or {}).get("last_prediction") or {}
                    bar_time = self._prediction_bar_time(prediction, latest)
                    pred_long = maybe_float(prediction.get("pred_long"))
                    pred_short = maybe_float(prediction.get("pred_short"))
                    if pred_long is not None and pred_long >= self.min_pred and bar_time:
                        self._remember(entry_key, "long", bar_time)
                    if pred_short is not None and pred_short >= self.min_pred and bar_time:
                        self._remember(entry_key, "short", bar_time)
                self._bootstrapped = True
                time.sleep(self.poll_sec)
                continue

            for entry_key, path in entries:
                latest = read_latest(path)
                if not latest:
                    continue
                latest = apply_dash_meta(latest, entry_key)
                display, _symbol = format_model_display(latest, entry_key, path)
                prediction = latest.get("prediction") or (latest.get("health") or {}).get("last_prediction") or {}
                bar_time = self._prediction_bar_time(prediction, latest)
                if not bar_time:
                    continue
                ts_text = latest.get("ts")

                pred_long = maybe_float(prediction.get("pred_long"))
                if pred_long is not None and pred_long >= self.min_pred:
                    last_seen = self._seen_key(entry_key, "long")
                    if last_seen is None or bar_time > last_seen:
                        message = self._format_prediction_message(
                            display,
                            "long",
                            pred_long,
                            self.min_pred,
                            bar_time,
                            ts_text,
                        )
                        self._send(message)
                        self._remember(entry_key, "long", bar_time)

                pred_short = maybe_float(prediction.get("pred_short"))
                if pred_short is not None and pred_short >= self.min_pred:
                    last_seen = self._seen_key(entry_key, "short")
                    if last_seen is None or bar_time > last_seen:
                        message = self._format_prediction_message(
                            display,
                            "short",
                            pred_short,
                            self.min_pred,
                            bar_time,
                            ts_text,
                        )
                        self._send(message)
                        self._remember(entry_key, "short", bar_time)

            time.sleep(self.poll_sec)


class DirectionNotifier:
    def __init__(
        self,
        webhook_url: str,
        telegram_token: str,
        telegram_chat_id: str,
        metrics_dir: Path,
        poll_sec: float = 5.0,
        fixed_files=None,
        symbols_filter=None,
    ):
        self.webhook_url = webhook_url
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.metrics_dir = metrics_dir
        self.poll_sec = poll_sec
        self.fixed_files = fixed_files
        self.symbols_filter = symbols_filter
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._bootstrapped = False
        self._seen = {}

    def start(self):
        if self.webhook_url or (self.telegram_token and self.telegram_chat_id):
            self._thread.start()

    def stop(self):
        self._stop.set()

    def _send_discord(self, content: str) -> None:
        if not self.webhook_url:
            return
        payload = json.dumps({"content": content}).encode("utf-8")
        req = Request(
            self.webhook_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "live-dashboard/1.0",
            },
        )
        try:
            with urlopen(req, timeout=10) as resp:
                resp.read()
        except HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            print(f"Direction notify (discord) failed: HTTP {exc.code} {exc.reason} {detail}")
        except URLError as exc:
            print(f"Direction notify (discord) failed: {exc}")
        except Exception as exc:
            print(f"Direction notify (discord) error: {exc}")

    def _send_telegram(self, content: str) -> None:
        if not self.telegram_token or not self.telegram_chat_id:
            return
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        payload = json.dumps({"chat_id": self.telegram_chat_id, "text": content}).encode("utf-8")
        req = Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "live-dashboard/1.0",
            },
        )
        try:
            with urlopen(req, timeout=10) as resp:
                resp.read()
        except HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            print(f"Direction notify (telegram) failed: HTTP {exc.code} {exc.reason} {detail}")
        except URLError as exc:
            print(f"Direction notify (telegram) failed: {exc}")
        except Exception as exc:
            print(f"Direction notify (telegram) error: {exc}")

    def _send(self, content: str) -> None:
        self._send_discord(content)
        self._send_telegram(content)

    def _seen_key(self, entry_key: str, side: str) -> Optional[int]:
        return self._seen.get(entry_key, {}).get(side)

    def _remember(self, entry_key: str, side: str, bar_time: int) -> None:
        if entry_key not in self._seen:
            self._seen[entry_key] = {}
        self._seen[entry_key][side] = bar_time

    def _format_direction_message(
        self,
        display: str,
        side: str,
        pred: float,
        aggressive: float,
        threshold: Optional[float],
        bar_time: Optional[int],
        ts: Optional[str],
    ) -> str:
        side_label = "LONG" if side == "long" else "SHORT"
        lines = [f"{display} DIR {side_label} {pred:.3f} (>= {aggressive:.2f})"]
        if threshold is not None:
            lines.append(f"Dir Threshold {threshold:.2f} | Bar {bar_time or 'n/a'}")
        else:
            lines.append(f"Bar {bar_time or 'n/a'}")
        if ts:
            lines.append(f"Time: {ts}")
        return "\n".join(lines)

    def _direction_bar_time(self, direction: dict, latest: dict) -> Optional[int]:
        bar_time = normalize_bar_time(direction.get("bar_time"))
        if bar_time is None:
            ts = parse_ts(latest.get("ts")) if latest else None
            if ts:
                bar_time = int(ts)
        return bar_time

    def _run(self) -> None:
        while not self._stop.is_set():
            if self.fixed_files:
                file_map = self.fixed_files
            else:
                file_map = discover_metrics_entries(self.metrics_dir, self.symbols_filter)
            entries = list(file_map.items())
            if not self._bootstrapped:
                for entry_key, path in entries:
                    latest = read_latest(path)
                    if not latest:
                        continue
                    latest = apply_dash_meta(latest, entry_key)
                    latest, meta = apply_model_meta(latest, path)
                    direction = resolve_direction_info(latest, meta)
                    if not direction:
                        continue
                    bar_time = self._direction_bar_time(direction, latest)
                    aggressive = maybe_float(direction.get("aggressive_threshold"))
                    if aggressive is None:
                        continue
                    threshold = maybe_float(direction.get("threshold"))
                    pred_long = maybe_float(direction.get("pred_long"))
                    pred_short = maybe_float(direction.get("pred_short"))
                    if pred_long is not None and pred_long >= aggressive and bar_time:
                        self._remember(entry_key, "long", bar_time)
                    if pred_short is not None and pred_short >= aggressive and bar_time:
                        self._remember(entry_key, "short", bar_time)
                self._bootstrapped = True
                time.sleep(self.poll_sec)
                continue

            for entry_key, path in entries:
                latest = read_latest(path)
                if not latest:
                    continue
                latest = apply_dash_meta(latest, entry_key)
                latest, meta = apply_model_meta(latest, path)
                direction = resolve_direction_info(latest, meta)
                if not direction:
                    continue
                bar_time = self._direction_bar_time(direction, latest)
                aggressive = maybe_float(direction.get("aggressive_threshold"))
                if aggressive is None:
                    continue
                threshold = maybe_float(direction.get("threshold"))
                display, _symbol = format_model_display(latest, entry_key, path)
                ts_text = latest.get("ts")

                pred_long = maybe_float(direction.get("pred_long"))
                if pred_long is not None and pred_long >= aggressive and bar_time:
                    last_seen = self._seen_key(entry_key, "long")
                    if last_seen is None or bar_time > last_seen:
                        message = self._format_direction_message(
                            display,
                            "long",
                            pred_long,
                            aggressive,
                            threshold,
                            bar_time,
                            ts_text,
                        )
                        self._send(message)
                        self._remember(entry_key, "long", bar_time)

                pred_short = maybe_float(direction.get("pred_short"))
                if pred_short is not None and pred_short >= aggressive and bar_time:
                    last_seen = self._seen_key(entry_key, "short")
                    if last_seen is None or bar_time > last_seen:
                        message = self._format_direction_message(
                            display,
                            "short",
                            pred_short,
                            aggressive,
                            threshold,
                            bar_time,
                            ts_text,
                        )
                        self._send(message)
                        self._remember(entry_key, "short", bar_time)

            time.sleep(self.poll_sec)


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Live Trading Dashboard")
    parser.add_argument("--metrics-dir", type=str, default=".")
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument("--symbol", type=str, default="")
    parser.add_argument("--metrics-file", type=str, default="")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--history-limit", type=int, default=200)
    parser.add_argument("--keys-file", type=str, default="key_profiles.json")
    parser.add_argument("--balance-poll-sec", type=float, default=30.0)
    parser.add_argument("--balance-history-limit", type=int, default=10000)
    parser.add_argument("--balance-testnet", action="store_true")
    parser.add_argument("--balance-flow-pct", type=float, default=2.0)
    parser.add_argument("--closed-pnl-poll-sec", type=float, default=60.0)
    parser.add_argument("--discord-webhook", type=str, default="")
    parser.add_argument("--telegram-token", type=str, default="")
    parser.add_argument("--telegram-chat-id", type=str, default="")
    parser.add_argument("--notify-offline-sec", type=float, default=120.0)
    parser.add_argument("--notify-poll-sec", type=float, default=5.0)
    parser.add_argument("--notify-cooldown-sec", type=float, default=60.0)
    parser.add_argument("--notify-drawdown-pct", type=float, default=10.0)
    parser.add_argument("--notify-signal-threshold", type=float, default=0.9)
    parser.add_argument(
        "--notify-system",
        dest="notify_system",
        action="store_true",
        help="Enable system alerts (default).",
    )
    parser.add_argument(
        "--no-notify-system",
        dest="notify_system",
        action="store_false",
        help="Disable system alerts.",
    )
    parser.add_argument(
        "--notify-signals",
        dest="notify_signals",
        action="store_true",
        help="Enable signal alerts (default).",
    )
    parser.add_argument(
        "--no-notify-signals",
        dest="notify_signals",
        action="store_false",
        help="Disable signal alerts.",
    )
    parser.add_argument(
        "--notify-direction",
        dest="notify_direction",
        action="store_true",
        help="Enable direction alerts (default).",
    )
    parser.add_argument(
        "--no-notify-direction",
        dest="notify_direction",
        action="store_false",
        help="Disable direction alerts.",
    )
    parser.add_argument("--reset-stats", action="store_true")
    parser.add_argument("--lang", type=str, default="", help="Language code (en, es). Auto-detects if not specified.")
    parser.set_defaults(notify_system=True, notify_signals=True, notify_direction=True)
    args = parser.parse_args(argv)

    # Initialize language
    if args.lang:
        set_language(args.lang)
    else:
        set_language(detect_system_language())
    print(f"Dashboard language: {get_language()}")

    MetricsHandler.metrics_dir = Path(args.metrics_dir)
    symbols_filter = None
    if args.metrics_file:
        file_path = Path(args.metrics_file)
        symbol = args.symbol or file_path.stem.replace("live_metrics_", "", 1)
        MetricsHandler.fixed_files = {symbol: file_path}
    else:
        if args.symbols:
            symbols_filter = {s.strip() for s in args.symbols.split(",") if s.strip()}
        elif args.symbol:
            symbols_filter = {args.symbol}
        MetricsHandler.symbols_filter = symbols_filter

    MetricsHandler.history_limit = args.history_limit

    keys_path = Path(args.keys_file).expanduser() if args.keys_file else None
    if args.reset_stats:
        if MetricsHandler.fixed_files:
            reset_files = MetricsHandler.fixed_files
        else:
            reset_files = discover_metrics_files(MetricsHandler.metrics_dir, symbols_filter)
        seed_trade_stats_cache(reset_files)
    balance_monitor = BalanceMonitor(
        keys_path=keys_path,
        metrics_dir=MetricsHandler.metrics_dir,
        poll_sec=max(5.0, args.balance_poll_sec),
        history_limit=args.balance_history_limit,
        testnet=args.balance_testnet,
        flow_threshold_pct=max(0.1, args.balance_flow_pct),
        reset_history=args.reset_stats,
    )
    MetricsHandler.balance_monitor = balance_monitor
    balance_monitor.start()

    global CLOSED_PNL_MONITOR
    closed_symbols_filter = symbols_filter
    if MetricsHandler.fixed_files:
        closed_symbols_filter = set(MetricsHandler.fixed_files.keys())
    closed_pnl_monitor = ClosedPnlMonitor(
        keys_path=keys_path,
        metrics_dir=MetricsHandler.metrics_dir,
        poll_sec=max(10.0, args.closed_pnl_poll_sec),
        testnet=args.balance_testnet,
        symbols_filter=closed_symbols_filter,
        reset_cutoff_ms=int(time.time() * 1000) if args.reset_stats else None,
    )
    CLOSED_PNL_MONITOR = closed_pnl_monitor
    closed_pnl_monitor.start()

    webhook = args.discord_webhook or os.getenv("DISCORD_WEBHOOK_URL", "")
    telegram_token = args.telegram_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id = args.telegram_chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
    if args.notify_system:
        notifier = DiscordNotifier(
            webhook_url=webhook,
            metrics_dir=MetricsHandler.metrics_dir,
            poll_sec=args.notify_poll_sec,
            offline_sec=args.notify_offline_sec,
            cooldown_sec=args.notify_cooldown_sec,
            drawdown_pct=args.notify_drawdown_pct,
            fixed_files=MetricsHandler.fixed_files,
            symbols_filter=MetricsHandler.symbols_filter,
        )
        notifier.start()
    if args.notify_signals:
        signal_threshold = args.notify_signal_threshold
        env_threshold = os.getenv("SIGNAL_NOTIFY_THRESHOLD", "")
        if env_threshold:
            signal_threshold = safe_float(env_threshold, signal_threshold)
        signal_notifier = SignalNotifier(
            webhook_url=webhook,
            telegram_token=telegram_token,
            telegram_chat_id=telegram_chat_id,
            metrics_dir=MetricsHandler.metrics_dir,
            poll_sec=args.notify_poll_sec,
            min_pred=signal_threshold,
            fixed_files=MetricsHandler.fixed_files,
            symbols_filter=MetricsHandler.symbols_filter,
            include_aggressive=True,
        )
        signal_notifier.start()
    if args.notify_direction and not args.notify_signals:
        direction_notifier = DirectionNotifier(
            webhook_url=webhook,
            telegram_token=telegram_token,
            telegram_chat_id=telegram_chat_id,
            metrics_dir=MetricsHandler.metrics_dir,
            poll_sec=args.notify_poll_sec,
            fixed_files=MetricsHandler.fixed_files,
            symbols_filter=MetricsHandler.symbols_filter,
        )
        direction_notifier.start()

    server = ThreadingHTTPServer((args.host, args.port), MetricsHandler)
    server.daemon_threads = True
    print(f"Dashboard running on http://{args.host}:{args.port}")
    if MetricsHandler.fixed_files:
        print(f"Reading metrics: {list(MetricsHandler.fixed_files.values())[0]}")
    else:
        print(f"Reading metrics dir: {MetricsHandler.metrics_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
