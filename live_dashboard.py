"""
Live Metrics Dashboard (read-only).
Serves a small web UI backed by live_metrics_<symbol>.jsonl.
"""
import argparse
import csv
import json
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

try:
    from pybit.unified_trading import HTTP as BybitHTTP
except Exception:
    BybitHTTP = None


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


def safe_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def safe_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


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
    exch_daily = exchange_daily_pnl(symbol)
    if exch_daily is not None:
        latest["daily_pnl"] = exch_daily
    price = extract_price(latest)
    resting = read_open_orders_latest(metrics_path, symbol)
    latest["dashboard"] = {
        "price": price,
        "resting_orders": resting,
        "resting_count": len(resting),
        "trade_stats": compute_trade_stats(metrics_path),
    }
    return latest


def discover_metrics_files(metrics_dir: Path, symbols=None):
    mapping = {}
    if not metrics_dir.exists():
        return mapping
    for path in metrics_dir.glob("live_metrics_*.jsonl"):
        symbol = path.stem.replace("live_metrics_", "", 1)
        if not symbol:
            continue
        if symbols and symbol not in symbols:
            continue
        mapping[symbol] = path
    return mapping


def summary_from_files(file_map):
    items = []
    for symbol in sorted(file_map.keys()):
        path = file_map[symbol]
        latest = read_latest(path)
        if not latest:
            latest = {"symbol": symbol, "ts": None}
        elif not latest.get("symbol"):
            latest["symbol"] = symbol
        latest = augment_latest(latest, symbol, path)
        items.append(latest)
    return items


EXTRA_SCRIPT = """
<script>
(function() {
  if (window.__dashExtrasInjected) return;
  window.__dashExtrasInjected = true;

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
      <h3>Market</h3>
      <div class="value" id="dashPrice">--</div>
      <div class="row"><span>Position Mode</span><span id="dashPosMode">--</span></div>
      <div class="row"><span>Resting Orders</span><span id="dashOrderCount">--</span></div>
      <div class="row"><span>Limits</span><span id="dashOrderList" style="white-space: pre-line; text-align: right;">--</span></div>
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
      <h3>Performance</h3>
      <div class="value" id="dashWinRate">--</div>
      <div class="row"><span>Trades</span><span id="dashTradeCount">--</span></div>
      <div class="row"><span>Sortino</span><span id="dashSortino">--</span></div>
      <div class="row"><span>Profit Factor</span><span id="dashProfitFactor">--</span></div>
      <div class="row"><span>Avg Trade</span><span id="dashAvgTrade">--</span></div>
      <div class="row"><span>Max DD</span><span id="dashMaxDD">--</span></div>
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
      <h3>Iceberg</h3>
      <div class="value" id="dashIcebergStatus">--</div>
      <div class="row"><span>Entry Buy</span><span id="dashIcebergBuy" style="white-space: pre-line; text-align: right;">--</span></div>
      <div class="row"><span>Entry Sell</span><span id="dashIcebergSell" style="white-space: pre-line; text-align: right;">--</span></div>
      <div class="row"><span>TP Long</span><span id="dashIcebergTpLong" style="white-space: pre-line; text-align: right;">--</span></div>
      <div class="row"><span>TP Short</span><span id="dashIcebergTpShort" style="white-space: pre-line; text-align: right;">--</span></div>
      <div class="row"><span>L1</span><span id="dashIcebergL1">--</span></div>
      <div class="row"><span>Controls</span>
        <span id="dashIcebergControls" class="iceberg-controls">
          <button class="mini-btn" id="icebergCancelBuy">Cancel Buy</button>
          <button class="mini-btn" id="icebergCancelSell">Cancel Sell</button>
          <button class="mini-btn" id="icebergCancelTpLong">Cancel TP Long</button>
          <button class="mini-btn" id="icebergCancelTpShort">Cancel TP Short</button>
        </span>
      </div>
    `;
    grid.appendChild(card);
  }

  let chart = null;
  let chartInitialized = false;
  let chartTimer = null;
  let chartFetchInFlight = false;
  let chartFullCandles = [];
  let chartIntervalSec = 60;
  let chartLastSymbol = null;
  let chartPricePrecision = 6;
  let chartLibLoading = false;
  let chartLibFallback = false;
  const MIN_PRICE_PRECISION = 5;
  let chartNeedsFit = true;
  let chartUserZoom = false;
  let chartZoomState = null;
  let chartOverlay = { lines: [], markers: [], livePrice: null };

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
        <h3>Iceberg Chart</h3>
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
          if (meta) meta.textContent = "Chart library failed to load.";
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
      if (meta) meta.textContent = "Chart library failed to load.";
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
      lines.push(`Price ${fmtPrice(lineValue)}`);
    }
    if (volume !== null && volume !== undefined && volume !== 0) {
      lines.push(`Vol ${fmtQty(volume)}`);
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
        name: "Price",
        type: "candlestick",
        data: candleData,
        encode: { x: 0, y: [1, 2, 3, 4] },
        itemStyle: { color: "#22c55e", color0: "#ef4444", borderColor: "#22c55e", borderColor0: "#ef4444" },
      },
      {
        id: "volume",
        name: "Volume",
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
        name: "Price",
        type: "line",
        data: lineData,
        encode: { x: 0, y: 1 },
        showSymbol: false,
        lineStyle: { color: "#38bdf8", width: 1.4 },
        emphasis: { focus: "series" },
      },
      {
        id: "volume",
        name: "Volume",
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
        name: `Last ${fmtPrice(lastPrice)}`,
        lineStyle: { color: "#38bdf8", width: 1, type: "solid" },
        label: { show: true, formatter: "{b}", color: "#38bdf8", fontSize: 10 },
      });
    }
    const markers = (chartOverlay.markers || [])
      .filter(m => Number.isFinite(m.time) && Number.isFinite(m.price))
      .map(m => ({
        coord: [m.time, m.price],
        name: m.title || "",
        itemStyle: { color: m.color || "#38bdf8" },
        label: { show: true, color: "#0b1220", fontSize: 9, formatter: "{b}" },
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
      setChartMeta("No chart data yet.");
      return;
    }
    renderCandles(filtered);
    const last = filtered[filtered.length - 1];
    applyPricePrecision(last.close);
    const interval = chartIntervalSec ? `${chartIntervalSec}s` : "--";
    const lastTime = new Date(last.time).toLocaleTimeString();
    const label = chartRangeLabel || "All";
    setChartMeta(`Range ${label} | Bars ${filtered.length} | Last ${fmtPrice(last.close)} | ${lastTime} | Int ${interval}`);
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

  async function fetchMetricHistory(symbol) {
    if (!chartInitialized) return;
    const sampleSec = window.__dashMetricsIntervalSec || 2;
    const targetRange = chartRangeSec || (24 * 60 * 60);
    const desired = Math.ceil(targetRange / sampleSec) + 200;
    const limit = Math.min(10000, Math.max(300, desired));
    const resp = await fetch(`/api/metrics?symbol=${encodeURIComponent(symbol)}&limit=${limit}`);
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
      setChartMeta("No chart data yet.");
      return;
    }
    const now = Date.now();
    const cutoff = chartRangeSec ? (now - chartRangeSec * 1000) : 0;
    const filtered = cutoff > 0 ? points.filter(p => p.time >= cutoff) : points;
    if (!filtered.length) {
      setChartMeta("No data in selected range.");
      return;
    }
    chartFullCandles = [];
    renderLine(filtered);
    const last = filtered[filtered.length - 1];
    applyPricePrecision(last.value);
    const label = chartRangeLabel || "All";
    setChartMeta(`Range ${label} | Points ${filtered.length} | Last ${fmtPrice(last.value)} | Metrics feed`);
  }

  async function fetchChartHistory(force = false) {
    if (!chartInitialized) return;
    const symbol = window.__dashSymbol;
    if (!symbol || chartFetchInFlight) return;
    if (force) {
      chartNeedsFit = true;
      chartUserZoom = false;
      chartZoomState = null;
    }
    chartFetchInFlight = true;
    const currentSymbol = symbol;
    try {
      const limit = computeCandleLimit();
      const resp = await fetch(`/api/candles?symbol=${encodeURIComponent(symbol)}&limit=${limit}`);
      if (!resp.ok) {
        await fetchMetricHistory(symbol);
        return;
      }
      const payload = await resp.json();
      if (currentSymbol !== window.__dashSymbol) return;
      if (payload && payload.interval_sec) {
        const interval = Number(payload.interval_sec);
        if (Number.isFinite(interval) && interval > 0) {
          chartIntervalSec = interval;
        }
      }
      const candles = normalizeCandles(payload ? payload.candles : []);
      if (!candles.length) {
        await fetchMetricHistory(symbol);
        return;
      }
      setCandlesData(candles);
    } catch (_err) {
      await fetchMetricHistory(symbol);
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

  function updateChartMarkers(data) {
    const markers = [];
    const ice = data.iceberg || {};
    const tp = data.iceberg_tp || {};
    const pushMarker = (time, price, color, title) => {
      const t = Number(time || 0);
      const p = Number(price || 0);
      if (!Number.isFinite(t) || t <= 0 || !Number.isFinite(p) || p <= 0) return;
      markers.push({ time: t, price: p, color, title });
    };
    if (ice.buy && ice.buy.active) {
      const time = resolveMarkerTime(ice.buy.last_post_ts || ice.buy.created_ts);
      const price = Number(ice.buy.last_price || ice.buy.target_price || 0);
      const info = ice.buy.last_post_ts ? icebergClipLabel(ice.buy) : icebergEntryLabel(ice.buy);
      const label = ice.buy.last_post_ts ? "Buy Clip" : "Buy Entry";
      pushMarker(time, price, "#22c55e", lineTitle(label, info));
    }
    if (ice.sell && ice.sell.active) {
      const time = resolveMarkerTime(ice.sell.last_post_ts || ice.sell.created_ts);
      const price = Number(ice.sell.last_price || ice.sell.target_price || 0);
      const info = ice.sell.last_post_ts ? icebergClipLabel(ice.sell) : icebergEntryLabel(ice.sell);
      const label = ice.sell.last_post_ts ? "Sell Clip" : "Sell Entry";
      pushMarker(time, price, "#f97316", lineTitle(label, info));
    }
    if (tp.long && (tp.long.active || tp.long.paused)) {
      const time = resolveMarkerTime(tp.long.tp_order_ts || tp.long.created_ts);
      const price = Number(tp.long.tp_order_price || tp.long.tp_target || 0);
      pushMarker(time, price, "#60a5fa", lineTitle("TP Long", tpInfoLabel(tp.long)));
    }
    if (tp.short && (tp.short.active || tp.short.paused)) {
      const time = resolveMarkerTime(tp.short.tp_order_ts || tp.short.created_ts);
      const price = Number(tp.short.tp_order_price || tp.short.tp_target || 0);
      pushMarker(time, price, "#f59e0b", lineTitle("TP Short", tpInfoLabel(tp.short)));
    }
    chartOverlay.markers = markers;
    applyChartOverlay();
  }

  function updateChartLines(data) {
    const levels = [];
    const ice = data.iceberg || {};
    if (ice.buy && ice.buy.active) {
      levels.push({ price: ice.buy.target_price, color: "#22c55e", title: priceTitle("Entry Buy", ice.buy.target_price, icebergEntryLabel(ice.buy)) });
      if (ice.buy.tp) levels.push({ price: ice.buy.tp, color: "#38bdf8", title: priceTitle("Entry TP", ice.buy.tp, "") });
      if (ice.buy.sl) levels.push({ price: ice.buy.sl, color: "#ef4444", title: priceTitle("Entry SL", ice.buy.sl, "") });
      if (ice.buy.order_id && ice.buy.last_price) {
        levels.push({ price: ice.buy.last_price, color: "#16a34a", title: priceTitle("Buy Clip", ice.buy.last_price, icebergClipLabel(ice.buy)) });
      }
    }
    if (ice.sell && ice.sell.active) {
      levels.push({ price: ice.sell.target_price, color: "#f97316", title: priceTitle("Entry Sell", ice.sell.target_price, icebergEntryLabel(ice.sell)) });
      if (ice.sell.tp) levels.push({ price: ice.sell.tp, color: "#38bdf8", title: priceTitle("Entry TP", ice.sell.tp, "") });
      if (ice.sell.sl) levels.push({ price: ice.sell.sl, color: "#ef4444", title: priceTitle("Entry SL", ice.sell.sl, "") });
      if (ice.sell.order_id && ice.sell.last_price) {
        levels.push({ price: ice.sell.last_price, color: "#ea580c", title: priceTitle("Sell Clip", ice.sell.last_price, icebergClipLabel(ice.sell)) });
      }
    }
    const tp = data.iceberg_tp || {};
    if (tp.long && (tp.long.active || tp.long.paused)) {
      if (tp.long.tp_target) levels.push({ price: tp.long.tp_target, color: "#60a5fa", title: priceTitle("TP Long", tp.long.tp_target, tpInfoLabel(tp.long)) });
      if (tp.long.sl_target) levels.push({ price: tp.long.sl_target, color: "#ef4444", title: priceTitle("SL Long", tp.long.sl_target, "") });
      if (tp.long.tp_order_price) levels.push({ price: tp.long.tp_order_price, color: "#0ea5e9", title: priceTitle("TP Clip", tp.long.tp_order_price, tpInfoLabel(tp.long)) });
    }
    if (tp.short && (tp.short.active || tp.short.paused)) {
      if (tp.short.tp_target) levels.push({ price: tp.short.tp_target, color: "#f59e0b", title: priceTitle("TP Short", tp.short.tp_target, tpInfoLabel(tp.short)) });
      if (tp.short.sl_target) levels.push({ price: tp.short.sl_target, color: "#ef4444", title: priceTitle("SL Short", tp.short.sl_target, "") });
      if (tp.short.tp_order_price) levels.push({ price: tp.short.tp_order_price, color: "#d97706", title: priceTitle("TP Clip", tp.short.tp_order_price, tpInfoLabel(tp.short)) });
    }
    chartOverlay.lines = levels.filter(level => Number(level.price) > 0);
    applyChartOverlay();
  }

  let controlsBound = false;
  let chartControlsBound = false;
  let chartRangeSec = 15 * 60;
  let chartRangeLabel = "15M";
  async function sendCommand(scope, side) {
    const symbol = window.__dashSymbol;
    if (!symbol) return;
    const payload = {
      ts: Date.now() / 1000,
      action: "cancel_iceberg",
      scope,
      side,
      symbol,
    };
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
      "all": { label: "All", sec: 0 },
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
    const state = ice.order_id ? "posted" : "waiting";
    const ready = (ice.activation_ready === null || ice.activation_ready === undefined)
      ? "--"
      : (ice.activation_ready ? "ready" : "wait");
    return `T ${target} TP ${tp} SL ${sl}\\nRem ${rem}/${total} Clip ${clip} ${state} ${ready}`;
  }

  function formatTpIceberg(tp) {
    if (!tp || (!tp.active && !tp.paused)) return "--";
    const target = fmtPrice(tp.tp_target);
    const sl = fmtPrice(tp.sl_target);
    const size = fmtQty(tp.size);
    const clipQty = fmtQty(tp.tp_order_qty);
    const clipPrice = fmtPrice(tp.tp_order_price);
    const clip = (tp.tp_order_qty && tp.tp_order_price) ? `${clipQty} @ ${clipPrice}` : "--";
    const state = tp.paused ? "paused" : (tp.tp_order_id ? "posted" : "waiting");
    const ready = (tp.activation_ready === null || tp.activation_ready === undefined)
      ? "--"
      : (tp.activation_ready ? "ready" : "wait");
    return `TP ${target} SL ${sl}\\nSize ${size} Clip ${clip} ${state} ${ready}`;
  }

  function icebergStateLabel(ice) {
    if (!ice || !ice.active) return "";
    if (ice.order_id) return "posted";
    if (ice.activation_ready === true) return "ready";
    if (ice.activation_ready === false) return "wait";
    return "wait";
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
    if (total > 0) parts.push(`Size ${fmtQty(total)}`);
    const fill = icebergFillLabel(ice);
    if (fill) parts.push(`Fill ${fill}`);
    const state = icebergStateLabel(ice);
    if (state) parts.push(state);
    return parts.join(" ");
  }

  function icebergClipLabel(ice) {
    if (!ice || !ice.active) return "";
    const parts = [];
    const clip = Number(ice.clip_qty || 0);
    if (clip > 0) parts.push(`Clip ${fmtQty(clip)}`);
    const state = icebergStateLabel(ice);
    if (state) parts.push(state);
    return parts.join(" ");
  }

  function tpStateLabel(tp) {
    if (!tp) return "";
    if (tp.paused) return "paused";
    if (tp.tp_order_id) return "posted";
    if (tp.activation_ready === true) return "ready";
    if (tp.activation_ready === false) return "wait";
    return "wait";
  }

  function tpInfoLabel(tp) {
    if (!tp || (!tp.active && !tp.paused)) return "";
    const parts = [];
    const size = Number(tp.size || 0);
    if (size > 0) parts.push(`Size ${fmtQty(size)}`);
    const clip = Number(tp.tp_order_qty || 0);
    if (clip > 0) parts.push(`Clip ${fmtQty(clip)}`);
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
    ensureChartSection();
    bindIcebergControls();
    bindChartRangeControls();
    if (data.symbol) {
      window.__dashSymbol = data.symbol;
    }
    if (data.metrics_interval_sec) {
      window.__dashMetricsIntervalSec = Number(data.metrics_interval_sec) || window.__dashMetricsIntervalSec;
    }
    const chartTitle = document.getElementById("chartTitle");
    if (chartTitle && data.symbol) chartTitle.textContent = `${data.symbol} Chart`;
    const chartSub = document.getElementById("chartSub");
    if (chartSub) chartSub.textContent = data.ts ? `Updated ${data.ts}` : "Waiting for data...";
    const chartStatus = document.getElementById("chartStatus");
    if (chartStatus) chartStatus.textContent = data.health?.status || "Live";
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
    if (buy.active) statusParts.push("Entry Buy");
    if (sell.active) statusParts.push("Entry Sell");
    if (tpLong.active) {
      statusParts.push("TP Long");
    } else if (tpLong.paused) {
      statusParts.push("TP Long (paused)");
    }
    if (tpShort.active) {
      statusParts.push("TP Short");
    } else if (tpShort.paused) {
      statusParts.push("TP Short (paused)");
    }
    const statusText = statusParts.length ? statusParts.join(" & ") : "Idle";
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
      ? `${fmtPrice(l1.bid)} / ${fmtPrice(l1.ask)} (${fmt(l1.age_sec, 1)}s | exch ${fmt(l1.exchange_age_sec, 1)}s | ${l1.source || "n/a"})`
      : "--";
    const l1El = document.getElementById("dashIcebergL1");
    if (l1El) l1El.textContent = l1Text;

    loadChartLib(() => {
      initChart();
      const symbol = window.__dashSymbol;
      const symbolChanged = symbol && symbol !== chartLastSymbol;
      if (symbolChanged) {
        chartLastSymbol = symbol;
        chartFullCandles = [];
        chartNeedsFit = true;
        chartUserZoom = false;
        chartZoomState = null;
        chartOverlay = { lines: [], markers: [], livePrice: null };
        if (chart) {
          chart.clear();
          chart.setOption(baseChartOption(), { notMerge: true, lazyUpdate: true });
        }
      }
      updateChartLines(data);
      updateChartMarkers(data);
      updateLivePriceLine(resolveLivePrice(data));
      if (!chartTimer) {
        fetchChartHistory(true);
        chartTimer = setInterval(fetchChartHistory, 10000);
      } else if (symbolChanged) {
        fetchChartHistory(true);
      }
    });
  }

  window.refreshIcebergChart = function() {
    loadChartLib(() => {
      initChart();
      fetchChartHistory(true);
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


def inject_dashboard_html(html: str) -> str:
    if "dashMarketCard" in html and "dashPerfCard" in html and "dashIcebergCard" in html:
        return html
    marker = "</body>"
    if marker in html:
        return html.replace(marker, EXTRA_SCRIPT + "\n" + marker)
    return html + EXTRA_SCRIPT


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Live Trading Dashboard</title>
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
      document.getElementById("symbol").textContent = `Live Trading Dashboard - ${data.symbol}`;
      document.getElementById("subline").textContent = `Startup: ${data.startup_time} - Uptime: ${Math.round(data.uptime_sec)}s`;
      document.getElementById("statusPill").textContent = data.trade_enabled ? "Trading Enabled" : "Trading Paused";
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
      document.getElementById("tradeEnabled").textContent = data.trade_enabled ? "Yes" : "No";
      const drift = data.drift.alerts && data.drift.alerts.length ? data.drift.alerts.join(", ") : "None";
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

HTML_TEMPLATE_PATH = Path(__file__).with_name("live_dashboard.html")
if HTML_TEMPLATE_PATH.exists():
    try:
        HTML_PAGE = HTML_TEMPLATE_PATH.read_text(encoding="utf-8")
    except Exception:
        pass
HTML_PAGE = inject_dashboard_html(HTML_PAGE)


class MetricsHandler(BaseHTTPRequestHandler):
    metrics_dir: Path = Path(".")
    symbols_filter = None
    fixed_files = None
    history_limit: int = 200
    balance_monitor = None
    static_dir: Path = Path(__file__).with_name("static")

    def _send(self, status, payload, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if isinstance(payload, (dict, list)):
            payload = json.dumps(payload, default=str).encode("utf-8")
        elif isinstance(payload, str):
            payload = payload.encode("utf-8")
        self.wfile.write(payload)

    def _resolve_files(self):
        if self.fixed_files:
            return self.fixed_files
        return discover_metrics_files(self.metrics_dir, self.symbols_filter)

    def _choose_symbol(self, file_map, symbol):
        if symbol and symbol in file_map:
            return symbol
        symbols = sorted(file_map.keys())
        return symbols[0] if symbols else None

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
        symbol = payload.get("symbol")
        symbol = self._choose_symbol(file_map, symbol)
        if not symbol:
            return self._send(400, {"error": "unknown symbol"})
        payload["ts"] = payload.get("ts") or time.time()
        cmd_path = self.metrics_dir / f"commands_{symbol}.jsonl"
        append_jsonl(cmd_path, payload)
        return self._send(200, {"ok": True})

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self._send(200, HTML_PAGE, "text/html; charset=utf-8")

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
            symbol = qs.get("symbol", [""])[0] or None
            symbol = self._choose_symbol(file_map, symbol)
            if not symbol:
                return self._send(200, {})
            latest = read_latest(file_map[symbol])
            if not latest:
                latest = {"symbol": symbol, "ts": None}
            elif not latest.get("symbol"):
                latest["symbol"] = symbol
            latest = augment_latest(latest, symbol, file_map[symbol])
            return self._send(200, latest)

        if parsed.path == "/api/metrics":
            qs = parse_qs(parsed.query)
            symbol = qs.get("symbol", [""])[0] or None
            symbol = self._choose_symbol(file_map, symbol)
            if not symbol:
                return self._send(200, [])
            limit = int(qs.get("limit", [self.history_limit])[0])
            max_bytes = max(512 * 1024, limit * 1024)
            max_bytes = min(max_bytes, 16 * 1024 * 1024)
            return self._send(200, tail_jsonl(file_map[symbol], limit=limit, max_bytes=max_bytes))

        if parsed.path == "/api/signals":
            qs = parse_qs(parsed.query)
            symbol = qs.get("symbol", [""])[0] or None
            symbol = self._choose_symbol(file_map, symbol)
            if not symbol:
                return self._send(200, [])
            limit = int(qs.get("limit", [40])[0])
            limit = max(1, min(200, limit))
            max_bytes = max(256 * 1024, limit * 1024)
            max_bytes = min(max_bytes, 4 * 1024 * 1024)
            signals_path = self.metrics_dir / f"signals_{symbol}.jsonl"
            return self._send(200, tail_jsonl(signals_path, limit=limit, max_bytes=max_bytes))

        if parsed.path == "/api/candles":
            qs = parse_qs(parsed.query)
            symbol = qs.get("symbol", [""])[0] or None
            symbol = self._choose_symbol(file_map, symbol)
            if not symbol:
                return self._send(200, {"candles": [], "interval_sec": None})
            limit = int(qs.get("limit", [2000])[0])
            limit = max(100, min(20000, limit))
            max_bytes = max(512 * 1024, limit * 256)
            max_bytes = min(max_bytes, 16 * 1024 * 1024)
            candles_path = self.metrics_dir / f"data_history_{symbol}.csv"
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

    def _cooldown_ok(self, symbol: str, event: str) -> bool:
        key = f"{symbol}:{event}"
        now = time.time()
        last = self.last_sent.get(key, 0.0)
        if now - last < self.cooldown_sec:
            return False
        self.last_sent[key] = now
        return True

    def _notify(self, symbol: str, event: str, message: str) -> None:
        if not self._cooldown_ok(symbol, event):
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

    def _check_symbol(self, symbol: str, latest: dict) -> None:
        state = self.last_state.setdefault(symbol, {})
        now = time.time()
        ts = parse_ts(latest.get("ts")) if latest else None
        online = ts is not None and (now - ts) <= self.offline_sec
        if state.get("online", True) and not online:
            age = now - ts if ts else None
            age_text = f"{age:.1f}s" if age is not None else "unknown"
            self._notify(symbol, "offline", f"{symbol} offline (last update {age_text} ago)")
        state["online"] = online

        if not latest or not online:
            return

        startup = latest.get("startup_time")
        if startup and startup != state.get("startup_time"):
            self._notify(symbol, "started", f"{symbol} started (startup {startup})")
            state["startup_time"] = startup

        errors = latest.get("errors") or {}
        runtime_count = int(errors.get("runtime_count") or 0)
        api_count = int(errors.get("api_count") or 0)
        if runtime_count > state.get("runtime_count", 0) or api_count > state.get("api_count", 0):
            err_text = errors.get("last_runtime_error") or errors.get("last_api_error") or "unknown error"
            self._notify(symbol, "error", f"{symbol} error: {err_text}")
            if self._is_order_error(err_text):
                self._notify(symbol, "order_error", f"{symbol} order error: {err_text}")
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
                symbol,
                "drawdown",
                f"{symbol} drawdown alert: {daily_pnl:.2f} ({pct:.1f}%)",
            )
            state["drawdown_alerted"] = True
        elif not drawdown:
            state["drawdown_alerted"] = False

    def _run(self) -> None:
        while not self._stop.is_set():
            if self.fixed_files:
                file_map = self.fixed_files
            else:
                file_map = discover_metrics_files(self.metrics_dir, self.symbols_filter)
            for symbol, path in file_map.items():
                latest = read_latest(path)
                if latest and not latest.get("symbol"):
                    latest["symbol"] = symbol
                self._check_symbol(symbol, latest or {})
            time.sleep(self.poll_sec)


def main():
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
    parser.add_argument("--notify-offline-sec", type=float, default=120.0)
    parser.add_argument("--notify-poll-sec", type=float, default=5.0)
    parser.add_argument("--notify-cooldown-sec", type=float, default=60.0)
    parser.add_argument("--notify-drawdown-pct", type=float, default=10.0)
    parser.add_argument("--reset-stats", action="store_true")
    args = parser.parse_args()

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

    server = HTTPServer((args.host, args.port), MetricsHandler)
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
