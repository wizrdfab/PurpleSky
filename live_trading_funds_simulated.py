import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import DEFAULT_CONFIG
from train import _load_train_config
from models import TrendFollowerModels
from data_loader import load_trades, preprocess_trades, create_multi_timeframe_bars
from feature_engine import calculate_multi_timeframe_features
from labels import create_training_dataset
from backtest import SimpleBacktester
from live_trading_funds import LiveFundsTrader
from live_trading import PaperPosition, CompletedTrade


@dataclass
class OrderResult:
    success: bool
    order_id: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class Position:
    is_open: bool
    side: str = ""
    size: float = 0.0
    entry_price: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class SimulatedSession:
    """Minimal session stub for get_closed_pnl."""

    def __init__(self, exchange: "SimulatedBybitClient"):
        self.exchange = exchange

    def get_closed_pnl(self, category: str = "linear", symbol: Optional[str] = None, limit: int = 1):
        closed = self.exchange.get_last_closed(symbol)
        items = [closed] if closed else []
        return {"retCode": 0, "result": {"list": items}}


class SimulatedBybitClient:
    """Simulated exchange that uses historical trades to trigger SL/TP exits."""

    def __init__(
        self,
        starting_balance: float,
        tick_size: float = 0.0,
        qty_step: float = 0.0,
        min_qty: float = 0.0,
        category: str = "linear",
    ):
        self.category = category
        self._balance = float(starting_balance)
        self._tick_size = float(tick_size)
        self._qty_step = float(qty_step)
        self._min_qty = float(min_qty)
        self._last_price: Optional[float] = None
        self.entry_price_override: Optional[float] = None
        self._position: Optional[dict] = None
        self._last_closed: Optional[dict] = None
        self._order_counter = 0
        self.last_order_error: Optional[str] = None
        self.session = SimulatedSession(self)

    def update_trade(self, price: float, timestamp: float) -> None:
        self._last_price = float(price)
        if not self._position:
            return

        direction = self._position["direction"]
        stop = self._position["stop_loss"]
        target = self._position["take_profit"]
        size = self._position["size"]
        entry = self._position["entry_price"]
        symbol = self._position["symbol"]

        exit_reason = None
        exit_price = None

        if direction == 1:
            if price <= stop:
                exit_reason = "stop_loss"
                exit_price = stop
            elif price >= target:
                exit_reason = "take_profit"
                exit_price = target
        else:
            if price >= stop:
                exit_reason = "stop_loss"
                exit_price = stop
            elif price <= target:
                exit_reason = "take_profit"
                exit_price = target

        if exit_reason is None:
            return

        pnl = direction * (exit_price - entry) * size
        self._balance += pnl
        self._last_closed = {
            "symbol": symbol,
            "side": "Buy" if direction == 1 else "Sell",
            "avgExitPrice": float(exit_price),
            "closedPnl": float(pnl),
            "updatedTime": int(timestamp * 1000),
            "exitReason": exit_reason,
        }
        self._position = None

    def get_last_closed(self, symbol: Optional[str] = None) -> Optional[dict]:
        if not self._last_closed:
            return None
        if symbol and self._last_closed.get("symbol") != symbol:
            return None
        return dict(self._last_closed)

    def get_available_balance(self, asset: str = "USDT", account_types: Optional[list] = None, logger=None) -> float:
        return float(self._balance)

    def get_instrument_info(self, symbol: str) -> Optional[dict]:
        return {
            "lotSizeFilter": {
                "qtyStep": str(self._qty_step),
                "minOrderQty": str(self._min_qty),
            },
            "priceFilter": {
                "tickSize": str(self._tick_size),
            },
        }

    def get_current_price(self, symbol: str) -> Optional[float]:
        return self._last_price

    def get_position(self, symbol: str) -> Optional[Position]:
        if not self._position or self._position.get("symbol") != symbol:
            return None
        side = "Buy" if self._position["direction"] == 1 else "Sell"
        return Position(
            is_open=True,
            side=side,
            size=float(self._position["size"]),
            entry_price=float(self._position["entry_price"]),
            stop_loss=float(self._position["stop_loss"]),
            take_profit=float(self._position["take_profit"]),
        )

    def set_leverage(self, symbol: str, leverage: int = 1) -> bool:
        return True

    def open_position(self, symbol: str, side: str, qty: float, stop_loss: float, take_profit: float, leverage: int = 1) -> OrderResult:
        self.last_order_error = None
        if self._position is not None:
            self.last_order_error = "position already open"
            return OrderResult(success=False, error_message=self.last_order_error)

        entry_price = self.entry_price_override if self.entry_price_override is not None else self._last_price
        self.entry_price_override = None
        if entry_price is None:
            self.last_order_error = "no price available"
            return OrderResult(success=False, error_message=self.last_order_error)

        direction = 1 if side.lower() == "buy" else -1
        self._order_counter += 1
        self._position = {
            "symbol": symbol,
            "direction": direction,
            "size": float(qty),
            "entry_price": float(entry_price),
            "stop_loss": float(stop_loss),
            "take_profit": float(take_profit),
        }
        return OrderResult(success=True, order_id=f"SIM-{self._order_counter}")


@dataclass
class SimResult:
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pnl_percent: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    sharpe_ratio: float = 0.0
    trades: List[CompletedTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


class SimulatedLiveFundsTrader(LiveFundsTrader):
    """LiveFundsTrader wired to a simulated exchange and trade replay."""

    def __init__(
        self,
        *args,
        starting_balance: float = 10000.0,
        max_entry_deviation_atr: float = 1.0,
        entry_fill_mode: str = "bar_close",
        ema_touch_mode: str = "base",
        test_start_bar_time: Optional[int] = None,
        test_end_bar_time: Optional[int] = None,
        **kwargs,
    ):
        super(LiveFundsTrader, self).__init__(*args, **kwargs)

        self.leverage = max(1, int(kwargs.get("leverage", 1)))
        self.balance_asset = kwargs.get("balance_asset", "USDT")
        self.dry_run = bool(kwargs.get("dry_run", False))
        self.exit_on_bar_close_only = bool(self.dry_run)

        self._instrument_info: Optional[dict] = None
        self._qty_step: float = 0.0
        self._min_qty: float = 0.0
        self._tick_size: float = 0.0
        self._last_entry_skip_log: Optional[datetime] = None
        self._max_entry_price_deviation_atr: float = max(0.0, float(max_entry_deviation_atr))

        self._bybit = SimulatedBybitClient(starting_balance=starting_balance)
        self.initial_capital = float(starting_balance)
        self.capital = float(starting_balance)

        self.entry_fill_mode = str(entry_fill_mode or "bar_close").lower()
        if self.entry_fill_mode not in ("bar_close", "last_trade"):
            self.entry_fill_mode = "bar_close"

        self.ema_touch_mode = str(ema_touch_mode or "base").lower()
        if self.ema_touch_mode not in ("base", "multi"):
            self.ema_touch_mode = "base"

        self.test_start_bar_time = test_start_bar_time
        self.test_end_bar_time = test_end_bar_time
        self.last_entry_guard_reason: Optional[str] = None
        self.last_entry_guard_details: dict = {}
        self.last_entry_guard_bar_time: Optional[int] = None

    def _connect_websocket(self):
        return

    def _check_entry(self, current_atr: float):
        if self._last_closed_bar_time is not None:
            if self.test_start_bar_time is not None and int(self._last_closed_bar_time) < int(self.test_start_bar_time):
                return
            if self.test_end_bar_time is not None and int(self._last_closed_bar_time) >= int(self.test_end_bar_time):
                return
        return super()._check_entry(current_atr)

    def _open_position(self, direction: int, quality: str, atr: float, entry_price: Optional[float] = None, expected_rr: Optional[float] = None):
        self.last_entry_guard_reason = None
        self.last_entry_guard_details = {}
        self.last_entry_guard_bar_time = None
        if isinstance(self._bybit, SimulatedBybitClient):
            if self.entry_fill_mode == "bar_close":
                signal_price = entry_price if entry_price is not None else self.current_price
                self._bybit.entry_price_override = float(signal_price)
            else:
                self._bybit.entry_price_override = None
        super()._open_position(direction, quality, atr, entry_price=entry_price, expected_rr=expected_rr)
        if self.position is not None and self._last_closed_bar_time is not None:
            try:
                self.position.entry_time = datetime.utcfromtimestamp(int(self._last_closed_bar_time))
            except Exception:
                pass
        if self.position is None:
            reason, details = self._diagnose_entry_guard(direction, atr, entry_price)
            if reason:
                self.last_entry_guard_reason = reason
                self.last_entry_guard_details = details
                self.last_entry_guard_bar_time = self._last_closed_bar_time

    def _compute_base_tf_touch(self) -> dict:
        ema_col = f'{self.base_tf}_ema_{self.config.labels.pullback_ema}'
        atr_col = f'{self.base_tf}_atr'
        slope_col = f'{self.base_tf}_ema_{self.config.labels.pullback_ema}_slope_norm'

        ema = super()._get_feature_value(ema_col)
        atr = super()._get_feature_value(atr_col)
        bar_high = super()._get_feature_value('high')
        bar_low = super()._get_feature_value('low')
        bar_close = super()._get_feature_value('close', default=self.current_price)

        if ema is None or atr is None or atr <= 0 or bar_high is None or bar_low is None:
            return {"ema_touch_detected": False, "ema_touch_direction": 0, "ema_touch_dist": None}

        slope_val = super()._get_feature_value(slope_col, default=0.0) or 0.0
        trend_dir = 1 if slope_val > 0 else -1 if slope_val < 0 else 0
        if trend_dir == 0:
            return {"ema_touch_detected": False, "ema_touch_direction": 0, "ema_touch_dist": None}

        threshold = getattr(self.config.labels, "touch_threshold_atr", 0.3)
        mid_bar = (bar_high + bar_low) / 2.0

        ema_touched = False
        touch_dist = None
        if trend_dir == 1:
            dist_low = (bar_low - ema) / atr
            if -threshold <= dist_low <= threshold and (bar_close >= ema or mid_bar >= ema):
                ema_touched = True
                touch_dist = dist_low
        else:
            dist_high = (bar_high - ema) / atr
            if -threshold <= dist_high <= threshold and (bar_close <= ema or mid_bar <= ema):
                ema_touched = True
                touch_dist = dist_high

        return {
            "ema_touch_detected": ema_touched,
            "ema_touch_direction": trend_dir if ema_touched else 0,
            "ema_touch_dist": touch_dist,
        }

    def _get_feature_value(self, feature_name: str, default=None):
        if (
            self.use_incremental
            and self.ema_touch_mode == "base"
            and feature_name in ("ema_touch_detected", "ema_touch_direction", "ema_touch_dist")
        ):
            base_touch = self._compute_base_tf_touch()
            return base_touch.get(feature_name, default)
        return super()._get_feature_value(feature_name, default)

    def _diagnose_entry_guard(
        self,
        direction: int,
        atr: float,
        entry_price: Optional[float],
    ) -> Tuple[Optional[str], dict]:
        details: dict = {}

        if self.position is not None:
            return "position_open", details

        if not self.dry_run and self._get_exchange_position_is_open():
            return "exchange_position_open", details

        signal_price = entry_price if entry_price is not None else self.current_price
        details["signal_price"] = signal_price
        details["atr"] = atr
        if signal_price <= 0 or atr <= 0:
            return "invalid_price_or_atr", details

        if not self.dry_run and self._bybit is not None:
            if self.last_trade_timestamp is None:
                return "no_last_trade_timestamp", details

            latest_bar_time = None
            if self.use_incremental:
                latest_bar_time = self.predictor.last_bar_time.get(self.base_tf)
            else:
                fc = self.predictor.features_cache
                if fc is None or len(fc) == 0 or "bar_time" not in fc.columns:
                    return "no_latest_bar", details
                try:
                    latest_bar_time = int(fc["bar_time"].iloc[-1])
                except Exception:
                    return "no_latest_bar", details

            if latest_bar_time is None:
                return "no_latest_bar", details
            try:
                latest_bar_time = int(latest_bar_time)
            except Exception:
                return "no_latest_bar", details

            current_bar_time = int(self.last_trade_timestamp // self.base_tf_seconds) * self.base_tf_seconds
            details["latest_bar_time"] = latest_bar_time
            details["current_bar_time"] = current_bar_time

            if latest_bar_time > (current_bar_time - self.base_tf_seconds):
                return "bar_not_closed", details

            if (current_bar_time - latest_bar_time) > (2 * self.base_tf_seconds):
                return "stale_bar", details

        live_price = None
        if not self.dry_run and self._bybit is not None:
            try:
                live_price = self._bybit.get_current_price(self.symbol)
            except Exception:
                live_price = None
        if live_price is not None and live_price > 0:
            live_price = float(live_price)
        details["live_price"] = live_price
        live_ref = live_price if live_price is not None else signal_price
        diff_atr = abs(live_ref - signal_price) / atr if atr > 0 else 0.0
        details["diff_atr"] = diff_atr
        details["max_entry_deviation_atr"] = self._max_entry_price_deviation_atr
        if (
            not self.dry_run
            and atr > 0
            and self._max_entry_price_deviation_atr > 0
            and diff_atr > self._max_entry_price_deviation_atr
        ):
            return "price_deviation", details

        if self.dry_run or self._bybit is None:
            return None, details

        balance = self._bybit.get_available_balance(asset=self.balance_asset, logger=self.logger)
        details["balance"] = balance
        if balance <= 0:
            return "no_balance", details

        stop_dist = (self.stop_loss_atr * atr) + (getattr(self, "stop_padding_pct", 0.0) * signal_price)
        stop_loss = signal_price - (direction * stop_dist)
        risk_per_unit = abs(signal_price - stop_loss)
        details["risk_per_unit"] = risk_per_unit
        if risk_per_unit <= 0:
            return "invalid_risk_per_unit", details

        risk_amount = balance * self.position_size_pct
        qty = risk_amount / risk_per_unit
        qty = self._round_qty(qty)
        details["qty"] = qty
        if qty <= 0:
            return "qty_below_min", details

        required_margin = (qty * signal_price) / float(self.leverage)
        details["required_margin"] = required_margin
        if required_margin > balance:
            scale = balance / required_margin
            scaled_qty = self._round_qty(qty * scale)
            details["scaled_qty"] = scaled_qty
            if scaled_qty <= 0:
                return "insufficient_margin", details

        if isinstance(self._bybit, SimulatedBybitClient) and self._bybit.last_order_error:
            details["order_error"] = self._bybit.last_order_error
            return "order_failed", details

        return None, details


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate LiveFundsTrader against historical trades and compare to full-feature backtest."
    )
    parser.add_argument("--model-dir", type=str, default="models/MONUSDT")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--lookback-days", type=int, default=None)
    parser.add_argument("--initial-capital", type=float, default=10000.0)
    parser.add_argument("--stop-loss-atr", type=float, default=1.0)
    parser.add_argument("--take-profit-rr", type=float, default=1.5)
    parser.add_argument("--min-bounce-prob", type=float, default=0.48)
    parser.add_argument("--max-bounce-prob", type=float, default=1.0)
    parser.add_argument("--touch-threshold-atr", type=float, default=0.3)
    parser.add_argument("--stop-padding-pct", type=float, default=0.0)
    parser.add_argument("--cooldown-bars-after-stop", type=int, default=0)
    parser.add_argument("--trade-side", type=str, default="both", choices=["long", "short", "both"])
    parser.add_argument("--use-dynamic-rr", action="store_true")
    parser.add_argument("--use-calibration", action="store_true")
    parser.add_argument("--warmup-trades", type=int, default=1000)
    parser.add_argument("--max-entry-deviation-atr", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument(
        "--entry-fill-mode",
        type=str,
        default="both",
        choices=["bar_close", "last_trade", "both"],
        help="Entry fill price for the simulated exchange (default: both).",
    )
    parser.add_argument(
        "--ema-touch-mode",
        type=str,
        default="base",
        choices=["base", "multi"],
        help="EMA touch detection mode for live simulation (default: base).",
    )
    parser.add_argument(
        "--dump-touch-diffs",
        action="store_true",
        help="Print EMA touch mismatches between full features and live replay.",
    )
    parser.add_argument(
        "--dump-touch-limit",
        type=int,
        default=50,
        help="Max EMA touch mismatches to print (default: 50).",
    )
    parser.add_argument(
        "--report-file",
        type=str,
        default=None,
        help="Write the comparison report to a file (also prints to console).",
    )
    parser.add_argument(
        "--print-all-trades",
        action="store_true",
        help="Print all trades for the full backtest and live simulations.",
    )
    return parser.parse_args()


def _make_report_printer(
    report_file: Optional[str],
) -> Tuple[Callable, Optional[Callable], Optional[object]]:
    if not report_file:
        return print, None, None

    report_path = Path(report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(report_path, "w", encoding="utf-8")

    def _rprint(*args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=handle)
        handle.flush()

    def _fprint(*args, **kwargs):
        print(*args, **kwargs, file=handle)
        handle.flush()

    return _rprint, _fprint, handle


def _noop_print(*args, **kwargs) -> None:
    return None


def _safe_float(value, default=np.nan) -> float:
    try:
        val = float(value)
    except Exception:
        return default
    if pd.isna(val) or np.isinf(val):
        return default
    return val


def _safe_int(value) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _normalize_touch_tf(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and (pd.isna(value) or np.isinf(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _count_valid_values(values: Iterable) -> int:
    count = 0
    for val in values:
        if val is None:
            continue
        if isinstance(val, float) and (pd.isna(val) or np.isinf(val)):
            continue
        count += 1
    return count


def _count_valid_dict(values: dict) -> Tuple[int, int]:
    if not values:
        return 0, 0
    total = len(values)
    valid = _count_valid_values(values.values())
    return valid, total


def _compute_quality(bounce_prob: float, trend_aligned: bool, is_pullback: bool) -> str:
    if bounce_prob > 0.6 and trend_aligned and is_pullback:
        return "A"
    if bounce_prob > 0.5 and (trend_aligned or is_pullback):
        return "B"
    return "C"


def _evaluate_signal(
    bounce_prob: Optional[float],
    trend_dir: int,
    ema_touch_detected: bool,
    allow_long: bool,
    allow_short: bool,
    min_bounce_prob: float,
    max_bounce_prob: float,
    signal_present: bool = True,
    cooldown_active: bool = False,
    position_open: bool = False,
) -> Tuple[bool, str]:
    if not signal_present:
        return False, "no_signal"
    if position_open:
        return False, "position_open"
    if cooldown_active:
        return False, "cooldown"
    if trend_dir == 0:
        return False, "trend_neutral"
    if trend_dir == 1 and not allow_long:
        return False, "long_disabled"
    if trend_dir == -1 and not allow_short:
        return False, "short_disabled"
    if not ema_touch_detected:
        return False, "no_ema_touch"
    if bounce_prob is None or pd.isna(bounce_prob):
        return False, "no_bounce_prob"
    if bounce_prob < min_bounce_prob:
        return False, "bounce_below_min"
    if bounce_prob > max_bounce_prob:
        return False, "bounce_above_max"
    return True, ""


def _resolve_touch_ema_value(get_value: Callable, tf_name: Optional[str], pullback_ema: int) -> Optional[float]:
    if not tf_name:
        return None
    ema_col = f"{tf_name}_ema_{pullback_ema}"
    ema_val = get_value(ema_col)
    if ema_val is None or (isinstance(ema_val, float) and pd.isna(ema_val)):
        return None
    try:
        return float(ema_val)
    except Exception:
        return None


def _trade_key(trade) -> Tuple:
    entry_time = trade.entry_time
    try:
        entry_time_key = entry_time.isoformat()
    except Exception:
        entry_time_key = str(entry_time)
    return (entry_time_key, int(trade.direction))


def _summarize_backtest(label: str, result, printer: Callable = print) -> None:
    printer(f"{label} RESULTS")
    printer(f"  total_trades: {result.total_trades}")
    printer(f"  win_rate: {result.win_rate:.4f}")
    printer(f"  total_pnl: {result.total_pnl:.2f}")
    printer(f"  total_pnl_percent: {result.total_pnl_percent:.4f}")
    printer(f"  profit_factor: {result.profit_factor:.4f}")
    printer(f"  max_drawdown: {result.max_drawdown:.2f}")
    printer(f"  max_drawdown_percent: {result.max_drawdown_percent:.4f}")
    printer(f"  sharpe_ratio: {result.sharpe_ratio:.4f}")
    printer(f"  avg_win: {result.avg_win:.2f}")
    printer(f"  avg_loss: {result.avg_loss:.2f}")
    printer("")


def _summarize_deltas(label: str, base, other, printer: Callable = print) -> None:
    printer(f"METRIC DELTAS ({label})")
    printer(f"  total_trades: {other.total_trades - base.total_trades}")
    printer(f"  win_rate: {(other.win_rate - base.win_rate):.6f}")
    printer(f"  total_pnl: {(other.total_pnl - base.total_pnl):.2f}")
    printer(f"  total_pnl_percent: {(other.total_pnl_percent - base.total_pnl_percent):.6f}")
    printer(f"  profit_factor: {(other.profit_factor - base.profit_factor):.6f}")
    printer(f"  max_drawdown: {(other.max_drawdown - base.max_drawdown):.2f}")
    printer(f"  max_drawdown_percent: {(other.max_drawdown_percent - base.max_drawdown_percent):.6f}")
    printer(f"  sharpe_ratio: {(other.sharpe_ratio - base.sharpe_ratio):.6f}")
    printer(f"  avg_win: {(other.avg_win - base.avg_win):.2f}")
    printer(f"  avg_loss: {(other.avg_loss - base.avg_loss):.2f}")
    printer("")


def _print_trades(label: str, trades: Iterable, printer: Callable = print) -> None:
    trade_list = list(trades)
    printer(f"{label} TRADES ({len(trade_list)})")
    for trade in trade_list:
        entry_time = trade.entry_time.isoformat() if hasattr(trade.entry_time, "isoformat") else str(trade.entry_time)
        exit_time = trade.exit_time.isoformat() if hasattr(trade.exit_time, "isoformat") else str(trade.exit_time)
        printer(
            f"  {entry_time} dir={trade.direction} entry={trade.entry_price:.8f} "
            f"exit={trade.exit_price:.8f} pnl={trade.pnl:.2f} "
            f"quality={trade.signal_quality} exit_reason={trade.exit_reason} exit_time={exit_time}"
        )
    printer("")


def _build_trade_intervals(trades: Iterable, base_tf_seconds: int) -> List[Tuple[int, int]]:
    intervals: List[Tuple[int, int]] = []
    for trade in trades:
        try:
            entry_ts = int(trade.entry_time.timestamp())
            exit_ts = int(trade.exit_time.timestamp())
        except Exception:
            continue
        entry_bar = int(entry_ts // base_tf_seconds) * base_tf_seconds
        exit_bar = int(exit_ts // base_tf_seconds) * base_tf_seconds
        if exit_bar < entry_bar:
            exit_bar = entry_bar
        intervals.append((entry_bar, exit_bar))
    return intervals


def _position_open_at(bar_time: int, intervals: List[Tuple[int, int]]) -> bool:
    for start, end in intervals:
        if start <= bar_time <= end:
            return True
    return False


def _build_full_bar_metrics(
    test: pd.DataFrame,
    feature_cols: List[str],
    bounce_probs: np.ndarray,
    expected_rrs: np.ndarray,
    cfg,
    args: argparse.Namespace,
    res_full,
    base_tf: str,
) -> dict:
    base_tf_seconds = int(cfg.features.timeframes[cfg.base_timeframe_idx])
    pullback_ema = int(cfg.labels.pullback_ema)
    ema_col = f"{base_tf}_ema_{pullback_ema}"
    atr_col = f"{base_tf}_atr"
    slope_col = f"{base_tf}_ema_{pullback_ema}_slope_norm"

    entry_trade_by_bar = {}
    stop_exit_bars = set()
    for trade in res_full.trades:
        try:
            entry_bar = int(trade.entry_time.timestamp() // base_tf_seconds) * base_tf_seconds
        except Exception:
            continue
        entry_trade_by_bar[entry_bar] = trade
        if trade.exit_reason == "stop_loss":
            try:
                exit_bar = int(trade.exit_time.timestamp() // base_tf_seconds) * base_tf_seconds
            except Exception:
                exit_bar = None
            if exit_bar is not None:
                stop_exit_bars.add(exit_bar)

    position_intervals = _build_trade_intervals(res_full.trades, base_tf_seconds)
    metrics = {}

    cooldown_until = None
    cooldown_bars = int(args.cooldown_bars_after_stop)

    for pos, (_, row) in enumerate(test.iterrows()):
        bar_time = _safe_int(row.get("bar_time"))
        if bar_time is None:
            continue

        if cooldown_until is not None and bar_time >= cooldown_until:
            cooldown_until = None

        cooldown_active = cooldown_until is not None and bar_time < cooldown_until
        if bar_time in stop_exit_bars and cooldown_bars > 0:
            cooldown_until = int(bar_time) + cooldown_bars * base_tf_seconds

        open_price = _safe_float(row.get("open"))
        high = _safe_float(row.get("high"))
        low = _safe_float(row.get("low"))
        close = _safe_float(row.get("close"))
        ema_val = _safe_float(row.get(ema_col))
        atr_val = _safe_float(row.get(atr_col))
        slope_val = _safe_float(row.get(slope_col), default=0.0)
        trend_dir = 1 if slope_val > 0 else -1 if slope_val < 0 else 0

        ema_touch_detected = False
        ema_touch_direction = 0
        ema_touch_dist = np.nan
        ema_touch_tf = None
        ema_touch_quality = np.nan
        ema_touch_slope = np.nan

        if str(args.ema_touch_mode).lower() == "multi" and "ema_touch_detected" in row.index:
            raw_detect = row.get("ema_touch_detected", False)
            if raw_detect is None or pd.isna(raw_detect):
                ema_touch_detected = False
            else:
                ema_touch_detected = bool(raw_detect)
            raw_dir = row.get("ema_touch_direction", 0)
            if raw_dir is None or pd.isna(raw_dir):
                raw_dir = 0
            try:
                raw_dir = int(raw_dir)
            except Exception:
                raw_dir = 0
            if ema_touch_detected and raw_dir not in (0, trend_dir):
                ema_touch_detected = False
            else:
                ema_touch_direction = raw_dir
            ema_touch_dist = _safe_float(row.get("ema_touch_dist"))
            ema_touch_tf = _normalize_touch_tf(row.get("ema_touch_tf"))
            ema_touch_quality = _safe_float(row.get("ema_touch_quality"))
            ema_touch_slope = _safe_float(row.get("ema_touch_slope"))
        else:
            base_touch = _compute_base_tf_touch_from_row(
                row,
                base_tf,
                pullback_ema,
                float(args.touch_threshold_atr),
            )
            ema_touch_detected = bool(base_touch.get("ema_touch_detected", False))
            ema_touch_direction = int(base_touch.get("ema_touch_direction", 0) or 0)
            ema_touch_dist = _safe_float(base_touch.get("ema_touch_dist"))
            if ema_touch_detected:
                ema_touch_tf = base_tf

        ema_touch_ema_value = _resolve_touch_ema_value(lambda key: row.get(key, None), ema_touch_tf, pullback_ema)

        bounce_prob = _safe_float(bounce_probs[pos])
        expected_rr = _safe_float(expected_rrs[pos])

        trend_aligned = trend_dir != 0
        quality = _compute_quality(bounce_prob, trend_aligned, ema_touch_detected)

        features_valid = _count_valid_values((row.get(col) for col in feature_cols if col in row.index))
        features_total = len(feature_cols)
        feature_values = {col: row.get(col) for col in feature_cols}

        position_open = _position_open_at(bar_time, position_intervals)
        signal_pass, skip_reason = _evaluate_signal(
            bounce_prob=bounce_prob,
            trend_dir=trend_dir,
            ema_touch_detected=ema_touch_detected,
            allow_long=bool(args.trade_side in ("long", "both")),
            allow_short=bool(args.trade_side in ("short", "both")),
            min_bounce_prob=float(args.min_bounce_prob),
            max_bounce_prob=float(args.max_bounce_prob),
            signal_present=not pd.isna(bounce_prob),
            cooldown_active=cooldown_active,
            position_open=position_open,
        )

        trade = entry_trade_by_bar.get(bar_time)
        entry_opened = trade is not None
        entry_price = trade.entry_price if trade else None
        stop_loss = trade.stop_loss if trade else None
        take_profit = trade.take_profit if trade else None
        size = trade.size if trade else None

        metrics[bar_time] = {
            "bar_time": bar_time,
            "bar_time_iso": datetime.utcfromtimestamp(int(bar_time)).isoformat(),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "atr": atr_val,
            "ema": ema_val,
            "ema_slope": slope_val,
            "trend_dir": trend_dir,
            "ema_touch_detected": ema_touch_detected,
            "ema_touch_direction": ema_touch_direction,
            "ema_touch_dist": ema_touch_dist,
            "ema_touch_tf": ema_touch_tf,
            "ema_touch_ema_value": ema_touch_ema_value,
            "ema_touch_quality": ema_touch_quality,
            "ema_touch_slope": ema_touch_slope,
            "bounce_prob": bounce_prob,
            "expected_rr": expected_rr,
            "quality": quality,
            "signal_pass": signal_pass,
            "skip_reason": skip_reason,
            "entry_opened": entry_opened,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "size": size,
            "position_open": position_open,
            "cooldown_active": cooldown_active,
            "features_valid": features_valid,
            "features_total": features_total,
            "feature_values": feature_values,
        }

    return metrics


def _capture_live_bar_snapshot(
    trader: SimulatedLiveFundsTrader,
    entry_signal,
    position_open_before: bool,
    position_open_after: bool,
    entry_opened_override: Optional[bool] = None,
) -> Optional[dict]:
    bar_time = trader._last_closed_bar_time
    if bar_time is None:
        return None
    try:
        bar_time = int(bar_time)
    except Exception:
        return None

    base_tf = trader.base_tf
    pullback_ema = int(trader.config.labels.pullback_ema)
    ema_col = f"{base_tf}_ema_{pullback_ema}"
    atr_col = f"{base_tf}_atr"
    slope_col = f"{base_tf}_ema_{pullback_ema}_slope_norm"

    open_price = _safe_float(trader._get_feature_value("open"))
    high = _safe_float(trader._get_feature_value("high"))
    low = _safe_float(trader._get_feature_value("low"))
    close = _safe_float(trader._get_feature_value("close"))
    ema_val = _safe_float(trader._get_feature_value(ema_col))
    atr_val = _safe_float(trader._get_feature_value(atr_col))
    slope_val = _safe_float(trader._get_feature_value(slope_col, default=0.0), default=0.0)
    trend_dir = 1 if slope_val > 0 else -1 if slope_val < 0 else 0

    ema_touch_detected = bool(trader._get_feature_value("ema_touch_detected", False))
    ema_touch_direction = int(trader._get_feature_value("ema_touch_direction", 0) or 0)
    ema_touch_dist = _safe_float(trader._get_feature_value("ema_touch_dist", np.nan))
    ema_touch_tf = _normalize_touch_tf(trader._get_feature_value("ema_touch_tf"))
    ema_touch_quality = _safe_float(trader._get_feature_value("ema_touch_quality", np.nan))
    ema_touch_slope = _safe_float(trader._get_feature_value("ema_touch_slope", np.nan))

    if trader.ema_touch_mode == "base":
        ema_touch_tf = base_tf if ema_touch_detected else None

    ema_touch_ema_value = _resolve_touch_ema_value(trader._get_feature_value, ema_touch_tf, pullback_ema)

    signal_present = entry_signal is not None
    bounce_prob = _safe_float(entry_signal.bounce_prob) if entry_signal else np.nan
    expected_rr = _safe_float(entry_signal.expected_rr) if entry_signal else np.nan
    model_quality = entry_signal.signal_quality if entry_signal else None

    trend_aligned = trend_dir != 0
    quality = _compute_quality(bounce_prob, trend_aligned, ema_touch_detected)

    cooldown_active = False
    if trader.cooldown_bars_after_stop > 0:
        if trader._next_entry_bar_time is not None and bar_time is not None:
            if int(bar_time) < int(trader._next_entry_bar_time):
                cooldown_active = True
        elif trader.cooldown_seconds > 0 and trader.last_stop_time:
            elapsed = (datetime.now() - trader.last_stop_time).total_seconds()
            if elapsed < trader.cooldown_seconds:
                cooldown_active = True

    signal_pass, skip_reason = _evaluate_signal(
        bounce_prob=bounce_prob,
        trend_dir=trend_dir,
        ema_touch_detected=ema_touch_detected,
        allow_long=bool(trader.allow_long),
        allow_short=bool(trader.allow_short),
        min_bounce_prob=float(trader.min_bounce_prob),
        max_bounce_prob=float(trader.max_bounce_prob),
        signal_present=signal_present,
        cooldown_active=cooldown_active,
        position_open=position_open_before,
    )

    if not signal_present:
        if position_open_before:
            skip_reason = "signal_not_calculated_position_open"
        else:
            skip_reason = "signal_not_available"

    entry_opened = position_open_before is False and position_open_after is True
    if entry_opened_override:
        entry_opened = True
    entry_price = None
    stop_loss = None
    take_profit = None
    size = None
    if entry_opened and trader.position is not None:
        entry_price = trader.position.entry_price
        stop_loss = trader.position.stop_loss
        take_profit = trader.position.take_profit
        size = trader.position.size

    features_valid = 0
    features_total = 0
    if trader.use_incremental and trader.predictor.incremental_features:
        features_valid, features_total = _count_valid_dict(trader.predictor.incremental_features)
    elif trader.predictor.features_cache is not None and len(trader.predictor.features_cache) > 0:
        latest = trader.predictor.features_cache.iloc[-1]
        features_total = len(latest)
        features_valid = _count_valid_values(latest.values)

    feature_values = {}
    feature_cols = []
    if trader.predictor is not None:
        feature_cols = list(getattr(trader.predictor, "feature_cols", []))
    if feature_cols:
        if trader.use_incremental and trader.predictor.incremental_features:
            src = trader.predictor.incremental_features
            for col in feature_cols:
                feature_values[col] = src.get(col)
        elif trader.predictor.features_cache is not None and len(trader.predictor.features_cache) > 0:
            latest = trader.predictor.features_cache.iloc[-1]
            for col in feature_cols:
                feature_values[col] = latest.get(col)

    guard_reason = None
    guard_details = {}
    if trader.last_entry_guard_bar_time is not None and int(trader.last_entry_guard_bar_time) == int(bar_time):
        guard_reason = trader.last_entry_guard_reason
        guard_details = trader.last_entry_guard_details or {}

    return {
        "bar_time": bar_time,
        "bar_time_iso": datetime.utcfromtimestamp(int(bar_time)).isoformat(),
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "atr": atr_val,
        "ema": ema_val,
        "ema_slope": slope_val,
        "trend_dir": trend_dir,
        "ema_touch_detected": ema_touch_detected,
        "ema_touch_direction": ema_touch_direction,
        "ema_touch_dist": ema_touch_dist,
        "ema_touch_tf": ema_touch_tf,
        "ema_touch_ema_value": ema_touch_ema_value,
        "ema_touch_quality": ema_touch_quality,
        "ema_touch_slope": ema_touch_slope,
        "bounce_prob": bounce_prob,
        "expected_rr": expected_rr,
        "quality": quality,
        "model_quality": model_quality,
        "signal_pass": signal_pass,
        "skip_reason": skip_reason,
        "entry_opened": entry_opened,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "size": size,
        "position_open": position_open_after,
        "cooldown_active": cooldown_active,
        "features_valid": features_valid,
        "features_total": features_total,
        "feature_values": feature_values,
        "entry_guard_reason": guard_reason,
        "entry_guard_signal_price": guard_details.get("signal_price"),
        "entry_guard_live_price": guard_details.get("live_price"),
        "entry_guard_diff_atr": guard_details.get("diff_atr"),
        "entry_guard_max_diff_atr": guard_details.get("max_entry_deviation_atr"),
        "entry_guard_balance": guard_details.get("balance"),
        "entry_guard_qty": guard_details.get("qty"),
        "entry_guard_required_margin": guard_details.get("required_margin"),
        "entry_guard_scaled_qty": guard_details.get("scaled_qty"),
        "entry_guard_order_error": guard_details.get("order_error"),
        "entry_guard_latest_bar_time": guard_details.get("latest_bar_time"),
        "entry_guard_current_bar_time": guard_details.get("current_bar_time"),
    }


def _capture_live_feature_snapshot(
    trader: SimulatedLiveFundsTrader,
    snapshot_time: int,
    min_bar_time: Optional[int] = None,
    max_bar_time: Optional[int] = None,
) -> Optional[dict]:
    try:
        snapshot_time = int(snapshot_time)
    except Exception:
        return None

    feature_cols = []
    if trader.predictor is not None:
        feature_cols = list(getattr(trader.predictor, "feature_cols", []))
    if not feature_cols:
        return None

    feature_values = {}
    if trader.use_incremental and trader.predictor.incremental_features:
        src = trader.predictor.incremental_features
        for col in feature_cols:
            feature_values[col] = src.get(col)
    elif trader.predictor.features_cache is not None and len(trader.predictor.features_cache) > 0:
        latest = trader.predictor.features_cache.iloc[-1]
        for col in feature_cols:
            feature_values[col] = latest.get(col)
    else:
        return None

    bar_time = trader._last_closed_bar_time
    bar_time_iso = ""
    if bar_time is not None:
        try:
            bar_time = int(bar_time)
            bar_time_iso = datetime.utcfromtimestamp(int(bar_time)).isoformat()
        except Exception:
            bar_time = None

    if bar_time is not None:
        if min_bar_time is not None and int(bar_time) < int(min_bar_time):
            return None
        if max_bar_time is not None and int(bar_time) >= int(max_bar_time):
            return None

    features_valid = _count_valid_values(feature_values.values())
    features_total = len(feature_values)

    return {
        "snapshot_time": snapshot_time,
        "snapshot_time_iso": datetime.utcfromtimestamp(int(snapshot_time)).isoformat(),
        "bar_time": bar_time,
        "bar_time_iso": bar_time_iso,
        "features_valid": features_valid,
        "features_total": features_total,
        "feature_values": feature_values,
    }


def _write_bar_comparison_csv(
    handle,
    title: str,
    base_tf: str,
    pullback_ema: int,
    full_metrics: dict,
    live_metrics: dict,
    diff_only: bool = False,
) -> None:
    if handle is None:
        return

    fields = [
        "open",
        "high",
        "low",
        "close",
        "atr",
        "ema",
        "ema_slope",
        "trend_dir",
        "ema_touch_detected",
        "ema_touch_direction",
        "ema_touch_dist",
        "ema_touch_tf",
        "ema_touch_ema_value",
        "ema_touch_quality",
        "ema_touch_slope",
        "bounce_prob",
        "expected_rr",
        "quality",
        "model_quality",
        "signal_pass",
        "entry_opened",
        "entry_price",
        "stop_loss",
        "take_profit",
        "size",
        "skip_reason",
        "position_open",
        "cooldown_active",
        "features_valid",
        "features_total",
        "entry_guard_reason",
        "entry_guard_signal_price",
        "entry_guard_live_price",
        "entry_guard_diff_atr",
        "entry_guard_max_diff_atr",
        "entry_guard_balance",
        "entry_guard_qty",
        "entry_guard_required_margin",
        "entry_guard_scaled_qty",
        "entry_guard_order_error",
        "entry_guard_latest_bar_time",
        "entry_guard_current_bar_time",
    ]

    keys = sorted(set(full_metrics.keys()) | set(live_metrics.keys()))
    handle.write(f"{title}\n")
    writer = csv.writer(handle)
    header = ["bar_time", "bar_time_iso", "base_tf", "ema_period"]
    header += [f"full_{f}" for f in fields]
    header += [f"live_{f}" for f in fields]
    writer.writerow(header)

    def has_diff(full_row: dict, live_row: dict) -> bool:
        if not full_row or not live_row:
            return True
        key_checks = [
            "ema_touch_detected",
            "ema_touch_direction",
            "ema_touch_tf",
            "bounce_prob",
            "entry_opened",
            "signal_pass",
        ]
        for key in key_checks:
            fval = full_row.get(key)
            lval = live_row.get(key)
            if isinstance(fval, float) and isinstance(lval, float):
                if pd.isna(fval) and pd.isna(lval):
                    continue
                if abs(float(fval) - float(lval)) > 1e-6:
                    return True
            else:
                if fval != lval:
                    return True
        return False

    for bar_time in keys:
        full_row = full_metrics.get(bar_time, {})
        live_row = live_metrics.get(bar_time, {})
        if diff_only and not has_diff(full_row, live_row):
            continue
        bar_time_iso = full_row.get("bar_time_iso") or live_row.get("bar_time_iso") or ""
        row = [bar_time, bar_time_iso, base_tf, pullback_ema]
        row += [full_row.get(field) for field in fields]
        row += [live_row.get(field) for field in fields]
        writer.writerow(row)

    handle.write("\n")


def _entry_bar_map(trades: Iterable, base_tf_seconds: int) -> dict:
    entries = {}
    for trade in trades:
        try:
            entry_ts = int(trade.entry_time.timestamp())
        except Exception:
            continue
        entry_bar = int(entry_ts // base_tf_seconds) * base_tf_seconds
        if entry_bar not in entries:
            entries[entry_bar] = trade
    return entries


def _write_entry_feature_diffs_csv(
    handle,
    title: str,
    feature_cols: List[str],
    full_metrics: dict,
    live_metrics: dict,
    full_trades: Iterable,
    live_trades: Iterable,
    base_tf_seconds: int,
) -> None:
    if handle is None:
        return

    full_entries = _entry_bar_map(full_trades, base_tf_seconds)
    live_entries = _entry_bar_map(live_trades, base_tf_seconds)
    entry_bars = sorted(set(full_entries.keys()) | set(live_entries.keys()))

    handle.write(f"{title}\n")
    writer = csv.writer(handle)
    header = [
        "bar_time",
        "bar_time_iso",
        "entry_source",
        "full_entry_time",
        "live_entry_time",
        "feature",
        "full_value",
        "live_value",
        "diff",
    ]
    writer.writerow(header)

    for bar_time in entry_bars:
        full_row = full_metrics.get(bar_time, {})
        live_row = live_metrics.get(bar_time, {})
        full_features = full_row.get("feature_values") or {}
        live_features = live_row.get("feature_values") or {}

        bar_time_iso = full_row.get("bar_time_iso") or live_row.get("bar_time_iso") or ""
        entry_source = "both" if bar_time in full_entries and bar_time in live_entries else "full"
        if bar_time not in full_entries and bar_time in live_entries:
            entry_source = "live"

        full_entry_time = ""
        if bar_time in full_entries:
            try:
                full_entry_time = full_entries[bar_time].entry_time.isoformat()
            except Exception:
                full_entry_time = str(full_entries[bar_time].entry_time)

        live_entry_time = ""
        if bar_time in live_entries:
            try:
                live_entry_time = live_entries[bar_time].entry_time.isoformat()
            except Exception:
                live_entry_time = str(live_entries[bar_time].entry_time)

        for feature in feature_cols:
            fval = full_features.get(feature)
            lval = live_features.get(feature)
            fnum = _safe_float(fval)
            lnum = _safe_float(lval)
            diff = ""
            if fnum is not None and lnum is not None and not pd.isna(fnum) and not pd.isna(lnum):
                diff = lnum - fnum
            writer.writerow([
                bar_time,
                bar_time_iso,
                entry_source,
                full_entry_time,
                live_entry_time,
                feature,
                fval,
                lval,
                diff,
            ])

    handle.write("\n")


def _write_feature_snapshot_10m_csv(
    handle,
    title: str,
    feature_cols: List[str],
    full_metrics: dict,
    live_snapshots: Optional[dict],
) -> None:
    if handle is None:
        return
    if not live_snapshots:
        return

    handle.write(f"{title}\n")
    writer = csv.writer(handle)
    header = [
        "snapshot_time",
        "snapshot_time_iso",
        "bar_time",
        "bar_time_iso",
        "feature",
        "full_value",
        "live_value",
        "diff",
    ]
    writer.writerow(header)

    for snapshot_time in sorted(live_snapshots.keys()):
        snapshot = live_snapshots[snapshot_time]
        bar_time = snapshot.get("bar_time")
        bar_time_iso = snapshot.get("bar_time_iso") or ""
        snapshot_iso = snapshot.get("snapshot_time_iso") or ""

        full_row = full_metrics.get(bar_time, {}) if bar_time is not None else {}
        full_features = full_row.get("feature_values") or {}
        live_features = snapshot.get("feature_values") or {}

        for feature in feature_cols:
            fval = full_features.get(feature)
            lval = live_features.get(feature)
            fnum = _safe_float(fval)
            lnum = _safe_float(lval)
            diff = ""
            if fnum is not None and lnum is not None and not pd.isna(fnum) and not pd.isna(lnum):
                diff = lnum - fnum
            writer.writerow([
                snapshot_time,
                snapshot_iso,
                bar_time,
                bar_time_iso,
                feature,
                fval,
                lval,
                diff,
            ])

    handle.write("\n")


def _write_trades_csv(
    handle,
    title: str,
    trades: Iterable,
    extra_by_bar_time: Optional[dict] = None,
    base_tf_seconds: int = 0,
) -> None:
    if handle is None:
        return
    handle.write(f"{title}\n")
    writer = csv.writer(handle)
    header = [
        "entry_time",
        "exit_time",
        "direction",
        "entry_price",
        "exit_price",
        "size",
        "pnl",
        "pnl_percent",
        "signal_quality",
        "exit_reason",
        "stop_loss",
        "take_profit",
        "trend_prob",
        "bounce_prob",
        "is_pullback",
        "trend_aligned",
        "dist_from_ema",
        "entry_bar_time",
        "extra_bounce_prob",
        "extra_expected_rr",
        "extra_quality",
    ]
    writer.writerow(header)
    for trade in trades:
        entry_time = trade.entry_time.isoformat() if hasattr(trade.entry_time, "isoformat") else str(trade.entry_time)
        exit_time = trade.exit_time.isoformat() if hasattr(trade.exit_time, "isoformat") else str(trade.exit_time)
        entry_bar_time = ""
        extra_bounce = ""
        extra_expected_rr = ""
        extra_quality = ""
        if base_tf_seconds > 0:
            try:
                entry_bar_time = int(trade.entry_time.timestamp() // base_tf_seconds) * base_tf_seconds
            except Exception:
                entry_bar_time = ""
        if extra_by_bar_time is not None and entry_bar_time in extra_by_bar_time:
            extra_bounce = extra_by_bar_time[entry_bar_time].get("bounce_prob")
            extra_expected_rr = extra_by_bar_time[entry_bar_time].get("expected_rr")
            extra_quality = extra_by_bar_time[entry_bar_time].get("quality")
        writer.writerow(
            [
                entry_time,
                exit_time,
                trade.direction,
                trade.entry_price,
                trade.exit_price,
                trade.size,
                trade.pnl,
                trade.pnl_percent,
                trade.signal_quality,
                trade.exit_reason,
                getattr(trade, "stop_loss", None),
                getattr(trade, "take_profit", None),
                getattr(trade, "trend_prob", None),
                getattr(trade, "bounce_prob", None),
                getattr(trade, "is_pullback", None),
                getattr(trade, "trend_aligned", None),
                getattr(trade, "dist_from_ema", None),
                entry_bar_time,
                extra_bounce,
                extra_expected_rr,
                extra_quality,
            ]
        )
    handle.write("\n")


def _predict_entry_probs(
    data: pd.DataFrame,
    feature_cols: List[str],
    models: TrendFollowerModels,
    use_calibration: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    feature_data = {}
    rows = len(data)
    for col in feature_cols:
        if col in data.columns:
            feature_data[col] = data[col].fillna(0).values
        else:
            feature_data[col] = np.zeros(rows)
    X = pd.DataFrame(feature_data, index=data.index)
    entry_pred = models.entry_model.predict(X, use_calibration=use_calibration)
    bounce_probs = np.array(entry_pred.get("bounce_prob", np.zeros(rows)), dtype=float)
    expected_rr = entry_pred.get("expected_rr", np.full(rows, np.nan))
    expected_rrs = np.array(expected_rr, dtype=float)
    return bounce_probs, expected_rrs


def _compute_base_tf_touch_from_row(
    row,
    base_tf: str,
    pullback_ema: int,
    touch_threshold_atr: float,
) -> dict:
    ema_col = f'{base_tf}_ema_{pullback_ema}'
    slope_col = f'{base_tf}_ema_{pullback_ema}_slope_norm'
    atr_col = f'{base_tf}_atr'

    ema = row.get(ema_col, None)
    atr = row.get(atr_col, None)
    slope_val = row.get(slope_col, 0.0)
    bar_high = row.get('high', None)
    bar_low = row.get('low', None)
    bar_close = row.get('close', None)

    if (
        ema is None
        or atr is None
        or bar_high is None
        or bar_low is None
        or bar_close is None
        or pd.isna(ema)
        or pd.isna(atr)
        or pd.isna(bar_high)
        or pd.isna(bar_low)
        or pd.isna(bar_close)
        or atr <= 0
    ):
        return {"ema_touch_detected": False, "ema_touch_direction": 0, "ema_touch_dist": None}

    if slope_val is None or pd.isna(slope_val):
        slope_val = 0.0
    trend_dir = 1 if slope_val > 0 else -1 if slope_val < 0 else 0
    if trend_dir == 0:
        return {"ema_touch_detected": False, "ema_touch_direction": 0, "ema_touch_dist": None}

    mid_bar = (bar_high + bar_low) / 2.0
    ema_touched = False
    touch_dist = None
    if trend_dir == 1:
        dist_low = (bar_low - ema) / atr
        if -touch_threshold_atr <= dist_low <= touch_threshold_atr and (bar_close >= ema or mid_bar >= ema):
            ema_touched = True
            touch_dist = dist_low
    else:
        dist_high = (bar_high - ema) / atr
        if -touch_threshold_atr <= dist_high <= touch_threshold_atr and (bar_close <= ema or mid_bar <= ema):
            ema_touched = True
            touch_dist = dist_high

    return {
        "ema_touch_detected": ema_touched,
        "ema_touch_direction": trend_dir if ema_touched else 0,
        "ema_touch_dist": touch_dist,
    }


def _summarize_touch_diffs(
    label: str,
    full_touch: dict,
    live_touch: dict,
    printer: Callable = print,
    limit: int = 50,
) -> None:
    full_keys = sorted(full_touch.keys())
    missing_live = [k for k in full_keys if k not in live_touch]
    mismatches = []

    for bar_time in full_keys:
        live = live_touch.get(bar_time)
        if live is None:
            continue
        full = full_touch[bar_time]
        full_detect = bool(full.get("ema_touch_detected", False))
        live_detect = bool(live.get("ema_touch_detected", False))
        full_dir = int(full.get("ema_touch_direction", 0) or 0)
        live_dir = int(live.get("ema_touch_direction", 0) or 0)

        mismatch = full_detect != live_detect
        if not mismatch and full_detect and live_detect and full_dir != live_dir:
            mismatch = True

        if mismatch:
            mismatches.append((bar_time, full, live))

    printer(f"{label} TOUCH DIFFS")
    printer(f"  full_bars: {len(full_keys)}")
    printer(f"  live_bars: {len(live_touch)}")
    printer(f"  missing_live: {len(missing_live)}")
    printer(f"  mismatches: {len(mismatches)}")
    printer("")

    limit = max(0, int(limit))
    if limit and mismatches:
        printer(f"{label} TOUCH MISMATCH DETAILS (limit {limit})")
        for bar_time, full, live in mismatches[:limit]:
            ts = datetime.utcfromtimestamp(int(bar_time)).isoformat()
            printer(
                f"  {ts} | full: det={full.get('ema_touch_detected')} "
                f"dir={full.get('ema_touch_direction')} dist={full.get('ema_touch_dist'):.4f} | "
                f"live: det={live.get('ema_touch_detected')} "
                f"dir={live.get('ema_touch_direction')} dist={live.get('ema_touch_dist')}"
            )

            # Print diagnostic info if available (for incremental mode)
            diag = live.get("diagnostic")
            batch_tf_values = full.get("batch_tf_values", {})

            if diag:
                config = diag.get("config", {})
                bar_info = diag.get("bar_info", {})
                atr = diag.get("atr", "N/A")
                skip_reason = diag.get("skip_reason")

                printer(f"    [DIAGNOSTIC] threshold_atr={config.get('touch_threshold_atr')}, min_slope={config.get('min_slope_norm')}, atr={atr}")
                printer(f"    [DIAGNOSTIC] bar: O={bar_info.get('open')}, H={bar_info.get('high')}, L={bar_info.get('low')}, C={bar_info.get('close')}")

                if skip_reason:
                    printer(f"    [DIAGNOSTIC] Skip reason: {skip_reason}")

                checked_tfs = diag.get("checked_tfs", [])
                if checked_tfs:
                    printer("    [DIAGNOSTIC] TF checks (INCR=incremental, BATCH=batch):")
                    for tf_info in checked_tfs:
                        tf_name = tf_info.get("tf", "?")
                        slope_val = tf_info.get("slope_val", "N/A")
                        min_slope = tf_info.get("min_slope", "N/A")
                        threshold = tf_info.get("threshold", "N/A")
                        decision = tf_info.get("decision", "?")
                        reason = tf_info.get("reason", "?")
                        dist = tf_info.get("dist", "N/A")
                        setup = tf_info.get("setup", "?")
                        ema_val = tf_info.get("ema_val", "N/A")

                        # Get batch values for comparison
                        batch_vals = batch_tf_values.get(tf_name, {})
                        batch_ema = batch_vals.get("ema", "N/A")
                        batch_slope = batch_vals.get("slope_norm", "N/A")
                        batch_atr = batch_vals.get("atr", "N/A")

                        if isinstance(slope_val, float) and not np.isnan(slope_val):
                            slope_str = f"{slope_val:.6f}"
                        else:
                            slope_str = str(slope_val)

                        if isinstance(dist, float) and not np.isnan(dist):
                            dist_str = f"{dist:.4f}"
                        else:
                            dist_str = str(dist)

                        printer(
                            f"      {tf_name}: slope={slope_str}, |slope|>{min_slope}? -> setup={setup}, "
                            f"dist={dist_str} in [-{threshold},{threshold}]? -> {decision} ({reason})"
                        )

                        # Print comparison of incremental vs batch values
                        if ema_val is not None or batch_ema is not None:
                            ema_incr_str = f"{ema_val:.8f}" if isinstance(ema_val, float) and not np.isnan(ema_val) else str(ema_val)
                            ema_batch_str = f"{batch_ema:.8f}" if isinstance(batch_ema, float) and not np.isnan(batch_ema) else str(batch_ema)
                            printer(f"        EMA: INCR={ema_incr_str} | BATCH={ema_batch_str}")

                            # Check for EMA divergence
                            if isinstance(ema_val, float) and isinstance(batch_ema, float) and not np.isnan(ema_val) and not np.isnan(batch_ema):
                                ema_diff = abs(ema_val - batch_ema)
                                if ema_diff > 1e-8:
                                    printer(f"        *** EMA DIVERGENCE: diff={ema_diff:.10f}")

                        if slope_val is not None or batch_slope is not None:
                            slope_incr_str = f"{slope_val:.8f}" if isinstance(slope_val, float) and not np.isnan(slope_val) else str(slope_val)
                            slope_batch_str = f"{batch_slope:.8f}" if isinstance(batch_slope, float) and not np.isnan(batch_slope) else str(batch_slope)
                            printer(f"        SLOPE_NORM: INCR={slope_incr_str} | BATCH={slope_batch_str}")

                            # Check for slope divergence
                            if isinstance(slope_val, float) and isinstance(batch_slope, float) and not np.isnan(slope_val) and not np.isnan(batch_slope):
                                slope_diff = abs(slope_val - batch_slope)
                                if slope_diff > 1e-8:
                                    printer(f"        *** SLOPE_NORM DIVERGENCE: diff={slope_diff:.10f}")

                                    # Check if divergence crosses the threshold
                                    if isinstance(min_slope, float):
                                        incr_passes = abs(slope_val) > min_slope
                                        batch_passes = abs(batch_slope) > min_slope
                                        if incr_passes != batch_passes:
                                            printer(f"        *** THRESHOLD MISMATCH: incr_passes={incr_passes}, batch_passes={batch_passes}")

        printer("")

def _summarize_trade_overlap(
    label: str,
    left_name: str,
    left_trades: Iterable,
    right_name: str,
    right_trades: Iterable,
    printer: Callable = print,
    show_only: bool = False,
) -> None:
    left_by_key = {_trade_key(t): t for t in left_trades}
    right_by_key = {_trade_key(t): t for t in right_trades}

    common_keys = sorted(set(left_by_key) & set(right_by_key))
    only_left_keys = sorted(set(left_by_key) - set(right_by_key))
    only_right_keys = sorted(set(right_by_key) - set(left_by_key))

    printer(f"{label} TRADE OVERLAP")
    printer(f"  common: {len(common_keys)}")
    printer(f"  only_{left_name}: {len(only_left_keys)}")
    printer(f"  only_{right_name}: {len(only_right_keys)}")
    printer("")

    if common_keys:
        quality_mismatch = 0
        exit_reason_mismatch = 0
        pnl_diffs = []
        for key in common_keys:
            left = left_by_key[key]
            right = right_by_key[key]
            if left.signal_quality != right.signal_quality:
                quality_mismatch += 1
            if left.exit_reason != right.exit_reason:
                exit_reason_mismatch += 1
            pnl_diffs.append(abs(left.pnl - right.pnl))

        diffs = np.array(pnl_diffs) if pnl_diffs else np.array([0.0])
        printer("OVERLAPPING TRADE DIFFS")
        printer(f"  pnl max_abs_diff: {float(diffs.max()):.6f}")
        printer(f"  pnl mean_abs_diff: {float(diffs.mean()):.6f}")
        printer(f"  pnl median_abs_diff: {float(np.median(diffs)):.6f}")
        printer(f"  pnl p95_abs_diff: {float(np.percentile(diffs, 95)):.6f}")
        printer(f"  quality_mismatch: {quality_mismatch}")
        printer(f"  exit_reason_mismatch: {exit_reason_mismatch}")
        printer("")

    if show_only:
        if only_left_keys:
            _print_trades(f"ONLY {left_name.upper()}", [left_by_key[k] for k in only_left_keys], printer=printer)
        if only_right_keys:
            _print_trades(f"ONLY {right_name.upper()}", [right_by_key[k] for k in only_right_keys], printer=printer)


def _build_sim_result(trader: SimulatedLiveFundsTrader) -> SimResult:
    result = SimResult()
    result.trades = list(trader.completed_trades)

    result.total_trades = len(result.trades)
    result.winning_trades = sum(1 for t in result.trades if t.pnl > 0)
    result.losing_trades = sum(1 for t in result.trades if t.pnl <= 0)
    result.win_rate = result.winning_trades / result.total_trades if result.total_trades > 0 else 0.0

    result.total_pnl = sum(t.pnl for t in result.trades)
    if trader.initial_capital:
        result.total_pnl_percent = (trader.capital - trader.initial_capital) / trader.initial_capital * 100

    wins = [t.pnl for t in result.trades if t.pnl > 0]
    losses = [t.pnl for t in result.trades if t.pnl <= 0]
    result.avg_win = float(np.mean(wins)) if wins else 0.0
    result.avg_loss = float(np.mean(losses)) if losses else 0.0

    total_wins = sum(wins)
    total_losses = abs(sum(losses))
    result.profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

    equity = [trader.initial_capital]
    for trade in result.trades:
        equity.append(equity[-1] + trade.pnl)
    result.equity_curve = equity

    if equity:
        equity_arr = np.array(equity)
        peak = np.maximum.accumulate(equity_arr)
        drawdown = equity_arr - peak
        result.max_drawdown = abs(min(drawdown))
        result.max_drawdown_percent = result.max_drawdown / max(peak) * 100 if max(peak) > 0 else 0.0

    returns = np.diff(np.array(equity)) / np.array(equity[:-1]) if len(equity) > 1 else np.array([])
    if len(returns) > 1 and np.std(returns) > 0:
        result.sharpe_ratio = float(np.mean(returns) / np.std(returns) * np.sqrt(252))

    return result


def _run_trade_replay(
    trader: SimulatedLiveFundsTrader,
    trades: pd.DataFrame,
    cfg,
    batch_size: int,
    printer: Callable = print,
    touch_snapshot: Optional[dict] = None,
    bar_diagnostics: Optional[dict] = None,
    feature_snapshots: Optional[dict] = None,
) -> SimResult:
    ts_col = cfg.data.timestamp_col
    price_col = cfg.data.price_col
    size_col = cfg.data.size_col
    side_col = cfg.data.side_col
    tick_col = cfg.data.tick_direction_col

    base_tf_seconds = trader.base_tf_seconds
    current_bar_time = None
    batch = []
    trade_count = 0
    total_trades = len(trades)
    last_touch_bar_time = None
    ten_min_seconds = 600
    last_snapshot_time = None

    def record_touch() -> None:
        nonlocal last_touch_bar_time
        if touch_snapshot is None:
            return
        bar_time = trader._last_closed_bar_time
        if bar_time is None:
            return
        try:
            bar_time = int(bar_time)
        except Exception:
            return
        if last_touch_bar_time == bar_time:
            return
        last_touch_bar_time = bar_time

        # Capture basic touch info
        touch_data = {
            "ema_touch_detected": bool(trader._get_feature_value("ema_touch_detected", False)),
            "ema_touch_direction": int(trader._get_feature_value("ema_touch_direction", 0) or 0),
            "ema_touch_dist": trader._get_feature_value("ema_touch_dist", np.nan),
        }

        # Capture diagnostic info from incremental engine if available
        if trader.use_incremental and trader.predictor and trader.predictor.incremental_engine:
            diag = getattr(trader.predictor.incremental_engine, "last_touch_diagnostic", None)
            if diag is not None:
                touch_data["diagnostic"] = diag

        touch_snapshot[bar_time] = touch_data

    def record_bar_snapshot(pos_before: bool, pos_after: bool, opened_this_tick: bool) -> None:
        if bar_diagnostics is None:
            return
        entry_signal = None
        if not pos_before or opened_this_tick:
            entry_signal = trader.predictor.last_entry_signal
        snapshot = _capture_live_bar_snapshot(
            trader,
            entry_signal,
            position_open_before=pos_before,
            position_open_after=pos_after,
            entry_opened_override=opened_this_tick,
        )
        if snapshot is None:
            return
        bar_diagnostics[snapshot["bar_time"]] = snapshot

    for row in trades.itertuples(index=False):
        ts = float(getattr(row, ts_col))
        trade_bar_time = int(ts // base_tf_seconds) * base_tf_seconds

        if current_bar_time is None:
            current_bar_time = trade_bar_time

        if trade_bar_time != current_bar_time:
            if batch:
                trader._handle_trade_message({"data": batch})
                batch = []
            if trade_count >= trader.warmup_trades:
                positions_opened_before = trader.stats.positions_opened
                pos_before = trader.position is not None
                trader._trading_tick()
                pos_after = trader.position is not None
                opened_this_tick = trader.stats.positions_opened > positions_opened_before
                record_touch()
                record_bar_snapshot(pos_before, pos_after, opened_this_tick)
            current_bar_time = trade_bar_time

        price = float(getattr(row, price_col))
        size = float(getattr(row, size_col))
        side = getattr(row, side_col, "Buy") or "Buy"
        tick_dir = getattr(row, tick_col, "ZeroPlusTick") or "ZeroPlusTick"
        symbol = getattr(row, "symbol", trader.symbol)

        if trader._bybit:
            trader._bybit.update_trade(price, ts)

        batch.append({"T": int(ts * 1000), "s": symbol, "S": side, "v": size, "p": price, "L": tick_dir})
        trade_count += 1

        if len(batch) >= batch_size:
            trader._handle_trade_message({"data": batch})
            batch = []

        if feature_snapshots is not None:
            snap_time = int(ts // ten_min_seconds) * ten_min_seconds
            if last_snapshot_time is None or snap_time > last_snapshot_time:
                snapshot = _capture_live_feature_snapshot(
                    trader,
                    snap_time,
                    min_bar_time=trader.test_start_bar_time,
                    max_bar_time=trader.test_end_bar_time,
                )
                if snapshot is not None:
                    feature_snapshots[snapshot["snapshot_time"]] = snapshot
                    last_snapshot_time = snapshot["snapshot_time"]

        if trade_count > 0 and trade_count % 250000 == 0:
            printer(f"  Processed {trade_count:,}/{total_trades:,} trades...")

    if batch:
        trader._handle_trade_message({"data": batch})
    if trade_count >= trader.warmup_trades:
        positions_opened_before = trader.stats.positions_opened
        pos_before = trader.position is not None
        trader._trading_tick()
        pos_after = trader.position is not None
        opened_this_tick = trader.stats.positions_opened > positions_opened_before
        record_touch()
        record_bar_snapshot(pos_before, pos_after, opened_this_tick)

    return _build_sim_result(trader)


def main() -> None:
    args = _parse_args()
    rprint, fprint, report_handle = _make_report_printer(args.report_file)
    if fprint is None:
        fprint = _noop_print
    if report_handle is not None:
        import atexit
        atexit.register(report_handle.close)
    model_dir = Path(args.model_dir)

    cfg = _load_train_config(model_dir) or DEFAULT_CONFIG
    cfg.model.model_dir = model_dir

    if args.data_dir:
        cfg.data.data_dir = Path(args.data_dir)
    if args.lookback_days is not None:
        cfg.data.lookback_days = int(args.lookback_days)
    cfg.labels.touch_threshold_atr = float(args.touch_threshold_atr)

    base_tf = cfg.features.timeframe_names[cfg.base_timeframe_idx]

    rprint(f"Model dir: {model_dir}")
    rprint(f"Data dir: {cfg.data.data_dir}")
    rprint(f"Base TF: {base_tf}")
    rprint(f"Lookback days: {cfg.data.lookback_days}")
    rprint("")

    rprint("Loading models...")
    models = TrendFollowerModels(cfg.model)
    models.load_all(model_dir)

    rprint("Loading trades...")
    raw_trades = load_trades(cfg.data, verbose=False)
    raw_trades = raw_trades.sort_values(cfg.data.timestamp_col).reset_index(drop=True)
    rprint(f"Trades loaded: {len(raw_trades):,}")

    rprint("Creating bars for full-feature backtest...")
    processed = preprocess_trades(raw_trades, cfg.data)
    bars = create_multi_timeframe_bars(
        processed,
        cfg.features.timeframes,
        cfg.features.timeframe_names,
        cfg.data,
    )

    rprint("Calculating full features...")
    featured = calculate_multi_timeframe_features(bars, base_tf, cfg.features)

    rprint("Labeling to get test split...")
    labeled, feature_cols = create_training_dataset(featured, cfg.labels, cfg.features, base_tf)

    start = int(len(labeled) * (cfg.model.train_ratio + cfg.model.val_ratio))
    test = labeled.iloc[start:].copy()
    rprint(f"Test bars: {len(test):,} (start index {start})")
    rprint(f"Feature count: {len(feature_cols)}")
    rprint("")

    test_start_bar_time = int(test["bar_time"].iloc[0]) if len(test) else None
    test_end_bar_time = int(test["bar_time"].iloc[-1]) if len(test) else None

    full_touch_by_time = {}
    if args.dump_touch_diffs and len(test):
        # Get EMA period and TF names for diagnostic capture
        pullback_ema = int(cfg.labels.pullback_ema)
        tf_names = cfg.features.timeframe_names

        if str(args.ema_touch_mode).lower() == "multi":
            for _, row in test.iterrows():
                bar_time = int(row["bar_time"])
                raw_detect = row.get("ema_touch_detected", False)
                if raw_detect is None or pd.isna(raw_detect):
                    raw_detect = False
                raw_dir = row.get("ema_touch_direction", 0)
                if raw_dir is None or pd.isna(raw_dir):
                    raw_dir = 0
                try:
                    raw_dir = int(raw_dir)
                except Exception:
                    raw_dir = 0
                raw_dist = row.get("ema_touch_dist", np.nan)

                # Capture batch feature values for each TF (for diagnostic comparison)
                batch_tf_values = {}
                for tf_name in tf_names:
                    ema_key = f"{tf_name}_ema_{pullback_ema}"
                    slope_key = f"{tf_name}_ema_{pullback_ema}_slope_norm"
                    atr_key = f"{tf_name}_atr"
                    batch_tf_values[tf_name] = {
                        "ema": row.get(ema_key),
                        "slope_norm": row.get(slope_key),
                        "atr": row.get(atr_key),
                    }

                full_touch_by_time[bar_time] = {
                    "ema_touch_detected": bool(raw_detect),
                    "ema_touch_direction": raw_dir,
                    "ema_touch_dist": raw_dist,
                    "batch_tf_values": batch_tf_values,
                }
        else:
            for _, row in test.iterrows():
                bar_time = int(row["bar_time"])
                full_touch_by_time[bar_time] = _compute_base_tf_touch_from_row(
                    row,
                    base_tf,
                    int(cfg.labels.pullback_ema),
                    float(args.touch_threshold_atr),
                )

    backtest_kwargs = dict(
        min_bounce_prob=float(args.min_bounce_prob),
        max_bounce_prob=float(args.max_bounce_prob),
        min_quality=getattr(cfg, "min_quality", "B"),
        stop_loss_atr=float(args.stop_loss_atr),
        stop_padding_pct=float(args.stop_padding_pct),
        take_profit_rr=float(args.take_profit_rr),
        cooldown_bars_after_stop=int(args.cooldown_bars_after_stop),
        trade_side=str(args.trade_side),
        use_dynamic_rr=bool(args.use_dynamic_rr),
        use_ema_touch_entry=True,
        ema_touch_mode=str(args.ema_touch_mode),
        touch_threshold_atr=float(args.touch_threshold_atr),
        raw_trades=processed,
        use_calibration=bool(args.use_calibration),
    )

    def run_backtest(data: pd.DataFrame):
        bt = SimpleBacktester(models, cfg, **backtest_kwargs)
        return bt.run(data, feature_cols)

    rprint("Running backtest (full features)...")
    res_full = run_backtest(test)
    full_bar_metrics = {}
    if report_handle is not None and len(test) > 0:
        bounce_probs, expected_rrs = _predict_entry_probs(
            test,
            feature_cols,
            models,
            use_calibration=bool(args.use_calibration),
        )
        full_bar_metrics = _build_full_bar_metrics(
            test=test,
            feature_cols=feature_cols,
            bounce_probs=bounce_probs,
            expected_rrs=expected_rrs,
            cfg=cfg,
            args=args,
            res_full=res_full,
            base_tf=base_tf,
        )

    symbol = raw_trades["symbol"].iloc[0] if "symbol" in raw_trades.columns and len(raw_trades) else "SIM"
    mode_arg = str(args.entry_fill_mode or "both").lower()
    if mode_arg == "both":
        entry_modes = ["bar_close", "last_trade"]
    else:
        entry_modes = [mode_arg]

    live_results = {}
    live_touch_maps = {}
    live_bar_metrics_by_mode = {}
    live_feature_snapshots_by_mode = {}
    for mode in entry_modes:
        rprint(f"Running simulated live funds replay (incremental features, entry_fill_mode={mode})...")
        sim_trader = SimulatedLiveFundsTrader(
            model_dir=model_dir,
            symbol=symbol,
            testnet=False,
            min_quality=getattr(cfg, "min_quality", "B"),
            min_trend_prob=0.0,
            min_bounce_prob=float(args.min_bounce_prob),
            max_bounce_prob=float(args.max_bounce_prob),
            trade_side=str(args.trade_side),
            stop_loss_atr=float(args.stop_loss_atr),
            stop_padding_pct=float(args.stop_padding_pct),
            take_profit_rr=float(args.take_profit_rr),
            use_dynamic_rr=bool(args.use_dynamic_rr),
            use_calibration=bool(args.use_calibration),
            use_incremental=True,
            cooldown_bars_after_stop=int(args.cooldown_bars_after_stop),
            update_interval=0.0,
            warmup_trades=int(args.warmup_trades),
            log_dir=Path("./live_results_simulated"),
            bootstrap_csv=None,
            lookback_days=args.lookback_days,
            starting_balance=float(args.initial_capital),
            max_entry_deviation_atr=float(args.max_entry_deviation_atr),
            entry_fill_mode=mode,
            ema_touch_mode=str(args.ema_touch_mode),
            test_start_bar_time=test_start_bar_time,
            test_end_bar_time=test_end_bar_time,
        )

        sim_trader.config.labels.touch_threshold_atr = float(args.touch_threshold_atr)
        if sim_trader.predictor and sim_trader.predictor.incremental_engine:
            sim_trader.predictor.incremental_engine.touch_threshold_atr = float(args.touch_threshold_atr)

        touch_snapshot = {} if args.dump_touch_diffs else None
        bar_snapshot = {} if report_handle is not None else None
        feature_snapshot = {} if report_handle is not None else None
        live_results[mode] = _run_trade_replay(
            sim_trader,
            raw_trades,
            cfg,
            batch_size=max(1, int(args.batch_size)),
            printer=rprint,
            touch_snapshot=touch_snapshot,
            bar_diagnostics=bar_snapshot,
            feature_snapshots=feature_snapshot,
        )
        if touch_snapshot is not None:
            live_touch_maps[mode] = touch_snapshot
        if bar_snapshot is not None:
            live_bar_metrics_by_mode[mode] = bar_snapshot
        if feature_snapshot is not None:
            live_feature_snapshots_by_mode[mode] = feature_snapshot

    rprint("")
    _summarize_backtest("FULL", res_full, printer=rprint)
    if args.print_all_trades:
        _print_trades("FULL", res_full.trades, printer=rprint)

    for mode in entry_modes:
        _summarize_backtest(f"LIVE SIM ({mode})", live_results[mode], printer=rprint)
        _summarize_deltas(f"LIVE {mode} - FULL", res_full, live_results[mode], printer=rprint)
        _summarize_trade_overlap(
            f"FULL vs LIVE ({mode})",
            "full",
            res_full.trades,
            mode,
            live_results[mode].trades,
            printer=rprint,
            show_only=True,
        )
        if args.print_all_trades:
            _print_trades(f"LIVE SIM ({mode})", live_results[mode].trades, printer=rprint)
        if args.dump_touch_diffs and full_touch_by_time:
            _summarize_touch_diffs(
                f"FULL vs LIVE ({mode})",
                full_touch_by_time,
                live_touch_maps.get(mode, {}),
                printer=rprint,
                limit=args.dump_touch_limit,
            )

    if len(entry_modes) == 2 and "bar_close" in live_results and "last_trade" in live_results:
        _summarize_deltas("LAST_TRADE - BAR_CLOSE", live_results["bar_close"], live_results["last_trade"], printer=rprint)
        _summarize_trade_overlap(
            "LIVE bar_close vs LIVE last_trade",
            "bar_close",
            live_results["bar_close"].trades,
            "last_trade",
            live_results["last_trade"].trades,
            printer=rprint,
            show_only=False,
        )

    if report_handle is not None:
        pullback_ema = int(cfg.labels.pullback_ema)
        base_tf_seconds = int(cfg.features.timeframes[cfg.base_timeframe_idx])
        fprint("REPORT CONFIG")
        fprint(f"model_dir: {model_dir}")
        fprint(f"data_dir: {cfg.data.data_dir}")
        fprint(f"base_tf: {base_tf}")
        fprint(f"lookback_days: {cfg.data.lookback_days}")
        fprint(f"ema_touch_mode: {args.ema_touch_mode}")
        fprint(f"pullback_ema: {pullback_ema}")
        fprint(f"touch_threshold_atr: {args.touch_threshold_atr}")
        fprint(f"stop_loss_atr: {args.stop_loss_atr}")
        fprint(f"take_profit_rr: {args.take_profit_rr}")
        fprint(f"min_bounce_prob: {args.min_bounce_prob}")
        fprint(f"max_bounce_prob: {args.max_bounce_prob}")
        fprint(f"trade_side: {args.trade_side}")
        fprint(f"use_dynamic_rr: {args.use_dynamic_rr}")
        fprint(f"use_calibration: {args.use_calibration}")
        fprint(f"cooldown_bars_after_stop: {args.cooldown_bars_after_stop}")
        fprint(f"entry_fill_mode: {args.entry_fill_mode}")
        fprint(f"feature_cols_count: {len(feature_cols)}")
        fprint("")

        fprint("FEATURE_COLUMNS")
        for col in feature_cols:
            fprint(col)
        fprint("")

        _write_trades_csv(
            report_handle,
            "FULL_TRADES_CSV",
            res_full.trades,
            extra_by_bar_time=full_bar_metrics,
            base_tf_seconds=base_tf_seconds,
        )
        for mode in entry_modes:
            _write_trades_csv(
                report_handle,
                f"LIVE_TRADES_CSV ({mode})",
                live_results[mode].trades,
                extra_by_bar_time=live_bar_metrics_by_mode.get(mode),
                base_tf_seconds=base_tf_seconds,
            )

        if full_bar_metrics:
            for mode in entry_modes:
                live_metrics = live_bar_metrics_by_mode.get(mode, {})
                _write_bar_comparison_csv(
                    report_handle,
                    f"BAR_COMPARISON_ALL ({mode})",
                    base_tf,
                    pullback_ema,
                    full_bar_metrics,
                    live_metrics,
                    diff_only=False,
                )
                _write_bar_comparison_csv(
                    report_handle,
                    f"BAR_COMPARISON_DIFFS ({mode})",
                    base_tf,
                    pullback_ema,
                    full_bar_metrics,
                    live_metrics,
                    diff_only=True,
                )
                _write_entry_feature_diffs_csv(
                    report_handle,
                    f"ENTRY_FEATURE_DIFFS ({mode})",
                    feature_cols,
                    full_bar_metrics,
                    live_metrics,
                    res_full.trades,
                    live_results[mode].trades,
                    base_tf_seconds,
                )
                _write_feature_snapshot_10m_csv(
                    report_handle,
                    f"FEATURE_SNAPSHOT_10M ({mode})",
                    feature_cols,
                    full_bar_metrics,
                    live_feature_snapshots_by_mode.get(mode),
                )


if __name__ == "__main__":
    main()
