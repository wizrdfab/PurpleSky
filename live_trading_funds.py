"""
Live trading with real funds on Bybit.

This script reuses the SAME market-data ingestion, bar building, feature calculation,
and signal logic as `live_trading.py`, but replaces paper execution with real Bybit
orders (Market) and attaches SL/TP on the exchange side.

Safety notes:
- Requires BYBIT_API_KEY / BYBIT_API_SECRET (env vars) or --api-key/--api-secret.
- Use --dry-run to disable real order placement (signals + paper execution only).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from pathlib import Path
from typing import Optional


# Ensure bundled `pybit-master` is importable (repo-local dependency).
_REPO_DIR = Path(__file__).resolve().parent
_PYBIT_DIR = _REPO_DIR / "pybit-master"
if _PYBIT_DIR.exists():
    sys.path.insert(0, str(_PYBIT_DIR))


from exchange_client import BybitClient  # noqa: E402
from live_trading import (  # noqa: E402
    DEFAULT_PARAMS,
    CompletedTrade,
    LivePaperTrader,
    PaperPosition,
)


class _Cp1252SafeFilter(logging.Filter):
    """Prevent Windows console crashes if a log message contains unsupported chars."""

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover
        try:
            msg = record.getMessage()
        except Exception:
            return True
        safe = msg.encode("cp1252", errors="replace").decode("cp1252", errors="replace")
        record.msg = safe
        record.args = ()
        return True


def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )

    log_filter = _Cp1252SafeFilter()
    root = logging.getLogger()
    for handler in root.handlers:
        handler.addFilter(log_filter)

    return logging.getLogger(__name__)


def _to_float(value, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return default


def _floor_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    d_value = Decimal(str(value))
    d_step = Decimal(str(step))
    return float((d_value / d_step).to_integral_value(rounding=ROUND_DOWN) * d_step)


def _round_stop_tp(
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    tick_size: float,
    direction: int,
) -> tuple[float, float]:
    if tick_size <= 0:
        return stop_loss, take_profit

    d_tick = Decimal(str(tick_size))

    def _round(value: float, rounding) -> float:
        d_val = Decimal(str(value))
        return float((d_val / d_tick).to_integral_value(rounding=rounding) * d_tick)

    if direction == 1:
        # Tighten (toward entry): stop up, TP down.
        stop = _round(stop_loss, ROUND_UP)
        tp = _round(take_profit, ROUND_DOWN)
        if stop >= entry_price:
            stop = _round(entry_price - tick_size, ROUND_DOWN)
        if tp <= entry_price:
            tp = _round(entry_price + tick_size, ROUND_UP)
        return stop, tp

    # direction == -1
    stop = _round(stop_loss, ROUND_DOWN)
    tp = _round(take_profit, ROUND_UP)
    if stop <= entry_price:
        stop = _round(entry_price + tick_size, ROUND_UP)
    if tp >= entry_price:
        tp = _round(entry_price - tick_size, ROUND_DOWN)
    return stop, tp


class LiveFundsTrader(LivePaperTrader):
    """
    Executes the same signals as LivePaperTrader, but on Bybit with real orders.

    - Market entries on base-bar close (same as profitable backtest logic in live_trading.py)
    - SL/TP attached on the exchange
    - Exchange position state is polled; paper exit simulation is disabled
    """

    def __init__(
        self,
        *args,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        leverage: int = 1,
        balance_asset: str = "USDT",
        dry_run: bool = False,
        max_entry_deviation_atr: float = 1.0,
        ema_touch_mode: str = "base",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.leverage = max(1, int(leverage))
        self.balance_asset = balance_asset
        self.dry_run = dry_run
        # In real-funds mode, poll exchange position state frequently (not only on bar closes).
        # In dry-run mode, keep bar-close-only exits to match backtest math.
        self.exit_on_bar_close_only = bool(self.dry_run)

        # EMA touch mode: "base" = only base TF touch detection, "multi" = multi-TF detection
        self.ema_touch_mode = str(ema_touch_mode or "base").lower()
        if self.ema_touch_mode not in ("base", "multi"):
            self.ema_touch_mode = "base"

        self._bybit: Optional[BybitClient] = None
        self._instrument_info: Optional[dict] = None
        self._qty_step: float = 0.0
        self._min_qty: float = 0.0
        self._tick_size: float = 0.0
        self._last_entry_skip_log: Optional[datetime] = None
        self._max_entry_price_deviation_atr: float = max(0.0, float(max_entry_deviation_atr))  # safety: skip if signal price is too far from live

        api_key = api_key or os.getenv("BYBIT_API_KEY", "")
        api_secret = api_secret or os.getenv("BYBIT_API_SECRET", "")

        if not api_key or not api_secret:
            self.logger.warning(
                "BYBIT_API_KEY/BYBIT_API_SECRET not set; running in --dry-run mode (no real orders)."
            )
            self.dry_run = True
            return

        self._bybit = BybitClient(api_key=api_key, api_secret=api_secret, testnet=self.testnet)

        if self.dry_run:
            return

        # Initialize paper capital from exchange available balance for accurate P&L tracking.
        bal = self._bybit.get_available_balance(asset=self.balance_asset, logger=self.logger)
        if bal > 0:
            self.initial_capital = bal
            self.capital = bal
            self.logger.info(f"Using exchange available balance as capital: {bal:.2f} {self.balance_asset}")

    # ----------------------------------------------------------------- helpers
    def _load_instrument_filters(self) -> None:
        if self._instrument_info is not None or self._bybit is None:
            return

        info = self._bybit.get_instrument_info(self.symbol)
        self._instrument_info = info or {}

        lot = (info or {}).get("lotSizeFilter", {}) if isinstance(info, dict) else {}
        price_filter = (info or {}).get("priceFilter", {}) if isinstance(info, dict) else {}

        self._qty_step = _to_float(lot.get("qtyStep"), 0.0) or 0.0
        self._min_qty = _to_float(lot.get("minOrderQty"), 0.0) or 0.0
        self._tick_size = _to_float(price_filter.get("tickSize"), 0.0) or 0.0

    def _round_qty(self, qty: float) -> float:
        self._load_instrument_filters()
        if self._qty_step > 0:
            qty = _floor_to_step(qty, self._qty_step)
        if self._min_qty > 0 and qty < self._min_qty:
            return 0.0
        return float(qty)

    def _compute_base_tf_touch(self) -> dict:
        """
        Compute base-TF-only EMA touch detection (matches backtest logic).
        This is used when ema_touch_mode='base' to ensure consistency with normal backtest.
        """
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
        """
        Override to use base-TF-only EMA touch detection when ema_touch_mode='base'.
        This ensures live trading matches backtest behavior when using base TF touch mode.
        """
        if (
            self.use_incremental
            and self.ema_touch_mode == "base"
            and feature_name in ("ema_touch_detected", "ema_touch_direction", "ema_touch_dist")
        ):
            base_touch = self._compute_base_tf_touch()
            return base_touch.get(feature_name, default)
        return super()._get_feature_value(feature_name, default)

    def _get_exchange_position_is_open(self) -> bool:
        if self._bybit is None or self.dry_run:
            return self.position is not None
        pos = self._bybit.get_position(self.symbol)
        return bool(pos and pos.is_open)

    def _sync_existing_position_on_start(self) -> None:
        if self._bybit is None or self.dry_run:
            return
        if self.position is not None:
            return
        pos = self._bybit.get_position(self.symbol)
        if not pos or not pos.is_open:
            return

        direction = 1 if pos.side.lower() == "buy" else -1
        self.position = PaperPosition(
            entry_time=datetime.now(),
            direction=direction,
            entry_price=pos.entry_price,
            size=pos.size,
            stop_loss=pos.stop_loss or 0.0,
            take_profit=pos.take_profit or 0.0,
            signal_quality="N/A",
            atr_at_entry=0.0,
            metadata={"adopted_exchange_position": True, "exchange_position": asdict(pos)},
        )
        self.logger.warning(
            "Detected an already-open exchange position at startup; adopting it and disabling new entries until it closes."
        )

    # --------------------------------------------------------------- lifecycle
    def start(self):
        self._sync_existing_position_on_start()
        super().start()

    def _connect_websocket(self):
        # Keep the exact WS/trade ingestion logic from LivePaperTrader, but prime the
        # displayed price from REST so we don't start from a stale/zero price.
        super()._connect_websocket()
        self._prime_price_from_exchange()

    def _reconnect_websocket(self) -> bool:
        """
        Override reconnection to also sync exchange position state after reconnection.
        Position may have been closed via SL/TP while we were disconnected.
        """
        result = super()._reconnect_websocket()

        if result and not self.dry_run and self._bybit is not None:
            # Re-sync position state from exchange
            try:
                pos = self._bybit.get_position(self.symbol)
                exchange_has_position = bool(pos and pos.is_open)

                if self.position is not None and not exchange_has_position:
                    # Our local state thinks we have a position but exchange doesn't
                    self.logger.warning(
                        "Exchange position was closed while disconnected. Resetting local state."
                    )
                    # Record as a completed trade with unknown result
                    from live_trading import CompletedTrade
                    closed = CompletedTrade(
                        entry_time=self.position.entry_time,
                        exit_time=datetime.now(),
                        direction=self.position.direction,
                        entry_price=self.position.entry_price,
                        exit_price=self.current_price,  # best guess
                        size=self.position.size,
                        pnl=0.0,  # unknown
                        pnl_pct=0.0,
                        exit_reason="connection_loss_unknown",
                        signal_quality=self.position.signal_quality,
                    )
                    self.completed_trades.append(closed)
                    self.position = None
                elif self.position is None and exchange_has_position:
                    # Exchange has a position but local state doesn't
                    self.logger.warning(
                        "Exchange has position but local state is None after reconnect. Adopting."
                    )
                    self._sync_existing_position_on_start()
            except Exception as e:
                self.logger.warning(f"Error syncing exchange position after reconnect: {e}")

        return result

    def _prime_price_from_exchange(self) -> None:
        if self._bybit is None or self.dry_run:
            return
        try:
            live_price = self._bybit.get_current_price(self.symbol)
        except Exception:
            live_price = None

        if live_price is None or live_price <= 0:
            return

        self.current_price = float(live_price)
        self.current_high = float(live_price) if self.current_high <= 0 else max(self.current_high, float(live_price))
        if self.current_low == float("inf"):
            self.current_low = float(live_price)
        else:
            self.current_low = min(self.current_low, float(live_price))
        self.logger.info(f"Primed current price from REST: {float(live_price):.6f}")

    # ---------------------------------------------------------------- execution
    def _open_position(self, direction: int, quality: str, atr: float, entry_price: Optional[float] = None, expected_rr: Optional[float] = None):
        # Enforce "one position at a time" using both local and exchange state.
        if self.position is not None:
            return

        if not self.dry_run and self._get_exchange_position_is_open():
            self.logger.warning("Exchange already has an open position; skipping new entry.")
            return

        signal_price = entry_price if entry_price is not None else self.current_price
        if signal_price <= 0 or atr <= 0:
            return

        # Safety: never trade on a stale last bar (e.g., when bootstrap data is old and there's a gap).
        # Require the latest feature bar to be (a) completed and (b) close in time to the current bar.
        if not self.dry_run and self._bybit is not None:
            if self.last_trade_timestamp is None:
                return

            latest_bar_time = None
            if self.use_incremental:
                latest_bar_time = self.predictor.last_bar_time.get(self.base_tf)
            else:
                fc = self.predictor.features_cache
                if fc is None or len(fc) == 0 or "bar_time" not in fc.columns:
                    return
                try:
                    latest_bar_time = int(fc["bar_time"].iloc[-1])
                except Exception:
                    return

            if latest_bar_time is None:
                return
            try:
                latest_bar_time = int(latest_bar_time)
            except Exception:
                return

            current_bar_time = int(self.last_trade_timestamp // self.base_tf_seconds) * self.base_tf_seconds

            # Only trade on a completed base bar close.
            if latest_bar_time > (current_bar_time - self.base_tf_seconds):
                return

            # And only if it's recent (avoid hours-old bootstrap bars).
            if (current_bar_time - latest_bar_time) > (2 * self.base_tf_seconds):
                now = datetime.now()
                if self._last_entry_skip_log is None or (now - self._last_entry_skip_log).total_seconds() > 30:
                    self._last_entry_skip_log = now
                    self.logger.warning(
                        "Skipping entry: latest completed bar is stale "
                        f"(latest_bar_time={latest_bar_time}, current_bar_time={current_bar_time})."
                    )
                return

        # For funds trading, use the live price for freshness checks but keep SL/TP
        # anchored to the completed bar close (matches backtest semantics).
        price = signal_price
        live_price = None
        if not self.dry_run and self._bybit is not None:
            try:
                live_price = self._bybit.get_current_price(self.symbol)
            except Exception:
                live_price = None
            if live_price is not None and live_price > 0:
                live_price = float(live_price)
                self.current_price = live_price
                self.current_high = max(self.current_high, live_price)
                self.current_low = min(self.current_low, live_price)

        live_ref = live_price if live_price is not None else price
        diff_atr = abs(live_ref - signal_price) / atr if atr > 0 else 0.0

        # Extra safety: if signal price is far from live price, skip (likely stale bars/bootstrapped gap).
        if (
            not self.dry_run
            and atr > 0
            and self._max_entry_price_deviation_atr > 0
            and diff_atr > self._max_entry_price_deviation_atr
        ):
            now = datetime.now()
            if self._last_entry_skip_log is None or (now - self._last_entry_skip_log).total_seconds() > 30:
                self._last_entry_skip_log = now
                self.logger.warning(
                    "Skipping entry: signal price too far from live price "
                    f"(signal={signal_price:.6f}, live={live_ref:.6f}, diff_atr={diff_atr:.2f})."
                )
            return

        # Calculate stop loss and take profit (SAME FORMULA AS BACKTEST/live_trading.py)
        stop_dist = (self.stop_loss_atr * atr) + (getattr(self, "stop_padding_pct", 0.0) * price)
        stop_loss = price - (direction * stop_dist)

        # Dynamic RR from model or fixed RR
        effective_rr = self.take_profit_rr
        if getattr(self, 'use_dynamic_rr', False) and expected_rr is not None and expected_rr > 0.5:
            effective_rr = min(max(expected_rr, 0.5), 5.0)

        take_profit = price + (direction * stop_dist * effective_rr)

        # Size is computed as risk-based, but bounded by available margin if needed.
        if self.dry_run or self._bybit is None:
            return super()._open_position(direction, quality, atr, entry_price=price, expected_rr=expected_rr)

        balance = self._bybit.get_available_balance(asset=self.balance_asset, logger=self.logger)
        if balance <= 0:
            self.logger.warning("Available balance is 0; cannot size a position.")
            return

        risk_amount = balance * self.position_size_pct
        risk_per_unit = abs(price - stop_loss)
        if risk_per_unit <= 0:
            return

        qty = risk_amount / risk_per_unit
        qty = self._round_qty(qty)
        if qty <= 0:
            self.logger.warning("Computed qty is below exchange minimum; skipping entry.")
            return

        # Ensure margin feasibility: margin ~= notional/leverage
        notional = qty * price
        required_margin = notional / float(self.leverage)
        if required_margin > balance:
            scale = balance / required_margin
            scaled_qty = self._round_qty(qty * scale)
            if scaled_qty <= 0:
                self.logger.warning("Insufficient margin even after scaling; skipping entry.")
                return
            self.logger.warning(
                f"Scaling qty due to margin: {qty:.6f} -> {scaled_qty:.6f} (leverage={self.leverage}, balance={balance:.2f})"
            )
            qty = scaled_qty

        # Round SL/TP to tick size (conservative rounding toward entry).
        self._load_instrument_filters()
        stop_loss, take_profit = _round_stop_tp(price, stop_loss, take_profit, self._tick_size, direction)

        side = "Buy" if direction == 1 else "Sell"
        order = self._bybit.open_position(
            symbol=self.symbol,
            side=side,
            qty=qty,
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=self.leverage,
        )
        if not order.success:
            self.logger.error(f"Bybit order failed: {order.error_message or 'unknown error'}")
            return

        self.position = PaperPosition(
            entry_time=datetime.now(),
            direction=direction,
            entry_price=price,
            size=qty,
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal_quality=quality,
            atr_at_entry=atr,
            metadata={
                "bybit_order_id": order.order_id,
                "bybit_testnet": self.testnet,
                "leverage": self.leverage,
                "balance_asset": self.balance_asset,
                "balance_available": balance,
            },
        )

        self.stats.positions_opened += 1
        self.stats.signals_generated += 1

        dir_name = "LONG" if direction == 1 else "SHORT"
        self.logger.info("=" * 70)
        self.logger.info(f"OPENED {quality}-grade {dir_name} POSITION (BYBIT)")
        self.logger.info(f"   Order ID:   {order.order_id}")
        self.logger.info(f"   Entry:      {price:.6f}")
        pad_pct = getattr(self, "stop_padding_pct", 0.0) * 100
        self.logger.info(f"   Stop Loss:  {stop_loss:.6f} ({self.stop_loss_atr} ATR + {pad_pct:.3f}% pad)")
        self.logger.info(f"   Take Profit:{take_profit:.6f} ({self.take_profit_rr}:1 R:R)")
        self.logger.info(f"   Qty:        {qty:.6f} units (risk ${risk_amount:.2f}, lev {self.leverage}x)")
        self.logger.info("=" * 70)

    def _check_exit(self, current_atr: float):
        # Paper simulation when dry-run.
        if self.dry_run or self._bybit is None:
            return super()._check_exit(current_atr)

        if self.position is None:
            return

        pos = self._bybit.get_position(self.symbol)
        if pos and pos.is_open:
            return

        # Exchange position is closed. Try to read the latest closed P&L record.
        close_info = None
        try:
            resp = self._bybit.session.get_closed_pnl(category=self._bybit.category, symbol=self.symbol, limit=1)
            if resp and resp.get("retCode", -1) == 0:
                items = resp.get("result", {}).get("list", [])
                close_info = items[0] if items else None
        except Exception as exc:
            self.logger.warning(f"Could not fetch closed PnL info: {exc}")

        exit_price = None
        exchange_pnl = None
        if isinstance(close_info, dict):
            exit_price = _to_float(
                close_info.get("avgExitPrice")
                or close_info.get("exitPrice")
                or close_info.get("markPrice")
                or close_info.get("closePrice"),
                None,
            )
            exchange_pnl = _to_float(close_info.get("closedPnl") or close_info.get("pnl"), None)

        # Fallback if API fields not present.
        exit_price = exit_price if exit_price is not None else self.current_price

        exit_reason = "take_profit"
        if exchange_pnl is not None:
            exit_reason = "stop_loss" if exchange_pnl < 0 else "take_profit"
        else:
            # Infer by proximity to configured stop/target.
            stop_dist = abs(exit_price - self.position.stop_loss)
            tp_dist = abs(exit_price - self.position.take_profit)
            exit_reason = "stop_loss" if stop_dist <= tp_dist else "take_profit"

        self._close_position_exchange(exit_price=exit_price, exit_reason=exit_reason, exchange_pnl=exchange_pnl, close_info=close_info)

    def _close_position_exchange(
        self,
        exit_price: float,
        exit_reason: str,
        exchange_pnl: Optional[float] = None,
        close_info: Optional[dict] = None,
    ):
        pos = self.position
        now = datetime.now()

        # Prefer exchange P&L when available; otherwise compute (no fees).
        calc_pnl = pos.direction * (exit_price - pos.entry_price) * pos.size
        pnl = exchange_pnl if exchange_pnl is not None else calc_pnl
        pnl_percent = pos.direction * (exit_price - pos.entry_price) / pos.entry_price * 100
        duration = (now - pos.entry_time).total_seconds()

        trade = CompletedTrade(
            entry_time=pos.entry_time,
            exit_time=now,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size=pos.size,
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
            pnl=pnl,
            pnl_percent=pnl_percent,
            signal_quality=pos.signal_quality,
            exit_reason=exit_reason,
            duration_seconds=duration,
        )

        # Attach exchange close info for auditability.
        if isinstance(close_info, dict):
            trade_dict = trade.to_dict()
            trade_dict["exchange_close_info"] = close_info
            self._save_trade_dict(trade_dict)
        else:
            self._save_trade(trade)

        self.completed_trades.append(trade)

        # Update capital and stats (tracks exchange P&L when available).
        self.capital += pnl
        self.stats.positions_closed += 1
        self.stats.total_trades += 1
        self.stats.total_pnl += pnl
        self.stats.total_pnl_percent = (self.capital - self.initial_capital) / self.initial_capital * 100

        grade = pos.signal_quality or "C"
        if grade not in self.stats.trades_by_grade:
            self.stats.trades_by_grade[grade] = 0
        if grade not in self.stats.wins_by_grade:
            self.stats.wins_by_grade[grade] = 0

        if pnl > 0:
            self.stats.winning_trades += 1
            self.stats.wins_by_grade[grade] += 1
        else:
            self.stats.losing_trades += 1

        self.stats.trades_by_grade[grade] += 1
        self.position = None

        result = "WIN" if pnl > 0 else "LOSS"
        dir_name = "LONG" if pos.direction == 1 else "SHORT"

        self.logger.info("=" * 70)
        self.logger.info(f"{result} - Closed {dir_name} ({exit_reason}) (BYBIT)")
        self.logger.info(f"   Entry:    {pos.entry_price:.6f}")
        self.logger.info(f"   Exit:     {exit_price:.6f}")
        if exchange_pnl is not None:
            self.logger.info(f"   P&L:      ${pnl:+.4f} (exchange)")
        else:
            self.logger.info(f"   P&L:      ${pnl:+.4f} (calc)")
        self.logger.info(f"   Duration: {duration:.0f}s")
        self.logger.info(f"   Capital:  ${self.capital:,.2f}")
        self.logger.info("-" * 70)
        self._log_running_stats()
        self.logger.info("=" * 70)

        if exit_reason == "stop_loss":
            self.last_stop_time = now

    def _save_trade_dict(self, trade_dict: dict) -> None:
        """Write a pre-built trade dict (used to include exchange metadata)."""
        try:
            trades = []
            if self.trades_file.exists():
                with open(self.trades_file, "r", encoding="utf-8") as f:
                    trades = json.load(f)
            trades.append(trade_dict)
            with open(self.trades_file, "w", encoding="utf-8") as f:
                json.dump(trades, f, indent=2, default=str)
        except Exception as exc:
            self.logger.warning(f"Could not save trade to file: {exc}")


def main():
    parser = argparse.ArgumentParser(
        description="TrendFollower Live Trading (Bybit Funds) - reuses live_trading.py logic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Env vars:
  BYBIT_API_KEY, BYBIT_API_SECRET

Examples:
  # Dry-run (signals + paper execution)
  python live_trading_funds.py --model-dir ./models --symbol MONUSDT --dry-run

  # Real trading (mainnet by default; use --testnet for testnet)
  python live_trading_funds.py --model-dir ./models --symbol MONUSDT --leverage 5
""",
    )

    # Core
    parser.add_argument("--model-dir", type=str, required=True, help="Directory containing trained models")
    parser.add_argument("--symbol", type=str, default="MONUSDT", help="Trading symbol (default: MONUSDT)")
    parser.add_argument("--testnet", action="store_true", help="Use Bybit testnet (public WS + trading HTTP)")

    # Trading params (defaults match live_trading.py/backtest)
    parser.add_argument("--min-quality", type=str, default=DEFAULT_PARAMS["min_quality"], choices=["A", "B", "C"])
    parser.add_argument("--min-trend-prob", type=float, default=DEFAULT_PARAMS["min_trend_prob"])
    parser.add_argument("--min-bounce-prob", type=float, default=DEFAULT_PARAMS["min_bounce_prob"])
    parser.add_argument("--max-bounce-prob", type=float, default=DEFAULT_PARAMS.get("max_bounce_prob", 1.0),
                       help="Maximum bounce probability for bucket filtering (default: 1.0 = no max)")
    parser.add_argument("--use-dynamic-rr", action="store_true",
                       help="Use expected RR from model for dynamic TP sizing")
    calib_group = parser.add_mutually_exclusive_group()
    calib_group.add_argument("--use-calibration", action="store_true",
                            help="Use calibrated probabilities (Isotonic Regression)")
    calib_group.add_argument("--use-raw-probabilities", action="store_true",
                            help="Use raw (uncalibrated) probabilities (default)")
    parser.add_argument(
        "--trade-side",
        type=str,
        default=DEFAULT_PARAMS.get("trade_side", "long"),
        choices=["long", "short", "both"],
        help=f"Trade direction filter (default: {DEFAULT_PARAMS.get('trade_side', 'long')})",
    )
    parser.add_argument("--stop-loss-atr", type=float, default=DEFAULT_PARAMS["stop_loss_atr"])
    parser.add_argument(
        "--stop-padding-pct",
        type=float,
        default=DEFAULT_PARAMS.get("stop_padding_pct", 0.0),
        help=f"Extra stop distance as fraction of entry (default: {DEFAULT_PARAMS.get('stop_padding_pct', 0.0):.6f}).",
    )
    parser.add_argument(
        "--cooldown-bars-after-stop",
        type=int,
        default=int(DEFAULT_PARAMS.get("cooldown_bars_after_stop", 0)),
        help="Cooldown after a stop-loss in base bars (default: 0 = disabled).",
    )
    parser.add_argument("--take-profit-rr", type=float, default=DEFAULT_PARAMS["take_profit_rr"])

    # System params (same as live_trading.py)
    parser.add_argument("--update-interval", type=float, default=5.0, help="Seconds between predictions (default: 5.0)")
    parser.add_argument("--warmup-trades", type=int, default=1000, help="Trades before starting (default: 1000)")
    parser.add_argument(
        "--lookback-days",
        type=float,
        default=None,
        help="Limit trade history to the most recent N days (default: use train_config if set)",
    )
    parser.add_argument(
        "--max-entry-deviation-atr",
        type=float,
        default=1.0,
        help="Max allowed abs(live_price - signal_price)/ATR before skipping entries (default: 1.0; 0 disables).",
    )
    parser.add_argument("--log-file", type=str, default=None, help="Log file path")
    parser.add_argument(
        "--bootstrap-csv",
        type=str,
        default=None,
        help="Optional CSV file OR directory of CSVs to seed the buffer (timestamp,price,size,side)",
    )
    parser.add_argument("--log-dir", type=str, default="./live_results_funds", help="Directory for JSON trade logs")
    parser.add_argument('--use-incremental', action='store_true', default=True,
                       help='Use incremental feature calculation for faster updates (default: True)')
    parser.add_argument('--no-incremental', dest='use_incremental', action='store_false',
                       help='Disable incremental features and use full recalculation')
    parser.add_argument(
        "--ema-touch-mode",
        type=str,
        default="base",
        choices=["base", "multi"],
        help="EMA touch detection mode: 'base' uses only base TF EMA touch (default), 'multi' uses multi-TF detection from incremental features.",
    )

    # Bybit execution
    parser.add_argument("--api-key", type=str, default=None, help="Bybit API key (or BYBIT_API_KEY env var)")
    parser.add_argument("--api-secret", type=str, default=None, help="Bybit API secret (or BYBIT_API_SECRET env var)")
    parser.add_argument("--leverage", type=int, default=1, help="Leverage to set on Bybit (default: 1)")
    parser.add_argument("--balance-asset", type=str, default="USDT", help="Balance asset for sizing (default: USDT)")
    parser.add_argument("--dry-run", action="store_true", help="Disable real order placement")

    args = parser.parse_args()

    use_calibration = bool(args.use_calibration)
    if args.use_raw_probabilities:
        use_calibration = False

    logger = setup_logging(args.log_file)

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        logger.error(f"Model directory not found: {model_dir}")
        return

    trader = LiveFundsTrader(
        model_dir=model_dir,
        symbol=args.symbol,
        testnet=args.testnet,
        min_quality=args.min_quality,
        min_trend_prob=args.min_trend_prob,
        min_bounce_prob=args.min_bounce_prob,
        max_bounce_prob=args.max_bounce_prob,
        trade_side=args.trade_side,
        stop_loss_atr=args.stop_loss_atr,
        stop_padding_pct=args.stop_padding_pct,
        take_profit_rr=args.take_profit_rr,
        use_dynamic_rr=args.use_dynamic_rr,
        use_calibration=use_calibration,
        use_incremental=args.use_incremental,
        cooldown_bars_after_stop=args.cooldown_bars_after_stop,
        update_interval=args.update_interval,
        warmup_trades=args.warmup_trades,
        log_dir=Path(args.log_dir),
        bootstrap_csv=args.bootstrap_csv,
        lookback_days=args.lookback_days,
        api_key=args.api_key,
        api_secret=args.api_secret,
        leverage=args.leverage,
        balance_asset=args.balance_asset,
        dry_run=args.dry_run,
        max_entry_deviation_atr=args.max_entry_deviation_atr,
        ema_touch_mode=args.ema_touch_mode,
    )

    try:
        trader.start()
    except KeyboardInterrupt:
        trader.stop()


if __name__ == "__main__":
    main()
