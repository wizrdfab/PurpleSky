"""
Live (real funds) trader for TrendFollower.

This script mirrors the "at close" backtest logic, but places REAL orders on Bybit:
- Uses Bybit public trade WebSocket (microstructure-based features from trades)
- Recomputes features only when a base timeframe bar closes (default: 5m)
- Long-only: enters only when EMA9 slope_norm > 0
- Entry gate: bounce_prob >= min_bounce_prob
- Entry: market order
- SL/TP: set on Bybit side via order parameters (takeProfit/stopLoss)

SAFETY:
- Defaults to TESTNET. Use --live to trade mainnet (REAL MONEY).
- When --live is used, the script requires an explicit confirmation.

Env vars required:
  BYBIT_API_KEY
  BYBIT_API_SECRET
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Local pybit fallback (repo ships pybit-master)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_LOCAL_PYBIT = _HERE / "pybit-master"
if (_LOCAL_PYBIT / "pybit").exists():
    sys.path.insert(0, str(_LOCAL_PYBIT))

try:
    from pybit.unified_trading import WebSocket
except Exception as exc:  # pragma: no cover - runtime dependency
    raise ImportError("pybit is required (pip install pybit) or use bundled pybit-master") from exc

from config import TrendFollowerConfig
from exchange_client import BybitClient
from models import TrendFollowerModels

# Reuse the streaming bar builder + feature manager to avoid recomputing from raw trades.
from bybit_live_trader import FeatureManager, StreamingBars


def _setup_logger(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("live_trader")
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


def _to_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _floor_to_step(value: float, step: float) -> float:
    if not step or step <= 0:
        return value
    return math.floor(value / step) * step


def _round_price_to_tick(price: float, tick_size: Optional[float]) -> float:
    if tick_size is None or tick_size <= 0:
        return price
    return round(price / tick_size) * tick_size


@dataclass
class LiveParams:
    position_size_pct: float = 0.02
    stop_loss_atr: float = 1.0
    take_profit_rr: float = 1.5
    min_bounce_prob: float = 0.48
    leverage: int = 1
    cooldown_bars_after_stop: int = 3


class LiveBybitTrader:
    """
    Real-money trader that mirrors the profitable "at close" backtest:
    - Only evaluates signals when a base bar closes
    - Long-only via EMA9 slope_norm
    - Bounce prob gate only
    """

    def __init__(
        self,
        *,
        model_dir: Path,
        symbol: str,
        testnet: bool,
        timeframes: List[int],
        timeframe_names: List[str],
        base_tf: str,
        update_interval: float,
        warmup_trades: int,
        save_trades_path: Optional[Path],
        bootstrap_csv: Optional[Path],
        params: LiveParams,
        max_bars: int = 5000,
        log_file: Optional[str] = None,
    ):
        self.symbol = symbol
        self.testnet = testnet
        self.update_interval = float(update_interval)
        self.warmup_trades = int(warmup_trades)
        self.save_trades_path = save_trades_path
        self.bootstrap_csv = bootstrap_csv
        self.params = params

        self.logger = _setup_logger(log_file=log_file)

        self.config = TrendFollowerConfig()
        self.config.features.timeframes = list(timeframes)
        self.config.features.timeframe_names = list(timeframe_names)
        if base_tf not in self.config.features.timeframe_names:
            raise ValueError("base_tf must be one of timeframe_names")
        self.config.base_timeframe_idx = self.config.features.timeframe_names.index(base_tf)
        self.base_tf = base_tf
        self.base_tf_seconds = int(self.config.features.timeframes[self.config.base_timeframe_idx])

        # Critical: align with EMA9-only + touch labels config used in training/backtests
        self.config.features.ema_periods = [9]
        self.config.labels.pullback_ema = 9
        self.config.labels.pullback_threshold = 0.02

        api_key = os.environ.get("BYBIT_API_KEY")
        api_secret = os.environ.get("BYBIT_API_SECRET")
        if not api_key or not api_secret:
            raise RuntimeError("Missing BYBIT_API_KEY / BYBIT_API_SECRET env vars")

        self.client = BybitClient(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
            category="linear",
        )

        self.models = TrendFollowerModels(self.config.model)
        self.models.load_all(model_dir)

        self.bars = StreamingBars(self.config.features.timeframes, self.config.features.timeframe_names, max_bars=max_bars)
        self.features = FeatureManager(self.config, self.base_tf)
        self.features.expected_features = self.models.entry_model.feature_names

        self.ws: Optional[WebSocket] = None
        self.trade_queue: Deque[pd.DataFrame] = deque()
        self.queue_lock = threading.Lock()
        self.running = False

        self.processed_trades = 0
        self.last_trade_time: Optional[float] = None

        self.balance_cache = 0.0
        self.last_balance_refresh = 0.0

        self.position_open = False
        self.position_seen_on_exchange = False
        self.last_order_time: Optional[float] = None
        self.last_order_id: Optional[str] = None
        self.last_stop_time: Optional[datetime] = None
        self.last_closed_pnl_created: Optional[int] = None

        self._save_buffer: List[pd.DataFrame] = []
        self._save_buffer_count: int = 0
        self._last_save_flush: float = 0.0

    # ------------------------------------------------------------------ websocket
    def _on_trade(self, message: dict):
        try:
            data = message.get("data", [])
            if not data:
                return
            trades = []
            for t in data:
                trades.append(
                    {
                        "timestamp": t["T"] / 1000,
                        "symbol": t["s"],
                        "side": t["S"],
                        "size": float(t["v"]),
                        "price": float(t["p"]),
                        "tickDirection": t.get("L", ""),
                    }
                )
            df = pd.DataFrame(trades)
            with self.queue_lock:
                self.trade_queue.append(df)
            self.last_trade_time = time.time()
        except Exception as exc:
            self.logger.error(f"Trade callback error: {exc}")

    def _start_stream(self):
        self.ws = WebSocket(testnet=self.testnet, channel_type="linear")
        self.ws.trade_stream(symbol=self.symbol, callback=self._on_trade)
        self.logger.info(f"Subscribed to public trade stream for {self.symbol} ({'testnet' if self.testnet else 'mainnet'})")

    def _stop_stream(self):
        if self.ws is None:
            return
        try:
            self.ws.exit()
        except Exception:
            pass
        self.ws = None

    # ------------------------------------------------------------------ helpers
    def _refresh_balance(self, force: bool = False):
        now = time.time()
        if not force and (now - self.last_balance_refresh) < 30:
            return
        self.balance_cache = self.client.get_available_balance(logger=self.logger)
        self.last_balance_refresh = now

    def _sync_position_state(self):
        """Keep local state aligned with the exchange position."""
        pos = self.client.get_position(self.symbol)
        is_open = bool(pos and pos.is_open)

        if is_open:
            if not self.position_open:
                self.logger.info(f"Detected open position on exchange: side={pos.side} size={pos.size} entry={pos.entry_price}")
            self.position_open = True
            self.position_seen_on_exchange = True
            return

        if self.position_open and not is_open:
            # Grace period after placing a market order: the position may not be visible instantly.
            now = time.time()
            if not self.position_seen_on_exchange and self.last_order_time and (now - self.last_order_time) < 30:
                return
            self.position_open = False
            self.position_seen_on_exchange = False
            self.logger.info("Position closed on exchange.")
            self._handle_position_closed()

    def _handle_position_closed(self):
        """
        Best-effort: fetch most recent closed PnL record to infer stop-loss vs take-profit.
        If closedPnl is negative -> treat as stop-loss and apply cooldown.
        """
        try:
            resp = self.client.session.get_closed_pnl(category=self.client.category, symbol=self.symbol, limit=1)
            if not resp or resp.get("retCode", -1) != 0:
                return
            recs = resp.get("result", {}).get("list", [])
            if not recs:
                return
            rec = recs[0]
            created = int(_to_float(rec.get("createdTime"), 0))
            if self.last_closed_pnl_created is not None and created <= self.last_closed_pnl_created:
                return
            self.last_closed_pnl_created = created

            closed_pnl = _to_float(rec.get("closedPnl"), 0.0)
            self.logger.info(f"Last closed PnL: {closed_pnl:.4f} (createdTime={created})")
            if closed_pnl < 0:
                self.last_stop_time = datetime.now()
        except Exception as exc:
            self.logger.warning(f"Unable to fetch closed PnL: {exc}")

    def _cooldown_active(self) -> bool:
        if not self.last_stop_time:
            return False
        cooldown_seconds = self.params.cooldown_bars_after_stop * self.base_tf_seconds
        return (datetime.now() - self.last_stop_time) < timedelta(seconds=cooldown_seconds)

    def _maybe_trade(self, features_df: pd.DataFrame):
        # Align state with exchange before evaluating a new entry
        self._sync_position_state()
        if self.position_open:
            return
        if self._cooldown_active():
            return

        latest = features_df.iloc[[-1]]
        expected = self.features.expected_features or self.features.feature_cols
        feature_data = {}
        missing = 0
        for col in expected:
            if col in latest.columns:
                feature_data[col] = latest[col].fillna(0).values
            else:
                feature_data[col] = [0]
                missing += 1
        X = pd.DataFrame(feature_data, index=latest.index)
        row = latest.iloc[0]

        entry_pred = self.models.entry_model.predict(X)
        bounce_prob = float(entry_pred["bounce_prob"][0])

        slope_col = f"{self.base_tf}_ema_{self.config.labels.pullback_ema}_slope_norm"
        ema_col = f"{self.base_tf}_ema_{self.config.labels.pullback_ema}"
        atr_col = f"{self.base_tf}_atr"

        slope_norm = float(row.get(slope_col, 0.0))
        trend_dir = 1 if slope_norm > 0 else 0

        price = float(row.get("close", 0.0))
        atr = float(row.get(atr_col, 0.0))
        if not atr or atr <= 0 or np.isnan(atr):
            atr = price * 0.02
        ema = row.get(ema_col, np.nan)
        is_pullback = False
        dist_atr = None
        if pd.notna(ema) and atr > 0:
            dist_atr = abs(price - float(ema)) / atr
            is_pullback = dist_atr <= float(self.config.labels.pullback_threshold)

        trend_aligned = trend_dir == 1
        if bounce_prob > 0.6 and trend_aligned and is_pullback:
            quality = "A"
        elif bounce_prob > 0.5 and (trend_aligned or is_pullback):
            quality = "B"
        else:
            quality = "C"

        bounce_ok = bounce_prob >= self.params.min_bounce_prob
        self.logger.info(
            f"Signal check | close={price:.6f} atr={atr:.6f} slope_norm={slope_norm:.3f} "
            f"bounce_prob={bounce_prob:.3f} dist_atr={dist_atr if dist_atr is not None else 'na'} "
            f"is_pullback={is_pullback} quality={quality} "
            f"features_used={len(X.columns)}/{len(expected)} missing={missing} "
            f"gates: trend_ok={trend_dir==1} bounce_ok={bounce_ok}"
        )

        if trend_dir != 1 or not bounce_ok:
            return

        stop_distance = self.params.stop_loss_atr * atr
        stop_loss = price - stop_distance
        take_profit = price + stop_distance * self.params.take_profit_rr

        # Refresh balance for sizing
        self._refresh_balance(force=False)
        if self.balance_cache <= 0:
            self.logger.warning("Balance is 0; skipping order.")
            return

        risk_amount = self.balance_cache * self.params.position_size_pct
        risk_per_unit = abs(price - stop_loss)
        qty = risk_amount / risk_per_unit if risk_per_unit > 0 else 0.0
        if qty <= 0:
            return

        # Best-effort rounding to instrument steps (avoid order rejections)
        info = self.client.get_instrument_info(self.symbol)
        qty_step = None
        min_qty = None
        tick_size = None
        if info:
            lot = info.get("lotSizeFilter", {}) or {}
            price_filter = info.get("priceFilter", {}) or {}
            qty_step = _to_float(lot.get("qtyStep"), 0.0) or None
            min_qty = _to_float(lot.get("minOrderQty"), 0.0) or None
            tick_size = _to_float(price_filter.get("tickSize"), 0.0) or None

        if qty_step:
            qty = _floor_to_step(qty, qty_step)
        if min_qty and qty < min_qty:
            self.logger.warning(f"Computed qty {qty:.6f} below minOrderQty {min_qty}; skipping.")
            return

        stop_loss = _round_price_to_tick(stop_loss, tick_size)
        take_profit = _round_price_to_tick(take_profit, tick_size)

        self.logger.info(
            f"Placing market order | side=Buy qty={qty:.6f} ref_price={price:.6f} "
            f"SL={stop_loss:.6f} TP={take_profit:.6f} balance={self.balance_cache:.2f}"
        )

        result = self.client.open_position(
            symbol=self.symbol,
            side="Buy",
            qty=qty,
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=int(self.params.leverage),
        )
        if result.success:
            self.position_open = True
            self.position_seen_on_exchange = False
            self.last_order_time = time.time()
            self.last_order_id = result.order_id
            self.logger.info(
                f"Opened {quality}-grade LONG | qty={qty:.6f} ref_price={price:.6f} "
                f"SL={stop_loss:.6f} TP={take_profit:.6f} bounce_prob={bounce_prob:.3f}"
            )
        else:
            self.logger.error(f"Order failed: {result.error_message}")

    def _append_trades_for_restart(self, trades: pd.DataFrame):
        if self.save_trades_path is None or trades.empty:
            return
        self._save_buffer.append(trades)
        self._save_buffer_count += len(trades)
        now = time.time()
        if self._save_buffer_count < 1000 and (now - self._last_save_flush) < 30:
            return
        self._flush_saved_trades()

    def _flush_saved_trades(self):
        if self.save_trades_path is None or not self._save_buffer:
            return
        try:
            df = pd.concat(self._save_buffer, ignore_index=True)
            path = Path(self.save_trades_path)
            header = not path.exists() or path.stat().st_size == 0
            df.to_csv(path, mode="a", index=False, header=header)
            self._save_buffer.clear()
            self._save_buffer_count = 0
            self._last_save_flush = time.time()
        except Exception as exc:
            self.logger.warning(f"Failed to append trades to {self.save_trades_path}: {exc}")

    # ------------------------------------------------------------------ bootstrap
    def _bootstrap_from_csv(self):
        if not self.bootstrap_csv:
            return
        path = Path(self.bootstrap_csv)
        if not path.exists():
            self.logger.warning(f"Bootstrap CSV not found: {path}")
            return
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            self.logger.warning(f"Failed reading bootstrap CSV: {exc}")
            return
        required = {"timestamp", "price", "size", "side"}
        if not required.issubset(set(df.columns)):
            self.logger.warning(f"Bootstrap CSV missing required columns: {required}")
            return
        if "tickDirection" not in df.columns:
            df["tickDirection"] = ""
        df = df.sort_values("timestamp")
        closed_base = self.bars.add_trades(df, self.base_tf)
        self.processed_trades += len(df)
        self.logger.info(f"Bootstrapped {len(df)} trades from {path} (closed_base={closed_base})")

    # ------------------------------------------------------------------ run loop
    def run(self):
        self.running = True
        self._refresh_balance(force=True)
        self._sync_position_state()
        self._bootstrap_from_csv()
        self._start_stream()

        self.logger.info(
            f"Live trader started | symbol={self.symbol} base_tf={self.base_tf} "
            f"min_bounce_prob={self.params.min_bounce_prob} SL_ATR={self.params.stop_loss_atr} "
            f"TP_RR={self.params.take_profit_rr} warmup_trades={self.warmup_trades}"
        )

        try:
            while self.running:
                batch = None
                with self.queue_lock:
                    if self.trade_queue:
                        batch = self.trade_queue.popleft()

                if batch is not None and not batch.empty:
                    self._append_trades_for_restart(batch)
                    closed_base = self.bars.add_trades(batch, self.base_tf)
                    self.processed_trades += len(batch)

                    if closed_base and self.processed_trades >= self.warmup_trades:
                        bars_dict = self.bars.get_bars_dict()
                        feats = self.features.compute(bars_dict)
                        if feats is not None and len(feats) > 0:
                            self._maybe_trade(feats)

                # Periodic housekeeping
                self._sync_position_state()
                self._refresh_balance(force=False)
                time.sleep(self.update_interval)
        finally:
            self._flush_saved_trades()
            self._stop_stream()

    def stop(self):
        self.running = False


def _parse_timeframes(tf_str: str) -> List[int]:
    return [int(x.strip()) for x in tf_str.split(",") if x.strip()]


def _parse_names(names_str: str) -> List[str]:
    return [x.strip() for x in names_str.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Live (real funds) Bybit trader for TrendFollower (EMA9 slope + bounce_prob).")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory containing trained models")
    parser.add_argument("--symbol", type=str, required=True, help="Trading symbol (e.g., ZECUSDT)")

    parser.add_argument("--testnet", action="store_true", help="Use Bybit testnet (default)")
    parser.add_argument("--live", action="store_true", help="Use Bybit mainnet (REAL MONEY)")

    parser.add_argument("--timeframes", type=str, default="60,300,900,3600,14400", help="Comma-separated timeframe sizes in seconds")
    parser.add_argument("--timeframe-names", type=str, default="1m,5m,15m,1h,4h", help="Comma-separated names matching timeframes")
    parser.add_argument("--base-tf", type=str, default="5m", help="Base timeframe name used for entries")
    parser.add_argument("--update-interval", type=float, default=1.0, help="Seconds between loop iterations")
    parser.add_argument("--warmup-trades", type=int, default=1000, help="Trades to accumulate before first prediction")
    parser.add_argument("--max-bars", type=int, default=5000, help="Max bars to keep per timeframe")

    parser.add_argument("--position-size-pct", type=float, default=0.02, help="Risk per trade as fraction of balance")
    parser.add_argument("--stop-loss-atr", type=float, default=1.0, help="Stop loss in ATR multiples")
    parser.add_argument("--take-profit-rr", type=float, default=1.5, help="Take profit R:R")
    parser.add_argument("--min-bounce-prob", type=float, default=0.48, help="Minimum bounce probability gate")
    parser.add_argument("--leverage", type=int, default=1, help="Leverage to set on orders")
    parser.add_argument("--cooldown-bars-after-stop", type=int, default=3, help="Bars to wait after a stop-loss")

    parser.add_argument("--save-trades", type=str, default=None, help="Optional CSV to append streamed trades for restart")
    parser.add_argument("--bootstrap-csv", type=str, default=None, help="Optional CSV of recent trades to seed bars")
    parser.add_argument("--log-file", type=str, default=None, help="Optional log file path")

    args = parser.parse_args()

    # Determine testnet mode
    use_testnet = True
    if args.live:
        use_testnet = False

    logger = _setup_logger(log_file=args.log_file)
    logger.info("=" * 60)
    logger.info("LIVE TRADER CONFIG")
    logger.info("=" * 60)
    logger.info(f"Mode:        {'TESTNET' if use_testnet else 'MAINNET'}")
    logger.info(f"Symbol:      {args.symbol}")
    logger.info(f"Model dir:   {args.model_dir}")
    logger.info(f"SL_ATR:      {args.stop_loss_atr}")
    logger.info(f"TP_RR:       {args.take_profit_rr}")
    logger.info(f"Min bounce:  {args.min_bounce_prob}")
    logger.info(f"Leverage:    {args.leverage}")
    logger.info("=" * 60)

    if not use_testnet:
        logger.warning("LIVE MODE SELECTED - REAL MONEY AT RISK")
        confirm = input("Type 'YES' to confirm live trading: ").strip()
        if confirm != "YES":
            logger.info("Aborted.")
            return

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        logger.error(f"Model directory not found: {model_dir}")
        return

    timeframes = _parse_timeframes(args.timeframes)
    names = _parse_names(args.timeframe_names)
    if len(timeframes) != len(names):
        logger.error("timeframes and timeframe-names must have the same length")
        return
    if args.base_tf not in names:
        logger.error("base-tf must be one of timeframe-names")
        return

    params = LiveParams(
        position_size_pct=float(args.position_size_pct),
        stop_loss_atr=float(args.stop_loss_atr),
        take_profit_rr=float(args.take_profit_rr),
        min_bounce_prob=float(args.min_bounce_prob),
        leverage=int(args.leverage),
        cooldown_bars_after_stop=int(args.cooldown_bars_after_stop),
    )

    trader = LiveBybitTrader(
        model_dir=model_dir,
        symbol=args.symbol,
        testnet=use_testnet,
        timeframes=timeframes,
        timeframe_names=names,
        base_tf=args.base_tf,
        update_interval=float(args.update_interval),
        warmup_trades=int(args.warmup_trades),
        save_trades_path=Path(args.save_trades) if args.save_trades else None,
        bootstrap_csv=Path(args.bootstrap_csv) if args.bootstrap_csv else None,
        params=params,
        max_bars=int(args.max_bars),
        log_file=args.log_file,
    )

    try:
        trader.run()
    except KeyboardInterrupt:
        trader.stop()


if __name__ == "__main__":
    main()
