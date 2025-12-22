#!/usr/bin/env python3
"""
Bybit live trader that mirrors TrendFollower backtest logic.

Key features:
- Streams trades from Bybit (pybit WebSocket), builds multi-timeframe bars from raw trades
- Calculates features with existing feature_engine to match backtest
- Loads trained models and applies the same entry/exit gating as SimpleBacktester
- Opens real orders via Bybit HTTP (through exchange_client.BybitClient)
- Entry decisions are made on base timeframe candle close (1m/5m/etc.)
"""
import argparse
import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pybit.unified_trading import WebSocket

from config import TrendFollowerConfig
from exchange_client import BybitClient
from feature_engine import calculate_multi_timeframe_features, get_feature_columns
from models import TrendFollowerModels


# -----------------------------------------------------------------------------
# Utility: trade archive (save only)
# -----------------------------------------------------------------------------
class TradeArchive:
    """Handles saving raw trades to CSV."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_trades(self, trades: pd.DataFrame, out_file: Path, mode: str = "a"):
        if trades.empty:
            return
        out_file = Path(out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        header = not out_file.exists() or mode == "w"
        trades.to_csv(out_file, mode=mode, header=header, index=False)


# -----------------------------------------------------------------------------
# Streaming bar builder (from raw trades)
# -----------------------------------------------------------------------------
@dataclass
class _BarState:
    bar_time: Optional[int] = None
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    value: float = 0.0
    net_side: float = 0.0
    net_volume: float = 0.0
    net_value: float = 0.0
    tick_dir_sum: float = 0.0
    trade_count: int = 0


class StreamingBars:
    """
    Builds OHLCV bars from raw trades for multiple timeframes.
    Produces the same columns as data_loader.aggregate_to_bars to keep feature parity.
    """

    TICK_MAP = {
        "PlusTick": 1.0,
        "ZeroPlusTick": 0.5,
        "MinusTick": -1.0,
        "ZeroMinusTick": -0.5,
    }

    def __init__(self, timeframes: List[int], timeframe_names: List[str], max_bars: int = 5000):
        self.timeframes = timeframes
        self.timeframe_names = timeframe_names
        self.max_bars = max_bars
        self.state: Dict[str, _BarState] = {tf_name: _BarState() for tf_name in timeframe_names}
        self.bars: Dict[str, List[Dict]] = {tf_name: [] for tf_name in timeframe_names}

    def _start_bar(self, tf_name: str, bar_time: int, price: float, size: float, side_num: float, tick_dir: float):
        st = _BarState(
            bar_time=bar_time,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=size,
            value=price * size,
            net_side=side_num,
            net_volume=size * side_num,
            net_value=price * size * side_num,
            tick_dir_sum=tick_dir,
            trade_count=1,
        )
        self.state[tf_name] = st

    def _update_bar(self, tf_name: str, price: float, size: float, side_num: float, tick_dir: float):
        st = self.state[tf_name]
        st.high = max(st.high, price)
        st.low = min(st.low, price)
        st.close = price
        st.volume += size
        st.value += price * size
        st.net_side += side_num
        st.net_volume += size * side_num
        st.net_value += price * size * side_num
        st.tick_dir_sum += tick_dir
        st.trade_count += 1

    def _finalize_bar(self, tf_name: str, tf_seconds: int):
        st = self.state[tf_name]
        if st.bar_time is None:
            return
        volume = st.volume if st.volume != 0 else np.nan
        trade_count = st.trade_count if st.trade_count != 0 else np.nan
        bar = {
            "bar_time": st.bar_time,
            "open": st.open,
            "high": st.high,
            "low": st.low,
            "close": st.close,
            "volume": st.volume,
            "value": st.value,
            "net_side": st.net_side,
            "net_volume": st.net_volume,
            "net_value": st.net_value,
            "avg_tick_dir": st.tick_dir_sum / trade_count if trade_count else 0.0,
            "trade_count": st.trade_count,
            "datetime": pd.to_datetime(st.bar_time, unit="s"),
        }
        bar["buy_volume"] = (bar["volume"] + bar["net_volume"]) / 2 if volume else 0.0
        bar["sell_volume"] = (bar["volume"] - bar["net_volume"]) / 2 if volume else 0.0
        bar["buy_sell_imbalance"] = bar["net_volume"] / volume if volume else 0.0
        bar["vwap"] = bar["value"] / volume if volume else bar["close"]
        bar["trade_intensity"] = bar["trade_count"] / tf_seconds if tf_seconds else 0.0
        bar["avg_trade_size"] = bar["volume"] / trade_count if trade_count else 0.0

        lst = self.bars[tf_name]
        lst.append(bar)
        if len(lst) > self.max_bars:
            del lst[: len(lst) - self.max_bars]
        self.state[tf_name] = _BarState()  # reset

    def add_trades(self, trades: pd.DataFrame, base_tf: str) -> bool:
        """
        Add a batch of trades. Returns True if a base timeframe bar was closed.
        Trades must have columns: timestamp, price, size, side (Buy/Sell), tickDirection.
        """
        if trades.empty:
            return False
        closed_base = False
        trades = trades.sort_values("timestamp")

        for _, row in trades.iterrows():
            ts = float(row["timestamp"])
            price = float(row["price"])
            size = float(row["size"])
            side_num = 1.0 if row["side"] == "Buy" else -1.0
            tick_dir = self.TICK_MAP.get(row.get("tickDirection", ""), 0.0)

            for tf_seconds, tf_name in zip(self.timeframes, self.timeframe_names):
                bucket = int(ts // tf_seconds) * tf_seconds
                st = self.state[tf_name]

                if st.bar_time is None:
                    self._start_bar(tf_name, bucket, price, size, side_num, tick_dir)
                    continue

                if bucket == st.bar_time:
                    self._update_bar(tf_name, price, size, side_num, tick_dir)
                elif bucket > st.bar_time:
                    self._finalize_bar(tf_name, tf_seconds)
                    if tf_name == base_tf:
                        closed_base = True
                    self._start_bar(tf_name, bucket, price, size, side_num, tick_dir)
                else:
                    # Out-of-order trade; ignore to preserve monotonic bars
                    continue
        return closed_base

    def get_bars_dict(self) -> Dict[str, pd.DataFrame]:
        """Return DataFrames of accumulated bars for each timeframe."""
        result = {}
        for tf_name, lst in self.bars.items():
            if lst:
                result[tf_name] = pd.DataFrame(lst).sort_values("bar_time")
        return result


# -----------------------------------------------------------------------------
# Feature manager
# -----------------------------------------------------------------------------
class FeatureManager:
    """Calculates multi-timeframe features and caches feature columns."""

    def __init__(self, config: TrendFollowerConfig, base_tf: str):
        self.config = config
        self.base_tf = base_tf
        self.feature_cols: List[str] = []
        self.features_cache: Optional[pd.DataFrame] = None
        self.expected_features: List[str] = []

    def compute(self, bars_dict: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
        if not bars_dict or self.base_tf not in bars_dict:
            return None
        featured = calculate_multi_timeframe_features(
            bars_dict,
            self.base_tf,
            self.config.features,
        )
        self.features_cache = featured
        if not self.feature_cols:
            self.feature_cols = get_feature_columns(featured)
        if not self.expected_features:
            self.expected_features = list(self.feature_cols)
        return featured


# -----------------------------------------------------------------------------
# Live Trader
# -----------------------------------------------------------------------------
class BybitLiveTrader:
    """Live trading orchestrator that aligns with backtest rules."""

    def __init__(
        self,
        model_dir: Path,
        symbol: str,
        timeframes: List[int],
        timeframe_names: List[str],
        base_tf: str,
        testnet: bool,
        params: Dict,
        save_trades_path: Optional[Path],
        update_interval: float,
        warmup_trades: int,
    ):
        self.symbol = symbol
        self.timeframes = timeframes
        self.timeframe_names = timeframe_names
        self.base_tf = base_tf
        self.testnet = testnet
        self.params = params
        self.update_interval = update_interval
        self.warmup_trades = warmup_trades
        self.save_trades_path = save_trades_path

        self.config = TrendFollowerConfig()
        self.config.features.timeframes = timeframes
        self.config.features.timeframe_names = timeframe_names
        self.config.base_timeframe_idx = timeframe_names.index(base_tf)
        self.base_tf_seconds = timeframes[self.config.base_timeframe_idx]

        self.logger = logging.getLogger(__name__)
        self.client = BybitClient(
            api_key=self._env("BYBIT_API_KEY"),
            api_secret=self._env("BYBIT_API_SECRET"),
            testnet=testnet,
            category="linear",
        )

        self.models = TrendFollowerModels(self.config.model)
        self.models.load_all(model_dir)

        self.bars = StreamingBars(timeframes, timeframe_names)
        self.features = FeatureManager(self.config, base_tf)
        self.features.expected_features = self.models.trend_classifier.feature_names

        self.trade_queue: Deque[pd.DataFrame] = deque()
        self.queue_lock = threading.Lock()
        self.ws: Optional[WebSocket] = None
        self.running = False

        self.position_open = False  # Single position policy
        self.balance_cache = 0.0
        self.last_save_time = time.time()
        self.last_trade_time: Optional[float] = None
        self.last_heartbeat: float = time.time()
        self.archive = TradeArchive(Path(save_trades_path).parent if save_trades_path else Path("./live_trading_logs"))

    # --------------------------------------------------------------------- util
    def _env(self, key: str) -> str:
        import os

        val = os.environ.get(key)
        if not val:
            raise RuntimeError(f"Missing env var: {key}")
        return val

    # ------------------------------------------------------------------ saving
    def _append_trades(self, trades: pd.DataFrame):
        if not self.save_trades_path:
            return
        self.archive.save_trades(trades, Path(self.save_trades_path), mode="a")

    # ---------------------------------------------------------------- websocket
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
                        "tickDirection": t["L"],
                    }
                )
            df = pd.DataFrame(trades)
            with self.queue_lock:
                self.trade_queue.append(df)
            self.last_trade_time = time.time()
        except Exception as exc:
            self.logger.error(f"Error in trade callback: {exc}")

    def start_stream(self):
        self.ws = WebSocket(testnet=self.testnet, channel_type="linear")
        self.ws.trade_stream(symbol=self.symbol, callback=self._on_trade)
        self.logger.info(f"Subscribed to public trade stream for {self.symbol} ({'testnet' if self.testnet else 'mainnet'})")

    # ------------------------------------------------------------------ trading
    def run(self):
        self.running = True
        self.balance_cache = self.client.get_available_balance(logger=self.logger)
        self.logger.info(f"Starting live trader. Balance: ${self.balance_cache:,.2f}")
        if self.balance_cache == 0:
            raw = self.client.get_balances_raw()
            self.logger.warning(f"Balance reported as 0. Raw balances: {raw}")
        self.start_stream()
        self.logger.info("Waiting for trades... (heartbeat every ~10s if idle)")
        self.logger.info(f"Model expected features: {len(self.features.expected_features)}")
        if self.features.expected_features:
            self.logger.info(f"First 5 expected features: {self.features.expected_features[:5]}")

        last_eval = time.time()
        processed_trades = 0

        while self.running:
            try:
                batch = None
                with self.queue_lock:
                    if self.trade_queue:
                        batch = self.trade_queue.popleft()
                if batch is not None:
                    self.logger.info(f"Received trade batch: {len(batch)} rows")
                    closed_base = self.bars.add_trades(batch, self.base_tf)
                    processed_trades += len(batch)
                    if self.save_trades_path and processed_trades >= 1000:
                        self._append_trades(batch)
                        self.logger.info(f"Appended {processed_trades} trades to {self.save_trades_path}")
                        processed_trades = 0

                    if closed_base:
                        base_bars = len(self.bars.bars.get(self.base_tf, []))
                        self.logger.info(f"Base timeframe bar closed ({self.base_tf}); recomputing features | base_bars={base_bars}")
                        bars_dict = self.bars.get_bars_dict()
                        feats = self.features.compute(bars_dict)
                        if feats is not None:
                            self.logger.info(
                                f"Features rows={len(feats)} latest close={feats['close'].iloc[-1]:.6f} "
                                f"atr={feats.get(f'{self.base_tf}_atr', pd.Series([0])).iloc[-1]:.6f}"
                            )
                            self._maybe_trade(feats)

                now = time.time()
                if now - last_eval < self.update_interval:
                    time.sleep(0.1)
                else:
                    last_eval = now

                # Heartbeat if idle to show liveness
                if now - self.last_heartbeat >= 10:
                    # refresh balance on heartbeat
                    refreshed = self.client.get_available_balance(logger=self.logger)
                    if refreshed != self.balance_cache:
                        self.balance_cache = refreshed
                        self.logger.info(f"Heartbeat balance refresh: ${self.balance_cache:,.2f}")
                        if self.balance_cache == 0:
                            raw = self.client.get_balances_raw()
                            self.logger.warning(f"Balance refresh still 0. Raw balances: {raw}")
                    queue_size = len(self.trade_queue)
                    last_trade_age = (now - self.last_trade_time) if self.last_trade_time else None
                    msg = f"Heartbeat | queue={queue_size} | balance={self.balance_cache:.2f} | position_open={self.position_open}"
                    if last_trade_age is None:
                        msg += " | last_trade=never"
                    else:
                        msg += f" | last_trade_age={last_trade_age:.1f}s"
                    self.logger.info(msg)
                    self.last_heartbeat = now
            except KeyboardInterrupt:
                self.logger.info("Stopping (KeyboardInterrupt)...")
                self.running = False
            except Exception as exc:
                self.logger.error(f"Error in main loop: {exc}")
                time.sleep(1.0)

    # ---------------------------------------------------------------- decisions
    def _prepare_features_row(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, int, int]:
        latest = features_df.iloc[[-1]]
        expected = self.features.expected_features or self.features.feature_cols
        feature_data = {}
        missing = 0
        for col in expected:
            if col in latest.columns:
                feature_data[col] = latest[col].fillna(0).values
            else:
                feature_data[col] = [0]  # keep parity with training/backtest
                missing += 1
        X = pd.DataFrame(feature_data, index=latest.index)
        return X, latest.iloc[0], missing, len(expected)

    def _grade_signal(self, bounce_prob: float, trend_dir: int, alignment: float, is_pullback: bool) -> str:
        trend_aligned = trend_dir == np.sign(alignment) and trend_dir != 0
        if bounce_prob > 0.6 and trend_aligned and is_pullback:
            return "A"
        if bounce_prob > 0.5 and (trend_aligned or is_pullback):
            return "B"
        return "C"

    def _maybe_trade(self, features_df: pd.DataFrame):
        # Candle-close decision: only at base TF close (already gated)
        if self.position_open:
            return  # single position policy; rely on exchange SL/TP

        X, latest_row, missing_features, expected_total = self._prepare_features_row(features_df)

        trend_pred = self.models.trend_classifier.predict(X)
        entry_pred = self.models.entry_model.predict(X)

        trend_dir = int(trend_pred["prediction"][0])
        prob_up = float(trend_pred["prob_up"][0])
        prob_down = float(trend_pred["prob_down"][0])
        bounce_prob = float(entry_pred["bounce_prob"][0])

        # Alignment and pullback calc
        alignment_col = f"{self.base_tf}_ema_alignment"
        ema_col = f"{self.base_tf}_ema_{self.config.labels.pullback_ema}"
        atr_col = f"{self.base_tf}_atr"

        alignment = latest_row.get(alignment_col, 0.0)
        price = latest_row.get("close", 0.0)
        atr = latest_row.get(atr_col, 0.0)
        ema = latest_row.get(ema_col, np.nan)
        is_pullback = False
        if atr and atr > 0 and not np.isnan(ema):
            dist = abs(price - ema) / atr
            is_pullback = dist <= self.config.labels.pullback_threshold

        quality = self._grade_signal(bounce_prob, trend_dir, alignment, is_pullback)

        # Apply thresholds (matching backtest)
        quality_ok = ord(quality) <= ord(self.params["min_quality"])
        if trend_dir == 1:
            trend_prob_ok = prob_up >= self.params["min_trend_prob"]
            trend_prob = prob_up
        elif trend_dir == -1:
            trend_prob_ok = prob_down >= self.params["min_trend_prob"]
            trend_prob = prob_down
        else:
            trend_prob_ok = False
            trend_prob = max(prob_up, prob_down)
        bounce_ok = bounce_prob >= self.params["min_bounce_prob"]

        self.logger.info(
            f"Signal check | close={price:.6f} atr={atr:.6f} trend_dir={trend_dir} "
            f"prob_up={prob_up:.3f} prob_down={prob_down:.3f} bounce_prob={bounce_prob:.3f} "
            f"alignment={alignment:.3f} pullback={is_pullback} quality={quality} "
            f"features_used={len(X.columns)}/{expected_total} missing={missing_features} "
            f"gates: quality_ok={quality_ok} trend_prob_ok={trend_prob_ok} bounce_ok={bounce_ok}"
        )
        if missing_features > 0:
            self.logger.warning(f"Missing {missing_features} expected features; filled with 0 to match training.")
        if pd.isna(atr) or atr <= 0:
            self.logger.warning(f"ATR is NaN/invalid (atr={atr}); proceeding with fallback (price * 0.02) for stops.")
            atr = price * 0.02

        if not (quality_ok and trend_prob_ok and bounce_ok and trend_dir != 0):
            return

        # Compute order prices
        stop_distance = self.params["stop_loss_atr"] * atr if atr else price * 0.02
        if trend_dir == 1:
            stop_loss = price - stop_distance
            take_profit = price + stop_distance * self.params["take_profit_rr"]
        else:
            stop_loss = price + stop_distance
            take_profit = price - stop_distance * self.params["take_profit_rr"]

        # Position sizing
        if time.time() - self.last_save_time > 30:
            self.balance_cache = self.client.get_available_balance()
            self.last_save_time = time.time()
        risk_amount = self.balance_cache * self.params["position_size_pct"]
        risk_per_unit = abs(price - stop_loss)
        qty = risk_amount / risk_per_unit if risk_per_unit > 0 else 0.0
        if qty <= 0:
            return

        self.logger.info(
            f"Placing order | side={'Buy' if trend_dir==1 else 'Sell'} qty={qty:.4f} "
            f"price={price:.6f} SL={stop_loss:.6f} TP={take_profit:.6f} "
            f"risk_amt={risk_amount:.2f} balance={self.balance_cache:.2f}"
        )

        # Place order
        side = "Buy" if trend_dir == 1 else "Sell"
        result = self.client.open_position(
            symbol=self.symbol,
            side=side,
            qty=qty,
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=self.params["leverage"],
        )
        if result.success:
            self.position_open = True
            self.logger.info(
                f"Opened {quality}-grade {side} | qty={qty:.4f} price={price:.6f} "
                f"SL={stop_loss:.6f} TP={take_profit:.6f} trend_prob={trend_prob:.2f} bounce_prob={bounce_prob:.2f}"
            )
        else:
            self.logger.error(f"Order failed: {result.error_message}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_timeframes(tf_str: str) -> List[int]:
    return [int(x.strip()) for x in tf_str.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Bybit live trader for TrendFollower (real funds).")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory containing trained models")
    parser.add_argument("--symbol", type=str, default="MONUSDT", help="Trading symbol")
    parser.add_argument(
        "--timeframes",
        type=str,
        default="60,300,900,3600,14400",
        help="Comma-separated timeframe sizes in seconds (default: 1m,5m,15m,1h,4h)",
    )
    parser.add_argument(
        "--timeframe-names",
        type=str,
        default="1m,5m,15m,1h,4h",
        help="Comma-separated names matching the timeframes list",
    )
    parser.add_argument("--base-tf", type=str, default="5m", help="Base timeframe name used for entries (must match names list)")
    parser.add_argument("--testnet", action="store_true", help="Use Bybit testnet (recommended)")
    parser.add_argument("--save-trades", type=str, default=None, help="Path to append streamed trades for restart (CSV)")
    parser.add_argument("--update-interval", type=float, default=1.0, help="Seconds between loop iterations")
    parser.add_argument("--warmup-trades", type=int, default=1000, help="Trades to accumulate before first prediction")

    # Trading params (must mirror backtest)
    parser.add_argument("--position-size-pct", type=float, default=0.02, help="Risk per trade as fraction of balance (default 0.02)")
    parser.add_argument("--stop-loss-atr", type=float, default=1.0, help="Stop loss in ATR multiples (default 1.0)")
    parser.add_argument("--take-profit-rr", type=float, default=2.0, help="Take profit R:R (default 2.0)")
    parser.add_argument("--min-quality", type=str, default="B", choices=["A", "B", "C"], help="Minimum signal quality (default B)")
    parser.add_argument("--min-trend-prob", type=float, default=0.5, help="Minimum trend probability (default 0.5)")
    parser.add_argument("--min-bounce-prob", type=float, default=0.5, help="Minimum bounce probability (default 0.5)")
    parser.add_argument("--leverage", type=int, default=1, help="Leverage to set on orders (default 1)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    timeframes = parse_timeframes(args.timeframes)
    tf_names = [n.strip() for n in args.timeframe_names.split(",")]
    if len(timeframes) != len(tf_names):
        raise ValueError("timeframes and timeframe-names must have the same length")
    if args.base_tf not in tf_names:
        raise ValueError("base-tf must be one of timeframe-names")

    params = {
        "position_size_pct": args.position_size_pct,
        "stop_loss_atr": args.stop_loss_atr,
        "take_profit_rr": args.take_profit_rr,
        "min_quality": args.min_quality,
        "min_trend_prob": args.min_trend_prob,
        "min_bounce_prob": args.min_bounce_prob,
        "leverage": args.leverage,
    }

    trader = BybitLiveTrader(
        model_dir=Path(args.model_dir),
        symbol=args.symbol,
        timeframes=timeframes,
        timeframe_names=tf_names,
        base_tf=args.base_tf,
        testnet=args.testnet,
        params=params,
        save_trades_path=Path(args.save_trades) if args.save_trades else None,
        update_interval=args.update_interval,
        warmup_trades=args.warmup_trades,
    )

    trader.run()


if __name__ == "__main__":
    main()
