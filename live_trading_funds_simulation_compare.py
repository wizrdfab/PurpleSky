import argparse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import DEFAULT_CONFIG
from train import _load_train_config
from models import TrendFollowerModels
from data_loader import load_trades, preprocess_trades, create_multi_timeframe_bars
from feature_engine import calculate_multi_timeframe_features
from labels import create_training_dataset
from backtest import SimpleBacktester
from live_trading import LivePaperTrader, PaperPosition, CompletedTrade


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


class SimulatedFundsTrader(LivePaperTrader):
    """Offline replay of the live trading logic using incremental features."""

    def __init__(
        self,
        *args,
        max_entry_deviation_atr: float = 1.0,
        test_start_bar_time: Optional[int] = None,
        test_end_bar_time: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._max_entry_price_deviation_atr = max(0.0, float(max_entry_deviation_atr))
        self.test_start_bar_time = test_start_bar_time
        self.test_end_bar_time = test_end_bar_time

    @staticmethod
    def _bar_time_to_datetime(bar_time: Optional[int]) -> datetime:
        try:
            if bar_time is not None:
                return datetime.utcfromtimestamp(int(bar_time))
        except Exception:
            pass
        return datetime.utcfromtimestamp(0)

    def _check_entry(self, current_atr: float):
        if self._last_closed_bar_time is not None:
            if self.test_start_bar_time is not None and int(self._last_closed_bar_time) < int(self.test_start_bar_time):
                return
            if self.test_end_bar_time is not None and int(self._last_closed_bar_time) >= int(self.test_end_bar_time):
                return
        return super()._check_entry(current_atr)

    def _open_position(
        self,
        direction: int,
        quality: str,
        atr: float,
        entry_price: Optional[float] = None,
        expected_rr: Optional[float] = None,
    ):
        price = entry_price if entry_price is not None else self.current_price
        if self._max_entry_price_deviation_atr > 0 and atr > 0:
            diff_atr = abs(self.current_price - price) / atr
            if diff_atr > self._max_entry_price_deviation_atr:
                self.logger.debug(
                    "Skipping entry: signal price too far from live price "
                    f"(signal={price:.6f}, live={self.current_price:.6f}, diff_atr={diff_atr:.2f})."
                )
                return

        stop_dist = (self.stop_loss_atr * atr) + (self.stop_padding_pct * price)
        stop_loss = price - (direction * stop_dist)

        effective_rr = self.take_profit_rr
        if self.use_dynamic_rr and expected_rr is not None and expected_rr > 0.5:
            effective_rr = min(max(expected_rr, 0.5), 5.0)

        take_profit = price + (direction * stop_dist * effective_rr)

        risk_amount = self.capital * self.position_size_pct
        risk_per_unit = abs(price - stop_loss)
        size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0

        entry_time = self._bar_time_to_datetime(self._last_closed_bar_time)
        self.position = PaperPosition(
            entry_time=entry_time,
            direction=direction,
            entry_price=price,
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal_quality=quality,
            atr_at_entry=atr,
        )

        self.stats.positions_opened += 1
        self.stats.signals_generated += 1

        dir_name = "LONG" if direction == 1 else "SHORT"
        self.logger.info("=" * 70)
        self.logger.info(f"OPENED {quality}-grade {dir_name} POSITION (SIM)")
        self.logger.info(f"   Entry:      {price:.6f}")
        self.logger.info(
            f"   Stop Loss:  {stop_loss:.6f} ({self.stop_loss_atr} ATR + {self.stop_padding_pct*100:.3f}% pad)"
        )
        rr_note = " (dynamic)" if self.use_dynamic_rr and expected_rr else ""
        self.logger.info(f"   Take Profit:{take_profit:.6f} ({effective_rr:.2f}:1 R:R{rr_note})")
        self.logger.info(f"   Size:       {size:.2f} units (${risk_amount:.2f} risk)")
        self.logger.info("=" * 70)

    def _close_position(self, exit_price: float, exit_reason: str, exit_bar_time: Optional[int] = None):
        pos = self.position
        if pos is None:
            return

        exit_time = self._bar_time_to_datetime(exit_bar_time or self._last_closed_bar_time)
        pnl = pos.direction * (exit_price - pos.entry_price) * pos.size
        pnl_percent = pos.direction * (exit_price - pos.entry_price) / pos.entry_price * 100
        duration = (exit_time - pos.entry_time).total_seconds()

        trade = CompletedTrade(
            entry_time=pos.entry_time,
            exit_time=exit_time,
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

        self.completed_trades.append(trade)

        self.capital += pnl
        self.stats.positions_closed += 1
        self.stats.total_trades += 1
        self.stats.total_pnl += pnl
        self.stats.total_pnl_percent = (self.capital - self.initial_capital) / self.initial_capital * 100

        if pnl > 0:
            self.stats.winning_trades += 1
            self.stats.wins_by_grade[pos.signal_quality] += 1
        else:
            self.stats.losing_trades += 1

        self.stats.trades_by_grade[pos.signal_quality] += 1
        self.position = None

        result = "WIN" if pnl > 0 else "LOSS"
        dir_name = "LONG" if pos.direction == 1 else "SHORT"

        self.logger.info("=" * 70)
        self.logger.info(f"{result} - Closed {dir_name} ({exit_reason}) (SIM)")
        self.logger.info(f"   Entry:    {pos.entry_price:.6f}")
        self.logger.info(f"   Exit:     {exit_price:.6f}")
        self.logger.info(f"   P&L:      ${pnl:+.2f} ({pnl_percent:+.2f}%)")
        self.logger.info(f"   Duration: {duration:.0f}s")
        self.logger.info(f"   Capital:  ${self.capital:,.2f}")
        self.logger.info("-" * 70)
        self._log_running_stats()
        self.logger.info("=" * 70)

        if exit_reason == "stop_loss":
            self.last_stop_time = exit_time
            if exit_bar_time is not None and self.cooldown_bars_after_stop > 0:
                try:
                    self._next_entry_bar_time = int(exit_bar_time) + int(self.cooldown_bars_after_stop) * int(self.base_tf_seconds)
                except Exception:
                    self._next_entry_bar_time = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare full-feature backtest results with a live-trading style incremental replay.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/MONUSDT",
        help="Model directory with trained models and train_config.json.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data directory (default: from train_config.json).",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=None,
        help="Limit trades to the most recent N days (default: config value).",
    )
    parser.add_argument(
        "--stop-loss-atr",
        type=float,
        default=1.0,
        help="Stop loss in ATR units (default: 1.0).",
    )
    parser.add_argument(
        "--take-profit-rr",
        type=float,
        default=1.5,
        help="Take profit reward:risk ratio (default: 1.5).",
    )
    parser.add_argument(
        "--min-bounce-prob",
        type=float,
        default=0.48,
        help="Minimum bounce probability gate (default: 0.48).",
    )
    parser.add_argument(
        "--max-bounce-prob",
        type=float,
        default=1.0,
        help="Maximum bounce probability for bucket filtering (default: 1.0).",
    )
    parser.add_argument(
        "--touch-threshold-atr",
        type=float,
        default=0.3,
        help="EMA touch detection threshold in ATR units (default: 0.3).",
    )
    parser.add_argument(
        "--stop-padding-pct",
        type=float,
        default=0.0,
        help="Extra stop distance as a fraction of entry (default: 0.0).",
    )
    parser.add_argument(
        "--cooldown-bars-after-stop",
        type=int,
        default=0,
        help="Cooldown after a stop-loss in base bars (default: 0).",
    )
    parser.add_argument(
        "--trade-side",
        type=str,
        default="both",
        choices=["long", "short", "both"],
        help="Trade direction to allow (default: both).",
    )
    parser.add_argument(
        "--use-dynamic-rr",
        action="store_true",
        help="Use expected RR from model for dynamic TP sizing.",
    )
    parser.add_argument(
        "--use-calibration",
        action="store_true",
        help="Use calibrated probabilities if available.",
    )
    parser.add_argument(
        "--warmup-trades",
        type=int,
        default=1000,
        help="Trades to ingest before starting decisions (default: 1000).",
    )
    parser.add_argument(
        "--max-entry-deviation-atr",
        type=float,
        default=1.0,
        help="Skip entries when abs(live_price - signal_price)/ATR exceeds this (default: 1.0; 0 disables).",
    )
    return parser.parse_args()


def _trade_key(trade) -> Tuple:
    entry_time = trade.entry_time
    try:
        entry_time_key = entry_time.isoformat()
    except Exception:
        entry_time_key = str(entry_time)
    return (entry_time_key, int(trade.direction), round(float(trade.entry_price), 8))


def _summarize_backtest(label: str, result) -> None:
    print(f"{label} RESULTS")
    print(f"  total_trades: {result.total_trades}")
    print(f"  win_rate: {result.win_rate:.4f}")
    print(f"  total_pnl: {result.total_pnl:.2f}")
    print(f"  total_pnl_percent: {result.total_pnl_percent:.4f}")
    print(f"  profit_factor: {result.profit_factor:.4f}")
    print(f"  max_drawdown: {result.max_drawdown:.2f}")
    print(f"  max_drawdown_percent: {result.max_drawdown_percent:.4f}")
    print(f"  sharpe_ratio: {result.sharpe_ratio:.4f}")
    print(f"  avg_win: {result.avg_win:.2f}")
    print(f"  avg_loss: {result.avg_loss:.2f}")
    print("")


def _summarize_deltas(full, live) -> None:
    print("METRIC DELTAS (LIVE - FULL)")
    print(f"  total_trades: {live.total_trades - full.total_trades}")
    print(f"  win_rate: {(live.win_rate - full.win_rate):.6f}")
    print(f"  total_pnl: {(live.total_pnl - full.total_pnl):.2f}")
    print(f"  total_pnl_percent: {(live.total_pnl_percent - full.total_pnl_percent):.6f}")
    print(f"  profit_factor: {(live.profit_factor - full.profit_factor):.6f}")
    print(f"  max_drawdown: {(live.max_drawdown - full.max_drawdown):.2f}")
    print(f"  max_drawdown_percent: {(live.max_drawdown_percent - full.max_drawdown_percent):.6f}")
    print(f"  sharpe_ratio: {(live.sharpe_ratio - full.sharpe_ratio):.6f}")
    print(f"  avg_win: {(live.avg_win - full.avg_win):.2f}")
    print(f"  avg_loss: {(live.avg_loss - full.avg_loss):.2f}")
    print("")


def _print_trades(label: str, trades: Iterable) -> None:
    trade_list = list(trades)
    print(f"{label} TRADES ({len(trade_list)})")
    for trade in trade_list:
        entry_time = trade.entry_time.isoformat() if hasattr(trade.entry_time, "isoformat") else str(trade.entry_time)
        exit_time = trade.exit_time.isoformat() if hasattr(trade.exit_time, "isoformat") else str(trade.exit_time)
        print(
            f"  {entry_time} dir={trade.direction} entry={trade.entry_price:.8f} "
            f"exit={trade.exit_price:.8f} pnl={trade.pnl:.2f} "
            f"quality={trade.signal_quality} exit_reason={trade.exit_reason} exit_time={exit_time}"
        )
    print("")


def _build_sim_result(trader: SimulatedFundsTrader) -> SimResult:
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


def _run_live_replay(
    trader: SimulatedFundsTrader,
    trades: pd.DataFrame,
    cfg,
) -> SimResult:
    ts_col = cfg.data.timestamp_col
    price_col = cfg.data.price_col
    size_col = cfg.data.size_col
    side_col = cfg.data.side_col
    tick_col = cfg.data.tick_direction_col

    base_tf_seconds = trader.base_tf_seconds
    last_bar_time: Optional[int] = None
    trade_count = 0

    total_trades = len(trades)
    for row in trades.itertuples(index=False):
        ts = float(getattr(row, ts_col))
        price = float(getattr(row, price_col))
        size = float(getattr(row, size_col))
        side = getattr(row, side_col, "Buy")
        tick_dir = getattr(row, tick_col, "ZeroPlusTick")
        symbol = getattr(row, "symbol", trader.symbol)

        trade_msg = {
            "T": int(ts * 1000),
            "s": symbol,
            "S": side,
            "v": size,
            "p": price,
            "L": tick_dir,
        }
        trader.trade_buffer.add_trade(trade_msg)

        trader.current_price = price
        if trader.current_high <= 0:
            trader.current_high = price
        else:
            trader.current_high = max(trader.current_high, price)
        if trader.current_low == float("inf"):
            trader.current_low = price
        else:
            trader.current_low = min(trader.current_low, price)
        trader.last_trade_timestamp = ts

        trade_count += 1
        if trade_count < trader.warmup_trades:
            continue

        current_bar_time = int(ts // base_tf_seconds) * base_tf_seconds
        if last_bar_time is None:
            last_bar_time = current_bar_time
        if current_bar_time > last_bar_time:
            last_bar_time = current_bar_time
            trader._trading_tick()

        if trade_count > 0 and trade_count % 250000 == 0:
            print(f"  Processed {trade_count:,}/{total_trades:,} trades...")

    # Flush the final bar so exits match backtest semantics.
    if trade_count >= trader.warmup_trades and trader.last_trade_timestamp is not None:
        trader.last_trade_timestamp = trader.last_trade_timestamp + trader.base_tf_seconds
        trader._trading_tick()

    return _build_sim_result(trader)


def main() -> None:
    args = _parse_args()
    model_dir = Path(args.model_dir)

    cfg = _load_train_config(model_dir) or DEFAULT_CONFIG
    cfg.model.model_dir = model_dir

    if args.data_dir:
        cfg.data.data_dir = Path(args.data_dir)
    if args.lookback_days is not None:
        cfg.data.lookback_days = int(args.lookback_days)
    cfg.labels.touch_threshold_atr = float(args.touch_threshold_atr)

    base_tf = cfg.features.timeframe_names[cfg.base_timeframe_idx]

    print(f"Model dir: {model_dir}")
    print(f"Data dir: {cfg.data.data_dir}")
    print(f"Base TF: {base_tf}")
    print(f"Lookback days: {cfg.data.lookback_days}")
    print("")

    print("Loading models...")
    models = TrendFollowerModels(cfg.model)
    models.load_all(model_dir)

    print("Loading trades...")
    raw_trades = load_trades(cfg.data, verbose=False)
    raw_trades = raw_trades.sort_values(cfg.data.timestamp_col).reset_index(drop=True)
    print(f"Trades loaded: {len(raw_trades):,}")

    print("Creating bars for full-feature backtest...")
    processed = preprocess_trades(raw_trades, cfg.data)
    bars = create_multi_timeframe_bars(
        processed,
        cfg.features.timeframes,
        cfg.features.timeframe_names,
        cfg.data,
    )

    print("Calculating full features...")
    featured = calculate_multi_timeframe_features(bars, base_tf, cfg.features)

    print("Labeling to get test split...")
    labeled, feature_cols = create_training_dataset(featured, cfg.labels, cfg.features, base_tf)

    start = int(len(labeled) * (cfg.model.train_ratio + cfg.model.val_ratio))
    test = labeled.iloc[start:].copy()
    print(f"Test bars: {len(test):,} (start index {start})")
    print(f"Feature count: {len(feature_cols)}")
    print("")

    test_start_bar_time = int(test["bar_time"].iloc[0]) if len(test) else None
    test_end_bar_time = int(test["bar_time"].iloc[-1]) if len(test) else None

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
        touch_threshold_atr=float(args.touch_threshold_atr),
        raw_trades=processed,
        use_calibration=bool(args.use_calibration),
    )

    def run_backtest(data: pd.DataFrame):
        bt = SimpleBacktester(models, cfg, **backtest_kwargs)
        return bt.run(data, feature_cols)

    print("Running backtest (full features)...")
    res_full = run_backtest(test)

    print("Running simulated live replay (incremental features)...")
    symbol = None
    if "symbol" in raw_trades.columns and len(raw_trades) > 0:
        symbol = raw_trades["symbol"].iloc[0]
    sim_trader = SimulatedFundsTrader(
        model_dir=model_dir,
        symbol=symbol or "SIM",
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
        log_dir=Path("./live_results_sim"),
        bootstrap_csv=None,
        lookback_days=args.lookback_days,
        max_entry_deviation_atr=float(args.max_entry_deviation_atr),
        test_start_bar_time=test_start_bar_time,
        test_end_bar_time=test_end_bar_time,
    )
    sim_trader.config.labels.touch_threshold_atr = float(args.touch_threshold_atr)
    if sim_trader.predictor and sim_trader.predictor.incremental_engine:
        sim_trader.predictor.incremental_engine.touch_threshold_atr = float(args.touch_threshold_atr)

    res_live = _run_live_replay(sim_trader, raw_trades, cfg)

    print("")
    _summarize_backtest("FULL", res_full)
    _summarize_backtest("LIVE SIM", res_live)
    _summarize_deltas(res_full, res_live)

    full_by_key = {_trade_key(t): t for t in res_full.trades}
    live_by_key = {_trade_key(t): t for t in res_live.trades}

    common_keys = sorted(set(full_by_key) & set(live_by_key))
    only_full_keys = sorted(set(full_by_key) - set(live_by_key))
    only_live_keys = sorted(set(live_by_key) - set(full_by_key))

    print("TRADE OVERLAP")
    print(f"  common: {len(common_keys)}")
    print(f"  only_full: {len(only_full_keys)}")
    print(f"  only_live: {len(only_live_keys)}")
    print("")

    if common_keys:
        quality_mismatch = 0
        exit_reason_mismatch = 0
        for key in common_keys:
            t_full = full_by_key[key]
            t_live = live_by_key[key]
            if t_full.signal_quality != t_live.signal_quality:
                quality_mismatch += 1
            if t_full.exit_reason != t_live.exit_reason:
                exit_reason_mismatch += 1

        print("OVERLAPPING TRADE DIFFS")
        print(f"  quality_mismatch: {quality_mismatch}")
        print(f"  exit_reason_mismatch: {exit_reason_mismatch}")
        print("")

    if only_full_keys:
        _print_trades("ONLY FULL", [full_by_key[k] for k in only_full_keys])

    if only_live_keys:
        _print_trades("ONLY LIVE", [live_by_key[k] for k in only_live_keys])


if __name__ == "__main__":
    main()
