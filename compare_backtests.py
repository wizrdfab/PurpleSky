import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from config import DEFAULT_CONFIG
from train import _load_train_config
from models import TrendFollowerModels
from data_loader import load_trades, preprocess_trades, create_multi_timeframe_bars
from feature_engine import calculate_multi_timeframe_features
from labels import create_training_dataset
from incremental_features import IncrementalFeatureEngine
from backtest import SimpleBacktester


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare backtest results between full and incremental features."
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
    return parser.parse_args()


def _build_feature_matrix(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Build model input matrix matching backtest behavior (fill missing with 0)."""
    data: Dict[str, np.ndarray] = {}
    n = len(df)
    for col in feature_cols:
        if col in df.columns:
            data[col] = df[col].fillna(0).values
        else:
            data[col] = np.zeros(n, dtype=float)
    return pd.DataFrame(data, index=df.index)


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


def _summarize_deltas(full, inc) -> None:
    print("METRIC DELTAS (INCREMENTAL - FULL)")
    print(f"  total_trades: {inc.total_trades - full.total_trades}")
    print(f"  win_rate: {(inc.win_rate - full.win_rate):.6f}")
    print(f"  total_pnl: {(inc.total_pnl - full.total_pnl):.2f}")
    print(f"  total_pnl_percent: {(inc.total_pnl_percent - full.total_pnl_percent):.6f}")
    print(f"  profit_factor: {(inc.profit_factor - full.profit_factor):.6f}")
    print(f"  max_drawdown: {(inc.max_drawdown - full.max_drawdown):.2f}")
    print(f"  max_drawdown_percent: {(inc.max_drawdown_percent - full.max_drawdown_percent):.6f}")
    print(f"  sharpe_ratio: {(inc.sharpe_ratio - full.sharpe_ratio):.6f}")
    print(f"  avg_win: {(inc.avg_win - full.avg_win):.2f}")
    print(f"  avg_loss: {(inc.avg_loss - full.avg_loss):.2f}")
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
            f"bounce_prob={trade.bounce_prob:.4f} quality={trade.signal_quality} "
            f"exit_reason={trade.exit_reason} exit_time={exit_time}"
        )
    print("")


def _summarize_bounce_diffs(full_probs: np.ndarray, inc_probs: np.ndarray) -> None:
    diffs = np.abs(full_probs - inc_probs)
    print("BOUNCE PROBABILITY DRIFT (per-bar)")
    print(f"  max_abs_diff: {float(np.max(diffs)):.6f}")
    print(f"  mean_abs_diff: {float(np.mean(diffs)):.6f}")
    print(f"  median_abs_diff: {float(np.median(diffs)):.6f}")
    print(f"  p95_abs_diff: {float(np.percentile(diffs, 95)):.6f}")
    for threshold in (1e-6, 1e-4, 1e-3, 1e-2):
        count = int(np.sum(diffs > threshold))
        print(f"  > {threshold:.0e}: {count}")
    print("")


def main() -> None:
    args = _parse_args()
    model_dir = Path(args.model_dir)

    cfg = _load_train_config(model_dir) or DEFAULT_CONFIG
    cfg.model.model_dir = model_dir

    if args.data_dir:
        cfg.data.data_dir = Path(args.data_dir)
    if args.lookback_days is not None:
        cfg.data.lookback_days = int(args.lookback_days)

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
    trades = load_trades(cfg.data, verbose=False)
    trades = preprocess_trades(trades, cfg.data)
    print(f"Trades loaded: {len(trades):,}")

    print("Creating bars...")
    bars = create_multi_timeframe_bars(
        trades,
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

    print("Calculating incremental features...")
    engine = IncrementalFeatureEngine(
        cfg.features,
        base_tf=base_tf,
        pullback_ema=cfg.labels.pullback_ema,
    )

    for tf_name in cfg.features.timeframe_names:
        if tf_name == base_tf:
            continue
        tf_bars = bars[tf_name].sort_values("bar_time")
        for _, row in tf_bars.iterrows():
            engine.update_timeframe(tf_name, row.to_dict())

    records = []
    base_bars = bars[base_tf].sort_values("bar_time")
    for _, row in base_bars.iterrows():
        records.append(engine.update_base_tf_bar(row.to_dict()))

    inc_df = pd.DataFrame(records)
    inc_test = pd.merge(test[["bar_time"]], inc_df, on="bar_time", how="left")

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
        raw_trades=trades,
        use_calibration=bool(args.use_calibration),
    )

    def run_backtest(data: pd.DataFrame):
        bt = SimpleBacktester(models, cfg, **backtest_kwargs)
        return bt.run(data, feature_cols)

    print("Running backtest (full features)...")
    res_full = run_backtest(test)
    print("Running backtest (incremental features)...")
    res_inc = run_backtest(inc_test)

    print("")
    _summarize_backtest("FULL", res_full)
    _summarize_backtest("INCREMENTAL", res_inc)
    _summarize_deltas(res_full, res_inc)

    # Trade comparisons
    full_by_key = {_trade_key(t): t for t in res_full.trades}
    inc_by_key = {_trade_key(t): t for t in res_inc.trades}

    common_keys = sorted(set(full_by_key) & set(inc_by_key))
    only_full_keys = sorted(set(full_by_key) - set(inc_by_key))
    only_inc_keys = sorted(set(inc_by_key) - set(full_by_key))

    print("TRADE OVERLAP")
    print(f"  common: {len(common_keys)}")
    print(f"  only_full: {len(only_full_keys)}")
    print(f"  only_incremental: {len(only_inc_keys)}")
    print("")

    if common_keys:
        diffs = []
        quality_mismatch = 0
        exit_reason_mismatch = 0
        for key in common_keys:
            t_full = full_by_key[key]
            t_inc = inc_by_key[key]
            diffs.append(abs(float(t_full.bounce_prob) - float(t_inc.bounce_prob)))
            if t_full.signal_quality != t_inc.signal_quality:
                quality_mismatch += 1
            if t_full.exit_reason != t_inc.exit_reason:
                exit_reason_mismatch += 1

        diffs_arr = np.array(diffs) if diffs else np.array([0.0])
        print("OVERLAPPING TRADE DIFFS")
        print(f"  bounce_prob max_abs_diff: {float(diffs_arr.max()):.6f}")
        print(f"  bounce_prob mean_abs_diff: {float(diffs_arr.mean()):.6f}")
        print(f"  quality_mismatch: {quality_mismatch}")
        print(f"  exit_reason_mismatch: {exit_reason_mismatch}")
        print("")

    if only_full_keys:
        _print_trades("ONLY FULL", [full_by_key[k] for k in only_full_keys])

    if only_inc_keys:
        _print_trades("ONLY INCREMENTAL", [inc_by_key[k] for k in only_inc_keys])

    # Per-bar bounce probability comparisons
    X_full = _build_feature_matrix(test, feature_cols)
    X_inc = _build_feature_matrix(inc_test, feature_cols)

    pred_full = models.entry_model.predict(X_full, use_calibration=bool(args.use_calibration))
    pred_inc = models.entry_model.predict(X_inc, use_calibration=bool(args.use_calibration))

    full_probs = np.asarray(pred_full["bounce_prob"], dtype=float)
    inc_probs = np.asarray(pred_inc["bounce_prob"], dtype=float)

    if len(full_probs) != len(inc_probs):
        print("WARNING: bounce_prob arrays differ in length.")
    else:
        _summarize_bounce_diffs(full_probs, inc_probs)


if __name__ == "__main__":
    main()
