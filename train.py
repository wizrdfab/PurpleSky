import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Tuple, Dict, List
import numpy as np
from config import TrendFollowerConfig
from trainer import run_training_pipeline, time_series_split
from models import TrendFollowerModels, EntryQualityModel, TrendClassifier, RegimeClassifier, append_context_features
from data_loader import load_trades, preprocess_trades, create_multi_timeframe_bars
from feature_engine import calculate_multi_timeframe_features, get_feature_columns
from labels import create_training_dataset
from backtest_tuned_config import (
    build_dataset_from_config,
    print_backtest_results,
    run_tuned_backtest,
    save_backtest_logs,
)
from optimizer import run_optimization, TrendFollowerOptimizer, OptimizerConfig


_TRAIN_CONFIG_FILENAME = "train_config.json"
_TUNING_SUMMARY_PREFIX = "tuning_summary"
_TRAIN_CONFIG_PREFIX = "train_config"


def _normalize_config(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _normalize_config(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_config(v) for v in obj]
    return obj


def _serialize_config(cfg: TrendFollowerConfig) -> dict:
    return _normalize_config(asdict(cfg))


def _apply_config_section(target, data: dict, path_fields: Optional[set] = None) -> None:
    path_fields = path_fields or set()
    for key, value in data.items():
        if key in path_fields and value is not None:
            value = Path(value)
        setattr(target, key, value)


def _load_train_config(model_dir: Path) -> Optional[TrendFollowerConfig]:
    config_path = Path(model_dir) / _TRAIN_CONFIG_FILENAME
    if not config_path.exists():
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cfg = TrendFollowerConfig()
    if isinstance(data, dict):
        if "data" in data:
            _apply_config_section(cfg.data, data["data"], {"data_dir"})
        if "features" in data:
            _apply_config_section(cfg.features, data["features"])
        if "labels" in data:
            _apply_config_section(cfg.labels, data["labels"])
        if "model" in data:
            _apply_config_section(cfg.model, data["model"], {"model_dir"})
        if "base_timeframe_idx" in data:
            cfg.base_timeframe_idx = int(data["base_timeframe_idx"])
        if "seed" in data:
            cfg.seed = int(data["seed"])
    return cfg


def _load_train_config_from_path(config_path: Path) -> Optional[TrendFollowerConfig]:
    if not config_path.exists():
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cfg = TrendFollowerConfig()
    if isinstance(data, dict):
        if "data" in data:
            _apply_config_section(cfg.data, data["data"], {"data_dir"})
        if "features" in data:
            _apply_config_section(cfg.features, data["features"])
        if "labels" in data:
            _apply_config_section(cfg.labels, data["labels"])
        if "model" in data:
            _apply_config_section(cfg.model, data["model"], {"model_dir"})
        if "base_timeframe_idx" in data:
            cfg.base_timeframe_idx = int(data["base_timeframe_idx"])
        if "seed" in data:
            cfg.seed = int(data["seed"])
    return cfg


def _save_train_config(
    cfg: TrendFollowerConfig,
    model_dir: Path,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> None:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    config_path = model_dir / _TRAIN_CONFIG_FILENAME
    with open(config_path, "w", encoding="utf-8") as f:
        payload = _serialize_config(cfg)
        if extra_meta:
            for key, value in extra_meta.items():
                if key in payload:
                    continue
                payload[key] = _normalize_config(value)
        json.dump(payload, f, indent=2)


def _save_train_config_path(
    cfg: TrendFollowerConfig,
    config_path: Path,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> None:
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _serialize_config(cfg)
    if extra_meta:
        for key, value in extra_meta.items():
            if key in payload:
                continue
            payload[key] = _normalize_config(value)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_train_config_meta(model_dir: Path) -> Dict[str, Any]:
    config_path = Path(model_dir) / _TRAIN_CONFIG_FILENAME
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    meta = {}
    if "tuning_summary_path" in data:
        meta["tuning_summary_path"] = data["tuning_summary_path"]
    if "entry_feature_readiness" in data:
        meta["entry_feature_readiness"] = data["entry_feature_readiness"]
    return meta


def _load_train_config_meta_from_path(config_path: Path) -> Dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    meta = {}
    if "tuning_summary_path" in data:
        meta["tuning_summary_path"] = data["tuning_summary_path"]
    if "entry_feature_readiness" in data:
        meta["entry_feature_readiness"] = data["entry_feature_readiness"]
    return meta


def _parse_trial_number_from_path(path: Path, prefix: str) -> Optional[int]:
    stem = Path(path).stem
    prefix = prefix.rstrip("_")
    if not stem.startswith(prefix + "_"):
        return None
    suffix = stem[len(prefix) + 1:]
    if not suffix.isdigit():
        return None
    try:
        return int(suffix)
    except Exception:
        return None


def _generate_study_name() -> str:
    adjectives = [
        "oblivious",
        "silent",
        "steady",
        "curious",
        "brisk",
        "calm",
        "lucid",
        "sable",
        "ivory",
        "gentle",
    ]
    animals = [
        "tripus",
        "otter",
        "hawk",
        "lynx",
        "finch",
        "manta",
        "yak",
        "puma",
        "ibis",
        "orca",
    ]
    rng = random.Random(time.time_ns())
    return f"{rng.choice(adjectives)}_{rng.choice(animals)}"


def _list_study_dirs(base_dir: Path) -> List[Path]:
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return []
    return sorted([p for p in base_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower())


def _prompt_for_study_and_trial(base_dir: Path, prefix: str) -> Tuple[str, int, Path]:
    base_dir = Path(base_dir)
    studies = _list_study_dirs(base_dir)
    if studies:
        print("Available studies:")
        for study in studies:
            print(f"  - {study.name}")
    study_name = input("Study name: ").strip()
    if not study_name:
        raise SystemExit("Study name is required.")
    trial_raw = input("Trial number: ").strip()
    if not trial_raw:
        raise SystemExit("Trial number is required.")
    try:
        trial_number = int(trial_raw)
    except Exception:
        raise SystemExit(f"Invalid trial number: {trial_raw}")
    file_name = f"{prefix}_{trial_number}.json"
    return study_name, trial_number, base_dir / study_name / file_name


def _arg_present(flag: str) -> bool:
    return any(arg == flag or arg.startswith(flag + "=") for arg in sys.argv)


def _get_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    if isinstance(value, (int, float)):
        return bool(float(value))
    return bool(value)


def _get_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _get_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _load_tuning_summary(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _apply_summary_to_config(cfg: TrendFollowerConfig, summary: dict) -> None:
    best_config = summary.get("best_config")
    if not isinstance(best_config, dict):
        return
    if "features" in best_config:
        _apply_config_section(cfg.features, best_config["features"])
    if "labels" in best_config:
        _apply_config_section(cfg.labels, best_config["labels"])
    if "model" in best_config:
        _apply_config_section(cfg.model, best_config["model"], {"model_dir"})
    if "base_timeframe_idx" in best_config:
        cfg.base_timeframe_idx = int(best_config["base_timeframe_idx"])
    if "seed" in best_config:
        cfg.seed = int(best_config["seed"])


def _extract_tuned_settings(summary: dict) -> Dict[str, Any]:
    best_config = summary.get("best_config", {}) if isinstance(summary, dict) else {}
    labels = best_config.get("labels", {}) if isinstance(best_config, dict) else {}
    metrics = summary.get("best_metrics", {}) if isinstance(summary, dict) else {}

    tuned_use_raw = _get_bool(metrics.get("use_raw_probabilities", labels.get("use_raw_probabilities")), False)
    tuned_use_cal = _get_bool(labels.get("use_calibration"), not tuned_use_raw)
    if tuned_use_raw:
        tuned_use_cal = False

    return {
        "best_threshold": _get_float(metrics.get("best_threshold", labels.get("best_threshold", 0.5)), 0.5),
        "stop_atr_multiple": _get_float(labels.get("stop_atr_multiple", metrics.get("stop_atr_multiple", 1.0)), 1.0),
        "target_rr": _get_float(labels.get("target_rr", metrics.get("target_rr", 1.5)), 1.5),
        "pullback_threshold": _get_float(labels.get("pullback_threshold", metrics.get("pullback_threshold", 0.3)), 0.3),
        "entry_forward_window": _get_int(labels.get("entry_forward_window", metrics.get("entry_forward_window", 0)), 0),
        "ev_margin_r": _get_float(labels.get("ev_margin_r", metrics.get("ev_margin_r", 0.0)), 0.0),
        "fee_percent": _get_float(labels.get("fee_percent", metrics.get("fee_percent", 0.0011)), 0.0011),
        "fee_per_trade_r": metrics.get("fee_per_trade_r", labels.get("fee_per_trade_r")),
        "use_expected_rr": _get_bool(labels.get("use_expected_rr", metrics.get("use_expected_rr", 0.0)), False),
        "use_ev_gate": _get_bool(labels.get("use_ev_gate", metrics.get("use_ev_gate", 1.0)), True),
        "use_trend_gate": _get_bool(labels.get("use_trend_gate", metrics.get("use_trend_gate", 0.0)), False),
        "min_trend_prob": _get_float(labels.get("min_trend_prob", metrics.get("min_trend_prob", 0.0)), 0.0),
        "use_regime_gate": _get_bool(labels.get("use_regime_gate", metrics.get("use_regime_gate", 0.0)), False),
        "min_regime_prob": _get_float(labels.get("min_regime_prob", metrics.get("min_regime_prob", 0.0)), 0.0),
        "allow_regime_ranging": _get_bool(labels.get("allow_regime_ranging", True), True),
        "allow_regime_trend_up": _get_bool(labels.get("allow_regime_trend_up", True), True),
        "allow_regime_trend_down": _get_bool(labels.get("allow_regime_trend_down", True), True),
        "allow_regime_volatile": _get_bool(labels.get("allow_regime_volatile", True), True),
        "regime_align_direction": _get_bool(labels.get("regime_align_direction", True), True),
        "use_raw_probabilities": tuned_use_raw,
        "use_calibration": tuned_use_cal,
        "ops_cost_enabled": _get_bool(metrics.get("ops_cost_enabled", 1.0), True),
        "ops_cost_target_trades_per_day": _get_float(metrics.get("ops_cost_target_trades_per_day", 30.0), 30.0),
        "ops_cost_c1": _get_float(metrics.get("ops_cost_c1", 0.01), 0.01),
        "ops_cost_alpha": _get_float(metrics.get("ops_cost_alpha", 1.7), 1.7),
    }


def _confirm_backtest_overrides(mismatches: List[Dict[str, Any]]) -> None:
    if not mismatches:
        return
    print("\n" + "!" * 70)
    print("WARNING: Backtest overrides differ from tuning summary:")
    for item in mismatches:
        name = item.get("name", "unknown")
        tuned = item.get("tuned")
        cli = item.get("cli")
        note = item.get("note", "")
        print(f"  - {name}: tuned={tuned} cli={cli} {note}".rstrip())
    print("!" * 70)
    if not sys.stdin.isatty():
        raise SystemExit("Backtest overrides require confirmation in interactive mode.")
    resp = input("Proceed with overrides? [y/N]: ").strip().lower()
    if resp not in {"y", "yes"}:
        raise SystemExit("Backtest aborted by user.")


def _binary_metrics_from_probs(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    if probs.size == 0:
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}
    preds = (probs >= threshold).astype(int)
    accuracy = float((preds == labels).mean())
    pred_pos = preds == 1
    if pred_pos.sum() == 0:
        precision = 0.0
    else:
        precision = float((labels[pred_pos] == 1).mean())
    actual_pos = labels == 1
    if actual_pos.sum() == 0:
        recall = 0.0
    else:
        recall = float((preds[actual_pos] == 1).mean())
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall}


def _evaluate_entry_holdout(
    trades,
    cfg: TrendFollowerConfig,
    use_noise_filtering: bool,
    use_seed_ensemble: bool,
    n_ensemble_seeds: int,
    use_rust_pipeline: bool = True,
    rust_cache_dir: str = "rust_cache",
    rust_write_intermediate: bool = False,
) -> Dict[str, float]:
    base_tf = cfg.features.timeframe_names[cfg.base_timeframe_idx]
    labeled = None
    feature_cols = None

    if use_rust_pipeline:
        try:
            import rust_pipeline_bridge as rust_bridge  # type: ignore
            if rust_bridge.is_available():
                labeled, feature_cols, _dataset_path = rust_bridge.build_dataset_from_config(
                    cfg,
                    cache_dir=rust_cache_dir,
                    write_intermediate=rust_write_intermediate,
                    force=False,
                )
        except Exception:
            labeled = None
            feature_cols = None

    if labeled is None or feature_cols is None:
        if trades is None:
            return {'skipped': True, 'reason': 'no_trades_for_python_pipeline'}
        bars = create_multi_timeframe_bars(
            trades,
            cfg.features.timeframes,
            cfg.features.timeframe_names,
            cfg.data,
        )
        featured = calculate_multi_timeframe_features(bars, base_tf, cfg.features)
        labeled, feature_cols = create_training_dataset(featured, cfg.labels, cfg.features, base_tf)

    train_df, val_df, test_df = time_series_split(
        labeled,
        train_ratio=float(cfg.model.train_ratio),
        val_ratio=float(cfg.model.val_ratio),
        test_ratio=float(cfg.model.test_ratio),
    )

    if len(test_df) == 0:
        return {'skipped': True, 'reason': 'no_test_split'}

    pullback_mask_train = ~train_df['pullback_success'].isna()
    pullback_mask_val = ~val_df['pullback_success'].isna()
    pullback_mask_test = ~test_df['pullback_success'].isna()

    def _slice_pred(pred, mask):
        if pred is None:
            return None
        mask_arr = np.asarray(mask, dtype=bool)
        sliced = {}
        for key, values in pred.items():
            arr = np.asarray(values)
            if arr.shape[0] == mask_arr.shape[0]:
                sliced[key] = arr[mask_arr]
        return sliced

    if pullback_mask_train.sum() < 10 or pullback_mask_test.sum() < 5:
        return {
            'skipped': True,
            'reason': 'insufficient_pullbacks',
            'train_pullbacks': float(pullback_mask_train.sum()),
            'test_pullbacks': float(pullback_mask_test.sum()),
        }

    X_train = train_df[feature_cols].fillna(0)
    X_val = val_df[feature_cols].fillna(0) if len(val_df) else None
    X_test = test_df[feature_cols].fillna(0)

    trend_pred_train = None
    trend_pred_val = None
    trend_pred_test = None
    if 'trend_label' in train_df.columns and 'trend_label' in val_df.columns:
        trend_model = TrendClassifier(cfg.model)
        trend_model.train(
            X_train,
            train_df['trend_label'],
            X_val,
            val_df['trend_label'],
            verbose=False,
        )
        trend_pred_train = trend_model.predict(X_train)
        trend_pred_val = trend_model.predict(X_val) if X_val is not None else None
        trend_pred_test = trend_model.predict(X_test)

    regime_pred_train = None
    regime_pred_val = None
    regime_pred_test = None
    if 'regime' in train_df.columns and 'regime' in val_df.columns and train_df['regime'].nunique() >= 2:
        regime_model = RegimeClassifier(cfg.model)
        regime_model.train(
            X_train,
            train_df['regime'],
            X_val,
            val_df['regime'],
            verbose=False,
        )
        regime_pred_train = regime_model.predict(X_train)
        regime_pred_val = regime_model.predict(X_val) if X_val is not None else None
        regime_pred_test = regime_model.predict(X_test)

    rr_col = 'pullback_win_r' if 'pullback_win_r' in train_df.columns else 'pullback_rr'
    y_rr_train = train_df.loc[pullback_mask_train, rr_col]
    y_success_train = train_df.loc[pullback_mask_train, 'pullback_success'].astype(int)
    X_entry_train = append_context_features(
        X_train[pullback_mask_train],
        _slice_pred(trend_pred_train, pullback_mask_train),
        _slice_pred(regime_pred_train, pullback_mask_train),
    )

    has_val_pullbacks = len(val_df) > 0 and pullback_mask_val.sum() > 0 and X_val is not None
    X_entry_val = (
        append_context_features(
            X_val[pullback_mask_val],
            _slice_pred(trend_pred_val, pullback_mask_val),
            _slice_pred(regime_pred_val, pullback_mask_val),
        )
        if has_val_pullbacks
        else None
    )
    y_success_val = val_df.loc[pullback_mask_val, 'pullback_success'].astype(int) if has_val_pullbacks else None
    rr_col_val = 'pullback_win_r' if 'pullback_win_r' in val_df.columns else 'pullback_rr'
    y_rr_val = val_df.loc[pullback_mask_val, rr_col_val] if has_val_pullbacks else None

    entry_model = EntryQualityModel(cfg.model)
    entry_model.train(
        X_entry_train,
        y_success_train,
        y_rr_train,
        X_entry_val,
        y_success_val,
        y_rr_val,
        verbose=False,
        calibrate=True,
        use_noise_filtering=use_noise_filtering,
        use_seed_ensemble=use_seed_ensemble,
        n_ensemble_seeds=n_ensemble_seeds,
    )

    X_entry_test = append_context_features(
        X_test[pullback_mask_test],
        _slice_pred(trend_pred_test, pullback_mask_test),
        _slice_pred(regime_pred_test, pullback_mask_test),
    )
    y_success_test = test_df.loc[pullback_mask_test, 'pullback_success'].astype(int)

    preds = entry_model.predict(X_entry_test, use_calibration=True)
    raw_probs = np.asarray(preds.get('bounce_prob_raw', preds.get('bounce_prob', [])), dtype=float)
    cal_probs = np.asarray(preds.get('bounce_prob', raw_probs), dtype=float)

    raw_metrics = _binary_metrics_from_probs(raw_probs, y_success_test.values)
    cal_metrics = _binary_metrics_from_probs(cal_probs, y_success_test.values)

    return {
        'test_pullbacks': float(pullback_mask_test.sum()),
        'test_base_rate': float(y_success_test.mean()),
        'test_accuracy_raw': float(raw_metrics['accuracy']),
        'test_precision_raw': float(raw_metrics['precision']),
        'test_recall_raw': float(raw_metrics['recall']),
        'test_accuracy_cal': float(cal_metrics['accuracy']),
        'test_precision_cal': float(cal_metrics['precision']),
        'test_recall_cal': float(cal_metrics['recall']),
    }


def main():
    parser = argparse.ArgumentParser()

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        '--backtest-only',
        action='store_true',
        help='Skip training and only run the backtest using existing saved models.',
    )
    mode.add_argument(
        '--train-only',
        action='store_true',
        help='Train models only (skip the backtest).',
    )
    parser.add_argument(
        '--backtest-mode',
        choices=['sim', 'tuning'],
        default='tuning',
        help='Backtest mode: tuning (tuning-aligned). sim is deprecated and ignored.',
    )
    parser.add_argument(
        '--backtest-tuning-summary',
        default=None,
        help='Path to tuning_summary JSON for tuning-style backtest.',
    )
    parser.add_argument(
        '--backtest-train-config',
        default=None,
        help='Path to train_config JSON for backtest-only runs.',
    )
    parser.add_argument(
        '--two-pass',
        action='store_true',
        help='Two-pass training: train with validation (early stopping), then retrain on Train+Val with best iterations.',
    )
    parser.add_argument(
        '--stop-loss-atr',
        type=float,
        default=1.0,
        help='Stop loss in ATR units. Uses tuned stop_atr_multiple from config if default.',
    )
    parser.add_argument(
        '--stop-padding-pct',
        type=float,
        default=0.0,
        help='Extra stop distance as a fraction of entry price (default: 0.0 = disabled).',
    )
    parser.add_argument(
        '--take-profit-rr',
        type=float,
        default=1.5,
        help='Take profit reward:risk ratio. Uses tuned target_rr from config if default.',
    )
    parser.add_argument(
        '--min-bounce-prob',
        type=float,
        default=0.48,
        help='Minimum bounce probability gate. Uses tuned best_threshold from config if default.',
    )
    parser.add_argument(
        '--min-trend-prob',
        type=float,
        default=None,
        help='Minimum trend probability gate (default: use tuned value).',
    )
    trend_gate_group = parser.add_mutually_exclusive_group()
    trend_gate_group.add_argument(
        '--use-trend-gate',
        action='store_true',
        help='Enable trend classifier gating for backtest.',
    )
    trend_gate_group.add_argument(
        '--no-trend-gate',
        action='store_true',
        help='Disable trend classifier gating for backtest.',
    )
    parser.add_argument(
        '--min-regime-prob',
        type=float,
        default=None,
        help='Minimum regime probability gate (default: use tuned value).',
    )
    regime_gate_group = parser.add_mutually_exclusive_group()
    regime_gate_group.add_argument(
        '--use-regime-gate',
        action='store_true',
        help='Enable regime classifier gating for backtest.',
    )
    regime_gate_group.add_argument(
        '--no-regime-gate',
        action='store_true',
        help='Disable regime classifier gating for backtest.',
    )
    parser.add_argument(
        '--max-bounce-prob',
        type=float,
        default=1.0,
        help='Maximum bounce probability for bucket filtering (default: 1.0 = no max).',
    )
    parser.add_argument(
        '--use-dynamic-rr',
        action='store_true',
        help='Use expected RR from model for dynamic TP sizing instead of fixed take-profit-rr.',
    )
    parser.add_argument(
        '--use-ev-gate',
        action='store_true',
        help='Use EV gating (expected_rr + costs) instead of min-bounce-prob threshold (default: on).',
    )
    parser.add_argument(
        '--no-ev-gate',
        action='store_true',
        help='Disable EV gating and fall back to min-bounce-prob threshold.',
    )
    parser.add_argument(
        '--backtest-ev-margin-r',
        type=float,
        default=None,
        help='Minimum EV margin in R units for backtest entry gating (default: use tuned value).',
    )
    parser.add_argument(
        '--backtest-fee-percent',
        type=float,
        default=None,
        help='Round-trip fee percent used for EV gating (default: use tuned value).',
    )
    parser.add_argument(
        '--backtest-no-expected-rr',
        action='store_true',
        help='Disable expected_rr in EV gating (use target_rr instead).',
    )
    parser.add_argument(
        '--touch-threshold-atr',
        type=float,
        default=0.3,
        help='EMA touch detection threshold in ATR units (default: 0.3).',
    )
    parser.add_argument(
        '--ema-touch-mode',
        type=str,
        default='multi',
        choices=['base', 'multi'],
        help='EMA touch detection mode for backtest (default: multi).',
    )
    parser.add_argument(
        '--backtest-ohlc-only',
        action='store_true',
        help='Disable intrabar TP/SL detection and use OHLC-only exits (for label alignment tests).',
    )
    parser.add_argument(
        '--backtest-max-hold-bars',
        type=int,
        default=None,
        help='Maximum holding bars for backtest (default: use tuned entry_forward_window).',
    )
    parser.add_argument(
        '--backtest-full-data',
        action='store_true',
        help='Run backtest on the full dataset (diagnostics only; deviates from tuning/test split).',
    )
    parser.add_argument(
        '--cooldown-bars-after-stop',
        type=int,
        default=0,
        help='Cooldown after a stop-loss in base bars (default: 0 = disabled).',
    )
    parser.add_argument(
        '--trade-side',
        type=str,
        default='both',
        choices=['long', 'short', 'both'],
        help='Trade direction to allow in backtest (default: both).',
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/MONUSDT',
        help="Directory containing the symbol's trade CSV files (default: data/MONUSDT).",
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models_ema9_touch',
        help='Directory to save/load the models.',
    )
    parser.add_argument(
        '--lookback-days',
        type=int,
        default=None,
        help='If set, only use the most recent N days of trades from the dataset.',
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=None,
        help='Train split ratio (fraction). Provide at least two ratios.',
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=None,
        help='Validation split ratio (fraction). Provide at least two ratios.',
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=None,
        help='Test split ratio (fraction). Provide at least two ratios.',
    )

    parser.add_argument('--learning-rate', type=float, default=None, help='LightGBM learning rate')
    parser.add_argument('--num-leaves', type=int, default=None, help='LightGBM num leaves')
    parser.add_argument('--n-estimators', type=int, default=None, help='LightGBM n estimators')
    parser.add_argument('--max-depth', type=int, default=None, help='LightGBM max depth')

    parser.add_argument('--feature-fraction', type=float, default=None, help='LightGBM feature fraction')
    parser.add_argument('--lambdaa-ele1', type=float, default=None, help='LightGBM lambda l1')
    parser.add_argument('--lambdaa-ele2', type=float, default=None, help='LightGBM lambda l2')
    parser.add_argument('--min-child-samples', type=int, default=None, help='LightGBM min child samples')

    # Custom optimizer
    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Run custom parameter optimizer to find best LightGBM params and min_bounce_prob.',
    )
    parser.add_argument(
        '--optimize-trials',
        type=int,
        default=30,
        help='Number of random search trials for optimization (default: 30).',
    )

    # =========================================================================
    # OPTUNA CONFIG TUNER
    # =========================================================================
    parser.add_argument(
        '--optuna-tune',
        action='store_true',
        help='Run Optuna tuning across config parameters to maximize robust profitability (or chosen objective).',
    )
    parser.add_argument(
        '--tune-scope',
        type=str,
        default='full',
        choices=['model', 'features', 'labels', 'full', 'all'],
        help='Which config sections to tune (default: full).',
    )
    parser.add_argument(
        '--tune-objective',
        type=str,
        default='profit',
        choices=['profit', 'precision', 'mixed'],
        help='Optimization objective for tuning: profit (Robust Profit Score), precision (Precision+Accuracy), or mixed (default: profit).',
    )
    parser.add_argument(
        '--tune-lgbm-only',
        action='store_true',
        help='Alias for --tune-scope model (only LightGBM parameters).',
    )
    parser.add_argument(
        '--tune-trials',
        type=int,
        default=200,
        help='Number of Optuna trials (default: 200).',
    )
    parser.add_argument(
        '--tune-parallel-jobs',
        type=int,
        default=1,
        help='Number of parallel Optuna jobs (default: 1).',
    )
    parser.add_argument(
        '--tune-storage',
        type=str,
        default=None,
        help='Optuna storage URL for shared RDB (e.g. sqlite:///optuna.db).',
    )
    parser.add_argument(
        '--tune-study-name',
        type=str,
        default=None,
        help='Optuna study name for shared storage.',
    )
    parser.add_argument(
        '--tune-storage-load-if-exists',
        action='store_true',
        help='Reuse an existing Optuna study when using --tune-storage.',
    )
    parser.add_argument(
        '--tune-timeout-min',
        type=float,
        default=None,
        help='Optional Optuna timeout in minutes (default: no timeout).',
    )
    parser.add_argument(
        '--tune-precision-weight',
        type=float,
        default=0.6,
        help='Weight for entry-model precision in the tuning objective (0-1).',
    )
    parser.add_argument(
        '--tune-trend-weight',
        type=float,
        default=0.0,
        help='Optional weight for trend validation accuracy in the objective (default: 0).',
    )
    parser.add_argument(
        '--tune-min-pullback-samples',
        type=int,
        default=100,
        help='Minimum pullback samples in train set to score a trial (default: 100).',
    )
    parser.add_argument(
        '--tune-min-pullback-val-samples',
        type=int,
        default=20,
        help='Minimum pullback samples in validation set to score a trial (default: 20).',
    )
    parser.add_argument(
        '--tune-min-trades',
        type=int,
        default=30,
        help='Minimum number of trades across folds to accept a trial (default: 30).',
    )
    parser.add_argument(
        '--tune-min-trades-fold',
        type=int,
        default=None,
        help='Minimum trades per fold (0 disables). Defaults to --tune-min-trades divided by folds.',
    )
    parser.add_argument(
        '--tune-min-coverage',
        type=float,
        default=0.0,
        help='Minimum trade coverage per fold (0-1, default: 0.0).',
    )
    parser.add_argument(
        '--tune-max-coverage',
        type=float,
        default=0.7,
        help='Maximum trade coverage per fold (0-1, default: 0.7).',
    )
    parser.add_argument(
        '--tune-ops-cost-target',
        type=float,
        default=30.0,
        help='Ops cost penalty target trades per day (default: 30).',
    )
    parser.add_argument(
        '--tune-ops-cost-c1',
        type=float,
        default=0.01,
        help='Ops cost penalty scale in R per trade (default: 0.01).',
    )
    parser.add_argument(
        '--tune-ops-cost-alpha',
        type=float,
        default=1.7,
        help='Ops cost penalty exponent (default: 1.7).',
    )
    parser.add_argument(
        '--tune-no-ops-cost',
        action='store_true',
        help='Disable ops cost penalty in tuning.',
    )
    parser.add_argument(
        '--tune-single-position',
        action='store_true',
        help='Enforce one position at a time during tuning evaluation (default).',
    )
    parser.add_argument(
        '--tune-multi-position',
        action='store_true',
        help='Allow overlapping positions during tuning evaluation (disables single-position).',
    )
    parser.add_argument(
        '--tune-opposite-policy',
        type=str,
        default='flip',
        choices=['ignore', 'close', 'flip'],
        help='When an opposite signal appears during an open trade: ignore, close, or flip (default: flip).',
    )
    parser.add_argument(
        '--tune-calibration-method',
        type=str,
        default='temperature',
        choices=['temperature', 'isotonic'],
        help='Calibration method for tuning (default: temperature).',
    )
    parser.add_argument(
        '--tune-fee-per-trade-r',
        type=float,
        default=None,
        help='Round-trip fee per trade in R units (overrides --tune-fee-percent).',
    )
    parser.add_argument(
        '--tune-fee-percent',
        type=float,
        default=0.0011,
        help='Round-trip fee as a decimal of price (default: 0.0011 = 0.11%%).',
    )
    parser.add_argument(
        '--tune-ev-margin-r',
        type=float,
        default=0.0,
        help='Minimum EV margin in R units to accept a trade (default: 0.0).',
    )
    parser.add_argument(
        '--tune-ev-margin-fixed',
        action='store_true',
        help='Use --tune-ev-margin-r as a fixed (non-swept) EV margin in tuning.',
    )
    parser.add_argument(
        '--tune-lcb-z',
        type=float,
        default=1.28,
        help='LCB z-score for mean R (default: 1.28 for ~90%% one-sided).',
    )
    parser.add_argument(
        '--tune-use-raw-probabilities',
        action='store_true',
        help='Use raw (uncalibrated) probabilities for tuning evaluation.',
    )
    parser.add_argument(
        '--tune-use-expected-rr',
        action='store_true',
        help='Enable expected_rr gating for EV (default: off).',
    )
    parser.add_argument(
        '--tune-no-expected-rr',
        action='store_true',
        help='Disable expected_rr gating; use target_rr only for EV gating.',
    )
    parser.add_argument(
        '--tune-no-prune',
        action='store_true',
        help='Disable Optuna early pruning (runs all folds for every trial).',
    )
    parser.add_argument(
        '--tune-seed',
        type=int,
        default=42,
        help='Random seed for Optuna sampler (default: 42).',
    )
    parser.add_argument(
        '--tune-save-results',
        type=str,
        default=None,
        help='Optional path to save tuning summary JSON (default: study-dir/tuning_summary_<trial>.json).',
    )
    parser.add_argument(
        '--train-from-tuning',
        nargs='?',
        const='__MODEL_DIR__',
        default=None,
        help='Train models from a tuning summary JSON (uses best_config). '
             'If no path is provided, you will be prompted for study/trial.',
    )
    parser.add_argument(
        '--tune-then-train',
        action='store_true',
        help='After tuning, train and save models using the best config.',
    )
    parser.add_argument(
        '--tune-no-progress',
        action='store_true',
        help='Disable Optuna progress bar.',
    )
    parser.add_argument(
        '--tune-no-trial-report',
        action='store_true',
        help='Disable per-trial fold/aggregate metrics reporting during tuning.',
    )

    # =========================================================================
    # MODEL IMPROVEMENT FLAGS (v1.2)
    # =========================================================================
    parser.add_argument(
        '--use-noise-filtering',
        action='store_true',
        help='Enable Noise Injection Feature Selection. Trains a quick model with a random '
             'noise column and removes features that rank below noise in importance.',
    )
    parser.add_argument(
        '--use-seed-ensemble',
        action='store_true',
        help='Enable Seed Ensembling (Bagging). Trains N LightGBM models with different '
             'random seeds and averages their predictions for more stable results.',
    )
    parser.add_argument(
        '--n-ensemble-seeds',
        type=int,
        default=5,
        help='Number of seeds for seed ensembling (default: 5). Only used if --use-seed-ensemble is set.',
    )

    # =========================================================================
    # PROBABILITY CALIBRATION FLAGS
    # =========================================================================
    parser.add_argument(
        '--use-calibration',
        action='store_true',
        help='Use calibrated probabilities (Isotonic Regression). This is now the DEFAULT behavior '
             'to match config_tuner.py. Use --use-raw-probabilities to disable.',
    )
    parser.add_argument(
        '--use-raw-probabilities',
        action='store_true',
        help='Use raw (uncalibrated) probabilities instead of calibrated. '
             'Note: Tuner uses calibration, so raw probs may give different threshold behavior.',
    )

    args = parser.parse_args()

    if args.tune_lgbm_only:
        args.tune_scope = 'model'

    # Validate mutually exclusive calibration flags
    if args.use_calibration and args.use_raw_probabilities:
        raise SystemExit("Cannot use both --use-calibration and --use-raw-probabilities. Choose one.")

    cfg = TrendFollowerConfig()
    if args.learning_rate is not None:
        cfg.model.learning_rate = args.learning_rate
    if args.num_leaves is not None:
        cfg.model.num_leaves = args.num_leaves
    if args.n_estimators is not None:
        cfg.model.n_estimators = args.n_estimators
    if args.max_depth is not None:
        cfg.model.max_depth = args.max_depth

    if args.feature_fraction is not None:
        cfg.model.feature_fraction = args.feature_fraction
    if args.lambdaa_ele1 is not None:
        cfg.model.lambdaa_ele1 = args.lambdaa_ele1
    if args.lambdaa_ele2 is not None:
        cfg.model.lambdaa_ele2 = args.lambdaa_ele2
    if args.min_child_samples is not None:
        cfg.model.min_child_samples = args.min_child_samples
        
    cfg.data.data_dir = Path(args.data_dir)
    cfg.model.model_dir = Path(args.model_dir)
    cfg.data.lookback_days = args.lookback_days
    cfg.base_timeframe_idx = 1  # 5m
    cfg.features.ema_periods = [9]
    cfg.labels.pullback_ema = 9
    cfg.sample_rate = 1.0
    cfg.enable_diagnostics = False

    # Optional: override train/val/test ratios
    ratio_override: Optional[Tuple[float, float, float]] = None
    ratios = {
        'train': args.train_ratio,
        'val': args.val_ratio,
        'test': args.test_ratio,
    }
    provided = {k: v for k, v in ratios.items() if v is not None}
    if provided:
        if len(provided) < 2:
            # Special-case: allow "train on the whole set" via --train-ratio 1.0
            if args.train_ratio is not None and float(args.train_ratio) == 1.0:
                train_ratio = 1.0
                val_ratio = 0.0
                test_ratio = 0.0
            else:
                raise SystemExit("Please provide at least two of --train-ratio/--val-ratio/--test-ratio.")
        else:
            train_ratio = args.train_ratio
            val_ratio = args.val_ratio
            test_ratio = args.test_ratio

            if train_ratio is None:
                train_ratio = 1.0 - float(val_ratio) - float(test_ratio)
            elif val_ratio is None:
                val_ratio = 1.0 - float(train_ratio) - float(test_ratio)
            elif test_ratio is None:
                test_ratio = 1.0 - float(train_ratio) - float(val_ratio)

        for name, value in [('train', train_ratio), ('val', val_ratio), ('test', test_ratio)]:
            if value is None:
                raise SystemExit("Could not infer missing ratio; provide at least two ratios.")
            if not (0.0 <= float(value) <= 1.0):
                raise SystemExit(f"Invalid {name} ratio: {value}. Ratios must be between 0 and 1 (inclusive).")

        total = float(train_ratio) + float(val_ratio) + float(test_ratio)
        if abs(total - 1.0) > 1e-6:
            raise SystemExit(f"Ratios must sum to 1.0. Got total={total:.6f}.")

        if not args.backtest_only and float(train_ratio) <= 0.0:
            raise SystemExit("Train ratio must be > 0 when training is enabled.")

        cfg.model.train_ratio = float(train_ratio)
        cfg.model.val_ratio = float(val_ratio)
        cfg.model.test_ratio = float(test_ratio)
        ratio_override = (cfg.model.train_ratio, cfg.model.val_ratio, cfg.model.test_ratio)

    if args.two_pass and not args.backtest_only and float(cfg.model.val_ratio) <= 0.0:
        raise SystemExit("--two-pass requires a non-zero validation split (set --val-ratio > 0).")
    if args.optuna_tune and float(cfg.model.val_ratio) <= 0.0:
        raise SystemExit("--optuna-tune requires a non-zero validation split (set --val-ratio > 0).")

    if args.optimize and args.optuna_tune:
        raise SystemExit("Choose only one: --optimize or --optuna-tune.")
    if args.optuna_tune and args.backtest_only:
        raise SystemExit("--optuna-tune requires training data (remove --backtest-only).")

    if not (0.0 <= float(args.tune_precision_weight) <= 1.0):
        raise SystemExit("--tune-precision-weight must be between 0 and 1.")
    if float(args.tune_trend_weight) < 0.0:
        raise SystemExit("--tune-trend-weight must be >= 0.")
    if not (0.0 <= float(args.tune_min_coverage) <= 1.0):
        raise SystemExit("--tune-min-coverage must be between 0 and 1.")
    if not (0.0 <= float(args.tune_max_coverage) <= 1.0):
        raise SystemExit("--tune-max-coverage must be between 0 and 1.")
    if float(args.tune_max_coverage) < float(args.tune_min_coverage):
        raise SystemExit("--tune-max-coverage must be >= --tune-min-coverage.")
    if float(args.tune_ops_cost_target) <= 0.0:
        raise SystemExit("--tune-ops-cost-target must be > 0.")
    if float(args.tune_ops_cost_c1) < 0.0:
        raise SystemExit("--tune-ops-cost-c1 must be >= 0.")
    if float(args.tune_ops_cost_alpha) <= 0.0:
        raise SystemExit("--tune-ops-cost-alpha must be > 0.")

    base_model_dir = Path(args.model_dir)
    train_meta: Dict[str, Any] = {}
    entry_readiness: Optional[Dict[str, Any]] = None
    train_config_path: Optional[Path] = None
    study_name: Optional[str] = None
    study_dir: Optional[Path] = None
    summary_path: Optional[Path] = None
    if args.backtest_only:
        train_config_trial: Optional[int] = None
        if args.backtest_train_config:
            train_config_path = Path(args.backtest_train_config)
            study_dir = train_config_path.parent
            study_name = study_dir.name
            train_config_trial = _parse_trial_number_from_path(train_config_path, _TRAIN_CONFIG_PREFIX)
        else:
            study_name, train_config_trial, train_config_path = _prompt_for_study_and_trial(
                base_model_dir,
                _TRAIN_CONFIG_PREFIX,
            )
            study_dir = base_model_dir / study_name

        if not train_config_path.exists():
            raise SystemExit(f"Train config not found: {train_config_path}")
        train_meta = _load_train_config_meta_from_path(train_config_path)
        entry_readiness = train_meta.get("entry_feature_readiness") if isinstance(train_meta, dict) else None
        loaded_cfg = _load_train_config_from_path(train_config_path)
        if loaded_cfg is None:
            raise SystemExit(f"Failed to load train config: {train_config_path}")
        cfg = loaded_cfg
        cfg.data.data_dir = Path(args.data_dir)
        cfg.data.lookback_days = args.lookback_days
        if ratio_override is not None:
            cfg.model.train_ratio, cfg.model.val_ratio, cfg.model.test_ratio = ratio_override
        print(f"Loaded training config from {train_config_path}")

        if args.backtest_mode == 'tuning':
            summary_path = None
            if args.backtest_tuning_summary:
                summary_path = Path(args.backtest_tuning_summary)
            elif train_meta.get("tuning_summary_path"):
                summary_path = Path(train_meta["tuning_summary_path"])
            elif train_config_trial is not None:
                summary_path = Path(train_config_path.parent) / f"{_TUNING_SUMMARY_PREFIX}_{train_config_trial}.json"
            else:
                raise SystemExit("Unable to infer tuning summary path. Provide --backtest-tuning-summary.")
            if not summary_path.exists():
                raise SystemExit(f"Tuning summary not found: {summary_path}")

        print("Backtest-only: skipping training. Using models from", cfg.model.model_dir)
    elif args.optimize:
        # Run custom parameter optimization
        print("Running parameter optimization...")
        print("Loading data for optimization...")

        trades = load_trades(cfg.data, verbose=True)
        trades = preprocess_trades(trades, cfg.data)
        bars = create_multi_timeframe_bars(trades, cfg.features.timeframes, cfg.features.timeframe_names, cfg.data)
        base_tf = cfg.features.timeframe_names[cfg.base_timeframe_idx]

        print("Calculating features...")
        featured = calculate_multi_timeframe_features(bars, base_tf, cfg.features)

        print("Labeling...")
        labeled, feature_cols = create_training_dataset(featured, cfg.labels, cfg.features, base_tf)

        # Split into train/val/test
        train_end = int(len(labeled) * cfg.model.train_ratio)
        val_end = int(len(labeled) * (cfg.model.train_ratio + cfg.model.val_ratio))

        train_df = labeled.iloc[:train_end]
        val_df = labeled.iloc[train_end:val_end]

        print(f"Train: {len(train_df)}, Val: {len(val_df)}")

        # Get pullback samples for entry model optimization
        pullback_mask_train = ~train_df['pullback_success'].isna()
        pullback_mask_val = ~val_df['pullback_success'].isna()

        X_train = train_df[feature_cols].loc[pullback_mask_train]
        y_train = train_df.loc[pullback_mask_train, 'pullback_success'].astype(int).values

        X_val = val_df[feature_cols].loc[pullback_mask_val]
        y_val = val_df.loc[pullback_mask_val, 'pullback_success'].astype(int).values

        print(f"Pullback samples - Train: {len(X_train)}, Val: {len(X_val)}")

        # Run optimization
        opt_config = OptimizerConfig(
            n_random_trials=args.optimize_trials,
            target_rr=float(args.take_profit_rr),
            min_trades_required=15,
        )

        optimizer = TrendFollowerOptimizer(X_train, y_train, X_val, y_val, config=opt_config)
        opt_results = optimizer.optimize(verbose=True)

        # Print recommended command
        if opt_results['best_params']:
            best_params = optimizer.get_best_config_params()
            print("\n" + "=" * 80)
            print("RECOMMENDED TRAINING COMMAND")
            print("=" * 80)
            cmd = (
                f"python train.py --data-dir {args.data_dir} --model-dir {args.model_dir} "
                f"--two-pass --trade-side {args.trade_side} "
                f"--min-bounce-prob {opt_results['best_min_bounce_prob']:.2f} "
                f"--learning-rate {best_params['learning_rate']:.4f} "
                f"--max-depth {best_params['max_depth']} "
                f"--num-leaves {best_params['num_leaves']} "
                f"--n-estimators {best_params['n_estimators']} "
                f"--min-child-samples {best_params['min_child_samples']} "
                f"--feature-fraction {best_params['feature_fraction']:.2f} "
                f"--lambdaa-ele1 {best_params['lambdaa_ele1']:.4f} "
                f"--lambdaa-ele2 {best_params['lambdaa_ele2']:.4f}"
            )
            print(cmd)
            print("=" * 80)

        return  # Don't continue to backtest after optimization
    elif args.train_from_tuning:
        tuning_path_value = args.train_from_tuning
        if tuning_path_value == '__MODEL_DIR__':
            study_name, trial_number, tuning_path = _prompt_for_study_and_trial(
                base_model_dir,
                _TUNING_SUMMARY_PREFIX,
            )
            study_dir = base_model_dir / study_name
        else:
            tuning_path = Path(tuning_path_value)
            study_dir = tuning_path.parent
            study_name = study_dir.name
            trial_number = _parse_trial_number_from_path(tuning_path, _TUNING_SUMMARY_PREFIX)
            if trial_number is None:
                trial_raw = input("Trial number (for train_config naming): ").strip()
                if not trial_raw:
                    raise SystemExit("Trial number is required.")
                try:
                    trial_number = int(trial_raw)
                except Exception:
                    raise SystemExit(f"Invalid trial number: {trial_raw}")
        if not tuning_path.exists():
            raise SystemExit(f"Tuning summary not found: {tuning_path}")
        with open(tuning_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        best_config = summary.get("best_config")
        if not isinstance(best_config, dict):
            raise SystemExit("Tuning summary does not contain best_config.")

        tuned_cfg = TrendFollowerConfig()
        _apply_config_section(tuned_cfg.data, best_config.get("data", {}), {"data_dir"})
        _apply_config_section(tuned_cfg.features, best_config.get("features", {}))
        _apply_config_section(tuned_cfg.labels, best_config.get("labels", {}))
        _apply_config_section(tuned_cfg.model, best_config.get("model", {}), {"model_dir"})
        if "base_timeframe_idx" in best_config:
            tuned_cfg.base_timeframe_idx = int(best_config["base_timeframe_idx"])
        if "seed" in best_config:
            tuned_cfg.seed = int(best_config["seed"])

        # Load best_threshold from best_metrics if available
        best_metrics = summary.get("best_metrics", {})
        if "best_threshold" in best_metrics:
            tuned_cfg.labels.best_threshold = float(best_metrics["best_threshold"])
        if "ev_margin_r" in best_metrics:
            tuned_cfg.labels.ev_margin_r = float(best_metrics["ev_margin_r"])
        if "fee_percent" in best_metrics:
            tuned_cfg.labels.fee_percent = float(best_metrics["fee_percent"])
        if "fee_per_trade_r" in best_metrics:
            tuned_cfg.labels.fee_per_trade_r = float(best_metrics["fee_per_trade_r"])
        if "use_expected_rr" in best_metrics:
            tuned_cfg.labels.use_expected_rr = bool(float(best_metrics["use_expected_rr"]) >= 0.5)
        if "use_raw_probabilities" in best_metrics:
            tuned_cfg.labels.use_calibration = bool(float(best_metrics["use_raw_probabilities"]) < 0.5)
        tuner_settings = summary.get("tuner_settings", {})
        if "calibration_method" in tuner_settings:
            tuned_cfg.labels.calibration_method = str(tuner_settings["calibration_method"])

        tuned_cfg.data.data_dir = Path(args.data_dir)
        tuned_cfg.model.model_dir = Path(study_dir) / f"model_{trial_number}"
        tuned_cfg.data.lookback_days = args.lookback_days

        if ratio_override is not None:
            tuned_cfg.model.train_ratio, tuned_cfg.model.val_ratio, tuned_cfg.model.test_ratio = ratio_override

        if float(tuned_cfg.model.val_ratio) <= 0.0:
            raise SystemExit("--train-from-tuning requires a non-zero validation split (set --val-ratio > 0).")

        print(f"Training models from tuning summary: {tuning_path}")
        train_results, _, _, _, _ = run_training_pipeline(
            tuned_cfg,
            enable_diagnostics=False,
            two_pass=bool(args.two_pass),
            use_noise_filtering=bool(args.use_noise_filtering),
            use_seed_ensemble=bool(args.use_seed_ensemble),
            n_ensemble_seeds=int(args.n_ensemble_seeds),
        )
        entry_readiness = train_results.get("entry_feature_readiness") if isinstance(train_results, dict) else None
        train_config_path = Path(study_dir) / f"{_TRAIN_CONFIG_PREFIX}_{trial_number}.json"
        _save_train_config_path(
            tuned_cfg,
            train_config_path,
            extra_meta={
                "tuning_summary_path": str(tuning_path),
                "entry_feature_readiness": entry_readiness,
            },
        )
        print("Training done. Saving models to", tuned_cfg.model.model_dir)
        return
    elif args.optuna_tune:
        print("Running Optuna config tuning...")
        print("Loading data for tuning...")
        trades = None
        rust_available = False
        try:
            import rust_pipeline_bridge as rust_bridge  # type: ignore
            rust_available = bool(rust_bridge.is_available())
        except Exception:
            rust_available = False
        if not rust_available:
            trades = load_trades(cfg.data, verbose=True)
            trades = preprocess_trades(trades, cfg.data)

        try:
            from config_tuner import run_config_tuning, serialize_config
        except ImportError as exc:
            raise SystemExit(str(exc))

        study_name = args.tune_study_name or _generate_study_name()
        study_dir = base_model_dir / study_name
        study_dir.mkdir(parents=True, exist_ok=True)

        storage_url = args.tune_storage
        if storage_url:
            if "://" not in storage_url:
                storage_path = Path(storage_url)
                if not storage_path.is_absolute():
                    storage_path = study_dir / storage_path
                storage_url = f"sqlite:///{storage_path}"
        else:
            storage_path = study_dir / "optuna.db"
            storage_url = f"sqlite:///{storage_path}"
        load_if_exists = bool(args.tune_storage_load_if_exists or storage_url)
        if bool(args.tune_multi_position):
            tune_single_position = False
        elif bool(args.tune_single_position):
            tune_single_position = True
        else:
            tune_single_position = True

        cfg.model.model_dir = study_dir
        tune_results = run_config_tuning(
            trades=trades,
            base_config=cfg,
            tune_scope=str(args.tune_scope),
            tuning_objective=str(args.tune_objective),
            n_trials=int(args.tune_trials),
            timeout_minutes=args.tune_timeout_min,
            precision_weight=float(args.tune_precision_weight),
            trend_weight=float(args.tune_trend_weight),
            min_pullback_samples=int(args.tune_min_pullback_samples),
            min_pullback_val_samples=int(args.tune_min_pullback_val_samples),
            use_noise_filtering=bool(args.use_noise_filtering),
            use_seed_ensemble=bool(args.use_seed_ensemble),
            n_ensemble_seeds=int(args.n_ensemble_seeds),
            seed=int(args.tune_seed),
            show_progress=not bool(args.tune_no_progress),
            report_trials=not bool(args.tune_no_trial_report),
            n_jobs=int(args.tune_parallel_jobs),
            storage=storage_url,
            study_name=study_name,
            load_if_exists=load_if_exists,
            fee_per_trade_r=args.tune_fee_per_trade_r if args.tune_fee_per_trade_r is not None else None,
            fee_percent=float(args.tune_fee_percent),
            ev_margin_r=float(args.tune_ev_margin_r),
            ev_margin_fixed=bool(args.tune_ev_margin_fixed),
            min_trades=int(args.tune_min_trades),
            min_trades_per_fold=args.tune_min_trades_fold,
            min_coverage=float(args.tune_min_coverage),
            max_coverage=float(args.tune_max_coverage),
            lcb_z=float(args.tune_lcb_z),
            use_raw_probabilities=bool(args.tune_use_raw_probabilities),
            use_expected_rr=bool(args.tune_use_expected_rr) and not bool(args.tune_no_expected_rr),
            ops_cost_enabled=not bool(args.tune_no_ops_cost),
            ops_cost_target_trades_per_day=float(args.tune_ops_cost_target),
            ops_cost_c1=float(args.tune_ops_cost_c1),
            ops_cost_alpha=float(args.tune_ops_cost_alpha),
            single_position=bool(tune_single_position),
            opposite_signal_policy=str(args.tune_opposite_policy),
            calibration_method=str(args.tune_calibration_method),
            enable_pruning=not bool(args.tune_no_prune),
            use_rust_pipeline=rust_available,
        )

        best_cfg = tune_results.best_config
        best_trial_number = int(tune_results.best_trial_number)
        best_cfg.model.model_dir = study_dir / f"model_{best_trial_number}"
        best_cfg.data.data_dir = Path(args.data_dir)
        best_margin = tune_results.best_metrics.get("ev_margin_r", float(args.tune_ev_margin_r))
        best_cfg.labels.ev_margin_r = float(best_margin)
        best_cfg.labels.fee_percent = float(args.tune_fee_percent)
        if tune_results.best_metrics.get("fee_per_trade_r") is not None:
            best_cfg.labels.fee_per_trade_r = float(tune_results.best_metrics["fee_per_trade_r"])
        best_cfg.labels.use_expected_rr = bool(args.tune_use_expected_rr) and not bool(args.tune_no_expected_rr)
        best_cfg.labels.use_ev_gate = True
        best_cfg.labels.calibration_method = str(args.tune_calibration_method)

        print("\n" + "=" * 80)
        print("OPTUNA TUNING SUMMARY")
        print("=" * 80)
        print(f"Best Score:        {tune_results.best_score:.6f}")
        print(f"Trials Completed:  {tune_results.trials_completed}")
        print(f"Elapsed Seconds:   {tune_results.elapsed_seconds:.1f}")
        print("Best Metrics:")
        for key, value in tune_results.best_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        print("=" * 80)

        holdout_entry_metrics = _evaluate_entry_holdout(
            trades,
            best_cfg,
            use_noise_filtering=bool(args.use_noise_filtering),
            use_seed_ensemble=bool(args.use_seed_ensemble),
            n_ensemble_seeds=int(args.n_ensemble_seeds),
        )
        print("\nHOLDOUT ENTRY METRICS (TEST SPLIT)")
        if holdout_entry_metrics.get('skipped'):
            reason = holdout_entry_metrics.get('reason', 'unknown')
            print(f"  Skipped: {reason}")
        else:
            print(f"  Test Pullbacks:   {holdout_entry_metrics.get('test_pullbacks', 0.0):.0f}")
            print(f"  Base Rate:        {holdout_entry_metrics.get('test_base_rate', 0.0):.3f}")
            print(f"  Raw Accuracy:     {holdout_entry_metrics.get('test_accuracy_raw', 0.0):.3f}")
            print(f"  Raw Precision:    {holdout_entry_metrics.get('test_precision_raw', 0.0):.3f}")
            print(f"  Raw Recall:       {holdout_entry_metrics.get('test_recall_raw', 0.0):.3f}")
            print(f"  Cal Accuracy:     {holdout_entry_metrics.get('test_accuracy_cal', 0.0):.3f}")
            print(f"  Cal Precision:    {holdout_entry_metrics.get('test_precision_cal', 0.0):.3f}")
            print(f"  Cal Recall:       {holdout_entry_metrics.get('test_recall_cal', 0.0):.3f}")
        print("=" * 80)

        if 'best_threshold' in tune_results.best_metrics:
            best_thresh = tune_results.best_metrics['best_threshold']
            # Save best_threshold to config so it gets persisted to train_config.json
            best_cfg.labels.best_threshold = best_thresh
            print("\nRECOMMENDED BACKTEST COMMAND (EV gate default):")
            print(f"python train.py --data-dir {args.data_dir} --model-dir {args.model_dir} --backtest-only")
            print("To force threshold gating:")
            print(
                f"python train.py --data-dir {args.data_dir} --model-dir {args.model_dir} "
                f"--backtest-only --no-ev-gate --min-bounce-prob {best_thresh:.2f}"
            )
            print("=" * 80)

        if args.tune_save_results:
            summary_path = Path(args.tune_save_results)
            if not summary_path.is_absolute():
                summary_path = study_dir / summary_path
        else:
            summary_path = study_dir / f"{_TUNING_SUMMARY_PREFIX}_{best_trial_number}.json"
        use_raw_prob = bool(tune_results.best_metrics.get("use_raw_probabilities", 0.0) >= 0.5)
        best_cfg.labels.use_calibration = not use_raw_prob
        summary = {
            'best_score': tune_results.best_score,
            'best_params': tune_results.best_params,
            'best_metrics': tune_results.best_metrics,
            'holdout_entry_metrics': holdout_entry_metrics,
            'best_config': serialize_config(best_cfg),
            'trials_completed': tune_results.trials_completed,
            'elapsed_seconds': tune_results.elapsed_seconds,
            'selected_trial': best_trial_number,
            'study_name': study_name,
        }
        summary['best_threshold'] = float(best_cfg.labels.best_threshold)
        summary['target_rr'] = float(best_cfg.labels.target_rr)
        summary['stop_atr_multiple'] = float(best_cfg.labels.stop_atr_multiple)
        summary['pullback_threshold'] = float(best_cfg.labels.pullback_threshold)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Tuning summary saved to {summary_path}")

        if args.tune_then_train:
            print("\nTraining models using best config from tuning...")
            train_results, _, _, _, _ = run_training_pipeline(
                best_cfg,
                enable_diagnostics=False,
                two_pass=bool(args.two_pass),
                use_noise_filtering=bool(args.use_noise_filtering),
                use_seed_ensemble=bool(args.use_seed_ensemble),
                n_ensemble_seeds=int(args.n_ensemble_seeds),
            )
            entry_readiness = train_results.get("entry_feature_readiness") if isinstance(train_results, dict) else None
            train_config_path = study_dir / f"{_TRAIN_CONFIG_PREFIX}_{best_trial_number}.json"
            _save_train_config_path(
                best_cfg,
                train_config_path,
                extra_meta={
                    "tuning_summary_path": str(summary_path),
                    "entry_feature_readiness": entry_readiness,
                },
            )
            print("Training done. Saving models to", best_cfg.model.model_dir)
        else:
            print("Tuning complete. Use --tune-then-train to train with the best config.")
        return
    else:
        print("Training models...")
        train_results, _, _, _, _ = run_training_pipeline(
            cfg,
            enable_diagnostics=False,
            two_pass=bool(args.two_pass),
            use_noise_filtering=bool(args.use_noise_filtering),
            use_seed_ensemble=bool(args.use_seed_ensemble),
            n_ensemble_seeds=int(args.n_ensemble_seeds),
        )
        entry_readiness = train_results.get("entry_feature_readiness") if isinstance(train_results, dict) else None
        _save_train_config(
            cfg,
            cfg.model.model_dir,
            extra_meta={"entry_feature_readiness": entry_readiness},
        )
        train_config_path = Path(cfg.model.model_dir) / _TRAIN_CONFIG_FILENAME
        train_meta = _load_train_config_meta(cfg.model.model_dir)
        entry_readiness = train_meta.get("entry_feature_readiness") if isinstance(train_meta, dict) else None
        print("Training done. Saving models to", cfg.model.model_dir)

    if args.train_only:
        print("Train-only: skipping backtest.")
        return

    print("Loading models for backtest...")
    models = TrendFollowerModels(cfg.model)
    models.load_all(cfg.model.model_dir)

    print("Loading data...")
    if summary_path is None:
        if args.backtest_tuning_summary:
            summary_path = Path(args.backtest_tuning_summary)
        elif train_meta.get("tuning_summary_path"):
            summary_path = Path(train_meta["tuning_summary_path"])
        elif train_config_path is not None:
            trial_number = _parse_trial_number_from_path(train_config_path, _TRAIN_CONFIG_PREFIX)
            if trial_number is not None:
                summary_path = Path(train_config_path.parent) / f"{_TUNING_SUMMARY_PREFIX}_{trial_number}.json"
            else:
                summary_path = base_model_dir / "tuning_summary.json"
        else:
            summary_path = base_model_dir / "tuning_summary.json"
    summary = _load_tuning_summary(summary_path) if summary_path else None
    if summary:
        _apply_summary_to_config(cfg, summary)

    cfg.data.data_dir = Path(args.data_dir)
    cfg.model.model_dir = Path(args.model_dir)
    cfg.data.lookback_days = args.lookback_days

    tuned_settings = _extract_tuned_settings(summary) if summary else {}
    use_tuned = bool(summary)
    mismatches: List[Dict[str, Any]] = []
    train_summary_path = train_meta.get("tuning_summary_path") if isinstance(train_meta, dict) else None
    if train_summary_path and summary_path:
        try:
            train_summary_resolved = Path(train_summary_path).resolve()
            summary_resolved = Path(summary_path).resolve()
        except Exception:
            train_summary_resolved = Path(train_summary_path)
            summary_resolved = Path(summary_path)
        if train_summary_resolved != summary_resolved:
            mismatches.append(
                {
                    "name": "tuning_summary_path",
                    "tuned": str(train_summary_resolved),
                    "cli": str(summary_resolved),
                    "note": "(using a different tuning summary than training)",
                }
            )

    if args.backtest_mode != "tuning":
        mismatches.append(
            {
                "name": "backtest_mode",
                "tuned": "tuning",
                "cli": args.backtest_mode,
                "note": "(tuning-style backtest only)",
            }
        )

    def _float_mismatch(a: float, b: float, tol: float = 1e-8) -> bool:
        return abs(float(a) - float(b)) > tol

    final_use_ev_gate = bool(tuned_settings.get("use_ev_gate", getattr(cfg.labels, "use_ev_gate", True)))
    ev_override = False
    if args.no_ev_gate:
        final_use_ev_gate = False
        ev_override = True
    elif args.use_ev_gate:
        final_use_ev_gate = True
        ev_override = True
    if ev_override and use_tuned:
        tuned_ev = bool(tuned_settings.get("use_ev_gate", True))
        if final_use_ev_gate != tuned_ev:
            mismatches.append({"name": "use_ev_gate", "tuned": tuned_ev, "cli": final_use_ev_gate})

    final_ev_margin_r = float(tuned_settings.get("ev_margin_r", getattr(cfg.labels, "ev_margin_r", 0.0)))
    if args.backtest_ev_margin_r is not None or _arg_present("--backtest-ev-margin-r"):
        final_ev_margin_r = float(args.backtest_ev_margin_r)
        if use_tuned and _float_mismatch(final_ev_margin_r, float(tuned_settings.get("ev_margin_r", 0.0))):
            mismatches.append({"name": "ev_margin_r", "tuned": tuned_settings.get("ev_margin_r"), "cli": final_ev_margin_r})

    final_fee_percent = float(tuned_settings.get("fee_percent", getattr(cfg.labels, "fee_percent", 0.0011)))
    if args.backtest_fee_percent is not None or _arg_present("--backtest-fee-percent"):
        final_fee_percent = float(args.backtest_fee_percent)
        if use_tuned and _float_mismatch(final_fee_percent, float(tuned_settings.get("fee_percent", 0.0011))):
            mismatches.append({"name": "fee_percent", "tuned": tuned_settings.get("fee_percent"), "cli": final_fee_percent})

    final_fee_per_trade_r = tuned_settings.get("fee_per_trade_r", getattr(cfg.labels, "fee_per_trade_r", None))
    if final_fee_per_trade_r is not None:
        final_fee_per_trade_r = float(final_fee_per_trade_r)

    final_min_bounce_prob = float(tuned_settings.get("best_threshold", getattr(cfg.labels, "best_threshold", args.min_bounce_prob)))
    if _arg_present("--min-bounce-prob"):
        final_min_bounce_prob = float(args.min_bounce_prob)
        if use_tuned and _float_mismatch(final_min_bounce_prob, float(tuned_settings.get("best_threshold", 0.5))):
            mismatches.append({"name": "min_bounce_prob", "tuned": tuned_settings.get("best_threshold"), "cli": final_min_bounce_prob})

    final_stop_loss_atr = float(tuned_settings.get("stop_atr_multiple", getattr(cfg.labels, "stop_atr_multiple", args.stop_loss_atr)))
    if _arg_present("--stop-loss-atr"):
        final_stop_loss_atr = float(args.stop_loss_atr)
        if use_tuned and _float_mismatch(final_stop_loss_atr, float(tuned_settings.get("stop_atr_multiple", 1.0))):
            mismatches.append({"name": "stop_loss_atr", "tuned": tuned_settings.get("stop_atr_multiple"), "cli": final_stop_loss_atr})

    final_take_profit_rr = float(tuned_settings.get("target_rr", getattr(cfg.labels, "target_rr", args.take_profit_rr)))
    if _arg_present("--take-profit-rr"):
        final_take_profit_rr = float(args.take_profit_rr)
        if use_tuned and _float_mismatch(final_take_profit_rr, float(tuned_settings.get("target_rr", 1.5))):
            mismatches.append({"name": "take_profit_rr", "tuned": tuned_settings.get("target_rr"), "cli": final_take_profit_rr})

    final_touch_threshold = float(tuned_settings.get("pullback_threshold", getattr(cfg.labels, "pullback_threshold", args.touch_threshold_atr)))
    if _arg_present("--touch-threshold-atr"):
        final_touch_threshold = float(args.touch_threshold_atr)
        if use_tuned and _float_mismatch(final_touch_threshold, float(tuned_settings.get("pullback_threshold", 0.3))):
            mismatches.append({"name": "touch_threshold_atr", "tuned": tuned_settings.get("pullback_threshold"), "cli": final_touch_threshold})

    if use_tuned:
        final_use_raw_probs = bool(tuned_settings.get("use_raw_probabilities", False))
        final_use_calibration = bool(tuned_settings.get("use_calibration", not final_use_raw_probs))
    else:
        final_use_calibration = bool(getattr(cfg.labels, "use_calibration", True))
        final_use_raw_probs = not final_use_calibration
    if args.use_raw_probabilities:
        final_use_raw_probs = True
        final_use_calibration = False
        if use_tuned and not bool(tuned_settings.get("use_raw_probabilities", False)):
            mismatches.append({"name": "use_raw_probabilities", "tuned": tuned_settings.get("use_raw_probabilities"), "cli": True})
    elif args.use_calibration:
        final_use_raw_probs = False
        final_use_calibration = True
        if use_tuned and bool(tuned_settings.get("use_raw_probabilities", False)):
            mismatches.append({"name": "use_calibration", "tuned": tuned_settings.get("use_calibration"), "cli": True})

    final_use_expected_rr = bool(tuned_settings.get("use_expected_rr", getattr(cfg.labels, "use_expected_rr", False)))
    if args.backtest_no_expected_rr:
        if use_tuned and final_use_expected_rr:
            mismatches.append({"name": "use_expected_rr", "tuned": True, "cli": False})
        final_use_expected_rr = False

    final_use_trend_gate = bool(tuned_settings.get("use_trend_gate", getattr(cfg.labels, "use_trend_gate", False)))
    trend_override = False
    if args.no_trend_gate:
        final_use_trend_gate = False
        trend_override = True
    elif args.use_trend_gate:
        final_use_trend_gate = True
        trend_override = True
    if trend_override and use_tuned:
        tuned_trend = bool(tuned_settings.get("use_trend_gate", False))
        if final_use_trend_gate != tuned_trend:
            mismatches.append({"name": "use_trend_gate", "tuned": tuned_trend, "cli": final_use_trend_gate})

    if args.min_trend_prob is None:
        final_min_trend_prob = float(tuned_settings.get("min_trend_prob", getattr(cfg.labels, "min_trend_prob", 0.0)))
    else:
        final_min_trend_prob = float(args.min_trend_prob)
        if use_tuned and _float_mismatch(final_min_trend_prob, float(tuned_settings.get("min_trend_prob", 0.0))):
            mismatches.append({"name": "min_trend_prob", "tuned": tuned_settings.get("min_trend_prob"), "cli": final_min_trend_prob})

    final_use_regime_gate = bool(tuned_settings.get("use_regime_gate", getattr(cfg.labels, "use_regime_gate", False)))
    regime_override = False
    if args.no_regime_gate:
        final_use_regime_gate = False
        regime_override = True
    elif args.use_regime_gate:
        final_use_regime_gate = True
        regime_override = True
    if regime_override and use_tuned:
        tuned_regime = bool(tuned_settings.get("use_regime_gate", False))
        if final_use_regime_gate != tuned_regime:
            mismatches.append({"name": "use_regime_gate", "tuned": tuned_regime, "cli": final_use_regime_gate})

    if args.min_regime_prob is None:
        final_min_regime_prob = float(tuned_settings.get("min_regime_prob", getattr(cfg.labels, "min_regime_prob", 0.0)))
    else:
        final_min_regime_prob = float(args.min_regime_prob)
        if use_tuned and _float_mismatch(final_min_regime_prob, float(tuned_settings.get("min_regime_prob", 0.0))):
            mismatches.append({"name": "min_regime_prob", "tuned": tuned_settings.get("min_regime_prob"), "cli": final_min_regime_prob})

    allow_regime_ranging = bool(tuned_settings.get("allow_regime_ranging", getattr(cfg.labels, "allow_regime_ranging", True)))
    allow_regime_trend_up = bool(tuned_settings.get("allow_regime_trend_up", getattr(cfg.labels, "allow_regime_trend_up", True)))
    allow_regime_trend_down = bool(tuned_settings.get("allow_regime_trend_down", getattr(cfg.labels, "allow_regime_trend_down", True)))
    allow_regime_volatile = bool(tuned_settings.get("allow_regime_volatile", getattr(cfg.labels, "allow_regime_volatile", True)))
    regime_align_direction = bool(tuned_settings.get("regime_align_direction", getattr(cfg.labels, "regime_align_direction", True)))

    ops_cost_enabled = bool(tuned_settings.get("ops_cost_enabled", True))
    ops_cost_target = float(tuned_settings.get("ops_cost_target_trades_per_day", 30.0))
    ops_cost_c1 = float(tuned_settings.get("ops_cost_c1", 0.01))
    ops_cost_alpha = float(tuned_settings.get("ops_cost_alpha", 1.7))

    use_full_data = bool(args.backtest_full_data)
    if use_full_data:
        mismatches.append({"name": "backtest_full_data", "tuned": "test_split", "cli": "full_data"})

    if args.backtest_ohlc_only:
        mismatches.append({"name": "backtest_ohlc_only", "tuned": False, "cli": True, "note": "(ignored in tuned backtest)"})

    if args.use_dynamic_rr:
        mismatches.append({"name": "use_dynamic_rr", "tuned": False, "cli": True, "note": "(ignored in tuned backtest)"})

    if args.ema_touch_mode != "multi":
        mismatches.append({"name": "ema_touch_mode", "tuned": "multi", "cli": args.ema_touch_mode})

    if args.cooldown_bars_after_stop != 0:
        mismatches.append(
            {
                "name": "cooldown_bars_after_stop",
                "tuned": 0,
                "cli": args.cooldown_bars_after_stop,
                "note": "(ignored in tuned backtest)",
            }
        )

    if args.trade_side != "both":
        mismatches.append({"name": "trade_side", "tuned": "both", "cli": args.trade_side})

    max_holding_bars = None
    tuned_max_hold = tuned_settings.get("entry_forward_window")
    if tuned_max_hold:
        max_holding_bars = int(tuned_max_hold)
    if args.backtest_max_hold_bars is not None:
        max_holding_bars = int(args.backtest_max_hold_bars)
        if use_tuned and tuned_max_hold is not None and max_holding_bars != int(tuned_max_hold):
            mismatches.append({"name": "backtest_max_hold_bars", "tuned": tuned_max_hold, "cli": max_holding_bars})

    final_max_bounce_prob = float(args.max_bounce_prob)
    if final_use_ev_gate and _arg_present("--max-bounce-prob") and final_max_bounce_prob < 1.0:
        mismatches.append({"name": "max_bounce_prob", "tuned": 1.0, "cli": final_max_bounce_prob, "note": "(ignored because EV gate is on)"})
        final_max_bounce_prob = 1.0

    _confirm_backtest_overrides(mismatches)

    cfg.labels.best_threshold = float(final_min_bounce_prob)
    cfg.labels.stop_atr_multiple = float(final_stop_loss_atr)
    cfg.labels.target_rr = float(final_take_profit_rr)
    cfg.labels.pullback_threshold = float(final_touch_threshold)
    cfg.labels.ev_margin_r = float(final_ev_margin_r)
    cfg.labels.fee_percent = float(final_fee_percent)
    if final_fee_per_trade_r is not None:
        cfg.labels.fee_per_trade_r = float(final_fee_per_trade_r)
    cfg.labels.use_expected_rr = bool(final_use_expected_rr)
    cfg.labels.use_calibration = bool(final_use_calibration)
    cfg.labels.use_ev_gate = bool(final_use_ev_gate)
    cfg.labels.use_trend_gate = bool(final_use_trend_gate)
    cfg.labels.min_trend_prob = float(final_min_trend_prob)
    cfg.labels.use_regime_gate = bool(final_use_regime_gate)
    cfg.labels.min_regime_prob = float(final_min_regime_prob)
    cfg.labels.allow_regime_ranging = bool(allow_regime_ranging)
    cfg.labels.allow_regime_trend_up = bool(allow_regime_trend_up)
    cfg.labels.allow_regime_trend_down = bool(allow_regime_trend_down)
    cfg.labels.allow_regime_volatile = bool(allow_regime_volatile)
    cfg.labels.regime_align_direction = bool(regime_align_direction)

    labeled, feature_cols = build_dataset_from_config(
        cfg,
        use_rust_pipeline=True,
        rust_cache_dir="rust_cache",
        rust_write_intermediate=False,
    )
    if labeled is None or feature_cols is None or len(labeled) == 0:
        raise SystemExit("Backtest dataset is empty; cannot run backtest.")

    split_idx = int(len(labeled) * (float(cfg.model.train_ratio) + float(cfg.model.val_ratio)))
    split_idx = max(0, min(len(labeled), split_idx))
    if not use_full_data and split_idx >= len(labeled):
        raise SystemExit(
            "Test split is empty (test_ratio=0). Use --train-only to just train, or set a non-zero --test-ratio."
        )
    if use_full_data:
        print(f"Backtest on full dataset: {len(labeled)} rows")
    else:
        print(f"Test set size: {len(labeled) - split_idx} bars from index {split_idx}")

    if final_use_ev_gate:
        print(
            "[INFO] Using EV gate: margin={:.4f}R fee_pct={:.5f} expected_rr={}".format(
                float(final_ev_margin_r),
                float(final_fee_percent),
                "on" if final_use_expected_rr else "off",
            )
        )

    res = run_tuned_backtest(
        labeled,
        feature_cols,
        models,
        cfg,
        use_full_data=use_full_data,
        trade_side=str(args.trade_side),
        use_ev_gate=bool(final_use_ev_gate),
        ev_margin_r=float(final_ev_margin_r),
        min_bounce_prob=float(final_min_bounce_prob),
        max_bounce_prob=float(final_max_bounce_prob),
        use_raw_probabilities=bool(final_use_raw_probs),
        use_calibration=bool(final_use_calibration),
        use_expected_rr=bool(final_use_expected_rr),
        fee_percent=float(final_fee_percent),
        fee_per_trade_r=final_fee_per_trade_r,
        ops_cost_enabled=bool(ops_cost_enabled),
        ops_cost_target_trades_per_day=float(ops_cost_target),
        ops_cost_c1=float(ops_cost_c1),
        ops_cost_alpha=float(ops_cost_alpha),
        single_position=True,
        opposite_signal_policy="flip",
        max_holding_bars=max_holding_bars,
        ema_touch_mode=str(args.ema_touch_mode),
        entry_feature_readiness=entry_readiness,
    )
    print_backtest_results(res)

    stop_stats = {}
    if res.trades:
        stops = []
        for t in res.trades:
            if t.stop_loss is not None and t.entry_price:
                dist_pct = abs(t.entry_price - t.stop_loss) / t.entry_price * 100
                stops.append(dist_pct)
        if stops:
            import pandas as pd
            s = pd.Series(stops)
            stop_stats = {
                'stop_distance_pct_mean': float(s.mean()),
                'stop_distance_pct_median': float(s.median()),
                'stop_distance_pct_min': float(s.min()),
                'stop_distance_pct_max': float(s.max()),
            }
            print(
                "Stop distance pct (entry to stop) -> "
                f"mean {stop_stats['stop_distance_pct_mean']:.3f}%, "
                f"median {stop_stats['stop_distance_pct_median']:.3f}%, "
                f"min {stop_stats['stop_distance_pct_min']:.3f}%, "
                f"max {stop_stats['stop_distance_pct_max']:.3f}%"
            )
        else:
            print("Stop distance pct: no data")

    log_params = {
        "backtest_only": bool(args.backtest_only),
        "use_full_data": bool(use_full_data),
        "min_bounce_prob": float(final_min_bounce_prob),
        "max_bounce_prob": float(final_max_bounce_prob),
        "min_trend_prob": float(final_min_trend_prob),
        "use_trend_gate": bool(final_use_trend_gate),
        "use_regime_gate": bool(final_use_regime_gate),
        "min_regime_prob": float(final_min_regime_prob),
        "allow_regime_ranging": bool(allow_regime_ranging),
        "allow_regime_trend_up": bool(allow_regime_trend_up),
        "allow_regime_trend_down": bool(allow_regime_trend_down),
        "allow_regime_volatile": bool(allow_regime_volatile),
        "regime_align_direction": bool(regime_align_direction),
        "stop_loss_atr": float(final_stop_loss_atr),
        "stop_padding_pct": float(args.stop_padding_pct),
        "take_profit_rr": float(final_take_profit_rr),
        "use_dynamic_rr": bool(args.use_dynamic_rr),
        "use_ev_gate": bool(final_use_ev_gate),
        "ev_margin_r": float(final_ev_margin_r),
        "fee_percent": float(final_fee_percent),
        "fee_per_trade_r": float(final_fee_per_trade_r) if final_fee_per_trade_r is not None else None,
        "use_expected_rr": float(1.0 if final_use_expected_rr else 0.0),
        "use_calibration": bool(final_use_calibration),
        "use_raw_probabilities": bool(final_use_raw_probs),
        "ops_cost_enabled": bool(ops_cost_enabled),
        "ops_cost_target_trades_per_day": float(ops_cost_target),
        "ops_cost_c1": float(ops_cost_c1),
        "ops_cost_alpha": float(ops_cost_alpha),
        "cooldown_bars_after_stop": int(args.cooldown_bars_after_stop),
        "trade_side": str(args.trade_side),
        "touch_threshold_atr": float(final_touch_threshold),
        "ema_touch_mode": str(args.ema_touch_mode),
        "backtest_ohlc_only": bool(args.backtest_ohlc_only),
        "backtest_max_holding_bars": int(max_holding_bars or 0),
        "base_timeframe_idx": int(cfg.base_timeframe_idx),
        "base_timeframe": cfg.features.timeframe_names[cfg.base_timeframe_idx],
        "ema_periods": list(cfg.features.ema_periods),
        "pullback_ema": int(cfg.labels.pullback_ema),
        "pullback_threshold_atr": float(cfg.labels.pullback_threshold),
        "test_bars": int(len(labeled) - split_idx if not use_full_data else len(labeled)),
        "test_start_index": int(0 if use_full_data else split_idx),
    }
    symbol_tag = Path(cfg.data.data_dir).name
    log_dir = Path("./backtests") / symbol_tag
    paths = save_backtest_logs(
        res,
        cfg,
        log_dir,
        model_dir=cfg.model.model_dir,
        driver=Path(__file__).name,
        parameters=log_params,
        extra_metrics=stop_stats,
    )
    print(f"Saved backtest logs -> {paths['summary']} and {paths['trades']}")


if __name__ == '__main__':
    main()
