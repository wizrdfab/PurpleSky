"""
Optuna-based configuration tuner for TrendFollower.

This module tunes model, feature, and label parameters from config.py
to maximize robust profitability (risk-adjusted), with optional
precision/accuracy objectives for comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from copy import deepcopy
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import time
import threading
import json

import pandas as pd
import numpy as np

from config import TrendFollowerConfig
from data_loader import create_multi_timeframe_bars
from feature_engine import calculate_multi_timeframe_features
from labels import create_training_dataset
from trainer import time_series_split, _oof_context_preds as _trainer_oof_context_preds
from backtest_tuned_config import run_tuned_backtest, print_backtest_results, _compute_trade_metrics
from models import (
    TrendClassifier,
    EntryQualityModel,
    RegimeClassifier,
    append_context_features,
    compute_expected_calibration_error,
    TrendFollowerModels,
)

try:
    from sklearn.metrics import roc_auc_score
    SKLEARN_METRICS_AVAILABLE = True
except ImportError:
    SKLEARN_METRICS_AVAILABLE = False

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


@dataclass
class ConfigTuningResult:
    best_score: float
    best_params: Dict[str, Any]
    best_metrics: Dict[str, float]
    best_config: TrendFollowerConfig
    best_trial_number: int
    trials_completed: int
    elapsed_seconds: float


def _serialize_dataclass(dc) -> Dict[str, Any]:
    data = asdict(dc)
    for key, value in data.items():
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def serialize_config(cfg: TrendFollowerConfig) -> Dict[str, Any]:
    return {
        'data': _serialize_dataclass(cfg.data),
        'features': _serialize_dataclass(cfg.features),
        'labels': _serialize_dataclass(cfg.labels),
        'model': _serialize_dataclass(cfg.model),
        'base_timeframe_idx': cfg.base_timeframe_idx,
        'seed': cfg.seed,
    }


class ConfigTuner:
    def __init__(
        self,
        trades: Optional[pd.DataFrame],
        base_config: TrendFollowerConfig,
        tune_scope: str = "full",
        tuning_objective: str = "profit",
        precision_weight: float = 0.6,
        trend_weight: float = 0.0,
        min_pullback_samples: int = 100,
        min_pullback_val_samples: int = 20,
        use_noise_filtering: bool = False,
        use_seed_ensemble: bool = False,
        n_ensemble_seeds: int = 5,
        seed: int = 42,
        report_trials: bool = True,
        fee_per_trade_r: Optional[float] = None,
        fee_percent: float = 0.0011,
        ev_margin_r: float = 0.0,
        ev_margin_fixed: bool = False,
        min_trades: int = 30,
        min_trades_per_fold: Optional[int] = None,
        min_coverage: float = 0.0,
        max_coverage: float = 0.7,
        lcb_z: float = 1.28,
        stability_score_iqr_weight: float = 0.5,
        stability_trade_iqr_weight: float = 0.1,
        stability_coverage_iqr_weight: float = 0.1,
        stability_penalty_multiplier: float = 1.5,
        stability_k: Optional[float] = None,
        coverage_k: Optional[float] = None,
        stability_eps: float = 1e-6,
        no_opportunity_penalty: float = 0.002,
        refusal_penalty: float = 0.01,
        ops_cost_enabled: bool = True,
        ops_cost_target_trades_per_day: float = 30.0,
        ops_cost_c1: float = 0.01,
        ops_cost_alpha: float = 1.7,
        single_position: bool = True,
        opposite_signal_policy: str = "flip",
        calibration_method: str = "temperature",
        use_raw_probabilities: bool = False,
        use_expected_rr: bool = False,
        use_rust_pipeline: bool = True,
        rust_cache_dir: str = "rust_cache",
        rust_write_intermediate: bool = False,
        lgbm_num_threads: Optional[int] = None,
        enable_pruning: bool = True,
        selection_enabled: bool = True,
        selection_top_n: int = -1,
        selection_test_pct: float = 0.10,
        selection_shadow_pct: float = 0.05,
        selection_min_trades: int = 30,
        selection_min_trades_shadow: Optional[int] = None,
        selection_shadow_cap: int = 30,
        selection_min_profit_factor: float = 1.1,
        selection_max_drawdown_pct: float = 5.0,
        selection_max_coverage: float = 0.7,
        selection_min_total_r: float = 0.0,
    ):
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required. Install with: pip install optuna")

        self.base_config = deepcopy(base_config)
        self.seed = int(seed)

        self.tune_scope = (tune_scope or "full").strip().lower()
        self.tune_model = self.tune_scope in {"model", "full", "all"}
        self.tune_features = self.tune_scope in {"features", "full", "all"}
        self.tune_labels = self.tune_scope in {"labels", "full", "all"}

        self.tuning_objective = (tuning_objective or "profit").strip().lower()
        if self.tuning_objective not in {"profit", "precision", "mixed"}:
            self.tuning_objective = "profit"

        self.precision_weight = float(precision_weight)
        self.trend_weight = float(trend_weight)
        self.min_pullback_samples = int(min_pullback_samples)
        self.min_pullback_val_samples = int(min_pullback_val_samples)

        self.use_noise_filtering = bool(use_noise_filtering)
        self.use_seed_ensemble = bool(use_seed_ensemble)
        self.n_ensemble_seeds = int(n_ensemble_seeds)
        self.report_trials = bool(report_trials)
        self.pause_requested = False
        self._pause_listener: Optional[threading.Thread] = None
        self._pause_listener_stop = threading.Event()
        self._pause_cond = threading.Condition()
        self._paused = False
        self._pause_menu_lock = threading.Lock()
        self._pause_active = False
        self._active_study = None
        self._tune_start_time: Optional[float] = None
        self._bars_cache: Dict[Tuple[Any, ...], Dict[str, pd.DataFrame]] = {}
        self._features_cache: Dict[Tuple[Any, ...], pd.DataFrame] = {}
        self._labels_cache: Dict[Tuple[Any, ...], Tuple[pd.DataFrame, List[str]]] = {}
        self._split_cache: Dict[Tuple[Any, ...], List[Tuple[np.ndarray, np.ndarray]]] = {}

        self.threshold_grid = np.arange(0.15, 0.86, 0.02)
        self.min_samples_for_folds = 500
        self.fee_per_trade_r = None if fee_per_trade_r is None else float(fee_per_trade_r)
        self.fee_percent = float(fee_percent)
        self.ev_margin_r = float(ev_margin_r)
        self.ev_margin_fixed = bool(ev_margin_fixed)
        self.min_trades = max(0, int(min_trades))
        self.min_trades_per_fold = None if min_trades_per_fold is None else max(0, int(min_trades_per_fold))
        self.min_coverage = float(min_coverage)
        self.max_coverage = float(max_coverage)
        self.lcb_z = float(lcb_z)
        self.stability_score_iqr_weight = float(stability_score_iqr_weight)
        self.stability_trade_iqr_weight = float(stability_trade_iqr_weight)
        self.stability_coverage_iqr_weight = float(stability_coverage_iqr_weight)
        self.stability_penalty_multiplier = float(stability_penalty_multiplier)
        if stability_k is None:
            self.stability_k = float(self.stability_penalty_multiplier)
        else:
            self.stability_k = float(stability_k)
        self.stability_eps = float(stability_eps)
        self.no_opportunity_penalty = float(no_opportunity_penalty)
        self.refusal_penalty = float(refusal_penalty)
        self.ops_cost_enabled = bool(ops_cost_enabled)
        self.ops_cost_target_trades_per_day = float(ops_cost_target_trades_per_day)
        self.ops_cost_c1 = float(ops_cost_c1)
        self.ops_cost_alpha = float(ops_cost_alpha)
        self.single_position = bool(single_position)
        self.opposite_signal_policy = str(opposite_signal_policy or "ignore").strip().lower()
        self.calibration_method = (calibration_method or "temperature").strip().lower()
        self.use_raw_probabilities = bool(use_raw_probabilities)
        self.use_expected_rr = bool(use_expected_rr)
        self.use_rust_pipeline = bool(use_rust_pipeline)
        self.rust_cache_dir = Path(rust_cache_dir)
        self.rust_write_intermediate = bool(rust_write_intermediate)
        if hasattr(self.base_config, "labels") and hasattr(self.base_config.labels, "calibration_method"):
            self.base_config.labels.calibration_method = self.calibration_method
        self.lgbm_num_threads_override = lgbm_num_threads
        self._rust_available: Optional[bool] = None
        self._rust_warned = False
        self.enable_pruning = bool(enable_pruning)
        self.selection_enabled = bool(selection_enabled)
        if selection_top_n is None:
            self.selection_top_n = -1
        else:
            self.selection_top_n = int(selection_top_n)
        self.selection_test_pct = float(selection_test_pct)
        self.selection_shadow_pct = float(selection_shadow_pct)
        self.selection_min_trades = max(0, int(selection_min_trades))
        self.selection_min_trades_shadow = None if selection_min_trades_shadow is None else max(0, int(selection_min_trades_shadow))
        self.selection_shadow_cap = max(0, int(selection_shadow_cap))
        self.selection_min_profit_factor = float(selection_min_profit_factor)
        self.selection_max_drawdown_pct = float(selection_max_drawdown_pct)
        self.selection_max_coverage = float(selection_max_coverage)
        self.selection_min_total_r = float(selection_min_total_r)
        if trades is not None and not isinstance(trades, pd.DataFrame):
            raise TypeError("trades must be a pandas DataFrame or None")
        self.trades = trades

        self.min_coverage = max(0.0, min(1.0, self.min_coverage))
        self.max_coverage = max(self.min_coverage, min(1.0, self.max_coverage))
        if not np.isfinite(self.ops_cost_target_trades_per_day) or self.ops_cost_target_trades_per_day <= 0:
            self.ops_cost_enabled = False
            self.ops_cost_target_trades_per_day = 0.0
        if not np.isfinite(self.ops_cost_c1) or self.ops_cost_c1 < 0:
            self.ops_cost_c1 = 0.0
        if not np.isfinite(self.ops_cost_alpha) or self.ops_cost_alpha <= 0:
            self.ops_cost_alpha = 1.0
        if not np.isfinite(self.stability_k) or self.stability_k < 0:
            self.stability_k = 0.0
        if not np.isfinite(self.stability_eps) or self.stability_eps <= 0:
            self.stability_eps = 1e-6
        coverage_band = max(0.05, self.max_coverage - self.min_coverage)
        if coverage_k is None:
            self.coverage_k = 2.0 / coverage_band
        else:
            self.coverage_k = float(coverage_k)
        if not np.isfinite(self.coverage_k) or self.coverage_k < 0:
            self.coverage_k = 0.0
        if not np.isfinite(self.selection_test_pct) or self.selection_test_pct < 0:
            self.selection_test_pct = 0.0
        if not np.isfinite(self.selection_shadow_pct) or self.selection_shadow_pct < 0:
            self.selection_shadow_pct = 0.0
        total_selection_pct = self.selection_test_pct + self.selection_shadow_pct
        if total_selection_pct > 1.0:
            scale = 1.0 / total_selection_pct
            self.selection_test_pct *= scale
            self.selection_shadow_pct *= scale
        if not np.isfinite(self.selection_min_profit_factor) or self.selection_min_profit_factor < 0:
            self.selection_min_profit_factor = 0.0
        if not np.isfinite(self.selection_max_drawdown_pct) or self.selection_max_drawdown_pct < 0:
            self.selection_max_drawdown_pct = 0.0
        if not np.isfinite(self.selection_max_coverage) or self.selection_max_coverage < 0:
            self.selection_max_coverage = 0.0
        if not np.isfinite(self.selection_min_total_r):
            self.selection_min_total_r = 0.0
        if self.selection_min_trades_shadow is not None and not np.isfinite(self.selection_min_trades_shadow):
            self.selection_min_trades_shadow = None
        if self.selection_min_trades_shadow is None:
            if self.selection_shadow_pct > 0 and self.selection_test_pct > 0:
                ratio = self.selection_shadow_pct / self.selection_test_pct
            else:
                ratio = 0.0
            derived = int(round(self.selection_min_trades * ratio))
            self.selection_min_trades_shadow = max(5 if self.selection_shadow_pct > 0 else 0, derived)
        if self.selection_shadow_cap > 0 and self.selection_min_trades_shadow is not None:
            self.selection_min_trades_shadow = min(self.selection_min_trades_shadow, self.selection_shadow_cap)
        if self.opposite_signal_policy not in {"ignore", "close", "flip"}:
            self.opposite_signal_policy = "ignore"

        if hasattr(self.base_config, "labels"):
            self.base_config.labels.ev_margin_r = float(ev_margin_r)
            self.base_config.labels.fee_percent = float(fee_percent)
            self.base_config.labels.use_expected_rr = bool(use_expected_rr)
            self.base_config.labels.use_ev_gate = True

        if self.trades is None and not self._rust_pipeline_available():
            raise ValueError("trades is required when Rust pipeline is unavailable")

    def _get_walk_forward_splits(
        self,
        data: pd.DataFrame,
        cfg: TrendFollowerConfig,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Walk-forward splits on the tuning range (train+val only).

        Default: 3 expanding-window folds over the tuning slice.
        Fallback: single split for small datasets.
        """
        split_indices = self._get_walk_forward_split_indices(len(data), cfg)
        if not split_indices:
            return []
        return [(data.iloc[train_idx], data.iloc[val_idx]) for train_idx, val_idx in split_indices]

    def _get_walk_forward_split_indices(
        self,
        n_rows: int,
        cfg: TrendFollowerConfig,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        purge_h = int(max(
            0,
            getattr(cfg.labels, "trend_forward_window", 0),
            getattr(cfg.labels, "entry_forward_window", 0),
        ))
        tuning_ratio = float(cfg.model.train_ratio) + float(cfg.model.val_ratio)
        cache_key = (
            int(n_rows),
            float(cfg.model.train_ratio),
            float(cfg.model.val_ratio),
            float(cfg.model.test_ratio),
            int(self.min_samples_for_folds),
            int(purge_h),
        )
        cached = self._split_cache.get(cache_key)
        if cached is not None:
            return cached

        if tuning_ratio <= 0.0 or n_rows <= 0:
            self._split_cache[cache_key] = []
            return []

        tuning_end = int(n_rows * tuning_ratio)
        if tuning_end <= 0:
            self._split_cache[cache_key] = []
            return []

        n_tune = tuning_end

        def _apply_purge_indices(
            train_idx: np.ndarray,
            val_idx: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray]:
            if purge_h <= 0:
                return train_idx, val_idx
            if train_idx.size > purge_h:
                train_idx = train_idx[:-purge_h]
            else:
                train_idx = np.array([], dtype=int)
            if val_idx.size > purge_h:
                val_idx = val_idx[purge_h:]
            else:
                val_idx = np.array([], dtype=int)
            return train_idx, val_idx

        splits: List[Tuple[np.ndarray, np.ndarray]] = []
        if n_tune < self.min_samples_for_folds:
            train_frac = float(cfg.model.train_ratio) / tuning_ratio
            train_end = int(n_tune * train_frac)
            if train_end <= 0 or train_end >= n_tune:
                self._split_cache[cache_key] = []
                return []
            train_idx = np.arange(0, train_end, dtype=int)
            val_idx = np.arange(train_end, n_tune, dtype=int)
            train_idx, val_idx = _apply_purge_indices(train_idx, val_idx)
            if train_idx.size > 0 and val_idx.size > 0:
                splits.append((train_idx, val_idx))
            self._split_cache[cache_key] = splits
            return splits

        val_starts = [0.55, 0.70, 0.85]
        val_ends = [0.70, 0.85, 1.0]
        for start_frac, end_frac in zip(val_starts, val_ends):
            train_end = int(n_tune * start_frac)
            val_end = int(n_tune * end_frac)
            if train_end <= 0 or val_end <= train_end:
                continue
            train_idx = np.arange(0, train_end, dtype=int)
            val_idx = np.arange(train_end, val_end, dtype=int)
            train_idx, val_idx = _apply_purge_indices(train_idx, val_idx)
            if train_idx.size > 0 and val_idx.size > 0:
                splits.append((train_idx, val_idx))

        self._split_cache[cache_key] = splits
        return splits

    def _bars_cache_key(self, cfg: TrendFollowerConfig) -> Tuple[Any, ...]:
        if self.trades is not None:
            trades_key = ("len", int(len(self.trades)))
        else:
            trades_key = (
                "data",
                str(cfg.data.data_dir),
                str(cfg.data.file_pattern),
                cfg.data.lookback_days,
                str(cfg.data.timestamp_col),
                str(cfg.data.price_col),
                str(cfg.data.size_col),
                str(cfg.data.side_col),
                str(cfg.data.tick_direction_col),
            )
        return (
            tuple(cfg.features.timeframes),
            tuple(cfg.features.timeframe_names),
            trades_key,
        )

    def _features_cache_key(
        self,
        bars_key: Tuple[Any, ...],
        cfg: TrendFollowerConfig,
        base_tf: str,
    ) -> Tuple[Any, ...]:
        return (
            bars_key,
            base_tf,
            tuple(sorted(cfg.features.ema_periods)),
            int(cfg.features.rsi_period),
            int(cfg.features.adx_period),
            int(cfg.features.atr_period),
            int(cfg.features.bb_period),
            float(cfg.features.bb_std),
            int(cfg.features.volume_ma_period),
            int(cfg.features.swing_lookback),
        )

    def _labels_cache_key(
        self,
        features_key: Tuple[Any, ...],
        cfg: TrendFollowerConfig,
        base_tf: str,
    ) -> Tuple[Any, ...]:
        return (
            features_key,
            base_tf,
            int(cfg.labels.trend_forward_window),
            int(cfg.labels.entry_forward_window),
            float(cfg.labels.trend_up_threshold),
            float(cfg.labels.trend_down_threshold),
            float(cfg.labels.max_adverse_for_trend),
            float(cfg.labels.target_rr),
            float(cfg.labels.stop_atr_multiple),
            int(cfg.labels.pullback_ema),
            float(cfg.labels.pullback_threshold),
        )

    def _get_cached_bars(self, cfg: TrendFollowerConfig) -> Tuple[Dict[str, pd.DataFrame], Tuple[Any, ...]]:
        if self.trades is None:
            raise ValueError("Trades are not loaded; Rust pipeline is required for bars.")
        bars_key = self._bars_cache_key(cfg)
        cached = self._bars_cache.get(bars_key)
        if cached is not None:
            return cached, bars_key
        bars_dict = create_multi_timeframe_bars(
            self.trades,
            cfg.features.timeframes,
            cfg.features.timeframe_names,
            cfg.data,
        )
        self._bars_cache[bars_key] = bars_dict
        return bars_dict, bars_key

    def _get_cached_features(
        self,
        bars_dict: Dict[str, pd.DataFrame],
        bars_key: Tuple[Any, ...],
        cfg: TrendFollowerConfig,
        base_tf: str,
    ) -> Tuple[pd.DataFrame, Tuple[Any, ...]]:
        features_key = self._features_cache_key(bars_key, cfg, base_tf)
        cached = self._features_cache.get(features_key)
        if cached is not None:
            return cached, features_key
        featured = calculate_multi_timeframe_features(
            bars_dict,
            base_tf,
            cfg.features,
        )
        self._features_cache[features_key] = featured
        return featured, features_key

    def _get_cached_labels(
        self,
        featured: pd.DataFrame,
        features_key: Tuple[Any, ...],
        cfg: TrendFollowerConfig,
        base_tf: str,
    ) -> Tuple[pd.DataFrame, List[str], Tuple[Any, ...]]:
        labels_key = self._labels_cache_key(features_key, cfg, base_tf)
        cached = self._labels_cache.get(labels_key)
        if cached is not None:
            labeled_data, feature_cols = cached
            return labeled_data, feature_cols, labels_key
        labeled_data, feature_cols = create_training_dataset(
            featured,
            cfg.labels,
            cfg.features,
            base_tf,
        )
        self._labels_cache[labels_key] = (labeled_data, feature_cols)
        return labeled_data, feature_cols, labels_key

    def _rust_pipeline_available(self) -> bool:
        if not self.use_rust_pipeline:
            return False
        if self._rust_available is None:
            try:
                import rust_pipeline_bridge as rust_bridge  # noqa: F401
                self._rust_available = bool(rust_bridge.is_available())
            except Exception:
                self._rust_available = False
        if not self._rust_available and not self._rust_warned:
            print("Rust pipeline unavailable; falling back to Python pipeline.")
            self._rust_warned = True
        return bool(self._rust_available)

    def _get_cached_labels_rust(
        self,
        cfg: TrendFollowerConfig,
        base_tf: str,
    ) -> Tuple[pd.DataFrame, List[str], Tuple[Any, ...]]:
        bars_key = self._bars_cache_key(cfg)
        features_key = self._features_cache_key(bars_key, cfg, base_tf)
        labels_key = self._labels_cache_key(features_key, cfg, base_tf)
        cached = self._labels_cache.get(labels_key)
        if cached is not None:
            labeled_data, feature_cols = cached
            return labeled_data, feature_cols, labels_key

        import rust_pipeline_bridge as rust_bridge

        labeled_data, feature_cols, _dataset_path = rust_bridge.build_dataset_from_config(
            cfg,
            cache_dir=self.rust_cache_dir,
            write_intermediate=self.rust_write_intermediate,
            force=False,
        )
        self._labels_cache[labels_key] = (labeled_data, feature_cols)
        return labeled_data, feature_cols, labels_key

    @staticmethod
    def _max_drawdown(returns: np.ndarray) -> float:
        if returns.size == 0:
            return 0.0
        equity = np.cumsum(returns)
        peak = np.maximum.accumulate(equity)
        drawdown = peak - equity
        return float(drawdown.max()) if drawdown.size else 0.0

    def _compute_trade_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        n_trades = int(returns.size)
        if n_trades == 0:
            return {
                'n_trades': 0.0,
                'total_pnl_r': 0.0,
                'avg_pnl_r': 0.0,
                'win_rate': 0.0,
                'avg_win_r': 0.0,
                'avg_loss_r': 0.0,
                'profit_factor': 0.0,
                'return_std': 0.0,
                'downside_std': 0.0,
                'sharpe': 0.0,
                'sortino': 0.0,
                'max_drawdown_r': 0.0,
                'calmar': 0.0,
            }

        total_pnl = float(np.sum(returns))
        avg_pnl = float(np.mean(returns))
        wins = returns[returns > 0]
        losses = returns[returns <= 0]
        win_rate = float(wins.size / n_trades)
        avg_win = float(np.mean(wins)) if wins.size > 0 else 0.0
        avg_loss = float(np.mean(losses)) if losses.size > 0 else 0.0
        gross_win = float(np.sum(wins)) if wins.size > 0 else 0.0
        gross_loss = float(np.sum(losses)) if losses.size > 0 else 0.0
        gross_loss_abs = abs(gross_loss)
        if gross_loss_abs > 0:
            profit_factor = float(gross_win / gross_loss_abs)
        else:
            profit_factor = float('inf') if gross_win > 0 else 0.0

        return_std = float(np.std(returns)) if n_trades > 1 else 0.0
        downside_std = float(np.std(losses)) if losses.size > 1 else 0.0
        sharpe = float(avg_pnl / return_std) if return_std > 0 else 0.0
        sortino = float(avg_pnl / downside_std) if downside_std > 0 else 0.0
        max_dd = self._max_drawdown(returns)
        calmar = float(total_pnl / max_dd) if max_dd > 0 else total_pnl

        return {
            'n_trades': float(n_trades),
            'total_pnl_r': total_pnl,
            'avg_pnl_r': avg_pnl,
            'win_rate': win_rate,
            'avg_win_r': avg_win,
            'avg_loss_r': avg_loss,
            'profit_factor': profit_factor,
            'return_std': return_std,
            'downside_std': downside_std,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown_r': max_dd,
            'calmar': calmar,
        }

    @staticmethod
    def _binary_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
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

    @staticmethod
    def _median_atr_percent(data: pd.DataFrame, base_tf: str) -> float:
        atr_col = f'{base_tf}_atr'
        if atr_col in data.columns and 'close' in data.columns:
            atr_vals = pd.to_numeric(data[atr_col], errors='coerce').to_numpy(dtype=float)
            close_vals = pd.to_numeric(data['close'], errors='coerce').to_numpy(dtype=float)
            mask = (atr_vals > 0) & (close_vals > 0)
            if mask.any():
                median_atr_pct = float(np.nanmedian(atr_vals[mask] / close_vals[mask]))
                if np.isfinite(median_atr_pct) and median_atr_pct > 0:
                    return median_atr_pct
        return 0.005

    @staticmethod
    def _estimate_span_days(data: pd.DataFrame, base_tf_seconds: Optional[float]) -> float:
        if data.empty:
            return 0.0
        if 'bar_time' in data.columns:
            times = pd.to_numeric(data['bar_time'], errors='coerce').to_numpy(dtype=float)
            mask = np.isfinite(times)
            if mask.any():
                span_sec = float(np.nanmax(times[mask]) - np.nanmin(times[mask]))
                if span_sec > 0:
                    return span_sec / 86400.0
        if 'datetime' in data.columns:
            dt = pd.to_datetime(data['datetime'], errors='coerce')
            if dt.notna().any():
                span_sec = (dt.max() - dt.min()).total_seconds()
                if span_sec > 0:
                    return span_sec / 86400.0
        if base_tf_seconds and base_tf_seconds > 0:
            span_sec = float(max(1, len(data) - 1)) * float(base_tf_seconds)
            return span_sec / 86400.0
        return 0.0

    @staticmethod
    def _compute_fee_r_series(
        data: pd.DataFrame,
        base_tf: str,
        fee_percent: float,
        stop_atr_multiple: float,
        fallback_fee_r: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        fee_r = np.full(len(data), fallback_fee_r, dtype=float)
        fallback_mask = np.ones(len(data), dtype=bool)
        atr_col = f'{base_tf}_atr'
        if atr_col not in data.columns or 'close' not in data.columns:
            return fee_r, fallback_mask

        atr_vals = pd.to_numeric(data[atr_col], errors='coerce').to_numpy(dtype=float)
        close_vals = pd.to_numeric(data['close'], errors='coerce').to_numpy(dtype=float)
        denom = stop_atr_multiple * atr_vals
        mask = (denom > 0) & (close_vals > 0) & np.isfinite(denom) & np.isfinite(close_vals)
        if mask.any():
            fee_r[mask] = (fee_percent * close_vals[mask]) / denom[mask]
            fallback_mask[mask] = False
        return fee_r, fallback_mask

    @staticmethod
    def _simulate_single_position_returns(
        data: pd.DataFrame,
        pullback_mask: pd.Series,
        trade_mask: np.ndarray,
        trend_dir: np.ndarray,
        base_tf: str,
        stop_atr_multiple: float,
        entry_forward_window: int,
        realized_r: Optional[np.ndarray],
        fallback_outcomes: np.ndarray,
        opposite_policy: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if trade_mask.size == 0:
            return np.array([], dtype=float), np.array([], dtype=int)
        if base_tf:
            atr_col = f'{base_tf}_atr'
        else:
            atr_col = 'atr'
        if 'close' not in data.columns or atr_col not in data.columns:
            idx = np.where(trade_mask)[0]
            returns = fallback_outcomes[trade_mask]
            return returns, idx

        pullback_mask_arr = pullback_mask.to_numpy(dtype=bool)
        pullback_pos = np.nonzero(pullback_mask_arr)[0]
        close_vals = pd.to_numeric(data['close'], errors='coerce').to_numpy(dtype=float)
        atr_vals = pd.to_numeric(data[atr_col], errors='coerce').to_numpy(dtype=float)

        bars_to_exit = None
        if 'pullback_bars_to_exit' in data.columns:
            bars_to_exit = pd.to_numeric(
                data.loc[pullback_mask, 'pullback_bars_to_exit'],
                errors='coerce',
            ).to_numpy(dtype=float)
        if bars_to_exit is None or bars_to_exit.size == 0:
            bars_to_exit = np.full_like(trend_dir, entry_forward_window, dtype=float)
        bars_to_exit = np.where(np.isfinite(bars_to_exit), bars_to_exit, entry_forward_window)

        if realized_r is None or realized_r.size == 0:
            realized_r = fallback_outcomes
        else:
            realized_r = np.where(np.isfinite(realized_r), realized_r, fallback_outcomes)

        trade_indices: List[int] = []
        returns: List[float] = []
        n = trade_mask.size
        i = 0

        while i < n:
            if not trade_mask[i]:
                i += 1
                continue
            direction = int(trend_dir[i])
            if direction == 0:
                i += 1
                continue
            entry_pos = int(pullback_pos[i])
            if entry_pos < 0 or entry_pos >= close_vals.size:
                i += 1
                continue
            current_atr = atr_vals[entry_pos]
            if not np.isfinite(current_atr) or current_atr <= 0:
                i += 1
                continue
            stop_dist = stop_atr_multiple * current_atr
            if not np.isfinite(stop_dist) or stop_dist <= 0:
                i += 1
                continue
            entry_price = close_vals[entry_pos]
            bars_exit = int(max(1, bars_to_exit[i]))
            natural_exit_pos = min(entry_pos + bars_exit, close_vals.size - 1)

            flipped = False
            if opposite_policy in {"flip", "close"}:
                j = i + 1
                while j < n and pullback_pos[j] <= natural_exit_pos:
                    if trade_mask[j] and int(trend_dir[j]) == -direction:
                        exit_pos = int(pullback_pos[j])
                        exit_price = close_vals[exit_pos]
                        if direction == 1:
                            realized = (exit_price - entry_price) / stop_dist
                        else:
                            realized = (entry_price - exit_price) / stop_dist
                        returns.append(float(realized))
                        trade_indices.append(i)
                        if opposite_policy == "flip":
                            i = j
                        else:
                            i = j + 1
                        flipped = True
                        break
                    j += 1

            if flipped:
                continue

            returns.append(float(realized_r[i]))
            trade_indices.append(i)
            k = i + 1
            while k < n and pullback_pos[k] <= natural_exit_pos:
                k += 1
            i = k

        return np.asarray(returns, dtype=float), np.asarray(trade_indices, dtype=int)

    def _evaluate_threshold(
        self,
        probs: np.ndarray,
        outcomes_r: np.ndarray,
        threshold: float,
        fee_per_trade_r: float,
    ) -> Dict[str, float]:
        mask = probs >= threshold
        returns = outcomes_r[mask]
        if fee_per_trade_r != 0.0:
            returns = returns - float(fee_per_trade_r)
        metrics = self._compute_trade_metrics(returns)
        coverage = float(returns.size / outcomes_r.size) if outcomes_r.size > 0 else 0.0
        metrics['coverage'] = coverage
        metrics['threshold'] = float(threshold)
        return metrics

    def _score_trade_metrics(self, metrics: Dict[str, float], *, lcb_z: Optional[float] = None) -> float:
        n_trades = int(metrics.get('n_trades', 0))
        if n_trades <= 0:
            return 0.0

        total_r = float(metrics.get('total_pnl_r', 0.0))
        std_r = float(metrics.get('return_std', 0.0))
        max_dd = float(metrics.get('max_drawdown_r', 0.0))

        if n_trades > 1 and std_r > 0:
            z_val = float(self.lcb_z) if lcb_z is None else float(lcb_z)
            lcb_total = total_r - (z_val * std_r * np.sqrt(n_trades))
        else:
            lcb_total = total_r

        dd_penalty = 1.0 / (1.0 + max_dd)
        score_curve = lcb_total * dd_penalty
        return float(score_curve)

    def _adaptive_lcb_z(self, n_trades: int) -> float:
        z_max = float(self.lcb_z)
        if not np.isfinite(z_max) or z_max <= 0.0:
            return 0.0
        if n_trades <= 0:
            return z_max

        if n_trades <= 20:
            z_val = 0.6
        elif n_trades <= 60:
            z_val = 0.6 + (0.9 - 0.6) * ((n_trades - 20) / 40.0)
        elif n_trades <= 150:
            z_val = 0.9 + (1.1 - 0.9) * ((n_trades - 60) / 90.0)
        elif n_trades <= 300:
            z_val = 1.1 + (1.28 - 1.1) * ((n_trades - 150) / 150.0)
        else:
            z_val = 1.28

        return max(0.0, min(z_val, z_max))

    def _print_trial_report(
        self,
        trial: Optional["optuna.Trial"],
        metrics: Dict[str, float],
        fold_details: List[Dict[str, Any]],
    ) -> None:
        if not self.report_trials:
            return

        trial_id = trial.number if trial is not None else "?"
        prob_mode = "raw" if self.use_raw_probabilities else "cal"
        header = (
            f"[TRIAL {trial_id}] objective={self.tuning_objective} prob={prob_mode} "
            f"score={metrics.get('score', 0.0):.6f} "
            f"profit_score={metrics.get('profit_score', 0.0):.6f} "
            f"entry_score={metrics.get('entry_score', 0.0):.6f}"
        )
        print("-" * 78)
        print(header)

        reason = metrics.get('reason')
        if reason:
            print(f"  reason: {reason}")

        print(
            "  samples: total={:.0f} tuning={:.0f} test={:.0f} features={:.0f} folds={:.0f}".format(
                metrics.get('total_samples', 0.0),
                metrics.get('tuning_samples', 0.0),
                metrics.get('test_samples', 0.0),
                metrics.get('feature_count', 0.0),
                metrics.get('folds', 0.0),
            )
        )

        breakeven = metrics.get('breakeven_threshold', metrics.get('best_threshold', 0.0))
        print(
            "  aggregate: breakeven_th={:.4f} pnl_r={:.4f} pnl_trd={:.4f} win={:.2f} "
            "pf={:.3f} dd={:.3f} sharpe={:.3f} sortino={:.3f} trades={:.0f} cov={:.3f} auc={:.3f}".format(
                breakeven,
                metrics.get('val_total_pnl_r', 0.0),
                metrics.get('val_pnl_per_trade_r', 0.0),
                metrics.get('val_win_rate', 0.0),
                metrics.get('val_profit_factor', 0.0),
                metrics.get('val_max_drawdown_r', 0.0),
                metrics.get('val_sharpe', 0.0),
                metrics.get('val_sortino', 0.0),
                metrics.get('val_trades', 0.0),
                metrics.get('val_trade_coverage', 0.0),
                metrics.get('val_auc', 0.0),
            )
        )
        print(
            "  costs: fee_r={:.4f} fee_pct={:.5f} atr_pct_med={:.5f} source={}".format(
                metrics.get('fee_per_trade_r', 0.0),
                metrics.get('fee_percent', 0.0),
                metrics.get('median_atr_percent', 0.0),
                metrics.get('fee_source', 'na'),
            )
        )
        if metrics.get('ops_cost_enabled', 0.0) > 0.0:
            print(
                "  ops_cost: target={:.1f} c1={:.4f} alpha={:.2f} rate_day_mean={:.2f} cost_mean={:.4f}".format(
                    metrics.get('ops_cost_target_trades_per_day', 0.0),
                    metrics.get('ops_cost_c1', 0.0),
                    metrics.get('ops_cost_alpha', 0.0),
                    metrics.get('trade_rate_day_mean', 0.0),
                    metrics.get('ops_cost_r_mean', 0.0),
                )
            )
        ev_grid = metrics.get('ev_margin_grid', '')
        if ev_grid:
            print(
                "  ev_margin: best={:.2f} grid=[{}]".format(
                    metrics.get('ev_margin_r', 0.0),
                    ev_grid,
                )
            )
        print(
            "  fee_fallback_pct={:.2f}".format(
                metrics.get('fee_fallback_pct', 0.0),
            )
        )

        print(
            "  stability: fold_score_iqr={:.4f} fold_trades_mean={:.1f} "
            "fold_trades_iqr={:.1f} fold_cov_mean={:.3f} fold_cov_iqr={:.3f} "
            "prob_spread={:.4f}".format(
                metrics.get('fold_score_iqr', 0.0),
                metrics.get('fold_trades_mean', 0.0),
                metrics.get('fold_trades_iqr', 0.0),
                metrics.get('fold_coverage_mean', 0.0),
                metrics.get('fold_coverage_iqr', 0.0),
                metrics.get('prob_spread', 0.0),
            )
        )
        print(
            "  constraints: min_trades_fold={:.0f} cov_min={:.3f} cov_max={:.3f} "
            "no_opp_folds={:.0f} refused_folds={:.0f} low_trade_folds={:.0f}".format(
                metrics.get('min_trades_per_fold', 0.0),
                metrics.get('coverage_min', 0.0),
                metrics.get('coverage_max', 0.0),
                metrics.get('no_opportunity_folds', 0.0),
                metrics.get('refused_trade_folds', 0.0),
                metrics.get('low_trade_folds', 0.0),
            )
        )
        opp_min = metrics.get('opportunity_min')
        if opp_min is not None:
            print(
                "  opportunity: min={:.0f} pb_val_med={:.1f} p_min={:.3f}".format(
                    metrics.get('opportunity_min', 0.0),
                    metrics.get('opportunity_pb_val_median', 0.0),
                    metrics.get('opportunity_p_min', 0.0),
                )
            )
        print(
            "  score_mult: stab={:.3f} cov_pen={:.3f} trade_conf={:.3f} fold_factor={:.2f}".format(
                metrics.get('stability_multiplier', 0.0),
                metrics.get('coverage_multiplier', 0.0),
                metrics.get('trade_confidence', 0.0),
                metrics.get('fold_factor', 0.0),
            )
        )
        trend_rej = metrics.get('trend_gate_rejected', 0.0)
        regime_rej = metrics.get('regime_gate_rejected', 0.0)
        dir_neutral = metrics.get('direction_neutral', 0.0)
        if (trend_rej or regime_rej or dir_neutral):
            print(
                "  gates: trend_rejects={:.0f} regime_rejects={:.0f} dir_neutral={:.0f}".format(
                    trend_rej,
                    regime_rej,
                    dir_neutral,
                )
            )
        print(
            "  entry: base={:.3f} raw_acc={:.3f} raw_prec={:.3f} raw_rec={:.3f} "
            "cal_acc={:.3f} cal_prec={:.3f} cal_rec={:.3f}".format(
                metrics.get('entry_val_base_rate', 0.0),
                metrics.get('entry_val_accuracy_raw', 0.0),
                metrics.get('entry_val_precision_raw', 0.0),
                metrics.get('entry_val_recall_raw', 0.0),
                metrics.get('entry_val_accuracy_cal', 0.0),
                metrics.get('entry_val_precision_cal', 0.0),
                metrics.get('entry_val_recall_cal', 0.0),
            )
        )
        print(
            "  entry_selected: prec={:.3f} ev_mean={:.3f} ev_p10={:.3f} cov={:.3f}".format(
                metrics.get('entry_val_precision_selected', 0.0),
                metrics.get('entry_val_ev_mean_selected', 0.0),
                metrics.get('entry_val_ev_p10_selected', 0.0),
                metrics.get('entry_val_coverage_selected', 0.0),
            )
        )
        print(
            "  selected_diag: brier={:.4f} logloss={:.4f} ece={:.4f} rr_ratio={:.3f} rr_mae={:.3f}".format(
                metrics.get('entry_selected_brier', 0.0),
                metrics.get('entry_selected_logloss', 0.0),
                metrics.get('entry_selected_ece', 0.0),
                metrics.get('expected_rr_bias_ratio', 0.0),
                metrics.get('expected_rr_mae', 0.0),
            )
        )
        ev_bins = metrics.get('ev_bin_summary', '')
        if ev_bins:
            print(f"  ev_bins: {ev_bins}")

        if fold_details:
            print("  folds:")
            for detail in fold_details:
                fold_id = detail.get('fold', '?')
                if detail.get('skipped'):
                    print(
                        "    fold {}: skipped reason={} train={} val={} pb_train={} pb_val={}".format(
                            fold_id,
                            detail.get('reason', 'unknown'),
                            detail.get('train_size', 0),
                            detail.get('val_size', 0),
                            detail.get('pullback_train', 0),
                            detail.get('pullback_val', 0),
                        )
                    )
                    continue
                print(
                    "    fold {}: train={} val={} pb_train={} pb_val={} breakeven={:.4f} "
                    "score={:.4f} auc={:.3f} trades={:.0f} cov={:.3f} pnl={:.3f} "
                    "win={:.3f} pf={:.3f} dd={:.3f}".format(
                        fold_id,
                        detail.get('train_size', 0),
                        detail.get('val_size', 0),
                        detail.get('pullback_train', 0),
                        detail.get('pullback_val', 0),
                        detail.get('breakeven_threshold', 0.0),
                        detail.get('fold_score', 0.0),
                        detail.get('auc', 0.0),
                        detail.get('trades', 0.0),
                        detail.get('coverage', 0.0),
                        detail.get('pnl_r', 0.0),
                        detail.get('win_rate', 0.0),
                        detail.get('profit_factor', 0.0),
                        detail.get('max_drawdown_r', 0.0),
                    )
                )
        print("-" * 78)

    def _start_pause_listener(self) -> None:
        if self._pause_listener and self._pause_listener.is_alive():
            return
        self._pause_listener_stop.clear()
        try:
            import msvcrt  # type: ignore
        except Exception:
            msvcrt = None

        def _listen_windows() -> None:
            if msvcrt is None:
                return
            while not self._pause_listener_stop.is_set():
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch in ("p", "P"):
                        self._request_pause()
                time.sleep(0.1)

        def _listen_posix() -> None:
            try:
                import sys
                import select
                import termios
                import tty
            except Exception:
                return
            if not sys.stdin.isatty():
                return
            fd = sys.stdin.fileno()
            try:
                old_settings = termios.tcgetattr(fd)
            except Exception:
                return
            try:
                tty.setcbreak(fd)
                while not self._pause_listener_stop.is_set():
                    readable, _, _ = select.select([fd], [], [], 0.1)
                    if readable:
                        ch = sys.stdin.read(1)
                        if ch in ("p", "P"):
                            self._request_pause()
            finally:
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                except Exception:
                    pass

        listener = _listen_windows if msvcrt is not None else _listen_posix
        self._pause_listener = threading.Thread(target=listener, daemon=True)
        self._pause_listener.start()

    def _stop_pause_listener(self) -> None:
        self._pause_listener_stop.set()
        if self._pause_listener and self._pause_listener.is_alive():
            self._pause_listener.join(timeout=1.0)

    def _pause_all_workers(self) -> None:
        with self._pause_cond:
            self._paused = True
            self._pause_active = True
            self._pause_cond.notify_all()

    def _resume_all_workers(self) -> None:
        with self._pause_cond:
            self._paused = False
            self._pause_active = False
            self._pause_cond.notify_all()

    def _wait_if_paused(self) -> None:
        if self.pause_requested and not self._pause_active:
            self._request_pause()
        with self._pause_cond:
            while self._paused:
                self._pause_cond.wait(timeout=0.5)

    def _request_pause(self) -> None:
        self.pause_requested = True
        if self._pause_active:
            return
        study = self._active_study
        if study is None:
            return
        if not self._pause_menu_lock.acquire(blocking=False):
            return

        def _run_menu() -> None:
            try:
                self._pause_menu(study)
            finally:
                self._pause_menu_lock.release()

        threading.Thread(target=_run_menu, daemon=True).start()

    def _get_top_trials(self, study: "optuna.Study", limit: int = 5) -> List["optuna.Trial"]:
        trials = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
        ]
        trials.sort(key=lambda t: float(t.value), reverse=True)
        return trials[:limit]

    def _trial_is_valid_for_selection(self, trial: "optuna.Trial") -> bool:
        if trial.value is None:
            return False
        if float(trial.value) <= -1e5:
            return False
        attrs = trial.user_attrs or {}
        reason = str(attrs.get("reason", "")).strip().lower()
        invalid_reasons = {
            "insufficient_pullbacks",
            "insufficient_data_for_folds",
            "no_features",
            "no_margin_candidates",
            "refused_with_opportunities",
            "insufficient_trades",
        }
        if reason in invalid_reasons:
            return False
        folds_used = float(attrs.get("folds_used", attrs.get("folds", 0.0)))
        if folds_used <= 0:
            return False
        val_trades = float(attrs.get("val_trades", 0.0))
        if val_trades <= 0:
            return False
        return True

    def _get_top_trials_by_p25(self, study: "optuna.Study", limit: int = 5) -> List["optuna.Trial"]:
        trials = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
        ]
        trials = [t for t in trials if self._trial_is_valid_for_selection(t)]
        if not trials:
            return []

        def _get_attr(trial: "optuna.Trial", key: str, default: float = 0.0) -> float:
            attrs = trial.user_attrs or {}
            value = attrs.get(key, default)
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        def _p25(trial: "optuna.Trial") -> float:
            attrs = trial.user_attrs or {}
            return float(attrs.get("fold_score_p25", trial.value or 0.0))

        def _p25_eps(trial: "optuna.Trial") -> float:
            p25_val = _p25(trial)
            iqr_val = _get_attr(trial, "fold_score_iqr", 0.0)
            return max(0.05, 0.25 * iqr_val, 0.05 * abs(p25_val))

        trials.sort(key=_p25, reverse=True)

        ranked: List["optuna.Trial"] = []
        i = 0
        while i < len(trials) and len(ranked) < limit:
            anchor = trials[i]
            anchor_p25 = _p25(anchor)
            eps = _p25_eps(anchor)
            group = [anchor]
            j = i + 1
            while j < len(trials):
                if anchor_p25 - _p25(trials[j]) <= eps:
                    group.append(trials[j])
                    j += 1
                else:
                    break

            # Tie-break within similar p25 using stability and sample confidence.
            group.sort(
                key=lambda t: (
                    _get_attr(t, "fold_score_iqr", 1e9),
                    -_get_attr(t, "trade_confidence", 0.0),
                    -_get_attr(t, "fold_score_median", 0.0),
                    -_get_attr(t, "val_pnl_per_trade_r", 0.0),
                    -_get_attr(t, "val_profit_factor", 0.0),
                    -_p25(t),
                )
            )
            ranked.extend(group)
            i = j

        return ranked[:limit]

    def _get_trial_by_number(
        self,
        study: "optuna.Study",
        trial_number: int,
    ) -> Optional["optuna.Trial"]:
        for trial in study.trials:
            if trial.number == trial_number:
                if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None:
                    return trial
                return None
        return None

    def _print_candidate_summary(self, trial: "optuna.Trial", index: int) -> None:
        attrs = trial.user_attrs or {}
        score = float(trial.value) if trial.value is not None else 0.0
        profit = float(attrs.get("profit_score", 0.0))
        trades = float(attrs.get("val_trades", 0.0))
        cov = float(attrs.get("val_trade_coverage", 0.0))
        ev_margin = float(attrs.get("ev_margin_r", 0.0))
        print(
            f"  {index}) Trial {trial.number} score={score:.4f} "
            f"profit={profit:.4f} trades={trades:.0f} cov={cov:.3f} ev_margin={ev_margin:.2f}"
        )

    def _print_candidate_details(self, trial: "optuna.Trial") -> None:
        attrs = trial.user_attrs or {}
        score = float(trial.value) if trial.value is not None else 0.0
        print("=" * 60)
        print(f"Trial {trial.number} details")
        print(f"  score:           {score:.6f}")
        print(f"  profit_score:    {float(attrs.get('profit_score', 0.0)):.6f}")
        print(f"  trades:          {float(attrs.get('val_trades', 0.0)):.0f}")
        print(f"  coverage:        {float(attrs.get('val_trade_coverage', 0.0)):.3f}")
        print(f"  pnl_r:           {float(attrs.get('val_total_pnl_r', 0.0)):.4f}")
        print(f"  pnl/trade_r:     {float(attrs.get('val_pnl_per_trade_r', 0.0)):.4f}")
        print(f"  win_rate:        {float(attrs.get('val_win_rate', 0.0)):.3f}")
        print(f"  profit_factor:   {float(attrs.get('val_profit_factor', 0.0)):.3f}")
        print(f"  max_dd_r:        {float(attrs.get('val_max_drawdown_r', 0.0)):.3f}")
        print(f"  ev_margin_r:     {float(attrs.get('ev_margin_r', 0.0)):.2f}")
        print(f"  use_expected_rr: {float(attrs.get('use_expected_rr', 0.0)):.0f}")
        print(f"  use_raw_probs:   {float(attrs.get('use_raw_probabilities', 0.0)):.0f}")
        print("=" * 60)

    def _build_candidate_config(self, trial: "optuna.Trial") -> Tuple[TrendFollowerConfig, Dict[str, Any]]:
        cfg, _ = self._apply_params(trial.params)
        attrs = trial.user_attrs or {}
        use_raw_probs = bool(float(attrs.get("use_raw_probabilities", 0.0)) >= 0.5)
        if "best_threshold" in attrs:
            cfg.labels.best_threshold = float(attrs["best_threshold"])
        if "target_rr" in attrs:
            cfg.labels.target_rr = float(attrs["target_rr"])
        if "stop_atr_multiple" in attrs:
            cfg.labels.stop_atr_multiple = float(attrs["stop_atr_multiple"])
        if "pullback_threshold" in attrs:
            cfg.labels.pullback_threshold = float(attrs["pullback_threshold"])
        if "ev_margin_r" in attrs:
            cfg.labels.ev_margin_r = float(attrs["ev_margin_r"])
        if "fee_percent" in attrs:
            cfg.labels.fee_percent = float(attrs["fee_percent"])
        if "fee_per_trade_r" in attrs:
            cfg.labels.fee_per_trade_r = float(attrs["fee_per_trade_r"])
        if "use_expected_rr" in attrs:
            cfg.labels.use_expected_rr = bool(float(attrs["use_expected_rr"]) >= 0.5)
        cfg.labels.use_calibration = not use_raw_probs
        cfg.labels.use_ev_gate = True
        return cfg, attrs

    def _prepare_candidate_backtest(
        self,
        trial: "optuna.Trial",
    ) -> Optional[Dict[str, Any]]:
        cfg, attrs = self._build_candidate_config(trial)
        use_raw_probs = bool(float(attrs.get("use_raw_probabilities", 0.0)) >= 0.5)
        use_expected_rr = bool(float(attrs.get("use_expected_rr", 0.0)) >= 0.5)
        fee_per_trade_r = None
        if "fee_per_trade_r" in attrs:
            fee_per_trade_r = float(attrs["fee_per_trade_r"])
        ops_cost_enabled = bool(float(attrs.get("ops_cost_enabled", 0.0)) >= 0.5)
        ops_cost_target = float(attrs.get("ops_cost_target_trades_per_day", 30.0))
        ops_cost_c1 = float(attrs.get("ops_cost_c1", 0.01))
        ops_cost_alpha = float(attrs.get("ops_cost_alpha", 1.7))

        base_tf = cfg.features.timeframe_names[cfg.base_timeframe_idx]
        if self._rust_pipeline_available():
            labeled, feature_cols, _labels_key = self._get_cached_labels_rust(cfg, base_tf)
        else:
            bars_dict = create_multi_timeframe_bars(
                self.trades,
                cfg.features.timeframes,
                cfg.features.timeframe_names,
                cfg.data,
            )
            featured = calculate_multi_timeframe_features(bars_dict, base_tf, cfg.features)
            labeled, feature_cols = create_training_dataset(featured, cfg.labels, cfg.features, base_tf)

        if not feature_cols:
            print("No features available; backtest skipped.")
            return None

        total_samples = len(labeled)
        tuning_ratio = float(cfg.model.train_ratio) + float(cfg.model.val_ratio)
        tuning_end = int(total_samples * tuning_ratio)
        tuning_end = max(0, min(total_samples, tuning_end))
        train_df = labeled.iloc[:tuning_end]
        if train_df.empty:
            print("Training split is empty; backtest skipped.")
            return None

        pullback_mask_train = ~train_df["pullback_success"].isna()
        if pullback_mask_train.sum() < 10:
            print("Insufficient pullbacks in training split; backtest skipped.")
            return None

        X_train = train_df[feature_cols].fillna(0)

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

        trend_pred_train = None
        trend_model = None
        if "trend_label" in train_df.columns and train_df["trend_label"].nunique() >= 2:
            trend_pred_train = _trainer_oof_context_preds(
                TrendClassifier,
                X_train,
                train_df["trend_label"],
                cfg,
                n_folds=3,
            )
            trend_model = TrendClassifier(cfg.model)
            trend_model.train(
                X_train,
                train_df["trend_label"],
                None,
                None,
                verbose=False,
            )

        regime_pred_train = None
        regime_model = None
        if "regime" in train_df.columns and train_df["regime"].nunique() >= 2:
            regime_pred_train = _trainer_oof_context_preds(
                RegimeClassifier,
                X_train,
                train_df["regime"],
                cfg,
                n_folds=3,
            )
            regime_model = RegimeClassifier(cfg.model)
            regime_model.train(
                X_train,
                train_df["regime"],
                None,
                None,
                verbose=False,
            )

        rr_col = "pullback_win_r" if "pullback_win_r" in train_df.columns else "pullback_rr"
        y_success_train = train_df.loc[pullback_mask_train, "pullback_success"].astype(int)
        y_rr_train = train_df.loc[pullback_mask_train, rr_col]
        X_entry_train = append_context_features(
            X_train[pullback_mask_train],
            _slice_pred(trend_pred_train, pullback_mask_train),
            _slice_pred(regime_pred_train, pullback_mask_train),
        )

        entry_model = EntryQualityModel(cfg.model)
        entry_model.train(
            X_entry_train,
            y_success_train,
            y_rr_train,
            None,
            None,
            None,
            verbose=False,
            calibrate=True,
            calibration_mode="oof",
            calibration_oof_folds=3,
            calibration_method=self.calibration_method,
            use_noise_filtering=self.use_noise_filtering,
            use_seed_ensemble=self.use_seed_ensemble,
            n_ensemble_seeds=self.n_ensemble_seeds,
        )

        models = TrendFollowerModels(cfg.model)
        models.entry_model = entry_model
        if trend_model is not None:
            models.trend_classifier = trend_model
        if regime_model is not None:
            models.regime_classifier = regime_model

        cfg.labels.use_calibration = not use_raw_probs
        cfg.labels.use_expected_rr = bool(use_expected_rr)
        cfg.labels.use_ev_gate = True
        if fee_per_trade_r is not None:
            cfg.labels.fee_per_trade_r = float(fee_per_trade_r)

        max_holding_bars = None
        if hasattr(cfg.labels, "entry_forward_window"):
            max_holding_bars = int(cfg.labels.entry_forward_window)

        return {
            "cfg": cfg,
            "attrs": attrs,
            "labeled": labeled,
            "feature_cols": feature_cols,
            "models": models,
            "tuning_end": tuning_end,
            "use_raw_probs": use_raw_probs,
            "use_expected_rr": use_expected_rr,
            "fee_per_trade_r": fee_per_trade_r,
            "ops_cost_enabled": ops_cost_enabled,
            "ops_cost_target": ops_cost_target,
            "ops_cost_c1": ops_cost_c1,
            "ops_cost_alpha": ops_cost_alpha,
            "max_holding_bars": max_holding_bars,
        }

    def _run_candidate_backtest(
        self,
        trial: "optuna.Trial",
        *,
        use_full_data: bool = False,
    ) -> Optional[Dict[str, Any]]:
        prepared = self._prepare_candidate_backtest(trial)
        if prepared is None:
            return None
        cfg = prepared["cfg"]
        labeled = prepared["labeled"]
        feature_cols = prepared["feature_cols"]
        models = prepared["models"]
        use_raw_probs = bool(prepared["use_raw_probs"])
        use_expected_rr = bool(prepared["use_expected_rr"])
        fee_per_trade_r = prepared["fee_per_trade_r"]
        ops_cost_enabled = bool(prepared["ops_cost_enabled"])
        ops_cost_target = float(prepared["ops_cost_target"])
        ops_cost_c1 = float(prepared["ops_cost_c1"])
        ops_cost_alpha = float(prepared["ops_cost_alpha"])
        max_holding_bars = prepared["max_holding_bars"]

        res = run_tuned_backtest(
            labeled,
            feature_cols,
            models,
            cfg,
            use_full_data=bool(use_full_data),
            trade_side="both",
            use_ev_gate=bool(cfg.labels.use_ev_gate),
            ev_margin_r=float(cfg.labels.ev_margin_r),
            min_bounce_prob=float(getattr(cfg.labels, "best_threshold", 0.5)),
            max_bounce_prob=1.0,
            use_raw_probabilities=bool(use_raw_probs),
            use_calibration=not use_raw_probs,
            use_expected_rr=bool(use_expected_rr),
            fee_percent=float(cfg.labels.fee_percent),
            fee_per_trade_r=fee_per_trade_r,
            ops_cost_enabled=bool(ops_cost_enabled),
            ops_cost_target_trades_per_day=float(ops_cost_target),
            ops_cost_c1=float(ops_cost_c1),
            ops_cost_alpha=float(ops_cost_alpha),
            single_position=bool(self.single_position),
            opposite_signal_policy=str(self.opposite_signal_policy),
            max_holding_bars=max_holding_bars,
            ema_touch_mode="multi",
        )
        print_backtest_results(res)

        return {
            "total_trades": res.total_trades,
            "pnl": res.total_pnl,
            "profit_factor": res.profit_factor,
            "max_drawdown": res.max_drawdown,
            "win_rate": res.win_rate,
            "ev_margin_r": float(cfg.labels.ev_margin_r),
        }

    def _split_test_shadow(
        self,
        labeled: pd.DataFrame,
        cfg: TrendFollowerConfig,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
        total_samples = len(labeled)
        tuning_ratio = float(cfg.model.train_ratio) + float(cfg.model.val_ratio)
        tuning_end = int(total_samples * tuning_ratio)
        tuning_end = max(0, min(total_samples, tuning_end))
        test_df = labeled.iloc[tuning_end:]
        if test_df.empty:
            return test_df, test_df.iloc[:0], tuning_end

        total_pct = float(self.selection_test_pct + self.selection_shadow_pct)
        if total_pct <= 0:
            return test_df, test_df.iloc[:0], tuning_end

        shadow_ratio = float(self.selection_shadow_pct) / total_pct
        shadow_size = int(round(len(test_df) * shadow_ratio))
        shadow_size = max(0, min(len(test_df) - 1, shadow_size))
        if shadow_size == 0:
            return test_df, test_df.iloc[:0], tuning_end

        test_main = test_df.iloc[:-shadow_size]
        shadow_df = test_df.iloc[-shadow_size:]
        return test_main, shadow_df, tuning_end

    def _summarize_backtest_result(self, res: "BacktestResult") -> Dict[str, float]:
        stats = res.signal_stats or {}
        checked = float(stats.get("signals_checked", 0.0))
        accepted = float(stats.get("accepted_signals", 0.0))
        coverage = (accepted / checked) if checked > 0 else 0.0
        diag = res.diagnostics or {}
        realized_r_net_mean = float(diag.get("realized_r_net_mean", 0.0))
        total_r_net = realized_r_net_mean * float(res.total_trades)
        return {
            "total_trades": float(res.total_trades),
            "coverage": coverage,
            "total_pnl_r": total_r_net,
            "pnl_per_trade_r": realized_r_net_mean,
            "profit_factor": float(res.profit_factor),
            "max_drawdown": float(res.max_drawdown),
            "max_drawdown_percent": float(res.max_drawdown_percent),
            "win_rate": float(res.win_rate),
            "total_pnl": float(res.total_pnl),
        }

    def _selection_passes(
        self,
        metrics: Dict[str, float],
        *,
        min_trades: Optional[int] = None,
    ) -> Tuple[bool, List[str]]:
        reasons = []
        min_trades = self.selection_min_trades if min_trades is None else int(min_trades)
        if metrics.get("total_trades", 0.0) < float(min_trades):
            reasons.append("min_trades")
        if metrics.get("total_pnl_r", 0.0) <= float(self.selection_min_total_r):
            reasons.append("pnl_r")
        if metrics.get("profit_factor", 0.0) <= float(self.selection_min_profit_factor):
            reasons.append("profit_factor")
        if metrics.get("max_drawdown_percent", 0.0) >= float(self.selection_max_drawdown_pct):
            reasons.append("max_drawdown_pct")
        if metrics.get("coverage", 0.0) >= float(self.selection_max_coverage):
            reasons.append("coverage")
        return (len(reasons) == 0), reasons

    def _run_candidate_selection(self, study: "optuna.Study") -> None:
        if not self.selection_enabled:
            print("Candidate selection is disabled.")
            return
        top_n = self._resolve_selection_top_n(study)
        top_trials = self._get_top_trials_by_p25(study, limit=top_n)
        if not top_trials:
            print("No completed trials available for selection.")
            return

        print("\nCandidate selection (test + shadow holdout):")
        print(f"  Top N:          {top_n}")
        print(f"  Test pct:       {self.selection_test_pct:.1%}")
        print(f"  Shadow pct:     {self.selection_shadow_pct:.1%}")
        print(
            f"  Min trades:     {self.selection_min_trades} (test) / "
            f"{self.selection_min_trades_shadow} (shadow, cap {self.selection_shadow_cap})"
        )
        print(f"  Min PF:         {self.selection_min_profit_factor:.2f}")
        print(f"  Max DD%:        {self.selection_max_drawdown_pct:.2f}")
        print(f"  Max coverage:   {self.selection_max_coverage:.2f}")

        passed_trials = []
        for trial in top_trials:
            prepared = self._prepare_candidate_backtest(trial)
            if prepared is None:
                print(f"  Trial {trial.number}: skipped (prep failed)")
                continue
            cfg = prepared["cfg"]
            labeled = prepared["labeled"]
            feature_cols = prepared["feature_cols"]
            models = prepared["models"]
            use_raw_probs = bool(prepared["use_raw_probs"])
            use_expected_rr = bool(prepared["use_expected_rr"])
            fee_per_trade_r = prepared["fee_per_trade_r"]
            ops_cost_enabled = bool(prepared["ops_cost_enabled"])
            ops_cost_target = float(prepared["ops_cost_target"])
            ops_cost_c1 = float(prepared["ops_cost_c1"])
            ops_cost_alpha = float(prepared["ops_cost_alpha"])
            max_holding_bars = prepared["max_holding_bars"]

            test_df, shadow_df, _tuning_end = self._split_test_shadow(labeled, cfg)
            if test_df.empty:
                print(f"  Trial {trial.number}: skipped (empty test split)")
                continue

            print(f"\nTrial {trial.number} selection backtest (test split):")
            test_res = run_tuned_backtest(
                test_df,
                feature_cols,
                models,
                cfg,
                use_full_data=True,
                trade_side="both",
                use_ev_gate=bool(cfg.labels.use_ev_gate),
                ev_margin_r=float(cfg.labels.ev_margin_r),
                min_bounce_prob=float(getattr(cfg.labels, "best_threshold", 0.5)),
                max_bounce_prob=1.0,
                use_raw_probabilities=bool(use_raw_probs),
                use_calibration=not use_raw_probs,
                use_expected_rr=bool(use_expected_rr),
                fee_percent=float(cfg.labels.fee_percent),
                fee_per_trade_r=fee_per_trade_r,
                ops_cost_enabled=bool(ops_cost_enabled),
                ops_cost_target_trades_per_day=float(ops_cost_target),
                ops_cost_c1=float(ops_cost_c1),
                ops_cost_alpha=float(ops_cost_alpha),
                single_position=bool(self.single_position),
                opposite_signal_policy=str(self.opposite_signal_policy),
                max_holding_bars=max_holding_bars,
                ema_touch_mode="multi",
                confirm_missing_models=False,
            )
            print_backtest_results(test_res)
            test_metrics = self._summarize_backtest_result(test_res)
            test_trades = int(test_metrics.get("total_trades", 0.0))
            test_pass, test_reasons = self._selection_passes(test_metrics, min_trades=self.selection_min_trades)

            shadow_metrics: Dict[str, float] = {}
            shadow_pass = False
            shadow_reasons: List[str] = []
            if not shadow_df.empty:
                print(f"\nTrial {trial.number} selection backtest (shadow holdout):")
                shadow_min_trades = self.selection_min_trades_shadow
                if self.selection_test_pct > 0:
                    shadow_min_trades = max(
                        5 if self.selection_shadow_pct > 0 else 0,
                        int(round(test_trades * (self.selection_shadow_pct / self.selection_test_pct))),
                    )
                if self.selection_shadow_cap > 0:
                    shadow_min_trades = min(shadow_min_trades, self.selection_shadow_cap)
                shadow_res = run_tuned_backtest(
                    shadow_df,
                    feature_cols,
                    models,
                    cfg,
                    use_full_data=True,
                    trade_side="both",
                    use_ev_gate=bool(cfg.labels.use_ev_gate),
                    ev_margin_r=float(cfg.labels.ev_margin_r),
                    min_bounce_prob=float(getattr(cfg.labels, "best_threshold", 0.5)),
                    max_bounce_prob=1.0,
                    use_raw_probabilities=bool(use_raw_probs),
                    use_calibration=not use_raw_probs,
                    use_expected_rr=bool(use_expected_rr),
                    fee_percent=float(cfg.labels.fee_percent),
                    fee_per_trade_r=fee_per_trade_r,
                    ops_cost_enabled=bool(ops_cost_enabled),
                    ops_cost_target_trades_per_day=float(ops_cost_target),
                    ops_cost_c1=float(ops_cost_c1),
                    ops_cost_alpha=float(ops_cost_alpha),
                    single_position=bool(self.single_position),
                    opposite_signal_policy=str(self.opposite_signal_policy),
                    max_holding_bars=max_holding_bars,
                    ema_touch_mode="multi",
                    confirm_missing_models=False,
                )
                print_backtest_results(shadow_res)
                shadow_metrics = self._summarize_backtest_result(shadow_res)
                shadow_pass, shadow_reasons = self._selection_passes(
                    shadow_metrics,
                    min_trades=shadow_min_trades,
                )

            shadow_required = bool(self.selection_shadow_pct > 0 and not shadow_df.empty)
            requires_review = bool(test_pass and shadow_required and not shadow_pass)
            passed = bool(test_pass and (shadow_pass if shadow_required else True))
            if passed:
                passed_trials.append(trial.number)

            print(
                f"Trial {trial.number} selection: "
                f"test={'PASS' if test_pass else 'FAIL'} "
                f"shadow={'PASS' if shadow_pass else 'FAIL'} "
                f"status={'REVIEW' if requires_review else ('PASS' if passed else 'FAIL')} "
                f"reasons={','.join(sorted(set(test_reasons + shadow_reasons))) or 'ok'}"
            )

            self._save_candidate_summary(
                trial,
                study,
                test_metrics=test_metrics,
                shadow_metrics=shadow_metrics,
                selection_passed=passed,
                selection_requires_review=requires_review,
                write_best=False,
            )

        if passed_trials:
            print(f"\nSelection passed trials: {', '.join(str(n) for n in passed_trials)}")
        else:
            print("\nSelection passed trials: none")

    def _resolve_selection_top_n(self, study: "optuna.Study") -> int:
        if not self.selection_top_n:
            return 5
        if self.selection_top_n < 0:
            total_trials = len(study.trials)
            adaptive = int(round(max(5.0, total_trials * 0.05)))
            return max(5, min(50, adaptive))
        return int(self.selection_top_n)

    def _run_candidate_backtest_folds(
        self,
        trial: "optuna.Trial",
    ) -> Optional[Dict[str, Any]]:
        cfg, attrs = self._build_candidate_config(trial)
        use_raw_probs = bool(float(attrs.get("use_raw_probabilities", 0.0)) >= 0.5)
        use_expected_rr = bool(float(attrs.get("use_expected_rr", 0.0)) >= 0.5)
        fee_per_trade_r = None
        if "fee_per_trade_r" in attrs:
            fee_per_trade_r = float(attrs["fee_per_trade_r"])
        ops_cost_enabled = bool(float(attrs.get("ops_cost_enabled", 0.0)) >= 0.5)
        ops_cost_target = float(attrs.get("ops_cost_target_trades_per_day", 30.0))
        ops_cost_c1 = float(attrs.get("ops_cost_c1", 0.01))
        ops_cost_alpha = float(attrs.get("ops_cost_alpha", 1.7))

        base_tf = cfg.features.timeframe_names[cfg.base_timeframe_idx]
        if self._rust_pipeline_available():
            labeled, feature_cols, _labels_key = self._get_cached_labels_rust(cfg, base_tf)
        else:
            bars_dict = create_multi_timeframe_bars(
                self.trades,
                cfg.features.timeframes,
                cfg.features.timeframe_names,
                cfg.data,
            )
            featured = calculate_multi_timeframe_features(bars_dict, base_tf, cfg.features)
            labeled, feature_cols = create_training_dataset(featured, cfg.labels, cfg.features, base_tf)

        if not feature_cols:
            print("No features available; backtest skipped.")
            return None

        folds = self._get_walk_forward_splits(labeled, cfg)
        if not folds:
            print("No folds available; backtest skipped.")
            return None

        cfg.labels.use_calibration = not use_raw_probs
        cfg.labels.use_expected_rr = bool(use_expected_rr)
        cfg.labels.use_ev_gate = True
        if fee_per_trade_r is not None:
            cfg.labels.fee_per_trade_r = float(fee_per_trade_r)

        max_holding_bars = None
        if hasattr(cfg.labels, "entry_forward_window"):
            max_holding_bars = int(cfg.labels.entry_forward_window)

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

        def _oof_context_preds(model_cls, X, y, n_folds: int = 3) -> Optional[Dict[str, np.ndarray]]:
            if X is None or len(X) == 0 or n_folds < 2:
                return None
            n = len(X)
            if n < max(30, n_folds * 10):
                return None
            y_series = pd.Series(y).reset_index(drop=True)
            if y_series.nunique() < 2:
                return None

            indices = np.arange(n)
            fold_indices = np.array_split(indices, n_folds)
            preds: Optional[Dict[str, np.ndarray]] = None

            for idx in fold_indices:
                if idx.size == 0:
                    continue
                train_end = int(idx[0])
                if train_end <= 1:
                    continue
                y_train = y_series.iloc[:train_end]
                if y_train.nunique() < 2:
                    continue
                model = model_cls(cfg.model)
                model.train(
                    X.iloc[:train_end],
                    y_train,
                    None,
                    None,
                    verbose=False,
                )
                fold_pred = model.predict(X.iloc[idx])
                if preds is None:
                    preds = {key: np.zeros(n, dtype=float) for key in fold_pred.keys()}
                for key, values in fold_pred.items():
                    arr = np.asarray(values)
                    if key in preds and arr.shape[0] == idx.size:
                        preds[key][idx] = arr

            return preds

        all_returns: List[float] = []
        total_checked = 0
        total_accepted = 0
        folds_run = 0
        skipped_folds = 0

        for fold_idx, (train_df, val_df) in enumerate(folds, start=1):
            if train_df.empty or val_df.empty:
                skipped_folds += 1
                print(f"Fold {fold_idx} skipped (empty split).")
                continue
            pullback_mask_train = ~train_df["pullback_success"].isna()
            pullback_mask_val = ~val_df["pullback_success"].isna()
            pullback_train = int(pullback_mask_train.sum())
            pullback_val = int(pullback_mask_val.sum())
            if pullback_train < self.min_pullback_samples or pullback_val < self.min_pullback_val_samples:
                skipped_folds += 1
                print(
                    f"Fold {fold_idx} skipped (insufficient pullbacks: "
                    f"train={pullback_train}, val={pullback_val})."
                )
                continue

            X_train = train_df[feature_cols].fillna(0)
            X_val = val_df[feature_cols].fillna(0)

            trend_pred_train = None
            trend_pred_val = None
            if "trend_label" in train_df.columns and "trend_label" in val_df.columns:
                if train_df["trend_label"].nunique() >= 2:
                    trend_pred_train = _oof_context_preds(
                        TrendClassifier,
                        X_train,
                        train_df["trend_label"],
                        n_folds=3,
                    )
                    trend_model = TrendClassifier(cfg.model)
                    trend_model.train(
                        X_train,
                        train_df["trend_label"],
                        X_val,
                        val_df["trend_label"],
                        verbose=False,
                    )
                    trend_pred_val = trend_model.predict(X_val)
                else:
                    trend_model = None
            else:
                trend_model = None

            regime_pred_train = None
            regime_pred_val = None
            if "regime" in train_df.columns and "regime" in val_df.columns and train_df["regime"].nunique() >= 2:
                regime_pred_train = _oof_context_preds(
                    RegimeClassifier,
                    X_train,
                    train_df["regime"],
                    n_folds=3,
                )
                regime_model = RegimeClassifier(cfg.model)
                regime_model.train(
                    X_train,
                    train_df["regime"],
                    X_val,
                    val_df["regime"],
                    verbose=False,
                )
                regime_pred_val = regime_model.predict(X_val)
            else:
                regime_model = None

            X_entry_train = append_context_features(
                X_train[pullback_mask_train],
                _slice_pred(trend_pred_train, pullback_mask_train),
                _slice_pred(regime_pred_train, pullback_mask_train),
            )
            y_success_train = train_df.loc[pullback_mask_train, "pullback_success"].astype(int)
            rr_col = "pullback_win_r" if "pullback_win_r" in train_df.columns else "pullback_rr"
            y_rr_train = train_df.loc[pullback_mask_train, rr_col]

            X_entry_val = append_context_features(
                X_val[pullback_mask_val],
                _slice_pred(trend_pred_val, pullback_mask_val),
                _slice_pred(regime_pred_val, pullback_mask_val),
            )
            y_success_val = val_df.loc[pullback_mask_val, "pullback_success"].astype(int)
            rr_col_val = "pullback_win_r" if "pullback_win_r" in val_df.columns else "pullback_rr"
            y_rr_val = val_df.loc[pullback_mask_val, rr_col_val]

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
                calibration_mode="oof",
                calibration_oof_folds=3,
                calibration_method=self.calibration_method,
                use_noise_filtering=self.use_noise_filtering,
                use_seed_ensemble=self.use_seed_ensemble,
                n_ensemble_seeds=self.n_ensemble_seeds,
            )

            models = TrendFollowerModels(cfg.model)
            models.entry_model = entry_model
            if trend_model is not None:
                models.trend_classifier = trend_model
            if regime_model is not None:
                models.regime_classifier = regime_model

            res = run_tuned_backtest(
                val_df,
                feature_cols,
                models,
                cfg,
                use_full_data=True,
                trade_side="both",
                use_ev_gate=bool(cfg.labels.use_ev_gate),
                ev_margin_r=float(cfg.labels.ev_margin_r),
                min_bounce_prob=float(getattr(cfg.labels, "best_threshold", 0.5)),
                max_bounce_prob=1.0,
                use_raw_probabilities=bool(use_raw_probs),
                use_calibration=not use_raw_probs,
                use_expected_rr=bool(use_expected_rr),
                fee_percent=float(cfg.labels.fee_percent),
                fee_per_trade_r=fee_per_trade_r,
                ops_cost_enabled=bool(ops_cost_enabled),
                ops_cost_target_trades_per_day=float(ops_cost_target),
                ops_cost_c1=float(ops_cost_c1),
                ops_cost_alpha=float(ops_cost_alpha),
                single_position=bool(self.single_position),
                opposite_signal_policy=str(self.opposite_signal_policy),
                max_holding_bars=max_holding_bars,
                ema_touch_mode="multi",
            )
            print(f"\nFold {fold_idx} backtest:")
            print_backtest_results(res)

            folds_run += 1
            if res.signal_stats:
                total_checked += int(res.signal_stats.get("signals_checked", 0))
                total_accepted += int(res.signal_stats.get("accepted_signals", 0))
            for trade in res.trades:
                all_returns.append(float(getattr(trade, "realized_r_net", 0.0)))

        if folds_run == 0:
            print("No folds backtested.")
            return None

        returns = np.asarray(all_returns, dtype=float)
        metrics = _compute_trade_metrics(returns)
        coverage = float(total_accepted / total_checked) if total_checked > 0 else 0.0

        print("\nFold Aggregate Summary (R units):")
        print(f"  Folds Run:      {folds_run} (skipped {skipped_folds})")
        print(f"  Trades:         {metrics.get('n_trades', 0.0):.0f}")
        print(f"  Coverage:       {coverage:.3f}")
        print(f"  Total PnL (R):  {metrics.get('total_pnl_r', 0.0):.4f}")
        print(f"  PnL/Trade (R):  {metrics.get('avg_pnl_r', 0.0):.4f}")
        print(f"  Win Rate:       {metrics.get('win_rate', 0.0):.3f}")
        print(f"  Profit Factor:  {metrics.get('profit_factor', 0.0):.3f}")
        print(f"  Max DD (R):     {metrics.get('max_drawdown_r', 0.0):.4f}")
        print(f"  Sharpe:         {metrics.get('sharpe', 0.0):.3f}")
        print(f"  Sortino:        {metrics.get('sortino', 0.0):.3f}")

        return {
            "folds_run": folds_run,
            "total_trades": int(metrics.get("n_trades", 0.0)),
            "coverage": coverage,
            "total_pnl_r": float(metrics.get("total_pnl_r", 0.0)),
            "avg_pnl_r": float(metrics.get("avg_pnl_r", 0.0)),
            "win_rate": float(metrics.get("win_rate", 0.0)),
            "profit_factor": float(metrics.get("profit_factor", 0.0)),
            "max_drawdown_r": float(metrics.get("max_drawdown_r", 0.0)),
            "ev_margin_r": float(cfg.labels.ev_margin_r),
        }

    def _save_candidate_summary(
        self,
        trial: "optuna.Trial",
        study: "optuna.Study",
        *,
        test_metrics: Optional[Dict[str, float]] = None,
        shadow_metrics: Optional[Dict[str, float]] = None,
        selection_passed: Optional[bool] = None,
        selection_requires_review: Optional[bool] = None,
        write_best: bool = True,
    ) -> Path:
        cfg, attrs = self._build_candidate_config(trial)
        summary = {
            'best_score': float(trial.value) if trial.value is not None else 0.0,
            'best_params': dict(trial.params),
            'best_metrics': dict(attrs),
            'best_config': serialize_config(cfg),
            'tuner_settings': {
                'min_pullback_samples': int(self.min_pullback_samples),
                'min_pullback_val_samples': int(self.min_pullback_val_samples),
                'min_trades': int(self.min_trades),
                'min_trades_per_fold': None if self.min_trades_per_fold is None else int(self.min_trades_per_fold),
                'min_coverage': float(self.min_coverage),
                'max_coverage': float(self.max_coverage),
                'lcb_z': float(self.lcb_z),
                'stability_k': float(self.stability_k),
                'coverage_k': float(self.coverage_k),
                'stability_eps': float(self.stability_eps),
                'no_opportunity_penalty': float(self.no_opportunity_penalty),
                'refusal_penalty': float(self.refusal_penalty),
                'ops_cost_enabled': bool(self.ops_cost_enabled),
                'ops_cost_target_trades_per_day': float(self.ops_cost_target_trades_per_day),
                'ops_cost_c1': float(self.ops_cost_c1),
                'ops_cost_alpha': float(self.ops_cost_alpha),
                'single_position': bool(self.single_position),
                'opposite_signal_policy': str(self.opposite_signal_policy),
                'use_noise_filtering': bool(self.use_noise_filtering),
                'use_seed_ensemble': bool(self.use_seed_ensemble),
                'n_ensemble_seeds': int(self.n_ensemble_seeds),
                'calibration_method': str(self.calibration_method),
                'use_raw_probabilities': bool(self.use_raw_probabilities),
                'use_expected_rr': bool(self.use_expected_rr),
                'fee_percent': float(self.fee_percent),
                'fee_per_trade_r': None if self.fee_per_trade_r is None else float(self.fee_per_trade_r),
                'selection_enabled': bool(self.selection_enabled),
                'selection_top_n': int(self.selection_top_n),
                'selection_test_pct': float(self.selection_test_pct),
                'selection_shadow_pct': float(self.selection_shadow_pct),
                'selection_min_trades': int(self.selection_min_trades),
                'selection_min_trades_shadow': None if self.selection_min_trades_shadow is None else int(self.selection_min_trades_shadow),
                'selection_shadow_cap': int(self.selection_shadow_cap),
                'selection_min_profit_factor': float(self.selection_min_profit_factor),
                'selection_max_drawdown_pct': float(self.selection_max_drawdown_pct),
                'selection_max_coverage': float(self.selection_max_coverage),
                'selection_min_total_r': float(self.selection_min_total_r),
            },
            'trials_completed': len(study.trials),
            'elapsed_seconds': float(time.time() - (self._tune_start_time or time.time())),
            'selected_trial': int(trial.number),
        }
        if test_metrics is not None:
            summary["selection_test_metrics"] = dict(test_metrics)
        if shadow_metrics is not None:
            summary["selection_shadow_metrics"] = dict(shadow_metrics)
        if selection_passed is not None:
            summary["selection_passed"] = bool(selection_passed)
        if selection_requires_review is not None:
            summary["selection_requires_review"] = bool(selection_requires_review)
        summary['best_threshold'] = float(getattr(cfg.labels, "best_threshold", 0.5))
        summary['target_rr'] = float(cfg.labels.target_rr)
        summary['stop_atr_multiple'] = float(cfg.labels.stop_atr_multiple)
        summary['pullback_threshold'] = float(cfg.labels.pullback_threshold)
        summary_dir = Path(cfg.model.model_dir)
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / f"tuning_summary_{trial.number}.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        if write_best:
            print(f"Saved tuning summary: {summary_path}")
        else:
            print(f"Saved trial summary: {summary_path}")
        return summary_path

    def _candidate_menu(self, study: "optuna.Study") -> None:
        while True:
            top_trials = self._get_top_trials(study, limit=5)
            if not top_trials:
                print("No completed trials available yet.")
                return
            print("\nTop candidates:")
            for i, trial in enumerate(top_trials, 1):
                self._print_candidate_summary(trial, i)
            print("  t) Select trial by number")
            choice = input("Select candidate #, 't' for trial number, or 'b' to go back: ").strip().lower()
            if choice in ("b", "back", ""):
                return
            if choice in ("t", "trial"):
                trial_input = input("Enter trial number: ").strip()
                if not trial_input.isdigit():
                    print("Invalid trial number.")
                    continue
                trial_num = int(trial_input)
                trial = self._get_trial_by_number(study, trial_num)
                if trial is None:
                    print("Trial not found or not complete.")
                    continue
                self._candidate_detail_menu(trial, study)
                continue
            if not choice.isdigit():
                continue
            idx = int(choice)
            if idx < 1 or idx > len(top_trials):
                continue
            self._candidate_detail_menu(top_trials[idx - 1], study)

    def _candidate_detail_menu(self, trial: "optuna.Trial", study: "optuna.Study") -> None:
        while True:
            self._print_candidate_details(trial)
            test_pct = float(self.base_config.model.test_ratio) * 100.0
            print(f"  1) Backtest this ({test_pct:.0f}% test set)")
            print("  2) Backtest this (full set)")
            print("  3) Backtest this (folds / tuning splits)")
            print("  4) Save to disk")
            print("  5) Go back")
            choice = input("Select: ").strip().lower()
            if choice == "1":
                res = self._run_candidate_backtest(trial, use_full_data=False)
                if res is not None:
                    while True:
                        post = input("Backtest done. Save to disk? (y/n, b=back): ").strip().lower()
                        if post in ("y", "yes"):
                            path = self._save_candidate_summary(trial, study)
                            print(f"Saved: {path}")
                            break
                        if post in ("n", "no", "b", "back", ""):
                            break
                continue
            if choice == "2":
                res = self._run_candidate_backtest(trial, use_full_data=True)
                if res is not None:
                    while True:
                        post = input("Backtest done. Save to disk? (y/n, b=back): ").strip().lower()
                        if post in ("y", "yes"):
                            path = self._save_candidate_summary(trial, study)
                            print(f"Saved: {path}")
                            break
                        if post in ("n", "no", "b", "back", ""):
                            break
                continue
            if choice == "3":
                res = self._run_candidate_backtest_folds(trial)
                if res is not None:
                    while True:
                        post = input("Backtest done. Save to disk? (y/n, b=back): ").strip().lower()
                        if post in ("y", "yes"):
                            path = self._save_candidate_summary(trial, study)
                            print(f"Saved: {path}")
                            break
                        if post in ("n", "no", "b", "back", ""):
                            break
                continue
            if choice == "4":
                path = self._save_candidate_summary(trial, study)
                print(f"Saved: {path}")
                continue
            if choice == "5" or choice in ("b", "back", ""):
                return

    def _pause_menu(self, study: "optuna.Study") -> None:
        self.pause_requested = False
        if not self._pause_active:
            self._pause_all_workers()
        self._stop_pause_listener()
        while True:
            print("\nTUNING PAUSED")
            print("  1) Backtest")
            print("  2) Run selection (test + shadow holdout)")
            print("  3) Resume tuning")
            choice = input("Select: ").strip().lower()
            if choice == "1":
                self._candidate_menu(study)
                continue
            if choice == "2":
                self._run_candidate_selection(study)
                continue
            if choice == "3" or choice in ("r", "resume", ""):
                break
        self._resume_all_workers()
        self._start_pause_listener()

    def _apply_trial(self, trial: "optuna.Trial") -> Tuple[TrendFollowerConfig, Dict[str, Any]]:
        cfg = deepcopy(self.base_config)
        param_log: Dict[str, Any] = {}

        if self.tune_labels:
            cfg.labels.trend_forward_window = trial.suggest_int("trend_forward_window", 10, 60)
            cfg.labels.entry_forward_window = trial.suggest_int("entry_forward_window", 5, 30)
            cfg.labels.trend_up_threshold = trial.suggest_float("trend_up_threshold", 0.8, 3.5)
            cfg.labels.trend_down_threshold = trial.suggest_float("trend_down_threshold", 0.8, 3.5)
            cfg.labels.max_adverse_for_trend = trial.suggest_float("max_adverse_for_trend", 0.5, 2.0)
            cfg.labels.target_rr = trial.suggest_float("target_rr", 1.0, 3.0)
            cfg.labels.stop_atr_multiple = trial.suggest_float("stop_atr_multiple", 0.5, 2.0)
            cfg.labels.pullback_ema = trial.suggest_int("pullback_ema", 5, 55)
            cfg.labels.pullback_threshold = trial.suggest_float("pullback_threshold", 0.02, 0.6)

            cfg.labels.use_trend_gate = False
            cfg.labels.min_trend_prob = 0.0
            cfg.labels.use_regime_gate = False
            cfg.labels.min_regime_prob = 0.0
            cfg.labels.regime_align_direction = False
            cfg.labels.allow_regime_ranging = True
            cfg.labels.allow_regime_trend_up = True
            cfg.labels.allow_regime_trend_down = True
            cfg.labels.allow_regime_volatile = True

            param_log.update({
                'trend_forward_window': cfg.labels.trend_forward_window,
                'entry_forward_window': cfg.labels.entry_forward_window,
                'trend_up_threshold': cfg.labels.trend_up_threshold,
                'trend_down_threshold': cfg.labels.trend_down_threshold,
                'max_adverse_for_trend': cfg.labels.max_adverse_for_trend,
                'target_rr': cfg.labels.target_rr,
                'stop_atr_multiple': cfg.labels.stop_atr_multiple,
                'pullback_ema': cfg.labels.pullback_ema,
                'pullback_threshold': cfg.labels.pullback_threshold,
                'use_trend_gate': cfg.labels.use_trend_gate,
                'min_trend_prob': cfg.labels.min_trend_prob,
                'use_regime_gate': cfg.labels.use_regime_gate,
                'min_regime_prob': cfg.labels.min_regime_prob,
                'regime_align_direction': cfg.labels.regime_align_direction,
                'allow_regime_ranging': cfg.labels.allow_regime_ranging,
                'allow_regime_trend_up': cfg.labels.allow_regime_trend_up,
                'allow_regime_trend_down': cfg.labels.allow_regime_trend_down,
                'allow_regime_volatile': cfg.labels.allow_regime_volatile,
            })

        if self.tune_features:
            max_idx = max(0, len(cfg.features.timeframes) - 1)
            cfg.base_timeframe_idx = trial.suggest_int("base_timeframe_idx", 0, max_idx)
            cfg.features.rsi_period = trial.suggest_int("rsi_period", 7, 21)
            cfg.features.adx_period = trial.suggest_int("adx_period", 7, 28)
            cfg.features.atr_period = trial.suggest_int("atr_period", 7, 28)
            cfg.features.bb_period = trial.suggest_int("bb_period", 10, 40)
            cfg.features.bb_std = trial.suggest_float("bb_std", 1.5, 3.0)
            cfg.features.volume_ma_period = trial.suggest_int("volume_ma_period", 10, 50)
            cfg.features.swing_lookback = trial.suggest_int("swing_lookback", 5, 30)

            ema_fast = trial.suggest_int("ema_fast", 5, 12)
            ema_mid = trial.suggest_int("ema_mid", 13, 34)
            ema_slow = trial.suggest_int("ema_slow", 35, 89)
            ema_long = trial.suggest_int("ema_long", 90, 200)
            ema_periods = sorted({ema_fast, ema_mid, ema_slow, ema_long, cfg.labels.pullback_ema})
            cfg.features.ema_periods = ema_periods

            param_log.update({
                'base_timeframe_idx': cfg.base_timeframe_idx,
                'rsi_period': cfg.features.rsi_period,
                'adx_period': cfg.features.adx_period,
                'atr_period': cfg.features.atr_period,
                'bb_period': cfg.features.bb_period,
                'bb_std': cfg.features.bb_std,
                'volume_ma_period': cfg.features.volume_ma_period,
                'swing_lookback': cfg.features.swing_lookback,
                'ema_fast': ema_fast,
                'ema_mid': ema_mid,
                'ema_slow': ema_slow,
                'ema_long': ema_long,
                'ema_periods': ema_periods,
            })

        if self.tune_model:
            cfg.model.n_estimators = trial.suggest_int("n_estimators", 200, 2000)
            cfg.model.max_depth = trial.suggest_int("max_depth", 3, 12)
            cfg.model.learning_rate = trial.suggest_float("learning_rate", 0.005, 0.2, log=True)
            cfg.model.num_leaves = trial.suggest_int("num_leaves", 16, 256)
            max_leaves = 2 ** int(cfg.model.max_depth)
            if max_leaves > 0:
                cfg.model.num_leaves = int(min(cfg.model.num_leaves, max_leaves))
            cfg.model.feature_fraction = trial.suggest_float("feature_fraction", 0.4, 1.0)
            cfg.model.bagging_fraction = trial.suggest_float("bagging_fraction", 0.4, 1.0)
            cfg.model.bagging_freq = trial.suggest_int("bagging_freq", 0, 10)
            cfg.model.min_child_samples = trial.suggest_int("min_child_samples", 10, 200)
            cfg.model.lambdaa_ele1 = trial.suggest_float("lambdaa_ele1", 1e-8, 10.0, log=True)
            cfg.model.lambdaa_ele2 = trial.suggest_float("lambdaa_ele2", 1e-8, 10.0, log=True)
            cfg.model.min_gain_to_split = trial.suggest_float("min_gain_to_split", 0.0, 1.0)

            param_log.update({
                'n_estimators': cfg.model.n_estimators,
                'max_depth': cfg.model.max_depth,
                'learning_rate': cfg.model.learning_rate,
                'num_leaves': cfg.model.num_leaves,
                'feature_fraction': cfg.model.feature_fraction,
                'bagging_fraction': cfg.model.bagging_fraction,
                'bagging_freq': cfg.model.bagging_freq,
                'min_child_samples': cfg.model.min_child_samples,
                'lambdaa_ele1': cfg.model.lambdaa_ele1,
                'lambdaa_ele2': cfg.model.lambdaa_ele2,
                'min_gain_to_split': cfg.model.min_gain_to_split,
            })

        if cfg.labels.pullback_ema not in cfg.features.ema_periods:
            cfg.features.ema_periods = sorted(
                set(cfg.features.ema_periods + [cfg.labels.pullback_ema])
            )
            param_log['ema_periods'] = cfg.features.ema_periods

        return cfg, param_log

    def _evaluate_config(
        self,
        cfg: TrendFollowerConfig,
        trial: Optional["optuna.Trial"] = None,
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if self.lgbm_num_threads_override is not None:
            cfg.model.num_threads = int(self.lgbm_num_threads_override)
        self._wait_if_paused()

        base_tf = cfg.features.timeframe_names[cfg.base_timeframe_idx]
        tf_seconds_map = dict(zip(cfg.features.timeframe_names, cfg.features.timeframes))
        base_tf_seconds = tf_seconds_map.get(base_tf)
        if self._rust_pipeline_available():
            labeled_data, feature_cols, _labels_key = self._get_cached_labels_rust(cfg, base_tf)
        else:
            bars_dict, bars_key = self._get_cached_bars(cfg)
            featured_data, features_key = self._get_cached_features(bars_dict, bars_key, cfg, base_tf)
            labeled_data, feature_cols, _labels_key = self._get_cached_labels(
                featured_data,
                features_key,
                cfg,
                base_tf,
            )

        if not feature_cols:
            metrics['score'] = -1.0
            metrics['reason'] = 'no_features'
            self._print_trial_report(trial, metrics, [])
            return metrics

        total_samples = len(labeled_data)
        tuning_ratio = float(cfg.model.train_ratio) + float(cfg.model.val_ratio)
        tuning_end = int(total_samples * tuning_ratio)
        tuning_end = max(0, min(total_samples, tuning_end))
        tuning_samples = float(tuning_end)
        test_samples = float(total_samples - tuning_end)

        metrics['total_samples'] = float(total_samples)
        metrics['tuning_samples'] = tuning_samples
        metrics['test_samples'] = test_samples
        metrics['feature_count'] = float(len(feature_cols))

        tuning_slice = labeled_data.iloc[:tuning_end] if tuning_end > 0 else labeled_data
        median_atr_percent = self._median_atr_percent(tuning_slice, base_tf)
        stop_atr_multiple = float(cfg.labels.stop_atr_multiple)
        if stop_atr_multiple <= 0:
            stop_atr_multiple = 1.0
        fallback_fee_r = float(self.fee_percent) / (stop_atr_multiple * median_atr_percent)
        if self.fee_per_trade_r is None:
            fee_source = "dynamic"
            dynamic_fee = True
        else:
            fallback_fee_r = float(self.fee_per_trade_r)
            fee_source = "explicit"
            dynamic_fee = False

        folds = self._get_walk_forward_splits(labeled_data, cfg)
        metrics['folds'] = float(len(folds))
        purge_h = int(max(
            0,
            getattr(cfg.labels, "trend_forward_window", 0),
            getattr(cfg.labels, "entry_forward_window", 0),
        ))
        metrics['purge_h'] = float(purge_h)
        if not folds:
            metrics['score'] = -1e6
            metrics['reason'] = 'insufficient_data_for_folds'
            self._print_trial_report(trial, metrics, [])
            return metrics

        fold_aucs = []
        fold_train_sizes = []
        fold_val_sizes = []
        fold_pullback_train = []
        fold_pullback_val = []
        skipped_folds = 0
        no_opportunity_folds = 0
        if self.min_trades_per_fold is None:
            if self.min_trades > 0 and len(folds) > 0:
                min_trades_per_fold = max(1, int(self.min_trades / len(folds)))
            else:
                min_trades_per_fold = 0
        else:
            min_trades_per_fold = max(0, int(self.min_trades_per_fold))

        pullback_val_counts = []
        for _, val_df in folds:
            if 'pullback_success' in val_df.columns:
                pullback_val_counts.append(int((~val_df['pullback_success'].isna()).sum()))
            else:
                pullback_val_counts.append(0)
        pb_vals_nonzero = [v for v in pullback_val_counts if v > 0]
        pb_val_median = float(np.median(pb_vals_nonzero)) if pb_vals_nonzero else 0.0

        coverage_bounds_active = self.min_coverage > 0.0 or self.max_coverage < 1.0
        alpha = 0.05
        if pb_val_median > 0 and min_trades_per_fold > 0:
            p_min = float(min_trades_per_fold) / float(pb_val_median)
        else:
            p_min = 0.0
        p_min = max(float(self.min_coverage), p_min)
        p_min = min(max(p_min, 0.0), 0.99)
        opp_min = 0
        if p_min > 0.0:
            denom = np.log(1.0 - p_min)
            if np.isfinite(denom) and denom < 0:
                opp_min = int(np.ceil(np.log(alpha) / denom))
        opp_min = max(int(self.min_pullback_val_samples), int(opp_min))
        base_margin = float(self.ev_margin_r)
        if self.ev_margin_fixed:
            if not np.isfinite(base_margin) or base_margin < 0.0:
                base_margin = 0.0
            margin_grid = [float(round(base_margin, 4))]
        else:
            margin_grid = [0.0, 0.02, 0.05, 0.08, 0.12, 0.2, 0.4]
            if np.isfinite(base_margin) and base_margin >= 0.0:
                margin_grid.append(base_margin)
            margin_grid = sorted({
                float(round(m, 4))
                for m in margin_grid
                if np.isfinite(m) and m >= 0.0
            })
            if not margin_grid:
                margin_grid = [0.0]
        prune_margin = base_margin if base_margin in margin_grid else margin_grid[0]

        margin_fold_scores = {m: [] for m in margin_grid}
        margin_fold_trade_counts = {m: [] for m in margin_grid}
        margin_fold_coverages = {m: [] for m in margin_grid}
        margin_fold_breakeven = {m: [] for m in margin_grid}
        margin_fold_trade_rates = {m: [] for m in margin_grid}
        margin_fold_ops_costs = {m: [] for m in margin_grid}
        margin_fold_span_days = {m: [] for m in margin_grid}
        margin_fold_selected_precision = {m: [] for m in margin_grid}
        margin_fold_selected_ev_mean = {m: [] for m in margin_grid}
        margin_fold_selected_ev_p10 = {m: [] for m in margin_grid}
        margin_fold_implied_iqr = {m: [] for m in margin_grid}
        margin_fold_implied_std = {m: [] for m in margin_grid}
        margin_fold_gap_selected_mean = {m: [] for m in margin_grid}
        margin_fold_gap_selected_p10 = {m: [] for m in margin_grid}
        margin_fold_details = {m: [] for m in margin_grid}
        margin_refused_folds = {m: 0 for m in margin_grid}
        margin_hard_refusal_folds = {m: 0 for m in margin_grid}
        margin_trade_violations = {m: 0 for m in margin_grid}
        margin_low_trade_penalty = {m: 0.0 for m in margin_grid}
        margin_coverage_violations = {m: 0 for m in margin_grid}

        all_probs = []
        all_probs_raw = []
        all_probs_cal = []
        all_labels = []
        all_fee_r = []
        all_fee_fallback = []
        margin_all_returns = {m: [] for m in margin_grid}
        margin_all_selected_labels = {m: [] for m in margin_grid}
        margin_all_selected_evs = {m: [] for m in margin_grid}
        margin_all_selected_probs = {m: [] for m in margin_grid}
        margin_all_selected_rr_mean = {m: [] for m in margin_grid}
        margin_all_selected_win_r = {m: [] for m in margin_grid}

        last_fold_train = None
        last_fold_val = None

        fold_details: List[Dict[str, Any]] = []

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

        def _oof_context_preds(model_cls, X, y, n_folds: int = 3) -> Optional[Dict[str, np.ndarray]]:
            if X is None or len(X) == 0 or n_folds < 2:
                return None
            n = len(X)
            if n < max(30, n_folds * 10):
                return None
            y_series = pd.Series(y).reset_index(drop=True)
            if y_series.nunique() < 2:
                return None

            indices = np.arange(n)
            fold_indices = np.array_split(indices, n_folds)
            preds: Optional[Dict[str, np.ndarray]] = None

            for idx in fold_indices:
                if idx.size == 0:
                    continue
                train_end = int(idx[0])
                if train_end <= 1:
                    continue
                y_train = y_series.iloc[:train_end]
                if y_train.nunique() < 2:
                    continue
                model = model_cls(cfg.model)
                model.train(
                    X.iloc[:train_end],
                    y_train,
                    None,
                    None,
                    verbose=False,
                )
                fold_pred = model.predict(X.iloc[idx])
                if preds is None:
                    preds = {key: np.zeros(n, dtype=float) for key in fold_pred.keys()}
                for key, values in fold_pred.items():
                    arr = np.asarray(values)
                    if key in preds and arr.shape[0] == idx.size:
                        preds[key][idx] = arr

            return preds

        for fold_idx, (train_df, val_df) in enumerate(folds, start=1):
            self._wait_if_paused()
            fold_train_sizes.append(len(train_df))
            fold_val_sizes.append(len(val_df))

            X_train = train_df[feature_cols].fillna(0)
            X_val = val_df[feature_cols].fillna(0)

            pullback_mask_train = ~train_df['pullback_success'].isna()
            pullback_mask_val = ~val_df['pullback_success'].isna()
            pullback_train = int(pullback_mask_train.sum())
            pullback_val = int(pullback_mask_val.sum())

            fold_pullback_train.append(pullback_train)
            fold_pullback_val.append(pullback_val)

            if pullback_train < self.min_pullback_samples or pullback_val < self.min_pullback_val_samples:
                skipped_folds += 1
                if pullback_val == 0:
                    no_opportunity_folds += 1
                fold_details.append({
                    'fold': fold_idx,
                    'train_size': len(train_df),
                    'val_size': len(val_df),
                    'pullback_train': pullback_train,
                    'pullback_val': pullback_val,
                    'skipped': True,
                    'reason': 'insufficient_pullbacks',
                })
                continue

            trend_pred_train = None
            trend_pred_val = None
            if 'trend_label' in train_df.columns and 'trend_label' in val_df.columns:
                if train_df['trend_label'].nunique() >= 2:
                    trend_pred_train = _oof_context_preds(
                        TrendClassifier,
                        X_train,
                        train_df['trend_label'],
                        n_folds=3,
                    )
                    trend_model = TrendClassifier(cfg.model)
                    trend_model.train(
                        X_train,
                        train_df['trend_label'],
                        X_val,
                        val_df['trend_label'],
                        verbose=False,
                    )
                    trend_pred_val = trend_model.predict(X_val)

            regime_pred_train = None
            regime_pred_val = None
            if 'regime' in train_df.columns and 'regime' in val_df.columns and train_df['regime'].nunique() >= 2:
                regime_pred_train = _oof_context_preds(
                    RegimeClassifier,
                    X_train,
                    train_df['regime'],
                    n_folds=3,
                )
                regime_model = RegimeClassifier(cfg.model)
                regime_model.train(
                    X_train,
                    train_df['regime'],
                    X_val,
                    val_df['regime'],
                    verbose=False,
                )
                regime_pred_val = regime_model.predict(X_val)

            X_entry_train = append_context_features(
                X_train[pullback_mask_train],
                _slice_pred(trend_pred_train, pullback_mask_train),
                _slice_pred(regime_pred_train, pullback_mask_train),
            )
            y_success_train = train_df.loc[pullback_mask_train, 'pullback_success'].astype(int)
            rr_col = 'pullback_win_r' if 'pullback_win_r' in train_df.columns else 'pullback_rr'
            y_rr_train = train_df.loc[pullback_mask_train, rr_col]

            X_entry_val = append_context_features(
                X_val[pullback_mask_val],
                _slice_pred(trend_pred_val, pullback_mask_val),
                _slice_pred(regime_pred_val, pullback_mask_val),
            )
            y_success_val = val_df.loc[pullback_mask_val, 'pullback_success'].astype(int)
            rr_col_val = 'pullback_win_r' if 'pullback_win_r' in val_df.columns else 'pullback_rr'
            y_rr_val = val_df.loc[pullback_mask_val, rr_col_val]

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
                calibration_mode="oof",
                calibration_oof_folds=3,
                calibration_method=self.calibration_method,
                use_noise_filtering=self.use_noise_filtering,
                use_seed_ensemble=self.use_seed_ensemble,
                n_ensemble_seeds=self.n_ensemble_seeds,
            )

            trend_gate_enabled = bool(getattr(cfg.labels, "use_trend_gate", True))
            min_trend_prob = float(getattr(cfg.labels, "min_trend_prob", 0.0))
            regime_gate_enabled = bool(getattr(cfg.labels, "use_regime_gate", True))
            min_regime_prob = float(getattr(cfg.labels, "min_regime_prob", 0.0))
            regime_align_direction = bool(getattr(cfg.labels, "regime_align_direction", True))
            allow_regime_ranging = bool(getattr(cfg.labels, "allow_regime_ranging", True))
            allow_regime_trend_up = bool(getattr(cfg.labels, "allow_regime_trend_up", True))
            allow_regime_trend_down = bool(getattr(cfg.labels, "allow_regime_trend_down", True))
            allow_regime_volatile = bool(getattr(cfg.labels, "allow_regime_volatile", True))

            touch_dir = np.zeros(pullback_val, dtype=int)
            if 'ema_touch_direction' in val_df.columns:
                touch_dir = (
                    val_df.loc[pullback_mask_val, 'ema_touch_direction']
                    .fillna(0)
                    .astype(int)
                    .values
                )

            slope_col = f'{base_tf}_ema_{cfg.labels.pullback_ema}_slope_norm'
            if slope_col in val_df.columns:
                slope_vals = val_df.loc[pullback_mask_val, slope_col].fillna(0).values.astype(float)
                slope_dir = np.sign(slope_vals).astype(int)
            else:
                slope_dir = np.zeros(pullback_val, dtype=int)

            # Align direction with EMA touch when available (matches labeling)
            trend_dir = np.where(touch_dir != 0, touch_dir, slope_dir).astype(int)
            direction_mask = trend_dir != 0

            trend_probs = _slice_pred(trend_pred_val, pullback_mask_val)
            trend_prob_dir = np.zeros_like(trend_dir, dtype=float)
            if trend_gate_enabled and trend_probs is not None:
                prob_up = np.asarray(trend_probs.get('prob_up', np.zeros_like(trend_dir, dtype=float)))
                prob_down = np.asarray(trend_probs.get('prob_down', np.zeros_like(trend_dir, dtype=float)))
                trend_prob_dir = np.where(trend_dir == 1, prob_up, np.where(trend_dir == -1, prob_down, 0.0))
                trend_gate_mask = (trend_prob_dir >= min_trend_prob) & direction_mask
            else:
                trend_gate_mask = direction_mask

            regime_gate_mask = np.ones_like(direction_mask, dtype=bool)
            regime_probs = _slice_pred(regime_pred_val, pullback_mask_val)
            if regime_gate_enabled and regime_probs is not None:
                regimes = np.asarray(regime_probs.get('regime', np.zeros_like(trend_dir, dtype=int)))
                prob_ranging = np.asarray(regime_probs.get('prob_ranging', np.zeros_like(trend_dir, dtype=float)))
                prob_trend_up = np.asarray(regime_probs.get('prob_trend_up', np.zeros_like(trend_dir, dtype=float)))
                prob_trend_down = np.asarray(regime_probs.get('prob_trend_down', np.zeros_like(trend_dir, dtype=float)))
                prob_volatile = np.asarray(regime_probs.get('prob_volatile', np.zeros_like(trend_dir, dtype=float)))

                regime_prob_dir = np.where(
                    regimes == 0,
                    prob_ranging,
                    np.where(
                        regimes == 1,
                        prob_trend_up,
                        np.where(regimes == 2, prob_trend_down, prob_volatile),
                    ),
                )

                allowed_mask = np.zeros_like(direction_mask, dtype=bool)
                allowed_mask = np.where(regimes == 0, allow_regime_ranging, allowed_mask)
                allowed_mask = np.where(regimes == 1, allow_regime_trend_up, allowed_mask)
                allowed_mask = np.where(regimes == 2, allow_regime_trend_down, allowed_mask)
                allowed_mask = np.where(regimes == 3, allow_regime_volatile, allowed_mask)

                if regime_align_direction:
                    align_mask = np.ones_like(direction_mask, dtype=bool)
                    align_mask = np.where(regimes == 1, trend_dir == 1, align_mask)
                    align_mask = np.where(regimes == 2, trend_dir == -1, align_mask)
                    allowed_mask = allowed_mask & align_mask

                regime_gate_mask = (regime_prob_dir >= min_regime_prob) & allowed_mask & direction_mask

            gate_mask = trend_gate_mask & regime_gate_mask & direction_mask
            gate_rejected_trend = int(np.sum(direction_mask & ~trend_gate_mask))
            gate_rejected_regime = int(np.sum(direction_mask & ~regime_gate_mask))
            gate_rejected_direction = int(np.sum(~direction_mask))

            preds = entry_model.predict(X_entry_val, use_calibration=True)
            prob_key = 'bounce_prob_raw' if self.use_raw_probabilities else 'bounce_prob'
            probs = np.asarray(preds.get(prob_key, preds.get('bounce_prob', [])), dtype=float)
            raw_probs = np.asarray(preds.get('bounce_prob_raw', probs), dtype=float)
            cal_probs = np.asarray(preds.get('bounce_prob', raw_probs), dtype=float)
            labels = y_success_val.values.astype(int)

            if probs.size == 0:
                continue

            if SKLEARN_METRICS_AVAILABLE and len(np.unique(labels)) > 1:
                fold_auc = float(roc_auc_score(labels, probs))
            else:
                fold_auc = 0.5
            fold_aucs.append(fold_auc)

            target_rr = float(cfg.labels.target_rr)
            win_r_used = None
            if self.use_expected_rr and 'pullback_win_r' in val_df.columns:
                win_r = pd.to_numeric(
                    val_df.loc[pullback_mask_val, 'pullback_win_r'],
                    errors='coerce'
                ).to_numpy(dtype=float)
                if win_r.size == 0:
                    win_r = np.full_like(labels, target_rr, dtype=float)
                win_r = np.where(np.isfinite(win_r), win_r, target_rr)
                win_r_used = np.where(labels == 1, win_r, np.nan)
                outcomes_r = np.where(labels == 1, win_r, -1.0).astype(float)
            else:
                outcomes_r = np.where(labels == 1, target_rr, -1.0).astype(float)

            realized_r = None
            if 'pullback_realized_r' in val_df.columns:
                realized_r = pd.to_numeric(
                    val_df.loc[pullback_mask_val, 'pullback_realized_r'],
                    errors='coerce'
                ).to_numpy(dtype=float)

            rr_mean = np.full_like(probs, target_rr, dtype=float)
            rr_cons = rr_mean
            if self.use_expected_rr:
                rr_mean = np.asarray(preds.get('expected_rr_mean', rr_mean), dtype=float)
                rr_cons = np.asarray(preds.get('expected_rr', rr_mean), dtype=float)

            if dynamic_fee:
                fee_r_all, fee_fallback_all = self._compute_fee_r_series(
                    val_df,
                    base_tf,
                    self.fee_percent,
                    stop_atr_multiple,
                    fallback_fee_r,
                )
                fee_r_entry = fee_r_all[pullback_mask_val.to_numpy()]
                fallback_entry = fee_fallback_all[pullback_mask_val.to_numpy()]
            else:
                fee_r_entry = np.full_like(probs, fallback_fee_r, dtype=float)
                fallback_entry = np.zeros_like(fee_r_entry, dtype=bool)

            all_probs.append(probs)
            all_probs_raw.append(raw_probs)
            all_probs_cal.append(cal_probs)
            all_labels.append(labels)
            if fee_r_entry.size > 0:
                all_fee_r.append(fee_r_entry)
                all_fee_fallback.append(fallback_entry)

            ev_components_base = entry_model.compute_expected_rr_components(
                probs,
                rr_mean,
                rr_conservative=rr_cons,
                cost_r=fee_r_entry,
            )
            ev_base = ev_components_base['ev_conservative_r']
            implied_base = np.clip(ev_components_base['implied_threshold'], 0.0, 1.0)
            threshold_gap_base = probs - implied_base
            span_days = self._estimate_span_days(val_df, base_tf_seconds)
            if span_days <= 0:
                span_days = 0.0

            prune_fold_score = 0.0
            prune_trade_count = 0
            for margin in margin_grid:
                ev = ev_base
                implied = implied_base
                threshold_gap = threshold_gap_base
                trade_mask = gate_mask & (ev > margin)
                trade_count = int(trade_mask.sum())
                coverage = float(trade_count / probs.size) if probs.size > 0 else 0.0
                trade_rate_day = float(trade_count / span_days) if span_days > 0 else 0.0
                ops_cost_r = 0.0
                if (
                    self.ops_cost_enabled
                    and trade_rate_day > self.ops_cost_target_trades_per_day
                    and self.ops_cost_target_trades_per_day > 0
                ):
                    excess = trade_rate_day - self.ops_cost_target_trades_per_day
                    ops_cost_r = self.ops_cost_c1 * (
                        (excess / self.ops_cost_target_trades_per_day) ** self.ops_cost_alpha
                    )

                if ops_cost_r > 0.0:
                    ev_components_pass2 = entry_model.compute_expected_rr_components(
                        probs,
                        rr_mean,
                        rr_conservative=rr_cons,
                        cost_r=fee_r_entry + ops_cost_r,
                    )
                    ev = ev_components_pass2['ev_conservative_r']
                    implied = np.clip(ev_components_pass2['implied_threshold'], 0.0, 1.0)
                    threshold_gap = probs - implied
                    trade_mask = gate_mask & (ev > margin)
                    trade_count = int(trade_mask.sum())
                    coverage = float(trade_count / probs.size) if probs.size > 0 else 0.0
                    trade_rate_day = float(trade_count / span_days) if span_days > 0 else 0.0

                trade_count_base = int(trade_mask.sum())
                coverage_base = float(trade_count_base / probs.size) if probs.size > 0 else 0.0
                trade_rate_day_base = float(trade_count_base / span_days) if span_days > 0 else 0.0

                implied_mean = float(np.mean(implied)) if implied.size > 0 else 0.0
                implied_std = float(np.std(implied)) if implied.size > 1 else 0.0
                implied_iqr = float(
                    np.percentile(implied, 75) - np.percentile(implied, 25)
                ) if implied.size > 1 else 0.0

                trade_indices = np.where(trade_mask)[0]
                if self.single_position:
                    returns, trade_indices = self._simulate_single_position_returns(
                        val_df,
                        pullback_mask_val,
                        trade_mask,
                        trend_dir,
                        base_tf,
                        stop_atr_multiple,
                        int(cfg.labels.entry_forward_window),
                        realized_r,
                        outcomes_r,
                        self.opposite_signal_policy,
                    )
                else:
                    if realized_r is not None and realized_r.size == probs.size:
                        returns = realized_r[trade_mask]
                    else:
                        returns = outcomes_r[trade_mask]
                trade_count = int(trade_indices.size)
                coverage = float(trade_count / probs.size) if probs.size > 0 else 0.0
                trade_rate_day = float(trade_count / span_days) if span_days > 0 else 0.0

                if returns.size > 0:
                    returns = returns - fee_r_entry[trade_indices]
                    if ops_cost_r != 0.0:
                        returns = returns - float(ops_cost_r)

                if trade_count > 0:
                    selected_labels = labels[trade_indices]
                    selected_probs = probs[trade_indices]
                    selected_precision = float(np.mean(selected_labels)) if selected_labels.size > 0 else 0.0
                    selected_ev = ev[trade_indices]
                    selected_ev_mean = float(np.mean(selected_ev)) if selected_ev.size > 0 else 0.0
                    selected_ev_p10 = float(np.percentile(selected_ev, 10)) if selected_ev.size > 0 else 0.0
                    gap_selected = threshold_gap[trade_indices]
                    gap_selected_mean = float(np.mean(gap_selected)) if gap_selected.size > 0 else 0.0
                    gap_selected_p10 = float(np.percentile(gap_selected, 10)) if gap_selected.size > 0 else 0.0
                    margin_all_selected_labels[margin].append(selected_labels)
                    margin_all_selected_evs[margin].append(selected_ev)
                    margin_all_selected_probs[margin].append(selected_probs)
                    if win_r_used is not None:
                        margin_all_selected_win_r[margin].append(win_r_used[trade_indices])
                    if rr_mean is not None:
                        margin_all_selected_rr_mean[margin].append(rr_mean[trade_indices])
                else:
                    selected_precision = 0.0
                    selected_ev_mean = 0.0
                    selected_ev_p10 = 0.0
                    gap_selected_mean = 0.0
                    gap_selected_p10 = 0.0

                if trade_count == 0:
                    margin_refused_folds[margin] += 1
                    if pullback_val >= opp_min:
                        margin_hard_refusal_folds[margin] += 1
                else:
                    if min_trades_per_fold > 0 and trade_count < min_trades_per_fold:
                        margin_trade_violations[margin] += 1
                        margin_low_trade_penalty[margin] += self.refusal_penalty * (
                            1.0 - (float(trade_count) / float(min_trades_per_fold))
                        ) ** 1.5
                    if coverage_bounds_active and (coverage < self.min_coverage or coverage > self.max_coverage):
                        margin_coverage_violations[margin] += 1

                fold_metrics = self._compute_trade_metrics(returns)
                lcb_z_used = self._adaptive_lcb_z(trade_count)
                fold_score = self._score_trade_metrics(fold_metrics, lcb_z=lcb_z_used)
                if trade_count == 0:
                    fold_score = 0.0

                if margin == prune_margin:
                    prune_fold_score = float(fold_score)
                    prune_trade_count = trade_count

                margin_fold_scores[margin].append(float(fold_score))
                margin_fold_trade_counts[margin].append(trade_count)
                margin_fold_coverages[margin].append(coverage)
                margin_fold_breakeven[margin].append(float(np.median(implied)))
                margin_fold_trade_rates[margin].append(trade_rate_day)
                margin_fold_ops_costs[margin].append(ops_cost_r)
                margin_fold_span_days[margin].append(span_days)
                margin_fold_selected_precision[margin].append(selected_precision)
                margin_fold_selected_ev_mean[margin].append(selected_ev_mean)
                margin_fold_selected_ev_p10[margin].append(selected_ev_p10)
                margin_fold_implied_iqr[margin].append(implied_iqr)
                margin_fold_implied_std[margin].append(implied_std)
                margin_fold_gap_selected_mean[margin].append(gap_selected_mean)
                margin_fold_gap_selected_p10[margin].append(gap_selected_p10)

                if returns.size > 0:
                    margin_all_returns[margin].append(returns)

                margin_fold_details[margin].append({
                    'fold': fold_idx,
                    'train_size': len(train_df),
                    'val_size': len(val_df),
                    'purge_h': purge_h,
                    'pullback_train': pullback_train,
                    'pullback_val': pullback_val,
                    'trend_gate_rejected': gate_rejected_trend,
                    'regime_gate_rejected': gate_rejected_regime,
                    'direction_neutral': gate_rejected_direction,
                    'breakeven_threshold': float(np.median(implied)),
                    'fold_score': float(fold_score),
                    'auc': float(fold_auc),
                    'trades': fold_metrics.get('n_trades', 0.0),
                    'pnl_r': fold_metrics.get('total_pnl_r', 0.0),
                    'win_rate': fold_metrics.get('win_rate', 0.0),
                    'profit_factor': fold_metrics.get('profit_factor', 0.0),
                    'max_drawdown_r': fold_metrics.get('max_drawdown_r', 0.0),
                    'coverage': coverage,
                    'selected_precision': selected_precision,
                    'selected_ev_mean': selected_ev_mean,
                    'selected_ev_p10': selected_ev_p10,
                    'implied_threshold_mean': implied_mean,
                    'implied_threshold_iqr': implied_iqr,
                    'implied_threshold_std': implied_std,
                    'prob_gap_mean_selected': gap_selected_mean,
                    'prob_gap_p10_selected': gap_selected_p10,
                    'trade_rate_day': trade_rate_day,
                    'ops_cost_r': ops_cost_r,
                    'span_days': span_days,
                    'lcb_z': float(self._adaptive_lcb_z(trade_count)),
                    'ev_margin_r': float(margin),
                })

            if trial is not None:
                try:
                    trial.report(prune_fold_score, step=fold_idx)
                except Exception:
                    pass

            if trial is not None and fold_idx == 1 and len(folds) > 1 and self.enable_pruning:
                prune = False
                if prune_trade_count == 0 and pullback_val >= opp_min:
                    prune = True
                elif min_trades_per_fold > 0 and prune_trade_count < min_trades_per_fold and prune_fold_score < -0.01:
                    prune = True
                elif prune_fold_score < -0.05:
                    prune = True
                if prune and OPTUNA_AVAILABLE:
                    raise optuna.TrialPruned()

            last_fold_train = train_df
            last_fold_val = val_df

        expected_folds = len(folds)
        missing_fold_types = []
        for detail in fold_details:
            if detail.get('skipped'):
                if int(detail.get('pullback_val', 0)) == 0:
                    missing_fold_types.append('no_opportunity')
                else:
                    missing_fold_types.append('insufficient')
        missing_no_opp = sum(1 for t in missing_fold_types if t == 'no_opportunity')
        missing_with_opps = sum(1 for t in missing_fold_types if t != 'no_opportunity')

        def _impute_missing_fold_scores(scores: List[float]) -> Tuple[List[float], int, int, float, float]:
            missing_total = max(0, expected_folds - len(scores))
            if missing_total <= 0:
                return list(scores), 0, 0, -self.no_opportunity_penalty, -self.refusal_penalty
            base_p25 = float(np.percentile(scores, 25)) if scores else 0.0
            no_opp_score = -self.no_opportunity_penalty
            opp_score = min(0.0, base_p25) - max(self.refusal_penalty, 0.5 * abs(base_p25))
            missing_no_opp_m = min(missing_total, missing_no_opp)
            missing_with_opps_m = max(0, missing_total - missing_no_opp_m)
            augmented = list(scores)
            if missing_no_opp_m > 0:
                augmented.extend([no_opp_score] * missing_no_opp_m)
            if missing_with_opps_m > 0:
                augmented.extend([opp_score] * missing_with_opps_m)
            return augmented, missing_no_opp_m, missing_with_opps_m, no_opp_score, opp_score

        if not any(margin_fold_scores[m] for m in margin_grid) or not all_probs:
            metrics['score'] = -1e6
            metrics['reason'] = 'insufficient_pullbacks'
            self._print_trial_report(trial, metrics, fold_details)
            return metrics

        margin_results: Dict[float, Dict[str, float]] = {}
        best_margin = None
        best_margin_score = None

        for margin in margin_grid:
            fold_scores_m = margin_fold_scores[margin]
            if not fold_scores_m:
                continue
            fold_trades_m = margin_fold_trade_counts[margin]
            fold_cov_m = margin_fold_coverages[margin]

            fold_scores_aug, missing_no_opp_m, missing_with_opps_m, no_opp_score, opp_score = _impute_missing_fold_scores(fold_scores_m)
            fold_score_iqr_m = float(np.percentile(fold_scores_aug, 75) - np.percentile(fold_scores_aug, 25)) if len(fold_scores_aug) > 1 else 0.0
            fold_trades_mean_m = float(np.mean(fold_trades_m)) if fold_trades_m else 0.0
            fold_trades_iqr_m = float(np.percentile(fold_trades_m, 75) - np.percentile(fold_trades_m, 25)) if len(fold_trades_m) > 1 else 0.0
            fold_cov_iqr_m = float(np.percentile(fold_cov_m, 75) - np.percentile(fold_cov_m, 25)) if len(fold_cov_m) > 1 else 0.0
            trade_iqr_norm_m = fold_trades_iqr_m / max(1.0, fold_trades_mean_m)
            coverage_distance_m = 0.0
            if coverage_bounds_active and fold_cov_m:
                for cov in fold_cov_m:
                    if cov < self.min_coverage:
                        coverage_distance_m += (self.min_coverage - cov)
                    elif cov > self.max_coverage:
                        coverage_distance_m += (cov - self.max_coverage)
                coverage_distance_m /= max(1, len(fold_cov_m))

            if fold_scores_aug:
                base_profit_score_m = float(np.percentile(fold_scores_aug, 25))
                fold_score_median_m = float(np.median(fold_scores_aug))
            else:
                base_profit_score_m = 0.0
                fold_score_median_m = 0.0
            fold_score_iqr_ratio_m = fold_score_iqr_m / (abs(fold_score_median_m) + self.stability_eps)
            stability_mult_m = 1.0 / (1.0 + (self.stability_k * fold_score_iqr_ratio_m))
            coverage_mult_m = 1.0 / (1.0 + (self.coverage_k * coverage_distance_m))
            if min_trades_per_fold > 0:
                trade_conf_m = float(np.sqrt(np.tanh(max(0.0, fold_trades_mean_m / min_trades_per_fold))))
            else:
                trade_conf_m = 1.0
            fold_count_m = len(fold_scores_aug)
            if fold_count_m > 0:
                fold_factor_m = min(1.0, (fold_count_m / 3.0) ** 2)
            else:
                fold_factor_m = 0.0
            profit_score_m = (
                base_profit_score_m
                * stability_mult_m
                * coverage_mult_m
                * trade_conf_m
                * fold_factor_m
            )

            margin_results[margin] = {
                'profit_score_base': base_profit_score_m,
                'profit_score': profit_score_m,
                'fold_factor': float(fold_factor_m),
                'folds_used': float(fold_count_m),
                'stability_multiplier': float(stability_mult_m),
                'fold_score_iqr_ratio': float(fold_score_iqr_ratio_m),
                'trade_iqr_norm': float(trade_iqr_norm_m),
                'coverage_multiplier': float(coverage_mult_m),
                'coverage_distance': float(coverage_distance_m),
                'trade_confidence': float(trade_conf_m),
                'fold_score_iqr': float(fold_score_iqr_m),
                'fold_score_median': float(fold_score_median_m),
                'fold_score_p25': float(base_profit_score_m),
                'fold_trades_mean': float(fold_trades_mean_m),
                'fold_trades_iqr': float(fold_trades_iqr_m),
                'fold_coverage_mean': float(np.mean(fold_cov_m)) if fold_cov_m else 0.0,
                'fold_coverage_iqr': float(fold_cov_iqr_m),
                'refused_folds': float(margin_refused_folds[margin]),
                'trade_violations': float(margin_trade_violations[margin]),
                'coverage_violations': float(margin_coverage_violations[margin]),
                'hard_refusal_folds': float(margin_hard_refusal_folds[margin]),
                'missing_folds': float(len(fold_scores_aug) - len(fold_scores_m)),
                'missing_no_opportunity': float(missing_no_opp_m),
                'missing_with_opportunity': float(missing_with_opps_m),
                'imputed_no_opp_score': float(no_opp_score),
                'imputed_missing_score': float(opp_score),
            }

            constraints_ok = margin_hard_refusal_folds[margin] == 0

            if constraints_ok:
                if best_margin_score is None or profit_score_m > best_margin_score:
                    best_margin_score = profit_score_m
                    best_margin = margin

        if best_margin is None:
            for margin, res in margin_results.items():
                if best_margin_score is None or res['profit_score'] > best_margin_score:
                    best_margin_score = res['profit_score']
                    best_margin = margin

        if best_margin is None:
            metrics['score'] = -1e6
            metrics['reason'] = 'no_margin_candidates'
            self._print_trial_report(trial, metrics, fold_details)
            return metrics

        fold_scores = margin_fold_scores[best_margin]
        fold_scores_aug, missing_no_opp_best, missing_with_opps_best, no_opp_score, opp_score = _impute_missing_fold_scores(fold_scores)
        fold_trade_counts = margin_fold_trade_counts[best_margin]
        fold_coverages = margin_fold_coverages[best_margin]
        fold_breakeven = margin_fold_breakeven[best_margin]
        fold_trade_rates = margin_fold_trade_rates[best_margin]
        fold_ops_costs = margin_fold_ops_costs[best_margin]
        fold_span_days = margin_fold_span_days[best_margin]
        fold_selected_precision = margin_fold_selected_precision[best_margin]
        fold_selected_ev_mean = margin_fold_selected_ev_mean[best_margin]
        fold_selected_ev_p10 = margin_fold_selected_ev_p10[best_margin]
        fold_implied_iqr = margin_fold_implied_iqr[best_margin]
        fold_implied_std = margin_fold_implied_std[best_margin]
        fold_gap_selected_mean = margin_fold_gap_selected_mean[best_margin]
        fold_gap_selected_p10 = margin_fold_gap_selected_p10[best_margin]
        fold_details = margin_fold_details[best_margin]

        refused_folds = margin_refused_folds[best_margin]
        trade_violations = margin_trade_violations[best_margin]
        low_trade_penalty = margin_low_trade_penalty[best_margin]
        low_trade_folds = margin_trade_violations[best_margin]
        coverage_violations = margin_coverage_violations[best_margin]
        hard_refusal_folds = margin_hard_refusal_folds[best_margin]
        coverage_distance = 0.0
        if coverage_bounds_active and fold_coverages:
            for cov in fold_coverages:
                if cov < self.min_coverage:
                    coverage_distance += (self.min_coverage - cov)
                elif cov > self.max_coverage:
                    coverage_distance += (cov - self.max_coverage)
            coverage_distance /= max(1, len(fold_coverages))

        metrics['trend_gate_rejected'] = float(sum(d.get('trend_gate_rejected', 0) for d in fold_details))
        metrics['regime_gate_rejected'] = float(sum(d.get('regime_gate_rejected', 0) for d in fold_details))
        metrics['direction_neutral'] = float(sum(d.get('direction_neutral', 0) for d in fold_details))

        all_probs_arr = np.concatenate(all_probs)
        all_probs_raw_arr = np.concatenate(all_probs_raw)
        all_probs_cal_arr = np.concatenate(all_probs_cal)
        all_labels_arr = np.concatenate(all_labels)
        all_returns_arr = (
            np.concatenate(margin_all_returns[best_margin])
            if margin_all_returns.get(best_margin)
            else np.array([], dtype=float)
        )
        all_fee_r_arr = np.concatenate(all_fee_r) if all_fee_r else np.array([fallback_fee_r], dtype=float)
        all_fee_fallback_arr = np.concatenate(all_fee_fallback) if all_fee_fallback else np.array([], dtype=bool)

        if SKLEARN_METRICS_AVAILABLE and len(np.unique(all_labels_arr)) > 1:
            val_auc = float(roc_auc_score(all_labels_arr, all_probs_arr))
        else:
            val_auc = 0.5

        agg_metrics = self._compute_trade_metrics(all_returns_arr)

        binary_05 = self._binary_metrics(all_probs_arr, all_labels_arr, threshold=0.5)
        entry_val_accuracy = float(binary_05['accuracy'])
        entry_val_precision = float(binary_05['precision'])
        entry_val_recall = float(binary_05['recall'])

        raw_binary = self._binary_metrics(all_probs_raw_arr, all_labels_arr, threshold=0.5)
        cal_binary = self._binary_metrics(all_probs_cal_arr, all_labels_arr, threshold=0.5)
        metrics['entry_val_base_rate'] = float(all_labels_arr.mean()) if all_labels_arr.size > 0 else 0.0
        metrics['entry_val_accuracy_raw'] = float(raw_binary['accuracy'])
        metrics['entry_val_precision_raw'] = float(raw_binary['precision'])
        metrics['entry_val_recall_raw'] = float(raw_binary['recall'])
        metrics['entry_val_accuracy_cal'] = float(cal_binary['accuracy'])
        metrics['entry_val_precision_cal'] = float(cal_binary['precision'])
        metrics['entry_val_recall_cal'] = float(cal_binary['recall'])

        selected_labels_arr = (
            np.concatenate(margin_all_selected_labels[best_margin])
            if margin_all_selected_labels.get(best_margin)
            else np.array([], dtype=float)
        )
        selected_probs_arr = (
            np.concatenate(margin_all_selected_probs[best_margin])
            if margin_all_selected_probs.get(best_margin)
            else np.array([], dtype=float)
        )
        selected_ev_arr = (
            np.concatenate(margin_all_selected_evs[best_margin])
            if margin_all_selected_evs.get(best_margin)
            else np.array([], dtype=float)
        )
        selected_rr_mean_arr = (
            np.concatenate(margin_all_selected_rr_mean[best_margin])
            if margin_all_selected_rr_mean.get(best_margin)
            else np.array([], dtype=float)
        )
        selected_win_r_arr = (
            np.concatenate(margin_all_selected_win_r[best_margin])
            if margin_all_selected_win_r.get(best_margin)
            else np.array([], dtype=float)
        )
        metrics['entry_val_precision_selected'] = float(selected_labels_arr.mean()) if selected_labels_arr.size > 0 else 0.0
        metrics['entry_val_ev_mean_selected'] = float(selected_ev_arr.mean()) if selected_ev_arr.size > 0 else 0.0
        metrics['entry_val_ev_p10_selected'] = float(np.percentile(selected_ev_arr, 10)) if selected_ev_arr.size > 0 else 0.0
        metrics['entry_val_coverage_selected'] = float(np.mean(fold_coverages)) if fold_coverages else 0.0

        selected_brier = 0.0
        selected_logloss = 0.0
        selected_ece = 0.0
        if selected_probs_arr.size > 0 and selected_labels_arr.size == selected_probs_arr.size:
            probs_clip = np.clip(selected_probs_arr.astype(float), 1e-6, 1 - 1e-6)
            labels_float = selected_labels_arr.astype(float)
            selected_brier = float(np.mean((probs_clip - labels_float) ** 2))
            selected_logloss = float(-np.mean(
                labels_float * np.log(probs_clip) + (1.0 - labels_float) * np.log(1.0 - probs_clip)
            ))
            if len(np.unique(labels_float)) > 1:
                try:
                    selected_ece = float(
                        compute_expected_calibration_error(labels_float, probs_clip).get('ece', 0.0)
                    )
                except Exception:
                    selected_ece = 0.0

        rr_ratio = 0.0
        rr_mae = 0.0
        if (
            selected_rr_mean_arr.size > 0
            and selected_win_r_arr.size == selected_rr_mean_arr.size
            and selected_labels_arr.size == selected_rr_mean_arr.size
        ):
            win_mask = (selected_labels_arr == 1) & np.isfinite(selected_win_r_arr)
            if win_mask.any():
                realized_win = selected_win_r_arr[win_mask]
                pred_win = selected_rr_mean_arr[win_mask]
                if np.mean(realized_win) > 0:
                    rr_ratio = float(np.mean(pred_win) / np.mean(realized_win))
                rr_mae = float(np.mean(np.abs(pred_win - realized_win)))

        ev_bin_summary = ""
        if selected_ev_arr.size > 0 and all_returns_arr.size > 0:
            n = min(selected_ev_arr.size, all_returns_arr.size)
            ev_vals = selected_ev_arr[:n]
            ret_vals = all_returns_arr[:n]
            if n > 0:
                n_bins = int(min(10, max(1, n)))
                if n_bins == 1:
                    ev_bin_summary = f"0:{ret_vals.mean():.4f}@{n}"
                else:
                    edges = np.quantile(ev_vals, np.linspace(0.0, 1.0, n_bins + 1))
                    if np.unique(edges).size == 1:
                        ev_bin_summary = f"0:{ret_vals.mean():.4f}@{n}"
                    else:
                        bins = np.digitize(ev_vals, edges[1:-1], right=True)
                        parts = []
                        for b in range(n_bins):
                            mask = bins == b
                            if mask.any():
                                parts.append(f"{b}:{ret_vals[mask].mean():.4f}@{int(mask.sum())}")
                        ev_bin_summary = "|".join(parts)

        metrics['entry_selected_brier'] = float(selected_brier)
        metrics['entry_selected_logloss'] = float(selected_logloss)
        metrics['entry_selected_ece'] = float(selected_ece)
        metrics['expected_rr_bias_ratio'] = float(rr_ratio)
        metrics['expected_rr_mae'] = float(rr_mae)
        metrics['ev_bin_summary'] = ev_bin_summary

        metrics['train_size_mean'] = float(np.mean(fold_train_sizes)) if fold_train_sizes else 0.0
        metrics['val_size_mean'] = float(np.mean(fold_val_sizes)) if fold_val_sizes else 0.0
        metrics['pullback_train_mean'] = float(np.mean(fold_pullback_train)) if fold_pullback_train else 0.0
        metrics['pullback_val_mean'] = float(np.mean(fold_pullback_val)) if fold_pullback_val else 0.0
        metrics['fold_score_median'] = float(np.median(fold_scores_aug)) if fold_scores_aug else 0.0
        metrics['fold_score_min'] = float(np.min(fold_scores_aug)) if fold_scores_aug else 0.0
        metrics['fold_score_iqr'] = float(np.percentile(fold_scores_aug, 75) - np.percentile(fold_scores_aug, 25)) if len(fold_scores_aug) > 1 else 0.0
        metrics['fold_score_p25'] = float(np.percentile(fold_scores_aug, 25)) if fold_scores_aug else 0.0
        metrics['fold_auc_mean'] = float(np.mean(fold_aucs)) if fold_aucs else 0.5
        metrics['fold_auc_std'] = float(np.std(fold_aucs)) if fold_aucs else 0.0
        metrics['fold_trades_mean'] = float(np.mean(fold_trade_counts)) if fold_trade_counts else 0.0
        metrics['fold_trades_iqr'] = float(np.percentile(fold_trade_counts, 75) - np.percentile(fold_trade_counts, 25)) if len(fold_trade_counts) > 1 else 0.0
        metrics['fold_coverage_mean'] = float(np.mean(fold_coverages)) if fold_coverages else 0.0
        metrics['fold_coverage_iqr'] = float(np.percentile(fold_coverages, 75) - np.percentile(fold_coverages, 25)) if len(fold_coverages) > 1 else 0.0
        metrics['fold_breakeven_median'] = float(np.median(fold_breakeven)) if fold_breakeven else 0.0
        metrics['fold_selected_precision_mean'] = float(np.mean(fold_selected_precision)) if fold_selected_precision else 0.0
        metrics['fold_selected_ev_mean'] = float(np.mean(fold_selected_ev_mean)) if fold_selected_ev_mean else 0.0
        metrics['fold_selected_ev_p10_mean'] = float(np.mean(fold_selected_ev_p10)) if fold_selected_ev_p10 else 0.0
        metrics['fold_implied_iqr_mean'] = float(np.mean(fold_implied_iqr)) if fold_implied_iqr else 0.0
        metrics['fold_implied_std_mean'] = float(np.mean(fold_implied_std)) if fold_implied_std else 0.0
        metrics['fold_gap_selected_mean'] = float(np.mean(fold_gap_selected_mean)) if fold_gap_selected_mean else 0.0
        metrics['fold_gap_selected_p10_mean'] = float(np.mean(fold_gap_selected_p10)) if fold_gap_selected_p10 else 0.0
        metrics['trade_rate_day_mean'] = float(np.mean(fold_trade_rates)) if fold_trade_rates else 0.0
        metrics['trade_rate_day_max'] = float(np.max(fold_trade_rates)) if fold_trade_rates else 0.0
        metrics['ops_cost_r_mean'] = float(np.mean(fold_ops_costs)) if fold_ops_costs else 0.0
        metrics['ops_cost_r_max'] = float(np.max(fold_ops_costs)) if fold_ops_costs else 0.0
        fold_lcb_z = [d.get('lcb_z', float(self.lcb_z)) for d in fold_details]
        metrics['fold_lcb_z_mean'] = float(np.mean(fold_lcb_z)) if fold_lcb_z else float(self.lcb_z)
        metrics['fold_lcb_z_min'] = float(np.min(fold_lcb_z)) if fold_lcb_z else float(self.lcb_z)
        metrics['fold_lcb_z_max'] = float(np.max(fold_lcb_z)) if fold_lcb_z else float(self.lcb_z)
        metrics['skipped_folds'] = float(skipped_folds)
        metrics['no_opportunity_folds'] = float(no_opportunity_folds)
        metrics['missing_folds'] = float(len(fold_scores_aug) - len(fold_scores))
        metrics['missing_no_opportunity_folds'] = float(missing_no_opp_best)
        metrics['missing_with_opportunity_folds'] = float(missing_with_opps_best)
        metrics['imputed_no_opp_score'] = float(no_opp_score)
        metrics['imputed_missing_score'] = float(opp_score)
        metrics['refused_trade_folds'] = float(refused_folds)
        metrics['low_trade_folds'] = float(low_trade_folds)
        metrics['min_trades_per_fold'] = float(min_trades_per_fold)
        metrics['coverage_min'] = float(self.min_coverage)
        metrics['coverage_max'] = float(self.max_coverage)

        metrics['entry_val_accuracy'] = entry_val_accuracy
        metrics['entry_val_precision'] = entry_val_precision
        metrics['entry_val_recall'] = entry_val_recall
        metrics['val_auc'] = val_auc

        fee_r_median = float(np.median(all_fee_r_arr)) if all_fee_r_arr.size else float(fallback_fee_r)
        breakeven_threshold = (1.0 + fee_r_median) / (target_rr + 1.0) if target_rr > 0 else 1.0
        metrics['best_threshold'] = float(breakeven_threshold)
        metrics['breakeven_threshold'] = float(breakeven_threshold)
        metrics['val_total_pnl_r'] = float(agg_metrics['total_pnl_r'])
        metrics['val_pnl_per_trade_r'] = float(agg_metrics['avg_pnl_r'])
        metrics['val_win_rate'] = float(agg_metrics['win_rate'])
        metrics['val_profit_factor'] = float(agg_metrics['profit_factor'])
        metrics['val_max_drawdown_r'] = float(agg_metrics['max_drawdown_r'])
        metrics['val_sharpe'] = float(agg_metrics['sharpe'])
        metrics['val_sortino'] = float(agg_metrics['sortino'])
        metrics['val_trades'] = float(agg_metrics['n_trades'])
        total_eval = float(all_probs_arr.size) if all_probs_arr.size > 0 else 0.0
        metrics['val_trade_coverage'] = float(agg_metrics['n_trades'] / total_eval) if total_eval > 0 else 0.0

        prob_spread = float(np.max(all_probs_arr) - np.min(all_probs_arr)) if all_probs_arr.size else 0.0
        metrics['prob_spread'] = prob_spread

        fold_score_iqr = float(metrics.get('fold_score_iqr', 0.0))
        fold_score_median = float(metrics.get('fold_score_median', 0.0))
        fold_score_p25 = float(metrics.get('fold_score_p25', 0.0))
        fold_trades_mean = float(metrics.get('fold_trades_mean', 0.0))
        fold_trades_iqr = float(metrics.get('fold_trades_iqr', 0.0))
        fold_coverage_iqr = float(metrics.get('fold_coverage_iqr', 0.0))
        trade_iqr_norm = fold_trades_iqr / max(1.0, fold_trades_mean)

        fold_score_iqr_ratio = fold_score_iqr / (abs(fold_score_median) + self.stability_eps)
        stability_multiplier = 1.0 / (1.0 + (self.stability_k * fold_score_iqr_ratio))
        coverage_multiplier = 1.0 / (1.0 + (self.coverage_k * coverage_distance))
        if min_trades_per_fold > 0:
            trade_confidence = float(np.sqrt(np.tanh(max(0.0, fold_trades_mean / min_trades_per_fold))))
        else:
            trade_confidence = 1.0

        metrics['fold_trades_iqr_norm'] = float(trade_iqr_norm)
        metrics['fold_score_iqr_ratio'] = float(fold_score_iqr_ratio)
        metrics['stability_multiplier'] = float(stability_multiplier)
        metrics['coverage_multiplier'] = float(coverage_multiplier)
        metrics['trade_confidence'] = float(trade_confidence)
        metrics['coverage_distance'] = float(coverage_distance)
        metrics['stability_k'] = float(self.stability_k)
        metrics['coverage_k'] = float(self.coverage_k)
        metrics['stability_eps'] = float(self.stability_eps)
        metrics['stability_penalty'] = 0.0
        metrics['stability_penalty_multiplier'] = float(self.stability_penalty_multiplier)
        metrics['no_opportunity_penalty'] = 0.0
        metrics['refusal_penalty'] = 0.0
        metrics['low_trade_penalty'] = float(low_trade_penalty)
        metrics['trade_violations'] = float(trade_violations)
        metrics['coverage_violations'] = float(coverage_violations)
        metrics['coverage_penalty'] = 0.0
        metrics['hard_refusal_folds'] = float(hard_refusal_folds)
        metrics['opportunity_min'] = float(opp_min)
        metrics['opportunity_alpha'] = float(alpha)
        metrics['opportunity_p_min'] = float(p_min)
        metrics['opportunity_pb_val_median'] = float(pb_val_median)

        fold_count = len(fold_scores_aug)
        if fold_count > 0:
            fold_factor = min(1.0, (fold_count / 3.0) ** 2)
        else:
            fold_factor = 0.0
        profit_score = fold_score_p25 * stability_multiplier * coverage_multiplier * trade_confidence * fold_factor

        if hard_refusal_folds > 0:
            metrics['score'] = -1e6
            metrics['reason'] = 'refused_with_opportunities'
            self._print_trial_report(trial, metrics, fold_details)
            return metrics

        if self.min_trades > 0 and agg_metrics['n_trades'] < self.min_trades:
            metrics['score'] = -1.0
            metrics['reason'] = 'insufficient_trades'
            self._print_trial_report(trial, metrics, fold_details)
            return metrics

        metrics['profit_score_base'] = float(fold_score_p25)
        metrics['fold_factor'] = float(fold_factor)
        metrics['folds_used'] = float(fold_count)
        metrics['floor_penalty_scale'] = 0.0
        metrics['pf_floor_penalty'] = 0.0
        metrics['pnl_floor_penalty'] = 0.0
        metrics['profit_score'] = float(profit_score)

        entry_score = float(metrics.get('entry_val_precision_selected', 0.0))
        metrics['entry_score'] = float(entry_score)

        trend_val_accuracy = 0.0
        if self.trend_weight > 0 and last_fold_train is not None and last_fold_val is not None:
            trend_model = TrendClassifier(cfg.model)
            trend_metrics = trend_model.train(
                last_fold_train[feature_cols].fillna(0),
                last_fold_train['trend_label'],
                last_fold_val[feature_cols].fillna(0),
                last_fold_val['trend_label'],
                verbose=False,
            )
            trend_val_accuracy = float(trend_metrics.get('val_accuracy', 0.0))

        metrics['trend_val_accuracy'] = trend_val_accuracy

        if self.tuning_objective == "precision":
            score = entry_score
        elif self.tuning_objective == "mixed":
            profit_component = float(np.tanh(profit_score))
            score = (0.7 * profit_component) + (0.3 * entry_score)
        else:
            score = profit_score

        metrics['score'] = float(score + (self.trend_weight * trend_val_accuracy))
        metrics['target_rr'] = float(cfg.labels.target_rr)
        metrics['stop_atr_multiple'] = float(cfg.labels.stop_atr_multiple)
        metrics['pullback_threshold'] = float(cfg.labels.pullback_threshold)
        metrics['fee_per_trade_r'] = float(fee_r_median)
        metrics['fee_percent'] = float(self.fee_percent)
        metrics['median_atr_percent'] = float(median_atr_percent)
        metrics['fee_source'] = fee_source
        if all_fee_fallback_arr.size > 0:
            metrics['fee_fallback_pct'] = float(all_fee_fallback_arr.mean() * 100.0)
        else:
            metrics['fee_fallback_pct'] = 0.0
        metrics['ops_cost_enabled'] = float(1.0 if self.ops_cost_enabled else 0.0)
        metrics['ops_cost_target_trades_per_day'] = float(self.ops_cost_target_trades_per_day)
        metrics['ops_cost_c1'] = float(self.ops_cost_c1)
        metrics['ops_cost_alpha'] = float(self.ops_cost_alpha)
        metrics['ev_margin_r'] = float(best_margin)
        metrics['ev_margin_grid'] = ",".join([f"{m:.2f}" for m in margin_grid])
        metrics['lcb_z'] = float(self.lcb_z)
        metrics['use_raw_probabilities'] = float(1.0 if self.use_raw_probabilities else 0.0)
        metrics['use_expected_rr'] = float(1.0 if self.use_expected_rr else 0.0)
        self._print_trial_report(trial, metrics, fold_details)
        return metrics

    def objective(self, trial: "optuna.Trial") -> float:
        self._wait_if_paused()
        cfg, param_log = self._apply_trial(trial)
        metrics = self._evaluate_config(cfg, trial=trial)

        for key, value in metrics.items():
            if isinstance(value, (int, float, str)):
                trial.set_user_attr(key, value)

        for key, value in param_log.items():
            if isinstance(value, (int, float, str, list)):
                trial.set_user_attr(f'param_{key}', value)

        score = float(metrics.get('score', 0.0))
        trial.report(score, step=0)

        return score

    def tune(
        self,
        n_trials: int = 200,
        timeout_minutes: Optional[float] = None,
        show_progress: bool = True,
        n_jobs: int = 1,
        storage: Optional[str] = None,
        study_name: Optional[str] = None,
        load_if_exists: bool = False,
    ) -> ConfigTuningResult:
        start_time = time.time()
        self._tune_start_time = start_time
        timeout_seconds = None
        if timeout_minutes is not None:
            timeout_seconds = float(timeout_minutes) * 60.0
        n_jobs = max(1, int(n_jobs))
        if n_jobs > 1 and self.lgbm_num_threads_override is None:
            self.lgbm_num_threads_override = 1

        sampler = TPESampler(seed=self.seed)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            storage=storage,
            study_name=study_name,
            load_if_exists=bool(load_if_exists),
        )

        if not show_progress:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        self._active_study = study
        self._start_pause_listener()
        print("Press 'p' to pause tuning and open the menu.")

        def _pause_callback(study_obj: "optuna.Study", trial_obj: "optuna.Trial") -> None:
            if self.pause_requested:
                self._request_pause()

        try:
            study.optimize(
                self.objective,
                n_trials=int(n_trials),
                timeout=timeout_seconds,
                show_progress_bar=show_progress,
                n_jobs=n_jobs,
                callbacks=[_pause_callback],
            )
        finally:
            self._stop_pause_listener()

        if self.selection_enabled:
            self._run_candidate_selection(study)

        best_cfg, _ = self._apply_params(study.best_trial.params)
        best_margin = study.best_trial.user_attrs.get('ev_margin_r')
        if isinstance(best_margin, (int, float)):
            best_cfg.labels.ev_margin_r = float(best_margin)
        elapsed = time.time() - start_time

        return ConfigTuningResult(
            best_score=float(study.best_value),
            best_params=dict(study.best_trial.params),
            best_metrics={
                key.replace('param_', ''): value
                for key, value in study.best_trial.user_attrs.items()
                if not key.startswith('param_')
            },
            best_config=best_cfg,
            best_trial_number=int(study.best_trial.number),
            trials_completed=len(study.trials),
            elapsed_seconds=elapsed,
        )

    def _apply_params(self, params: Dict[str, Any]) -> Tuple[TrendFollowerConfig, Dict[str, Any]]:
        cfg = deepcopy(self.base_config)
        param_log: Dict[str, Any] = {}

        if self.tune_labels:
            cfg.labels.trend_forward_window = int(params['trend_forward_window'])
            cfg.labels.entry_forward_window = int(params['entry_forward_window'])
            cfg.labels.trend_up_threshold = float(params['trend_up_threshold'])
            cfg.labels.trend_down_threshold = float(params['trend_down_threshold'])
            cfg.labels.max_adverse_for_trend = float(params['max_adverse_for_trend'])
            cfg.labels.target_rr = float(params['target_rr'])
            cfg.labels.stop_atr_multiple = float(params['stop_atr_multiple'])
            cfg.labels.pullback_ema = int(params['pullback_ema'])
            cfg.labels.pullback_threshold = float(params['pullback_threshold'])

        if self.tune_features:
            cfg.base_timeframe_idx = int(params['base_timeframe_idx'])
            cfg.features.rsi_period = int(params['rsi_period'])
            cfg.features.adx_period = int(params['adx_period'])
            cfg.features.atr_period = int(params['atr_period'])
            cfg.features.bb_period = int(params['bb_period'])
            cfg.features.bb_std = float(params['bb_std'])
            cfg.features.volume_ma_period = int(params['volume_ma_period'])
            cfg.features.swing_lookback = int(params['swing_lookback'])

            ema_fast = int(params['ema_fast'])
            ema_mid = int(params['ema_mid'])
            ema_slow = int(params['ema_slow'])
            ema_long = int(params['ema_long'])
            cfg.features.ema_periods = sorted(
                {ema_fast, ema_mid, ema_slow, ema_long, cfg.labels.pullback_ema}
            )

        if self.tune_model:
            cfg.model.n_estimators = int(params['n_estimators'])
            cfg.model.max_depth = int(params['max_depth'])
            cfg.model.learning_rate = float(params['learning_rate'])
            cfg.model.num_leaves = int(params['num_leaves'])
            cfg.model.feature_fraction = float(params['feature_fraction'])
            cfg.model.bagging_fraction = float(params['bagging_fraction'])
            cfg.model.bagging_freq = int(params['bagging_freq'])
            cfg.model.min_child_samples = int(params['min_child_samples'])
            cfg.model.lambdaa_ele1 = float(params['lambdaa_ele1'])
            cfg.model.lambdaa_ele2 = float(params['lambdaa_ele2'])
            cfg.model.min_gain_to_split = float(params['min_gain_to_split'])

        if cfg.labels.pullback_ema not in cfg.features.ema_periods:
            cfg.features.ema_periods = sorted(
                set(cfg.features.ema_periods + [cfg.labels.pullback_ema])
            )

        param_log['ema_periods'] = cfg.features.ema_periods
        return cfg, param_log


def run_config_tuning(
    trades: Optional[pd.DataFrame],
    base_config: TrendFollowerConfig,
    tune_scope: str = "full",
    tuning_objective: str = "profit",
    n_trials: int = 200,
    timeout_minutes: Optional[float] = None,
    precision_weight: float = 0.6,
    trend_weight: float = 0.0,
    min_pullback_samples: int = 100,
    min_pullback_val_samples: int = 20,
    use_noise_filtering: bool = False,
    use_seed_ensemble: bool = False,
    n_ensemble_seeds: int = 5,
    seed: int = 42,
    show_progress: bool = True,
    report_trials: bool = True,
    n_jobs: int = 1,
    storage: Optional[str] = None,
    study_name: Optional[str] = None,
    load_if_exists: bool = False,
    fee_per_trade_r: Optional[float] = None,
    fee_percent: float = 0.0011,
    ev_margin_r: float = 0.0,
    ev_margin_fixed: bool = False,
    min_trades: int = 30,
    min_trades_per_fold: Optional[int] = None,
    min_coverage: float = 0.0,
    max_coverage: float = 0.7,
    lcb_z: float = 1.28,
    stability_score_iqr_weight: float = 0.5,
    stability_trade_iqr_weight: float = 0.1,
    stability_coverage_iqr_weight: float = 0.1,
    stability_penalty_multiplier: float = 1.5,
    stability_k: Optional[float] = None,
    coverage_k: Optional[float] = None,
    stability_eps: float = 1e-6,
    no_opportunity_penalty: float = 0.002,
    refusal_penalty: float = 0.01,
    ops_cost_enabled: bool = True,
    ops_cost_target_trades_per_day: float = 30.0,
    ops_cost_c1: float = 0.01,
    ops_cost_alpha: float = 1.7,
    single_position: bool = True,
    opposite_signal_policy: str = "flip",
    calibration_method: str = "temperature",
    use_raw_probabilities: bool = False,
    use_expected_rr: bool = False,
    use_rust_pipeline: bool = True,
    rust_cache_dir: str = "rust_cache",
    rust_write_intermediate: bool = False,
    enable_pruning: bool = True,
    selection_enabled: bool = True,
    selection_top_n: int = -1,
    selection_test_pct: float = 0.10,
    selection_shadow_pct: float = 0.05,
    selection_min_trades: int = 30,
    selection_min_trades_shadow: Optional[int] = None,
    selection_shadow_cap: int = 30,
    selection_min_profit_factor: float = 1.1,
    selection_max_drawdown_pct: float = 5.0,
    selection_max_coverage: float = 0.7,
    selection_min_total_r: float = 0.0,
) -> ConfigTuningResult:
    tuner = ConfigTuner(
        trades=trades,
        base_config=base_config,
        tune_scope=tune_scope,
        tuning_objective=tuning_objective,
        precision_weight=precision_weight,
        trend_weight=trend_weight,
        min_pullback_samples=min_pullback_samples,
        min_pullback_val_samples=min_pullback_val_samples,
        use_noise_filtering=use_noise_filtering,
        use_seed_ensemble=use_seed_ensemble,
        n_ensemble_seeds=n_ensemble_seeds,
        seed=seed,
        report_trials=report_trials,
        fee_per_trade_r=fee_per_trade_r,
        fee_percent=fee_percent,
        ev_margin_r=ev_margin_r,
        ev_margin_fixed=ev_margin_fixed,
        min_trades=min_trades,
        min_trades_per_fold=min_trades_per_fold,
        min_coverage=min_coverage,
        max_coverage=max_coverage,
        lcb_z=lcb_z,
        stability_score_iqr_weight=stability_score_iqr_weight,
        stability_trade_iqr_weight=stability_trade_iqr_weight,
        stability_coverage_iqr_weight=stability_coverage_iqr_weight,
        stability_penalty_multiplier=stability_penalty_multiplier,
        stability_k=stability_k,
        coverage_k=coverage_k,
        stability_eps=stability_eps,
        no_opportunity_penalty=no_opportunity_penalty,
        refusal_penalty=refusal_penalty,
        ops_cost_enabled=ops_cost_enabled,
        ops_cost_target_trades_per_day=ops_cost_target_trades_per_day,
        ops_cost_c1=ops_cost_c1,
        ops_cost_alpha=ops_cost_alpha,
        single_position=single_position,
        opposite_signal_policy=opposite_signal_policy,
        calibration_method=calibration_method,
        use_raw_probabilities=use_raw_probabilities,
        use_expected_rr=use_expected_rr,
        use_rust_pipeline=use_rust_pipeline,
        rust_cache_dir=rust_cache_dir,
        rust_write_intermediate=rust_write_intermediate,
        enable_pruning=enable_pruning,
        selection_enabled=selection_enabled,
        selection_top_n=selection_top_n,
        selection_test_pct=selection_test_pct,
        selection_shadow_pct=selection_shadow_pct,
        selection_min_trades=selection_min_trades,
        selection_min_trades_shadow=selection_min_trades_shadow,
        selection_shadow_cap=selection_shadow_cap,
        selection_min_profit_factor=selection_min_profit_factor,
        selection_max_drawdown_pct=selection_max_drawdown_pct,
        selection_max_coverage=selection_max_coverage,
        selection_min_total_r=selection_min_total_r,
    )
    return tuner.tune(
        n_trials=n_trials,
        timeout_minutes=timeout_minutes,
        show_progress=show_progress,
        n_jobs=n_jobs,
        storage=storage,
        study_name=study_name,
        load_if_exists=load_if_exists,
    )
