"""
Optuna-based configuration tuner for TrendFollower.

This module tunes model, feature, and label parameters from config.py
to maximize validation accuracy/precision (entry model, optionally trend).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from copy import deepcopy
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import time

import pandas as pd

from config import TrendFollowerConfig
from data_loader import create_multi_timeframe_bars
from feature_engine import calculate_multi_timeframe_features
from labels import create_training_dataset
from models import TrendClassifier, EntryQualityModel
from trainer import time_series_split

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
        trades: pd.DataFrame,
        base_config: TrendFollowerConfig,
        tune_scope: str = "full",
        precision_weight: float = 0.6,
        trend_weight: float = 0.0,
        min_pullback_samples: int = 100,
        min_pullback_val_samples: int = 20,
        use_noise_filtering: bool = False,
        use_seed_ensemble: bool = False,
        n_ensemble_seeds: int = 5,
        seed: int = 42,
    ):
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required. Install with: pip install optuna")

        self.trades = trades
        self.base_config = deepcopy(base_config)
        self.seed = int(seed)

        self.tune_scope = (tune_scope or "full").strip().lower()
        self.tune_model = self.tune_scope in {"model", "full", "all"}
        self.tune_features = self.tune_scope in {"features", "full", "all"}
        self.tune_labels = self.tune_scope in {"labels", "full", "all"}

        self.precision_weight = float(precision_weight)
        self.trend_weight = float(trend_weight)
        self.min_pullback_samples = int(min_pullback_samples)
        self.min_pullback_val_samples = int(min_pullback_val_samples)

        self.use_noise_filtering = bool(use_noise_filtering)
        self.use_seed_ensemble = bool(use_seed_ensemble)
        self.n_ensemble_seeds = int(n_ensemble_seeds)

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

    def _evaluate_config(self, cfg: TrendFollowerConfig) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        bars_dict = create_multi_timeframe_bars(
            self.trades,
            cfg.features.timeframes,
            cfg.features.timeframe_names,
            cfg.data,
        )
        base_tf = cfg.features.timeframe_names[cfg.base_timeframe_idx]
        featured_data = calculate_multi_timeframe_features(
            bars_dict,
            base_tf,
            cfg.features,
        )
        labeled_data, feature_cols = create_training_dataset(
            featured_data,
            cfg.labels,
            cfg.features,
            base_tf,
        )

        if not feature_cols:
            metrics['score'] = 0.0
            metrics['reason'] = 'no_features'
            return metrics

        train_df, val_df, _ = time_series_split(
            labeled_data,
            cfg.model.train_ratio,
            cfg.model.val_ratio,
            cfg.model.test_ratio,
        )

        metrics['train_size'] = float(len(train_df))
        metrics['val_size'] = float(len(val_df))
        metrics['feature_count'] = float(len(feature_cols))

        if len(train_df) == 0 or len(val_df) == 0:
            metrics['score'] = 0.0
            metrics['reason'] = 'empty_split'
            return metrics

        X_train = train_df[feature_cols].fillna(0)
        X_val = val_df[feature_cols].fillna(0)

        pullback_mask_train = ~train_df['pullback_success'].isna()
        pullback_mask_val = ~val_df['pullback_success'].isna()
        pullback_train = int(pullback_mask_train.sum())
        pullback_val = int(pullback_mask_val.sum())

        metrics['pullback_train'] = float(pullback_train)
        metrics['pullback_val'] = float(pullback_val)

        if pullback_train < self.min_pullback_samples or pullback_val < self.min_pullback_val_samples:
            metrics['score'] = 0.0
            metrics['reason'] = 'insufficient_pullbacks'
            return metrics

        X_entry_train = X_train[pullback_mask_train]
        y_success_train = train_df.loc[pullback_mask_train, 'pullback_success'].astype(int)
        y_rr_train = train_df.loc[pullback_mask_train, 'pullback_rr']

        X_entry_val = X_val[pullback_mask_val]
        y_success_val = val_df.loc[pullback_mask_val, 'pullback_success'].astype(int)
        y_rr_val = val_df.loc[pullback_mask_val, 'pullback_rr']

        entry_model = EntryQualityModel(cfg.model)
        entry_metrics = entry_model.train(
            X_entry_train,
            y_success_train,
            y_rr_train,
            X_entry_val,
            y_success_val,
            y_rr_val,
            verbose=False,
            calibrate=True,
            use_noise_filtering=self.use_noise_filtering,
            use_seed_ensemble=self.use_seed_ensemble,
            n_ensemble_seeds=self.n_ensemble_seeds,
        )

        entry_val_accuracy = float(entry_metrics.get('val_accuracy', 0.0))
        entry_val_precision = float(entry_metrics.get('val_precision', 0.0))

        metrics['entry_val_accuracy'] = entry_val_accuracy
        metrics['entry_val_precision'] = entry_val_precision

        entry_score = (
            (1.0 - self.precision_weight) * entry_val_accuracy
            + self.precision_weight * entry_val_precision
        )

        trend_val_accuracy = 0.0
        if self.trend_weight > 0:
            trend_model = TrendClassifier(cfg.model)
            trend_metrics = trend_model.train(
                X_train,
                train_df['trend_label'],
                X_val,
                val_df['trend_label'],
                verbose=False,
            )
            trend_val_accuracy = float(trend_metrics.get('val_accuracy', 0.0))

        metrics['trend_val_accuracy'] = trend_val_accuracy
        metrics['score'] = entry_score + (self.trend_weight * trend_val_accuracy)
        return metrics

    def objective(self, trial: "optuna.Trial") -> float:
        cfg, param_log = self._apply_trial(trial)
        metrics = self._evaluate_config(cfg)

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
    ) -> ConfigTuningResult:
        start_time = time.time()
        timeout_seconds = None
        if timeout_minutes is not None:
            timeout_seconds = float(timeout_minutes) * 60.0

        sampler = TPESampler(seed=self.seed)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
        )

        if not show_progress:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        study.optimize(
            self.objective,
            n_trials=int(n_trials),
            timeout=timeout_seconds,
            show_progress_bar=show_progress,
        )

        best_cfg, _ = self._apply_params(study.best_trial.params)
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
    trades: pd.DataFrame,
    base_config: TrendFollowerConfig,
    tune_scope: str = "full",
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
) -> ConfigTuningResult:
    tuner = ConfigTuner(
        trades=trades,
        base_config=base_config,
        tune_scope=tune_scope,
        precision_weight=precision_weight,
        trend_weight=trend_weight,
        min_pullback_samples=min_pullback_samples,
        min_pullback_val_samples=min_pullback_val_samples,
        use_noise_filtering=use_noise_filtering,
        use_seed_ensemble=use_seed_ensemble,
        n_ensemble_seeds=n_ensemble_seeds,
        seed=seed,
    )
    return tuner.tune(
        n_trials=n_trials,
        timeout_minutes=timeout_minutes,
        show_progress=show_progress,
    )
