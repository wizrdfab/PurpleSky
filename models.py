"""
ML Models for trend following.
Uses LightGBM for gradient boosting classification and regression.

Improvements (v1.1):
- Probability calibration for EntryQualityModel (Isotonic/Platt scaling)
- Expected Calibration Error (ECE) diagnostic
- Multi-tier quality prediction support

Improvements (v1.2):
- Noise Injection Feature Selection: Compare features against random noise
- Seed Ensembling: Train multiple models with different seeds and average predictions
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import pickle
import json
from copy import deepcopy

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not installed. Run: pip install lightgbm")

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    SKLEARN_CALIBRATION_AVAILABLE = True
except ImportError:
    SKLEARN_CALIBRATION_AVAILABLE = False
    print("Warning: sklearn calibration not available. Run: pip install scikit-learn")

from config import ModelConfig

CONTEXT_FEATURE_NAMES = [
    "trend_prob_up",
    "trend_prob_down",
    "trend_prob_neutral",
    "regime_prob_ranging",
    "regime_prob_trend_up",
    "regime_prob_trend_down",
    "regime_prob_volatile",
]


def _apply_num_threads(params: Dict[str, Any], config: ModelConfig) -> None:
    num_threads = getattr(config, "num_threads", 0)
    if isinstance(num_threads, (int, float)) and int(num_threads) > 0:
        params["num_threads"] = int(num_threads)


def append_context_features(
    X: pd.DataFrame,
    trend_pred: Optional[Dict[str, np.ndarray]] = None,
    regime_pred: Optional[Dict[str, np.ndarray]] = None,
) -> pd.DataFrame:
    """Append trend/regime probability outputs as entry-model features."""
    X_ctx = X.copy()
    rows = len(X_ctx)

    def _safe_series(pred: Optional[Dict[str, np.ndarray]], key: str) -> np.ndarray:
        if pred is None:
            return np.zeros(rows, dtype=float)
        values = pred.get(key)
        if values is None:
            return np.zeros(rows, dtype=float)
        arr = np.asarray(values, dtype=float)
        if arr.shape[0] != rows:
            return np.zeros(rows, dtype=float)
        return arr

    X_ctx["trend_prob_up"] = _safe_series(trend_pred, "prob_up")
    X_ctx["trend_prob_down"] = _safe_series(trend_pred, "prob_down")
    X_ctx["trend_prob_neutral"] = _safe_series(trend_pred, "prob_neutral")
    X_ctx["regime_prob_ranging"] = _safe_series(regime_pred, "prob_ranging")
    X_ctx["regime_prob_trend_up"] = _safe_series(regime_pred, "prob_trend_up")
    X_ctx["regime_prob_trend_down"] = _safe_series(regime_pred, "prob_trend_down")
    X_ctx["regime_prob_volatile"] = _safe_series(regime_pred, "prob_volatile")

    return X_ctx


class TemperatureScaler:
    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        eps = 1e-6
        p = np.clip(p.astype(float), eps, 1.0 - eps)
        return np.log(p / (1.0 - p))

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-x))

    def predict(self, probs: np.ndarray) -> np.ndarray:
        logits = self._logit(probs)
        scaled = self._sigmoid(logits / float(self.temperature))
        return np.clip(scaled, 0.0, 1.0)

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        probs = np.asarray(probs, dtype=float)
        labels = np.asarray(labels, dtype=float)
        logits = self._logit(probs)
        eps = 1e-6

        def nll(temp: float) -> float:
            if temp <= 0:
                return float("inf")
            scaled = self._sigmoid(logits / temp)
            scaled = np.clip(scaled, eps, 1.0 - eps)
            return float(-np.mean(labels * np.log(scaled) + (1.0 - labels) * np.log(1.0 - scaled)))

        temps = np.exp(np.linspace(np.log(0.25), np.log(10.0), 25))
        losses = [nll(t) for t in temps]
        best_idx = int(np.argmin(losses))
        best_temp = float(temps[best_idx])

        low = max(0.05, best_temp / 2.0)
        high = min(20.0, best_temp * 2.0)
        refine = np.exp(np.linspace(np.log(low), np.log(high), 15))
        refine_losses = [nll(t) for t in refine]
        best_refine = int(np.argmin(refine_losses))
        self.temperature = float(refine[best_refine])
        return self


def compute_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict:
    """
    Compute Expected Calibration Error (ECE) and calibration diagnostics.

    ECE measures how well predicted probabilities match actual outcomes.
    A perfectly calibrated model has ECE = 0.

    Args:
        y_true: Binary true labels (0 or 1)
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration analysis

    Returns:
        Dictionary with ECE, bin details, and calibration summary
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_details = []

    total_samples = len(y_true)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Find samples in this bin
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        bin_size = in_bin.sum()

        if bin_size > 0:
            # Average predicted probability in bin
            avg_confidence = y_prob[in_bin].mean()
            # Actual accuracy in bin
            actual_accuracy = y_true[in_bin].mean()
            # Calibration error for this bin
            bin_error = abs(avg_confidence - actual_accuracy)

            # Weighted contribution to ECE
            ece += (bin_size / total_samples) * bin_error

            bin_details.append({
                'bin': f'{bin_lower:.1f}-{bin_upper:.1f}',
                'count': int(bin_size),
                'avg_confidence': float(avg_confidence),
                'actual_accuracy': float(actual_accuracy),
                'calibration_error': float(bin_error),
            })

    return {
        'ece': float(ece),
        'n_bins': n_bins,
        'bin_details': bin_details,
        'is_well_calibrated': ece < 0.05,  # < 5% ECE is generally good
    }


def add_noise_feature(X: pd.DataFrame, seed: int = 42) -> Tuple[pd.DataFrame, str]:
    """
    Add a random noise column to the feature DataFrame for feature selection.

    Features that rank below this noise column in importance should be considered
    for removal as they contribute less than random noise.

    Args:
        X: Feature DataFrame
        seed: Random seed for reproducibility

    Returns:
        Tuple of (DataFrame with noise column, noise column name)
    """
    np.random.seed(seed)
    noise_col = 'random_noise_baseline'
    X_with_noise = X.copy()
    X_with_noise[noise_col] = np.random.randn(len(X))
    return X_with_noise, noise_col


def identify_features_below_noise(
    feature_importance: Dict[str, float],
    noise_col: str = 'random_noise_baseline',
    verbose: bool = True
) -> List[str]:
    """
    Identify features that rank below the random noise column.

    Args:
        feature_importance: Dictionary mapping feature names to importance scores
        noise_col: Name of the noise column
        verbose: Print identified features

    Returns:
        List of feature names that rank below noise
    """
    if noise_col not in feature_importance:
        return []

    noise_importance = feature_importance[noise_col]
    features_below_noise = [
        feat for feat, imp in feature_importance.items()
        if imp < noise_importance and feat != noise_col
    ]

    if verbose:
        print(f"\n  Noise Injection Feature Selection:")
        print(f"    Noise column importance: {noise_importance:.0f}")
        print(f"    Features below noise: {len(features_below_noise)} / {len(feature_importance) - 1}")
        if features_below_noise:
            print(f"    Weak features (first 20): {features_below_noise[:20]}")

    return features_below_noise


def filter_features_by_noise(
    X_train: pd.DataFrame,
    X_val: Optional[pd.DataFrame],
    y_train: pd.Series,
    config,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], List[str]]:
    """
    Train a quick model with noise feature and filter out features below noise.

    Args:
        X_train: Training features
        X_val: Validation features (optional)
        y_train: Training labels
        config: ModelConfig
        verbose: Print progress

    Returns:
        Tuple of (filtered X_train, filtered X_val, list of removed features)
    """
    if not LIGHTGBM_AVAILABLE:
        return X_train, X_val, []

    # Add noise feature
    X_train_noise, noise_col = add_noise_feature(X_train)

    # Train quick model to get feature importance
    quick_params = {
        'objective': 'binary',
        'n_estimators': min(100, config.n_estimators),
        'max_depth': config.max_depth,
        'learning_rate': 0.1,  # Faster
        'verbose': -1,
        'random_state': 42,
    }
    _apply_num_threads(quick_params, config)

    quick_model = lgb.LGBMClassifier(**quick_params)
    quick_model.fit(X_train_noise, y_train)

    # Get feature importance
    importance_dict = dict(zip(X_train_noise.columns, quick_model.feature_importances_))

    # Find features below noise
    features_below = identify_features_below_noise(importance_dict, noise_col, verbose)

    # Filter features
    features_to_keep = [col for col in X_train.columns if col not in features_below]

    X_train_filtered = X_train[features_to_keep]
    X_val_filtered = X_val[features_to_keep] if X_val is not None else None

    if verbose:
        print(f"    Features kept: {len(features_to_keep)} / {len(X_train.columns)}")

    return X_train_filtered, X_val_filtered, features_below


class TrendClassifier:
    """
    Predicts whether a tradeable trend is starting.
    
    Output: probability of uptrend, downtrend, or no trend
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.feature_names: List[str] = []
        self.feature_importance: Optional[pd.DataFrame] = None
        
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        verbose: bool = True,
        n_estimators: Optional[int] = None,
    ) -> Dict:
        """
        Train the trend classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels (-1, 0, 1)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            verbose: Print training progress
            
        Returns:
            Dictionary with training metrics
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is required for training")
        
        self.feature_names = list(X_train.columns)
        
        # Convert labels: -1, 0, 1 -> 0, 1, 2 for LightGBM
        y_train_adj = y_train + 1
        
        n_estimators = int(n_estimators) if n_estimators is not None else int(self.config.n_estimators)
        n_estimators = max(1, n_estimators)

        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'n_estimators': n_estimators,
            'max_depth': self.config.max_depth,
            'learning_rate': self.config.learning_rate,
            'num_leaves': self.config.num_leaves,
            'feature_fraction': self.config.feature_fraction,
            'bagging_fraction': self.config.bagging_fraction,
            'bagging_freq': self.config.bagging_freq,
            'min_child_samples': self.config.min_child_samples,
            'reg_alpha': self.config.lambdaa_ele1,
            'reg_lambda': self.config.lambdaa_ele2,
            'min_gain_to_split': self.config.min_gain_to_split,
            'verbose': -1,
            'force_row_wise': True,
            'random_state': 42,  # For reproducibility
            'deterministic': True,  # For reproducibility
        }
        _apply_num_threads(params, self.config)
        
        callbacks = []
        if verbose:
            callbacks.append(lgb.log_evaluation(period=50))
        
        eval_set = [(X_train, y_train_adj)]
        eval_names = ['train']
        
        if X_val is not None and y_val is not None:
            y_val_adj = y_val + 1
            eval_set.append((X_val, y_val_adj))
            eval_names.append('valid')
            callbacks.append(lgb.early_stopping(stopping_rounds=50, verbose=verbose))
        
        self.model = lgb.LGBMClassifier(**params)
        self.model.fit(
            X_train, y_train_adj,
            eval_set=eval_set,
            eval_names=eval_names,
            callbacks=callbacks
        )
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Calculate metrics
        train_pred = self.model.predict(X_train) - 1  # Convert back
        metrics = {
            'train_accuracy': (train_pred == y_train).mean(),
        }
        
        if X_val is not None:
            val_pred = self.model.predict(X_val) - 1
            metrics['val_accuracy'] = (val_pred == y_val).mean()

        metrics['best_iteration'] = getattr(self.model, 'best_iteration_', None)
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Predict trend probabilities.
        
        Args:
            X: Features
            
        Returns:
            Dictionary with predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        proba = self.model.predict_proba(X)
        pred = self.model.predict(X) - 1  # Convert back to -1, 0, 1
        
        return {
            'prediction': pred,
            'prob_down': proba[:, 0],    # Class 0 = -1 (downtrend)
            'prob_neutral': proba[:, 1],  # Class 1 = 0 (no trend)
            'prob_up': proba[:, 2],       # Class 2 = 1 (uptrend)
        }
    
    def save(self, path: Path):
        """Save model to disk"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'feature_importance': self.feature_importance,
                'config': self.config
            }, f)
    
    def load(self, path: Path):
        """Load model from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.feature_importance = data['feature_importance']
        self.config = data['config']


class EntryQualityModel:
    """
    Predicts the quality of a pullback entry.

    Output: probability of successful bounce, expected R:R

    Improvements (v1.1):
    - Probability calibration using Isotonic Regression
    - ECE (Expected Calibration Error) diagnostics
    - Multi-tier quality prediction support

    Improvements (v1.2):
    - Noise Injection Feature Selection: Filter out features worse than random noise
    - Seed Ensembling: Train N models with different seeds and average predictions
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.classifier = None  # Binary: will it bounce? (single model or None if ensembling)
        self.ensemble_classifiers: List = []  # List of classifiers for seed ensembling
        self.regressor = None   # Continuous: expected win R (mean)
        self.regressor_q25 = None  # Conservative win R (quantile)
        self.tier_classifier = None  # Multi-class: quality tier (0-3)
        self.calibrator = None  # Isotonic calibrator for bounce_prob
        self.feature_names: List[str] = []
        self.filtered_feature_names: List[str] = []  # Features after noise filtering
        self.removed_features: List[str] = []  # Features removed by noise filtering
        self.calibration_stats: Optional[Dict] = None
        self.calibration_shrink_k: float = 300.0
        self.ensemble_seeds: List[int] = []  # Seeds used for ensembling
        self.use_ensemble: bool = False

    @staticmethod
    def compute_expected_rr_components(
        probs: np.ndarray,
        rr_mean: np.ndarray,
        rr_conservative: Optional[np.ndarray] = None,
        cost_r: Optional[np.ndarray] = None,
        p_clip: Tuple[float, float] = (0.01, 0.99),
        rr_clip: Tuple[float, float] = (0.05, 5.0),
    ) -> Dict[str, np.ndarray]:
        p_min, p_max = p_clip
        rr_min, rr_max = rr_clip
        p = np.clip(probs.astype(float), p_min, p_max)
        rr_mean_clipped = np.clip(rr_mean.astype(float), rr_min, rr_max)
        rr_cons = rr_mean_clipped
        if rr_conservative is not None:
            rr_cons = np.minimum(
                rr_mean_clipped,
                np.clip(rr_conservative.astype(float), rr_min, rr_max)
            )
        if cost_r is None:
            cost = np.zeros_like(p, dtype=float)
        else:
            cost = np.maximum(cost_r.astype(float), 0.0)

        ev_mean = (p * rr_mean_clipped) - ((1.0 - p) * 1.0) - cost
        ev_cons = (p * rr_cons) - ((1.0 - p) * 1.0) - cost
        implied_th = (1.0 + cost) / (rr_cons + 1.0)

        return {
            'ev_mean_r': ev_mean,
            'ev_conservative_r': ev_cons,
            'implied_threshold': implied_th,
            'p_clipped': p,
            'rr_mean_clipped': rr_mean_clipped,
            'rr_conservative': rr_cons,
            'cost_r': cost,
        }

    def train(
        self,
        X_train: pd.DataFrame,
        y_success: pd.Series,
        y_rr: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_success_val: Optional[pd.Series] = None,
        y_rr_val: Optional[pd.Series] = None,
        y_tier_train: Optional[pd.Series] = None,
        y_tier_val: Optional[pd.Series] = None,
        verbose: bool = True,
        n_estimators: Optional[int] = None,
        calibrate: bool = True,
        calibration_mode: str = "val",
        calibration_oof_folds: int = 3,
        calibration_shrink_k: float = 300.0,
        calibration_method: str = "temperature",
        use_noise_filtering: bool = False,
        use_seed_ensemble: bool = False,
        n_ensemble_seeds: int = 5,
    ) -> Dict:
        """
        Train classifier, regressor, and optional tier classifier with calibration.

        Args:
            X_train, y_success, y_rr: Training data
            X_val, y_success_val, y_rr_val: Validation data (optional)
            y_tier_train, y_tier_val: Multi-tier labels (0-3) for tier classifier
            verbose: Print training progress
            n_estimators: Override default n_estimators
            calibrate: Whether to apply probability calibration
            calibration_mode: Calibration data source ("val" or "oof").
            calibration_oof_folds: Number of OOF folds for time-ordered calibration.
            calibration_shrink_k: Shrinkage strength for blending calibrated probs with raw probs.
            calibration_method: Calibration model ("temperature" or "isotonic").
            use_noise_filtering: If True, filter out features ranking below random noise
            use_seed_ensemble: If True, train N models with different seeds and average
            n_ensemble_seeds: Number of seeds to use for ensembling (default: 5)

        Returns:
            Dictionary with training metrics and calibration stats
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is required for training")

        if not np.isfinite(calibration_shrink_k) or calibration_shrink_k < 0:
            calibration_shrink_k = 0.0
        self.calibration_shrink_k = float(calibration_shrink_k)

        self.feature_names = list(X_train.columns)
        self.use_ensemble = use_seed_ensemble

        # =====================================================
        # NOISE INJECTION FEATURE SELECTION (optional)
        # =====================================================
        X_train_filtered = X_train
        X_val_filtered = X_val
        self.removed_features = []

        if use_noise_filtering:
            if verbose:
                print("  Running Noise Injection Feature Selection...")
            X_train_filtered, X_val_filtered, self.removed_features = filter_features_by_noise(
                X_train, X_val, y_success, self.config, verbose=verbose
            )
            self.filtered_feature_names = list(X_train_filtered.columns)
        else:
            self.filtered_feature_names = list(X_train.columns)

        # Train classifier
        n_estimators = int(n_estimators) if n_estimators is not None else int(self.config.n_estimators)
        n_estimators = max(1, n_estimators)

        clf_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'n_estimators': n_estimators,
            'max_depth': self.config.max_depth,
            'learning_rate': self.config.learning_rate,
            'num_leaves': self.config.num_leaves,
            'feature_fraction': self.config.feature_fraction,
            'bagging_fraction': self.config.bagging_fraction,
            'bagging_freq': self.config.bagging_freq,

            'reg_alpha': self.config.lambdaa_ele1,
            'reg_lambda': self.config.lambdaa_ele2,
            'min_child_samples': self.config.min_child_samples,
            'min_gain_to_split': self.config.min_gain_to_split,

            'verbose': -1,
            # Handle class imbalance automatically (help recall without destroying precision)
            'class_weight': 'balanced',
        }
        _apply_num_threads(clf_params, self.config)

        # =====================================================
        # SEED ENSEMBLING (optional)
        # =====================================================
        if use_seed_ensemble:
            if verbose:
                print(f"  Training Seed Ensemble with {n_ensemble_seeds} models...")

            self.ensemble_seeds = [42 + i * 17 for i in range(n_ensemble_seeds)]  # Deterministic seeds
            self.ensemble_classifiers = []
            best_iterations = []

            for i, seed in enumerate(self.ensemble_seeds):
                if verbose:
                    print(f"    Training model {i+1}/{n_ensemble_seeds} (seed={seed})...")

                seed_params = clf_params.copy()
                seed_params['random_state'] = seed

                clf = lgb.LGBMClassifier(**seed_params)

                eval_set_clf = [(X_train_filtered, y_success)]
                if X_val_filtered is not None:
                    eval_set_clf.append((X_val_filtered, y_success_val))

                callbacks = []
                if X_val_filtered is not None:
                    callbacks.append(lgb.early_stopping(stopping_rounds=50, verbose=False))

                clf.fit(
                    X_train_filtered, y_success,
                    eval_set=eval_set_clf,
                    callbacks=callbacks if callbacks else None
                )

                self.ensemble_classifiers.append(clf)
                best_iterations.append(getattr(clf, 'best_iteration_', n_estimators))

            # Use first classifier as the "main" one for compatibility
            self.classifier = self.ensemble_classifiers[0]

            if verbose:
                print(f"    Ensemble trained. Best iterations: {best_iterations}")
                print(f"    Avg best iteration: {np.mean(best_iterations):.0f}")

        else:
            # Single model training (original behavior)
            self.classifier = lgb.LGBMClassifier(**clf_params)

            eval_set_clf = [(X_train_filtered, y_success)]
            if X_val_filtered is not None:
                eval_set_clf.append((X_val_filtered, y_success_val))

            callbacks = [lgb.log_evaluation(period=50)] if verbose else []
            if X_val_filtered is not None:
                callbacks.append(lgb.early_stopping(stopping_rounds=50, verbose=verbose))

            self.classifier.fit(
                X_train_filtered, y_success,
                eval_set=eval_set_clf,
                callbacks=callbacks
            )

        # =====================================================
        # PROBABILITY CALIBRATION
        # =====================================================
        if calibrate and SKLEARN_CALIBRATION_AVAILABLE:
            cal_mode = (calibration_mode or "val").strip().lower()
            use_oof = cal_mode in {"oof", "oof_time", "time_oof"}
            cal_method_name = (calibration_method or "temperature").strip().lower()
            if cal_method_name in {"temp", "temperature", "temperature_scaling"}:
                calibrator_name = "Temperature Scaling"
                use_temperature = True
            else:
                calibrator_name = "Isotonic Regression"
                use_temperature = False

            if use_oof:
                if verbose:
                    print("  Using OOF time-ordered calibration (Train Only)...")

                n_samples = len(X_train_filtered)
                k_folds = int(calibration_oof_folds)
                if k_folds < 2:
                    k_folds = 2
                if n_samples < max(200, k_folds * 30):
                    cal_raw_probs = self._get_raw_probs(X_train_filtered)
                    cal_y = y_success
                    cal_method = "OOF Time (Fallback Full)"
                else:
                    fold_sizes = np.full(k_folds, n_samples // k_folds, dtype=int)
                    fold_sizes[: n_samples % k_folds] += 1
                    indices = np.arange(n_samples)
                    cal_raw_probs = np.zeros(n_samples, dtype=float)
                    filled_mask = np.zeros(n_samples, dtype=bool)
                    start = 0
                    for fold_idx, fold_size in enumerate(fold_sizes, start=1):
                        stop = start + fold_size
                        val_idx = indices[start:stop]
                        train_idx = indices[:start]
                        if train_idx.size < 10 or val_idx.size == 0:
                            start = stop
                            continue
                        fold_model = lgb.LGBMClassifier(**clf_params)
                        fold_model.fit(
                            X_train_filtered.iloc[train_idx],
                            y_success.iloc[train_idx],
                        )
                        fold_probs = fold_model.predict_proba(X_train_filtered.iloc[val_idx])[:, 1]
                        cal_raw_probs[val_idx] = fold_probs
                        filled_mask[val_idx] = True
                        start = stop
                    if filled_mask.sum() < max(10, int(0.2 * n_samples)):
                        cal_raw_probs = self._get_raw_probs(X_train_filtered)
                        cal_y = y_success
                        cal_method = "OOF Time (Fallback Full)"
                    else:
                        cal_raw_probs = cal_raw_probs[filled_mask]
                        cal_y = y_success.iloc[filled_mask]
                        cal_method = f"OOF Time ({k_folds} folds)"
            else:
                if X_val_filtered is not None and y_success_val is not None and len(y_success_val) > 0:
                    cal_X = X_val_filtered
                    cal_y = y_success_val
                    cal_method = "Validation Set"
                else:
                    cal_X = X_train_filtered
                    cal_y = y_success
                    cal_method = "Train Set"

                cal_raw_probs = self._get_raw_probs(cal_X)

            if len(np.unique(cal_y)) < 2:
                if verbose:
                    print(f"  Skipping calibration ({cal_method}): Target contains only one class.")
                self.calibrator = None
            else:
                if verbose:
                    print(f"  Calibrating probabilities ({calibrator_name}) using {cal_method}...")

                if use_temperature:
                    self.calibrator = TemperatureScaler().fit(cal_raw_probs, cal_y.values)
                else:
                    self.calibrator = IsotonicRegression(out_of_bounds='clip')
                    self.calibrator.fit(cal_raw_probs, cal_y.values)

                # Compute ECE
                calibrated_probs = self.calibrator.predict(cal_raw_probs)
                n_cal = int(len(cal_y))
                shrink_k = float(self.calibration_shrink_k)
                shrink_w = 1.0
                if n_cal > 0 and shrink_k > 0:
                    shrink_w = float(n_cal / (n_cal + shrink_k))
                    calibrated_probs = (shrink_w * calibrated_probs) + ((1.0 - shrink_w) * cal_raw_probs)

                ece_res = compute_expected_calibration_error(cal_y.values, calibrated_probs)
                
                self.calibration_stats = {
                    'method': cal_method,
                    'calibrator': calibrator_name,
                    'ece': ece_res['ece'],
                    'details': ece_res,
                    'n_cal': float(n_cal),
                    'shrink_k': float(shrink_k),
                    'shrink_w': float(shrink_w),
                }
                if use_temperature:
                    self.calibration_stats['temperature'] = float(self.calibrator.temperature)

                if verbose:
                    print(f"    Calibration ECE: {ece_res['ece']:.4f}")

            if verbose and self.calibration_stats:
                if 'pre_calibration_ece' in self.calibration_stats:
                    print(f"    Pre-calibration ECE:  {self.calibration_stats['pre_calibration_ece']:.4f}")
                    print(f"    Post-calibration ECE: {self.calibration_stats['post_calibration_ece']:.4f}")
                    print(f"    ECE Improvement:      {self.calibration_stats['ece_improvement']:.4f}")
                else:
                    # CV / Self-Calibration case
                    print(f"    Calibration ECE:      {self.calibration_stats['ece']:.4f} ({self.calibration_stats['method']})")

        # =====================================================
        # MULTI-TIER QUALITY CLASSIFIER (optional)
        # =====================================================
        if y_tier_train is not None and y_tier_train.notna().sum() > 100:
            if verbose:
                print("  Training multi-tier quality classifier...")

            # Filter to samples with tier labels (use filtered features)
            tier_mask_train = y_tier_train.notna()
            X_tier_train = X_train_filtered[tier_mask_train]
            y_tier_train_clean = y_tier_train[tier_mask_train].astype(int)
            if y_tier_train_clean.nunique() < 2:
                if verbose:
                    print("    Skipping tier classifier (need >=2 classes in train).")
            else:
                tier_params = {
                    'objective': 'multiclass',
                    'num_class': 4,  # Tiers 0, 1, 2, 3
                    'metric': 'multi_logloss',
                    'n_estimators': n_estimators // 2,
                    'max_depth': self.config.max_depth,
                    'learning_rate': self.config.learning_rate,
                    'num_leaves': self.config.num_leaves,
                    'feature_fraction': self.config.feature_fraction,
                    'bagging_fraction': self.config.bagging_fraction,
                    'bagging_freq': self.config.bagging_freq,
                    'min_child_samples': self.config.min_child_samples,
                    'reg_alpha': self.config.lambdaa_ele1,
                    'reg_lambda': self.config.lambdaa_ele2,
                    'min_gain_to_split': self.config.min_gain_to_split,
                    'verbose': -1,
                }
                _apply_num_threads(tier_params, self.config)

                self.tier_classifier = lgb.LGBMClassifier(**tier_params)

                tier_eval_set = [(X_tier_train, y_tier_train_clean)]
                tier_callbacks = []

                if y_tier_val is not None and y_tier_val.notna().sum() > 50:
                    tier_mask_val = y_tier_val.notna()
                    X_tier_val = X_val_filtered[tier_mask_val]
                    y_tier_val_clean = y_tier_val[tier_mask_val].astype(int)
                    train_classes = set(y_tier_train_clean.unique())
                    val_mask = y_tier_val_clean.isin(train_classes).to_numpy(dtype=bool)
                    if val_mask.any():
                        X_tier_val = X_tier_val[val_mask]
                        y_tier_val_clean = y_tier_val_clean[val_mask]
                    if y_tier_val_clean.size > 50:
                        tier_eval_set.append((X_tier_val, y_tier_val_clean))
                        tier_callbacks.append(lgb.early_stopping(stopping_rounds=30, verbose=False))

                self.tier_classifier.fit(
                    X_tier_train, y_tier_train_clean,
                    eval_set=tier_eval_set,
                    callbacks=tier_callbacks if tier_callbacks else None
                )

                if verbose:
                    tier_train_acc = (self.tier_classifier.predict(X_tier_train) == y_tier_train_clean).mean()
                    print(f"    Tier classifier train accuracy: {tier_train_acc:.3f}")

        # Train regressors for win magnitude (mean + conservative quantile)
        success_mask = (y_success == 1) & y_rr.notna()
        if success_mask.sum() > 100:
            base_reg_params = {
                'n_estimators': max(50, self.config.n_estimators // 2),
                'max_depth': self.config.max_depth,
                'learning_rate': self.config.learning_rate,
                'num_leaves': self.config.num_leaves,
                'feature_fraction': self.config.feature_fraction,
                'bagging_fraction': self.config.bagging_fraction,
                'bagging_freq': self.config.bagging_freq,
                'min_child_samples': self.config.min_child_samples,
                'reg_alpha': self.config.lambdaa_ele1,
                'reg_lambda': self.config.lambdaa_ele2,
                'min_gain_to_split': self.config.min_gain_to_split,
                'verbose': -1,
            }
            _apply_num_threads(base_reg_params, self.config)

            mean_params = dict(base_reg_params)
            mean_params.update({'objective': 'regression', 'metric': 'rmse'})
            self.regressor = lgb.LGBMRegressor(**mean_params)
            self.regressor.fit(
                X_train_filtered[success_mask],
                y_rr[success_mask]
            )

            q_params = dict(base_reg_params)
            q_params.update({'objective': 'quantile', 'alpha': 0.25, 'metric': 'quantile'})
            self.regressor_q25 = lgb.LGBMRegressor(**q_params)
            self.regressor_q25.fit(
                X_train_filtered[success_mask],
                y_rr[success_mask]
            )

        # Metrics (use filtered features)
        train_pred = self._get_predictions(X_train_filtered)
        metrics = {
            'train_accuracy': (train_pred == y_success).mean(),
            'train_precision': self._precision(train_pred, y_success),
        }

        if X_val_filtered is not None:
            val_pred = self._get_predictions(X_val_filtered)
            metrics['val_accuracy'] = (val_pred == y_success_val).mean()
            metrics['val_precision'] = self._precision(val_pred, y_success_val)

        raw_train_probs = self._get_raw_probs(X_train_filtered)
        train_raw_metrics = self._binary_metrics_from_probs(raw_train_probs, y_success.values)
        metrics['train_base_rate'] = float(y_success.mean())
        metrics['train_accuracy_raw'] = train_raw_metrics['accuracy']
        metrics['train_precision_raw'] = train_raw_metrics['precision']
        metrics['train_recall_raw'] = train_raw_metrics['recall']

        cal_train_probs = self._apply_calibration(raw_train_probs)
        train_cal_metrics = self._binary_metrics_from_probs(cal_train_probs, y_success.values)
        metrics['train_accuracy_cal'] = train_cal_metrics['accuracy']
        metrics['train_precision_cal'] = train_cal_metrics['precision']
        metrics['train_recall_cal'] = train_cal_metrics['recall']

        if X_val_filtered is not None and y_success_val is not None:
            raw_val_probs = self._get_raw_probs(X_val_filtered)
            val_raw_metrics = self._binary_metrics_from_probs(raw_val_probs, y_success_val.values)
            metrics['val_base_rate'] = float(y_success_val.mean())
            metrics['val_accuracy_raw'] = val_raw_metrics['accuracy']
            metrics['val_precision_raw'] = val_raw_metrics['precision']
            metrics['val_recall_raw'] = val_raw_metrics['recall']

            cal_val_probs = self._apply_calibration(raw_val_probs)
            val_cal_metrics = self._binary_metrics_from_probs(cal_val_probs, y_success_val.values)
            metrics['val_accuracy_cal'] = val_cal_metrics['accuracy']
            metrics['val_precision_cal'] = val_cal_metrics['precision']
            metrics['val_recall_cal'] = val_cal_metrics['recall']

        metrics['best_iteration'] = getattr(self.classifier, 'best_iteration_', None)

        # Add feature filtering stats
        if use_noise_filtering:
            metrics['noise_filtering'] = {
                'features_removed': len(self.removed_features),
                'features_kept': len(self.filtered_feature_names),
                'removed_features': self.removed_features[:50],  # First 50 for brevity
            }

        # Add ensemble stats
        if use_seed_ensemble:
            metrics['seed_ensemble'] = {
                'n_models': len(self.ensemble_classifiers),
                'seeds': self.ensemble_seeds,
            }

        if self.calibration_stats:
            metrics['calibration'] = self.calibration_stats

        return metrics

    def _get_raw_probs(self, X: pd.DataFrame) -> np.ndarray:
        """Get raw probabilities, using ensemble average if applicable."""
        if self.use_ensemble and self.ensemble_classifiers:
            # Average probabilities from all ensemble members
            probs = np.zeros(len(X))
            for clf in self.ensemble_classifiers:
                probs += clf.predict_proba(X)[:, 1]
            return probs / len(self.ensemble_classifiers)
        else:
            return self.classifier.predict_proba(X)[:, 1]

    def _apply_calibration(self, raw_probs: np.ndarray) -> np.ndarray:
        if self.calibrator is None:
            return raw_probs
        cal_probs = self.calibrator.predict(raw_probs)
        if not self.calibration_stats:
            return cal_probs
        n_cal = self.calibration_stats.get('n_cal')
        shrink_k = self.calibration_stats.get('shrink_k')
        if n_cal is None or shrink_k is None:
            return cal_probs
        n_cal = float(n_cal)
        shrink_k = float(shrink_k)
        if n_cal <= 0 or shrink_k <= 0:
            return cal_probs
        shrink_w = n_cal / (n_cal + shrink_k)
        return (shrink_w * cal_probs) + ((1.0 - shrink_w) * raw_probs)

    def _get_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """Get binary predictions, using ensemble average if applicable."""
        probs = self._get_raw_probs(X)
        return (probs >= 0.5).astype(int)

    @staticmethod
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

    def _precision(self, pred, true):
        """Calculate precision"""
        pred_positive = pred == 1
        if pred_positive.sum() == 0:
            return 0.0
        return (pred[pred_positive] == true[pred_positive]).mean()

    def predict(
        self,
        X: pd.DataFrame,
        use_calibration: bool = False,
        cost_r: Optional[np.ndarray] = None,
        compute_ev: bool = False,
        p_clip: Tuple[float, float] = (0.01, 0.99),
        rr_clip: Tuple[float, float] = (0.05, 5.0),
    ) -> Dict[str, np.ndarray]:
        """
        Predict entry quality.

        Args:
            X: Features DataFrame
            use_calibration: Whether to apply probability calibration (if available)

        Returns:
            Dictionary with predictions including:
            - bounce_prob: Probability of successful bounce (calibrated if available)
            - bounce_prob_raw: Raw uncalibrated probability
            - bounce_pred: Binary prediction
            - expected_rr: Conservative expected win R (if regressor trained)
            - expected_rr_mean: Mean predicted win R (if regressor trained)
            - expected_rr_q25: 25th percentile win R (if quantile regressor trained)
            - ev_mean_r / ev_conservative_r: Expected value components (if compute_ev=True)
            - tier_prob_*: Tier probabilities (if tier classifier trained)
        """
        if self.classifier is None:
            raise ValueError("Model not trained yet")

        # Filter to only the features used during training (if noise filtering was applied)
        X_filtered = X
        if self.filtered_feature_names and len(self.filtered_feature_names) < len(X.columns):
            # Only use features that were kept after noise filtering
            available_cols = [c for c in self.filtered_feature_names if c in X.columns]
            X_filtered = X[available_cols]

        # Get raw probabilities (handles ensemble averaging internally)
        raw_prob = self._get_raw_probs(X_filtered)

        result = {
            'bounce_prob_raw': raw_prob,
            'bounce_pred': self._get_predictions(X_filtered),
        }

        # Apply calibration if available
        if use_calibration and self.calibrator is not None:
            result['bounce_prob'] = self._apply_calibration(raw_prob)
        else:
            result['bounce_prob'] = raw_prob

        if self.regressor is not None:
            rr_mean = self.regressor.predict(X_filtered)
            rr_q25 = None
            if self.regressor_q25 is not None:
                rr_q25 = self.regressor_q25.predict(X_filtered)

            rr_min, rr_max = rr_clip
            rr_mean_clipped = np.clip(rr_mean, rr_min, rr_max)
            if rr_q25 is not None:
                rr_q25_clipped = np.clip(rr_q25, rr_min, rr_max)
                rr_cons = np.minimum(rr_mean_clipped, rr_q25_clipped)
                result['expected_rr_q25'] = rr_q25_clipped
            else:
                rr_cons = rr_mean_clipped

            result['expected_rr_mean'] = rr_mean_clipped
            result['expected_rr'] = rr_cons

            if compute_ev and cost_r is not None:
                ev = self.compute_expected_rr_components(
                    result['bounce_prob'],
                    rr_mean_clipped,
                    rr_conservative=rr_cons,
                    cost_r=cost_r,
                    p_clip=p_clip,
                    rr_clip=rr_clip,
                )
                result.update(ev)

        # Tier predictions if available
        if self.tier_classifier is not None:
            tier_proba = self.tier_classifier.predict_proba(X_filtered)
            result['tier_pred'] = self.tier_classifier.predict(X_filtered)
            # Handle case where not all tiers were seen during training
            classes = self.tier_classifier.classes_
            for i, cls in enumerate(classes):
                result[f'tier_prob_{cls}'] = tier_proba[:, i]

        return result

    def save(self, path: Path):
        """Save model to disk"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump({
                'classifier': self.classifier,
                'ensemble_classifiers': self.ensemble_classifiers,
                'regressor': self.regressor,
                'regressor_q25': self.regressor_q25,
                'tier_classifier': self.tier_classifier,
                'calibrator': self.calibrator,
                'calibration_stats': self.calibration_stats,
                'feature_names': self.feature_names,
                'filtered_feature_names': self.filtered_feature_names,
                'removed_features': self.removed_features,
                'ensemble_seeds': self.ensemble_seeds,
                'use_ensemble': self.use_ensemble,
                'config': self.config
            }, f)

    def load(self, path: Path):
        """Load model from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.classifier = data['classifier']
        self.ensemble_classifiers = data.get('ensemble_classifiers', [])
        self.regressor = data.get('regressor')
        self.regressor_q25 = data.get('regressor_q25')
        self.tier_classifier = data.get('tier_classifier')
        self.calibrator = data.get('calibrator')
        self.calibration_stats = data.get('calibration_stats')
        self.feature_names = data['feature_names']
        self.filtered_feature_names = data.get('filtered_feature_names', data['feature_names'])
        self.removed_features = data.get('removed_features', [])
        self.ensemble_seeds = data.get('ensemble_seeds', [])
        self.use_ensemble = data.get('use_ensemble', False)
        self.config = data['config']


class RegimeClassifier:
    """
    Classifies current market regime.
    
    Output: regime type (ranging, trending up, trending down, volatile)
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.feature_names: List[str] = []
        
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        verbose: bool = True,
        n_estimators: Optional[int] = None,
    ) -> Dict:
        """Train regime classifier"""
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is required for training")
        
        self.feature_names = list(X_train.columns)
        
        n_estimators = int(n_estimators) if n_estimators is not None else int(self.config.n_estimators)
        n_estimators = max(1, n_estimators)

        params = {
            'objective': 'multiclass',
            'num_class': 4,
            'metric': 'multi_logloss',
            'n_estimators': n_estimators,
            'max_depth': self.config.max_depth,
            'learning_rate': self.config.learning_rate,
            'num_leaves': self.config.num_leaves,
            'feature_fraction': self.config.feature_fraction,
            'bagging_fraction': self.config.bagging_fraction,
            'bagging_freq': self.config.bagging_freq,
            'min_child_samples': self.config.min_child_samples,
            'reg_alpha': self.config.lambdaa_ele1,
            'reg_lambda': self.config.lambdaa_ele2,
            'min_gain_to_split': self.config.min_gain_to_split,
            'verbose': -1,
        }
        _apply_num_threads(params, self.config)
        
        self.model = lgb.LGBMClassifier(**params)
        
        eval_set = [(X_train, y_train)]
        callbacks = [lgb.log_evaluation(period=50)] if verbose else []
        
        if X_val is not None:
            eval_set.append((X_val, y_val))
            callbacks.append(lgb.early_stopping(stopping_rounds=50, verbose=verbose))
        
        self.model.fit(X_train, y_train, eval_set=eval_set, callbacks=callbacks)
        
        metrics = {
            'train_accuracy': (self.model.predict(X_train) == y_train).mean()
        }
        
        if X_val is not None:
            metrics['val_accuracy'] = (self.model.predict(X_val) == y_val).mean()

        metrics['best_iteration'] = getattr(self.model, 'best_iteration_', None)
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Predict regime"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        proba = self.model.predict_proba(X)
        predictions = self.model.predict(X)
        
        # Get the actual classes the model knows about
        classes = self.model.classes_
        n_samples = len(X)
        
        # Initialize probabilities for all possible regimes (0-3)
        prob_ranging = np.zeros(n_samples)
        prob_trend_up = np.zeros(n_samples)
        prob_trend_down = np.zeros(n_samples)
        prob_volatile = np.zeros(n_samples)
        
        # Map probabilities based on which classes the model learned
        for i, cls in enumerate(classes):
            if cls == 0:
                prob_ranging = proba[:, i]
            elif cls == 1:
                prob_trend_up = proba[:, i]
            elif cls == 2:
                prob_trend_down = proba[:, i]
            elif cls == 3:
                prob_volatile = proba[:, i]
        
        return {
            'regime': predictions,
            'prob_ranging': prob_ranging,
            'prob_trend_up': prob_trend_up,
            'prob_trend_down': prob_trend_down,
            'prob_volatile': prob_volatile,
        }
    
    def save(self, path: Path):
        """Save model to disk"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'config': self.config
            }, f)
    
    def load(self, path: Path):
        """Load model from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.config = data['config']


class TrendFollowerModels:
    """
    Container for all trend follower models.
    Provides unified interface for training and prediction.
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.trend_classifier = TrendClassifier(config)
        self.entry_model = EntryQualityModel(config)
        self.regime_classifier = RegimeClassifier(config)
        
    def save_all(self, model_dir: Path):
        """Save all models"""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        self.trend_classifier.save(model_dir / 'trend_classifier.pkl')
        self.entry_model.save(model_dir / 'entry_model.pkl')
        self.regime_classifier.save(model_dir / 'regime_classifier.pkl')
        
        print(f"Models saved to {model_dir}")
    
    def load_all(self, model_dir: Path):
        """Load all models"""
        model_dir = Path(model_dir)
        
        self.trend_classifier.load(model_dir / 'trend_classifier.pkl')
        self.entry_model.load(model_dir / 'entry_model.pkl')
        self.regime_classifier.load(model_dir / 'regime_classifier.pkl')
        
        print(f"Models loaded from {model_dir}")


if __name__ == "__main__":
    from config import DEFAULT_CONFIG
    
    print("Models module loaded successfully")
    print(f"LightGBM available: {LIGHTGBM_AVAILABLE}")
    
    if LIGHTGBM_AVAILABLE:
        models = TrendFollowerModels(DEFAULT_CONFIG.model)
        print("Model container initialized")
