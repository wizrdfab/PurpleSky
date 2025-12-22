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
from typing import Dict, List, Tuple, Optional
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
    SKLEARN_CALIBRATION_AVAILABLE = True
except ImportError:
    SKLEARN_CALIBRATION_AVAILABLE = False
    print("Warning: sklearn calibration not available. Run: pip install scikit-learn")

from config import ModelConfig


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
        self.regressor = None   # Continuous: expected R:R
        self.tier_classifier = None  # Multi-class: quality tier (0-3)
        self.calibrator = None  # Isotonic calibrator for bounce_prob
        self.feature_names: List[str] = []
        self.filtered_feature_names: List[str] = []  # Features after noise filtering
        self.removed_features: List[str] = []  # Features removed by noise filtering
        self.calibration_stats: Optional[Dict] = None
        self.ensemble_seeds: List[int] = []  # Seeds used for ensembling
        self.use_ensemble: bool = False

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
            use_noise_filtering: If True, filter out features ranking below random noise
            use_seed_ensemble: If True, train N models with different seeds and average
            n_ensemble_seeds: Number of seeds to use for ensembling (default: 5)

        Returns:
            Dictionary with training metrics and calibration stats
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is required for training")

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
        # PROBABILITY CALIBRATION (using validation set)
        # =====================================================
        if calibrate and X_val_filtered is not None and SKLEARN_CALIBRATION_AVAILABLE:
            if verbose:
                print("  Calibrating probabilities (Isotonic Regression)...")

            # Get raw probabilities on validation set (use ensemble average if applicable)
            raw_probs = self._get_raw_probs(X_val_filtered)

            # Compute pre-calibration ECE
            pre_cal_ece = compute_expected_calibration_error(
                y_success_val.values, raw_probs
            )

            # Fit isotonic calibrator
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(raw_probs, y_success_val.values)

            # Compute post-calibration ECE
            calibrated_probs = self.calibrator.predict(raw_probs)
            post_cal_ece = compute_expected_calibration_error(
                y_success_val.values, calibrated_probs
            )

            self.calibration_stats = {
                'pre_calibration_ece': pre_cal_ece['ece'],
                'post_calibration_ece': post_cal_ece['ece'],
                'ece_improvement': pre_cal_ece['ece'] - post_cal_ece['ece'],
                'pre_calibration_details': pre_cal_ece,
                'post_calibration_details': post_cal_ece,
            }

            if verbose:
                print(f"    Pre-calibration ECE:  {pre_cal_ece['ece']:.4f}")
                print(f"    Post-calibration ECE: {post_cal_ece['ece']:.4f}")
                print(f"    ECE Improvement:      {pre_cal_ece['ece'] - post_cal_ece['ece']:.4f}")

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

            self.tier_classifier = lgb.LGBMClassifier(**tier_params)

            tier_eval_set = [(X_tier_train, y_tier_train_clean)]
            tier_callbacks = []

            if y_tier_val is not None and y_tier_val.notna().sum() > 50:
                tier_mask_val = y_tier_val.notna()
                X_tier_val = X_val_filtered[tier_mask_val]
                y_tier_val_clean = y_tier_val[tier_mask_val].astype(int)
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

        # Train regressor (on successful trades only for better R:R prediction)
        success_mask = y_success == 1
        if success_mask.sum() > 100:
            reg_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'n_estimators': self.config.n_estimators // 2,
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

            self.regressor = lgb.LGBMRegressor(**reg_params)
            self.regressor.fit(
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

    def _get_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """Get binary predictions, using ensemble average if applicable."""
        probs = self._get_raw_probs(X)
        return (probs >= 0.5).astype(int)

    def _precision(self, pred, true):
        """Calculate precision"""
        pred_positive = pred == 1
        if pred_positive.sum() == 0:
            return 0.0
        return (pred[pred_positive] == true[pred_positive]).mean()

    def predict(self, X: pd.DataFrame, use_calibration: bool = False) -> Dict[str, np.ndarray]:
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
            - expected_rr: Expected reward:risk ratio (if regressor trained)
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
            result['bounce_prob'] = self.calibrator.predict(raw_prob)
        else:
            result['bounce_prob'] = raw_prob

        if self.regressor is not None:
            result['expected_rr'] = self.regressor.predict(X_filtered)

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
