"""
Feature noise filtering utilities.

This module provides stronger alternatives to the simple "compare against a random
noise column" approach, while still keeping the workflow lightweight enough for
iterative research.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:  # pragma: no cover
    LIGHTGBM_AVAILABLE = False


def _timewise_subsample(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    max_rows: int,
) -> Tuple[pd.DataFrame, pd.Series, bool]:
    """
    Deterministic timewise subsample for very large training sets.
    Keeps chronological structure (no random shuffle) and reduces compute.
    """
    max_rows = int(max_rows)
    if max_rows <= 0 or len(X) <= max_rows:
        return X, y, False

    stride = max(1, len(X) // max_rows)
    X_sub = X.iloc[::stride].head(max_rows)
    y_sub = y.iloc[::stride].head(max_rows)
    return X_sub, y_sub, True


def _train_lgbm_importance(
    X: pd.DataFrame,
    y: pd.Series,
    config,
    *,
    seed: int,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
) -> Dict[str, float]:
    if not LIGHTGBM_AVAILABLE:
        return {c: 0.0 for c in X.columns}

    y_unique = pd.Series(y).dropna().unique()
    if len(y_unique) < 2:
        # Can't train a binary classifier if there's only one class.
        return {c: 0.0 for c in X.columns}

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'n_estimators': int(n_estimators),
        'max_depth': int(getattr(config, 'max_depth', -1)),
        'learning_rate': float(learning_rate),
        'num_leaves': int(getattr(config, 'num_leaves', 31)),
        'feature_fraction': float(getattr(config, 'feature_fraction', 0.8)),
        'bagging_fraction': float(getattr(config, 'bagging_fraction', 0.8)),
        'bagging_freq': int(getattr(config, 'bagging_freq', 5)),
        'min_child_samples': int(getattr(config, 'min_child_samples', 20)),
        'reg_alpha': float(getattr(config, 'lambdaa_ele1', 0.0)),
        'reg_lambda': float(getattr(config, 'lambdaa_ele2', 0.0)),
        'verbose': -1,
        'force_row_wise': True,
        'random_state': int(seed),
        'deterministic': True,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(X, y)

    try:
        booster = model.booster_
        importance = booster.feature_importance(importance_type='gain')
        names = booster.feature_name()
        return {str(n): float(v) for n, v in zip(names, importance)}
    except Exception:
        return {str(n): float(v) for n, v in zip(X.columns, model.feature_importances_)}


def filter_features_by_null_importance(
    X_train: pd.DataFrame,
    X_val: Optional[pd.DataFrame],
    y_train: pd.Series,
    config,
    *,
    n_shuffles: int = 20,
    null_quantile: float = 75.0,
    min_features_to_keep: int = 50,
    max_rows: int = 20000,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], List[str]]:
    """
    Null-importance feature selection.

    Idea:
      - Train a model on TRUE labels and record each feature's importance.
      - Train N models on SHUFFLED labels and record importance distributions.
      - Keep features whose true importance beats a chosen percentile of the null distribution.

    This is usually more robust than a single "random noise column" threshold,
    especially when you have lots of data and many correlated indicators.
    """
    if not LIGHTGBM_AVAILABLE:
        return X_train, X_val, []

    n_shuffles = max(1, int(n_shuffles))
    null_quantile = float(null_quantile)
    min_features_to_keep = max(1, int(min_features_to_keep))
    max_rows = int(max_rows)

    if verbose:
        print("\n  Null-Importance Feature Selection:")
        print(f"    shuffles={n_shuffles}, null_quantile={null_quantile:.1f}, max_rows={max_rows:,}")

    X_fs, y_fs, used_subsample = _timewise_subsample(X_train, y_train, max_rows=max_rows)
    if used_subsample and verbose:
        print(f"    Using timewise subsample: {len(X_fs):,} / {len(X_train):,} rows")

    # Real importance on true labels.
    real_imp = _train_lgbm_importance(X_fs, y_fs, config, seed=42)

    # Null importances (label shuffled).
    null_imps: Dict[str, List[float]] = {c: [] for c in X_fs.columns}
    for i in range(n_shuffles):
        seed = 1000 + i * 17
        y_shuf = y_fs.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        X_shuf = X_fs.reset_index(drop=True)

        imp = _train_lgbm_importance(X_shuf, y_shuf, config, seed=seed)
        for c in X_fs.columns:
            null_imps[c].append(float(imp.get(c, 0.0)))

        if verbose and (i + 1) % max(1, n_shuffles // 5) == 0:
            print(f"    Shuffles: {i + 1}/{n_shuffles}")

    # Decide keep/remove.
    scores: Dict[str, float] = {}
    for c in X_train.columns:
        real = float(real_imp.get(c, 0.0))
        dist = np.asarray(null_imps.get(c, [0.0]), dtype=np.float64)
        thr = float(np.percentile(dist, null_quantile)) if dist.size else 0.0
        scores[c] = real - thr

    keep = [c for c, s in scores.items() if s > 0.0]
    removed = [c for c in X_train.columns if c not in keep]

    if len(keep) < min_features_to_keep:
        # Keep the strongest features by score to satisfy the minimum.
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        keep = [c for c, _ in ranked[:min_features_to_keep]]
        removed = [c for c in X_train.columns if c not in keep]
        if verbose:
            print(f"    Keeping top {min_features_to_keep} features to enforce minimum.")

    X_train_filtered = X_train[keep]
    X_val_filtered = X_val[keep] if X_val is not None else None

    if verbose:
        print(f"    Total features: {len(X_train.columns)}")
        print(f"    Features kept:  {len(keep)}")
        print(f"    Features removed: {len(removed)}")
        if removed:
            print(f"    Removed (first 20): {removed[:20]}")

    return X_train_filtered, X_val_filtered, removed

