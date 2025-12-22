"""
Hyperparameter optimization and threshold tuning for TrendFollower.

This module provides:
1. LightGBM hyperparameter tuning using Optuna (Optimizing for Total PnL)
2. min_bounce_prob threshold optimization
3. Training diagnostics showing returns by probability bucket
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import json
from pathlib import Path

# Scikit-learn metrics
try:
    from sklearn.metrics import precision_score, accuracy_score, log_loss
    SKLEARN_METRICS_AVAILABLE = True
except ImportError:
    SKLEARN_METRICS_AVAILABLE = False

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not installed. Run: pip install optuna")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


@dataclass
class ThresholdAnalysis:
    """Results from threshold optimization."""
    threshold: float
    n_trades: int
    win_rate: float
    total_pnl: float
    avg_pnl_per_trade: float
    profit_factor: float
    avg_winner: float
    avg_loser: float


def analyze_threshold_performance(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    pnl_per_trade: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> List[ThresholdAnalysis]:
    """
    Analyze trading performance at different probability thresholds.
    """
    if thresholds is None:
        thresholds = [0.40, 0.45, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65, 0.70]

    results = []

    for thresh in thresholds:
        mask = y_prob >= thresh
        n_trades = mask.sum()

        if n_trades == 0:
            results.append(ThresholdAnalysis(
                threshold=thresh,
                n_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                avg_pnl_per_trade=0.0,
                profit_factor=0.0,
                avg_winner=0.0,
                avg_loser=0.0,
            ))
            continue

        wins = y_true[mask] == 1
        losses = y_true[mask] == 0
        win_rate = wins.mean()

        pnl_selected = pnl_per_trade[mask]
        total_pnl = pnl_selected.sum()
        avg_pnl = pnl_selected.mean()

        winners = pnl_selected[pnl_selected > 0]
        losers = pnl_selected[pnl_selected < 0]

        avg_winner = winners.mean() if len(winners) > 0 else 0.0
        avg_loser = losers.mean() if len(losers) > 0 else 0.0

        gross_profit = winners.sum() if len(winners) > 0 else 0.0
        gross_loss = abs(losers.sum()) if len(losers) > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        results.append(ThresholdAnalysis(
            threshold=thresh,
            n_trades=int(n_trades),
            win_rate=float(win_rate),
            total_pnl=float(total_pnl),
            avg_pnl_per_trade=float(avg_pnl),
            profit_factor=float(profit_factor),
            avg_winner=float(avg_winner),
            avg_loser=float(avg_loser),
        ))

    return results


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    pnl_per_trade: np.ndarray,
    min_trades: int = 20,
    optimize_for: str = 'total_pnl',
) -> Tuple[float, List[ThresholdAnalysis]]:
    """
    Find the optimal probability threshold for trading.
    """
    # Fine-grained search
    thresholds = np.arange(0.40, 0.75, 0.01).tolist()
    results = analyze_threshold_performance(y_true, y_prob, pnl_per_trade, thresholds)

    # Filter by minimum trades
    valid_results = [r for r in results if r.n_trades >= min_trades]

    if not valid_results:
        # Fallback: use result with most trades
        valid_results = sorted(results, key=lambda x: x.n_trades, reverse=True)[:1]

    # Find optimal
    if optimize_for == 'total_pnl':
        best = max(valid_results, key=lambda x: x.total_pnl)
    elif optimize_for == 'profit_factor':
        best = max(valid_results, key=lambda x: x.profit_factor if x.profit_factor != float('inf') else 0)
    elif optimize_for == 'win_rate':
        best = max(valid_results, key=lambda x: x.win_rate)
    else:
        best = max(valid_results, key=lambda x: x.total_pnl)

    return best.threshold, results


def print_threshold_analysis(results: List[ThresholdAnalysis], optimal_threshold: float = None):
    """Print a formatted table of threshold analysis results."""
    print("\n" + "=" * 90)
    print("PROBABILITY THRESHOLD ANALYSIS")
    print("=" * 90)
    print(f"{'Threshold':>10} {'Trades':>8} {'Win Rate':>10} {'Total PnL':>12} {'Avg PnL':>10} {'PF':>8} {'Avg Win':>10} {'Avg Loss':>10}")
    print("-" * 90)

    for r in results:
        if r.n_trades == 0:
            continue

        marker = " <-- OPTIMAL" if optimal_threshold and abs(r.threshold - optimal_threshold) < 0.005 else ""
        pf_str = f"{r.profit_factor:.2f}" if r.profit_factor != float('inf') else "inf"

        print(f"{r.threshold:>10.2f} {r.n_trades:>8} {r.win_rate:>9.1%} {r.total_pnl:>12.2f} "
              f"{r.avg_pnl_per_trade:>10.2f} {pf_str:>8} {r.avg_winner:>10.2f} {r.avg_loser:>10.2f}{marker}")

    print("=" * 90)


def analyze_probability_buckets(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    pnl_per_trade: np.ndarray,
    n_buckets: int = 10,
) -> pd.DataFrame:
    """
    Analyze performance by probability bucket.
    """
    # Create buckets
    bucket_edges = np.linspace(0, 1, n_buckets + 1)
    bucket_labels = [f"{bucket_edges[i]:.1f}-{bucket_edges[i+1]:.1f}" for i in range(n_buckets)]

    # Assign each prediction to a bucket
    bucket_idx = np.digitize(y_prob, bucket_edges[1:-1])

    results = []
    for i in range(n_buckets):
        mask = bucket_idx == i
        n_samples = mask.sum()

        if n_samples == 0:
            results.append({
                'bucket': bucket_labels[i],
                'n_samples': 0,
                'actual_win_rate': np.nan,
                'avg_prob': np.nan,
                'total_pnl': 0.0,
                'avg_pnl': np.nan,
            })
            continue

        actual_win_rate = y_true[mask].mean()
        avg_prob = y_prob[mask].mean()
        bucket_pnl = pnl_per_trade[mask]

        results.append({
            'bucket': bucket_labels[i],
            'n_samples': int(n_samples),
            'actual_win_rate': float(actual_win_rate),
            'avg_prob': float(avg_prob),
            'total_pnl': float(bucket_pnl.sum()),
            'avg_pnl': float(bucket_pnl.mean()),
        })

    return pd.DataFrame(results)


def print_probability_bucket_analysis(bucket_df: pd.DataFrame):
    """Print probability bucket analysis."""
    print("\n" + "=" * 85)
    print("PROBABILITY BUCKET ANALYSIS (Model Calibration Check)")
    print("=" * 85)
    print(f"{'Bucket':>12} {'Samples':>10} {'Avg Prob':>10} {'Actual WR':>12} {'Calibration':>12} {'Total PnL':>12}")
    print("-" * 85)

    for _, row in bucket_df.iterrows():
        if row['n_samples'] == 0:
            continue

        if pd.notna(row['actual_win_rate']) and pd.notna(row['avg_prob']):
            cal_error = row['actual_win_rate'] - row['avg_prob']
            cal_str = f"{cal_error:+.1%}"
            if abs(cal_error) > 0.1:
                cal_str += " (!)"  # Flag large calibration errors
        else:
            cal_str = "N/A"

        print(f"{row['bucket']:>12} {row['n_samples']:>10} {row['avg_prob']:>10.1%} "
              f"{row['actual_win_rate']:>11.1%} {cal_str:>12} {row['total_pnl']:>12.2f}")

    print("=" * 85)
    print("Calibration: (Actual Win Rate - Predicted Prob). Positive = underestimates wins.")


class LightGBMTuner:
    """
    Hyperparameter tuner for LightGBM using Optuna.
    Now optimized for TOTAL PnL (Risk Adjusted) rather than just AUC/Accuracy.
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        task: str = 'binary',  # 'binary' or 'multiclass'
        n_classes: int = 2,
    ):
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for hyperparameter tuning. Run: pip install optuna")

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.task = task
        self.n_classes = n_classes
        self.best_params = None
        self.study = None

    def objective(self, trial: 'optuna.Trial') -> float:
        """Optuna objective function."""
        # ... (Keep your existing params dictionary exactly as is) ...
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 16, 128),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 150),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 1.0),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 5.0),
            'verbose': -1,
            'force_row_wise': True,
            'random_state': 42,
        }

        if self.task == 'binary':
            params['objective'] = 'binary'
            params['metric'] = 'binary_logloss'
            model = lgb.LGBMClassifier(**params)
        else:
            params['objective'] = 'multiclass'
            params['metric'] = 'multi_logloss'
            params['num_class'] = self.n_classes
            model = lgb.LGBMClassifier(**params)

        # Train with early stopping
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )

        # Evaluate on validation set
        if self.task == 'binary':
            val_prob = model.predict_proba(self.X_val)[:, 1]
            
            # --- DEBUGGING INFO ---
            max_prob = val_prob.max()
            mean_prob = val_prob.mean()
            # ----------------------

            # --- DYNAMIC THRESHOLD STRATEGY ---
            # 1. Try a hard threshold first (e.g., 0.50)
            threshold = 0.50
            preds = (val_prob > threshold).astype(int)
            n_trades = preds.sum()

            # 2. If 0 trades, force the top 1% (or at least 5 trades) to be evaluated
            # This prevents the "50.0" score loop by forcing the model to show its best ideas
            if n_trades < 5:
                # Find the threshold that gives us the top 2% of predictions
                threshold = np.percentile(val_prob, 98) 
                preds = (val_prob > threshold).astype(int)
                n_trades = preds.sum()

            # Convert series to numpy if needed
            y_val_np = self.y_val.to_numpy() if hasattr(self.y_val, 'to_numpy') else self.y_val
            
            tp = ((preds == 1) & (y_val_np == 1)).sum()
            fp = ((preds == 1) & (y_val_np == 0)).sum()
            
            total_pnl = (tp * 1.5) - (fp * 1.0)
            
            # Print status to help you debug live
            print(f"    [Debug] MaxProb: {max_prob:.3f} | Trades: {n_trades} | PnL: {total_pnl:.2f}")

            if n_trades == 0:
                score = 50.0 # Still 0 trades? Impossible with dynamic threshold, but just in case.
            else:
                score = -total_pnl 

        else:
            val_pred = model.predict_proba(self.X_val)
            score = log_loss(self.y_val, val_pred)

        return score

    def tune(
        self,
        n_trials: int = 50,
        timeout: int = None,
        show_progress: bool = True,
    ) -> Dict:
        """
        Run hyperparameter optimization.
        """
        sampler = TPESampler(seed=42)

        self.study = optuna.create_study(
            direction='minimize',
            sampler=sampler,
        )

        if not show_progress:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        self.study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress,
        )

        self.best_params = self.study.best_params

        return {
            'best_params': self.best_params,
            'best_value': self.study.best_value,
            'n_trials': len(self.study.trials),
        }

    def get_best_params_for_config(self) -> Dict:
        """Get best params formatted for ModelConfig."""
        if self.best_params is None:
            raise ValueError("Must run tune() first")

        return {
            'n_estimators': self.best_params['n_estimators'],
            'max_depth': self.best_params['max_depth'],
            'learning_rate': self.best_params['learning_rate'],
            'num_leaves': self.best_params['num_leaves'],
            'feature_fraction': self.best_params['feature_fraction'],
            'bagging_fraction': self.best_params['bagging_fraction'],
            'bagging_freq': self.best_params['bagging_freq'],
            'min_child_samples': self.best_params['min_child_samples'],
            'lambda_l1': self.best_params['lambda_l1'],
            'lambda_l2': self.best_params['lambda_l2'],
            'min_gain_to_split': self.best_params['min_gain_to_split'],
            'scale_pos_weight': self.best_params['scale_pos_weight'], # Added this
        }

    def print_results(self):
        """Print tuning results."""
        if self.study is None:
            print("No tuning results available. Run tune() first.")
            return

        print("\n" + "=" * 60)
        print("HYPERPARAMETER TUNING RESULTS (Optimized for Total PnL)")
        print("=" * 60)
        
        # Invert score to show actual PnL
        best_pnl = -self.study.best_value
        print(f"Best Validation PnL (Estimated): {best_pnl:.2f} R")
        print(f"Trials completed: {len(self.study.trials)}")
        
        print("\nBest parameters:")
        for key, value in self.best_params.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        print("=" * 60)


def estimate_pnl_from_labels(
    y_true: np.ndarray,
    target_rr: float = 1.5,
    stop_loss_value: float = 1.0,
) -> np.ndarray:
    """
    Estimate P&L per trade from binary labels.
    """
    pnl = np.where(y_true == 1, target_rr * stop_loss_value, -stop_loss_value)
    return pnl


def run_full_optimization(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    target_rr: float = 1.5,
    tune_hyperparams: bool = True,
    n_trials: int = 50,
    verbose: bool = True,
) -> Dict:
    """
    Run complete optimization: hyperparameters + threshold.
    """
    results = {}

    # Step 1: Hyperparameter tuning
    if tune_hyperparams and OPTUNA_AVAILABLE:
        if verbose:
            print("\n[1/3] Tuning LightGBM hyperparameters (PnL Objective)...")

        tuner = LightGBMTuner(X_train, y_train, X_val, y_val, task='binary')
        tune_results = tuner.tune(n_trials=n_trials, show_progress=verbose)

        results['best_lgb_params'] = tuner.get_best_params_for_config()
        results['tuning_score'] = tune_results['best_value']

        if verbose:
            tuner.print_results()

        # Train model with best params for threshold analysis
        best_params = tuner.best_params.copy()
        best_params['objective'] = 'binary'
        best_params['metric'] = 'binary_logloss'
        best_params['verbose'] = -1

        model = lgb.LGBMClassifier(**best_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )
        y_prob = model.predict_proba(X_val)[:, 1]
    else:
        if verbose:
            print("\n[1/3] Skipping hyperparameter tuning (using defaults or Optuna not available)")

        # Train with default params
        model = lgb.LGBMClassifier(
            objective='binary',
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            verbose=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )
        y_prob = model.predict_proba(X_val)[:, 1]
        results['best_lgb_params'] = None

    # Step 2: Threshold optimization
    if verbose:
        print("\n[2/3] Optimizing probability threshold...")

    pnl_per_trade = estimate_pnl_from_labels(y_val, target_rr=target_rr)
    optimal_thresh, thresh_results = find_optimal_threshold(
        y_val, y_prob, pnl_per_trade,
        min_trades=20,
        optimize_for='total_pnl'
    )

    results['best_threshold'] = optimal_thresh
    results['threshold_analysis'] = thresh_results

    if verbose:
        print_threshold_analysis(thresh_results, optimal_thresh)

    # Step 3: Probability bucket analysis
    if verbose:
        print("\n[3/3] Analyzing probability buckets...")

    bucket_df = analyze_probability_buckets(y_val, y_prob, pnl_per_trade)
    results['bucket_analysis'] = bucket_df

    if verbose:
        print_probability_bucket_analysis(bucket_df)

    return results


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 50

    X = pd.DataFrame(np.random.randn(n_samples, n_features), columns=[f'f{i}' for i in range(n_features)])
    y = (np.random.rand(n_samples) > 0.6).astype(int)  # ~40% win rate

    # Split
    split = int(0.7 * n_samples)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y[:split], y[split:]

    print("Testing hyperopt module with synthetic data...")
    results = run_full_optimization(
        X_train, y_train, X_val, y_val,
        target_rr=1.5,
        tune_hyperparams=OPTUNA_AVAILABLE,
        n_trials=10,  # Quick test
        verbose=True
    )

    print(f"\nOptimal threshold: {results['best_threshold']:.2f}")
