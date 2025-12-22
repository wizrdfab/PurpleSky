"""
Custom Parameter Optimizer for TrendFollower.

Optimizes LightGBM hyperparameters AND min_bounce_prob threshold
to maximize cumulative profit on the validation set.

No external dependencies (no Optuna) - uses grid search + random search hybrid.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import time
import random
from copy import deepcopy

import lightgbm as lgb


@dataclass
class OptimizationResult:
    """Results from a single optimization trial."""
    params: Dict[str, Any]
    min_bounce_prob: float
    n_trades: int
    win_rate: float
    total_pnl: float  # In R units (risk units)
    profit_factor: float
    sharpe_ratio: float = 0.0


@dataclass
class OptimizerConfig:
    """Configuration for the optimizer."""
    # Search space for LightGBM parameters
    n_estimators_range: Tuple[int, int] = (50, 500)
    max_depth_range: Tuple[int, int] = (3, 10)
    num_leaves_range: Tuple[int, int] = (16, 96)
    learning_rate_range: Tuple[float, float] = (0.01, 0.15)
    min_child_samples_range: Tuple[int, int] = (20, 100)
    feature_fraction_range: Tuple[float, float] = (0.5, 1.0)
    bagging_fraction_range: Tuple[float, float] = (0.5, 1.0)
    lambda_l1_range: Tuple[float, float] = (0.0, 5.0)
    lambda_l2_range: Tuple[float, float] = (0.0, 5.0)
    min_gain_to_split_range: Tuple[float, float] = (0.0, 0.5)

    # Search space for min_bounce_prob
    min_bounce_prob_range: Tuple[float, float] = (0.40, 0.70)
    min_bounce_prob_step: float = 0.02

    # Optimization settings
    n_random_trials: int = 30  # Random search trials
    min_trades_required: int = 15  # Minimum trades to consider valid
    target_rr: float = 1.5  # Risk:Reward ratio for P&L calculation

    # Early stopping
    early_stop_no_improvement: int = 10  # Stop if no improvement for N trials


class TrendFollowerOptimizer:
    """
    Custom optimizer for TrendFollower model.

    Optimizes both LightGBM hyperparameters and min_bounce_prob threshold
    to maximize cumulative profit on validation data.
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        config: Optional[OptimizerConfig] = None,
        seed: int = 42,
    ):
        """
        Initialize the optimizer.

        Args:
            X_train: Training features
            y_train: Training labels (binary: 1=success/TP, 0=failure/SL)
            X_val: Validation features
            y_val: Validation labels
            config: Optimizer configuration
            seed: Random seed for reproducibility
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.config = config or OptimizerConfig()
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)

        self.results: List[OptimizationResult] = []
        self.best_result: Optional[OptimizationResult] = None

    def _sample_params(self) -> Dict[str, Any]:
        """Sample random hyperparameters from search space."""
        cfg = self.config
        return {
            'n_estimators': random.randint(*cfg.n_estimators_range),
            'max_depth': random.randint(*cfg.max_depth_range),
            'num_leaves': random.randint(*cfg.num_leaves_range),
            'learning_rate': random.uniform(*cfg.learning_rate_range),
            'min_child_samples': random.randint(*cfg.min_child_samples_range),
            'feature_fraction': random.uniform(*cfg.feature_fraction_range),
            'bagging_fraction': random.uniform(*cfg.bagging_fraction_range),
            'bagging_freq': random.randint(1, 7),
            'lambda_l1': random.uniform(*cfg.lambda_l1_range),
            'lambda_l2': random.uniform(*cfg.lambda_l2_range),
            'min_gain_to_split': random.uniform(*cfg.min_gain_to_split_range),
        }

    def _train_model(self, params: Dict[str, Any]) -> lgb.LGBMClassifier:
        """Train a LightGBM model with given parameters."""
        model_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbose': -1,
            'force_row_wise': True,
            'random_state': self.seed,
            **params
        }

        model = lgb.LGBMClassifier(**model_params)

        # Train with early stopping
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )

        return model

    def _evaluate_threshold(
        self,
        val_probs: np.ndarray,
        threshold: float,
    ) -> Dict[str, float]:
        """
        Evaluate performance at a given probability threshold.

        Returns metrics in R-units (risk units) assuming:
        - Win = +target_rr R
        - Loss = -1 R
        """
        mask = val_probs >= threshold
        n_trades = mask.sum()

        if n_trades < self.config.min_trades_required:
            return {
                'n_trades': n_trades,
                'win_rate': 0.0,
                'total_pnl': -999999.0,  # Invalid
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
            }

        y_pred = mask.astype(int)
        y_actual = self.y_val[mask]

        # Calculate wins and losses
        n_wins = (y_actual == 1).sum()
        n_losses = (y_actual == 0).sum()

        win_rate = n_wins / n_trades if n_trades > 0 else 0.0

        # P&L in R-units
        win_pnl = n_wins * self.config.target_rr
        loss_pnl = n_losses * 1.0
        total_pnl = win_pnl - loss_pnl

        profit_factor = win_pnl / loss_pnl if loss_pnl > 0 else float('inf')

        # Sharpe-like ratio (P&L per trade normalized by std)
        pnl_per_trade = []
        for outcome in y_actual:
            if outcome == 1:
                pnl_per_trade.append(self.config.target_rr)
            else:
                pnl_per_trade.append(-1.0)

        if len(pnl_per_trade) > 1:
            sharpe = np.mean(pnl_per_trade) / (np.std(pnl_per_trade) + 1e-8)
        else:
            sharpe = 0.0

        return {
            'n_trades': n_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
        }

    def _find_best_threshold(
        self,
        val_probs: np.ndarray,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Find the best min_bounce_prob threshold for given predictions.

        Returns the threshold that maximizes total P&L.
        """
        cfg = self.config
        thresholds = np.arange(
            cfg.min_bounce_prob_range[0],
            cfg.min_bounce_prob_range[1] + cfg.min_bounce_prob_step,
            cfg.min_bounce_prob_step
        )

        best_threshold = 0.50
        best_metrics = None
        best_pnl = -float('inf')

        for thresh in thresholds:
            metrics = self._evaluate_threshold(val_probs, thresh)

            if metrics['total_pnl'] > best_pnl and metrics['n_trades'] >= cfg.min_trades_required:
                best_pnl = metrics['total_pnl']
                best_threshold = thresh
                best_metrics = metrics

        if best_metrics is None:
            # No valid threshold found, use default
            best_metrics = self._evaluate_threshold(val_probs, 0.50)

        return best_threshold, best_metrics

    def _evaluate_params(self, params: Dict[str, Any]) -> OptimizationResult:
        """
        Evaluate a set of parameters by training a model and finding optimal threshold.
        """
        # Train model
        model = self._train_model(params)

        # Get validation probabilities
        val_probs = model.predict_proba(self.X_val)[:, 1]

        # Find best threshold
        best_thresh, metrics = self._find_best_threshold(val_probs)

        return OptimizationResult(
            params=params,
            min_bounce_prob=best_thresh,
            n_trades=metrics['n_trades'],
            win_rate=metrics['win_rate'],
            total_pnl=metrics['total_pnl'],
            profit_factor=metrics['profit_factor'],
            sharpe_ratio=metrics['sharpe_ratio'],
        )

    def optimize(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run the optimization process.

        Args:
            verbose: Print detailed progress information

        Returns:
            Dictionary with best parameters and results
        """
        start_time = time.time()

        if verbose:
            print("\n" + "=" * 80)
            print("CUSTOM PARAMETER OPTIMIZER")
            print("=" * 80)
            print(f"Training samples: {len(self.X_train):,}")
            print(f"Validation samples: {len(self.X_val):,}")
            print(f"Validation win rate (baseline): {self.y_val.mean():.1%}")
            print(f"Random trials: {self.config.n_random_trials}")
            print(f"Min trades required: {self.config.min_trades_required}")
            print(f"Target R:R: {self.config.target_rr}")
            print("=" * 80)

        self.results = []
        self.best_result = None
        no_improvement_count = 0

        # Progress tracking
        trial_results = []

        for trial in range(self.config.n_random_trials):
            # Sample parameters
            params = self._sample_params()

            try:
                result = self._evaluate_params(params)
                self.results.append(result)

                # Track if this is the best result
                is_best = False
                if self.best_result is None or result.total_pnl > self.best_result.total_pnl:
                    self.best_result = result
                    is_best = True
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                # Progress output
                if verbose:
                    status = " ** NEW BEST **" if is_best else ""
                    print(f"\n  Trial {trial + 1}/{self.config.n_random_trials}{status}")
                    print(f"    min_bounce_prob: {result.min_bounce_prob:.2f}")
                    print(f"    Trades: {result.n_trades}, Win Rate: {result.win_rate:.1%}")
                    print(f"    Total P&L: {result.total_pnl:+.1f} R, PF: {result.profit_factor:.2f}")

                    # Show key params
                    print(f"    Params: lr={params['learning_rate']:.3f}, "
                          f"depth={params['max_depth']}, leaves={params['num_leaves']}, "
                          f"L1={params['lambda_l1']:.2f}, L2={params['lambda_l2']:.2f}")

                trial_results.append({
                    'trial': trial + 1,
                    'pnl': result.total_pnl,
                    'threshold': result.min_bounce_prob,
                    'trades': result.n_trades,
                    'win_rate': result.win_rate,
                })

            except Exception as e:
                if verbose:
                    print(f"\n  Trial {trial + 1} FAILED: {e}")
                continue

            # Early stopping
            if no_improvement_count >= self.config.early_stop_no_improvement:
                if verbose:
                    print(f"\n  Early stopping: No improvement for {no_improvement_count} trials")
                break

        elapsed = time.time() - start_time

        # Print final summary
        if verbose and self.best_result:
            self._print_summary(elapsed)
            self._print_threshold_distribution()

        return {
            'best_params': self.best_result.params if self.best_result else {},
            'best_min_bounce_prob': self.best_result.min_bounce_prob if self.best_result else 0.50,
            'best_total_pnl': self.best_result.total_pnl if self.best_result else 0.0,
            'best_win_rate': self.best_result.win_rate if self.best_result else 0.0,
            'best_n_trades': self.best_result.n_trades if self.best_result else 0,
            'best_profit_factor': self.best_result.profit_factor if self.best_result else 0.0,
            'trials_completed': len(self.results),
            'elapsed_seconds': elapsed,
        }

    def _print_summary(self, elapsed: float):
        """Print optimization summary."""
        r = self.best_result

        print("\n" + "=" * 80)
        print("OPTIMIZATION COMPLETE")
        print("=" * 80)
        print(f"Time elapsed: {elapsed:.1f}s")
        print(f"Trials completed: {len(self.results)}")

        print("\n" + "-" * 40)
        print("BEST RESULT")
        print("-" * 40)
        print(f"  min_bounce_prob: {r.min_bounce_prob:.2f}")
        print(f"  Total P&L:       {r.total_pnl:+.1f} R")
        print(f"  Trades:          {r.n_trades}")
        print(f"  Win Rate:        {r.win_rate:.1%}")
        print(f"  Profit Factor:   {r.profit_factor:.2f}")
        print(f"  Sharpe Ratio:    {r.sharpe_ratio:.2f}")

        print("\n" + "-" * 40)
        print("BEST LIGHTGBM PARAMETERS")
        print("-" * 40)
        for key, value in r.params.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print("=" * 80)

    def _print_threshold_distribution(self):
        """Print detailed P&L distribution by threshold."""
        if not self.best_result:
            return

        # Retrain best model to get probabilities
        model = self._train_model(self.best_result.params)
        val_probs = model.predict_proba(self.X_val)[:, 1]

        print("\n" + "=" * 80)
        print("P&L DISTRIBUTION BY min_bounce_prob THRESHOLD")
        print("=" * 80)
        print(f"{'Threshold':>10} {'Trades':>8} {'Win Rate':>10} {'Total P&L':>12} {'PF':>8} {'Avg P&L':>10}")
        print("-" * 80)

        cfg = self.config
        thresholds = np.arange(0.40, 0.72, 0.02)

        for thresh in thresholds:
            # Direct calculation - avoid the -999999 sentinel value
            mask = val_probs >= thresh
            n_trades = mask.sum()

            if n_trades >= self.config.min_trades_required:
                y_actual = self.y_val[mask]
                n_wins = (y_actual == 1).sum()
                n_losses = (y_actual == 0).sum()
                win_rate = n_wins / n_trades if n_trades > 0 else 0.0

                win_pnl = n_wins * self.config.target_rr
                loss_pnl = n_losses * 1.0
                total_pnl = win_pnl - loss_pnl
                profit_factor = win_pnl / loss_pnl if loss_pnl > 0 else float('inf')
                avg_pnl = total_pnl / n_trades if n_trades > 0 else 0

                marker = " <-- OPTIMAL" if abs(thresh - self.best_result.min_bounce_prob) < 0.01 else ""

                print(f"{thresh:>10.2f} {n_trades:>8} {win_rate:>10.1%} "
                      f"{total_pnl:>+12.1f} R {profit_factor:>8.2f} "
                      f"{avg_pnl:>+10.2f} R{marker}")
            elif n_trades >= 3:
                # Show partial info for small samples
                y_actual = self.y_val[mask]
                n_wins = (y_actual == 1).sum()
                win_rate = n_wins / n_trades if n_trades > 0 else 0.0
                print(f"{thresh:>10.2f} {n_trades:>8} {win_rate:>10.1%} {'(too few trades)':>12}")
            else:
                print(f"{thresh:>10.2f} {n_trades:>8} {'(insufficient)':>10}")

        print("=" * 80)

        # Probability distribution
        print("\n" + "=" * 80)
        print("RAW PROBABILITY DISTRIBUTION (Validation Set)")
        print("=" * 80)

        percentiles = [10, 25, 50, 75, 90, 95, 99]
        print(f"{'Percentile':>12} {'Probability':>12}")
        print("-" * 30)
        for p in percentiles:
            val = np.percentile(val_probs, p)
            print(f"{p:>10}th {val:>12.3f}")

        print(f"\n  Min:  {val_probs.min():.3f}")
        print(f"  Max:  {val_probs.max():.3f}")
        print(f"  Mean: {val_probs.mean():.3f}")
        print(f"  Std:  {val_probs.std():.3f}")
        print("=" * 80)

    def get_best_config_params(self) -> Dict[str, Any]:
        """Get best parameters formatted for config.ModelConfig."""
        if not self.best_result:
            return {}

        p = self.best_result.params
        return {
            'n_estimators': p['n_estimators'],
            'max_depth': p['max_depth'],
            'num_leaves': p['num_leaves'],
            'learning_rate': p['learning_rate'],
            'min_child_samples': p['min_child_samples'],
            'feature_fraction': p['feature_fraction'],
            'bagging_fraction': p['bagging_fraction'],
            'bagging_freq': p['bagging_freq'],
            'lambdaa_ele1': p['lambda_l1'],
            'lambdaa_ele2': p['lambda_l2'],
            'min_gain_to_split': p['min_gain_to_split'],
        }


def run_optimization(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    n_trials: int = 30,
    target_rr: float = 1.5,
    min_trades: int = 15,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function to run optimization.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        n_trials: Number of random search trials
        target_rr: Target risk:reward ratio
        min_trades: Minimum trades to consider valid result
        verbose: Print progress

    Returns:
        Dictionary with optimization results
    """
    config = OptimizerConfig(
        n_random_trials=n_trials,
        target_rr=target_rr,
        min_trades_required=min_trades,
    )

    optimizer = TrendFollowerOptimizer(
        X_train, y_train,
        X_val, y_val,
        config=config,
    )

    return optimizer.optimize(verbose=verbose)


if __name__ == "__main__":
    print("Optimizer module loaded successfully")
    print("Usage: from optimizer import run_optimization")
