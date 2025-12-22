"""
Hyperparameter Tuner for TrendFollower.
Optimizes LightGBM parameters and Bounce Probability thresholds.
"""
import pandas as pd
import numpy as np
import argparse
import itertools
from pathlib import Path
from config import DEFAULT_CONFIG
from data_loader import load_trades, preprocess_trades, create_multi_timeframe_bars
from feature_engine import calculate_multi_timeframe_features, get_feature_columns
from labels import create_training_dataset
from models import TrendFollowerModels
from trainer import time_series_split

def get_validation_stats(y_true, y_prob, threshold):
    """
    Calculates detailed statistics for a specific threshold.
    Returns: dict of stats
    """
    # Create binary predictions based on threshold
    y_pred = (y_prob >= threshold).astype(int)
    
    # "Support" is the number of samples predicted as Positive (1)
    # This answers "How many samples are these metrics derived from?"
    predicted_positives = y_pred.sum()
    
    if predicted_positives == 0:
        return {
            'threshold': threshold,
            'precision': 0.0,
            'trades_found': 0,
            'valid_sample': False
        }

    # True Positives: We predicted 1, and it was actually 1
    true_positives = ((y_pred == 1) & (y_true == 1)).sum()
    
    precision = true_positives / predicted_positives
    
    return {
        'threshold': threshold,
        'precision': precision,
        'trades_found': int(predicted_positives),
        'valid_sample': True
    }

def run_tuning(args):
    # 1. Load and Prepare Data (Same as trainer.py)
    print("Loading data for tuning...")
    config = DEFAULT_CONFIG
    config.data.data_dir = Path(args.data_dir)
    config.base_timeframe_idx = args.base_tf
    
    # Load
    trades = load_trades(config.data)
    trades = preprocess_trades(trades, config.data)
    bars = create_multi_timeframe_bars(trades, config.features.timeframes, config.features.timeframe_names, config.data)
    base_tf = config.features.timeframe_names[config.base_timeframe_idx]
    featured = calculate_multi_timeframe_features(bars, base_tf, config.features)
    
    # Label
    print("Generating labels...")
    labeled, feature_cols = create_training_dataset(featured, config.labels, config.features, base_tf)
    
    # Split
    print("Splitting data...")
    train_df, val_df, _ = time_series_split(labeled, 0.7, 0.15, 0.15)
    
    # Filter for Pullback entries only (Entry Model specific)
    mask_train = train_df['pullback_success'].notna()
    mask_val = val_df['pullback_success'].notna()
    
    X_train = train_df.loc[mask_train, feature_cols].fillna(0)
    y_train = train_df.loc[mask_train, 'pullback_success'].astype(int)
    y_rr_train = train_df.loc[mask_train, 'pullback_rr']
    
    X_val = val_df.loc[mask_val, feature_cols].fillna(0)
    y_val = val_df.loc[mask_val, 'pullback_success'].astype(int)
    
    print(f"\n--- DATASET STATISTICS ---")
    print(f"Total Validation Set Size: {len(val_df)} bars")
    print(f"Validation Pullback Opportunities: {len(X_val)} samples")
    print(f"(These are the samples metrics are derived from)")
    print("-" * 30)

    # 2. Define Hyperparameter Grid
    # Add/Remove parameters here to tune different things
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [20, 31, 50],
        'n_estimators': [100, 200, 500],
        'max_depth': [-1, 7, 10]
    }
    
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"\nStarting Grid Search over {len(combinations)} model configurations...")
    
    best_config = None
    best_score = -1.0
    best_stats = {}
    
    # 3. Tuning Loop
    for i, params in enumerate(combinations):
        print(f"\n[{i+1}/{len(combinations)}] Testing params: {params}")
        
        # Update Config with new params
        config.model.learning_rate = params['learning_rate']
        config.model.num_leaves = params['num_leaves']
        config.model.n_estimators = params['n_estimators']
        config.model.max_depth = params['max_depth']
        
        # Train Model
        models = TrendFollowerModels(config.model)
        
        # We assume raw probabilities (calibration=False) for tuning stability
        models.entry_model.train(
            X_train, y_train, y_rr_train,
            X_val, y_val, None, # Validation data for early stopping
            verbose=False,
            calibrate=False 
        )
        
        # Get Raw Predictions on Validation Set
        preds = models.entry_model.predict(X_val, use_calibration=False)
        probs = preds['bounce_prob_raw'] # Always use raw for tuning
        
        # Inner Loop: Tune Bounce Probability Threshold
        # Scan from 0.50 to 0.70
        for thresh in np.arange(0.50, 0.70, 0.01):
            stats = get_validation_stats(y_val, probs, thresh)
            
            if not stats['valid_sample']:
                continue
                
            # Filter: Ignore thresholds with too few trades (statistical noise)
            if stats['trades_found'] < 10:
                continue
                
            # Objective: Maximize Precision (Win Rate)
            # You could also optimize for Profit Factor if you included PnL data
            score = stats['precision']
            
            if score > best_score:
                best_score = score
                best_config = params
                best_stats = stats
                print(f"  New Best! Precision: {score:.1%} | Thresh: {thresh:.2f} | Trades: {stats['trades_found']}")

    # 4. Final Report
    print("\n" + "="*60)
    print("TUNING RESULTS")
    print("="*60)
    print("Best LightGBM Parameters:")
    for k, v in best_config.items():
        print(f"  {k}: {v}")
    
    print("\nBest Bounce Probability Threshold:")
    print(f"  {best_stats['threshold']:.2f} (Use this for --min-bounce-prob)")
    
    print("\nDetailed Validation Statistics (at best settings):")
    print(f"  Validation Set Pullbacks:  {len(X_val)} (Total potential entries)")
    print(f"  Trades Triggered:          {best_stats['trades_found']} (Samples derived from)")
    print(f"  Precision (Win Rate):      {best_stats['precision']:.1%} ({int(best_stats['precision']*best_stats['trades_found'])} Wins / {best_stats['trades_found'] - int(best_stats['precision']*best_stats['trades_found'])} Losses)")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--base-tf', type=int, default=1)
    args = parser.parse_args()
    
    run_tuning(args)
