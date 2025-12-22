"""
Hyperparameter Tuner (STRICT & OPTIMIZED)
1. Optimizes LightGBM (w/ Regularization).
2. Scans min_bounce_prob (Inner Loop).
3. Enforces Strict Backtest Logic (Matches backtest.py).
"""
import argparse
import optuna
import pandas as pd
import numpy as np
from pathlib import Path
import copy
import logging
import sys

# Import your existing modules
from config import DEFAULT_CONFIG
from models import EntryQualityModel
from data_loader import load_trades, preprocess_trades, create_multi_timeframe_bars
from feature_engine import calculate_multi_timeframe_features
from labels import create_training_dataset

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO, 
    format='%(message)s', 
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# STRICT FAST BACKTESTER
# -----------------------------------------------------------------------------
class FastBacktester:
    """
    Vectorized Backtester with Strict Logic.
    Matches backtest.py: 
    - Requires valid setup (ema_touch_detected).
    - Pessimistic Exit (SL hit before TP on same bar).
    """
    @staticmethod
    def run(df, prob_arr, min_bounce_prob, stop_loss_atr, take_profit_rr):
        # --- 1. STRICT FILTERING ---
        # Only take trades where:
        # A) Model Probability >= Threshold
        # B) A valid setup was detected (ema_touch_detected)
        
        if 'ema_touch_detected' in df.columns:
            setup_mask = df['ema_touch_detected'].values
        else:
            # Fallback/Warning if column missing (shouldn't happen with correct labels.py)
            # We default to False to avoid testing on garbage data
            setup_mask = np.zeros(len(df), dtype=bool) 

        # Signal = High Confidence AND Valid Setup
        signal_mask = (prob_arr >= min_bounce_prob) & (setup_mask)
        
        candidate_indices = np.flatnonzero(signal_mask)
        
        if len(candidate_indices) == 0:
            return 0, 0.0, 0.0

        # --- PREPARE DATA ---
        opens = df['open'].values
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        
        # Get ATR for dynamic stops
        # We try to grab the specific timeframe ATR, or fallback to any ATR column
        atr_col_name = f"{DEFAULT_CONFIG.features.timeframe_names[1]}_atr"
        if atr_col_name in df.columns:
            atrs = df[atr_col_name].values
        else:
            # Fallback: grab the first column containing 'atr'
            valid_cols = [c for c in df.columns if 'atr' in c]
            if valid_cols:
                atrs = df[valid_cols[0]].values
            else:
                return 0, 0.0, 0.0 # Cannot trade without ATR

        # Get Direction (Long vs Short)
        if 'ema_touch_direction' in df.columns:
            directions = df['ema_touch_direction'].values
        else:
            # Default to Long if missing
            directions = np.ones(len(df))

        trades_count = 0
        total_r = 0.0
        wins = 0
        last_exit_idx = -1
        n_bars = len(df)
        
        # --- 2. TRADE LOOP ---
        for idx in candidate_indices:
            # Skip if we are already in a trade
            if idx <= last_exit_idx: continue
            # Skip if end of data
            if idx >= n_bars - 1: break

            direction = directions[idx]
            entry_price = closes[idx]
            current_atr = atrs[idx]
            
            # --- RISK SETTINGS ---
            stop_dist = current_atr * stop_loss_atr
            target_dist = stop_dist * take_profit_rr
            
            if direction == 1: # LONG
                stop_price = entry_price - stop_dist
                target_price = entry_price + target_dist
            else: # SHORT (Assuming -1 direction)
                stop_price = entry_price + stop_dist
                target_price = entry_price - target_dist
            
            # --- FAST LOOKAHEAD (Exit Detection) ---
            look_ahead = 200 # Max bars to hold
            end_search = min(idx + look_ahead, n_bars)
            
            future_lows = lows[idx+1 : end_search]
            future_highs = highs[idx+1 : end_search]
            
            # Identify where price hits SL or TP
            if direction == 1: # LONG
                sl_hits = future_lows <= stop_price
                tp_hits = future_highs >= target_price
            else: # SHORT
                sl_hits = future_highs >= stop_price
                tp_hits = future_lows <= target_price
            
            # np.argmax returns the *first* index of True. 
            # If no True exists, it returns 0. We must check np.any() first.
            has_sl = np.any(sl_hits)
            has_tp = np.any(tp_hits)
            
            first_sl = np.argmax(sl_hits) if has_sl else 99999
            first_tp = np.argmax(tp_hits) if has_tp else 99999
            
            exit_pnl = 0.0
            
            if first_sl == 99999 and first_tp == 99999:
                # Timeout Exit
                exit_price = closes[end_search-1]
                exit_pnl = (exit_price - entry_price) * direction
                last_exit_idx = end_search
                
            elif first_sl <= first_tp:
                # LOSS (SL hit first OR on same bar) -> Pessimistic
                exit_pnl = (stop_price - entry_price) * direction
                last_exit_idx = idx + 1 + first_sl
            else:
                # WIN (TP hit strictly before SL)
                exit_pnl = (target_price - entry_price) * direction
                wins += 1
                last_exit_idx = idx + 1 + first_tp
            
            # Normalize to R-Multiple (Risk Units)
            # e.g. -1.0 for SL, +1.5 for TP
            r_multiple = exit_pnl / stop_dist
            total_r += r_multiple
            trades_count += 1

        win_rate = wins / trades_count if trades_count > 0 else 0.0
        return trades_count, win_rate, total_r

# -----------------------------------------------------------------------------
# OPTIMIZATION FLOW
# -----------------------------------------------------------------------------

def prepare_data(data_dir: str):
    print("--- Loading and Preparing Data ---")
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.data.data_dir = Path(data_dir)
    cfg.base_timeframe_idx = 1 # 5m
    
    # Load
    trades = load_trades(cfg.data, verbose=False)
    trades = preprocess_trades(trades, cfg.data)
    
    # Feature Engineering
    bars = create_multi_timeframe_bars(trades, cfg.features.timeframes, cfg.features.timeframe_names, cfg.data)
    base_tf = cfg.features.timeframe_names[cfg.base_timeframe_idx]
    featured = calculate_multi_timeframe_features(bars, base_tf, cfg.features)
    
    # Labeling
    labeled, feature_cols = create_training_dataset(featured, cfg.labels, cfg.features, base_tf)
    
    # Split Data
    n = len(labeled)
    train_size = int(n * 0.60)
    
    # Train set (Only valid outcomes for training)
    train_df = labeled.iloc[:train_size].dropna(subset=['pullback_success']).copy()
    
    # Validation Full (For Backtest - preserves sequence)
    val_df_full = labeled.iloc[train_size:].copy()
    
    # Validation Model (For Early Stopping - filtered)
    val_df_model = val_df_full.dropna(subset=['pullback_success']).copy()
    
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples (Model): {len(val_df_model)}")
    print(f"Validation bars (Backtest): {len(val_df_full)}")
    print("----------------------------------\n")
    
    return train_df, val_df_model, val_df_full, feature_cols, cfg

def objective(trial, train_df, val_df_model, val_df_full, feature_cols, base_config):
    # --- 1. Tuning LIGHTGBM Only ---
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        # Regularization to prevent overfitting
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'verbose': -1,
        'random_state': 42
    }
    
    # --- 2. FIXED Risk Settings (From Config) ---
    stop_loss_atr = base_config.labels.stop_atr_multiple
    take_profit_rr = base_config.labels.target_rr
    
    # --- 3. Train Model ---
    cfg = copy.deepcopy(base_config)
    for k, v in params.items():
        setattr(cfg.model, k, v)
        
    model = EntryQualityModel(cfg.model)
    try:
        model.train(
            train_df[feature_cols], train_df['pullback_success'], train_df['pullback_rr'],
            val_df_model[feature_cols], val_df_model['pullback_success'], val_df_model['pullback_rr'],
            verbose=False, calibrate=True
        )
    except Exception as e:
        return -999.0

    # --- 4. Get Probabilities (Vectorized) ---
    # We predict on the FULL validation set to simulate real timeline
    val_probs = model.predict(val_df_full[feature_cols], use_calibration=True)['bounce_prob']
    
    # --- 5. Inner Loop: Find BEST Min Bounce Prob ---
    # Scan every 1% probability step
    thresholds_to_test = np.arange(0.50, 0.86, 0.01)
    
    best_total_r = -999.0
    best_thresh = 0.0
    best_stats = {'trades': 0, 'wr': 0.0, 'avg_r': 0.0}

    for thresh in thresholds_to_test:
        trades, win_rate, total_r = FastBacktester.run(
            val_df_full, val_probs, thresh, stop_loss_atr, take_profit_rr
        )
        
        # Penalize inactivity: We want robust stats
        if trades < 3:
            score = -100.0 # Soft penalty
        else:
            score = total_r
            
        if score > best_total_r:
            best_total_r = score
            best_thresh = thresh
            avg_r = total_r / trades if trades > 0 else 0
            best_stats = {'trades': trades, 'wr': win_rate, 'avg_r': avg_r}

    # --- 6. Logging & Saving ---
    if best_stats['trades'] >= 3:
        logger.info(
            f"Trial {trial.number:03d} | "
            f"Thresh: {best_thresh:.2f} | "
            f"Trades: {best_stats['trades']:03d} | "
            f"WR: {best_stats['wr']:.1%} | "
            f"Total R: {best_total_r:6.2f} | "
            f"Avg R: {best_stats['avg_r']:.2f}"
        )
    else:
        logger.info(f"Trial {trial.number:03d} | Inactive (Trades < 3)")

    # Save best threshold to this trial's attributes
    trial.set_user_attr("best_min_bounce_prob", float(best_thresh))
    trial.set_user_attr("trades", int(best_stats['trades']))
    
    return best_total_r

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/MONUSDT')
    parser.add_argument('--n-trials', type=int, default=50)
    args = parser.parse_args()
    
    train_df, val_df_model, val_df_full, feat_cols, cfg = prepare_data(args.data_dir)
    
    print(f"Optimization Settings:")
    print(f"  Fixed Stop Loss:   {cfg.labels.stop_atr_multiple} ATR")
    print(f"  Fixed Take Profit: {cfg.labels.target_rr} R")
    print("----------------------------------\n")
    
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda t: objective(t, train_df, val_df_model, val_df_full, feat_cols, cfg), 
        n_trials=args.n_trials
    )
    
    print("\n==================================")
    print("       OPTIMIZATION RESULTS       ")
    print("==================================")
    print(f"Best Total R-Multiple: {study.best_value:.2f}")
    
    if len(study.trials) > 0:
        best_trial = study.best_trial
        print(f"Best Threshold (min_bounce_prob): {best_trial.user_attrs['best_min_bounce_prob']:.2f}")
        print(f"Trade Count: {best_trial.user_attrs['trades']}")
        
        print("\nBest Model Hyperparameters:")
        for k, v in best_trial.params.items():
            print(f"  {k}: {v}")
        
        # Create final config dict
        final_config = best_trial.params.copy()
        final_config['min_bounce_prob'] = best_trial.user_attrs['best_min_bounce_prob']
        
        import json
        with open('best_params.json', 'w') as f:
            json.dump(final_config, f, indent=4)
        print("\nSaved to best_params.json")
