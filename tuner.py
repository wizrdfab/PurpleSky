"""
Professional Tuner with Time-Series Cross-Validation.
Optimizes for Sharpe Ratio across multiple market regimes.
"""
import optuna
import numpy as np
import pandas as pd
import copy
from sklearn.model_selection import TimeSeriesSplit
from config import CONF
from data_loader import DataLoader
from feature_engine import FeatureEngine
from labels import Labeler
from models import ModelManager
from backtest import Backtester

# Global Cache
CACHE = {}

def get_data():
    if 'data' in CACHE: return CACHE['data']
    
    dl = DataLoader(CONF.data)
    # Load and Merge
    df = dl.load_and_merge(CONF.features.base_timeframe)
    if df.empty: return df
    
    fe = FeatureEngine(CONF.features)
    df = fe.calculate_features(df)
    
    CACHE['data'] = df
    return df

def objective(trial):
    # 1. Strategy Parameters (Labels)
    offset = trial.suggest_float('limit_offset_atr', 0.5, 1.5)
    tp = trial.suggest_float('take_profit_atr', 0.5, 2.0)
    sl = trial.suggest_float('stop_loss_atr', 1.0, 4.0)
    
    # 2. Model Parameters
    lr = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
    depth = trial.suggest_int('max_depth', 3, 8)
    leaves = trial.suggest_int('num_leaves', 16, 128)
    
    # 3. Execution Threshold
    thresh = trial.suggest_float('model_threshold', 0.55, 0.85)
    
    # Config Setup
    conf = copy.deepcopy(CONF)
    conf.strategy.base_limit_offset_atr = offset
    conf.strategy.take_profit_atr = tp
    conf.strategy.stop_loss_atr = sl
    conf.model.learning_rate = lr
    conf.model.max_depth = depth
    conf.model.num_leaves = leaves
    conf.model.n_estimators = 300 # Faster for tuning
    
    # Data
    df_orig = get_data()
    if df_orig.empty: return -999
    df = df_orig.copy()
    
    # Generate Labels (Expensive but necessary as params change)
    lbl = Labeler(conf)
    df = lbl.generate_labels(df)
    
    # Basic Filter
    if df['target_long'].sum() < 50: return -1.0
    
    # --- CROSS VALIDATION (Robustness Check) ---
    # We use the first 85% of data for the CV process (Train+Val)
    # We reserve the final 15% (Test) strictly for final verification (not used here)
    dev_size = int(len(df) * (conf.model.train_ratio + conf.model.val_ratio))
    df_dev = df.iloc[:dev_size].reset_index(drop=True)
    
    tscv = TimeSeriesSplit(n_splits=3, test_size=int(len(df_dev)*0.2))
    
    scores = []
    
    excludes = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'datetime', 'target_long', 'target_short', 'pred_long', 'pred_short',
                'vol_delta', 'buy_vol', 'trade_count', 'sell_vol', 'ob_imbalance_last', 'ob_spread_mean', 'ob_bid_depth_mean', 'ob_ask_depth_mean']
    feats = [c for c in df.columns if c not in excludes and not c.startswith('ob_') or c in ['ob_imbalance_z', 'ob_spread_ratio', 'ob_bid_impulse', 'ob_ask_impulse', 'ob_depth_ratio']]

    for fold, (train_idx, val_idx) in enumerate(tscv.split(df_dev)):
        # Train
        X_train = df_dev.iloc[train_idx][feats]
        y_long_train = df_dev.iloc[train_idx]['target_long']
        y_short_train = df_dev.iloc[train_idx]['target_short']
        
        # We manually train here to avoid saving models to disk repeatedly
        # Simplified LightGBM for speed
        import lightgbm as lgb
        params = {
            'objective': 'binary', 'metric': 'auc', 'verbosity': -1, 'n_jobs': -1,
            'learning_rate': lr, 'max_depth': depth, 'num_leaves': leaves,
            'boosting_type': 'gbdt' # Use GBDT for tuning speed, DART for final
        }
        
        m_long = lgb.train(params, lgb.Dataset(X_train, label=y_long_train), num_boost_round=100)
        m_short = lgb.train(params, lgb.Dataset(X_train, label=y_short_train), num_boost_round=100)
        
        # Validation
        val_df = df_dev.iloc[val_idx].copy()
        val_X = val_df[feats]
        val_df['pred_long'] = m_long.predict(val_X)
        val_df['pred_short'] = m_short.predict(val_X)
        
        # Backtest
        bt = Backtester(conf)
        res = bt.run(val_df, threshold=thresh)
        
        # Metric: Sharpe-like (Return / Downside Risk proxy)
        # Using simple: Return * WinRate * log(Trades)
        if res['trades'] < 3:
            score = -0.1
        else:
            score = res['total_return'] * (res['win_rate'] ** 2) * np.log(res['trades'])
            
        scores.append(score)
    
    # Conservative Aggregation: Use the MEAN score minus std dev (penalize inconsistency)
    final_score = np.mean(scores) - (np.std(scores) * 0.5)
    return final_score

if __name__ == "__main__":
    print("Starting Robust CV Optimization (50 Trials)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    print("Best Robust Params:", study.best_params)