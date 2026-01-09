"""
Stepwise Auto-ML Pipeline.
Phase 1: Feature Selection (MDA/Importance).
Phase 2: Alpha Modeling (Maximize IC).
Phase 3: Strategy Optimization (Maximize PnL on Fixed Alpha).
"""
import argparse
import optuna
import pandas as pd
import numpy as np
import copy
import joblib
import json
import lightgbm as lgb
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

from config import CONF, GlobalConfig
from data_loader import DataLoader
from feature_engine import FeatureEngine
from labels import Labeler
from models import ModelManager
from backtest import Backtester

# --- HELPER FUNCTIONS ---

def get_data(config: GlobalConfig, timeframe: str):
    print(f"[Data] Loading {timeframe} data...")
    dl = DataLoader(config.data)
    df = dl.load_and_merge(timeframe)
    if df.empty: raise Exception("No data found!")
    
    print(f"[Features] Generating Stationary Features...")
    fe = FeatureEngine(config.features)
    df = fe.calculate_features(df)
    
    print(f"[Labels] Generating Targets...")
    lbl = Labeler(config)
    df = lbl.generate_labels(df)
    
    # Drop NaNs created by lags/diffs
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

def get_feature_list(df):
    # Exclude non-feature columns
    excludes = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'datetime', 
                'target_long', 'target_short', 'target_return', 'target_dir_long', 'target_dir_short',
                'pred_long', 'pred_short', 'pred_return', 'tr', 'atr', 'log_close', 'log_vol', 'volatility']
    
    return [c for c in df.columns if c not in excludes and not c.startswith('target_')]

# --- PHASE 1: FEATURE SELECTION ---

def select_features(df, initial_feats, top_k=30):
    print(f"\n=== Phase 1: Feature Selection (Input: {len(initial_feats)}) ===")
    
    # Train a quick model to get importance
    # Target: We use the continuous return target for feature selection too
    X = df[initial_feats]
    y = df['target_return']
    
    # Fast parameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'n_estimators': 100,
        'learning_rate': 0.1,
        'num_leaves': 31,
        'verbosity': -1,
        'n_jobs': -1
    }
    
    model = lgb.train(params, lgb.Dataset(X, label=y))
    
    # Get Importance
    imp = model.feature_importance(importance_type='gain')
    feat_imp = pd.DataFrame({'feature': initial_feats, 'gain': imp})
    feat_imp = feat_imp.sort_values('gain', ascending=False)
    
    selected = feat_imp.head(top_k)['feature'].tolist()
    print(f"Selected Top {top_k} Features:")
    print(selected)
    
    return selected

# --- PHASE 2: ALPHA MODELING (IC) ---

def custom_ic_metric(preds, train_data):
    # LightGBM custom metric: (name, value, is_higher_better)
    labels = train_data.get_label()
    # Spearman Correlation
    corr, _ = spearmanr(preds, labels)
    return 'ic', corr, True

def train_alpha_model(df, features, trials=20):
    print(f"\n=== Phase 2: Alpha Modeling (Maximizing IC) ===")
    
    # Split Train/Val (Time Series Split)
    split_idx = int(len(df) * 0.80)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    X_train = train_df[features]
    y_train = train_df['target_return']
    X_val = val_df[features]
    y_val = val_df['target_return']
    
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'None', # We use custom IC
            'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 16, 64),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'verbosity': -1,
            'n_jobs': -1
        }
        
        model = lgb.train(
            params, 
            lgb.Dataset(X_train, label=y_train),
            valid_sets=[lgb.Dataset(X_val, label=y_val)],
            feval=custom_ic_metric,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Return best IC score
        return model.best_score['valid_0']['ic']

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=trials)
    
    print("Best Alpha Params:", study.best_params)
    print("Best IC:", study.best_value)
    
    # Train Final Model with Best Params
    best_params = study.best_params
    best_params.update({
        'objective': 'regression', 
        'metric': 'None', 
        'verbosity': -1, 
        'n_jobs': -1
    })
    
    final_model = lgb.train(
        best_params, 
        lgb.Dataset(df[features], label=df['target_return']), # Train on full data? Or keep split?
        # Let's train on Train+Val for the Strategy Phase
        num_boost_round=int(best_params['n_estimators'] * 1.2)
    )
    
    return final_model

# --- PHASE 3: STRATEGY OPTIMIZATION ---

def optimize_strategy(df, model, features, trials=30):
    print(f"\n=== Phase 3: Strategy Optimization (Fixed Alpha) ===")
    
    # Generate Predictions for the whole dataset
    print("Generating predictions...")
    preds = model.predict(df[features])
    df['pred_return'] = preds
    
    # Using the last chunk for Strategy Optimization to avoid lookahead bias relative to Alpha Training?
    # Actually, we should ideally have: Train(Alpha) -> Val(Alpha) -> Opt(Strategy) -> Test(Final).
    # For now, let's use the Validation set from Phase 2 for Strategy Opt.
    split_idx = int(len(df) * 0.80)
    opt_df = df.iloc[split_idx:].copy() # Use the most recent 20%
    
    def objective(trial):
        # Strategy Parameters
        threshold = trial.suggest_float('model_threshold', 0.5, 2.0) # Threshold on predicted return (z-score like)
        tp = trial.suggest_float('take_profit_atr', 0.5, 3.0)
        sl = trial.suggest_float('stop_loss_atr', 1.0, 5.0)
        offset = trial.suggest_float('limit_offset_atr', 0.0, 1.0)
        
        # Configure Backtester
        conf = copy.deepcopy(CONF)
        conf.strategy.take_profit_atr = tp
        conf.strategy.stop_loss_atr = sl
        conf.strategy.base_limit_offset_atr = offset
        
        # Map Continuous Prediction to Signals
        # pred_return is roughly Z-Score of return.
        # Long if > threshold, Short if < -threshold
        opt_df['pred_long'] = (opt_df['pred_return'] > threshold).astype(float)
        opt_df['pred_short'] = (opt_df['pred_return'] < -threshold).astype(float)
        
        # Run Backtest
        res = Backtester(conf).run(opt_df, threshold=0.5) # Threshold handled above
        
        if res['trades'] < 5: return -1.0
        return res['sortino']

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=trials)
    
    print("Best Strategy Params:", study.best_params)
    print("Best Sortino:", study.best_value)
    
    return study.best_params

# --- MAIN ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="RAVEUSDT")
    parser.add_argument("--timeframe", type=str, default="5m")
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()
    
    CONF.data.symbol = args.symbol
    
    # 1. Load Data
    df = get_data(CONF, args.timeframe)
    all_feats = get_feature_list(df)
    
    # 2. Feature Selection
    selected_feats = select_features(df, all_feats, top_k=25)
    
    # 3. Alpha Model
    model = train_alpha_model(df, selected_feats, trials=args.trials)
    
    # 4. Strategy Opt
    strat_params = optimize_strategy(df, model, selected_feats, trials=args.trials)
    
    # 5. Save Everything
    save_path = CONF.model.model_dir / args.symbol / "stepwise_v1"
    save_path.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, save_path / "alpha_model.pkl")
    joblib.dump(selected_feats, save_path / "features.pkl")
    with open(save_path / "strategy_params.json", "w") as f:
        json.dump(strat_params, f, indent=4)
        
    print(f"Done. Saved to {save_path}")

if __name__ == "__main__":
    main()
