"""
Copyright (C) 2026 wizrdfab

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
import argparse
import optuna
import pandas as pd
import numpy as np
import copy
import joblib
import json
import os
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit

from config import CONF, GlobalConfig
from data_loader import DataLoader
from feature_engine import FeatureEngine
from labels import Labeler
from models import ModelManager
from backtest import Backtester

# Global Cache
CACHE = {}

class CombinatorialPurgedKFold:
    """
    Combinatorial Purged Cross-Validation (CPCV).
    Splits data into N groups, takes k combinations for testing.
    Purges overlap at boundaries to prevent leakage.
    """
    def __init__(self, n_splits=5, n_test_splits=2, purge_overlap=50):
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge = purge_overlap
        
    def split(self, X):
        n = len(X)
        indices = np.arange(n)
        fold_size = n // self.n_splits
        
        # Split indices into N chunks
        chunks = [indices[i*fold_size : (i+1)*fold_size] for i in range(self.n_splits)]
        # Handle remainder in the last chunk
        if n % self.n_splits != 0:
            chunks[-1] = np.concatenate([chunks[-1], indices[self.n_splits*fold_size:]])
        
        from itertools import combinations
        chunk_indices = range(self.n_splits)
        
        for test_chunk_idxs in combinations(chunk_indices, self.n_test_splits):
            test_chunk_idxs = set(test_chunk_idxs)
            
            # Construct Test Set
            test_idx = np.concatenate([chunks[i] for i in test_chunk_idxs])
            
            # Construct Train Set with Purging
            train_indices_list = []
            
            for i in range(self.n_splits):
                if i in test_chunk_idxs:
                    continue
                
                # This chunk is TRAIN. Check neighbors to see if we need to purge.
                # A chunk i needs purging if it borders a TEST chunk.
                
                c_idx = chunks[i]
                start = c_idx[0]
                end = c_idx[-1]
                
                # Check PREVIOUS chunk (i-1)
                # If i-1 was TEST, we must embargo the start of i (optional, but good for safety)
                # For simplicity in this implementation, we focus on the main leakage:
                # Labels in Train that look forward into Test.
                # This happens when Train comes BEFORE Test.
                
                # Masking array for this chunk
                mask = np.ones(len(c_idx), dtype=bool)
                
                # 1. Forward Leakage: Train i is followed by Test i+1
                if (i + 1) in test_chunk_idxs:
                    # Remove the LAST 'purge' items from this train chunk
                    # because their labels might look into the start of the next (Test) chunk
                    mask[-self.purge:] = False
                    
                # 2. Backward Leakage (Lookback features): Train i follows Test i-1
                # If features use lagged data from Test, it's usually fine (market memory),
                # but we can embargo start to be safe.
                # (Skipped for standard label leakage protection, strictly focused on forward labels)
                
                train_indices_list.append(c_idx[mask])
            
            if not train_indices_list:
                continue
                
            train_idx = np.concatenate(train_indices_list)
            yield train_idx, test_idx

def get_data(config: GlobalConfig, timeframe: str):
    key = f"{timeframe}_{config.data.ob_levels}"
    if key in CACHE: return CACHE[key]
    print(f"[DataPrep] Loading {timeframe} data (OB Levels: {config.data.ob_levels})...")
    dl = DataLoader(config.data)
    df = dl.load_and_merge(timeframe)
    if df.empty: raise Exception("No data found!")
    print(f"[DataPrep] Generating Features for {len(df)} bars...")
    fe = FeatureEngine(config.features)
    df = fe.calculate_features(df)
    CACHE[key] = df
    return df

class AutoML:
    def __init__(self, args):
        self.args = args
        CONF.data.data_dir = Path(args.data_dir)
        CONF.data.symbol = args.symbol
        CONF.model.model_dir = Path(args.model_dir) / args.symbol
        CONF.data.ob_levels = args.ob_levels
        CONF.model.extra_trees = args.extra_trees
        self.raw_df = get_data(CONF, args.timeframe)
        
    def run(self):
        # 1. Tuning Phase
        print(f"\n=== Phase 1: Optimization ({self.args.trials} trials) ===")
        print("Objective: Maximize Sharpe/Sortino via CPCV on first 85% of data.")
        
        sampler = optuna.samplers.TPESampler(n_startup_trials=int(self.args.trials * 0.5), seed=CONF.seed)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(self.objective, n_trials=self.args.trials)
        
        best = study.best_trial
        print("\n" + "="*60)
        print(f"BEST TRIAL FOUND (Trial {best.number})")
        print(f"Value (CV Score): {best.value:.4f}")
        print("Params:", json.dumps(best.params, indent=2))
        print("="*60 + "\n")
        
        # 2. Final Training & Test (Last 15%)
        print("=== Phase 2: Champion Validation (Train 85% -> Test 15%) ===")
        
        # Apply Best Params to Config
        conf = copy.deepcopy(CONF)
        p = best.params
        conf.strategy.base_limit_offset_atr = p['limit_offset_atr']
        conf.strategy.take_profit_atr = p['take_profit_atr']
        conf.strategy.stop_loss_atr = p['stop_loss_atr']
        conf.model.learning_rate = p['learning_rate']
        conf.model.max_depth = p['max_depth']
        conf.model.num_leaves = p['num_leaves']
        conf.model.model_threshold = p['model_threshold']
        conf.model.min_child_samples = 40
        
        # Regenerate Labels with OPTIMIZED Strategy Params
        print("Regenerating labels with optimized strategy parameters...")
        df_full = self.raw_df.copy()
        df_full = Labeler(conf).generate_labels(df_full)
        feats = self.get_feature_list(df_full)
        
        # Split 85/15
        split_idx = int(len(df_full) * 0.85)
        df_train = df_full.iloc[:split_idx]
        df_test = df_full.iloc[split_idx:]
        
        print(f"Training Final Model on {len(df_train)} bars...")
        mm = ModelManager(conf.model)
        mm.train(df_train, feats)
        
        # Print Feature Importance
        for name, model in [("LONG", mm.model_long), ("SHORT", mm.model_short)]:
            print("\n" + f" [ {name} FEATURE IMPORTANCE ] ".center(60, "="))
            importances = model.feature_importance(importance_type='gain')
            feat_imp = sorted(zip(feats, importances), key=lambda x: x[1], reverse=True)
            for f, imp in feat_imp:
                print(f"{f:<30} | {imp:>10.2f}")
            print("="*60)
        
        print(f"Backtesting on Holdout Set ({len(df_test)} bars)...")
        # Predict on Test
        df_test_pred = mm.predict(df_test.copy())
        
        # Backtest
        # Note: We pass the optimized threshold to the backtester if needed, 
        # but the Backtester class usually takes it from 'pred_long' > thresh or passed arg.
        # We'll pass the threshold explicitly.
        res = Backtester(conf).run(df_test_pred, threshold=conf.model.model_threshold)
        
        print("\n" + " [ CHAMPION PERFORMANCE - 15% HOLDOUT ] ".center(60, "="))
        metrics = [
            ("Total Return", f"{res['total_return']:.2%}"),
            ("Win Rate", f"{res['win_rate']:.2%}"),
            ("Total Trades", f"{res['trades']}"),
            ("Max Drawdown", f"{res['max_drawdown']:.2%}"),
            ("Sortino Ratio", f"{res['sortino']:.2f}"),
            ("Final Equity", f"${res['final_equity']:.2f}")
        ]
        for label, val in metrics:
            print(f"{label:<30} | {val:>15}")
        print("="*60 + "\n")
        
        # Save Champion (Rank 1)
        print("Saving Champion to rank_1...")
        self.save_champion({
            'model_manager': mm,
            'features': feats,
            'params': p,
            # Dummy placeholders for the save function structure
            'trial_number': best.number,
            'oos_return': res['total_return'],
            'win_rate': res['win_rate'],
            'trades': res['trades'],
            'sortino': res['sortino']
        }, rank=1)
        
        # Clean up other ranks if they exist (to avoid confusion)
        # (Optional, but good practice if switching modes)

    def backtest_council(self, council, df):
        # Deprecated in this mode
        pass

    def get_feature_list(self, df):
        excludes = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'datetime', 'target_long', 'target_short', 'pred_long', 'pred_short',
                    'vol_delta', 'buy_vol', 'vol_sell', 'trade_count', 'sell_vol', 'dollar_val', 'total_val',
                    'ob_imbalance_last', 'ob_spread_mean', 'ob_bid_depth_mean', 'ob_ask_depth_mean',
                    'ob_micro_dev_std', 'ob_micro_dev_last', 'ob_micro_dev_mean', 'ob_imbalance_mean',
                    'target_dir_long', 'target_dir_short', 'pred_dir_long', 'pred_dir_short', 'atr', 'price_path']
        ob_whitelist = ['ob_imbalance_z', 'ob_spread_ratio', 'ob_bid_impulse', 'ob_ask_impulse', 'ob_depth_ratio', 'ob_spread_bps', 'ob_depth_log_ratio', 'price_liq_div', 'liq_dominance', 'micro_dev_vol', 'ob_imb_trend', 'micro_pressure', 'bid_depth_chg', 'ask_depth_chg', 'spread_z', 'ob_slope_ratio', 'bid_slope_z', 'ob_bid_elasticity', 'ob_ask_elasticity', 'ob_bid_integrity_mean', 'ob_ask_integrity_mean', 'ob_integrity_skew', 'bid_integrity_chg', 'ask_integrity_chg']
        
        all_features = [c for c in df.columns if (c not in excludes and not c.startswith('ob_')) or c in ob_whitelist]
        
        if self.args.microstructure_only:
            tech_blacklist = ['ema_', 'dist_ema_', 'rsi', 'atr', 'natr']
            filtered = [f for f in all_features if not any(b in f for b in tech_blacklist)]
            
            print(f"[Research] Microstructure Only: Reduced features from {len(all_features)} to {len(filtered)}")
            return filtered
            
        return all_features

    def objective(self, trial):
        offset, tp, sl = trial.suggest_float('limit_offset_atr', 0.5, 1.5), trial.suggest_float('take_profit_atr', 0.5, 2.5), trial.suggest_float('stop_loss_atr', 1.0, 4.0)
        
        # Constrained Search Space to prevent Overfitting (SNIPER MODE)
        lr = trial.suggest_float('learning_rate', 0.005, 0.05, log=True)
        depth = trial.suggest_int('max_depth', 2, 4)
        leaves = trial.suggest_int('num_leaves', 4, 16)
        min_child = 40
        
        thresh = trial.suggest_float('model_threshold', 0.60, 0.93)
        dir_thresh = trial.suggest_float('direction_threshold', 0.35, 0.65)
        agg_thresh = trial.suggest_float('aggressive_threshold', 0.75, 0.98)
        
        conf = copy.deepcopy(CONF)
        conf.strategy.base_limit_offset_atr, conf.strategy.take_profit_atr, conf.strategy.stop_loss_atr = offset, tp, sl
        conf.model.learning_rate, conf.model.max_depth, conf.model.num_leaves = lr, depth, leaves
        conf.model.min_child_samples = min_child
        conf.model.direction_threshold = dir_thresh
        conf.model.aggressive_threshold = agg_thresh
        
        df = self.raw_df.copy()
        df = Labeler(conf).generate_labels(df)
        if df['target_long'].sum() < 50: return -1.0
        
        dev_size = int(len(df) * 0.85)
        df_dev = df.iloc[:dev_size].reset_index(drop=True)
        
        # CPCV Split
        tscv = CombinatorialPurgedKFold(n_splits=5, n_test_splits=1) 
        
        feats = self.get_feature_list(df)
        scores = []
        
        # We only train Execution Model inside optimization loop for speed
        # Direction Model is implicit or trained once, but here we just optimize Execution Logic
        import lightgbm as lgb
        for train_idx, val_idx in tscv.split(df_dev):
            X_t = df_dev.iloc[train_idx][feats]
            y_l, y_s = df_dev.iloc[train_idx]['target_long'], df_dev.iloc[train_idx]['target_short']
            y_dir_l, y_dir_s = df_dev.iloc[train_idx]['target_dir_long'], df_dev.iloc[train_idx]['target_dir_short']
            
            p = {'objective':'binary','metric':'auc','verbosity':-1,'n_jobs':-1,
                 'learning_rate':lr,'max_depth':depth,'num_leaves':leaves,
                 'min_child_samples':conf.model.min_child_samples,'boosting_type':'gbdt'}
            
            # Train Execution Models
            m_l = lgb.train(p, lgb.Dataset(X_t, label=y_l), num_boost_round=100)
            m_s = lgb.train(p, lgb.Dataset(X_t, label=y_s), num_boost_round=100)
            
            # Train Direction Models (Lighter version for speed)
            m_dir_l = lgb.train(p, lgb.Dataset(X_t, label=y_dir_l), num_boost_round=50)
            m_dir_s = lgb.train(p, lgb.Dataset(X_t, label=y_dir_s), num_boost_round=50)
            
            val_f = df_dev.iloc[val_idx].copy()
            val_f['pred_long'], val_f['pred_short'] = m_l.predict(val_f[feats]), m_s.predict(val_f[feats])
            val_f['pred_dir_long'], val_f['pred_dir_short'] = m_dir_l.predict(val_f[feats]), m_dir_s.predict(val_f[feats])
            
            # Simple Backtest (Direction logic is not applied in optimization loop to keep it fast)
            res = Backtester(conf).run(val_f, threshold=thresh)
            
            # Revert to standard selective requirement
            scores.append(res['sortino'] * np.log(res['trades']) if res['trades'] >= 3 else -0.1)
            
        return np.mean(scores)

    def verify_candidate(self, params, trial_number, val_score):
        conf = copy.deepcopy(CONF)
        conf.strategy.base_limit_offset_atr, conf.strategy.take_profit_atr, conf.strategy.stop_loss_atr = params['limit_offset_atr'], params['take_profit_atr'], params['stop_loss_atr']
        conf.model.learning_rate, conf.model.max_depth, conf.model.num_leaves, conf.model.model_threshold = params['learning_rate'], params['max_depth'], params['num_leaves'], params['model_threshold']
        conf.model.min_child_samples = 40
        conf.model.direction_threshold = params['direction_threshold']
        conf.model.aggressive_threshold = params['aggressive_threshold']
        
        df = self.raw_df.copy()
        df = Labeler(conf).generate_labels(df)
        feats = self.get_feature_list(df)
        
        # Full Training (Direction + Execution)
        mm = ModelManager(conf.model)
        mm.train(df, feats) # Trains both A and B
        df = mm.predict(df)
        
        # OOS-1 Slice: 85% to 95%
        test_start = int(len(df) * 0.85)
        test_end = int(len(df) * 0.95)
        oos1_df = df.iloc[test_start:test_end]
        
        res = Backtester(conf).run(oos1_df, threshold=conf.model.model_threshold)
        
        return {
            'trial_number': trial_number, 'val_score': val_score, 'params': params,
            'oos_return': res['total_return'], 'win_rate': res['win_rate'],
            'trades': res['trades'], 'max_drawdown': res['max_drawdown'],
            'sortino': res['sortino'], 
            'oos_score': res['sortino'] * np.log(res['trades']) if res['trades'] >= 2 else -1.0,
            'model_manager': mm, 'features': feats # Return full manager
        }

    def save_champion(self, res, rank):
        save_path = CONF.model.model_dir / f"rank_{rank}"
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save ModelManager (Models A & B + LSTM & Gate)
        # Note: res['model_manager'] is the object
        mm = res['model_manager']
        
        # Point ModelManager to the correct rank folder
        mm.config.model_dir = save_path
        mm.save_models()
            
        # Params saving remains the same
        params = dict(res.get('params', {}))
        params.setdefault("symbol", self.args.symbol)
        params.setdefault("timeframe", self.args.timeframe)
        params.setdefault("ob_levels", self.args.ob_levels)
        with open(save_path / "params.json", "w") as f:
            json.dump(params, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--candidate-n", type=int, default=100, help="Number of top trials to move to OOS-1")
    parser.add_argument("--timeframe", type=str, default="5m")
    parser.add_argument("--ob-levels", type=int, default=CONF.data.ob_levels)
    parser.add_argument("--min-child", type=int, default=50) 
    parser.add_argument("--data-dir", type=str, default="data/RAVEUSDT")
    parser.add_argument("--symbol", type=str, default="RAVEUSDT")
    parser.add_argument("--model-dir", type=str, default="models_v4")
    parser.add_argument("--microstructure-only", action="store_true", help="Blacklist technical indicators")
    parser.add_argument("--extra-trees", action="store_true", help="Enable LightGBM extra_trees mode")
    args = parser.parse_args()
    AutoML(args).run()