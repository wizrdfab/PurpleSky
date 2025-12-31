"""
Auto-ML Training & Optimization CLI.
Full Pipeline: Load -> Tune -> Select Top 5 -> Tournament (OOS) -> Save Champions.
Strictly uses Config Timeframe (e.g. 5m).
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

def get_data(config: GlobalConfig):
    if 'data' in CACHE: return CACHE['data']
    
    logger_name = "DataPrep"
    print(f"[{logger_name}] Loading {config.features.base_timeframe} data from {config.data.data_dir}...")
    
    dl = DataLoader(config.data)
    df = dl.load_and_merge(config.features.base_timeframe)
    
    if df.empty:
        raise Exception("No data found!")
        
    print(f"[{logger_name}] Generating Features for {len(df)} bars...")
    fe = FeatureEngine(config.features)
    df = fe.calculate_features(df)
    
    CACHE['data'] = df
    return df

class AutoML:
    def __init__(self, args):
        self.args = args
        CONF.data.data_dir = Path(args.data_dir)
        CONF.data.symbol = args.symbol
        CONF.model.model_dir = Path(args.model_dir) / args.symbol
        
        # Load Data Once
        self.raw_df = get_data(CONF)
        
    def run(self):
        # 1. Tuning Phase
        print(f"\n=== Starting Optimization ({self.args.trials} trials) ===")
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.args.trials)
        
        print("\n=== Optimization Complete. Starting Tournament ===")
        
        # Select Top 10 Candidates (Validation Score)
        valid_trials = [t for t in study.trials if t.value is not None and t.value > -1.0]
        valid_trials.sort(key=lambda x: x.value, reverse=True)
        top_candidates = valid_trials[:10]
        
        tournament_results = []
        
        # 2. Tournament Phase (OOS Backtest)
        for i, trial in enumerate(top_candidates):
            print(f"\nEvaluating Candidate #{i+1} (Trial {trial.number})...")
            res = self.evaluate_candidate(trial.params, trial_number=trial.number)
            
            # Tournament Score: Sortino * log(Trades)
            if res['trades'] < 2:
                res['tournament_score'] = -1.0
            else:
                res['tournament_score'] = res['sortino'] * np.log(res['trades'])
                
            tournament_results.append(res)
            
        # 3. Save Champions
        print("\n=== Tournament Results (Ranked by Tournament Score) ===")
        tournament_results.sort(key=lambda x: x['tournament_score'], reverse=True)
        
        for i in range(min(3, len(tournament_results))):
            self.save_champion(tournament_results[i], rank=i+1)
            
        for res in tournament_results:
            print(f"Trial {res['trial_number']:<3} | Score: {res['tournament_score']:<6.3f} | Ret: {res['oos_return']:.2%} | WR: {res['win_rate']:.1%} | DD: {res['max_drawdown']:.2%} | Trades: {res['trades']}")

    def get_feature_list(self, df):
        # Quant Lead: Strictly exclude raw non-stationary values
        excludes = [
            'open', 'high', 'low', 'close', 'volume', 'vwap', 'datetime', 
            'target_long', 'target_short', 'pred_long', 'pred_short',
            'vol_delta', 'buy_vol', 'vol_sell', 'trade_count', 'sell_vol', 'dollar_val', 'total_val',
            'ob_imbalance_last', 'ob_spread_mean', 'ob_bid_depth_mean', 'ob_ask_depth_mean',
            'ob_micro_dev_std', 'ob_micro_dev_last', 'ob_micro_dev_mean', 'ob_imbalance_mean'
        ]
        
        # Whitelist of engineered Orderbook features
        ob_whitelist = [
            'ob_imbalance_z', 'ob_spread_ratio', 'ob_bid_impulse', 'ob_ask_impulse', 
            'ob_depth_ratio', 'ob_spread_bps', 'ob_depth_log_ratio', 'price_liq_div', 
            'liq_dominance', 'micro_dev_vol', 'ob_imb_trend', 'micro_pressure', 
            'bid_depth_chg', 'ask_depth_chg', 'spread_z'
        ]
        
        feats = [c for c in df.columns if (c not in excludes and not c.startswith('ob_')) or c in ob_whitelist]
        return feats

    def objective(self, trial):
        # Params
        offset = trial.suggest_float('limit_offset_atr', 0.5, 1.5)
        tp = trial.suggest_float('take_profit_atr', 0.5, 2.5)
        sl = trial.suggest_float('stop_loss_atr', 1.0, 4.0)
        lr = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
        depth = trial.suggest_int('max_depth', 3, 8)
        leaves = trial.suggest_int('num_leaves', 16, 128)
        thresh = trial.suggest_float('model_threshold', 0.55, 0.85)
        
        conf = copy.deepcopy(CONF)
        conf.strategy.base_limit_offset_atr = offset
        conf.strategy.take_profit_atr = tp
        conf.strategy.stop_loss_atr = sl
        conf.model.learning_rate = lr
        conf.model.max_depth = depth
        conf.model.num_leaves = leaves
        conf.model.n_estimators = 300
        
        df = self.raw_df.copy()
        lbl = Labeler(conf)
        df = lbl.generate_labels(df)
        
        if df['target_long'].sum() < 50: return -1.0
        
        # CV Logic
        dev_size = int(len(df) * (conf.model.train_ratio + conf.model.val_ratio))
        df_dev = df.iloc[:dev_size].reset_index(drop=True)
        tscv = TimeSeriesSplit(n_splits=3, test_size=int(len(df_dev)*0.2))
        
        feats = self.get_feature_list(df)
        scores = []

        import lightgbm as lgb
        for train_idx, val_idx in tscv.split(df_dev):
            X_train, y_l = df_dev.iloc[train_idx][feats], df_dev.iloc[train_idx]['target_long']
            y_s = df_dev.iloc[train_idx]['target_short']
            
            p = {'objective':'binary', 'metric':'auc', 'verbosity':-1, 'n_jobs':-1,
                 'learning_rate':lr, 'max_depth':depth, 'num_leaves':leaves, 'boosting_type':'gbdt'}
            
            m_l = lgb.train(p, lgb.Dataset(X_train, label=y_l), num_boost_round=100)
            m_s = lgb.train(p, lgb.Dataset(X_train, label=y_s), num_boost_round=100)
            
            val_f = df_dev.iloc[val_idx].copy()
            val_f['pred_long'] = m_l.predict(val_f[feats])
            val_f['pred_short'] = m_s.predict(val_f[feats])
            
            bt = Backtester(conf)
            res = bt.run(val_f, threshold=thresh)
            scores.append(res['sortino'] if res['trades'] >= 3 else -0.1)
                
        return np.mean(scores)

    def evaluate_candidate(self, params, trial_number):
        conf = copy.deepcopy(CONF)
        conf.strategy.base_limit_offset_atr = params['limit_offset_atr']
        conf.strategy.take_profit_atr = params['take_profit_atr']
        conf.strategy.stop_loss_atr = params['stop_loss_atr']
        conf.model.learning_rate = params['learning_rate']
        conf.model.max_depth = params['max_depth']
        conf.model.num_leaves = params['num_leaves']
        conf.model.model_threshold = params['model_threshold']
        
        df = self.raw_df.copy()
        lbl = Labeler(conf)
        df = lbl.generate_labels(df)
        
        feats = self.get_feature_list(df)
        
        mm = ModelManager(conf.model)
        mm.train(df, feats)
        
        df = mm.predict(df)
        test_start = int(len(df) * (conf.model.train_ratio + conf.model.val_ratio))
        test_df = df.iloc[test_start:]
        
        bt = Backtester(conf)
        res = bt.run(test_df, threshold=conf.model.model_threshold)
        
        return {
            'trial_number': trial_number, 'params': params,
            'oos_return': res['total_return'], 'win_rate': res['win_rate'],
            'trades': res['trades'], 'max_drawdown': res['max_drawdown'],
            'sortino': res['sortino'], 'model_long': mm.model_long,
            'model_short': mm.model_short, 'features': feats
        }

    def save_champion(self, res, rank):
        save_path = CONF.model.model_dir / f"rank_{rank}"
        save_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(res['model_long'], save_path / "model_long.pkl")
        joblib.dump(res['model_short'], save_path / "model_short.pkl")
        joblib.dump(res['features'], save_path / "features.pkl")
        with open(save_path / "params.json", "w") as f:
            json.dump(res['params'], f, indent=4)
        with open(save_path / "metrics.json", "w") as f:
            metrics = {k: v for k,v in res.items() if k not in ['model_long', 'model_short', 'features', 'params']}
            json.dump(metrics, f, indent=4)
        print(f"  -> Saved Rank {rank} Champion to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--data-dir", type=str, default="data/RAVEUSDT")
    parser.add_argument("--symbol", type=str, default="RAVEUSDT")
    parser.add_argument("--model-dir", type=str, default="models_v2")
    args = parser.parse_args()
    
    pipeline = AutoML(args)
    pipeline.run()