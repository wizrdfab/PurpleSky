"""
Auto-ML Training: Lone Champion Edition.
Pipeline: 1. Optimize -> 2. OOS-1 Qualification -> 3. Super-OOS Verification.
NO COUNCILS.
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
        self.raw_df = get_data(CONF, args.timeframe)
        
    def run(self):
        # 1. Tuning Phase
        print(f"\n=== Phase 1: Optimization ({self.args.trials} trials) ===")
        sampler = optuna.samplers.TPESampler(n_startup_trials=int(self.args.trials * 0.5), seed=CONF.seed)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(self.objective, n_trials=self.args.trials)
        
        # 2. Select Top 10 for OOS-1
        valid_trials = [t for t in study.trials if t.value is not None and t.value > -1.0]
        valid_trials.sort(key=lambda x: x.value, reverse=True)
        candidates = valid_trials[:10]
        
        # 3. Individual Tournament (OOS-1)
        print("\n=== Phase 2: OOS-1 Qualification (10% slice) ===")
        results = []
        for i, trial in enumerate(candidates):
            print(f"Testing Candidate #{i+1} (Trial {trial.number})...")
            res = self.verify_candidate(trial.params, trial_number=trial.number, val_score=trial.value)
            results.append(res)
            
        results.sort(key=lambda x: x['oos_score'], reverse=True)
        
        # 4. Pick THE Best
        champion = results[0]
        print(f"\n🏆 LONE CHAMPION SELECTED: Trial {champion['trial_number']}")
        self.save_champion(champion, rank=1)
        
        # 5. Final Verification (Super-OOS)
        print("\n=== Phase 3: SUPER-OOS VERIFICATION (Final 5%) ===")
        self.run_final_test(champion)

        # Report
        print("\n" + "="*90)
        print(f"{ 'RANK':<5} | {'TRIAL':<6} | {'OOS-1 SCORE':<12} | {'OOS-1 RET':<10} | {'WINRATE':<8} | {'MAXDD'}")
        print("-" * 90)
        for i, res in enumerate(results):
            print(f"#{i+1:<4} | {res['trial_number']:<6} | {res['oos_score']:<12.3f} | {res['oos_return']:<10.2%} | {res['win_rate']:<8.1%} | {res['max_drawdown']:<8.2%}")
        print("="*90)

    def run_final_test(self, champ):
        df = self.raw_df.copy()
        start_idx = int(len(df) * 0.95)
        test_df = df.iloc[start_idx:].copy()
        
        mm = ModelManager(CONF.model)
        mm.model_long, mm.model_short, mm.feature_cols = champ['model_long'], champ['model_short'], champ['features']
        df_pred = mm.predict(test_df)
        
        # Backtest with Champ's specific params
        conf = copy.deepcopy(CONF)
        conf.strategy.base_limit_offset_atr = champ['params']['limit_offset_atr']
        conf.strategy.take_profit_atr = champ['params']['take_profit_atr']
        conf.strategy.stop_loss_atr = champ['params']['stop_loss_atr']
        
        bt = Backtester(conf)
        res = bt.run(df_pred, threshold=champ['params']['model_threshold'])
        
        score = -1.0
        if res['trades'] >= 2: score = res['sortino'] * np.log(res['trades'])
        
        print("\n" + "!"*50)
        print(f"CHAMPION FINAL EXAM (SUPER-OOS)")
        print("!"*50)
        print(f"Robust Score:   {score:.3f}")
        print(f"Return:         {res['total_return']:.2%}")
        print(f"Win Rate:       {res['win_rate']:.1%}")
        print(f"Trades:         {res['trades']}")
        print(f"Max Drawdown:   {res['max_drawdown']:.2%}")
        print("!"*50 + "\n")

    def run_ensemble_experiment(self, members, label="COUNCIL", target_slice="super_oos"):
        """Wrapper for backtesting a list of models."""
        df = self.raw_df.copy()
        total_len = len(df)
        
        if target_slice == "oos1":
            start, end = int(total_len * 0.85), int(total_len * 0.95)
            test_df = df.iloc[start:end].copy()
            slice_name = "OOS-1 (10%)"
        else: # super_oos
            start = int(total_len * 0.95)
            test_df = df.iloc[start:].copy()
            slice_name = "SUPER-OOS (5%)"
        
        preds_l, preds_s = [], []
        offsets, tps, sls, thresholds = [], [], [], []
        
        for m in members:
            mm = ModelManager(CONF.model)
            mm.model_long, mm.model_short, mm.feature_cols = m['model_long'], m['model_short'], m['features']
            df_pred = mm.predict(test_df.copy())
            preds_l.append(df_pred['pred_long'])
            preds_s.append(df_pred['pred_short'])
            offsets.append(m['params']['limit_offset_atr'])
            tps.append(m['params']['take_profit_atr'])
            sls.append(m['params']['stop_loss_atr'])
            thresholds.append(m['params']['model_threshold'])
            
        final_df = test_df.copy()
        final_df['pred_long'] = np.mean(preds_l, axis=0)
        final_df['pred_short'] = np.mean(preds_s, axis=0)
        avg_thresh = np.mean(thresholds)
        
        ens_conf = copy.deepcopy(CONF)
        ens_conf.strategy.base_limit_offset_atr = np.mean(offsets)
        ens_conf.strategy.take_profit_atr = np.mean(tps)
        ens_conf.strategy.stop_loss_atr = np.mean(sls)
        
        bt = Backtester(ens_conf)
        res = bt.run(final_df, threshold=avg_thresh)
        
        score = -1.0
        if res['trades'] >= 2: score = res['sortino'] * np.log(res['trades'])
        
        print("\n" + "!"*50)
        print(f"VERDICT: {label} [{slice_name}]")
        print("!"*50)
        print(f"Robust Score:   {score:.3f}")
        print(f"Return:         {res['total_return']:.2%}")
        print(f"Win Rate:       {res['win_rate']:.1%}")
        print(f"Trades:         {res['trades']}")
        print(f"Max Drawdown:   {res['max_drawdown']:.2%}")
        print("!"*50)

    def get_feature_list(self, df):
        excludes = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'datetime', 'target_long', 'target_short', 'pred_long', 'pred_short',
                    'vol_delta', 'buy_vol', 'vol_sell', 'trade_count', 'sell_vol', 'dollar_val', 'total_val',
                    'ob_imbalance_last', 'ob_spread_mean', 'ob_bid_depth_mean', 'ob_ask_depth_mean',
                    'ob_micro_dev_std', 'ob_micro_dev_last', 'ob_micro_dev_mean', 'ob_imbalance_mean']
        ob_whitelist = ['ob_imbalance_z', 'ob_spread_ratio', 'ob_bid_impulse', 'ob_ask_impulse', 'ob_depth_ratio', 'ob_spread_bps', 'ob_depth_log_ratio', 'price_liq_div', 'liq_dominance', 'micro_dev_vol', 'ob_imb_trend', 'micro_pressure', 'bid_depth_chg', 'ask_depth_chg', 'spread_z', 'ob_slope_ratio', 'bid_slope_z', 'ob_bid_elasticity', 'ob_ask_elasticity', 'ob_bid_integrity_mean', 'ob_ask_integrity_mean', 'ob_integrity_skew', 'bid_integrity_chg', 'ask_integrity_chg']
        
        all_features = [c for c in df.columns if (c not in excludes and not c.startswith('ob_')) or c in ob_whitelist]
        
        if self.args.microstructure_only:
            tech_blacklist = ['ema_', 'dist_ema_', 'rsi', 'atr']
            filtered = [f for f in all_features if not any(b in f for b in tech_blacklist)]
            print(f"[Research] Microstructure Only: Reduced features from {len(all_features)} to {len(filtered)}")
            return filtered
            
        return all_features

    def objective(self, trial):
        offset, tp, sl = trial.suggest_float('limit_offset_atr', 0.5, 1.5), trial.suggest_float('take_profit_atr', 0.5, 2.5), trial.suggest_float('stop_loss_atr', 1.0, 4.0)
        lr, depth, leaves = trial.suggest_float('learning_rate', 0.01, 0.1, log=True), trial.suggest_int('max_depth', 3, 8), trial.suggest_int('num_leaves', 16, 128)
        thresh = trial.suggest_float('model_threshold', 0.55, 0.85)
        
        min_child = self.args.min_child
        
        conf = copy.deepcopy(CONF)
        conf.strategy.base_limit_offset_atr, conf.strategy.take_profit_atr, conf.strategy.stop_loss_atr = offset, tp, sl
        conf.model.learning_rate, conf.model.max_depth, conf.model.num_leaves = lr, depth, leaves
        conf.model.min_child_samples = min_child
        
        df = self.raw_df.copy()
        df = Labeler(conf).generate_labels(df)
        if df['target_long'].sum() < 50: return -1.0
        
        dev_size = int(len(df) * 0.85)
        df_dev = df.iloc[:dev_size].reset_index(drop=True)
        tscv = TimeSeriesSplit(n_splits=3, test_size=int(len(df_dev)*0.2))
        
        feats = self.get_feature_list(df)
        scores = []
        import lightgbm as lgb
        for train_idx, val_idx in tscv.split(df_dev):
            X_t, y_l, y_s = df_dev.iloc[train_idx][feats], df_dev.iloc[train_idx]['target_long'], df_dev.iloc[train_idx]['target_short']
            p = {'objective':'binary','metric':'auc','verbosity':-1,'n_jobs':-1,
                 'learning_rate':lr,'max_depth':depth,'num_leaves':leaves,
                 'min_child_samples':min_child,'boosting_type':'gbdt',
                 'colsample_bytree': CONF.model.colsample_bytree}
            m_l = lgb.train(p, lgb.Dataset(X_t, label=y_l), num_boost_round=100)
            m_s = lgb.train(p, lgb.Dataset(X_t, label=y_s), num_boost_round=100)
            val_f = df_dev.iloc[val_idx].copy()
            val_f['pred_long'], val_f['pred_short'] = m_l.predict(val_f[feats]), m_s.predict(val_f[feats])
            res = Backtester(conf).run(val_f, threshold=thresh)
            scores.append(res['sortino'] * np.log(res['trades']) if res['trades'] >= 3 else -0.1)
        return np.mean(scores)

    def verify_candidate(self, params, trial_number, val_score):
        conf = copy.deepcopy(CONF)
        conf.strategy.base_limit_offset_atr, conf.strategy.take_profit_atr, conf.strategy.stop_loss_atr = params['limit_offset_atr'], params['take_profit_atr'], params['stop_loss_atr']
        conf.model.learning_rate, conf.model.max_depth, conf.model.num_leaves, conf.model.model_threshold = params['learning_rate'], params['max_depth'], params['num_leaves'], params['model_threshold']
        conf.model.min_child_samples = self.args.min_child
        
        df = self.raw_df.copy()
        df = Labeler(conf).generate_labels(df)
        feats = self.get_feature_list(df)
        mm = ModelManager(conf.model)
        mm.train(df, feats)
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
            'model_long': mm.model_long, 'model_short': mm.model_short, 'features': feats
        }

    def save_champion(self, res, rank):
        save_path = CONF.model.model_dir / f"rank_{rank}"
        save_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(res['model_long'], save_path / "model_long.pkl")
        joblib.dump(res['model_short'], save_path / "model_short.pkl")
        joblib.dump(res['features'], save_path / "features.pkl")
        with open(save_path / "params.json", "w") as f: json.dump(res['params'], f, indent=4)
        with open(save_path / "metrics.json", "w") as f:
            m = {k: v for k,v in res.items() if k not in ['model_long', 'model_short', 'features', 'params']}
            json.dump(m, f, indent=4)
        print(f"  -> Champion Saved to rank_1/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--timeframe", type=str, default="5m")
    parser.add_argument("--ob-levels", type=int, default=200)
    parser.add_argument("--min-child", type=int, default=50) 
    parser.add_argument("--data-dir", type=str, default="data/RAVEUSDT")
    parser.add_argument("--symbol", type=str, default="RAVEUSDT")
    parser.add_argument("--model-dir", type=str, default="models_v4")
    parser.add_argument("--microstructure-only", action="store_true", help="Blacklist technical indicators")
    args = parser.parse_args()
    AutoML(args).run()
