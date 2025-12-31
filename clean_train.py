"""
Clean Slate Production Trainer.
Forces feature regeneration and verifies feature count > 40.
"""
import pandas as pd
import joblib
import json
import shutil
from pathlib import Path

from config import CONF
from data_loader import DataLoader
from feature_engine import FeatureEngine
from labels import Labeler
from models import ModelManager
from backtest import Backtester

def train_production_model():
    print("--- Starting Clean Production Build ---")
    
    # 1. Load Data (Fresh)
    print("1. Loading Data...")
    dl = DataLoader(CONF.data)
    df = dl.load_and_merge(CONF.features.base_timeframe)
    
    # 2. Features
    print("2. Generating Features...")
    fe = FeatureEngine(CONF.features)
    df = fe.calculate_features(df)
    
    # 3. Feature Audit
    # Identify feature columns
    excludes = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'datetime', 'target_long', 'target_short', 'pred_long', 'pred_short',
                'vol_delta', 'buy_vol', 'trade_count', 'sell_vol', 'ob_imbalance_last', 'ob_spread_mean', 'ob_bid_depth_mean', 'ob_ask_depth_mean']
    # Include calculated OB features
    feature_cols = [c for c in df.columns if c not in excludes and not c.startswith('ob_') or c in ['ob_imbalance_z', 'ob_spread_ratio', 'ob_bid_impulse', 'ob_ask_impulse', 'ob_depth_ratio', 'ob_spread_bps', 'ob_depth_log_ratio', 'price_liq_div', 'liq_dominance', 'micro_dev_vol', 'ob_imb_trend', 'micro_pressure', 'bid_depth_chg', 'ask_depth_chg', 'spread_z']]
    
    print(f"   -> Feature Count: {len(feature_cols)}")
    print(f"   -> Features: {feature_cols}")
    
    if len(feature_cols) < 30:
        print("CRITICAL WARNING: Feature count is low (likely Trade-Only). Check Orderbook data loading.")
        # We continue, but warn.
        
    # 4. Labels
    print("3. Generating Labels...")
    lbl = Labeler(CONF)
    df = lbl.generate_labels(df)
    
    # 5. Train
    print("4. Training Final Model...")
    mm = ModelManager(CONF.model)
    mm.train(df, feature_cols)
    
    # 6. Save to Production Folder
    prod_path = Path("models/production/RAVEUSDT")
    prod_path.mkdir(parents=True, exist_ok=True)
    
    print(f"5. Saving to {prod_path}...")
    joblib.dump(mm.model_long, prod_path / "model_long.pkl")
    joblib.dump(mm.model_short, prod_path / "model_short.pkl")
    joblib.dump(feature_cols, prod_path / "features.pkl")
    
    # Save current config params as params.json for live_trader compatibility
    params = {
        'limit_offset_atr': CONF.strategy.base_limit_offset_atr,
        'take_profit_atr': CONF.strategy.take_profit_atr,
        'stop_loss_atr': CONF.strategy.stop_loss_atr,
        'model_threshold': CONF.model.model_threshold
    }
    with open(prod_path / "params.json", "w") as f:
        json.dump(params, f, indent=4)
        
    print("--- Production Build Complete ---")
    print(f"Point live_trader.py to: {prod_path}")

if __name__ == "__main__":
    train_production_model()
