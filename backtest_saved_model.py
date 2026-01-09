"""
Backtest Saved Model.
Loads a specific model/params and runs a full simulation.
"""
import argparse
import pandas as pd
import numpy as np
import joblib
import json
import lightgbm as lgb
from pathlib import Path

from config import CONF
from data_loader import DataLoader
from feature_engine import FeatureEngine
from backtest import Backtester
from labels import Labeler

def load_data(symbol, timeframe):
    print(f"Loading data for {symbol} {timeframe}...")
    dl = DataLoader(CONF.data)
    # Force reload to ensure we get fresh data if available
    df = dl.load_and_merge(timeframe)
    
    print("Generating Features...")
    fe = FeatureEngine(CONF.features)
    df = fe.calculate_features(df)
    
    # We need the 'target_long'/'target_short' for the backtester logic (even if we don't train on them)
    # The current backtester uses them for debug or simple logic, but primarily uses 'pred_long'/'pred_short'
    # derived below.
    # However, the Labeler also adds 'target_dir_long' etc which might be needed if the backtester checks them.
    # Let's run Labeler just to be safe and consistent with training structure.
    lbl = Labeler(CONF)
    df = lbl.generate_labels(df)
    
    return df

def run_backtest(model_path):
    path = Path(model_path)
    
    # 1. Load Artifacts
    print(f"Loading model from {path}...")
    model = joblib.load(path / "alpha_model.pkl")
    features = joblib.load(path / "features.pkl")
    with open(path / "strategy_params.json", "r") as f:
        strat_params = json.load(f)
        
    print(f"Strategy Params: {strat_params}")
    
    # 2. Setup Configuration
    # Apply saved strategy params to Global Config
    CONF.strategy.take_profit_atr = strat_params['take_profit_atr']
    CONF.strategy.stop_loss_atr = strat_params['stop_loss_atr']
    CONF.strategy.base_limit_offset_atr = strat_params['limit_offset_atr']
    threshold = strat_params['model_threshold']
    
    # 3. Load Data
    # Assuming symbol/timeframe are in the path or we default
    df = load_data("RAVEUSDT", "5m")
    
    # 4. Generate Predictions
    print("Predicting...")
    # Check for missing columns
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features in dataset: {missing}")
        
    preds = model.predict(df[features])
    df['pred_return'] = preds
    
    # 5. Signal Generation
    # Long if predicted return > threshold, Short if < -threshold
    df['pred_long'] = (df['pred_return'] > threshold).astype(float)
    df['pred_short'] = (df['pred_return'] < -threshold).astype(float)
    
    # 6. Run Simulation
    print("Running Event-Driven Backtest...")
    bt = Backtester(CONF)
    res = bt.run(df, threshold=0.5) # Threshold already applied to cols
    
    # 7. Report
    print("\n" + "="*50)
    print(f"FULL BACKTEST RESULTS ({len(df)} bars)")
    print("="*50)
    print(f"Total Return: {res['total_return']:.2%}")
    print(f"Final Equity: ${res['final_equity']:.2f}")
    print(f"Trades:       {res['trades']}")
    print(f"Win Rate:     {res['win_rate']:.2%}")
    print(f"Max Drawdown: {res['max_drawdown']:.2%}")
    print(f"Sortino:      {res['sortino']:.2f}")
    print("="*50 + "\n")
    
    # Split Analysis (Train vs Test approximation)
    split_idx = int(len(df) * 0.80)
    
    # We can't easily split the Backtester object results retrospectively 
    # without modifying Backtester to return per-trade data.
    # But we can inspect the 'history' from the backtester if we modify it or access it.
    # The current Backtester returns a dict summary.
    # Let's hacking access to internal list if possible, or just rely on full output.
    # For now, full output is good.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="models_v3/RAVEUSDT/stepwise_v1")
    args = parser.parse_args()
    
    run_backtest(args.model_path)
