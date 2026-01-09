"""
Robustness Check: Rolling Information Coefficient (IC).
This tells us if the model actually 'knows' something or just got lucky.
"""
import argparse
import pandas as pd
import numpy as np
import joblib
import json
from scipy.stats import spearmanr
from pathlib import Path

from config import CONF
from data_loader import DataLoader
from feature_engine import FeatureEngine
from labels import Labeler

def calculate_robustness(model_path):
    path = Path(model_path)
    
    # 1. Load Artifacts
    print(f"Loading model from {path}...")
    model = joblib.load(path / "alpha_model.pkl")
    features = joblib.load(path / "features.pkl")
    
    # 2. Load Data
    print("Loading data...")
    dl = DataLoader(CONF.data)
    df = dl.load_and_merge("5m")
    
    print("Generating Features & Targets...")
    fe = FeatureEngine(CONF.features)
    df = fe.calculate_features(df)
    lbl = Labeler(CONF)
    df = lbl.generate_labels(df)
    
    # Drop NaNs for correlation check
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # 3. Predict
    print("Generating Alpha Predictions...")
    preds = model.predict(df[features])
    df['pred_alpha'] = preds
    
    # 4. Rolling IC Analysis
    # We calculate the Spearman Correlation between Prediction and Real Future Return
    # over a rolling window (e.g., every 500 bars ~ 2 days)
    
    window_size = 500
    rolling_ic = []
    dates = []
    
    # We iterate through the data
    for i in range(window_size, len(df), window_size):
        chunk = df.iloc[i-window_size:i]
        
        # Calculate IC for this chunk
        ic, _ = spearmanr(chunk['pred_alpha'], chunk['target_return'])
        rolling_ic.append(ic)
        dates.append(chunk.index[-1])
        
    # 5. Statistics
    avg_ic = np.mean(rolling_ic)
    std_ic = np.std(rolling_ic)
    ic_sharpe = avg_ic / std_ic if std_ic > 0 else 0
    
    print("\n" + "="*60)
    print("ROBUSTNESS REPORT (Rolling IC)")
    print("="*60)
    print(f"Global IC:      {avg_ic:.4f}  (> 0.05 is good, > 0.10 is excellent)")
    print(f"IC Volatility:  {std_ic:.4f}  (Lower is better)")
    print(f"IC Sharpe:      {ic_sharpe:.2f}  (Stability metric, > 1.0 is robust)")
    print("-" * 60)
    print("Rolling IC by Period (Every ~48 hours):")
    
    pos_periods = 0
    total_periods = len(rolling_ic)
    
    for date, ic in zip(dates, rolling_ic):
        status = "[PASS]" if ic > 0.05 else "[WEAK]" if ic > 0 else "[FAIL]"
        print(f"{date}: {ic:+.4f} {status}")
        if ic > 0: pos_periods += 1
        
    consistency = pos_periods / total_periods
    print("-" * 60)
    print(f"Consistency: {consistency:.1%}")
    
    if avg_ic > 0.05 and consistency > 0.7:
        print("\nVERDICT: [SAFE TO DEPLOY] - Model shows consistent predictive power.")
    elif avg_ic > 0.02 and consistency > 0.5:
        print("\nVERDICT: [RISKY] - Model has edge but is unstable.")
    else:
        print("\nVERDICT: [DO NOT DEPLOY] - Model results likely due to overfitting.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="models_v3/RAVEUSDT/stepwise_v1")
    args = parser.parse_args()
    
    calculate_robustness(args.model_path)
