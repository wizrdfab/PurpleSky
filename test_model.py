
import argparse
import json
import pandas as pd
import numpy as np
import torch
import joblib
from pathlib import Path
from config import GlobalConfig, ModelConfig
from data_loader import DataLoader
from feature_engine import FeatureEngine
from models import ModelManager, LSTMModel, GatingNetwork
from backtest import Backtester

def load_mm_from_path(model_dir: Path, conf: ModelConfig):
    mm = ModelManager(conf)
    
    # Load feature list
    mm.feature_cols = joblib.load(model_dir / "features.pkl")
    context_keywords = ['atr', 'vol', 'rsi', 'std', 'slope']
    mm.context_cols = [c for c in mm.feature_cols if any(k in c.lower() for k in context_keywords)]
    if not mm.context_cols: mm.context_cols = mm.feature_cols[:10] 
    
    # Load LightGBM models
    mm.model_long = joblib.load(model_dir / "model_long.pkl")
    mm.model_short = joblib.load(model_dir / "model_short.pkl")
    
    if (model_dir / "dir_model_long.pkl").exists():
        mm.dir_model_long = joblib.load(model_dir / "dir_model_long.pkl")
        mm.dir_model_short = joblib.load(model_dir / "dir_model_short.pkl")
        
    # Load LSTM components
    if (model_dir / "lstm_model.pth").exists():
        mm.lstm_model = LSTMModel(
            input_size=len(mm.feature_cols),
            hidden_size=conf.lstm_hidden_size,
            num_layers=conf.lstm_layers,
            dropout=conf.lstm_dropout
        ).to(mm.device)
        mm.lstm_model.load_state_dict(torch.load(model_dir / "lstm_model.pth", map_location=mm.device, weights_only=True))
        
        mm.gating_model = GatingNetwork(input_size=len(mm.context_cols)).to(mm.device)
        mm.gating_model.load_state_dict(torch.load(model_dir / "gating_model.pth", map_location=mm.device, weights_only=True))
        
        mm.scaler = joblib.load(model_dir / "scaler.pkl")
        
    return mm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str, help="Path to rank_1 folder")
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--timeframe", type=str, default="5m")
    parser.add_argument("--ob-levels", type=int, default=50)
    parser.add_argument("--aggressive-only", action="store_true")
    args = parser.parse_args()
    
    model_path = Path(args.model_dir)
    params_path = model_path / "params.json"
    if not params_path.exists():
        print(f"Error: {params_path} not found")
        return
        
    with open(params_path, "r") as f:
        params = json.load(f)
        
    # Initialize Config
    from config import GlobalConfig
    conf = GlobalConfig()
    conf.data.symbol = args.symbol
    conf.data.data_dir = Path(args.data_dir)
    conf.data.ob_levels = args.ob_levels
    
    # Strategy Params from saved model
    conf.strategy.base_limit_offset_atr = params.get('limit_offset_atr', conf.strategy.base_limit_offset_atr)
    conf.strategy.take_profit_atr = params.get('take_profit_atr', conf.strategy.take_profit_atr)
    conf.strategy.stop_loss_atr = params.get('stop_loss_atr', conf.strategy.stop_loss_atr)
    
    # Model Config
    conf.model.learning_rate = params.get('learning_rate', conf.model.learning_rate)
    conf.model.max_depth = params.get('max_depth', conf.model.max_depth)
    conf.model.num_leaves = params.get('num_leaves', conf.model.num_leaves)
    conf.model.model_threshold = params.get('model_threshold', conf.model.model_threshold)
    conf.model.aggressive_threshold = params.get('aggressive_threshold', conf.model.aggressive_threshold)
    conf.model.direction_threshold = params.get('direction_threshold', conf.model.direction_threshold)
    
    # Load Data
    print(f"Loading data for {args.symbol} from {args.data_dir}...")
    dl = DataLoader(conf.data)
    df = dl.load_and_merge(args.timeframe)
    if df.empty:
        print("Error: No data loaded")
        return
        
    fe = FeatureEngine(conf.features)
    df = fe.calculate_features(df)
    
    # Slice last 15%
    split_idx = int(len(df) * 0.85)
    df_test = df.iloc[split_idx:].copy()
    print(f"Holdout Set: {len(df_test)} bars")
    
    # Load Model
    mm = load_mm_from_path(model_path, conf.model)
    
    # Predict
    print("Generating predictions...")
    df_pred = mm.predict(df_test)
    
    # Backtest
    bt = Backtester(conf)
    
    # Aggressive Only Logic
    threshold = conf.model.model_threshold
    if args.aggressive_only:
        print("MODE: Aggressive Only (Limit orders disabled)")
        # Force threshold > 1.0 to disable limit entries in the Numba kernel
        threshold = 2.0 
    
    res = bt.run(df_pred, threshold=threshold)
    
    print("\n" + " [ MODEL TEST PERFORMANCE - 15% HOLDOUT ] ".center(60, "="))
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

if __name__ == "__main__":
    main()
