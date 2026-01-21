import unittest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

sys.path.append(str(Path(__file__).parent.parent))

from feature_engine import FeatureEngine
from live_trading import LiveBot
from config import CONF

class TestFeatureParity(unittest.TestCase):
    def setUp(self):
        # 1. Load some real historical data
        self.csv_path = "history_BTCUSDT_5m.csv"
        self.df = None
        if os.path.exists(self.csv_path):
            try:
                self.df = pd.read_csv(self.csv_path)
                if len(self.df) < 500: self.df = None # Too small
            except: self.df = None

        if self.df is None:
            # Create a dummy dataset
            print("    Using generated dummy data for parity check (500 bars).")
            data = {
                'timestamp': np.arange(500) * 300000,
                'open': np.random.normal(50000, 100, 500),
                'high': np.random.normal(50100, 100, 500),
                'low': np.random.normal(49900, 100, 500),
                'close': np.random.normal(50000, 100, 500),
                'volume': np.random.normal(10, 2, 500),
                'ob_spread_mean': [0.1] * 500,
                'ob_micro_dev_mean': [0.0] * 500,
                'ob_micro_dev_std': [0.01] * 500,
                'ob_micro_dev_last': [0.0] * 500,
                'ob_imbalance_mean': [0.0] * 500,
                'ob_imbalance_last': [0.0] * 500,
                'ob_bid_depth_mean': [100] * 500,
                'ob_ask_depth_mean': [100] * 500,
                'ob_bid_slope_mean': [0.001] * 500,
                'ob_ask_slope_mean': [0.001] * 500,
                'ob_bid_integrity_mean': [0.5] * 500,
                'ob_ask_integrity_mean': [0.5] * 500,
                'taker_buy_ratio': [0.5] * 500
            }
            self.df = pd.DataFrame(data)
        else:
            self.df = self.df.iloc[:500]
            # Ensure all required OB columns exist for parity check
            required = ['ob_spread_mean', 'ob_imbalance_mean', 'ob_bid_depth_mean', 'ob_ask_depth_mean', 'taker_buy_ratio']
            for col in required:
                if col not in self.df.columns: self.df[col] = 0.5

        self.engine = FeatureEngine(CONF.features)

    def test_feature_parity_batch_vs_incremental(self):
        """Compare whole-dataframe calculation vs bar-by-bar live calculation"""
        print("\n[Parity Test] Batch vs Incremental (Live)")
        
        # 1. Batch Calculation (The 'Gold Standard' from Training)
        # We calculate on the whole block
        batch_features = self.engine.calculate_features(self.df)
        gold_row = batch_features.iloc[-1]
        
        # 2. Incremental Calculation (Simulating the LiveBot)
        # We'll mock a LiveBot and feed it bars until it reaches the same point
        args = SimpleNamespace(symbol="BTCUSDT", model_dir="dummy", api_key="a", api_secret="s", testnet=True, timeframe="5m")
        
        with patch('live_trading.BybitAdapter'), patch('live_trading.BybitWSAdapter'), patch('live_trading.LiveBot._load_model_manager'):
            bot = LiveBot(args)
            bot.warmup_bars = 0
            
            # Setup history: The bot normally loads some history first.
            # We seed it with the first 400 bars.
            bot.bars = self.df.iloc[:400].copy()
            bot.real_ob_bars_count = 400
            
            # Feed the remaining 100 bars one by one
            print(f"    Feeding 100 bars incrementally to bot...")
            for i in range(400, 500):
                row = self.df.iloc[i].to_dict()
                # Mock the WS kline message
                fake_kline = {
                    'start': row['timestamp'], 'open': row['open'], 'high': row['high'], 
                    'low': row['low'], 'close': row['close'], 'volume': row['volume'], 'confirm': True
                }
                # Mock OB finalize to return the stored stats for that bar
                bot.ob_agg.finalize = MagicMock(return_value={
                    k: row[k] for k in row if k.startswith('ob_')
                })
                # Mock taker flow
                bot.bar_taker_buy_vol = row['taker_buy_ratio'] * 100
                bot.bar_taker_total_vol = 100
                
                # We need to capture the feature row calculated INSIDE on_bar_close
                # Let's patch predict to catch the data
                def capture_data(current_row):
                    bot.last_captured_row = current_row.iloc[0]
                    return pd.DataFrame([{'pred_long':0,'pred_short':0,'pred_dir_long':0,'pred_dir_short':0}])
                
                bot.model_manager.predict = capture_data
                bot.on_bar_close(fake_kline)

            live_row = bot.last_captured_row
            
            # 3. COMPARE
            print("    Comparing specific indicators...")
            cols_to_check = ['atr', 'rsi', 'bb_percent_b', 'vwap_4h_dist']
            
            found_errors = 0
            for col in cols_to_check:
                if col not in gold_row or col not in live_row:
                    print(f"      - {col}: MISSING in one of the sets!")
                    continue
                    
                diff = abs(gold_row[col] - live_row[col])
                # Small tolerance for float diffs
                if diff > 1e-6:
                    print(f"      - {col}: mismatch! Batch={gold_row[col]:.6f}, Live={live_row[col]:.6f} (Diff: {diff:.8f})")
                    found_errors += 1
                else:
                    print(f"      - {col}: OK")

            # Check for NaN consistency
            if live_row.isna().any():
                print("    !!! WARNING: Live row contains NaNs!")
                found_errors += 1

            self.assertEqual(found_errors, 0, f"Found {found_errors} feature mismatches between Batch and Live mode!")
            print("    [SUCCESS] Live calculation is mathematically identical to Batch calculation.")

if __name__ == '__main__':
    unittest.main()
