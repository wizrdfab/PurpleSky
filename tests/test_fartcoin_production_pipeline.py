import unittest
import logging
import sys
import os
import pandas as pd
import time
from pathlib import Path
from types import SimpleNamespace

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from live_trading import LiveBot
from config import CONF

# API CREDENTIALS (TESTNET)
API_KEY = os.getenv("BYBIT_API_KEY", "DZh7qdh2dTfw328ASi")
API_SECRET = os.getenv("BYBIT_API_SECRET", "Y4Jhn5z2MMi0LGN174RRNXLeE2NrRpGZ3mcf")
SYMBOL = "FARTCOINUSDT"
MODEL_DIR = "models/FARTCOINUSDT/rank_1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FartcoinProdTest")

class TestFartcoinProductionPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(f"\n=== STARTING PRODUCTION PIPELINE TEST - {SYMBOL} ===")
        cls.args = SimpleNamespace(
            symbol=SYMBOL,
            model_dir=MODEL_DIR, 
            api_key=API_KEY,
            api_secret=API_SECRET,
            testnet=True,
            timeframe="5m"
        )
        
    def setUp(self):
        # NO MOCKS for ModelManager, FeatureEngine, or joblib/torch
        # We want the real code to run.
        
        print("    Loading Bot with real Model & Features...")
        self.bot = LiveBot(self.args)
        self.bot.warmup_bars = 0 # Disable warmup for test
        
        # 1. Populate History with REAL data from exchange
        print("    Fetching real history for feature calculation...")
        klines = self.bot.rest_api.get_public_klines(SYMBOL, "5m", limit=300)
        if not klines:
            self.fail("Failed to fetch history from exchange.")
        
        self.bot.bars = pd.DataFrame(klines)
        # Ensure OB columns exist so calc doesn't fail
        for col in ['ob_spread_mean', 'ob_micro_dev_std']:
            self.bot.bars[col] = 0.0001 # Realistic dummy value
            
        print(f"    History loaded: {len(self.bot.bars)} bars.")

    def test_production_inference(self):
        """Run the full production pipeline: History -> Features -> Model -> Prediction"""
        print("\n[Test] Production Inference Pipeline")
        
        # Latest bar from our fetched history
        last_bar = self.bot.bars.iloc[-1].to_dict()
        
        # Prepare a 'fake' kline message that matches the last bar
        fake_kline = {
            'start': last_bar['timestamp'],
            'open': last_bar['open'],
            'high': last_bar['high'],
            'low': last_bar['low'],
            'close': last_bar['close'],
            'volume': last_bar['volume'],
            'confirm': True
        }
        
        # Manually trigger bar close
        # This will run: finalize() -> concat -> calculate_features -> predict -> execute_logic
        print("    Triggering bar close processing...")
        
        # We catch the logs to verify everything happened
        with self.assertLogs('LiveTrading', level='INFO') as cm:
            self.bot.on_bar_close(fake_kline)
            
            # Check if prediction happened
            pred_logs = [l for l in cm.output if "Preds: Long=" in l]
            self.assertTrue(len(pred_logs) > 0, "Model prediction log not found!")
            
            print(f"    RESULT: {pred_logs[0]}")
            
            # Check if features were printed (our new diagnostic report)
            feat_logs = [l for l in cm.output if ">>> [ALL MODEL FEATURES]" in l]
            self.assertTrue(len(feat_logs) > 0, "Feature diagnostic report not found!")
            print("    [OK] Feature engine and Diagnostic report worked correctly.")

    def test_instrument_sync(self):
        """Verify the bot correctly loaded Fartcoin's specific exchange rules"""
        print("\n[Test] Instrument Specification Sync")
        info = self.bot.instrument_info
        print(f"    Fartcoin Tick Size: {info['tick_size']}")
        print(f"    Fartcoin Min Qty:  {info['min_qty']}")
        
        self.assertIn('tick_size', info)
        self.assertGreater(info['tick_size'], 0)

    def test_taker_flow_accumulation(self):
        """Verify that public trades correctly accumulate volume and calculate ratio"""
        print("\n[Test] Taker Flow Accumulation")
        
        # 1. Simulate some public trades
        trades = {
            'data': [
                {'v': '100', 'S': 'Buy'},
                {'v': '50', 'S': 'Sell'},
                {'v': '150', 'S': 'Buy'}
            ]
        }
        self.bot.on_public_trade(trades)
        
        self.assertEqual(self.bot.bar_taker_total_vol, 300)
        self.assertEqual(self.bot.bar_taker_buy_vol, 250)
        print(f"    Accumulated Vol: Total={self.bot.bar_taker_total_vol}, Buy={self.bot.bar_taker_buy_vol}")
        
        # 2. Trigger bar close and verify ratio calculation in logs
        last_bar = self.bot.bars.iloc[-1].to_dict()
        fake_kline = {
            'start': last_bar['timestamp'], 'open': last_bar['open'], 'high': last_bar['high'], 
            'low': last_bar['low'], 'close': last_bar['close'], 'volume': last_bar['volume'], 
            'confirm': True
        }
        
        with self.assertLogs('LiveTrading', level='INFO') as cm:
            self.bot.on_bar_close(fake_kline)
            
            # Check for Taker Flow log
            flow_logs = [l for l in cm.output if "Taker Flow: Buy Vol=250.00, Total Vol=300.00, Ratio=0.8333" in l]
            self.assertTrue(len(flow_logs) > 0, "Taker Flow ratio log mismatch or missing!")
            print("    [OK] Taker Buy Ratio calculated correctly (0.8333).")
            
            # Check if counters were reset
            self.assertEqual(self.bot.bar_taker_total_vol, 0)
            self.assertEqual(self.bot.bar_taker_buy_vol, 0)
            print("    [OK] Volume counters reset for next bar.")

if __name__ == '__main__':
    unittest.main()
