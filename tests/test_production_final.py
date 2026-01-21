import unittest
import logging
import sys
import os
import pandas as pd
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
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
logger = logging.getLogger("FinalProdTest")

class TestProductionFinal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(f"\n=== STARTING FINAL PRODUCTION READINESS TEST - {SYMBOL} ===")
        cls.args = SimpleNamespace(
            symbol=SYMBOL,
            model_dir=MODEL_DIR, 
            api_key=API_KEY,
            api_secret=API_SECRET,
            testnet=True,
            timeframe="5m"
        )
        
    def setUp(self):
        # Fresh state for VPM and History
        if os.path.exists("virtual_positions.json"):
            os.remove("virtual_positions.json")
        
        hist_file = f"history_{SYMBOL}_{self.args.timeframe}.csv"
        if os.path.exists(hist_file):
            os.remove(hist_file)
        
        # Cleanup exchange first
        temp_bot = LiveBot(self.args)
        temp_bot.rest_api.cancel_all_orders(SYMBOL)
        pos = temp_bot.rest_api.get_positions(SYMBOL)
        for p in pos:
            if float(p['size']) > 0:
                side = "Sell" if p['side'] == "Buy" else "Buy"
                temp_bot.rest_api.place_order(SYMBOL, side, "Market", p['size'], reduce_only=True, position_idx=p['position_idx'])
        time.sleep(2)
        del temp_bot

    def test_full_production_cycle(self):
        """End-to-End Test: History -> Integrity -> LSTM -> Execution -> Persistence"""
        
        # 1. Initialize Bot & Fetch 10,000 bars
        print("\n[Step 1] Initializing Bot & Fetching 10,000 bars...")
        bot = LiveBot(self.args)
        bot.warmup_bars = 0
        
        # Explicitly trigger history load (it's normally inside bot.run())
        print("    Starting recursive history download (this may take 10-20 seconds)...")
        bot._load_history()
        
        self.assertGreaterEqual(len(bot.bars), 10000, "Should have fetched 10,000 bars")
        print(f"    [OK] Fetched {len(bot.bars)} bars successfully.")

        # 2. Trigger Bar Close & Check Integrity/Inference
        print("\n[Step 2] Triggering bar close (Integrity & LSTM Inference)...")
        last_bar = bot.bars.iloc[-1].to_dict()
        fake_kline = {
            'start': last_bar['timestamp'], 'open': last_bar['open'], 'high': last_bar['high'], 
            'low': last_bar['low'], 'close': last_bar['close'], 'volume': last_bar['volume'], 
            'confirm': True
        }
        
        # We catch logs to verify the Data Integrity report
        with self.assertLogs('LiveTrading', level='INFO') as cm:
            bot.on_bar_close(fake_kline)
            
            # Verify Integrity Report headers exist in logs
            self.assertTrue(any(">>> [SYSTEM METRICS]" in l for l in cm.output))
            self.assertTrue(any(">>> [ALL MODEL FEATURES]" in l for l in cm.output))
            
            # Verify Prediction happened
            pred_logs = [l for l in cm.output if "Preds: Long=" in l]
            self.assertTrue(len(pred_logs) > 0)
            print(f"    [OK] Inference Success: {pred_logs[0]}")

        # 3. Force a real Trade to test Execution & Stop placement
        print("\n[Step 3] Executing a real Market Trade on Testnet...")
        # Manually call _execute_trade to force action regardless of current signal
        # Use min size for Fartcoin (usually 1.0)
        bot._execute_trade("Buy", last_bar['close'], last_bar['close']*0.05, "Market", check_debounce=False)
        time.sleep(5) # Increased from 3s to 5s for Testnet stability
        
        # Verify Exchange Position
        pos = bot.rest_api.get_positions(SYMBOL)
        long_pos = next((p for p in pos if p['side'] == 'Buy'), None)
        self.assertIsNotNone(long_pos, "Real position should be open on exchange")
        print(f"    [OK] Position open on Bybit: {long_pos['size']} {SYMBOL}")
        
        # Verify Stop Loss
        orders = bot.rest_api.get_open_orders(SYMBOL)
        stops = [o for o in orders if float(o.get('triggerPrice') or 0) > 0]
        self.assertGreaterEqual(len(stops), 1, "Should have at least 1 Stop Loss order on exchange")
        print(f"    [OK] Stop Loss placed on Bybit: {stops[0]['triggerPrice']}")

        # 4. Verify Persistence (Restart)
        print("\n[Step 4] Verifying Persistence (Bot Restart)...")
        num_trades = len(bot.vpm.trades)
        trade_ids = [t.trade_id for t in bot.vpm.trades]
        del bot
        
        bot_new = LiveBot(self.args)
        self.assertEqual(len(bot_new.vpm.trades), num_trades, f"New bot instance should load {num_trades} existing trades")
        for tid in trade_ids:
            self.assertTrue(any(t.trade_id == tid for t in bot_new.vpm.trades))
        print(f"    [OK] New bot instance recovered {num_trades} trades.")

        # Cleanup
        print("\n[Cleanup] Closing test positions...")
        bot_new.rest_api.cancel_all_orders(SYMBOL)
        pos = bot_new.rest_api.get_positions(SYMBOL)
        for p in pos:
            if float(p['size']) > 0:
                side = "Sell" if p['side'] == "Buy" else "Buy"
                bot_new.rest_api.place_order(SYMBOL, side, "Market", p['size'], reduce_only=True, position_idx=p['position_idx'])
        print("=== FINAL TEST PASSED: SYSTEM IS 100% PRODUCTION READY ===")

if __name__ == '__main__':
    unittest.main()
