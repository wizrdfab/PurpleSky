import unittest
import logging
import sys
import time
import os
import pandas as pd
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

sys.path.append(str(Path(__file__).parent.parent))

from live_trading import LiveBot
from config import CONF

# API CREDENTIALS (TESTNET)
API_KEY = os.getenv("BYBIT_API_KEY", "DZh7qdh2dTfw328ASi")
API_SECRET = os.getenv("BYBIT_API_SECRET", "Y4Jhn5z2MMi0LGN174RRNXLeE2NrRpGZ3mcf")
SYMBOL = "BTCUSDT"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestLiveLifecycle")

class TestLiveLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n=== STARTING LIVE LIFECYCLE TEST (PERSISTENCE & EXITS) ===")
        cls.args = SimpleNamespace(
            symbol=SYMBOL,
            model_dir="dummy_model_dir", 
            api_key=API_KEY,
            api_secret=API_SECRET,
            testnet=True,
            timeframe="5m"
        )
        
    def setUp(self):
        # We DO NOT delete virtual_positions.json automatically here for all tests,
        # because Test 02 relies on the file created by Test 01?
        # Actually, standard unittest philosophy is isolation.
        # So we should test persistence within a SINGLE test method to be safe.
        
        if os.path.exists("virtual_positions.json"):
            os.remove("virtual_positions.json")

        self.patcher1 = patch('live_trading.joblib.load')
        self.patcher2 = patch('live_trading.torch.load')
        self.patcher3 = patch('live_trading.FeatureEngine')
        self.patcher4 = patch('live_trading.CONF.strategy.max_positions', 3) 
        self.patcher5 = patch('live_trading.CONF.strategy.risk_per_trade', 0.001)
        self.patcher6 = patch('live_trading.CONF.strategy.base_limit_offset_atr', -0.05)
        # Short holding period for test (e.g., 5 seconds equivalent)
        self.patcher7 = patch('live_trading.CONF.strategy.max_holding_bars', 1) 

        self.mock_joblib = self.patcher1.start()
        self.mock_torch = self.patcher2.start()
        self.mock_fe = self.patcher3.start()
        self.patcher4.start()
        self.patcher5.start()
        self.patcher6.start()
        self.patcher7.start()
        
        self.create_bot()
        self.cleanup_exchange()

    def create_bot(self):
        with patch('live_trading.ModelManager') as MockMM:
            self.bot = LiveBot(self.args)
            self.bot.model_manager = MockMM.return_value
            self.bot.model_manager.feature_cols = ['close', 'atr']
        self.bot.warmup_bars = 0

    def tearDown(self):
        self.patcher1.stop()
        self.patcher2.stop()
        self.patcher3.stop()
        self.patcher4.stop()
        self.patcher5.stop()
        self.patcher6.stop()
        self.patcher7.stop()
        self.cleanup_exchange()

    def cleanup_exchange(self):
        print("    [Cleanup] Clearing orders and positions...")
        self.bot.rest_api.cancel_all_orders(SYMBOL)
        time.sleep(1)
        positions = self.bot.rest_api.get_positions(SYMBOL)
        if positions:
            for p in positions:
                sz = float(p['size'])
                if sz > 0:
                    side = "Sell" if p['side'] == "Buy" else "Buy"
                    idx = p.get('position_idx', 0)
                    self.bot.rest_api.place_order(SYMBOL, side, "Market", sz, reduce_only=True, position_idx=idx)
        time.sleep(2)

    def trigger_buy(self):
        real_price = self.bot.rest_api.get_current_price(SYMBOL)
        self.bot.model_manager.predict.return_value = pd.DataFrame([{
            'pred_long': 0.8, 'pred_short': 0.1, 'pred_dir_long': 0.9, 'pred_dir_short': 0.1
        }])
        self.mock_fe.return_value.calculate_features.return_value = pd.DataFrame([{'close': real_price, 'atr': real_price * 0.05}])
        
        fake_kline = {
            'start': int(time.time()*1000), 
            'open': real_price, 'high': real_price, 'low': real_price, 'close': real_price, 
            'volume': 1000, 'confirm': True
        }
        self.bot.on_bar_close(fake_kline)
        return real_price

    def test_01_persistence_restart(self):
        """Test: Bot Crash & Restart -> Remembers Trade"""
        print("\n[Test 01] Persistence & Restart")
        
        # 1. Open Trade with Bot A
        print("    Bot A: Buying...")
        self.trigger_buy()
        time.sleep(2)
        
        trade_id = self.bot.vpm.trades[0].trade_id
        print(f"    Trade Opened: {trade_id}")
        
        # Verify File Exists
        self.assertTrue(os.path.exists("virtual_positions.json"), "JSON file should exist")
        
        # 2. Kill Bot A (Simulate Crash)
        print("    Simulating Crash (Deleting Bot A)...")
        del self.bot
        
        # 3. Start Bot B (New Instance)
        print("    Bot B: Starting up...")
        self.create_bot() # Reloads from disk
        
        # 4. Verify Bot B knows the trade
        self.assertEqual(len(self.bot.vpm.trades), 1, "Bot B should load 1 trade")
        self.assertEqual(self.bot.vpm.trades[0].trade_id, trade_id, "Trade ID should match")
        print("    [OK] Bot B remembered the trade.")
        
        # 5. Verify Bot B manages it (Reconcile)
        # Should NOT open new positions, should just verify match
        print("    Bot B: Reconciling...")
        self.bot.reconcile_positions()
        time.sleep(2)
        
        positions = self.bot.rest_api.get_positions(SYMBOL)
        long_pos = next((p for p in positions if p['side'] == 'Buy'), None)
        self.assertIsNotNone(long_pos)
        # Should be exactly what we opened (0.001 or min qty), not double
        # Since logic is "Diff = Target - Actual", if Target=1 and Actual=1, Diff=0. No trade.
        print(f"    Exchange Position: {long_pos['size']}")
        # We can't assert exact size easily without knowing min qty logic, but we assume it didn't double.
        # If it doubled, we'd see 2x. 
        
    def test_02_time_exit(self):
        """Test: Trade expires after max_holding_bars"""
        print("\n[Test 02] Time Based Exit")
        
        # 1. Open Trade
        self.trigger_buy()
        time.sleep(2)
        self.assertEqual(len(self.bot.vpm.trades), 1)
        
        # 2. Simulate Time Passing
        # Config max_holding_bars = 1 (Patched in setUp)
        # Timeframe = 5m = 300s.
        # We need to simulate: Entry Time + 301 seconds.
        
        entry_time = self.bot.vpm.trades[0].entry_time
        future_time = entry_time + 305
        
        print(f"    Simulating Future: {future_time} (Entry: {entry_time})")
        
        # We need to trigger the check. It happens in manage_exits -> check_time_exits
        # We can call manage_exits directly with a patched time
        
        with patch('time.time', return_value=future_time):
            self.bot.manage_exits()
            
        # 3. Verify Closure in VPM
        self.assertEqual(len(self.bot.vpm.trades), 0, "Trade should be closed due to time exit")
        print("    [OK] VPM closed the trade.")
        
        # 4. Verify Execution (Reconcile)
        self.bot.reconcile_positions()
        time.sleep(2)
        
        positions = self.bot.rest_api.get_positions(SYMBOL)
        long_qty = sum(float(p['size']) for p in positions if p['side'] == 'Buy')
        self.assertEqual(long_qty, 0.0, "Exchange position should be closed")
        print("    [OK] Exchange position closed.")

if __name__ == '__main__':
    unittest.main()
