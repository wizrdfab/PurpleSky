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
logger = logging.getLogger("TestLiveTP")

class TestLiveTakeProfits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n=== STARTING LIVE TAKE PROFIT TEST (TESTNET) ===")
        cls.args = SimpleNamespace(
            symbol=SYMBOL,
            model_dir="dummy_model_dir", 
            api_key=API_KEY,
            api_secret=API_SECRET,
            testnet=True,
            timeframe="5m"
        )
        
    def setUp(self):
        if os.path.exists("virtual_positions.json"):
            os.remove("virtual_positions.json")

        self.patcher1 = patch('live_trading.joblib.load')
        self.patcher2 = patch('live_trading.torch.load')
        self.patcher3 = patch('live_trading.FeatureEngine')
        self.patcher4 = patch('live_trading.CONF.strategy.max_positions', 3) 
        self.patcher5 = patch('live_trading.CONF.strategy.risk_per_trade', 0.001) 
        # Aggressive Limit Offset to ensure immediate fill (Marketable Limit)
        self.patcher6 = patch('live_trading.CONF.strategy.base_limit_offset_atr', -0.05) 
        
        # Patch TP ATR to be TINY so it triggers almost instantly
        # standard is 1.2, we use 0.001 for test
        self.patcher7 = patch('live_trading.CONF.strategy.take_profit_atr', 0.001)

        self.mock_joblib = self.patcher1.start()
        self.mock_torch = self.patcher2.start()
        self.mock_fe = self.patcher3.start()
        self.patcher4.start()
        self.patcher5.start()
        self.patcher6.start()
        self.patcher7.start()
        
        with patch('live_trading.ModelManager') as MockMM:
            self.bot = LiveBot(self.args)
            self.bot.model_manager = MockMM.return_value
            self.bot.model_manager.feature_cols = ['close', 'atr']
            
        self.bot.warmup_bars = 0
        self.cleanup_exchange()

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

    def get_real_price(self):
        return self.bot.rest_api.get_current_price(SYMBOL)

    def trigger_signal(self, pred_row, price_override=None):
        real_price = price_override if price_override else self.get_real_price()
        
        self.bot.model_manager.predict.return_value = pd.DataFrame([pred_row])
        # Use LARGER ATR to prevent premature TP hit from market noise
        # 0.05 * 0.001 (TP factor) = 0.00005 * Price. Still small, but safer.
        # Actually, let's make it robust. 1.0% ATR.
        self.mock_fe.return_value.calculate_features.return_value = pd.DataFrame([{'close': real_price, 'atr': real_price * 0.05}])
        
        fake_kline = {
            'start': int(time.time()*1000), 
            'open': real_price, 'high': real_price, 'low': real_price, 'close': real_price, 
            'volume': 1000, 'confirm': True
        }
        self.bot.on_bar_close(fake_kline)
        return real_price

    def test_01_standard_take_profit(self):
        """Test: Enter -> TP Hit -> Close"""
        print("\n[Test 01] Standard Take Profit Trigger")
        
        # 1. Enter Long
        print("    Entering Long...")
        entry_price = self.trigger_signal({
            'pred_long': 0.8, 'pred_short': 0.1, 'pred_dir_long': 0.9, 'pred_dir_short': 0.1
        })
        time.sleep(2)
        
        # Check VPM
        self.assertEqual(len(self.bot.vpm.trades), 1)
        trade = self.bot.vpm.trades[0]
        print(f"    Trade Entry: {trade.entry_price}, TP: {trade.take_profit}")
        
        # 2. Simulate Price Move to Hit TP
        # We manually trigger 'prune_dead_trades' with a price above TP
        print("    Simulating Price Hit > TP...")
        
        # Force the bot's internal price check to see the TP price
        # LiveBot.reconcile_positions calls 'get_current_price'
        # We can just call 'prune_dead_trades' directly for verification, 
        # OR patch get_current_price to return a winner.
        
        winning_price = trade.take_profit + 100.0
        
        # Let's verify via the actual bot logic path
        # The bot checks dead trades at end of reconcile or on bar close?
        # It checks in 'reconcile_positions' -> 'prune_dead_trades'
        
        with patch.object(self.bot.rest_api, 'get_current_price', return_value=winning_price):
            self.bot.reconcile_positions()
            
        time.sleep(2)
        
        # 3. Verify Closure
        self.assertEqual(len(self.bot.vpm.trades), 0, "Trade should be closed in VPM")
        
        # 4. Verify Exchange Position Reduced
        positions = self.bot.rest_api.get_positions(SYMBOL)
        long_qty = sum(float(p['size']) for p in positions if p['side'] == 'Buy')
        print(f"    Remaining Exchange Long: {long_qty}")
        self.assertEqual(long_qty, 0.0, "Exchange position should be closed")

    def test_02_partial_fill_tp(self):
        """Test: Partial Fill -> TP Set for Partial Size -> TP Hit"""
        print("\n[Test 02] Partial Fill Take Profit")
        
        # 1. Place Limit Buy (Standard)
        print("    Placing Limit Buy...")
        # Use standard params, not aggressive market
        start_price = self.get_real_price()
        self.trigger_signal({
            'pred_long': 0.7, 'pred_short': 0.1, 'pred_dir_long': 0.6, 'pred_dir_short': 0.1
        })
        
        self.assertEqual(len(self.bot.pending_limits), 1)
        oid = list(self.bot.pending_limits.keys())[0]
        link_id = self.bot.pending_limits[oid]['link_id']
        
        # 2. Inject Partial Fill (e.g., 0.002 BTC)
        print("    Injecting Partial Fill (0.002)...")
        exec_msg = {
            'data': [{
                'symbol': SYMBOL, 'side': 'Buy', 'execQty': '0.002', 
                'execPrice': str(start_price), 'orderLinkId': link_id
            }]
        }
        self.bot.on_execution_update(exec_msg)
        self.bot.reconcile_positions()
        time.sleep(2)
        
        # Verify VPM has partial trade
        self.assertEqual(len(self.bot.vpm.trades), 1)
        trade = self.bot.vpm.trades[0]
        self.assertAlmostEqual(trade.size, 0.002)
        print(f"    Partial Trade Recorded: {trade.size} @ {trade.entry_price}")
        
        # 3. Trigger TP for this partial
        print("    Triggering TP for Partial...")
        winning_price = trade.take_profit + 100.0
        
        with patch.object(self.bot.rest_api, 'get_current_price', return_value=winning_price):
            self.bot.reconcile_positions()
            
        time.sleep(2)
        
        # 4. Verify Partial Closed
        self.assertEqual(len(self.bot.vpm.trades), 0)
        
        # Verify Exchange: Should be 0 (since we filled 0.002 and closed 0.002)
        positions = self.bot.rest_api.get_positions(SYMBOL)
        long_qty = sum(float(p['size']) for p in positions if p['side'] == 'Buy')
        print(f"    Remaining Exchange Long: {long_qty}")
        self.assertEqual(long_qty, 0.0)
        
        # 5. Cancel the rest of the limit (Cleanup)
        self.bot.rest_api.cancel_order(SYMBOL, oid)

    def test_03_multi_fill_multi_tp(self):
        """Test: 2 Separate Fills -> 2 Separate TPs -> Hit 1, Keep 1"""
        print("\n[Test 03] Multi-Fill Multi-TP Logic")
        
        # 1. Place Limit Buy
        start_price = self.get_real_price()
        self.trigger_signal({
            'pred_long': 0.7, 'pred_short': 0.1, 'pred_dir_long': 0.6, 'pred_dir_short': 0.1
        })
        oid = list(self.bot.pending_limits.keys())[0]
        link_id = self.bot.pending_limits[oid]['link_id']
        
        # 2. Fill #1 @ Price X
        print("    Fill #1 (0.001 @ Base)...")
        self.bot.on_execution_update({'data': [{'symbol': SYMBOL, 'side': 'Buy', 'execQty': '0.001', 'execPrice': str(start_price), 'orderLinkId': link_id}]})
        
        # 3. Fill #2 @ Price X+50 (Worse price, Higher TP)
        print("    Fill #2 (0.001 @ Base+50)...")
        worse_price = start_price + 50.0
        self.bot.on_execution_update({'data': [{'symbol': SYMBOL, 'side': 'Buy', 'execQty': '0.001', 'execPrice': str(worse_price), 'orderLinkId': link_id}]})
        
        self.bot.reconcile_positions()
        time.sleep(2)
        
        # Verify 2 Trades in VPM
        self.assertEqual(len(self.bot.vpm.trades), 2)
        t1 = self.bot.vpm.trades[0]
        t2 = self.bot.vpm.trades[1]
        print(f"    T1 TP: {t1.take_profit}, T2 TP: {t2.take_profit}")
        self.assertNotEqual(t1.take_profit, t2.take_profit)
        
        # 4. Price moves to Hit T1 but NOT T2
        # T1 entry was lower, so T1 TP should be lower
        # T2 entry was higher, so T2 TP is higher
        
        target_price = t1.take_profit + 5.0 # Definitely beats T1
        # Ensure it's below T2? 
        # Since we used tiny ATR (0.001), the TPs are super close to entry.
        # entry1 ~ X, TP1 ~ X+small
        # entry2 ~ X+50, TP2 ~ X+50+small
        # So a price of X+10 will hit TP1 but NOT TP2.
        
        print(f"    Simulating Price {target_price} (Hits T1, Misses T2)...")
        
        with patch.object(self.bot.rest_api, 'get_current_price', return_value=target_price):
            self.bot.reconcile_positions()
            
        time.sleep(2)
        
        # 5. Verify T1 Closed, T2 Open
        self.assertEqual(len(self.bot.vpm.trades), 1, "Should have 1 trade left")
        self.assertEqual(self.bot.vpm.trades[0].entry_price, worse_price, "Trade 2 should remain")
        
        # Verify Exchange Size: Should be 0.001 (0.002 filled - 0.001 closed)
        positions = self.bot.rest_api.get_positions(SYMBOL)
        long_qty = sum(float(p['size']) for p in positions if p['side'] == 'Buy')
        print(f"    Remaining Exchange Long: {long_qty}")
        self.assertAlmostEqual(long_qty, 0.001)

if __name__ == '__main__':
    unittest.main()
