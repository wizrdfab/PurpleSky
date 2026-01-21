import unittest
import logging
import sys
import time
import os
import shutil
import pandas as pd
from unittest.mock import MagicMock, patch
from pathlib import Path
from types import SimpleNamespace

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import the module to test
from live_trading import LiveBot
from config import CONF

# API CREDENTIALS (TESTNET)
API_KEY = os.getenv("BYBIT_API_KEY", "DZh7qdh2dTfw328ASi")
API_SECRET = os.getenv("BYBIT_API_SECRET", "Y4Jhn5z2MMi0LGN174RRNXLeE2NrRpGZ3mcf")
SYMBOL = "BTCUSDT"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestLiveBotModule")

class TestLiveBotModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n=== STARTING COMPREHENSIVE LIVE BOT MODULE TEST ===")
        # 1. Setup Mock Args
        cls.args = SimpleNamespace(
            symbol=SYMBOL,
            model_dir="dummy_model_dir", 
            api_key=API_KEY,
            api_secret=API_SECRET,
            testnet=True,
            timeframe="5m"
        )
        
    def setUp(self):
        # Patch dependencies
        self.patcher1 = patch('live_trading.joblib.load')
        self.patcher2 = patch('live_trading.torch.load')
        self.patcher3 = patch('live_trading.FeatureEngine')
        
        self.mock_joblib = self.patcher1.start()
        self.mock_torch = self.patcher2.start()
        self.mock_fe = self.patcher3.start()
        
        # Mock Feature Engine to return valid data
        # Close = 50000, ATR = 500
        self.current_price = 50000.0
        self.mock_fe.return_value.calculate_features.return_value = pd.DataFrame([{'close': self.current_price, 'atr': 500.0}])
        
        # Instantiate Bot with mocked ModelManager
        with patch('live_trading.ModelManager') as MockMM:
            self.bot = LiveBot(self.args)
            self.bot.model_manager = MockMM.return_value
            self.bot.model_manager.feature_cols = ['close', 'atr']
            
        # Bypass Warmup
        self.bot.warmup_bars = 0
            
        # Inject Custom Instrument Info for Testing Rules
        # Tick Size = 0.5, Step = 0.002, Min Qty = 0.002
        self.bot.instrument_info = {
            'tick_size': 0.5,
            'qty_step': 0.002,
            'min_qty': 0.002,
            'min_notional': 5.0
        }
        
        # Clean Exchange
        print(">> Setup: Cleaning Exchange State...")
        self.bot.rest_api.cancel_all_orders(SYMBOL)
        self.close_all_positions()
        self.bot.vpm.trades = []
        self.bot.pending_limits = {}

    def tearDown(self):
        self.patcher1.stop()
        self.patcher2.stop()
        self.patcher3.stop()
        print("\n>> Teardown: Cleaning up...")
        self.bot.rest_api.cancel_all_orders(SYMBOL)
        self.close_all_positions()

    def close_all_positions(self):
        print("    [Cleanup] Closing all positions...")
        for _ in range(5):
            positions = self.bot.rest_api.get_positions(SYMBOL)
            if not positions: break
            
            active = False
            for p in positions:
                sz = float(p['size'])
                if sz > 0:
                    active = True
                    side = "Sell" if p['side'] == "Buy" else "Buy"
                    idx = p.get('position_idx', 0)
                    if idx == 0: continue
                    print(f"      Closing {p['side']} {sz}")
                    self.bot.rest_api.place_order(SYMBOL, side, "Market", sz, reduce_only=True, position_idx=idx)
            
            if not active: break
            time.sleep(2)

    def trigger_bar_close(self, pred_row):
        # Mock Prediction
        self.bot.model_manager.predict.return_value = pd.DataFrame([pred_row])
        
        # Fetch Real Price to ensure orders are valid relative to market
        real_price = self.bot.rest_api.get_current_price(SYMBOL)
        if real_price == 0: real_price = 50000.0
        self.current_price = real_price
        
        # Mock FE with Real Price
        self.mock_fe.return_value.calculate_features.return_value = pd.DataFrame([{'close': real_price, 'atr': real_price * 0.01}])
        
        # Fake Kline
        fake_kline = {
            'start': int(time.time()*1000), 'open': real_price, 'high': real_price, 
            'low': real_price, 'close': real_price, 'volume': 1000, 'confirm': True
        }
        self.bot.on_bar_close(fake_kline)
        return real_price

    def test_01_instrument_rules_rounding(self):
        """Test if Price/Qty are rounded according to custom instrument info"""
        print("\n[Test 01] Instrument Rules (Rounding)")
        
        # Signal: Aggressive Buy (Market)
        # We want to check the Qty calculated inside _execute_trade
        # We can spy on rest_api.place_order
        
        with patch.object(self.bot.rest_api, 'place_order', wraps=self.bot.rest_api.place_order) as mock_place:
            self.trigger_bar_close({
                'pred_long': 0.8, 'pred_short': 0.1, 'pred_dir_long': 0.9, 'pred_dir_short': 0.1
            })
            
            # Verify call args
            self.assertTrue(mock_place.called)
            args, kwargs = mock_place.call_args
            
            qty = float(kwargs['qty'])
            print(f"  Calculated Qty: {qty}")
            
            # Check Step Size (0.002)
            # Use division check to avoid float modulo issues
            steps = qty / 0.002
            self.assertAlmostEqual(steps, round(steps), delta=0.001, msg="Qty not multiple of step 0.002")
            
            # Since it's Market, Price is None or ignored, but let's check Limit logic next

    def test_02_limit_order_flow_sync(self):
        """Test Standard Signal -> Limit Order -> Fill -> Sync"""
        print("\n[Test 02] Limit Order Flow & Sync")
        self.close_all_positions()
        
        # Verify Clean
        pos = self.bot.rest_api.get_positions(SYMBOL)
        total = sum(float(p['size']) for p in pos)
        self.assertEqual(total, 0, f"Failed to clean positions: {pos}")
        
        # 1. Standard Buy Signal
        print("  >> Triggering Standard Signal...")
        real_price = self.trigger_bar_close({
            'pred_long': 0.7, 'pred_short': 0.1, 'pred_dir_long': 0.6, 'pred_dir_short': 0.1
        })
        
        # 2. Verify Limit Order
        self.assertEqual(len(self.bot.pending_limits), 1)
        oid = list(self.bot.pending_limits.keys())[0]
        data = self.bot.pending_limits[oid]
        
        # Verify Price Rounding (Tick 0.5)
        orders = self.bot.rest_api.get_open_orders(SYMBOL)
        my_order = next(o for o in orders if o['orderId'] == oid)
        order_price = float(my_order['price'])
        print(f"  Limit Price: {order_price}")
        self.assertAlmostEqual(order_price % 0.5, 0, delta=0.000001, msg="Price not rounded to tick 0.5")
        
        # 3. Simulate Execution
        print("  >> Simulating Fill...")
        qty = float(my_order['qty'])
        exec_msg = {
            'data': [{
                'symbol': SYMBOL, 'side': 'Buy', 'execQty': str(qty), 
                'execPrice': str(order_price), 'orderLinkId': data['link_id']
            }]
        }
        self.bot.on_execution_update(exec_msg)
        
        # 4. Verify VPM
        self.assertEqual(len(self.bot.vpm.trades), 1)
        self.assertEqual(self.bot.vpm.trades[0].size, qty)
        print("  [OK] VPM updated via Execution")
        
        # 5. Verify Reconciliation (Should be Stable)
        # Note: Since we only simulated fill, Real Exchange has Open Order (0 pos).
        # VPM has 1 pos. Reconcile will try to Open Market.
        # It will also try to place Stop Loss (Sell).
        
        with patch.object(self.bot.rest_api, 'place_order', wraps=self.bot.rest_api.place_order) as mock_place:
            self.bot.reconcile_positions()
            
            # Find the Entry Order (Buy, not ReduceOnly) in the calls
            entry_found = False
            for call in mock_place.call_args_list:
                args, kwargs = call
                
                # Extract params from args or kwargs
                side = kwargs.get('side')
                if not side and len(args) > 1: side = args[1]
                
                order_type = kwargs.get('order_type')
                if not order_type and len(args) > 2: order_type = args[2]
                
                reduce_only = kwargs.get('reduce_only', False)
                
                # Check for Market Buy that is NOT reduce_only (Entry)
                if side == 'Buy' and order_type == 'Market' and not reduce_only:
                    entry_found = True
                    print(f"  [OK] Found Entry Order: args={args}, kwargs={kwargs}")
                    break
            
            if not entry_found:
                print(f"  [DEBUG] All Calls: {mock_place.call_args_list}")
            
            self.assertTrue(entry_found, "Reconciliation did not place Entry Market Buy")
            print("  [OK] Reconciliation attempted to sync (Simulated fill scenario)")

    def test_03_min_notional_rejection(self):
        """Test that trades below Min Notional are skipped"""
        print("\n[Test 03] Min Notional Rejection")
        
        # Set Min Notional high (e.g. $1,000,000)
        self.bot.instrument_info['min_notional'] = 1_000_000.0
        
        # Trigger Aggressive Buy
        self.trigger_bar_close({
            'pred_long': 0.8, 'pred_short': 0.1, 'pred_dir_long': 0.9, 'pred_dir_short': 0.1
        })
        
        # Verify No Order Placed
        self.assertEqual(len(self.bot.vpm.trades), 0)
        orders = self.bot.rest_api.get_open_orders(SYMBOL)
        self.assertEqual(len(orders), 0)
        print("  [OK] Trade skipped due to Min Notional")

if __name__ == '__main__':
    unittest.main()