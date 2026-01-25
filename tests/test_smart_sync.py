
import unittest
from unittest.mock import MagicMock, patch, call
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from live_trading import LiveBot

class TestSmartSync(unittest.TestCase):
    def setUp(self):
        self.args = MagicMock()
        self.args.symbol = "TESTSYNC"
        self.args.model_dir = "models/mock"
        self.args.api_key = "test"
        
        # Patch everything
        self.patcher_rest = patch('live_trading.BybitAdapter')
        self.patcher_ws = patch('live_trading.BybitWSAdapter')
        self.patcher_vpm = patch('live_trading.VirtualPositionManager')
        self.patcher_fe = patch('live_trading.FeatureEngine')
        self.patcher_mm = patch('live_trading.ModelManager')
        self.patcher_lmm = patch.object(LiveBot, '_load_model_manager')
        
        self.MockRest = self.patcher_rest.start()
        self.MockWS = self.patcher_ws.start()
        self.MockVPM = self.patcher_vpm.start()
        self.patcher_fe.start()
        self.patcher_mm.start()
        self.MockLoadMM = self.patcher_lmm.start()
        
        self.bot = LiveBot(self.args)
        # Fix instrument info after init (since init call returned Mock)
        self.bot.instrument_info = {'tick_size': 0.1, 'qty_step': 0.1, 'min_qty': 0.1, 'min_notional': 5.0}
        
        # Setup common mock returns
        self.bot.rest_api.get_current_price.return_value = 100.0
        self.bot.pending_limits = {}
        
        # Mock Local Book for Spread Check
        self.bot.local_book = MagicMock()
        self.bot.local_book.get_snapshot.return_value = {
            'bids': [[99.95, 1.0]], 
            'asks': [[100.05, 1.0]], 
            'timestamp': 123456789
        }

    def tearDown(self):
        patch.stopall()

    def test_clears_position_stops(self):
        """Test: It clears global Position SL/TP if position exists."""
        # Setup: We have a Long position
        self.bot.rest_api.get_positions.return_value = [
            {
                'symbol': 'TESTSYNC', 'side': 'Buy', 'size': 100, 'position_idx': 1
            }
        ]
        
        self.bot.reconcile_positions()
        
        # Verify call
        # Should call set_trading_stop(..., sl=0, tp=0)
        self.bot.rest_api.set_trading_stop.assert_called_with(
            'TESTSYNC', position_idx=1, sl=0, tp=0
        )
        print("\n[PASS] Clears Position SL/TP correctly.")

    def test_smart_matching_logic(self):
        """Test: Keeps matching orders, cancels stale ones, places new ones."""
        
        # 1. Setup VPM Requirements (2 Stops needed)
        # Stop A: Sell 100 @ 90 (SL)
        # Stop B: Sell 100 @ 110 (TP)
        self.bot.vpm.get_active_stops.return_value = [
            {'id': 't1', 'qty': 100, 'trigger_price': 90, 'side': 'Sell', 'type': 'sl'},
            {'id': 't1', 'qty': 100, 'trigger_price': 110, 'side': 'Sell', 'type': 'tp'}
        ]
        
        # 2. Setup Exchange State (Open Orders)
        # Order 1: Matches Stop A perfectly (Keep)
        # Order 2: Stale Order (Wrong Price 85) (Cancel)
        self.bot.rest_api.get_open_orders.return_value = [
            {'orderId': 'oid_match', 'qty': '100', 'triggerPrice': '90', 'side': 'Sell', 'orderType': 'Market'}, # Match
            {'orderId': 'oid_stale', 'qty': '100', 'triggerPrice': '85', 'side': 'Sell', 'orderType': 'Market'}  # Stale
        ]
        
        # Mock Position exists so we place orders
        self.bot.rest_api.get_positions.return_value = [{'size': 100, 'position_idx': 1, 'side': 'Buy'}]
        
        # Mock VPM trades to match position so we don't trigger a Reduce order
        trade_mock = MagicMock()
        trade_mock.side = 'Buy'
        trade_mock.size = 100.0
        self.bot.vpm.trades = [trade_mock]
        
        # Run Sync
        self.bot.reconcile_positions()
        
        # 3. Assertions
        
        # A. Cancellation
        # Should cancel 'oid_stale'
        # Should NOT cancel 'oid_match'
        self.bot.rest_api.cancel_order.assert_called_once_with('TESTSYNC', 'oid_stale')
        
        # B. Placement
        # Should place Stop B (TP @ 110)
        # Should NOT place Stop A (already exists)
        self.bot.rest_api.place_order.assert_called_once()
        args, kwargs = self.bot.rest_api.place_order.call_args
        
        self.assertEqual(kwargs['trigger_price'], 110)
        self.assertEqual(kwargs['trigger_direction'], 1) # Rise (TP)
        self.assertEqual(kwargs['reduce_only'], True)
        
        print("[PASS] Smart Match Logic: Kept 1, Cancelled 1, Placed 1.")

    def test_trigger_direction_logic(self):
        """Test: SL uses Fall (2), TP uses Rise (1) for Longs."""
        self.bot.rest_api.get_current_price.return_value = 100.0
        self.bot.rest_api.get_open_orders.return_value = [] # Clean slate
        self.bot.rest_api.get_positions.return_value = [{'size': 100, 'position_idx': 1, 'side': 'Buy'}]
        
        # Mock VPM trades to match position so we don't trigger a Reduce order
        trade_mock = MagicMock()
        trade_mock.side = 'Buy'
        trade_mock.size = 100.0
        self.bot.vpm.trades = [trade_mock]
        
        # VPM has SL and TP
        self.bot.vpm.get_active_stops.return_value = [
            {'id': 't1', 'qty': 100, 'trigger_price': 90, 'side': 'Sell', 'type': 'sl'}, # SL < Price
            {'id': 't1', 'qty': 100, 'trigger_price': 110, 'side': 'Sell', 'type': 'tp'} # TP > Price
        ]
        
        self.bot.reconcile_positions()
        
        # Check Calls
        calls = self.bot.rest_api.place_order.call_args_list
        self.assertEqual(len(calls), 2)
        
        # First call (SL 90) -> Trigger Dir 2 (Fall)
        sl_call = [c for c in calls if c[1]['trigger_price'] == 90][0]
        self.assertEqual(sl_call[1]['trigger_direction'], 2)
        
        # Second call (TP 110) -> Trigger Dir 1 (Rise)
        tp_call = [c for c in calls if c[1]['trigger_price'] == 110][0]
        self.assertEqual(tp_call[1]['trigger_direction'], 1)
        
        print("[PASS] Trigger Directions calculated correctly.")

if __name__ == '__main__':
    unittest.main()
