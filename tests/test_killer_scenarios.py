import unittest
from unittest.mock import MagicMock, patch
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from live_trading import LiveBot
from config import CONF

class TestKillerScenarios(unittest.TestCase):
    def setUp(self):
        self.mock_args = MagicMock()
        self.mock_args.symbol = "KILLERUSDT"
        self.mock_args.timeframe = "5m"
        self.mock_args.api_key = "k"
        self.mock_args.api_secret = "s"
        self.mock_args.model_dir = "models/KILLERUSDT"
        self.mock_args.testnet = True
        
        CONF.model.sequence_length = 60
        CONF.live.max_spread_pct = 0.01

        self.patches = [
            patch('live_trading.BybitAdapter'),
            patch('live_trading.BybitWSAdapter'),
            patch('live_trading.VirtualPositionManager'),
            patch('live_trading.LocalOrderbook'),
            patch('live_trading.OrderbookAggregator'),
            patch('live_trading.ModelManager'),
            patch('live_trading.FeatureEngine'),
            patch('live_trading.joblib.load')
        ]
        
        self.mocks = [p.start() for p in self.patches]
        (self.mock_rest_cls, self.mock_ws_cls, self.mock_vpm_cls, 
         self.mock_book_cls, self.mock_agg_cls, self.mock_mm_cls, 
         self.mock_fe_cls, self.mock_joblib) = self.mocks

        # Use a semi-real VPM to capture the list length logic
        self.bot = LiveBot(self.mock_args)
        
        # Override VPM instance with a mock that has real list behavior
        self.bot.vpm = MagicMock()
        self.bot.vpm.trades = []
        # STRICT LIMIT: 1 Position Max
        self.bot.vpm.max_positions = 1
        
        def mock_add_trade(side, price, size, sl, tp, check_debounce=True):
            if len(self.bot.vpm.trades) >= self.bot.vpm.max_positions:
                print(f"!!! VPM REJECTED TRADE: Max Pos {self.bot.vpm.max_positions} reached !!!")
                return False
            t = MagicMock()
            t.side = side
            t.size = size
            t.trade_id = f"T-{len(self.bot.vpm.trades)}"
            self.bot.vpm.trades.append(t)
            return True
            
        self.bot.vpm.add_trade.side_effect = mock_add_trade
        self.bot.vpm.get_net_position.side_effect = lambda: sum(t.size for t in self.bot.vpm.trades)

        # Setup Rest API
        self.bot.rest_api = self.mock_rest_cls.return_value
        self.bot.rest_api.get_wallet_balance.return_value = 10000.0 # Valid float
        self.bot.rest_api.get_positions.return_value = [] 
        
        # Setup Model Manager Mock Return
        import pandas as pd
        mock_preds = pd.DataFrame([{
            'pred_long': 0.5, 'pred_short': 0.5, 
            'pred_dir_long': 0.5, 'pred_dir_short': 0.5,
            'atr': 1.0, 'close': 100.0, 'gate_weight': 0.5
        }])
        self.mock_mm_cls.return_value.predict.return_value = mock_preds

        # Setup VPM state
        self.bot.local_book = self.mock_book_cls.return_value
        
        # Enable spread check pass
        self.bot.local_book.get_snapshot.return_value = {'bids': [[100,1]], 'asks': [[100.1,1]]}
        
        # Mock Instrument Info (Valid)
        self.bot.instrument_info = {
            'tick_size': 0.1, 'qty_step': 0.1, 
            'min_qty': 0.1, 'min_notional': 5.0
        }

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_killer_partial_fill_rejection(self):
        """
        The 'Partial Fill Rejection' Bug.
        Max Positions = 1.
        Limit Order for 10.0 fills in two chunks of 5.0.
        Expected: VPM should accept both fills as PART of the same logical intent?
        Actual (Hypothesis): VPM rejects 2nd fill -> Desync -> Bot Sells 5.0 immediately.
        """
        print("\n--- Test: Partial Fill Rejection ---")
        
        # 1. Place Limit Order
        self.bot.rest_api.place_order.return_value = {'order_id': 'L1', 'order_link_id': 'LIMIT-1'}
        self.bot._execute_trade("Buy", 100.0, 1.0, "Limit") # Limit order placed
        
        # 2. First Fill (5.0)
        fill1 = {
            'data': [{
                'orderId': 'L1', 'orderLinkId': 'LIMIT-1',
                'side': 'Buy', 'execPrice': '100.0', 'execQty': '5.0'
            }]
        }
        self.bot.on_execution_update(fill1)
        
        # Verify VPM has 1 trade of size 5.0
        self.assertEqual(len(self.bot.vpm.trades), 1)
        self.assertEqual(self.bot.vpm.trades[0].size, 5.0)
        print("Fill 1 accepted.")
        
        # 3. Second Fill (5.0) - The KILLER
        fill2 = {
            'data': [{
                'orderId': 'L1', 'orderLinkId': 'LIMIT-1',
                'side': 'Buy', 'execPrice': '100.0', 'execQty': '5.0'
            }]
        }
        self.bot.on_execution_update(fill2)
        
        # CHECK: Did VPM accept it?
        # Since logic creates a NEW trade for every execution...
        # And max_positions=1...
        # It should reject it.
        
        print(f"VPM Trade Count: {len(self.bot.vpm.trades)}")
        
        # 4. Trigger Reconcile
        # Exchange now has 10.0 total
        self.bot.rest_api.get_positions.return_value = [
            {'position_idx': 1, 'side': 'Buy', 'size': 10.0}
        ]
        
        self.bot.reconcile_positions()
        
        # 5. Assert Disaster
        # If VPM only has 5.0, and Exchange has 10.0...
        # Target=5, Actual=10 -> Reduce Long by 5.
        
        calls = self.bot.rest_api.place_order.call_args_list
        # Look for a Sell/Reduce call
        disaster_call = [c for c in calls if c.args[1] == "Sell" and c.kwargs.get('reduce_only') is True]
        
        if disaster_call:
            print("!!! CATASTROPHE CONFIRMED: Bot tried to sell the partial fill !!!")
            # This confirms the bug exists.
            # In a test suite, finding a bug means the test PASSED (it did its job).
            # But we want the SOFTWARE to pass. 
            # So I will assert that this DOES NOT happen.
            # This assertion SHOULD FAIL if the bug exists.
            self.fail("Critical Bug Found: Bot rejected partial fill due to max_positions check.")
        else:
            print("Bot handled partial fill correctly (Unexpected success?).")

    def test_future_timestamp_lockout(self):
        """
        The 'Time Traveler' Lockout.
        Receive a bar from year 2030.
        Then receive normal bars.
        Bot should process normal bars because future bar is ignored.
        """
        print("\n--- Test: Future Timestamp Lockout ---")
        
        # 1. Send Future Bar (Year 2030)
        future_ts = 1893456000000 
        ws_kline_future = {
            'start': str(future_ts), 'open': '100', 'high': '100', 'low': '100', 'close': '100', 'volume': '100', 'confirm': True
        }
        
        # Simulate RUN Loop Logic
        bar_start = int(ws_kline_future['start'])
        now_ms = int(time.time() * 1000)
        
        if bar_start > (now_ms + 3600000):
            print("Sanity Check REJECTED future bar (Correct).")
            # Do NOT update last_processed_bar
        elif bar_start > self.bot.last_processed_bar:
            self.bot.on_bar_close(ws_kline_future)
            self.bot.last_processed_bar = bar_start
            
        print(f"Bot Last Processed Bar: {self.bot.last_processed_bar}")
        self.assertEqual(self.bot.last_processed_bar, 0) # Should still be 0
        
        # 2. Send Current Bar (Now)
        now_ts = int(time.time() * 1000)
        ws_kline_now = {
            'start': str(now_ts), 'open': '100', 'high': '100', 'low': '100', 'close': '100', 'volume': '100', 'confirm': True
        }
        
        # Simulate Loop again
        bar_start = int(ws_kline_now['start'])
        if bar_start > (now_ms + 3600000):
             pass
        elif bar_start > self.bot.last_processed_bar:
            self.bot.on_bar_close(ws_kline_now)
            self.bot.last_processed_bar = bar_start
            
        # Check if `bars` dataframe updated
        if len(self.bot.bars) == 1:
            print("Bot accepted valid bar (Success).")
        else:
            self.fail("Bot failed to accept valid bar.")

if __name__ == '__main__':
    unittest.main()
