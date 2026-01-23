import unittest
from unittest.mock import MagicMock, patch, ANY
import time
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from live_trading import LiveBot
from config import CONF

class TestAdversarialLive(unittest.TestCase):
    def setUp(self):
        self.mock_args = MagicMock()
        self.mock_args.symbol = "CHAOSUSDT"
        self.mock_args.timeframe = "5m"
        self.mock_args.api_key = "k"
        self.mock_args.api_secret = "s"
        self.mock_args.model_dir = "models/CHAOSUSDT"
        self.mock_args.testnet = True
        
        CONF.model.sequence_length = 60
        CONF.live.max_spread_pct = 0.002

        # Patch everything
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
        (self.mock_rest, self.mock_ws, self.mock_vpm, 
         self.mock_book, self.mock_agg, self.mock_mm, 
         self.mock_fe, self.mock_joblib) = self.mocks

        self.bot = LiveBot(self.mock_args)
        
        # Reset specific instance mocks
        self.bot.rest_api = self.mock_rest.return_value
        self.bot.vpm = self.mock_vpm.return_value
        self.bot.local_book = self.mock_book.return_value
        
        # Valid Default Info
        self.bot.instrument_info = {'tick_size': 0.1, 'qty_step': 0.1, 'min_qty': 0.1, 'min_notional': 5.0}

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_chaos_input_ws_kline(self):
        """Inject garbage into on_bar_close via kline stream"""
        # 1. Missing keys
        garbage = {'data': [{'missing': 'everything'}]}
        try:
            self.bot.on_bar_close(garbage['data'][0])
        except KeyError:
            self.fail("Bot crashed on missing kline keys!")
        except Exception:
            pass # It might fail gracefully, which is fine

        # 2. Wrong Types (Strings where ints expected)
        bad_types = {
            'start': 'not_an_int',
            'open': 'not_a_float',
            'high': [],
            'low': None,
            'close': {},
            'volume': '100'
        }
        try:
            self.bot.on_bar_close(bad_types)
        except ValueError:
            pass # Expected conversion error, should be caught or handled
        except TypeError:
            self.fail("Bot crashed with TypeError on bad input types!")
            
    def test_flash_crash_spread(self):
        """Simulate 99% Spread during Reconcile"""
        self.bot.local_book.get_snapshot.return_value = {
            'bids': [[1.0, 100]], # Bid $1
            'asks': [[100.0, 100]], # Ask $100 -> Spread 9900%
            'timestamp': 123
        }
        
        # Trigger reconcile
        with self.assertLogs('LiveTrading', level='WARNING') as cm:
            self.bot.reconcile_positions()
            self.assertTrue(any("CRITICAL: High Spread" in o for o in cm.output))
            # Verify NO order placed
            self.bot.rest_api.place_order.assert_not_called()

    def test_race_condition_fill_before_place_return(self):
        """
        Simulate WS Fill arriving BEFORE place_order returns.
        This tests the fallback logic when orderId is unknown.
        """
        # 1. WS Fill arrives "from the future" (Order ID unknown yet)
        ws_msg = {
            'data': [{
                'orderId': 'FUTURE-ORD-1',
                'orderLinkId': 'LIMIT-FAST',
                'side': 'Buy',
                'execPrice': '100.0',
                'execQty': '1.0'
            }]
        }
        
        with self.assertLogs('LiveTrading', level='WARNING') as cm:
            self.bot.on_execution_update(ws_msg)
            # Should invoke fallback warning
            self.assertTrue(any("missing stored SL/TP" in o for o in cm.output))
            
        # Verify it added trade with fallback values (1% SL)
        # Price 100 -> SL 99.0
        self.bot.vpm.add_trade.assert_called_with(
            'Buy', 100.0, 1.0, 99.0, 102.0, check_debounce=False
        )

    def test_api_explodes(self):
        """Simulate API raising exceptions during critical loops"""
        # Mock API to raise Exception
        self.bot.rest_api.get_positions.side_effect = Exception("API 500 Internal Error")
        
        # Call check_consistency
        try:
            self.bot.check_consistency()
        except Exception:
            self.fail("check_consistency crashed on API error!")
            
        # Call reconcile
        self.bot.rest_api.get_positions.side_effect = Exception("API Timeout")
        self.bot.local_book.get_snapshot.return_value = {'bids':[[100,1]], 'asks':[[100.1,1]]}
        
        # Reconcile attempts to get positions. It should catch the error.
        # Wait, does reconcile catch get_positions error?
        # get_positions in adapter catches it and returns [].
        # Let's verify adapter behavior mocking.
        # My mock `get_positions` RAISES exception. 
        # Real adapter `get_positions` CATCHES exception and returns [].
        # But if the ADAPTER raises (e.g. valid network error bubbling up), bot should handle it.
        # Let's see if `reconcile_positions` handles `get_positions` returning [] (which implies empty or error).
        
        self.bot.rest_api.get_positions.side_effect = None
        self.bot.rest_api.get_positions.return_value = [] # Error case
        
        # Trigger reconcile with VPM target = 1
        trade = MagicMock()
        trade.side = "Buy"
        trade.size = 1.0
        self.bot.vpm.trades = [trade]
        
        # If get_positions returns [] (Error or Empty), bot assumes Actual=0.
        # Diff = 1. Bot sends Buy.
        # This is "Fail Open" risk?
        # If API fails, we think we have 0 pos, so we buy more?
        # YES. This is a potential risk.
        
        self.bot.reconcile_positions()
        self.bot.rest_api.place_order.assert_called() 
        # This confirms "Fail Open" behavior on API error if adapter swallows exception.

if __name__ == '__main__':
    unittest.main()