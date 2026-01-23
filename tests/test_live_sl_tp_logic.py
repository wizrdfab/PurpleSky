import unittest
from unittest.mock import MagicMock, patch
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from live_trading import LiveBot
from config import CONF

class TestLiveSLTPLogic(unittest.TestCase):
    def setUp(self):
        self.mock_args = MagicMock()
        self.mock_args.symbol = "TESTUSDT"
        self.mock_args.timeframe = "5m"
        self.mock_args.api_key = "k"
        self.mock_args.api_secret = "s"
        self.mock_args.model_dir = "models/TESTUSDT"
        self.mock_args.testnet = True
        
        CONF.model.sequence_length = 60
        CONF.live.max_spread_pct = 0.002

        with patch('live_trading.BybitAdapter') as mock_rest, \
             patch('live_trading.BybitWSAdapter') as mock_ws, \
             patch('live_trading.VirtualPositionManager') as mock_vpm, \
             patch('live_trading.LocalOrderbook') as mock_book, \
             patch('live_trading.OrderbookAggregator') as mock_agg, \
             patch('live_trading.ModelManager') as mock_mm, \
             patch('live_trading.FeatureEngine') as mock_fe, \
             patch('live_trading.joblib.load') as mock_joblib:
            
            self.bot = LiveBot(self.mock_args)
            self.bot.rest_api = MagicMock()
            self.bot.vpm = MagicMock()
            # Mock instrument info for normalization
            self.bot.instrument_info = {
                'tick_size': 0.1, 'qty_step': 0.001, 
                'min_qty': 0.001, 'min_notional': 5.0
            }

    def test_sl_tp_persistence(self):
        """Test that SL/TP are stored on placement and retrieved on execution"""
        
        # 1. Place Limit Order
        # Setup mock return for place_order
        self.bot.rest_api.place_order.return_value = {
            'order_id': 'ORD-123',
            'order_link_id': 'LIMIT-ABC'
        }
        self.bot.rest_api.get_wallet_balance.return_value = 1000.0
        
        # Call _execute_trade (Buy Limit)
        # Price=100, ATR=1.0. 
        # Config (default): SL=3.9 ATR -> 3.9 dist. TP=1.2 ATR -> 1.2 dist.
        CONF.strategy.stop_loss_atr = 3.9
        CONF.strategy.take_profit_atr = 1.2
        
        price = 100.0
        atr = 1.0
        
        expected_sl = 100.0 - 3.9 # 96.1
        expected_tp = 100.0 + 1.2 # 101.2
        
        self.bot._execute_trade("Buy", price, atr, "Limit")
        
        # Verify Storage
        self.assertIn('ORD-123', self.bot.pending_limits)
        stored = self.bot.pending_limits['ORD-123']
        self.assertAlmostEqual(stored['sl'], expected_sl)
        self.assertAlmostEqual(stored['tp'], expected_tp)
        
        # 2. Simulate WS Execution Update
        ws_msg = {
            'data': [{
                'orderId': 'ORD-123',
                'orderLinkId': 'LIMIT-ABC',
                'side': 'Buy',
                'execPrice': '100.0',
                'execQty': '0.1'
            }]
        }
        
        self.bot.on_execution_update(ws_msg)
        
        # Verify VPM call used correct values
        self.bot.vpm.add_trade.assert_called_with(
            'Buy', 100.0, 0.1, 
            expected_sl, expected_tp, 
            check_debounce=False
        )

    def test_sl_tp_fallback(self):
        """Test fallback logic when Limit Order ID is unknown"""
        
        ws_msg = {
            'data': [{
                'orderId': 'UNKNOWN-999',
                'orderLinkId': 'LIMIT-XYZ',
                'side': 'Buy',
                'execPrice': '100.0',
                'execQty': '0.1'
            }]
        }
        
        self.bot.on_execution_update(ws_msg)
        
        # Fallback: SL 1% -> 99.0, TP 2% -> 102.0
        self.bot.vpm.add_trade.assert_called_with(
            'Buy', 100.0, 0.1, 
            99.0, 102.0, 
            check_debounce=False
        )

if __name__ == '__main__':
    unittest.main()
