import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import time
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from live_trading import LiveBot
from config import CONF

class TestLiveSafetyAndWarmup(unittest.TestCase):
    def setUp(self):
        # Mock dependencies
        self.mock_args = MagicMock()
        self.mock_args.symbol = "TESTUSDT"
        self.mock_args.timeframe = "5m"
        self.mock_args.api_key = "test_key"
        self.mock_args.api_secret = "test_secret"
        self.mock_args.model_dir = "models/TESTUSDT"
        self.mock_args.testnet = True

        # Mock CONF
        CONF.model.sequence_length = 60
        CONF.live.max_spread_pct = 0.002 # 0.2%

        # Patch internal components
        with patch('live_trading.BybitAdapter') as mock_rest, \
             patch('live_trading.BybitWSAdapter') as mock_ws, \
             patch('live_trading.VirtualPositionManager') as mock_vpm, \
             patch('live_trading.LocalOrderbook') as mock_book, \
             patch('live_trading.OrderbookAggregator') as mock_agg, \
             patch('live_trading.ModelManager') as mock_mm, \
             patch('live_trading.FeatureEngine') as mock_fe, \
             patch('live_trading.joblib.load') as mock_joblib_load:
            
            # Setup mock return for joblib to avoid attribute errors if accessed
            mock_joblib_load.return_value = ["mock_feature"] 

            self.bot = LiveBot(self.mock_args)
            
            # Reset dependencies for specific test setup
            self.bot.rest_api = MagicMock()
            self.bot.vpm = MagicMock()
            self.bot.local_book = MagicMock()
            self.bot.history_file = Path("test_history.csv")
            
            # Set valid instrument info for calculation logic
            self.bot.instrument_info = {
                'tick_size': 0.1, 
                'qty_step': 0.001, 
                'min_qty': 0.001, 
                'min_notional': 5.0
            }

    def test_warmup_calculation_logic(self):
        """Test _calculate_warmup_status with various scenarios"""
        
        # Scenario 1: Empty history
        self.bot.bars = pd.DataFrame()
        self.assertEqual(self.bot._calculate_warmup_status(), 0)
        
        # Scenario 2: Stale history (Last bar too old)
        now = int(time.time() * 1000)
        old_ts = now - (60 * 60 * 1000) # 1 hour ago
        self.bot.bars = pd.DataFrame([
            {'timestamp': old_ts, 'ob_spread_mean': 1.0}
        ])
        self.assertEqual(self.bot._calculate_warmup_status(), 0) # Should be 0 due to staleness
        
        # Scenario 3: Continuous valid bars
        # Create 60 bars ending now, all valid
        bars = []
        for i in range(60):
            bars.append({
                'timestamp': now - ((59-i) * 300 * 1000), # 5m intervals
                'ob_spread_mean': 1.0
            })
        self.bot.bars = pd.DataFrame(bars)
        self.assertEqual(self.bot._calculate_warmup_status(), 60)
        
        # Scenario 4: Broken continuity (Gap)
        bars = []
        # First 30 bars are fine
        for i in range(30):
            bars.append({
                'timestamp': now - ((59-i) * 300 * 1000),
                'ob_spread_mean': 1.0
            })
        # Gap of 10 mins (skip one 5m interval)
        for i in range(31, 60):
             bars.append({
                'timestamp': now - ((59-i) * 300 * 1000),
                'ob_spread_mean': 1.0
            })
        self.bot.bars = pd.DataFrame(bars)
        # Should count backwards from end until gap
        # Indices 31..59 are 29 bars? No range(31,60) is 29 items.
        # Let's verify manually. 
        # Last bar is index 59 (time=now).
        # Previous bar is index 58 (time=now-5m).
        # ...
        # Bar 31 (time=now - 28*5m).
        # Bar 30 is MISSING (Gap).
        # So we should find continuous block from 31..59. Length = 29.
        self.assertEqual(self.bot._calculate_warmup_status(), 29)

        # Scenario 5: Invalid OB Data (Zeros)
        bars = []
        for i in range(60):
            bars.append({
                'timestamp': now - ((59-i) * 300 * 1000),
                'ob_spread_mean': 1.0 if i >= 10 else 0.0 # First 10 are invalid
            })
        self.bot.bars = pd.DataFrame(bars)
        # Should count valid ones from end: 60 - 10 = 50
        self.assertEqual(self.bot._calculate_warmup_status(), 50)

    def test_startup_consistency_check(self):
        """Test check_consistency logic"""
        
        # Case 1: Match (Both 0)
        self.bot.vpm.get_net_position.return_value = 0.0
        self.bot.rest_api.get_positions.return_value = []
        
        with self.assertLogs('LiveTrading', level='INFO') as cm:
            self.bot.check_consistency()
            self.assertTrue(any("Startup Consistency Check: OK" in o for o in cm.output))
            
        # Case 2: Mismatch (VPM=1, Exch=0)
        self.bot.vpm.get_net_position.return_value = 1.0
        self.bot.rest_api.get_positions.return_value = [] # Exch 0
        
        with self.assertLogs('LiveTrading', level='WARNING') as cm:
            self.bot.check_consistency()
            self.assertTrue(any("STARTUP INCONSISTENCY DETECTED" in o for o in cm.output))
            
        # Case 3: Match (Both 1 Long)
        self.bot.vpm.get_net_position.return_value = 1.0
        self.bot.rest_api.get_positions.return_value = [
            {'position_idx': 1, 'size': 1.0, 'side': 'Buy'}
        ]
        
        with self.assertLogs('LiveTrading', level='INFO') as cm:
            self.bot.check_consistency()
            self.assertTrue(any("OK" in o for o in cm.output))

    def test_spread_check_safety(self):
        """Test spread check in reconcile_positions"""
        
        # Setup VPM to trigger a trade logic (if check passes)
        # We assume reconcile logic continues if check passes
        
        # Case 1: High Spread -> Abort
        self.bot.local_book.get_snapshot.return_value = {
            'bids': [[100.0, 1.0]],
            'asks': [[101.0, 1.0]], # Spread = 1/100 = 1% > 0.2%
            'timestamp': 123
        }
        
        with self.assertLogs('LiveTrading', level='WARNING') as cm:
            self.bot.reconcile_positions()
            # Should log Critical Warning
            self.assertTrue(any("CRITICAL: High Spread" in o for o in cm.output))
            
            # Verify NO orders were placed (mock rest api shouldn't be called)
            self.bot.rest_api.place_order.assert_not_called()

        # Case 2: Low Spread -> Proceed
        self.bot.local_book.get_snapshot.return_value = {
            'bids': [[100.0, 1.0]],
            'asks': [[100.1, 1.0]], # Spread = 0.1% < 0.2%
            'timestamp': 123
        }
        
        # Mock VPM trades to trigger an action (Target=1, Actual=0)
        trade = MagicMock()
        trade.side = "Buy"
        trade.size = 1.0
        self.bot.vpm.trades = [trade]
        self.bot.rest_api.get_positions.return_value = [] # Actual 0
        
        # Mock current price for final check
        self.bot.rest_api.get_current_price.return_value = 100.0
        
        # Prevent recursion by ensuring no dead trades are found
        self.bot.vpm.prune_dead_trades.return_value = []
        
        self.bot.reconcile_positions()
        
        # Should call place_order
        self.bot.rest_api.place_order.assert_called()

if __name__ == '__main__':
    unittest.main()
