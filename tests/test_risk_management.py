import unittest
import sys
import os
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from live_trading import LiveBot
from config import CONF

class MockAPI:
    def __init__(self):
        self.wallet = 1000.0
        self.min_notional = 5.0
        self.positions = []
    
    def get_wallet_balance(self, coin): return self.wallet
    def get_instrument_info(self, symbol):
        return {'tick_size': 0.01, 'qty_step': 1.0, 'min_qty': 1.0, 'min_notional': self.min_notional}
    def check_clock_drift(self): return 0.0
    def get_positions(self, symbol): return self.positions
    def get_open_orders(self, symbol): return []
    def place_order(self, *args, **kwargs): return {'order_id': 'test_id', 'order_link_id': 'test_link'}
    def cancel_all_orders(self, symbol): return True
    def set_leverage(self, symbol, leverage): return True
    def switch_position_mode(self, symbol, mode): return True

class TestRiskManagement(unittest.TestCase):
    def setUp(self):
        # Ensure a clean state for virtual_positions.json
        if os.path.exists("virtual_positions.json"):
            os.remove("virtual_positions.json")

        self.args = SimpleNamespace(
            symbol="BTCUSDT", model_dir="dummy", api_key="k", api_secret="s", 
            testnet=True, timeframe="5m"
        )
        self.mock_api = MockAPI()
        
        with patch('live_trading.BybitAdapter', return_value=self.mock_api):
            with patch('live_trading.BybitWSAdapter'):
                with patch('live_trading.LiveBot._load_model_manager'):
                    self.bot = LiveBot(self.args)
                    self.bot.warmup_bars = 0

    def test_max_positions_guard(self):
        """Verify that VPM respects max_positions limit."""
        print("\n[Risk Test] Max Positions Guard")
        self.bot.vpm.max_positions = 2
        
        # Add 2 trades
        self.bot.vpm.add_trade("Buy", 100, 10, 90, 110, check_debounce=False)
        self.bot.vpm.add_trade("Buy", 100, 10, 90, 110, check_debounce=False)
        self.assertEqual(len(self.bot.vpm.trades), 2)
        
        # Try 3rd trade - should be rejected
        success = self.bot.vpm.add_trade("Buy", 100, 10, 90, 110, check_debounce=False)
        self.assertFalse(success)
        self.assertEqual(len(self.bot.vpm.trades), 2)
        print("    [OK] Correctly blocked 3rd trade due to max_positions=2")

    def test_min_notional_guard(self):
        """Verify that trades below min_notional are skipped."""
        print("\n[Risk Test] Min Notional Guard")
        # risk_amt = 13.0 (1000 * 0.013)
        # sl_dist = atr * 3.9
        # raw_qty = 13.0 / (3.33 * 3.9) = 13.0 / 12.987 = 1.001
        # size_qty = 1.0
        # notional = 1.0 * 2.0 = 2.0 < 5.0 (min_notional)
        
        with self.assertLogs('LiveTrading', level='WARNING') as cm:
            self.bot._execute_trade("Buy", 2.0, 3.33, "Market", check_debounce=False)
            self.assertTrue(any("Min Notional" in l for l in cm.output))
        print("    [OK] Correctly skipped trade below Min Notional")

    def test_wallet_depletion_guard(self):
        """Verify that bot doesn't trade if wallet is empty."""
        print("\n[Risk Test] Wallet Depletion Guard")
        self.mock_api.wallet = 0.0
        
        initial_trades = len(self.bot.vpm.trades)
        self.bot._execute_trade("Buy", 100.0, 1.0, "Market", check_debounce=False)
        self.assertEqual(len(self.bot.vpm.trades), initial_trades)
        print("    [OK] Correctly skipped trade due to 0 wallet balance")

if __name__ == '__main__':
    unittest.main()
