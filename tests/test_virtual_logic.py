import unittest
import shutil
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from virtual_position_manager import VirtualPositionManager

class TestVirtualManager(unittest.TestCase):
    def setUp(self):
        # Use a temporary file for testing
        self.test_file = Path("test_virtual_positions.json")
        self.symbol = "BTCUSDT"
        self.vpm = VirtualPositionManager(
            symbol=self.symbol, 
            max_positions=3, 
            storage_file=self.test_file
        )
        # Clean start
        self.vpm.trades = []
        self.vpm.save()

    def tearDown(self):
        if self.test_file.exists():
            self.test_file.unlink()

    def test_01_add_max_positions(self):
        print("\n[VPM] Testing Max Positions...")
        # Add 3 trades
        self.assertTrue(self.vpm.add_trade("Buy", 50000, 0.1, 49000, 52000))
        # Hack to bypass time debounce for testing
        self.vpm.trades[-1].entry_time -= 100 
        
        self.assertTrue(self.vpm.add_trade("Buy", 51000, 0.2, 50000, 53000))
        self.vpm.trades[-1].entry_time -= 100
        
        self.assertTrue(self.vpm.add_trade("Sell", 48000, 0.1, 49000, 46000))
        
        # Verify 3 trades exist
        self.assertEqual(len(self.vpm.trades), 3)
        
        # Try adding 4th (Should fail)
        self.vpm.trades[-1].entry_time -= 100
        self.assertFalse(self.vpm.add_trade("Buy", 50500, 0.1, 49500, 51500))
        print("      Max positions limit respected.")

    def test_02_persistence(self):
        print("\n[VPM] Testing Persistence (Save/Load)...")
        self.vpm.add_trade("Buy", 50000, 0.5, 49000, 51000)
        
        # Create new instance loading same file
        vpm2 = VirtualPositionManager(symbol=self.symbol, storage_file=self.test_file)
        self.assertEqual(len(vpm2.trades), 1)
        self.assertEqual(vpm2.trades[0].size, 0.5)
        print("      Data loaded correctly from disk.")

    def test_03_active_stops_generation(self):
        print("\n[VPM] Testing Stop Loss Generation...")
        self.vpm.add_trade("Buy", 50000, 0.1, 49000, 55000) # Stop @ 49000 (Sell)
        # Bypass debounce
        self.vpm.trades[-1].entry_time -= 100
        self.vpm.add_trade("Sell", 50000, 0.2, 51000, 45000) # Stop @ 51000 (Buy)
        
        stops = self.vpm.get_active_stops()
        self.assertEqual(len(stops), 2)
        
        stop_long = next(s for s in stops if s['side'] == 'Sell')
        stop_short = next(s for s in stops if s['side'] == 'Buy')
        
        self.assertEqual(stop_long['trigger_price'], 49000)
        self.assertEqual(stop_long['qty'], 0.1)
        
        self.assertEqual(stop_short['trigger_price'], 51000)
        self.assertEqual(stop_short['qty'], 0.2)
        print("      Stop orders generated correctly.")

    def test_04_pruning(self):
        print("\n[VPM] Testing Dead Trade Pruning...")
        self.vpm.add_trade("Buy", 50000, 0.1, 49000, 52000)
        
        # Price hits 50500 (No action)
        closed = self.vpm.prune_dead_trades(50500)
        self.assertEqual(len(closed), 0)
        
        # Price hits SL (48999)
        closed = self.vpm.prune_dead_trades(48999)
        self.assertEqual(len(closed), 1)
        self.assertEqual(len(self.vpm.trades), 0)
        print("      Trade closed via Stop Loss logic.")

if __name__ == '__main__':
    unittest.main()
