import unittest
import logging
import sys
import time
import os
import shutil
from pathlib import Path
from dataclasses import dataclass

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from bybit_adapter import BybitAdapter
from virtual_position_manager import VirtualPositionManager

# API CREDENTIALS (TESTNET)
API_KEY = os.getenv("BYBIT_API_KEY", "DZh7qdh2dTfw328ASi")
API_SECRET = os.getenv("BYBIT_API_SECRET", "Y4Jhn5z2MMi0LGN174RRNXLeE2NrRpGZ3mcf")
SYMBOL = "BTCUSDT"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestStateSync")

class TestStateSync(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n=== STARTING STATE SYNC & STRESS TEST ===")
        cls.rest = BybitAdapter(api_key=API_KEY, api_secret=API_SECRET, testnet=True)
        
        # Setup Hedge Mode
        cls.rest.switch_position_mode(SYMBOL, mode=3)
        cls.rest.set_leverage(SYMBOL, leverage=10)
        
        # Clean Start
        cls.rest.cancel_all_orders(SYMBOL)
        cls._close_all_positions(cls)
        
        # VPM Setup (Max 5 positions for stress test)
        cls.test_file = Path("test_sync_vpm.json")
        cls.vpm = VirtualPositionManager(symbol=SYMBOL, max_positions=5, storage_file=cls.test_file)
        cls.vpm.trades = []
        cls.vpm.save()

    @classmethod
    def tearDownClass(cls):
        print("\n=== TEARDOWN ===")
        cls.rest.cancel_all_orders(SYMBOL)
        cls._close_all_positions(cls)
        if cls.test_file.exists():
            cls.test_file.unlink()

    @staticmethod
    def _close_all_positions(cls):
        positions = cls.rest.get_positions(SYMBOL)
        for p in positions:
            side = "Sell" if p['side'] == "Buy" else "Buy"
            idx = p.get('position_idx', 0)
            if idx == 0: continue # Skip if bad data
            cls.rest.place_order(SYMBOL, side, "Market", p['size'], reduce_only=True, position_idx=idx)
        time.sleep(2)

    def reconcile_logic(self):
        """Replicates live_trading.py logic for syncing VPM -> Exchange"""
        # 1. Calc Targets
        target_long = sum(t.size for t in self.vpm.trades if t.side == "Buy")
        target_short = sum(t.size for t in self.vpm.trades if t.side == "Sell")
        
        # 2. Get Actuals
        positions = self.rest.get_positions(SYMBOL)
        actual_long = 0.0
        actual_short = 0.0
        long_entry = 0.0
        short_entry = 0.0
        
        for p in positions:
            idx = p.get('position_idx', 0)
            if idx == 1: 
                actual_long = p['size']
                long_entry = p['entry_price']
            elif idx == 2: 
                actual_short = p['size']
                short_entry = p['entry_price']
                
        # 3. Execute
        # Long
        diff_long = round(target_long - actual_long, 3)
        if abs(diff_long) > 0:
            qty = abs(diff_long)
            if diff_long > 0:
                self.rest.place_order(SYMBOL, "Buy", "Market", qty, position_idx=1)
            else:
                self.rest.place_order(SYMBOL, "Sell", "Market", qty, reduce_only=True, position_idx=1)
        
        # Short
        diff_short = round(target_short - actual_short, 3)
        if abs(diff_short) > 0:
            qty = abs(diff_short)
            if diff_short > 0:
                self.rest.place_order(SYMBOL, "Sell", "Market", qty, position_idx=2)
            else:
                self.rest.place_order(SYMBOL, "Buy", "Market", qty, reduce_only=True, position_idx=2)
                
        time.sleep(2) # Wait for fill
        return actual_long, actual_short, long_entry, short_entry

    def test_stress_and_sync(self):
        print("\n[Test] Stress Test & State Audit (4+ Positions)")
        
        price = self.rest.get_current_price(SYMBOL)
        print(f"  Current Price: {price}")
        
        # --- SCENARIO 1: Open 4 Positions (2 Long, 2 Short) ---
        print("\n  >> Opening 4 Virtual Positions...")
        
        # L1: 0.01
        self.vpm.add_trade("Buy", price, 0.01, price*0.9, price*1.1)
        self.vpm.trades[-1].entry_time -= 100 # Debounce hack
        
        # L2: 0.02
        self.vpm.add_trade("Buy", price, 0.02, price*0.9, price*1.1)
        self.vpm.trades[-1].entry_time -= 100
        
        # S1: 0.01
        self.vpm.add_trade("Sell", price, 0.01, price*1.1, price*0.9)
        self.vpm.trades[-1].entry_time -= 100
        
        # S2: 0.02
        self.vpm.add_trade("Sell", price, 0.02, price*1.1, price*0.9)
        
        # Sync
        self.reconcile_logic()
        
        # AUDIT
        positions = self.rest.get_positions(SYMBOL)
        real_long = next((p for p in positions if p['position_idx']==1), None)
        real_short = next((p for p in positions if p['position_idx']==2), None)
        
        self.assertIsNotNone(real_long)
        self.assertIsNotNone(real_short)
        
        # Size Check
        print(f"  [Audit 1] Long Size: Virtual=0.03, Real={real_long['size']}")
        self.assertAlmostEqual(real_long['size'], 0.03, delta=0.001)
        
        print(f"  [Audit 1] Short Size: Virtual=0.03, Real={real_short['size']}")
        self.assertAlmostEqual(real_short['size'], 0.03, delta=0.001)
        
        # Entry Price Check (Approximate)
        # Since we bought at market close to 'price', avg entry should be close to 'price'
        # Slippage might occur, checking within 0.5%
        print(f"  [Audit 1] Long Entry: {real_long['entry_price']} (Ref: {price})")
        self.assertAlmostEqual(real_long['entry_price'], price, delta=price*0.005)
        
        # --- SCENARIO 2: Close One Short (S1: 0.01) ---
        print("\n  >> Closing Virtual Short S1 (0.01)...")
        # Identify S1 (first sell)
        s1 = next(t for t in self.vpm.trades if t.side == "Sell" and t.size == 0.01)
        self.vpm.close_trade(s1.trade_id)
        
        self.reconcile_logic()
        
        # AUDIT
        positions = self.rest.get_positions(SYMBOL)
        real_short = next((p for p in positions if p['position_idx']==2), None)
        
        print(f"  [Audit 2] Short Size: Virtual=0.02, Real={real_short['size']}")
        self.assertAlmostEqual(real_short['size'], 0.02, delta=0.001)
        
        # --- SCENARIO 3: Add Large Long (L3: 0.05) ---
        print("\n  >> Adding Large Long (0.05)...")
        self.vpm.trades[-1].entry_time -= 100
        self.vpm.add_trade("Buy", price, 0.05, price*0.9, price*1.1)
        
        self.reconcile_logic()
        
        # AUDIT
        positions = self.rest.get_positions(SYMBOL)
        real_long = next((p for p in positions if p['position_idx']==1), None)
        
        # Total Long: 0.01 + 0.02 + 0.05 = 0.08
        print(f"  [Audit 3] Long Size: Virtual=0.08, Real={real_long['size']}")
        self.assertAlmostEqual(real_long['size'], 0.08, delta=0.001)
        
        # --- SCENARIO 4: Close All ---
        print("\n  >> Closing All Virtual Positions...")
        # Manually close all in VPM
        for t in list(self.vpm.trades):
            self.vpm.close_trade(t.trade_id)
            
        self.reconcile_logic()
        
        # AUDIT
        positions = self.rest.get_positions(SYMBOL)
        self.assertEqual(len(positions), 0, f"Positions remained: {positions}")
        print("  [Audit 4] Exchange positions cleared.")

if __name__ == '__main__':
    unittest.main()
