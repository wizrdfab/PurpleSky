import unittest
import logging
import sys
import time
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from bybit_adapter import BybitAdapter

# API CREDENTIALS (TESTNET)
API_KEY = os.getenv("BYBIT_API_KEY", "DZh7qdh2dTfw328ASi")
API_SECRET = os.getenv("BYBIT_API_SECRET", "Y4Jhn5z2MMi0LGN174RRNXLeE2NrRpGZ3mcf")
SYMBOL = "BTCUSDT"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestLiveIntegration")

class TestLiveIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n--- Setting up Bybit Live Integration Test ---")
        cls.adapter = BybitAdapter(api_key=API_KEY, api_secret=API_SECRET, testnet=True)
        
        # 1. Check Connectivity by fetching Time/Price
        price = cls.adapter.get_current_price(SYMBOL)
        if price == 0:
            raise Exception("Failed to connect to Bybit Testnet (Price is 0). Check internet or keys.")
        print(f"Connection Successful. {SYMBOL} Price: {price}")
        
        # 2. Setup Account (Hedge Mode, Leverage)
        print("Configuring Account...")
        cls.adapter.switch_position_mode(SYMBOL, mode=3) # 3 = Hedge Mode
        cls.adapter.set_leverage(SYMBOL, leverage=10)
        
        # 3. Clean start
        print("Cleaning up existing orders/positions...")
        cls.adapter.cancel_all_orders(SYMBOL)
        cls._close_all_positions(cls)

    @classmethod
    def tearDownClass(cls):
        print("\n--- Teardown: Closing all positions ---")
        cls.adapter.cancel_all_orders(SYMBOL)
        cls._close_all_positions(cls)

    @staticmethod
    def _close_all_positions(cls):
        positions = cls.adapter.get_positions(SYMBOL)
        for p in positions:
            side = "Sell" if p['side'] == "Buy" else "Buy" # Close side is opposite
            idx = p['position_idx']
            print(f"Closing {p['side']} (Idx: {idx}) {p['size']}...")
            cls.adapter.place_order(
                symbol=SYMBOL,
                side=side,
                order_type="Market",
                qty=p['size'],
                reduce_only=True,
                position_idx=idx
            )
        time.sleep(1) # Wait for execution

    def test_01_wallet_balance(self):
        print("\n[Test 01] Checking Wallet Balance")
        bal = self.adapter.get_wallet_balance("USDT")
        print(f"USDT Balance: {bal}")
        self.assertIsInstance(bal, float)
        self.assertGreaterEqual(bal, 0.0)

    def test_02_open_long(self):
        print("\n[Test 02] Open Long Position")
        qty = 0.01 
        
        # Position Index 1 = Long Side in Hedge Mode
        res = self.adapter.place_order(SYMBOL, "Buy", "Market", qty, position_idx=1)
        self.assertNotIn('error', res, f"Order failed: {res}")
        print(f"Long Order Placed: {res}")
        
        time.sleep(2) 
        
        positions = self.adapter.get_positions(SYMBOL)
        long_pos = next((p for p in positions if p['side'] == 'Buy'), None)
        
        self.assertIsNotNone(long_pos, "Long position not found")
        self.assertEqual(long_pos['size'], qty)
        print(f"Long Position Verified: {long_pos}")

    def test_03_open_short_hedge(self):
        print("\n[Test 03] Open Short Position (Hedge Mode Check)")
        # We assume Long is still open from test_02
        qty = 0.01
        
        # Position Index 2 = Short Side in Hedge Mode
        res = self.adapter.place_order(SYMBOL, "Sell", "Market", qty, position_idx=2)
        self.assertNotIn('error', res)
        print(f"Short Order Placed: {res}")
        
        time.sleep(2)
        
        positions = self.adapter.get_positions(SYMBOL)
        long_pos = next((p for p in positions if p['side'] == 'Buy'), None)
        short_pos = next((p for p in positions if p['side'] == 'Sell'), None)
        
        self.assertIsNotNone(long_pos, "Long position disappeared!")
        self.assertIsNotNone(short_pos, "Short position not found")
        
        print(f"Hedge Mode Confirmed. Long: {long_pos['size']}, Short: {short_pos['size']}")

    def test_04_limit_orders(self):
        print("\n[Test 04] Place and Cancel Limit Order")
        price = self.adapter.get_current_price(SYMBOL)
        limit_price = round(price * 0.9, 1) # Deep OTM Buy
        
        # Limit Buy for Long Side (idx=1)
        res = self.adapter.place_order(SYMBOL, "Buy", "Limit", 0.005, price=limit_price, position_idx=1)
        self.assertNotIn('error', res)
        order_id = res['order_id']
        print(f"Limit Order Placed: {order_id} @ {limit_price}")
        
        # Verify cancellation
        cancel_success = self.adapter.cancel_order(SYMBOL, order_id)
        self.assertTrue(cancel_success, "Failed to cancel limit order")
        print("Limit Order Cancelled Successfully")

    def test_05_reduce_position(self):
        print("\n[Test 05] Reduce Short Position")
        # Current state: Long 0.01, Short 0.01
        # Close Short fully
        positions = self.adapter.get_positions(SYMBOL)
        short_pos = next((p for p in positions if p['side'] == 'Sell'), None)
        
        if short_pos:
            idx = short_pos['position_idx']
            res = self.adapter.place_order(SYMBOL, "Buy", "Market", short_pos['size'], reduce_only=True, position_idx=idx)
            self.assertNotIn('error', res)
            print("Closed Short Position")
            
        time.sleep(2)
        positions = self.adapter.get_positions(SYMBOL)
        short_pos = next((p for p in positions if p['side'] == 'Sell'), None)
        self.assertIsNone(short_pos, "Short position should be gone")

if __name__ == '__main__':
    unittest.main()