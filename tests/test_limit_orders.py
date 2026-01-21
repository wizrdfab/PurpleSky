import unittest
import logging
import sys
import time
import os
import threading
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from bybit_adapter import BybitAdapter, BybitWSAdapter

# API CREDENTIALS
API_KEY = os.getenv("BYBIT_API_KEY", "DZh7qdh2dTfw328ASi")
API_SECRET = os.getenv("BYBIT_API_SECRET", "Y4Jhn5z2MMi0LGN174RRNXLeE2NrRpGZ3mcf")
SYMBOL = "BTCUSDT"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestLimitOrders")

class TestLimitOrders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n=== STARTING LIMIT ORDER TESTS ===")
        cls.rest = BybitAdapter(api_key=API_KEY, api_secret=API_SECRET, testnet=True)
        cls.ws = BybitWSAdapter(api_key=API_KEY, api_secret=API_SECRET, testnet=True)
        
        # Setup Hedge Mode
        cls.rest.switch_position_mode(SYMBOL, mode=3)
        cls.rest.cancel_all_orders(SYMBOL)
        cls._close_all_positions(cls)
        
        # Subscribe to Execution Stream
        cls.filled_orders = []
        def on_exec(msg):
            # Capture fills
            for item in msg.get('data', []):
                print(f"    [WS] Execution: {item.get('side')} {item.get('execQty')} @ {item.get('execPrice')}")
                cls.filled_orders.append(item)
                
        cls.ws.subscribe_private(on_execution=on_exec)
        time.sleep(2) # Connect

    @classmethod
    def tearDownClass(cls):
        print("\n=== TEARDOWN ===")
        cls.rest.cancel_all_orders(SYMBOL)
        cls._close_all_positions(cls)
        cls.ws.disconnect()

    @staticmethod
    def _close_all_positions(cls):
        positions = cls.rest.get_positions(SYMBOL)
        for p in positions:
            side = "Sell" if p['side'] == "Buy" else "Buy"
            idx = p.get('position_idx', 0)
            if idx == 0: continue
            cls.rest.place_order(SYMBOL, side, "Market", p['size'], reduce_only=True, position_idx=idx)

    def test_01_place_limit_maker(self):
        print("\n[Test 01] Place Maker Limit Order (Passive)")
        price = self.rest.get_current_price(SYMBOL)
        
        # Buy Limit well below price (Maker)
        limit_price = round(price * 0.95, 1)
        qty = 0.005
        
        print(f"  - Placing Limit Buy @ {limit_price}...")
        res = self.rest.place_order(SYMBOL, "Buy", "Limit", qty, price=limit_price, position_idx=1)
        self.assertNotIn('error', res)
        oid = res['order_id']
        
        # Verify Open
        orders = self.rest.get_open_orders(SYMBOL)
        target = next((o for o in orders if o['orderId'] == oid), None)
        self.assertIsNotNone(target)
        self.assertEqual(float(target['price']), limit_price)
        print("    [OK] Order is Open and Passive")
        
        # Cancel
        self.rest.cancel_order(SYMBOL, oid)
        time.sleep(1)
        orders = self.rest.get_open_orders(SYMBOL)
        self.assertFalse(any(o['orderId'] == oid for o in orders))
        print("    [OK] Order Cancelled")

    def test_02_place_limit_taker_fill(self):
        print("\n[Test 02] Place Aggressive Limit Order (Immediate Fill)")
        # We want to fill immediately to test "Limit Position" creation
        
        # Fetch Orderbook to find Best Ask
        ob = self.rest.get_orderbook(SYMBOL)
        best_ask = ob['asks'][0][0]
        
        # Place Buy Limit @ Best Ask + buffer (to ensure fill)
        limit_price = round(best_ask * 1.001, 1)
        qty = 0.01
        
        print(f"  - Placing Limit Buy @ {limit_price} (Best Ask: {best_ask})...")
        
        # Clear capture list
        self.filled_orders.clear()
        
        res = self.rest.place_order(SYMBOL, "Buy", "Limit", qty, price=limit_price, position_idx=1)
        self.assertNotIn('error', res)
        
        # Wait for Fill (WS)
        print("  - Waiting for execution...")
        start = time.time()
        while time.time() - start < 5:
            if len(self.filled_orders) > 0: break
            time.sleep(0.1)
            
        self.assertGreater(len(self.filled_orders), 0, "No execution received via WS")
        fill = self.filled_orders[0]
        self.assertEqual(fill['symbol'], SYMBOL)
        self.assertEqual(fill['side'], 'Buy')
        print(f"    [OK] Order Filled via WS: {fill['execQty']} @ {fill['execPrice']}")
        
        # Verify Position on Exchange
        positions = self.rest.get_positions(SYMBOL)
        long_pos = next((p for p in positions if p['position_idx'] == 1), None)
        self.assertIsNotNone(long_pos)
        self.assertAlmostEqual(long_pos['size'], qty, delta=0.001)
        print(f"    [OK] Position Created: {long_pos['size']}")

    def test_03_limit_reduce_only(self):
        print("\n[Test 03] Limit Close (ReduceOnly)")
        # We have Long 0.01 from Test 02
        
        ob = self.rest.get_orderbook(SYMBOL)
        best_bid = ob['bids'][0][0]
        
        # Place Sell Limit @ Best Bid - buffer (Aggressive Close)
        limit_price = round(best_bid * 0.999, 1)
        qty = 0.01
        
        print(f"  - Placing Limit Sell (ReduceOnly) @ {limit_price}...")
        self.filled_orders.clear()
        
        res = self.rest.place_order(SYMBOL, "Sell", "Limit", qty, price=limit_price, reduce_only=True, position_idx=1)
        self.assertNotIn('error', res)
        
        # Wait for Fill
        start = time.time()
        while time.time() - start < 5:
            if len(self.filled_orders) > 0: break
            time.sleep(0.1)
            
        self.assertGreater(len(self.filled_orders), 0)
        print("    [OK] Close Filled")
        
        # Verify Position Gone
        time.sleep(1)
        positions = self.rest.get_positions(SYMBOL)
        long_pos = next((p for p in positions if p['position_idx'] == 1), None)
        self.assertTrue(long_pos is None or long_pos['size'] == 0)
        print("    [OK] Position Closed")

if __name__ == '__main__':
    unittest.main()
