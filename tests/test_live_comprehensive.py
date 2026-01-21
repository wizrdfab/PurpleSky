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
from local_orderbook import LocalOrderbook

# API CREDENTIALS (TESTNET)
API_KEY = os.getenv("BYBIT_API_KEY", "DZh7qdh2dTfw328ASi")
API_SECRET = os.getenv("BYBIT_API_SECRET", "Y4Jhn5z2MMi0LGN174RRNXLeE2NrRpGZ3mcf")
SYMBOL = "BTCUSDT"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestComprehensive")

class TestComprehensive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n=== STARTING COMPREHENSIVE BYBIT TESTS ===")
        cls.rest = BybitAdapter(api_key=API_KEY, api_secret=API_SECRET, testnet=True)
        cls.ws = BybitWSAdapter(api_key=API_KEY, api_secret=API_SECRET, testnet=True)
        
        # Ensure Hedge Mode
        print(">> Setup: Switching to Hedge Mode...")
        cls.rest.switch_position_mode(SYMBOL, mode=3)
        cls.rest.set_leverage(SYMBOL, leverage=10)
        cls.rest.cancel_all_orders(SYMBOL)
        
        # Close all positions
        positions = cls.rest.get_positions(SYMBOL)
        for p in positions:
            side = "Sell" if p['side'] == "Buy" else "Buy"
            cls.rest.place_order(SYMBOL, side, "Market", p['size'], reduce_only=True, position_idx=p['position_idx'])
        
        time.sleep(2)

    @classmethod
    def tearDownClass(cls):
        print("\n=== TEARDOWN ===")
        cls.rest.cancel_all_orders(SYMBOL)
        cls.ws.disconnect()

    def test_01_rest_market_data(self):
        print("\n[Test 01] REST Market Data")
        
        # Klines
        print("  - Fetching Klines...")
        klines = self.rest.get_public_klines(SYMBOL, "15m", limit=5)
        self.assertIsInstance(klines, list)
        self.assertGreater(len(klines), 0)
        print(f"    Received {len(klines)} klines. Latest Close: {klines[-1]['close']}")
        
        # Orderbook
        print("  - Fetching Orderbook...")
        ob = self.rest.get_orderbook(SYMBOL)
        self.assertIn('bids', ob)
        self.assertGreater(len(ob['bids']), 0)
        print(f"    Top Bid: {ob['bids'][0][0]}")

    def test_02_websocket_stream(self):
        print("\n[Test 02] WebSocket Streams")
        
        # Local Orderbook
        local_ob = LocalOrderbook(SYMBOL)
        print("  - Subscribing to Orderbook...")
        self.ws.subscribe_orderbook(SYMBOL, local_ob)
        
        # Klines
        kline_data = []
        def on_kline_update(msg):
            kline_data.append(msg)
            
        print("  - Subscribing to Klines...")
        self.ws.subscribe_kline(SYMBOL, "1m") # 1m for frequent updates? (Might not update if no candle close/tick)
        # Actually Pybit sends updates on candle change or interval.
        # We'll hook into ws_public directly to inject a callback if needed, 
        # but the adapter stores `latest_kline`.
        
        # Wait for data
        print("  - Waiting for WS data (5s)...")
        start = time.time()
        has_ob = False
        has_kline = False
        
        while time.time() - start < 5:
            if local_ob.initialized: has_ob = True
            if self.ws.latest_kline: has_kline = True
            if has_ob and has_kline: break
            time.sleep(0.5)
            
        if not has_ob: print("    [WARN] No Orderbook snapshot received (Geo restriction?)")
        else: print("    [OK] Orderbook synced via WS")
        
        if not has_kline: print("    [WARN] No Kline update received")
        else: print(f"    [OK] Kline received: {self.ws.latest_kline.get('close')}")
        
        self.assertTrue(has_ob, "WS Orderbook failed")

    def test_03_advanced_orders(self):
        print("\n[Test 03] Advanced Orders")
        price = self.rest.get_current_price(SYMBOL)
        
        # 1. Limit Order
        limit_price = round(price * 0.8, 1)
        print(f"  - Placing Limit Buy @ {limit_price}...")
        res = self.rest.place_order(SYMBOL, "Buy", "Limit", 0.005, price=limit_price, position_idx=1)
        self.assertNotIn('error', res)
        oid = res['order_id']
        
        # 2. Check Open Orders
        orders = self.rest.get_open_orders(SYMBOL)
        self.assertTrue(any(o['orderId'] == oid for o in orders), "Limit order not found in open orders")
        print("    [OK] Order found in open orders")
        
        # 3. Conditional Order (Stop Loss simulation)
        # Trigger price below current price for Sell Stop (protecting Long)
        trigger = round(price * 0.7, 1)
        # Since Trigger < Price, Direction = 2 (Fall)
        print(f"  - Placing Conditional Sell Stop @ {trigger} (Dir=2)...")
        res_cond = self.rest.place_order(
            SYMBOL, "Sell", "Market", 0.005, 
            trigger_price=trigger, 
            reduce_only=True, 
            position_idx=1,
            trigger_direction=2
        )
        self.assertNotIn('error', res_cond)
        cond_oid = res_cond['order_id']
        print(f"    Conditional Order Placed: {cond_oid}")
        
        # 4. Cancel All
        print("  - Executing Cancel All...")
        self.rest.cancel_all_orders(SYMBOL)
        time.sleep(1)
        
        orders_after = self.rest.get_open_orders(SYMBOL)
        self.assertEqual(len(orders_after), 0, f"Orders remained after Cancel All: {orders_after}")
        print("    [OK] All orders cancelled")

    def test_04_hedge_logic_verify(self):
        print("\n[Test 04] Hedge Mode Logic verification")
        
        # Open Long
        print("  - Opening Long (0.01)...")
        self.rest.place_order(SYMBOL, "Buy", "Market", 0.01, position_idx=1)
        
        # Open Short
        print("  - Opening Short (0.01)...")
        self.rest.place_order(SYMBOL, "Sell", "Market", 0.01, position_idx=2)
        
        time.sleep(2)
        positions = self.rest.get_positions(SYMBOL)
        
        long_p = next((p for p in positions if p['position_idx'] == 1), None)
        short_p = next((p for p in positions if p['position_idx'] == 2), None)
        
        self.assertIsNotNone(long_p)
        self.assertIsNotNone(short_p)
        print(f"    [OK] Simultaneous positions held: L={long_p['size']}, S={short_p['size']}")
        
        # Close Long
        print("  - Closing Long...")
        self.rest.place_order(SYMBOL, "Sell", "Market", 0.01, reduce_only=True, position_idx=1)
        
        # Close Short
        print("  - Closing Short...")
        self.rest.place_order(SYMBOL, "Buy", "Market", 0.01, reduce_only=True, position_idx=2)
        
        time.sleep(1)
        positions_final = self.rest.get_positions(SYMBOL)
        self.assertEqual(len(positions_final), 0, f"Positions remained: {positions_final}")
        print("    [OK] Positions closed cleanly")

    def test_05_multi_position_aggregation(self):
        print("\n[Test 05] Multi-Position Aggregation (Virtual -> Real)")
        
        # 1. Open First Long (0.01)
        print("  - Virtual: Add Long 0.01")
        self.rest.place_order(SYMBOL, "Buy", "Market", 0.01, position_idx=1)
        
        # 2. Add Second Long (0.02) - Simulating adding to position
        print("  - Virtual: Add Long 0.02")
        self.rest.place_order(SYMBOL, "Buy", "Market", 0.02, position_idx=1)
        
        time.sleep(2)
        positions = self.rest.get_positions(SYMBOL)
        long_p = next((p for p in positions if p['position_idx'] == 1), None)
        self.assertIsNotNone(long_p)
        # 0.01 + 0.02 = 0.03
        self.assertAlmostEqual(long_p['size'], 0.03, delta=0.001)
        print(f"    [OK] Aggregated Long Size: {long_p['size']}")
        
        # 3. Add Short (0.01)
        print("  - Virtual: Add Short 0.01")
        self.rest.place_order(SYMBOL, "Sell", "Market", 0.01, position_idx=2)
        
        time.sleep(2)
        positions = self.rest.get_positions(SYMBOL)
        short_p = next((p for p in positions if p['position_idx'] == 2), None)
        self.assertIsNotNone(short_p)
        self.assertEqual(short_p['size'], 0.01)
        print(f"    [OK] Short Size: {short_p['size']}")
        
        # 4. Partial Close Long (0.01) - Simulating closing one virtual trade
        print("  - Virtual: Close First Long (0.01)")
        self.rest.place_order(SYMBOL, "Sell", "Market", 0.01, reduce_only=True, position_idx=1)
        
        time.sleep(2)
        positions = self.rest.get_positions(SYMBOL)
        long_p = next((p for p in positions if p['position_idx'] == 1), None)
        self.assertAlmostEqual(long_p['size'], 0.02, delta=0.001)
        print(f"    [OK] Remaining Long Size: {long_p['size']}")
        
        # Cleanup
        print("  - Closing remaining positions...")
        self.rest.place_order(SYMBOL, "Sell", "Market", 0.02, reduce_only=True, position_idx=1)
        self.rest.place_order(SYMBOL, "Buy", "Market", 0.01, reduce_only=True, position_idx=2)

if __name__ == '__main__':
    unittest.main()
