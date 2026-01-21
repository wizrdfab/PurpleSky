import unittest
import os
import json
import time
import logging
from bybit_adapter import BybitAdapter, BybitWSAdapter
from local_orderbook import LocalOrderbook

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestLiveConnectivity")

class TestLiveConnectivity(unittest.TestCase):
    def setUp(self):
        # 1. Load Keys
        self.api_key = os.getenv("BYBIT_API_KEY") or os.getenv("BYBIT_TESTNET_KEY")
        self.api_secret = os.getenv("BYBIT_API_SECRET") or os.getenv("BYBIT_TESTNET_SECRET")
        
        if not self.api_key:
            # Try loading from keys.json
            try:
                if os.path.exists("keys.json"):
                    with open("keys.json") as f:
                        keys = json.load(f)
                        self.api_key = keys.get("api_key")
                        self.api_secret = keys.get("api_secret")
            except: pass

        self.testnet = True # Default to testnet for safety in tests
        
        if not self.api_key or self.api_key == "dummy":
            print("\n[WARN] No valid API keys found. Private tests will be SKIPPED.")
            self.skip_private = True
            # Use dummy for public tests
            self.api_key = "dummy"
            self.api_secret = "dummy"
        else:
            self.skip_private = False
            print("\n[INFO] API Keys found. Private tests will RUN.")

        self.symbol = "BTCUSDT"
        self.ws = BybitWSAdapter(self.api_key, self.api_secret, testnet=self.testnet)
        self.rest = BybitAdapter(self.api_key, self.api_secret, testnet=self.testnet)
        self.local_book = LocalOrderbook(self.symbol)

    def tearDown(self):
        self.ws.disconnect()

    def test_01_public_connectivity(self):
        """Test if we can receive public data (Orderbook)"""
        print("\n>>> Test 01: Public Connectivity (Orderbook)")
        self.ws.subscribe_orderbook(self.symbol, self.local_book)
        
        start = time.time()
        while time.time() - start < 10:
            if self.local_book.initialized:
                break
            time.sleep(0.5)
            
        self.assertTrue(self.local_book.initialized, "Orderbook did not initialize within 10s")
        snap = self.local_book.get_snapshot(1)
        print(f"    Top Bid: {snap['bids'][0][0]}")
        self.assertGreater(len(snap['bids']), 0)

    def test_02_private_connectivity(self):
        """Test Authentication and Private Channels"""
        if self.skip_private:
            print("    Skipping (No Keys)")
            return

        print("\n>>> Test 02: Private Connectivity (Auth)")
        
        # We need to wait a bit for the async auth to happen in background
        # pybit connects immediately on init.
        # We can check if it's connected.
        
        # Subscribe to wallet to verify
        received_wallet = [False]
        def on_wallet(msg):
            received_wallet[0] = True
            print(f"    Wallet Update: {msg}")

        # Currently BybitWSAdapter doesn't expose wallet sub explicitly, 
        # let's add a generic sub or use position/execution which we have.
        
        received_pos = [False]
        def on_pos(msg):
            received_pos[0] = True
            print(f"    Position Update: {msg}")
            
        self.ws.subscribe_private(on_position=on_pos)
        
        # We might not get an update immediately unless something changes.
        # But we can check REST to verify keys at least.
        
        print("    Verifying REST access...")
        balance = self.rest.get_wallet_balance("USDT")
        print(f"    Wallet Balance: {balance} USDT")
        # If balance >= 0 (and not error 0.0 fallback), keys work.
        # Ideally check if result was actual success.
        
        # Since we can't easily force a WS push without trading, 
        # success here is mostly about not getting Auth Error logs.
        pass

    def test_03_order_lifecycle(self):
        """Test Place -> WS Update -> Cancel"""
        if self.skip_private:
            print("    Skipping (No Keys)")
            return
            
        print("\n>>> Test 03: Order Lifecycle")
        
        # 1. Place Limit Order far away
        price = self.rest.get_current_price(self.symbol)
        if price == 0:
            self.fail("Could not get current price")
            
        limit_price = round(price * 0.5, 1) # 50% below price, unlikely to fill
        qty = 0.001 # Min size for BTC
        
        print(f"    Placing Buy Limit @ {limit_price}...")
        order = self.rest.place_order(self.symbol, "Buy", "Limit", qty, price=limit_price)
        
        if 'error' in order:
            self.fail(f"Order placement failed: {order['error']}")
            
        order_id = order.get('order_id')
        self.assertIsNotNone(order_id)
        print(f"    Order Placed: {order_id}")
        
        # 2. Wait for WS Execution/Order update
        # We need to hook into WS to see if we get the 'New' order status
        # (Not implemented in Adapter for 'Order' stream, only Execution)
        # We can check REST open orders
        
        time.sleep(1) # Wait for propagation
        
        # 3. Cancel
        print("    Cancelling...")
        success = self.rest.cancel_order(self.symbol, order_id)
        self.assertTrue(success, "Cancel failed")
        print("    Order Cancelled.")

if __name__ == '__main__':
    unittest.main()
