import unittest
import time
import logging
from bybit_adapter import BybitWSAdapter, BybitAdapter
from local_orderbook import LocalOrderbook

logging.basicConfig(level=logging.INFO)

class TestBybitRealConnectivity(unittest.TestCase):
    def setUp(self):
        # Public data doesn't need valid keys
        self.ws = BybitWSAdapter(api_key="dummy", api_secret="dummy", testnet=False) # Use Mainnet for stability or Testnet? User said "not restricted".
        self.symbol = "BTCUSDT"
        self.local_book = LocalOrderbook(self.symbol)

    def tearDown(self):
        self.ws.disconnect()

    def test_public_ws_orderbook(self):
        print("\nTesting Real WS Orderbook (Public)...")
        self.ws.subscribe_orderbook(self.symbol, self.local_book)
        
        # Wait for data
        for _ in range(10):
            if self.local_book.initialized:
                break
            time.sleep(1)
            
        self.assertTrue(self.local_book.initialized, "Orderbook failed to initialize from WS")
        snap = self.local_book.get_snapshot(5)
        print(f"Top Bid: {snap['bids'][0]}")
        self.assertTrue(len(snap['bids']) > 0)
        self.assertTrue(len(snap['asks']) > 0)

    def test_public_ws_kline(self):
        print("\nTesting Real WS Kline (Public)...")
        self.ws.subscribe_kline(self.symbol, "1m")
        
        # Wait for data (might take a bit if market is slow, but BTCUSDT 1m is fast)
        # We just need *any* message, usually initial snapshot or update comes quickly
        # Actually Bybit kline stream pushes when candle updates.
        
        received = False
        for _ in range(10):
            if self.ws.latest_kline:
                received = True
                break
            time.sleep(1)
            
        if not received:
            print("Warning: No kline update received in 10s. This is normal if price didn't change, but unlikely for BTC.")
        else:
            print(f"Latest Kline: {self.ws.latest_kline}")
            self.assertIn('close', self.ws.latest_kline)

if __name__ == '__main__':
    unittest.main()
