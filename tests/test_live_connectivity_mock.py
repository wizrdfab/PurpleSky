import unittest
import time
import logging
from unittest.mock import MagicMock, patch
from bybit_adapter import BybitWSAdapter, BybitAdapter
from local_orderbook import LocalOrderbook

logging.basicConfig(level=logging.INFO)

class TestLiveConnectivityMock(unittest.TestCase):
    def setUp(self):
        self.symbol = "BTCUSDT"
        self.api_key = "dummy"
        self.api_secret = "dummy"
        self.local_book = LocalOrderbook(self.symbol)
        
        # Patch pybit WebSocket to prevent actual connection attempt
        self.patcher = patch('bybit_adapter.WebSocket')
        self.MockWebSocket = self.patcher.start()
        
        self.ws = BybitWSAdapter(self.api_key, self.api_secret, testnet=True)
        # Mock the underlying public/private objects
        self.ws.ws_public = MagicMock()
        self.ws.ws_private = MagicMock()
        self.ws.connected = True # Force success

    def tearDown(self):
        self.patcher.stop()

    def test_01_public_orderbook_flow(self):
        print("\n>>> Test 01: Mocked Public Orderbook Flow")
        
        # 1. Subscribe
        self.ws.subscribe_orderbook(self.symbol, self.local_book)
        
        # Verify call
        self.ws.ws_public.orderbook_stream.assert_called_with(
            depth=50, 
            symbol=self.symbol, 
            callback=unittest.mock.ANY
        )
        
        # 2. Simulate Callback (Snapshot)
        callback = self.ws.ws_public.orderbook_stream.call_args[1]['callback']
        snapshot_msg = {
            "topic": f"orderbook.50.{self.symbol}", 
            "type": "snapshot", 
            "ts": 1672304486868, 
            "data": {
                "s": self.symbol,
                "b": [["16800.00", "0.5"], ["16799.00", "1.0"]],
                "a": [["16801.00", "0.2"], ["16802.00", "0.8"]]
            }
        }
        callback(snapshot_msg)
        
        # Verify LocalBook
        self.assertTrue(self.local_book.initialized)
        snap = self.local_book.get_snapshot(1)
        self.assertEqual(snap['bids'][0][0], 16800.0)
        print("    Snapshot applied successfully.")
        
        # 3. Simulate Delta (Update)
        delta_msg = {
            "topic": f"orderbook.50.{self.symbol}", 
            "type": "delta", 
            "ts": 1672304486900, 
            "data": {
                "s": self.symbol,
                "b": [["16800.00", "0"]], # Delete 16800
                "a": []
            }
        }
        callback(delta_msg)
        
        snap_after = self.local_book.get_snapshot(1)
        self.assertEqual(snap_after['bids'][0][0], 16799.0) # Next best bid
        print("    Delta applied successfully.")

    def test_02_private_execution_flow(self):
        print("\n>>> Test 02: Mocked Private Execution Flow")
        
        received = []
        def on_exec(msg):
            received.append(msg)
            
        # 1. Subscribe
        self.ws.subscribe_private(on_execution=on_exec)
        
        # Verify call
        self.ws.ws_private.execution_stream.assert_called()
        
        # 2. Simulate Callback
        callback = self.ws.ws_private.execution_stream.call_args[1]['callback']
        exec_msg = {
            "topic": "execution", 
            "data": [{"symbol": self.symbol, "execQty": "0.1", "side": "Buy"}]
        }
        callback(exec_msg)
        
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]['data'][0]['execQty'], "0.1")
        print("    Execution callback triggered successfully.")

if __name__ == '__main__':
    unittest.main()
