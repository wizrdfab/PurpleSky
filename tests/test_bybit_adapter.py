import unittest
from unittest.mock import MagicMock
from bybit_adapter import BybitAdapter

class TestBybitAdapter(unittest.TestCase):
    def setUp(self):
        self.exchange = BybitAdapter(api_key="dummy", api_secret="dummy", testnet=True)
        self.exchange.session = MagicMock() # Mock the HTTP session
        self.symbol = "BTCUSDT"

    def test_get_public_klines(self):
        print("\nTesting Public Klines (Mock)પૂર્ણ...")
        # Mock Response
        self.exchange.session.get_kline.return_value = {
            'retCode': 0,
            'result': {
                'list': [['1670000000000', '100', '110', '90', '105', '1000', '0']]
            }
        }
        klines = self.exchange.get_public_klines(self.symbol, "1h", limit=5)
        self.assertEqual(len(klines), 1)
        self.assertEqual(klines[0]['close'], 105.0)
        print("Success.")

    def test_get_positions(self):
        print("\nTesting Get Positions (Mock)...")
        self.exchange.session.get_positions.return_value = {
            'retCode': 0,
            'result': {
                'list': [{
                    'symbol': 'BTCUSDT', 'side': 'Buy', 'size': '0.1', 
                    'avgPrice': '20000', 'unrealisedPnl': '50', 'positionIdx': 0
                }]
            }
        }
        positions = self.exchange.get_positions(self.symbol)
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]['size'], 0.1)
        print("Success.")

if __name__ == '__main__':
    unittest.main()