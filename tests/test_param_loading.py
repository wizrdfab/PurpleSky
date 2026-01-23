import unittest
import json
import os
from pathlib import Path
from live_trading import LiveBot
from config import CONF

class TestParamLoading(unittest.TestCase):
    def setUp(self):
        # Create a mock params.json for testing
        self.model_dir = Path("test_model_params")
        self.model_dir.mkdir(exist_ok=True)
        self.params_file = self.model_dir / "params.json"
        
        self.mock_params = {
            "model_threshold": 0.99,
            "direction_threshold": 0.88,
            "aggressive_threshold": 0.77,
            "take_profit_atr": 5.5,
            "stop_loss_atr": 9.9,
            "limit_offset_atr": 1.1
        }
        
        with open(self.params_file, 'w') as f:
            json.dump(self.mock_params, f)
            
        # We also need a dummy features.pkl, model_long.pkl, model_short.pkl to satisfy _load_model_manager
        import joblib
        joblib.dump(['close'], self.model_dir / "features.pkl")
        joblib.dump(None, self.model_dir / "model_long.pkl")
        joblib.dump(None, self.model_dir / "model_short.pkl")

        class Args:
            symbol = "TESTUSDT"
            model_dir = "test_model_params"
            api_key = ""
            api_secret = ""
            testnet = False
            timeframe = "5m"
        self.args = Args()

    def tearDown(self):
        # Clean up files
        for f in self.model_dir.glob("*"):
            f.unlink()
        self.model_dir.rmdir()

    def test_parameter_override(self):
        """Verify that LiveBot overrides GlobalConfig with values from params.json"""
        # Save original values
        orig_agg = CONF.model.aggressive_threshold
        
        # Initialize Bot (this triggers _load_model_manager)
        bot = LiveBot(self.args)
        
        # Assertions
        self.assertEqual(CONF.model.model_threshold, 0.99)
        self.assertEqual(CONF.model.direction_threshold, 0.88)
        self.assertEqual(CONF.model.aggressive_threshold, 0.77)
        self.assertEqual(CONF.strategy.take_profit_atr, 5.5)
        self.assertEqual(CONF.strategy.stop_loss_atr, 9.9)
        self.assertEqual(CONF.strategy.base_limit_offset_atr, 1.1)
        
        print(f"\n[OK] Loaded Aggressive Threshold: {CONF.model.aggressive_threshold}")
        print(f"[OK] Loaded Stop Loss ATR: {CONF.strategy.stop_loss_atr}")

if __name__ == "__main__":
    unittest.main()
