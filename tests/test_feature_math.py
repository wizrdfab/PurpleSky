import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from feature_engine import FeatureEngine
from config import FeatureConfig

class TestFeatureMath(unittest.TestCase):
    def setUp(self):
        self.config = FeatureConfig()
        self.engine = FeatureEngine(self.config)

    def test_rsi_calculation(self):
        """Verify RSI against a known sequence."""
        # RSI 14-period
        prices = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00]
        df = pd.DataFrame({
            'open': prices, 'high': prices, 'low': prices, 'close': prices, 'volume': [100]*len(prices)
        })
        
        # We need a longer sequence for EMA/RSI to stabilize, but we check if it's moving in right direction
        feats = self.engine.calculate_features(df)
        last_rsi = feats['rsi'].iloc[-1]
        
        # With this upward sequence, RSI should be > 50
        self.assertGreater(last_rsi, 50)
        self.assertLessEqual(last_rsi, 100)
        print(f"    [OK] RSI math producing sane values: {last_rsi:.2f}")

    def test_ob_imbalance_math(self):
        """Verify Orderbook Imbalance logic."""
        df = pd.DataFrame({
            'close': [100], 'high': [101], 'low': [99], 'volume': [1000],
            'ob_bid_depth_mean': [1000],
            'ob_ask_depth_mean': [3000],
            'ob_spread_mean': [0.1],
            'ob_micro_dev_mean': [0.01],
            'ob_micro_dev_std': [0.001],
            'ob_micro_dev_last': [0.01],
            'ob_imbalance_mean': [-0.5], # 1000 vs 3000 -> (1-3)/(1+3) = -0.5
            'ob_imbalance_last': [-0.5],
            'ob_bid_slope_mean': [0.1], 'ob_ask_slope_mean': [0.1],
            'ob_bid_integrity_mean': [0.5], 'ob_ask_integrity_mean': [0.5]
        })
        
        # Calculate features (requires some history for rolling)
        df_long = pd.concat([df]*30, ignore_index=True)
        feats = self.engine.calculate_features(df_long)
        
        # Depth Log Ratio: log(1000/3000) = log(0.333) approx -1.09
        expected_log_ratio = np.log(1000/3000)
        actual_log_ratio = feats['ob_depth_log_ratio'].iloc[-1]
        self.assertAlmostEqual(actual_log_ratio, expected_log_ratio, places=4)
        print(f"    [OK] OB Imbalance/Depth math correct: {actual_log_ratio:.4f}")

if __name__ == '__main__':
    unittest.main()
