import unittest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from feature_engine import FeatureEngine
from config import FeatureConfig, CONF

class TestComprehensiveFeatures(unittest.TestCase):
    def setUp(self):
        self.config = FeatureConfig()
        self.engine = FeatureEngine(self.config)
        self.expected_features = [
            'close', 'taker_buy_ratio', 'ob_bid_integrity_mean', 'ob_ask_integrity_mean', 
            'natr', 'dist_ema_9', 'dist_ema_21', 'dist_ema_50', 'rsi', 'bb_percent_b', 
            'ema_9_slope', 'ema_21_slope', 'ema_50_slope', 'taker_buy_z', 'vol_z', 
            'ob_imb_trend', 'micro_pressure', 'bid_depth_chg', 'ask_depth_chg', 
            'spread_z', 'ob_spread_bps', 'ob_depth_log_ratio', 'ob_imbalance_z', 
            'price_liq_div', 'liq_dominance', 'micro_dev_vol', 'ob_bid_elasticity', 
            'ob_ask_elasticity', 'ob_slope_ratio', 'bid_slope_z', 'ob_integrity_skew', 
            'bid_integrity_chg', 'ask_integrity_chg', 'vwap_4h_dist', 'vwap_24h_dist', 
            'vol_intraday', 'vol_macro', 'atr_z', 'macro_dist_3d', 'macro_dist_1d', 
            'atr_regime', 'atr_macro'
        ]

    def test_all_features_exist_and_sane(self):
        """Audit every single feature required by the production model."""
        print(f"\n[Comprehensive Audit] Verifying {len(self.expected_features)} features...")
        
        # 1. Create a 1000-bar dataset with realistic movements
        n = 1000
        data = {
            'timestamp': np.arange(n) * 300000,
            'open': 100.0 + np.cumsum(np.random.normal(0, 0.1, n)),
            'high': 100.5 + np.cumsum(np.random.normal(0, 0.1, n)),
            'low': 99.5 + np.cumsum(np.random.normal(0, 0.1, n)),
            'close': 100.0 + np.cumsum(np.random.normal(0, 0.1, n)),
            'volume': np.random.uniform(100, 1000, n),
            'taker_buy_ratio': np.random.uniform(0.4, 0.6, n),
            # Orderbook Base Metrics (Simulated from Aggregator)
            'ob_spread_mean': np.random.uniform(0.01, 0.05, n),
            'ob_micro_dev_mean': np.random.normal(0, 0.001, n),
            'ob_micro_dev_std': np.random.uniform(0.0001, 0.001, n),
            'ob_imbalance_mean': np.random.uniform(-0.5, 0.5, n),
            'ob_bid_depth_mean': np.random.uniform(1000, 5000, n),
            'ob_ask_depth_mean': np.random.uniform(1000, 5000, n),
            'ob_bid_slope_mean': np.random.uniform(0.001, 0.01, n),
            'ob_ask_slope_mean': np.random.uniform(0.001, 0.01, n),
            'ob_bid_integrity_mean': np.random.uniform(0.1, 0.9, n),
            'ob_ask_integrity_mean': np.random.uniform(0.1, 0.9, n),
            'dollar_val': np.random.uniform(10000, 100000, n)
        }
        df = pd.DataFrame(data)
        
        # 2. Run Engine
        feats = self.engine.calculate_features(df)
        
        # 3. Audit
        missing = []
        nans = []
        inf_values = []
        
        # We check the last 100 rows to ensure rolling windows are somewhat active
        check_slice = feats.iloc[-100:]
        
        for f in self.expected_features:
            if f not in feats.columns:
                missing.append(f)
                continue
            
            if check_slice[f].isna().any():
                nans.append(f)
            
            if np.isinf(check_slice[f]).any():
                inf_values.append(f)
        
        # Reports
        if missing: print(f"    !!! MISSING: {missing}")
        if nans: print(f"    !!! NaNs:    {nans}")
        if inf_values: print(f"    !!! INFs:    {inf_values}")
        
        self.assertEqual(len(missing), 0, f"Features missing: {missing}")
        self.assertEqual(len(nans), 0, f"Features contain NaNs: {nans}")
        self.assertEqual(len(inf_values), 0, f"Features contain Infs: {inf_values}")
        
        print(f"    [SUCCESS] All {len(self.expected_features)} features are present, finite, and calculated.")

if __name__ == '__main__':
    unittest.main()
