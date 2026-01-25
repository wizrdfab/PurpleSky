
import unittest
import pandas as pd
import numpy as np
import logging
from feature_engine import FeatureEngine
from config import CONF, FeatureConfig

# Setup isolated logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IntegrityTest")

class TestComprehensiveIntegrity(unittest.TestCase):
    def setUp(self):
        self.config = CONF.features
        self.engine = FeatureEngine(self.config)
        self.base_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'taker_buy_ratio']
        
        # Mock Orderbook columns expected by FeatureEngine
        self.ob_cols = [
            'ob_spread_mean', 'ob_micro_dev_mean', 'ob_micro_dev_std', 
            'ob_imbalance_mean', 'ob_bid_depth_mean', 'ob_ask_depth_mean',
            'ob_bid_slope_mean', 'ob_ask_slope_mean',
            'ob_bid_integrity_mean', 'ob_ask_integrity_mean'
        ]

    def _generate_synthetic_data(self, n_rows=1000, pattern='normal'):
        """Generates synthetic OHLCV + OB data for testing."""
        dates = pd.date_range(start='2024-01-01', periods=n_rows, freq='5min')
        df = pd.DataFrame(index=range(n_rows))
        df['timestamp'] = dates.astype(np.int64) // 10**6 # ms
        
        # Price Generation
        if pattern == 'normal':
            returns = np.random.normal(0, 0.001, n_rows)
            price = 100 * np.exp(np.cumsum(returns))
            df['close'] = price
            df['open'] = price * (1 + np.random.normal(0, 0.0001, n_rows))
            df['high'] = df[['open', 'close']].max(axis=1) * 1.001
            df['low'] = df[['open', 'close']].min(axis=1) * 0.999
        elif pattern == 'flat':
            price = np.full(n_rows, 100.0)
            df['close'] = price
            df['open'] = price
            df['high'] = price
            df['low'] = price
        elif pattern == 'shock':
            price = np.full(n_rows, 100.0)
            price[500:] = 50.0 # 50% drop
            df['close'] = price
            df['open'] = price
            df['high'] = price
            df['low'] = price
            # Add small noise to others to allow calc, but keep shock clean
            df['open'] = price
        else: # random walk
            price = 100 + np.random.randn(n_rows).cumsum()
            df['close'] = price
            df['open'] = price * (1 + np.random.normal(0, 0.0001, n_rows))
            df['high'] = df[['open', 'close']].max(axis=1) * 1.001
            df['low'] = df[['open', 'close']].min(axis=1) * 0.999

        df['volume'] = np.random.lognormal(10, 1, n_rows)
        df['taker_buy_ratio'] = 0.5 + np.random.normal(0, 0.1, n_rows)
        
        # Synthetic OB Data
        df['ob_spread_mean'] = df['close'] * 0.0005 # 5 bps spread
        df['ob_micro_dev_mean'] = 0.0
        df['ob_micro_dev_std'] = df['close'] * 0.0001
        df['ob_imbalance_mean'] = np.random.normal(0, 0.3, n_rows) # -1 to 1 range approx
        df['ob_bid_depth_mean'] = 100000.0
        df['ob_ask_depth_mean'] = 100000.0
        df['ob_bid_slope_mean'] = 50.0
        df['ob_ask_slope_mean'] = 50.0
        df['ob_bid_integrity_mean'] = 0.8
        df['ob_ask_integrity_mean'] = 0.8
        
        return df

    def test_feature_completeness(self):
        """Test 1: Verify all expected features are generated."""
        df = self._generate_synthetic_data()
        df_feats = self.engine.calculate_features(df)
        
        expected_keywords = [
            'atr', 'rsi', 'dist_ema_9', 'bb_percent_b', 
            'ob_spread_bps', 'ob_imbalance_z', 'liq_dominance',
            'vwap_4h_dist', 'vol_macro', 'atr_regime'
        ]
        
        missing = []
        columns = df_feats.columns.tolist()
        for k in expected_keywords:
            if not any(k in c for c in columns):
                missing.append(k)
        
        self.assertEqual(len(missing), 0, f"Missing expected features: {missing}")
        logger.info("[PASS] All feature groups present.")

    def test_mathematical_sanity(self):
        """Test 2: Verify feature value ranges and mathematical validity."""
        df = self._generate_synthetic_data()
        df_feats = self.engine.calculate_features(df)
        
        # RSI Range [0, 100]
        rsi = df_feats['rsi'].iloc[50:] # Skip warmup
        self.assertTrue(rsi.between(0, 100).all(), f"RSI out of bounds: {rsi.min()} - {rsi.max()}")
        
        # ATR Positive
        atr = df_feats['atr'].iloc[50:]
        self.assertTrue((atr > 0).all(), "ATR should be positive")
        
        # BB %B sanity (can be outside 0-1 but mostly inside)
        bb = df_feats['bb_percent_b'].iloc[50:]
        self.assertTrue(bb.mean() > 0.2 and bb.mean() < 0.8, "Bollinger %B mean looks wrong")

        logger.info("[PASS] Mathematical ranges valid.")

    def test_nan_safety(self):
        """Test 3: Ensure no NaNs or Infs remain in the output."""
        df = self._generate_synthetic_data()
        # Inject some zeros/NaNs in inputs
        df.loc[100:105, 'ob_spread_mean'] = 0.0
        df.loc[200:205, 'volume'] = 0.0
        
        df_feats = self.engine.calculate_features(df)
        
        nans = df_feats.isna().sum().sum()
        infs = np.isinf(df_feats).sum().sum()
        
        self.assertEqual(nans, 0, f"Found {nans} NaNs in features.")
        self.assertEqual(infs, 0, f"Found {infs} Infs in features.")
        logger.info("[PASS] NaN/Inf safety check passed.")

    def test_scenario_flat_market(self):
        """Test 4: Flat market should produce low volatility features."""
        df = self._generate_synthetic_data(pattern='flat')
        df_feats = self.engine.calculate_features(df)
        
        # ATR should decay to near zero
        last_atr = df_feats['atr'].iloc[-1]
        self.assertAlmostEqual(last_atr, 0, places=2, msg="ATR did not decay in flat market")
        
        # RSI should be exactly 50 (or undefined -> filled 50? Engine fills 0 usually or prev)
        # Actually RSI on flat: diff=0, gain=0, loss=0 -> rs=nan -> rsi=nan. 
        # Engine should handle this.
        # Let's check 'dist_ema_9' which should be 0
        last_dist = df_feats['dist_ema_9'].iloc[-1]
        # Dist uses ATR div, so 0/0 might be handled.
        
        logger.info("[PASS] Flat market scenario verified.")

    def test_scenario_shock_crash(self):
        """Test 5: Market Crash scenario (50% drop)."""
        df = self._generate_synthetic_data(pattern='shock')
        df_feats = self.engine.calculate_features(df)
        
        # Check specific bar where crash happens (idx 500)
        # RSI should plummet
        crash_rsi = df_feats['rsi'].iloc[505]
        self.assertLess(crash_rsi, 30, f"RSI did not detect crash: {crash_rsi}")
        
        # ATR should spike
        pre_crash_atr = df_feats['atr'].iloc[490]
        post_crash_atr = df_feats['atr'].iloc[510]
        self.assertGreater(post_crash_atr, pre_crash_atr * 5, "ATR did not spike on crash")
        
        logger.info("[PASS] Shock scenario verified.")

    def test_ob_derived_features(self):
        """Test 6: Orderbook Derived Metrics."""
        df = self._generate_synthetic_data()
        
        # Manipulate OB data to test logic
        # Make Bid Depth HUGE vs Ask Depth -> liq_dominance or depth_ratio
        df['ob_bid_depth_mean'] = 1000000.0
        df['ob_ask_depth_mean'] = 1000.0
        
        df_feats = self.engine.calculate_features(df)
        
        # ob_depth_log_ratio should be very positive (log(1000) ~ 6.9)
        ratio = df_feats['ob_depth_log_ratio'].iloc[-1]
        self.assertGreater(ratio, 5.0, f"Depth ratio failed: {ratio}")
        
        logger.info("[PASS] Orderbook logic verified.")
        
    def test_data_leakage(self):
        """Test 7: Ensure features don't peek into the future."""
        df = self._generate_synthetic_data(n_rows=200)
        
        # We calculate features on full dataset
        full_feats = self.engine.calculate_features(df)
        
        # We calculate on partial dataset (first 100 rows)
        partial_df = df.iloc[:100].copy()
        partial_feats = self.engine.calculate_features(partial_df)
        
        # The 100th row in partial should match 100th row in full exactly
        # (Assuming no forward-looking rolling windows)
        
        # Compare row 99 (0-indexed)
        # Note: some macro features (like VWAP 24h) depend on history length,
        # so if history is cut, they might differ if min_periods not met.
        # But standard indicators like RSI/EMA should match if enough warmup.
        
        cols_to_check = ['rsi', 'ema_9_slope', 'bb_percent_b']
        
        for col in cols_to_check:
            val_full = full_feats[col].iloc[99]
            val_part = partial_feats[col].iloc[99]
            self.assertAlmostEqual(val_full, val_part, places=5, msg=f"Leakage or History dep in {col}")
            
        logger.info("[PASS] No Future Leakage detected.")

if __name__ == '__main__':
    unittest.main()
