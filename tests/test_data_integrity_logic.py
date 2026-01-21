import unittest
import logging
import sys
import os
import pandas as pd
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

sys.path.append(str(Path(__file__).parent.parent))

from live_trading import LiveBot
from config import CONF

# Mock API for fast testing
class MockAPI:
    def get_instrument_info(self, symbol):
        return {'tick_size': 0.1, 'qty_step': 0.01, 'min_qty': 0.01, 'min_notional': 5.0}
    def get_wallet_balance(self, coin): return 1000.0
    def check_clock_drift(self): return 0.0

class TestDataIntegrityLogic(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            symbol="BTCUSDT",
            model_dir="dummy", 
            api_key="key",
            api_secret="secret",
            testnet=True,
            timeframe="5m"
        )
        
        # Patch heavy loads
        with patch('live_trading.BybitAdapter', return_value=MockAPI()):
            with patch('live_trading.BybitWSAdapter'):
                with patch('live_trading.LiveBot._load_model_manager') as mock_mm:
                    self.bot = LiveBot(self.args)
                    self.bot.model_manager = mock_mm.return_value
                    self.bot.model_manager.feature_cols = ['close', 'atr']
                    # Mock prediction to return real numbers
                    self.bot.model_manager.predict = MagicMock(return_value=pd.DataFrame([{
                        'pred_long': 0.5, 'pred_short': 0.5, 'pred_dir_long': 0.5, 'pred_dir_short': 0.5
                    }]))
                    self.bot.warmup_bars = 0

    def trigger_bar(self, timestamp, close=100.0, ob_spread=0.0001, volume=1000):
        # Setup OB aggregator to return specific spread
        self.bot.ob_agg.finalize = MagicMock(return_value={
            'ob_spread_mean': ob_spread, 'ob_micro_dev_mean': 0, 'ob_micro_dev_std': 0,
            'ob_micro_dev_last': 0, 'ob_imbalance_mean': 0, 'ob_imbalance_last': 0,
            'ob_bid_depth_mean': 10, 'ob_ask_depth_mean': 10, 'ob_bid_slope_mean': 0,
            'ob_ask_slope_mean': 0, 'ob_bid_integrity_mean': 0, 'ob_ask_integrity_mean': 0
        })
        
        fake_kline = {
            'start': timestamp, 'open': close, 'high': close, 'low': close, 
            'close': close, 'volume': volume, 'confirm': True
        }
        
        # Mock Feature Engine to return what we gave it + basic placeholders
        def mock_calc(df):
            res = df.copy()
            if 'atr' not in res.columns: res['atr'] = res['close'] * 0.01
            return res
            
        self.bot.feature_engine.calculate_features = MagicMock(side_effect=mock_calc)
        
        with self.assertLogs('LiveTrading', level='INFO') as cm:
            self.bot.on_bar_close(fake_kline)
            return cm.output

    def test_gap_detection(self):
        """Verify that missing bars are flagged"""
        print("\n[Integrity Test] Gap Detection")
        start_ts = 1000000000
        tf_ms = 300 * 1000 # 5m
        
        # Bar 1
        self.trigger_bar(start_ts)
        
        # Bar 2 - CORRECT (start_ts + 5m)
        logs = self.trigger_bar(start_ts + tf_ms)
        self.assertTrue(any("Continuity: OK (Continuous)" in l for l in logs))
        print("    [OK] Continuous data identified correctly.")
        
        # Bar 3 - GAP (start_ts + 15m instead of +10m)
        logs = self.trigger_bar(start_ts + (tf_ms * 3))
        self.assertTrue(any("!!! GAP DETECTED" in l for l in logs))
        print("    [OK] Gap identified correctly.")

    def test_nan_detection(self):
        """Verify that NaN features are flagged"""
        print("\n[Integrity Test] NaN Detection")
        
        # Trigger bar with a NaN close
        logs = self.trigger_bar(time.time()*1000, close=float('nan'))
        self.assertTrue(any("NaNs Detected" in l for l in logs))
        print("    [OK] NaN value identified correctly.")

    def test_stale_orderbook(self):
        """Verify that zero spread flags stale orderbook"""
        print("\n[Integrity Test] Stale Orderbook")
        
        # Trigger bar with 0 spread
        logs = self.trigger_bar(time.time()*1000, ob_spread=0.0)
        self.assertTrue(any("!!! STALE/EMPTY ORDERBOOK !!!" in l for l in logs))
        print("    [OK] Stale orderbook identified correctly.")

    def test_latency_warning(self):
        """Verify that slow processing is flagged"""
        print("\n[Integrity Test] Latency Warning")
        
        # Bar timestamp is 10 seconds ago
        old_ts = int((time.time() - 310) * 1000) # 5m bar would have ended 10s ago
        
        logs = self.trigger_bar(old_ts)
        # Latency check looks at (BarStart + 5m) vs Now.
        # If BarStart was 310s ago, BarEnd was 10s ago. 
        # Processing now means lag is ~10,000ms.
        self.assertTrue(any("(HIGH LATENCY)" in l for l in logs))
        print("    [OK] High latency identified correctly.")

if __name__ == '__main__':
    unittest.main()
