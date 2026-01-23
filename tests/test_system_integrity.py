import unittest
import pandas as pd
import numpy as np
import os
import shutil
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from live_trading import LiveBot
from config import GlobalConfig, CONF

class TestSystemIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Setup temporary test directory
        cls.test_dir = Path("tests/temp_integrity_test")
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Mock Config
        cls.symbol = "TESTUSDT"
        cls.model_dir = "models/FARTCOINUSDT/rank_1" # Use existing model dir for features
        
        # Ensure features.pkl exists (it should, based on file listing)
        if not Path(cls.model_dir).exists():
            print(f"WARNING: Model dir {cls.model_dir} not found. Tests may fail.")

    @classmethod
    def tearDownClass(cls):
        # Cleanup
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)

    def setUp(self):
        # Reset Config for each test
        self.args = MagicMock()
        self.args.symbol = self.symbol
        self.args.model_dir = self.model_dir
        self.args.api_key = "test_key"
        self.args.api_secret = "test_secret"
        self.args.testnet = True
        self.args.timeframe = "5m"
        
        # Mock Paths inside LiveBot
        # We need to patch the Path creation in LiveBot or subclass it to redirect files
        pass

    def create_dummy_bars(self, start_ts, count, price=100.0):
        # Create realistic-looking bars with movement
        bars = []
        for i in range(count):
            ts = start_ts + (i * 300 * 1000) # 5m intervals
            # Add sine wave movement
            current_price = price + (np.sin(i / 10.0) * 5)
            
            row = {
                'timestamp': ts,
                'open': current_price, 
                'high': current_price + 1, 
                'low': current_price - 1, 
                'close': current_price + (np.random.random() - 0.5), # Small noise
                'volume': 1000.0,
                'taker_buy_ratio': 0.5,
                # OB Stats (Simulated)
                'ob_spread_mean': 0.01,
                'ob_micro_dev_mean': 0.001,
                'ob_micro_dev_std': 0.0001, # Added missing column
                'ob_imbalance_mean': 0.1,
                'ob_bid_depth_mean': 50000,
                'ob_ask_depth_mean': 50000,
                'ob_bid_slope_mean': 10,
                'ob_ask_slope_mean': 10,
                'ob_bid_integrity_mean': 0.8,
                'ob_ask_integrity_mean': 0.8
            }
            bars.append(row)
        return pd.DataFrame(bars)

    def test_atomic_write(self):
        print("\n--- Test: Atomic Write ---")
        # Initialize Bot (Mocked)
        with patch('live_trading.BybitAdapter') as MockRest, \
             patch('live_trading.BybitWSAdapter') as MockWS:
            
            bot = LiveBot(self.args)
            # Redirect history file
            bot.history_file = self.test_dir / "atomic_test.csv"
            
            # Create Data
            bot.bars = self.create_dummy_bars(1000000, 10)
            
            # Save
            bot._save_history_atomic()
            
            # Verify
            self.assertTrue(bot.history_file.exists())
            loaded = pd.read_csv(bot.history_file)
            self.assertEqual(len(loaded), 10)
            print("Atomic write successful.")

    def test_auto_backfill(self):
        print("\n--- Test: Auto-Backfill ---")
        with patch('live_trading.BybitAdapter') as MockRest, \
             patch('live_trading.BybitWSAdapter') as MockWS:
            
            bot = LiveBot(self.args)
            bot.history_file = self.test_dir / "backfill_test.csv"
            
            # 1. Create Old History (Gap of 1 hour = 12 bars)
            now_ms = int(time.time() * 1000)
            gap_duration = 3600 * 1000 # 1 hour
            last_ts = now_ms - gap_duration - (300 * 1000 * 10) # 10 bars before the gap
            
            old_bars = self.create_dummy_bars(last_ts, 10)
            old_bars.to_csv(bot.history_file, index=False)
            
            # Mock API Response
            missing_bars = []
            for i in range(12): # 12 bars to fill 1 hour
                ts = last_ts + (10 * 300 * 1000) + ((i+1) * 300 * 1000)
                missing_bars.append({
                    'startTime': str(ts), # API returns string usually
                    'open': '101', 'high': '102', 'low': '100', 'close': '101', 'volume': '500'
                })
            
            # Adapters usually format this before returning.
            # Assuming BybitAdapter.get_public_klines returns a list of DICTS or DataFrame.
            # LiveBot code: "self.bars = pd.DataFrame(klines)" -> implies list of dicts with cleaned keys
            # Let's verify BybitAdapter behavior if possible, but assuming standard dict structure for now.
            # BUT LiveBot expects keys: 'timestamp', 'open'...
            
            clean_missing_bars = []
            for m in missing_bars:
                clean_missing_bars.append({
                    'timestamp': int(m['startTime']),
                    'open': float(m['open']),
                    'high': float(m['high']),
                    'low': float(m['low']),
                    'close': float(m['close']),
                    'volume': float(m['volume'])
                })

            bot.rest_api.get_public_klines.return_value = clean_missing_bars
            
            # 3. Trigger Load (which triggers backfill)
            bot._load_history()
            
            # 4. Verify
            self.assertEqual(len(bot.bars), 22) # 10 old + 12 new
            print(f"Backfilled {len(bot.bars) - 10} bars successfully.")
            
            # Check if OB columns are 0 in the new bars
            last_bar = bot.bars.iloc[-1]
            if 'ob_spread_mean' in last_bar:
                self.assertEqual(last_bar['ob_spread_mean'], 0.0)
                print("OB columns correctly initialized to 0.0 for backfilled data.")

    def test_backup_rescue(self):
        print("\n--- Test: Backup Rescue ---")
        with patch('live_trading.BybitAdapter') as MockRest, \
             patch('live_trading.BybitWSAdapter') as MockWS:
            
            bot = LiveBot(self.args)
            bot.history_file = self.test_dir / "rescue_main.csv"
            
            # Use a filename that LiveBot will actually look for
            backup_filename = f"backup_history_{self.symbol}_{self.args.timeframe}.csv"
            # We create it in the CURRENT directory because LiveBot looks there
            backup_file = Path(backup_filename)
            
            # 1. Main File: Short/Cold (5 bars)
            now_ms = int(time.time() * 1000)
            start_ts = now_ms - (300 * 1000 * 5)
            main_bars = self.create_dummy_bars(start_ts, 5)
            main_bars.to_csv(bot.history_file, index=False)
            
            # 2. Backup File: Long/Warm (100 bars)
            # It must OVERLAP and EXTEND back
            start_ts_warm = now_ms - (300 * 1000 * 100)
            backup_bars = self.create_dummy_bars(start_ts_warm, 100)
            backup_bars.to_csv(backup_file, index=False)
            
            try:
                # 3. Trigger Load
                # Config Warmup is usually 60. 5 < 60 -> Rescue Triggered.
                bot._load_history()
                
                # 4. Verify
                self.assertGreater(len(bot.bars), 90)
                # self.assertTrue(bot.real_ob_bars_count >= 60) # This depends on _calculate_warmup_status implementation
                print(f"Rescue successful. Bars: {len(bot.bars)}")
                
            finally:
                # Cleanup backup file
                if backup_file.exists():
                    os.remove(backup_file)

    def test_feature_parity(self):
        print("\n--- Test: Feature Parity ---")
        # Load real features list
        import joblib
        features_path = Path(self.model_dir) / "features.pkl"
        if not features_path.exists():
            print("Skipping feature parity test (features.pkl not found)")
            return

        feature_cols = joblib.load(features_path)
        
        with patch('live_trading.BybitAdapter') as MockRest, \
             patch('live_trading.BybitWSAdapter') as MockWS:
            
            bot = LiveBot(self.args)
            # Create enough bars to calc features (e.g. 300 for EMAs)
            bot.bars = self.create_dummy_bars(1000000, 300)
            
            # Calculate Features
            df_feats = bot.feature_engine.calculate_features(bot.bars)
            
            # Verify columns
            missing = [c for c in feature_cols if c not in df_feats.columns]
            self.assertEqual(len(missing), 0, f"Missing features: {missing}")
            
            # Verify not all zeros (for standard columns)
            if 'rsi' in df_feats.columns:
                self.assertNotEqual(df_feats['rsi'].iloc[-1], 0)
                
            print(f"Feature parity confirmed. {len(feature_cols)} features calculated.")

    def test_csv_columns_integrity(self):
        print("\n--- Test: CSV Column Integrity ---")
        with patch('live_trading.BybitAdapter') as MockRest, \
             patch('live_trading.BybitWSAdapter') as MockWS:
            
            bot = LiveBot(self.args)
            bot.history_file = self.test_dir / "column_test.csv"
            bot.bars = self.create_dummy_bars(1000000, 10)
            
            # Calculate features to populate the DataFrame
            # (In real run, this happens before save)
            df_feats = bot.feature_engine.calculate_features(bot.bars)
            
            # Fix: Ensure columns exist before update (mimic LiveBot logic)
            for col in df_feats.columns:
                if col not in bot.bars.columns:
                    bot.bars[col] = 0.0
            
            bot.bars.update(df_feats)
            
            # Save
            bot._save_history_atomic()
            
            # Load and Check
            saved_df = pd.read_csv(bot.history_file)
            saved_cols = set(saved_df.columns)
            
            # 1. Check Standard OHLCV
            required_ohlcv = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
            self.assertTrue(required_ohlcv.issubset(saved_cols), f"Missing OHLCV: {required_ohlcv - saved_cols}")
            
            # 2. Check Raw OB Stats (from dummy generator)
            required_ob = {'ob_spread_mean', 'ob_imbalance_mean'}
            self.assertTrue(required_ob.issubset(saved_cols), f"Missing OB Stats: {required_ob - saved_cols}")
            
            # 3. Check Calculated Features
            # We assume features.pkl is correct (validated in previous test)
            import joblib
            features_path = Path(self.model_dir) / "features.pkl"
            if features_path.exists():
                feature_cols = set(joblib.load(features_path))
                missing_feats = feature_cols - saved_cols
                self.assertEqual(len(missing_feats), 0, f"Saved CSV missing features: {missing_feats}")
                
            print(f"CSV Integrity Check Passed: {len(saved_cols)} columns verified.")

    def test_watchdog_logic(self):
        print("\n--- Test: Watchdog Logic ---")
        # Dynamic import of watchdog because it's a script not a module usually
        import importlib.util
        spec = importlib.util.spec_from_file_location("watchdog", "watchdog.py")
        wd_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wd_module)
        ServiceManager = wd_module.ServiceManager
        
        # Mocking
        mock_webhook = "http://fake.url"
        log_file = self.test_dir / "test_service.log"
        
        # 1. Test Initialization
        sm = ServiceManager("TEST_SVC", "test_script.py", str(log_file), ["--arg"], mock_webhook)
        self.assertTrue(log_file.exists())
        
        # 2. Test Start (New Process)
        with patch('subprocess.Popen') as mock_popen:
            mock_proc = MagicMock()
            mock_proc.pid = 12345
            mock_popen.return_value = mock_proc
            
            # Mock find_existing to return None so it starts new
            with patch.object(sm, 'find_existing_process', return_value=None):
                sm.start()
                
            self.assertEqual(sm.process_pid, 12345)
            self.assertTrue(mock_popen.called)
            print("Watchdog: Start logic ok.")

        # 3. Test Log Reading (Error Detection)
        with open(log_file, 'w') as f:
            f.write("Some normal log line\n")
            f.flush()
            
        sm.open_log_reader()
        # Move cursor to end
        sm.file_handle.seek(0, 2)
        
        # Simulate writing an error
        with open(log_file, 'a') as f:
            f.write("CRITICAL: !!! GAP DETECTED !!!\n")
            f.flush()
            
        # Simulate Tick
        # We need to mock is_running to return True so it doesn't try to restart immediately
        with patch.object(sm, 'is_running', return_value=True), \
             patch.object(sm, 'stop') as mock_stop:
             
            sm.tick(network_ok=True, disk_ok=True)
            
            # Expect stop to be called because of the error pattern
            mock_stop.assert_called()
            args, _ = mock_stop.call_args
            self.assertIn("GAP", args[0] if args else "") # Check reason contains "GAP" (case sensitive)
            print("Watchdog: Error detection logic ok.")
            
        # Cleanup file handle
        if sm.file_handle:
            sm.file_handle.close()

if __name__ == '__main__':
    unittest.main()
