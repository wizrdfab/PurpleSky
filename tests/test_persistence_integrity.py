
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

class TestPersistenceIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Setup temporary test directory
        cls.test_dir = Path("tests/temp_persistence_test")
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        cls.symbol = "TESTPERSIST"
        cls.tf = "5m"

    @classmethod
    def tearDownClass(cls):
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)

    def setUp(self):
        self.args = MagicMock()
        self.args.symbol = self.symbol
        self.args.model_dir = "models/FARTCOINUSDT/rank_1" # Mock dir
        self.args.api_key = "test_key"
        self.args.api_secret = "test_secret"
        self.args.testnet = True
        self.args.timeframe = self.tf
        
        # Mock Feature Engine to avoid needing real model dir
        self.patcher = patch('live_trading.FeatureEngine')
        self.MockEngine = self.patcher.start()
        
    def tearDown(self):
        self.patcher.stop()
        # Clean specific files
        for f in self.test_dir.glob("*"):
            try: os.remove(f)
            except: pass

    def create_dummy_bars(self, start_ts, count):
        bars = []
        for i in range(count):
            ts = start_ts + (i * 300 * 1000)
            row = {
                'timestamp': ts,
                'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0,
                'taker_buy_ratio': 0.5,
                'ob_spread_mean': 0.1, # Valid OB data
                'ob_imbalance_mean': 0.0
            }
            bars.append(row)
        return pd.DataFrame(bars)

    def test_backup_rejection_insufficient(self):
        """Test: Backup file has data, but not enough to warmup. Should SKIP merge."""
        with patch('live_trading.BybitAdapter'), patch('live_trading.BybitWSAdapter'):
            bot = LiveBot(self.args)
            bot.history_file = self.test_dir / "main.csv"
            bot.warmup_bars = 60
            
            # 1. Main File: Empty
            
            # 2. Backup File: Short (10 bars)
            backup_file = Path(f"backup_history_{self.symbol}_{self.tf}.csv")
            now_ms = int(time.time() * 1000)
            backup_bars = self.create_dummy_bars(now_ms - (300*1000*10), 10)
            backup_bars.to_csv(backup_file, index=False)
            
            try:
                bot._load_history()
                
                # Should remain empty or match main (empty)
                # Because backup (10) < warmup (60), it should be rejected.
                self.assertEqual(len(bot.bars), 0, "Backup should have been rejected (insufficient)")
                
            finally:
                if backup_file.exists(): os.remove(backup_file)

    def test_backup_rejection_stale(self):
        """Test: Backup file has enough bars, but is too old. Should SKIP merge."""
        with patch('live_trading.BybitAdapter'), patch('live_trading.BybitWSAdapter'):
            bot = LiveBot(self.args)
            bot.history_file = self.test_dir / "main.csv"
            bot.warmup_bars = 60
            
            # 1. Backup File: Long (100 bars) BUT Old (2 hours ago)
            backup_file = Path(f"backup_history_{self.symbol}_{self.tf}.csv")
            
            # 2 hours ago = 120 mins = 24 * 5m bars.
            # Max lag allowed in code is 2 * TF = 10 mins.
            
            old_ts = int(time.time() * 1000) - (3600 * 1000 * 2) 
            backup_bars = self.create_dummy_bars(old_ts, 100)
            backup_bars.to_csv(backup_file, index=False)
            
            try:
                bot._load_history()
                
                # Backup is 100 bars (enough), but last bar is 2 hours old (stale).
                # _calculate_warmup_status checks recency.
                # It returns 0 if stale.
                # So merge logic sees 0 < 60 -> Reject.
                
                self.assertEqual(len(bot.bars), 0, "Backup should have been rejected (stale)")
                
            finally:
                if backup_file.exists(): os.remove(backup_file)

    def test_backup_acceptance_healthy(self):
        """Test: Backup file is valid, fresh, and long enough. Should MERGE."""
        with patch('live_trading.BybitAdapter'), patch('live_trading.BybitWSAdapter'):
            bot = LiveBot(self.args)
            bot.history_file = self.test_dir / "main.csv"
            bot.warmup_bars = 60
            
            # 1. Main: Empty/Cold
            
            # 2. Backup: Fresh & Long (70 bars)
            backup_file = Path(f"backup_history_{self.symbol}_{self.tf}.csv")
            now_ms = int(time.time() * 1000)
            start_ts = now_ms - (300 * 1000 * 70)
            backup_bars = self.create_dummy_bars(start_ts, 70)
            backup_bars.to_csv(backup_file, index=False)
            
            try:
                bot._load_history()
                
                # Should merge
                self.assertEqual(len(bot.bars), 70, "Backup should have been merged")
                self.assertEqual(bot.real_ob_bars_count, 70)
                
            finally:
                if backup_file.exists(): os.remove(backup_file)

    def test_deduplication_logic(self):
        """Test: Merge overlaps correctly without duplicates."""
        with patch('live_trading.BybitAdapter'), patch('live_trading.BybitWSAdapter'):
            bot = LiveBot(self.args)
            bot.history_file = self.test_dir / "main_dedup.csv"
            bot.warmup_bars = 10 # Lower threshold for test convenience
            
            # 1. Main: Bars 0-20
            now_ms = int(time.time() * 1000)
            start_ts = now_ms - (300 * 1000 * 50)
            
            main_bars = self.create_dummy_bars(start_ts, 20) # 0 to 19
            # Force main to appear "cold" (e.g. invalid OB data in last few bars)
            # so that it triggers rescue logic.
            # We explicitly set a gap in continuity to fail the internal warmup check
            # main_bars.loc[15:, 'ob_spread_mean'] = 0 # Invalid tail
            # Actually, live_trading checks continuity. Let's create a GAP in timestamps inside main.
            # Or just set count=0 by making it very old.
            # But we want to test overlap logic.
            
            # Strategy: Make Main EMPTY so it tries to load backup. 
            # WAIT. The code merges `self.bars` (loaded from disk) + `backup_bars`.
            # So we save `main_bars` to disk.
            
            # To trigger rescue, `real_ob_bars_count` must be < `warmup_bars`.
            # We set `warmup_bars` = 100.
            # Main has 20 bars. 20 < 100. Trigger!
            bot.warmup_bars = 100
            
            main_bars.to_csv(bot.history_file, index=False)
            
            # 2. Backup: Bars 10-40 (Overlap 10-19, New 20-40)
            # Backup is perfectly valid
            backup_file = Path(f"backup_history_{self.symbol}_{self.tf}.csv")
            
            # Start of backup corresponds to index 10 of main
            backup_start = start_ts + (10 * 300 * 1000)
            backup_bars = self.create_dummy_bars(backup_start, 110) # 110 bars to satisfy > 100 req
            backup_bars.to_csv(backup_file, index=False)
            
            try:
                bot._load_history()
                
                # Check Deduplication
                # We expect 0-19 from Main, 20-119 from Backup?
                # Total bars should be 130 (0-129)
                # Overlap: 10-19 exists in both.
                
                # Assert Unique Timestamps
                timestamps = bot.bars['timestamp']
                self.assertTrue(timestamps.is_unique, "Timestamps should be unique after merge")
                self.assertTrue(timestamps.is_monotonic_increasing, "Timestamps should be sorted")
                
            finally:
                if backup_file.exists(): os.remove(backup_file)

if __name__ == '__main__':
    unittest.main()
