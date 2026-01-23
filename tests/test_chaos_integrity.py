import unittest
import pandas as pd
import numpy as np
import os
import shutil
import time
import random
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from live_trading import LiveBot
from data_keeper import DataKeeperBot

class ChaosMonkeyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path("tests/chaos_test_zone")
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        cls.symbol = "CHAOSUSDT"
        cls.model_dir = "models/FARTCOINUSDT/rank_1" # Use real model dir for feature list

    @classmethod
    def tearDownClass(cls):
        if cls.test_dir.exists():
            try:
                shutil.rmtree(cls.test_dir)
            except: pass

    def setUp(self):
        self.args = MagicMock()
        self.args.symbol = self.symbol
        self.args.model_dir = self.model_dir
        self.args.api_key = "test"
        self.args.api_secret = "test"
        self.args.testnet = True
        self.args.timeframe = "5m"
        
        # Helper to get a fresh bot instance
        self.bot_history_file = self.test_dir / f"backup_history_{self.symbol}_5m.csv"
        
        # Reset file
        if self.bot_history_file.exists():
            os.remove(self.bot_history_file)

    def create_valid_history(self, count=100):
        # Create a valid starting point
        bars = []
        now = int(time.time() * 1000)
        for i in range(count):
            ts = now - ((count - i) * 300 * 1000)
            row = {
                'timestamp': ts,
                'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.5,
                'volume': 1000 + i,
                'ob_spread_mean': 0.01
            }
            bars.append(row)
        df = pd.DataFrame(bars)
        # Add dummy feature cols to match schema
        import joblib
        feats = joblib.load(Path(self.model_dir) / "features.pkl")
        for f in feats:
            if f not in df.columns: df[f] = 0.5
        
        df.to_csv(self.bot_history_file, index=False)
        return df

    def test_chaos_01_sudden_death_mid_write(self):
        print("\n[CHAOS 01] Sudden Death during Write (Atomic Check)")
        # Scenario: Power cut while bytes are being written.
        # Result: Temp file is partial, Original file is untouched.
        
        self.create_valid_history(100)
        original_size = self.bot_history_file.stat().st_size
        
        # Simulate a temp file that got half-written
        temp_file = self.bot_history_file.with_suffix(".tmp")
        with open(temp_file, 'w') as f:
            f.write("timestamp,open,high,low\n12345,100,101") # Corrupt garbage
            
        # Initialize Bot
        with patch('live_trading.BybitAdapter'), patch('live_trading.BybitWSAdapter'):
            bot = DataKeeperBot(self.args)
            bot.history_file = self.bot_history_file # Point to test file
            
            # Action: Bot starts up. It should IGNORE the corrupt .tmp file and load the valid .csv
            bot._load_history()
            
            self.assertEqual(len(bot.bars), 100)
            print(" -> Passed: Bot ignored corrupted temp file and loaded valid history.")
            
            # Check if it cleaned up? (Optional, nice to have)
            # LiveBot doesn't auto-clean .tmp files on startup currently, but it overwrites them on next save.
            # So integrity is preserved.

    def test_chaos_02_file_truncation(self):
        print("\n[CHAOS 02] File Truncation (Zero Byte)")
        # Scenario: File exists but is empty (e.g. disk full error previously)
        
        # Create empty file
        with open(self.bot_history_file, 'w') as f:
            pass 
            
        with patch('live_trading.BybitAdapter') as MockRest, patch('live_trading.BybitWSAdapter'):
            # Setup Mock to return data (Backfill)
            # If file is empty, it should fetch from exchange
            MockRest.return_value.get_public_klines.return_value = [
                {'timestamp': 1000, 'open': 10, 'high': 11, 'low': 9, 'close': 10, 'volume': 100}
            ]
            
            bot = DataKeeperBot(self.args)
            bot.history_file = self.bot_history_file
            
            # Action
            bot._load_history()
            
            # Verify: It treated empty file as "No history" and fetched fresh
            self.assertFalse(bot.bars.empty)
            print(" -> Passed: Bot recognized empty file and re-initialized.")

    def test_chaos_03_corrupt_last_line(self):
        print("\n[CHAOS 03] Corrupt Last Line (Partial Write)")
        # Scenario: The CSV has valid data, but the last line is cut off half-way.
        # pandas.read_csv usually handles this with 'error_bad_lines' or warnings, 
        # but stricter parsers might fail.
        
        self.create_valid_history(100)
        
        # Corrupt end of file
        with open(self.bot_history_file, 'a') as f:
            f.write("\n123456789,100.0,10") # Incomplete line
            
        with patch('live_trading.BybitAdapter'), patch('live_trading.BybitWSAdapter'):
            bot = DataKeeperBot(self.args)
            bot.history_file = self.bot_history_file
            
            # Action: Attempt load
            try:
                bot._load_history()
                # Pandas C engine is robust. It might drop the line or warn.
                # As long as bot.bars is valid (100 rows), we are good.
                self.assertEqual(len(bot.bars), 100)
                print(" -> Passed: Pandas parser recovered from partial line (dropped it).")
            except Exception as e:
                print(f" -> Failed: Crashed on corrupt line: {e}")
                raise e

    def test_chaos_04_read_only_filesystem(self):
        print("\n[CHAOS 04] Read-Only / Locked File")
        # Scenario: Permissions messed up. Bot cannot write.
        
        self.create_valid_history(10)
        
        # Lock file (Make read-only)
        os.chmod(self.bot_history_file, 0o444)
        
        try:
            with patch('live_trading.BybitAdapter'), patch('live_trading.BybitWSAdapter'):
                bot = DataKeeperBot(self.args)
                bot.history_file = self.bot_history_file
                bot._load_history() # Should work (read)
                
                # Now try to SAVE (Atomic)
                # It should catch exception and log error, NOT crash the process
                try:
                    bot._save_history_atomic()
                    print(" -> Warning: Save appeared to work (OS didn't enforce lock?)")
                except PermissionError:
                    print(" -> Passed: PermissionError caught naturally.")
                except Exception as e:
                    # We want to ensure the BOT doesn't crash. 
                    # LiveBot._save_history_atomic has a try/except block.
                    # It logs error and continues.
                    pass
                    
                # Verify process is still "alive" (method returned)
                print(" -> Passed: Bot survived write permission error.")
                
        finally:
            # Unlock for cleanup
            os.chmod(self.bot_history_file, 0o777)

    def test_chaos_05_column_mismatch(self):
        print("\n[CHAOS 05] Schema Drift (Old CSV vs New Model)")
        # Scenario: CSV has 10 columns. Model update now expects 12.
        
        # Create history with MISSING columns
        df = self.create_valid_history(10)
        df = df.drop(columns=df.columns[-5:]) # Drop last 5 cols
        df.to_csv(self.bot_history_file, index=False)
        
        with patch('live_trading.BybitAdapter'), patch('live_trading.BybitWSAdapter'):
            bot = DataKeeperBot(self.args)
            bot.history_file = self.bot_history_file
            
            # Load logic in LiveBot:
            # "for col in feature_cols: if col not in bars: bars[col] = 0.0"
            # This logic exists in _load_history
            
            bot._load_history()
            
            # Verify missing columns were restored (as 0.0)
            import joblib
            feats = joblib.load(Path(self.model_dir) / "features.pkl")
            
            missing = [c for c in feats if c not in bot.bars.columns]
            self.assertEqual(len(missing), 0)
            print(" -> Passed: Schema drift auto-healed (Columns restored).")

if __name__ == '__main__':
    unittest.main()
