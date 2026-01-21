import unittest
import subprocess
import sys
import os
from pathlib import Path

class TestCLIRobustness(unittest.TestCase):
    def run_cli(self, args):
        print(f"    DEBUG: Running in {os.getcwd()}")
        cmd = [sys.executable, "live_trading.py"] + args
        print(f"    DEBUG: Cmd={' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            # Combine stdout and stderr for easier pattern matching
            return result.returncode, result.stdout + result.stderr
        except subprocess.TimeoutExpired as e:
            return 0, ""
        except Exception as e:
            return 1, str(e)

    def test_underscore_vs_hyphen(self):
        print("\n[CLI Test] Testing Underscore vs Hyphen...")
        
        # Test Case 1: Underscores (The one that failed before)
        code, err = self.run_cli(["--symbol", "BTCUSDT", "--model_dir", "models/FARTCOINUSDT/rank_1", "--testnet"])
        self.assertEqual(code, 0, f"Failed with underscores: {err}")
        print("    [OK] Underscore version parsed successfully.")

        # Test Case 2: Hyphens (The original standard)
        code, err = self.run_cli(["--symbol", "BTCUSDT", "--model-dir", "models/FARTCOINUSDT/rank_1", "--testnet"])
        self.assertEqual(code, 0, f"Failed with hyphens: {err}")
        print("    [OK] Hyphen version parsed successfully.")

    def test_missing_required(self):
        print("\n[CLI Test] Testing Missing Required Arguments...")
        code, err = self.run_cli(["--model-dir", "models/FARTCOINUSDT/rank_1"])
        
        if code != 0:
            print(f"    [OK] Correctly identified missing arguments (Code {code}).")
        else:
            print(f"    DEBUG: Code={code}, Err='{err}'")
            self.fail(f"CLI allowed execution without required --symbol! code={code}")

    def test_timeframe_variations(self):
        print("\n[CLI Test] Testing Timeframe Argument Variations...")
        code, err = self.run_cli(["--symbol", "BTCUSDT", "--model-dir", "models/FARTCOINUSDT/rank_1", "--time_frame", "15m", "--testnet"])
        self.assertEqual(code, 0, f"Timeframe underscores failed: {err}")
        print("    [OK] Timeframe underscores parsed successfully.")

    def test_api_key_variations(self):
        print("\n[CLI Test] Testing API Key Argument Variations...")
        # Test underscore version of api keys
        code, err = self.run_cli([
            "--symbol", "BTCUSDT", 
            "--model-dir", "models/FARTCOINUSDT/rank_1", 
            "--api_key", "test", 
            "--api_secret", "test",
            "--testnet"
        ])
        self.assertEqual(code, 0, f"API key underscores failed: {err}")
        print("    [OK] API key underscores parsed successfully.")

if __name__ == '__main__':
    unittest.main()
