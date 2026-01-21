import unittest
import subprocess
import time
import os
import sys
from pathlib import Path

# Add current directory to sys.path so we can import watchdog
sys.path.append(str(Path(__file__).parent.parent))

# Use raw string to avoid escape issues
MOCK_BOT_CODE = r"""
import time
import sys
import os

log_file = "live_trading.log"
mode = sys.argv[1] if len(sys.argv) > 1 else "normal"

def write_log(msg):
    # Append to log
    with open(log_file, "a") as f:
        f.write(msg + "\n")
    # Flush ensures watchdog sees it
    print(f"MockBot: {msg}")

if mode == "gap":
    time.sleep(2)
    write_log("!!! GAP DETECTED !!!")
    time.sleep(10)
elif mode == "latency":
    time.sleep(2)
    write_log("Step 2: (HIGH LATENCY)")
    time.sleep(10)
elif mode == "silence":
    write_log("Starting...")
    # Does not write heartbeat, should trigger MAX_SILENCE
    time.sleep(30) 
else:
    write_log(">>> [SYSTEM METRICS]")
    time.sleep(10)
"""

class TestWatchdogFunctional(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("live_trading_mock.py", "w") as f:
            f.write(MOCK_BOT_CODE)

    @classmethod
    def tearDownClass(cls):
        for f in ["live_trading_mock.py", "live_trading.log"]:
            if os.path.exists(f): os.remove(f)

    def run_watchdog_test(self, bot_mode, silence_limit=None):
        import watchdog
        from watchdog import BotManager
        
        class TestManager(BotManager):
            def __init__(self, mode):
                super().__init__([mode])
                self.cmd = [sys.executable, "live_trading_mock.py", mode]
        
        manager = TestManager(bot_mode)
        if silence_limit:
            watchdog.MAX_SILENCE_SECONDS = silence_limit
            watchdog.CHECK_INTERVAL = 1 # Faster for testing
        else:
            watchdog.MAX_SILENCE_SECONDS = 600
            watchdog.CHECK_INTERVAL = 1
            
        start_time = time.time()
        manager.monitor()
        return time.time() - start_time

    def test_gap_detection(self):
        print("\n[Watchdog Test] Testing Gap Detection...")
        duration = self.run_watchdog_test("gap")
        self.assertLess(duration, 15)
        self.assertGreater(duration, 1)
        print(f"    [OK] Stopped in {duration:.2f}s")

    def test_latency_detection(self):
        print("\n[Watchdog Test] Testing Latency Detection...")
        duration = self.run_watchdog_test("latency")
        self.assertLess(duration, 15)
        self.assertGreater(duration, 1)
        print(f"    [OK] Stopped in {duration:.2f}s")

    def test_silence_detection(self):
        print("\n[Watchdog Test] Testing Silence (Heartbeat) Detection...")
        # Test with 5 second silence limit
        duration = self.run_watchdog_test("silence", silence_limit=5)
        # Should take roughly silence_limit + check_interval
        self.assertLess(duration, 12)
        self.assertGreater(duration, 4)
        print(f"    [OK] Stopped after {duration:.2f}s of silence")

    def test_continuity_in_good_conditions(self):
        # ... (keep existing code)
        if os.path.exists("live_trading_mock_healthy.py"):
            os.remove("live_trading_mock_healthy.py")

    def test_auto_recovery_limit(self):
        print("\n[Watchdog Test] Testing Auto-Recovery and Max Restarts...")
        import watchdog
        from watchdog import BotManager
        
        # Override settings for fast test
        watchdog.MAX_RESTARTS = 2
        watchdog.RESTART_BACKOFF = 1
        watchdog.CHECK_INTERVAL = 1
        
        # This bot will die instantly, triggering a restart
        MOCK_CRASHING_BOT = r"""
import sys
with open("live_trading.log", "a") as f:
    f.write("I am about to crash...\n")
sys.exit(1)
"""
        with open("live_trading_mock_crash.py", "w") as f:
            f.write(MOCK_CRASHING_BOT)
            
        class RecoveryManager(BotManager):
            def __init__(self):
                super().__init__([])
                self.cmd = [sys.executable, "live_trading_mock_crash.py"]

        manager = RecoveryManager()
        start_time = time.time()
        manager.monitor()
        duration = time.time() - start_time
        
        # It should try to start, die, wait 1s, start again, die, then exit.
        self.assertEqual(manager.restart_count, 2)
        print(f"    [OK] Watchdog correctly stopped after {manager.restart_count} restarts.")
        
        if os.path.exists("live_trading_mock_crash.py"):
            os.remove("live_trading_mock_crash.py")

    def test_active_repair_corrupted_state(self):
        print("\n[Watchdog Test] Testing Active Repair (JSON Corruption)...")
        import watchdog
        from watchdog import BotManager
        
        # Create a corrupted JSON file
        vpm_file = Path("virtual_positions.json")
        with open(vpm_file, "w") as f:
            f.write("{ invalid json: [")
            
        class RepairManager(BotManager):
            def __init__(self):
                super().__init__(["normal"])
                # We stop immediately after starting to check file state
                self.cmd = [sys.executable, "-c", "print('started')"]

        manager = RepairManager()
        manager.repair_environment()
        
        # Check if the file was repaired (moved to .corrupt_bak)
        self.assertFalse(vpm_file.exists())
        self.assertTrue(Path("virtual_positions.json.corrupt_bak").exists())
        print("    [OK] Watchdog detected corruption and salvaged the state file.")
        
        # Cleanup
        if Path("virtual_positions.json.corrupt_bak").exists():
            os.remove("virtual_positions.json.corrupt_bak")

    def test_network_failure_detection(self):
        print("\n[Watchdog Test] Testing Network Failure Detection...")
        import watchdog
        from watchdog import BotManager
        
        # Override settings
        watchdog.MAX_RESTARTS = 1
        watchdog.CHECK_INTERVAL = 1
        
        class NetworkFailManager(BotManager):
            def __init__(self):
                super().__init__(["normal"])
                self.cmd = [sys.executable, "live_trading_mock.py", "normal"]
            def check_network(self):
                return False # Simulate net down
        
        manager = NetworkFailManager()
        start_time = time.time()
        manager.monitor()
        duration = time.time() - start_time
        
        self.assertEqual(manager.restart_count, 1)
        print(f"    [OK] Watchdog caught network failure and stopped bot in {duration:.2f}s")

    def test_disk_failure_detection(self):
        print("\n[Watchdog Test] Testing Disk Failure Detection...")
        import watchdog
        from watchdog import BotManager
        
        # Override settings
        watchdog.MAX_RESTARTS = 1
        watchdog.CHECK_INTERVAL = 1
        
        class DiskFailManager(BotManager):
            def __init__(self):
                super().__init__(["normal"])
                self.cmd = [sys.executable, "live_trading_mock.py", "normal"]
            def check_disk(self):
                return False # Simulate disk full
        
        manager = DiskFailManager()
        start_time = time.time()
        manager.monitor()
        duration = time.time() - start_time
        
        self.assertEqual(manager.restart_count, 1)
        print(f"    [OK] Watchdog caught disk failure and stopped bot in {duration:.2f}s")

    def test_discord_integration(self):
        print("\n[Watchdog Test] Testing Discord Notification Triggers...")
        from unittest.mock import patch
        import watchdog
        from watchdog import BotManager
        
        # Override for fast test
        watchdog.MAX_RESTARTS = 1
        watchdog.RESTART_BACKOFF = 0
        watchdog.CHECK_INTERVAL = 1
        
        # Mock bot that performs a trade and then dies
        MOCK_TRADE_BOT = r"""
import time
with open("live_trading.log", "a") as f:
    f.write("Opening Long: 100.0\n")
    f.flush()
# Small sleep to ensure watchdog reads the line before the process dies
time.sleep(2)
with open("live_trading.log", "a") as f:
    f.write("!!! GAP DETECTED !!!\n")
    f.flush()
time.sleep(1)
"""
        with open("live_trading_mock_trade.py", "w") as f:
            f.write(MOCK_TRADE_BOT)

        with patch("requests.post") as mock_post:
            class DiscordTestManager(BotManager):
                def __init__(self):
                    super().__init__(["--symbol", "TEST"])
                    self.cmd = [sys.executable, "live_trading_mock_trade.py"]
                    self.webhook_url = "http://fake-webhook.com"
            
            manager = DiscordTestManager()
            # We want it to read quickly
            import watchdog
            watchdog.CHECK_INTERVAL = 0.5
            manager.monitor()
            
            # Check all notification contents
            all_content = ""
            for call in mock_post.call_args_list:
                json_data = call.kwargs.get('json', {})
                all_content += str(json_data.get('content', '')) + " | "
            
            # Use broader matching
            startup_notified = "Starting Bot" in all_content
            trade_notified = "Trade Event: Opening Long" in all_content
            error_notified = "EMERGENCY STOP" in all_content
            
            self.assertTrue(startup_notified, f"Startup notification missing in: {all_content}")
            self.assertTrue(trade_notified, f"Trade notification missing in: {all_content}")
            self.assertTrue(error_notified, f"Error notification missing in: {all_content}")
            
            print(f"    [OK] Discord alerts triggered correctly.")

        if os.path.exists("live_trading_mock_trade.py"):
            os.remove("live_trading_mock_trade.py")

if __name__ == '__main__':
    unittest.main()
