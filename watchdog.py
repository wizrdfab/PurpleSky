import subprocess
import time
import os
import sys
import signal
import logging
import socket
import shutil
from pathlib import Path

import json
import requests

# --- Configuration ---
LOG_FILE = "live_trading.log"
CHECK_INTERVAL = 5  
MAX_SILENCE_SECONDS = 600 
MIN_DISK_MB = 500         
NET_CHECK_HOST = "8.8.8.8" 

MAX_RESTARTS = 5          # Safety limit
RESTART_BACKOFF = 30      # Seconds to wait before restart

CRITICAL_PATTERNS = [
    "!!! GAP DETECTED !!!",
    "!!! STALE/EMPTY SPREAD !!!",
    "!!! LOW LIQUIDITY/FROZEN",
    "(HIGH LATENCY)",
    "Traceback (most recent call last):",
    "ConnectionError",
    "Rate limit hit",
    "Insufficient margin",
    "Account blocked",
    "[ERROR]"
]

TRADE_PATTERNS = [
    "Opening Long:",
    "Opening Short:",
    "EXECUTION:",
    "Virtual Trade"
]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [WATCHDOG] %(message)s'
)
logger = logging.getLogger("Watchdog")

class BotManager:
    def __init__(self, cmd_args):
        self.cmd = [sys.executable, "live_trading.py"] + cmd_args
        self.process = None
        self.last_health_report = time.time()
        self.log_path = Path(LOG_FILE)
        self.restart_count = 0
        self.webhook_url = self.load_webhook_url()
        
        self.ensure_log_exists()

    def load_webhook_url(self):
        url = os.getenv("DISCORD_WEBHOOK_URL")
        if not url:
            try:
                with open("keys.json", "r") as f:
                    keys = json.load(f)
                    url = keys.get("discord_webhook_url")
            except: pass
        return url

    def notify_discord(self, message, is_error=False):
        if not self.webhook_url:
            return
        
        prefix = "🚨 **[CRITICAL]**" if is_error else "📈 **[INFO]**"
        payload = {
            "content": f"{prefix} {message}"
        }
        try:
            requests.post(self.webhook_url, json=payload, timeout=5)
        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")

    def ensure_log_exists(self):
        if not self.log_path.exists():
            with open(self.log_path, 'w') as f: f.write("")
        else:
            with open(self.log_path, 'w') as f: f.write("")

    def check_network(self):
        # 1. General Internet Check
        try:
            socket.setdefaulttimeout(3)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((NET_CHECK_HOST, 53))
        except:
            return False
            
        # 2. Exchange Reachability Check
        try:
            # Check if Bybit API is reachable
            socket.gethostbyname("api.bybit.com")
            return True
        except:
            logger.warning("Internet is UP, but api.bybit.com is UNREACHABLE.")
            return False

    def check_disk(self):
        total, used, free = shutil.disk_usage(".")
        return (free // (2**20)) > MIN_DISK_MB

    def start_bot(self):
        self.repair_environment() # --- ACTIVE REPAIR ---
        self.ensure_log_exists()
        logger.info(f"Starting Bot (Attempt {self.restart_count + 1}/{MAX_RESTARTS}): {' '.join(self.cmd)}")
        self.notify_discord(f"Starting Bot (Attempt {self.restart_count + 1}/{MAX_RESTARTS})")
        self.process = subprocess.Popen(self.cmd)
        self.last_health_report = time.time()

    def repair_environment(self):
        """Actively fix common issues before starting the bot."""
        logger.info("Running Active Repair Engine...")
        
        # 1. Zombie Cleanup (Force kill any leaked bot processes)
        if sys.platform == "win32":
            # Find and kill processes running live_trading.py
            subprocess.run(["taskkill", "/F", "/IM", "python.exe", "/FI", "WINDOWTITLE eq live_trading.py*"], 
                           capture_output=True, shell=True)
        
        # 2. Disk Salvage (If low, truncate the log file)
        if not self.check_disk():
            logger.warning("Disk space low. Truncating log file to salvage space...")
            if self.log_path.exists():
                with open(self.log_path, 'w') as f: f.write("--- DISK SALVAGE PERFORMED ---\n")

        # 3. State Integrity (Check for corrupted JSON)
        vpm_file = Path("virtual_positions.json")
        if vpm_file.exists():
            try:
                import json
                with open(vpm_file, 'r') as f:
                    json.load(f)
            except Exception as e:
                logger.error(f"CORRUPTION DETECTED in {vpm_file}: {e}. Repairing...")
                bak_file = vpm_file.with_suffix(".json.corrupt_bak")
                vpm_file.replace(bak_file)
                logger.info(f"Corrupted state moved to {bak_file}. Bot will reconcile from exchange.")

    def stop_bot(self, reason):
        if self.process:
            logger.error(f"EMERGENCY STOP: {reason}")
            self.notify_discord(f"EMERGENCY STOP: {reason}", is_error=True)
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            logger.info("Bot process terminated.")

    def wait_for_recovery(self):
        """Wait until environment is healthy again."""
        logger.info("Entering Recovery Mode. Checking environment...")
        self.notify_discord("⚠️ Bot entered Recovery Mode. Waiting for healthy environment...")
        while True:
            net_ok = self.check_network()
            disk_ok = self.check_disk()
            
            if net_ok and disk_ok:
                logger.info("Environment is HEALTHY. Preparing to restart...")
                self.notify_discord("✅ Environment recovered. Restarting bot soon.")
                time.sleep(RESTART_BACKOFF)
                return True
            
            logger.warning(f"Recovery pending: Network={'OK' if net_ok else 'DOWN'}, Disk={'OK' if disk_ok else 'LOW'}")
            time.sleep(30)

    def monitor(self):
        while self.restart_count < MAX_RESTARTS:
            self.start_bot()
            
            # Inner monitor loop
            stop_reason = None
            try:
                with open(self.log_path, 'r') as f:
                    f.seek(0, os.SEEK_END)
                    
                    while True:
                        if self.process.poll() is not None:
                            stop_reason = "Bot process died unexpectedly."
                            break

                        if not self.check_network():
                            stop_reason = "Network Connectivity Lost"
                            break

                        if not self.check_disk():
                            stop_reason = "Disk Space Critical"
                            break

                        line = f.readline()
                        if line:
                            if ">>> [SYSTEM METRICS]" in line:
                                self.last_health_report = time.time()
                            
                            for pattern in CRITICAL_PATTERNS:
                                if pattern in line:
                                    stop_reason = f"Anomaly in logs: {line.strip()}"
                                    break
                            
                            # 2. Trade Notifications
                            for pattern in TRADE_PATTERNS:
                                if pattern in line:
                                    self.notify_discord(f"Trade Event: {line.strip()}")
                            
                            if stop_reason: break
                        else:
                            silence = time.time() - self.last_health_report
                            if silence > MAX_SILENCE_SECONDS:
                                stop_reason = f"Bot frozen (No heartbeat for {silence:.0f}s)"
                                break
                            time.sleep(CHECK_INTERVAL)
                            
            except Exception as e:
                stop_reason = f"Watchdog Error: {e}"

            # If we reached here, we need to stop and potentially restart
            self.stop_bot(stop_reason)
            self.restart_count += 1
            
            if self.restart_count < MAX_RESTARTS:
                self.wait_for_recovery()
            else:
                logger.critical("MAX RESTARTS REACHED. Watchdog exiting. Manual intervention required.")
                self.notify_discord("💀 **MAX RESTARTS REACHED.** Watchdog exiting. Manual intervention required!", is_error=True)
                break

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python watchdog.py [live_trading_args...]")
        sys.exit(1)
    manager = BotManager(sys.argv[1:])
    manager.monitor()
