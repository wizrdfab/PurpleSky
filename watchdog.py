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
CHECK_INTERVAL = 5
MAX_SILENCE_SECONDS = 600
MIN_DISK_MB = 500
NET_CHECK_HOST = "8.8.8.8"
MAX_RESTARTS = 100
RESTART_BACKOFF = 30

# Common Patterns
CRITICAL_PATTERNS = [
    "!!! GAP DETECTED !!!",
    "!!! STALE/EMPTY SPREAD !!!",
    "!!! LOW LIQUIDITY/FROZEN",
    "(HIGH LATENCY)",
    "Traceback (most recent call last):",
    "ConnectionError",
    "Rate limit hit",
    "[ERROR]"
]

IGNORE_PATTERNS = [
    "WinError 10054",
    "Connection reset",
    "Connection aborted",
    "RemoteDisconnected",
    "Authorization for Unified V5 (Auth) failed"
]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [SUPERVISOR] %(message)s'
)
logger = logging.getLogger("Supervisor")

class ServiceManager:
    """Manages a single child process (Trader or Keeper)."""
    
    def __init__(self, name, script_name, log_file, cmd_args, webhook_url, clean_start=False):
        self.name = name
        self.script_name = script_name
        self.log_path = Path(log_file)
        self.cmd = [sys.executable, script_name] + cmd_args
        self.webhook_url = webhook_url
        self.clean_start = clean_start
        
        self.process_pid = None
        self.last_health_report = time.time()
        self.restart_count = 0
        self.file_handle = None # For reading logs
        
        self.ensure_log_exists()
        
    def ensure_log_exists(self):
        if not self.log_path.exists():
            with open(self.log_path, 'w') as f: f.write("")

    def notify(self, message, is_error=False):
        if not self.webhook_url: return
        prefix = f"**[{self.name}]**"
        if is_error: prefix += " 🚨"
        try:
            requests.post(self.webhook_url, json={"content": f"{prefix} {message}"}, timeout=5)
        except: pass

    def find_existing_process(self):
        """Finds a process running the specific script with same args."""
        if sys.platform != "win32": return None

        try:
            signature_script = self.script_name
            # We look for the script name in the command line
            cmd = 'wmic process where "name=\'python.exe\'" get ProcessId,CommandLine /format:csv'
            result = subprocess.check_output(cmd, shell=True).decode('utf-8', errors='ignore')
            
            lines = result.strip().splitlines()
            for line in lines:
                parts = line.split(',')
                if len(parts) < 2: continue
                pid_str = parts[-1].strip()
                if not pid_str.isdigit(): continue
                
                command_line = line
                # Match script name AND args (simple check)
                if signature_script in command_line:
                    # Avoid self
                    if int(pid_str) != os.getpid():
                         return int(pid_str)
        except Exception as e:
            logger.error(f"[{self.name}] Discovery Error: {e}")
        return None

    def is_running(self):
        if not self.process_pid: return False
        if sys.platform == "win32":
            try:
                cmd = f'tasklist /FI "PID eq {self.process_pid}" /NH'
                output = subprocess.check_output(cmd, shell=True).decode()
                return str(self.process_pid) in output
            except: return False
        else:
            try:
                os.kill(self.process_pid, 0)
                return True
            except: return False

    def start(self):
        # 0. Clean Start (One-off)
        if self.clean_start:
            existing = self.find_existing_process()
            while existing:
                logger.warning(f"[{self.name}] CLEAN START: Killing existing PID {existing}...")
                if sys.platform == "win32":
                    subprocess.call(['taskkill', '/F', '/T', '/PID', str(existing)])
                else:
                    try: os.kill(existing, signal.SIGKILL)
                    except: pass
                time.sleep(1)
                existing = self.find_existing_process()
            self.clean_start = False # Done

        # 1. Try to adopt
        existing = self.find_existing_process()
        if existing:
            self.process_pid = existing
            logger.info(f"[{self.name}] Adopted PID {existing}")
            self.notify(f"Adopted active PID {existing}")
            self.open_log_reader()
            return

        # 2. Start new detached
        logger.info(f"[{self.name}] Starting new process...")
        self.notify("Starting new session")
        
        creationflags = 0x00000008 | 0x00000200 # DETACHED + NEW_GROUP
        
        try:
            # We rely on the script's internal FileHandler for logging.
            # We also redirect stdout/stderr to the file to capture crashes/unhandled exceptions.
            self.output_file = open(self.log_path, 'a')
            
            proc = subprocess.Popen(
                self.cmd,
                creationflags=creationflags,
                close_fds=True,
                stdout=self.output_file,
                stderr=subprocess.STDOUT
            )
            self.process_pid = proc.pid
            self.last_health_report = time.time()
            self.open_log_reader()
        except Exception as e:
            logger.error(f"[{self.name}] Start Failed: {e}")
            if hasattr(self, 'output_file') and self.output_file:
                self.output_file.close()

    def stop(self, reason):
        logger.warning(f"[{self.name}] Stopping PID {self.process_pid}: {reason}")
        self.notify(f"Stopping ({reason})", is_error=True)
        
        if self.process_pid:
            if sys.platform == "win32":
                subprocess.call(['taskkill', '/F', '/T', '/PID', str(self.process_pid)])
            else:
                try: os.kill(self.process_pid, signal.SIGKILL)
                except: pass
        
        self.process_pid = None
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
            
        if hasattr(self, 'output_file') and self.output_file:
            try: self.output_file.close()
            except: pass
            self.output_file = None

    def open_log_reader(self):
        if self.file_handle: self.file_handle.close()
        try:
            self.file_handle = open(self.log_path, 'r')
            self.file_handle.seek(0, os.SEEK_END)
        except Exception as e:
            logger.error(f"[{self.name}] Log Open Error: {e}")

    def tick(self, network_ok, disk_ok):
        """Called every loop iteration."""
        # 1. Ensure Running
        if not self.process_pid or not self.is_running():
            if self.process_pid:
                self.stop("Process Vanished")
            self.start()
            return

        # 2. Environment Checks (Global)
        if not network_ok:
            # We might treat net loss differently for Trader vs Keeper?
            # For now, strict: kill and restart when net back
            self.stop("Network Lost")
            return

        if not disk_ok:
            self.stop("Disk Full")
            return

        # 3. Read Logs
        if not self.file_handle: return

        try:
            lines = self.file_handle.readlines() # Read all new lines
            if not lines:
                # Silence Check
                if (time.time() - self.last_health_report) > MAX_SILENCE_SECONDS:
                    self.stop(f"Frozen ({MAX_SILENCE_SECONDS}s silence)")
                return

            for line in lines:
                # ECHO TO CONSOLE (Capture Output)
                print(f"[{self.name}] {line.strip()}")
                
                if ">>> [SYSTEM METRICS]" in line:
                    self.last_health_report = time.time()
                
                if any(p in line for p in IGNORE_PATTERNS): continue
                
                for pattern in CRITICAL_PATTERNS:
                    if pattern in line:
                        self.stop(f"Log Error: {line.strip()}")
                        return
        except Exception as e:
            logger.error(f"[{self.name}] Log Read Error: {e}")


class Supervisor:
    def __init__(self, cmd_args, clean_start=False):
        webhook_url = self.load_webhook_url()
        
        # Initialize Services
        self.trader = ServiceManager(
            "TRADER", "live_trading.py", "live_trading.log", 
            cmd_args, webhook_url, clean_start
        )
        self.keeper = ServiceManager(
            "KEEPER", "data_keeper.py", "data_keeper.log", 
            cmd_args, webhook_url, clean_start
        )
        
    def load_webhook_url(self):
        url = os.getenv("DISCORD_WEBHOOK_URL")
        if not url:
            try:
                with open("keys.json", "r") as f:
                    keys = json.load(f)
                    url = keys.get("discord_webhook_url")
            except: pass
        return url

    def check_network(self):
        try:
            socket.setdefaulttimeout(3)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((NET_CHECK_HOST, 53))
            return True
        except: return False

    def check_disk(self):
        total, used, free = shutil.disk_usage(".")
        return (free // (2**20)) > MIN_DISK_MB

    def run(self):
        logger.info("Supervisor Started. Monitoring Trader & Keeper...")
        
        while True:
            # 1. Global Checks
            net_ok = self.check_network()
            disk_ok = self.check_disk()
            
            # 2. Tick Services
            self.trader.tick(net_ok, disk_ok)
            self.keeper.tick(net_ok, disk_ok)
            
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python watchdog.py [shared_args...] [--start-clean]")
        sys.exit(1)
    
    args = sys.argv[1:]
    clean_start = False
    
    if "--start-clean" in args:
        clean_start = True
        args.remove("--start-clean")
    
    sup = Supervisor(args, clean_start=clean_start)
    sup.run()