
import subprocess
import sys
import time

def run_collector():
    print("Starting collector for 15s...")
    # Start the collector as a subprocess
    process = subprocess.Popen([sys.executable, "data_collector.py", "PIEVERSEUSDT"])
    
    # Let it run
    time.sleep(15)
    
    # Terminate
    print("Stopping collector...")
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
    print("Collector stopped.")

if __name__ == "__main__":
    run_collector()
