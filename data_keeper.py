import sys
import logging
import argparse
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from live_trading import LiveBot

# --- FIX LOGGING INTERFERENCE ---
# live_trading.py sets up logging on import. We must clear it.
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Silence Parent Class (LiveBot) Noise
logging.getLogger("LiveTrading").setLevel(logging.WARNING)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [DATA_KEEPER] %(message)s',
    handlers=[
        logging.FileHandler("data_keeper.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("DataKeeper")

class DataKeeperBot(LiveBot):
    """
    A specialized version of LiveBot that ONLY handles data.
    It inherits all connection, download, and feature calculation logic,
    but overrides trading methods to be no-ops (do nothing).
    """
    def __init__(self, args):
        super().__init__(args)
        # Override history file to prevent write conflicts with the main bot
        self.history_file = Path(f"backup_history_{self.symbol}_{self.timeframe}.csv")
        logger.info(f"Keeper initialized. Saving to backup file: {self.history_file}")

    def execute_logic(self, p_long, p_short, p_dir_long, p_dir_short, close, atr):
        """
        Override: Do not execute trades.
        This method is called after every bar close and feature save.
        """
        logger.info(f"✓ Data synced & features updated. (History file: {self.history_file})")

    def reconcile_positions(self):
        """Override: Do not sync positions with exchange."""
        pass

    def manage_exits(self):
        # We sleep briefly to simulate the Trader's network lag.
        # This averages the loop speed to ~9Hz (2700 snapshots/bar).
        time.sleep(0.11)
    
    def check_order_expiry(self):
        """Override: Do not cancel orders."""
        pass
        
    def check_consistency(self):
        """Override: Do not check VPM vs Exchange consistency."""
        pass
    
    def on_execution_update(self, msg):
        """Override: Ignore execution updates."""
        pass

    def on_bar_close(self, ws_kline):
        """Override: Suppress verbose info logs from the parent class."""
        # The parent methods use the 'LiveTrading' logger. We silence it temporarily.
        lt_logger = logging.getLogger("LiveTrading")
        previous_level = lt_logger.level
        lt_logger.setLevel(logging.WARNING)
        
        try:
            super().on_bar_close(ws_kline)
        finally:
            lt_logger.setLevel(previous_level)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Keeper - Maintains history file using LiveBot logic.")
    parser.add_argument("--symbol", type=str, required=True, help="Trading Pair (e.g., BTCUSDT)")
    parser.add_argument("--model-dir", "--model_dir", type=str, required=True, dest="model_dir", help="Path to model directory (required for feature list)")
    parser.add_argument("--api-key", "--api_key", type=str, default="", dest="api_key", help="Bybit API Key")
    parser.add_argument("--api-secret", "--api_secret", type=str, default="", dest="api_secret", help="Bybit API Secret")
    parser.add_argument("--testnet", action="store_true", help="Use Testnet")
    parser.add_argument("--timeframe", "--time_frame", type=str, default="5m", dest="timeframe", help="Timeframe (e.g., 5m)")
    
    args = parser.parse_args()
    
    # Load keys if not provided (same logic as live_trading)
    if not args.api_key:
        try:
            with open("keys.json") as f:
                keys = json.load(f)
                args.api_key = keys.get("api_key")
                args.api_secret = keys.get("api_secret")
        except:
            logger.warning("No API keys found in keys.json.")

    logger.info(f"Starting Data Keeper for {args.symbol}...")
    
    try:
        bot = DataKeeperBot(args)
        bot.run()
    except KeyboardInterrupt:
        logger.info("Data Keeper stopped by user.")
    except Exception as e:
        logger.error(f"Data Keeper crashed: {e}")
