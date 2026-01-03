
import os
import time
import json
import logging
from datetime import datetime
from config import CONF
from exchange_client import ExchangeClient

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("DebugBot")

def run_debug():
    key = os.getenv("BYBIT_API_KEY")
    secret = os.getenv("BYBIT_API_SECRET")
    
    if not key or not secret:
        logger.warning("No API keys found in env. Cannot connect to real exchange.")
        return

    symbol = CONF.data.symbol
    logger.info(f"Connecting to Bybit for {symbol}...")
    client = ExchangeClient(key, secret, symbol)
    
    # 1. Check Time Sync
    logger.info("-" * 40)
    logger.info("Checking Time Synchronization...")
    drift = client.check_time_sync()
    local_now = datetime.utcnow()
    logger.info(f"Local UTC Time: {local_now}")
    logger.info(f"Drift: {drift:.2f} ms")
    
    # 2. Check Positions
    logger.info("-" * 40)
    logger.info("Fetching Position Details...")
    try:
        # We access the internal session to get raw response for debugging
        resp = client.session.get_positions(category="linear", symbol=symbol)
        positions = resp.get('result', {}).get('list', [])
        
        if not positions:
            logger.info("No positions found on exchange.")
        
        for p in positions:
            logger.info(f"Found Position: {p['symbol']} {p['side']} {p['size']}")
            logger.info(f"Raw Data: {json.dumps(p, indent=2)}")
            
            created_time_str = p.get('createdTime')
            updated_time_str = p.get('updatedTime')
            
            logger.info(f"createdTime (Raw): '{created_time_str}' (Type: {type(created_time_str)})")
            logger.info(f"updatedTime (Raw): '{updated_time_str}' (Type: {type(updated_time_str)})")
            
            # Simulate Logic from live_trader.py
            current_time = datetime.utcnow()
            
            created_ms = int(created_time_str) if created_time_str else 0
            if created_ms > 0:
                entry_time = datetime.utcfromtimestamp(created_ms / 1000.0)
                hold_seconds = (current_time - entry_time).total_seconds()
                
                logger.info(f"Entry Time (Parsed): {entry_time}")
                logger.info(f"Current Time (UTC):  {current_time}")
                logger.info(f"Hold Seconds: {hold_seconds:.2f}s ({hold_seconds/60:.2f}m)")
                
                max_hold_bars = CONF.strategy.max_holding_bars # 144
                max_hold_seconds = max_hold_bars * 300 # 5m * 60s
                
                logger.info(f"Max Hold Allowed: {max_hold_seconds}s ({max_hold_seconds/60:.2f}m)")
                
                if hold_seconds > max_hold_seconds:
                    logger.warning("!!! TIMEOUT CONDITION MET !!!")
                else:
                    logger.info("Timeout Condition: False")
            else:
                logger.error("createdTime is 0 or Invalid! This would cause infinite hold time or errors.")

    except Exception as e:
        logger.error(f"Error fetching positions: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_debug()
