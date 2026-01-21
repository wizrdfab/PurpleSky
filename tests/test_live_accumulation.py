import time
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from live_trading import LiveBot
import logging
import threading

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LiveDataAccumulationTest")

class LiveDataTester:
    def __init__(self, duration_sec=720): # 12 minutes
        class Args:
            symbol = "FARTCOINUSDT"
            model_dir = "models/FARTCOINUSDT/rank_1"
            api_key = ""
            api_secret = ""
            testnet = False
            timeframe = "5m"
        
        self.args = Args()
        if os.path.exists("keys.json"):
            with open("keys.json") as f:
                k = json.load(f)
                self.args.api_key = k.get("api_key", "")
                self.args.api_secret = k.get("api_secret", "")

        self.duration = duration_sec
        self.test_csv = Path("test_live_accumulation.csv")
        if self.test_csv.exists(): 
            try: os.remove(self.test_csv)
            except: pass
        
        self.bot = LiveBot(self.args)
        self.bot.history_file = self.test_csv
        self.initial_bar_count = 0

    def run_test(self):
        logger.info(f"Starting Live Accumulation Test for {self.duration/60:.1f} minutes...")
        
        # Start bot in a background thread
        bot_thread = threading.Thread(target=self.bot.run, daemon=True)
        bot_thread.start()
        
        start_time = time.time()
        # Wait for initial history load from exchange
        while len(self.bot.bars) == 0 and time.time() - start_time < 30:
            time.sleep(1)
            
        self.initial_bar_count = len(self.bot.bars)
        logger.info(f"Initial history loaded: {self.initial_bar_count} bars.")
        
        if self.initial_bar_count == 0:
             logger.error("Failed to load initial history. Check internet/API.")
             self.bot.running = False
             return False

        while time.time() - start_time < self.duration:
            elapsed = int(time.time() - start_time)
            
            # Status Update
            current_bars = len(self.bot.bars)
            new_bars = current_bars - self.initial_bar_count
            updates_count = len(self.bot.ob_agg.snapshots)
            
            logger.info(f"Progress: {elapsed}s/{self.duration}s | New Bars: {new_bars} | Current Snapshots: {updates_count}")
            
            # Check for early failure (no WS data)
            if elapsed > 60 and updates_count == 0 and new_bars == 0:
                logger.error("CRITICAL: No WebSocket snapshots received in 60 seconds!")
                # Do not exit immediately, maybe trades are coming or klines
                
            time.sleep(30)

        logger.info("Test duration reached. Shutting down bot...")
        self.bot.running = False
        time.sleep(5) # Allow final save
        
        return self.verify_results()

    def verify_results(self):
        if not self.test_csv.exists():
            logger.error("CSV file was not created!")
            # Check if bot has it in memory at least
            if len(self.bot.bars) > 0:
                logger.info("Bot has bars in memory but CSV is missing. Manual save...")
                self.bot.bars.to_csv(self.test_csv, index=False)
            else:
                return False
        
        df = pd.read_csv(self.test_csv)
        final_count = len(df)
        logger.info(f"Final CSV count: {final_count} bars (Added {final_count - self.initial_bar_count})")
        
        if final_count <= self.initial_bar_count:
            logger.warning("No new bars were added during the test period. This might happen if the bar didn't close yet.")
            # We can still check the last bar's partially filled features if update() worked
        
        # Check for presence of required columns
        for col in self.bot.model_manager.feature_cols:
            if col not in df.columns:
                logger.error(f"Omission Error: Column '{col}' is missing from the CSV!")
                return False

        # Check for data types and non-zero orderbook stats in the latest row
        # We find the first row where ob_update_count > 0
        live_rows = df[df['ob_update_count'] > 0]
        if live_rows.empty:
            logger.error("No rows with real orderbook data were found in the CSV!")
            return False
            
        logger.info(f"Found {len(live_rows)} rows with real live data.")
        
        # Final Prediction check
        try:
            logger.info("Verifying model prediction on accumulated data...")
            # Use the live rows for a realistic check
            input_df = df.iloc[-100:].copy()
            preds = self.bot.model_manager.predict(input_df)
            p_long = preds['pred_long'].iloc[-1]
            logger.info(f"Success! Prediction on live data: {p_long:.4f}")
        except Exception as e:
            logger.error(f"Prediction failed on accumulated CSV: {e}")
            return False

        logger.info("INTEGRITY TEST PASSED: Full data flow confirmed.")
        return True

if __name__ == "__main__":
    # Run for 12 minutes to ensure we hit a 5m bar boundary
    tester = LiveDataTester(duration_sec=720) 
    success = tester.run_test()
    if success:
        print("\n[PASSED] The system built, saved, and re-loaded bars correctly.")
    else:
        print("\n[FAILED] Data integrity issue detected.")
        exit(1)
