import unittest
import pandas as pd
import numpy as np
import os
import time
import json
from pathlib import Path
from live_trading import LiveBot
from orderbook_aggregator import OrderbookAggregator
from local_orderbook import LocalOrderbook
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RealDataFlowTest")

class TestRealDataFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class Args:
            symbol = "FARTCOINUSDT"
            model_dir = "models/FARTCOINUSDT/rank_1"
            api_key = ""
            api_secret = ""
            testnet = False
            timeframe = "5m"
        cls.args = Args()
        if os.path.exists("keys.json"):
            with open("keys.json") as f:
                k = json.load(f)
                cls.args.api_key = k.get("api_key", "")
                cls.args.api_secret = k.get("api_secret", "")

    def setUp(self):
        self.bot = LiveBot(self.args)
        self.test_csv = Path("test_real_flow.csv")
        self.bot.history_file = self.test_csv

    def tearDown(self):
        if self.test_csv.exists():
            try: os.remove(self.test_csv)
            except: pass

    def test_end_to_end_real_data(self):
        """
        Tests: Real Snapshots -> Aggregator -> Bar -> Features -> CSV
        """
        logger.info("Step 1: Fetching REAL Live Orderbook Snapshots from REST...")
        agg = OrderbookAggregator(ob_levels=50)
        
        # Collect 5 real snapshots with slight delays to simulate time passing
        for i in range(5):
            snap = self.bot.rest_api.get_orderbook(self.args.symbol, limit=50)
            self.assertTrue(snap and 'bids' in snap, "Failed to fetch real orderbook")
            
            # Feed to Aggregator
            agg.process_snapshot(snap['bids'], snap['asks'], snap['timestamp'])
            logger.info(f"  Snapshot {i+1} processed. Bid[0]: {snap['bids'][0][0]}")
            time.sleep(0.5)

        # 2. Finalize Aggregator
        ob_stats = agg.finalize()
        logger.info(f"Step 2: Aggregated Stats: {ob_stats}")
        self.assertGreater(ob_stats['ob_update_count'], 0)
        self.assertGreater(ob_stats['ob_bid_depth_mean'], 0)

        # 3. Create a Bar using Real OHLCV + Real OB Stats
        logger.info("Step 3: Creating Bar with Real Data...")
        klines = self.bot.rest_api.get_public_klines(self.args.symbol, self.args.timeframe, limit=100)
        df_base = pd.DataFrame(klines)
        
        # Construct the 'Live' row exactly as on_bar_close does
        new_row = {
            'timestamp': int(time.time() * 1000),
            'open': df_base.iloc[-1]['close'],
            'high': df_base.iloc[-1]['close'] * 1.001,
            'low': df_base.iloc[-1]['close'] * 0.999,
            'close': df_base.iloc[-1]['close'],
            'volume': 1000.0,
            'taker_buy_ratio': 0.55
        }
        new_row.update(ob_stats)
        
        # Append to history
        self.bot.bars = pd.concat([df_base, pd.DataFrame([new_row])], ignore_index=True)

        # 4. Calculate Features
        logger.info("Step 4: Calculating Features on Real Data Flow...")
        df_feats = self.bot.feature_engine.calculate_features(self.bot.bars)
        
        # Check if OB features were calculated from the real data
        # We check the last row specifically
        last_row = df_feats.iloc[-1]
        self.assertNotEqual(last_row['ob_depth_log_ratio'], 0, "OB Features failed to calculate from real snapshots")
        logger.info(f"  Real OB Feature Sample (Log Ratio): {last_row['ob_depth_log_ratio']}")

        # 5. Save and Reload (Ordering Check)
        logger.info("Step 5: Verifying CSV Persistence and Column Order...")
        df_feats.to_csv(self.test_csv, index=False)
        df_loaded = pd.read_csv(self.test_csv)
        
        # Compare columns
        self.assertEqual(list(df_feats.columns), list(df_loaded.columns), "Column order jumbled in real flow!")
        
        # 6. Model Prediction
        logger.info("Step 6: Final Prediction Test...")
        preds = self.bot.model_manager.predict(df_loaded.iloc[-100:].copy())
        self.assertIn('pred_long', preds.columns)
        logger.info(f"  Prediction Successful. Long Prob: {preds['pred_long'].iloc[-1]:.4f}")

if __name__ == "__main__":
    unittest.main()
