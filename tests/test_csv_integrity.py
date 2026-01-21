import unittest
import pandas as pd
import numpy as np
import os
import time
import json
import argparse
import sys
from pathlib import Path
from live_trading import LiveBot
from config import CONF
import logging

# Set up logging for the test
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CSVIntegrityTest")

class TestCSVIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Mocking minimal args for LiveBot
        class Args:
            symbol = "FARTCOINUSDT"
            model_dir = "models/FARTCOINUSDT/rank_1"
            api_key = ""
            api_secret = ""
            testnet = False
            timeframe = "5m"
        
        cls.args = Args()
        # Load keys if available
        if os.path.exists("keys.json"):
            with open("keys.json") as f:
                k = json.load(f)
                cls.args.api_key = k.get("api_key", "")
                cls.args.api_secret = k.get("api_secret", "")

    def setUp(self):
        self.test_csv = Path(f"test_integrity_{self.args.symbol}.csv")
        # Initialize Bot
        self.bot = LiveBot(self.args)
        # Override history file to avoid touching production data
        self.bot.history_file = self.test_csv

    def tearDown(self):
        if self.test_csv.exists():
            try:
                os.remove(self.test_csv)
            except: pass

    def inject_raw_columns(self, df):
        """Simulates the raw data columns the bot adds live before feature calculation."""
        df = df.copy()
        df['taker_buy_ratio'] = 0.5
        ob_raw_cols = [
            'ob_spread_mean', 'ob_micro_dev_mean', 'ob_micro_dev_std', 'ob_micro_dev_last',
            'ob_imbalance_mean', 'ob_imbalance_last', 'ob_bid_depth_mean', 'ob_ask_depth_mean',
            'ob_bid_slope_mean', 'ob_ask_slope_mean', 'ob_bid_integrity_mean', 'ob_ask_integrity_mean'
        ]
        for col in ob_raw_cols:
            df[col] = 1.0 # Use 1.0 to avoid division by zero in some tests
        return df

    def test_full_lifecycle(self):
        """
        Comprehensive test: Fetch -> Calculate -> Save -> Load -> Predict
        """
        logger.info("Step 1: Fetching 500 bars from exchange...")
        klines = self.bot.rest_api.get_public_klines(self.args.symbol, self.args.timeframe, limit=500)
        self.assertGreater(len(klines), 0, "Failed to fetch data from exchange.")
        
        df_raw = pd.DataFrame(klines)
        logger.info(f"Fetched {len(df_raw)} bars.")

        logger.info("Step 2: Injecting raw data and calculating features...")
        df_raw_enriched = self.inject_raw_columns(df_raw)
        df_feats = self.bot.feature_engine.calculate_features(df_raw_enriched)
        
        # Ensure all model features exist
        missing = [c for c in self.bot.model_manager.feature_cols if c not in df_feats.columns]
        self.assertEqual(len(missing), 0, f"Feature Engine omitted critical columns: {missing}")

        logger.info("Step 3: Simulating Live Update & Save...")
        self.bot.bars = df_feats.copy()
        self.bot.bars.to_csv(self.test_csv, index=False)
        self.assertTrue(self.test_csv.exists(), "CSV was not saved")

        logger.info("Step 4: Reloading and verifying Alignment...")
        df_loaded = pd.read_csv(self.test_csv)
        
        # Check if all required features are present
        loaded_cols = list(df_loaded.columns)
        for col in self.bot.model_manager.feature_cols:
            self.assertIn(col, loaded_cols, f"Feature {col} missing after CSV reload")

        # Check for numeric corruption
        df_loaded = df_loaded.replace([np.inf, -np.inf], 0).fillna(0)
        nan_count = df_loaded[self.bot.model_manager.feature_cols].isna().sum().sum()
        self.assertEqual(nan_count, 0, f"Found {nan_count} NaNs in reloaded CSV")

        logger.info("Step 5: Verifying Model Compatibility...")
        try:
            # The model expects specific features, pandas slice [lgb_feats] handles order
            input_df = df_loaded.iloc[-100:].copy()
            preds = self.bot.model_manager.predict(input_df)
            logger.info("Prediction successful on reloaded CSV data.")
            
            self.assertIn('pred_long', preds.columns)
            self.assertIn('pred_short', preds.columns)
            
            p1 = preds['pred_long'].iloc[-1]
            self.assertTrue(0 <= p1 <= 1, f"Invalid prediction value: {p1}")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.fail(f"Model failed to process reloaded CSV data: {e}")

    def test_column_consistency(self):
        """Checks if the calculated features match the model's expected input set."""
        logger.info("Checking Feature Set alignment...")
        
        model_req = self.bot.model_manager.feature_cols
        
        klines = self.bot.rest_api.get_public_klines(self.args.symbol, self.args.timeframe, limit=20)
        df_raw = pd.DataFrame(klines)
        df_raw_enriched = self.inject_raw_columns(df_raw)
        df_sample = self.bot.feature_engine.calculate_features(df_raw_enriched)
        
        for feat in model_req:
            self.assertIn(feat, df_sample.columns, f"Critical feature '{feat}' is missing from Engine output!")
        logger.info("Feature Set Alignment: OK")

if __name__ == "__main__":
    unittest.main()