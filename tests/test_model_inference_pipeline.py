import unittest
import pandas as pd
import numpy as np
import os
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import ModelManager
from feature_engine import FeatureEngine
from config import CONF, ModelConfig

class TestRealModelEndToEnd(unittest.TestCase):
    def setUp(self):
        # 1. Setup Isolation
        self.test_csv_path = Path("tests/temp_inference_data.csv")
        
        # 2. Point to REAL model
        self.real_model_path = Path("models/FARTCOINUSDT/rank_1")
        if not self.real_model_path.exists():
            self.skipTest("Real model directory not found")

        # 3. Load Real Manager (CPU forced)
        self.config = CONF.model
        self.config.model_dir = self.real_model_path
        self.manager = self._load_manager()
        
        # 4. Initialize Feature Engine
        self.engine = FeatureEngine(CONF.features)

    def tearDown(self):
        # Cleanup isolated file
        if self.test_csv_path.exists():
            os.remove(self.test_csv_path)

    def _load_manager(self):
        import joblib
        mm = ModelManager(self.config)
        mm.device = torch.device('cpu') 
        
        mm.feature_cols = joblib.load(self.real_model_path / "features.pkl")
        
        context_keywords = ['atr', 'vol', 'rsi', 'std', 'slope']
        mm.context_cols = [c for c in mm.feature_cols if any(k in c.lower() for k in context_keywords)]
        if not mm.context_cols: mm.context_cols = mm.feature_cols[:10]
        
        mm.model_long = joblib.load(self.real_model_path / "model_long.pkl")
        mm.model_short = joblib.load(self.real_model_path / "model_short.pkl")
        
        if (self.real_model_path / "dir_model_long.pkl").exists():
            mm.dir_model_long = joblib.load(self.real_model_path / "dir_model_long.pkl")
            mm.dir_model_short = joblib.load(self.real_model_path / "dir_model_short.pkl")
            
        if (self.real_model_path / "lstm_model.pth").exists():
            from models import LSTMModel, GatingNetwork
            mm.lstm_model = LSTMModel(len(mm.feature_cols), self.config.lstm_hidden_size, 1, 0.0).to(mm.device)
            mm.lstm_model.load_state_dict(torch.load(self.real_model_path / "lstm_model.pth", map_location=mm.device, weights_only=True))
            
            mm.gating_model = GatingNetwork(len(mm.context_cols)).to(mm.device)
            mm.gating_model.load_state_dict(torch.load(self.real_model_path / "gating_model.pth", map_location=mm.device, weights_only=True))
            
            mm.scaler = joblib.load(self.real_model_path / "scaler.pkl")
            
        return mm

    def test_csv_roundtrip_inference(self):
        """
        Full Lifecycle Test:
        1. Generate Raw Data
        2. Save to CSV (Mimic LiveBot storage)
        3. Load from CSV (Mimic LiveBot startup)
        4. Calculate Features
        5. Ingest into Model
        6. Verify Predictions
        """
        print("\n--- Starting End-to-End CSV Roundtrip Test ---")
        
        # 1. Generate Raw Data
        n_rows = 300
        dates = pd.date_range(start='2024-01-01', periods=n_rows, freq='5min')
        df = pd.DataFrame(index=range(n_rows))
        df['timestamp'] = dates.astype(np.int64) // 10**6
        df['open'] = 100.0
        df['high'] = 101.0
        df['low'] = 99.0
        df['close'] = 100.0 + np.random.randn(n_rows).cumsum() # Random walk
        df['volume'] = np.random.randint(1000, 10000, n_rows).astype(float)
        df['taker_buy_ratio'] = 0.5
        
        # Add Raw OB Stats (required for feature calc)
        raw_ob_cols = [
            'ob_spread_mean', 'ob_micro_dev_mean', 'ob_micro_dev_std', 'ob_micro_dev_last',
            'ob_imbalance_mean', 'ob_imbalance_last', 'ob_bid_depth_mean', 'ob_ask_depth_mean',
            'ob_bid_slope_mean', 'ob_ask_slope_mean', 'ob_bid_integrity_mean', 'ob_ask_integrity_mean'
        ]
        for c in raw_ob_cols:
            df[c] = np.random.random(n_rows)

        # 2. Save to CSV (Simulating DataKeeper/LiveBot persistence)
        df.to_csv(self.test_csv_path, index=False)
        print(f"Saved synthetic history to {self.test_csv_path}")
        
        # 3. Load from CSV
        loaded_df = pd.read_csv(self.test_csv_path)
        print(f"Loaded {len(loaded_df)} bars from disk.")
        
        # 4. Feature Engineering
        # Calculate features on the loaded data
        df_feats = self.engine.calculate_features(loaded_df)
        print(f"Calculated {len(df_feats.columns)} columns (Features + Raw).")
        
        # 5. Ingestion & Prediction
        # This calls 'prepare_sequences' internally which scales the data
        try:
            preds = self.manager.predict(df_feats)
            print("Model inference successful.")
        except KeyError as e:
            self.fail(f"Model Ingestion Failed! Missing columns: {e}")
        except ValueError as e:
            self.fail(f"Model Ingestion Failed! Value Error (NaNs/Shape): {e}")
            
        # 6. Verification
        # Check standard output
        self.assertTrue('pred_long' in preds.columns)
        self.assertTrue('pred_short' in preds.columns)
        
        # Check values are valid probabilities
        p_long = preds['pred_long'].iloc[-1]
        p_short = preds['pred_short'].iloc[-1]
        
        print(f"Sample Prediction -> Long: {p_long:.4f}, Short: {p_short:.4f}")
        
        self.assertTrue(0 <= p_long <= 1, "Long prediction out of range")
        self.assertTrue(0 <= p_short <= 1, "Short prediction out of range")
        
        # Check LSTM Gating (if applicable)
        if 'gate_weight' in preds.columns:
            gate = preds['gate_weight'].iloc[-1]
            print(f"LSTM Gate Weight: {gate:.4f}")
            self.assertTrue(0 <= gate <= 1, "Gate weight out of range")

if __name__ == '__main__':
    unittest.main()