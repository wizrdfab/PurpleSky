"""
Copyright (C) 2026 Fabián Zúñiga Franck

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import lightgbm as lgb
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from config import ModelConfig

# --- PyTorch Components ---

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
        # nn.LSTM dropout only applies between layers (requires num_layers > 1)
        # We'll use 0 for the internal LSTM dropout if num_layers == 1 to avoid the warning
        internal_dropout = dropout if num_layers > 1 else 0
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=internal_dropout)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc_long = nn.Linear(hidden_size, 1)
        self.fc_short = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        # Take last time step
        last_out = out[:, -1, :]
        
        # Apply explicit dropout
        last_out = self.dropout_layer(last_out)
        
        pred_long = self.sigmoid(self.fc_long(last_out))
        pred_short = self.sigmoid(self.fc_short(last_out))
        return pred_long, pred_short

class GatingNetwork(nn.Module):
    """
    Mixture of Experts Gating Network.
    Input: Market Context Features (subset of features)
    Output: Weight (alpha) for GBM. (1-alpha) for LSTM.
    """
    def __init__(self, input_size):
        super(GatingNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

class SequenceDataset(Dataset):
    def __init__(self, X, y_long, y_short, seq_len):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_long = torch.tensor(y_long, dtype=torch.float32)
        self.y_short = torch.tensor(y_short, dtype=torch.float32)
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.X) - self.seq_len
    
    def __getitem__(self, idx):
        # Input: X[idx : idx+seq_len]
        # Target: y[idx+seq_len-1] (Predicting for the last bar in sequence)
        return (self.X[idx : idx+self.seq_len], 
                self.y_long[idx + self.seq_len - 1],
                self.y_short[idx + self.seq_len - 1])

# --- Main Manager ---

class ModelManager:
    def __init__(self, config: ModelConfig):
        self.config = config
        
        # LightGBM Models (Execution)
        self.model_long = None
        self.model_short = None
        
        # LightGBM Models (Direction - Meta Labeling)
        self.dir_model_long = None
        self.dir_model_short = None
        
        # LSTM Ensemble Components
        self.lstm_model = None
        self.gating_model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.feature_cols = []
        self.context_cols = [] # For Gating

    def prepare_sequences(self, df, fit_scaler=False):
        # Extract features
        X = df[self.feature_cols].values
        # Handle NaN/Inf
        X = np.nan_to_num(X)
        
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
            
        return X_scaled

    def train(self, df: pd.DataFrame, feature_cols: list):
        self.feature_cols = feature_cols
        # Identify context features for Gating (e.g., volatility, regime)
        # Using a heuristic list of keywords
        context_keywords = ['atr', 'vol', 'rsi', 'std', 'slope']
        self.context_cols = [c for c in feature_cols if any(k in c.lower() for k in context_keywords)]
        if not self.context_cols: self.context_cols = feature_cols[:10] # Fallback
        
        # Triple Split Logic
        n = len(df)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))
        
        lgb_feats = [f for f in feature_cols if f != 'close']
        X_train_lgb = df[lgb_feats].iloc[:train_end]
        X_val_lgb = df[lgb_feats].iloc[train_end:val_end]
        
        y_long = df['target_long']
        y_short = df['target_short']
        
        y_dir_long = df.get('target_dir_long', pd.Series(0, index=df.index))
        y_dir_short = df.get('target_dir_short', pd.Series(0, index=df.index))
        
        print(f"Train: {len(X_train_lgb)} | Val: {len(X_val_lgb)}")
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'dart', 
            'learning_rate': self.config.learning_rate,
            'num_leaves': self.config.num_leaves,
            'max_depth': self.config.max_depth,
            'min_child_samples': self.config.min_child_samples,
            'subsample': self.config.subsample,
            'colsample_bytree': self.config.colsample_bytree,
            'rate_drop': self.config.rate_drop,
            'extra_trees': self.config.extra_trees,
            'skip_drop': 0.5,
            'n_jobs': -1,
            'verbosity': -1
        }
        
        # --- 1. Train Direction Models (Model A) ---
        if self.config.use_meta_labeling:
            print("Training Direction Models (Model A)...")
            self.dir_model_long = lgb.train(
                params, lgb.Dataset(X_train_lgb, label=y_dir_long.iloc[:train_end]),
                num_boost_round=self.config.n_estimators // 2,
                valid_sets=[lgb.Dataset(X_val_lgb, label=y_dir_long.iloc[train_end:val_end])],
                callbacks=[lgb.early_stopping(stopping_rounds=self.config.early_stopping_rounds), lgb.log_evaluation(0)]
            )
            self.dir_model_short = lgb.train(
                params, lgb.Dataset(X_train_lgb, label=y_dir_short.iloc[:train_end]),
                num_boost_round=self.config.n_estimators // 2,
                valid_sets=[lgb.Dataset(X_val_lgb, label=y_dir_short.iloc[train_end:val_end])],
                callbacks=[lgb.early_stopping(stopping_rounds=self.config.early_stopping_rounds), lgb.log_evaluation(0)]
            )
        
        # --- 2. Train Execution Models (Model B - LightGBM) ---
        print("Training LightGBM Execution Models (Model B)...")
        self.model_long = lgb.train(
            params,
            lgb.Dataset(X_train_lgb, label=y_long.iloc[:train_end]),
            num_boost_round=self.config.n_estimators,
            valid_sets=[lgb.Dataset(X_val_lgb, label=y_long.iloc[train_end:val_end])],
            callbacks=[lgb.early_stopping(stopping_rounds=self.config.early_stopping_rounds), lgb.log_evaluation(0)]
        )
        
        self.model_short = lgb.train(
            params,
            lgb.Dataset(X_train_lgb, label=y_short.iloc[:train_end]),
            num_boost_round=self.config.n_estimators,
            valid_sets=[lgb.Dataset(X_val_lgb, label=y_short.iloc[train_end:val_end])],
            callbacks=[lgb.early_stopping(stopping_rounds=self.config.early_stopping_rounds), lgb.log_evaluation(0)]
        )

        # --- 3. Train LSTM & Gating (Model C & D) ---
        if self.config.use_lstm_ensemble:
            print("Training LSTM & Gating Network (Hybrid Ensemble)...")
            
            # Prepare Data for LSTM
            # Fit scaler ONLY on training data to avoid leakage
            train_df = df.iloc[:train_end]
            val_df = df.iloc[train_end:val_end]
            
            train_np = self.prepare_sequences(train_df, fit_scaler=True)
            val_np = self.prepare_sequences(val_df, fit_scaler=False)
            
            y_long_np = y_long.values[:train_end]
            y_short_np = y_short.values[:train_end]
            
            y_long_val = y_long.values[train_end:val_end]
            y_short_val = y_short.values[train_end:val_end]

            # Create Datasets
            train_dataset = SequenceDataset(train_np, y_long_np, y_short_np, self.config.sequence_length)
            train_loader = DataLoader(train_dataset, batch_size=self.config.lstm_batch_size, shuffle=True)
            
            val_dataset = SequenceDataset(val_np, y_long_val, y_short_val, self.config.sequence_length)
            val_loader = DataLoader(val_dataset, batch_size=self.config.lstm_batch_size, shuffle=False)

            # Init Models
            self.lstm_model = LSTMModel(
                input_size=len(feature_cols),
                hidden_size=self.config.lstm_hidden_size,
                num_layers=self.config.lstm_layers,
                dropout=self.config.lstm_dropout
            ).to(self.device)
            
            self.gating_model = GatingNetwork(
                input_size=len(self.context_cols)
            ).to(self.device)
            
            # Optimizers
            optimizer = optim.Adam(
                list(self.lstm_model.parameters()) + list(self.gating_model.parameters()), 
                lr=self.config.lstm_learning_rate,
                weight_decay=self.config.lstm_weight_decay
            )
            criterion = nn.BCELoss()
            
            # Generate LightGBM predictions for the training set (needed for Gating training)
            # We use raw scores? No, probabilities.
            print("  > Generating LightGBM priors for Gating training...")
            lgb_pred_long = self.model_long.predict(X_train_lgb)
            lgb_pred_short = self.model_short.predict(X_train_lgb)
            
            # Map context columns indices
            ctx_indices = [feature_cols.index(c) for c in self.context_cols]
            
            # Early Stopping Variables
            best_val_loss = float('inf')
            patience_counter = 0
            best_model_state = None

            for epoch in range(self.config.lstm_epochs):
                self.lstm_model.train()
                epoch_loss = 0
                count = 0
                
                for x_seq, y_l, y_s in train_loader:
                    x_seq, y_l, y_s = x_seq.to(self.device), y_l.to(self.device), y_s.to(self.device)
                    
                    # Optional: Add small noise to inputs for regularization
                    if self.lstm_model.training:
                        x_seq = x_seq + torch.randn_like(x_seq) * 0.05

                    # 1. LSTM Forward
                    lstm_l, lstm_s = self.lstm_model(x_seq)
                    
                    # Loss for LSTM (Pure LSTM Training)
                    loss_l = criterion(lstm_l.squeeze(), y_l)
                    loss_s = criterion(lstm_s.squeeze(), y_s)
                    loss = loss_l + loss_s
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    count += 1
                
                # Validation Phase
                self.lstm_model.eval()
                val_loss = 0
                val_count = 0
                with torch.no_grad():
                    for x_v, y_vl, y_vs in val_loader:
                        x_v, y_vl, y_vs = x_v.to(self.device), y_vl.to(self.device), y_vs.to(self.device)
                        pl, ps = self.lstm_model(x_v)
                        v_loss = criterion(pl.squeeze(), y_vl) + criterion(ps.squeeze(), y_vs)
                        val_loss += v_loss.item()
                        val_count += 1
                
                avg_val_loss = val_loss / val_count if val_count > 0 else float('inf')
                print(f"  > Epoch {epoch+1}/{self.config.lstm_epochs} | Train Loss: {epoch_loss/count:.4f} | Val Loss: {avg_val_loss:.4f}")

                # Early Stopping Logic
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_model_state = self.lstm_model.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.lstm_patience:
                        print(f"  > Early stopping triggered at epoch {epoch+1}")
                        break
            
            # Restore best model
            if best_model_state:
                self.lstm_model.load_state_dict(best_model_state)

            self.train_gate_on_validation(df.iloc[train_end:val_end], feature_cols)


    def train_gate_on_validation(self, df_val, feature_cols):
        print("Optimizing Gating Network on Validation Set...")
        lgb_feats = [f for f in feature_cols if f != 'close']
        
        # Get LGB Preds
        p_lgb_l = self.model_long.predict(df_val[lgb_feats])
        p_lgb_s = self.model_short.predict(df_val[lgb_feats])
        
        # Get Targets
        y_val_l = df_val['target_long']
        y_val_s = df_val['target_short']
        
        # Get LSTM Preds
        X_scaled = self.prepare_sequences(df_val, fit_scaler=False)
        
        seq_len = self.config.sequence_length
        if len(df_val) <= seq_len: return
        
        dataset = SequenceDataset(X_scaled, y_val_l.values, y_val_s.values, seq_len)
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        
        # We need LGB preds aligned.
        p_lgb_l = p_lgb_l[seq_len:]
        p_lgb_s = p_lgb_s[seq_len:]
        
        optimizer = optim.Adam(self.gating_model.parameters(), lr=0.01)
        criterion = nn.BCELoss()
        
        ctx_indices = [feature_cols.index(c) for c in self.context_cols]
        
        self.lstm_model.eval()
        self.gating_model.train()
        
        batch_start = 0
        for epoch in range(5):
            batch_start = 0
            for x_seq, y_l, y_s in loader:
                x_seq = x_seq.to(self.device)
                bs = x_seq.size(0)
                
                # LGB Batch
                lgb_l = torch.tensor(p_lgb_l[batch_start : batch_start+bs], dtype=torch.float32).to(self.device)
                lgb_s = torch.tensor(p_lgb_s[batch_start : batch_start+bs], dtype=torch.float32).to(self.device)
                y_l_t = y_l.to(self.device)
                y_s_t = y_s.to(self.device)
                
                batch_start += bs
                
                with torch.no_grad():
                    lstm_l, lstm_s = self.lstm_model(x_seq)
                
                # Gate
                last_step_feats = x_seq[:, -1, :]
                context = last_step_feats[:, ctx_indices]
                alpha = self.gating_model(context)
                
                # Ensemble
                ens_l = alpha.squeeze() * lgb_l + (1 - alpha.squeeze()) * lstm_l.squeeze()
                ens_s = alpha.squeeze() * lgb_s + (1 - alpha.squeeze()) * lstm_s.squeeze()
                
                loss = criterion(ens_l, y_l_t) + criterion(ens_s, y_s_t)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.model_long: raise Exception("Not trained")
        lgb_feats = [f for f in self.feature_cols if f != 'close']
        
        # 1. LightGBM Preds
        lgb_long = self.model_long.predict(df[lgb_feats])
        lgb_short = self.model_short.predict(df[lgb_feats])
        
        if self.config.use_lstm_ensemble and self.lstm_model:
            # 2. LSTM Preds
            # Scale
            X_scaled = self.prepare_sequences(df, fit_scaler=False)
            
            # Need to create sequences.
            # Efficient inference: Rolling window?
            # For backtesting (batch), we can use the Dataset approach but we need to pad the start
            # so output length matches df length.
            # Start: fill with 0 or LGB preds.
            
            n = len(df)
            seq_len = self.config.sequence_length
            
            lstm_long = np.zeros(n)
            lstm_short = np.zeros(n)
            gate_vals = np.ones(n) * 0.5 # Default 0.5
            
            # We can batch inference
            # We only generate predictions where we have enough history
            if n > seq_len:
                # Create dummy targets for dataset (not used)
                dummy_y = np.zeros(n)
                dataset = SequenceDataset(X_scaled, dummy_y, dummy_y, seq_len)
                loader = DataLoader(dataset, batch_size=1024, shuffle=False)
                
                preds_l_list = []
                preds_s_list = []
                gates_list = []
                
                ctx_indices = [self.feature_cols.index(c) for c in self.context_cols]
                
                self.lstm_model.eval()
                self.gating_model.eval()
                
                with torch.no_grad():
                    for x_seq, _, _ in loader:
                        x_seq = x_seq.to(self.device)
                        
                        # LSTM
                        p_l, p_s = self.lstm_model(x_seq)
                        preds_l_list.append(p_l.cpu().numpy())
                        preds_s_list.append(p_s.cpu().numpy())
                        
                        # Gate
                        last_step_feats = x_seq[:, -1, :]
                        ctx = last_step_feats[:, ctx_indices]
                        g = self.gating_model(ctx)
                        gates_list.append(g.cpu().numpy())
                
                # Concatenate
                valid_l = np.concatenate(preds_l_list).flatten()
                valid_s = np.concatenate(preds_s_list).flatten()
                valid_g = np.concatenate(gates_list).flatten()
                
                # Fill the valid range [seq_len:]
                # (Dataset length is N - seq_len, shifted by seq_len)
                lstm_long[seq_len:] = valid_l
                lstm_short[seq_len:] = valid_s
                gate_vals[seq_len:] = valid_g
                
                # Fallback for start: Use LGB only (Gate = 1.0)
                gate_vals[:seq_len] = 1.0
            
            # 3. Combine
            df['pred_long'] = gate_vals * lgb_long + (1 - gate_vals) * lstm_long
            df['pred_short'] = gate_vals * lgb_short + (1 - gate_vals) * lstm_short
            
            # Debug/Vis columns
            df['gate_weight'] = gate_vals
            df['lstm_long'] = lstm_long
            
        else:
            df['pred_long'] = lgb_long
            df['pred_short'] = lgb_short
        
        # Direction Preds
        if self.dir_model_long:
            df['pred_dir_long'] = self.dir_model_long.predict(df[lgb_feats])
            df['pred_dir_short'] = self.dir_model_short.predict(df[lgb_feats])
        else:
            df['pred_dir_long'] = 1.0 
            df['pred_dir_short'] = 1.0
            
        return df

    def save_models(self):
        self.config.model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model_long, self.config.model_dir / "model_long.pkl")
        joblib.dump(self.model_short, self.config.model_dir / "model_short.pkl")
        
        if self.lstm_model:
            torch.save(self.lstm_model.state_dict(), self.config.model_dir / "lstm_model.pth")
            torch.save(self.gating_model.state_dict(), self.config.model_dir / "gating_model.pth")
            joblib.dump(self.scaler, self.config.model_dir / "scaler.pkl")
            
        if self.dir_model_long:
            joblib.dump(self.dir_model_long, self.config.model_dir / "dir_model_long.pkl")
            joblib.dump(self.dir_model_short, self.config.model_dir / "dir_model_short.pkl")
        joblib.dump(self.feature_cols, self.config.model_dir / "features.pkl")
        print("Models saved.")