"""
LightGBM DART Manager.
"""
import lightgbm as lgb
import pandas as pd
import joblib
from config import ModelConfig

class ModelManager:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model_long = None
        self.model_short = None
        self.feature_cols = []

    def train(self, df: pd.DataFrame, feature_cols: list):
        self.feature_cols = feature_cols
        
        # Triple Split Logic
        n = len(df)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))
        
        X = df[feature_cols]
        y_long = df['target_long']
        y_short = df['target_short']
        
        X_train, y_long_train, y_short_train = X.iloc[:train_end], y_long.iloc[:train_end], y_short.iloc[:train_end]
        X_val, y_long_val, y_short_val = X.iloc[train_end:val_end], y_long.iloc[train_end:val_end], y_short.iloc[train_end:val_end]
        
        print(f"Train: {len(X_train)} | Val: {len(X_val)}")
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'dart', # DART for robustness
            'learning_rate': self.config.learning_rate,
            'num_leaves': self.config.num_leaves,
            'max_depth': self.config.max_depth,
            'min_child_samples': self.config.min_child_samples,
            'subsample': self.config.subsample,
            'colsample_bytree': self.config.colsample_bytree,
            'rate_drop': self.config.rate_drop,
            'skip_drop': 0.5,
            'n_jobs': -1,
            'verbosity': -1
        }
        
        print("Training Long (DART)...")
        self.model_long = lgb.train(
            params,
            lgb.Dataset(X_train, label=y_long_train),
            num_boost_round=self.config.n_estimators,
            valid_sets=[lgb.Dataset(X_val, label=y_long_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=self.config.early_stopping_rounds), lgb.log_evaluation(0)]
        )
        
        print("Training Short (DART)...")
        self.model_short = lgb.train(
            params,
            lgb.Dataset(X_train, label=y_short_train),
            num_boost_round=self.config.n_estimators,
            valid_sets=[lgb.Dataset(X_val, label=y_short_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=self.config.early_stopping_rounds), lgb.log_evaluation(0)]
        )

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.model_long: raise Exception("Not trained")
        X = df[self.feature_cols]
        df['pred_long'] = self.model_long.predict(X)
        df['pred_short'] = self.model_short.predict(X)
        return df

    def save_models(self):
        self.config.model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model_long, self.config.model_dir / "model_long.pkl")
        joblib.dump(self.model_short, self.config.model_dir / "model_short.pkl")
        joblib.dump(self.feature_cols, self.config.model_dir / "features.pkl")
        print("Models saved.")
