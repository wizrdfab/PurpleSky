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
        # Execution Models (Model B)
        self.model_long = None
        self.model_short = None
        
        # Direction Models (Model A - Meta Labeling)
        self.dir_model_long = None
        self.dir_model_short = None
        
        self.feature_cols = []

    def train(self, df: pd.DataFrame, feature_cols: list):
        self.feature_cols = feature_cols
        
        # Triple Split Logic
        n = len(df)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))
        
        X = df[feature_cols]
        # Execution Targets
        y_long = df['target_long']
        y_short = df['target_short']
        
        # Direction Targets
        y_dir_long = df.get('target_dir_long', pd.Series(0, index=df.index))
        y_dir_short = df.get('target_dir_short', pd.Series(0, index=df.index))
        
        X_train = X.iloc[:train_end]
        X_val = X.iloc[train_end:val_end]
        
        print(f"Train: {len(X_train)} | Val: {len(X_val)}")
        
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
                params, lgb.Dataset(X_train, label=y_dir_long.iloc[:train_end]),
                num_boost_round=self.config.n_estimators // 2, # Lighter model
                valid_sets=[lgb.Dataset(X_val, label=y_dir_long.iloc[train_end:val_end])],
                callbacks=[lgb.early_stopping(stopping_rounds=self.config.early_stopping_rounds), lgb.log_evaluation(0)]
            )
            self.dir_model_short = lgb.train(
                params, lgb.Dataset(X_train, label=y_dir_short.iloc[:train_end]),
                num_boost_round=self.config.n_estimators // 2,
                valid_sets=[lgb.Dataset(X_val, label=y_dir_short.iloc[train_end:val_end])],
                callbacks=[lgb.early_stopping(stopping_rounds=self.config.early_stopping_rounds), lgb.log_evaluation(0)]
            )
        
        # --- 2. Train Execution Models (Model B) ---
        print("Training Execution Models (Model B)...")
        self.model_long = lgb.train(
            params,
            lgb.Dataset(X_train, label=y_long.iloc[:train_end]),
            num_boost_round=self.config.n_estimators,
            valid_sets=[lgb.Dataset(X_val, label=y_long.iloc[train_end:val_end])],
            callbacks=[lgb.early_stopping(stopping_rounds=self.config.early_stopping_rounds), lgb.log_evaluation(0)]
        )
        
        self.model_short = lgb.train(
            params,
            lgb.Dataset(X_train, label=y_short.iloc[:train_end]),
            num_boost_round=self.config.n_estimators,
            valid_sets=[lgb.Dataset(X_val, label=y_short.iloc[train_end:val_end])],
            callbacks=[lgb.early_stopping(stopping_rounds=self.config.early_stopping_rounds), lgb.log_evaluation(0)]
        )

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.model_long: raise Exception("Not trained")
        X = df[self.feature_cols]
        
        # Execution Preds
        df['pred_long'] = self.model_long.predict(X)
        df['pred_short'] = self.model_short.predict(X)
        
        # Direction Preds
        if self.dir_model_long:
            df['pred_dir_long'] = self.dir_model_long.predict(X)
            df['pred_dir_short'] = self.dir_model_short.predict(X)
        else:
            df['pred_dir_long'] = 1.0 # Pass-through if no meta-labeling
            df['pred_dir_short'] = 1.0
            
        return df

    def save_models(self):
        self.config.model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model_long, self.config.model_dir / "model_long.pkl")
        joblib.dump(self.model_short, self.config.model_dir / "model_short.pkl")
        if self.dir_model_long:
            joblib.dump(self.dir_model_long, self.config.model_dir / "dir_model_long.pkl")
            joblib.dump(self.dir_model_short, self.config.model_dir / "dir_model_short.pkl")
        joblib.dump(self.feature_cols, self.config.model_dir / "features.pkl")
        print("Models saved.")
