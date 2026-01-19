"""
Copyright (C) 2026 wizrdfab

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

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

@dataclass
class DataConfig:
    data_dir: Path = Path("data/RAVEUSDT")
    symbol: str = "RAVEUSDT"
    trade_subdir: str = "Trade"
    orderbook_subdir: str = "Orderbook"
    trade_pattern: str = "*.csv"
    orderbook_pattern: str = "*.data"
    timestamp_col: str = "timestamp"
    price_col: str = "price"
    size_col: str = "size"
    side_col: str = "side"
    ob_levels: int = 50

@dataclass
class StrategyConfig:
    base_limit_offset_atr: float = 0.88
    
    time_limit_bars: int = 36        
    max_holding_bars: int = 144       
    stop_loss_atr: float = 3.9       
    take_profit_atr: float = 0.8     
    maker_fee: float = 0.0002
    taker_fee: float = 0.0006
    risk_per_trade: float = 0.04    
    max_positions: int = 3

@dataclass
class LiveSettings:
    """Production Safety Settings"""
    max_daily_drawdown_pct: float = 0.03  
    max_spread_pct: float = 0.002         
    max_api_errors: int = 5               
    poll_interval_sec: int = 1           
    reconcile_interval_sec: int = 60      
    dry_run: bool = True #Momentary                 

@dataclass
class FeatureConfig:
    base_timeframe: str = "5m"      
    ema_periods: List[int] = field(default_factory=lambda: [9, 21, 50])
    atr_period: int = 14
    rsi_period: int = 14
    micro_windows: List[int] = field(default_factory=lambda: [1, 4]) 

@dataclass
class ModelConfig:
    model_type: str = "lightgbm_dart"
    model_dir: Path = Path("models_v3")
    n_estimators: int = 1500
    learning_rate: float = 0.05
    max_depth: int = 5           
    num_leaves: int = 32         
    min_child_samples: int = 20  # Relaxed from 50
    subsample: float = 0.7
    colsample_bytree: float = 0.2
    rate_drop: float = 0.1       
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    model_threshold: float = 0.60
    early_stopping_rounds: int = 50
    extra_trees: bool = False
    
    # --- Council & Meta-Labeling ---
    ensemble_size: int = 5       # Size of the Council
    voting_threshold: int = 1    # Min votes to execute
    use_meta_labeling: bool = True
    direction_threshold: float = 0.5 # Dynamic Threshold for Direction Model
    aggressive_threshold: float = 0.8 # Threshold for Market Order Override

    # --- LSTM Hybrid Ensemble ---
    use_lstm_ensemble: bool = True
    lstm_hidden_size: int = 32
    lstm_layers: int = 1
    sequence_length: int = 60
    lstm_dropout: float = 0.8
    lstm_epochs: int = 50
    lstm_batch_size: int = 64
    lstm_weight_decay: float = 1e-5
    lstm_patience: int = 10
    lstm_learning_rate: float = 0.0001

@dataclass
class GlobalConfig:
    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    live: LiveSettings = field(default_factory=LiveSettings)
    seed: int = 42

CONF = GlobalConfig()
