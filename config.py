"""
Configuration for TrendFollower ML system.
Adjust these parameters based on your setup and experimentation.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class DataConfig:
    """Data loading configuration"""
    data_dir: Path = Path("./data")  # Directory containing CSV files
    file_pattern: str = "*.csv"       # Pattern to match trade files
    lookback_days: Optional[int] = None  # If set, only use the most recent N days of trades
    
    # Column mapping (adjust if your CSV has different names)
    timestamp_col: str = "timestamp"
    price_col: str = "price"
    size_col: str = "size"
    side_col: str = "side"
    tick_direction_col: str = "tickDirection"


@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    # Timeframes to generate (in seconds)
    timeframes: List[int] = field(default_factory=lambda: [
        60,      # 1m
        300,     # 5m
        900,     # 15m
        3600,    # 1h
        14400,   # 4h
    ])
    timeframe_names: List[str] = field(default_factory=lambda: [
        '1m', '5m', '15m', '1h', '4h'
    ])
    
    # EMA periods
    ema_periods: List[int] = field(default_factory=lambda: [10, 21, 50, 100])
    
    # Other indicator periods
    rsi_period: int = 14
    adx_period: int = 14
    atr_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    
    # Volume features
    volume_ma_period: int = 20
    
    # Lookback for structure detection (in bars)
    swing_lookback: int = 10


@dataclass
class LabelConfig:
    """Label generation configuration"""
    # Forward windows for labeling (in bars of the base timeframe)
    trend_forward_window: int = 20      # How far ahead to look for trend
    entry_forward_window: int = 10      # How far ahead to look for entry success
    
    # Trend classification thresholds (in ATR units)
    trend_up_threshold: float = 2.0     # Must move up this much
    trend_down_threshold: float = 2.0   # Must move down this much
    max_adverse_for_trend: float = 1.0  # Max drawdown to still count as trend
    
    # Entry success thresholds
    target_rr: float = 1.5              # Target reward:risk ratio
    stop_atr_multiple: float = 1.0      # Stop loss in ATR units
    
    # Pullback detection
    pullback_ema: int = 21              # EMA to detect pullbacks to
    pullback_threshold: float = 0.5     # Price within X ATR of EMA


@dataclass
class ModelConfig:
    """ML model configuration"""
    # LightGBM parameters
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    num_leaves: int = 31
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    min_child_samples: int = 50

    #Test
    # --- Regularization (The "Penalties") ---
    lambdaa_ele1: float = 0.5          # <--- Added this (testing regularization, may need to be changed)
    lambdaa_ele2: float = 0.5          # <--- Added this (testing regularization, may need to be changed)
    min_gain_to_split: float = 0.1  # <--- Added this (testing regularization, may need to be changed)
    
    # Training
    train_ratio: float = 0.7            # 70% train
    val_ratio: float = 0.15             # 15% validation
    test_ratio: float = 0.15            # 15% test
    
    # Model paths
    model_dir: Path = Path("./models")


@dataclass
class TrendFollowerConfig:
    """Main configuration combining all sub-configs"""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # Base timeframe for training (index into timeframes list)
    base_timeframe_idx: int = 1  # 5m by default
    
    # Random seed for reproducibility
    seed: int = 42


# Default configuration instance
DEFAULT_CONFIG = TrendFollowerConfig()
