"""
Data loader for raw trade data from Bybit.
Handles loading multiple CSV files and basic preprocessing.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
from config import DataConfig


def load_trades(config: DataConfig, verbose: bool = True, sample_rate: float = 1.0) -> pd.DataFrame:
    """
    Load all trade CSV files from the data directory.
    
    Args:
        config: DataConfig with paths and column names
        verbose: Print progress information
        sample_rate: Fraction of trades to keep (1.0 = all, 0.1 = 10%)
        
    Returns:
        DataFrame with all trades, sorted by timestamp
    """
    data_path = Path(config.data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    # Find all matching files
    files = sorted(data_path.glob(config.file_pattern))
    
    if not files:
        raise FileNotFoundError(f"No files matching '{config.file_pattern}' in {data_path}")
    
    if verbose:
        print(f"Found {len(files)} trade files:")
        for f in files:
            print(f"  - {f.name}")
    
    # Only load columns we need to save memory
    use_cols = [
        config.timestamp_col,
        config.price_col,
        config.size_col,
        config.side_col,
        config.tick_direction_col,
    ]
    
    # Load files one at a time, already sorted
    all_trades = []
    total_original = 0
    
    for file in files:
        # Load only needed columns with optimized dtypes
        df = pd.read_csv(
            file, 
            usecols=lambda c: c in use_cols or c == 'symbol',
            dtype={
                config.price_col: 'float32',
                config.size_col: 'float32',
            }
        )
        
        total_original += len(df)
        
        # Sample if needed
        if sample_rate < 1.0:
            df = df.sample(frac=sample_rate, random_state=42)
        
        # Sort this file's trades
        df = df.sort_values(config.timestamp_col)
        
        all_trades.append(df)
        
        if verbose:
            print(f"  Loaded {file.name}: {len(df):,} trades")
        
        # Clear memory
        import gc
        gc.collect()
    
    # Concatenate (files are already sorted, so merge-sort is efficient)
    if verbose:
        print("  Merging files...")
    
    trades = pd.concat(all_trades, ignore_index=True)
    del all_trades
    
    # Final sort (needed because files might overlap in time)
    trades = trades.sort_values(config.timestamp_col).reset_index(drop=True)

    # Optional: limit to the most recent N days of trades.
    lookback_days = getattr(config, 'lookback_days', None)
    if lookback_days is not None:
        try:
            lookback_days_int = int(lookback_days)
        except Exception:
            lookback_days_int = None

        if lookback_days_int is not None and lookback_days_int > 0 and len(trades) > 0:
            max_ts = float(trades[config.timestamp_col].max())
            cutoff_ts = max_ts - (lookback_days_int * 86400)
            pre_filter_len = len(trades)
            trades = trades[trades[config.timestamp_col] >= cutoff_ts].reset_index(drop=True)
            if verbose:
                print(
                    f"Applied lookback: last {lookback_days_int} days -> "
                    f"{len(trades):,} trades (from {pre_filter_len:,})"
                )
    
    if verbose:
        print(f"\nTotal trades: {len(trades):,}" + 
              (f" (sampled from {total_original:,})" if sample_rate < 1.0 else ""))
        print(f"Date range: {pd.to_datetime(trades[config.timestamp_col].min(), unit='s')} "
              f"to {pd.to_datetime(trades[config.timestamp_col].max(), unit='s')}")
        
        # Memory usage
        mem_mb = trades.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"Memory usage: {mem_mb:.1f} MB")
    
    return trades


def preprocess_trades(trades: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    """
    Preprocess raw trade data.
    
    Args:
        trades: Raw trade DataFrame
        config: DataConfig with column names
        
    Returns:
        Preprocessed DataFrame
    """
    df = trades.copy()
    
    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df[config.timestamp_col], unit='s')
    
    # Encode side as numeric (-1 for Sell, +1 for Buy)
    df['side_num'] = df[config.side_col].map({'Buy': 1, 'Sell': -1})
    
    # Encode tick direction
    tick_map = {
        'PlusTick': 1,
        'ZeroPlusTick': 0.5,
        'MinusTick': -1,
        'ZeroMinusTick': -0.5
    }
    df['tick_dir_num'] = df[config.tick_direction_col].map(tick_map).fillna(0)
    
    # Calculate trade value
    df['value'] = df[config.price_col] * df[config.size_col]
    
    # Calculate signed volume (positive for buys, negative for sells)
    df['signed_size'] = df[config.size_col] * df['side_num']
    df['signed_value'] = df['value'] * df['side_num']
    
    return df


def aggregate_to_bars(
    trades: pd.DataFrame, 
    timeframe_seconds: int,
    config: DataConfig
) -> pd.DataFrame:
    """
    Aggregate raw trades into OHLCV bars.
    
    Args:
        trades: Preprocessed trade DataFrame
        timeframe_seconds: Bar size in seconds
        config: DataConfig with column names
        
    Returns:
        DataFrame with OHLCV bars and additional metrics
    """
    df = trades.copy()
    
    # Create time buckets
    df['bar_time'] = (df[config.timestamp_col] // timeframe_seconds) * timeframe_seconds
    df['bar_datetime'] = pd.to_datetime(df['bar_time'], unit='s')
    
    # Aggregate
    bars = df.groupby('bar_time').agg({
        config.price_col: ['first', 'max', 'min', 'last'],
        config.size_col: 'sum',
        'value': 'sum',
        'side_num': 'sum',  # Net buy/sell count
        'signed_size': 'sum',  # Net volume
        'signed_value': 'sum',  # Net value
        'tick_dir_num': 'mean',  # Average tick direction
        config.timestamp_col: 'count',  # Trade count
    })
    
    # Flatten column names
    bars.columns = [
        'open', 'high', 'low', 'close',
        'volume', 'value',
        'net_side', 'net_volume', 'net_value',
        'avg_tick_dir', 'trade_count'
    ]
    
    bars = bars.reset_index()
    bars['datetime'] = pd.to_datetime(bars['bar_time'], unit='s')
    
    # Calculate additional metrics
    bars['buy_volume'] = (bars['volume'] + bars['net_volume']) / 2
    bars['sell_volume'] = (bars['volume'] - bars['net_volume']) / 2
    bars['buy_sell_imbalance'] = bars['net_volume'] / bars['volume'].replace(0, np.nan)
    
    # VWAP
    bars['vwap'] = bars['value'] / bars['volume'].replace(0, np.nan)
    
    # Trade intensity (trades per second)
    bars['trade_intensity'] = bars['trade_count'] / timeframe_seconds
    
    # Average trade size
    bars['avg_trade_size'] = bars['volume'] / bars['trade_count'].replace(0, np.nan)
    
    # Fill NaN values
    bars = bars.fillna(method='ffill')
    
    return bars


def create_multi_timeframe_bars(
    trades: pd.DataFrame,
    timeframes_seconds: List[int],
    timeframe_names: List[str],
    config: DataConfig
) -> dict:
    """
    Create bars for multiple timeframes.
    
    Args:
        trades: Preprocessed trade DataFrame
        timeframes_seconds: List of timeframe sizes in seconds
        timeframe_names: Names for each timeframe
        config: DataConfig
        
    Returns:
        Dictionary mapping timeframe name to bar DataFrame
    """
    bars_dict = {}
    
    for tf_seconds, tf_name in zip(timeframes_seconds, timeframe_names):
        print(f"  Creating {tf_name} bars...")
        bars = aggregate_to_bars(trades, tf_seconds, config)
        bars_dict[tf_name] = bars
        print(f"    -> {len(bars):,} bars")
    
    return bars_dict


if __name__ == "__main__":
    # Test the data loader
    from config import DEFAULT_CONFIG
    
    config = DEFAULT_CONFIG.data
    
    # Adjust path for testing
    config.data_dir = Path("./data")
    
    print("Loading trades...")
    trades = load_trades(config)
    
    print("\nPreprocessing...")
    trades = preprocess_trades(trades, config)
    
    print("\nCreating multi-timeframe bars...")
    bars = create_multi_timeframe_bars(
        trades,
        DEFAULT_CONFIG.features.timeframes,
        DEFAULT_CONFIG.features.timeframe_names,
        config
    )
    
    print("\nSample 5m bars:")
    print(bars['5m'].head(10))
