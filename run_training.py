#!/usr/bin/env python3
"""
Main entry point for TrendFollower training and backtesting.

Usage:
    # Train models
    python run_training.py --data-dir /path/to/trade/csvs
    
    # Train with custom settings
    python run_training.py --data-dir ./data --base-tf 2 --model-dir ./my_models
    
    # Run backtest after training
    python run_training.py --data-dir ./data --backtest
"""
import argparse
from pathlib import Path
import sys

from config import TrendFollowerConfig, DEFAULT_CONFIG


def main():
    parser = argparse.ArgumentParser(
        description='Train TrendFollower ML models on trade data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with default settings
    python run_training.py --data-dir ./data
    
    # Train on 15m timeframe
    python run_training.py --data-dir ./data --base-tf 2
    
    # Train and backtest
    python run_training.py --data-dir ./data --backtest
    
Timeframe indices:
    0 = 1m
    1 = 5m (default)
    2 = 15m
    3 = 1h
    4 = 4h
        """
    )
    
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing trade CSV files')
    parser.add_argument('--model-dir', type=str, default='./models',
                       help='Directory to save trained models (default: ./models)')
    parser.add_argument('--base-tf', type=int, default=1,
                       help='Base timeframe index: 0=1m, 1=5m, 2=15m, 3=1h, 4=4h (default: 1)')
    parser.add_argument('--sample-rate', type=float, default=1.0,
                       help='Fraction of trades to use (0.1 = 10%%, 1.0 = all). Use lower values if running out of memory.')
    parser.add_argument('--backtest', action='store_true',
                       help='Run backtest after training')
    parser.add_argument('--backtest-only', action='store_true',
                       help='Only run backtest (requires existing models)')
    parser.add_argument('--no-diagnostics', action='store_true',
                       help='Disable diagnostic logging (enabled by default)')
    parser.add_argument('--min-quality', type=str, default='B', choices=['A', 'B', 'C'],
                       help='Minimum signal quality for backtest trades (A=best, C=all). Default: B')
    
    args = parser.parse_args()
    
    # Validate data directory
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"Error: Data directory not found: {data_path}")
        sys.exit(1)
    
    # Check for CSV files
    csv_files = list(data_path.glob('*.csv'))
    if not csv_files:
        print(f"Error: No CSV files found in {data_path}")
        sys.exit(1)
    
    print(f"Found {len(csv_files)} CSV files in {data_path}")
    
    # Create config
    config = TrendFollowerConfig()
    config.data.data_dir = data_path
    config.model.model_dir = Path(args.model_dir)
    config.base_timeframe_idx = args.base_tf
    config.sample_rate = args.sample_rate  # Add sample rate
    config.enable_diagnostics = not args.no_diagnostics  # Enabled by default
    config.min_quality = args.min_quality  # Minimum signal quality for backtest
    
    print(f"\nConfiguration:")
    print(f"  Data directory:   {config.data.data_dir}")
    print(f"  Model directory:  {config.model.model_dir}")
    print(f"  Base timeframe:   {config.features.timeframe_names[config.base_timeframe_idx]}")
    if args.sample_rate < 1.0:
        print(f"  Sample rate:      {args.sample_rate:.0%} of trades")
    print(f"  Diagnostics:      {'Enabled' if config.enable_diagnostics else 'Disabled'}")
    print(f"  Min quality:      {config.min_quality}")
    
    if args.backtest_only:
        # Only run backtest
        print("\nRunning backtest only...")
        run_backtest_only(config)
    else:
        # Run training
        from trainer import run_training_pipeline
        training_output = run_training_pipeline(config, enable_diagnostics=config.enable_diagnostics)
        
        # Handle both old and new return format
        if isinstance(training_output, tuple):
            results, models, test_df, X_test, feature_cols = training_output
        else:
            results = training_output
            models = None
        
        if args.backtest:
            print("\nRunning backtest on test data...")
            run_backtest_after_training(config, results)
        
        # Print diagnostic log location if available
        if 'diagnostic_log' in results:
            print(f"\nDiagnostic log saved to: {results['diagnostic_log']}")
    
    print("\nDone!")


def run_backtest_only(config: TrendFollowerConfig):
    """Run backtest using existing models"""
    from models import TrendFollowerModels
    from data_loader import load_trades, preprocess_trades, create_multi_timeframe_bars
    from feature_engine import calculate_multi_timeframe_features, get_feature_columns
    from labels import create_training_dataset
    from backtest import run_backtest
    
    # Load models
    model_path = config.model.model_dir
    if not model_path.exists():
        print(f"Error: Model directory not found: {model_path}")
        sys.exit(1)
    
    models = TrendFollowerModels(config.model)
    models.load_all(model_path)
    
    # Load and process data
    print("Loading data...")
    trades = load_trades(config.data)
    trades = preprocess_trades(trades, config.data)
    
    print("Creating bars...")
    bars_dict = create_multi_timeframe_bars(
        trades,
        config.features.timeframes,
        config.features.timeframe_names,
        config.data
    )
    
    base_tf = config.features.timeframe_names[config.base_timeframe_idx]
    
    print("Calculating features...")
    featured_data = calculate_multi_timeframe_features(
        bars_dict,
        base_tf,
        config.features
    )
    
    print("Generating labels...")
    labeled_data, feature_cols = create_training_dataset(
        featured_data,
        config.labels,
        config.features,
        base_tf
    )
    
    # Use last 20% as test
    test_start = int(len(labeled_data) * 0.8)
    test_data = labeled_data.iloc[test_start:]
    
    print(f"\nBacktesting on {len(test_data):,} bars...")
    run_backtest(test_data, models, feature_cols, config)


def run_backtest_after_training(config: TrendFollowerConfig, training_results: dict):
    """Run backtest after training completes"""
    from models import TrendFollowerModels
    from data_loader import load_trades, preprocess_trades, create_multi_timeframe_bars
    from feature_engine import calculate_multi_timeframe_features, get_feature_columns
    from labels import create_training_dataset
    from backtest import run_backtest
    from diagnostic_logger import DiagnosticLogger
    
    # Get or create diagnostic logger
    diag = None
    if getattr(config, 'enable_diagnostics', True):
        diag = DiagnosticLogger('./logs')
        diag.log_section("Backtest After Training")
    
    # Load the just-trained models
    models = TrendFollowerModels(config.model)
    models.load_all(config.model.model_dir)
    
    # Reload data (could optimize by caching)
    trades = load_trades(config.data, verbose=False)
    trades = preprocess_trades(trades, config.data)
    
    bars_dict = create_multi_timeframe_bars(
        trades,
        config.features.timeframes,
        config.features.timeframe_names,
        config.data
    )
    
    base_tf = config.features.timeframe_names[config.base_timeframe_idx]
    
    featured_data = calculate_multi_timeframe_features(
        bars_dict,
        base_tf,
        config.features
    )
    
    labeled_data, feature_cols = create_training_dataset(
        featured_data,
        config.labels,
        config.features,
        base_tf
    )
    
    # Use last portion as test
    test_start = int(len(labeled_data) * (config.model.train_ratio + config.model.val_ratio))
    test_data = labeled_data.iloc[test_start:]
    
    if diag:
        diag.log_metric("backtest_test_samples", len(test_data))
        diag.log_metric("backtest_test_start_idx", test_start)
    
    print(f"\nBacktesting on {len(test_data):,} bars (test set)...")
    
    # Get min_quality from config (default to 'B')
    min_quality = getattr(config, 'min_quality', 'B')
    
    run_backtest(test_data, models, feature_cols, config, 
                 diagnostic_logger=diag, min_quality=min_quality)
    
    if diag:
        diag.save()


if __name__ == "__main__":
    main()
