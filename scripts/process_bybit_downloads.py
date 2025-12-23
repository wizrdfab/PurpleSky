#!/usr/bin/env python3
"""
Process downloaded Bybit CSV files and merge them

Usage:
    python process_bybit_downloads.py --input data/bybit_downloads --output data/BTCUSDT_ticks.csv
"""

import argparse
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_bybit_csvs(input_dir: str, output_file: str, symbol_filter: str = None):
    """
    Process and merge downloaded Bybit CSV files

    Args:
        input_dir: Directory containing downloaded CSV files
        output_file: Output file path for merged data
        symbol_filter: Only process files for this symbol (optional)
    """
    input_path = Path(input_dir)
    csv_files = sorted(input_path.glob("*.csv"))

    if not csv_files:
        logger.error(f"No CSV files found in {input_path}")
        return

    logger.info(f"Found {len(csv_files)} CSV files")

    all_data = []
    skipped = 0

    for csv_file in csv_files:
        try:
            # Check if file matches symbol filter
            if symbol_filter and symbol_filter not in csv_file.name:
                skipped += 1
                continue

            logger.info(f"Processing {csv_file.name}...")
            df = pd.read_csv(csv_file)

            # Check if file is empty
            if df.empty:
                logger.warning(f"  Skipped (empty): {csv_file.name}")
                skipped += 1
                continue

            # Convert timestamp to datetime if needed
            if 'timestamp' in df.columns:
                # Bybit timestamps are in milliseconds
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')

            all_data.append(df)
            logger.info(f"  Loaded {len(df):,} rows")

        except Exception as e:
            logger.error(f"  Error processing {csv_file.name}: {e}")
            skipped += 1

    if not all_data:
        logger.error("No valid data found")
        return

    # Combine all dataframes
    logger.info("Merging all data...")
    combined = pd.concat(all_data, ignore_index=True)

    # Sort by timestamp
    if 'timestamp' in combined.columns:
        combined = combined.sort_values('timestamp')

    # Remove duplicates based on trade ID
    before_dedup = len(combined)
    if 'trdMatchID' in combined.columns:
        combined = combined.drop_duplicates(subset=['trdMatchID'])
        logger.info(f"Removed {before_dedup - len(combined):,} duplicate trades")

    # Save to output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing Complete!")
    logger.info(f"  Input files processed: {len(all_data)}")
    logger.info(f"  Input files skipped: {skipped}")
    logger.info(f"  Total rows: {len(combined):,}")
    logger.info(f"  Output file: {output_path}")
    logger.info(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    if 'timestamp' in combined.columns:
        logger.info(f"  Date range: {combined['timestamp'].min()} to {combined['timestamp'].max()}")

    logger.info(f"{'='*60}\n")

    # Show sample
    logger.info("Sample data (first 5 rows):")
    print(combined.head())


def main():
    parser = argparse.ArgumentParser(
        description="Process downloaded Bybit CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all CSVs in download directory
    python process_bybit_downloads.py --input data/bybit_downloads --output data/merged_ticks.csv

    # Process only BTCUSDT files
    python process_bybit_downloads.py --input data/bybit_downloads --output data/BTCUSDT_ticks.csv --symbol BTCUSDT

    # Process and save to specific location
    python process_bybit_downloads.py -i ./downloads -o ./data/historical/BTC_1month.csv -s BTCUSDT
        """
    )

    parser.add_argument("-i", "--input", required=True, help="Input directory with CSV files")
    parser.add_argument("-o", "--output", required=True, help="Output CSV file path")
    parser.add_argument("-s", "--symbol", help="Filter by symbol (e.g., BTCUSDT)")

    args = parser.parse_args()

    process_bybit_csvs(
        input_dir=args.input,
        output_file=args.output,
        symbol_filter=args.symbol
    )


if __name__ == "__main__":
    main()
