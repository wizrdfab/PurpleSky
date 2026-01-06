import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

DATA_ROOT = Path("data")
TIMEFRAME = "1h"

def load_symbol_prices(symbol_path: Path):
    """
    Loads trade data for a symbol and returns a Series of 1H closing prices.
    Reads all Trade CSVs found.
    """
    symbol = symbol_path.name
    trade_path = symbol_path / "Trade"
    files = sorted(glob.glob(str(trade_path / "*.csv")))
    
    if not files:
        return None
    
    print(f"Loading {symbol} ({len(files)} files)...")
    
    dfs = []
    # Limit to last 3 files for speed if many exist, correlation is usually stable
    for f in files[-5:]:
        try:
            # Minimal read: timestamp, price
            df = pd.read_csv(f, usecols=['timestamp', 'price'])
            # Coerce errors
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df.dropna(inplace=True)
            
            # Detect unit
            if not df.empty and df['timestamp'].iloc[0] > 3000000000:
                unit = 'ms'
            else:
                unit = 's'
                
            df['datetime'] = pd.to_datetime(df['timestamp'], unit=unit)
            df.set_index('datetime', inplace=True)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not dfs: return None
    
    full_df = pd.concat(dfs).sort_index()
    # Resample to 1H close
    resampled = full_df['price'].resample(TIMEFRAME).last().ffill()
    return resampled

def main():
    print(f"--- Correlation Matrix Tool (Timeframe: {TIMEFRAME}) ---")
    
    symbols = [d for d in DATA_ROOT.iterdir() if d.is_dir()]
    price_series = {}
    
    for s in symbols:
        try:
            series = load_symbol_prices(s)
            if series is not None and not series.empty:
                price_series[s.name] = series
        except Exception as e:
            print(f"Skipping {s.name}: {e}")
            
    if len(price_series) < 2:
        print("Not enough data found to calculate correlation (need at least 2 symbols).")
        return

    # Create DataFrame
    df = pd.DataFrame(price_series)
    
    # Calculate Returns (Price levels are not stationary, returns are)
    returns = df.pct_change().dropna()
    
    # Correlation
    corr_matrix = returns.corr()
    
    print("\n" + "="*50)
    print("CORRELATION MATRIX (Returns)")
    print("="*50)
    print(corr_matrix.round(2))
    print("\n" + "="*50)
    
    # Interpretation
    print("INTERPRETATION GUIDE:")
    print("  > 0.80  : HIGHLY CORRELATED. (Danger: Do not trade both)")
    print("  0.5-0.8 : Moderately Correlated. (Acceptable)")
    print("  < 0.50  : Uncorrelated/Diversified. (Ideal)")
    print("  < 0.00  : Negative Correlation. (Hedging behavior)")
    print("="*50)

    # Pairs to avoid
    high_corr_pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            val = corr_matrix.iloc[i, j]
            if val > 0.8:
                high_corr_pairs.append((cols[i], cols[j], val))
                
    if high_corr_pairs:
        print("\n⚠️  WARNING: REDUNDANT PAIRS DETECTED ⚠️")
        for s1, s2, v in high_corr_pairs:
            print(f"  - {s1} <-> {s2}: {v:.2f}")
        print("Recommendation: Pick only ONE from each pair to trade.")

if __name__ == "__main__":
    main()
