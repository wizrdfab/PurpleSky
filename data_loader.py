"""
Advanced Data Loader with Stream Processing. Aggregates Trades AND Orderbook snapshots into high-fidelity bars.
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
import pandas as pd
import polars as pl
import numpy as np
import glob
import json
import os
from pathlib import Path
from joblib import Parallel, delayed
from config import DataConfig

def _process_single_ob_file(f, timeframe, data_dir, ob_levels, rule):
    """
    Standalone function to process a single Orderbook file.
    Must be top-level for joblib pickling.
    """
    source_file = Path(f)
    # Unique cache name including timeframe (so 5m and 1h don't conflict)
    cache_name = f"{source_file.stem}_{timeframe}.pkl"
    cache_dir = data_dir / "Orderbook_reconstruction"
    cache_path = cache_dir / cache_name
    
    # --- CACHE CHECK ---
    try:
        if cache_path.exists():
            source_mtime = source_file.stat().st_mtime
            cache_mtime = cache_path.stat().st_mtime
            if cache_mtime > source_mtime:
                # print(f"  [CACHE] Loading {cache_name}...") 
                # Reduced logging for parallel execution
                return pd.read_pickle(cache_path)
    except Exception as e:
        print(f"  [CACHE] Check failed for {f}: {e}")

    # --- RECONSTRUCTION (Cache Miss) ---
    try:
        buffer = []
        # Local Orderbook State: Dict[Price(str), Size(float)]
        # Using string keys for precision/matching, floats for values
        local_bids = {}
        local_asks = {}
        
        line_counter = 0
        
        with open(f, 'r') as file:
            for line in file:
                line_counter += 1
                try:
                    raw = json.loads(line)
                    msg_type = raw.get('type') # 'snapshot' or 'delta'
                    data = raw.get('data', {})
                    if not data: continue
                    
                    ts = raw.get('ts')
                    bids = data.get('b', [])
                    asks = data.get('a', [])
                    
                    # 1. Update Local State
                    if msg_type == 'snapshot':
                        local_bids = {p: float(s) for p, s in bids}
                        local_asks = {p: float(s) for p, s in asks}
                    else:
                        # Process Deltas
                        for p, s in bids:
                            size = float(s)
                            if size == 0:
                                local_bids.pop(p, None)
                            else:
                                local_bids[p] = size
                                
                        for p, s in asks:
                            size = float(s)
                            if size == 0:
                                local_asks.pop(p, None)
                            else:
                                local_asks[p] = size
                    
                    # 2. Sample State (e.g., every 50 updates to save RAM, or ~200ms)
                    # We MUST sample from the *reconstructed* state, not the raw line
                    if line_counter % 50 != 0: continue
                    
                    if not local_bids or not local_asks: continue
                    
                    # 3. Sort & Slice (Reconstruct the L2 View)
                    # Bids: Descending Price
                    sorted_bids = sorted(local_bids.items(), key=lambda x: float(x[0]), reverse=True)
                    # Asks: Ascending Price
                    sorted_asks = sorted(local_asks.items(), key=lambda x: float(x[0]))
                    
                    # Slice to configured depth
                    bids_slice = sorted_bids[:ob_levels]
                    asks_slice = sorted_asks[:ob_levels]
                    
                    if not bids_slice or not asks_slice: continue

                    # --- Calculate Features on RECONSTRUCTED Book ---
                    
                    bb_p = float(bids_slice[0][0])
                    bb_s = bids_slice[0][1]
                    ba_p = float(asks_slice[0][0])
                    ba_s = asks_slice[0][1]
                    
                    # Spread & Micro
                    spread = ba_p - bb_p
                    mid_price = (ba_p + bb_p) / 2
                    micro_price = (ba_p * bb_s + bb_p * ba_s) / (bb_s + ba_s + 1e-9)
                    micro_deviation = (micro_price - mid_price)
                    
                    # Depth
                    bid_depth = sum([s for _, s in bids_slice])
                    ask_depth = sum([s for _, s in asks_slice])
                    imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-9)
                    
                    # Gradient (Slope)
                    bid_slope = 0.0
                    if bid_depth > 0 and len(bids_slice) > 1:
                        deepest_bid = float(bids_slice[-1][0])
                        bid_slope = (bb_p - deepest_bid) / bid_depth
                        
                    ask_slope = 0.0
                    if ask_depth > 0 and len(asks_slice) > 1:
                        deepest_ask = float(asks_slice[-1][0])
                        ask_slope = (deepest_ask - ba_p) / ask_depth
                        
                    # NEW: Wall Integrity (Intention)
                    # Ratio of Best_Bid_Size to Total_Bid_Depth
                    # High = Concentrated Intent (Front-loaded)
                    # Low = Scattered/Layered
                    bid_integrity = bb_s / bid_depth if bid_depth > 0 else 0
                    ask_integrity = ba_s / ask_depth if ask_depth > 0 else 0
                    
                    buffer.append({
                        'timestamp': ts,
                        'ob_spread': spread,
                        'ob_micro_dev': micro_deviation,
                        'ob_imbalance': imbalance,
                        'ob_bid_depth': bid_depth,
                        'ob_ask_depth': ask_depth,
                        'ob_bid_slope': bid_slope,
                        'ob_ask_slope': ask_slope,
                        'ob_bid_integrity': bid_integrity,
                        'ob_ask_integrity': ask_integrity
                    })
                    
                except Exception as e:
                    continue
                    
        if not buffer: return None
        
        df_chunk = pd.DataFrame(buffer)
        df_chunk['datetime'] = pd.to_datetime(df_chunk['timestamp'], unit='ms')
        df_chunk.set_index('datetime', inplace=True)
        
        # Aggregation
        chunk_agg = df_chunk.resample(rule).agg({
            'ob_spread': 'mean',
            'ob_micro_dev': ['mean', 'std', 'last'],
            'ob_imbalance': ['mean', 'last'],
            'ob_bid_depth': 'mean',
            'ob_ask_depth': 'mean',
            'ob_bid_slope': 'mean',
            'ob_ask_slope': 'mean',
            'ob_bid_integrity': 'mean',
            'ob_ask_integrity': 'mean'
        })
        
        chunk_agg.columns = ['_'.join(col).strip() for col in chunk_agg.columns.values]
        
        # --- SAVE TO CACHE ---
        try:
            chunk_agg.to_pickle(cache_path)
            print(f"  [SAVE] Reconstructed & Cached {cache_name}")
        except Exception as e:
            print(f"  [WARN] Failed to cache {cache_name}: {e}")
        
        return chunk_agg
        
    except Exception as e:
        print(f"  Error {f}: {e}")
        return None

class DataLoader:
    def __init__(self, config: DataConfig):
        self.config = config

    def load_and_merge(self, timeframe: str) -> pd.DataFrame:
        """Main entry point: Loads trades, loads OB, merges them."""
        print(f"--- Data Loader ({timeframe}) ---")
        
        # 1. Load Trades (Returns Polars DataFrame)
        trades_df = self._load_trades()
        if trades_df.is_empty():
            return pd.DataFrame()
            
        # 2. Aggregate Trades to Bars (Returns Polars DataFrame)
        bars_pl = self._agg_trades(trades_df, timeframe)
        print(f"Trade Bars: {bars_pl.height}")
        
        # Convert to Pandas for merging with Orderbook data (which is still Pandas based)
        bars = bars_pl.to_pandas()
        
        # CRITICAL: Convert price_path from Python List to Numpy Array to match legacy behavior
        if 'price_path' in bars.columns:
            bars['price_path'] = bars['price_path'].apply(np.array)
        
        # 3. Stream & Aggregate Orderbook
        ob_bars = self._process_orderbook(timeframe)
        
        # 4. Merge
        if ob_bars.empty:
            print("Warning: No Orderbook data found. Using Trade data only.")
            return bars
            
        print("Merging Orderbook data...")
        # Ensure indexes match
        if 'datetime' in bars.columns: bars.set_index('datetime', inplace=True)
        # ob_bars already has datetime index
        
        merged = bars.join(ob_bars, how='left')
        
        # Forward fill OB data (snapshots stick until changed)
        merged.ffill(inplace=True) # Updated from fillna(method='ffill')
        merged.fillna(0, inplace=True) # Handle startup NaNs
        
        return merged.reset_index()

    def _load_trades(self) -> pl.DataFrame:
        """
        Loads trades using Polars.
        Checks for a 'trades_cache.parquet' first.
        If not found, loads CSVs, creates the cache, and returns.
        """
        # 1. Check for Parquet Cache
        cache_path = self.config.data_dir / "trades_cache.parquet"
        
        # Check if cache is fresh enough? 
        # For simplicity, if it exists, we use it (assuming immutable historical data).
        # To be robust, one might check mtimes, but that's expensive for many files.
        if cache_path.exists():
            print(f"Loading cached trades from {cache_path}...")
            try:
                return pl.read_parquet(cache_path)
            except Exception as e:
                print(f"Failed to load cache: {e}. Reloading from CSVs.")
        
        # 2. Load from CSVs
        search_path = self.config.data_dir / self.config.trade_subdir / self.config.trade_pattern
        files = sorted(glob.glob(str(search_path)))
        
        if not files:
            print("No trade files found.")
            return pl.DataFrame()
            
        print(f"Loading {len(files)} trade files with Polars...")
        
        # We define a schema or at least the columns we want
        # Polars scan_csv is fast.
        
        q_list = []
        for f in files:
            # We lazy scan each file to handle potential missing columns per file safely
            try:
                # Basic scan with forced schema to prevent inference errors (e.g. scientific notation)
                overrides = {
                    self.config.price_col: pl.Float64,
                    self.config.size_col: pl.Float64,
                    self.config.timestamp_col: pl.Float64
                }
                q = pl.scan_csv(f, schema_overrides=overrides)
                
                # Check columns (Lazy schema check)
                schema = q.collect_schema()
                
                # Ensure side column exists
                if self.config.side_col not in schema.names():
                     q = q.with_columns(pl.lit("Buy").alias(self.config.side_col))
                
                # Select only needed columns and cast types
                q = q.select([
                    pl.col(self.config.timestamp_col).cast(pl.Int64),
                    pl.col(self.config.price_col).cast(pl.Float64),
                    pl.col(self.config.size_col).cast(pl.Float64),
                    pl.col(self.config.side_col).cast(pl.Utf8)
                ])
                
                q_list.append(q)
            except Exception as e:
                print(f"Error preparing {f}: {e}")

        if not q_list:
            return pl.DataFrame()

        # Concatenate all lazy frames
        full_lazy = pl.concat(q_list)
        
        # Sort by timestamp
        full_lazy = full_lazy.sort(self.config.timestamp_col)
        
        # Collect
        full_df = full_lazy.collect()
        
        # Save to Parquet Cache
        try:
            full_df.write_parquet(cache_path)
            print(f"Saved trades cache to {cache_path}")
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")
            
        return full_df

    def _agg_trades(self, df: pl.DataFrame, timeframe: str) -> pl.DataFrame:
        tf_map = {'1m': '1m', '5m': '5m', '15m': '15m', '1h': '1h', '4h': '4h'}
        rule = tf_map.get(timeframe, '15m')
        
        # Auto-detect timestamp unit and convert to datetime
        # Assuming ms if > 3e9 (year 2065 in seconds, safely assumes ms for current timestamps)
        ts_val = df.select(pl.col(self.config.timestamp_col).first()).item()
        
        # Convert to microseconds (us) for Polars Datetime
        if ts_val > 3000000000:
            # It's milliseconds
            # Multiply by 1000 to get us
            df = df.with_columns(
                (pl.col(self.config.timestamp_col) * 1000).cast(pl.Int64).cast(pl.Datetime("us")).alias("datetime")
            )
        else:
            # It's seconds
            # Multiply by 1_000_000 to get us
            df = df.with_columns(
                (pl.col(self.config.timestamp_col) * 1_000_000).cast(pl.Int64).cast(pl.Datetime("us")).alias("datetime")
            )
        
        # Microstructure Pre-calc
        # side_map = {'Buy': 1, 'Sell': -1, 'buy': 1, 'sell': -1}
        # Polars efficient mapping
        df = df.with_columns(
            pl.when(pl.col(self.config.side_col).str.to_lowercase() == "buy").then(1)
            .when(pl.col(self.config.side_col).str.to_lowercase() == "sell").then(-1)
            .otherwise(0)
            .alias("side_num")
        )
        
        # Metrics
        df = df.with_columns([
            pl.when(pl.col("side_num") == 1).then(pl.col(self.config.size_col)).otherwise(0).alias("vol_buy"),
            pl.when(pl.col("side_num") == -1).then(pl.col(self.config.size_col)).otherwise(0).alias("vol_sell"),
            (pl.col(self.config.price_col) * pl.col(self.config.size_col)).alias("dollar_val")
        ])
        
        # Aggregation using group_by_dynamic
        # rule matches Polars duration string format (e.g. "15m", "1h")
        
        # Ensure we sort by datetime before dynamic groupby (usually required/recommended)
        df = df.sort("datetime")
        
        agg_df = df.group_by_dynamic("datetime", every=rule).agg([
            pl.col(self.config.price_col).first().alias("open"),
            pl.col(self.config.price_col).max().alias("high"),
            pl.col(self.config.price_col).min().alias("low"),
            pl.col(self.config.price_col).last().alias("close"),
            
            pl.col(self.config.size_col).sum().alias("volume"),
            pl.col("vol_buy").sum().alias("vol_buy"),
            pl.col("vol_sell").sum().alias("vol_sell"),
            pl.col("side_num").count().alias("trade_count"),
            pl.col("dollar_val").sum().alias("dollar_val"),
            
            # Capture Real Price Path (List of prices)
            pl.col(self.config.price_col).alias("price_path") 
        ])
        
        # Post-Aggregation Metrics
        agg_df = agg_df.with_columns([
            (pl.col("dollar_val") / pl.col("volume").replace(0, 1)).alias("vwap"),
            (pl.col("vol_buy") / pl.col("volume").replace(0, 1)).alias("taker_buy_ratio")
        ])
        
        # Drop empty/incomplete (equivalent to dropna(subset=['close']))
        agg_df = agg_df.drop_nulls(subset=['close'])
        
        return agg_df

    def _process_orderbook(self, timeframe: str) -> pd.DataFrame:
        search_path = self.config.data_dir / self.config.orderbook_subdir / self.config.orderbook_pattern
        files = sorted(glob.glob(str(search_path)))
        
        if not files: return pd.DataFrame()
        
        print(f"Processing {len(files)} Orderbook files (Parallel)...")
        
        tf_map = {'1m': '1min', '5m': '5min', '15m': '15min', '1h': '1h'}
        rule = tf_map.get(timeframe, '15min')
        
        # Setup Cache Directory
        cache_dir = self.config.data_dir / "Orderbook_reconstruction"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Parallel Execution
        # We pass necessary simple types or paths, avoid passing complex objects if possible
        
        aggs = Parallel(n_jobs=-1)(
            delayed(_process_single_ob_file)(
                f, timeframe, self.config.data_dir, self.config.ob_levels, rule
            ) for f in files
        )
        
        # Filter None results
        aggs = [df for df in aggs if df is not None]
                
        if not aggs: return pd.DataFrame()
        
        full_ob = pd.concat(aggs).sort_index()
        return full_ob
