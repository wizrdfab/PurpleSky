"""
Advanced Data Loader with Stream Processing.
Aggregates Trades AND Orderbook snapshots into high-fidelity bars.
"""
import pandas as pd
import numpy as np
import glob
import json
import os
from config import DataConfig

class DataLoader:
    def __init__(self, config: DataConfig):
        self.config = config

    def load_and_merge(self, timeframe: str) -> pd.DataFrame:
        """Main entry point: Loads trades, loads OB, merges them."""
        print(f"--- Data Loader ({timeframe}) ---")
        
        # 1. Load Trades
        trades_df = self._load_trades()
        if trades_df.empty:
            return pd.DataFrame()
            
        # 2. Aggregate Trades to Bars
        bars = self._agg_trades(trades_df, timeframe)
        print(f"Trade Bars: {len(bars)}")
        
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
        merged.fillna(method='ffill', inplace=True)
        merged.fillna(0, inplace=True) # Handle startup NaNs
        
        return merged.reset_index()

    def _load_trades(self) -> pd.DataFrame:
        search_path = self.config.data_dir / self.config.trade_subdir / self.config.trade_pattern
        files = sorted(glob.glob(str(search_path)))
        
        if not files:
            print("No trade files found.")
            return pd.DataFrame()
            
        dfs = []
        for f in files:
            try:
                # Minimal read
                preview = pd.read_csv(f, nrows=1)
                has_side = self.config.side_col in preview.columns
                
                cols = [self.config.timestamp_col, self.config.price_col, self.config.size_col]
                if has_side:
                    cols.append(self.config.side_col)
                
                df = pd.read_csv(f, usecols=cols)
                if not has_side:
                    df[self.config.side_col] = "Buy" # Default
                
                # Force numeric types (handle bad data)
                df[self.config.price_col] = pd.to_numeric(df[self.config.price_col], errors='coerce')
                df[self.config.size_col] = pd.to_numeric(df[self.config.size_col], errors='coerce')
                
                before_len = len(df)
                df.dropna(subset=[self.config.price_col, self.config.size_col], inplace=True)
                after_len = len(df)
                
                if after_len < before_len:
                    print(f"  Dropped {before_len - after_len} rows with invalid numeric data in {os.path.basename(f)}")
                
                if not df.empty:
                    dfs.append(df)
                else:
                    print(f"  File {os.path.basename(f)} resulted in empty dataframe.")
                    
            except Exception as e:
                print(f"Error reading {f}: {e}")
                
        if not dfs: return pd.DataFrame()
        
        full = pd.concat(dfs, ignore_index=True)
        full.sort_values(self.config.timestamp_col, inplace=True)
        return full

    def _agg_trades(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        tf_map = {'1m': '1min', '5m': '5min', '15m': '15min', '1h': '1h', '4h': '4h'}
        rule = tf_map.get(timeframe, '15min')
        
        df = df.copy()
        
        # Auto-detect timestamp unit
        if df[self.config.timestamp_col].iloc[0] > 3000000000:
            unit = 'ms'
        else:
            unit = 's'
        df['datetime'] = pd.to_datetime(df[self.config.timestamp_col], unit=unit)
        df.set_index('datetime', inplace=True)
        
        # Microstructure Pre-calc
        side_map = {'Buy': 1, 'Sell': -1, 'buy': 1, 'sell': -1}
        df['side_num'] = df[self.config.side_col].map(side_map).fillna(0)
        
        # Metrics
        df['vol_buy'] = np.where(df['side_num']==1, df[self.config.size_col], 0)
        df['vol_sell'] = np.where(df['side_num']==-1, df[self.config.size_col], 0)
        df['dollar_val'] = df[self.config.price_col] * df[self.config.size_col]
        
        # Aggregation
        ohlcv = df[self.config.price_col].resample(rule).ohlc()
        
        micro = df.resample(rule).agg({
            self.config.size_col: 'sum',      # Total Volume
            'vol_buy': 'sum',                 # Buy Volume
            'vol_sell': 'sum',                # Sell Volume
            'side_num': 'count',              # Trade Count
            'dollar_val': 'sum'               # For VWAP
        })
        
        micro.rename(columns={self.config.size_col: 'volume', 'side_num': 'trade_count'}, inplace=True)
        
        # Feature: Taker Buy/Sell Ratio
        # Feature: VWAP
        bars = pd.concat([ohlcv, micro], axis=1)
        bars['vwap'] = bars['dollar_val'] / bars['volume'].replace(0, 1)
        bars['taker_buy_ratio'] = bars['vol_buy'] / bars['volume'].replace(0, 1)
        
        # Drop empty
        bars.dropna(subset=['close'], inplace=True)
        
        return bars

    def _process_orderbook(self, timeframe: str) -> pd.DataFrame:
        search_path = self.config.data_dir / self.config.orderbook_subdir / self.config.orderbook_pattern
        files = sorted(glob.glob(str(search_path)))
        
        if not files: return pd.DataFrame()
        
        print(f"Processing {len(files)} Orderbook files (Full Reconstruction)...")
        
        tf_map = {'1m': '1min', '5m': '5min', '15m': '15min', '1h': '1h'}
        rule = tf_map.get(timeframe, '15min')
        
        aggs = []
        
        for f in files:
            try:
                buffer = []
                # Local Orderbook State: Dict[Price(str), Size(float)]
                # Using string keys for precision/matching, floats for values
                local_bids = {}
                local_asks = {}
                
                line_counter = 0
                snapshot_counter = 0
                
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
                            bids_slice = sorted_bids[:self.config.ob_levels]
                            asks_slice = sorted_asks[:self.config.ob_levels]
                            
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
                            
                if not buffer: continue
                
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
                aggs.append(chunk_agg)
                print(f"  Processed {os.path.basename(f)} ({line_counter} msgs)")
                
            except Exception as e:
                print(f"  Error {f}: {e}")
                
        if not aggs: return pd.DataFrame()
        
        full_ob = pd.concat(aggs).sort_index()
        return full_ob