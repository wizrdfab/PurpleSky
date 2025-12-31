"""
Feature Engine.
Calculates Technicals + Advanced Orderbook Microstructure Features.
"""
import pandas as pd
import numpy as np
from config import FeatureConfig

class FeatureEngine:
    def __init__(self, config: FeatureConfig):
        self.config = config

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        df = df.copy()
        
        # 1. Volatility (ATR)
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].ewm(span=self.config.atr_period, adjust=False).mean()
        
        # 2. EMAs
        for p in self.config.ema_periods:
            df[f'ema_{p}'] = df['close'].ewm(span=p).mean()
            df[f'dist_ema_{p}'] = (df['close'] - df[f'ema_{p}']) / df['atr']
        
        # 3. RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 4. Microstructure (Trade Based)
        # Taker Flow Z-Score
        if 'taker_buy_ratio' in df.columns:
            df['taker_buy_z'] = (df['taker_buy_ratio'] - 0.5) / df['taker_buy_ratio'].rolling(24).std()
            
        # Volume Anomaly
        df['vol_z'] = (df['volume'] - df['volume'].rolling(24).mean()) / df['volume'].rolling(24).std()
        
        # 5. Advanced Orderbook Features
        if 'ob_imbalance_mean' in df.columns:
            # --- Base Metrics ---
            # Imbalance Regime
            df['ob_imb_trend'] = df['ob_imbalance_mean'].rolling(4).mean()
            
            # Micro-Price Deviation (The "Hidden" Trend)
            df['micro_pressure'] = df['ob_micro_dev_mean'] / df['atr'] # Normalized
            
            # Liquidity Impulse
            df['bid_depth_chg'] = df['ob_bid_depth_mean'].pct_change()
            df['ask_depth_chg'] = df['ob_ask_depth_mean'].pct_change()
            
            # Spread Regime (Historical Z-Score)
            df['spread_z'] = (df['ob_spread_mean'] - df['ob_spread_mean'].rolling(24).mean()) / df['ob_spread_mean'].rolling(24).std()

            # --- NEW Advanced Metrics ---
            
            # 1. Spread in Basis Points (Normalized Cost)
            df['ob_spread_bps'] = (df['ob_spread_mean'] / df['close']) * 10000
            
            # 2. Depth Log Ratio (Symmetric Pressure)
            # Log transform makes +10% bids comparable to +10% asks
            # Clip ratio to avoid log(0)
            depth_ratio = df['ob_bid_depth_mean'] / df['ob_ask_depth_mean'].replace(0, 1)
            df['ob_depth_log_ratio'] = np.log(depth_ratio.replace(0, 1e-9))
            
            # 3. Imbalance Z-Score (Shock Detection)
            # Is current imbalance unusual compared to last 4 hours (16 bars)?
            roll_mean = df['ob_imbalance_mean'].rolling(window=16).mean()
            roll_std = df['ob_imbalance_mean'].rolling(window=16).std()
            df['ob_imbalance_z'] = (df['ob_imbalance_mean'] - roll_mean) / roll_std.replace(0, 1)
            
            # 4. Price-Liquidity Divergence
            # Sign(Price Chg) * Sign(Imbalance Chg)
            # Positive = Confirming (Price follows Liquidity)
            # Negative = Divergent (Price moves against Liquidity -> Exhaustion/Spoof)
            df['price_chg'] = df['close'].diff()
            df['imb_chg'] = df['ob_imbalance_mean'].diff()
            df['price_liq_div'] = np.sign(df['price_chg']) * np.sign(df['imb_chg'])
            
            # 5. Liquidity Dominance (Depth vs Volume)
            # Ratio of Passive Liquidity to Active Volume
            # High = Absorptive (Mean Reversion), Low = Fragile (Trend)
            total_depth = df['ob_bid_depth_mean'] + df['ob_ask_depth_mean']
            df['liq_dominance'] = total_depth / df['volume'].replace(0, 1)
            
            # 6. Maker Uncertainty (Micro-Price Volatility)
            # Normalized by ATR. High values = Makers are adjusting rapidly = Risk
            df['micro_dev_vol'] = df['ob_micro_dev_std'] / df['atr'].replace(0, 1)

        cols_to_drop = ['tr0', 'tr1', 'tr2', 'tr', 'up_move', 'down_move', 'plus_dm', 'minus_dm', 'price_chg', 'imb_chg']
        df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
        
        return df.fillna(0)