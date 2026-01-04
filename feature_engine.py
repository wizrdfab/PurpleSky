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

            # 7. Liquidity Elasticity (Slope/Gradient)
            # Captures "Strength" vs "Fragility" of the walls
            if 'ob_bid_slope_mean' in df.columns:
                # Normalize slopes by Price/ATR to make them comparable across symbols
                # A slope is (Price_Diff / Volume). 
                # We want (Price_Diff_in_Basis_Points / Volume)
                df['ob_bid_elasticity'] = (df['ob_bid_slope_mean'] / df['close'].replace(0, 1)) * 10000
                df['ob_ask_elasticity'] = (df['ob_ask_slope_mean'] / df['close'].replace(0, 1)) * 10000
                
                # Relative Elasticity (The Alpha Signal)
                # If Ratio > 1.0: Bids are "Thin" (Steep slope), Asks are "Thick" (Flat slope).
                df['ob_slope_ratio'] = df['ob_bid_elasticity'] / df['ob_ask_elasticity'].replace(0, 1)
                
                # Slope Shock (Z-Score)
                # Detecting when the wall suddenly becomes "Thin/Glassy"
                df['bid_slope_z'] = (df['ob_bid_elasticity'] - df['ob_bid_elasticity'].rolling(24).mean()) / df['ob_bid_elasticity'].rolling(24).std().replace(0, 1)

            # 8. Wall Integrity (Intention)
            # High = Concentrated liquidity at the front (Strong Intent)
            # Low = Scattered/Layered liquidity (Weak/Passive Intent)
            if 'ob_bid_integrity_mean' in df.columns:
                df['ob_integrity_skew'] = df['ob_bid_integrity_mean'] - df['ob_ask_integrity_mean']
                
                # Integrity Momentum
                df['bid_integrity_chg'] = df['ob_bid_integrity_mean'].diff()
                df['ask_integrity_chg'] = df['ob_ask_integrity_mean'].diff()

        # 6. Inventory Context (VWAP Deviations)
        # Using 'dollar_val' for VWAP. 4h = 48 bars, 24h = 288 bars (for 5m timeframe)
        if 'dollar_val' not in df.columns:
            df['dollar_val'] = df['close'] * df['volume']
            
        # 4-Hour VWAP
        v_4h = df['dollar_val'].rolling(48, min_periods=12).sum()
        q_4h = df['volume'].rolling(48, min_periods=12).sum()
        df['vwap_4h'] = v_4h / q_4h.replace(0, 1)
        df['vwap_4h_dist'] = (df['close'] - df['vwap_4h']) / df['atr'].replace(0, 1)
        
        # 24-Hour VWAP (The "Cost Basis" Proxy)
        v_24h = df['dollar_val'].rolling(288, min_periods=24).sum()
        q_24h = df['volume'].rolling(288, min_periods=24).sum()
        df['vwap_24h'] = v_24h / q_24h.replace(0, 1)
        df['vwap_24h_dist'] = (df['close'] - df['vwap_24h']) / df['atr'].replace(0, 1)
        
        # 7. Volume Regimes (Context)
        # Intraday Regime: Is today active relative to yesterday?
        vol_1h = df['volume'].rolling(12, min_periods=3).mean()
        vol_24h = df['volume'].rolling(288, min_periods=24).mean()
        df['vol_intraday'] = vol_1h / vol_24h.replace(0, 1)
        
        # Macro Regime: Is this week/month active relative to the last 30 days?
        # 30 days * 24h * 12 bars/h = 8640 bars
        vol_30d = df['volume'].rolling(8640, min_periods=288).mean()
        df['vol_macro'] = vol_24h / vol_30d.replace(0, 1)

        # 8. Volatility Regimes (Expansion/Compression)
        # Volatility Z-Score (Shock detection)
        df['atr_z'] = (df['atr'] - df['atr'].rolling(24).mean()) / df['atr'].rolling(24).std().replace(0, 1)
        
        # Volatility Regime (Intraday vs Daily)
        # Ratio > 1.0 = Expanding/High Vol -> Trend Risk
        # Ratio < 1.0 = Compressing/Low Vol -> Breakout Risk or Mean Reversion
        atr_1h = df['atr'].rolling(12).mean()
        atr_24h = df['atr'].rolling(288).mean()
        df['atr_regime'] = atr_1h / atr_24h.replace(0, 1)
        
        # Macro Volatility (Secular Trend)
        # Is the market waking up from a long slumber?
        atr_30d = df['atr'].rolling(8640, min_periods=288).mean()
        df['atr_macro'] = atr_24h / atr_30d.replace(0, 1)

        cols_to_drop = ['tr0', 'tr1', 'tr2', 'tr', 'up_move', 'down_move', 'plus_dm', 'minus_dm', 'price_chg', 'imb_chg', 'vol_buy', 'vol_sell', 'volume', 'vwap_4h', 'vwap_24h', 'dollar_val']
        df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
        
        # FINAL SAFETY: Drop any rows that have 0.0 or NaN in 'close' 
        # (prevents bootstrap holes or empty history from corrupting features)
        df = df[df['close'] > 0].copy()
        
        return df.fillna(0)