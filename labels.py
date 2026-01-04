"""
Multi-Level Label Generation.
Creates targets for "Will a Buy Limit at X ATR Offset be profitable?"
"""
import pandas as pd
import numpy as np
from config import GlobalConfig

class Labeler:
    def __init__(self, config: GlobalConfig):
        self.strat = config.strategy

    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Target: Binary 1/0.
        Logic:
        1. Place Limit Buy at Close - (ATR * offset).
        2. Check if Low < Limit (Fill).
        3. If Filled: Check if High > Limit + TP *before* Low < Limit - SL.
        4. Also check Time Limit (if not hit TP/SL in N bars, it's a Fail).
        """
        if df.empty: return df
        df = df.copy()
        
        offset = self.strat.base_limit_offset_atr
        tp = self.strat.take_profit_atr
        sl = self.strat.stop_loss_atr
        time_limit = self.strat.time_limit_bars
        max_hold = self.strat.max_holding_bars
        
        n = len(df)
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        opens = df['open'].values
        atrs = df['atr'].values
        
        # Long Target
        y_long = np.zeros(n)
        # Short Target
        y_short = np.zeros(n)
        
        # Meta-Labeling Direction Targets
        df['target_dir_long'] = 0.0
        df['target_dir_short'] = 0.0
        
        for i in range(n - max_hold - 1):
            if atrs[i] == 0: continue
            
            # --- Directional Target (Meta-Labeling Model A) ---
            # Simply: Is the price higher/lower in the future?
            # We check the horizon equal to time_limit_bars (the window we are willing to wait)
            horizon = min(i + time_limit, n - 1)
            if closes[horizon] > closes[i] + (atrs[i] * 0.5): # Significant move required
                df.at[i, 'target_dir_long'] = 1
            if closes[horizon] < closes[i] - (atrs[i] * 0.5):
                df.at[i, 'target_dir_short'] = 1

            # --- LONG EXECUTION ---
            limit = closes[i] - (atrs[i] * offset)
            filled = False
            fill_price = 0.0
            fill_idx = -1
            
            # Check Fill (Time Limit)
            for j in range(i+1, min(i+time_limit+1, n)):
                if lows[j] <= limit:
                    filled = True
                    # Fill at Limit or Open if Open < Limit (Gap)
                    fill_price = opens[j] if opens[j] < limit else limit
                    fill_idx = j
                    break
            
            if filled:
                target_price = fill_price + (atrs[i] * tp)
                stop_price = fill_price - (atrs[i] * sl)
                
                # Check Outcome (Holding Limit)
                for k in range(fill_idx, min(fill_idx + max_hold, n)):
                    # Check SL
                    if lows[k] <= stop_price:
                        # Fail
                        break
                    # Check TP
                    if highs[k] >= target_price:
                        y_long[i] = 1 # Success
                        break
                        
            # --- SHORT ---
            limit_s = closes[i] + (atrs[i] * offset)
            filled_s = False
            fill_price_s = 0.0
            fill_idx_s = -1
            
            for j in range(i+1, min(i+time_limit+1, n)):
                if highs[j] >= limit_s:
                    filled_s = True
                    fill_price_s = opens[j] if opens[j] > limit_s else limit_s
                    fill_idx_s = j
                    break
            
            if filled_s:
                target_s = fill_price_s - (atrs[i] * tp)
                stop_s = fill_price_s + (atrs[i] * sl)
                
                for k in range(fill_idx_s, min(fill_idx_s + max_hold, n)):
                    if highs[k] >= stop_s:
                        break
                    if lows[k] <= target_s:
                        y_short[i] = 1
                        break
        
        df['target_long'] = y_long
        df['target_short'] = y_short
        return df