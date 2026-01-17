"""
Multi-Level Label Generation. Creates targets for "Will a Buy Limit at X ATR Offset be profitable?"
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
import numpy as np
from numba import njit
from config import GlobalConfig

@njit(cache=True)
def generate_labels_kernel(highs, lows, closes, opens, atrs, 
                          base_offset, tp_mult, sl_mult, time_limit, max_hold):
    n = len(highs)
    y_long = np.zeros(n, dtype=np.float64)
    y_short = np.zeros(n, dtype=np.float64)
    t_dir_long = np.zeros(n, dtype=np.float64)
    t_dir_short = np.zeros(n, dtype=np.float64)
    
    # Pre-calc direction targets to avoid redundant checks
    # (Though inside the main loop is fine for cache locality)
    
    for i in range(n - max_hold - 1):
        if atrs[i] <= 0: continue
        
        # --- Directional Target ---
        horizon = min(i + time_limit, n - 1)
        if closes[horizon] > closes[i] + (atrs[i] * 0.5):
            t_dir_long[i] = 1.0
        if closes[horizon] < closes[i] - (atrs[i] * 0.5):
            t_dir_short[i] = 1.0

        # --- LONG EXECUTION ---
        limit = closes[i] - (atrs[i] * base_offset)
        filled = False
        fill_price = 0.0
        fill_idx = -1
        
        # Check Fill
        end_fill_window = min(i + time_limit + 1, n)
        for j in range(i + 1, end_fill_window):
            if lows[j] <= limit:
                filled = True
                fill_price = opens[j] if opens[j] < limit else limit
                fill_idx = j
                break
        
        if filled:
            target_price = fill_price + (atrs[i] * tp_mult)
            stop_price = fill_price - (atrs[i] * sl_mult)
            
            end_hold_window = min(fill_idx + max_hold, n)
            for k in range(fill_idx, end_hold_window):
                if lows[k] <= stop_price: # SL Hit
                    break
                if highs[k] >= target_price: # TP Hit
                    y_long[i] = 1.0
                    break
                    
        # --- SHORT EXECUTION ---
        limit_s = closes[i] + (atrs[i] * base_offset)
        filled_s = False
        fill_price_s = 0.0
        fill_idx_s = -1
        
        for j in range(i + 1, end_fill_window):
            if highs[j] >= limit_s:
                filled_s = True
                fill_price_s = opens[j] if opens[j] > limit_s else limit_s
                fill_idx_s = j
                break
        
        if filled_s:
            target_s = fill_price_s - (atrs[i] * tp_mult)
            stop_s = fill_price_s + (atrs[i] * sl_mult)
            
            end_hold_window = min(fill_idx_s + max_hold, n)
            for k in range(fill_idx_s, end_hold_window):
                if highs[k] >= stop_s: # SL Hit
                    break
                if lows[k] <= target_s: # TP Hit
                    y_short[i] = 1.0
                    break
                    
    return y_long, y_short, t_dir_long, t_dir_short

class Labeler:
    def __init__(self, config: GlobalConfig):
        self.strat = config.strategy

    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        df = df.copy()
        
        highs = df['high'].values.astype(np.float64)
        lows = df['low'].values.astype(np.float64)
        closes = df['close'].values.astype(np.float64)
        opens = df['open'].values.astype(np.float64)
        atrs = df['atr'].values.astype(np.float64)
        
        y_l, y_s, td_l, td_s = generate_labels_kernel(
            highs, lows, closes, opens, atrs,
            self.strat.base_limit_offset_atr,
            self.strat.take_profit_atr,
            self.strat.stop_loss_atr,
            self.strat.time_limit_bars,
            self.strat.max_holding_bars
        )
        
        df['target_long'] = y_l
        df['target_short'] = y_s
        df['target_dir_long'] = td_l
        df['target_dir_short'] = td_s
        
        return df