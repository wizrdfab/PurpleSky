"""
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
from dataclasses import dataclass
from config import GlobalConfig
from numba import njit, float64, int64, boolean
from numba.types import Tuple

@njit(cache=True)
def run_backtest_kernel(
    highs, lows, opens, closes, atrs,
    p_long, p_short, p_dir_long, p_dir_short,
    ticks_flat, tick_offsets, tick_counts,
    threshold,
    # Strategy Params
    base_limit_offset_atr, time_limit_bars, max_holding_bars,
    stop_loss_atr, take_profit_atr, maker_fee, taker_fee,
    risk_per_trade, max_positions,
    # Model Params
    aggressive_threshold, direction_threshold
):
    n = len(highs)
    equity = 10000.0
    
    # Position Struct: [entry_price, size, tp, sl, entry_idx, side]
    # side: 1 (Long), -1 (Short), 0 (Empty)
    positions = np.zeros((max_positions, 6), dtype=np.float64)
    pos_count = 0
    
    # Active Orders: [limit_price, index, active_flag(0/1)]
    active_buy = np.zeros(3, dtype=np.float64) 
    active_sell = np.zeros(3, dtype=np.float64)
    
    equity_curve = np.zeros(n, dtype=np.float64)
    
    # Pre-allocate history (Resize later if needed, assume max 10k trades)
    history_pnl = np.zeros(10000, dtype=np.float64)
    history_side = np.zeros(10000, dtype=np.int64)
    history_equity = np.zeros(10000, dtype=np.float64)
    history_count = 0
    
    for i in range(n):
        # 0. Tick Processing Setup
        has_ticks = tick_counts[i] > 0
        current_ticks = ticks_flat[tick_offsets[i] : tick_offsets[i] + tick_counts[i]] if has_ticks else None
        
        # Determine Simulation Mode: Tick-by-Tick OR OHLC Fallback
        # Optimization: Bounding Box Check
        # If High/Low of bar doesn't touch any TP/SL/Limit, skip ticks.
        
        use_ticks = False
        if has_ticks:
            bar_low, bar_high = lows[i], highs[i]
            
            # Check Positions
            for p_idx in range(max_positions):
                if positions[p_idx, 5] != 0: # Active
                    p_sl = positions[p_idx, 3]
                    p_tp = positions[p_idx, 2]
                    p_side = positions[p_idx, 5]
                    
                    if p_side == 1: # Long
                        if p_sl >= bar_low or p_tp <= bar_high:
                            use_ticks = True; break
                    else: # Short
                        if p_sl <= bar_high or p_tp >= bar_low:
                            use_ticks = True; break
                            
            # Check Orders
            if not use_ticks:
                if active_buy[2] == 1 and active_buy[0] >= bar_low: use_ticks = True
                elif active_sell[2] == 1 and active_sell[0] <= bar_high: use_ticks = True
        
        # --- SIMULATION LOOP ---
        
        # Define simulation steps: If using ticks, iterate ticks. Else, iterate once (OHLC Logic).
        iterations = len(current_ticks) if use_ticks else 1
        
        for t_step in range(iterations):
            
            # Current Price Context
            if use_ticks:
                curr_price = current_ticks[t_step]
            else:
                # OHLC Logic relies on checking Lows/Highs globally, 
                # but inside this loop we just use placeholders since we check explicit OHLC later
                curr_price = closes[i] 

            # 1. Manage Positions
            for p_idx in range(max_positions):
                if positions[p_idx, 5] == 0: continue # Empty slot
                
                side = positions[p_idx, 5]
                entry_p = positions[p_idx, 0]
                size = positions[p_idx, 1]
                tp = positions[p_idx, 2]
                sl = positions[p_idx, 3]
                entry_idx = int(positions[p_idx, 4])
                
                closed = False
                pnl = 0.0
                fee = 0.0
                
                # Logic: Tick Mode vs OHLC Mode
                if use_ticks:
                    # Tick Check
                    if side == 1:
                        if curr_price <= sl:
                            fee = sl * size * taker_fee
                            pnl = (sl - entry_p) * size - fee
                            closed = True
                        elif curr_price >= tp:
                            fee = tp * size * maker_fee
                            pnl = (tp - entry_p) * size - fee
                            closed = True
                    else:
                        if curr_price >= sl:
                            fee = sl * size * taker_fee
                            pnl = (entry_p - sl) * size - fee
                            closed = True
                        elif curr_price <= tp:
                            fee = tp * size * maker_fee
                            pnl = (entry_p - tp) * size - fee
                            closed = True
                else:
                    # OHLC Check (Only runs once per bar)
                    # Time Exit Check (Common to both, but checked at end of ticks usually)
                    # Here we check SL/TP against High/Low
                    if side == 1:
                        if lows[i] <= sl:
                            fee = sl * size * taker_fee
                            pnl = (sl - entry_p) * size - fee
                            closed = True
                        elif highs[i] >= tp:
                            fee = tp * size * maker_fee
                            pnl = (tp - entry_p) * size - fee
                            closed = True
                    else:
                        if highs[i] >= sl:
                            fee = sl * size * taker_fee
                            pnl = (entry_p - sl) * size - fee
                            closed = True
                        elif lows[i] <= tp:
                            fee = tp * size * maker_fee
                            pnl = (entry_p - tp) * size - fee
                            closed = True
                            
                if closed:
                    equity += pnl
                    if history_count < 10000:
                        history_pnl[history_count] = pnl
                        history_side[history_count] = int(side)
                        history_equity[history_count] = equity
                        history_count += 1
                    # Clear Position
                    positions[p_idx, :] = 0
                    pos_count -= 1
                    
            # 2. Manage Orders (Fills)
            # Buy Limit
            if active_buy[2] == 1:
                limit = active_buy[0]
                idx = int(active_buy[1])
                
                # Check Expiry
                if (i - idx) > time_limit_bars:
                    active_buy[2] = 0 # Cancel
                else:
                    filled = False
                    fill_p = 0.0
                    
                    if use_ticks:
                        if curr_price <= limit:
                            fill_p = curr_price
                            filled = True
                    else:
                        if lows[i] <= limit:
                            fill_p = opens[i] if opens[i] < limit else limit
                            filled = True
                            
                    if filled:
                        # Enter Long
                        size = (equity * risk_per_trade) / (atrs[idx] * stop_loss_atr)
                        fee = fill_p * size * maker_fee
                        equity -= fee
                        
                        tp = fill_p + (atrs[idx] * take_profit_atr)
                        sl = fill_p - (atrs[idx] * stop_loss_atr)
                        
                        # Find Empty Slot
                        for k in range(max_positions):
                            if positions[k, 5] == 0:
                                positions[k, 0] = fill_p
                                positions[k, 1] = size
                                positions[k, 2] = tp
                                positions[k, 3] = sl
                                positions[k, 4] = i
                                positions[k, 5] = 1
                                pos_count += 1
                                break
                        active_buy[2] = 0 # Remove Order

            # Sell Limit
            if active_sell[2] == 1:
                limit = active_sell[0]
                idx = int(active_sell[1])
                
                if (i - idx) > time_limit_bars:
                    active_sell[2] = 0
                else:
                    filled = False
                    fill_p = 0.0
                    
                    if use_ticks:
                        if curr_price >= limit:
                            fill_p = curr_price
                            filled = True
                    else:
                        if highs[i] >= limit:
                            fill_p = opens[i] if opens[i] > limit else limit
                            filled = True
                            
                    if filled:
                        # Enter Short
                        size = (equity * risk_per_trade) / (atrs[idx] * stop_loss_atr)
                        fee = fill_p * size * maker_fee
                        equity -= fee
                        
                        tp = fill_p - (atrs[idx] * take_profit_atr)
                        sl = fill_p + (atrs[idx] * stop_loss_atr)
                        
                        for k in range(max_positions):
                            if positions[k, 5] == 0:
                                positions[k, 0] = fill_p
                                positions[k, 1] = size
                                positions[k, 2] = tp
                                positions[k, 3] = sl
                                positions[k, 4] = i
                                positions[k, 5] = -1
                                pos_count += 1
                                break
                        active_sell[2] = 0
        
        # End of Bar: Time Exit Check
        # (This runs once per bar, after all ticks or OHLC logic)
        for p_idx in range(max_positions):
            if positions[p_idx, 5] != 0:
                entry_idx = int(positions[p_idx, 4])
                if (i - entry_idx) >= max_holding_bars:
                    # Close at Close
                    exit_price = closes[i]
                    size = positions[p_idx, 1]
                    side = positions[p_idx, 5]
                    entry_p = positions[p_idx, 0]
                    
                    fee = exit_price * size * taker_fee
                    pnl = (exit_price - entry_p) * size * side - fee
                    
                    equity += pnl
                    if history_count < 10000:
                        history_pnl[history_count] = pnl
                        history_side[history_count] = int(side)
                        history_equity[history_count] = equity
                        history_count += 1
                    
                    positions[p_idx, :] = 0
                    pos_count -= 1

        equity_curve[i] = equity

        # 3. Place Orders (Signal Logic)
        if pos_count < max_positions and atrs[i] > 0:
            
            # Aggressive Long
            if p_dir_long[i] > aggressive_threshold:
                active_buy[2] = 0 # Cancel pending
                # Market Buy
                fill_p = closes[i]
                size = (equity * risk_per_trade) / (atrs[i] * stop_loss_atr)
                fee = fill_p * size * taker_fee
                equity -= fee
                
                tp = fill_p + (atrs[i] * take_profit_atr * 10.0)
                sl = fill_p - (atrs[i] * stop_loss_atr)
                
                for k in range(max_positions):
                    if positions[k, 5] == 0:
                        positions[k, 0] = fill_p
                        positions[k, 1] = size
                        positions[k, 2] = tp
                        positions[k, 3] = sl
                        positions[k, 4] = i
                        positions[k, 5] = 1
                        pos_count += 1
                        break
                        
            # Limit Long
            elif active_buy[2] == 0 and p_long[i] > threshold and p_dir_long[i] > direction_threshold:
                limit = closes[i] - (atrs[i] * base_limit_offset_atr)
                active_buy[0] = limit
                active_buy[1] = i
                active_buy[2] = 1
            
            # Aggressive Short
            if p_dir_short[i] > aggressive_threshold:
                active_sell[2] = 0
                # Market Sell
                fill_p = closes[i]
                size = (equity * risk_per_trade) / (atrs[i] * stop_loss_atr)
                fee = fill_p * size * taker_fee
                equity -= fee
                
                tp = fill_p - (atrs[i] * take_profit_atr * 10.0)
                sl = fill_p + (atrs[i] * stop_loss_atr)
                
                for k in range(max_positions):
                    if positions[k, 5] == 0:
                        positions[k, 0] = fill_p
                        positions[k, 1] = size
                        positions[k, 2] = tp
                        positions[k, 3] = sl
                        positions[k, 4] = i
                        positions[k, 5] = -1
                        pos_count += 1
                        break
                        
            # Limit Short
            elif active_sell[2] == 0 and p_short[i] > threshold and p_dir_short[i] > direction_threshold:
                limit = closes[i] + (atrs[i] * base_limit_offset_atr)
                active_sell[0] = limit
                active_sell[1] = i
                active_sell[2] = 1

    return equity_curve, history_pnl[:history_count], history_side[:history_count], history_equity[:history_count]


class Backtester:
    def __init__(self, config: GlobalConfig):
        self.conf = config
        self.strat = config.strategy
        self.equity = 10000.0
        
    def run(self, df: pd.DataFrame, threshold: float) -> dict:
        # Prepare Data for Numba
        highs = df['high'].values.astype(np.float64)
        lows = df['low'].values.astype(np.float64)
        opens = df['open'].values.astype(np.float64)
        closes = df['close'].values.astype(np.float64)
        atrs = df['atr'].values.astype(np.float64)
        p_long = df['pred_long'].values.astype(np.float64)
        p_short = df['pred_short'].values.astype(np.float64)
        
        # Safe Get with defaults
        if 'pred_dir_long' in df.columns:
            p_dir_long = df['pred_dir_long'].values.astype(np.float64)
        else:
            p_dir_long = np.ones(len(df), dtype=np.float64)
            
        if 'pred_dir_short' in df.columns:
            p_dir_short = df['pred_dir_short'].values.astype(np.float64)
        else:
            p_dir_short = np.ones(len(df), dtype=np.float64)

        # Prepare Price Paths (Flattened)
        # We need: ticks_flat (1D), tick_offsets (1D), tick_counts (1D)
        
        has_paths = 'price_path' in df.columns
        n = len(df)
        
        if has_paths:
            # Vectorized Flattening
            # This handles the object column of arrays efficiently
            paths = df['price_path'].values
            
            # Calculate lengths
            lengths = np.array([len(p) if isinstance(p, (np.ndarray, list)) else 0 for p in paths], dtype=np.int64)
            offsets = np.zeros(n, dtype=np.int64)
            offsets[1:] = np.cumsum(lengths)[:-1]
            total_ticks = np.sum(lengths)
            
            ticks_flat = np.zeros(total_ticks, dtype=np.float64)
            
            # We still need a loop to fill, but it's just copying
            # (Or use np.concatenate which is fast)
            valid_paths = [p for p in paths if isinstance(p, (np.ndarray, list)) and len(p) > 0]
            if valid_paths:
                ticks_flat = np.concatenate(valid_paths).astype(np.float64)
            else:
                ticks_flat = np.zeros(0, dtype=np.float64) # Safety
                
            tick_offsets = offsets
            tick_counts = lengths
        else:
            ticks_flat = np.zeros(0, dtype=np.float64)
            tick_offsets = np.zeros(n, dtype=np.int64)
            tick_counts = np.zeros(n, dtype=np.int64)

        # Run Kernel
        eq_curve, h_pnl, h_side, h_eq = run_backtest_kernel(
            highs, lows, opens, closes, atrs,
            p_long, p_short, p_dir_long, p_dir_short,
            ticks_flat, tick_offsets, tick_counts,
            threshold,
            # Params
            self.strat.base_limit_offset_atr,
            self.strat.time_limit_bars,
            self.strat.max_holding_bars,
            self.strat.stop_loss_atr,
            self.strat.take_profit_atr,
            self.strat.maker_fee,
            self.strat.taker_fee,
            self.strat.risk_per_trade,
            self.strat.max_positions,
            self.conf.model.aggressive_threshold,
            self.conf.model.direction_threshold
        )
        
        # Post-Process Results
        self.equity_curve = eq_curve
        self.equity = eq_curve[-1]
        
        # Reconstruct History
        self.history = []
        for j in range(len(h_pnl)):
            self.history.append({
                'pnl': h_pnl[j],
                'side': h_side[j],
                'equity': h_eq[j]
            })
            
        return self._stats()

    def _stats(self):
        if not self.history:
            return {
                'total_return': 0.0, 'trades': 0, 'win_rate': 0.0, 
                'final_equity': 10000.0, 'max_drawdown': 0.0, 'sortino': 0.0
            }
        
        # Basic
        wins = len([x for x in self.history if x['pnl'] > 0])
        total = len(self.history)
        ret = (self.equity - 10000) / 10000
        
        # Drawdown
        equity_curve = self.equity_curve
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        max_dd = abs(np.min(drawdown))
        
        # Sortino (Professional Downside Deviation)
        pnls = np.array([x['pnl'] for x in self.history])
        avg_ret = np.mean(pnls)
        
        # Downside Deviation: RMS of negative returns only
        downside_pnls = pnls[pnls < 0]
        if len(downside_pnls) == 0:
            sortino = 10.0 if avg_ret > 0 else -10.0
        else:
            # Standard Sortino uses 0 as target return
            downside_dev = np.sqrt(np.mean(downside_pnls**2))
            sortino = avg_ret / downside_dev if downside_dev > 0 else 0
            
        return {
            'total_return': ret,
            'trades': total,
            'win_rate': wins/total,
            'final_equity': self.equity,
            'max_drawdown': max_dd,
            'sortino': sortino
        }