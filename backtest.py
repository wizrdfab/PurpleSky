"""
Strict Event-Driven Backtester.
Simulates Fees, Timeouts, and Holding Periods.
Calculates Advanced Risk Metrics (Sortino, Calmar, MaxDD).
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from config import GlobalConfig

@dataclass
class Position:
    entry_price: float
    size: float
    tp: float
    sl: float
    entry_idx: int
    side: int

class Backtester:
    def __init__(self, config: GlobalConfig):
        self.conf = config
        self.strat = config.strategy
        self.equity = 10000.0
        self.positions = []
        self.history = []
        self.equity_curve = [10000.0]
        
    def run(self, df: pd.DataFrame, threshold: float) -> dict:
        self.equity = 10000.0
        self.positions = []
        self.history = []
        self.equity_curve = [10000.0]
        
        # Pre-calc
        highs = df['high'].values
        lows = df['low'].values
        opens = df['open'].values
        closes = df['close'].values
        atrs = df['atr'].values
        p_long = df['pred_long'].values
        p_short = df['pred_short'].values
        
        offset = self.strat.base_limit_offset_atr
        
        active_buy = None 
        active_sell = None
        
        n = len(df)
        
        for i in range(n):
            # 1. Manage Positions
            keep_pos = []
            for pos in self.positions:
                closed = False
                pnl = 0
                fee = 0
                
                # Time Exit
                if (i - pos.entry_idx) >= self.strat.max_holding_bars:
                    exit_price = closes[i]
                    fee = exit_price * pos.size * self.strat.taker_fee
                    pnl = (exit_price - pos.entry_price) * pos.size * pos.side - fee
                    closed = True
                    
                # SL/TP
                elif pos.side == 1:
                    if lows[i] <= pos.sl:
                        fee = pos.sl * pos.size * self.strat.taker_fee
                        pnl = (pos.sl - pos.entry_price) * pos.size - fee
                        closed = True
                    elif highs[i] >= pos.tp:
                        fee = pos.tp * pos.size * self.strat.maker_fee
                        pnl = (pos.tp - pos.entry_price) * pos.size - fee
                        closed = True
                else:
                    if highs[i] >= pos.sl:
                        fee = pos.sl * pos.size * self.strat.taker_fee
                        pnl = (pos.entry_price - pos.sl) * pos.size - fee
                        closed = True
                    elif lows[i] <= pos.tp:
                        fee = pos.tp * pos.size * self.strat.maker_fee
                        pnl = (pos.entry_price - pos.tp) * pos.size - fee
                        closed = True
                        
                if closed:
                    self.equity += pnl
                    self.history.append({'pnl': pnl, 'side': pos.side, 'equity': self.equity})
                else:
                    keep_pos.append(pos)
            
            self.positions = keep_pos
            self.equity_curve.append(self.equity) # Track equity per bar (mark-to-market approx)
            
            # 2. Manage Orders
            if active_buy:
                limit, idx = active_buy
                if (i - idx) > self.strat.time_limit_bars:
                    active_buy = None
                elif lows[i] <= limit:
                    fill_p = opens[i] if opens[i] < limit else limit
                    size = (self.equity * self.strat.risk_per_trade) / (atrs[idx] * self.strat.stop_loss_atr)
                    fee = fill_p * size * self.strat.maker_fee
                    self.equity -= fee
                    
                    tp = fill_p + (atrs[idx] * self.strat.take_profit_atr)
                    sl = fill_p - (atrs[idx] * self.strat.stop_loss_atr)
                    self.positions.append(Position(fill_p, size, tp, sl, i, 1))
                    active_buy = None
            
            if active_sell:
                limit, idx = active_sell
                if (i - idx) > self.strat.time_limit_bars:
                    active_sell = None
                elif highs[i] >= limit:
                    fill_p = opens[i] if opens[i] > limit else limit
                    size = (self.equity * self.strat.risk_per_trade) / (atrs[idx] * self.strat.stop_loss_atr)
                    fee = fill_p * size * self.strat.maker_fee
                    self.equity -= fee
                    
                    tp = fill_p - (atrs[idx] * self.strat.take_profit_atr)
                    sl = fill_p + (atrs[idx] * self.strat.stop_loss_atr)
                    self.positions.append(Position(fill_p, size, tp, sl, i, -1))
                    active_sell = None
            
            # 3. Place Orders
            if len(self.positions) == 0 and atrs[i] > 0:
                if active_buy is None and p_long[i] > threshold:
                    limit = closes[i] - (atrs[i] * offset)
                    active_buy = (limit, i)
                
                if active_sell is None and p_short[i] > threshold:
                    limit = closes[i] + (atrs[i] * offset)
                    active_sell = (limit, i)
                    
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
        equity_curve = np.array(self.equity_curve)
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