import json
import time
import uuid
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from pathlib import Path

@dataclass
class VirtualTrade:
    symbol: str
    side: str # "Buy" or "Sell"
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    entry_time: int = field(default_factory=lambda: int(time.time()))
    status: str = "open" # open, closed

@dataclass
class VirtualPositionManager:
    symbol: str
    max_positions: int = 3
    risk_per_trade: float = 0.01
    trades: List[VirtualTrade] = field(default_factory=list)
    storage_file: Path = Path("virtual_positions.json")
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("VirtualPositionManager"))

    def __post_init__(self):
        self.load()

    def load(self):
        if self.storage_file.exists():
            try:
                with open(self.storage_file, "r") as f:
                    data = json.load(f)
                    self.trades = [VirtualTrade(**t) for t in data if t['symbol'] == self.symbol and t['status'] == 'open']
                self.logger.info(f"Loaded {len(self.trades)} active trades for {self.symbol}")
            except Exception as e:
                self.logger.error(f"Failed to load virtual positions: {e}")

    def save(self):
        try:
            # We load all, update ours, and save back to preserve other symbols if shared file
            all_trades = []
            if self.storage_file.exists():
                try:
                    with open(self.storage_file, "r") as f:
                        all_trades = json.load(f)
                except: pass
            
            # Remove current symbol's open trades from list
            all_trades = [t for t in all_trades if not (t['symbol'] == self.symbol and t['status'] == 'open')]
            
            # Add current state
            all_trades.extend([asdict(t) for t in self.trades])
            
            with open(self.storage_file, "w") as f:
                json.dump(all_trades, f, indent=4)
        except Exception as e:
            self.logger.error(f"Failed to save virtual positions: {e}")

    def add_trade(self, side: str, price: float, size: float, sl: float, tp: float) -> bool:
        if len(self.trades) >= self.max_positions:
            self.logger.warning(f"Max positions ({self.max_positions}) reached. Cannot add trade.")
            return False
        
        # Check against existing trades to prevent duplicate signals (simple time deboucing)
        # If we added a trade in the last 60 seconds, ignore
        now = int(time.time())
        if any(abs(t.entry_time - now) < 60 for t in self.trades):
             self.logger.info("Signal debounced (trade added recently).")
             return False

        trade = VirtualTrade(
            symbol=self.symbol,
            side=side,
            entry_price=price,
            size=size,
            stop_loss=sl,
            take_profit=tp
        )
        self.trades.append(trade)
        self.save()
        self.logger.info(f"Opened Virtual Trade {trade.trade_id}: {side} {size} @ {price}")
        return True

    def close_trade(self, trade_id: str):
        for t in self.trades:
            if t.trade_id == trade_id:
                t.status = "closed"
                self.trades.remove(t)
                self.save()
                self.logger.info(f"Closed Virtual Trade {trade_id}")
                return
        self.logger.warning(f"Trade {trade_id} not found to close.")

    def get_net_position(self) -> float:
        net = 0.0
        for t in self.trades:
            if t.side == "Buy":
                net += t.size
            else:
                net -= t.size
        return net
    
    def get_active_stops(self) -> List[Dict]:
        """Returns list of needed Stop orders: [{'qty': float, 'trigger_price': float, 'side': str}]"""
        stops = []
        for t in self.trades:
            # If Long, Stop is Sell
            side = "Sell" if t.side == "Buy" else "Buy"
            stops.append({
                'id': t.trade_id, # Link back to virtual trade
                'qty': t.size,
                'trigger_price': t.stop_loss,
                'side': side,
                'type': 'sl'
            })
            # We could add TP here too
        return stops

    def prune_dead_trades(self, current_price: float):
        """
        Check if any active trades have hit their SL/TP based on current price.
        Used for simulation or fallback safety.
        Returns list of closed trade IDs.
        """
        closed = []
        for t in list(self.trades):
            if t.side == "Buy":
                if current_price <= t.stop_loss or current_price >= t.take_profit:
                    self.close_trade(t.trade_id)
                    closed.append(t.trade_id)
            else:
                if current_price >= t.stop_loss or current_price <= t.take_profit:
                    self.close_trade(t.trade_id)
                    closed.append(t.trade_id)
        return closed
