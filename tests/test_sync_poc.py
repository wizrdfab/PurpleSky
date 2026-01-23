
import logging
import sys
from dataclasses import dataclass, field
from typing import List, Dict

# --- MOCKS ---

@dataclass
class MockTrade:
    trade_id: str
    side: str
    size: float
    order_id: str = None # The new field

@dataclass
class MockVPM:
    trades: List[MockTrade] = field(default_factory=list)
    
    def close_trade(self, trade_id):
        print(f"  [VPM] Closing Trade {trade_id} (Pruned)")
        self.trades = [t for t in self.trades if t.trade_id != trade_id]

class MockAdapter:
    def __init__(self):
        self.open_orders = [] # List of dicts
        self.positions = []   # List of dicts

    def get_open_orders(self, symbol):
        return self.open_orders
    
    def get_positions(self, symbol):
        return self.positions

# --- THE LOGIC TO TEST ---

def reconcile_logic_demo(vpm, adapter, symbol="BTCUSDT"):
    print("\n--- Running Reconciliation Logic ---")
    
    # 1. Snapshot Reality
    open_orders = adapter.get_open_orders(symbol)
    open_order_ids = set(o['orderId'] for o in open_orders)
    
    positions = adapter.get_positions(symbol)
    actual_long = 0.0
    actual_short = 0.0
    for p in positions:
        if p['side'] == 'Buy': actual_long = p['size']
        if p['side'] == 'Sell': actual_short = p['size']
        
    print(f"  [Reality] Open Orders: {open_order_ids}")
    print(f"  [Reality] Position: Long={actual_long}, Short={actual_short}")
    
    # 2. Iterate Virtual Trades
    # We copy the list to safely modify the original during iteration
    for trade in list(vpm.trades):
        print(f"  [Check] Trade {trade.trade_id} (Order: {trade.order_id})...")
        
        if not trade.order_id: 
            print("    -> No Order ID. Skipping (Legacy).")
            continue
            
        # A. Is Pending?
        is_pending = trade.order_id in open_order_ids
        
        # B. Is Valid Logic?
        # If NOT pending, it must be in position.
        if not is_pending:
            has_position = (trade.side == "Buy" and actual_long > 0) or \
                           (trade.side == "Sell" and actual_short > 0)
            
            if not has_position:
                print("    -> CRITICAL: Not Pending AND No Position.")
                print("    -> CONCLUSION: Trade is Dead. PRUNING.")
                vpm.close_trade(trade.trade_id)
            else:
                print("    -> OK: Not pending, but Position exists (Filled).")
        else:
            print("    -> OK: Order is Pending.")


# --- TEST CASES ---

def run_tests():
    # Setup
    adapter = MockAdapter()
    vpm = MockVPM()
    
    print("=== TEST 1: Pending Order (Normal) ===")
    t1 = MockTrade(trade_id="v1", side="Buy", size=1.0, order_id="ord_123")
    vpm.trades.append(t1)
    
    # Reality: Order is open, No position yet
    adapter.open_orders = [{'orderId': 'ord_123'}]
    adapter.positions = []
    
    reconcile_logic_demo(vpm, adapter)
    assert len(vpm.trades) == 1
    print("RESULT: PASS (Trade kept)")
    
    print("\n=== TEST 2: Filled Order (Normal) ===")
    # Reality: Order gone (filled), Position exists
    adapter.open_orders = []
    adapter.positions = [{'side': 'Buy', 'size': 1.0}]
    
    reconcile_logic_demo(vpm, adapter)
    assert len(vpm.trades) == 1
    print("RESULT: PASS (Trade kept)")
    
    print("\n=== TEST 3: External Close (The Fix) ===")
    # Reality: Order gone, Position gone (Manually closed)
    adapter.open_orders = []
    adapter.positions = [] # Empty!
    
    reconcile_logic_demo(vpm, adapter)
    assert len(vpm.trades) == 0
    print("RESULT: PASS (Trade Pruned)")

if __name__ == "__main__":
    run_tests()
