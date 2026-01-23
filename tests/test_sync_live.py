
import logging
import time
import json
import sys
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

# Import the REAL Adapter
sys.path.append(".")
from bybit_adapter import BybitAdapter

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("TestSyncLive")

# --- PROPOSED STRUCTURES (MOCKS) ---

@dataclass
class MockTrade:
    trade_id: str
    symbol: str
    side: str
    size: float
    order_id: Optional[str] = None
    
@dataclass
class MockVPM:
    trades: List[MockTrade] = field(default_factory=list)
    
    def add_trade(self, t):
        self.trades.append(t)
        
    def close_trade(self, trade_id):
        logger.info(f"[VPM] Closing Trade {trade_id} (DETERMINISTIC KILL)")
        self.trades = [t for t in self.trades if t.trade_id != trade_id]

# --- THE LOGIC TO TEST ---

def deterministic_reconcile(vpm, adapter, symbol):
    logger.info(">>> Running Deterministic Reconcile Logic...")
    
    # 1. Fetch Real State
    open_orders = adapter.get_open_orders(symbol)
    open_order_ids = set(o['orderId'] for o in open_orders)
    
    positions = adapter.get_positions(symbol)
    actual_long = 0.0
    actual_short = 0.0
    for p in positions:
        # Bybit Linear: Side=Buy -> Long, Side=Sell -> Short (in Hedge Mode idx 1/2) 
        # or just side based in One-Way.
        # Let's simplify and trust the parsed size since we aren't testing the adapter parsing today
        if p['side'] == 'Buy': actual_long = p['size']
        if p['side'] == 'Sell': actual_short = p['size']
        
    logger.info(f"    Reality: OpenOrders={len(open_order_ids)}, Long={actual_long}, Short={actual_short}")

    # 2. Iterate Virtual
    for trade in list(vpm.trades):
        if not trade.order_id:
            logger.warning(f"    Trade {trade.trade_id} has no Order ID. Skipping.")
            continue
            
        is_pending = trade.order_id in open_order_ids
        
        # LOGIC GATES
        if is_pending:
            logger.info(f"    Trade {trade.trade_id} (Ord: {trade.order_id}): PENDING. (OK)")
            continue
            
        # Not Pending. Must be in Position.
        has_position = False
        if trade.side == "Buy" and actual_long > 0: has_position = True
        if trade.side == "Sell" and actual_short > 0: has_position = True
        
        if has_position:
             logger.info(f"    Trade {trade.trade_id} (Ord: {trade.order_id}): FILLED/ACTIVE. (OK)")
        else:
             logger.error(f"    Trade {trade.trade_id} (Ord: {trade.order_id}): GONE!")
             logger.error("    -> Not Pending AND No Position. PRUNING.")
             vpm.close_trade(trade.trade_id)

def run_live_test():
    # Load Keys
    with open("keys.json") as f:
        keys = json.load(f)
        
    # Connect
    adapter = BybitAdapter(keys['api_key'], keys['api_secret'], testnet=True)
    symbol = "BTCUSDT"  # Use major pair
    
    # Check Connection
    price = adapter.get_current_price(symbol)
    if price == 0:
        logger.error("Failed to connect or fetch price. Check keys/network.")
        return
    logger.info(f"Connected. {symbol} Price: {price}")
    
    # Cleanup First
    adapter.cancel_all_orders(symbol)
    # Ensure no positions (Manual closes required if any exist, skipping for safety)
    
    vpm = MockVPM()
    
    # --- SCENARIO 1: PENDING ORDER LIFECYCLE ---
    logger.info("\n=== TEST 1: Pending Order Clean ===")
    
    # 1. Place Limit Buy (Deep OTM)
    limit_price = round(price * 0.5, 1)
    logger.info(f"Placing Limit Buy @ {limit_price}")
    res = adapter.place_order(symbol, "Buy", "Limit", 0.001, price=limit_price)
    
    if 'order_id' not in res:
        logger.error(f"Failed to place order: {res}")
        return
        
    oid = res['order_id']
    logger.info(f"Order Placed: {oid}")
    
    # 2. Register in VPM
    t1 = MockTrade(trade_id="v1", symbol=symbol, side="Buy", size=0.001, order_id=oid)
    vpm.add_trade(t1)
    
    # 3. Check (Should exist)
    deterministic_reconcile(vpm, adapter, symbol)
    if len(vpm.trades) != 1: raise Exception("Failed: Trade pruned prematurely!")
    
    # 4. CANCEL ORDER (Simulate External Action)
    logger.info("Externally Cancelling Order...")
    adapter.cancel_order(symbol, oid)
    time.sleep(2) # Wait for propagation
    
    # 5. Check (Should be pruned)
    deterministic_reconcile(vpm, adapter, symbol)
    if len(vpm.trades) != 0: raise Exception("Failed: Trade NOT pruned after cancel!")
    logger.info("PASS: Trade correctly pruned.")
    
    # --- SCENARIO 2: FILLED POSITION LIFECYCLE ---
    logger.info("\n=== TEST 2: Position Sync ===")
    
    # 1. Place Market Buy
    logger.info("Placing Market Buy (0.001)...")
    res = adapter.place_order(symbol, "Buy", "Market", 0.001)
    if 'order_id' not in res:
        logger.error("Failed to place market order.")
        return
        
    oid = res['order_id']
    t2 = MockTrade(trade_id="v2", symbol=symbol, side="Buy", size=0.001, order_id=oid)
    vpm.add_trade(t2)
    time.sleep(2) # Wait for fill
    
    # 2. Check (Should exist as 'Active')
    deterministic_reconcile(vpm, adapter, symbol)
    if len(vpm.trades) != 1: raise Exception("Failed: Filled trade pruned!")
    
    # 3. CLOSE POSITION (Simulate Stop Loss)
    logger.info("Externally Closing Position...")
    # Place Sell to close
    adapter.place_order(symbol, "Sell", "Market", 0.001, reduce_only=True)
    time.sleep(2)
    
    # 4. Check (Should be pruned)
    deterministic_reconcile(vpm, adapter, symbol)
    if len(vpm.trades) != 0: raise Exception("Failed: Closed trade NOT pruned!")
    logger.info("PASS: Trade correctly pruned.")
    
    logger.info("\nALL TESTS PASSED.")

if __name__ == "__main__":
    run_live_test()
