import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from live_trading import LiveBot
from config import CONF

class MockArgs:
    symbol = "FARTCOINUSDT"
    model_dir = "models/FARTCOINUSDT"
    timeframe = "5m"
    api_key = "test"
    api_secret = "test"
    testnet = True

@pytest.fixture
def bot():
    with patch('live_trading.BybitAdapter') as MockAdapter, \
         patch('live_trading.BybitWSAdapter') as MockWS, \
         patch('live_trading.VirtualPositionManager') as MockVPM, \
         patch('live_trading.LocalOrderbook') as MockOB, \
         patch('live_trading.OrderbookAggregator') as MockAgg, \
         patch('live_trading.ModelManager') as MockModel, \
         patch('live_trading.FeatureEngine') as MockFeat, \
         patch('joblib.load') as MockJoblib: # Patch joblib to avoid file load
        
        # Setup Mocks
        adapter = MockAdapter.return_value
        adapter.get_instrument_info.return_value = {
            'tick_size': 0.0001, 'qty_step': 1.0, 'min_qty': 1.0, 'min_notional': 5.0
        }
        adapter.get_wallet_balance.return_value = 50.0
        adapter.get_current_price.return_value = 0.30
        adapter.get_positions.return_value = [] # No initial positions
        adapter.get_open_orders.return_value = []
        
        # Mock LocalOrderbook Snapshot (Valid Spread)
        MockOB.return_value.get_snapshot.return_value = {
            'bids': [[0.2999, 1000]], 
            'asks': [[0.3000, 1000]], 
            'timestamp': 123456789
        }
        
        # Mock Joblib to return dummy data for features/models
        MockJoblib.return_value = ['close', 'open', 'high', 'low', 'volume'] # Dummy feature list
        
        # Initialize Bot
        bot = LiveBot(MockArgs())
        # Disable VPM loading from file for test
        bot.vpm.trades = []
        return bot

def test_aggressive_sell_execution_flow(bot):
    """
    Test that an aggressive sell signal correctly:
    1. Calculates sizing
    2. Adds trade to VPM
    3. Executes MARKET order WITH SL/TP attached
    """
    # 1. Setup Context
    close_price = 0.30
    atr = 0.005
    
    # Configure VPM to store the trade when added
    def mock_add_trade(side, price, size, sl, tp, check_debounce=True):
        bot.vpm.trades.append(MagicMock(
            side=side, entry_price=price, size=size, stop_loss=sl, take_profit=tp
        ))
        return True
    bot.vpm.add_trade.side_effect = mock_add_trade
    
    # 2. Trigger Signal
    # Risk = 1.3% of $50 = $0.65
    # SL Dist = ATR * 2.4 (from params) approx 0.012
    # Size = 0.65 / 0.012 ~= 54 units
    
    bot._execute_trade(
        side="Sell", 
        price=close_price, 
        atr=atr, 
        order_type="Market", 
        aggressive=True
    )
    
    # 3. Verify VPM State
    assert len(bot.vpm.trades) == 1
    trade = bot.vpm.trades[0]
    assert trade.side == "Sell"
    assert trade.size > 0
    # Check SL logic (Sell SL is ABOVE price)
    assert trade.stop_loss > close_price 
    # Check Aggressive TP logic (Sell TP is BELOW price, boosted by multiplier)
    assert trade.take_profit < close_price
    
    print(f"\n[Test] VPM Trade: {trade.size} @ {trade.entry_price} | SL: {trade.stop_loss} | TP: {trade.take_profit}")

    # 4. Verify Exchange Execution (Reconciliation)
    # The bot should have called place_order
    # We need to verify that 'sl' and 'tp' were passed to place_order
    
    # Find the place_order call
    calls = bot.rest_api.place_order.call_args_list
    assert len(calls) > 0, "No order placed on exchange"
    
    # Check args of the last call
    args, kwargs = calls[-1]
    
    print(f"[Test] Place Order Args: {args}")
    print(f"[Test] Place Order Kwargs: {kwargs}")
    
    # Check Positional Args: symbol, side, order_type, qty
    assert args[1] == "Sell"
    assert args[2] == "Market"
    assert float(args[3]) == trade.size
    
    # CRITICAL CHECK: Were SL and TP attached?
    assert kwargs.get('sl') is not None, "Stop Loss NOT attached to Market Order!"
    assert kwargs.get('tp') is not None, "Take Profit NOT attached to Market Order!"
    
    assert float(kwargs['sl']) == trade.stop_loss
    assert float(kwargs['tp']) == trade.take_profit
    
    print("[Test] SUCCESS: SL/TP were correctly attached to the Market Order.")

def test_naked_fill_recovery(bot):
    """
    Simulates a scenario where Bybit accepts the Market Order but SILENTLY ignores/drops 
    the SL/TP params (e.g. due to 'Trigger Price too close').
    
    The bot must:
    1. Place the Market Order (simulated 'Naked' fill).
    2. REFRESH the Open Orders state.
    3. DETECT that SL/TP are missing.
    4. PLACE separate orders to fix it.
    """
    # 1. Setup VPM with a trade that NEEDS stops
    # Since VPM is mocked, we must manually populate the list AND the get_active_stops method
    trade = MagicMock(side="Sell", entry_price=0.30, size=100.0, stop_loss=0.31, take_profit=0.25)
    bot.vpm.trades.append(trade)
    
    # Mock get_active_stops to return the dicts for SL/TP
    bot.vpm.get_active_stops.return_value = [
        {'id': '1', 'qty': 100.0, 'trigger_price': 0.31, 'side': 'Buy', 'type': 'sl'},
        {'id': '1', 'qty': 100.0, 'trigger_price': 0.25, 'side': 'Buy', 'type': 'tp'}
    ]
    
    # 2. Mock Bybit State
    # A. Initial State: No orders, No positions
    bot.rest_api.get_open_orders.side_effect = [[], []] 
    
    # Positions: First call (Empty), Second call (Filled)
    bot.rest_api.get_positions.side_effect = [
        [], 
        [{'symbol': 'FARTCOINUSDT', 'side': 'Sell', 'size': 100.0, 'entry_price': 0.30, 'unrealized_pnl': 0, 'position_idx': 2}]
    ]
    
    # B. Mock Place Order to succeed
    bot.rest_api.place_order.return_value = {'order_id': '123', 'order_link_id': 'AUTO-1'}
    
    # 3. Run Reconciliation
    bot.reconcile_positions()
    
    # 4. Verify
    # We expect AT LEAST 3 calls to place_order:
    # 1. The Market Sell (Entry)
    # 2. The Stop Loss (Fix)
    # 3. The Take Profit (Fix)
    
    calls = bot.rest_api.place_order.call_args_list
    print(f"\n[Test] Total Place Order Calls: {len(calls)}")
    for i, c in enumerate(calls):
        print(f"  Call {i+1}: {c[1]}")
        
    assert len(calls) >= 3, f"Failed to recover! Expected 3 calls (Entry + SL + TP), got {len(calls)}"
    
    # Verify Call 1 is Entry (Positional Args)
    # Args: (self.symbol, "Sell", "Market", qty)
    args_entry, kwargs_entry = calls[0]
    assert args_entry[1] == 'Sell'
    assert args_entry[2] == 'Market'
    
    # Verify Call 2/3 are conditional fixes (Keyword Args)
    # These use place_order(symbol=..., side=..., ...) so 'side' IS in kwargs
    fix_orders = [c[1] for c in calls[1:]]
    
    # Verify we sent ReduceOnly orders
    fix_types = [o.get('reduce_only') for o in fix_orders]
    assert all(fix_types), "Recovery orders must be ReduceOnly!"
    
    # Check prices
    prices = [float(o.get('trigger_price', 0)) for o in fix_orders]
    assert 0.31 in prices, "Missing SL placement"
    assert 0.25 in prices, "Missing TP placement"
    
    print("[Test] SUCCESS: Bot detected naked position and placed fixes.")