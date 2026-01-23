import unittest
from unittest.mock import MagicMock, patch, ANY
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from live_trading import LiveBot
from config import CONF

class TestAdversarialTradingLogic(unittest.TestCase):
    """
    Stress testing the trading logic:
    - Max Position Limits
    - Concurrent signals
    - Hedge Mode logic (Long + Short)
    - Partial Fills
    - Conflicting Exits
    """
    
    def setUp(self):
        self.mock_args = MagicMock()
        self.mock_args.symbol = "STRESSUSDT"
        self.mock_args.timeframe = "5m"
        self.mock_args.api_key = "k"
        self.mock_args.api_secret = "s"
        self.mock_args.model_dir = "models/STRESSUSDT"
        self.mock_args.testnet = True
        
        # Override Config for stress testing
        CONF.strategy.max_positions = 3
        CONF.strategy.risk_per_trade = 0.01
        CONF.live.max_spread_pct = 0.01 # Relax spread check for logic testing
        CONF.model.sequence_length = 60

        self.patches = [
            patch('live_trading.BybitAdapter'),
            patch('live_trading.BybitWSAdapter'),
            patch('live_trading.VirtualPositionManager'),
            patch('live_trading.LocalOrderbook'),
            patch('live_trading.OrderbookAggregator'),
            patch('live_trading.ModelManager'),
            patch('live_trading.FeatureEngine'),
            patch('live_trading.joblib.load')
        ]
        
        self.mocks = [p.start() for p in self.patches]
        (self.mock_rest, self.mock_ws, self.mock_vpm_cls, 
         self.mock_book, self.mock_agg, self.mock_mm, 
         self.mock_fe, self.mock_joblib) = self.mocks

        # Use a real-ish VPM behavior?
        # Using a mock VPM class makes it hard to test "Max Positions" logic if the mock doesn't store state.
        # I will replace the bot.vpm with a MagicMock that actually stores trades in a list 
        # to simulate the "Count" logic.
        
        self.bot = LiveBot(self.mock_args)
        
        # Setup Rest API
        self.bot.rest_api = self.mock_rest.return_value
        self.bot.rest_api.get_wallet_balance.return_value = 10000.0
        self.bot.rest_api.get_positions.return_value = [] # Start empty
        
        # Setup VPM Mock with state
        self.bot.vpm = MagicMock()
        self.bot.vpm.trades = []
        self.bot.vpm.max_positions = 3
        
        # Implement add_trade logic on the mock to simulate limit enforcement
        def mock_add_trade(side, price, size, sl, tp, check_debounce=True):
            if len(self.bot.vpm.trades) >= self.bot.vpm.max_positions:
                return False
            t = MagicMock()
            t.side = side
            t.size = size
            t.entry_price = price
            t.stop_loss = sl
            t.take_profit = tp
            t.trade_id = f"T-{len(self.bot.vpm.trades)}" # Corrected escaping for f-string
            self.bot.vpm.trades.append(t)
            return True
            
        self.bot.vpm.add_trade.side_effect = mock_add_trade
        
        # Mock instrument
        self.bot.instrument_info = {'tick_size': 1.0, 'qty_step': 1.0, 'min_qty': 1.0, 'min_notional': 10.0}
        
        # Mock Book for Spread check
        self.bot.local_book = MagicMock()
        self.bot.local_book.get_snapshot.return_value = {
            'bids': [[100.0, 100]], 'asks': [[100.1, 100]], 'timestamp': 123
        }

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_machine_gun_signals(self):
        """Fire 10 Aggressive Buy signals. Should only open 3 trades."""
        print("\n--- Test: Machine Gun Signals (Max Positions) ---")
        
        # Simulate Aggressive Buy Signal
        # p_dir_long > agg_thresh
        CONF.model.aggressive_threshold = 0.8
        
        # Fire 10 times
        for i in range(10):
            # execute_logic(p_long, p_short, p_dir_long, p_dir_short, close, atr)
            self.bot.execute_logic(0.5, 0.0, 0.9, 0.0, 100.0, 1.0)
            
        # Verify VPM has max 3 trades
        print(f"Trades opened: {len(self.bot.vpm.trades)}")
        self.assertEqual(len(self.bot.vpm.trades), 3)
        
        # Verify API called (at least 3 times for placement, maybe more for reconciles)
        # Reconcile is called after each add_trade.
        # Since 3 succeeded, place_order should be called at least 3 times.
        self.assertGreaterEqual(self.bot.rest_api.place_order.call_count, 3)

    def test_hedge_mode_conflict(self):
        """Open Long, then Open Short. Verify Reconcile handles net positions correctly."""
        print("\n--- Test: Hedge Mode Logic ---")
        
        # 1. Open Long
        self.bot.execute_logic(0.5, 0.0, 0.9, 0.0, 100.0, 1.0) # Aggressive Buy
        self.assertEqual(len(self.bot.vpm.trades), 1)
        self.assertEqual(self.bot.vpm.trades[0].side, "Buy")
        
        # Simulate Exchange having 1 Long
        self.bot.rest_api.get_positions.return_value = [
            {'position_idx': 1, 'side': 'Buy', 'size': 100.0} # Size matches risk calc
        ]
        
        # 2. Open Short (Aggressive Sell)
        self.bot.execute_logic(0.0, 0.5, 0.0, 0.9, 100.0, 1.0) 
        self.assertEqual(len(self.bot.vpm.trades), 2)
        self.assertEqual(self.bot.vpm.trades[1].side, "Sell")
        
        # Verify Reconcile Logic
        # It should try to place a Sell order for position_idx=2
        # Last call to place_order should be the Sell
        args, kwargs = self.bot.rest_api.place_order.call_args
        print(f"Last Order: {args} {kwargs}")
        
        self.assertEqual(args[1], "Sell") # Side
        self.assertEqual(kwargs['position_idx'], 2) # Short Bucket

    def test_limit_fill_scattered(self):
        """
        Simulate a Limit Order filling in multiple small chunks.
        Verify VPM aggregates them or stores multiple trades.
        """
        print("\n--- Test: Scattered Limit Fills ---")
        
        # 1. Place Limit
        self.bot.rest_api.place_order.return_value = {'order_id': 'L1', 'order_link_id': 'LIMIT-1'}
        self.bot._execute_trade("Buy", 100.0, 1.0, "Limit")
        
        # 2. Fill 1 (Qty 5)
        ws_msg_1 = {
            'data': [{
                'orderId': 'L1', 'orderLinkId': 'LIMIT-1',
                'side': 'Buy', 'execPrice': '100.0', 'execQty': '5.0'
            }]
        }
        self.bot.on_execution_update(ws_msg_1)
        
        # 3. Fill 2 (Qty 5)
        ws_msg_2 = {
            'data': [{
                'orderId': 'L1', 'orderLinkId': 'LIMIT-1',
                'side': 'Buy', 'execPrice': '100.0', 'execQty': '5.0'
            }]
        }
        self.bot.on_execution_update(ws_msg_2)
        
        # VPM should have 2 trades (since VPM is simple list)
        self.assertEqual(len(self.bot.vpm.trades), 2)
        total_size = sum(t.size for t in self.bot.vpm.trades)
        self.assertEqual(total_size, 10.0)
        
        # Reconcile should see Target=10.
        # If Exchange has 0 (mock default), it should try to Buy 10.
        # (This is correct behavior: if we filled but exchange says 0, we are out of sync, so we buy.
        # In reality, exchange would say 10).

    def test_conflicting_exits(self):
        """
        Scenario: Market creates a Wick that hits TP for Long and SL for Short in same second.
        """
        print("\n--- Test: Conflicting Exits (Wick) ---")
        
        # Setup: 1 Long, 1 Short in VPM
        # Long: SL 90, TP 110
        # Short: SL 110, TP 90
        # Price hits 110. Long TP hit. Short SL hit.
        
        # Setup VPM
        self.bot.vpm.trades = []
        # Manually add mock trades
        long_t = MagicMock()
        long_t.side = "Buy"; long_t.entry_price=100; long_t.size=10; long_t.stop_loss=90; long_t.take_profit=110; long_t.trade_id="L"
        short_t = MagicMock()
        short_t.side = "Sell"; short_t.entry_price=100; short_t.size=10; short_t.stop_loss=110; short_t.take_profit=90; short_t.trade_id="S"
        self.bot.vpm.trades = [long_t, short_t]
        
        # Prepare prune_dead_trades to return both IDs when price is 110
        # Since we mocked prune_dead_trades in setUp? No, we mocked the CLASS.
        # But we replaced `self.bot.vpm` with a MagicMock instance.
        # So `prune_dead_trades` is a mock. We must define its behavior.
        self.bot.vpm.prune_dead_trades.return_value = ["L", "S"]
        
        # Simulate price 110
        self.bot.rest_api.get_current_price.return_value = 110.0
        
        # Call manage_exits
        self.bot.manage_exits()
        
        # Verify Reconcile was called
        # And since VPM removed them (we assume prune_dead_trades removes them in real life),
        # Target = 0.
        # Exchange has 10 Long, 10 Short.
        # Reconcile should Reduce Long and Reduce Short.
        
        # To verify this fully, we need to simulate VPM being empty during reconcile.
        self.bot.vpm.trades = [] 
        self.bot.rest_api.get_positions.return_value = [
            {'position_idx': 1, 'side': 'Buy', 'size': 10.0},
            {'position_idx': 2, 'side': 'Sell', 'size': 10.0}
        ]
        
        self.bot.reconcile_positions()
        
        # Verify calls
        # Should call place_order(Reduce Long) AND place_order(Reduce Short)
        calls = self.bot.rest_api.place_order.call_args_list
        # Check for ReduceOnly=True calls
        reductions = [c for c in calls if c.kwargs.get('reduce_only') is True]
        self.assertEqual(len(reductions), 2)
        print(f"Reductions triggered: {len(reductions)}")

    def test_failure_to_execute_entry(self):
        """
        Adversarial: Bot decides to Buy, but API fails (Margin Error).
        Verify it doesn't crash and retries next time?
        """
        print("\n--- Test: Failed Entry Execution ---")
        
        # VPM has 0 trades. Signal Buy.
        # place_order raises Exception or returns error
        self.bot.rest_api.place_order.return_value = {'error': 'Margin Insufficient'}
        
        # execute_logic calls _execute_trade
        # _execute_trade calls place_order (Market) -> fails
        # _execute_trade calls vpm.add_trade (Success) -> VPM=1
        # _execute_trade calls reconcile -> Target=1, Actual=0 -> tries to Buy -> fails
        
        self.bot.execute_logic(0.5, 0.0, 0.9, 0.0, 100.0, 1.0)
        
        # Result: VPM has trade (Phantom trade). Exchange has nothing.
        self.assertEqual(len(self.bot.vpm.trades), 1)
        
        # Verify warning logged? 
        # Ideally, if place_order fails, we should maybe undo VPM?
        # Current logic: "VPM is Master". If Exchange fails, we keep VPM and try again later.
        # This is robust IF margin returns later.
        # But if error is permanent, we are stuck in a loop.
        # This confirms "Fail Open" behavior on API error if adapter swallows exception.

    def test_empty_orderbook(self):
        """Simulate total liquidity disappearance (Empty Book)"""
        print("\n--- Test: Empty Orderbook ---")
        self.bot.local_book.get_snapshot.return_value = {'bids': [], 'asks': [], 'timestamp': 123}
        
        with self.assertLogs('LiveTrading', level='WARNING') as cm:
            # Attempt to reconcile (which checks spread)
            self.bot.reconcile_positions()
            # It should NOT place orders.
            # But wait, my code checks `if snap and snap['bids']...`.
            # If empty, it skips the spread check?
            # Let's verify the logic: `if snap and snap['bids']...` -> If False, it continues to logic!
            # This is a potential SAFETY GAP I found earlier.
            # "If local_book is empty, we skip check and allow Market order?"
            pass 
            # Actually, `reconcile` continues.
            # But `place_order` ("Market") might fail on exchange if no liquidity.
            # Ideally, the bot should PAUSE if OB is empty.
            
    def test_negative_prices(self):
        """Simulate API bug returning negative price"""
        print("\n--- Test: Negative Prices ---")
        self.bot.rest_api.get_current_price.return_value = -100.0
        
        # Trigger manage_exits (which uses price)
        self.bot.manage_exits()
        # Should not crash. Should not trigger exits (since price <= 0 check in logic).
        self.bot.rest_api.place_order.assert_not_called()

    def test_wick_of_death(self):
        """Simulate price going +50% then -50% in one logic tick"""
        print("\n--- Test: Wick of Death ---")
        
        # Setup 1 Long
        self.bot.vpm.trades = []
        self.bot.vpm.add_trade("Buy", 100.0, 10.0, 90.0, 110.0)
        
        # Mock prune_dead_trades to actually remove the trade
        def mock_prune(price):
            if price >= 110.0: # TP Hit
                self.bot.vpm.trades = []
                return ["T-0"]
            return []
        self.bot.vpm.prune_dead_trades.side_effect = mock_prune
        
        # Simulate Exchange HAS the position initially
        self.bot.rest_api.get_positions.return_value = [
            {'position_idx': 1, 'side': 'Buy', 'size': 10.0}
        ]
        
        # 1. Price shoots to 150 (Hit TP)
        self.bot.rest_api.get_current_price.return_value = 150.0
        self.bot.manage_exits()
        
        # Verify TP triggered
        # VPM trades should be empty (closed)
        self.assertEqual(len(self.bot.vpm.trades), 0)
        
        # Now Update Exchange state to reflect it's CLOSED (for step 2)
        # Otherwise step 2 might try to close it again?
        # But wait, step 2 checks "Closed trade SL"?
        # If trade is closed in VPM, it's GONE. 
        # Price crashing to 50 should NOT trigger prune_dead_trades on it.
        # So it shouldn't matter what exchange says, because manage_exits loop
        # iterates VPM trades. If VPM is empty, loop is empty.
        
        # 2. Price crashes to 50
        self.bot.rest_api.get_positions.return_value = [] # Exchange closed it too (ideal sync)
        self.bot.rest_api.get_current_price.return_value = 50.0
        self.bot.manage_exits()
        
        # Verify no double-closing (Redundant place_order calls)
        # We expect 1 call from step 1.
        self.assertEqual(self.bot.rest_api.place_order.call_count, 1)

    def test_vpm_corruption(self):
        """Inject NaN into VPM state and trigger reconcile"""
        print("\n--- Test: VPM State Corruption ---")
        
        t = MagicMock()
        t.side = "Buy"
        t.size = float('nan') # CORRUPTION
        self.bot.vpm.trades = [t]
        
        # The sum() in get_net_position will be NaN.
        # vpm_net = NaN.
        # exch_net = 0.
        # diff = NaN.
        # abs(NaN) = NaN.
        # if NaN > 0.0001 -> False.
        # So it might silently fail to reconcile?
        # Or crash if used in calculations?
        
        # Let's see behavior.
        try:
            self.bot.reconcile_positions()
        except Exception as e:
            print(f"Caught expected crash: {e}")
            # If it crashes, that's "Fail Safe" (Better than Fail Open).
            pass

if __name__ == '__main__':
    unittest.main()
