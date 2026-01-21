import time
import logging
import argparse
import pandas as pd
import numpy as np
import json
import traceback
import signal
import sys
import uuid
from pathlib import Path
from datetime import datetime
from collections import deque

from config import CONF, GlobalConfig
from bybit_adapter import BybitAdapter, BybitWSAdapter
from virtual_position_manager import VirtualPositionManager
from orderbook_aggregator import OrderbookAggregator
from feature_engine import FeatureEngine
from models import ModelManager
from local_orderbook import LocalOrderbook
import joblib
import torch

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("live_trading.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("LiveTrading")

class LiveBot:
    def __init__(self, args):
        self.args = args
        self.symbol = args.symbol
        self.timeframe = args.timeframe
        self.tf_seconds = self._tf_to_seconds(self.timeframe)
        self.max_positions = CONF.strategy.max_positions
        self.risk_per_trade = CONF.strategy.risk_per_trade
        
        # Components
        self.rest_api = BybitAdapter(
            api_key=args.api_key, 
            api_secret=args.api_secret, 
            testnet=args.testnet
        )
        self.ws_api = BybitWSAdapter(
            api_key=args.api_key,
            api_secret=args.api_secret,
            testnet=args.testnet
        )
        
        self.vpm = VirtualPositionManager(symbol=self.symbol, max_positions=self.max_positions)
        self.local_book = LocalOrderbook(self.symbol)
        self.ob_agg = OrderbookAggregator(ob_levels=50)
        
        # Data State
        self.history_file = Path(f"history_{self.symbol}_{self.timeframe}.csv")
        self.bars = pd.DataFrame()
        self.last_bar_time = 0
        self.last_processed_bar = 0
        
        # Load Model
        logger.info(f"Loading model from {args.model_dir}...")
        self.model_manager = self._load_model_manager(args.model_dir)
        self.feature_engine = FeatureEngine(CONF.features)
        
        # Runtime Flags
        self.running = True
        self.warmup_bars = 50 
        self.real_ob_bars_count = 0
        self.pending_limits = {} # {oid: {'entry_time': ts, 'side': ..., 'size': ...}}
        
        # Taker Flow Tracking
        self.bar_taker_buy_vol = 0.0
        self.bar_taker_total_vol = 0.0
        
        # Load Instrument Info
        self.instrument_info = self.rest_api.get_instrument_info(self.symbol)
        if not self.instrument_info:
            logger.warning("Failed to fetch instrument info. Using defaults.")
            self.instrument_info = {'tick_size': 0.1, 'qty_step': 0.001, 'min_qty': 0.001, 'min_notional': 5.0}
        
        logger.info(f"Instrument Info: {self.instrument_info}")
        
        # --- AUTO-CONFIGURATION ---
        logger.info("Auto-configuring exchange settings...")
        self.rest_api.switch_position_mode(self.symbol, mode=3) # Force Hedge Mode
        self.rest_api.set_leverage(self.symbol, leverage=10.0) # Default to 10x safety
        
        # Check Clock Drift
        self.rest_api.check_clock_drift()
        
        # Graceful Shutdown
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    def normalize_price(self, price):
        tick = self.instrument_info['tick_size']
        return round(round(price / tick) * tick, 8)

    def normalize_qty(self, qty):
        step = self.instrument_info['qty_step']
        min_q = self.instrument_info['min_qty']
        if qty < min_q: return 0.0
        return round(round(qty / step) * step, 8)

    def _tf_to_seconds(self, tf):
        units = {'m': 60, 'h': 3600, 'd': 86400}
        try:
            return int(tf[:-1]) * units[tf[-1]]
        except:
            return 300 # Default 5m

    def _load_model_manager(self, model_dir):
        path = Path(model_dir)
        mm = ModelManager(CONF.model)
        mm.feature_cols = joblib.load(path / "features.pkl")
        
        context_keywords = ['atr', 'vol', 'rsi', 'std', 'slope']
        mm.context_cols = [c for c in mm.feature_cols if any(k in c.lower() for k in context_keywords)]
        if not mm.context_cols: mm.context_cols = mm.feature_cols[:10]
        
        mm.model_long = joblib.load(path / "model_long.pkl")
        mm.model_short = joblib.load(path / "model_short.pkl")
        
        if (path / "dir_model_long.pkl").exists():
            mm.dir_model_long = joblib.load(path / "dir_model_long.pkl")
            mm.dir_model_short = joblib.load(path / "dir_model_short.pkl")
            
        if (path / "lstm_model.pth").exists():
            from models import LSTMModel, GatingNetwork
            mm.lstm_model = LSTMModel(len(mm.feature_cols), CONF.model.lstm_hidden_size, 1, 0.0).to(mm.device)
            mm.lstm_model.load_state_dict(torch.load(path / "lstm_model.pth", map_location=mm.device, weights_only=True))
            
            mm.gating_model = GatingNetwork(len(mm.context_cols)).to(mm.device)
            mm.gating_model.load_state_dict(torch.load(path / "gating_model.pth", map_location=mm.device, weights_only=True))
            
            mm.scaler = joblib.load(path / "scaler.pkl")
            
        return mm

    def shutdown(self, signum, frame):
        logger.info("Shutdown signal received. Exiting...")
        self.running = False
        self.ws_api.disconnect()

    def _load_history(self):
        if self.history_file.exists():
            logger.info("Loading history from disk...")
            self.bars = pd.read_csv(self.history_file)
            if not self.bars.empty:
                self.last_bar_time = int(self.bars.iloc[-1]['timestamp'])
        else:
            logger.info("No history file found. Fetching 10,000 bars from exchange...")
            klines = self.rest_api.get_public_klines(self.symbol, self.timeframe, limit=10000)
            if klines:
                self.bars = pd.DataFrame(klines)
                for col in self.model_manager.feature_cols:
                    if col not in self.bars.columns and col not in ['open','high','low','close','volume','timestamp']:
                        self.bars[col] = 0.0
                self.last_bar_time = int(self.bars.iloc[-1]['timestamp'])
            else:
                logger.warning("Exchange returned no history. Starting with empty buffer.")
                self.bars = pd.DataFrame()
        
        if not self.bars.empty:
            self.bars['timestamp'] = self.bars['timestamp'].astype(int)
            
            # --- FULL HISTORY CONTINUITY CHECK ---
            if len(self.bars) >= 2:
                timestamps = self.bars['timestamp'].values
                deltas = np.diff(timestamps)
                expected = self.tf_seconds * 1000
                # Only flag real gaps (missing data), ignore duplicates/overlaps
                gaps = np.where(deltas > expected)[0]
                
                if len(gaps) > 0:
                    logger.warning(f"!!! HISTORY GAPS DETECTED: {len(gaps)} missing periods found !!!")
                    for g_idx in gaps[:5]: # Show first 5 gaps
                        logger.warning(f"    Gap at {datetime.fromtimestamp(timestamps[g_idx]/1000)} -> {datetime.fromtimestamp(timestamps[g_idx+1]/1000)}")
                else:
                    logger.info(f"History Continuity Check: OK (Checked {len(self.bars)} bars)")
            # -------------------------------------

    def run(self):
        self._load_history()
        
        # Connect WS
        logger.info("Connecting WS...")
        self.ws_api.subscribe_orderbook(self.symbol, self.local_book)
        self.ws_api.subscribe_kline(self.symbol, self.timeframe)
        self.ws_api.subscribe_trades(self.symbol, self.on_public_trade)
        self.ws_api.subscribe_private(self.on_position_update, self.on_execution_update)
        
        logger.info("Starting Main Loop...")
        
        while self.running:
            try:
                # 1. Process Orderbook Stream (High Frequency)
                # Take snapshot from local book maintained by WS
                snap = self.local_book.get_snapshot(limit=50)
                if snap:
                    # Feed aggregator
                    self.ob_agg.process_snapshot(snap['bids'], snap['asks'], snap['timestamp'])
                
                # 2. Check Bar Closure (Low Frequency)
                # We use WS kline stream as trigger, or fall back to system time
                # Ideally WS kline message has 'confirm': True when bar closes
                
                ws_kline = self.ws_api.latest_kline # {'start': ..., 'confirm': bool}
                if ws_kline and ws_kline.get('confirm'):
                    bar_start = int(ws_kline['start'])
                    
                    if bar_start > self.last_processed_bar:
                        self.on_bar_close(ws_kline)
                        self.last_processed_bar = bar_start
                
                time.sleep(0.1) # Fast loop
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                traceback.print_exc()
                time.sleep(5)

    def on_public_trade(self, msg):
        """Processes public trade stream to track taker flow."""
        for trade in msg.get('data', []):
            try:
                sz = float(trade.get('v', 0))
                side = trade.get('S') # 'Buy' or 'Sell'
                self.bar_taker_total_vol += sz
                if side == 'Buy':
                    self.bar_taker_buy_vol += sz
            except Exception as e:
                logger.error(f"Error processing public trade: {e}")

    def on_position_update(self, msg):
        # Update Virtual Position Manager? 
        # For now, VPM is master, exchange is slave. We reconcile periodically.
        pass

    def on_execution_update(self, msg):
        # Log fills
        for item in msg.get('data', []):
            logger.info(f"EXECUTION: {item.get('side')} {item.get('execQty')} @ {item.get('execPrice')}")
            
            link_id = item.get('orderLinkId', '')
            if link_id and link_id.startswith('LIMIT-'):
                logger.info(f"Detected Limit Fill {link_id}. Adding to VPM.")
                
                side = item['side']
                price = float(item['execPrice'])
                qty = float(item['execQty'])
                
                # Fallback SL/TP since we don't have ATR access here easily
                # Assuming 1% SL, 2% TP
                sl_dist = price * 0.01
                if side == "Buy":
                    sl = price - sl_dist
                    tp = price + (sl_dist * 2)
                else:
                    sl = price + sl_dist
                    tp = price - (sl_dist * 2)
                    
                self.vpm.add_trade(side, price, qty, sl, tp, check_debounce=False)

    def on_bar_close(self, ws_kline):
        logger.info(f"Bar Closing: {datetime.fromtimestamp(int(ws_kline['start'])/1000)}")
        
        # 1. Finalize OB Stats
        ob_stats = self.ob_agg.finalize()
        if not ob_stats:
             # Comprehensive fallbacks for all metrics the FeatureEngine expects
            keys = [
                'ob_spread_mean', 'ob_micro_dev_mean', 'ob_micro_dev_std', 'ob_micro_dev_last',
                'ob_imbalance_mean', 'ob_imbalance_last', 'ob_bid_depth_mean', 'ob_ask_depth_mean',
                'ob_bid_slope_mean', 'ob_ask_slope_mean', 'ob_bid_integrity_mean', 'ob_ask_integrity_mean'
            ]
            ob_stats = {k: 0.0 for k in keys} 
        
        # 2. Calculate Taker Buy Ratio
        taker_buy_ratio = 0.5
        if self.bar_taker_total_vol > 0:
            taker_buy_ratio = self.bar_taker_buy_vol / self.bar_taker_total_vol
        
        logger.info(f"Taker Flow: Buy Vol={self.bar_taker_buy_vol:.2f}, Total Vol={self.bar_taker_total_vol:.2f}, Ratio={taker_buy_ratio:.4f}")
        
        # Reset counters for next bar
        self.bar_taker_buy_vol = 0.0
        self.bar_taker_total_vol = 0.0

        # 3. Construct Bar
        row = {
            'timestamp': int(ws_kline['start']),
            'open': float(ws_kline['open']),
            'high': float(ws_kline['high']),
            'low': float(ws_kline['low']),
            'close': float(ws_kline['close']),
            'volume': float(ws_kline['volume']),
            'taker_buy_ratio': taker_buy_ratio
        }
        row.update(ob_stats)
        
        new_row_df = pd.DataFrame([row])
        self.bars = pd.concat([self.bars, new_row_df], ignore_index=True)
        self.last_bar_time = row['timestamp']
        
        # 4. Feature Calc (Using 10,000 bar window for macro stability)
        df_feats = self.feature_engine.calculate_features(self.bars.iloc[-10000:]) 
        
        # --- SAFETY SHIELD ---
        # Clean up extreme values (Inf/-Inf) and NaNs caused by history transition.
        # This prevents the scaler overflow that leads to 'nan' predictions.
        df_feats = df_feats.replace([np.inf, -np.inf], 0).fillna(0)
        
        # Update self.bars with calculated features so they are saved to CSV
        self.bars.update(df_feats)
        
        # Save History
        self.bars.to_csv(self.history_file, index=False)
        
        current_row = df_feats.iloc[[-1]]
        
        # --- COMPREHENSIVE DATA INTEGRITY & FEATURE REPORT ---
        logger.info(">>> [SYSTEM METRICS]")
        
        # 1. History & Gap Check
        hist_len = len(self.bars)
        gap_msg = "First Bar"
        if hist_len >= 2:
            delta = self.bars.iloc[-1]['timestamp'] - self.bars.iloc[-2]['timestamp']
            expected = self.tf_seconds * 1000
            if delta == expected:
                gap_msg = "OK (Continuous)"
            else:
                gap_msg = f"!!! GAP DETECTED (Delta={delta}ms, Exp={expected}ms) !!!"

        # 2. Latency Check (Processing Lag)
        bar_end_time = row['timestamp'] + (self.tf_seconds * 1000)
        now_ms = int(time.time() * 1000)
        lag_ms = now_ms - bar_end_time
        lag_msg = f"{lag_ms}ms"
        if lag_ms > 5000: lag_msg += " (HIGH LATENCY)"

        # 3. Data Quality (NaNs & OB)
        nan_count = current_row.isna().sum().sum()
        nan_msg = "OK" if nan_count == 0 else f"!!! WARNING: {nan_count} NaNs Detected !!!"
        
        ob_spread = row.get('ob_spread_mean', 0)
        ob_updates = row.get('ob_update_count', 0)
        
        ob_status = "OK"
        if ob_spread <= 0:
            ob_status = "!!! STALE/EMPTY SPREAD !!!"
        elif ob_updates < 5: # Expect at least 5 unique snapshots in 5 minutes
            ob_status = f"!!! LOW LIQUIDITY/FROZEN ({ob_updates} updates) !!!"

        logger.info(f"    History:       {hist_len} bars | Continuity: {gap_msg}")
        logger.info(f"    Processing Lag:{lag_msg}")
        logger.info(f"    Data Quality:  {nan_msg}")
        logger.info(f"    OB State:      {ob_status} (Spread: {ob_spread:.8f})")
        logger.info(f"    Warmup Status: {self.real_ob_bars_count}/{self.warmup_bars}")
        
        logger.info(">>> [ALL MODEL FEATURES]")
        feature_data = []
        for col in self.model_manager.feature_cols:
            if col in current_row.columns:
                val = current_row[col].iloc[0]
                feature_data.append(f"{col}: {val:.6f}")
            else:
                feature_data.append(f"{col}: MISSING!")
        
        # Print features in chunks of 4 per line for readability
        for i in range(0, len(feature_data), 4):
            logger.info("    " + " | ".join(feature_data[i:i+4]))
        
        logger.info(">>> [END REPORT]")
        # ------------------------------------------------------
        
        # 5. Predict (Optimized: use .copy() to satisfy LSTM and avoid SettingWithCopy)
        preds_full = self.model_manager.predict(df_feats.iloc[-100:].copy())
        preds = preds_full.iloc[[-1]]
        
        p_long = preds['pred_long'].iloc[0]
        p_short = preds['pred_short'].iloc[0]
        p_dir_long = preds['pred_dir_long'].iloc[0] if 'pred_dir_long' in preds else 1.0
        p_dir_short = preds['pred_dir_short'].iloc[0] if 'pred_dir_short' in preds else 1.0
        
        atr = preds['atr'].iloc[0]
        close = preds['close'].iloc[0]
        
        # Log specific LSTM metrics if they exist
        if 'gate_weight' in preds.columns:
            logger.info(f"LSTM Context: Gate Weight={preds['gate_weight'].iloc[0]:.4f} (LGB vs LSTM)")

        logger.info(f"Preds: Long={p_long:.2f} (Dir={p_dir_long:.2f}), Short={p_short:.2f} (Dir={p_dir_short:.2f})")

        # 6. Check Warmup
        if any(ob_stats.values()):
            self.real_ob_bars_count += 1
            
        if self.real_ob_bars_count < self.warmup_bars:
            logger.info(f"WARMUP ACTIVE: Trading disabled until {self.warmup_bars} real OB bars reached.")
            return

        # 7. Execute Strategy
        self.execute_logic(p_long, p_short, p_dir_long, p_dir_short, close, atr)
        
        # 8. Management
        self.check_order_expiry()
        self.manage_exits()

    def execute_logic(self, p_long, p_short, p_dir_long, p_dir_short, close, atr):
        thresh = CONF.model.model_threshold
        dir_thresh = CONF.model.direction_threshold
        agg_thresh = CONF.model.aggressive_threshold
        
        # Check Long
        if p_dir_long > agg_thresh:
            logger.info("Signal: Aggressive Buy")
            self._execute_trade("Buy", close, atr, "Market")
        elif p_long > thresh and p_dir_long > dir_thresh:
            logger.info("Signal: Standard Buy (Limit)")
            limit_p = close - (atr * CONF.strategy.base_limit_offset_atr)
            self._execute_trade("Buy", limit_p, atr, "Limit")
            
        # Check Short
        if p_dir_short > agg_thresh:
            logger.info("Signal: Aggressive Sell")
            self._execute_trade("Sell", close, atr, "Market")
        elif p_short > thresh and p_dir_short > dir_thresh:
            logger.info("Signal: Standard Sell (Limit)")
            limit_p = close + (atr * CONF.strategy.base_limit_offset_atr)
            self._execute_trade("Sell", limit_p, atr, "Limit")

    def _execute_trade(self, side, price, atr, order_type, check_debounce=True):
        logger.info(f"Execute {side} {order_type} @ {price:.2f}")
        wallet = self.rest_api.get_wallet_balance("USDT")
        if wallet <= 0: return

        risk_amt = wallet * self.risk_per_trade
        sl_dist = atr * CONF.strategy.stop_loss_atr
        if sl_dist == 0: sl_dist = price * 0.01
        
        raw_qty = risk_amt / sl_dist
        size_qty = self.normalize_qty(raw_qty)
        
        if size_qty <= 0: return
        
        # Check Min Notional
        if (size_qty * price) < self.instrument_info['min_notional']:
            logger.warning(f"Trade value {size_qty*price:.2f} < Min Notional {self.instrument_info['min_notional']}. Skipping.")
            return

        price = self.normalize_price(price)

        sl_price = 0
        tp_price = 0
        if side == "Buy":
            sl_price = price - sl_dist
            tp_price = price + (atr * CONF.strategy.take_profit_atr)
        else:
            sl_price = price + sl_dist
            tp_price = price - (atr * CONF.strategy.take_profit_atr)
            
        sl_price = self.normalize_price(sl_price)
        tp_price = self.normalize_price(tp_price)

        if order_type == "Market":
            success = self.vpm.add_trade(side, price, size_qty, sl_price, tp_price, check_debounce=check_debounce)
            if success:
                self.reconcile_positions()
        else:
            # Limit Order
            # Place on Exchange + Track
            oid = f"LIMIT-{uuid.uuid4().hex[:8]}"
            logger.info(f"Placing Limit {side} {size_qty} @ {price}")
            
            # For Limit, we map Position Index same as Market
            p_idx = 1 if side == "Buy" else 2
            
            res = self.rest_api.place_order(
                self.symbol, side, "Limit", size_qty, price=price, 
                position_idx=p_idx, order_link_id=oid,
                sl=sl_price, tp=tp_price
            )
            
            logger.info(f"DEBUG: Limit Res: {res}")
            
            if 'order_id' in res:
                self.pending_limits[res['order_id']] = {
                    'entry_time': time.time(),
                    'link_id': oid
                }
                logger.info(f"DEBUG: Added to pending_limits. Count: {len(self.pending_limits)}")
            else:
                logger.info("DEBUG: Order ID missing in response")

    def check_order_expiry(self):
        limit_sec = CONF.strategy.time_limit_bars * self.tf_seconds
        now = time.time()
        
        to_remove = []
        for oid, data in self.pending_limits.items():
            if (now - data['entry_time']) > limit_sec:
                logger.info(f"Limit Order {oid} expired. Cancelling.")
                self.rest_api.cancel_order(self.symbol, oid)
                to_remove.append(oid)
        
        for oid in to_remove:
            del self.pending_limits[oid]
            
    def manage_exits(self):
        # Time Exits (VPM)
        max_duration = CONF.strategy.max_holding_bars * self.tf_seconds
        closed = self.vpm.check_time_exits(int(time.time()), max_duration)
        if closed:
            logger.info(f"Time Exits triggered: {closed}. Reconciling.")
            self.reconcile_positions()

    def reconcile_positions(self):
        logger.info("Reconciling Positions (Hedge Mode)...")
        
        # 1. Calculate Targets from Virtual Trades
        target_long = 0.0
        target_short = 0.0
        
        for t in self.vpm.trades:
            if t.side == "Buy": target_long += t.size
            elif t.side == "Sell": target_short += t.size
            
        # 2. Get Actual Positions from Exchange
        positions = self.rest_api.get_positions(self.symbol)
        actual_long = 0.0
        actual_short = 0.0
        
        # Map Bybit Positions to Long/Short buckets
        if positions:
            for p in positions:
                idx = p.get('position_idx', 0)
                sz = p['size']
                side = p['side'] # Buy or Sell
                
                # Hedge Mode: Idx 1=Buy(Long), Idx 2=Sell(Short)
                if idx == 1:
                    actual_long = sz
                elif idx == 2:
                    actual_short = sz
                elif idx == 0:
                    # Fallback for One-Way mode (treating as hedge buckets)
                    if side == 'Buy': actual_long = sz
                    elif side == 'Sell': actual_short = sz
                    
        # 3. Execute Deltas
        # Long Side (Idx 1)
        diff_long = target_long - actual_long
        norm_diff_long = self.normalize_qty(abs(diff_long))
        
        if norm_diff_long > 0:
            qty = norm_diff_long
            oid = f"AUTO-{uuid.uuid4().hex[:8]}"
            if diff_long > 0:
                logger.info(f"Opening Long: {qty}")
                # Attach SL/TP from latest trade for initial protection
                last_buy = next((t for t in reversed(self.vpm.trades) if t.side == "Buy"), None)
                sl = last_buy.stop_loss if last_buy else None
                tp = last_buy.take_profit if last_buy else None
                
                self.rest_api.place_order(self.symbol, "Buy", "Market", qty, position_idx=1, order_link_id=oid, sl=sl, tp=tp)
            else:
                logger.info(f"Reducing Long: {qty}")
                self.rest_api.place_order(self.symbol, "Sell", "Market", qty, reduce_only=True, position_idx=1, order_link_id=oid)

        # Short Side (Idx 2)
        diff_short = target_short - actual_short
        norm_diff_short = self.normalize_qty(abs(diff_short))
        
        if norm_diff_short > 0:
            qty = norm_diff_short
            oid = f"AUTO-{uuid.uuid4().hex[:8]}"
            if diff_short > 0:
                logger.info(f"Opening Short: {qty}")
                # Attach SL/TP
                last_sell = next((t for t in reversed(self.vpm.trades) if t.side == "Sell"), None)
                sl = last_sell.stop_loss if last_sell else None
                tp = last_sell.take_profit if last_sell else None
                
                self.rest_api.place_order(self.symbol, "Sell", "Market", qty, position_idx=2, order_link_id=oid, sl=sl, tp=tp)
            else:
                logger.info(f"Reducing Short: {qty}")
                self.rest_api.place_order(self.symbol, "Buy", "Market", qty, reduce_only=True, position_idx=2, order_link_id=oid)

        # Sync Stops
        # Cancel existing stops/untracked orders, but PRESERVE active Limit Orders
        # AND preserve attached stops if position is still forming (size 0)
        open_orders = self.rest_api.get_open_orders(self.symbol)
        for o in open_orders:
            oid = o['orderId']
            if oid in self.pending_limits:
                continue
            
            # Skip Market Orders (they fill instantly, usually race condition if we see them here)
            if o.get('orderType') == 'Market':
                continue

            # Check if it is a Stop Order
            trig = float(o.get('triggerPrice') or 0)
            if trig > 0:
                # It's a Stop. Preserve if corresponding position is 0 (likely attached to pending)
                # Stop Sell protects Long
                if o['side'] == 'Sell' and actual_long == 0: continue
                # Stop Buy protects Short
                if o['side'] == 'Buy' and actual_short == 0: continue
            
            self.rest_api.cancel_order(self.symbol, oid)
        
        # Fetch price for trigger direction calculation
        current_price = self.rest_api.get_current_price(self.symbol)
        
        stops = self.vpm.get_active_stops()
        for s in stops:
             # Skip if position doesn't exist yet to avoid ReduceOnly rejection
             if s['side'] == 'Sell' and actual_long == 0: continue
             if s['side'] == 'Buy' and actual_short == 0: continue
             
             p_idx = 0
             if s['side'] == 'Sell': p_idx = 1
             elif s['side'] == 'Buy': p_idx = 2
             
             # Calculate Trigger Direction
             t_dir = None
             if current_price > 0:
                 if s['trigger_price'] > current_price:
                     t_dir = 1 # Rise
                 else:
                     t_dir = 2 # Fall
             
             self.rest_api.place_order(
                symbol=self.symbol,
                side=s['side'],
                order_type="Market",
                qty=s['qty'],
                reduce_only=True,
                trigger_price=s['trigger_price'],
                position_idx=p_idx,
                trigger_direction=t_dir
            )

        # Prune Dead Trades
        if current_price > 0:
            closed_ids = self.vpm.prune_dead_trades(current_price)
            if closed_ids:
                logger.info(f"Software SL/TP triggered for {closed_ids}. Re-running reconcile.")
                self.reconcile_positions()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--model-dir", "--model_dir", type=str, required=True, dest="model_dir")
    parser.add_argument("--api-key", "--api_key", type=str, default="", dest="api_key")
    parser.add_argument("--api-secret", "--api_secret", type=str, default="", dest="api_secret")
    parser.add_argument("--testnet", action="store_true")
    parser.add_argument("--timeframe", "--time_frame", type=str, default="5m", dest="timeframe")
    args = parser.parse_args()
    
    # --- MANUAL VALIDATION (Double Safety) ---
    if not args.symbol or not args.model_dir:
        logger.error("CRITICAL: Missing --symbol or --model-dir. Bot cannot start.")
        parser.print_help()
        sys.exit(1)
    
    if not args.api_key:
        try:
            with open("keys.json") as f:
                keys = json.load(f)
                args.api_key = keys.get("api_key")
                args.api_secret = keys.get("api_secret")
        except:
            logger.warning("No API keys found.")

    bot = LiveBot(args)
    bot.run()