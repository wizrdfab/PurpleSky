import time
import logging
import argparse
import pandas as pd
import numpy as np
import json
import traceback
import signal
import sys
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
        
        # Graceful Shutdown
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

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
            logger.info("No history file found. Fetching from exchange...")
            klines = self.rest_api.get_public_klines(self.symbol, self.timeframe, limit=200)
            if klines:
                self.bars = pd.DataFrame(klines)
                for col in self.model_manager.feature_cols:
                    if col not in self.bars.columns and col not in ['open','high','low','close','volume','timestamp']:
                        self.bars[col] = 0.0
                self.last_bar_time = int(self.bars.iloc[-1]['timestamp'])
        
        self.bars['timestamp'] = self.bars['timestamp'].astype(int)

    def run(self):
        self._load_history()
        
        # Connect WS
        logger.info("Connecting WS...")
        self.ws_api.subscribe_orderbook(self.symbol, self.local_book)
        self.ws_api.subscribe_kline(self.symbol, self.timeframe)
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

    def on_position_update(self, msg):
        # Update Virtual Position Manager? 
        # For now, VPM is master, exchange is slave. We reconcile periodically.
        pass

    def on_execution_update(self, msg):
        # Log fills
        for item in msg.get('data', []):
            logger.info(f"EXECUTION: {item.get('side')} {item.get('execQty')} @ {item.get('execPrice')}")

    def on_bar_close(self, ws_kline):
        logger.info(f"Bar Closing: {datetime.fromtimestamp(int(ws_kline['start'])/1000)}")
        
        # 1. Finalize OB Stats
        ob_stats = self.ob_agg.finalize()
        if not ob_stats:
             # Minimal patch
            ob_stats = {k: 0.0 for k in ['ob_spread_mean', 'ob_micro_dev_std']} 
        
        # 2. Construct Bar
        row = {
            'timestamp': int(ws_kline['start']),
            'open': float(ws_kline['open']),
            'high': float(ws_kline['high']),
            'low': float(ws_kline['low']),
            'close': float(ws_kline['close']),
            'volume': float(ws_kline['volume'])
        }
        row.update(ob_stats)
        
        new_row_df = pd.DataFrame([row])
        self.bars = pd.concat([self.bars, new_row_df], ignore_index=True)
        self.last_bar_time = row['timestamp']
        
        # Save History
        self.bars.to_csv(self.history_file, index=False)
        
        # 3. Check Warmup
        if any(ob_stats.values()):
            self.real_ob_bars_count += 1
            
        if self.real_ob_bars_count < self.warmup_bars:
            logger.info(f"Warmup: {self.real_ob_bars_count}/{self.warmup_bars} real OB bars.")
            return

        # 4. Feature Calc
        df_feats = self.feature_engine.calculate_features(self.bars.iloc[-300:]) 
        current_row = df_feats.iloc[[-1]].copy()
        
        # 5. Predict
        preds = self.model_manager.predict(current_row)
        p_long = preds['pred_long'].iloc[0]
        p_short = preds['pred_short'].iloc[0]
        p_dir_long = preds['pred_dir_long'].iloc[0] if 'pred_dir_long' in preds else 1.0
        p_dir_short = preds['pred_dir_short'].iloc[0] if 'pred_dir_short' in preds else 1.0
        
        atr = current_row['atr'].iloc[0]
        close = current_row['close'].iloc[0]
        
        logger.info(f"Preds: Long={p_long:.2f} (Dir={p_dir_long:.2f}), Short={p_short:.2f} (Dir={p_dir_short:.2f})")
        
        # 6. Execute Strategy
        self.execute_logic(p_long, p_short, p_dir_long, p_dir_short, close, atr)

    def execute_logic(self, p_long, p_short, p_dir_long, p_dir_short, close, atr):
        thresh = CONF.model.model_threshold
        dir_thresh = CONF.model.direction_threshold
        agg_thresh = CONF.model.aggressive_threshold
        
        signal_side = None
        
        if p_dir_long > agg_thresh:
            signal_side = "Buy"
            logger.info("Signal: Aggressive Buy")
        elif p_dir_short > agg_thresh:
            signal_side = "Sell"
            logger.info("Signal: Aggressive Sell")
        elif p_long > thresh and p_dir_long > dir_thresh:
            signal_side = "Buy"
            logger.info("Signal: Standard Buy")
        elif p_short > thresh and p_dir_short > dir_thresh:
            signal_side = "Sell"
            logger.info("Signal: Standard Sell")
            
        if signal_side:
            wallet = self.rest_api.get_wallet_balance("USDT")
            if wallet <= 0: return

            risk_amt = wallet * self.risk_per_trade
            sl_dist = atr * CONF.strategy.stop_loss_atr
            if sl_dist == 0: sl_dist = close * 0.01
            
            size_qty = risk_amt / sl_dist
            size_qty = round(size_qty, 3) 
            
            if size_qty <= 0: return

            if signal_side == "Buy":
                sl_price = close - sl_dist
                tp_price = close + (atr * CONF.strategy.take_profit_atr)
            else:
                sl_price = close + sl_dist
                tp_price = close - (atr * CONF.strategy.take_profit_atr)
                
            success = self.vpm.add_trade(signal_side, close, size_qty, sl_price, tp_price)
            if success:
                self.reconcile_positions()

    def reconcile_positions(self):
        logger.info("Reconciling Positions...")
        target_net = self.vpm.get_net_position()
        
        positions = self.rest_api.get_positions(self.symbol)
        actual_net = 0.0
        if positions:
            for p in positions:
                s = p['size']
                if p['side'] == 'Sell': s = -s
                actual_net += s
        
        diff = target_net - actual_net
        
        if abs(diff) > 0.001: 
            side = "Buy" if diff > 0 else "Sell"
            qty = abs(diff)
            logger.info(f"Drift detected. Target: {target_net}, Actual: {actual_net}. Executing {side} {qty}")
            self.rest_api.place_order(self.symbol, side, "Market", qty)
            
        # Sync Stops
        self.rest_api.cancel_all_orders(self.symbol)
        
        stops = self.vpm.get_active_stops()
        for s in stops:
             self.rest_api.place_order(
                symbol=self.symbol,
                side=s['side'],
                order_type="Market",
                qty=s['qty'],
                reduce_only=True,
                trigger_price=s['trigger_price']
            )

        # Prune Dead Trades
        curr_price = self.rest_api.get_current_price(self.symbol)
        if curr_price > 0:
            closed_ids = self.vpm.prune_dead_trades(curr_price)
            if closed_ids:
                logger.info(f"Software SL/TP triggered for {closed_ids}. Re-running reconcile.")
                self.reconcile_positions()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--api-secret", type=str, default="")
    parser.add_argument("--testnet", action="store_true")
    parser.add_argument("--timeframe", type=str, default="5m")
    args = parser.parse_args()
    
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