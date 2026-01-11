"""
Live Trading V3 - Binance Futures version.
Uses ExchangeClientBinance and Binance Websockets.
"""

import argparse
import json
import logging
import math
import os
import re
import time
import threading
import traceback
import uuid
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from config import CONF
from feature_engine import FeatureEngine
from exchange_client_binance import ExchangeClientBinance as ExchangeClient
from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient

# Reuse existing classes from live_trading_v2 (simulated here for brevity or import them)
# In a real scenario, I would refactor common logic to a base class.
# For now, I will include necessary components or assume they are available.

from live_trading_v2 import (
    BAR_COLUMNS,
    StateStore,
    InstrumentInfo,
    LatencyTracker,
    RollingWindow,
    DriftMonitor,
    HealthMonitor,
    SanityCheck,
    FeatureBaseline,
    TradeBarBuilder,
    OrderbookAggregator,
    BarWindow,
    SafeExchange,
    utc_now_str,
    safe_float,
    safe_int,
    normalize_ts_ms,
    timeframe_to_seconds,
    bar_time_from_ts,
    round_down,
    round_up
)

class LiveTradingV3:
    def __init__(self, args: argparse.Namespace):
        self.logger = self._setup_logger(args.log_level)
        self.config = CONF
        self.config.data.data_dir = Path(args.data_dir)
        self.config.data.symbol = args.symbol
        self.config.data.ob_levels = args.ob_levels
        self.testnet = args.testnet
        self.position_mode = (args.position_mode or "oneway").lower()
        self.hedge_mode = self.position_mode == "hedge"
        
        self.tf_seconds = timeframe_to_seconds(self.config.features.base_timeframe)
        self.window_size = args.window
        
        # Core Components
        self.state = StateStore(Path(f"bot_state_{self.config.data.symbol}_v3.json"), self.config.data.symbol, self.logger)
        self.state.load()
        
        self.exchange = self._init_exchange()
        if not args.dry_run:
            if not self.exchange.startup_check(self.position_mode):
                raise RuntimeError("Startup checks failed.")
        
        self.api = SafeExchange(self.exchange, self.logger)
        self.instrument = InstrumentInfo()
        self.feature_engine = FeatureEngine(self.config.features)
        
        # Load Model
        self.model_dir = Path(args.model_dir)
        self.model_long = joblib.load(self.model_dir / "model_long.pkl")
        self.model_short = joblib.load(self.model_dir / "model_short.pkl")
        self.model_features = joblib.load(self.model_dir / "features.pkl")
        
        self.bar_window = BarWindow(self.tf_seconds, self.config.data.ob_levels, self.window_size, self.logger)
        self.health = HealthMonitor()
        self.sanity = SanityCheck()
        
        self.ws_client = None
        self.ws_trade_queue = deque(maxlen=5000)
        self.ws_ob_queue = deque(maxlen=200)
        self.ws_lock = threading.Lock()
        
        self.running = False
        self.dry_run = args.dry_run
        
        self._refresh_instrument_info(force=True)
        self._bootstrap_data()

    def _setup_logger(self, level):
        logger = logging.getLogger("LiveTradingV3")
        logger.setLevel(getattr(logging, level.upper()))
        if not logger.handlers:
            ch = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        return logger

    def _init_exchange(self) -> ExchangeClient:
        # Load keys from env or file
        api_key = os.getenv("BINANCE_API_KEY", "dummy")
        api_secret = os.getenv("BINANCE_API_SECRET", "dummy")
        return ExchangeClient(api_key, api_secret, self.config.data.symbol, testnet=self.testnet)

    def _refresh_instrument_info(self, force=False):
        info = self.api.get_instrument_info()
        if info:
            self.instrument.min_qty = info['min_qty']
            self.instrument.qty_step = info['qty_step']
            self.instrument.tick_size = info['tick_size']
            self.instrument.min_notional = info['min_notional']

    def _bootstrap_data(self):
        self.logger.info("Bootstrapping historical data from Binance...")
        # Fetch klines
        klines = self.api.fetch_kline(interval=self.config.features.base_timeframe, limit=500)
        self.bar_window.bootstrap_from_klines(klines)
        self.logger.info(f"Bootstrapped {len(self.bar_window.bars)} bars.")

    def start(self):
        self.running = True
        self._init_websockets()
        
        self.main_thread = threading.Thread(target=self._main_loop)
        self.main_thread.start()

    def _init_websockets(self):
        self.logger.info("Initializing Binance Websockets...")
        self.ws_client = UMFuturesWebsocketClient(on_message=self._on_ws_message)
        
        symbol_low = self.config.data.symbol.lower()
        self.ws_client.agg_trade(symbol=symbol_low)
        self.ws_client.partial_book_depth(symbol=symbol_low, level=20, speed=100)

    def _on_ws_message(self, _, msg):
        try:
            data = json.loads(msg)
            event = data.get('e')
            
            with self.ws_lock:
                if event == 'aggTrade':
                    self.ws_trade_queue.append({
                        "timestamp": int(data['T']) / 1000.0,
                        "price": float(data['p']),
                        "size": float(data['q']),
                        "side": "Sell" if data['m'] else "Buy"
                    })
                elif 'b' in data and 'a' in data:
                    # Partial depth
                    self.ws_ob_queue.append({
                        "ts": data.get('E') or int(time.time() * 1000),
                        "b": data['b'],
                        "a": data['a']
                    })
        except Exception as e:
            self.logger.error(f"WS error: {e}")

    def _main_loop(self):
        while self.running:
            try:
                now_ts = time.time()
                
                # 1. Process WS queues
                with self.ws_lock:
                    trades = list(self.ws_trade_queue)
                    self.ws_trade_queue.clear()
                    obs = list(self.ws_ob_queue)
                    self.ws_ob_queue.clear()
                
                # Ingest Orderbook
                for ob in obs:
                    self.bar_window.ingest_orderbook(ob)
                
                # Ingest Trades & check for closed bars
                closed_times = self.bar_window.ingest_trades(trades, now_ts)
                
                for bar_time in closed_times:
                    self._on_bar_closed(bar_time)
                
                time.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Main loop error: {e}")
                traceback.print_exc()
                time.sleep(1)

    def _on_bar_closed(self, bar_time):
        self.logger.info(f"Bar closed: {pd.to_datetime(bar_time, unit='s')}")
        
        # 1. Feature Engineering
        df = self.bar_window.bars.copy()
        features_df = self.feature_engine.generate(df)
        if features_df.empty: return
        
        last_row = features_df.iloc[-1]
        
        # 2. Prediction
        feat_vector = last_row[self.model_features].values.reshape(1, -1)
        prob_long = self.model_long.predict_proba(feat_vector)[0][1]
        prob_short = self.model_short.predict_proba(feat_vector)[0][1]
        
        self.logger.info(f"Preds - Long: {prob_long:.3f}, Short: {prob_short:.3f}")
        
        # 3. Execution Logic (Simple placeholder, should be robust like v2)
        if self.dry_run: return
        
        threshold = 0.6
        if prob_long > threshold:
            self._execute_signal("Buy", last_row['close'])
        elif prob_short > threshold:
            self._execute_signal("Sell", last_row['close'])

    def _execute_signal(self, side, price):
        # Basic execution
        qty = self.instrument.min_qty * 10 # Example
        self.logger.info(f"Executing {side} signal at {price}...")
        self.api.place_market_order(side=side, qty=qty)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--ob_levels", type=int, default=20)
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--testnet", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--position_mode", type=str, default="oneway")
    
    args = parser.parse_args()
    bot = LiveTradingV3(args)
    bot.start()
