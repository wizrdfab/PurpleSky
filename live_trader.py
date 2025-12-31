"""
Production-Grade Live Trading System.
Features: State Persistence, Risk Management, Exchange Reconciliation, Verbose Dashboard.
"""
import os
import time
import json
import joblib
import logging
import traceback
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

from config import CONF, GlobalConfig
from exchange_client import ExchangeClient
from feature_engine import FeatureEngine
from models import ModelManager

# --- Setup Logging ---
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "production_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ProdBot")

class StateManager:
    """Handles persistent state (disk I/O) to survive restarts."""
    def __init__(self, symbol: str):
        self.file_path = Path(f"bot_state_{symbol}.json")
        self.state = {
            "total_pnl": 0.0,
            "daily_pnl": 0.0,
            "last_reset_date": datetime.utcnow().strftime('%Y-%m-%d'),
            "active_orders": {},  # {order_id: {entry_bar: int, side: str}}
            "position_entry_bar": None,
            "last_processed_bar": None
        }
        self.load()

    def load(self):
        if self.file_path.exists():
            try:
                with open(self.file_path, 'r') as f:
                    self.state.update(json.load(f))
                logger.info("State restored from disk.")
            except Exception as e:
                logger.error(f"Failed to load state: {e}")

    def save(self):
        try:
            with open(self.file_path, 'w') as f:
                json.dump(self.state, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def update_pnl(self, realized_pnl: float):
        # Reset daily PnL if new day
        today = datetime.utcnow().strftime('%Y-%m-%d')
        if self.state["last_reset_date"] != today:
            self.state["daily_pnl"] = 0.0
            self.state["last_reset_date"] = today
        
        self.state["total_pnl"] += realized_pnl
        self.state["daily_pnl"] += realized_pnl
        self.save()

class RiskManager:
    """Safety checks before execution."""
    def __init__(self, config: GlobalConfig, state: StateManager):
        self.config = config
        self.state = state

    def can_trade(self, spread_pct: float) -> bool:
        if spread_pct > self.config.live.max_spread_pct:
            logger.warning(f"Risk: Spread too high ({spread_pct:.4%}). No trade.")
            return False
        return True

    def check_drawdown(self, equity: float) -> bool:
        limit = -1 * equity * self.config.live.max_daily_drawdown_pct
        if self.state.state["daily_pnl"] < limit:
            logger.critical(f"KILL SWITCH: Daily Drawdown Limit Hit ({self.state.state['daily_pnl']:.2f}). Stopping.")
            return False
        return True

class LiveDataManager:
    """
    Maintains a rolling window of OHLCV + Orderbook bars.
    Bootstraps history on startup for immediate readiness.
    Accumulates snapshots to build high-fidelity bars.
    """
    def __init__(self, config: GlobalConfig, exchange: ExchangeClient, window_size: int = 500):
        self.config = config
        self.exchange = exchange
        self.window_size = window_size
        self.bars = pd.DataFrame()
        self.ob_buffer = [] # Accumulator for current bar snapshots
        self.current_bar_idx = None
        
        self.tf_seconds = 15 * 60 
        if "5m" in config.features.base_timeframe: self.tf_seconds = 300
        elif "15m" in config.features.base_timeframe: self.tf_seconds = 900
        elif "1h" in config.features.base_timeframe: self.tf_seconds = 3600
        
        self.bootstrap_history()

    def bootstrap_history(self):
        """Fetch historical klines to pre-fill the bar window."""
        logger.info(f"Bootstrapping history for {self.window_size} bars...")
        
        interval_map = {300: "5", 900: "15", 3600: "60", 14400: "240"}
        bybit_interval = interval_map.get(self.tf_seconds, "15")
        
        klines = self.exchange.fetch_kline(interval=bybit_interval, limit=200)
        
        if klines.empty:
            logger.error("Failed to bootstrap klines.")
            return

        df = pd.DataFrame()
        df['datetime'] = pd.to_datetime(klines['timestamp'], unit='s')
        df.set_index('datetime', inplace=True)
        
        df['open'] = klines['open']
        df['high'] = klines['high']
        df['low'] = klines['low']
        df['close'] = klines['close']
        df['volume'] = klines['volume']
        
        df['vol_delta'] = 0.0
        df['vol_buy'] = df['volume'] / 2.0
        df['vol_sell'] = df['volume'] / 2.0
        df['sell_vol'] = df['volume'] / 2.0
        df['trade_count'] = 1 
        df['total_val'] = klines['turnover']
        df['vwap'] = df['total_val'] / df['volume'].replace(0, 1)
        df['taker_buy_ratio'] = 0.5
        
        # Neutral init
        df['ob_spread_mean'] = df['close'] * 0.0001
        df['ob_imbalance_mean'] = 0.0
        df['ob_bid_depth_mean'] = df['volume'] 
        df['ob_ask_depth_mean'] = df['volume']
        df['ob_micro_dev_mean'] = 0.0
        df['ob_micro_dev_std'] = 0.0
        
        self.bars = df
        if not self.bars.empty:
            self.current_bar_idx = self.bars.index[-1]
            
        logger.info(f"History bootstrapped. Current bars: {len(self.bars)}")

    def update(self, ob_snapshot: dict):
        """
        Polls trades and accumulates OB snapshots.
        """
        trades = self.exchange.fetch_recent_trades(limit=100) 
        self._process_new_trades(trades, is_bootstrap=False)
        
        if self.bars.empty: return self.bars
        
        # Check if we moved to a new bar
        last_idx = self.bars.index[-1]
        if self.current_bar_idx != last_idx:
            # New bar started, reset buffer
            self.ob_buffer = []
            self.current_bar_idx = last_idx
            
        if ob_snapshot:
            self._buffer_snapshot(ob_snapshot, last_idx)
            
        return self.bars
        
    def _process_new_trades(self, trades_df: pd.DataFrame, is_bootstrap: bool = False):
        if trades_df.empty: return
        
        resample_rule = '5min' if self.tf_seconds == 300 else '15min'
        df = trades_df.copy()
        
        # Standardize
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('datetime', inplace=True)
        
        side_map = {'Buy': 1, 'Sell': -1}
        df['side_num'] = df['side'].map(side_map).fillna(0)
        df['vol_buy'] = np.where(df['side_num']==1, df['size'], 0)
        df['vol_sell'] = np.where(df['side_num']==-1, df['size'], 0)
        df['dollar_val'] = df['price'] * df['size']
        df['vol_delta_calc'] = df['size'] * df['side_num']
        
        # Aggregate
        ohlcv = df['price'].resample(resample_rule).ohlc()
        agg = df.resample(resample_rule).agg({
            'size': 'sum',
            'vol_buy': 'sum',
            'vol_sell': 'sum',
            'side_num': 'count',
            'dollar_val': 'sum',
            'vol_delta_calc': 'sum'
        }).rename(columns={'size': 'volume', 'side_num': 'trade_count', 'vol_delta_calc':'vol_delta'})

        current_bars = pd.concat([ohlcv, agg], axis=1)
        current_bars['vwap'] = current_bars['dollar_val'] / current_bars['volume'].replace(0, 1)
        current_bars['taker_buy_ratio'] = current_bars['vol_buy'] / current_bars['volume'].replace(0, 1)
        current_bars['sell_vol'] = current_bars['vol_sell']
        
        current_bars.dropna(subset=['close'], inplace=True)

        if is_bootstrap:
            self.bars = current_bars
        else:
            if not self.bars.empty:
                self.bars = self.bars[:-1]
            self.bars = pd.concat([self.bars, current_bars])
            self.bars = self.bars[~self.bars.index.duplicated(keep='last')]
            
        if len(self.bars) > self.window_size:
            self.bars = self.bars.iloc[-self.window_size:]
        
        # Fix deprecated fillna
        self.bars.ffill(inplace=True)
        self.bars.fillna(0, inplace=True)

    def _buffer_snapshot(self, ob, idx):
        bids = ob.get('b', [])
        asks = ob.get('a', [])
        if not bids or not asks: 
            return
        
        bid_depth = sum([float(b[1]) for b in bids[:self.config.data.ob_levels]])
        ask_depth = sum([float(a[1]) for a in asks[:self.config.data.ob_levels]])
        bb = float(bids[0][0])
        ba = float(asks[0][0])
        
        # Micro
        bb_s = float(bids[0][1])
        ba_s = float(asks[0][1])
        micro = (ba * bb_s + bb * ba_s) / (bb_s + ba_s + 1e-9)
        mid = (ba + bb) / 2
        
        snap = {
            'spread': ba - bb,
            'imbalance': (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-9),
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'micro_dev': micro - mid
        }
        self.ob_buffer.append(snap)
        
        # Update Current Bar State
        df_buf = pd.DataFrame(self.ob_buffer)
        self.bars.loc[idx, 'ob_spread_mean'] = df_buf['spread'].mean()
        self.bars.loc[idx, 'ob_imbalance_mean'] = df_buf['imbalance'].mean()
        self.bars.loc[idx, 'ob_bid_depth_mean'] = df_buf['bid_depth'].mean()
        self.bars.loc[idx, 'ob_ask_depth_mean'] = df_buf['ask_depth'].mean()
        self.bars.loc[idx, 'ob_micro_dev_mean'] = df_buf['micro_dev'].mean()
        self.bars.loc[idx, 'ob_micro_dev_std'] = df_buf['micro_dev'].std() if len(df_buf) > 1 else 0.0

class ProductionBot:
    def __init__(self, rank_dir: str):
        self.config = CONF
        
        # 1. API Keys
        key = os.getenv("BYBIT_API_KEY")
        secret = os.getenv("BYBIT_API_SECRET")
        if not key or not secret:
            logger.warning("API Keys not found in ENV. Using Dry Run mode.")
            self.config.live.dry_run = True
            key = "dummy"
            secret = "dummy"
            
        self.exchange = ExchangeClient(key, secret, self.config.data.symbol)
        
        # Fetch Instrument Rules
        self.instr_info = {}
        if not self.config.live.dry_run:
            self.instr_info = self.exchange.fetch_instrument_info()
            logger.info(f"Instrument Rules: {self.instr_info}")
        
        # 2. Components
        self.state = StateManager(self.config.data.symbol)
        self.risk = RiskManager(self.config, self.state)
        self.data = LiveDataManager(self.config, self.exchange)
        self.fe = FeatureEngine(self.config.features)
        self.mm = ModelManager(self.config.model)
        
        # 3. Load Model
        self._load_champion(rank_dir)
        self._print_system_manifest(rank_dir)
        
        self.error_count = 0

    def _load_champion(self, rank_dir):
        path = Path(rank_dir)
        if not path.exists():
            raise FileNotFoundError(f"Model path {path} not found!")
            
        with open(path / "params.json") as f:
            p = json.load(f)
            
        self.config.strategy.base_limit_offset_atr = p['limit_offset_atr']
        self.config.strategy.take_profit_atr = p['take_profit_atr']
        self.config.strategy.stop_loss_atr = p['stop_loss_atr']
        self.config.model.model_threshold = p['model_threshold']
        
        self.mm.model_long = joblib.load(path / "model_long.pkl")
        self.mm.model_short = joblib.load(path / "model_short.pkl")
        self.mm.feature_cols = joblib.load(path / "features.pkl")

    def _print_system_manifest(self, rank_dir):
        """Prints a detailed report of the bot's configuration and health on startup."""
        logger.info("\n" + "="*60)
        logger.info("PRE-FLIGHT SYSTEM MANIFEST")
        logger.info("="*60)
        
        # 1. Environment & Paths
        logger.info(f"[PATHS]")
        logger.info(f"  Symbol:         {self.config.data.symbol}")
        logger.info(f"  Model Folder:   {rank_dir}")
        logger.info(f"  Data Folder:    {self.config.data.data_dir}")
        logger.info(f"  State File:     {self.state.file_path}")
        
        # 2. Model & Features
        logger.info(f"\n[MODEL BRAIN]")
        logger.info(f"  Feature Count:  {len(self.mm.feature_cols)}")
        logger.info(f"  Model Type:     {self.config.model.model_type}")
        logger.info(f"  Threshold:      {self.config.model.model_threshold:.3f}")
        
        # 3. Strategy Parameters
        logger.info(f"\n[STRATEGY]")
        logger.info(f"  Limit Offset:   {self.config.strategy.base_limit_offset_atr:.3f} ATR")
        logger.info(f"  Take Profit:    {self.config.strategy.take_profit_atr:.3f} ATR")
        logger.info(f"  Stop Loss:      {self.config.strategy.stop_loss_atr:.3f} ATR")
        logger.info(f"  Order Timeout:  {self.config.strategy.time_limit_bars} bars")
        
        # 4. Account & Network Health
        logger.info(f"\n[ACCOUNT & NETWORK]")
        if not self.config.live.dry_run:
            t0 = time.time()
            bal = self.exchange.get_wallet_balance()
            latency = (time.time() - t0) * 1000
            pos = self.exchange.get_position()
            
            logger.info(f"  Latency (Ping): {latency:.1f}ms")
            logger.info(f"  USDT Equity:    ${bal.get('equity', 0.0):.2f}")
            logger.info(f"  Available:      ${bal.get('available', 0.0):.2f}")
            logger.info(f"  Current Pos:    {pos}")
        else:
            logger.info("  MODE: DRY RUN (Simulated)")
            logger.info("  Latency: N/A")
            logger.info("  Balance: $10,000.00 (Dummy)")
            
        logger.info("="*60 + "\n")

    def run(self):
        logger.info(f"Bot Started. Dry Run: {self.config.live.dry_run}")
        
        last_poll = 0
        last_reconcile = 0
        
        while True:
            try:
                now = time.time()
                
                # 1. High-Frequency Polling (Every 1s)
                if now - last_poll >= 1.0:
                    self._tick_fast()
                    last_poll = now
                
                # 2. Reconcile (Every 60s)
                if now - last_reconcile >= 60.0:
                    self._reconcile()
                    last_reconcile = now
                    
                # 3. Bar Close Logic
                self._check_bar_close()
                
                # SUCCESS: Reset error counters
                self.error_count = 0
                
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                logger.info("Stopping...")
                break
            except Exception as e:
                err_str = str(e).lower()
                
                # 1. Check for Fatal Errors (Auth/IP)
                if "10003" in err_str or "ip" in err_str or "permission" in err_str or "auth" in err_str:
                    logger.critical(f"FATAL API ERROR: {e}. Checking API Permissions/IP Binding.")
                    break
                
                # 2. Network/Transient Errors -> Exponential Backoff
                self.error_count += 1
                
                # Backoff: 5s, 10s, 20s, 40s... max 60s
                sleep_time = min(5 * (2 ** (self.error_count - 1)), 60)
                
                logger.error(f"Network/API Error ({self.error_count}/{self.config.live.max_api_errors}): {e}")
                logger.info(f"Retrying in {sleep_time}s...")
                
                if self.error_count >= self.config.live.max_api_errors:
                    logger.critical("Max errors exceeded. Connection lost for too long. Shutting down.")
                    break
                    
                time.sleep(sleep_time)
                
                # Attempt to Re-Initialize Client if connection completely broke
                if self.error_count > 2:
                    try:
                        logger.info("Attempting to re-initialize Exchange Client...")
                        key = os.getenv("BYBIT_API_KEY")
                        secret = os.getenv("BYBIT_API_SECRET")
                        if key and secret:
                            self.exchange = ExchangeClient(key, secret, self.config.data.symbol)
                            # Update references
                            self.data.exchange = self.exchange
                    except:
                        pass # Keep retrying loop

    def _tick_fast(self):
        ob = self.exchange.fetch_orderbook()
        self.data.update(ob)

    def _check_bar_close(self):
        if len(self.data.bars) < 60: return

        now = datetime.utcnow()
        # Trigger window
        bar_min = 5 if self.data.tf_seconds == 300 else 15
        is_close_window = (now.minute % bar_min == 0) and (now.second < 10)
        
        if is_close_window:
            df = self.data.bars
            last_closed_time = df.index[-2]
            
            if self.state.state["last_processed_bar"] != str(last_closed_time):
                logger.info(f"\n{'='*20} BAR CLOSE: {last_closed_time} {'='*20}")
                
                self._reconcile()
                
                df_feat = self.fe.calculate_features(df)
                df_pred = self.mm.predict(df_feat)
                latest = df_pred.iloc[-2]
                
                # Measuring Latency for Dashboard
                t0 = time.time()
                bal = self.exchange.get_wallet_balance() if not self.config.live.dry_run else {'equity':10000}
                latency = (time.time() - t0) * 1000
                equity = bal.get('equity', 0.0)
                
                self._log_dashboard(latest, equity, latency)
                
                spread_pct = latest['ob_spread_mean'] / latest['close']
                if self.risk.can_trade(spread_pct) and self.risk.check_drawdown(equity):
                    self._execute_logic(latest)
                
                self.state.state["last_processed_bar"] = str(last_closed_time)
                self.state.save()

    def _log_dashboard(self, row, equity, latency):
        pos_size = self.exchange.get_position() if not self.config.live.dry_run else 0
        
        logger.info(f"--- SYSTEM HEALTH ---")
        logger.info(f"Latency: {latency:.1f}ms | Balance: ${equity:.2f} | Position: {pos_size}")
        
        logger.info(f"--- MARKET STATE ---")
        logger.info(f"Price: {row['close']:.4f} | ATR: {row['atr']:.4f} | Spread: {row['ob_spread_mean']:.5f}")
        logger.info(f"Imbalance Z: {row.get('ob_imbalance_z', 0):.2f} | MicroDev: {row.get('ob_micro_dev_mean', 0):.4f}")
        
        logger.info(f"--- MODEL BRAIN ---")
        logger.info(f"Long Pred:  {row['pred_long']:.3f} (Thresh: {self.config.model.model_threshold:.3f})")
        logger.info(f"Short Pred: {row['pred_short']:.3f}")
        
        if row['pred_long'] > self.config.model.model_threshold:
            logger.info(">>> SIGNAL: BULLISH")
        elif row['pred_short'] > self.config.model.model_threshold:
            logger.info(">>> SIGNAL: BEARISH")
        else:
            logger.info(">>> SIGNAL: NEUTRAL")
        logger.info(f"{ '='*50}\n")

    def _reconcile(self):
        if self.config.live.dry_run: return
        open_orders = self.exchange.get_open_orders()
        active_ids = [o['orderId'] for o in open_orders]
        pos_size = self.exchange.get_position()
        
        tracked = list(self.state.state["active_orders"].keys())
        for oid in tracked:
            if oid not in active_ids:
                if abs(pos_size) > 0:
                    logger.info(f"Order {oid} filled.")
                    if self.state.state["position_entry_bar"] is None:
                        self.state.state["position_entry_bar"] = self.state.state["active_orders"][oid]['idx']
                else:
                    logger.info(f"Order {oid} gone.")
                del self.state.state["active_orders"][oid]
        
        if pos_size == 0:
            self.state.state["position_entry_bar"] = None
        self.state.save()

    def _execute_logic(self, row):
        current_idx = len(self.data.bars)
        
        # 1. Position Timeout
        pos_size = self.exchange.get_position() if not self.config.live.dry_run else 0
        entry_bar = self.state.state["position_entry_bar"]
        
        if abs(pos_size) > 0 and entry_bar is not None:
            elapsed = current_idx - entry_bar
            if elapsed >= self.config.strategy.max_holding_bars:
                logger.warning(f"POSITION TIMEOUT. Closing.")
                side = "Buy" if pos_size > 0 else "Sell"
                if not self.config.live.dry_run:
                    self.exchange.market_close(side, abs(pos_size))
                return

        # 2. Order Timeout
        to_cancel = []
        for oid, info in self.state.state["active_orders"].items():
            if (current_idx - info['idx']) > self.config.strategy.time_limit_bars:
                logger.info(f"Order {oid} timed out. Canceling.")
                to_cancel.append(oid)
        
        if to_cancel and not self.config.live.dry_run:
            self.exchange.cancel_all_orders() 
            self.state.state["active_orders"] = {}
            
        # 3. Place New
        has_orders = len(self.state.state["active_orders"]) > 0
        
        if pos_size == 0 and not has_orders:
            threshold = self.config.model.model_threshold
            if row['pred_long'] > threshold:
                self._place_order(row, "Buy", current_idx)

    def _place_order(self, row, side, idx):
        price = float(row['close']) - (float(row['atr']) * self.config.strategy.base_limit_offset_atr)
        
        # Tick Rounding
        tick_size = self.instr_info.get('tick_size', 0.0001)
        price = round(price / tick_size) * tick_size
        price = round(price, 5) 
        
        # Get Equity for Sizing
        equity = 10000.0 # Default for Dry Run
        if not self.config.live.dry_run:
            bal = self.exchange.get_wallet_balance()
            equity = bal.get('equity', 0.0)
            
        # Calc Size: Risk % of Equity
        # Position Size = (Equity * Risk%) / StopDist%
        stop_atr = self.config.strategy.stop_loss_atr
        risk_pct = self.config.strategy.risk_per_trade
        
        # Stop Distance in Price
        stop_dist_price = float(row['atr']) * stop_atr
        
        if stop_dist_price <= 0 or equity <= 0: return
        
        # Max Dollar Risk (e.g. $10 * 0.01 = $0.10 risk)
        risk_dollars = equity * risk_pct
        
        # Qty = Risk$ / StopDist$
        qty = risk_dollars / stop_dist_price
        
        # Leverage Check (Safety Cap)
        # Don't exceed 5x leverage even if stop is tight
        max_qty_lev = (equity * 5.0) / price
        qty = min(qty, max_qty_lev)
        
        # Step Rounding
        qty_step = self.instr_info.get('qty_step', 0.1)
        qty = int(qty / qty_step) * qty_step
        qty = round(qty, 5)
        
        # Min Qty Check
        min_qty = self.instr_info.get('min_qty', 0.0)
        if qty < min_qty:
            if min_qty > 0:
                logger.info(f"Calculated Qty {qty} < Min {min_qty}. Flooring to Min Qty to enable execution.")
                qty = min_qty
            else:
                logger.warning(f"Could not determine Min Qty. Skipping.")
                return
            
        logger.info(f"SIGNAL: {side} Limit @ {price} (Qty: {qty}) [Eq: ${equity:.2f} Risk: ${risk_dollars:.2f}]")
        
        if not self.config.live.dry_run:
            resp = self.exchange.place_limit_order(side, price, qty)
            if resp:
                oid = resp.get('result', {}).get('orderId')
                if oid:
                    self.state.state["active_orders"][oid] = {'idx': idx, 'side': side}
                    self.state.save()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Liquidity Provision Live Trader")
    default_model = f"models_v2/{CONF.data.symbol}/rank_1"
    parser.add_argument("--model-dir", type=str, default=default_model, help=f"Path to model (default: {default_model})")
    args = parser.parse_args()
    
    logger.info(f"Starting Live Trader for {CONF.data.symbol}")
    bot = ProductionBot(args.model_dir)
    bot.run()