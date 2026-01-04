"""
Sofia Lone Champion - Professional Live Trading System.
Optimized for single-model high-fidelity execution.
"""
import os
import time
import json
import joblib
import logging
import signal
import traceback
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from config import CONF, GlobalConfig
from exchange_client import ExchangeClient
from feature_engine import FeatureEngine
from models import ModelManager
from notifier import Notifier

# --- Setup Logging ---
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "lone_champion.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ChampionBot")

class HealthMonitor:
    """Tracks strategy health live to detect Concept Drift."""
    def __init__(self, window=20):
        self.window = window
        self.outcomes = [] 
        self.confidences = [] 
        
    def record_trade(self, pnl: float):
        self.outcomes.append(1 if pnl > 0 else 0)
        if len(self.outcomes) > self.window: self.outcomes.pop(0)
        
    def record_prediction(self, pred_val: float):
        self.confidences.append(pred_val)
        if len(self.confidences) > 12: self.confidences.pop(0)
        
    def check_sentiment(self):
        if not self.confidences: return "CALIBRATING"
        avg_conf = sum(self.confidences) / len(self.confidences)
        state = "NEUTRAL (Healthy)"
        if avg_conf > 0.75: state = "PINNED BULLISH (Trend Risk)"
        elif avg_conf < 0.25: state = "PINNED BEARISH (Trend Risk)"
        return f"{state} | 1H Avg: {avg_conf:.3f}"

    def check_regime(self, df: pd.DataFrame):
        """Detects if Orderbook Structure has drifted from the 24h baseline."""
        if len(df) < 50: return "CALIBRATING"
        try:
            # Monitor Bid Slope (Elasticity) for structural drift
            col = 'ob_bid_slope_mean'
            if col not in df.columns: return "DATA MISSING"
            
            valid_df = df[df[col] > 0]
            if len(valid_df) < 24: return "COLLECTING BASELINE"
            
            baseline_mean = valid_df[col].tail(288).mean()
            baseline_std = valid_df[col].tail(288).std()
            current_mean = valid_df[col].tail(12).mean()
            
            if baseline_std == 0: return "STABLE (Zero Vol)"
            z_drift = (current_mean - baseline_mean) / (baseline_std + 1e-9)
            
            status = "STABLE"
            if abs(z_drift) > 3.0: status = "CRITICAL DRIFT"
            elif abs(z_drift) > 2.0: status = "WARNING (Regime Shift)"
            return f"{status} | Drift Z: {z_drift:.2f}"
        except Exception as e: return f"ERROR: {str(e)[:20]}"

    def check_health(self):
        if len(self.outcomes) < 5: return "CALIBRATING"
        wr = sum(self.outcomes) / len(self.outcomes)
        status = "HEALTHY"
        if wr < 0.40: status = "CRITICAL (Drift Detected)"
        elif wr < 0.50: status = "WARNING (Low WR)"
        return f"{status} | Rolling WR: {wr:.1%}"

class StateManager:
    def __init__(self, symbol: str):
        self.file_path = Path(f"bot_state_{symbol}_champion.json")
        self.state = {
            "total_pnl": 0.0,
            "max_equity": 0.0,
            "active_orders": {},
            "position_entry_time": None,
            "last_processed_bar": None
        }
        self.load()

    def load(self):
        if self.file_path.exists():
            try:
                with open(self.file_path, 'r') as f: self.state.update(json.load(f))
                logger.info("Champion state restored.")
            except: pass

    def save(self):
        """
        Atomic Save to prevent corruption during crash/power loss.
        Works on Linux (POSIX) and Windows.
        """
        tmp_path = self.file_path.with_suffix('.tmp')
        try:
            with open(tmp_path, 'w') as f: 
                json.dump(self.state, f, indent=4)
                # Ensure data is physically on disk before renaming
                f.flush()
                os.fsync(f.fileno())
            
            # Atomic replacement
            os.replace(tmp_path, self.file_path)
        except Exception as e: 
            logger.error(f"Failed to save state: {e}")
        
    def update_pnl(self, pnl):
        self.state["total_pnl"] += pnl
        self.save()
        
    def update_hwm(self, current_equity):
        if current_equity > self.state["max_equity"]:
            self.state["max_equity"] = current_equity
            self.save()

class RiskManager:
    def __init__(self, config: GlobalConfig, state: StateManager, notifier: Notifier):
        self.config, self.state, self.notifier = config, state, notifier
        
    def check_drawdown(self, equity: float) -> bool:
        self.state.update_hwm(equity)
        hwm = self.state.state["max_equity"]
        if hwm > 0:
            dd_pct = (equity - hwm) / hwm
            if dd_pct < -0.10:
                msg = f"!!! KILL SWITCH TRIGGERED !!! Drawdown: {dd_pct:.2%}. Peak: ${hwm}."
                self.notifier.send(msg, "CRITICAL")
                logger.critical(msg)
                return False
        return True

class LiveDataManager:
    def __init__(self, config: GlobalConfig, exchange: ExchangeClient):
        self.config, self.exchange = config, exchange
        self.trade_bars = pd.DataFrame()
        self.ob_bars = pd.DataFrame()
        self.ob_buffer, self.current_bar_idx, self.window_size = [], None, 500
        self.total_snapshots = 0
        self.new_bar_event = False
        self.history_file = Path(f"data_history_{config.data.symbol}.csv")
        
        # Continuity Tracker: Bars collected WITHOUT a gap
        self.continuous_bars = 0
        
        # New Microstructure columns
        self.micro_cols = [
            'ob_spread_mean', 'ob_imbalance_mean', 'ob_bid_depth_mean', 'ob_ask_depth_mean', 
            'ob_micro_dev_mean', 'ob_micro_dev_std',
            'ob_bid_slope_mean', 'ob_ask_slope_mean',
            'ob_bid_integrity_mean', 'ob_ask_integrity_mean'
        ]
        
        # Transient columns (needed for runtime updates but often dropped in history)
        self.transient_cols = ['vol_buy', 'vol_sell', 'vol_delta', 'taker_buy_ratio']
        self.has_medium_term_gap = False
        
        self.load_history()
        self.bootstrap()

    def get_bars(self):
        """Merge Trades and OB data on demand."""
        if self.trade_bars.empty: return pd.DataFrame()
        
        # Left join trades with OB (OB might be sparse or lagged)
        merged = self.trade_bars.join(self.ob_bars, how='left')
        
        # Forward fill OB data (last known state) and fill remaining with 0
        merged.fillna(method='ffill', inplace=True)
        merged.fillna(0, inplace=True)
        return merged

    def load_history(self):
        if self.history_file.exists():
            try:
                df = pd.read_csv(self.history_file, index_col='datetime', parse_dates=True)
                if df.empty: return

                # Check for "Gap": If last bar > 60m old, reset continuity but keep data for baseline     
                last_ts = df.index[-1]
                time_diff_min = (datetime.now(timezone.utc).replace(tzinfo=None) - last_ts).total_seconds() / 60.0

                if time_diff_min > 60.0:
                    logger.warning(f"History gap detected ({time_diff_min:.1f}m). Keeping baseline, but resetting warmup.")
                    self.continuous_bars = 0
                else:
                    logger.info(f"Minor history gap ({time_diff_min:.1f}m). Resuming with existing warmup state.")
                    self.continuous_bars = len(df)
                    if self.continuous_bars < 12:
                        logger.warning(f"Resuming with SHORT history ({self.continuous_bars} bars < 1h). Data may be fragmented.")

                    # Check for gap in the 1h-2h window
                    two_h_ago = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=2)
                    one_h_ago = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1)
                    mid_history = df.loc[two_h_ago:one_h_ago]
                    if mid_history.empty and len(df) > 0:
                         logger.warning("History Gap Detected: No data found in the 1h-2h window. Medium-term features may be unstable.")
                         self.has_medium_term_gap = True

                # Filter for only existing columns
                ob_present = [c for c in self.micro_cols if c in df.columns]
                trade_cols = [c for c in df.columns if c not in self.micro_cols]
                
                self.trade_bars = df[trade_cols].copy()
                self.ob_bars = df[ob_present].copy()
                
                # Schema Migration: Fill missing micro columns with 0
                for col in self.micro_cols:
                    if col not in self.ob_bars.columns:
                        self.ob_bars[col] = 0.0
                        
                # Schema Migration: Fill missing transient columns (vol_buy, etc.)
                if not self.trade_bars.empty:
                    for col in self.transient_cols:
                        if col not in self.trade_bars.columns:
                            self.trade_bars[col] = 0.0
                
                if not self.trade_bars.empty: self.current_bar_idx = self.trade_bars.index[-1]
                logger.info(f"Loaded {len(df)} bars from history for {self.config.data.symbol}. Continuous: {self.continuous_bars}")
            except Exception as e: logger.error(f"Failed to load history: {e}")

    def save_history(self):
        try:
            df = self.get_bars()
            if not df.empty:
                save_df = df.iloc[-self.window_size:]
                save_df.to_csv(self.history_file)
        except Exception as e: logger.error(f"Failed to save history: {e}")

    def bootstrap(self):
        """
        Deep Bootstrap: Fetch ~30 days of history for Macro Features using pagination.
        """
        # 30 days * 24h * 12 bars = 8640. We target ~10,000 bars.
        target_bars = 10000
        
        # Increase window size to hold this history
        if self.window_size < target_bars:
            self.window_size = target_bars
            
        if len(self.trade_bars) >= target_bars: 
            return

        logger.info(f"Bootstrapping {self.config.data.symbol} history (Target: 30 days / {target_bars} bars)...")
        
        fetched_dfs = []
        limit = 200 # Safer limit
        current_end = None # Start from Now
        
        for i in range(60): # 60 * 200 = 12000 bars
            try:
                # Prepare kwargs
                kwargs = {}
                if current_end is not None:
                    kwargs['end'] = int(current_end) # Bybit expects ms timestamp
                
                # logger.info(f"Fetching batch {i} (End: {current_end})...")
                klines = self.exchange.fetch_kline(interval="5", limit=limit, **kwargs)
                if klines.empty: 
                    logger.warning(f"Batch {i} returned empty. Stopping bootstrap.")
                    break
                
                df = pd.DataFrame()
                df['datetime'] = pd.to_datetime(klines['timestamp'], unit='s')
                # Fix: Use .values to avoid index alignment mismatch (DatetimeIndex vs RangeIndex)
                df['open'] = klines['open'].values
                df['high'] = klines['high'].values
                df['low'] = klines['low'].values
                df['close'] = klines['close'].values
                df['volume'] = klines['volume'].values
                
                df.set_index('datetime', inplace=True)
                
                # Fill missing/synthetic columns
                df['vol_delta'], df['vol_buy'], df['vol_sell'], df['sell_vol'], df['trade_count'] = 0.0, df['volume'].values/2, df['volume'].values/2, df['volume'].values/2, 100
                df['total_val'], df['taker_buy_ratio'] = klines['turnover'].values, 0.5
                df['vwap'] = (klines['turnover'].values / klines['volume'].values.clip(min=1e-9))
                
                # Fix: Explicitly set dollar_val (Turnover) so FeatureEngine doesn't see 0.0s
                df['dollar_val'] = klines['turnover'].values
                
                # DEBUG: Log shape before filter
                rows_before = len(df)
                
                # Filter out zero-price bars (Bybit sometimes returns empty/zero klines)
                df = df[df['close'] > 0].copy()
                rows_after = len(df)
                
                if rows_after < rows_before:
                    logger.warning(f"Batch {i}: Dropped {rows_before - rows_after} zero-price rows.")

                if df.empty: 
                    logger.warning(f"Batch {i}: Data became empty after filtering!")
                    pass 
                else:
                    fetched_dfs.append(df)
                
                # Update cursor: Oldest timestamp in this batch becomes the END for the next batch
                # timestamp is in seconds, convert to ms
                oldest_ts = klines['timestamp'].min()
                current_end = oldest_ts * 1000
                
                # Safety: If we reached back far enough (e.g. 35 days), stop
                # But loop limit 12 is safe enough.
                time.sleep(0.1) # Rate limit kindness

            except Exception as e:
                logger.error(f"Bootstrap batch {i} failed: {e}")
                break
        
        if not fetched_dfs: 
            logger.warning("Bootstrap complete but NO data was fetched/kept.")
            return
        
        # Concatenate all batches (they are chronological segments)
        full_history = pd.concat(fetched_dfs)
        full_history.sort_index(inplace=True)
        # Remove duplicates just in case pagination overlapped
        full_history = full_history[~full_history.index.duplicated(keep='last')]
        
        logger.info(f"Bootstrap complete. Fetched {len(full_history)} bars.")

        # Merge with existing local history
        if not self.trade_bars.empty:
            # combine_first prefers self.trade_bars (local) over full_history (remote)
            self.trade_bars = self.trade_bars.combine_first(full_history)
        else:
            self.trade_bars = full_history
            
        self.trade_bars.sort_index(inplace=True)
        # Ensure we keep the full window
        self.trade_bars = self.trade_bars.iloc[-self.window_size:]
        
        if not self.trade_bars.empty: self.current_bar_idx = self.trade_bars.index[-1]

    def get_quality_report(self) -> str:
        """
        Diagnostic: Check data health for feature calculation.
        """
        if self.trade_bars.empty: return "Data: EMPTY"
        
        n = len(self.trade_bars)
        
        # 1. Macro Readiness (Target: 30 days = 8640 bars)
        macro_target = 8640
        macro_pct = min(100.0, (n / macro_target) * 100)
        
        # 2. Orderbook Density (Last 2h = 24 bars)
        target_window = 24
        if self.ob_bars.empty:
            ob_density = 0.0
        else:
            # Focus on the 2-hour window
            recent_ob = self.ob_bars.tail(target_window)
            
            # Robust Check: Real OB data always has a Spread > 0.
            if 'ob_spread_mean' in recent_ob.columns:
                valid_count = (recent_ob['ob_spread_mean'] > 0.0).sum()
                # Denominator is the TARGET window, not the current length
                ob_density = (valid_count / target_window * 100)
            else:
                ob_density = 0.0
        
        # Cap at 100%
        ob_density = min(100.0, ob_density)
        
        return f"Data Health: [Bars: {n} | Macro: {macro_pct:.0f}% | OB-Density(2h): {ob_density:.0f}%]"

    def update(self, ob_snapshot: dict) -> pd.DataFrame:
        trades = self.exchange.fetch_recent_trades(limit=1000)
        self._process_trades(trades)
        
        if self.trade_bars.empty: return pd.DataFrame()
        last_idx = self.trade_bars.index[-1]
        
        if self.current_bar_idx is None: self.current_bar_idx = last_idx
        if self.current_bar_idx != last_idx:
            self.new_bar_event = True
            self.ob_buffer = []
            self.current_bar_idx = last_idx
            self.continuous_bars += 1
            
        if ob_snapshot: 
            self._buffer_snapshot(ob_snapshot, last_idx)
            self.total_snapshots += 1
        return self.get_bars()

    def _process_trades(self, trades_df):
        if trades_df.empty: return
        df = trades_df.copy()
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('datetime', inplace=True)
        side_map = {'Buy': 1, 'Sell': -1}
        df['side_num'] = df['side'].map(side_map).fillna(0)
        df['vol_buy'], df['vol_sell'], df['dollar_val'] = np.where(df['side_num']==1, df['size'], 0), np.where(df['side_num']==-1, df['size'], 0), df['price'] * df['size']
        df['vol_delta_calc'] = df['size'] * df['side_num']
        resample_rule = '5min'
        ohlcv = df['price'].resample(resample_rule).ohlc()
        agg = df.resample(resample_rule).agg({'size':'sum','vol_buy':'sum','vol_sell':'sum','side_num':'count','dollar_val':'sum','vol_delta_calc':'sum'}).rename(columns={'size':'volume','side_num':'trade_count','vol_delta_calc':'vol_delta'})
        new_bars = pd.concat([ohlcv, agg], axis=1)
        new_bars['vwap'] = new_bars['dollar_val'] / new_bars['volume'].replace(0, 1)
        new_bars['taker_buy_ratio'] = new_bars['vol_buy'] / new_bars['volume'].replace(0, 1)
        new_bars['sell_vol'] = new_bars['vol_sell']
        new_bars.dropna(subset=['close'], inplace=True)
        
        if not self.trade_bars.empty:
            last_idx = self.trade_bars.index[-1]
            new_idx = new_bars.index[-1]
            if new_idx == last_idx:
                # Update in place
                for col in new_bars.columns:
                    self.trade_bars.loc[last_idx, col] = new_bars.loc[new_idx, col]
            else:
                self.trade_bars = pd.concat([self.trade_bars, new_bars])
        else:
            self.trade_bars = new_bars
            
        self.trade_bars = self.trade_bars[~self.trade_bars.index.duplicated(keep='last')]
        if len(self.trade_bars) > self.window_size: self.trade_bars = self.trade_bars.iloc[-self.window_size:]
        self.trade_bars.ffill(inplace=True)

    def _buffer_snapshot(self, ob, idx):
        bids, asks = ob.get('b', []), ob.get('a', [])
        if not bids or not asks: return
        
        # Consistent depth levels
        depth_lvls = self.config.data.ob_levels
        bids_slice = bids[:depth_lvls]
        asks_slice = asks[:depth_lvls]
        
        bid_depth = sum([float(b[1]) for b in bids_slice])
        ask_depth = sum([float(a[1]) for a in asks_slice])
        
        bb, ba = float(bids[0][0]), float(asks[0][0])
        bb_s, ba_s = float(bids[0][1]), float(asks[0][1])
        
        # 1. Spread & Micro-Price
        spread = ba - bb
        mid = (ba + bb) / 2
        micro = (ba * bb_s + bb * ba_s) / (bb_s + ba_s + 1e-9)
        micro_dev = micro - mid
        
        # 2. Imbalance
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-9)
        
        # 3. Slope (Gradient)
        bid_slope = 0.0
        if bid_depth > 0 and len(bids_slice) > 1:
            bid_slope = (bb - float(bids_slice[-1][0])) / bid_depth
            
        ask_slope = 0.0
        if ask_depth > 0 and len(asks_slice) > 1:
            ask_slope = (float(asks_slice[-1][0]) - ba) / ask_depth
            
        # 4. Integrity (Intention)
        bid_integrity = bb_s / bid_depth if bid_depth > 0 else 0
        ask_integrity = ba_s / ask_depth if ask_depth > 0 else 0

        self.ob_buffer.append({
            'spread': spread,
            'imbalance': imbalance,
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'micro_dev': micro_dev,
            'bid_slope': bid_slope,
            'ask_slope': ask_slope,
            'bid_integrity': bid_integrity,
            'ask_integrity': ask_integrity
        })
        
        df_buf = pd.DataFrame(self.ob_buffer)
        
        # Map to DataFrame columns
        vals = [
            df_buf['spread'].mean(), 
            df_buf['imbalance'].mean(), 
            df_buf['bid_depth'].mean(), 
            df_buf['ask_depth'].mean(), 
            df_buf['micro_dev'].mean(),
            df_buf['bid_slope'].mean(),
            df_buf['ask_slope'].mean(),
            df_buf['bid_integrity'].mean(),
            df_buf['ask_integrity'].mean()
        ]
        
        # The target columns in self.ob_bars
        target_cols = [
            'ob_spread_mean', 'ob_imbalance_mean', 'ob_bid_depth_mean', 'ob_ask_depth_mean', 
            'ob_micro_dev_mean', 'ob_bid_slope_mean', 'ob_ask_slope_mean', 
            'ob_bid_integrity_mean', 'ob_ask_integrity_mean'
        ]
        
        if idx not in self.ob_bars.index:
            # Create new row (initialized with zeros)
            new_row = pd.DataFrame(0.0, columns=self.micro_cols, index=[idx])
            self.ob_bars = pd.concat([self.ob_bars, new_row])
        
        # Update row
        self.ob_bars.loc[idx, target_cols] = vals
        self.ob_bars.loc[idx, 'ob_micro_dev_std'] = df_buf['micro_dev'].std() if len(df_buf) > 1 else 0.0
            
        if len(self.ob_bars) > self.window_size: self.ob_bars = self.ob_bars.iloc[-self.window_size:]

class SanityCheck:
    """
    Data Scientist Guardrails.
    Checks for statistical anomalies that indicate data corruption.
    """
    def check(self, row: pd.Series) -> List[str]:
        warnings = []
        
        # 1. Z-Score Explosion Check (Variance Collapse)
        z_cols = [c for c in row.index if c.endswith('_z')]
        for c in z_cols:
            val = row[c]
            if abs(val) > 10.0:
                warnings.append(f"EXTREME Z-SCORE: {c} = {val:.2f} (Likely Zero Variance in History)")

        # 2. VWAP Dislocation Check (Missing Price/Turnover)
        vwap_cols = [c for c in row.index if 'vwap' in c and 'dist' in c]
        for c in vwap_cols:
            val = row[c]
            if abs(val) > 20.0:
                warnings.append(f"VWAP DISLOCATION: {c} = {val:.2f} ATRs (Possible Missing DollarVal)")

        # 3. Synthetic Data Leak (The "0.5" Ratio)
        if 'taker_buy_ratio' in row.index:
            if row['taker_buy_ratio'] == 0.5:
                warnings.append("SYNTHETIC DATA: Taker Buy Ratio is exactly 0.5 (Bootstrap Artifact)")
                
        # 4. Missing Microstructure (Zero Elasticity)
        if 'ob_bid_elasticity' in row.index:
            if row['ob_bid_elasticity'] == 0.0:
                 warnings.append("MISSING ORDERBOOK: Elasticity is 0.0 (No Depth Data)")

        return warnings

class ChampionBot:
    def __init__(self, model_root: str):
        self.config = CONF
        self.is_placing_order = False
        self.last_order_ts = 0
        key, secret = os.getenv("BYBIT_API_KEY"), os.getenv("BYBIT_API_SECRET")
        if not key or not secret:
            self.config.live.dry_run = True
            key, secret = "dummy", "dummy"
        else: self.config.live.dry_run = False
            
        self.exchange = ExchangeClient(key, secret, self.config.data.symbol)
        if not self.config.live.dry_run:
            if not self.exchange.startup_check():
                logger.critical("Startup Checks Failed. Exiting...")
                exit(1)
            self.exchange.set_leverage(10)
            
        self.instr_info = self.exchange.fetch_instrument_info() if not self.config.live.dry_run else {'min_qty':1.0,'qty_step':0.1,'tick_size':0.0001}
        self.state = StateManager(self.config.data.symbol)
        self.notifier = Notifier(prefix=f"[{self.config.data.symbol}]")
        self.risk = RiskManager(self.config, self.state, self.notifier)
        self.health = HealthMonitor()
        self.sanity = SanityCheck()
        self.data = LiveDataManager(self.config, self.exchange)
        self.fe = FeatureEngine(self.config.features)
        
        # Load Lone Champion
        self._load_champion(model_root)
        self._register_signals()
        
        # --- Reality Reconciliation ---
        if not self.config.live.dry_run:
            bal = self.exchange.get_wallet_balance()
            self.start_equity = bal.get('equity', 0.0)
            self.start_pnl = self.state.state.get("total_pnl", 0.0)
            logger.info(f"Reality Baseline: Start Equity: ${self.start_equity:.2f} | Start PnL: ${self.start_pnl:.2f}")
        else:
            self.start_equity = 10000.0
            self.start_pnl = 0.0

        self._print_manifest(model_root)
        
        # Initial Notification
        status = "WARMING UP" if self.data.continuous_bars < 24 else "ACTIVE"
        gap_msg = " | !! MEDIUM-TERM GAP DETECTED !!" if self.data.has_medium_term_gap else ""
        msg = f"Champion Bot Started [{status}: {self.data.continuous_bars}/24 bars]{gap_msg}"
        
        level = "WARNING" if (status == "WARMING UP" or self.data.has_medium_term_gap) else "INFO"
        self.notifier.send(msg, level)

    def _register_signals(self):
        """Register signal handlers for graceful shutdown on Linux/Docker."""
        signal.signal(signal.SIGINT, self._handle_exit_signal)
        try:
            signal.signal(signal.SIGTERM, self._handle_exit_signal)
        except AttributeError:
            pass # SIGTERM not available on Windows

    def _handle_exit_signal(self, signum, frame):
        logger.warning(f"Received Termination Signal ({signum}). Exiting safely...")
        raise KeyboardInterrupt

    def _load_champion(self, root):
        root_path = Path(root)
        rank_path = root_path / "rank_1"
        if not rank_path.exists(): raise Exception("No champion found in rank_1!")
        logger.info(f"Loading Champion from {rank_path}...")
        with open(rank_path / "params.json") as f: p = json.load(f)
        self.champion = {
            'ml': joblib.load(rank_path / "model_long.pkl"),
            'ms': joblib.load(rank_path / "model_short.pkl"),
            'features': joblib.load(rank_path / "features.pkl"),
            'params': p
        }

    def _print_manifest(self, root):
        logger.info("\n" + "="*60 + f"\nLONE CHAMPION STARTUP MANIFEST\n" + "="*60)
        logger.info(f"Symbol: {self.config.data.symbol} | Model Root: {root}")
        logger.info(f"Clock Drift: {self.exchange.check_time_sync():.1f}ms")
        
        active_orders = self.state.state.get("active_orders", {})
        logger.info(f"State: {len(active_orders)} Active Orders Restored")
        
        if not self.config.live.dry_run:
            bal = self.exchange.get_wallet_balance()
            logger.info(f"Account: ${bal['equity']:.2f} Equity | ${bal['available']:.2f} Available")
        else: logger.info("MODE: DRY RUN")
        logger.info("="*60 + "\n")

    def run(self):
        last_poll, last_reconcile, last_status, last_risk_check, last_sync = 0, 0, 0, 0, time.time()
        while True:
            try:
                now = time.time()
                # 1. Periodic Time Sync (Every 1 Hour)
                if now - last_sync >= 3600.0:
                    drift = self.exchange.check_time_sync()
                    logger.info(f"Periodic Time Sync: Current Drift: {drift:.1f}ms")
                    last_sync = now

                if now - last_risk_check >= 5.0:
                    if not self._monitor_risk(): break
                    last_risk_check = now
                if now - last_poll >= 1.0:
                    self.data.update(self.exchange.fetch_orderbook())
                    last_poll = now
                if now - last_status >= 30.0:
                    # Use trade_bars for quick stats (faster than get_bars merge)
                    curr_p = self.data.trade_bars['close'].iloc[-1] if not self.data.trade_bars.empty else 0.0
                    pos_s = self.exchange.get_position() if not self.config.live.dry_run else 0.0
                    
                    # Reality Check
                    pnl_str = ""
                    if not self.config.live.dry_run:
                        bal = self.exchange.get_wallet_balance()
                        curr_equity = bal.get('equity', 0.0)
                        curr_cash = bal.get('total_balance', 0.0)
                        curr_total_pnl = self.state.state.get("total_pnl", 0.0)
                        
                        # Friction = Cash - (StartEquity + RealizedPnLDelta)
                        # This stays stable even when a trade is open
                        expected_cash = self.start_equity + (curr_total_pnl - self.start_pnl)
                        friction = curr_cash - expected_cash
                        
                        # Get Unrealized PnL for the heartbeat
                        pos_details = self.exchange.get_position_details()
                        unreal = pos_details.get('unrealized_pnl', 0.0)
                        order_count = len(self.state.state.get("active_orders", {}))
                        
                        pnl_str = f"| Eq: ${curr_equity:.2f} | Unreal: ${unreal:.2f} | Friction: {friction:.2f} | Orders: {order_count}"
                    
                    data_health = self.data.get_quality_report()
                    logger.info(f"[HEARTBEAT] Price: {curr_p:.4f} | Pos: {pos_s} {pnl_str} | {data_health} | Health: {self.health.check_health()}")
                    last_status = now
                if now - last_reconcile >= 10.0:
                    self._reconcile()
                    last_reconcile = now
                self._check_bar_close()
                time.sleep(0.1)
            except KeyboardInterrupt: 
                self.notifier.send("Bot Stopped by User", "WARNING")
                break
            except Exception as e:
                err_msg = f"Loop Error: {e}"
                logger.error(f"{err_msg}\n{traceback.format_exc()}")
                self.notifier.send(err_msg, "CRITICAL")
                time.sleep(5)

    def _monitor_risk(self) -> bool:
        if self.config.live.dry_run: return True
        try:
            bal = self.exchange.get_wallet_balance()
            equity = bal.get('equity', 0.0)
            if not self.risk.check_drawdown(equity):
                self._emergency_shutdown()
                return False
            return True
        except Exception as e:
            logger.error(f"Risk Monitor Error: {e}")
            return True

    def _emergency_shutdown(self):
        msg = "!!! EMERGENCY SHUTDOWN PROTOCOL INITIATED !!!"
        self.notifier.send(msg, "CRITICAL")
        logger.critical(msg)
        for i in range(1, 11):
            try:
                self.exchange.cancel_all_orders()
                self.exchange.close_all_positions()
                time.sleep(2.0)
                if self.exchange.get_position() == 0:
                    logger.info("SUCCESS: Account Flat. Exiting.")
                    os._exit(0)
            except Exception as e: logger.error(f"Retry {i} failed: {e}")
        while True:
            try:
                import winsound
                winsound.Beep(1000, 500); winsound.Beep(500, 500)
            except: print('\a')
            self.exchange.close_all_positions()
            time.sleep(5)

    def _check_bar_close(self):
        if self.data.new_bar_event:
            self.data.new_bar_event = False
            self.data.save_history() # Save immediately to persist warmup progress
            
            # Use get_bars() for the full picture
            full_bars = self.data.get_bars()
            
            # 1. Price History Gate (Technical Indicators)
            if len(full_bars) < 60:
                logger.warning(f"Price History too short ({len(full_bars)} bars). Waiting for 60.")
                return

            # 2. Microstructure Gate (Z-Score Stability)
            # We need ~24 bars (2 hours) of CONTINUOUS data for rolling Z-scores to stabilize.
            # If we have a gap, we must wait for new warmup.
            if self.data.continuous_bars < 24:
                logger.info(f"Microstructure Warmup: {self.data.continuous_bars}/24 continuous bars collected. Trading Paused.")
                return
                
            last_closed_time = full_bars.index[-2]
            
            logger.info(f"\n>>> CHAMPION DECISION FOR BAR: {last_closed_time} <<<")
            regime = self.health.check_regime(full_bars)
            logger.info(f"[DIAG] Bars: {len(full_bars)} | Continuous: {self.data.continuous_bars} | Regime: {regime} | Sentiment: {self.health.check_sentiment()}")
            
            self._reconcile()
            df_feat = self.fe.calculate_features(full_bars)
            self._execute_champion(df_feat.iloc[-2])
            
            self.state.state["last_processed_bar"] = str(last_closed_time)
            self.state.save()

    def _execute_champion(self, row):
        c = self.champion
        feature_names = c['features']
        
        # Log Feature Inputs (Transparency)
        if logger.isEnabledFor(logging.INFO):
            msg = f"\n[FEATURES] Input Vector ({len(feature_names)} features):\n"
            # Format in columns of 3
            for i in range(0, len(feature_names), 3):
                chunk = feature_names[i:i+3]
                line = " | ".join([f"{name}: {row[name]:.6f}" for name in chunk])
                msg += f"  {line}\n"
            logger.info(msg)
            
        # Data Scientist Guardrails
        sanity_warnings = self.sanity.check(row)
        for w in sanity_warnings:
            logger.warning(f"[DATA INTEGRITY] {w}")

        X = row[feature_names].values.reshape(1, -1)
        pred_l, pred_s = c['ml'].predict(X)[0], c['ms'].predict(X)[0]
        thresh = c['params']['model_threshold']
        self.health.record_prediction(max(pred_l, pred_s))
        logger.info(f"Verdict: Long {pred_l:.3f} | Short {pred_s:.3f} | Thresh {thresh:.3f}")
        
        if pred_l > thresh: self._place_trade(row, "Buy", c['params'])
        elif pred_s > thresh: self._place_trade(row, "Sell", c['params'])

    def _place_trade(self, row, side, p):
        if self.exchange.get_position() != 0 or self.state.state["active_orders"]: return
        if self.is_placing_order: return
        if time.time() - self.last_order_ts < 2.0: return
        
        self.is_placing_order = True
        try:
            price = row['close'] - (row['atr'] * p['limit_offset_atr']) if side == "Buy" else row['close'] + (row['atr'] * p['limit_offset_atr'])
            tick = self.instr_info.get('tick_size', 0.0001)
            price = round(round(price / tick) * tick, 5)
            bal = self.exchange.get_wallet_balance() if not self.config.live.dry_run else {'equity':10000}
            risk_d = bal['equity'] * self.config.strategy.risk_per_trade
            qty = risk_d / (row['atr'] * p['stop_loss_atr'])
            qty_n = 6.0 / price
            final_qty = max(qty, self.instr_info.get('min_qty', 1.0), qty_n)
            max_lev_qty = (bal['equity'] * 5.0) / price
            final_qty = min(final_qty, max_lev_qty)
            step = self.instr_info.get('qty_step', 0.1)
            final_qty = round(int(final_qty/step)*step, 5)
            tp_p = price + (row['atr'] * p['take_profit_atr']) if side == "Buy" else price - (row['atr'] * p['take_profit_atr'])
            sl_p = price - (row['atr'] * p['stop_loss_atr']) if side == "Buy" else price + (row['atr'] * p['stop_loss_atr'])
            tp_p = round(round(tp_p / tick) * tick, 5); sl_p = round(round(sl_p / tick) * tick, 5)
            
            msg = f"!!! TRADING: {side} @ {price} [TP: {tp_p} | SL: {sl_p}] Notional: ${final_qty*price:.2f} !!!"
            logger.info(msg)
            
            if not self.config.live.dry_run:
                resp = self.exchange.place_limit_order(side, price, final_qty, tp=tp_p, sl=sl_p)
                if resp: 
                    self.last_order_ts = time.time()
                    self.notifier.send(msg, "SUCCESS")
                    # Store timestamp (ISO format) for robust time tracking
                    ts = row.name.isoformat() if hasattr(row.name, 'isoformat') else datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
                    self.state.state["active_orders"][resp['result']['orderId']] = {
                        'created_at': ts, 
                        'side': side,
                        'price': price,
                        'qty': final_qty
                    }
                    self.state.save()
        finally:
            self.is_placing_order = False

    def _reconcile(self):
        """
        Expert Reconciliation:
        1. Syncs Local Orders <-> Exchange Orders (Strict Mode)
        2. Manages Position Timeouts (Local Time Authority)
        3. Detects PnL
        """
        if self.config.live.dry_run: return
        
        # A. Fetch State
        remote_orders_list = self.exchange.get_open_orders()
        pos_details = self.exchange.get_position_details()
        pos_size = pos_details.get('size', 0.0)
        abs_size = pos_details.get('abs_size', 0.0)
        
        # Build Remote Map for O(1) Access
        # Key: orderId -> Value: Order Dict
        remote_orders = {o['orderId']: o for o in remote_orders_list}
        local_orders = self.state.state["active_orders"]
        
        # authoritative current time synced to Bybit
        current_time = self.exchange.get_exchange_now()
        tf_seconds = 300 # 5m
        
        # B. SYNC LOCAL -> REMOTE (Clean Zombies & Update Drift)
        # We iterate a COPY of keys to allow deletion
        for oid in list(local_orders.keys()):
            local_info = local_orders[oid]
            
            # Case 1: Order exists on Exchange
            if oid in remote_orders:
                remote_info = remote_orders[oid]
                
                # Check for "Drift" (External Modification)
                # Note: API returns strings, we compare as floats
                rem_price = float(remote_info.get('price', 0))
                loc_price = float(local_info.get('price', 0))
                
                if loc_price > 0 and abs(rem_price - loc_price) > self.instr_info['tick_size']:
                    logger.warning(f"Order {oid} drift detected! Local: {loc_price} -> Remote: {rem_price}. Syncing.")
                    local_orders[oid]['price'] = rem_price
                    # We could also check Qty here
                
                # Check Expiry (Time Limit)
                if 'created_at' in local_info:
                    try:
                        created_at = pd.to_datetime(local_info['created_at'])
                        age_seconds = (current_time - created_at).total_seconds()
                        limit_seconds = self.config.strategy.time_limit_bars * tf_seconds
                        
                        if age_seconds > limit_seconds:
                            logger.info(f"Order {oid} expired ({age_seconds:.0f}s). Cancelling.")
                            self.exchange.cancel_order(order_id=oid)
                            # Will be removed from local state in next pass or below
                            # For now, we can optimistically remove it or wait for next sync
                    except Exception: pass

            # Case 2: Order MISSING on Exchange (Zombie / Filled / Cancelled)
            else:
                # If we have a position now, it likely filled.
                # If not, it was cancelled or rejected.
                # Either way, it's no longer "Active"
                logger.info(f"Order {oid} missing on exchange. Removing from local state.")
                del local_orders[oid]

        # C. SYNC REMOTE -> LOCAL (Adopt Orphans)
        for oid, rem_info in remote_orders.items():
            if oid not in local_orders:
                try:
                    c_time = int(rem_info.get('createdTime', 0))
                    # Fallback to current synced time if createdTime is missing
                    ts_iso = datetime.fromtimestamp(c_time / 1000.0, tz=timezone.utc).replace(tzinfo=None).isoformat() if c_time > 0 else self.exchange.get_exchange_now().isoformat()
                    
                    logger.info(f"Adopting Orphan Order {oid} @ {rem_info.get('price')}")
                    local_orders[oid] = {
                        'created_at': ts_iso,
                        'side': rem_info.get('side'),
                        'price': float(rem_info.get('price', 0)),
                        'qty': float(rem_info.get('qty', 0)),
                        'adopted': True
                    }
                except Exception as e:
                    logger.error(f"Failed to adopt {oid}: {e}")
        
        # D. MANAGE POSITION (Timeouts)
        if abs_size > 0:
            try:
                # Detect New Entry
                if self.state.state.get("last_pos_size", 0) == 0:
                    logger.info("Position Detected. initializing Entry Time tracker.")
                    self.state.state["position_entry_time"] = self.exchange.get_exchange_now().isoformat()
                    self.state.save()
                
                # Determine Entry Time
                entry_time = None
                if self.state.state.get("position_entry_time"):
                    entry_time = pd.to_datetime(self.state.state["position_entry_time"])
                else:
                    # Fallback
                    created_ms = pos_details.get('created_time', 0)
                    if created_ms > 0:
                        entry_time = datetime.fromtimestamp(created_ms / 1000.0, tz=timezone.utc).replace(tzinfo=None)
                        self.state.state["position_entry_time"] = entry_time.isoformat()
                        self.state.save()
                
                # Check Timeout
                if entry_time:
                    hold_seconds = (current_time - entry_time).total_seconds()
                    max_hold_seconds = self.config.strategy.max_holding_bars * tf_seconds
                    
                    if hold_seconds > max_hold_seconds:
                        logger.warning(f"Max Holding Time Exceeded ({hold_seconds:.0f}s). Force Closing.")
                        side = "Buy" if pos_size > 0 else "Sell"
                        self.exchange.market_close(side, abs_size)
                         
            except Exception as e:
                logger.error(f"Position timer error: {e}")

        # E. DETECT CLOSE (PnL)
        if abs_size == 0 and self.state.state.get("last_pos_size", 0) != 0:
             try:
                # Expert PnL: Aggregate ALL fills since entry
                entry_ts_iso = self.state.state.get("position_entry_time")
                # Convert ISO to ms timestamp
                entry_ms = 0
                if entry_ts_iso:
                    dt_entry = pd.to_datetime(entry_ts_iso)
                    entry_ms = dt_entry.timestamp() * 1000.0
                
                # Fetch last 20 records (buffer for fragmentation)
                closed_records = self.exchange.fetch_closed_pnl(limit=20)
                
                cycle_pnl = 0.0
                trade_count = 0
                
                for r in closed_records:
                    # Filter: Only include PnL generated AFTER we entered this position
                    # We add a 30-second "drift buffer" (30000ms) to ensure we don't miss records
                    # due to exchange/local timestamp misalignments.
                    r_time = int(r.get('createdTime', 0))
                    if r_time > (entry_ms - 30000):
                        cycle_pnl += float(r.get('closedPnl', 0))
                        trade_count += 1
                
                if trade_count > 0:
                    self.health.record_trade(cycle_pnl)
                    self.state.update_pnl(cycle_pnl)
                    logger.info(f"Trade Cycle Closed. Aggregated PnL: {cycle_pnl:.4f} USDT ({trade_count} fills) | Health: {self.health.check_health()}")
                else:
                    logger.warning("Position closed but no recent PnL records found! (Timing mismatch?)")
                    
             except Exception as e:
                logger.error(f"PnL Calculation Error: {e}")
             
             # Clear Entry Time AFTER processing
             self.state.state["position_entry_time"] = None
        
        # Update local state tracker for next loop
        self.state.state["last_pos_size"] = abs_size
        self.state.save()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="RAVEUSDT", help="Trading Pair (e.g., RAVEUSDT, MONUSDT)")
    parser.add_argument("--model-root", type=str, default="models_v4/RAVEUSDT", help="Path to model directory")
    args = parser.parse_args()
    
    # OVERRIDE GLOBAL CONFIG
    CONF.data.symbol = args.symbol
    # Also update model root if it looks like a default pattern, 
    # but respect user input if they point to a specific folder.
    # We leave model-root as is, assuming the user points to the right model for the symbol.
    
    logger.info(f"Starting ChampionBot for {CONF.data.symbol}...")
    ChampionBot(args.model_root).run()
