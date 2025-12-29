#!/usr/bin/env python3
"""
Standalone Live Paper Trading Test for TrendFollower

This script connects to Bybit via WebSocket, receives real-time trades,
feeds them to the TrendFollower ML model, opens paper positions,
tracks them to completion (stop loss or take profit), and logs results.

The goal is to verify that live performance matches backtest results.

Usage:
    python live_trading.py --model-dir ./models --symbol MONUSDT

Requirements:
    pip install pybit
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from time import sleep, time
from typing import Optional, List, Dict
from collections import deque
from dataclasses import dataclass, field, asdict
import threading
import logging
import argparse
import json
import io
import os

# Pybit imports
from pybit.unified_trading import WebSocket

# Our imports
from predictor import TrendFollowerPredictor
from config import TrendFollowerConfig, DEFAULT_CONFIG
from data_loader import aggregate_to_bars, preprocess_trades


# =============================================================================
# Training Config Loader (train_config.json)
# =============================================================================

_TRAIN_CONFIG_FILENAME = "train_config.json"


def _apply_config_section(target, data: dict, path_fields: Optional[set] = None) -> None:
    if not isinstance(data, dict):
        return
    path_fields = path_fields or set()
    for key, value in data.items():
        if key in path_fields and value is not None:
            value = Path(value)
        setattr(target, key, value)


def _load_train_config(model_dir: Path) -> Optional[TrendFollowerConfig]:
    config_path = Path(model_dir) / _TRAIN_CONFIG_FILENAME
    if not config_path.exists():
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise ValueError(f"Failed to read {config_path}: {exc}")

    if not isinstance(data, dict):
        raise ValueError(f"Invalid config format in {config_path}")

    cfg = TrendFollowerConfig()
    _apply_config_section(cfg.data, data.get("data", {}), {"data_dir"})
    _apply_config_section(cfg.features, data.get("features", {}))
    _apply_config_section(cfg.labels, data.get("labels", {}))
    _apply_config_section(cfg.model, data.get("model", {}), {"model_dir"})
    if "base_timeframe_idx" in data:
        cfg.base_timeframe_idx = int(data["base_timeframe_idx"])
    if "seed" in data:
        cfg.seed = int(data["seed"])
    return cfg


# =============================================================================
# Default Parameters - MUST MATCH BACKTEST
# =============================================================================

# These defaults match SimpleBacktester in backtest.py
DEFAULT_PARAMS = {
    'initial_capital': 10000.0,
    'position_size_pct': 0.02,      # 2% risk per trade
    'stop_loss_atr': 1.0,           # Stop in ATR units
    'stop_padding_pct': 0.0,        # Extra stop distance as fraction of entry (0.0 = disabled)
    'take_profit_rr': 1.5,          # Target reward:risk ratio (matches backtest)
    'min_quality': 'C',             # Quality not gated in backtest
    'use_trend_gate': False,        # Gate by trend classifier probability
    'min_trend_prob': 0.0,          # Minimum trend probability
    'use_regime_gate': False,       # Gate by regime classifier
    'min_regime_prob': 0.0,         # Minimum regime probability
    'allow_regime_ranging': True,
    'allow_regime_trend_up': True,
    'allow_regime_trend_down': True,
    'allow_regime_volatile': True,
    'regime_align_direction': True,
    'min_bounce_prob': 0.48,        # Minimum bounce probability (matches touch backtest)
    'max_bounce_prob': 1.0,         # Maximum bounce probability for bucket filtering (1.0 = no max)
    'use_ev_gate': True,            # Use EV gate instead of probability threshold
    'ev_margin_r': 0.0,             # Minimum EV margin in R units
    'fee_percent': 0.0011,          # Round-trip fee as a decimal of price
    'use_expected_rr': False,       # Use expected_rr in EV gating when available
    'cooldown_bars_after_stop': 0,  # Cooldown after stop-loss in base bars (0 = disabled)
    'trade_side': 'long',           # long | short | both
    'use_dynamic_rr': False,        # Use expected_rr from model for TP sizing
    'use_calibration': False,       # Use calibrated probabilities (Isotonic Regression)
    'use_incremental': True,        # Use incremental feature calculation (faster)
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PaperPosition:
    """An open paper trading position"""
    entry_time: datetime
    direction: int              # 1 for long, -1 for short
    entry_price: float
    size: float                 # In base currency units
    stop_loss: float
    take_profit: float
    signal_quality: str
    atr_at_entry: float
    metadata: dict = field(default_factory=dict)


@dataclass
class CompletedTrade:
    """A completed paper trade"""
    entry_time: datetime
    exit_time: datetime
    direction: int
    entry_price: float
    exit_price: float
    size: float
    stop_loss: float
    take_profit: float
    pnl: float
    pnl_percent: float
    signal_quality: str
    exit_reason: str            # 'stop_loss', 'take_profit'
    duration_seconds: float
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d['entry_time'] = self.entry_time.isoformat()
        d['exit_time'] = self.exit_time.isoformat()
        return d


@dataclass
class LiveStats:
    """Live trading statistics"""
    start_time: datetime = None
    trades_received: int = 0
    predictions_made: int = 0
    signals_generated: int = 0
    positions_opened: int = 0
    positions_closed: int = 0
    
    # Performance metrics (updated as trades close)
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_pnl_percent: float = 0.0
    
    # By grade
    trades_by_grade: Dict[str, int] = field(default_factory=lambda: {'A': 0, 'B': 0, 'C': 0})
    wins_by_grade: Dict[str, int] = field(default_factory=lambda: {'A': 0, 'B': 0, 'C': 0})


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(log_file: Optional[str] = None):
    """Setup logging configuration"""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)


# =============================================================================
# Trade Buffer
# =============================================================================

class TradeBuffer:
    """Thread-safe buffer for accumulating trades."""

    def __init__(self, max_trades: int = 100000000, max_age_seconds: Optional[float] = None):
        self.max_trades = max_trades
        self.max_age_seconds = max_age_seconds if max_age_seconds and max_age_seconds > 0 else None
        self.trades: deque = deque(maxlen=max_trades)
        self._pending: List[dict] = []
        self.lock = threading.Lock()
        self.trade_count = 0

    def set_max_age_seconds(self, max_age_seconds: Optional[float]) -> None:
        with self.lock:
            self.max_age_seconds = max_age_seconds if max_age_seconds and max_age_seconds > 0 else None
            self._trim_old()

    def _trim_old(self, latest_ts: Optional[float] = None) -> None:
        if not self.max_age_seconds:
            return
        if latest_ts is None:
            if not self.trades:
                return
            latest_ts = self.trades[-1].get('timestamp')
        if latest_ts is None:
            return
        cutoff = float(latest_ts) - float(self.max_age_seconds)
        while self.trades:
            ts = self.trades[0].get('timestamp')
            if ts is None or float(ts) >= cutoff:
                break
            self.trades.popleft()
    
    def add_trade(self, trade: dict):
        """Add a single trade from WebSocket message"""
        with self.lock:
            tick_dir = trade.get('L') or trade.get('tickDirection') or 'ZeroPlusTick'
            normalized = {
                'timestamp': trade['T'] / 1000,
                'symbol': trade['s'],
                'side': trade['S'],
                'size': float(trade['v']),
                'price': float(trade['p']),
                'tickDirection': tick_dir,
            }
            self.trades.append(normalized)
            self._pending.append(normalized)
            self.trade_count += 1
            self._trim_old(latest_ts=normalized.get('timestamp'))
    
    def add_trades_batch(self, trades: List[dict]):
        """Add multiple trades from WebSocket message"""
        for trade in trades:
            self.add_trade(trade)
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get all trades as DataFrame"""
        with self.lock:
            if not self.trades:
                return pd.DataFrame()
            return pd.DataFrame(list(self.trades))

    def drain_pending_dataframe(self) -> pd.DataFrame:
        """Get new (unconsumed) trades as DataFrame and clear the pending buffer."""
        with self.lock:
            if not self._pending:
                return pd.DataFrame()
            df = pd.DataFrame(self._pending)
            self._pending = []
            return df
    
    def get_trade_count(self) -> int:
        return self.trade_count

    def get_active_trade_count(self) -> int:
      """Number of trades currently kept (used to build bars/features)."""
      with self.lock:
          return len(self.trades)


# =============================================================================
# Live Paper Trader
# =============================================================================

class LivePaperTrader:
    """
    Live paper trading system that matches backtest logic exactly.
    
    - Opens positions based on ML signals (same criteria as backtest)
    - Tracks positions and closes on stop loss or take profit
    - Logs all trades and calculates metrics
    """
    
    def __init__(
        self,
        model_dir: Path,
        symbol: str = 'MONUSDT',
        testnet: bool = False,
        # Trading parameters - MATCH BACKTEST DEFAULTS
        initial_capital: float = DEFAULT_PARAMS['initial_capital'],
        position_size_pct: float = DEFAULT_PARAMS['position_size_pct'],
        stop_loss_atr: float = DEFAULT_PARAMS['stop_loss_atr'],
        stop_padding_pct: float = DEFAULT_PARAMS['stop_padding_pct'],
        take_profit_rr: float = DEFAULT_PARAMS['take_profit_rr'],
        min_quality: str = DEFAULT_PARAMS['min_quality'],
        min_trend_prob: float = DEFAULT_PARAMS['min_trend_prob'],
        use_trend_gate: bool = DEFAULT_PARAMS['use_trend_gate'],
        use_regime_gate: bool = DEFAULT_PARAMS['use_regime_gate'],
        min_regime_prob: float = DEFAULT_PARAMS['min_regime_prob'],
        allow_regime_ranging: bool = DEFAULT_PARAMS['allow_regime_ranging'],
        allow_regime_trend_up: bool = DEFAULT_PARAMS['allow_regime_trend_up'],
        allow_regime_trend_down: bool = DEFAULT_PARAMS['allow_regime_trend_down'],
        allow_regime_volatile: bool = DEFAULT_PARAMS['allow_regime_volatile'],
        regime_align_direction: bool = DEFAULT_PARAMS['regime_align_direction'],
        min_bounce_prob: float = DEFAULT_PARAMS['min_bounce_prob'],
        max_bounce_prob: float = DEFAULT_PARAMS['max_bounce_prob'],
        use_ev_gate: bool = DEFAULT_PARAMS['use_ev_gate'],
        ev_margin_r: float = DEFAULT_PARAMS['ev_margin_r'],
        fee_percent: float = DEFAULT_PARAMS['fee_percent'],
        use_expected_rr: bool = DEFAULT_PARAMS['use_expected_rr'],
        cooldown_bars_after_stop: int = DEFAULT_PARAMS['cooldown_bars_after_stop'],
        trade_side: str = DEFAULT_PARAMS['trade_side'],
        use_dynamic_rr: bool = DEFAULT_PARAMS['use_dynamic_rr'],
        use_calibration: bool = DEFAULT_PARAMS['use_calibration'],
        use_incremental: bool = DEFAULT_PARAMS['use_incremental'],
        # System parameters
        update_interval: float = 5.0,
        warmup_trades: int = 1000,
        log_dir: Path = Path('./live_results'),
        bootstrap_csv: Optional[str] = None,
        lookback_days: Optional[float] = None,
    ):
        self.symbol = symbol
        self.testnet = testnet
        
        # Trading parameters (MUST MATCH BACKTEST)
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position_size_pct = position_size_pct
        self.stop_loss_atr = stop_loss_atr
        self.stop_padding_pct = stop_padding_pct
        self.take_profit_rr = take_profit_rr
        self.min_quality = min_quality
        self.min_trend_prob = min_trend_prob
        self.use_trend_gate = bool(use_trend_gate)
        self.use_regime_gate = bool(use_regime_gate)
        self.min_regime_prob = float(min_regime_prob)
        self.allow_regime_ranging = bool(allow_regime_ranging)
        self.allow_regime_trend_up = bool(allow_regime_trend_up)
        self.allow_regime_trend_down = bool(allow_regime_trend_down)
        self.allow_regime_volatile = bool(allow_regime_volatile)
        self.regime_align_direction = bool(regime_align_direction)
        self.min_bounce_prob = min_bounce_prob
        self.max_bounce_prob = max_bounce_prob
        self.use_ev_gate = bool(use_ev_gate)
        self.ev_margin_r = float(ev_margin_r)
        self.fee_percent = max(0.0, float(fee_percent))
        self.use_expected_rr = bool(use_expected_rr)
        self.use_dynamic_rr = use_dynamic_rr
        self.use_calibration = bool(use_calibration)
        self.cooldown_bars_after_stop = max(0, int(cooldown_bars_after_stop))
        self.cooldown_seconds: float = 0.0
        self.last_stop_time: Optional[datetime] = None

        # Trade direction filter
        self.trade_side = (trade_side or "long").strip().lower()
        if self.trade_side in {"long", "buy"}:
            self.allow_long = True
            self.allow_short = False
            self.trade_side = "long"
        elif self.trade_side in {"short", "sell"}:
            self.allow_long = False
            self.allow_short = True
            self.trade_side = "short"
        elif self.trade_side in {"both", "all", "long+short", "long_short"}:
            self.allow_long = True
            self.allow_short = True
            self.trade_side = "both"
        else:
            raise ValueError("trade_side must be one of: long, short, both")
        
        # System parameters
        self.update_interval = update_interval
        self.warmup_trades = warmup_trades
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.bootstrap_csv = Path(bootstrap_csv) if bootstrap_csv else None
        self.lookback_days: Optional[float] = lookback_days
        self.lookback_seconds: Optional[float] = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize ML predictor
        self.use_incremental = use_incremental
        self.logger.info(f"Loading models from {model_dir}...")
        train_config = None
        try:
            train_config = _load_train_config(model_dir)
        except Exception as exc:
            self.logger.warning(f"Could not load training config: {exc}")

        if train_config is not None:
            self.config = train_config
            self.config.model.model_dir = Path(model_dir)
            self.logger.info(f"Loaded training config from {Path(model_dir) / _TRAIN_CONFIG_FILENAME}")
        else:
            self.logger.warning(
                f"Training config not found at {Path(model_dir) / _TRAIN_CONFIG_FILENAME}; "
                "falling back to defaults."
            )
            self.config = TrendFollowerConfig()
            # Align config with EMA9-only, pullback 0.3 ATR setup used in backtests
            self.config.features.ema_periods = [9]
            self.config.labels.pullback_ema = 9
            # Use config.py default pullback_threshold if not set (0.5 ATR)
            # Note: Trained models will have their own tuned value loaded from train_config
            if not hasattr(self.config.labels, 'pullback_threshold'):
                self.config.labels.pullback_threshold = 0.5

        # Use tuned thresholds/targets when CLI defaults are left unchanged.
        if stop_loss_atr == DEFAULT_PARAMS['stop_loss_atr']:
            tuned_stop = getattr(self.config.labels, 'stop_atr_multiple', None)
            if tuned_stop is not None:
                self.stop_loss_atr = float(tuned_stop)
                self.logger.info(f"Using tuned stop_atr_multiple for stop_loss_atr: {self.stop_loss_atr:.4f}")
        if take_profit_rr == DEFAULT_PARAMS['take_profit_rr']:
            tuned_rr = getattr(self.config.labels, 'target_rr', None)
            if tuned_rr is not None:
                self.take_profit_rr = float(tuned_rr)
                self.logger.info(f"Using tuned target_rr for take_profit_rr: {self.take_profit_rr:.4f}")
        if min_bounce_prob == DEFAULT_PARAMS['min_bounce_prob']:
            tuned_threshold = getattr(self.config.labels, 'best_threshold', None)
            if tuned_threshold is not None:
                self.min_bounce_prob = float(tuned_threshold)
                self.logger.info(f"Using tuned best_threshold for min_bounce_prob: {self.min_bounce_prob:.4f}")
        if use_ev_gate == DEFAULT_PARAMS['use_ev_gate']:
            tuned_use_ev_gate = getattr(self.config.labels, 'use_ev_gate', None)
            if tuned_use_ev_gate is not None:
                self.use_ev_gate = bool(tuned_use_ev_gate)
                self.logger.info(f"Using tuned use_ev_gate: {self.use_ev_gate}")
        if ev_margin_r == DEFAULT_PARAMS['ev_margin_r']:
            tuned_ev_margin = getattr(self.config.labels, 'ev_margin_r', None)
            if tuned_ev_margin is not None:
                self.ev_margin_r = float(tuned_ev_margin)
                self.logger.info(f"Using tuned ev_margin_r: {self.ev_margin_r:.4f}")
        if fee_percent == DEFAULT_PARAMS['fee_percent']:
            tuned_fee = getattr(self.config.labels, 'fee_percent', None)
            if tuned_fee is not None:
                self.fee_percent = float(tuned_fee)
                self.logger.info(f"Using tuned fee_percent: {self.fee_percent:.5f}")
        if use_expected_rr == DEFAULT_PARAMS['use_expected_rr']:
            tuned_use_rr = getattr(self.config.labels, 'use_expected_rr', None)
            if tuned_use_rr is not None:
                self.use_expected_rr = bool(tuned_use_rr)
                self.logger.info(f"Using tuned use_expected_rr: {self.use_expected_rr}")
        if use_calibration == DEFAULT_PARAMS['use_calibration']:
            tuned_use_cal = getattr(self.config.labels, 'use_calibration', None)
            if tuned_use_cal is not None:
                self.use_calibration = bool(tuned_use_cal)
                self.logger.info(f"Using tuned use_calibration: {self.use_calibration}")
        if use_trend_gate == DEFAULT_PARAMS['use_trend_gate']:
            tuned_use_trend = getattr(self.config.labels, 'use_trend_gate', None)
            if tuned_use_trend is not None:
                self.use_trend_gate = bool(tuned_use_trend)
                self.logger.info(f"Using tuned use_trend_gate: {self.use_trend_gate}")
        if min_trend_prob == DEFAULT_PARAMS['min_trend_prob']:
            tuned_min_trend = getattr(self.config.labels, 'min_trend_prob', None)
            if tuned_min_trend is not None:
                self.min_trend_prob = float(tuned_min_trend)
                self.logger.info(f"Using tuned min_trend_prob: {self.min_trend_prob:.4f}")
        if use_regime_gate == DEFAULT_PARAMS['use_regime_gate']:
            tuned_use_regime = getattr(self.config.labels, 'use_regime_gate', None)
            if tuned_use_regime is not None:
                self.use_regime_gate = bool(tuned_use_regime)
                self.logger.info(f"Using tuned use_regime_gate: {self.use_regime_gate}")
        if min_regime_prob == DEFAULT_PARAMS['min_regime_prob']:
            tuned_min_regime = getattr(self.config.labels, 'min_regime_prob', None)
            if tuned_min_regime is not None:
                self.min_regime_prob = float(tuned_min_regime)
                self.logger.info(f"Using tuned min_regime_prob: {self.min_regime_prob:.4f}")
        if allow_regime_ranging == DEFAULT_PARAMS['allow_regime_ranging']:
            tuned_allow = getattr(self.config.labels, 'allow_regime_ranging', None)
            if tuned_allow is not None:
                self.allow_regime_ranging = bool(tuned_allow)
        if allow_regime_trend_up == DEFAULT_PARAMS['allow_regime_trend_up']:
            tuned_allow = getattr(self.config.labels, 'allow_regime_trend_up', None)
            if tuned_allow is not None:
                self.allow_regime_trend_up = bool(tuned_allow)
        if allow_regime_trend_down == DEFAULT_PARAMS['allow_regime_trend_down']:
            tuned_allow = getattr(self.config.labels, 'allow_regime_trend_down', None)
            if tuned_allow is not None:
                self.allow_regime_trend_down = bool(tuned_allow)
        if allow_regime_volatile == DEFAULT_PARAMS['allow_regime_volatile']:
            tuned_allow = getattr(self.config.labels, 'allow_regime_volatile', None)
            if tuned_allow is not None:
                self.allow_regime_volatile = bool(tuned_allow)
        if regime_align_direction == DEFAULT_PARAMS['regime_align_direction']:
            tuned_align = getattr(self.config.labels, 'regime_align_direction', None)
            if tuned_align is not None:
                self.regime_align_direction = bool(tuned_align)

        effective_lookback = self.lookback_days
        if effective_lookback is None:
            cfg_lookback = getattr(self.config.data, "lookback_days", None)
            effective_lookback = cfg_lookback
        if effective_lookback is not None:
            try:
                effective_lookback = float(effective_lookback)
            except Exception:
                effective_lookback = None
        if effective_lookback is not None and effective_lookback > 0:
            self.lookback_days = effective_lookback
            self.lookback_seconds = float(effective_lookback) * 86400.0
        else:
            self.lookback_days = None
            self.lookback_seconds = None

        # Initialize components
        self.trade_buffer = TradeBuffer(max_age_seconds=self.lookback_seconds)
        self.predictor = TrendFollowerPredictor(
            self.config,
            use_incremental=use_incremental,
            use_calibration=self.use_calibration,
        )
        self.predictor.load_models(model_dir)
        if use_incremental:
            self.logger.info("Incremental feature mode ENABLED (fast updates)")
        self.base_tf = self.config.features.timeframe_names[self.config.base_timeframe_idx]
        self.base_tf_seconds = self.config.features.timeframes[self.config.base_timeframe_idx]
        self.cooldown_seconds = float(self.cooldown_bars_after_stop) * float(self.base_tf_seconds)
        
        # Require price to be essentially on the EMA to enter (in ATR units)
        # Use the tuned pullback_threshold from config if available, else default to 0.5 (config.py default)
        self.ema_touch_tolerance_atr = float(getattr(self.config.labels, 'pullback_threshold', 0.5))
        self.logger.info(f"Using EMA touch threshold: {self.ema_touch_tolerance_atr} ATR")
        
        # WebSocket
        self.ws: Optional[WebSocket] = None
        
        # State
        self.running = False
        self.current_price: float = 0.0
        self.current_high: float = 0.0  # Track high since last check
        self.current_low: float = float('inf')  # Track low since last check
        self.position: Optional[PaperPosition] = None
        self.completed_trades: List[CompletedTrade] = []
        self.stats = LiveStats()
        # Exit semantics: when True, exits are evaluated only on completed base bars (matches backtest math).
        # When False, exits are evaluated continuously using intratick highs/lows.
        self.exit_on_bar_close_only: bool = True
        self.last_trade_timestamp: Optional[float] = None
        self._predictor_last_cutoff_bar_time: Optional[int] = None
        self._predictor_carryover_df: Optional[pd.DataFrame] = None
        # Base-bar metadata for backtest-equivalent semantics
        self._last_closed_bar_time: Optional[int] = None  # epoch seconds (bar OPEN time)
        self._next_entry_bar_time: Optional[int] = None   # epoch seconds (bar OPEN time) when entries are allowed again
        
        # Session log file
        session_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.trades_file = self.log_dir / f'trades_{session_time}.json'
        self.stats_file = self.log_dir / f'stats_{session_time}.json'

        # Connection resilience settings
        self._stale_data_threshold_sec: float = 60.0  # Consider data stale after N seconds of no trades
        self._last_trade_wallclock: Optional[float] = None  # When we last received a trade (time.time())
        self._reconnect_attempts: int = 0
        self._max_reconnect_attempts: int = 10
        self._reconnect_backoff_base: float = 2.0  # Exponential backoff base (seconds)
        self._last_reconnect_time: Optional[float] = None

    def start(self):
        """Start the live paper trading system"""
        self._print_config()
        
        self.running = True
        self.stats.start_time = datetime.now()

        # Optional bootstrap from CSV to avoid cold start gaps
        self._load_bootstrap_trades()
        
        # Connect to WebSocket
        self._connect_websocket()
        
        # Start main loop
        self._main_loop()
    
    def stop(self):
        """Stop the system"""
        self.logger.info("Stopping...")
        self.running = False
    
    def _print_config(self):
        """Print configuration at startup"""
        self.logger.info("=" * 70)
        self.logger.info("LIVE PAPER TRADING - TrendFollower")
        self.logger.info("=" * 70)
        self.logger.info("")
        self.logger.info("TRADING PARAMETERS (matching backtest):")
        self.logger.info(f"  Initial Capital:    ${self.initial_capital:,.2f}")
        self.logger.info(f"  Position Size:      {self.position_size_pct:.1%} of capital")
        self.logger.info(f"  Stop Loss:          {self.stop_loss_atr} ATR")
        self.logger.info(f"  Stop Padding:       {self.stop_padding_pct*100:.3f}% of entry")
        self.logger.info(f"  Take Profit:        {self.take_profit_rr}:1 R:R")
        self.logger.info(f"  Min Quality:        {self.min_quality}")
        self.logger.info(f"  Min Trend Prob:     {self.min_trend_prob:.0%}")
        self.logger.info(f"  Min Bounce Prob:    {self.min_bounce_prob:.0%}")
        self.logger.info(f"  Use Calibration:   {'Yes' if self.use_calibration else 'No'}")
        self.logger.info(f"  Trade Side:         {self.trade_side}")
        self.logger.info("")
        self.logger.info("SYSTEM PARAMETERS:")
        self.logger.info(f"  Symbol:             {self.symbol}")
        self.logger.info(f"  Testnet:            {self.testnet}")
        self.logger.info(f"  Update Interval:    {self.update_interval}s")
        self.logger.info(f"  Warmup Trades:      {self.warmup_trades}")
        if self.lookback_days:
            self.logger.info(f"  Lookback Days:      {self.lookback_days}")
        else:
            self.logger.info("  Lookback Days:      disabled")
        if self.cooldown_bars_after_stop > 0:
            self.logger.info(
                f"  Cooldown After SL:  {self.cooldown_bars_after_stop} bars ({int(self.cooldown_seconds)}s)"
            )
        else:
            self.logger.info("  Cooldown After SL:  disabled")
        self.logger.info(f"  Log Directory:      {self.log_dir}")
        if self.bootstrap_csv:
            self.logger.info(f"  Bootstrap CSV:      {self.bootstrap_csv}")
        self.logger.info("=" * 70)

    def _load_bootstrap_trades(self):
        """Load historical trades from CSV (file or directory) to seed the predictor."""
        if not self.bootstrap_csv or not self.bootstrap_csv.exists():
            return 0

        # Allow passing either a single file or a directory (e.g. ./data/SYMBOL).
        if self.bootstrap_csv.is_dir():
            files = sorted(self.bootstrap_csv.glob("*.csv"))
            if not files:
                self.logger.warning(f"Bootstrap path is a directory with no .csv files: {self.bootstrap_csv}")
                return 0
        else:
            files = [self.bootstrap_csv]

        want_cols = {"timestamp", "price", "size", "side", "tickdirection", "symbol"}
        dfs: List[pd.DataFrame] = []

        for path in files:
            header_line = ""
            try:
                with open(path, "rb") as f:
                    header_line = f.readline().decode("utf-8", errors="ignore").lstrip("\ufeff").strip()
            except Exception:
                header_line = ""

            header_lower = header_line.lower()
            has_header = all(k in header_lower for k in ("timestamp", "price", "size", "side"))

            try:
                if has_header:
                    df_part = pd.read_csv(
                        path,
                        usecols=lambda c: str(c).lower() in want_cols,
                    )
                else:
                    # Headerless collector-style rows, e.g.:
                    # timestamp,symbol,side,size,price,tickDirection,...
                    num_cols = len(header_line.split(",")) if header_line else 0
                    if num_cols >= 6:
                        df_part = pd.read_csv(
                            path,
                            header=None,
                            usecols=[0, 1, 2, 3, 4, 5],
                            names=["timestamp", "symbol", "side", "size", "price", "tickDirection"],
                        )
                    elif num_cols == 5:
                        df_part = pd.read_csv(
                            path,
                            header=None,
                            usecols=[0, 1, 2, 3, 4],
                            names=["timestamp", "symbol", "side", "size", "price"],
                        )
                    elif num_cols == 4:
                        df_part = pd.read_csv(
                            path,
                            header=None,
                            usecols=[0, 1, 2, 3],
                            names=["timestamp", "price", "size", "side"],
                        )
                    else:
                        df_part = pd.read_csv(path)
            except Exception as exc:
                self.logger.warning(f"Failed to read bootstrap CSV {path}: {exc}")
                continue

            if df_part is None or df_part.empty:
                continue

            # Normalize column names (case-insensitive)
            rename_map = {}
            for c in df_part.columns:
                lc = str(c).lower()
                if lc == "tickdirection":
                    rename_map[c] = "tickDirection"
                elif lc == "timestamp":
                    rename_map[c] = "timestamp"
                elif lc == "price":
                    rename_map[c] = "price"
                elif lc == "size":
                    rename_map[c] = "size"
                elif lc == "side":
                    rename_map[c] = "side"
                elif lc == "symbol":
                    rename_map[c] = "symbol"
            if rename_map:
                df_part = df_part.rename(columns=rename_map)

            if "price" in df_part.columns:
                df_part["price"] = pd.to_numeric(df_part["price"], errors="coerce").astype("float32")
            if "size" in df_part.columns:
                df_part["size"] = pd.to_numeric(df_part["size"], errors="coerce").astype("float32")
            if "timestamp" in df_part.columns:
                df_part["timestamp"] = pd.to_numeric(df_part["timestamp"], errors="coerce")

            if "tickDirection" not in df_part.columns:
                df_part["tickDirection"] = "ZeroPlusTick"
            else:
                df_part["tickDirection"] = df_part["tickDirection"].fillna("ZeroPlusTick")

            if "symbol" in df_part.columns:
                try:
                    df_part = df_part[df_part["symbol"].astype(str) == str(self.symbol)]
                except Exception:
                    pass

            dfs.append(df_part)

        if not dfs:
            self.logger.warning(f"No bootstrap trades loaded from {self.bootstrap_csv}")
            return 0

        df = pd.concat(dfs, ignore_index=True)
        del dfs

        required = {"timestamp", "price", "size", "side", "tickDirection"}
        if not required.issubset(df.columns):
            self.logger.warning(f"Bootstrap CSV missing required columns {required}; found {df.columns.tolist()}")
            return 0

        try:
            df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        except Exception:
            pass
        
        if self.lookback_seconds and not df.empty and "timestamp" in df.columns:
            try:
                max_ts = float(df["timestamp"].max())
                cutoff = max_ts - float(self.lookback_seconds)
                pre_filter_len = len(df)
                df = df[df["timestamp"] >= cutoff].reset_index(drop=True)
                self.logger.info(
                    f"Applied lookback: last {self.lookback_days} days -> "
                    f"{len(df):,} trades (from {pre_filter_len:,})"
                )
            except Exception as exc:
                self.logger.warning(f"Failed to apply lookback window: {exc}")

        # Seed predictor directly (do not inflate the live trade buffer).
        try:
            if self.use_incremental:
                # For incremental mode: preprocess trades and build bars, then warm up
                import time
                t0 = time.time()
                self.logger.info("Preprocessing bootstrap trades for incremental warmup...")

                # Preprocess trades (same as training pipeline)
                processed_df = preprocess_trades(df, self.config.data)

                # Build bars for each timeframe
                bars_dict = {}
                for tf_seconds, tf_name in zip(self.config.features.timeframes,
                                               self.config.features.timeframe_names):
                    self.logger.info(f"  Building {tf_name} bars from {len(processed_df)} trades...")
                    bars = aggregate_to_bars(processed_df, tf_seconds, self.config.data)
                    bars_dict[tf_name] = bars
                    self.logger.info(f"    -> {len(bars)} {tf_name} bars")

                # Warm up the incremental engine
                self.predictor.warm_up_incremental(bars_dict)

                t1 = time.time()
                self.logger.info(f"Incremental warmup completed in {t1-t0:.1f}s")

                # Store last bar time for each TF for incremental updates
                for tf_name, bars in bars_dict.items():
                    if not bars.empty:
                        self.predictor.last_bar_time[tf_name] = int(bars['bar_time'].iloc[-1])

            else:
                # Legacy mode: full recalculation
                self.predictor.add_trades(df)
        except Exception as exc:
            self.logger.warning(f"Failed to seed predictor from bootstrap data {self.bootstrap_csv}: {exc}")
            import traceback
            traceback.print_exc()
            return 0

        self.logger.info(f"Loaded {len(df)} bootstrap trades from {self.bootstrap_csv}")
        return int(len(df))
    
    def _connect_websocket(self):
        """Connect to Bybit WebSocket"""
        self.logger.info(f"Connecting to Bybit {'testnet' if self.testnet else 'mainnet'}...")
        
        self.ws = WebSocket(
            testnet=self.testnet,
            channel_type="linear",
        )
        
        self.ws.trade_stream(
            symbol=self.symbol,
            callback=self._handle_trade_message
        )
        
        self.logger.info(f"Subscribed to publicTrade.{self.symbol}")
        # Initialize wallclock tracking
        self._last_trade_wallclock = time()

    def _check_connection_health(self) -> bool:
        """
        Check if the WebSocket connection is healthy.
        Returns True if healthy, False if reconnection is needed.
        """
        # Skip check if we haven't started receiving data yet
        if self._last_trade_wallclock is None:
            return True

        now = time()
        time_since_trade = now - self._last_trade_wallclock

        # Check if data is stale
        if time_since_trade < self._stale_data_threshold_sec:
            return True  # Data is fresh

        # Data is stale - check WebSocket state
        ws_connected = False
        try:
            if self.ws is not None:
                ws_connected = self.ws.is_connected()
        except Exception as e:
            self.logger.warning(f"Error checking WebSocket connection: {e}")
            ws_connected = False

        if not ws_connected:
            self.logger.warning(
                f"Connection appears dead: no trades for {time_since_trade:.0f}s, "
                f"WebSocket is_connected={ws_connected}"
            )
            return False

        # WebSocket claims connected but no data - could be a silent disconnect
        if time_since_trade > self._stale_data_threshold_sec * 2:
            self.logger.warning(
                f"Possible silent disconnect: no trades for {time_since_trade:.0f}s "
                f"despite WebSocket reporting connected"
            )
            return False

        return True

    def _reconnect_websocket(self) -> bool:
        """
        Attempt to reconnect the WebSocket with exponential backoff.
        Returns True if reconnection was attempted, False if max attempts exceeded.
        """
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            self.logger.error(
                f"Max reconnection attempts ({self._max_reconnect_attempts}) exceeded. "
                f"Manual intervention required."
            )
            return False

        # Calculate backoff delay
        backoff = min(
            self._reconnect_backoff_base ** self._reconnect_attempts,
            60.0  # Cap at 60 seconds
        )

        self.logger.info(
            f"Reconnection attempt {self._reconnect_attempts + 1}/{self._max_reconnect_attempts} "
            f"(waiting {backoff:.1f}s)..."
        )

        # Wait before reconnecting
        sleep(backoff)

        # Try to cleanly close existing connection
        try:
            if self.ws is not None:
                self.ws.exit()
        except Exception as e:
            self.logger.warning(f"Error closing existing WebSocket: {e}")

        self.ws = None

        # Attempt reconnection
        try:
            self._connect_websocket()
            self._reconnect_attempts += 1
            self._last_reconnect_time = time()
            self.logger.info("Reconnection initiated - waiting for data...")
            return True
        except Exception as e:
            self.logger.error(f"Reconnection failed: {e}")
            self._reconnect_attempts += 1
            return True  # Still return True to allow retry

    def _handle_trade_message(self, message: dict):
        """Callback for WebSocket trade messages"""
        try:
            if 'data' in message:
                trades = message['data']
                self.trade_buffer.add_trades_batch(trades)

                # Track when we last received data (wallclock time)
                self._last_trade_wallclock = time()
                # Reset reconnect counter on successful data receipt
                if self._reconnect_attempts > 0:
                    self.logger.info("Connection restored - receiving data again")
                    self._reconnect_attempts = 0

                # Update price tracking
                for trade in trades:
                    price = float(trade['p'])
                    self.current_price = price
                    self.current_high = max(self.current_high, price)
                    self.current_low = min(self.current_low, price)
                    try:
                        self.last_trade_timestamp = float(trade.get('T', 0)) / 1000.0
                    except Exception:
                        pass

                self.stats.trades_received = self.trade_buffer.get_trade_count()
        except Exception as e:
            self.logger.error(f"Error handling trade message: {e}")
    
    def _main_loop(self):
        """Main trading loop"""
        self.logger.info(f"Waiting for {self.warmup_trades} trades to warm up...")

        last_prediction_time = None
        last_health_check = time()

        while self.running:
            try:
                # === CONNECTION HEALTH CHECK ===
                # Check periodically (every ~10 seconds) whether connection is healthy
                now_wallclock = time()
                if now_wallclock - last_health_check >= 10.0:
                    last_health_check = now_wallclock
                    if not self._check_connection_health():
                        # Connection is unhealthy - attempt reconnection
                        if not self._reconnect_websocket():
                            # Max retries exceeded - stop trading
                            self.logger.error("Connection lost and could not recover. Stopping.")
                            self.stop()
                            break
                        # After reconnection attempt, continue loop to wait for data
                        continue

                trade_count = self.trade_buffer.get_trade_count()

                # Warmup phase
                if trade_count < self.warmup_trades:
                    self.logger.info(f"Warming up... {trade_count}/{self.warmup_trades} trades")
                    sleep(self.update_interval)
                    continue

                # Check timing
                now = datetime.now()
                if last_prediction_time is not None:
                    elapsed = (now - last_prediction_time).total_seconds()
                    if elapsed < self.update_interval:
                        sleep(0.5)
                        continue

                # Run trading logic
                self._trading_tick()
                last_prediction_time = now

                # Reset high/low tracking for next interval
                self.current_high = self.current_price
                self.current_low = self.current_price

            except KeyboardInterrupt:
                self.stop()
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                import traceback
                traceback.print_exc()
                sleep(1)

        self._save_results()
        self._print_summary()
    
    def _trading_tick(self):
        """Single trading tick - check exits, check entries"""
        
        # Update predictor ONLY when a new base bar completes. This avoids
        # rebuilding features from the entire trade history every tick, which
        # becomes extremely slow after a couple hours.
        predictor_updated = False
        if self.last_trade_timestamp is not None:
            current_bar_time = int(self.last_trade_timestamp // self.base_tf_seconds) * self.base_tf_seconds
            need_predictor_update = (
                self._predictor_last_cutoff_bar_time is None
                or current_bar_time > self._predictor_last_cutoff_bar_time
                or self.predictor.features_cache is None
                or len(self.predictor.features_cache) == 0
            )
            if need_predictor_update:
                new_df = self.trade_buffer.drain_pending_dataframe()
                if not new_df.empty:
                    if 'tickDirection' not in new_df.columns:
                        new_df['tickDirection'] = 'ZeroPlusTick'
                    new_df['tickDirection'] = new_df['tickDirection'].fillna('ZeroPlusTick')

                combined = new_df
                if self._predictor_carryover_df is not None and not self._predictor_carryover_df.empty:
                    if combined.empty:
                        combined = self._predictor_carryover_df
                    else:
                        combined = pd.concat([self._predictor_carryover_df, combined], ignore_index=True)

                cutoff = current_bar_time
                if not combined.empty and 'timestamp' in combined.columns:
                    to_add = combined[combined['timestamp'] < cutoff]
                    carry = combined[combined['timestamp'] >= cutoff]
                else:
                    to_add = combined
                    carry = pd.DataFrame()

                # If we have no completed-bar trades yet (e.g., just started mid-bar),
                # fall back to building features from whatever we have so the system
                # can start producing signals.
                has_features = False
                if self.use_incremental:
                    has_features = bool(self.predictor.incremental_features)
                else:
                    has_features = self.predictor.features_cache is not None and len(self.predictor.features_cache) > 0

                if to_add.empty and not has_features:
                    to_add = combined
                    carry = pd.DataFrame()

                self._predictor_carryover_df = carry if not carry.empty else None

                if not to_add.empty:
                    to_add = to_add.sort_values('timestamp', kind='mergesort')

                    if self.use_incremental:
                        # Incremental mode: build bars and update incrementally
                        try:
                            processed = preprocess_trades(to_add, self.config.data)
                            new_bars_dict = {}
                            for tf_seconds, tf_name in zip(self.config.features.timeframes,
                                                           self.config.features.timeframe_names):
                                bars = aggregate_to_bars(processed, tf_seconds, self.config.data)
                                if not bars.empty:
                                    new_bars_dict[tf_name] = bars

                            if new_bars_dict:
                                self.predictor.add_bars_batch(new_bars_dict)
                                predictor_updated = True
                        except Exception as e:
                            self.logger.warning(f"Incremental update failed: {e}")
                    else:
                        # Legacy mode: full recalculation
                        self.predictor.add_trades(to_add)
                        predictor_updated = True

                    self.stats.predictions_made += 1

                self._predictor_last_cutoff_bar_time = cutoff
        
        # Get current ATR (latest completed base bar)
        current_atr = self._get_current_atr()

        # If a new base bar just completed, capture its timestamp (and optionally its OHLC range).
        if predictor_updated:
            if self.use_incremental:
                # Incremental mode: use last_bar_time from predictor
                if self.predictor.last_bar_time.get(self.base_tf):
                    self._last_closed_bar_time = self.predictor.last_bar_time[self.base_tf]
            elif self.predictor.features_cache is not None and len(self.predictor.features_cache) > 0:
                latest_bar = self.predictor.features_cache.iloc[-1]
                if 'bar_time' in latest_bar.index and pd.notna(latest_bar['bar_time']):
                    try:
                        self._last_closed_bar_time = int(latest_bar['bar_time'])
                    except Exception:
                        self._last_closed_bar_time = None
                if self.exit_on_bar_close_only:
                    # Bar-close exit mode: use the completed bar's high/low for stop/TP checks.
                    if 'high' in latest_bar.index and pd.notna(latest_bar['high']):
                        self.current_high = float(latest_bar['high'])
                    if 'low' in latest_bar.index and pd.notna(latest_bar['low']):
                        self.current_low = float(latest_bar['low'])

        # Step 1: Exit checks (bar-close only, to match backtest math)
        if self.position is not None:
            if self.exit_on_bar_close_only:
                if predictor_updated:
                    self._check_exit(current_atr)
            else:
                self._check_exit(current_atr)

        # Step 2: Entry checks (only when a completed base bar is available)
        if self.position is None and predictor_updated:
            self._check_entry(current_atr)
        
        # Log current state
        self._log_state()

        # Reset intratick high/low to current price for next tick window
        # (kept for diagnostics; exits are evaluated on completed bars only).
        self.current_high = self.current_price
        self.current_low = self.current_price
    
    def _check_entry(self, current_atr: float):
        """Check if we should enter a position - MATCHES BACKTEST LOGIC"""
        
        # Get predictions
        entry_signal = self.predictor.get_entry_signal()
        
        if entry_signal is None:
            return
        
        # Debug: feature availability and buffer size
        has_features = False
        warmup_status = ""
        if self.use_incremental:
            if self.predictor.incremental_features:
                feats = self.predictor.incremental_features
                usable_features = sum(1 for v in feats.values() if v is not None and not (isinstance(v, float) and np.isnan(v)))
                total_features = len(feats)
                has_features = True
                # Get warmup status from engine
                if self.predictor.incremental_engine:
                    warmup_status = f" | Warmup: {self.predictor.incremental_engine.get_warmup_summary()}"
        else:
            if self.predictor.features_cache is not None and len(self.predictor.features_cache) > 0:
                latest = self.predictor.features_cache.iloc[-1]
                usable_features = int((latest.notna() & (latest != 0)).sum())
                total_features = len(latest)
                has_features = True

        if has_features:
            buf_trades = self.trade_buffer.get_active_trade_count()
            self.logger.info(
                f"Features: {usable_features}/{total_features}{warmup_status} | "
                f"Trades in buffer: {buf_trades} | bounce_prob: {entry_signal.bounce_prob:.3f}"
            )
        
        # Cooldown after stop-loss (optional; set --cooldown-bars-after-stop > 0)
        # Prefer bar-time based gating to match backtest; fall back to wall-clock if needed.
        if self.cooldown_bars_after_stop > 0:
            if self._next_entry_bar_time is not None and self._last_closed_bar_time is not None:
                if int(self._last_closed_bar_time) < int(self._next_entry_bar_time):
                    return
            elif self.cooldown_seconds > 0 and self.last_stop_time:
                elapsed = (datetime.now() - self.last_stop_time).total_seconds()
                if elapsed < self.cooldown_seconds:
                    return
        
        # Get model predictions
        trend_dir = entry_signal.direction  # from EMA slope (long-only when >0)
        bounce_prob = entry_signal.bounce_prob
        expected_rr = getattr(entry_signal, 'expected_rr', None)
        expected_rr_mean = getattr(entry_signal, 'expected_rr_mean', None)
        if expected_rr is not None and isinstance(expected_rr, float) and np.isnan(expected_rr):
            expected_rr = None
        if expected_rr_mean is not None and isinstance(expected_rr_mean, float) and np.isnan(expected_rr_mean):
            expected_rr_mean = None
        
        # Determine quality grade (same logic as backtest)
        atr_val = current_atr
        is_pullback = False
        trend_aligned = False
        slope_col = f'{self.base_tf}_ema_{self.config.labels.pullback_ema}_slope_norm'
        slope_val = self._get_feature_value(slope_col, default=0.0) or 0.0
        dist_atr = None
        ema_col = f'{self.base_tf}_ema_{self.config.labels.pullback_ema}'
        atr_col = f'{self.base_tf}_atr'

        # Get values using helper that handles both incremental and legacy modes
        ema = self._get_feature_value(ema_col)
        atr = self._get_feature_value(atr_col)
        # Use the completed base-bar close when available (matches backtest timing)
        price = self._get_feature_value('close', default=self.current_price)

        if ema is not None and atr is not None and atr > 0:
            atr_val = float(atr)
            dist_atr = abs(price - ema) / atr_val

        ema_touched = False
        if self.use_incremental:
            ema_touched = bool(self._get_feature_value('ema_touch_detected', default=False))
            ema_touch_dir = int(self._get_feature_value('ema_touch_direction', default=0) or 0)
            if ema_touched and ema_touch_dir not in (0, trend_dir):
                ema_touched = False
        else:
            bar_high = self._get_feature_value('high')
            bar_low = self._get_feature_value('low')
            if (
                ema is not None
                and atr is not None
                and atr > 0
                and bar_high is not None
                and bar_low is not None
            ):
                threshold = self.ema_touch_tolerance_atr
                mid_bar = (bar_high + bar_low) / 2.0
                if trend_dir == 1:
                    dist_low = (bar_low - ema) / atr_val
                    if -threshold <= dist_low <= threshold and (price >= ema or mid_bar >= ema):
                        ema_touched = True
                elif trend_dir == -1:
                    dist_high = (bar_high - ema) / atr_val
                    if -threshold <= dist_high <= threshold and (price <= ema or mid_bar <= ema):
                        ema_touched = True

        is_pullback = ema_touched

        trend_aligned = (trend_dir != 0 and slope_val != 0 and (slope_val * trend_dir) > 0)
        self.logger.info(
            f"slope_norm={slope_val:.3f}, pullback_dist_atr={dist_atr if dist_atr is not None else 'na'}, "
            f"ema_touch={ema_touched}, trend_aligned={trend_aligned}, bounce_prob={bounce_prob:.3f}"
        )
        
        if bounce_prob > 0.6 and trend_aligned and is_pullback:
            quality = 'A'
        elif bounce_prob > 0.5 and (trend_aligned or is_pullback):
            quality = 'B'
        else:
            quality = 'C'
        
        # Check entry criteria (SAME AS BACKTEST)
        quality_ok = True  # do not gate by quality
        trend_prob_ok = True
        regime_prob_ok = True
        trend_prob = None
        trend_signal = None
        if self.use_trend_gate or self.use_regime_gate:
            try:
                trend_signal = self.predictor.get_trend_signal()
            except Exception as exc:
                self.logger.debug(f"Trend/regime prediction unavailable: {exc}")
                trend_signal = None

        if trend_signal is not None:
            trend_prob = trend_signal.prob_up if trend_dir == 1 else trend_signal.prob_down if trend_dir == -1 else 0.0

        if self.use_trend_gate:
            if trend_prob is None:
                self.logger.debug("Skipping entry: trend_prob unavailable")
                return
            if trend_prob < float(self.min_trend_prob):
                self.logger.debug(
                    "Skipping entry: trend_prob {:.3f} < min {:.3f}".format(
                        trend_prob,
                        float(self.min_trend_prob),
                    )
                )
                return

        if self.use_regime_gate:
            if trend_signal is None:
                self.logger.debug("Skipping entry: regime prediction unavailable")
                return
            regime_id = int(trend_signal.regime)
            if regime_id == 0:
                regime_prob = float(trend_signal.prob_regime_ranging)
                allowed = bool(self.allow_regime_ranging)
            elif regime_id == 1:
                regime_prob = float(trend_signal.prob_regime_trend_up)
                allowed = bool(self.allow_regime_trend_up)
                if self.regime_align_direction and trend_dir != 1:
                    allowed = False
            elif regime_id == 2:
                regime_prob = float(trend_signal.prob_regime_trend_down)
                allowed = bool(self.allow_regime_trend_down)
                if self.regime_align_direction and trend_dir != -1:
                    allowed = False
            else:
                regime_prob = float(trend_signal.prob_regime_volatile)
                allowed = bool(self.allow_regime_volatile)

            if (not allowed) or (regime_prob < float(self.min_regime_prob)):
                self.logger.debug(
                    "Skipping entry: regime {} prob {:.3f} < min {:.3f} allowed={}".format(
                        regime_id,
                        regime_prob,
                        float(self.min_regime_prob),
                        allowed,
                    )
                )
                return

        bounce_min_ok = bounce_prob >= self.min_bounce_prob
        bounce_max_ok = bounce_prob <= self.max_bounce_prob

        # Market entry at the completed base-bar close (matches profitable backtest runs).
        if price is None:
            return
        if trend_dir not in (-1, 1):
            return
        if trend_dir == 1 and not self.allow_long:
            return
        if trend_dir == -1 and not self.allow_short:
            return
        if not ema_touched:
            self.logger.debug("Skipping entry: ema_touch not detected")
            return

        if self.use_ev_gate:
            entry_model = getattr(getattr(self.predictor, "models", None), "entry_model", None)
            if entry_model is None:
                self.logger.debug("Skipping EV gate: entry_model unavailable")
            else:
                stop_dist = (self.stop_loss_atr * atr_val) + (self.stop_padding_pct * price)
                fee_r = 0.0
                if np.isfinite(stop_dist) and stop_dist > 0 and price > 0:
                    fee_r = (self.fee_percent * price) / stop_dist

                rr_mean = float(self.take_profit_rr)
                rr_cons = rr_mean
                if self.use_expected_rr:
                    if expected_rr_mean is not None:
                        rr_mean = float(expected_rr_mean)
                    if expected_rr is not None:
                        rr_cons = float(expected_rr)

                ev_components = entry_model.compute_expected_rr_components(
                    np.asarray([bounce_prob], dtype=float),
                    np.asarray([rr_mean], dtype=float),
                    rr_conservative=np.asarray([rr_cons], dtype=float),
                    cost_r=np.asarray([fee_r], dtype=float),
                )
                ev_value = float(ev_components['ev_conservative_r'][0])
                if ev_value <= float(self.ev_margin_r):
                    self.logger.debug(
                        "Skipping entry: EV {:.4f} <= margin {:.4f}".format(
                            ev_value,
                            float(self.ev_margin_r),
                        )
                    )
                    return
        else:
            if not bounce_min_ok:
                self.logger.debug(f"Skipping entry: bounce_prob {bounce_prob:.3f} < min {self.min_bounce_prob}")
                return
            if not bounce_max_ok:
                self.logger.debug(f"Skipping entry: bounce_prob {bounce_prob:.3f} > max {self.max_bounce_prob}")
                return

        # Get expected RR from model if dynamic RR is enabled
        if self.use_dynamic_rr and expected_rr is None:
            entry_signal = self.predictor.get_entry_signal()
            if entry_signal and hasattr(entry_signal, 'expected_rr'):
                expected_rr = entry_signal.expected_rr

        if quality_ok and trend_prob_ok and regime_prob_ok:
            self._open_position(trend_dir, quality, atr_val, entry_price=price, expected_rr=expected_rr)
    
    def _open_position(self, direction: int, quality: str, atr: float, entry_price: Optional[float] = None, expected_rr: Optional[float] = None):
        """Open a new paper position"""

        price = entry_price if entry_price is not None else self.current_price

        # Calculate stop loss and take profit (SAME AS BACKTEST)
        stop_dist = (self.stop_loss_atr * atr) + (self.stop_padding_pct * price)
        stop_loss = price - (direction * stop_dist)

        # Use dynamic RR from model if available, else use fixed take_profit_rr
        effective_rr = self.take_profit_rr
        if self.use_dynamic_rr and expected_rr is not None and expected_rr > 0.5:
            # Use model's expected RR, but cap it at reasonable values
            effective_rr = min(max(expected_rr, 0.5), 5.0)

        take_profit = price + (direction * stop_dist * effective_rr)
        
        # Calculate position size (SAME AS BACKTEST)
        risk_amount = self.capital * self.position_size_pct
        risk_per_unit = abs(price - stop_loss)
        size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
        
        self.position = PaperPosition(
            entry_time=datetime.now(),
            direction=direction,
            entry_price=price,
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal_quality=quality,
            atr_at_entry=atr,
        )
        
        self.stats.positions_opened += 1
        self.stats.signals_generated += 1
        
        dir_name = "LONG" if direction == 1 else "SHORT"
        self.logger.info("=" * 70)
        self.logger.info(f"OPENED {quality}-grade {dir_name} POSITION")
        self.logger.info(f"   Entry:      {price:.6f}")
        self.logger.info(
            f"   Stop Loss:  {stop_loss:.6f} ({self.stop_loss_atr} ATR + {self.stop_padding_pct*100:.3f}% pad)"
        )
        rr_note = f" (dynamic)" if self.use_dynamic_rr and expected_rr else ""
        self.logger.info(f"   Take Profit:{take_profit:.6f} ({effective_rr:.2f}:1 R:R{rr_note})")
        self.logger.info(f"   Size:       {size:.2f} units (${risk_amount:.2f} risk)")
        self.logger.info("=" * 70)
    
    def _check_exit(self, current_atr: float):
        """Check if we should exit current position (bar-based, matches backtest)."""
        
        if self.position is None:
            return
        
        direction = self.position.direction
        stop = self.position.stop_loss
        target = self.position.take_profit
        
        exit_reason = None
        exit_price = None
        
        # Use high/low since last tick to simulate intrabar price action
        high = self.current_high
        low = self.current_low
        
        # Check stop loss (SAME AS BACKTEST)
        if direction == 1 and low <= stop:
            exit_reason = 'stop_loss'
            exit_price = stop
        elif direction == -1 and high >= stop:
            exit_reason = 'stop_loss'
            exit_price = stop
        
        # Check take profit (SAME AS BACKTEST)
        elif direction == 1 and high >= target:
            exit_reason = 'take_profit'
            exit_price = target
        elif direction == -1 and low <= target:
            exit_reason = 'take_profit'
            exit_price = target
        
        if exit_reason:
            self._close_position(exit_price, exit_reason, exit_bar_time=self._last_closed_bar_time)
    
    def _close_position(self, exit_price: float, exit_reason: str, exit_bar_time: Optional[int] = None):
        """Close current position and record trade"""
        
        pos = self.position
        now = datetime.now()
        
        # Calculate P&L (SAME AS BACKTEST)
        pnl = pos.direction * (exit_price - pos.entry_price) * pos.size
        pnl_percent = pos.direction * (exit_price - pos.entry_price) / pos.entry_price * 100
        duration = (now - pos.entry_time).total_seconds()
        
        # Record completed trade
        trade = CompletedTrade(
            entry_time=pos.entry_time,
            exit_time=now,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size=pos.size,
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
            pnl=pnl,
            pnl_percent=pnl_percent,
            signal_quality=pos.signal_quality,
            exit_reason=exit_reason,
            duration_seconds=duration,
        )
        
        self.completed_trades.append(trade)
        
        # Update capital
        self.capital += pnl
        
        # Update stats
        self.stats.positions_closed += 1
        self.stats.total_trades += 1
        self.stats.total_pnl += pnl
        self.stats.total_pnl_percent = (self.capital - self.initial_capital) / self.initial_capital * 100
        
        if pnl > 0:
            self.stats.winning_trades += 1
            self.stats.wins_by_grade[pos.signal_quality] += 1
        else:
            self.stats.losing_trades += 1
        
        self.stats.trades_by_grade[pos.signal_quality] += 1
        
        # Clear position
        self.position = None
        
        # Log
        result = "WIN" if pnl > 0 else "LOSS"
        dir_name = "LONG" if pos.direction == 1 else "SHORT"

        self.logger.info("=" * 70)
        self.logger.info(f"{result} - Closed {dir_name} ({exit_reason})")
        self.logger.info(f"   Entry:    {pos.entry_price:.6f}")
        self.logger.info(f"   Exit:     {exit_price:.6f}")
        self.logger.info(f"   P&L:      ${pnl:+.2f} ({pnl_percent:+.2f}%)")
        self.logger.info(f"   Duration: {duration:.0f}s")
        self.logger.info(f"   Capital:  ${self.capital:,.2f}")
        self.logger.info("-" * 70)
        self._log_running_stats()
        self.logger.info("=" * 70)

        if exit_reason == 'stop_loss':
            self.last_stop_time = now
            if exit_bar_time is not None and self.cooldown_bars_after_stop > 0:
                try:
                    self._next_entry_bar_time = int(exit_bar_time) + int(self.cooldown_bars_after_stop) * int(self.base_tf_seconds)
                except Exception:
                    self._next_entry_bar_time = None

        # Save trade to file
        self._save_trade(trade)
    
    def _log_state(self):
        """Log current market state (simplified)"""
        pos_str = "OPEN" if self.position else "NO POSITION"
        self.logger.info(f"Price: {self.current_price:.6f} | {pos_str}")
    
    def _log_running_stats(self):
        """Log running statistics"""
        if self.stats.total_trades == 0:
            return
        
        win_rate = self.stats.winning_trades / self.stats.total_trades
        
        self.logger.info(f"   RUNNING STATS:")
        self.logger.info(f"   Trades: {self.stats.total_trades} | "
                        f"Win Rate: {win_rate:.1%} | "
                        f"Total P&L: ${self.stats.total_pnl:+.2f} ({self.stats.total_pnl_percent:+.2f}%)")
    
    def _get_feature_value(self, feature_name: str, default=None):
        """Get a feature value from predictor (handles both incremental and legacy modes)"""
        if self.use_incremental:
            if not self.predictor.incremental_features:
                return default
            val = self.predictor.incremental_features.get(feature_name)
            return val if val is not None and not (isinstance(val, float) and np.isnan(val)) else default
        else:
            if self.predictor.features_cache is None or len(self.predictor.features_cache) == 0:
                return default
            if feature_name not in self.predictor.features_cache.columns:
                return default
            val = self.predictor.features_cache[feature_name].iloc[-1]
            return val if pd.notna(val) else default

    def _get_current_atr(self) -> float:
        """Get current ATR from predictor"""
        atr_col = f'{self.base_tf}_atr'
        atr = self._get_feature_value(atr_col)
        if atr is not None and atr > 0:
            return atr
        return self.current_price * 0.02  # Fallback

    def _get_ema_alignment(self) -> float:
        """Get EMA slope-based alignment from predictor"""
        col = f'{self.base_tf}_ema_{self.config.labels.pullback_ema}_slope_norm'
        val = self._get_feature_value(col, default=0)
        return val if val is not None else 0

    def _is_pullback_zone(self, atr: float) -> bool:
        """
        Check if price is essentially touching EMA (very tight band).
        Uses a small ATR tolerance to avoid float issues.
        """
        ema_col = f'{self.base_tf}_ema_{self.config.labels.pullback_ema}'
        ema = self._get_feature_value(ema_col)

        if ema is None or atr <= 0:
            return False

        dist = abs(self.current_price - ema) / atr
        return dist <= self.ema_touch_tolerance_atr
    
    def _save_trade(self, trade: CompletedTrade):
        """Save trade to file"""
        with open(self.trades_file, 'a') as f:
            f.write(json.dumps(trade.to_dict()) + '\n')
    
    def _save_results(self):
        """Save final results"""
        results = {
            'session_start': self.stats.start_time.isoformat() if self.stats.start_time else None,
            'session_end': datetime.now().isoformat(),
            'symbol': self.symbol,
            'parameters': {
                'initial_capital': self.initial_capital,
                'position_size_pct': self.position_size_pct,
                'stop_loss_atr': self.stop_loss_atr,
                'stop_padding_pct': self.stop_padding_pct,
                'take_profit_rr': self.take_profit_rr,
                'min_quality': self.min_quality,
                'min_trend_prob': self.min_trend_prob,
                'use_trend_gate': self.use_trend_gate,
                'use_regime_gate': self.use_regime_gate,
                'min_regime_prob': self.min_regime_prob,
                'allow_regime_ranging': self.allow_regime_ranging,
                'allow_regime_trend_up': self.allow_regime_trend_up,
                'allow_regime_trend_down': self.allow_regime_trend_down,
                'allow_regime_volatile': self.allow_regime_volatile,
                'regime_align_direction': self.regime_align_direction,
                'min_bounce_prob': self.min_bounce_prob,
                'use_ev_gate': self.use_ev_gate,
                'ev_margin_r': self.ev_margin_r,
                'fee_percent': self.fee_percent,
                'use_expected_rr': self.use_expected_rr,
                'cooldown_bars_after_stop': self.cooldown_bars_after_stop,
                'trade_side': self.trade_side,
            },
            'results': {
                'total_trades': self.stats.total_trades,
                'winning_trades': self.stats.winning_trades,
                'losing_trades': self.stats.losing_trades,
                'win_rate': self.stats.winning_trades / self.stats.total_trades if self.stats.total_trades > 0 else 0,
                'total_pnl': self.stats.total_pnl,
                'total_pnl_percent': self.stats.total_pnl_percent,
                'final_capital': self.capital,
                'trades_by_grade': self.stats.trades_by_grade,
                'wins_by_grade': self.stats.wins_by_grade,
            },
            'trades_file': str(self.trades_file),
        }
        
        with open(self.stats_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Results saved to {self.stats_file}")
    
    def _print_summary(self):
        """Print final session summary"""
        duration = datetime.now() - self.stats.start_time if self.stats.start_time else timedelta(0)
        
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("SESSION SUMMARY")
        self.logger.info("=" * 70)
        self.logger.info(f"Duration:           {duration}")
        self.logger.info(f"Trades Received:    {self.stats.trades_received:,}")
        self.logger.info(f"Predictions Made:   {self.stats.predictions_made}")
        self.logger.info(f"Signals Generated:  {self.stats.signals_generated}")
        self.logger.info("")
        self.logger.info("TRADING RESULTS:")
        self.logger.info(f"  Total Trades:     {self.stats.total_trades}")
        self.logger.info(f"  Winning Trades:   {self.stats.winning_trades}")
        self.logger.info(f"  Losing Trades:    {self.stats.losing_trades}")
        
        if self.stats.total_trades > 0:
            win_rate = self.stats.winning_trades / self.stats.total_trades
            self.logger.info(f"  Win Rate:         {win_rate:.1%}")
        
        self.logger.info(f"  Total P&L:        ${self.stats.total_pnl:+,.2f}")
        self.logger.info(f"  Return:           {self.stats.total_pnl_percent:+.2f}%")
        self.logger.info(f"  Final Capital:    ${self.capital:,.2f}")
        
        if self.stats.total_trades > 0:
            self.logger.info("")
            self.logger.info("BY SIGNAL GRADE:")
            for grade in ['A', 'B', 'C']:
                count = self.stats.trades_by_grade.get(grade, 0)
                wins = self.stats.wins_by_grade.get(grade, 0)
                wr = wins / count if count > 0 else 0
                self.logger.info(f"  Grade {grade}: {count} trades, {wr:.1%} win rate")
        
        self.logger.info("=" * 70)
        self.logger.info(f"Trades log: {self.trades_file}")
        self.logger.info(f"Stats file: {self.stats_file}")
        self.logger.info("=" * 70)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='TrendFollower Live Paper Trading',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with backtest-matching defaults
    python live_trading.py --model-dir ./models --symbol MONUSDT
    
    # Use testnet
    python live_trading.py --model-dir ./models --symbol MONUSDT --testnet
    
    Note: Default parameters match the backtest exactly:
        - min_quality: B
        - min_trend_prob: 0.5 (50%)
        - min_bounce_prob: 0.48 (48%)
        - stop_loss: 1.0 ATR
        - take_profit: 2.0:1 R:R
        """
    )
    
    parser.add_argument('--model-dir', type=str, required=True,
                       help='Directory containing trained models')
    parser.add_argument('--symbol', type=str, default='MONUSDT',
                       help='Trading symbol (default: MONUSDT)')
    parser.add_argument('--testnet', action='store_true',
                       help='Use Bybit testnet')
    
    # Trading parameters (defaults match backtest)
    parser.add_argument('--min-quality', type=str, default=DEFAULT_PARAMS['min_quality'],
                       choices=['A', 'B', 'C'],
                       help=f"Minimum signal quality (default: {DEFAULT_PARAMS['min_quality']})")
    parser.add_argument('--min-trend-prob', type=float, default=DEFAULT_PARAMS['min_trend_prob'],
                       help=f"Minimum trend probability (default: {DEFAULT_PARAMS['min_trend_prob']})")
    trend_gate_group = parser.add_mutually_exclusive_group()
    trend_gate_group.add_argument(
        '--use-trend-gate',
        action='store_true',
        help='Enable trend classifier gating (override train_config).',
    )
    trend_gate_group.add_argument(
        '--no-trend-gate',
        action='store_true',
        help='Disable trend classifier gating (override train_config).',
    )
    parser.add_argument('--min-regime-prob', type=float, default=DEFAULT_PARAMS['min_regime_prob'],
                       help=f"Minimum regime probability (default: {DEFAULT_PARAMS['min_regime_prob']})")
    regime_gate_group = parser.add_mutually_exclusive_group()
    regime_gate_group.add_argument(
        '--use-regime-gate',
        action='store_true',
        help='Enable regime classifier gating (override train_config).',
    )
    regime_gate_group.add_argument(
        '--no-regime-gate',
        action='store_true',
        help='Disable regime classifier gating (override train_config).',
    )
    parser.add_argument('--min-bounce-prob', type=float, default=DEFAULT_PARAMS['min_bounce_prob'],
                       help=f"Minimum bounce probability (default: {DEFAULT_PARAMS['min_bounce_prob']})")
    parser.add_argument('--max-bounce-prob', type=float, default=DEFAULT_PARAMS['max_bounce_prob'],
                       help=f"Maximum bounce probability for bucket filtering (default: {DEFAULT_PARAMS['max_bounce_prob']})")
    parser.add_argument('--use-dynamic-rr', action='store_true',
                       help='Use expected RR from model for dynamic TP sizing')
    calib_group = parser.add_mutually_exclusive_group()
    calib_group.add_argument('--use-calibration', action='store_true',
                            help='Use calibrated probabilities (Isotonic Regression)')
    calib_group.add_argument('--use-raw-probabilities', action='store_true',
                            help='Use raw (uncalibrated) probabilities (default)')
    parser.add_argument(
        '--trade-side',
        type=str,
        default=DEFAULT_PARAMS['trade_side'],
        choices=['long', 'short', 'both'],
        help=f"Trade direction filter (default: {DEFAULT_PARAMS['trade_side']})",
    )
    parser.add_argument('--stop-loss-atr', type=float, default=DEFAULT_PARAMS['stop_loss_atr'],
                       help=f"Stop loss in ATR (default: {DEFAULT_PARAMS['stop_loss_atr']})")
    parser.add_argument(
        '--stop-padding-pct',
        type=float,
        default=DEFAULT_PARAMS['stop_padding_pct'],
        help=f"Extra stop distance as fraction of entry (default: {DEFAULT_PARAMS['stop_padding_pct']:.6f}).",
    )
    parser.add_argument(
        '--cooldown-bars-after-stop',
        type=int,
        default=DEFAULT_PARAMS['cooldown_bars_after_stop'],
        help=f"Cooldown after a stop-loss in base bars (default: {DEFAULT_PARAMS['cooldown_bars_after_stop']}).",
    )
    parser.add_argument('--take-profit-rr', type=float, default=DEFAULT_PARAMS['take_profit_rr'],
                       help=f"Take profit R:R (default: {DEFAULT_PARAMS['take_profit_rr']})")
    
    # System parameters
    parser.add_argument('--update-interval', type=float, default=5.0,
                       help='Seconds between predictions (default: 5.0)')
    parser.add_argument('--warmup-trades', type=int, default=1000,
                       help='Trades before starting (default: 1000)')
    parser.add_argument(
        '--lookback-days',
        type=float,
        default=None,
        help='Limit trade history to the most recent N days (default: use train_config if set)',
    )
    parser.add_argument('--log-file', type=str, default=None,
                       help='Log file path')
    parser.add_argument('--bootstrap-csv', type=str, default=None,
                       help='Optional CSV file OR directory of CSVs to seed the buffer (timestamp,price,size,side)')
    parser.add_argument('--use-incremental', action='store_true', default=True,
                       help='Use incremental feature calculation for faster updates (default: True)')
    parser.add_argument('--no-incremental', dest='use_incremental', action='store_false',
                       help='Disable incremental features and use full recalculation')

    args = parser.parse_args()

    use_calibration = bool(args.use_calibration)
    if args.use_raw_probabilities:
        use_calibration = False
    
    # Setup logging
    setup_logging(args.log_file)
    logger = logging.getLogger(__name__)
    
    # Check model directory
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        logger.error(f"Model directory not found: {model_dir}")
        return
    
    # Create and start trader
    use_trend_gate = DEFAULT_PARAMS['use_trend_gate']
    if args.use_trend_gate:
        use_trend_gate = True
    elif args.no_trend_gate:
        use_trend_gate = False

    use_regime_gate = DEFAULT_PARAMS['use_regime_gate']
    if args.use_regime_gate:
        use_regime_gate = True
    elif args.no_regime_gate:
        use_regime_gate = False

    trader = LivePaperTrader(
        model_dir=model_dir,
        symbol=args.symbol,
        testnet=args.testnet,
        min_quality=args.min_quality,
        min_trend_prob=args.min_trend_prob,
        use_trend_gate=use_trend_gate,
        use_regime_gate=use_regime_gate,
        min_regime_prob=args.min_regime_prob,
        min_bounce_prob=args.min_bounce_prob,
        max_bounce_prob=args.max_bounce_prob,
        trade_side=args.trade_side,
        stop_loss_atr=args.stop_loss_atr,
        stop_padding_pct=args.stop_padding_pct,
        take_profit_rr=args.take_profit_rr,
        use_dynamic_rr=args.use_dynamic_rr,
        use_calibration=use_calibration,
        use_incremental=args.use_incremental,
        cooldown_bars_after_stop=args.cooldown_bars_after_stop,
        update_interval=args.update_interval,
        warmup_trades=args.warmup_trades,
        bootstrap_csv=args.bootstrap_csv,
        lookback_days=args.lookback_days,
    )
    
    try:
        trader.start()
    except KeyboardInterrupt:
        trader.stop()


if __name__ == "__main__":
    main()
