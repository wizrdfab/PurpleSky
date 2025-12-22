"""
Simple backtester for TrendFollower strategy.
Evaluates model predictions on historical data.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from config import TrendFollowerConfig, DEFAULT_CONFIG
from models import TrendFollowerModels
from feature_engine import get_feature_columns
from diagnostic_logger import DiagnosticLogger


@dataclass
class Trade:
    """Record of a single trade"""
    entry_time: datetime
    exit_time: datetime
    direction: int  # 1 for long, -1 for short
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_percent: float
    signal_quality: str
    exit_reason: str
    stop_loss: float = None
    take_profit: float = None
    # Additional diagnostic info
    trend_prob: float = 0.0
    bounce_prob: float = 0.0
    is_pullback: bool = False
    trend_aligned: bool = False
    dist_from_ema: float = 0.0


@dataclass
class BacktestResult:
    """Results from a backtest run"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pnl_percent: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    sharpe_ratio: float = 0.0
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    trades_by_grade: Dict[str, int] = field(default_factory=dict)
    win_rate_by_grade: Dict[str, float] = field(default_factory=dict)


class SimpleBacktester:
    """Simple event-driven backtester for model predictions."""

    def __init__(
        self,
        models: TrendFollowerModels,
        config: TrendFollowerConfig = DEFAULT_CONFIG,
        initial_capital: float = 10000.0,
        position_size_pct: float = 0.02,
        stop_loss_atr: float = 1.0,
        stop_padding_pct: float = 0.0,  # Extra stop distance as fraction of entry price (0.0 = disabled)
        take_profit_rr: float = 1.5,
        min_quality: str = 'B',
        min_trend_prob: float = 0.5,
        min_bounce_prob: float = 0.5,
        max_bounce_prob: float = 1.0,  # Max bounce prob for bucket filtering
        diagnostic_logger: Optional[DiagnosticLogger] = None,
        cooldown_bars_after_stop: int = 0,
        trade_side: str = "long",
        use_raw_probabilities: bool = False,
        use_dynamic_rr: bool = False,  # Use expected_rr from model for TP sizing
        use_ema_touch_entry: bool = True,  # Use touch-based EMA entry detection
        touch_threshold_atr: float = 0.3,  # How close to EMA counts as touch
        ema_touch_mode: str = "base",  # "base" or "multi" (uses ema_touch_detected)
        raw_trades: Optional[pd.DataFrame] = None,  # Raw trades for precise TP/SL
        use_calibration: bool = False,  # Use Isotonic Regression calibrated probabilities
    ):
        self.models = models
        self.config = config
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.stop_loss_atr = stop_loss_atr
        self.stop_padding_pct = stop_padding_pct
        self.take_profit_rr = take_profit_rr
        self.min_quality = min_quality
        self.min_trend_prob = min_trend_prob
        self.min_bounce_prob = min_bounce_prob
        self.max_bounce_prob = max_bounce_prob
        self.cooldown_bars_after_stop = max(0, int(cooldown_bars_after_stop))
        self.trade_side = (trade_side or "long").strip().lower()
        self.use_dynamic_rr = use_dynamic_rr
        self.use_ema_touch_entry = use_ema_touch_entry
        self.touch_threshold_atr = touch_threshold_atr
        self.ema_touch_mode = (ema_touch_mode or "base").strip().lower()
        if self.ema_touch_mode not in {"base", "multi"}:
            self.ema_touch_mode = "base"
        self.raw_trades = raw_trades
        self.use_calibration = use_calibration
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
        self.base_tf = config.features.timeframe_names[config.base_timeframe_idx]
        self.diag = diagnostic_logger
        
        self.capital = initial_capital
        self.position: Optional[Dict] = None
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [initial_capital]
        self.next_entry_bar: int = 0
        
        # Diagnostic tracking
        self._signal_stats = {
            'total_bars': 0,
            'signals_checked': 0,
            'trend_up_signals': 0,
            'trend_down_signals': 0,
            'trend_neutral_signals': 0,
            'quality_A': 0,
            'quality_B': 0,
            'quality_C': 0,
            'rejected_quality': 0,
            'rejected_trend_prob': 0,
            'rejected_bounce_prob': 0,
            'rejected_max_bounce_prob': 0,
            'rejected_no_ema_touch': 0,
            'rejected_cooldown': 0,
            'rejected_trade_side': 0,
            'accepted_signals': 0,
        }

        # Build index for raw trades if provided (for precise intrabar TP/SL detection)
        self._trades_by_bar = None
        if self.raw_trades is not None:
            self._build_trades_index()

        # Grade analysis tracking
        self._grade_analysis = {
            'A': {'signals': [], 'trades': []},
            'B': {'signals': [], 'trades': []},
            'C': {'signals': [], 'trades': []},
        }

    def _get_entry_feature_names(self) -> Optional[List[str]]:
        entry_model = getattr(self.models, "entry_model", None)
        if entry_model is None:
            return None
        filtered = getattr(entry_model, "filtered_feature_names", None)
        if filtered:
            return list(filtered)
        feature_names = getattr(entry_model, "feature_names", None)
        if feature_names:
            return list(feature_names)
        return None

    def _validate_feature_columns(self, feature_cols: List[str]) -> List[str]:
        expected = self._get_entry_feature_names()
        if not expected:
            return feature_cols

        missing = [col for col in expected if col not in feature_cols]
        extra = [col for col in feature_cols if col not in expected]
        if missing or extra:
            message = (
                "Feature mismatch between backtest data and entry model. "
                f"Missing: {len(missing)}, Extra: {len(extra)}. "
                "Rebuild features with the same config used for training "
                "(see train_config.json in the model directory)."
            )
            raise ValueError(message)

        if feature_cols != expected:
            return expected
        return feature_cols

    def _build_trades_index(self):
        """Build an index mapping bar_time to trades for fast lookup."""
        if self.raw_trades is None:
            return

        # Get the base timeframe in seconds
        base_tf_seconds = self.config.features.timeframes[self.config.base_timeframe_idx]

        # Create bar_time column for trades
        trades_df = self.raw_trades.copy()
        timestamp_col = self.config.data.timestamp_col
        trades_df['bar_time'] = (trades_df[timestamp_col] // base_tf_seconds) * base_tf_seconds

        # Group trades by bar_time
        self._trades_by_bar = {}
        for bar_time, group in trades_df.groupby('bar_time'):
            # Sort trades by timestamp within the bar
            sorted_trades = group.sort_values(timestamp_col)
            self._trades_by_bar[bar_time] = sorted_trades

    def _check_exit_with_trades(self, row, bar_time: int) -> Tuple[Optional[str], Optional[float]]:
        """
        Check exit using raw trade data for precise intrabar TP/SL detection.

        This provides exact determination of whether TP or SL was hit first
        by walking through actual trades that occurred during the bar.

        Returns:
            Tuple of (exit_reason, exit_price) or (None, None) if no exit
        """
        if self._trades_by_bar is None or bar_time not in self._trades_by_bar:
            return None, None

        trades_in_bar = self._trades_by_bar[bar_time]
        price_col = self.config.data.price_col

        direction = self.position['direction']
        stop = self.position['stop_loss']
        target = self.position['take_profit']

        # Walk through each trade in order
        for _, trade in trades_in_bar.iterrows():
            trade_price = trade[price_col]

            if direction == 1:  # Long position
                if trade_price <= stop:
                    return 'stop_loss', stop
                if trade_price >= target:
                    return 'take_profit', target
            else:  # Short position
                if trade_price >= stop:
                    return 'stop_loss', stop
                if trade_price <= target:
                    return 'take_profit', target

        return None, None

    def run(self, data: pd.DataFrame, feature_cols: List[str]) -> BacktestResult:
        """Run backtest on historical data."""
        self.capital = self.initial_capital
        self.position = None
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.next_entry_bar = 0
        
        # Reset signal stats
        self._signal_stats = {k: 0 for k in self._signal_stats}
        self._signal_stats['total_bars'] = len(data)
        
        print(f"Running backtest on {len(data):,} bars...")
        
        if self.diag:
            self.diag.log_section("Backtest Execution")
            self.diag.log_metric("backtest_bars", len(data))
            self.diag.log_metric("backtest_initial_capital", self.initial_capital)
            self.diag.log_metric("backtest_min_quality", self.min_quality)
            self.diag.log_metric("backtest_min_trend_prob", self.min_trend_prob)
            self.diag.log_metric("backtest_min_bounce_prob", self.min_bounce_prob)
            self.diag.log_metric("backtest_stop_loss_atr", self.stop_loss_atr)
            self.diag.log_metric("backtest_stop_padding_pct", self.stop_padding_pct)
            self.diag.log_metric("backtest_take_profit_rr", self.take_profit_rr)
            self.diag.log_metric("backtest_cooldown_bars_after_stop", self.cooldown_bars_after_stop)
            self.diag.log_metric("backtest_use_calibration", self.use_calibration)
            self.diag.log_raw(f"\n  trade_side: {self.trade_side}\n")
            self.diag.log_raw(f"  use_calibration: {self.use_calibration}\n")
            
            # VERIFICATION: Confirm feature_cols doesn't contain label columns
            label_patterns = ['label', 'success', 'outcome', '_mfe', '_mae', '_rr', 'regime']
            leaky_features = [col for col in feature_cols if any(p in col.lower() for p in label_patterns)]
            if leaky_features:
                self.diag.log_error(f"LEAKAGE DETECTED! Feature columns contain labels: {leaky_features}")
            else:
                self.diag.log_raw("\n  ✓ VERIFIED: No label columns in feature_cols\n")
            
            # Log which columns are being used
            self.diag.log_metric("feature_count", len(feature_cols))
            self.diag.log_raw(f"\n  First 10 feature columns: {feature_cols[:10]}\n")

        feature_cols = self._validate_feature_columns(feature_cols)

        atr_col = f'{self.base_tf}_atr'
        
        n = len(data)
        for i in range(n):
            row = data.iloc[i]
            
            current_price = row['close']
            current_atr = row[atr_col] if atr_col in row else current_price * 0.02
            
            if self.position is not None:
                # Exit decisions are evaluated on the CURRENT bar's high/low.
                # This avoids skipping the first bar after entry (off-by-one),
                # and matches "enter on close, exit during subsequent bars" logic.
                self._check_exit(row, current_atr, i)
            
            if self.position is None:
                if self.cooldown_bars_after_stop > 0 and i < self.next_entry_bar:
                    self._signal_stats['rejected_cooldown'] += 1
                else:
                    # Do not open on the final bar (no future bar to realize stop/TP).
                    if i < (n - 1):
                        self._check_entry(row, data.iloc[[i]], feature_cols, current_price, current_atr, i)
            
            if self.position is not None:
                unrealized = self._calculate_unrealized_pnl(current_price)
                self.equity_curve.append(self.capital + unrealized)
            else:
                self.equity_curve.append(self.capital)
        
        return self._calculate_results()
    
    def _check_entry(self, row, features_df, feature_cols, price, atr, bar_idx):
        """Check if we should enter a position."""
        self._signal_stats['signals_checked'] += 1

        # Build feature DataFrame efficiently (avoid fragmentation)
        feature_data = {}
        for col in feature_cols:
            if col in features_df.columns:
                feature_data[col] = features_df[col].fillna(0).values
            else:
                feature_data[col] = [0]

        X = pd.DataFrame(feature_data, index=features_df.index)

        entry_pred = self.models.entry_model.predict(X, use_calibration=self.use_calibration)

        bounce_prob = entry_pred['bounce_prob'][0]

        # Get expected RR from model if available and dynamic RR is enabled
        expected_rr = None
        if self.use_dynamic_rr and 'expected_rr' in entry_pred:
            expected_rr = entry_pred['expected_rr'][0]

        # Use EMA slope (normalized) for directional bias instead of EMA stacking
        slope_col = f'{self.base_tf}_ema_{self.config.labels.pullback_ema}_slope_norm'
        alignment = row.get(slope_col, 0)
        if alignment is None or pd.isna(alignment) or float(alignment) == 0.0:
            self._signal_stats['trend_neutral_signals'] += 1
            return

        alignment = float(alignment)
        trend_dir = 1 if alignment > 0 else -1
        if trend_dir == 1:
            self._signal_stats['trend_up_signals'] += 1
            if not self.allow_long:
                self._signal_stats['rejected_trade_side'] += 1
                return
            prob_up = 1.0
            prob_down = 0.0
        else:
            self._signal_stats['trend_down_signals'] += 1
            if not self.allow_short:
                self._signal_stats['rejected_trade_side'] += 1
                return
            prob_up = 0.0
            prob_down = 1.0

        # =========================================================
        # EMA TOUCH DETECTION (intrabar touch vs close-based)
        # =========================================================
        ema_col = f'{self.base_tf}_ema_{self.config.labels.pullback_ema}'
        ema_touched = False
        dist_from_ema = 999

        if ema_col in row.index and atr > 0:
            ema = row[ema_col]

            if self.use_ema_touch_entry:
                if self.ema_touch_mode == "multi" and "ema_touch_detected" in row.index:
                    raw_touch = row.get("ema_touch_detected", False)
                    if raw_touch is None or pd.isna(raw_touch):
                        ema_touched = False
                    else:
                        ema_touched = bool(raw_touch)

                    touch_dir = row.get("ema_touch_direction", 0)
                    if touch_dir is None or pd.isna(touch_dir):
                        touch_dir = 0
                    try:
                        touch_dir = int(touch_dir)
                    except Exception:
                        touch_dir = 0
                    if ema_touched and touch_dir not in (0, trend_dir):
                        ema_touched = False

                    touch_dist = row.get("ema_touch_dist", None)
                    if touch_dist is not None and not pd.isna(touch_dist):
                        dist_from_ema = abs(float(touch_dist))
                else:
                    # TOUCH-BASED (base TF): Check if bar's high/low touched EMA within threshold
                    # For LONG: low should touch EMA (price dips down to EMA)
                    # For SHORT: high should touch EMA (price rises up to EMA)
                    if trend_dir == 1:
                        # Long setup: check if LOW touched EMA
                        dist_low_to_ema = (row['low'] - ema) / atr
                        if -self.touch_threshold_atr <= dist_low_to_ema <= self.touch_threshold_atr:
                            # Verify close is above or at EMA (bullish context)
                            if row['close'] >= ema or (row['high'] + row['low']) / 2 >= ema:
                                ema_touched = True
                                dist_from_ema = abs(dist_low_to_ema)
                    else:
                        # Short setup: check if HIGH touched EMA
                        dist_high_to_ema = (row['high'] - ema) / atr
                        if -self.touch_threshold_atr <= dist_high_to_ema <= self.touch_threshold_atr:
                            # Verify close is below or at EMA (bearish context)
                            if row['close'] <= ema or (row['high'] + row['low']) / 2 <= ema:
                                ema_touched = True
                                dist_from_ema = abs(dist_high_to_ema)
            else:
                # CLOSE-BASED (legacy): use close price distance
                dist_from_ema = abs(price - ema) / atr
                ema_touched = dist_from_ema <= self.config.labels.pullback_threshold

        is_pullback = ema_touched
        trend_aligned = True  # by construction (direction follows slope sign)

        # Reject if EMA touch is required but not detected
        if self.use_ema_touch_entry and not ema_touched:
            self._signal_stats['rejected_no_ema_touch'] += 1
            return

        # Calculate trend_prob early for signal tracking (not used for gating)
        trend_prob = prob_up if trend_dir == 1 else prob_down

        if bounce_prob > 0.6 and trend_aligned and is_pullback:
            quality = 'A'
        elif bounce_prob > 0.5 and (trend_aligned or is_pullback):
            quality = 'B'
        else:
            quality = 'C'

        # Track quality distribution
        self._signal_stats[f'quality_{quality}'] += 1

        # Track detailed grade analysis for all signals (not just accepted)
        signal_info = {
            'time': row.get('datetime', None),
            'price': price,
            'direction': trend_dir,
            'trend_prob': trend_prob,
            'bounce_prob': bounce_prob,
            'is_pullback': is_pullback,
            'trend_aligned': trend_aligned,
            'alignment': alignment,
            'dist_from_ema': dist_from_ema,
            'atr': atr,
        }
        self._grade_analysis[quality]['signals'].append(signal_info)

        # Quality is informational; do not block entries
        quality_ok = True

        # Check probability thresholds (both min AND max for bucket filtering)
        trend_prob_ok = True
        bounce_min_ok = bounce_prob >= self.min_bounce_prob
        bounce_max_ok = bounce_prob <= self.max_bounce_prob

        # Track rejection reasons
        if not bounce_min_ok:
            self._signal_stats['rejected_bounce_prob'] += 1
            return
        if not bounce_max_ok:
            self._signal_stats['rejected_max_bounce_prob'] += 1
            return

        if quality_ok:
            self._signal_stats['accepted_signals'] += 1

            entry_price = price
            stop_dist = (self.stop_loss_atr * atr) + (self.stop_padding_pct * entry_price)
            stop_loss = entry_price - (trend_dir * stop_dist)

            # Use dynamic RR from model if available, else use fixed take_profit_rr
            effective_rr = self.take_profit_rr
            if self.use_dynamic_rr and expected_rr is not None and expected_rr > 0.5:
                # Use model's expected RR, but cap it at reasonable values
                effective_rr = min(max(expected_rr, 0.5), 5.0)

            take_profit = entry_price + (trend_dir * stop_dist * effective_rr)

            risk_amount = self.capital * self.position_size_pct
            risk_per_unit = abs(entry_price - stop_loss)
            size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0

            self.position = {
                'direction': trend_dir,
                'entry_price': price,
                'entry_time': row.get('datetime', datetime.now()),
                'size': size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'quality': quality,
                # Additional diagnostic info
                'trend_prob': trend_prob,
                'bounce_prob': bounce_prob,
                'is_pullback': is_pullback,
                'trend_aligned': trend_aligned,
                'dist_from_ema': dist_from_ema,
                'expected_rr': expected_rr,
            }
    
    def _check_exit(self, row, atr, bar_idx):
        """
        Check if we should exit current position using this bar's OHLC.

        If raw trade data is available, uses precise intrabar detection.
        Otherwise, uses open price to infer which level was hit first.
        """
        # Try precise detection with raw trades first
        if self._trades_by_bar is not None and 'bar_time' in row.index:
            bar_time = int(row['bar_time'])
            exit_reason, exit_price = self._check_exit_with_trades(row, bar_time)
            if exit_reason is not None:
                self._close_position(exit_price, row.get('datetime', datetime.now()), exit_reason, bar_idx)
                return

        # Fallback to OHLC-based detection
        high = row['high']
        low = row['low']
        open_price = row['open']

        direction = self.position['direction']
        stop = self.position['stop_loss']
        target = self.position['take_profit']

        exit_reason = None
        exit_price = None

        if direction == 1:  # Long position
            sl_hit = low <= stop
            tp_hit = high >= target

            if sl_hit and tp_hit:
                # BOTH levels hit on same bar - use open to infer which was first
                dist_to_sl = abs(open_price - stop)
                dist_to_tp = abs(open_price - target)

                if dist_to_tp <= dist_to_sl:
                    # Open closer to TP or equidistant -> assume TP hit first
                    exit_reason = 'take_profit'
                    exit_price = target
                else:
                    # Open closer to SL -> assume SL hit first
                    exit_reason = 'stop_loss'
                    exit_price = stop
            elif sl_hit:
                exit_reason = 'stop_loss'
                exit_price = stop
            elif tp_hit:
                exit_reason = 'take_profit'
                exit_price = target

        else:  # Short position (direction == -1)
            sl_hit = high >= stop
            tp_hit = low <= target

            if sl_hit and tp_hit:
                # BOTH levels hit on same bar - use open to infer which was first
                dist_to_sl = abs(open_price - stop)
                dist_to_tp = abs(open_price - target)

                if dist_to_tp <= dist_to_sl:
                    exit_reason = 'take_profit'
                    exit_price = target
                else:
                    exit_reason = 'stop_loss'
                    exit_price = stop
            elif sl_hit:
                exit_reason = 'stop_loss'
                exit_price = stop
            elif tp_hit:
                exit_reason = 'take_profit'
                exit_price = target

        if exit_reason:
            self._close_position(exit_price, row.get('datetime', datetime.now()), exit_reason, bar_idx)
    
    def _close_position(self, exit_price, exit_time, exit_reason, bar_idx):
        """Close current position and record trade."""
        direction = self.position['direction']
        entry_price = self.position['entry_price']
        size = self.position['size']
        
        pnl = direction * (exit_price - entry_price) * size
        pnl_percent = direction * (exit_price - entry_price) / entry_price * 100
        
        trade = Trade(
            entry_time=self.position['entry_time'],
            exit_time=exit_time,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            size=size,
            pnl=pnl,
            pnl_percent=pnl_percent,
            signal_quality=self.position['quality'],
            exit_reason=exit_reason,
            stop_loss=self.position.get('stop_loss'),
            take_profit=self.position.get('take_profit'),
            trend_prob=self.position.get('trend_prob', 0.0),
            bounce_prob=self.position.get('bounce_prob', 0.0),
            is_pullback=self.position.get('is_pullback', False),
            trend_aligned=self.position.get('trend_aligned', False),
            dist_from_ema=self.position.get('dist_from_ema', 0.0),
        )
        
        # Track trade in grade analysis
        grade = self.position['quality']
        self._grade_analysis[grade]['trades'].append({
            'pnl': pnl,
            'win': pnl > 0,
            'exit_reason': exit_reason,
            'trend_prob': self.position.get('trend_prob', 0.0),
            'bounce_prob': self.position.get('bounce_prob', 0.0),
            'is_pullback': self.position.get('is_pullback', False),
            'trend_aligned': self.position.get('trend_aligned', False),
            'dist_from_ema': self.position.get('dist_from_ema', 0.0),
            'direction': direction,
        })
        
        self.trades.append(trade)
        self.capital += pnl
        self.position = None
        
        if exit_reason == 'stop_loss' and self.cooldown_bars_after_stop > 0:
            self.next_entry_bar = max(self.next_entry_bar, bar_idx + self.cooldown_bars_after_stop)
    
    def _calculate_unrealized_pnl(self, current_price):
        """Calculate unrealized P&L for open position."""
        if self.position is None:
            return 0.0
        
        direction = self.position['direction']
        entry_price = self.position['entry_price']
        size = self.position['size']
        
        return direction * (current_price - entry_price) * size
    
    def _calculate_results(self) -> BacktestResult:
        """Calculate backtest results and metrics."""
        result = BacktestResult()
        result.trades = self.trades
        result.equity_curve = self.equity_curve
        
        # Log signal statistics to diagnostics
        if self.diag:
            self.diag.log_raw("\n  Signal Statistics:\n")
            for key, value in self._signal_stats.items():
                self.diag.log_metric(f"backtest_{key}", value)
            
            # Calculate percentages for better understanding
            total = self._signal_stats['signals_checked']
            if total > 0:
                self.diag.log_raw(f"\n  Signal Breakdown (of {total} bars checked):\n")
                self.diag.log_raw(f"    Trend UP predictions:    {self._signal_stats['trend_up_signals']} ({self._signal_stats['trend_up_signals']/total:.1%})\n")
                self.diag.log_raw(f"    Trend DOWN predictions:  {self._signal_stats['trend_down_signals']} ({self._signal_stats['trend_down_signals']/total:.1%})\n")
                self.diag.log_raw(f"    Trend NEUTRAL predictions: {self._signal_stats['trend_neutral_signals']} ({self._signal_stats['trend_neutral_signals']/total:.1%})\n")
                self.diag.log_raw(f"\n    Quality A signals: {self._signal_stats['quality_A']}\n")
                self.diag.log_raw(f"    Quality B signals: {self._signal_stats['quality_B']}\n")
                self.diag.log_raw(f"    Quality C signals: {self._signal_stats['quality_C']}\n")
                self.diag.log_raw(f"\n    Rejected (quality):      {self._signal_stats['rejected_quality']}\n")
                self.diag.log_raw(f"    Rejected (trend prob):   {self._signal_stats['rejected_trend_prob']}\n")
                self.diag.log_raw(f"    Rejected (bounce prob):  {self._signal_stats['rejected_bounce_prob']}\n")
                self.diag.log_raw(f"    ACCEPTED signals:        {self._signal_stats['accepted_signals']}\n")
        
        if not self.trades:
            if self.diag:
                self.diag.log_warning("No trades were executed in backtest!")
            return result
        
        result.total_trades = len(self.trades)
        result.winning_trades = sum(1 for t in self.trades if t.pnl > 0)
        result.losing_trades = sum(1 for t in self.trades if t.pnl <= 0)
        result.win_rate = result.winning_trades / result.total_trades
        
        result.total_pnl = sum(t.pnl for t in self.trades)
        result.total_pnl_percent = (self.capital - self.initial_capital) / self.initial_capital * 100
        
        wins = [t.pnl for t in self.trades if t.pnl > 0]
        losses = [t.pnl for t in self.trades if t.pnl <= 0]
        
        result.avg_win = np.mean(wins) if wins else 0.0
        result.avg_loss = np.mean(losses) if losses else 0.0
        
        total_wins = sum(wins)
        total_losses = abs(sum(losses))
        result.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = equity - peak
        result.max_drawdown = abs(min(drawdown))
        result.max_drawdown_percent = result.max_drawdown / max(peak) * 100 if max(peak) > 0 else 0.0
        
        returns = np.diff(equity) / equity[:-1]
        if len(returns) > 1 and np.std(returns) > 0:
            result.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        
        for grade in ['A', 'B', 'C']:
            grade_trades = [t for t in self.trades if t.signal_quality == grade]
            result.trades_by_grade[grade] = len(grade_trades)
            if grade_trades:
                result.win_rate_by_grade[grade] = sum(1 for t in grade_trades if t.pnl > 0) / len(grade_trades)
        
        # Log detailed trade info to diagnostics
        if self.diag:
            self.diag.log_raw("\n  Trade Details:\n")
            self.diag.log_raw(f"    {'#':<3} {'Entry Time':<20} {'Exit Time':<20} {'Dir':<5} {'Entry':<10} {'Exit':<10} {'P&L':<10} {'Result':<6} {'Grade':<5} {'Exit Reason':<12}\n")
            self.diag.log_raw(f"    {'-'*115}\n")
            
            for i, t in enumerate(self.trades):
                dir_str = "LONG" if t.direction == 1 else "SHORT"
                result_str = "WIN" if t.pnl > 0 else "LOSS"
                
                # Format times
                entry_time_str = t.entry_time.strftime('%Y-%m-%d %H:%M') if hasattr(t.entry_time, 'strftime') else str(t.entry_time)[:16]
                exit_time_str = t.exit_time.strftime('%Y-%m-%d %H:%M') if hasattr(t.exit_time, 'strftime') else str(t.exit_time)[:16]
                
                self.diag.log_raw(f"    {i+1:<3} {entry_time_str:<20} {exit_time_str:<20} {dir_str:<5} {t.entry_price:<10.6f} {t.exit_price:<10.6f} ${t.pnl:<9.2f} {result_str:<6} {t.signal_quality:<5} {t.exit_reason:<12}\n")
            
            # Additional trade statistics
            self.diag.log_raw(f"\n  Trade Probabilities:\n")
            self.diag.log_raw(f"    {'#':<3} {'TrendP':<8} {'BounceP':<9} {'Pullback':<10} {'Aligned':<10}\n")
            self.diag.log_raw(f"    {'-'*45}\n")
            for i, t in enumerate(self.trades):
                pb_str = "Yes" if t.is_pullback else "No"
                aligned_str = "Yes" if t.trend_aligned else "No"
                self.diag.log_raw(f"    {i+1:<3} {t.trend_prob:<8.1%} {t.bounce_prob:<9.1%} {pb_str:<10} {aligned_str:<10}\n")
            
            self.diag.log_metric("backtest_total_trades", result.total_trades)
            self.diag.log_metric("backtest_win_rate", result.win_rate, warn_if='high')
            self.diag.log_metric("backtest_total_pnl", result.total_pnl)
            self.diag.log_metric("backtest_profit_factor", result.profit_factor)
            
            # Log position handling verification
            self.diag.log_raw(f"\n  ✓ VERIFIED: One position at a time (matches live trading)\n")
            
            # =====================================================
            # GRADE ANALYSIS - WHY ARE A-GRADES UNDERPERFORMING?
            # =====================================================
            self.diag.log_raw(f"\n\n{'='*80}\n")
            self.diag.log_raw(f"GRADE ANALYSIS - INVESTIGATING A-GRADE PERFORMANCE\n")
            self.diag.log_raw(f"{'='*80}\n")
            
            # Grade definitions reminder
            self.diag.log_raw(f"\n  Grade Definitions:\n")
            self.diag.log_raw(f"    A: bounce_prob > 0.6 AND trend_aligned AND is_pullback\n")
            self.diag.log_raw(f"    B: bounce_prob > 0.5 AND (trend_aligned OR is_pullback)\n")
            self.diag.log_raw(f"    C: Everything else\n")
            
            # Analyze each grade
            for grade in ['A', 'B', 'C']:
                signals = self._grade_analysis[grade]['signals']
                trades = self._grade_analysis[grade]['trades']
                
                self.diag.log_raw(f"\n  --- Grade {grade} Analysis ---\n")
                self.diag.log_raw(f"    Total signals: {len(signals)}\n")
                self.diag.log_raw(f"    Trades executed: {len(trades)}\n")
                
                if trades:
                    wins = [t for t in trades if t['win']]
                    losses = [t for t in trades if not t['win']]
                    win_rate = len(wins) / len(trades) * 100
                    
                    self.diag.log_raw(f"    Win rate: {win_rate:.1f}% ({len(wins)}W / {len(losses)}L)\n")
                    
                    # Analyze winning vs losing trades
                    if wins:
                        avg_trend_prob_wins = sum(t['trend_prob'] for t in wins) / len(wins)
                        avg_bounce_prob_wins = sum(t['bounce_prob'] for t in wins) / len(wins)
                        avg_dist_ema_wins = sum(t['dist_from_ema'] for t in wins) / len(wins)
                        pullback_pct_wins = sum(1 for t in wins if t['is_pullback']) / len(wins) * 100
                        aligned_pct_wins = sum(1 for t in wins if t['trend_aligned']) / len(wins) * 100
                        
                        self.diag.log_raw(f"\n    WINNING trades ({len(wins)}):\n")
                        self.diag.log_raw(f"      Avg trend_prob:   {avg_trend_prob_wins:.1%}\n")
                        self.diag.log_raw(f"      Avg bounce_prob:  {avg_bounce_prob_wins:.1%}\n")
                        self.diag.log_raw(f"      Avg dist_from_ema: {avg_dist_ema_wins:.2f} ATR\n")
                        self.diag.log_raw(f"      % in pullback:    {pullback_pct_wins:.0f}%\n")
                        self.diag.log_raw(f"      % trend aligned:  {aligned_pct_wins:.0f}%\n")
                        
                        # Exit reasons for wins
                        exit_reasons_wins = {}
                        for t in wins:
                            r = t['exit_reason']
                            exit_reasons_wins[r] = exit_reasons_wins.get(r, 0) + 1
                        self.diag.log_raw(f"      Exit reasons: {exit_reasons_wins}\n")
                    
                    if losses:
                        avg_trend_prob_losses = sum(t['trend_prob'] for t in losses) / len(losses)
                        avg_bounce_prob_losses = sum(t['bounce_prob'] for t in losses) / len(losses)
                        avg_dist_ema_losses = sum(t['dist_from_ema'] for t in losses) / len(losses)
                        pullback_pct_losses = sum(1 for t in losses if t['is_pullback']) / len(losses) * 100
                        aligned_pct_losses = sum(1 for t in losses if t['trend_aligned']) / len(losses) * 100
                        
                        self.diag.log_raw(f"\n    LOSING trades ({len(losses)}):\n")
                        self.diag.log_raw(f"      Avg trend_prob:   {avg_trend_prob_losses:.1%}\n")
                        self.diag.log_raw(f"      Avg bounce_prob:  {avg_bounce_prob_losses:.1%}\n")
                        self.diag.log_raw(f"      Avg dist_from_ema: {avg_dist_ema_losses:.2f} ATR\n")
                        self.diag.log_raw(f"      % in pullback:    {pullback_pct_losses:.0f}%\n")
                        self.diag.log_raw(f"      % trend aligned:  {aligned_pct_losses:.0f}%\n")
                        
                        # Exit reasons for losses
                        exit_reasons_losses = {}
                        for t in losses:
                            r = t['exit_reason']
                            exit_reasons_losses[r] = exit_reasons_losses.get(r, 0) + 1
                        self.diag.log_raw(f"      Exit reasons: {exit_reasons_losses}\n")
                    
                    # Direction analysis
                    longs = [t for t in trades if t['direction'] == 1]
                    shorts = [t for t in trades if t['direction'] == -1]
                    if longs:
                        long_wr = sum(1 for t in longs if t['win']) / len(longs) * 100
                        self.diag.log_raw(f"\n    LONG trades: {len(longs)}, win rate: {long_wr:.0f}%\n")
                    if shorts:
                        short_wr = sum(1 for t in shorts if t['win']) / len(shorts) * 100
                        self.diag.log_raw(f"    SHORT trades: {len(shorts)}, win rate: {short_wr:.0f}%\n")
                
                else:
                    self.diag.log_raw(f"    No trades executed for this grade\n")
            
            # Summary comparison
            self.diag.log_raw(f"\n  --- Summary Comparison ---\n")
            self.diag.log_raw(f"    {'Grade':<6} {'Signals':<10} {'Trades':<8} {'Win Rate':<10} {'Avg BounceP':<12}\n")
            self.diag.log_raw(f"    {'-'*50}\n")
            
            for grade in ['A', 'B', 'C']:
                signals = self._grade_analysis[grade]['signals']
                trades = self._grade_analysis[grade]['trades']
                
                if trades:
                    wr = sum(1 for t in trades if t['win']) / len(trades) * 100
                    avg_bp = sum(t['bounce_prob'] for t in trades) / len(trades) * 100
                else:
                    wr = 0
                    avg_bp = 0
                
                self.diag.log_raw(f"    {grade:<6} {len(signals):<10} {len(trades):<8} {wr:<10.1f}% {avg_bp:<12.1f}%\n")
            
            # Hypothesis about A-grade underperformance
            a_trades = self._grade_analysis['A']['trades']
            b_trades = self._grade_analysis['B']['trades']
            
            if a_trades and b_trades:
                a_avg_bounce = sum(t['bounce_prob'] for t in a_trades) / len(a_trades)
                b_avg_bounce = sum(t['bounce_prob'] for t in b_trades) / len(b_trades)
                
                self.diag.log_raw(f"\n  --- Hypothesis Analysis ---\n")
                self.diag.log_raw(f"    A-grade avg bounce_prob: {a_avg_bounce:.1%}\n")
                self.diag.log_raw(f"    B-grade avg bounce_prob: {b_avg_bounce:.1%}\n")
                
                if a_avg_bounce > b_avg_bounce:
                    self.diag.log_raw(f"\n    ⚠️ A-grades have HIGHER bounce_prob but LOWER win rate!\n")
                    self.diag.log_raw(f"    Possible causes:\n")
                    self.diag.log_raw(f"    1. Entry model (bounce_prob) is overfitted and unreliable\n")
                    self.diag.log_raw(f"    2. 'is_pullback' requirement catches reversals, not pullbacks\n")
                    self.diag.log_raw(f"    3. High-confidence signals occur at market turning points\n")
                    self.diag.log_raw(f"    4. Small sample size - need more trades to validate\n")
        
        return result


def print_backtest_results(result: BacktestResult):
    """Pretty print backtest results."""
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    
    print(f"\n  Total Trades:     {result.total_trades}")
    print(f"  Winning Trades:   {result.winning_trades}")
    print(f"  Losing Trades:    {result.losing_trades}")
    print(f"  Win Rate:         {result.win_rate:.1%}")
    
    print(f"\n  Total P&L:        ${result.total_pnl:,.2f}")
    print(f"  Total Return:     {result.total_pnl_percent:.2f}%")
    print(f"  Avg Win:          ${result.avg_win:,.2f}")
    print(f"  Avg Loss:         ${result.avg_loss:,.2f}")
    print(f"  Profit Factor:    {result.profit_factor:.2f}")
    
    print(f"\n  Max Drawdown:     ${result.max_drawdown:,.2f}")
    print(f"  Max DD %:         {result.max_drawdown_percent:.2f}%")
    print(f"  Sharpe Ratio:     {result.sharpe_ratio:.2f}")
    
    print("\n  Performance by Signal Grade:")
    for grade in ['A', 'B', 'C']:
        count = result.trades_by_grade.get(grade, 0)
        wr = result.win_rate_by_grade.get(grade, 0)
        print(f"    Grade {grade}: {count} trades, {wr:.1%} win rate")
    
    print("=" * 60)


def run_backtest(data, models, feature_cols, config=DEFAULT_CONFIG, **kwargs) -> BacktestResult:
    """Convenience function to run a backtest."""
    backtester = SimpleBacktester(models, config, **kwargs)
    result = backtester.run(data, feature_cols)
    print_backtest_results(result)
    return result


def _trade_to_dict(trade: Trade, *, timeframe_seconds: Optional[int] = None) -> dict:
    """Serialize Trade dataclass into a JSON-friendly dict."""
    def _as_py(v):
        if isinstance(v, np.generic):
            return v.item()
        return v

    entry_bar_open_time = trade.entry_time
    exit_bar_open_time = trade.exit_time

    entry_bar_close_time = None
    exit_bar_close_time = None
    if timeframe_seconds is not None:
        if entry_bar_open_time is not None:
            entry_bar_close_time = entry_bar_open_time + timedelta(seconds=int(timeframe_seconds))
        if exit_bar_open_time is not None:
            exit_bar_close_time = exit_bar_open_time + timedelta(seconds=int(timeframe_seconds))

    return {
        # NOTE: Our bars are timestamped by OPEN time, but entries are executed on bar CLOSE (price = close).
        # To avoid confusion, we log both bar-open and bar-close timestamps.
        'entry_time': entry_bar_close_time.isoformat() if entry_bar_close_time else (entry_bar_open_time.isoformat() if entry_bar_open_time else None),
        'entry_bar_open_time': entry_bar_open_time.isoformat() if entry_bar_open_time else None,
        'entry_bar_close_time': entry_bar_close_time.isoformat() if entry_bar_close_time else None,
        'exit_time': exit_bar_open_time.isoformat() if exit_bar_open_time else None,
        'exit_bar_open_time': exit_bar_open_time.isoformat() if exit_bar_open_time else None,
        'exit_bar_close_time': exit_bar_close_time.isoformat() if exit_bar_close_time else None,
        'direction': trade.direction,
        'entry_price': _as_py(trade.entry_price),
        'exit_price': _as_py(trade.exit_price),
        'size': _as_py(trade.size),
        'pnl': _as_py(trade.pnl),
        'pnl_percent': _as_py(trade.pnl_percent),
        'signal_quality': trade.signal_quality,
        'exit_reason': trade.exit_reason,
        'stop_loss': _as_py(trade.stop_loss),
        'take_profit': _as_py(trade.take_profit),
        'trend_prob': _as_py(trade.trend_prob),
        'bounce_prob': _as_py(trade.bounce_prob),
        'is_pullback': _as_py(trade.is_pullback),
        'trend_aligned': _as_py(trade.trend_aligned),
        'dist_from_ema': _as_py(trade.dist_from_ema),
    }


def save_backtest_logs(
    result: BacktestResult,
    config: TrendFollowerConfig,
    log_dir: Path,
    *,
    model_dir: Optional[Path] = None,
    driver: Optional[str] = None,
    parameters: Optional[dict] = None,
    extra_metrics: Optional[dict] = None,
) -> Dict[str, Path]:
    """
    Save a backtest run summary + per-trade JSONL to disk.

    Intended for keeping an audit trail of which positions were opened and how they performed.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    symbol_tag = Path(getattr(config.data, 'data_dir', 'data')).name if hasattr(config, 'data') else 'data'
    base_tf = config.features.timeframe_names[config.base_timeframe_idx] if hasattr(config, 'features') else 'tf'
    prefix = f'backtest_{symbol_tag}_{base_tf}_{run_id}'

    trades_path = log_dir / f'{prefix}_trades.jsonl'
    summary_path = log_dir / f'{prefix}_summary.json'

    base_tf_seconds = None
    if hasattr(config, 'features') and hasattr(config.features, 'timeframes'):
        base_tf_seconds = int(config.features.timeframes[config.base_timeframe_idx])

    with trades_path.open('w', encoding='utf-8') as f:
        for t in result.trades:
            f.write(json.dumps(_trade_to_dict(t, timeframe_seconds=base_tf_seconds), ensure_ascii=False) + '\n')

    summary = {
        'run_id': run_id,
        'created_at': datetime.now().isoformat(),
        'driver': driver,
        'model_dir': str(model_dir) if model_dir is not None else None,
        'data_dir': str(getattr(config.data, 'data_dir', None)) if hasattr(config, 'data') else None,
        'base_timeframe': base_tf,
        'base_timeframe_seconds': base_tf_seconds,
        'parameters': parameters or {},
        'metrics': {
            'total_trades': result.total_trades,
            'winning_trades': result.winning_trades,
            'losing_trades': result.losing_trades,
            'win_rate': result.win_rate,
            'total_pnl': result.total_pnl,
            'total_return_percent': result.total_pnl_percent,
            'avg_win': result.avg_win,
            'avg_loss': result.avg_loss,
            'profit_factor': result.profit_factor,
            'max_drawdown': result.max_drawdown,
            'max_drawdown_percent': result.max_drawdown_percent,
            'sharpe_ratio': result.sharpe_ratio,
            'trades_by_grade': result.trades_by_grade,
            'win_rate_by_grade': result.win_rate_by_grade,
        },
        'extra_metrics': extra_metrics or {},
        'files': {
            'trades_jsonl': str(trades_path),
        },
    }

    with summary_path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return {'summary': summary_path, 'trades': trades_path}


if __name__ == "__main__":
    print("Backtest module loaded successfully")
