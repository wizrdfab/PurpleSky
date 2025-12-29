"""
Simple backtester for TrendFollower strategy.
Evaluates model predictions on historical data.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from config import TrendFollowerConfig, DEFAULT_CONFIG
from models import TrendFollowerModels, CONTEXT_FEATURE_NAMES, compute_expected_calibration_error, append_context_features
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
    expected_rr: float = 0.0
    expected_rr_mean: float = 0.0
    ev_value: float = 0.0
    implied_threshold: float = 0.0
    fee_r: float = 0.0
    stop_dist: float = 0.0
    realized_r: float = 0.0
    realized_r_net: float = 0.0


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
    signal_stats: Dict[str, int] = field(default_factory=dict)
    bounce_prob_stats: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


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
        min_trend_prob: Optional[float] = None,
        min_bounce_prob: float = 0.5,
        max_bounce_prob: float = 1.0,  # Max bounce prob for bucket filtering
        diagnostic_logger: Optional[DiagnosticLogger] = None,
        cooldown_bars_after_stop: int = 0,
        trade_side: str = "both",
        use_raw_probabilities: bool = False,
        use_dynamic_rr: bool = False,  # Use expected_rr from model for TP sizing
        use_ev_gate: bool = True,  # Use EV (expected_rr + costs) gating for entries
        ev_margin_r: float = 0.0,
        fee_percent: float = 0.0011,  # Round-trip fee as percent of price (0.0011 = 0.11%)
        fee_per_trade_r: Optional[float] = None,  # Explicit fee in R units (overrides fee_percent)
        use_expected_rr: bool = True,  # Use expected_rr in EV gating when available
        ops_cost_enabled: bool = True,
        ops_cost_target_trades_per_day: float = 30.0,
        ops_cost_c1: float = 0.01,
        ops_cost_alpha: float = 1.7,
        single_position: bool = True,
        opposite_signal_policy: str = "flip",
        use_trend_gate: Optional[bool] = None,  # Gate entries by trend classifier probabilities
        use_regime_gate: Optional[bool] = None,  # Gate entries by regime classifier
        min_regime_prob: Optional[float] = None,
        allow_regime_ranging: Optional[bool] = None,
        allow_regime_trend_up: Optional[bool] = None,
        allow_regime_trend_down: Optional[bool] = None,
        allow_regime_volatile: Optional[bool] = None,
        regime_align_direction: Optional[bool] = None,
        use_ema_touch_entry: bool = True,  # Use touch-based EMA entry detection
        touch_threshold_atr: float = 0.3,  # How close to EMA counts as touch
        ema_touch_mode: str = "multi",  # "base" or "multi" (uses ema_touch_detected)
        use_intrabar_exits: bool = True,  # Use intrabar TP/SL detection when available
        raw_trades: Optional[pd.DataFrame] = None,  # Raw trades for precise TP/SL
        use_calibration: bool = False,  # Use Isotonic Regression calibrated probabilities
        max_holding_bars: Optional[int] = None,  # Max bars to hold a position (timeout)
    ):
        self.models = models
        self.config = config
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.stop_loss_atr = stop_loss_atr
        self.stop_padding_pct = stop_padding_pct
        self.take_profit_rr = take_profit_rr
        self.min_quality = min_quality
        if min_trend_prob is None:
            min_trend_prob = getattr(config.labels, "min_trend_prob", 0.0)
        self.min_trend_prob = float(min_trend_prob)
        self.min_bounce_prob = min_bounce_prob
        self.max_bounce_prob = max_bounce_prob
        self.cooldown_bars_after_stop = max(0, int(cooldown_bars_after_stop))
        self.trade_side = (trade_side or "long").strip().lower()
        self.use_raw_probabilities = bool(use_raw_probabilities)
        self.use_dynamic_rr = use_dynamic_rr
        self.use_ev_gate = bool(use_ev_gate)
        self.ev_margin_r = float(ev_margin_r)
        self.fee_percent = max(0.0, float(fee_percent))
        if fee_per_trade_r is None or not np.isfinite(float(fee_per_trade_r)):
            self.fee_per_trade_r = None
        else:
            self.fee_per_trade_r = max(0.0, float(fee_per_trade_r))
        self.use_expected_rr = bool(use_expected_rr)
        self.ops_cost_enabled = bool(ops_cost_enabled)
        self.ops_cost_target_trades_per_day = float(ops_cost_target_trades_per_day)
        self.ops_cost_c1 = float(ops_cost_c1)
        self.ops_cost_alpha = float(ops_cost_alpha)
        self.single_position = bool(single_position)
        self.opposite_signal_policy = str(opposite_signal_policy or "ignore").strip().lower()
        if (
            not np.isfinite(self.ops_cost_target_trades_per_day)
            or self.ops_cost_target_trades_per_day <= 0
        ):
            self.ops_cost_enabled = False
            self.ops_cost_target_trades_per_day = 0.0
        if not np.isfinite(self.ops_cost_c1) or self.ops_cost_c1 < 0:
            self.ops_cost_c1 = 0.0
        if not np.isfinite(self.ops_cost_alpha) or self.ops_cost_alpha <= 0:
            self.ops_cost_alpha = 1.0
        if self.opposite_signal_policy not in {"ignore", "close", "flip"}:
            self.opposite_signal_policy = "ignore"
        if use_trend_gate is None:
            use_trend_gate = getattr(config.labels, "use_trend_gate", True)
        if use_regime_gate is None:
            use_regime_gate = getattr(config.labels, "use_regime_gate", True)
        self.use_trend_gate = bool(use_trend_gate)
        self.use_regime_gate = bool(use_regime_gate)
        if min_regime_prob is None:
            min_regime_prob = getattr(config.labels, "min_regime_prob", 0.0)
        self.min_regime_prob = float(min_regime_prob)
        if allow_regime_ranging is None:
            allow_regime_ranging = getattr(config.labels, "allow_regime_ranging", True)
        if allow_regime_trend_up is None:
            allow_regime_trend_up = getattr(config.labels, "allow_regime_trend_up", True)
        if allow_regime_trend_down is None:
            allow_regime_trend_down = getattr(config.labels, "allow_regime_trend_down", True)
        if allow_regime_volatile is None:
            allow_regime_volatile = getattr(config.labels, "allow_regime_volatile", True)
        if regime_align_direction is None:
            regime_align_direction = getattr(config.labels, "regime_align_direction", True)
        self.allow_regime_ranging = bool(allow_regime_ranging)
        self.allow_regime_trend_up = bool(allow_regime_trend_up)
        self.allow_regime_trend_down = bool(allow_regime_trend_down)
        self.allow_regime_volatile = bool(allow_regime_volatile)
        self.regime_align_direction = bool(regime_align_direction)
        self.use_ema_touch_entry = use_ema_touch_entry
        self.touch_threshold_atr = touch_threshold_atr
        self.ema_touch_mode = (ema_touch_mode or "multi").strip().lower()
        if self.ema_touch_mode not in {"base", "multi"}:
            self.ema_touch_mode = "multi"
        
        # If using base touch mode, sync the threshold with the config if it's available
        # This allows the tuned pullback_threshold to control the backtest entry sensitivity
        if self.ema_touch_mode == "base" and hasattr(config, "labels") and hasattr(config.labels, "pullback_threshold"):
             self.touch_threshold_atr = float(config.labels.pullback_threshold)

        self.use_intrabar_exits = bool(use_intrabar_exits)
        self.raw_trades = raw_trades if self.use_intrabar_exits else None
        if max_holding_bars is None and hasattr(config, "labels") and hasattr(config.labels, "entry_forward_window"):
            max_holding_bars = int(getattr(config.labels, "entry_forward_window", 0))
        self.max_holding_bars = int(max_holding_bars or 0)
        self.use_calibration = bool(use_calibration) and not self.use_raw_probabilities
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
        self.base_tf_seconds = 0
        if hasattr(config, "features") and hasattr(config.features, "timeframes"):
            try:
                self.base_tf_seconds = int(config.features.timeframes[config.base_timeframe_idx])
            except Exception:
                self.base_tf_seconds = 0
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
            'rejected_trend_gate': 0,
            'rejected_regime_gate': 0,
            'rejected_bounce_prob': 0,
            'rejected_max_bounce_prob': 0,
            'rejected_ev_gate': 0,
            'rejected_no_ema_touch': 0,
            'rejected_cooldown': 0,
            'rejected_trade_side': 0,
            'ema_touch_raw': 0,
            'ema_touch_passed': 0,
            'ema_touch_dir_mismatch': 0,
            'accepted_signals': 0,
        }

        self._bounce_probs_all: List[float] = []
        self._bounce_probs_touch: List[float] = []
        self._trend_pred: Optional[Dict[str, np.ndarray]] = None
        self._regime_pred: Optional[Dict[str, np.ndarray]] = None
        self._trend_prob_checked: List[float] = []
        self._regime_prob_checked: List[float] = []
        self._regime_id_checked: List[int] = []
        self._rust_trade_index = None

        # Build index for raw trades if provided (for precise intrabar TP/SL detection)
        self._trades_by_bar = None
        if self.use_intrabar_exits and self.raw_trades is not None:
            self._build_trades_index()
        elif self.use_intrabar_exits:
            try:
                import rust_pipeline_bridge as rust_bridge  # type: ignore
                if rust_bridge.is_available():
                    self._rust_trade_index = rust_bridge.build_trade_index(self.config)
            except Exception:
                self._rust_trade_index = None

        # Grade analysis tracking
        self._grade_analysis = {
            'A': {'signals': [], 'trades': []},
            'B': {'signals': [], 'trades': []},
            'C': {'signals': [], 'trades': []},
        }
        self._feature_audit: Dict[str, Any] = {}

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

    def _get_context_feature_cols(self) -> List[str]:
        expected = self._get_entry_feature_names() or []
        return [col for col in CONTEXT_FEATURE_NAMES if col in expected]

    def _get_trend_feature_names(self) -> Optional[List[str]]:
        trend_model = getattr(self.models, "trend_classifier", None)
        if trend_model is None:
            return None
        feature_names = getattr(trend_model, "feature_names", None)
        if feature_names:
            return list(feature_names)
        return None

    def _get_regime_feature_names(self) -> Optional[List[str]]:
        regime_model = getattr(self.models, "regime_classifier", None)
        if regime_model is None:
            return None
        feature_names = getattr(regime_model, "feature_names", None)
        if feature_names:
            return list(feature_names)
        return None

    def _build_feature_frame(self, data: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
        if not feature_names:
            return pd.DataFrame(index=data.index)
        X = data.reindex(columns=feature_names)
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0
        return X.fillna(0)

    def _validate_feature_columns(self, feature_cols: List[str]) -> List[str]:
        expected = self._get_entry_feature_names()
        if not expected:
            self._feature_audit = {
                'expected_count': 0,
                'data_count': len(feature_cols),
                'missing': [],
                'extra': [],
            }
            return feature_cols

        context_cols = set(CONTEXT_FEATURE_NAMES)
        missing = [col for col in expected if col not in feature_cols and col not in context_cols]
        extra = [col for col in feature_cols if col not in expected and col not in context_cols]
        self._feature_audit = {
            'expected_count': len(expected),
            'data_count': len(feature_cols),
            'missing': missing,
            'extra': extra,
        }
        if missing or extra:
            missing_str = ", ".join(missing) if missing else "none"
            extra_str = ", ".join(extra) if extra else "none"
            message = (
                "Feature mismatch between backtest data and entry model. "
                f"Missing: {len(missing)} [{missing_str}]. "
                f"Extra: {len(extra)} [{extra_str}]. "
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

    def _check_exit_with_rust(self, bar_time: int) -> Tuple[Optional[str], Optional[float]]:
        """Check exit using the Rust trade index."""
        if self._rust_trade_index is None:
            return None, None

        direction = int(self.position['direction'])
        stop = float(self.position['stop_loss'])
        target = float(self.position['take_profit'])

        try:
            exit_code, exit_price = self._rust_trade_index.check_exit(
                int(bar_time),
                direction,
                stop,
                target,
            )
        except Exception:
            return None, None

        if exit_code == 1:
            return 'stop_loss', float(exit_price)
        if exit_code == 2:
            return 'take_profit', float(exit_price)
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
        self._bounce_probs_all = []
        self._bounce_probs_touch = []
        self._trend_prob_checked = []
        self._regime_prob_checked = []
        self._regime_id_checked = []
        
        print(f"Running backtest on {len(data):,} bars...")
        
        if self.diag:
            self.diag.log_section("Backtest Execution")
            self.diag.log_metric("backtest_bars", len(data))
            self.diag.log_metric("backtest_initial_capital", self.initial_capital)
            self.diag.log_metric("backtest_min_quality", self.min_quality)
            self.diag.log_metric("backtest_min_trend_prob", self.min_trend_prob)
            self.diag.log_metric("backtest_use_trend_gate", float(self.use_trend_gate))
            self.diag.log_metric("backtest_use_regime_gate", float(self.use_regime_gate))
            self.diag.log_metric("backtest_min_regime_prob", self.min_regime_prob)
            self.diag.log_metric("backtest_regime_align_direction", float(self.regime_align_direction))
            self.diag.log_raw(
                "  regime_allowed: ranging={} trend_up={} trend_down={} volatile={}\n".format(
                    self.allow_regime_ranging,
                    self.allow_regime_trend_up,
                    self.allow_regime_trend_down,
                    self.allow_regime_volatile,
                )
            )
            self.diag.log_metric("backtest_min_bounce_prob", self.min_bounce_prob)
            self.diag.log_metric("backtest_use_ev_gate", self.use_ev_gate)
            self.diag.log_metric("backtest_ev_margin_r", self.ev_margin_r)
            self.diag.log_metric("backtest_fee_percent", self.fee_percent)
            if self.fee_per_trade_r is not None:
                self.diag.log_metric("backtest_fee_per_trade_r", self.fee_per_trade_r)
            self.diag.log_metric("backtest_use_expected_rr", self.use_expected_rr)
            self.diag.log_metric("backtest_ops_cost_enabled", float(self.ops_cost_enabled))
            self.diag.log_metric("backtest_ops_cost_target_trades_per_day", self.ops_cost_target_trades_per_day)
            self.diag.log_metric("backtest_ops_cost_c1", self.ops_cost_c1)
            self.diag.log_metric("backtest_ops_cost_alpha", self.ops_cost_alpha)
            self.diag.log_metric("backtest_stop_loss_atr", self.stop_loss_atr)
            self.diag.log_metric("backtest_stop_padding_pct", self.stop_padding_pct)
            self.diag.log_metric("backtest_take_profit_rr", self.take_profit_rr)
            self.diag.log_metric("backtest_cooldown_bars_after_stop", self.cooldown_bars_after_stop)
            self.diag.log_metric("backtest_use_calibration", self.use_calibration)
            self.diag.log_raw(f"\n  trade_side: {self.trade_side}\n")
            self.diag.log_raw(f"  use_calibration: {self.use_calibration}\n")
            
            # VERIFICATION: Confirm feature_cols doesn't contain label columns
            label_patterns = ['label', 'success', 'outcome', '_mfe', '_mae', '_rr']
            leaky_features = [col for col in feature_cols if any(p in col.lower() for p in label_patterns)]
            if 'regime' in feature_cols:
                leaky_features.append('regime')
            if leaky_features:
                self.diag.log_error(f"LEAKAGE DETECTED! Feature columns contain labels: {leaky_features}")
            else:
                self.diag.log_raw("\n  ✓ VERIFIED: No label columns in feature_cols\n")
            
            # Log which columns are being used
            self.diag.log_metric("feature_count", len(feature_cols))
            self.diag.log_raw(f"\n  First 10 feature columns: {feature_cols[:10]}\n")

        feature_cols = self._validate_feature_columns(feature_cols)

        self._trend_pred = None
        self._regime_pred = None
        context_cols = self._get_context_feature_cols()
        need_trend_context = any(col.startswith("trend_prob_") for col in context_cols)
        need_regime_context = any(col.startswith("regime_prob_") for col in context_cols)
        if self.use_trend_gate or need_trend_context:
            trend_model = getattr(self.models, "trend_classifier", None)
            if trend_model is None or getattr(trend_model, "model", None) is None:
                raise ValueError("Trend classifier not loaded; retrain models or disable trend context/gating.")
            trend_cols = self._get_trend_feature_names() or feature_cols
            X_trend = self._build_feature_frame(data, trend_cols)
            self._trend_pred = trend_model.predict(X_trend)

        if self.use_regime_gate or need_regime_context:
            regime_model = getattr(self.models, "regime_classifier", None)
            if regime_model is None or getattr(regime_model, "model", None) is None:
                raise ValueError("Regime classifier not loaded; retrain models or disable regime context/gating.")
            regime_cols = self._get_regime_feature_names() or feature_cols
            X_regime = self._build_feature_frame(data, regime_cols)
            self._regime_pred = regime_model.predict(X_regime)

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

                if self.position is not None and self.max_holding_bars > 0:
                    entry_bar_idx = int(self.position.get('entry_bar_idx', i))
                    if (i - entry_bar_idx) >= self.max_holding_bars:
                        self._close_position(row.get('close', current_price), row.get('datetime', datetime.now()), 'timeout', i)
            
            if self.position is None:
                if self.cooldown_bars_after_stop > 0 and i < self.next_entry_bar:
                    self._signal_stats['rejected_cooldown'] += 1
                else:
                    # Do not open on the final bar (no future bar to realize stop/TP).
                    if i < (n - 1):
                        if self.max_holding_bars > 0 and (i + self.max_holding_bars) >= n:
                            pass
                        else:
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

        context_cols = self._get_context_feature_cols()
        if "trend_prob_up" in context_cols:
            prob_up = 0.0
            prob_down = 0.0
            prob_neutral = 0.0
            if self._trend_pred is not None:
                try:
                    prob_up = float(self._trend_pred.get("prob_up", [0.0])[bar_idx])
                    prob_down = float(self._trend_pred.get("prob_down", [0.0])[bar_idx])
                    prob_neutral = float(self._trend_pred.get("prob_neutral", [0.0])[bar_idx])
                except Exception:
                    prob_up = 0.0
                    prob_down = 0.0
                    prob_neutral = 0.0
            feature_data["trend_prob_up"] = [prob_up]
            feature_data["trend_prob_down"] = [prob_down]
            feature_data["trend_prob_neutral"] = [prob_neutral]

        if "regime_prob_ranging" in context_cols:
            prob_ranging = 0.0
            prob_trend_up = 0.0
            prob_trend_down = 0.0
            prob_volatile = 0.0
            if self._regime_pred is not None:
                try:
                    prob_ranging = float(self._regime_pred.get("prob_ranging", [0.0])[bar_idx])
                    prob_trend_up = float(self._regime_pred.get("prob_trend_up", [0.0])[bar_idx])
                    prob_trend_down = float(self._regime_pred.get("prob_trend_down", [0.0])[bar_idx])
                    prob_volatile = float(self._regime_pred.get("prob_volatile", [0.0])[bar_idx])
                except Exception:
                    prob_ranging = 0.0
                    prob_trend_up = 0.0
                    prob_trend_down = 0.0
                    prob_volatile = 0.0
            feature_data["regime_prob_ranging"] = [prob_ranging]
            feature_data["regime_prob_trend_up"] = [prob_trend_up]
            feature_data["regime_prob_trend_down"] = [prob_trend_down]
            feature_data["regime_prob_volatile"] = [prob_volatile]

        X = pd.DataFrame(feature_data, index=features_df.index)

        entry_pred = self.models.entry_model.predict(X, use_calibration=self.use_calibration)

        bounce_key = "bounce_prob_raw" if self.use_raw_probabilities else "bounce_prob"
        bounce_arr = entry_pred.get(bounce_key)
        if bounce_arr is None:
            bounce_arr = entry_pred.get("bounce_prob", entry_pred.get("bounce_prob_raw", [0.0]))
        bounce_prob = float(bounce_arr[0])
        self._bounce_probs_all.append(float(bounce_prob))

        # Get expected RR from model if available
        expected_rr = None
        expected_rr_mean = None
        if 'expected_rr' in entry_pred:
            expected_rr = entry_pred['expected_rr'][0]
        if 'expected_rr_mean' in entry_pred:
            expected_rr_mean = entry_pred['expected_rr_mean'][0]

        slope_col = f'{self.base_tf}_ema_{self.config.labels.pullback_ema}_slope_norm'
        alignment = row.get(slope_col, 0)
        slope_dir = 0
        if alignment is not None and not pd.isna(alignment):
            try:
                alignment = float(alignment)
                if alignment > 0:
                    slope_dir = 1
                elif alignment < 0:
                    slope_dir = -1
            except Exception:
                slope_dir = 0

        touch_dir = 0
        if "ema_touch_direction" in row.index:
            touch_val = row.get("ema_touch_direction", 0)
            if touch_val is None or pd.isna(touch_val):
                touch_dir = 0
            else:
                try:
                    touch_dir = int(touch_val)
                except Exception:
                    touch_dir = 0
            if touch_dir not in (-1, 1):
                touch_dir = 0

        # Align trade direction with EMA touch direction when available (matches labeling)
        trend_dir = touch_dir if touch_dir in (-1, 1) else slope_dir
        if trend_dir == 0:
            self._signal_stats['trend_neutral_signals'] += 1
            return
        if trend_dir == 1:
            self._signal_stats['trend_up_signals'] += 1
            if not self.allow_long:
                self._signal_stats['rejected_trade_side'] += 1
                return
        else:
            self._signal_stats['trend_down_signals'] += 1
            if not self.allow_short:
                self._signal_stats['rejected_trade_side'] += 1
                return
        prob_up = 1.0 if trend_dir == 1 else 0.0
        prob_down = 1.0 if trend_dir == -1 else 0.0
        if self._trend_pred is not None:
            try:
                prob_up = float(self._trend_pred.get('prob_up', [prob_up])[bar_idx])
                prob_down = float(self._trend_pred.get('prob_down', [prob_down])[bar_idx])
            except Exception:
                pass

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
                        if ema_touched:
                            self._signal_stats['ema_touch_raw'] += 1

                    # touch_dir already computed above
                    if ema_touched and touch_dir not in (0, trend_dir):
                        self._signal_stats['ema_touch_dir_mismatch'] += 1
                        ema_touched = False

                    touch_dist = row.get("ema_touch_dist", None)
                    if touch_dist is not None and not pd.isna(touch_dist):
                        dist_from_ema = abs(float(touch_dist))
                    if ema_touched:
                        self._signal_stats['ema_touch_passed'] += 1
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
                    if ema_touched:
                        self._signal_stats['ema_touch_raw'] += 1
                        self._signal_stats['ema_touch_passed'] += 1
            else:
                # CLOSE-BASED (legacy): use close price distance
                dist_from_ema = abs(price - ema) / atr
                ema_touched = dist_from_ema <= self.config.labels.pullback_threshold
                if ema_touched:
                    self._signal_stats['ema_touch_raw'] += 1
                    self._signal_stats['ema_touch_passed'] += 1

        is_pullback = ema_touched
        if ema_touched:
            self._bounce_probs_touch.append(float(bounce_prob))
        trend_aligned = True
        if slope_dir != 0:
            trend_aligned = (trend_dir == slope_dir)

        # Reject if EMA touch is required but not detected
        if self.use_ema_touch_entry and not ema_touched:
            self._signal_stats['rejected_no_ema_touch'] += 1
            return

        trend_prob = prob_up if trend_dir == 1 else prob_down
        self._trend_prob_checked.append(float(trend_prob))
        if self.use_trend_gate and trend_prob < self.min_trend_prob:
            self._signal_stats['rejected_trend_gate'] += 1
            return

        if self.use_regime_gate and self._regime_pred is not None:
            try:
                regime_id = int(self._regime_pred.get('regime', [0])[bar_idx])
                prob_ranging = float(self._regime_pred.get('prob_ranging', [0.0])[bar_idx])
                prob_trend_up = float(self._regime_pred.get('prob_trend_up', [0.0])[bar_idx])
                prob_trend_down = float(self._regime_pred.get('prob_trend_down', [0.0])[bar_idx])
                prob_volatile = float(self._regime_pred.get('prob_volatile', [0.0])[bar_idx])
            except Exception:
                regime_id = 0
                prob_ranging = 0.0
                prob_trend_up = 0.0
                prob_trend_down = 0.0
                prob_volatile = 0.0

            if regime_id == 0:
                regime_prob = prob_ranging
                allowed = self.allow_regime_ranging
            elif regime_id == 1:
                regime_prob = prob_trend_up
                allowed = self.allow_regime_trend_up
                if self.regime_align_direction and trend_dir != 1:
                    allowed = False
            elif regime_id == 2:
                regime_prob = prob_trend_down
                allowed = self.allow_regime_trend_down
                if self.regime_align_direction and trend_dir != -1:
                    allowed = False
            else:
                regime_prob = prob_volatile
                allowed = self.allow_regime_volatile

            self._regime_id_checked.append(int(regime_id))
            self._regime_prob_checked.append(float(regime_prob))
            if (not allowed) or (regime_prob < self.min_regime_prob):
                self._signal_stats['rejected_regime_gate'] += 1
                return

        stop_dist = (self.stop_loss_atr * atr) + (self.stop_padding_pct * price)
        fee_r = 0.0
        if self.fee_per_trade_r is not None:
            fee_r = float(self.fee_per_trade_r)
        elif np.isfinite(stop_dist) and stop_dist > 0 and price > 0:
            fee_r = (self.fee_percent * price) / stop_dist
        fee_r = max(0.0, float(fee_r))

        ops_cost_r = 0.0
        if (
            self.ops_cost_enabled
            and self.ops_cost_target_trades_per_day > 0
            and self.base_tf_seconds > 0
        ):
            elapsed_days = ((bar_idx + 1) * self.base_tf_seconds) / 86400.0
            if elapsed_days > 0:
                trade_rate_day = float(self._signal_stats.get('accepted_signals', 0)) / elapsed_days
                if trade_rate_day > self.ops_cost_target_trades_per_day:
                    excess = trade_rate_day - self.ops_cost_target_trades_per_day
                    ops_cost_r = self.ops_cost_c1 * (
                        (excess / self.ops_cost_target_trades_per_day) ** self.ops_cost_alpha
                    )

        rr_mean = float(self.take_profit_rr)
        rr_cons = rr_mean
        if self.use_expected_rr:
            if expected_rr_mean is not None:
                rr_mean = float(expected_rr_mean)
            if expected_rr is not None:
                rr_cons = float(expected_rr)

        ev_value = 0.0
        implied_threshold = 0.0
        if np.isfinite(stop_dist) and stop_dist > 0:
            ev_components = self.models.entry_model.compute_expected_rr_components(
                np.asarray([bounce_prob], dtype=float),
                np.asarray([rr_mean], dtype=float),
                rr_conservative=np.asarray([rr_cons], dtype=float),
                cost_r=np.asarray([fee_r + ops_cost_r], dtype=float),
            )
            ev_value = float(ev_components['ev_conservative_r'][0])
            implied_threshold = float(ev_components['implied_threshold'][0])

        # EV gating (matches tuner): use expected_rr + costs instead of a raw probability threshold
        if self.use_ev_gate:
            if ev_value <= self.ev_margin_r:
                self._signal_stats['rejected_ev_gate'] += 1
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
        if not self.use_ev_gate:
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
                'entry_bar_idx': bar_idx,
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
                'expected_rr_mean': expected_rr_mean,
                'ev_value': ev_value,
                'implied_threshold': implied_threshold,
                'fee_r': fee_r,
                'stop_dist': stop_dist,
            }
    
    def _check_exit(self, row, atr, bar_idx):
        """
        Check if we should exit current position using this bar's OHLC.

        If raw trade data is available, uses precise intrabar detection.
        Otherwise, uses open price to infer which level was hit first.
        """
        # Try precise detection with Rust trade index first
        if self.use_intrabar_exits and self._rust_trade_index is not None and 'bar_time' in row.index:
            bar_time = int(row['bar_time'])
            exit_reason, exit_price = self._check_exit_with_rust(bar_time)
            if exit_reason is not None:
                self._close_position(exit_price, row.get('datetime', datetime.now()), exit_reason, bar_idx)
                return

        # Try precise detection with raw trades
        if self.use_intrabar_exits and self._trades_by_bar is not None and 'bar_time' in row.index:
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
            expected_rr=self.position.get('expected_rr', 0.0) or 0.0,
            expected_rr_mean=self.position.get('expected_rr_mean', 0.0) or 0.0,
            ev_value=self.position.get('ev_value', 0.0) or 0.0,
            implied_threshold=self.position.get('implied_threshold', 0.0) or 0.0,
            fee_r=self.position.get('fee_r', 0.0) or 0.0,
            stop_dist=self.position.get('stop_dist', 0.0) or 0.0,
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
        if trade.stop_dist and trade.stop_dist > 0:
            trade.realized_r = (direction * (exit_price - entry_price)) / trade.stop_dist
        else:
            trade.realized_r = 0.0
        trade.realized_r_net = trade.realized_r - trade.fee_r
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
        result.signal_stats = dict(self._signal_stats)

        def _summarize(values: List[float]) -> Dict[str, float]:
            if not values:
                return {
                    'count': 0.0,
                    'min': 0.0,
                    'mean': 0.0,
                    'p25': 0.0,
                    'p50': 0.0,
                    'p75': 0.0,
                    'p90': 0.0,
                    'max': 0.0,
                    'pct_above_threshold': 0.0,
                }
            arr = np.asarray(values, dtype=float)
            pct_above = float((arr >= float(self.min_bounce_prob)).mean())
            return {
                'count': float(arr.size),
                'min': float(arr.min()),
                'mean': float(arr.mean()),
                'p25': float(np.percentile(arr, 25)),
                'p50': float(np.percentile(arr, 50)),
                'p75': float(np.percentile(arr, 75)),
                'p90': float(np.percentile(arr, 90)),
                'max': float(arr.max()),
                'pct_above_threshold': pct_above,
            }

        result.bounce_prob_stats = {
            'min_bounce_prob': float(self.min_bounce_prob),
            'all': _summarize(self._bounce_probs_all),
            'ema_touch': _summarize(self._bounce_probs_touch),
        }
        result.diagnostics = self._build_diagnostics()
        
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
                self.diag.log_raw(f"    Rejected (trend gate):   {self._signal_stats['rejected_trend_gate']}\n")
                self.diag.log_raw(f"    Rejected (regime gate):  {self._signal_stats['rejected_regime_gate']}\n")
                self.diag.log_raw(f"    Rejected (bounce prob):  {self._signal_stats['rejected_bounce_prob']}\n")
                self.diag.log_raw(f"    Rejected (EV gate):      {self._signal_stats['rejected_ev_gate']}\n")
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

    def _build_diagnostics(self) -> Dict[str, Any]:
        trades = self.trades or []
        diagnostics: Dict[str, Any] = {
            'trade_count': len(trades),
        }
        if self._feature_audit:
            diagnostics['feature_audit'] = dict(self._feature_audit)
        trend_probs = np.asarray(self._trend_prob_checked, dtype=float)
        if trend_probs.size > 0:
            diagnostics['trend_prob_checked_count'] = float(trend_probs.size)
            diagnostics['trend_prob_checked_mean'] = float(trend_probs.mean())
            diagnostics['trend_prob_checked_p10'] = float(np.percentile(trend_probs, 10))
            diagnostics['trend_prob_checked_p50'] = float(np.percentile(trend_probs, 50))
            diagnostics['trend_prob_checked_p90'] = float(np.percentile(trend_probs, 90))

        regime_probs = np.asarray(self._regime_prob_checked, dtype=float)
        if regime_probs.size > 0:
            diagnostics['regime_prob_checked_count'] = float(regime_probs.size)
            diagnostics['regime_prob_checked_mean'] = float(regime_probs.mean())
            diagnostics['regime_prob_checked_p10'] = float(np.percentile(regime_probs, 10))
            diagnostics['regime_prob_checked_p50'] = float(np.percentile(regime_probs, 50))
            diagnostics['regime_prob_checked_p90'] = float(np.percentile(regime_probs, 90))
        if self._regime_id_checked:
            regime_counts: Dict[int, int] = {}
            for rid in self._regime_id_checked:
                regime_counts[rid] = regime_counts.get(rid, 0) + 1
            diagnostics['regime_id_counts'] = regime_counts
        if not trades:
            return diagnostics

        bounce_probs = np.asarray([t.bounce_prob for t in trades], dtype=float)
        ev_values = np.asarray([t.ev_value for t in trades], dtype=float)
        implied = np.asarray([t.implied_threshold for t in trades], dtype=float)
        expected_rr_mean = np.asarray([t.expected_rr_mean for t in trades], dtype=float)
        expected_rr = np.asarray([t.expected_rr for t in trades], dtype=float)
        realized_r = np.asarray([t.realized_r for t in trades], dtype=float)
        realized_r_net = np.asarray([t.realized_r_net for t in trades], dtype=float)
        fee_r = np.asarray([t.fee_r for t in trades], dtype=float)
        wins = (realized_r > 0).astype(float)

        probs_clip = np.clip(bounce_probs, 1e-6, 1 - 1e-6)
        diagnostics['brier'] = float(np.mean((probs_clip - wins) ** 2))
        diagnostics['logloss'] = float(-np.mean(
            wins * np.log(probs_clip) + (1.0 - wins) * np.log(1.0 - probs_clip)
        ))
        if len(np.unique(wins)) > 1:
            try:
                diagnostics['ece'] = float(compute_expected_calibration_error(wins, probs_clip).get('ece', 0.0))
            except Exception:
                diagnostics['ece'] = 0.0
        else:
            diagnostics['ece'] = 0.0

        if wins.sum() > 0:
            realized_win = realized_r[wins == 1]
            pred_win = expected_rr_mean[wins == 1]
            if np.mean(realized_win) > 0:
                diagnostics['expected_rr_bias_ratio'] = float(np.mean(pred_win) / np.mean(realized_win))
            else:
                diagnostics['expected_rr_bias_ratio'] = 0.0
            diagnostics['expected_rr_mae'] = float(np.mean(np.abs(pred_win - realized_win)))
        else:
            diagnostics['expected_rr_bias_ratio'] = 0.0
            diagnostics['expected_rr_mae'] = 0.0

        diagnostics['ev_mean'] = float(ev_values.mean()) if ev_values.size else 0.0
        diagnostics['ev_p10'] = float(np.percentile(ev_values, 10)) if ev_values.size else 0.0
        diagnostics['threshold_gap_mean'] = float((bounce_probs - implied).mean()) if implied.size else 0.0
        diagnostics['threshold_gap_p10'] = float(np.percentile((bounce_probs - implied), 10)) if implied.size else 0.0
        diagnostics['fee_r_mean'] = float(fee_r.mean()) if fee_r.size else 0.0
        diagnostics['realized_r_mean'] = float(realized_r.mean()) if realized_r.size else 0.0
        diagnostics['realized_r_net_mean'] = float(realized_r_net.mean()) if realized_r_net.size else 0.0
        diagnostics['expected_rr_mean'] = float(expected_rr_mean.mean()) if expected_rr_mean.size else 0.0
        diagnostics['expected_rr_cons_mean'] = float(expected_rr.mean()) if expected_rr.size else 0.0

        if ev_values.size > 0 and realized_r_net.size == ev_values.size:
            n = ev_values.size
            n_bins = int(min(10, max(1, n)))
            bins_summary = ""
            if n_bins == 1:
                bins_summary = f"0:{realized_r_net.mean():.4f}@{n}"
            else:
                edges = np.quantile(ev_values, np.linspace(0.0, 1.0, n_bins + 1))
                if np.unique(edges).size == 1:
                    bins_summary = f"0:{realized_r_net.mean():.4f}@{n}"
                else:
                    buckets = np.digitize(ev_values, edges[1:-1], right=True)
                    parts = []
                    for b in range(n_bins):
                        mask = buckets == b
                        if mask.any():
                            parts.append(f"{b}:{realized_r_net[mask].mean():.4f}@{int(mask.sum())}")
                    bins_summary = "|".join(parts)
            diagnostics['ev_bin_summary'] = bins_summary

        return diagnostics


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
    if result.trades:
        exit_counts = {}
        for trade in result.trades:
            exit_counts[trade.exit_reason] = exit_counts.get(trade.exit_reason, 0) + 1
        parts = [f"{reason}={count}" for reason, count in sorted(exit_counts.items(), key=lambda kv: (-kv[1], kv[0]))]
        print(f"\n  Exit Reasons:     {', '.join(parts)}")

    if result.signal_stats:
        stats = result.signal_stats
        checked = float(stats.get('signals_checked', 0))
        total_bars = float(stats.get('total_bars', 0))
        accepted = stats.get('accepted_signals', 0)
        print("\n  Signal Gate Diagnostics:")
        print(
            "    Bars Checked:   {} / {} ({:.1f}%)".format(
                int(checked),
                int(total_bars),
                (checked / total_bars * 100.0) if total_bars > 0 else 0.0,
            )
        )
        print(f"    Accepted:       {accepted}")
        print(
            "    EMA Touch:      raw={} passed={} dir_mismatch={}".format(
                stats.get('ema_touch_raw', 0),
                stats.get('ema_touch_passed', 0),
                stats.get('ema_touch_dir_mismatch', 0),
            )
        )
        print(
            "    Rejected:       no_ema_touch={} bounce_prob={} max_bounce_prob={} ev_gate={} trade_side={}".format(
                stats.get('rejected_no_ema_touch', 0),
                stats.get('rejected_bounce_prob', 0),
                stats.get('rejected_max_bounce_prob', 0),
                stats.get('rejected_ev_gate', 0),
                stats.get('rejected_trade_side', 0),
            )
        )
        if checked > 0:
            print(
                "    Trend Signals:  up={} down={} neutral={}".format(
                    stats.get('trend_up_signals', 0),
                    stats.get('trend_down_signals', 0),
                    stats.get('trend_neutral_signals', 0),
                )
            )

    if result.diagnostics:
        diag = result.diagnostics
        if 'feature_audit' in diag:
            fa = diag.get('feature_audit', {})
            print("\n  Feature Audit:")
            print(
                "    Expected features: {}  Data features: {}".format(
                    fa.get('expected_count', 0),
                    fa.get('data_count', 0),
                )
            )
            missing = fa.get('missing') or []
            extra = fa.get('extra') or []
            if missing:
                print("    Missing features:")
                for name in missing:
                    print(f"      - {name}")
            if extra:
                print("    Extra features:")
                for name in extra:
                    print(f"      - {name}")
        if 'trend_prob_checked_count' in diag or 'regime_prob_checked_count' in diag:
            print("\n  Trend/Regime Prob Diagnostics (signals checked):")
            if 'trend_prob_checked_count' in diag:
                print(
                    "    Trend prob: count={:.0f} mean={:.3f} p10={:.3f} p50={:.3f} p90={:.3f}".format(
                        diag.get('trend_prob_checked_count', 0.0),
                        diag.get('trend_prob_checked_mean', 0.0),
                        diag.get('trend_prob_checked_p10', 0.0),
                        diag.get('trend_prob_checked_p50', 0.0),
                        diag.get('trend_prob_checked_p90', 0.0),
                    )
                )
            if 'regime_prob_checked_count' in diag:
                print(
                    "    Regime prob: count={:.0f} mean={:.3f} p10={:.3f} p50={:.3f} p90={:.3f}".format(
                        diag.get('regime_prob_checked_count', 0.0),
                        diag.get('regime_prob_checked_mean', 0.0),
                        diag.get('regime_prob_checked_p10', 0.0),
                        diag.get('regime_prob_checked_p50', 0.0),
                        diag.get('regime_prob_checked_p90', 0.0),
                    )
                )
            if 'regime_id_counts' in diag:
                print(f"    Regime id counts: {diag.get('regime_id_counts')}")

    if result.bounce_prob_stats:
        bp_stats = result.bounce_prob_stats
        min_thr = float(bp_stats.get('min_bounce_prob', 0.0))
        all_stats = bp_stats.get('all', {})
        touch_stats = bp_stats.get('ema_touch', {})

        def _print_bp(label: str, s: dict) -> None:
            print(
                "    {}: count={:.0f} mean={:.3f} p50={:.3f} p90={:.3f} max={:.3f} pct>=thr={:.1f}%".format(
                    label,
                    s.get('count', 0.0),
                    s.get('mean', 0.0),
                    s.get('p50', 0.0),
                    s.get('p90', 0.0),
                    s.get('max', 0.0),
                    s.get('pct_above_threshold', 0.0) * 100.0,
                )
            )

        print("\n  Bounce Prob Diagnostics:")
        print(f"    Threshold: {min_thr:.4f}")
        _print_bp("All Signals", all_stats)
        _print_bp("EMA Touch", touch_stats)

    if result.diagnostics:
        diag = result.diagnostics
        print("\n  EV Diagnostics:")
        print(f"    Trades:          {diag.get('trade_count', 0)}")
        print(
            "    Brier:           {brier:.4f}  LogLoss: {logloss:.4f}  ECE: {ece:.4f}".format(
                brier=diag.get('brier', 0.0),
                logloss=diag.get('logloss', 0.0),
                ece=diag.get('ece', 0.0),
            )
        )
        print(
            "    EV mean/p10:     {mean:.4f} / {p10:.4f}".format(
                mean=diag.get('ev_mean', 0.0),
                p10=diag.get('ev_p10', 0.0),
            )
        )
        print(
            "    Gap mean/p10:    {mean:.4f} / {p10:.4f}".format(
                mean=diag.get('threshold_gap_mean', 0.0),
                p10=diag.get('threshold_gap_p10', 0.0),
            )
        )
        print(
            "    RR mean/cons:    {mean:.4f} / {cons:.4f}".format(
                mean=diag.get('expected_rr_mean', 0.0),
                cons=diag.get('expected_rr_cons_mean', 0.0),
            )
        )
        print(
            "    Realized R:      {mean:.4f} (net {net:.4f}) fee_r {fee:.4f}".format(
                mean=diag.get('realized_r_mean', 0.0),
                net=diag.get('realized_r_net_mean', 0.0),
                fee=diag.get('fee_r_mean', 0.0),
            )
        )
        print(
            "    RR bias/mae:     {ratio:.3f} / {mae:.3f}".format(
                ratio=diag.get('expected_rr_bias_ratio', 0.0),
                mae=diag.get('expected_rr_mae', 0.0),
            )
        )
        ev_bins = diag.get('ev_bin_summary', '')
        if ev_bins:
            print(f"    EV bins (Rnet):  {ev_bins}")

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
    def _to_datetime(value):
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (int, float)):
            ts = float(value)
            if ts > 1e11:
                ts /= 1000.0
            return datetime.fromtimestamp(ts)
        return None

    entry_bar_open_time = _to_datetime(trade.entry_time)
    exit_bar_open_time = _to_datetime(trade.exit_time)

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
        'expected_rr': _as_py(trade.expected_rr),
        'expected_rr_mean': _as_py(trade.expected_rr_mean),
        'ev_value': _as_py(trade.ev_value),
        'implied_threshold': _as_py(trade.implied_threshold),
        'fee_r': _as_py(trade.fee_r),
        'stop_dist': _as_py(trade.stop_dist),
        'realized_r': _as_py(trade.realized_r),
        'realized_r_net': _as_py(trade.realized_r_net),
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
        'signal_stats': result.signal_stats,
        'bounce_prob_stats': result.bounce_prob_stats,
        'diagnostics': result.diagnostics,
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
