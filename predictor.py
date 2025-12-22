"""
Real-time predictor for TrendFollower.
Use trained models to generate predictions on live data.

Supports two modes:
1. Full recalculation (legacy) - recalculates all features from trade buffer
2. Incremental mode - updates features bar-by-bar for ~100x speedup
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from config import TrendFollowerConfig, DEFAULT_CONFIG
from data_loader import aggregate_to_bars
from feature_engine import calculate_features_for_timeframe, get_feature_columns
from models import TrendFollowerModels

# Import incremental engine
try:
    from incremental_features import (
        IncrementalFeatureEngine,
        IncrementalBarAggregator,
        warm_up_from_bars
    )
    INCREMENTAL_AVAILABLE = True
except ImportError:
    INCREMENTAL_AVAILABLE = False


@dataclass
class TrendSignal:
    """Signal from the trend classifier"""
    timestamp: datetime
    direction: int  # -1, 0, 1
    prob_up: float
    prob_down: float
    prob_neutral: float
    confidence: float  # max probability
    regime: int
    regime_name: str


@dataclass
class EntrySignal:
    """Signal for a potential entry"""
    timestamp: datetime
    direction: int
    bounce_prob: float
    expected_rr: float
    is_pullback_zone: bool
    trend_aligned: bool
    signal_quality: str  # 'A', 'B', 'C'


class TrendFollowerPredictor:
    """
    Real-time predictor using trained models.

    Usage:
        predictor = TrendFollowerPredictor()
        predictor.load_models('./models')

        # Feed new trades
        predictor.add_trades(new_trades_df)

        # Get predictions
        trend_signal = predictor.get_trend_signal()
        entry_signal = predictor.get_entry_signal()

    Incremental Mode (faster):
        predictor = TrendFollowerPredictor(use_incremental=True)
        predictor.load_models('./models')

        # Initialize from historical bars
        predictor.warm_up_incremental(bars_dict)

        # Then add bars incrementally
        predictor.add_bar(tf_name, bar_dict)
    """

    REGIME_NAMES = {
        0: 'ranging',
        1: 'trending_up',
        2: 'trending_down',
        3: 'volatile'
    }

    def __init__(
        self,
        config: TrendFollowerConfig = DEFAULT_CONFIG,
        use_incremental: bool = False,
        use_calibration: bool = False,
    ):
        self.config = config
        self.models: Optional[TrendFollowerModels] = None
        self.use_calibration = bool(use_calibration)

        # Incremental mode settings
        self.use_incremental = use_incremental and INCREMENTAL_AVAILABLE
        self.incremental_engine: Optional[IncrementalFeatureEngine] = None
        self.bar_aggregators: Dict[str, IncrementalBarAggregator] = {}

        self.base_tf = config.features.timeframe_names[config.base_timeframe_idx]

        if self.use_incremental:
            touch_threshold = getattr(config.labels, 'touch_threshold_atr', 0.3)
            if touch_threshold is None:
                touch_threshold = 0.3
            min_slope_norm = getattr(config.labels, 'min_slope_norm', 0.03)
            if min_slope_norm is None:
                min_slope_norm = 0.03
            self.incremental_engine = IncrementalFeatureEngine(
                config.features,
                base_tf=self.base_tf,
                pullback_ema=getattr(config.labels, 'pullback_ema', None),
                touch_threshold_atr=float(touch_threshold),
                min_slope_norm=float(min_slope_norm),
            )
            # Create bar aggregators for each timeframe
            for tf_name, tf_seconds in zip(config.features.timeframe_names,
                                           config.features.timeframes):
                self.bar_aggregators[tf_name] = IncrementalBarAggregator(tf_seconds)

        # Trade buffer (for legacy mode)
        self.trades_buffer: List[pd.DataFrame] = []
        self.max_buffer_size = 100000000  # Max trades to keep in memory

        # Bar caches
        self.bars_cache: Dict[str, pd.DataFrame] = {}
        self.features_cache: Optional[pd.DataFrame] = None

        # Incremental features cache (dict for faster access)
        self.incremental_features: Dict[str, float] = {}

        # Last prediction
        self.last_trend_signal: Optional[TrendSignal] = None
        self.last_entry_signal: Optional[EntrySignal] = None

        self.feature_cols: List[str] = []
        self.trend_feature_cols: List[str] = []

        # Track last bar time per TF for incremental updates
        self.last_bar_time: Dict[str, Optional[int]] = {
            tf: None for tf in config.features.timeframe_names
        }
    
    def load_models(self, model_dir: Path):
        """Load trained models"""
        model_dir = Path(model_dir)
        
        self.models = TrendFollowerModels(self.config.model)
        self.models.load_all(model_dir)

        # Align entry features to the entry model (backtest uses this list).
        self.trend_feature_cols = list(self.models.trend_classifier.feature_names)
        self.feature_cols = self._get_entry_feature_names()
        if not self.feature_cols:
            self.feature_cols = list(self.trend_feature_cols)

        print(f"Loaded models from {model_dir}")
        print(f"Expected features: {len(self.feature_cols)}")

    def _get_entry_feature_names(self) -> List[str]:
        entry_model = getattr(self.models, "entry_model", None)
        if entry_model is None:
            return []
        filtered = getattr(entry_model, "filtered_feature_names", None)
        if filtered:
            return list(filtered)
        feature_names = getattr(entry_model, "feature_names", None)
        if feature_names:
            return list(feature_names)
        return []
    
    def add_trades(self, trades: pd.DataFrame):
        """
        Add new trades to the buffer and update predictions.

        Args:
            trades: DataFrame with trade data (same format as training)
        """
        self.trades_buffer.append(trades)

        # Trim buffer if too large
        total_trades = sum(len(df) for df in self.trades_buffer)
        while total_trades > self.max_buffer_size and len(self.trades_buffer) > 1:
            removed = self.trades_buffer.pop(0)
            total_trades -= len(removed)

        # Rebuild bars and features
        self._update_features()

    def _update_features(self):
        """Rebuild bars and features from trade buffer"""
        if not self.trades_buffer:
            return

        # Combine all trades
        all_trades = pd.concat(self.trades_buffer, ignore_index=True)
        all_trades = all_trades.sort_values(self.config.data.timestamp_col)

        # Preprocess
        from data_loader import preprocess_trades
        all_trades = preprocess_trades(all_trades, self.config.data)

        # Create bars for each timeframe
        for tf_seconds, tf_name in zip(self.config.features.timeframes,
                                        self.config.features.timeframe_names):
            bars = aggregate_to_bars(all_trades, tf_seconds, self.config.data)
            self.bars_cache[tf_name] = bars

        # Calculate features
        from feature_engine import calculate_multi_timeframe_features
        self.features_cache = calculate_multi_timeframe_features(
            self.bars_cache,
            self.base_tf,
            self.config.features
        )

    # =========================================================================
    # Incremental Mode Methods
    # =========================================================================

    def warm_up_incremental(self, bars_dict: Dict[str, pd.DataFrame]) -> None:
        """
        Initialize incremental engine from historical bars.

        This should be called once at startup with historical bar data
        to warm up all the rolling indicators.

        Args:
            bars_dict: Dictionary mapping timeframe names to bar DataFrames
        """
        if not self.use_incremental or self.incremental_engine is None:
            raise RuntimeError("Incremental mode not enabled")

        print("Warming up incremental feature engine...")

        # Get all timeframes sorted by seconds (smallest first)
        tf_seconds = dict(zip(self.config.features.timeframe_names,
                              self.config.features.timeframes))
        sorted_tfs = sorted(self.config.features.timeframe_names,
                            key=lambda x: tf_seconds[x])

        base_tf = self.base_tf
        non_base_tfs = [tf for tf in sorted_tfs if tf != base_tf]

        # Enqueue non-base TF bars first (prevents HTF alignment drift)
        for tf_name in non_base_tfs:
            if tf_name in bars_dict:
                tf_bars = bars_dict[tf_name].sort_values('bar_time')
                print(f"  Processing {len(tf_bars)} {tf_name} bars...")
                for _, row in tf_bars.iterrows():
                    bar = row.to_dict()
                    self.incremental_engine.update_timeframe(tf_name, bar)
                if len(tf_bars):
                    self.last_bar_time[tf_name] = int(tf_bars['bar_time'].iloc[-1])

        # Then process base TF bars to publish shift-aligned HTF features
        if base_tf in bars_dict:
            base_bars = bars_dict[base_tf].sort_values('bar_time')
            print(f"  Processing {len(base_bars)} {base_tf} bars...")
            for _, row in base_bars.iterrows():
                bar = row.to_dict()
                self.incremental_engine.update_base_tf_bar(bar)
            if len(base_bars):
                self.last_bar_time[base_tf] = int(base_bars['bar_time'].iloc[-1])

        # Store current features
        self.incremental_features = self.incremental_engine.get_all_features()
        print(f"  Warmed up with {len(self.incremental_features)} features")

    def add_bar(self, tf_name: str, bar: Dict) -> Dict[str, float]:
        """
        Add a completed bar and update features incrementally.

        This is the main method for incremental updates during live trading.
        Call this whenever a new bar completes for any timeframe.

        Args:
            tf_name: Timeframe name (e.g., '1m', '5m')
            bar: Bar data dictionary with keys: bar_time, open, high, low, close,
                 volume, buy_sell_imbalance, trade_intensity, avg_trade_size

        Returns:
            Updated feature dictionary
        """
        if not self.use_incremental or self.incremental_engine is None:
            raise RuntimeError("Incremental mode not enabled")

        # Check if this bar is newer than the last one
        bar_time = bar.get('bar_time')
        if bar_time is not None:
            try:
                bar_time = int(bar_time)
            except (TypeError, ValueError):
                pass
            if self.last_bar_time[tf_name] is not None and bar_time <= self.last_bar_time[tf_name]:
                # Already processed this bar
                return self.incremental_features
            self.last_bar_time[tf_name] = bar_time

        # Update the incremental engine
        if tf_name == self.base_tf:
            # Base TF - also updates partial HTF and cross-TF features
            self.incremental_features = self.incremental_engine.update_base_tf_bar(bar)
        else:
            # Higher TF - just update this TF's features
            self.incremental_engine.update_timeframe(tf_name, bar)
            self.incremental_features = self.incremental_engine.get_all_features()

        return self.incremental_features

    def add_bars_batch(self, bars_dict: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Add multiple bars across timeframes (e.g., from a 5-minute update).

        This efficiently processes new bars for all timeframes.

        Args:
            bars_dict: Dictionary mapping timeframe names to NEW bar DataFrames

        Returns:
            Updated feature dictionary
        """
        if not self.use_incremental or self.incremental_engine is None:
            raise RuntimeError("Incremental mode not enabled")

        # Process in order of timeframe size (smallest first)
        tf_seconds = dict(zip(self.config.features.timeframe_names,
                              self.config.features.timeframes))
        sorted_tfs = sorted(self.config.features.timeframe_names,
                            key=lambda x: tf_seconds[x])
        base_tf = self.base_tf
        non_base_tfs = [tf for tf in sorted_tfs if tf != base_tf]

        # Enqueue non-base TF bars first to avoid HTF alignment drift
        for tf_name in non_base_tfs:
            if tf_name in bars_dict:
                new_bars = bars_dict[tf_name].sort_values('bar_time')

                # Only process bars newer than the last one
                if self.last_bar_time[tf_name] is not None:
                    new_bars = new_bars[new_bars['bar_time'] > self.last_bar_time[tf_name]]

                for _, row in new_bars.iterrows():
                    bar = row.to_dict()
                    self.add_bar(tf_name, bar)

        # Process base TF last so it publishes shift-aligned HTF features
        if base_tf in bars_dict:
            new_bars = bars_dict[base_tf].sort_values('bar_time')

            if self.last_bar_time[base_tf] is not None:
                new_bars = new_bars[new_bars['bar_time'] > self.last_bar_time[base_tf]]

            for _, row in new_bars.iterrows():
                bar = row.to_dict()
                self.add_bar(base_tf, bar)

        return self.incremental_features

    def get_incremental_features_df(self) -> pd.DataFrame:
        """Get incremental features as a single-row DataFrame for prediction"""
        if not self.use_incremental:
            raise RuntimeError("Incremental mode not enabled")
        return pd.DataFrame([self.incremental_features])

    def _compute_touch_and_bounce(self, row: pd.Series) -> Dict[str, float]:
        """Compute multi-TF EMA touch and bounce features for a single bar."""
        touch_info = {
            'ema_touch_detected': False,
            'ema_touch_tf': None,
            'ema_touch_direction': 0,
            'ema_touch_quality': 0.0,
            'ema_touch_dist': np.nan,
            'ema_touch_slope': np.nan,
        }
        bounce_info = {
            'bounce_bar_body_ratio': np.nan,
            'bounce_bar_wick_ratio': np.nan,
            'bounce_bar_direction': np.nan,
            'bounce_volume_ratio': np.nan,
        }

        ema_period = getattr(self.config.labels, 'pullback_ema', None)
        if ema_period is None:
            return {**touch_info, **bounce_info}

        atr_col = f'{self.base_tf}_atr'
        atr_val = row.get(atr_col)
        if atr_val is None or pd.isna(atr_val) or atr_val <= 0:
            return {**touch_info, **bounce_info}

        bar_open = row.get('open')
        bar_high = row.get('high')
        bar_low = row.get('low')
        bar_close = row.get('close')
        bar_volume = row.get('volume')
        if any(val is None or pd.isna(val) for val in [bar_open, bar_high, bar_low, bar_close]):
            return {**touch_info, **bounce_info}

        mid_bar = (bar_high + bar_low) / 2.0
        threshold = getattr(self.config.labels, 'touch_threshold_atr', 0.3)
        if threshold is None:
            threshold = 0.3
        min_slope = getattr(self.config.labels, 'min_slope_norm', 0.03)
        if min_slope is None:
            min_slope = 0.03

        for tf_name in self.config.features.timeframe_names:
            ema_key = f'{tf_name}_ema_{ema_period}'
            slope_key = f'{tf_name}_ema_{ema_period}_slope_norm'
            if ema_key not in row or slope_key not in row:
                continue

            ema_val = row.get(ema_key)
            slope_val = row.get(slope_key)
            if ema_val is None or slope_val is None:
                continue
            if pd.isna(ema_val) or pd.isna(slope_val):
                continue

            if slope_val > min_slope:
                dist_low = (bar_low - ema_val) / atr_val
                if -threshold <= dist_low <= threshold:
                    if bar_close >= ema_val or mid_bar >= ema_val:
                        dist_score = 1.0 - min(abs(dist_low) / threshold, 1.0) if threshold > 0 else 0.0
                        wick_score = (bar_close - bar_low) / (bar_high - bar_low) if bar_high > bar_low else 0.0
                        quality = (dist_score + wick_score) / 2.0
                        touch_info.update({
                            'ema_touch_detected': True,
                            'ema_touch_tf': tf_name,
                            'ema_touch_direction': 1,
                            'ema_touch_quality': quality,
                            'ema_touch_dist': dist_low,
                            'ema_touch_slope': slope_val,
                        })
                        break
            elif slope_val < -min_slope:
                dist_high = (bar_high - ema_val) / atr_val
                if -threshold <= dist_high <= threshold:
                    if bar_close <= ema_val or mid_bar <= ema_val:
                        dist_score = 1.0 - min(abs(dist_high) / threshold, 1.0) if threshold > 0 else 0.0
                        wick_score = (bar_high - bar_close) / (bar_high - bar_low) if bar_high > bar_low else 0.0
                        quality = (dist_score + wick_score) / 2.0
                        touch_info.update({
                            'ema_touch_detected': True,
                            'ema_touch_tf': tf_name,
                            'ema_touch_direction': -1,
                            'ema_touch_quality': quality,
                            'ema_touch_dist': dist_high,
                            'ema_touch_slope': slope_val,
                        })
                        break

        touch_direction = int(touch_info.get('ema_touch_direction', 0) or 0)
        if touch_direction == 0:
            return {**touch_info, **bounce_info}

        bar_range = bar_high - bar_low
        if bar_range > 0:
            body_size = abs(bar_close - bar_open)
            bounce_info['bounce_bar_body_ratio'] = body_size / bar_range
            if touch_direction > 0:
                favorable_wick = min(bar_open, bar_close) - bar_low
            else:
                favorable_wick = bar_high - max(bar_open, bar_close)
            bounce_info['bounce_bar_wick_ratio'] = favorable_wick / bar_range

        bounce_info['bounce_bar_direction'] = 1 if bar_close > bar_open else -1

        volume_ma_col = f'{self.base_tf}_volume_sma'
        volume_ma = row.get(volume_ma_col)
        if volume_ma is not None and not pd.isna(volume_ma) and volume_ma > 0 and bar_volume is not None:
            bounce_info['bounce_volume_ratio'] = bar_volume / volume_ma

        return {**touch_info, **bounce_info}
    
    def get_trend_signal(self) -> Optional[TrendSignal]:
        """
        Get current trend prediction.

        Returns:
            TrendSignal with direction and probabilities
        """
        if self.models is None:
            raise ValueError("Models not loaded. Call load_models() first.")

        # Get feature DataFrame (incremental or legacy mode)
        if self.use_incremental:
            if not self.incremental_features:
                return None
            latest = pd.DataFrame([self.incremental_features])
        else:
            if self.features_cache is None or len(self.features_cache) == 0:
                return None
            latest = self.features_cache.iloc[[-1]].copy()
            touch_bounce = self._compute_touch_and_bounce(latest.iloc[0])
            for key, value in touch_bounce.items():
                latest.at[latest.index[0], key] = value

        # Prepare features efficiently (avoid fragmentation)
        feature_data = {}
        feature_cols = self.trend_feature_cols or self.feature_cols
        for col in feature_cols:
            if col in latest.columns:
                val = latest[col].iloc[0]
                feature_data[col] = [0 if pd.isna(val) else val]
            else:
                feature_data[col] = [0]

        X = pd.DataFrame(feature_data)

        # Predict trend
        trend_pred = self.models.trend_classifier.predict(X)

        # Predict regime
        regime_pred = self.models.regime_classifier.predict(X)

        # Create signal
        signal = TrendSignal(
            timestamp=datetime.now(),
            direction=int(trend_pred['prediction'][0]),
            prob_up=float(trend_pred['prob_up'][0]),
            prob_down=float(trend_pred['prob_down'][0]),
            prob_neutral=float(trend_pred['prob_neutral'][0]),
            confidence=float(max(trend_pred['prob_up'][0],
                                trend_pred['prob_down'][0],
                                trend_pred['prob_neutral'][0])),
            regime=int(regime_pred['regime'][0]),
            regime_name=self.REGIME_NAMES.get(regime_pred['regime'][0], 'unknown')
        )

        self.last_trend_signal = signal
        return signal
    
    def get_entry_signal(self) -> Optional[EntrySignal]:
        """
        Get entry quality prediction for current bar.

        Returns:
            EntrySignal with bounce probability and quality grade
        """
        if self.models is None:
            raise ValueError("Models not loaded. Call load_models() first.")

        # Get feature DataFrame (incremental or legacy mode)
        if self.use_incremental:
            if not self.incremental_features:
                return None
            latest = pd.DataFrame([self.incremental_features])
        else:
            if self.features_cache is None or len(self.features_cache) == 0:
                return None
            latest = self.features_cache.iloc[[-1]].copy()
            touch_bounce = self._compute_touch_and_bounce(latest.iloc[0])
            for key, value in touch_bounce.items():
                latest.at[latest.index[0], key] = value

        # Prepare features efficiently (avoid fragmentation)
        feature_data = {}
        for col in self.feature_cols:
            if col in latest.columns:
                val = latest[col].iloc[0]
                feature_data[col] = [0 if pd.isna(val) else val]
            else:
                feature_data[col] = [0]

        X = pd.DataFrame(feature_data)

        # Check if in pullback zone
        ema_col = f'{self.base_tf}_ema_{self.config.labels.pullback_ema}'
        atr_col = f'{self.base_tf}_atr'
        slope_col = f'{self.base_tf}_ema_{self.config.labels.pullback_ema}_slope_norm'

        is_pullback = False
        trend_dir = 0

        if all(col in latest.columns for col in [ema_col, atr_col]):
            # Get close price - from incremental features or original bars
            if self.use_incremental:
                # In incremental mode, we track the last close via the base_tf bars
                price = self.incremental_features.get(f'{self.base_tf}_close',
                         self.incremental_features.get('close', 0))
                # If no explicit close, estimate from EMA
                if price == 0:
                    price = latest[ema_col].iloc[0]
            else:
                price = latest['close'].iloc[0]

            ema = latest[ema_col].iloc[0]
            atr = latest[atr_col].iloc[0]

            slope_val = latest[slope_col].iloc[0] if slope_col in latest.columns else 0
            if pd.isna(slope_val):
                slope_val = 0

            if atr > 0 and not pd.isna(atr):
                dist = abs(price - ema) / atr if not pd.isna(ema) else 0
                is_pullback = dist <= self.config.labels.pullback_threshold
                # Long-only bias: any non-zero positive slope counts as trend_dir=1
                trend_dir = int(np.sign(slope_val)) if slope_val != 0 else 0

        # Predict entry quality
        entry_pred = self.models.entry_model.predict(X, use_calibration=self.use_calibration)

        bounce_prob = float(entry_pred['bounce_prob'][0])
        expected_rr = float(entry_pred.get('expected_rr', [1.0])[0])

        # Determine signal quality
        trend_aligned = (trend_dir != 0)

        if bounce_prob > 0.6 and trend_aligned and is_pullback:
            quality = 'A'
        elif bounce_prob > 0.5 and (trend_aligned or is_pullback):
            quality = 'B'
        else:
            quality = 'C'

        signal = EntrySignal(
            timestamp=datetime.now(),
            direction=int(trend_dir),
            bounce_prob=bounce_prob,
            expected_rr=expected_rr,
            is_pullback_zone=is_pullback,
            trend_aligned=trend_aligned,
            signal_quality=quality
        )

        self.last_entry_signal = signal
        return signal
    
    def get_full_prediction(self) -> Dict:
        """
        Get complete prediction summary.
        
        Returns:
            Dictionary with all predictions and context
        """
        trend = self.get_trend_signal()
        entry = self.get_entry_signal()
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'trend': None,
            'entry': None,
            'recommendation': 'WAIT'
        }
        
        if trend:
            result['trend'] = {
                'direction': trend.direction,
                'direction_name': {-1: 'DOWN', 0: 'NEUTRAL', 1: 'UP'}[trend.direction],
                'prob_up': trend.prob_up,
                'prob_down': trend.prob_down,
                'confidence': trend.confidence,
                'regime': trend.regime_name
            }
        
        if entry:
            result['entry'] = {
                'direction': entry.direction,
                'bounce_prob': entry.bounce_prob,
                'expected_rr': entry.expected_rr,
                'is_pullback': entry.is_pullback_zone,
                'trend_aligned': entry.trend_aligned,
                'quality': entry.signal_quality
            }
            
            # Generate recommendation
            if entry.signal_quality == 'A' and entry.direction != 0:
                result['recommendation'] = f"ENTER {'LONG' if entry.direction > 0 else 'SHORT'} (A-grade)"
            elif entry.signal_quality == 'B' and entry.direction != 0:
                result['recommendation'] = f"CONSIDER {'LONG' if entry.direction > 0 else 'SHORT'} (B-grade)"
            else:
                result['recommendation'] = 'WAIT'
        
        return result


def create_live_predictor(
    model_dir: Path,
    config: TrendFollowerConfig = DEFAULT_CONFIG,
    use_calibration: bool = False,
) -> TrendFollowerPredictor:
    """
    Create and initialize a live predictor.
    
    Args:
        model_dir: Path to saved models
        config: Configuration
        use_calibration: Apply probability calibration if available
        
    Returns:
        Initialized TrendFollowerPredictor
    """
    predictor = TrendFollowerPredictor(config, use_calibration=use_calibration)
    predictor.load_models(model_dir)
    return predictor


if __name__ == "__main__":
    print("Predictor module loaded successfully")
    
    # Example usage (requires trained models)
    # predictor = create_live_predictor(Path('./models'))
    # predictor.add_trades(new_trades_df)
    # prediction = predictor.get_full_prediction()
    # print(prediction)
