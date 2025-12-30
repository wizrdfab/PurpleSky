"""
Incremental feature calculation engine for TrendFollower.

This module provides efficient incremental updates for ALL features instead of
recalculating from scratch. It maintains rolling state for each indicator.

EXHAUSTIVE FEATURE LIST (per timeframe):
========================================
EMA-based (4 EMAs: 10, 21, 50, 100):
  - ema_{period}: Exponential Moving Average - INCREMENTAL via EMA formula
  - ema_{period}_slope: EMA difference - INCREMENTAL (diff of EMA)
  - price_vs_ema_{period}: (close - EMA) / ATR - INCREMENTAL (derived)
  - ema_{period}_slope_norm: slope / ATR - INCREMENTAL (derived)
  - ema_alignment: EMA stacking score - INCREMENTAL (derived from EMAs)

RSI (period=14):
  - rsi: Relative Strength Index - INCREMENTAL via Wilder smoothing
  - rsi_slope: RSI change over 3 bars - INCREMENTAL (rolling buffer)

ADX (period=14):
  - adx: Average Directional Index - INCREMENTAL via smoothed averages
  - plus_di: Positive DI - INCREMENTAL
  - minus_di: Negative DI - INCREMENTAL
  - di_diff: plus_di - minus_di - INCREMENTAL (derived)
  - adx_slope: ADX change over 3 bars - INCREMENTAL (rolling buffer)

Bollinger Bands (period=20, std=2):
  - bb_width: (upper - lower) / mid - INCREMENTAL via rolling stats
  - bb_position: (close - lower) / (upper - lower) - INCREMENTAL

ATR (period=14):
  - atr: Average True Range - INCREMENTAL via rolling mean
  - atr_percentile: ATR rank over 100 bars - INCREMENTAL via deque

Volume (period=20):
  - volume_sma: Simple MA of volume - INCREMENTAL via rolling sum
  - volume_ratio: volume / volume_sma - INCREMENTAL (derived)
  - obv: On-Balance Volume - INCREMENTAL via cumsum
  - obv_slope: OBV change over 5 bars - INCREMENTAL (rolling buffer)

Microstructure (derived from bar aggregation):
  - imbalance_ma: SMA of buy_sell_imbalance (10) - INCREMENTAL
  - imbalance_slope: imbalance diff over 3 bars - INCREMENTAL
  - intensity_ma: SMA of trade_intensity (20) - INCREMENTAL
  - intensity_ratio: intensity / intensity_ma - INCREMENTAL
  - size_ma: SMA of avg_trade_size (20) - INCREMENTAL
  - size_ratio: avg_trade_size / size_ma - INCREMENTAL

Structure (swing_lookback=10):
  - dist_from_high: Distance from recent swing high - INCREMENTAL (rolling buffer)
  - dist_from_low: Distance from recent swing low - INCREMENTAL (rolling buffer)
  - swing_high: Swing high detection - INCREMENTAL (rolling buffer)
  - swing_low: Swing low detection - INCREMENTAL (rolling buffer)

Price dynamics:
  - returns: pct_change - INCREMENTAL (just prev close)
  - returns_volatility: rolling std of returns (20) - INCREMENTAL
  - momentum_5: pct_change(5) - INCREMENTAL (rolling buffer)
  - momentum_10: pct_change(10) - INCREMENTAL (rolling buffer)
  - momentum_20: pct_change(20) - INCREMENTAL (rolling buffer)

Candle features:
  - body_size: |close - open| / ATR - INCREMENTAL (derived)
  - upper_wick: (high - max(open,close)) / ATR - INCREMENTAL (derived)
  - lower_wick: (min(open,close) - low) / ATR - INCREMENTAL (derived)
  - candle_direction: sign(close - open) - INCREMENTAL (derived)

Cross-timeframe features:
  - tf_trend_agreement: Mean of ema_alignment across TFs - INCREMENTAL (derived)
  - tf_trending_count: Count of ADX > 25 across TFs - INCREMENTAL (derived)
  - tf_avg_rsi: Mean of RSI across TFs - INCREMENTAL (derived)

Partial HTF features (for each HTF relative to base):
  - partial_{tf}_open: First open in rolling window - INCREMENTAL
  - partial_{tf}_high: Max high in rolling window - INCREMENTAL
  - partial_{tf}_low: Min low in rolling window - INCREMENTAL
  - partial_{tf}_close: Current close - INCREMENTAL
  - partial_{tf}_volume: Sum of volume in rolling window - INCREMENTAL

TOTAL FEATURES: ~263 (5 timeframes × ~48 features + 23 cross/partial features)
"""

import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from config import FeatureConfig


@dataclass
class RollingState:
    """State for a rolling window calculation"""
    buffer: deque = field(default_factory=lambda: deque(maxlen=100))
    sum_val: float = 0.0
    sum_sq: float = 0.0
    count: int = 0
    nan_count: int = 0

    def add(self, value: float):
        """Add a value to the rolling window"""
        if len(self.buffer) == self.buffer.maxlen:
            old_val = self.buffer.popleft()
            self.count -= 1
            if old_val is None or (isinstance(old_val, float) and np.isnan(old_val)):
                self.nan_count -= 1
            else:
                self.sum_val -= old_val
                self.sum_sq -= old_val * old_val
        self.buffer.append(value)
        self.count += 1
        if value is None or (isinstance(value, float) and np.isnan(value)):
            self.nan_count += 1
        else:
            self.sum_val += value
            self.sum_sq += value * value

    def mean(self, min_count: int = 1) -> float:
        non_nan = self.count - self.nan_count
        if non_nan < min_count:
            return np.nan
        return self.sum_val / non_nan

    def std(self, ddof: int = 0, min_count: Optional[int] = None) -> float:
        non_nan = self.count - self.nan_count
        if min_count is None:
            min_count = ddof + 1
        if non_nan < min_count:
            return np.nan
        denom = non_nan - ddof
        if denom <= 0:
            return np.nan
        variance = (self.sum_sq - (self.sum_val * self.sum_val) / non_nan) / denom
        return np.sqrt(max(0.0, variance))

    def max_val(self) -> float:
        if not self.buffer:
            return np.nan
        vals = [v for v in self.buffer if not (isinstance(v, float) and np.isnan(v))]
        return max(vals) if vals else np.nan

    def min_val(self) -> float:
        if not self.buffer:
            return np.nan
        vals = [v for v in self.buffer if not (isinstance(v, float) and np.isnan(v))]
        return min(vals) if vals else np.nan

    def first(self) -> float:
        return self.buffer[0] if self.buffer else np.nan

    def last(self) -> float:
        return self.buffer[-1] if self.buffer else np.nan


class IncrementalEMA:
    """Incrementally updated EMA"""
    def __init__(self, period: int):
        self.period = period
        self.alpha = 2.0 / (period + 1)
        self.value: Optional[float] = None
        self.prev_value: Optional[float] = None

    def update(self, price: float) -> float:
        """Update EMA with new price and return current value"""
        if self.value is None:
            self.value = price
            self.prev_value = None
            return self.value

        self.prev_value = self.value
        self.value = self.alpha * price + (1 - self.alpha) * self.value
        return self.value

    def slope(self) -> float:
        """Get slope (diff from previous value)"""
        if self.prev_value is None or self.value is None:
            return np.nan
        return self.value - self.prev_value


class IncrementalRSI:
    """Incrementally updated RSI using simple rolling means (matches feature_engine.py)"""
    def __init__(self, period: int = 14):
        self.period = period
        self.prev_close: Optional[float] = None
        self.gains = RollingState()
        self.gains.buffer = deque(maxlen=period)
        self.losses = RollingState()
        self.losses.buffer = deque(maxlen=period)
        # Buffer for slope calculation
        self.history = deque(maxlen=4)  # Need 4 values for diff(3)

    def update(self, close: float) -> float:
        """Update RSI with new close price"""
        if self.prev_close is None:
            delta = 0.0
        else:
            delta = close - self.prev_close

        gain = max(0.0, delta)
        loss = max(0.0, -delta)
        self.prev_close = close

        self.gains.add(gain)
        self.losses.add(loss)

        avg_gain = self.gains.mean(min_count=self.period)
        avg_loss = self.losses.mean(min_count=self.period)

        if np.isnan(avg_gain) or np.isnan(avg_loss) or avg_loss == 0:
            rsi = np.nan
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        self.history.append(rsi)
        return rsi

    def slope(self) -> float:
        """RSI change over 3 bars"""
        if len(self.history) >= 4:
            current = self.history[-1]
            prior = self.history[-4]
            if (isinstance(current, float) and np.isnan(current)) or (isinstance(prior, float) and np.isnan(prior)):
                return np.nan
            return current - prior
        return np.nan


class IncrementalATR:
    """Incrementally updated ATR"""
    def __init__(self, period: int = 14):
        self.period = period
        self.prev_close: Optional[float] = None
        self.value: Optional[float] = None
        self.tr_state = RollingState()
        self.tr_state.buffer = deque(maxlen=period)
        # For percentile calculation (ATR values)
        self.atr_buffer = deque(maxlen=100)
        self.atr_nan_count = 0

    def update(self, high: float, low: float, close: float) -> float:
        """Update ATR with new bar data"""
        # Calculate True Range
        if self.prev_close is None:
            tr = high - low
        else:
            tr1 = high - low
            tr2 = abs(high - self.prev_close)
            tr3 = abs(low - self.prev_close)
            tr = max(tr1, tr2, tr3)

        self.prev_close = close
        self.tr_state.add(tr)
        self.value = self.tr_state.mean(min_count=self.period)

        # Update percentile buffer
        if len(self.atr_buffer) == self.atr_buffer.maxlen:
            old_val = self.atr_buffer.popleft()
            if isinstance(old_val, float) and np.isnan(old_val):
                self.atr_nan_count -= 1
        self.atr_buffer.append(self.value)
        if isinstance(self.value, float) and np.isnan(self.value):
            self.atr_nan_count += 1

        return self.value

    def percentile(self) -> float:
        """Get ATR percentile rank over last 100 bars"""
        if len(self.atr_buffer) < self.atr_buffer.maxlen or self.atr_nan_count > 0:
            return np.nan
        current = self.atr_buffer[-1]
        if isinstance(current, float) and np.isnan(current):
            return np.nan
        vals = np.array(self.atr_buffer, dtype=float)
        less = np.sum(vals < current)
        equal = np.sum(vals == current)
        n = len(vals)
        rank = less + (equal + 1) / 2.0
        return rank / n


class IncrementalADX:
    """Incrementally updated ADX with +DI and -DI (matches feature_engine.py)"""
    def __init__(self, period: int = 14):
        self.period = period
        self.prev_high: Optional[float] = None
        self.prev_low: Optional[float] = None
        self.prev_close: Optional[float] = None
        self.tr_state = RollingState()
        self.tr_state.buffer = deque(maxlen=period)
        self.plus_dm_state = RollingState()
        self.plus_dm_state.buffer = deque(maxlen=period)
        self.minus_dm_state = RollingState()
        self.minus_dm_state.buffer = deque(maxlen=period)
        self.dx_state = RollingState()
        self.dx_state.buffer = deque(maxlen=period)

        # For slope
        self.history = deque(maxlen=4)

    def update(self, high: float, low: float, close: float) -> Dict[str, float]:
        """Update ADX and return all values"""
        # Calculate True Range
        if self.prev_close is None:
            tr = high - low
            plus_dm = 0.0
            minus_dm = 0.0
        else:
            tr1 = high - low
            tr2 = abs(high - self.prev_close)
            tr3 = abs(low - self.prev_close)
            tr = max(tr1, tr2, tr3)

            # Directional Movement
            up_move = high - self.prev_high if self.prev_high is not None else 0.0
            down_move = self.prev_low - low if self.prev_low is not None else 0.0

            if up_move > down_move and up_move > 0:
                plus_dm = up_move
            else:
                plus_dm = 0.0

            if down_move > up_move and down_move > 0:
                minus_dm = down_move
            else:
                minus_dm = 0.0

        self.prev_high = high
        self.prev_low = low
        self.prev_close = close
        self.tr_state.add(tr)
        self.plus_dm_state.add(plus_dm)
        self.minus_dm_state.add(minus_dm)

        if self.tr_state.count >= self.period:
            atr_smooth = self.tr_state.sum_val
            plus_dm_smooth = self.plus_dm_state.sum_val
            minus_dm_smooth = self.minus_dm_state.sum_val
        else:
            atr_smooth = np.nan
            plus_dm_smooth = np.nan
            minus_dm_smooth = np.nan

        def _is_nan(value: Optional[float]) -> bool:
            try:
                return np.isnan(value)
            except TypeError:
                return False

        if atr_smooth is None or _is_nan(atr_smooth) or atr_smooth == 0:
            plus_di = np.nan
            minus_di = np.nan
        else:
            plus_di = 100 * plus_dm_smooth / atr_smooth
            minus_di = 100 * minus_dm_smooth / atr_smooth

        if _is_nan(plus_di) or _is_nan(minus_di):
            di_diff = np.nan
            di_sum = np.nan
        else:
            di_diff = plus_di - minus_di
            di_sum = plus_di + minus_di

        if di_sum is None or _is_nan(di_sum) or di_sum == 0:
            dx = np.nan
        else:
            dx = 100 * abs(di_diff) / di_sum

        self.dx_state.add(dx)
        adx_val = self.dx_state.mean(min_count=self.period)

        self.history.append(adx_val)

        return {
            'adx': adx_val,
            'plus_di': plus_di,
            'minus_di': minus_di,
            'di_diff': di_diff
        }

    def slope(self) -> float:
        """ADX change over 3 bars"""
        if len(self.history) >= 4:
            current = self.history[-1]
            prior = self.history[-4]
            if (isinstance(current, float) and np.isnan(current)) or (isinstance(prior, float) and np.isnan(prior)):
                return np.nan
            return current - prior
        return np.nan


class IncrementalBB:
    """Incrementally updated Bollinger Bands"""
    def __init__(self, period: int = 20, std_mult: float = 2.0):
        self.period = period
        self.std_mult = std_mult
        self.state = RollingState()
        self.state.buffer = deque(maxlen=period)

    def update(self, close: float) -> Dict[str, float]:
        """Update Bollinger Bands with new close"""
        self.state.add(close)

        mid = self.state.mean(min_count=self.period)
        std_dev = self.state.std(ddof=1, min_count=self.period)

        if np.isnan(mid) or np.isnan(std_dev) or std_dev == 0 or mid == 0:
            return {
                'bb_width': np.nan,
                'bb_position': np.nan
            }

        upper = mid + self.std_mult * std_dev
        lower = mid - self.std_mult * std_dev

        width = (upper - lower) / mid if mid != 0 else np.nan
        position = (close - lower) / (upper - lower) if (upper - lower) != 0 else np.nan

        return {
            'bb_width': width,
            'bb_position': position
        }


class IncrementalSMA:
    """Incrementally updated SMA using rolling sum"""
    def __init__(self, period: int):
        self.period = period
        self.buffer = deque(maxlen=period)
        self.sum_val = 0.0

    def update(self, value: float) -> float:
        """Update SMA with new value"""
        if len(self.buffer) == self.period:
            self.sum_val -= self.buffer[0]
        self.buffer.append(value)
        self.sum_val += value
        if len(self.buffer) < self.period:
            return np.nan
        return self.sum_val / self.period


class IncrementalOBV:
    """Incrementally updated On-Balance Volume"""
    def __init__(self):
        self.prev_close: Optional[float] = None
        self.value: float = 0.0
        self.history = deque(maxlen=6)  # For slope over 5 bars

    def update(self, close: float, volume: float) -> float:
        """Update OBV with new bar"""
        if self.prev_close is None:
            self.prev_close = close
            self.history.append(np.nan)
            return np.nan

        if close > self.prev_close:
            self.value += volume
        elif close < self.prev_close:
            self.value -= volume

        self.prev_close = close
        self.history.append(self.value)
        return self.value

    def slope(self, volume_sma: float) -> float:
        """OBV change over 5 bars normalized by volume SMA"""
        if len(self.history) >= 6 and volume_sma > 0:
            current = self.history[-1]
            prior = self.history[-6]
            if (isinstance(current, float) and np.isnan(current)) or (isinstance(prior, float) and np.isnan(prior)):
                return np.nan
            return (current - prior) / volume_sma
        return np.nan


class IncrementalMomentum:
    """Incrementally updated momentum (pct_change over N bars)"""
    def __init__(self, periods: List[int]):
        self.periods = periods
        max_period = max(periods) + 1
        self.buffer = deque(maxlen=max_period)

    def update(self, close: float) -> Dict[int, float]:
        """Update momentum and return all period values"""
        self.buffer.append(close)

        result = {}
        for period in self.periods:
            if len(self.buffer) > period:
                prev_close = self.buffer[-(period + 1)]
                if prev_close != 0:
                    result[period] = (close - prev_close) / prev_close
                else:
                    result[period] = np.nan
            else:
                result[period] = np.nan

        return result


class IncrementalReturnsVol:
    """Incrementally updated returns volatility"""
    def __init__(self, period: int = 20):
        self.period = period
        self.prev_close: Optional[float] = None
        self.state = RollingState()
        self.state.buffer = deque(maxlen=period)

    def update(self, close: float) -> Tuple[float, float]:
        """Update and return (return, volatility)"""
        if self.prev_close is None or self.prev_close == 0:
            ret = np.nan
        else:
            ret = (close - self.prev_close) / self.prev_close

        self.prev_close = close
        self.state.add(ret)

        return ret, self.state.std(ddof=1, min_count=self.period)


class IncrementalSwingDetector:
    """Incrementally updated swing high/low detection"""
    def __init__(self, lookback: int = 10, distance_lookback: int = 50):
        self.lookback = lookback
        self.distance_lookback = distance_lookback
        self.high_buffer = deque(maxlen=lookback * 2 + 1)
        self.low_buffer = deque(maxlen=lookback * 2 + 1)
        self.high_values = deque()  # (index, value)
        self.low_values = deque()   # (index, value)
        self.bar_index = -1

    def update(self, high: float, low: float, close: float, atr: float) -> Dict[str, float]:
        """Update swing detection and return distances"""
        self.bar_index += 1
        self.high_buffer.append(high)
        self.low_buffer.append(low)

        result = {
            'dist_from_high': np.nan,
            'dist_from_low': np.nan,
            'is_swing_high': False,
            'is_swing_low': False
        }

        # Check for confirmed swing high (lookback bars ago)
        if len(self.high_buffer) >= self.lookback * 2 + 1:
            mid_idx = self.lookback
            mid_val = self.high_buffer[mid_idx]
            if mid_val == max(self.high_buffer):
                swing_idx = self.bar_index - self.lookback
                self.high_values.append((swing_idx, mid_val))
                result['is_swing_high'] = True

        # Check for confirmed swing low
        if len(self.low_buffer) >= self.lookback * 2 + 1:
            mid_idx = self.lookback
            mid_val = self.low_buffer[mid_idx]
            if mid_val == min(self.low_buffer):
                swing_idx = self.bar_index - self.lookback
                self.low_values.append((swing_idx, mid_val))
                result['is_swing_low'] = True

        # Drop swings outside the distance lookback window
        cutoff = self.bar_index - self.distance_lookback
        while self.high_values and self.high_values[0][0] < cutoff:
            self.high_values.popleft()
        while self.low_values and self.low_values[0][0] < cutoff:
            self.low_values.popleft()

        # Calculate distances from recent swings
        if self.high_values and atr > 0 and not np.isnan(atr):
            recent_high = self.high_values[-1][1]
            result['dist_from_high'] = (recent_high - close) / atr

        if self.low_values and atr > 0 and not np.isnan(atr):
            recent_low = self.low_values[-1][1]
            result['dist_from_low'] = (close - recent_low) / atr

        return result


class IncrementalPartialHTF:
    """Incrementally updated partial higher timeframe OHLCV"""
    def __init__(self, window_size: int):
        self.window = window_size
        self.open_buffer = deque(maxlen=window_size)
        self.high_buffer = deque(maxlen=window_size)
        self.low_buffer = deque(maxlen=window_size)
        self.close_buffer = deque(maxlen=window_size)
        self.volume_buffer = deque(maxlen=window_size)
        self.volume_sum = 0.0

    def update(self, open_: float, high: float, low: float,
               close: float, volume: float) -> Dict[str, float]:
        """Update partial HTF and return current values"""
        # Handle volume sum
        if len(self.volume_buffer) == self.window:
            self.volume_sum -= self.volume_buffer[0]

        self.open_buffer.append(open_)
        self.high_buffer.append(high)
        self.low_buffer.append(low)
        self.close_buffer.append(close)
        self.volume_buffer.append(volume)
        self.volume_sum += volume

        return {
            'open': self.open_buffer[0] if self.open_buffer else np.nan,
            'high': max(self.high_buffer) if self.high_buffer else np.nan,
            'low': min(self.low_buffer) if self.low_buffer else np.nan,
            'close': close,
            'volume': self.volume_sum
        }


class TimeframeState:
    """Complete state for one timeframe"""
    def __init__(self, config: FeatureConfig):
        self.config = config

        # EMA indicators
        self.emas: Dict[int, IncrementalEMA] = {
            period: IncrementalEMA(period) for period in config.ema_periods
        }

        # RSI
        self.rsi = IncrementalRSI(config.rsi_period)

        # ATR
        self.atr = IncrementalATR(config.atr_period)

        # ADX
        self.adx = IncrementalADX(config.adx_period)

        # Bollinger Bands
        self.bb = IncrementalBB(config.bb_period, config.bb_std)

        # Volume features
        self.volume_sma = IncrementalSMA(config.volume_ma_period)
        self.obv = IncrementalOBV()

        # Microstructure SMAs
        self.imbalance_sma = IncrementalSMA(10)
        self.intensity_sma = IncrementalSMA(20)
        self.size_sma = IncrementalSMA(20)

        # Imbalance slope buffer
        self.imbalance_history = deque(maxlen=4)

        # Returns and momentum
        self.returns_vol = IncrementalReturnsVol(20)
        self.momentum = IncrementalMomentum([5, 10, 20])

        # Swing detection
        self.swing = IncrementalSwingDetector(config.swing_lookback)

        # Last bar data for candle features
        self.last_bar: Optional[Dict] = None

        # Track bar count for warmup reporting
        self.bar_count: int = 0

    def is_warmed_up(self) -> bool:
        """
        Check if all indicators have sufficient warmup data.
        Returns True if all key indicators are properly initialized.
        """
        # Key warmup requirements:
        # - RSI: 14 bars
        # - ADX: 28 bars (2x period for proper DX smoothing)
        # - ATR: 14 bars
        # - BB: 20 bars
        # - ATR percentile: 100 bars (for meaningful ranking)
        # - Volume SMA: 20 bars
        # - Momentum: 20 bars
        required_bars = max(
            self.config.rsi_period + 1,      # RSI
            self.config.adx_period * 2,      # ADX
            self.config.atr_period,          # ATR
            self.config.bb_period,           # Bollinger Bands
            self.config.volume_ma_period,    # Volume SMA
            100,                              # ATR percentile buffer
            20,                               # Momentum
        )
        return self.bar_count >= required_bars

    def get_warmup_progress(self) -> tuple:
        """
        Get warmup progress as (current_bars, required_bars).
        """
        required_bars = max(
            self.config.rsi_period + 1,
            self.config.adx_period * 2,
            self.config.atr_period,
            self.config.bb_period,
            self.config.volume_ma_period,
            100,
            20,
        )
        return (self.bar_count, required_bars)

    def update(self, bar: Dict) -> Dict[str, float]:
        """
        Update all indicators for this timeframe with new bar data.

        Args:
            bar: Dictionary with keys: open, high, low, close, volume,
                 buy_sell_imbalance, trade_intensity, avg_trade_size, etc.

        Returns:
            Dictionary of all calculated features
        """
        features = {}

        # Track bar count for warmup
        self.bar_count += 1

        o = bar['open']
        h = bar['high']
        l = bar['low']
        c = bar['close']
        v = bar['volume']

        # === RAW BAR FEATURES (pass-through from aggregate_to_bars) ===
        # These are features calculated during bar aggregation, not from indicators
        raw_bar_features = [
            'value', 'net_side', 'net_volume', 'net_value',
            'avg_tick_dir', 'trade_count', 'buy_volume', 'sell_volume',
            'buy_sell_imbalance', 'vwap', 'trade_intensity', 'avg_trade_size'
        ]
        for feat in raw_bar_features:
            if feat in bar:
                features[feat] = bar[feat]

        # ATR first (needed for normalization)
        atr_val = self.atr.update(h, l, c)
        features['atr'] = atr_val
        features['atr_percentile'] = self.atr.percentile()

        # EMAs and derived features
        ema_vals = {}
        for period, ema in self.emas.items():
            ema_val = ema.update(c)
            ema_vals[period] = ema_val
            features[f'ema_{period}'] = ema_val
            features[f'ema_{period}_slope'] = ema.slope()

            if atr_val > 0:
                features[f'price_vs_ema_{period}'] = (c - ema_val) / atr_val
                slope = ema.slope()
                features[f'ema_{period}_slope_norm'] = slope / atr_val if not np.isnan(slope) else np.nan
            else:
                features[f'price_vs_ema_{period}'] = np.nan
                features[f'ema_{period}_slope_norm'] = np.nan

        # EMA alignment
        sorted_periods = sorted(self.config.ema_periods)
        if len(sorted_periods) >= 3:
            alignment = 0
            for i in range(len(sorted_periods) - 1):
                if ema_vals[sorted_periods[i]] > ema_vals[sorted_periods[i + 1]]:
                    alignment += 1
            max_score = len(sorted_periods) - 1
            features['ema_alignment'] = (alignment - max_score / 2) / (max_score / 2)

        # RSI
        features['rsi'] = self.rsi.update(c)
        features['rsi_slope'] = self.rsi.slope()

        # ADX
        adx_result = self.adx.update(h, l, c)
        features['adx'] = adx_result['adx']
        features['plus_di'] = adx_result['plus_di']
        features['minus_di'] = adx_result['minus_di']
        features['di_diff'] = adx_result['di_diff']
        features['adx_slope'] = self.adx.slope()

        # Bollinger Bands
        bb_result = self.bb.update(c)
        features['bb_width'] = bb_result['bb_width']
        features['bb_position'] = bb_result['bb_position']

        # Volume features
        vol_sma = self.volume_sma.update(v)
        features['volume_sma'] = vol_sma
        features['volume_ratio'] = v / vol_sma if vol_sma > 0 else np.nan

        obv_val = self.obv.update(c, v)
        features['obv'] = obv_val
        features['obv_slope'] = self.obv.slope(vol_sma)

        # Microstructure features
        if 'buy_sell_imbalance' in bar:
            imb = bar['buy_sell_imbalance']
            imb_ma = self.imbalance_sma.update(imb)
            features['imbalance_ma'] = imb_ma
            self.imbalance_history.append(imb)
            if len(self.imbalance_history) >= 4:
                current = self.imbalance_history[-1]
                prior = self.imbalance_history[-4]
                if (isinstance(current, float) and np.isnan(current)) or (isinstance(prior, float) and np.isnan(prior)):
                    features['imbalance_slope'] = np.nan
                else:
                    features['imbalance_slope'] = current - prior
            else:
                features['imbalance_slope'] = np.nan

        if 'trade_intensity' in bar:
            intensity = bar['trade_intensity']
            int_ma = self.intensity_sma.update(intensity)
            features['intensity_ma'] = int_ma
            features['intensity_ratio'] = intensity / int_ma if int_ma > 0 else np.nan

        if 'avg_trade_size' in bar:
            size = bar['avg_trade_size']
            size_ma = self.size_sma.update(size)
            features['size_ma'] = size_ma
            features['size_ratio'] = size / size_ma if size_ma > 0 else np.nan

        # Returns and momentum
        ret, vol = self.returns_vol.update(c)
        features['returns'] = ret
        features['returns_volatility'] = vol

        mom_result = self.momentum.update(c)
        features['momentum_5'] = mom_result.get(5, np.nan)
        features['momentum_10'] = mom_result.get(10, np.nan)
        features['momentum_20'] = mom_result.get(20, np.nan)

        # Swing detection and distances
        swing_result = self.swing.update(h, l, c, atr_val)
        features['dist_from_high'] = swing_result['dist_from_high']
        features['dist_from_low'] = swing_result['dist_from_low']
        features['swing_high'] = bool(swing_result.get('is_swing_high', False))
        features['swing_low'] = bool(swing_result.get('is_swing_low', False))

        # Candle features
        if atr_val > 0:
            features['body_size'] = abs(c - o) / atr_val
            features['upper_wick'] = (h - max(o, c)) / atr_val
            features['lower_wick'] = (min(o, c) - l) / atr_val
        else:
            features['body_size'] = np.nan
            features['upper_wick'] = np.nan
            features['lower_wick'] = np.nan

        features['candle_direction'] = np.sign(c - o)

        self.last_bar = bar

        return features


class IncrementalFeatureEngine:
    """
    Main incremental feature calculation engine.

    Maintains state for all timeframes and provides efficient updates
    when new bars arrive.
    """

    def __init__(
        self,
        config: FeatureConfig,
        base_tf: Optional[str] = None,
        pullback_ema: Optional[int] = None,
        touch_threshold_atr: float = 0.3,
        min_slope_norm: float = 0.03,
    ):
        self.config = config
        self.base_tf = base_tf if base_tf in config.timeframe_names else config.timeframe_names[0]
        self.pullback_ema = pullback_ema
        if self.pullback_ema is None:
            self.pullback_ema = config.ema_periods[0] if config.ema_periods else None
        self.touch_threshold_atr = float(touch_threshold_atr)
        self.min_slope_norm = float(min_slope_norm)
        self._tf_seconds_map = dict(zip(config.timeframe_names, config.timeframes))

        # State for each timeframe
        self.tf_states: Dict[str, TimeframeState] = {
            tf_name: TimeframeState(config)
            for tf_name in config.timeframe_names
        }

        # Partial HTF trackers (for each HTF relative to base)
        base_seconds = self._tf_seconds_map[self.base_tf]

        self.partial_htf: Dict[str, IncrementalPartialHTF] = {}
        for tf_name, tf_seconds in self._tf_seconds_map.items():
            if tf_name != self.base_tf and tf_seconds > base_seconds:
                if tf_seconds % base_seconds == 0:
                    window = tf_seconds // base_seconds
                    self.partial_htf[tf_name] = IncrementalPartialHTF(window)

        # Current features cache
        self.features_cache: Dict[str, float] = {}

        # Higher TF bar caches (for merged features)
        self.htf_features: Dict[str, Dict[str, float]] = {
            tf_name: {} for tf_name in config.timeframe_names
        }
        self._pending_htf: Dict[str, deque] = {
            tf_name: deque() for tf_name in config.timeframe_names if tf_name != self.base_tf
        }

        # Track last bar time for each TF
        self.last_bar_time: Dict[str, Optional[int]] = {
            tf_name: None for tf_name in config.timeframe_names
        }

        # EMA touch diagnostic info (captures decision details for debugging)
        self.last_touch_diagnostic: Dict = {
            "bar_time": None,
            "checked_tfs": [],  # List of (tf_name, slope, threshold_comparison, distance, reason)
            "result": None,  # Final result dict
        }

    def update_timeframe(self, tf_name: str, bar: Dict) -> Dict[str, float]:
        """
        Update features for a specific timeframe.

        Args:
            tf_name: Timeframe name (e.g., '1m', '5m')
            bar: Bar data dictionary

        Returns:
            Updated features for this timeframe (prefixed with tf_name)
        """
        if tf_name not in self.tf_states:
            raise ValueError(f"Unknown timeframe: {tf_name}")

        bar_time = bar.get('bar_time')
        if bar_time is not None:
            try:
                bar_time = int(bar_time)
            except (TypeError, ValueError):
                pass

        last_time = self.last_bar_time.get(tf_name)
        if bar_time is not None and last_time is not None:
            try:
                last_time_val = int(last_time)
            except (TypeError, ValueError):
                last_time_val = last_time
            if bar_time <= last_time_val:
                existing = self.htf_features.get(tf_name, {})
                return {f'{tf_name}_{k}': v for k, v in existing.items()}

        # Update the timeframe state
        raw_features = self.tf_states[tf_name].update(bar)
        prefixed = {f'{tf_name}_{k}': v for k, v in raw_features.items()}

        # Track bar time
        self.last_bar_time[tf_name] = bar_time

        if tf_name == self.base_tf:
            # Base TF: apply immediately
            self.htf_features[tf_name] = raw_features
            self.features_cache.update(prefixed)
        else:
            # Non-base TF: queue until its close time is visible to base bars
            if bar_time is None:
                self.htf_features[tf_name] = raw_features
                self.features_cache.update(prefixed)
            else:
                self._pending_htf[tf_name].append((int(bar_time), raw_features))

        return prefixed

    def _publish_pending_htf(self, base_bar_time: Optional[int]) -> None:
        """Publish higher timeframe bars once their close time is visible to base bars."""
        if base_bar_time is None:
            for tf_name, queue in self._pending_htf.items():
                if not queue:
                    continue
                _, raw_features = queue[-1]
                queue.clear()
                self.htf_features[tf_name] = raw_features
                prefixed = {f'{tf_name}_{k}': v for k, v in raw_features.items()}
                self.features_cache.update(prefixed)
            return

        for tf_name, queue in self._pending_htf.items():
            tf_seconds = self._tf_seconds_map.get(tf_name, 0)
            while queue and (queue[0][0] + tf_seconds) <= base_bar_time:
                _, raw_features = queue.popleft()
                self.htf_features[tf_name] = raw_features
            if tf_name in self.htf_features and self.htf_features[tf_name]:
                prefixed = {f'{tf_name}_{k}': v for k, v in self.htf_features[tf_name].items()}
                self.features_cache.update(prefixed)

    def _compute_ema_touch_features(self, bar: Dict) -> Dict[str, float]:
        """Compute multi-TF EMA touch diagnostics for the current base bar."""
        results = {
            'ema_touch_detected': False,
            'ema_touch_tf': None,
            'ema_touch_direction': 0,
            'ema_touch_quality': 0.0,
            'ema_touch_dist': np.nan,
            'ema_touch_slope': np.nan,
        }

        # Reset diagnostic info for this bar
        bar_time = bar.get('bar_time')
        self.last_touch_diagnostic = {
            "bar_time": bar_time,
            "checked_tfs": [],
            "result": None,
            "config": {
                "touch_threshold_atr": self.touch_threshold_atr,
                "min_slope_norm": self.min_slope_norm,
                "pullback_ema": self.pullback_ema,
            },
            "bar_info": {
                "open": bar.get('open'),
                "high": bar.get('high'),
                "low": bar.get('low'),
                "close": bar.get('close'),
            },
        }

        ema_period = self.pullback_ema
        if ema_period is None:
            self.last_touch_diagnostic["result"] = results
            self.last_touch_diagnostic["skip_reason"] = "no_pullback_ema"
            return results

        atr_key = f'{self.base_tf}_atr'
        atr_val = self.features_cache.get(atr_key)
        if atr_val is None or (isinstance(atr_val, float) and np.isnan(atr_val)) or atr_val <= 0:
            self.last_touch_diagnostic["result"] = results
            self.last_touch_diagnostic["skip_reason"] = f"invalid_atr: {atr_val}"
            return results

        self.last_touch_diagnostic["atr"] = atr_val

        bar_open = bar.get('open')
        bar_high = bar.get('high')
        bar_low = bar.get('low')
        bar_close = bar.get('close')
        if bar_open is None or bar_high is None or bar_low is None or bar_close is None:
            self.last_touch_diagnostic["result"] = results
            self.last_touch_diagnostic["skip_reason"] = "missing_bar_data"
            return results

        mid_bar = (bar_high + bar_low) / 2.0
        threshold = self.touch_threshold_atr
        min_slope = self.min_slope_norm

        for tf_name in self.config.timeframe_names:
            ema_key = f'{tf_name}_ema_{ema_period}'
            slope_key = f'{tf_name}_ema_{ema_period}_slope_norm'
            ema_val = self.features_cache.get(ema_key)
            slope_val = self.features_cache.get(slope_key)

            tf_diag = {
                "tf": tf_name,
                "ema_key": ema_key,
                "ema_val": ema_val,
                "slope_key": slope_key,
                "slope_val": slope_val,
                "min_slope": min_slope,
                "threshold": threshold,
                "decision": None,
                "reason": None,
            }

            if ema_val is None or slope_val is None:
                tf_diag["decision"] = "skip"
                tf_diag["reason"] = "missing_ema_or_slope"
                self.last_touch_diagnostic["checked_tfs"].append(tf_diag)
                continue
            if (isinstance(ema_val, float) and np.isnan(ema_val)) or (isinstance(slope_val, float) and np.isnan(slope_val)):
                tf_diag["decision"] = "skip"
                tf_diag["reason"] = "nan_ema_or_slope"
                self.last_touch_diagnostic["checked_tfs"].append(tf_diag)
                continue

            if slope_val > min_slope:
                # Long setup: low touches EMA from above
                dist_low = (bar_low - ema_val) / atr_val
                tf_diag["setup"] = "long"
                tf_diag["dist"] = dist_low
                tf_diag["dist_in_range"] = -threshold <= dist_low <= threshold

                if -threshold <= dist_low <= threshold:
                    close_or_mid_above = bar_close >= ema_val or mid_bar >= ema_val
                    tf_diag["close_or_mid_above_ema"] = close_or_mid_above
                    if close_or_mid_above:
                        dist_score = 1.0 - min(abs(dist_low) / threshold, 1.0) if threshold > 0 else 0.0
                        wick_score = (bar_close - bar_low) / (bar_high - bar_low) if bar_high > bar_low else 0.0
                        quality = (dist_score + wick_score) / 2.0

                        results.update({
                            'ema_touch_detected': True,
                            'ema_touch_tf': tf_name,
                            'ema_touch_direction': 1,
                            'ema_touch_quality': quality,
                            'ema_touch_dist': dist_low,
                            'ema_touch_slope': slope_val,
                        })
                        tf_diag["decision"] = "TOUCH_DETECTED"
                        tf_diag["reason"] = "long_touch"
                        tf_diag["quality"] = quality
                        self.last_touch_diagnostic["checked_tfs"].append(tf_diag)
                        self.last_touch_diagnostic["result"] = results
                        return results
                    else:
                        tf_diag["decision"] = "no_touch"
                        tf_diag["reason"] = "close_and_mid_below_ema"
                else:
                    tf_diag["decision"] = "no_touch"
                    tf_diag["reason"] = f"dist_out_of_range: {dist_low:.4f} not in [-{threshold}, {threshold}]"

            elif slope_val < -min_slope:
                # Short setup: high touches EMA from below
                dist_high = (bar_high - ema_val) / atr_val
                tf_diag["setup"] = "short"
                tf_diag["dist"] = dist_high
                tf_diag["dist_in_range"] = -threshold <= dist_high <= threshold

                if -threshold <= dist_high <= threshold:
                    close_or_mid_below = bar_close <= ema_val or mid_bar <= ema_val
                    tf_diag["close_or_mid_below_ema"] = close_or_mid_below
                    if close_or_mid_below:
                        dist_score = 1.0 - min(abs(dist_high) / threshold, 1.0) if threshold > 0 else 0.0
                        wick_score = (bar_high - bar_close) / (bar_high - bar_low) if bar_high > bar_low else 0.0
                        quality = (dist_score + wick_score) / 2.0

                        results.update({
                            'ema_touch_detected': True,
                            'ema_touch_tf': tf_name,
                            'ema_touch_direction': -1,
                            'ema_touch_quality': quality,
                            'ema_touch_dist': dist_high,
                            'ema_touch_slope': slope_val,
                        })
                        tf_diag["decision"] = "TOUCH_DETECTED"
                        tf_diag["reason"] = "short_touch"
                        tf_diag["quality"] = quality
                        self.last_touch_diagnostic["checked_tfs"].append(tf_diag)
                        self.last_touch_diagnostic["result"] = results
                        return results
                    else:
                        tf_diag["decision"] = "no_touch"
                        tf_diag["reason"] = "close_and_mid_above_ema"
                else:
                    tf_diag["decision"] = "no_touch"
                    tf_diag["reason"] = f"dist_out_of_range: {dist_high:.4f} not in [-{threshold}, {threshold}]"
            else:
                tf_diag["decision"] = "no_touch"
                tf_diag["reason"] = f"slope_neutral: |{slope_val:.6f}| <= {min_slope}"

            self.last_touch_diagnostic["checked_tfs"].append(tf_diag)

        self.last_touch_diagnostic["result"] = results
        return results

    def _compute_bounce_features(self, bar: Dict, touch_direction: int) -> Dict[str, float]:
        """Compute bounce bar features only when an EMA touch is detected."""
        results = {
            'bounce_bar_body_ratio': np.nan,
            'bounce_bar_wick_ratio': np.nan,
            'bounce_bar_direction': np.nan,
            'bounce_volume_ratio': np.nan,
        }

        if touch_direction == 0:
            return results

        bar_open = bar.get('open')
        bar_high = bar.get('high')
        bar_low = bar.get('low')
        bar_close = bar.get('close')
        bar_volume = bar.get('volume')
        if bar_open is None or bar_high is None or bar_low is None or bar_close is None:
            return results

        bar_range = bar_high - bar_low
        if bar_range > 0:
            body_size = abs(bar_close - bar_open)
            results['bounce_bar_body_ratio'] = body_size / bar_range
            if touch_direction > 0:
                favorable_wick = min(bar_open, bar_close) - bar_low
            else:
                favorable_wick = bar_high - max(bar_open, bar_close)
            results['bounce_bar_wick_ratio'] = favorable_wick / bar_range

        results['bounce_bar_direction'] = 1 if bar_close > bar_open else -1

        volume_ma_key = f'{self.base_tf}_volume_sma'
        volume_ma = self.features_cache.get(volume_ma_key)
        if (
            bar_volume is not None
            and volume_ma is not None
            and not (isinstance(volume_ma, float) and np.isnan(volume_ma))
            and volume_ma > 0
        ):
            results['bounce_volume_ratio'] = bar_volume / volume_ma

        return results

    def update_base_tf_bar(self, bar: Dict) -> Dict[str, float]:
        """
        Update with a new base timeframe bar.
        This also updates partial HTF features.

        Args:
            bar: Base timeframe bar data

        Returns:
            All current features
        """
        bar_time = bar.get('bar_time')
        if bar_time is not None:
            try:
                bar_time = int(bar_time)
            except (TypeError, ValueError):
                pass
        last_time = self.last_bar_time.get(self.base_tf)
        if bar_time is not None and last_time is not None:
            try:
                last_time_val = int(last_time)
            except (TypeError, ValueError):
                last_time_val = last_time
            if bar_time <= last_time_val:
                return self.features_cache.copy()

        # Update base TF features
        self.update_timeframe(self.base_tf, bar)

        # Store base bar fields (unprefixed, matches feature_engine output)
        for key in ('bar_time', 'datetime', 'open', 'high', 'low', 'close', 'volume'):
            if key in bar:
                self.features_cache[key] = bar[key]

        # Publish higher timeframe features that have closed relative to this base bar
        self._publish_pending_htf(bar_time)

        # Update cross-TF features
        self._update_cross_tf_features()

        # EMA touch diagnostics + bounce features (match labels.py)
        touch_info = self._compute_ema_touch_features(bar)
        self.features_cache.update(touch_info)
        bounce_info = self._compute_bounce_features(bar, touch_info.get('ema_touch_direction', 0))
        self.features_cache.update(bounce_info)

        # Update partial HTF trackers
        for tf_name, tracker in self.partial_htf.items():
            partial = tracker.update(
                bar['open'], bar['high'], bar['low'],
                bar['close'], bar['volume']
            )
            self.features_cache[f'partial_{tf_name}_open'] = partial['open']
            self.features_cache[f'partial_{tf_name}_high'] = partial['high']
            self.features_cache[f'partial_{tf_name}_low'] = partial['low']
            self.features_cache[f'partial_{tf_name}_close'] = partial['close']
            self.features_cache[f'partial_{tf_name}_volume'] = partial['volume']

        return self.features_cache.copy()

    def _update_cross_tf_features(self):
        """Update features that combine multiple timeframes"""
        def _is_finite(value: Optional[float]) -> bool:
            return value is not None and not (isinstance(value, float) and np.isnan(value))

        # Trend agreement (mean of ema_alignment)
        alignment_vals = []
        for tf_name in self.config.timeframe_names:
            key = f'{tf_name}_ema_alignment'
            if key in self.features_cache:
                val = self.features_cache[key]
                if _is_finite(val):
                    alignment_vals.append(val)

        if alignment_vals:
            self.features_cache['tf_trend_agreement'] = np.mean(alignment_vals)

        # Trending count (ADX > 25)
        trending_count = 0
        for tf_name in self.config.timeframe_names:
            key = f'{tf_name}_adx'
            if key in self.features_cache:
                val = self.features_cache[key]
                if _is_finite(val) and val > 25:
                    trending_count += 1

        self.features_cache['tf_trending_count'] = trending_count

        # Average RSI
        rsi_vals = []
        for tf_name in self.config.timeframe_names:
            key = f'{tf_name}_rsi'
            if key in self.features_cache:
                val = self.features_cache[key]
                if _is_finite(val):
                    rsi_vals.append(val)

        if rsi_vals:
            self.features_cache['tf_avg_rsi'] = np.mean(rsi_vals)

    def get_features_df(self) -> pd.DataFrame:
        """Get current features as a single-row DataFrame"""
        return pd.DataFrame([self.features_cache])

    def get_feature(self, name: str) -> float:
        """Get a specific feature value"""
        return self.features_cache.get(name, np.nan)

    def is_warmed_up(self) -> bool:
        """
        Check if ALL timeframes have sufficient warmup data for valid features.
        Returns True only when all timeframe indicators are properly initialized.
        """
        for tf_name, tf_state in self.tf_states.items():
            if not tf_state.is_warmed_up():
                return False
        return True

    def get_warmup_status(self) -> Dict[str, tuple]:
        """
        Get warmup progress for each timeframe.
        Returns dict of {tf_name: (current_bars, required_bars)}.
        """
        return {
            tf_name: tf_state.get_warmup_progress()
            for tf_name, tf_state in self.tf_states.items()
        }

    def get_warmup_summary(self) -> str:
        """
        Get a human-readable warmup status summary.
        """
        status = self.get_warmup_status()
        parts = []
        all_ready = True
        for tf_name in self.config.timeframe_names:
            if tf_name in status:
                current, required = status[tf_name]
                if current >= required:
                    parts.append(f"{tf_name}:OK")
                else:
                    parts.append(f"{tf_name}:{current}/{required}")
                    all_ready = False
        if all_ready:
            return "All TFs warmed up"
        return " | ".join(parts)

    def get_all_features(self) -> Dict[str, float]:
        """Get all current features"""
        return self.features_cache.copy()


class IncrementalBarAggregator:
    """
    Aggregates raw trades into bars incrementally.
    Maintains state for a single timeframe.
    """

    def __init__(self, timeframe_seconds: int):
        self.tf_seconds = timeframe_seconds
        self.current_bar_time: Optional[int] = None

        # Current bar accumulator
        self.trades_in_bar: int = 0
        self.open_price: Optional[float] = None
        self.high_price: float = 0.0
        self.low_price: float = float('inf')
        self.close_price: Optional[float] = None
        self.volume: float = 0.0
        self.value: float = 0.0
        self.net_volume: float = 0.0  # signed volume (buy - sell)
        self.net_value: float = 0.0   # signed value
        self.net_side: float = 0.0    # count of buys - sells
        self.tick_dir_sum: float = 0.0

    def add_trade(self, timestamp: float, price: float, size: float,
                  side: int, tick_dir: float) -> Optional[Dict]:
        """
        Add a trade and return completed bar if timeframe boundary crossed.

        Args:
            timestamp: Unix timestamp
            price: Trade price
            size: Trade size
            side: 1 for Buy, -1 for Sell
            tick_dir: Tick direction numeric value

        Returns:
            Completed bar dictionary if boundary crossed, None otherwise
        """
        bar_time = int(timestamp // self.tf_seconds) * self.tf_seconds

        completed_bar = None

        # Check if we crossed a bar boundary
        if self.current_bar_time is not None and bar_time > self.current_bar_time:
            # Complete the current bar
            completed_bar = self._finalize_bar()

        # Start new bar if needed
        if self.current_bar_time != bar_time:
            self._reset_bar(bar_time, price)

        # Accumulate trade
        self.trades_in_bar += 1
        self.high_price = max(self.high_price, price)
        self.low_price = min(self.low_price, price)
        self.close_price = price

        self.volume += size
        self.value += price * size
        self.net_volume += size * side
        self.net_value += price * size * side
        self.net_side += side  # +1 for buy, -1 for sell
        self.tick_dir_sum += tick_dir

        return completed_bar

    def _reset_bar(self, bar_time: int, price: float):
        """Reset accumulator for new bar"""
        self.current_bar_time = bar_time
        self.trades_in_bar = 0
        self.open_price = price
        self.high_price = price
        self.low_price = price
        self.close_price = price
        self.volume = 0.0
        self.value = 0.0
        self.net_volume = 0.0
        self.net_value = 0.0
        self.net_side = 0.0
        self.tick_dir_sum = 0.0

    def _finalize_bar(self) -> Dict:
        """Finalize current bar and return it"""
        # Calculate buy/sell volumes from net_volume and total volume
        # net_volume = buy_volume - sell_volume
        # volume = buy_volume + sell_volume
        # Therefore: buy_volume = (volume + net_volume) / 2
        #            sell_volume = (volume - net_volume) / 2
        buy_volume = (self.volume + self.net_volume) / 2
        sell_volume = (self.volume - self.net_volume) / 2

        bar = {
            'bar_time': self.current_bar_time,
            'open': self.open_price,
            'high': self.high_price,
            'low': self.low_price,
            'close': self.close_price,
            'volume': self.volume,
            'value': self.value,
            'net_volume': self.net_volume,
            'net_value': self.net_value,
            'net_side': self.net_side,
            'trade_count': self.trades_in_bar,
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
        }

        # Derived fields
        if self.volume > 0:
            bar['buy_sell_imbalance'] = self.net_volume / self.volume
            bar['vwap'] = self.value / self.volume
            bar['avg_trade_size'] = self.volume / self.trades_in_bar
        else:
            bar['buy_sell_imbalance'] = 0.0
            bar['vwap'] = self.close_price
            bar['avg_trade_size'] = 0.0

        bar['trade_intensity'] = self.trades_in_bar / self.tf_seconds
        bar['avg_tick_dir'] = self.tick_dir_sum / self.trades_in_bar if self.trades_in_bar > 0 else 0.0

        return bar

    def get_current_bar(self) -> Optional[Dict]:
        """Get the current (incomplete) bar"""
        if self.current_bar_time is None:
            return None
        return self._finalize_bar()


class IncrementalPredictor:
    """
    Full incremental prediction pipeline.

    Replaces the expensive full-recalculation in TrendFollowerPredictor
    with efficient incremental updates.
    """

    def __init__(self, config: FeatureConfig, base_tf: Optional[str] = None):
        self.config = config

        # Bar aggregators for each timeframe
        self.aggregators: Dict[str, IncrementalBarAggregator] = {
            tf_name: IncrementalBarAggregator(tf_seconds)
            for tf_name, tf_seconds in zip(config.timeframe_names, config.timeframes)
        }

        # Base timeframe name
        self.base_tf = base_tf if base_tf in config.timeframe_names else config.timeframe_names[0]

        # Feature engine
        self.feature_engine = IncrementalFeatureEngine(config, base_tf=self.base_tf)

    def add_trade(self, timestamp: float, price: float, size: float,
                  side: int, tick_dir: float) -> Optional[Dict[str, float]]:
        """
        Add a trade and update features if any bar completes.

        Returns:
            Updated features if base TF bar completed, None otherwise
        """
        features_updated = None

        completed = []

        # Update all timeframe aggregators
        for tf_name, agg in self.aggregators.items():
            completed_bar = agg.add_trade(timestamp, price, size, side, tick_dir)
            if completed_bar is not None:
                completed.append((tf_name, completed_bar))

        # Apply non-base TF updates first (for shift-aligned publishing)
        for tf_name, bar in completed:
            if tf_name != self.base_tf:
                self.feature_engine.update_timeframe(tf_name, bar)

        # Apply base TF update last
        for tf_name, bar in completed:
            if tf_name == self.base_tf:
                features_updated = self.feature_engine.update_base_tf_bar(bar)

        return features_updated

    def add_bar(self, tf_name: str, bar: Dict) -> Dict[str, float]:
        """
        Add a completed bar directly (for bulk loading).

        Args:
            tf_name: Timeframe name
            bar: Bar data dictionary

        Returns:
            Updated features
        """
        if tf_name == self.base_tf:
            return self.feature_engine.update_base_tf_bar(bar)

        self.feature_engine.update_timeframe(tf_name, bar)
        return self.feature_engine.get_all_features()

    def get_features(self) -> Dict[str, float]:
        """Get current feature values"""
        return self.feature_engine.get_all_features()

    def get_features_df(self) -> pd.DataFrame:
        """Get features as DataFrame for model prediction"""
        return self.feature_engine.get_features_df()


def warm_up_from_bars(predictor: IncrementalPredictor,
                       bars_dict: Dict[str, pd.DataFrame]) -> None:
    """
    Warm up the incremental predictor from historical bars.

    This is used during startup to initialize the incremental state
    from existing data without recalculating everything.

    Args:
        predictor: IncrementalPredictor to warm up
        bars_dict: Dictionary of bar DataFrames by timeframe
    """
    # Get all timeframes sorted by seconds (smallest first)
    tf_seconds = dict(zip(predictor.config.timeframe_names, predictor.config.timeframes))
    sorted_tfs = sorted(predictor.config.timeframe_names, key=lambda x: tf_seconds[x])

    base_tf = predictor.base_tf
    non_base_tfs = [tf for tf in sorted_tfs if tf != base_tf]

    # Enqueue non-base TF bars first
    for tf_name in non_base_tfs:
        if tf_name in bars_dict:
            tf_bars = bars_dict[tf_name].sort_values('bar_time')
            for _, row in tf_bars.iterrows():
                bar = row.to_dict()
                predictor.feature_engine.update_timeframe(tf_name, bar)

    # Then process base TF bars to publish shift-aligned features
    if base_tf in bars_dict:
        base_bars = bars_dict[base_tf].sort_values('bar_time')
        for _, row in base_bars.iterrows():
            bar = row.to_dict()
            predictor.feature_engine.update_base_tf_bar(bar)
