"""
Label generation for trend following ML model.
Creates training labels from historical price data.

Improvements (v1.1):
- Sequential TP/SL labeling: matches actual trade execution (TP hit before SL)
- Multi-tier quality labels: 0=stopped, 1=breakeven-1R, 2=1R-2R, 3=2R+
- Time-to-target features: bars_to_exit, early momentum
- Bounce reaction features: candle quality at the bounce point
- Probability calibration support via sklearn

Improvements (v1.2 - Multi-Timeframe EMA Touch):
- Detect EMA touches across ALL timeframes (1m, 5m, 15m, 1h, 4h)
- Use intrabar HIGH/LOW to detect when price "kissed" the EMA
- Directional setup: EMA slope positive + price above + dips to EMA = long setup
- Generates 10-50x more training samples by scanning all timeframes
- Each touch is labeled with which timeframe EMA was touched
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict
from config import LabelConfig, FeatureConfig


def label_trend_opportunities(
    df: pd.DataFrame,
    config: LabelConfig,
    base_tf: str
) -> pd.DataFrame:
    """
    Label each bar with whether a tradeable trend followed.
    
    Labels:
        1: Strong uptrend followed (price went up significantly)
       -1: Strong downtrend followed (price went down significantly)
        0: No clear trend (choppy, ranging, or small move)
    
    Args:
        df: DataFrame with OHLCV and features
        config: LabelConfig with thresholds
        base_tf: Base timeframe prefix
        
    Returns:
        DataFrame with 'trend_label' column added
    """
    result = df.copy()
    n = len(result)
    window = config.trend_forward_window
    
    # Get ATR column
    atr_col = f'{base_tf}_atr'
    if atr_col not in result.columns:
        raise ValueError(f"ATR column '{atr_col}' not found")
    
    labels = np.zeros(n)
    max_favorable = np.zeros(n)
    max_adverse = np.zeros(n)
    
    for i in range(n - window):
        current_price = result['close'].iloc[i]
        current_atr = result[atr_col].iloc[i]
        
        if current_atr <= 0 or pd.isna(current_atr):
            continue
        
        # Look forward
        future_highs = result['high'].iloc[i+1:i+window+1]
        future_lows = result['low'].iloc[i+1:i+window+1]
        
        # Max move up and down
        max_up = (future_highs.max() - current_price) / current_atr
        max_down = (current_price - future_lows.min()) / current_atr
        
        max_favorable[i] = max(max_up, max_down)
        
        # Strong uptrend: went up significantly without big drawdown first
        if max_up >= config.trend_up_threshold and max_down < config.max_adverse_for_trend:
            labels[i] = 1
            max_adverse[i] = max_down
        # Strong downtrend: went down significantly without big drawup first
        elif max_down >= config.trend_down_threshold and max_up < config.max_adverse_for_trend:
            labels[i] = -1
            max_adverse[i] = max_up
        else:
            labels[i] = 0
            max_adverse[i] = min(max_up, max_down)
    
    result['trend_label'] = labels
    result['trend_max_favorable'] = max_favorable
    result['trend_max_adverse'] = max_adverse
    
    return result


def detect_pullback_zones(
    df: pd.DataFrame,
    config: LabelConfig,
    base_tf: str
) -> pd.Series:
    """
    LEGACY: Detect bars where price is pulling back to a key EMA in a trend.
    Use detect_multi_tf_ema_touches() for the new multi-timeframe approach.
    """
    ema_col = f'{base_tf}_ema_{config.pullback_ema}'
    atr_col = f'{base_tf}_atr'
    alignment_col = f'{base_tf}_ema_{config.pullback_ema}_slope_norm'

    for col in [ema_col, atr_col]:
        if col not in df.columns:
            print(f"Warning: Column {col} not found, using fallback")
            return pd.Series(False, index=df.index)
    if alignment_col not in df.columns:
        df = df.copy()
        df[alignment_col] = 0

    ohlc = df[['open', 'high', 'low', 'close']]
    ema_vals = df[ema_col]
    dist_ema = (ohlc.sub(ema_vals, axis=0)).abs()
    min_dist = dist_ema.min(axis=1)
    dist_from_ema = min_dist / df[atr_col].replace(0, np.nan)

    is_close_to_ema = dist_from_ema <= config.pullback_threshold
    has_trend = abs(df[alignment_col]) > 0.05

    return is_close_to_ema & has_trend


def detect_multi_tf_ema_touches(
    df: pd.DataFrame,
    config: LabelConfig,
    feature_config: FeatureConfig,
    base_tf: str,
    touch_threshold_atr: float = 0.3,
    min_slope_norm: float = 0.03,
) -> pd.DataFrame:
    """
    Detect EMA touches across ALL timeframes to generate more training samples.

    Trading Logic (your intuition):
    - For LONG setup: EMA slope positive, price generally above EMA, price dips DOWN to kiss EMA
    - For SHORT setup: EMA slope negative, price generally below EMA, price rises UP to kiss EMA
    - "Kiss" = intrabar low (for long) or high (for short) touches or slightly crosses EMA

    This function scans:
    - 1m 9EMA, 5m 9EMA, 15m 9EMA, 1h 9EMA, 4h 9EMA
    - For each bar, checks if ANY of these EMAs was touched with a valid directional setup

    Args:
        df: DataFrame with multi-TF features (must have columns like '5m_ema_9', '15m_ema_9', etc.)
        config: LabelConfig
        feature_config: FeatureConfig with timeframe names
        base_tf: Base timeframe prefix (e.g., '5m')
        touch_threshold_atr: How close to EMA counts as a "touch" (in ATR units, default 0.3)
        min_slope_norm: Minimum slope to consider trending (default 0.03)

    Returns:
        DataFrame with new columns:
        - ema_touch_detected: Boolean, any valid touch found
        - ema_touch_tf: Which timeframe's EMA was touched (e.g., '5m', '1h')
        - ema_touch_direction: 1 for long setup, -1 for short setup
        - ema_touch_quality: How clean the touch was (0-1 score)
        - ema_touch_dist: Distance from EMA in ATR units at touch
    """
    result = df.copy()
    n = len(result)

    # Initialize output columns
    result['ema_touch_detected'] = False
    result['ema_touch_tf'] = None
    result['ema_touch_direction'] = 0
    result['ema_touch_quality'] = 0.0
    result['ema_touch_dist'] = np.nan
    result['ema_touch_slope'] = np.nan

    # Get all available timeframes from the data
    # Look for columns like '5m_ema_9', '15m_ema_9', etc.
    ema_period = config.pullback_ema  # Usually 9
    timeframe_names = feature_config.timeframe_names  # ['1m', '5m', '15m', '1h', '4h']

    # We need ATR for normalization - use base TF's ATR
    atr_col = f'{base_tf}_atr'
    if atr_col not in result.columns:
        print(f"Warning: ATR column {atr_col} not found")
        return result

    touches_found = 0
    touches_by_tf = {}

    for tf in timeframe_names:
        ema_col = f'{tf}_ema_{ema_period}'
        slope_col = f'{tf}_ema_{ema_period}_slope_norm'

        if ema_col not in result.columns:
            continue
        if slope_col not in result.columns:
            continue

        touches_by_tf[tf] = 0

        for i in range(n):
            # Skip if already found a touch for this bar (prioritize higher TFs)
            if result['ema_touch_detected'].iloc[i]:
                continue

            ema_val = result[ema_col].iloc[i]
            slope_val = result[slope_col].iloc[i]
            atr_val = result[atr_col].iloc[i]

            if pd.isna(ema_val) or pd.isna(slope_val) or pd.isna(atr_val) or atr_val <= 0:
                continue

            bar_high = result['high'].iloc[i]
            bar_low = result['low'].iloc[i]
            bar_open = result['open'].iloc[i]
            bar_close = result['close'].iloc[i]

            # Determine if this is a valid directional setup
            # LONG SETUP: slope positive, looking for price to dip DOWN to EMA
            # SHORT SETUP: slope negative, looking for price to rise UP to EMA

            if slope_val > min_slope_norm:
                # LONG SETUP
                # Price should come down to touch EMA from above
                # The LOW of the bar should be near the EMA
                dist_low_to_ema = (bar_low - ema_val) / atr_val

                # Valid touch: low is within threshold of EMA (can be slightly below)
                # AND the bar shows some rejection (close above low)
                if -touch_threshold_atr <= dist_low_to_ema <= touch_threshold_atr:
                    # Check that price is generally above EMA (bullish context)
                    # Close should be above EMA, or at least mid-bar above EMA
                    mid_bar = (bar_high + bar_low) / 2
                    if bar_close >= ema_val or mid_bar >= ema_val:
                        # Calculate touch quality
                        # Better quality = smaller distance + bigger rejection wick
                        dist_score = 1.0 - min(abs(dist_low_to_ema) / touch_threshold_atr, 1.0)
                        wick_score = (bar_close - bar_low) / (bar_high - bar_low) if bar_high > bar_low else 0

                        quality = (dist_score + wick_score) / 2

                        result.iloc[i, result.columns.get_loc('ema_touch_detected')] = True
                        result.iloc[i, result.columns.get_loc('ema_touch_tf')] = tf
                        result.iloc[i, result.columns.get_loc('ema_touch_direction')] = 1
                        result.iloc[i, result.columns.get_loc('ema_touch_quality')] = quality
                        result.iloc[i, result.columns.get_loc('ema_touch_dist')] = dist_low_to_ema
                        result.iloc[i, result.columns.get_loc('ema_touch_slope')] = slope_val

                        touches_found += 1
                        touches_by_tf[tf] += 1

            elif slope_val < -min_slope_norm:
                # SHORT SETUP
                # Price should come up to touch EMA from below
                # The HIGH of the bar should be near the EMA
                dist_high_to_ema = (bar_high - ema_val) / atr_val

                # Valid touch: high is within threshold of EMA (can be slightly above)
                if -touch_threshold_atr <= dist_high_to_ema <= touch_threshold_atr:
                    # Check that price is generally below EMA (bearish context)
                    mid_bar = (bar_high + bar_low) / 2
                    if bar_close <= ema_val or mid_bar <= ema_val:
                        # Calculate touch quality
                        dist_score = 1.0 - min(abs(dist_high_to_ema) / touch_threshold_atr, 1.0)
                        wick_score = (bar_high - bar_close) / (bar_high - bar_low) if bar_high > bar_low else 0

                        quality = (dist_score + wick_score) / 2

                        result.iloc[i, result.columns.get_loc('ema_touch_detected')] = True
                        result.iloc[i, result.columns.get_loc('ema_touch_tf')] = tf
                        result.iloc[i, result.columns.get_loc('ema_touch_direction')] = -1
                        result.iloc[i, result.columns.get_loc('ema_touch_quality')] = quality
                        result.iloc[i, result.columns.get_loc('ema_touch_dist')] = dist_high_to_ema
                        result.iloc[i, result.columns.get_loc('ema_touch_slope')] = slope_val

                        touches_found += 1
                        touches_by_tf[tf] += 1

    print(f"    Multi-TF EMA touches detected: {touches_found:,}")
    for tf, count in touches_by_tf.items():
        if count > 0:
            print(f"      {tf}: {count:,} touches")

    return result


def detect_ema_touch_simple(
    df: pd.DataFrame,
    ema_col: str,
    slope_col: str,
    atr_col: str,
    touch_threshold_atr: float = 0.3,
    min_slope_norm: float = 0.03,
) -> pd.Series:
    """
    Simplified vectorized EMA touch detection for a single timeframe.

    Returns a Series of direction: 1 (long setup), -1 (short setup), or 0 (no touch)
    """
    n = len(df)
    directions = np.zeros(n)

    ema = df[ema_col].values
    slope = df[slope_col].values
    atr = df[atr_col].values
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values

    for i in range(n):
        if pd.isna(ema[i]) or pd.isna(slope[i]) or pd.isna(atr[i]) or atr[i] <= 0:
            continue

        if slope[i] > min_slope_norm:
            # LONG: check if low touched EMA
            dist = (low[i] - ema[i]) / atr[i]
            if -touch_threshold_atr <= dist <= touch_threshold_atr:
                if close[i] >= ema[i]:
                    directions[i] = 1

        elif slope[i] < -min_slope_norm:
            # SHORT: check if high touched EMA
            dist = (high[i] - ema[i]) / atr[i]
            if -touch_threshold_atr <= dist <= touch_threshold_atr:
                if close[i] <= ema[i]:
                    directions[i] = -1

    return pd.Series(directions, index=df.index)


def label_pullback_outcomes(
    df: pd.DataFrame,
    pullback_mask: pd.Series,
    config: LabelConfig,
    base_tf: str,
    use_ema_touch_direction: bool = True,
) -> pd.DataFrame:
    """
    Label pullback zones with their outcomes using SEQUENTIAL TP/SL logic.

    This matches actual trade execution: a trade is successful only if
    take-profit is hit BEFORE stop-loss, walking bar-by-bar through price action.

    Args:
        df: DataFrame with features
        pullback_mask: Boolean mask of pullback zones
        config: LabelConfig
        base_tf: Base timeframe prefix
        use_ema_touch_direction: If True, use 'ema_touch_direction' column for trend_dir

    Returns:
        DataFrame with pullback outcome labels
    """
    result = df.copy()
    n = len(result)
    window = config.entry_forward_window

    atr_col = f'{base_tf}_atr'
    alignment_col = f'{base_tf}_ema_{config.pullback_ema}_slope_norm'

    # Initialize label columns - core labels
    result['pullback_success'] = np.nan
    result['pullback_mfe'] = np.nan
    result['pullback_mae'] = np.nan
    result['pullback_rr'] = np.nan

    # Sequential execution labels
    result['pullback_hit_tp_first'] = np.nan
    result['pullback_bars_to_exit'] = np.nan
    result['pullback_exit_type'] = None

    # Multi-tier quality labels
    result['pullback_tier'] = np.nan

    # Time-to-target / early momentum
    result['pullback_early_momentum'] = np.nan
    result['pullback_immediate_range'] = np.nan

    # Bounce reaction features
    result['bounce_bar_body_ratio'] = np.nan
    result['bounce_bar_wick_ratio'] = np.nan
    result['bounce_bar_direction'] = np.nan
    result['bounce_volume_ratio'] = np.nan

    pullback_indices = result.index[pullback_mask].tolist()

    volume_ma_col = f'{base_tf}_volume_sma' if f'{base_tf}_volume_sma' in result.columns else None

    # Check if we have ema_touch_direction column
    has_touch_direction = 'ema_touch_direction' in result.columns and use_ema_touch_direction

    for idx in pullback_indices:
        i = result.index.get_loc(idx)

        if i + window >= n:
            continue

        entry_price = result['close'].iloc[i]
        current_atr = result[atr_col].iloc[i]

        # Get direction from either ema_touch_direction or slope
        if has_touch_direction:
            trend_dir = result['ema_touch_direction'].iloc[i]
        else:
            trend_dir = np.sign(result[alignment_col].iloc[i])

        if current_atr <= 0 or pd.isna(current_atr) or trend_dir == 0:
            continue

        trend_dir = int(trend_dir)

        # BOUNCE REACTION FEATURES
        bar_open = result['open'].iloc[i]
        bar_high = result['high'].iloc[i]
        bar_low = result['low'].iloc[i]
        bar_close = result['close'].iloc[i]
        bar_range = bar_high - bar_low

        if bar_range > 0:
            body_size = abs(bar_close - bar_open)
            result.loc[idx, 'bounce_bar_body_ratio'] = body_size / bar_range

            if trend_dir > 0:
                favorable_wick = min(bar_open, bar_close) - bar_low
            else:
                favorable_wick = bar_high - max(bar_open, bar_close)
            result.loc[idx, 'bounce_bar_wick_ratio'] = favorable_wick / bar_range

        result.loc[idx, 'bounce_bar_direction'] = 1 if bar_close > bar_open else -1

        if volume_ma_col and volume_ma_col in result.columns:
            vol_ma = result[volume_ma_col].iloc[i]
            if vol_ma > 0:
                result.loc[idx, 'bounce_volume_ratio'] = result['volume'].iloc[i] / vol_ma

        # SEQUENTIAL TP/SL SIMULATION
        stop_dist = config.stop_atr_multiple * current_atr
        tp_dist = config.target_rr * stop_dist

        if trend_dir > 0:
            stop_level = entry_price - stop_dist
            tp_level = entry_price + tp_dist
        else:
            stop_level = entry_price + stop_dist
            tp_level = entry_price - tp_dist

        hit_tp = False
        hit_sl = False
        bars_to_exit = None
        exit_type = 'timeout'
        mfe = 0.0
        mae = 0.0

        for j in range(i + 1, min(i + window + 1, n)):
            future_high = result['high'].iloc[j]
            future_low = result['low'].iloc[j]
            future_open = result['open'].iloc[j]

            if trend_dir > 0:
                mfe = max(mfe, (future_high - entry_price) / current_atr)
                mae = max(mae, (entry_price - future_low) / current_atr)

                # Check if BOTH levels are hit on same bar
                sl_hit = future_low <= stop_level
                tp_hit = future_high >= tp_level

                if sl_hit and tp_hit:
                    # BOTH hit on same bar - use open price to infer which was first
                    # If open is closer to stop, assume SL hit first
                    # If open is closer to target, assume TP hit first
                    dist_to_sl = abs(future_open - stop_level)
                    dist_to_tp = abs(future_open - tp_level)

                    if dist_to_tp <= dist_to_sl:
                        # Open closer to TP or equidistant -> assume TP hit first
                        hit_tp = True
                        bars_to_exit = j - i
                        exit_type = 'tp'
                    else:
                        # Open closer to SL -> assume SL hit first
                        hit_sl = True
                        bars_to_exit = j - i
                        exit_type = 'sl'
                    break
                elif sl_hit:
                    hit_sl = True
                    bars_to_exit = j - i
                    exit_type = 'sl'
                    break
                elif tp_hit:
                    hit_tp = True
                    bars_to_exit = j - i
                    exit_type = 'tp'
                    break
            else:
                mfe = max(mfe, (entry_price - future_low) / current_atr)
                mae = max(mae, (future_high - entry_price) / current_atr)

                # Check if BOTH levels are hit on same bar
                sl_hit = future_high >= stop_level
                tp_hit = future_low <= tp_level

                if sl_hit and tp_hit:
                    # BOTH hit on same bar - use open price to infer which was first
                    dist_to_sl = abs(future_open - stop_level)
                    dist_to_tp = abs(future_open - tp_level)

                    if dist_to_tp <= dist_to_sl:
                        hit_tp = True
                        bars_to_exit = j - i
                        exit_type = 'tp'
                    else:
                        hit_sl = True
                        bars_to_exit = j - i
                        exit_type = 'sl'
                    break
                elif sl_hit:
                    hit_sl = True
                    bars_to_exit = j - i
                    exit_type = 'sl'
                    break
                elif tp_hit:
                    hit_tp = True
                    bars_to_exit = j - i
                    exit_type = 'tp'
                    break

        # EARLY MOMENTUM
        early_bars = min(3, window)
        if i + early_bars < n:
            early_close = result['close'].iloc[i + early_bars]
            early_momentum = (early_close - entry_price) / current_atr
            result.loc[idx, 'pullback_early_momentum'] = early_momentum * trend_dir

            early_highs = result['high'].iloc[i+1:i+early_bars+1]
            early_lows = result['low'].iloc[i+1:i+early_bars+1]
            if len(early_highs) > 0:
                immediate_range = (early_highs.max() - early_lows.min()) / current_atr
                result.loc[idx, 'pullback_immediate_range'] = immediate_range

        # SUCCESS LABEL
        success = 1 if hit_tp else 0
        rr = mfe / max(mae, 0.1)

        result.loc[idx, 'pullback_success'] = success
        result.loc[idx, 'pullback_mfe'] = mfe
        result.loc[idx, 'pullback_mae'] = mae
        result.loc[idx, 'pullback_rr'] = rr
        result.loc[idx, 'pullback_hit_tp_first'] = int(hit_tp)
        result.loc[idx, 'pullback_bars_to_exit'] = bars_to_exit
        result.loc[idx, 'pullback_exit_type'] = exit_type

        # MULTI-TIER QUALITY LABEL
        if hit_sl or mfe < 0.5:
            tier = 0
        elif mfe < 1.0:
            tier = 1
        elif mfe < 2.0:
            tier = 2
        else:
            tier = 3

        result.loc[idx, 'pullback_tier'] = tier

    return result


def label_regime(
    df: pd.DataFrame,
    base_tf: str,
    feature_config: FeatureConfig,
    lookback: int = 20
) -> pd.DataFrame:
    """
    Label market regime for each bar.
    
    Regimes:
        0: Ranging/Choppy
        1: Trending Up
        2: Trending Down
        3: High Volatility (no direction)
    
    Args:
        df: DataFrame with features
        base_tf: Base timeframe prefix
        lookback: Bars to look back for regime detection
        
    Returns:
        DataFrame with 'regime' column
    """
    result = df.copy()
    
    adx_col = f'{base_tf}_adx'
    ema_period = feature_config.ema_periods[0] if feature_config and feature_config.ema_periods else 9
    alignment_col = f'{base_tf}_ema_{ema_period}_slope_norm'
    atr_pct_col = f'{base_tf}_atr_percentile'
    
    # Default to ranging
    result['regime'] = 0
    
    # Trending up: ADX > 25 and bullish alignment
    trending_up = (result[adx_col] > 25) & (result[alignment_col] > 0.05)
    result.loc[trending_up, 'regime'] = 1
    
    # Trending down: ADX > 25 and bearish alignment
    trending_down = (result[adx_col] > 25) & (result[alignment_col] < -0.05)
    result.loc[trending_down, 'regime'] = 2
    
    # High volatility: high ATR but no direction
    if atr_pct_col in result.columns:
        high_vol = (result[atr_pct_col] > 0.8) & (abs(result[alignment_col]) < 0.05)
        result.loc[high_vol, 'regime'] = 3
    
    return result


def create_training_dataset(
    df: pd.DataFrame,
    config: LabelConfig,
    feature_config: FeatureConfig,
    base_tf: str,
    use_multi_tf_touches: bool = True,
    touch_threshold_atr: float = 0.3,
    min_slope_norm: float = 0.03,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create complete training dataset with all labels.

    Args:
        df: DataFrame with all features
        config: LabelConfig
        feature_config: FeatureConfig
        base_tf: Base timeframe prefix
        use_multi_tf_touches: If True, use new multi-TF EMA touch detection
        touch_threshold_atr: Distance threshold for EMA touch (in ATR units)
        min_slope_norm: Minimum EMA slope to consider trending

    Returns:
        Tuple of (labeled DataFrame, list of feature columns)
    """
    print("Generating labels...")

    # Label trend opportunities
    print("  Labeling trend opportunities...")
    result = label_trend_opportunities(df, config, base_tf)

    # Detect pullback zones - use new multi-TF approach if enabled
    if use_multi_tf_touches:
        print("  Detecting multi-TF EMA touches...")
        result = detect_multi_tf_ema_touches(
            result, config, feature_config, base_tf,
            touch_threshold_atr=touch_threshold_atr,
            min_slope_norm=min_slope_norm,
        )
        pullback_mask = result['ema_touch_detected']
        print(f"    Total EMA touches found: {pullback_mask.sum():,}")

        # Show breakdown by timeframe
        if 'ema_touch_tf' in result.columns:
            tf_counts = result.loc[pullback_mask, 'ema_touch_tf'].value_counts()
            for tf, count in tf_counts.items():
                long_count = ((result['ema_touch_tf'] == tf) & (result['ema_touch_direction'] == 1)).sum()
                short_count = ((result['ema_touch_tf'] == tf) & (result['ema_touch_direction'] == -1)).sum()
                print(f"      {tf}: {count:,} touches (L:{long_count:,} / S:{short_count:,})")
    else:
        print("  Detecting pullback zones (legacy mode)...")
        pullback_mask = detect_pullback_zones(result, config, base_tf)
        print(f"    Found {pullback_mask.sum():,} pullback zones")

    # Label pullback outcomes
    print("  Labeling pullback outcomes...")
    result = label_pullback_outcomes(
        result, pullback_mask, config, base_tf,
        use_ema_touch_direction=use_multi_tf_touches
    )

    # Show outcome distribution
    if 'pullback_success' in result.columns:
        successes = result['pullback_success'].dropna()
        if len(successes) > 0:
            win_rate = successes.mean() * 100
            print(f"    Labeled {len(successes):,} bounce outcomes")
            print(f"    Win rate (TP before SL): {win_rate:.1f}%")

            if 'pullback_exit_type' in result.columns:
                exit_types = result['pullback_exit_type'].dropna().value_counts()
                for exit_type, count in exit_types.items():
                    print(f"      {exit_type}: {count:,}")

    # Label regime
    print("  Labeling regime...")
    result = label_regime(result, base_tf, feature_config)

    # Get feature columns
    from feature_engine import get_feature_columns
    feature_cols = get_feature_columns(result)

    # Remove rows with NaN in key columns
    print("  Cleaning data...")
    key_cols = ['trend_label', f'{base_tf}_atr', f'{base_tf}_adx']
    result = result.dropna(subset=[c for c in key_cols if c in result.columns])

    print(f"  Final dataset: {len(result):,} samples")
    print(f"  Features: {len(feature_cols)}")

    # Label distribution
    if 'trend_label' in result.columns:
        print(f"\n  Trend label distribution:")
        print(f"    Up trends:   {(result['trend_label'] == 1).sum():,}")
        print(f"    Down trends: {(result['trend_label'] == -1).sum():,}")
        print(f"    No trend:    {(result['trend_label'] == 0).sum():,}")

    return result, feature_cols


if __name__ == "__main__":
    from config import DEFAULT_CONFIG
    print("Label generation module loaded successfully")
    print(f"Trend threshold: {DEFAULT_CONFIG.labels.trend_up_threshold} ATR")
    print(f"Forward window: {DEFAULT_CONFIG.labels.trend_forward_window} bars")
