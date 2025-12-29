use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Context, Result};
use csv::ReaderBuilder;
use glob::glob;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct DataConfig {
    pub data_dir: String,
    pub file_pattern: String,
    pub lookback_days: Option<i64>,
    pub timestamp_col: String,
    pub price_col: String,
    pub size_col: String,
    pub side_col: String,
    pub tick_direction_col: String,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            data_dir: "./data".to_string(),
            file_pattern: "*.csv".to_string(),
            lookback_days: None,
            timestamp_col: "timestamp".to_string(),
            price_col: "price".to_string(),
            size_col: "size".to_string(),
            side_col: "side".to_string(),
            tick_direction_col: "tickDirection".to_string(),
        }
    }
}

fn default_timeframes() -> Vec<i64> {
    vec![60, 300, 900, 3600, 14400]
}

fn default_timeframe_names() -> Vec<String> {
    vec!["1m", "5m", "15m", "1h", "4h"]
        .into_iter()
        .map(|s| s.to_string())
        .collect()
}

fn default_ema_periods() -> Vec<i64> {
    vec![10, 21, 50, 100]
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct FeatureConfig {
    pub timeframes: Vec<i64>,
    pub timeframe_names: Vec<String>,
    pub ema_periods: Vec<i64>,
    pub rsi_period: i64,
    pub adx_period: i64,
    pub atr_period: i64,
    pub bb_period: i64,
    pub bb_std: f64,
    pub volume_ma_period: i64,
    pub swing_lookback: i64,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            timeframes: default_timeframes(),
            timeframe_names: default_timeframe_names(),
            ema_periods: default_ema_periods(),
            rsi_period: 14,
            adx_period: 14,
            atr_period: 14,
            bb_period: 20,
            bb_std: 2.0,
            volume_ma_period: 20,
            swing_lookback: 10,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct LabelConfig {
    pub trend_forward_window: i64,
    pub entry_forward_window: i64,
    pub trend_up_threshold: f64,
    pub trend_down_threshold: f64,
    pub max_adverse_for_trend: f64,
    pub target_rr: f64,
    pub stop_atr_multiple: f64,
    pub pullback_ema: i64,
    pub pullback_threshold: f64,
    pub best_threshold: f64,
    pub ev_margin_r: f64,
    pub fee_percent: f64,
    pub fee_per_trade_r: Option<f64>,
    pub use_expected_rr: bool,
    pub use_ev_gate: bool,
    pub use_calibration: bool,
    pub use_trend_gate: bool,
    pub min_trend_prob: f64,
    pub use_regime_gate: bool,
    pub min_regime_prob: f64,
    pub allow_regime_ranging: bool,
    pub allow_regime_trend_up: bool,
    pub allow_regime_trend_down: bool,
    pub allow_regime_volatile: bool,
    pub regime_align_direction: bool,
}

impl Default for LabelConfig {
    fn default() -> Self {
        Self {
            trend_forward_window: 20,
            entry_forward_window: 10,
            trend_up_threshold: 2.0,
            trend_down_threshold: 2.0,
            max_adverse_for_trend: 1.0,
            target_rr: 1.5,
            stop_atr_multiple: 1.0,
            pullback_ema: 21,
            pullback_threshold: 0.5,
            best_threshold: 0.5,
            ev_margin_r: 0.0,
            fee_percent: 0.0011,
            fee_per_trade_r: None,
            use_expected_rr: false,
            use_ev_gate: true,
            use_calibration: true,
            use_trend_gate: false,
            min_trend_prob: 0.0,
            use_regime_gate: false,
            min_regime_prob: 0.0,
            allow_regime_ranging: true,
            allow_regime_trend_up: true,
            allow_regime_trend_down: true,
            allow_regime_volatile: true,
            regime_align_direction: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct PipelineConfig {
    pub data: DataConfig,
    pub features: FeatureConfig,
    pub labels: LabelConfig,
    pub base_timeframe_idx: usize,
    pub seed: Option<i64>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            data: DataConfig::default(),
            features: FeatureConfig::default(),
            labels: LabelConfig::default(),
            base_timeframe_idx: 1,
            seed: Some(42),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Trade {
    timestamp: f64,
    price: f64,
    size: f64,
    side_num: f64,
    tick_dir_num: f64,
    value: f64,
    signed_size: f64,
    signed_value: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TradeCacheKey {
    data_dir: String,
    file_pattern: String,
    lookback_days: Option<i64>,
    timestamp_col: String,
    price_col: String,
    size_col: String,
    side_col: String,
    tick_direction_col: String,
}

#[derive(Debug, Clone)]
struct TradeCache {
    key: TradeCacheKey,
    trades: Arc<Vec<Trade>>,
}

static TRADES_CACHE: OnceLock<Mutex<Option<TradeCache>>> = OnceLock::new();
static TRADES_INDEX_CACHE: OnceLock<
    Mutex<HashMap<TradeCacheKey, HashMap<i64, Arc<HashMap<i64, Vec<f64>>>>>>,
> = OnceLock::new();

fn trades_cache() -> &'static Mutex<Option<TradeCache>> {
    TRADES_CACHE.get_or_init(|| Mutex::new(None))
}

fn trades_index_cache(
) -> &'static Mutex<HashMap<TradeCacheKey, HashMap<i64, Arc<HashMap<i64, Vec<f64>>>>>> {
    TRADES_INDEX_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn trade_cache_key(config: &DataConfig) -> TradeCacheKey {
    TradeCacheKey {
        data_dir: config.data_dir.clone(),
        file_pattern: config.file_pattern.clone(),
        lookback_days: config.lookback_days,
        timestamp_col: config.timestamp_col.clone(),
        price_col: config.price_col.clone(),
        size_col: config.size_col.clone(),
        side_col: config.side_col.clone(),
        tick_direction_col: config.tick_direction_col.clone(),
    }
}

#[derive(Debug, Clone)]
struct Bar {
    bar_time: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    value: f64,
    net_side: f64,
    net_volume: f64,
    net_value: f64,
    avg_tick_dir_sum: f64,
    trade_count: i64,
}

#[derive(Debug, Clone, Default)]
struct Frame {
    len: usize,
    f64_cols: HashMap<String, Vec<f64>>,
    i64_cols: HashMap<String, Vec<i64>>,
    bool_cols: HashMap<String, Vec<bool>>,
    str_cols: HashMap<String, Vec<Option<String>>>,
}

impl Frame {
    fn new(len: usize) -> Self {
        Self {
            len,
            f64_cols: HashMap::new(),
            i64_cols: HashMap::new(),
            bool_cols: HashMap::new(),
            str_cols: HashMap::new(),
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn set_f64(&mut self, name: &str, values: Vec<f64>) {
        self.len = values.len();
        self.f64_cols.insert(name.to_string(), values);
    }

    fn set_i64(&mut self, name: &str, values: Vec<i64>) {
        self.len = values.len();
        self.i64_cols.insert(name.to_string(), values);
    }

    fn set_bool(&mut self, name: &str, values: Vec<bool>) {
        self.len = values.len();
        self.bool_cols.insert(name.to_string(), values);
    }

    fn set_str(&mut self, name: &str, values: Vec<Option<String>>) {
        self.len = values.len();
        self.str_cols.insert(name.to_string(), values);
    }

    fn get_f64(&self, name: &str) -> Result<&Vec<f64>> {
        self.f64_cols
            .get(name)
            .with_context(|| format!("Missing f64 column: {}", name))
    }

    fn get_i64(&self, name: &str) -> Result<&Vec<i64>> {
        self.i64_cols
            .get(name)
            .with_context(|| format!("Missing i64 column: {}", name))
    }

    fn get_bool(&self, name: &str) -> Result<&Vec<bool>> {
        self.bool_cols
            .get(name)
            .with_context(|| format!("Missing bool column: {}", name))
    }

    fn get_str(&self, name: &str) -> Result<&Vec<Option<String>>> {
        self.str_cols
            .get(name)
            .with_context(|| format!("Missing string column: {}", name))
    }

    fn column_names(&self) -> Vec<String> {
        let mut out = Vec::new();
        out.extend(self.i64_cols.keys().cloned());
        out.extend(self.f64_cols.keys().cloned());
        out.extend(self.bool_cols.keys().cloned());
        out.extend(self.str_cols.keys().cloned());
        out
    }

    fn filter_rows(&self, mask: &[bool]) -> Self {
        let mut out = Frame::new(0);
        for (name, col) in &self.f64_cols {
            let mut vals = Vec::new();
            for (i, v) in col.iter().enumerate() {
                if mask[i] {
                    vals.push(*v);
                }
            }
            out.set_f64(name, vals);
        }
        for (name, col) in &self.i64_cols {
            let mut vals = Vec::new();
            for (i, v) in col.iter().enumerate() {
                if mask[i] {
                    vals.push(*v);
                }
            }
            out.set_i64(name, vals);
        }
        for (name, col) in &self.bool_cols {
            let mut vals = Vec::new();
            for (i, v) in col.iter().enumerate() {
                if mask[i] {
                    vals.push(*v);
                }
            }
            out.set_bool(name, vals);
        }
        for (name, col) in &self.str_cols {
            let mut vals = Vec::new();
            for (i, v) in col.iter().enumerate() {
                if mask[i] {
                    vals.push(v.clone());
                }
            }
            out.set_str(name, vals);
        }
        out
    }
}

fn is_nan(v: f64) -> bool {
    v.is_nan()
}

fn safe_div(numer: f64, denom: f64) -> f64 {
    if is_nan(numer) || is_nan(denom) || denom == 0.0 {
        f64::NAN
    } else {
        numer / denom
    }
}

fn sign(v: f64) -> f64 {
    if is_nan(v) {
        f64::NAN
    } else if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        0.0
    }
}
fn ffill(values: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(values.len());
    let mut last = f64::NAN;
    for v in values {
        if is_nan(*v) {
            out.push(last);
        } else {
            last = *v;
            out.push(*v);
        }
    }
    out
}

fn diff(values: &[f64], periods: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; values.len()];
    if periods == 0 {
        return out;
    }
    for i in periods..values.len() {
        let a = values[i];
        let b = values[i - periods];
        if is_nan(a) || is_nan(b) {
            out[i] = f64::NAN;
        } else {
            out[i] = a - b;
        }
    }
    out
}

fn pct_change(values: &[f64], periods: usize) -> Vec<f64> {
    let filled = ffill(values);
    let mut out = vec![f64::NAN; values.len()];
    if periods == 0 {
        return out;
    }
    for i in periods..values.len() {
        let cur = filled[i];
        let prev = filled[i - periods];
        if is_nan(cur) || is_nan(prev) {
            out[i] = f64::NAN;
        } else {
            out[i] = cur / prev - 1.0;
        }
    }
    out
}

fn cumsum_skipna(values: &[f64]) -> Vec<f64> {
    let mut out = vec![f64::NAN; values.len()];
    let mut acc = 0.0;
    let mut has_value = false;
    for (i, v) in values.iter().enumerate() {
        if is_nan(*v) {
            if has_value {
                out[i] = acc;
            } else {
                out[i] = f64::NAN;
            }
        } else {
            acc += *v;
            has_value = true;
            out[i] = acc;
        }
    }
    out
}

fn rolling_sum(values: &[f64], window: usize, min_periods: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; values.len()];
    for i in 0..values.len() {
        if i + 1 < window {
            continue;
        }
        let start = i + 1 - window;
        let mut count = 0usize;
        let mut sum = 0.0;
        for v in &values[start..=i] {
            if !is_nan(*v) {
                sum += *v;
                count += 1;
            }
        }
        if count >= min_periods {
            out[i] = sum;
        }
    }
    out
}

fn rolling_mean(values: &[f64], window: usize, min_periods: usize) -> Vec<f64> {
    let sums = rolling_sum(values, window, min_periods);
    let mut out = vec![f64::NAN; values.len()];
    for i in 0..values.len() {
        let start = if i + 1 >= window { i + 1 - window } else { 0 };
        let mut count = 0usize;
        for v in &values[start..=i] {
            if !is_nan(*v) {
                count += 1;
            }
        }
        if count >= min_periods {
            let s = sums[i];
            if !is_nan(s) {
                out[i] = s / count as f64;
            }
        }
    }
    out
}

fn rolling_max(values: &[f64], window: usize, min_periods: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; values.len()];
    for i in 0..values.len() {
        if i + 1 < window {
            continue;
        }
        let start = i + 1 - window;
        let mut count = 0usize;
        let mut max_v = f64::NAN;
        for v in &values[start..=i] {
            if !is_nan(*v) {
                if is_nan(max_v) || *v > max_v {
                    max_v = *v;
                }
                count += 1;
            }
        }
        if count >= min_periods {
            out[i] = max_v;
        }
    }
    out
}

fn rolling_min(values: &[f64], window: usize, min_periods: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; values.len()];
    for i in 0..values.len() {
        if i + 1 < window {
            continue;
        }
        let start = i + 1 - window;
        let mut count = 0usize;
        let mut min_v = f64::NAN;
        for v in &values[start..=i] {
            if !is_nan(*v) {
                if is_nan(min_v) || *v < min_v {
                    min_v = *v;
                }
                count += 1;
            }
        }
        if count >= min_periods {
            out[i] = min_v;
        }
    }
    out
}

fn rolling_std(values: &[f64], window: usize, min_periods: usize, ddof: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; values.len()];
    for i in 0..values.len() {
        if i + 1 < window {
            continue;
        }
        let start = i + 1 - window;
        let mut vals = Vec::new();
        for v in &values[start..=i] {
            if !is_nan(*v) {
                vals.push(*v);
            }
        }
        let count = vals.len();
        if count < min_periods || count <= ddof {
            continue;
        }
        let mean = vals.iter().sum::<f64>() / count as f64;
        let mut var = 0.0;
        for v in &vals {
            let d = v - mean;
            var += d * d;
        }
        var /= (count - ddof) as f64;
        out[i] = var.sqrt();
    }
    out
}

fn rolling_first(values: &[f64], window: usize, min_periods: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; values.len()];
    for i in 0..values.len() {
        let start = if i + 1 >= window { i + 1 - window } else { 0 };
        let count = i + 1 - start;
        if count < min_periods {
            continue;
        }
        out[i] = values[start];
    }
    out
}

fn rolling_rank_last(values: &[f64], window: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; values.len()];
    if window == 0 {
        return out;
    }
    for i in 0..values.len() {
        if i + 1 < window {
            continue;
        }
        let start = i + 1 - window;
        let slice = &values[start..=i];
        let mut count = 0usize;
        for v in slice {
            if !is_nan(*v) {
                count += 1;
            }
        }
        if count < window {
            continue;
        }
        let last = values[i];
        if is_nan(last) {
            continue;
        }
        let mut less = 0usize;
        let mut equal = 0usize;
        for v in slice {
            if is_nan(*v) {
                continue;
            }
            if *v < last {
                less += 1;
            } else if *v == last {
                equal += 1;
            }
        }
        let rank = less as f64 + (equal as f64 + 1.0) / 2.0;
        out[i] = rank / count as f64;
    }
    out
}
fn load_trades(config: &DataConfig) -> Result<Vec<Trade>> {
    let data_path = Path::new(&config.data_dir);
    if !data_path.exists() {
        return Err(anyhow::anyhow!("Data directory not found: {}", data_path.display()));
    }
    let pattern = data_path.join(&config.file_pattern);
    let pattern_str = pattern.to_string_lossy().to_string();
    let mut files: Vec<PathBuf> = Vec::new();
    for entry in glob(&pattern_str).context("glob failed")? {
        let path = entry?;
        files.push(path);
    }
    files.sort();
    if files.is_empty() {
        return Err(anyhow::anyhow!("No files matching '{}' in {}", config.file_pattern, data_path.display()));
    }

    let mut all_trades: Vec<Trade> = Vec::new();

    for file in files {
        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .from_path(&file)
            .with_context(|| format!("Failed to open {}", file.display()))?;
        let headers = reader
            .headers()
            .with_context(|| format!("Failed to read headers for {}", file.display()))?
            .clone();

        let mut idx_ts = None;
        let mut idx_price = None;
        let mut idx_size = None;
        let mut idx_side = None;
        let mut idx_tick = None;

        for (i, h) in headers.iter().enumerate() {
            if h == config.timestamp_col {
                idx_ts = Some(i);
            }
            if h == config.price_col {
                idx_price = Some(i);
            }
            if h == config.size_col {
                idx_size = Some(i);
            }
            if h == config.side_col {
                idx_side = Some(i);
            }
            if h == config.tick_direction_col {
                idx_tick = Some(i);
            }
        }
        let idx_ts = idx_ts.context("Missing timestamp column")?;
        let idx_price = idx_price.context("Missing price column")?;
        let idx_size = idx_size.context("Missing size column")?;
        let idx_side = idx_side.context("Missing side column")?;
        let idx_tick = idx_tick.context("Missing tick direction column")?;

        for result in reader.records() {
            let record = result?;
            let ts_raw: f64 = record
                .get(idx_ts)
                .context("Missing timestamp value")?
                .parse()
                .unwrap_or(f64::NAN);
            if is_nan(ts_raw) {
                continue;
            }
            let ts = ts_raw;
            let price: f64 = record
                .get(idx_price)
                .context("Missing price value")?
                .parse()
                .unwrap_or(f64::NAN);
            let size: f64 = record
                .get(idx_size)
                .context("Missing size value")?
                .parse()
                .unwrap_or(f64::NAN);
            let side = record.get(idx_side).unwrap_or("");
            let tick = record.get(idx_tick).unwrap_or("");

            let side_num = match side {
                "Buy" => 1.0,
                "Sell" => -1.0,
                _ => 0.0,
            };
            let tick_dir_num = match tick {
                "PlusTick" => 1.0,
                "ZeroPlusTick" => 0.5,
                "MinusTick" => -1.0,
                "ZeroMinusTick" => -0.5,
                _ => 0.0,
            };
            let value = price * size;
            let signed_size = size * side_num;
            let signed_value = value * side_num;

            all_trades.push(Trade {
                timestamp: ts,
                price,
                size,
                side_num,
                tick_dir_num,
                value,
                signed_size,
                signed_value,
            });
        }
    }

    all_trades.sort_by(|a, b| {
        a.timestamp
            .partial_cmp(&b.timestamp)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if let Some(days) = config.lookback_days {
        if days > 0 && !all_trades.is_empty() {
            let max_ts = all_trades.last().map(|t| t.timestamp).unwrap_or(0.0);
            let cutoff = max_ts - (days as f64 * 86400.0);
            all_trades.retain(|t| t.timestamp >= cutoff);
        }
    }

    Ok(all_trades)
}

pub fn clear_trades_cache() {
    if let Ok(mut guard) = trades_cache().lock() {
        *guard = None;
    }
    if let Ok(mut guard) = trades_index_cache().lock() {
        guard.clear();
    }
}

pub(crate) fn load_trades_cached(config: &DataConfig) -> Result<Arc<Vec<Trade>>> {
    let key = trade_cache_key(config);
    if let Ok(guard) = trades_cache().lock() {
        if let Some(cache) = guard.as_ref() {
            if cache.key == key {
                return Ok(cache.trades.clone());
            }
        }
    }

    let trades = load_trades(config)?;
    let arc_trades = Arc::new(trades);
    if let Ok(mut guard) = trades_cache().lock() {
        *guard = Some(TradeCache {
            key,
            trades: arc_trades.clone(),
        });
    }
    Ok(arc_trades)
}

fn build_trades_index(trades: &[Trade], timeframe_seconds: i64) -> HashMap<i64, Vec<f64>> {
    let mut out: HashMap<i64, Vec<f64>> = HashMap::new();
    if timeframe_seconds <= 0 {
        return out;
    }
    let tf = timeframe_seconds as f64;
    for trade in trades {
        if is_nan(trade.timestamp) {
            continue;
        }
        let bar_time = ((trade.timestamp / tf).floor() as i64) * timeframe_seconds;
        out.entry(bar_time).or_insert_with(Vec::new).push(trade.price);
    }
    out
}

pub(crate) fn load_trades_index_cached(
    config: &DataConfig,
    timeframe_seconds: i64,
) -> Result<Arc<HashMap<i64, Vec<f64>>>> {
    if timeframe_seconds <= 0 {
        return Err(anyhow::anyhow!("Invalid timeframe_seconds: {}", timeframe_seconds));
    }
    let key = trade_cache_key(config);
    if let Ok(guard) = trades_index_cache().lock() {
        if let Some(tf_map) = guard.get(&key) {
            if let Some(index) = tf_map.get(&timeframe_seconds) {
                return Ok(index.clone());
            }
        }
    }

    let trades = load_trades_cached(config)?;
    let index = build_trades_index(&trades, timeframe_seconds);
    let arc_index = Arc::new(index);
    if let Ok(mut guard) = trades_index_cache().lock() {
        let tf_map = guard.entry(key).or_insert_with(HashMap::new);
        tf_map.insert(timeframe_seconds, arc_index.clone());
    }
    Ok(arc_index)
}

pub(crate) fn check_exit_in_bar(
    config: &DataConfig,
    timeframe_seconds: i64,
    bar_time: i64,
    direction: i64,
    stop: f64,
    target: f64,
) -> Result<(i32, f64)> {
    if is_nan(stop) || is_nan(target) {
        return Ok((0, 0.0));
    }
    let index = load_trades_index_cached(config, timeframe_seconds)?;
    let trades_in_bar = match index.get(&bar_time) {
        Some(trades) => trades,
        None => return Ok((0, 0.0)),
    };
    if direction == 0 {
        return Ok((0, 0.0));
    }
    if direction > 0 {
        for price in trades_in_bar {
            if *price <= stop {
                return Ok((1, stop));
            }
            if *price >= target {
                return Ok((2, target));
            }
        }
    } else {
        for price in trades_in_bar {
            if *price >= stop {
                return Ok((1, stop));
            }
            if *price <= target {
                return Ok((2, target));
            }
        }
    }
    Ok((0, 0.0))
}

fn aggregate_to_bars(trades: &[Trade], timeframe_seconds: i64) -> Frame {
    let mut bars: Vec<Bar> = Vec::new();
    let mut current: Option<Bar> = None;

    for t in trades {
        let bar_time = (t.timestamp / timeframe_seconds as f64).floor() as i64 * timeframe_seconds;
        if current.is_none() || current.as_ref().unwrap().bar_time != bar_time {
            if let Some(bar) = current.take() {
                bars.push(bar);
            }
            current = Some(Bar {
                bar_time,
                open: t.price,
                high: t.price,
                low: t.price,
                close: t.price,
                volume: t.size,
                value: t.value,
                net_side: t.side_num,
                net_volume: t.signed_size,
                net_value: t.signed_value,
                avg_tick_dir_sum: t.tick_dir_num,
                trade_count: 1,
            });
        } else if let Some(ref mut bar) = current {
            if t.price > bar.high {
                bar.high = t.price;
            }
            if t.price < bar.low {
                bar.low = t.price;
            }
            bar.close = t.price;
            bar.volume += t.size;
            bar.value += t.value;
            bar.net_side += t.side_num;
            bar.net_volume += t.signed_size;
            bar.net_value += t.signed_value;
            bar.avg_tick_dir_sum += t.tick_dir_num;
            bar.trade_count += 1;
        }
    }
    if let Some(bar) = current.take() {
        bars.push(bar);
    }

    let mut frame = Frame::new(bars.len());
    let mut bar_time = Vec::with_capacity(bars.len());
    let mut open = Vec::with_capacity(bars.len());
    let mut high = Vec::with_capacity(bars.len());
    let mut low = Vec::with_capacity(bars.len());
    let mut close = Vec::with_capacity(bars.len());
    let mut volume = Vec::with_capacity(bars.len());
    let mut value = Vec::with_capacity(bars.len());
    let mut net_side = Vec::with_capacity(bars.len());
    let mut net_volume = Vec::with_capacity(bars.len());
    let mut net_value = Vec::with_capacity(bars.len());
    let mut avg_tick_dir = Vec::with_capacity(bars.len());
    let mut trade_count = Vec::with_capacity(bars.len());

    for bar in &bars {
        bar_time.push(bar.bar_time);
        open.push(bar.open);
        high.push(bar.high);
        low.push(bar.low);
        close.push(bar.close);
        volume.push(bar.volume);
        value.push(bar.value);
        net_side.push(bar.net_side);
        net_volume.push(bar.net_volume);
        net_value.push(bar.net_value);
        avg_tick_dir.push(bar.avg_tick_dir_sum / bar.trade_count as f64);
        trade_count.push(bar.trade_count as f64);
    }

    frame.set_i64("bar_time", bar_time);
    frame.set_i64("datetime", frame.get_i64("bar_time").unwrap().clone());
    frame.set_f64("open", open);
    frame.set_f64("high", high);
    frame.set_f64("low", low);
    frame.set_f64("close", close);
    frame.set_f64("volume", volume);
    frame.set_f64("value", value);
    frame.set_f64("net_side", net_side);
    frame.set_f64("net_volume", net_volume);
    frame.set_f64("net_value", net_value);
    frame.set_f64("avg_tick_dir", avg_tick_dir);
    frame.set_f64("trade_count", trade_count);

    let volume = frame.get_f64("volume").unwrap();
    let net_volume = frame.get_f64("net_volume").unwrap();
    let value = frame.get_f64("value").unwrap();
    let trade_count = frame.get_f64("trade_count").unwrap();

    let mut buy_volume = Vec::with_capacity(frame.len());
    let mut sell_volume = Vec::with_capacity(frame.len());
    let mut buy_sell_imbalance = Vec::with_capacity(frame.len());
    let mut vwap = Vec::with_capacity(frame.len());
    let mut trade_intensity = Vec::with_capacity(frame.len());
    let mut avg_trade_size = Vec::with_capacity(frame.len());

    for i in 0..frame.len() {
        let vol = volume[i];
        let net_vol = net_volume[i];
        buy_volume.push((vol + net_vol) / 2.0);
        sell_volume.push((vol - net_vol) / 2.0);
        buy_sell_imbalance.push(safe_div(net_vol, vol));
        vwap.push(safe_div(value[i], vol));
        trade_intensity.push(trade_count[i] / timeframe_seconds as f64);
        avg_trade_size.push(safe_div(vol, trade_count[i]));
    }

    frame.set_f64("buy_volume", buy_volume);
    frame.set_f64("sell_volume", sell_volume);
    frame.set_f64("buy_sell_imbalance", buy_sell_imbalance);
    frame.set_f64("vwap", vwap);
    frame.set_f64("trade_intensity", trade_intensity);
    frame.set_f64("avg_trade_size", avg_trade_size);

    for name in frame.f64_cols.clone().keys() {
        let vals = frame.get_f64(name).unwrap();
        let filled = ffill(vals);
        frame.set_f64(name, filled);
    }

    frame
}

fn create_multi_timeframe_bars(
    trades: &[Trade],
    timeframes_seconds: &[i64],
    timeframe_names: &[String],
) -> HashMap<String, Frame> {
    let mut out = HashMap::new();
    for (tf_seconds, tf_name) in timeframes_seconds.iter().zip(timeframe_names.iter()) {
        let bars = aggregate_to_bars(trades, *tf_seconds);
        out.insert(tf_name.clone(), bars);
    }
    out
}
fn ema(values: &[f64], period: i64) -> Vec<f64> {
    let mut out = vec![f64::NAN; values.len()];
    if period <= 0 {
        return out;
    }
    let span = period as f64;
    let alpha = 2.0 / (span + 1.0);
    for i in 0..values.len() {
        let v = values[i];
        if i == 0 {
            out[i] = v;
            continue;
        }
        let prev = out[i - 1];
        if is_nan(v) {
            out[i] = prev;
        } else if is_nan(prev) {
            out[i] = v;
        } else {
            out[i] = alpha * v + (1.0 - alpha) * prev;
        }
    }
    out
}

fn sma(values: &[f64], period: i64) -> Vec<f64> {
    let window = period.max(1) as usize;
    rolling_mean(values, window, window)
}

fn rsi(values: &[f64], period: i64) -> Vec<f64> {
    let diff_vals = diff(values, 1);
    let mut gain = Vec::with_capacity(values.len());
    let mut loss = Vec::with_capacity(values.len());
    for v in &diff_vals {
        if is_nan(*v) {
            gain.push(f64::NAN);
            loss.push(f64::NAN);
        } else if *v > 0.0 {
            gain.push(*v);
            loss.push(0.0);
        } else {
            gain.push(0.0);
            loss.push(-*v);
        }
    }
    let window = period.max(1) as usize;
    let avg_gain = rolling_mean(&gain, window, window);
    let avg_loss = rolling_mean(&loss, window, window);
    let mut out = vec![f64::NAN; values.len()];
    for i in 0..values.len() {
        let g = avg_gain[i];
        let l = avg_loss[i];
        if is_nan(g) || is_nan(l) || l == 0.0 {
            out[i] = f64::NAN;
        } else {
            let rs = g / l;
            out[i] = 100.0 - (100.0 / (1.0 + rs));
        }
    }
    out
}

fn atr(high: &[f64], low: &[f64], close: &[f64], period: i64) -> Vec<f64> {
    let mut tr = vec![f64::NAN; high.len()];
    for i in 0..high.len() {
        let prev_close = if i == 0 { f64::NAN } else { close[i - 1] };
        let tr1 = high[i] - low[i];
        let tr2 = if is_nan(prev_close) { f64::NAN } else { (high[i] - prev_close).abs() };
        let tr3 = if is_nan(prev_close) { f64::NAN } else { (low[i] - prev_close).abs() };
        let mut max_v = tr1;
        if !is_nan(tr2) && (is_nan(max_v) || tr2 > max_v) {
            max_v = tr2;
        }
        if !is_nan(tr3) && (is_nan(max_v) || tr3 > max_v) {
            max_v = tr3;
        }
        tr[i] = max_v;
    }
    let window = period.max(1) as usize;
    rolling_mean(&tr, window, window)
}

fn adx(high: &[f64], low: &[f64], close: &[f64], period: i64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut tr = vec![f64::NAN; high.len()];
    let mut plus_dm = vec![0.0; high.len()];
    let mut minus_dm = vec![0.0; high.len()];

    for i in 0..high.len() {
        let prev_close = if i == 0 { f64::NAN } else { close[i - 1] };
        let tr1 = high[i] - low[i];
        let tr2 = if is_nan(prev_close) { f64::NAN } else { (high[i] - prev_close).abs() };
        let tr3 = if is_nan(prev_close) { f64::NAN } else { (low[i] - prev_close).abs() };
        let mut max_v = tr1;
        if !is_nan(tr2) && (is_nan(max_v) || tr2 > max_v) {
            max_v = tr2;
        }
        if !is_nan(tr3) && (is_nan(max_v) || tr3 > max_v) {
            max_v = tr3;
        }
        tr[i] = max_v;

        if i > 0 {
            let up_move = high[i] - high[i - 1];
            let down_move = low[i - 1] - low[i];
            if up_move > down_move && up_move > 0.0 {
                plus_dm[i] = up_move;
            }
            if down_move > up_move && down_move > 0.0 {
                minus_dm[i] = down_move;
            }
        }
    }

    let window = period.max(1) as usize;
    let atr_smooth = rolling_sum(&tr, window, window);
    let plus_dm_smooth = rolling_sum(&plus_dm, window, window);
    let minus_dm_smooth = rolling_sum(&minus_dm, window, window);

    let mut plus_di = vec![f64::NAN; high.len()];
    let mut minus_di = vec![f64::NAN; high.len()];
    let mut dx = vec![f64::NAN; high.len()];

    for i in 0..high.len() {
        let atr_v = atr_smooth[i];
        if is_nan(atr_v) || atr_v == 0.0 {
            continue;
        }
        plus_di[i] = 100.0 * plus_dm_smooth[i] / atr_v;
        minus_di[i] = 100.0 * minus_dm_smooth[i] / atr_v;
        let di_sum = plus_di[i] + minus_di[i];
        let di_diff = (plus_di[i] - minus_di[i]).abs();
        if di_sum != 0.0 {
            dx[i] = 100.0 * di_diff / di_sum;
        }
    }

    let adx_vals = rolling_mean(&dx, window, window);
    (adx_vals, plus_di, minus_di)
}

fn bollinger_bands(
    close: &[f64],
    period: i64,
    std_mult: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let window = period.max(1) as usize;
    let mid = rolling_mean(close, window, window);
    let std = rolling_std(close, window, window, 1);
    let mut upper = vec![f64::NAN; close.len()];
    let mut lower = vec![f64::NAN; close.len()];
    let mut width = vec![f64::NAN; close.len()];
    let mut position = vec![f64::NAN; close.len()];
    for i in 0..close.len() {
        if is_nan(mid[i]) || is_nan(std[i]) {
            continue;
        }
        upper[i] = mid[i] + std_mult * std[i];
        lower[i] = mid[i] - std_mult * std[i];
        width[i] = safe_div(upper[i] - lower[i], mid[i]);
        position[i] = safe_div(close[i] - lower[i], upper[i] - lower[i]);
    }
    (mid, upper, lower, width, position)
}

fn obv(close: &[f64], volume: &[f64]) -> Vec<f64> {
    let diff_vals = diff(close, 1);
    let mut direction = Vec::with_capacity(close.len());
    for v in &diff_vals {
        direction.push(sign(*v));
    }
    let mut signed = Vec::with_capacity(close.len());
    for i in 0..close.len() {
        let dir = direction[i];
        let vol = volume[i];
        if is_nan(dir) || is_nan(vol) {
            signed.push(f64::NAN);
        } else {
            signed.push(dir * vol);
        }
    }
    cumsum_skipna(&signed)
}

fn detect_swing_highs(high: &[f64], lookback: usize) -> Vec<bool> {
    let mut swing = vec![false; high.len()];
    if high.len() == 0 || lookback == 0 {
        return swing;
    }
    for i in lookback..(high.len().saturating_sub(lookback)) {
        let start = i - lookback;
        let end = i + lookback;
        let mut max_v = f64::NAN;
        for v in &high[start..=end] {
            if !is_nan(*v) && (is_nan(max_v) || *v > max_v) {
                max_v = *v;
            }
        }
        if !is_nan(max_v) && high[i] == max_v {
            swing[i] = true;
        }
    }
    swing
}

fn detect_swing_lows(low: &[f64], lookback: usize) -> Vec<bool> {
    let mut swing = vec![false; low.len()];
    if low.len() == 0 || lookback == 0 {
        return swing;
    }
    for i in lookback..(low.len().saturating_sub(lookback)) {
        let start = i - lookback;
        let end = i + lookback;
        let mut min_v = f64::NAN;
        for v in &low[start..=end] {
            if !is_nan(*v) && (is_nan(min_v) || *v < min_v) {
                min_v = *v;
            }
        }
        if !is_nan(min_v) && low[i] == min_v {
            swing[i] = true;
        }
    }
    swing
}

fn get_recent_swing_high(
    high: &[f64],
    swing_highs: &[bool],
    current_idx: usize,
    lookback: usize,
    confirmation_bars: usize,
) -> Option<f64> {
    let start_idx = current_idx.saturating_sub(lookback);
    let end_idx = current_idx.saturating_sub(confirmation_bars) + 1;
    if end_idx <= start_idx {
        return None;
    }
    let mut last_idx: Option<usize> = None;
    for i in start_idx..end_idx {
        if swing_highs[i] {
            last_idx = Some(i);
        }
    }
    last_idx.map(|idx| high[idx])
}

fn get_recent_swing_low(
    low: &[f64],
    swing_lows: &[bool],
    current_idx: usize,
    lookback: usize,
    confirmation_bars: usize,
) -> Option<f64> {
    let start_idx = current_idx.saturating_sub(lookback);
    let end_idx = current_idx.saturating_sub(confirmation_bars) + 1;
    if end_idx <= start_idx {
        return None;
    }
    let mut last_idx: Option<usize> = None;
    for i in start_idx..end_idx {
        if swing_lows[i] {
            last_idx = Some(i);
        }
    }
    last_idx.map(|idx| low[idx])
}
fn calculate_features_for_timeframe(bars: &Frame, config: &FeatureConfig) -> Result<Frame> {
    let mut df = bars.clone();
    let close = df.get_f64("close")?.clone();
    let high = df.get_f64("high")?.clone();
    let low = df.get_f64("low")?.clone();
    let open = df.get_f64("open")?.clone();
    let volume = df.get_f64("volume")?.clone();

    for period in &config.ema_periods {
        let ema_vals = ema(&close, *period);
        let slope = diff(&ema_vals, 1);
        df.set_f64(&format!("ema_{}", period), ema_vals.clone());
        df.set_f64(&format!("ema_{}_slope", period), slope);
    }

    let atr_vals = atr(&high, &low, &close, config.atr_period);
    df.set_f64("atr", atr_vals.clone());

    for period in &config.ema_periods {
        let ema_vals = df.get_f64(&format!("ema_{}", period))?.clone();
        let ema_slope = df.get_f64(&format!("ema_{}_slope", period))?.clone();
        let mut price_vs = Vec::with_capacity(close.len());
        let mut slope_norm = Vec::with_capacity(close.len());
        for i in 0..close.len() {
            price_vs.push(safe_div(close[i] - ema_vals[i], atr_vals[i]));
            slope_norm.push(safe_div(ema_slope[i], atr_vals[i]));
        }
        df.set_f64(&format!("price_vs_ema_{}", period), price_vs);
        df.set_f64(&format!("ema_{}_slope_norm", period), slope_norm);
    }

    if config.ema_periods.len() >= 3 {
        let mut periods = config.ema_periods.clone();
        periods.sort();
        let mut alignment = vec![0.0; close.len()];
        for i in 0..(periods.len() - 1) {
            let a = df.get_f64(&format!("ema_{}", periods[i]))?;
            let b = df.get_f64(&format!("ema_{}", periods[i + 1]))?;
            for j in 0..close.len() {
                if !is_nan(a[j]) && !is_nan(b[j]) && a[j] > b[j] {
                    alignment[j] += 1.0;
                }
            }
        }
        let max_score = (periods.len() - 1) as f64;
        for j in 0..alignment.len() {
            alignment[j] = (alignment[j] - max_score / 2.0) / (max_score / 2.0);
        }
        df.set_f64("ema_alignment", alignment);
    }

    let rsi_vals = rsi(&close, config.rsi_period);
    let rsi_slope = diff(&rsi_vals, 3);
    df.set_f64("rsi", rsi_vals);
    df.set_f64("rsi_slope", rsi_slope);

    let (adx_vals, plus_di, minus_di) = adx(&high, &low, &close, config.adx_period);
    let di_diff = plus_di
        .iter()
        .zip(minus_di.iter())
        .map(|(a, b)| if is_nan(*a) || is_nan(*b) { f64::NAN } else { a - b })
        .collect::<Vec<f64>>();
    let adx_slope = diff(&adx_vals, 3);
    df.set_f64("adx", adx_vals);
    df.set_f64("plus_di", plus_di);
    df.set_f64("minus_di", minus_di);
    df.set_f64("di_diff", di_diff);
    df.set_f64("adx_slope", adx_slope);

    let (_mid, _upper, _lower, bb_width, bb_position) =
        bollinger_bands(&close, config.bb_period, config.bb_std);
    df.set_f64("bb_width", bb_width);
    df.set_f64("bb_position", bb_position);

    let atr_percentile = rolling_rank_last(&atr_vals, 100);
    df.set_f64("atr_percentile", atr_percentile);

    let volume_sma = sma(&volume, config.volume_ma_period);
    let mut volume_ratio = Vec::with_capacity(volume.len());
    for i in 0..volume.len() {
        volume_ratio.push(safe_div(volume[i], volume_sma[i]));
    }
    df.set_f64("volume_sma", volume_sma.clone());
    df.set_f64("volume_ratio", volume_ratio);

    let obv_vals = obv(&close, &volume);
    let obv_slope = diff(&obv_vals, 5);
    let mut obv_slope_norm = Vec::with_capacity(obv_vals.len());
    for i in 0..obv_vals.len() {
        obv_slope_norm.push(safe_div(obv_slope[i], volume_sma[i]));
    }
    df.set_f64("obv", obv_vals);
    df.set_f64("obv_slope", obv_slope_norm);

    let swing_high = detect_swing_highs(&high, config.swing_lookback as usize);
    let swing_low = detect_swing_lows(&low, config.swing_lookback as usize);
    let swing_high_f: Vec<f64> = swing_high.iter().map(|v| if *v { 1.0 } else { 0.0 }).collect();
    let swing_low_f: Vec<f64> = swing_low.iter().map(|v| if *v { 1.0 } else { 0.0 }).collect();
    df.set_f64("swing_high", swing_high_f);
    df.set_f64("swing_low", swing_low_f);

    let mut dist_from_high = vec![f64::NAN; close.len()];
    let mut dist_from_low = vec![f64::NAN; close.len()];
    for i in 0..close.len() {
        let confirm = config.swing_lookback.max(0) as usize;
        let recent_high = get_recent_swing_high(&high, &swing_high, i, 50, confirm);
        let recent_low = get_recent_swing_low(&low, &swing_low, i, 50, confirm);
        if let Some(rh) = recent_high {
            if !is_nan(atr_vals[i]) && atr_vals[i] > 0.0 {
                dist_from_high[i] = (rh - close[i]) / atr_vals[i];
            }
        }
        if let Some(rl) = recent_low {
            if !is_nan(atr_vals[i]) && atr_vals[i] > 0.0 {
                dist_from_low[i] = (close[i] - rl) / atr_vals[i];
            }
        }
    }
    df.set_f64("dist_from_high", dist_from_high);
    df.set_f64("dist_from_low", dist_from_low);

    if df.f64_cols.contains_key("buy_sell_imbalance") {
        let imb = df.get_f64("buy_sell_imbalance")?.clone();
        let imb_ma = sma(&imb, 10);
        let imb_slope = diff(&imb, 3);
        df.set_f64("imbalance_ma", imb_ma);
        df.set_f64("imbalance_slope", imb_slope);
    }

    if df.f64_cols.contains_key("trade_intensity") {
        let intensity = df.get_f64("trade_intensity")?.clone();
        let intensity_ma = sma(&intensity, 20);
        let mut ratio = Vec::with_capacity(intensity.len());
        for i in 0..intensity.len() {
            ratio.push(safe_div(intensity[i], intensity_ma[i]));
        }
        df.set_f64("intensity_ma", intensity_ma);
        df.set_f64("intensity_ratio", ratio);
    }

    if df.f64_cols.contains_key("avg_trade_size") {
        let size = df.get_f64("avg_trade_size")?.clone();
        let size_ma = sma(&size, 20);
        let mut ratio = Vec::with_capacity(size.len());
        for i in 0..size.len() {
            ratio.push(safe_div(size[i], size_ma[i]));
        }
        df.set_f64("size_ma", size_ma);
        df.set_f64("size_ratio", ratio);
    }

    let returns = pct_change(&close, 1);
    let returns_vol = rolling_std(&returns, 20, 20, 1);
    let momentum_5 = pct_change(&close, 5);
    let momentum_10 = pct_change(&close, 10);
    let momentum_20 = pct_change(&close, 20);
    df.set_f64("returns", returns);
    df.set_f64("returns_volatility", returns_vol);
    df.set_f64("momentum_5", momentum_5);
    df.set_f64("momentum_10", momentum_10);
    df.set_f64("momentum_20", momentum_20);

    let mut body_size = Vec::with_capacity(close.len());
    let mut upper_wick = Vec::with_capacity(close.len());
    let mut lower_wick = Vec::with_capacity(close.len());
    let mut candle_direction = Vec::with_capacity(close.len());
    for i in 0..close.len() {
        body_size.push(safe_div((close[i] - open[i]).abs(), atr_vals[i]));
        let upper = high[i] - nanmax(open[i], close[i]);
        let lower = nanmin(open[i], close[i]) - low[i];
        upper_wick.push(safe_div(upper, atr_vals[i]));
        lower_wick.push(safe_div(lower, atr_vals[i]));
        candle_direction.push(sign(close[i] - open[i]));
    }
    df.set_f64("body_size", body_size);
    df.set_f64("upper_wick", upper_wick);
    df.set_f64("lower_wick", lower_wick);
    df.set_f64("candle_direction", candle_direction);

    Ok(df)
}

fn calculate_cross_tf_features(df: &mut Frame, config: &FeatureConfig) -> Result<()> {
    let tf_names = &config.timeframe_names;
    let mut alignment_cols = Vec::new();
    let mut adx_cols = Vec::new();
    let mut rsi_cols = Vec::new();

    for tf in tf_names {
        let align = format!("{}_ema_alignment", tf);
        if df.f64_cols.contains_key(&align) {
            alignment_cols.push(align);
        }
        let adx = format!("{}_adx", tf);
        if df.f64_cols.contains_key(&adx) {
            adx_cols.push(adx);
        }
        let rsi = format!("{}_rsi", tf);
        if df.f64_cols.contains_key(&rsi) {
            rsi_cols.push(rsi);
        }
    }

    if !alignment_cols.is_empty() {
        let mut vals = vec![0.0; df.len()];
        let mut counts = vec![0usize; df.len()];
        for col in &alignment_cols {
            let series = df.get_f64(col)?;
            for i in 0..df.len() {
                if !is_nan(series[i]) {
                    vals[i] += series[i];
                    counts[i] += 1;
                }
            }
        }
        for i in 0..df.len() {
            if counts[i] > 0 {
                vals[i] /= counts[i] as f64;
            } else {
                vals[i] = f64::NAN;
            }
        }
        df.set_f64("tf_trend_agreement", vals);
    }

    if !adx_cols.is_empty() {
        let mut vals = vec![0.0; df.len()];
        for col in &adx_cols {
            let series = df.get_f64(col)?;
            for i in 0..df.len() {
                if !is_nan(series[i]) && series[i] > 25.0 {
                    vals[i] += 1.0;
                }
            }
        }
        df.set_f64("tf_trending_count", vals);
    }

    if !rsi_cols.is_empty() {
        let mut vals = vec![0.0; df.len()];
        let mut counts = vec![0usize; df.len()];
        for col in &rsi_cols {
            let series = df.get_f64(col)?;
            for i in 0..df.len() {
                if !is_nan(series[i]) {
                    vals[i] += series[i];
                    counts[i] += 1;
                }
            }
        }
        for i in 0..df.len() {
            if counts[i] > 0 {
                vals[i] /= counts[i] as f64;
            } else {
                vals[i] = f64::NAN;
            }
        }
        df.set_f64("tf_avg_rsi", vals);
    }

    Ok(())
}

fn merge_asof(base: &Frame, right: &Frame, on: &str) -> Result<Frame> {
    let base_times = base.get_i64(on)?;
    let right_times = right.get_i64(on)?;
    let mut out = base.clone();

    let mut right_idx = 0usize;
    let right_len = right.len();

    let right_f64_cols: Vec<String> = right
        .f64_cols
        .keys()
        .filter(|k| k.as_str() != on)
        .cloned()
        .collect();

    for col in &right_f64_cols {
        out.set_f64(col, vec![f64::NAN; base.len()]);
    }

    for i in 0..base.len() {
        let bt = base_times[i];
        while right_idx + 1 < right_len && right_times[right_idx + 1] <= bt {
            right_idx += 1;
        }
        if right_len == 0 || right_times[right_idx] > bt {
            continue;
        }
        for col in &right_f64_cols {
            let series = right.get_f64(col)?;
            if let Some(out_col) = out.f64_cols.get_mut(col) {
                out_col[i] = series[right_idx];
            }
        }
    }

    Ok(out)
}

fn calculate_multi_timeframe_features(
    bars_dict: &HashMap<String, Frame>,
    base_tf: &str,
    config: &FeatureConfig,
) -> Result<Frame> {
    let mut featured = HashMap::new();
    for (tf_name, bars) in bars_dict {
        let ft = calculate_features_for_timeframe(bars, config)?;
        featured.insert(tf_name.clone(), ft);
    }

    let mut result = featured
        .get(base_tf)
        .context("Missing base timeframe")?
        .clone();

    let base_seconds = config
        .timeframes
        .get(
            config
                .timeframe_names
                .iter()
                .position(|t| t == base_tf)
                .unwrap_or(0),
        )
        .cloned()
        .unwrap_or(0);

    let base_feature_cols: Vec<String> = result
        .f64_cols
        .keys()
        .filter(|c| {
            let name = c.as_str();
            !matches!(name, "bar_time" | "datetime" | "open" | "high" | "low" | "close" | "volume")
        })
        .cloned()
        .collect();

    for col in &base_feature_cols {
        if let Some(values) = result.f64_cols.remove(col) {
            result.set_f64(&format!("{}_{}", base_tf, col), values);
        }
    }

    for (tf_name, bars) in &featured {
        if tf_name == base_tf {
            continue;
        }
        let mut tf_features = Frame::new(bars.len());
        tf_features.set_i64("bar_time", bars.get_i64("bar_time")?.clone());
        let feature_cols: Vec<String> = bars
            .f64_cols
            .keys()
            .filter(|c| {
                let name = c.as_str();
                !matches!(name, "bar_time" | "datetime" | "open" | "high" | "low" | "close" | "volume")
            })
            .cloned()
            .collect();
        for col in &feature_cols {
            let values = bars.get_f64(col)?.clone();
            tf_features.set_f64(&format!("{}_{}", tf_name, col), values);
        }
        let tf_seconds = config
            .timeframes
            .get(
                config
                    .timeframe_names
                    .iter()
                    .position(|t| t == tf_name)
                    .unwrap_or(0),
            )
            .cloned()
            .unwrap_or(0);
        if tf_seconds > 0 {
            let times = tf_features.get_i64("bar_time")?.clone();
            let shifted: Vec<i64> = times.iter().map(|t| t + tf_seconds).collect();
            tf_features.set_i64("bar_time", shifted);
        }
        result = merge_asof(&result, &tf_features, "bar_time")?;
    }

    let open = result.get_f64("open")?.clone();
    let high = result.get_f64("high")?.clone();
    let low = result.get_f64("low")?.clone();
    let close = result.get_f64("close")?.clone();
    let volume = result.get_f64("volume")?.clone();

    for (tf_name, tf_seconds) in config.timeframe_names.iter().zip(config.timeframes.iter()) {
        if tf_name == base_tf {
            continue;
        }
        if *tf_seconds <= base_seconds || base_seconds == 0 || tf_seconds % base_seconds != 0 {
            continue;
        }
        let window = (*tf_seconds / base_seconds) as usize;
        let partial_open = rolling_first(&open, window, 1);
        let partial_high = rolling_max(&high, window, 1);
        let partial_low = rolling_min(&low, window, 1);
        let partial_close = close.clone();
        let partial_volume = rolling_sum(&volume, window, 1);

        result.set_f64(&format!("partial_{}_open", tf_name), partial_open);
        result.set_f64(&format!("partial_{}_high", tf_name), partial_high);
        result.set_f64(&format!("partial_{}_low", tf_name), partial_low);
        result.set_f64(&format!("partial_{}_close", tf_name), partial_close);
        result.set_f64(&format!("partial_{}_volume", tf_name), partial_volume);
    }

    calculate_cross_tf_features(&mut result, config)?;

    Ok(result)
}
fn label_trend_opportunities(df: &Frame, config: &LabelConfig, base_tf: &str) -> Result<Frame> {
    let mut result = df.clone();
    let n = result.len();
    let window = config.trend_forward_window as usize;
    let atr_col = format!("{}_atr", base_tf);
    let atr_vals = result.get_f64(&atr_col)?.clone();
    let close = result.get_f64("close")?.clone();
    let high = result.get_f64("high")?.clone();
    let low = result.get_f64("low")?.clone();

    let mut labels = vec![0.0; n];
    let mut max_favorable = vec![0.0; n];
    let mut max_adverse = vec![0.0; n];

    for i in 0..n.saturating_sub(window) {
        let current_price = close[i];
        let current_atr = atr_vals[i];
        if current_atr <= 0.0 || is_nan(current_atr) || is_nan(current_price) {
            continue;
        }
        let mut future_high = f64::NAN;
        let mut future_low = f64::NAN;
        for j in (i + 1)..=i + window {
            let h = high[j];
            let l = low[j];
            if !is_nan(h) && (is_nan(future_high) || h > future_high) {
                future_high = h;
            }
            if !is_nan(l) && (is_nan(future_low) || l < future_low) {
                future_low = l;
            }
        }
        if is_nan(future_high) || is_nan(future_low) {
            continue;
        }
        let max_up = (future_high - current_price) / current_atr;
        let max_down = (current_price - future_low) / current_atr;
        max_favorable[i] = if max_up > max_down { max_up } else { max_down };

        if max_up >= config.trend_up_threshold && max_down < config.max_adverse_for_trend {
            labels[i] = 1.0;
            max_adverse[i] = max_down;
        } else if max_down >= config.trend_down_threshold && max_up < config.max_adverse_for_trend {
            labels[i] = -1.0;
            max_adverse[i] = max_up;
        } else {
            labels[i] = 0.0;
            max_adverse[i] = if max_up < max_down { max_up } else { max_down };
        }
    }

    result.set_f64("trend_label", labels);
    result.set_f64("trend_max_favorable", max_favorable);
    result.set_f64("trend_max_adverse", max_adverse);
    Ok(result)
}

fn detect_multi_tf_ema_touches(
    df: &Frame,
    config: &LabelConfig,
    feature_config: &FeatureConfig,
    base_tf: &str,
    touch_threshold_atr: f64,
    min_slope_norm: f64,
) -> Result<Frame> {
    let mut result = df.clone();
    let n = result.len();
    let ema_period = config.pullback_ema;
    let atr_col = format!("{}_atr", base_tf);
    let atr_vals = result.get_f64(&atr_col)?.clone();
    let high = result.get_f64("high")?.clone();
    let low = result.get_f64("low")?.clone();
    let open = result.get_f64("open")?.clone();
    let close = result.get_f64("close")?.clone();

    let mut detected = vec![false; n];
    let mut touch_tf = vec![None; n];
    let mut touch_dir = vec![0.0; n];
    let mut touch_quality = vec![0.0; n];
    let mut touch_dist = vec![f64::NAN; n];
    let mut touch_slope = vec![f64::NAN; n];

    for tf in &feature_config.timeframe_names {
        let ema_col = format!("{}_ema_{}", tf, ema_period);
        let slope_col = format!("{}_ema_{}_slope_norm", tf, ema_period);
        if !result.f64_cols.contains_key(&ema_col) || !result.f64_cols.contains_key(&slope_col) {
            continue;
        }
        let ema_vals = result.get_f64(&ema_col)?.clone();
        let slope_vals = result.get_f64(&slope_col)?.clone();

        for i in 0..n {
            if detected[i] {
                continue;
            }
            let ema_val = ema_vals[i];
            let slope_val = slope_vals[i];
            let atr_val = atr_vals[i];
            if is_nan(ema_val) || is_nan(slope_val) || is_nan(atr_val) || atr_val <= 0.0 {
                continue;
            }
            let bar_high = high[i];
            let bar_low = low[i];
            let bar_close = close[i];
            let _bar_open = open[i];

            if slope_val > min_slope_norm {
                let dist_low = (bar_low - ema_val) / atr_val;
                if dist_low >= -touch_threshold_atr && dist_low <= touch_threshold_atr {
                    let mid_bar = (bar_high + bar_low) / 2.0;
                    if bar_close >= ema_val || mid_bar >= ema_val {
                        let dist_score = 1.0 - (dist_low.abs() / touch_threshold_atr).min(1.0);
                        let wick_score = if bar_high > bar_low {
                            (bar_close - bar_low) / (bar_high - bar_low)
                        } else {
                            0.0
                        };
                        let quality = (dist_score + wick_score) / 2.0;
                        detected[i] = true;
                        touch_tf[i] = Some(tf.clone());
                        touch_dir[i] = 1.0;
                        touch_quality[i] = quality;
                        touch_dist[i] = dist_low;
                        touch_slope[i] = slope_val;
                    }
                }
            } else if slope_val < -min_slope_norm {
                let dist_high = (bar_high - ema_val) / atr_val;
                if dist_high >= -touch_threshold_atr && dist_high <= touch_threshold_atr {
                    let mid_bar = (bar_high + bar_low) / 2.0;
                    if bar_close <= ema_val || mid_bar <= ema_val {
                        let dist_score = 1.0 - (dist_high.abs() / touch_threshold_atr).min(1.0);
                        let wick_score = if bar_high > bar_low {
                            (bar_high - bar_close) / (bar_high - bar_low)
                        } else {
                            0.0
                        };
                        let quality = (dist_score + wick_score) / 2.0;
                        detected[i] = true;
                        touch_tf[i] = Some(tf.clone());
                        touch_dir[i] = -1.0;
                        touch_quality[i] = quality;
                        touch_dist[i] = dist_high;
                        touch_slope[i] = slope_val;
                    }
                }
            }
        }
    }

    result.set_bool("ema_touch_detected", detected);
    result.set_str("ema_touch_tf", touch_tf);
    result.set_f64("ema_touch_direction", touch_dir);
    result.set_f64("ema_touch_quality", touch_quality);
    result.set_f64("ema_touch_dist", touch_dist);
    result.set_f64("ema_touch_slope", touch_slope);

    Ok(result)
}
fn label_pullback_outcomes(
    df: &Frame,
    pullback_mask: &[bool],
    config: &LabelConfig,
    base_tf: &str,
    use_ema_touch_direction: bool,
) -> Result<Frame> {
    let mut result = df.clone();
    let n = result.len();
    let window = config.entry_forward_window as usize;
    let atr_col = format!("{}_atr", base_tf);
    let alignment_col = format!("{}_ema_{}_slope_norm", base_tf, config.pullback_ema);

    let close = result.get_f64("close")?.clone();
    let open = result.get_f64("open")?.clone();
    let high = result.get_f64("high")?.clone();
    let low = result.get_f64("low")?.clone();
    let volume = result.get_f64("volume")?.clone();
    let atr_vals = result.get_f64(&atr_col)?.clone();

    let volume_ma_col = format!("{}_volume_sma", base_tf);
    let volume_ma = if result.f64_cols.contains_key(&volume_ma_col) {
        Some(result.get_f64(&volume_ma_col)?.clone())
    } else {
        None
    };

    let touch_dir = if use_ema_touch_direction && result.f64_cols.contains_key("ema_touch_direction") {
        Some(result.get_f64("ema_touch_direction")?.clone())
    } else {
        None
    };
    let alignment = if result.f64_cols.contains_key(&alignment_col) {
        Some(result.get_f64(&alignment_col)?.clone())
    } else {
        None
    };

    let mut pullback_success = vec![f64::NAN; n];
    let mut pullback_mfe = vec![f64::NAN; n];
    let mut pullback_mae = vec![f64::NAN; n];
    let mut pullback_rr = vec![f64::NAN; n];
    let mut pullback_win_r = vec![f64::NAN; n];
    let mut pullback_realized_r = vec![f64::NAN; n];
    let mut pullback_hit_tp_first = vec![f64::NAN; n];
    let mut pullback_bars_to_exit = vec![f64::NAN; n];
    let mut pullback_exit_type: Vec<Option<String>> = vec![None; n];
    let mut pullback_tier = vec![f64::NAN; n];
    let mut pullback_early_momentum = vec![f64::NAN; n];
    let mut pullback_immediate_range = vec![f64::NAN; n];
    let mut bounce_bar_body_ratio = vec![f64::NAN; n];
    let mut bounce_bar_wick_ratio = vec![f64::NAN; n];
    let mut bounce_bar_direction = vec![f64::NAN; n];
    let mut bounce_volume_ratio = vec![f64::NAN; n];

    for i in 0..n {
        if !pullback_mask[i] {
            continue;
        }
        if i + window >= n {
            continue;
        }
        let entry_price = close[i];
        let current_atr = atr_vals[i];
        let mut trend_dir = 0.0;
        if let Some(ref td) = touch_dir {
            trend_dir = td[i];
        } else if let Some(ref al) = alignment {
            trend_dir = sign(al[i]);
        }
        if current_atr <= 0.0 || is_nan(current_atr) || trend_dir == 0.0 || is_nan(trend_dir) {
            continue;
        }
        let trend_dir_i = if trend_dir > 0.0 { 1.0 } else { -1.0 };

        let bar_open = open[i];
        let bar_high = high[i];
        let bar_low = low[i];
        let bar_close = close[i];
        let bar_range = bar_high - bar_low;
        if bar_range > 0.0 {
            let body_size = (bar_close - bar_open).abs();
            bounce_bar_body_ratio[i] = body_size / bar_range;
            let favorable_wick = if trend_dir_i > 0.0 {
                bar_open.min(bar_close) - bar_low
            } else {
                bar_high - bar_open.max(bar_close)
            };
            bounce_bar_wick_ratio[i] = favorable_wick / bar_range;
        }
        bounce_bar_direction[i] = if bar_close > bar_open { 1.0 } else { -1.0 };
        if let Some(ref vol_ma) = volume_ma {
            if !is_nan(vol_ma[i]) && vol_ma[i] > 0.0 {
                bounce_volume_ratio[i] = safe_div(volume[i], vol_ma[i]);
            }
        }

        let stop_dist = config.stop_atr_multiple * current_atr;
        let tp_dist = config.target_rr * stop_dist;

        let (stop_level, tp_level) = if trend_dir_i > 0.0 {
            (entry_price - stop_dist, entry_price + tp_dist)
        } else {
            (entry_price + stop_dist, entry_price - tp_dist)
        };

        let mut hit_tp = false;
        let mut hit_sl = false;
        let mut bars_to_exit: Option<i64> = None;
        let mut exit_type = "timeout".to_string();
        let mut mfe: f64 = 0.0;
        let mut mae: f64 = 0.0;

        for j in (i + 1)..=usize::min(i + window, n - 1) {
            let future_high = high[j];
            let future_low = low[j];
            let future_open = open[j];

            if trend_dir_i > 0.0 {
                mfe = mfe.max((future_high - entry_price) / current_atr);
                mae = mae.max((entry_price - future_low) / current_atr);
                let sl_hit = future_low <= stop_level;
                let tp_hit = future_high >= tp_level;
                if sl_hit && tp_hit {
                    let dist_to_sl = (future_open - stop_level).abs();
                    let dist_to_tp = (future_open - tp_level).abs();
                    if dist_to_tp <= dist_to_sl {
                        hit_tp = true;
                        bars_to_exit = Some((j - i) as i64);
                        exit_type = "tp".to_string();
                    } else {
                        hit_sl = true;
                        bars_to_exit = Some((j - i) as i64);
                        exit_type = "sl".to_string();
                    }
                    break;
                } else if sl_hit {
                    hit_sl = true;
                    bars_to_exit = Some((j - i) as i64);
                    exit_type = "sl".to_string();
                    break;
                } else if tp_hit {
                    hit_tp = true;
                    bars_to_exit = Some((j - i) as i64);
                    exit_type = "tp".to_string();
                    break;
                }
            } else {
                mfe = mfe.max((entry_price - future_low) / current_atr);
                mae = mae.max((future_high - entry_price) / current_atr);
                let sl_hit = future_high >= stop_level;
                let tp_hit = future_low <= tp_level;
                if sl_hit && tp_hit {
                    let dist_to_sl = (future_open - stop_level).abs();
                    let dist_to_tp = (future_open - tp_level).abs();
                    if dist_to_tp <= dist_to_sl {
                        hit_tp = true;
                        bars_to_exit = Some((j - i) as i64);
                        exit_type = "tp".to_string();
                    } else {
                        hit_sl = true;
                        bars_to_exit = Some((j - i) as i64);
                        exit_type = "sl".to_string();
                    }
                    break;
                } else if sl_hit {
                    hit_sl = true;
                    bars_to_exit = Some((j - i) as i64);
                    exit_type = "sl".to_string();
                    break;
                } else if tp_hit {
                    hit_tp = true;
                    bars_to_exit = Some((j - i) as i64);
                    exit_type = "tp".to_string();
                    break;
                }
            }
        }

        let early_bars = std::cmp::min(3, window);
        if i + early_bars < n {
            let early_close = close[i + early_bars];
            let early_momentum = (early_close - entry_price) / current_atr;
            pullback_early_momentum[i] = early_momentum * trend_dir_i;
            let mut early_high = f64::NAN;
            let mut early_low = f64::NAN;
            for j in (i + 1)..=i + early_bars {
                let h = high[j];
                let l = low[j];
                if !is_nan(h) && (is_nan(early_high) || h > early_high) {
                    early_high = h;
                }
                if !is_nan(l) && (is_nan(early_low) || l < early_low) {
                    early_low = l;
                }
            }
            if !is_nan(early_high) && !is_nan(early_low) {
                pullback_immediate_range[i] = (early_high - early_low) / current_atr;
            }
        }

        let success = if hit_tp { 1.0 } else { 0.0 };
        let rr = mfe / mae.max(0.1);
        let stop_mult = if config.stop_atr_multiple > 0.0 { config.stop_atr_multiple } else { 1.0 };
        let win_r = mfe / stop_mult;

        pullback_success[i] = success;
        pullback_mfe[i] = mfe;
        pullback_mae[i] = mae;
        pullback_rr[i] = rr;
        if success == 1.0 {
            pullback_win_r[i] = win_r;
        }
        pullback_hit_tp_first[i] = if hit_tp { 1.0 } else { 0.0 };
        if let Some(b) = bars_to_exit {
            pullback_bars_to_exit[i] = b as f64;
        }
        pullback_exit_type[i] = Some(exit_type.clone());

        let realized_r = if hit_tp {
            config.target_rr
        } else if hit_sl {
            -1.0
        } else {
            let exit_idx = std::cmp::min(i + window, n - 1);
            let exit_price = close[exit_idx];
            if stop_dist > 0.0 {
                if trend_dir_i > 0.0 {
                    (exit_price - entry_price) / stop_dist
                } else {
                    (entry_price - exit_price) / stop_dist
                }
            } else {
                0.0
            }
        };
        pullback_realized_r[i] = realized_r;

        let tier = if hit_sl || mfe < 0.5 {
            0.0
        } else if mfe < 1.0 {
            1.0
        } else if mfe < 2.0 {
            2.0
        } else {
            3.0
        };
        pullback_tier[i] = tier;
    }

    result.set_f64("pullback_success", pullback_success);
    result.set_f64("pullback_mfe", pullback_mfe);
    result.set_f64("pullback_mae", pullback_mae);
    result.set_f64("pullback_rr", pullback_rr);
    result.set_f64("pullback_win_r", pullback_win_r);
    result.set_f64("pullback_realized_r", pullback_realized_r);
    result.set_f64("pullback_hit_tp_first", pullback_hit_tp_first);
    result.set_f64("pullback_bars_to_exit", pullback_bars_to_exit);
    result.set_str("pullback_exit_type", pullback_exit_type);
    result.set_f64("pullback_tier", pullback_tier);
    result.set_f64("pullback_early_momentum", pullback_early_momentum);
    result.set_f64("pullback_immediate_range", pullback_immediate_range);
    result.set_f64("bounce_bar_body_ratio", bounce_bar_body_ratio);
    result.set_f64("bounce_bar_wick_ratio", bounce_bar_wick_ratio);
    result.set_f64("bounce_bar_direction", bounce_bar_direction);
    result.set_f64("bounce_volume_ratio", bounce_volume_ratio);

    Ok(result)
}

fn label_regime(df: &Frame, base_tf: &str, feature_config: &FeatureConfig) -> Result<Frame> {
    let mut result = df.clone();
    let adx_col = format!("{}_adx", base_tf);
    let ema_period = feature_config.ema_periods.get(0).cloned().unwrap_or(9);
    let alignment_col = format!("{}_ema_{}_slope_norm", base_tf, ema_period);
    let atr_pct_col = format!("{}_atr_percentile", base_tf);

    let adx_vals = result.get_f64(&adx_col)?.clone();
    let alignment = result.get_f64(&alignment_col)?.clone();
    let atr_pct = if result.f64_cols.contains_key(&atr_pct_col) {
        Some(result.get_f64(&atr_pct_col)?.clone())
    } else {
        None
    };

    let mut regime = vec![0.0; result.len()];

    for i in 0..result.len() {
        if !is_nan(adx_vals[i]) && !is_nan(alignment[i]) && adx_vals[i] > 25.0 && alignment[i] > 0.05 {
            regime[i] = 1.0;
        }
        if !is_nan(adx_vals[i]) && !is_nan(alignment[i]) && adx_vals[i] > 25.0 && alignment[i] < -0.05 {
            regime[i] = 2.0;
        }
        if let Some(ref atrp) = atr_pct {
            if !is_nan(atrp[i]) && !is_nan(alignment[i]) && atrp[i] > 0.8 && alignment[i].abs() < 0.05 {
                regime[i] = 3.0;
            }
        }
    }

    result.set_f64("regime", regime);
    Ok(result)
}
fn get_feature_columns(df: &Frame) -> Vec<String> {
    let mut exclude: Vec<&str> = vec![
        "bar_time", "datetime", "open", "high", "low", "close", "volume", "value", "swing_high",
        "swing_low", "obv",
    ];
    let label_columns = vec![
        "trend_label",
        "trend_max_favorable",
        "trend_max_adverse",
        "pullback_label",
        "pullback_outcome",
        "pullback_success",
        "pullback_mfe",
        "pullback_mae",
        "pullback_rr",
        "pullback_win_r",
        "pullback_realized_r",
        "is_pullback_zone",
        "pullback_hit_tp_first",
        "pullback_bars_to_exit",
        "pullback_exit_type",
        "pullback_tier",
        "pullback_early_momentum",
        "pullback_immediate_range",
        "ema_touch_detected",
        "ema_touch_tf",
        "ema_touch_direction",
        "ema_touch_quality",
        "ema_touch_dist",
        "ema_touch_slope",
        "regime",
    ];
    for col in label_columns {
        exclude.push(col);
    }

    let exclude_patterns = vec![
        "_count",
        "_time",
        "trdMatchID",
        "symbol",
        "_label",
        "_outcome",
        "_success",
        "_mfe",
        "_mae",
        "_rr",
        "_hit_tp",
        "_exit_type",
        "_tier",
        "_early_momentum",
        "_immediate_range",
        "_bars_to_exit",
        "ema_touch_",
    ];

    let valid_bounce_features = vec![
        "bounce_bar_body_ratio",
        "bounce_bar_wick_ratio",
        "bounce_bar_direction",
        "bounce_volume_ratio",
    ];

    let mut feature_cols = Vec::new();
    for col in df.f64_cols.keys() {
        if exclude.contains(&col.as_str()) {
            continue;
        }
        if valid_bounce_features.contains(&col.as_str()) {
            feature_cols.push(col.clone());
            continue;
        }
        if exclude_patterns.iter().any(|pat| col.contains(pat)) {
            continue;
        }
        feature_cols.push(col.clone());
    }

    feature_cols.sort();
    feature_cols
}

fn create_training_dataset(
    df: &Frame,
    config: &LabelConfig,
    feature_config: &FeatureConfig,
    base_tf: &str,
) -> Result<(Frame, Vec<String>)> {
    let mut result = label_trend_opportunities(df, config, base_tf)?;

    let final_touch_threshold = config.pullback_threshold;
    result = detect_multi_tf_ema_touches(
        &result,
        config,
        feature_config,
        base_tf,
        final_touch_threshold,
        0.03,
    )?;

    let pullback_mask = result.get_bool("ema_touch_detected")?.clone();
    result = label_pullback_outcomes(&result, &pullback_mask, config, base_tf, true)?;

    result = label_regime(&result, base_tf, feature_config)?;

    let feature_cols = get_feature_columns(&result);

    let key_cols = vec![
        "trend_label".to_string(),
        format!("{}_atr", base_tf),
        format!("{}_adx", base_tf),
    ];

    let mut mask = vec![true; result.len()];
    for col in key_cols {
        if !result.f64_cols.contains_key(&col) {
            continue;
        }
        let vals = result.get_f64(&col)?.clone();
        for i in 0..result.len() {
            if is_nan(vals[i]) {
                mask[i] = false;
            }
        }
    }
    result = result.filter_rows(&mask);

    Ok((result, feature_cols))
}
use std::fs;
use std::io::Cursor;

use arrow::array::{ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::FileWriter;
use arrow::record_batch::RecordBatch;

fn parse_config_value(value: serde_json::Value) -> Result<PipelineConfig> {
    if let Some(best_cfg) = value.get("best_config") {
        let cfg: PipelineConfig =
            serde_json::from_value(best_cfg.clone()).context("Failed to parse best_config")?;
        Ok(cfg)
    } else {
        let cfg: PipelineConfig =
            serde_json::from_value(value).context("Failed to parse config")?;
        Ok(cfg)
    }
}

pub fn load_config(path: &Path) -> Result<PipelineConfig> {
    let raw = fs::read_to_string(path).with_context(|| format!("Failed to read {}", path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&raw).context("Failed to parse config JSON")?;
    parse_config_value(value)
}

pub fn load_config_from_str(raw: &str) -> Result<PipelineConfig> {
    let value: serde_json::Value = serde_json::from_str(raw).context("Failed to parse config JSON")?;
    parse_config_value(value)
}

pub fn resolve_data_dir(config: &mut PipelineConfig, config_path: &Path) -> Result<()> {
    let data_dir = PathBuf::from(&config.data.data_dir);
    if data_dir.is_absolute() {
        if data_dir.exists() {
            return Ok(());
        }
        return Err(anyhow::anyhow!(
            "Data directory not found: {}",
            data_dir.display()
        ));
    }

    if data_dir.exists() {
        config.data.data_dir = data_dir.to_string_lossy().to_string();
        return Ok(());
    }

    let cwd = std::env::current_dir().context("Failed to read current directory")?;
    let mut candidates: Vec<PathBuf> = Vec::new();

    for ancestor in cwd.ancestors() {
        candidates.push(ancestor.join(&data_dir));
    }
    if let Some(parent) = config_path.parent() {
        for ancestor in parent.ancestors() {
            candidates.push(ancestor.join(&data_dir));
        }
    }

    for cand in candidates {
        if cand.exists() {
            config.data.data_dir = cand.to_string_lossy().to_string();
            return Ok(());
        }
    }

    Err(anyhow::anyhow!(
        "Data directory not found: {}",
        data_dir.display()
    ))
}

fn nanmax(a: f64, b: f64) -> f64 {
    if is_nan(a) {
        b
    } else if is_nan(b) {
        a
    } else {
        a.max(b)
    }
}

fn nanmin(a: f64, b: f64) -> f64 {
    if is_nan(a) {
        b
    } else if is_nan(b) {
        a
    } else {
        a.min(b)
    }
}

fn write_arrow(frame: &Frame, path: &Path) -> Result<()> {
    let mut fields: Vec<Field> = Vec::new();
    let mut arrays: Vec<ArrayRef> = Vec::new();

    let mut col_names = frame.column_names();
    col_names.sort();

    for name in &col_names {
        if let Some(col) = frame.i64_cols.get(name) {
            let arr = Int64Array::from(col.clone());
            fields.push(Field::new(name, DataType::Int64, false));
            arrays.push(std::sync::Arc::new(arr) as ArrayRef);
        } else if let Some(col) = frame.f64_cols.get(name) {
            let arr = Float64Array::from(col.clone());
            fields.push(Field::new(name, DataType::Float64, true));
            arrays.push(std::sync::Arc::new(arr) as ArrayRef);
        } else if let Some(col) = frame.bool_cols.get(name) {
            let arr = BooleanArray::from(col.clone());
            fields.push(Field::new(name, DataType::Boolean, false));
            arrays.push(std::sync::Arc::new(arr) as ArrayRef);
        } else if let Some(col) = frame.str_cols.get(name) {
            let arr = StringArray::from(col.clone());
            fields.push(Field::new(name, DataType::Utf8, true));
            arrays.push(std::sync::Arc::new(arr) as ArrayRef);
        }
    }

    let schema = Schema::new(fields);
    let batch = RecordBatch::try_new(std::sync::Arc::new(schema), arrays)?;
    let file = fs::File::create(path).with_context(|| format!("Failed to create {}", path.display()))?;
    let mut writer = FileWriter::try_new(file, &batch.schema())?;
    writer.write(&batch)?;
    writer.finish()?;
    Ok(())
}

fn write_arrow_bytes(frame: &Frame) -> Result<Vec<u8>> {
    let mut fields: Vec<Field> = Vec::new();
    let mut arrays: Vec<ArrayRef> = Vec::new();

    let mut col_names = frame.column_names();
    col_names.sort();

    for name in &col_names {
        if let Some(col) = frame.i64_cols.get(name) {
            let arr = Int64Array::from(col.clone());
            fields.push(Field::new(name, DataType::Int64, false));
            arrays.push(std::sync::Arc::new(arr) as ArrayRef);
        } else if let Some(col) = frame.f64_cols.get(name) {
            let arr = Float64Array::from(col.clone());
            fields.push(Field::new(name, DataType::Float64, true));
            arrays.push(std::sync::Arc::new(arr) as ArrayRef);
        } else if let Some(col) = frame.bool_cols.get(name) {
            let arr = BooleanArray::from(col.clone());
            fields.push(Field::new(name, DataType::Boolean, false));
            arrays.push(std::sync::Arc::new(arr) as ArrayRef);
        } else if let Some(col) = frame.str_cols.get(name) {
            let arr = StringArray::from(col.clone());
            fields.push(Field::new(name, DataType::Utf8, true));
            arrays.push(std::sync::Arc::new(arr) as ArrayRef);
        }
    }

    let schema = Schema::new(fields);
    let batch = RecordBatch::try_new(std::sync::Arc::new(schema), arrays)?;
    let mut buffer = Vec::new();
    {
        let cursor = Cursor::new(&mut buffer);
        let mut writer = FileWriter::try_new(cursor, &batch.schema())?;
        writer.write(&batch)?;
        writer.finish()?;
    }
    Ok(buffer)
}

pub fn run_pipeline(config: &PipelineConfig, output_dir: &Path, write_intermediate: bool) -> Result<()> {
    fs::create_dir_all(output_dir)?;
    let trades = load_trades_cached(&config.data)?;
    let bars = create_multi_timeframe_bars(
        &trades,
        &config.features.timeframes,
        &config.features.timeframe_names,
    );
    let base_tf = config
        .features
        .timeframe_names
        .get(config.base_timeframe_idx)
        .context("Invalid base_timeframe_idx")?
        .clone();

    let features = calculate_multi_timeframe_features(&bars, &base_tf, &config.features)?;
    let (dataset, feature_cols) = create_training_dataset(
        &features,
        &config.labels,
        &config.features,
        &base_tf,
    )?;

    let dataset_path = output_dir.join("dataset.arrow");
    write_arrow(&dataset, &dataset_path)?;

    let feature_cols_path = output_dir.join("feature_cols.json");
    let cols_json = serde_json::to_string_pretty(&feature_cols)?;
    fs::write(&feature_cols_path, cols_json)?;

    if write_intermediate {
        for (tf_name, frame) in bars {
            let path = output_dir.join(format!("bars_{}.arrow", tf_name));
            write_arrow(&frame, &path)?;
        }
        let features_path = output_dir.join("features.arrow");
        write_arrow(&features, &features_path)?;
    }

    Ok(())
}

pub fn run_pipeline_in_memory(config: &PipelineConfig) -> Result<(Vec<u8>, Vec<String>)> {
    let trades = load_trades_cached(&config.data)?;
    let bars = create_multi_timeframe_bars(
        &trades,
        &config.features.timeframes,
        &config.features.timeframe_names,
    );
    let base_tf = config
        .features
        .timeframe_names
        .get(config.base_timeframe_idx)
        .context("Invalid base_timeframe_idx")?
        .clone();

    let features = calculate_multi_timeframe_features(&bars, &base_tf, &config.features)?;
    let (dataset, feature_cols) = create_training_dataset(
        &features,
        &config.labels,
        &config.features,
        &base_tf,
    )?;

    let bytes = write_arrow_bytes(&dataset)?;
    Ok((bytes, feature_cols))
}
