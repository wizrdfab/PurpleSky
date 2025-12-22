# TrendFollower ML Trading System - Usage Manual

## Overview

TrendFollower is a machine learning-based trading system that identifies EMA pullback/bounce opportunities using LightGBM models. The system trains on historical trade data, backtests the strategy, and can execute trades live on Bybit (paper or real funds).

## Table of Contents

1. [Data Requirements](#1-data-requirements)
2. [Training Models](#2-training-models)
3. [Backtesting](#3-backtesting)
4. [Parameter Optimization](#4-parameter-optimization)
5. [Live Paper Trading](#5-live-paper-trading)
6. [Live Trading with Real Funds](#6-live-trading-with-real-funds)
7. [Key Parameters Reference](#7-key-parameters-reference)

---

## 1. Data Requirements

### Data Format
The system requires raw trade data in CSV format with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | float | Unix timestamp in seconds |
| `price` | float | Trade price |
| `size` | float | Trade size/volume |
| `side` | string | "Buy" or "Sell" |
| `tickDirection` | string | "PlusTick", "MinusTick", "ZeroPlusTick", "ZeroMinusTick" |

### Data Organization
Place your CSV files in a directory structure like:
```
data/
  MONUSDT/
    MONUSDT2025-11-25.csv
    MONUSDT2025-11-26.csv
    ...
  BEATUSDT/
    BEATUSDT2025-11-25.csv
    ...
```

### Collecting Data
You can collect trade data from Bybit using WebSocket streams. The system expects at least 7-14 days of data for meaningful model training.

---

## 2. Training Models

### Basic Training
```bash
python train.py --data-dir data/MONUSDT --model-dir models_my_symbol
```

### Two-Pass Training (Recommended)
Uses validation set for early stopping, then retrains on combined train+val with optimal iterations:
```bash
python train.py --data-dir data/MONUSDT --model-dir models_my_symbol --two-pass
```

### Training Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | data/MONUSDT | Directory with trade CSV files |
| `--model-dir` | models_ema9_touch | Where to save trained models |
| `--two-pass` | false | Enable two-pass training |
| `--train-only` | false | Skip backtest after training |
| `--train-ratio` | 0.70 | Training data ratio |
| `--val-ratio` | 0.15 | Validation data ratio |
| `--test-ratio` | 0.15 | Test data ratio |
| `--lookback-days` | None | Use only last N days of data |

### LightGBM Hyperparameters
```bash
python train.py --data-dir data/MONUSDT --model-dir models_my_symbol \
  --learning-rate 0.05 \
  --num-leaves 31 \
  --n-estimators 500 \
  --max-depth 6 \
  --feature-fraction 0.8 \
  --min-child-samples 50 \
  --lambdaa-ele1 0.5 \
  --lambdaa-ele2 0.5
```

---

## 3. Backtesting

### Backtest with Existing Models
```bash
python train.py --backtest-only --data-dir data/MONUSDT --model-dir models_my_symbol
```

### Key Backtest Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--min-bounce-prob` | 0.48 | Minimum bounce probability to enter |
| `--max-bounce-prob` | 1.0 | Maximum bounce probability (for bucket filtering) |
| `--stop-loss-atr` | 1.0 | Stop loss distance in ATR units |
| `--take-profit-rr` | 1.5 | Take profit as reward:risk ratio |
| `--trade-side` | long | "long", "short", or "both" |
| `--use-dynamic-rr` | false | Use model's expected RR for TP sizing |
| `--touch-threshold-atr` | 0.3 | EMA touch detection threshold |

### Example: Backtest with Probability Bucket
Filter trades to only those with bounce probability between 0.42 and 0.55:
```bash
python train.py --backtest-only --data-dir data/MONUSDT --model-dir models_my_symbol \
  --min-bounce-prob 0.42 --max-bounce-prob 0.55 --trade-side both
```

### Example: Backtest with Dynamic RR
Use the model's expected R:R prediction instead of fixed take-profit:
```bash
python train.py --backtest-only --data-dir data/MONUSDT --model-dir models_my_symbol \
  --use-dynamic-rr
```

---

## 4. Parameter Optimization

### Run the Custom Optimizer
Find optimal LightGBM hyperparameters AND min_bounce_prob threshold:
```bash
python train.py --data-dir data/MONUSDT --model-dir models_test \
  --optimize --optimize-trials 30 --trade-side both
```

The optimizer will:
1. Run random search over LightGBM hyperparameters
2. For each model, find the optimal min_bounce_prob threshold
3. Maximize cumulative profit (in R-units) on the validation set
4. Print a recommended training command with best parameters

### Optimizer Output Example
```
================================================================================
P&L DISTRIBUTION BY min_bounce_prob THRESHOLD
================================================================================
 Threshold   Trades   Win Rate     Total P&L       PF     Avg P&L
--------------------------------------------------------------------------------
      0.40       85      64.7%        +24.5 R     2.75      +0.29 R
      0.42       72      65.3%        +22.8 R     2.83      +0.32 R <-- OPTIMAL
      0.44       58      62.1%        +16.2 R     2.45      +0.28 R
      ...

================================================================================
RECOMMENDED TRAINING COMMAND
================================================================================
python train.py --data-dir data/MONUSDT --model-dir models_test \
  --two-pass --trade-side both \
  --min-bounce-prob 0.42 \
  --learning-rate 0.0421 \
  --max-depth 5 \
  ...
================================================================================
```

---

## 5. Live Paper Trading

Paper trading simulates live execution without real funds. Use this to verify the model performs as expected before risking real money.

### Start Paper Trading
```bash
python live_trading.py --model-dir models_my_symbol --symbol MONUSDT
```

### With Bootstrap Data
Pre-load historical trades for faster warmup:
```bash
python live_trading.py --model-dir models_my_symbol --symbol MONUSDT \
  --bootstrap-csv data/MONUSDT
```

### Paper Trading Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-dir` | required | Directory with trained models |
| `--symbol` | MONUSDT | Trading symbol |
| `--testnet` | false | Use Bybit testnet |
| `--min-bounce-prob` | 0.48 | Minimum bounce probability |
| `--max-bounce-prob` | 1.0 | Maximum bounce probability |
| `--stop-loss-atr` | 1.0 | Stop loss in ATR units |
| `--take-profit-rr` | 1.5 | Take profit R:R ratio |
| `--use-dynamic-rr` | false | Use model's expected RR |
| `--trade-side` | long | "long", "short", or "both" |
| `--warmup-trades` | 1000 | Trades to collect before starting |
| `--bootstrap-csv` | None | CSV/directory to pre-load trades |

### Example: Paper Trading with Bucket Filter
```bash
python live_trading.py --model-dir models_my_symbol --symbol MONUSDT \
  --min-bounce-prob 0.42 --max-bounce-prob 0.55 --trade-side both
```

---

## 6. Live Trading with Real Funds

**WARNING: Live trading with real funds carries significant risk. Only proceed if you fully understand the system and accept potential losses.**

### Prerequisites
1. Bybit API key with trading permissions
2. Set environment variables:
   ```bash
   export BYBIT_API_KEY="your_api_key"
   export BYBIT_API_SECRET="your_api_secret"
   ```

### Dry Run (Recommended First)
Test the full pipeline without placing real orders:
```bash
python live_trading_funds.py --model-dir models_my_symbol --symbol MONUSDT --dry-run
```

### Live Trading (Real Orders)
```bash
python live_trading_funds.py --model-dir models_my_symbol --symbol MONUSDT \
  --leverage 1 --min-bounce-prob 0.48
```

### Live Trading Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-dir` | required | Directory with trained models |
| `--symbol` | MONUSDT | Trading symbol |
| `--testnet` | false | Use Bybit testnet |
| `--dry-run` | false | Disable real order placement |
| `--leverage` | 1 | Leverage on Bybit |
| `--balance-asset` | USDT | Asset for balance/sizing |
| `--api-key` | env var | Bybit API key |
| `--api-secret` | env var | Bybit API secret |
| `--min-bounce-prob` | 0.48 | Minimum bounce probability |
| `--max-bounce-prob` | 1.0 | Maximum bounce probability |
| `--use-dynamic-rr` | false | Use model's expected RR |
| `--stop-loss-atr` | 1.0 | Stop loss in ATR units |
| `--take-profit-rr` | 1.5 | Take profit R:R ratio |
| `--trade-side` | long | "long", "short", or "both" |

### Safety Features
- **Position sync**: Detects existing exchange positions on startup
- **Entry deviation check**: Skips entries if signal price is too far from live price
- **Stale bar protection**: Won't trade on old/bootstrap data
- **Exchange SL/TP**: Stop loss and take profit are attached on Bybit's side

### Example: Conservative Live Setup
```bash
python live_trading_funds.py --model-dir models_my_symbol --symbol MONUSDT \
  --leverage 1 \
  --min-bounce-prob 0.50 \
  --max-bounce-prob 0.65 \
  --stop-loss-atr 1.0 \
  --take-profit-rr 1.5 \
  --trade-side long \
  --bootstrap-csv data/MONUSDT
```

---

## 7. Key Parameters Reference

### Entry Filters

| Parameter | Description | Recommendation |
|-----------|-------------|----------------|
| `min-bounce-prob` | Minimum model confidence to enter | Start at 0.48, optimize with `--optimize` |
| `max-bounce-prob` | Maximum confidence (bucket filtering) | Use if certain prob ranges perform better |
| `trade-side` | Direction filter | "long" for uptrends, "short" for downtrends, "both" for all |

### Risk Management

| Parameter | Description | Recommendation |
|-----------|-------------|----------------|
| `stop-loss-atr` | Stop distance in ATR units | 1.0 is standard; lower = tighter stops |
| `stop-padding-pct` | Extra stop distance as % of price | 0.0 (disabled) or small value like 0.001 |
| `take-profit-rr` | Take profit as multiple of risk | 1.5-2.0 typical; higher = fewer wins but bigger |
| `use-dynamic-rr` | Let model predict optimal R:R | Enable if model's expected_rr is reliable |

### Cooldown

| Parameter | Description | Recommendation |
|-----------|-------------|----------------|
| `cooldown-bars-after-stop` | Bars to wait after stop-loss | 0 (disabled) or 1-5 to avoid revenge trading |

---

## Workflow Summary

1. **Collect Data**: Gather 7-14+ days of trade data for your symbol
2. **Train**: `python train.py --data-dir data/SYMBOL --model-dir models_symbol --two-pass`
3. **Optimize**: `python train.py --optimize --optimize-trials 30 ...` (optional but recommended)
4. **Backtest**: `python train.py --backtest-only ...` with various parameters
5. **Paper Trade**: `python live_trading.py ...` for at least a few days
6. **Live Trade**: `python live_trading_funds.py --dry-run ...` first, then without `--dry-run`

---

## Troubleshooting

### No Trades in Backtest
- Lower `--min-bounce-prob` threshold
- Check `--trade-side` matches your data (use "both" for bidirectional)
- Ensure data has enough EMA touch events

### Poor Backtest Performance
- Try `--optimize` to find better hyperparameters
- Experiment with `--max-bounce-prob` bucket filtering
- Check if `--use-dynamic-rr` helps or hurts

### Live Trading Issues
- Ensure `--bootstrap-csv` points to recent data for faster warmup
- Check API credentials are set correctly
- Use `--dry-run` first to verify signals are generated

---

## File Reference

| File | Purpose |
|------|---------|
| `train.py` | Training, backtesting, optimization |
| `live_trading.py` | Paper trading simulation |
| `live_trading_funds.py` | Real fund trading on Bybit |
| `config.py` | Configuration dataclasses |
| `models.py` | ML model definitions |
| `backtest.py` | Backtester implementation |
| `optimizer.py` | Custom parameter optimizer |
| `predictor.py` | Live prediction interface |
| `feature_engine.py` | Feature calculation |
| `labels.py` | Label generation |
| `data_loader.py` | Data loading utilities |
