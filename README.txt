# Sofia LightGBM (Brox) - TrendFollower ML System

A multi-timeframe, ML-driven trend follower for crypto. The pipeline ingests raw trades, builds OHLCV bars, engineers indicators, labels EMA-touch pullbacks, and tunes model + rule parameters with Optuna. Backtests are tuning-aligned by default and use the same EV gating and cost model as tuning.

## Highlights

- Multi-timeframe feature set (1m/5m/15m/1h/4h)
- EntryQualityModel for pullback bounce probability and EV gating
- Optional Trend/Regime gating (when enabled in tuning)
- Optuna tuning with robust scoring + candidate selection
- Tuning-aligned backtest (single-position, flip-on-opposite by default)
- Optional Rust pipeline for fast bars/features/labels

## Installation

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux

pip install -r requirements.txt
```

## Data Format

CSV files of raw trades under `data/<SYMBOL>/` with columns:

```
timestamp,price,size,side,tickDirection
```

Required:
- `timestamp` (Unix seconds, float ok)
- `price`
- `size`
- `side` (Buy/Sell)
- `tickDirection` (PlusTick/ZeroPlusTick/MinusTick/ZeroMinusTick)

## Common Workflow

### 1) Tuning

```bash
python train.py --data-dir ./data/RAVEUSDT --model-dir ./models/RAVEUSDT --optuna-tune --tune-trials 200
```

This creates a study folder under the model directory, for example:

```
models/RAVEUSDT/<study>/
  optuna.db
  tuning_summary_<trial>.json
  train_config_<trial>.json
  model_<trial>/
```

### 2) Train from a tuning summary

```bash
python train.py --data-dir ./data/RAVEUSDT --model-dir ./models/RAVEUSDT --train-from-tuning
```

If a summary path is not provided, the CLI prompts for study name and trial id.

### 3) Backtest (tuning-aligned)

```bash
python train.py --data-dir ./data/RAVEUSDT --model-dir ./models/RAVEUSDT --backtest-only
```

Backtests use the tuning-aligned engine (`backtest_tuned_config.py`) by default and inherit gate settings, EV margin, fees, and expected-rr flags from the saved `train_config_<trial>.json` + tuning summary.

## Rust Pipeline (Optional)

A Rust extension can accelerate bars/features/labels. When available, it is used automatically.

Build example (Windows/Linux):

```bash
# from repo root
maturin develop -m rust_pipeline/Cargo.toml
```

If the module is not available, the system falls back to the Python pipeline.

## Live Trading (Optional)

Live trading components exist for exchange integration. They rely on `pybit` (either the bundled `pybit-master/` or pip package). These are not required for tuning/backtesting.

## Notes

- Backtests and tuning are aligned by design. Use `PROGRAM_OVERVIEW.md` for the full workflow.
- See `AGENTS.md` for operating instructions and logging requirements.

## License

MIT
