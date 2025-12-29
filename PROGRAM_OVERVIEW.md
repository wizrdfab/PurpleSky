# Program Overview (Sofia LightGBM)

Purpose:
This system trains, tunes, and evaluates a multi-timeframe EMA touch strategy using
LightGBM models. It enforces consistent decision logic across tuning, backtests,
and live/paper trading.

High-Level Flow (data -> production):

  Trade CSVs
     |
     v
  Bars / Features / Labels  (Rust pipeline if available; Python fallback)
     |
     v
  Optuna Tuning (walk-forward folds + EV gate + costs)
     |
     v
  Candidate Selection (test split + shadow holdout)
     |
     v
  Train from Tuning (model_<trial>)
     |
     v
  Backtest (tuning-aligned)
     |
     v
  Stress Tests (optional)
     |
     v
  Live Trading / Paper Trading


Key Modules:
- data_loader.py: Load trade CSVs and create OHLCV bars (Python path).
- rust_pipeline_bridge.py + rust_pipeline/: Rust acceleration for bars/features/labels.
- feature_engine.py: Technical features + multi-timeframe features.
- labels.py: Pullback labeling, trend labels, regimes.
- models.py: TrendClassifier, EntryQualityModel, RegimeClassifier.
- trainer.py: Training pipeline, time series splits, seed ensembles.
- config_tuner.py: Optuna tuning, EV gating, costs, selection logic.
- backtest_tuned_config.py: Tuning-aligned backtest (default for --backtest-only).
- train.py: CLI entrypoint for tuning, training, and backtesting.
- live_trading_funds_simulated.py: Simulation of exchange plus trading on historical data
- live_trading_funds.py : Live trading (production)

Study-Scoped Artifacts (model_dir/<study>/):
- optuna.db: Default Optuna storage (sqlite).
- tuning_summary_<trial>.json: Full tuning summary per trial.
- train_config_<trial>.json: Training config that references tuning summary.
- model_<trial>/ : Trained LightGBM models for that trial.


Workflows

1) Data Gathering
- Trade CSVs live in data/<SYMBOL>/ (timestamp, price, size, side, tickDirection).
- Rust pipeline loads CSVs and generates bars, features, labels if available.
- Python fallback uses data_loader.py + feature_engine.py + labels.py.

2) Tuning (Optuna)
- Entry point: train.py --optuna-tune
- Base config (features/labels/model) is tuned by config_tuner.py.
- Uses walk-forward folds over tuning data with EV gating and costs.
- Selection uses tuning-aligned backtest on test + shadow splits.
- Outputs are stored in model_dir/<study>/.

3) Candidate Selection
- Top-N trials ranked by robust fold score (p25) with stability/confidence tie-breaker.
- Candidates are evaluated on test split; survivors may also run on shadow split.
- Selection status is recorded in tuning_summary_<trial>.json.

4) Train from Tuning
- Entry point: train.py --train-from-tuning [path]
- If no path is provided, prompts for study + trial.
- Produces model_dir/<study>/model_<trial>/ and train_config_<trial>.json.

5) Backtest (Tuning-Aligned)
- Entry point: train.py --backtest-only
- If no train config path is provided, prompts for study + trial.
- Loads train_config_<trial>.json and the matching tuning_summary_<trial>.json.
- Uses the same EV gate settings (fees, ops cost, expected_rr) and gates used in tuning.
- Warns if CLI overrides differ from tuned settings.

6) Stress Tests (Optional)
- Run tuning-style backtests on full dataset, folds for diagnostics or extra data.
- Use backtest_tuned_config.py to ensure consistency with tuning logic.

7) Live trading simulation
- Uses live_trading_funds_simulated.py to simulate a exchange that feeds trade data into the trading engine.

7) Live / Paper Trading
- live_trading_funds.py
- Use the same config fields (EV gate, fees, gates, timeouts) as tuning/backtests.


Consistency Guards
- Train configs store tuning_summary_path; backtests use it by default.
- Backtest warns when CLI overrides diverge from tuned settings.
- Feature audit compares expected vs data features.
- EV gating uses fee_percent and fee_per_trade_r from tuning.
- Single-position (default) and flip logic (default) is stored in tuning settings and reused.


Operational Notes
- Study naming: If no study name is provided, a random one is generated.
- Optuna storage: Defaults to sqlite in the study folder unless overridden.
- If you see feature mismatch errors, confirm backtest is using the matching
  train_config_<trial>.json and tuning_summary_<trial>.json from the same study.
