# PurpleSky - Quantitative Trading System

## Project Overview
**PurpleSky** (formerly Sofia Brox) is a high-frequency, limit-order based quantitative trading system designed for **Bybit Linear Perpetuals** (USDT). It employs a **Mean Reversion** strategy reinforced by **LightGBM DART** models and advanced **Orderbook Microstructure** features.

The system is designed to act as a "Directional Liquidity Provider," placing maker-only limit orders to capture spreads and rebates while using ML models to avoid toxic flow (adverse selection).

## Core Architecture

### 1. Alpha & Features (`feature_engine.py`)
The system generates ~40 features per 5-minute bar, focusing on two categories:
*   **Technicals:** ATR, EMAs (9, 21, 50), RSI, Volume Z-Scores.
*   **Microstructure (The "Secret Sauce"):**
    *   `ob_imbalance_z`: Z-Score of Orderbook Imbalance (Shock detection).
    *   `micro_pressure`: Deviation of Volume-Weighted Micro-Price from Mid-Price.
    *   `taker_buy_z`: Detection of aggressive market buying (Toxic Flow).
    *   `liq_dominance`: Ratio of Book Depth to Trade Volume.

### 2. Training Pipeline "Lone Champion" (`train.py`)
A strict AutoML pipeline designed to prevent overfitting:
1.  **Optimization:** Uses Optuna (`TPESampler`) to tune Strategy Params (Offsets, TP/SL) and Model Params simultaneously.
2.  **OOS-1 Qualification:** Top 10 candidates are tested on a 10% Out-Of-Sample slice.
3.  **Super-OOS Verification:** The winner is validated on a final 5% holdout ("Champion Final Exam").
4.  **Artifacts:** Best models are saved to `models_vX/<SYMBOL>/rank_1/`.

### 3. Execution Engines
*   **`live_trader.py` (Active):** The "Champion" runner. Loads the specific `rank_1` model and features. Focuses on high-fidelity execution of the trained strategy.
*   **`live_trading_v2.py` (Robust):** A highly defensive execution engine with drift monitoring, state reconciliation, and strict safety checks.

### 4. Backtesting (`backtest.py`)
An event-driven backtester that simulates:
*   **Maker/Taker Fees:** Critical for this low-margin strategy.
*   **Order Timeouts:** Cancels orders if not filled within `time_limit_bars`.
*   **Position Timeouts:** Force-closes positions after `max_holding_bars`.

## Key Commands

### Live Trading
Run the watchdog batch script (Windows):
```batch
.\RAVEUSDT_Trader_MAIN.bat
```
Or directly via Python:
```bash
python live_trader.py --model-root "models_v5/RAVEUSDT"
```

### Training
Train a new champion:
```bash
python train.py --symbol RAVEUSDT --trials 50 --timeframe 5m
```

### Backtesting
Run a simulation:
```bash
python backtest.py --config "models_v5/RAVEUSDT/rank_1/params.json"
```

## Configuration (`config.py`)
All settings are centralized in dataclasses:
*   **`StrategyConfig`:** `base_limit_offset_atr`, `tp_atr`, `sl_atr`.
*   **`ModelConfig`:** LightGBM params (`num_leaves`, `learning_rate`).
*   **`LiveSettings`:** `max_daily_drawdown_pct` (3%), `max_api_errors`.

## Development Guidelines
*   **AGENTS.md:** strictly follow instructions in this file for logging changes.
*   **Maker Only:** The strategy strictly uses `PostOnly` orders. Never use Market orders for entry.
*   **State Management:** Local state is persisted in `bot_state_*.json`. Do not manually edit these while the bot is running.
*   **Safety:** The `secrets.bat` file contains API keys and must **NEVER** be committed to version control.

## Directory Structure
```
├── data/                   # Historical CSV data (Ignored)
├── models_v*/              # Model artifacts (rank_1/, etc.)
├── logs/                   # Runtime logs
├── pybit-master/           # Vendored Bybit V5 API wrapper
├── train.py                # AutoML Training Entry Point
├── live_trader.py          # Primary Live Execution Script
├── feature_engine.py       # Feature Engineering Logic
├── exchange_client.py      # Bybit API Client
└── config.py               # Central Configuration
```
