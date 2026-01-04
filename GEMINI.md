# Sofia / PurpleSky Trading System

## Project Overview

**Sofia (aka PurpleSky)** is a sophisticated, AI-powered algorithmic trading system designed for the **Bybit** cryptocurrency exchange. It focuses on **Liquidity Provision** and **Directional Trading** using a "Council" of Machine Learning models (LightGBM) driven by granular **Orderbook Microstructure** features.

The system is engineered for:
*   **High-Fidelity Execution:** Uses Limit Orders with dynamic offsets based on volatility (ATR).
*   **Robustness:** Features a `HealthMonitor` to detect concept drift and a `RiskManager` with a hard drawdown kill-switch.
*   **Ensemble Logic:** "The Council" architecture allows multiple optimized models to vote on trade execution, increasing confidence and reducing variance.

## Architecture & Core Components

### 1. Live Execution (`live_trader.py`)
The `ChampionBot` class orchestrates the production lifecycle:
*   **`LiveDataManager`:** Manages real-time data ingestion (Trades & Orderbook snapshots), bootstrapping history, and maintaining data continuity.
*   **`FeatureEngine`:** Calculates complex features on-the-fly, including Orderbook Imbalance, Depth Slope, Spread Integrity, and Micro-Price deviations.
*   **`The Council`:** A collection of top-performing ML models (loaded from `models_v*`) that vote on `Buy` or `Sell` signals.
*   **`RiskManager` & `HealthMonitor`:** Real-time safety guards against regime shifts and account drawdowns.
*   **`ExchangeClient`:** Wrapper around `pybit` for API interaction.

### 2. Training Pipeline (`train.py`)
An AutoML pipeline powered by **Optuna**:
*   **Objective:** Optimizes Strategy parameters (TP/SL/Offsets) and Model hyperparameters (LightGBM) simultaneously.
*   **Validation:** Uses **Combinatorial Purged K-Fold** cross-validation to prevent overfitting and leakage.
*   **Output:** Generates a ranked list of models. The top N models form "The Council" stored in `models_v*/rank_N`.

### 3. Configuration (`config.py`)
Centralized configuration using Python `dataclasses`:
*   **`GlobalConfig`:** Master config object.
*   **`StrategyConfig`:** Risk settings, fees, and trade parameters.
*   **`ModelConfig`:** Hyperparameters for LightGBM and Council settings.
*   **`DataConfig`:** Paths and symbols.

## Key Files

*   `live_trader.py`: **Main Entry Point** for live trading.
*   `train.py`: Script for training and selecting model councils.
*   `config.py`: System-wide configuration.
*   `requirements.txt`: Python dependencies (`pybit`, `lightgbm`, `optuna`, `pandas`).
*   `Manual_live_trader.txt` / `MONUSDT_Trader_MAIN.bat`: Launch scripts/notes.

## Usage

### 1. Setup
Ensure Python 3.10+ is installed.
```bash
pip install -r requirements.txt
```

### 2. Training a New Council
To train a new ensemble for a symbol (e.g., `RAVEUSDT`):
```bash
python train.py --symbol RAVEUSDT --trials 50 --timeframe 5m --ob-levels 200
```
*   Outputs models to `models_v*/RAVEUSDT/rank_*`.

### 3. Running Live
To start the bot in production mode, you must provide your API keys.
*   **Environment Variables:** Set `BYBIT_API_KEY`, `BYBIT_API_SECRET`, and optionally `DISCORD_WEBHOOK_URL`.
*   **Windows Helper:** You can create a `secrets.bat` file (ignored by git) to set these variables automatically when using the provided batch scripts.

```bash
python live_trader.py --symbol RAVEUSDT --model-root models_v4/RAVEUSDT
```
*   **Dry Run:** Defaults to `True` in `config.py` unless API keys are present and valid.
*   **State:** Persists active orders and PnL to `bot_state_{symbol}_champion.json`.

## Development Conventions

*   **Code Style:** Pythonic, using `dataclasses` for configuration and type hinting for function signatures.
*   **Logging:** extensive logging to `logs/lone_champion.log` and stdout.
*   **Safety:** rigorous "Sanity Checks" (`SanityCheck` class) on data before inference to prevent GIGO (Garbage In, Garbage Out).
*   **Persistence:** Atomic state saving to prevent corruption during crashes.
