# PurpleSky User Manual

This manual provides detailed instructions on how to operate the **PurpleSky Quantitative Trading System**. It covers the entire lifecycle from data collection to live execution.

---

## 1. Environment Setup

### 1.1 Prerequisites
*   **Operating System:** Windows, Linux, or macOS.
*   **Python:** Version 3.10 or higher.
*   **RAM:** Minimum 8GB (16GB recommended for training large datasets).
*   **Storage:** SSD recommended for faster data processing.

### 1.2 Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/wizrdfab/PurpleSky.git
    cd PurpleSky
    ```
2.  Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    # Windows:
    .\venv\Scripts\activate
    # Linux/Mac:
    source venv/bin/activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### 1.3 API Keys
Create a file named `keys.json` in the root directory. This file stores your exchange credentials. **Never commit this file to Git.**

**Format:**
```json
{
    "profiles": {
        "default": {
            "api_key": "YOUR_BYBIT_API_KEY",
            "api_secret": "YOUR_BYBIT_API_SECRET"
        },
        "paper": {
            "api_key": "YOUR_TESTNET_KEY",
            "api_secret": "YOUR_TESTNET_SECRET"
        }
    }
}
```

---

## 2. Data Acquisition (The Fuel)

Before you can train a model, you need high-quality Tick and Orderbook data. PurpleSky includes a robust collector.

### 2.1 Running the Collector
The collector connects to Binance Futures (UM) WebSocket and saves data to the `data/` directory.

```bash
# Syntax: python data_collector.py [SYMBOL1] [SYMBOL2] ...
python data_collector.py FARTCOINUSDT
```

*   **Duration:** Let this run for at least **3-7 days** to gather enough data for a meaningful model. The more, the better.
*   **Output:** Data is saved in `data/FARTCOINUSDT_Binance/Trade/` and `data/FARTCOINUSDT_Binance/Orderbook/`.

### 2.2 Data Structure
*   `Trade/*.csv`: Tick-by-tick trade data (Price, Size, Side).
*   `Orderbook/*.data`: JSONL snapshots of the Level 2 Orderbook (200 levels).

---

## 3. Training & Optimization (The Engine)

Once you have data, you use the AutoML pipeline to build the "Brain" of the bot.

### 3.1 The `train.py` Script
This script performs Feature Engineering, Cross-Validation, and Hyperparameter Optimization.

**Command:**
```bash
python train.py --symbol FARTCOINUSDT --data-dir data/FARTCOINUSDT_Binance --trials 50
```

**Key Arguments:**
*   `--symbol`: The asset name (e.g., `FARTCOINUSDT`).
*   `--data-dir`: Path to the collected data (e.g., `data/FARTCOINUSDT_Binance`).
*   `--trials`: Number of optimization trials (Default: 20). Use 50-100 for better results.
*   `--timeframe`: Base timeframe for bars (Default: `5m`).

### 3.2 What happens during training?
1.  **Feature Generation:** Calculates complex metrics (Liquidity Elasticity, VWAP Distances).
2.  **Purged K-Fold:** Splits data into chunks, removing overlaps to prevent cheating.
3.  **Optuna Optimization:** Tries different combinations of:
    *   **Financials:** Stop Loss, Take Profit, Limit Offset.
    *   **Logic:** Aggressive Threshold (When to switch to Market Orders).
    *   **Model:** Tree Depth, Learning Rate.
4.  **Selection:** Picks the best trial based on Sortino Ratio.
5.  **Validation:** Tests the winner on the final 15% of data (Holdout).
6.  **Saving:** Saves the model to `models_v9/FARTCOINUSDT/...`.

---

## 4. Live Operations (Driving)

The `live_trading_v2.py` script is the execution engine.

### 4.1 Dry Run (Simulation)
Always start with a Dry Run to verify connections and logic without risking funds.

```bash
python live_trading_v2.py --symbol FARTCOINUSDT --keys-file keys.json --dry-run
```
*   **Behavior:** Simulates orders locally. No API calls are sent to the exchange for placement.
*   **Logs:** Check the logs to see "Placing Buy Limit..." messages.

### 4.2 Live Trading (Real Money)
When confident, remove the `--dry-run` flag.

```bash
python live_trading_v2.py --symbol FARTCOINUSDT --keys-file keys.json
```

**Optional Arguments:**
*   `--log-features`: Logs the feature vector for every bar (useful for debugging).
*   `--testnet`: Connects to Bybit Testnet instead of Mainnet.

### 4.3 Understanding the Hybrid Strategy in Action
The bot will operate in two modes automatically:
1.  **Sniper Mode (Limit Orders):** You will see it placing orders *below* current price (for Buys). It waits for a dip. If the dip doesn't happen, it cancels and moves the order.
2.  **Aggressor Mode (Market Orders):** If the trend signal spikes (e.g., > 0.87), you will see immediate Market entries with a much wider Take Profit (10x).

---

## 5. Monitoring & Dashboard

### 5.1 The Dashboard
PurpleSky includes a lightweight local web dashboard.

1.  Run in a separate terminal:
    ```bash
    python live_dashboard.py
    ```
2.  Open your browser to: `http://localhost:8050`

**Features:**
*   **Live Chart:** Candlestick chart with Buy/Sell markers.
*   **Model Confidence:** Real-time gauge of the Execution and Direction model probabilities.
*   **PnL Curve:** Tracks your session performance.

### 5.2 Log Files
*   `live_trading_v2.log`: General operational logs (Heartbeats, Errors).
*   `signals_SYMBOL.jsonl`: Structured log of every signal generated by the model.
*   `open_orders_SYMBOL.jsonl`: Raw snapshot of open orders (for debugging reconciliation).

---

## 6. Troubleshooting

**Q: "History insufficient / bootstrapping..."**
*   **Cause:** The bot needs historical data to calculate features (e.g., 24h VWAP).
*   **Fix:** The bot attempts to download this from the exchange automatically. If it fails, ensure your internet connection is stable or let the `data_collector.py` run longer before starting live trading.

**Q: "Daily drawdown limit exceeded"**
*   **Cause:** The safety fuse tripped. You lost too much money today (Default: 3% of equity).
*   **Fix:** The bot pauses trading. Review your strategy. To reset, restart the script (equity baseline resets).

**Q: "Order not confirmed resting"**
*   **Cause:** API Lag. The bot placed an order but the exchange didn't report it back immediately via WebSocket.
*   **Fix:** The bot will retry confirmation via REST API. Usually resolves itself.

**Q: "Drift alerts"**
*   **Cause:** The current market behavior (volatility, spread) is very different from the Training Data.
*   **Action:** If this persists, the model might be stale. Consider collecting new data and retraining (`train.py`).

---

**Happy Trading!**
*The PurpleSky Team*
