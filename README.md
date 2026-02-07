# 🌌 PurpleSky

⚠️The current model for FARTCOIN is somewhat old, it may be a good idea to train your own with the tools provided.

PurpleSky is a state-of-the-art automated trading system made 100% by AI and me, designed for high-frequency crypto futures trading (currently there is a well developed connector for Bybit, but it has the needed abstraction to make a connector for any other exchange). It leverages a hybrid machine learning architecture that combines Gradient Boosting (LightGBM) with Deep Learning (LSTM) and an intelligent Gating Network to navigate complex market microstructures.

There is a strategy already the model FARTCOINUSDT deploys correctly in Bybit (may also work in Binance though if you train on Bybit data and deploy on Binance), however you can make any strategy you want; it has quite a bit of flexibility, specially with the direction models and the LTSM model :)

If you want a different one out of the box, you may try changing the TP ATR search space when doing optuna optimization, and you may also remove the drawdown penalty on the score function for the trials ;)

## 🚀 Key Features

- **Hybrid Model Architecture**: Combines the structural efficiency of LightGBM with the temporal memory of LSTM.
- **Mixture of Experts (MoE)**: A neural Gating Network dynamically weights the GBM and LSTM predictions based on real-time market context (volatility, regime, etc.).
- **Meta-Labeling**: Secondary confirmation models verify the "directionality" of signals to filter out low-probability trades.
- **Real-Time Market Microstructure**: Processes high-frequency orderbook snapshots (50 levels) and taker flow to calculate advanced features like Liquidity Elasticity, Price-Liquidity Divergence, and Wall Integrity.
- **Production-Ready Core**: Built-in data watchdog, automatic clock synchronization, and graceful handling of exchange-specific artifacts.

---

## 🏗 Architecture Overview

PurpleSky doesn't rely on a single model. It uses a "Stacked Ensemble" approach:

1.  **LightGBM Execution (Model B)**: Predicts trade success based on immediate features.
2.  **LSTM Temporal (Model C)**: Analyzes the last 60 bars of sequence data to find hidden patterns.
3.  **Gating Network (Model D)**: An "Arbitrator" that decides which model to trust more given the current market regime.
4.  **Directional Confirmation (Model A)**: A high-threshold filter that ensures the general trend matches the execution signal.

---

## 🛠 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/wizrdfab/PurpleSky
    cd PurpleSky
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure API Keys**:
    Create a `keys.json` file in the root directory:
    ```json
    {
        "api_key": "YOUR_BYBIT_API_KEY",
        "api_secret": "YOUR_BYBIT_API_SECRET"
    }
    ```

---

## 📈 Usage

### 1. Training a Model
To train your own model on historical data, use `train.py`. The system expects data in the `data/` directory.

```bash
python train.py --symbol FARTCOINUSDT --timeframe 5m --trials 50 --model-dir models/FARTCOINUSDT/rank_1
```

**Options**:
- `--trials`: Number of Optuna optimization trials.
- `--candidate-n`: Number of top trials to evaluate in the out-of-sample set.
- `--microstructure-only`: Focus purely on orderbook data, ignoring technical indicators.

### 2. Live Trading
Once you have a trained model, start the bot with `live_trading.py`.

```bash
python live_trading.py --symbol FARTCOINUSDT --model-dir models/FARTCOINUSDT/rank_1
```

**Production Flags**:
- `--testnet`: Run on Bybit Testnet.
- `--timeframe`: Set the trading timeframe (must match your trained model).

---

## 🧪 Development & Testing
A comprehensive test suite is included to ensure data integrity and connectivity.

```bash
# Run CSV and Data Flow tests
python tests/test_csv_integrity.py
python tests/test_real_data_flow.py

# Run Live Data Accumulation test (12 minutes)
python tests/test_live_accumulation.py
```

---

## 📞 Contact & Support

**Bug Reports**: If you find a bug, please open an Issue here: [Issues](../../issues)  
**Feature Requests**: Have an idea? Start a discussion in the Issues tab.  
**Business Inquiries**: [contact@purple-sky.online](mailto:contact@purple-sky.online)

### Community
Feel free to contact me in case you need help or you just want to have a chat! I'm always willing to help if you need, and I would be happy to help train your own models! Feel free to contact me anytime.

- **Twitter/X**: [@Fabb_998](https://twitter.com/Fabb_998)
- **Telegram**: [Join the Channel](https://t.me/PurpleSkymm)
- **Discord**: [Join the Server](https://discord.gg/JjSC23Cv)

---

## ❤️ Support the Project

If you find this project useful, consider making a donation or signing up with my referral code in Binance:

- **Bitcoin (BTC)**: `1PucNiXsUCzfrMqUGCPfwgdyE3BL8Xnrrp`
- **Ethereum (ETH)**: `0x58ef00f47d6e94dfc486a2ed9b3dd3cfaf3c9714`
- **Binance Referral** [Create an account using my link](https://accounts.binance.com/register?ref=1204138773) (Referral Code: '1204138773')

Thanks! <3

Special aknowledgements to the guys from this paper I got some ideas from: https://arxiv.org/html/2505.23084v1

---
*Disclaimer: Trading cryptocurrencies involves significant risk. PurpleSky is provided for educational purposes only. Use it at your own risk.*
