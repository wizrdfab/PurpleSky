![Status](https://img.shields.io/badge/Status-Pre--Alpha-orange)

PurpleSky - Advanced Quantitative Trading System
================================================

PurpleSky is a high-performance, event-driven algorithmic trading engine designed for cryptocurrency derivatives markets (Bybit/Binance). It implements a novel Hybrid Strategy that dynamically switches between conservative Mean Reversion scalping and aggressive Trend Following based on a dual-layer machine learning architecture.

--------------------------------------------------------------------------------

Core Philosophy
---------------

PurpleSky solves the "Scalper's Dilemma" (winning small often but missing the big move) by separating Execution Probability from Directional Bias:

1. Execution Model (The Sniper):
   * Objective: Predicts the probability of a limit order getting filled and hitting a take-profit target before a stop-loss.
   * Behavior: Contrarian. Seeks liquidity imbalances and overextensions (Mean Reversion).
   * Output: "Safe" entry signals for choppy/ranging markets.

2. Direction Model (The Trend Filter):
   * Objective: Predicts purely where the price will be in the future (e.g., t+30m), ignoring entry mechanics.
   * Behavior: Momentum-based. Identifies breakouts and strong directional flow.
   * Output: A "Trend Confidence" score (0.0 to 1.0).

The Hybrid Logic:
The system fuses these two signals into a dynamic execution strategy:

Regime           | Trend Confidence | Strategy         | Execution Type      | TP Target
-----------------|------------------|------------------|---------------------|----------------------
Neutral / Chop   | Moderate (0.5-0.8)| Mean Reversion   | Limit Order (Maker) | Standard (0.8 ATR)
Breakout         | Extreme (> 0.87) | Trend Following  | Market Order (Taker) | Aggressive (10x Std)

This allows PurpleSky to "scalp the noise" for consistent income while "hammering the breakout" for massive windfall profits.

--------------------------------------------------------------------------------

System Architecture
-------------------

1. Data Pipeline & Feature Engineering (feature_engine.py)
PurpleSky does not rely on simple OHLCV indicators. It ingests high-frequency Tick and Orderbook data to generate Microstructure Alpha:
* Orderbook Elasticity: Measures how "soft" the liquidity walls are (slope of bid/ask depth).
* Liquidity Dominance: Ratio of passive depth to active taker volume.
* Regime Z-Scores: Normalizes volatility and volume against a 30-day baseline to detect "Wake-up" events.
* Micro-Price Deviation: Detects hidden buy/sell pressure inside the spread.

2. Machine Learning Core (models.py, train.py)
* Algorithm: LightGBM (Gradient Boosting) with DART (Dropout) to prevent overfitting.
* Validation: Uses Combinatorial Purged K-Fold Cross-Validation (CPCV).
    * Purging: Removes data points where training labels overlap with test data to prevent look-ahead bias (Leakage).
    * Embargo: Adds safety buffers between splits.
* Optimization: Uses Optuna to optimize not just model hyperparameters (depth, leaves) but also financial parameters (Stop Loss ATR, Take Profit ATR, Aggressive Threshold).

3. Event-Driven Backtester (backtest.py)
A strict, realistic simulation engine:
* Latency Simulation: Accounts for processing time.
* Fee Structure: Models Maker vs. Taker fees accurately.
* Timeouts: Cancels limit orders if not filled within N bars (preventing "toxic fills" later).
* Mark-to-Market: Tracks equity curve tick-by-tick.

4. Production Engine (live_trading_v2.py)
The battle-hardened execution script:
* WebSocket First: Listens to real-time public (Trade/Book) and private (Order/Position) streams.
* State Recovery: Maintains a local JSON state database to survive crashes or restarts without losing track of orders.
* Drift Monitoring: Alerts if live data distribution diverges significantly from training data (Concept Drift).
* Multi-Position Management: Can manage up to 3 overlapping positions to scale into trends.

--------------------------------------------------------------------------------

Installation & Setup
--------------------

Prerequisites:
* Python 3.10+
* Bybit or Binance Futures Account (Currently, there is an implementation of automated trading only for Bybit, API Keys required for live trading)

Installation:
```bashd
git clone https://github.com/wizrdfab/PurpleSky.git
cd PurpleSky
pip install -r requirements.txt
```

Configuration:
Edit config.py to adjust global settings, or use command-line arguments.

--------------------------------------------------------------------------------

Usage Workflow
--------------

1. Data Collection
Collect raw tick and depth data for your target asset. You can do so directly from Bybit https://www.bybit.com/derivatives/en/history-data (raw trade data and orderbook data) or you can download the data from binance using a collector (it builds history, so it may take a bit before you have enough to train the models):

```bash
python data_collector.py FARTCOINUSDT SOLUSDT ADAUSDT ZECUSDT
```
Stores data in data/FARTCOINUSDT_Binance/ data/SOLUSDT_Binance ...

2. Model Training (AutoML)
Run the optimization pipeline. This will:
1. Load data and generate features.
2. Run Optuna trials to find the best Hybrid Strategy parameters.
3. Validate the champion model on a held-out test set.
4. Save the model to models_v9/.
```bash
python train.py --symbol FARTCOINUSDT --data-dir data/FARTCOINUSDT_Binance --trials 50
```

3. Simulation / Dry Run
Test the model live without using real money.
```bash

python live_trading_v2.py --symbol FARTCOINUSDT --keys-file key_profiles.json --dry-run
python live_trading_v2.py --symbol FARTCOINUSDT --keys-file key_profiles.json --testnet
```

4. Live Trading (Real Money)
Warning: Ensure you understand the risks.
```bash
python live_trading_v2.py --symbol FARTCOINUSDT --keys-file keys.json
```

5. Monitoring Dashboard
Launch the local web dashboard to view live signals and PnL. Dashboard has a chart where signals can be seen clearly.
```bash
python live_dashboard.py
```
Access at http://localhost:8787

Dashboard supports discord/telegram notifications. You can pass them as 

python live_dashboard.py --discord-webhook {URL} --telegram-token {TOKEN} --telegram-chat-id {CHATID}

This Will send notificationsof max drawdown, errors, startups and high confidence signals (agressive trend or scalping). 

--------------------------------------------------------------------------------

Performance Metrics (Sample)
----------------------------

Target: FARTCOINUSDT (High Volatility Altcoin)
Period: Out-of-Sample Holdout (15% of dataset, 4.5~ days of trading)

Metric           | Value         | Notes
-----------------|---------------|----------------------------------------------
Total Return     | > 15.8%       | Over short validation window
Win Rate         | ~81%          | High consistency due to selective entry
Sortino Ratio    | 0.23          | Improved by "Home Run" logic
Max Drawdown     | 6.9%          | Managed via strict SL

Results vary by market regime. Past performance is not indicative of future results.

--------------------------------------------------------------------------------

It's greatly suggested that you train the model for certain coins with data from binance (using the data_collector for example) 
because in some coins Binance will lead from orderbook/volume and the model uses that to make predictions. You can though use data from Binance to get signals to trade in Bybit manually or use that data to trade automatically on Bybit by just training the model on Binance data and running it connected to Bybit. Using data from Bybit for FARTCOINUSDT is ok
because Bybit leads this coin hence the results are more reliable. I may soon add a connector to Binance API for automated trading.

Contact & Support
-----------------

* Bug Reports: If you find a bug, please open an Issue here: ../../issues
* Feature Requests: Have an idea? Start a discussion in the Issues tab.
* Business Inquiries: servicios.de.mercado.fzf@gmail.com

Community
---------

Feel free to contact me in case you need help or you just want to have a chat!

* Twitter/X: @Fabb_998 (https://twitter.com/Fabb_998)
* Telegram: Join the Channel (https://t.me/PurpleSkymm)
* Discord: Join the Server (https://discord.gg/JjSC23Cv)

--------------------------------------------------------------------------------

Support & Donations
-------------------

If you find this project useful, consider making a donation or signing with my referral code in Bybit:

* Bitcoin (BTC): 1PucNiXsUCzfrMqUGCPfwgdyE3BL8Xnrrp
* Ethereum (ETH): 0x58ef00f47d6e94dfc486a2ed9b3dd3cfaf3c9714

Create a Bybit account using my referral link:
https://www.bybit.com/invite?ref=14VP14Z

Thanks <3

--------------------------------------------------------------------------------

Disclaimer
----------

This software is for educational purposes only. Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.

We strongly recommend you test the software on the Bybit Testnet before using real money.

Legal Note:
This software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.
