# TrendFollower ML System

An ML-based trend following expert for crypto trading. Uses gradient boosting models trained on raw trade data to predict trends and identify high-probability entry points.

## Philosophy

This system follows the principle: **provide value to the market rather than extract from it**.

The TrendFollower serves the market by:
- Pushing price toward new fair value faster (price discovery)
- Providing liquidity to those exiting old regimes
- Absorbing counter-trend flow from trapped participants

## Features

- **Multi-timeframe analysis**: Analyzes 1m, 5m, 15m, 1h, and 4h timeframes simultaneously
- **Raw trade data**: Uses tick-level data for richer microstructure features
- **Three ML models**:
  - TrendClassifier: Predicts trend direction and probability
  - EntryQualityModel: Predicts pullback bounce probability
  - RegimeClassifier: Identifies market regime (trending/ranging/volatile)
- **Signal grading**: A/B/C quality grades based on confluence
- **Built-in backtester**: Evaluate strategy performance before going live

## Installation

```bash
# Clone or copy the trend_follower directory to your machine
cd trend_follower

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Format

The system expects CSV files with raw trade data in the following format:

```csv
timestamp,symbol,side,size,price,tickDirection,trdMatchID,grossValue,homeNotional,foreignNotional,RPI
1764056925.401,MONUSDT,Buy,521,0.031654,PlusTick,e549e450-...,1.6491734e+09,521,16.491734,0
```

Required columns:
- `timestamp`: Unix timestamp (seconds with decimals for milliseconds)
- `price`: Trade price
- `size`: Trade size
- `side`: "Buy" or "Sell"
- `tickDirection`: "PlusTick", "ZeroPlusTick", "MinusTick", "ZeroMinusTick"

## Usage

### 1. Prepare Your Data

Place your trade CSV files in a directory:

```
data/
├── trades_2024-11-25.csv
├── trades_2024-11-26.csv
├── trades_2024-11-27.csv
└── ...
```

### 2. Train Models

```bash
# Basic training (uses 5m as base timeframe)
python run_training.py --data-dir ./data

# Train on different timeframe
python run_training.py --data-dir ./data --base-tf 2  # 15m

# Train and run backtest
python run_training.py --data-dir ./data --backtest

# Custom model directory
python run_training.py --data-dir ./data --model-dir ./my_models
```

### 3. Backtest Only (after training)

```bash
python run_training.py --data-dir ./data --backtest-only
```

### 4. Use in Your Trading System

```python
from predictor import create_live_predictor
import pandas as pd

# Initialize predictor
predictor = create_live_predictor('./models')

# Feed new trades as they arrive
new_trades = pd.DataFrame(...)  # Your incoming trade data
predictor.add_trades(new_trades)

# Get predictions
prediction = predictor.get_full_prediction()
print(prediction)

# Output:
# {
#     'timestamp': '2024-12-03T10:30:00',
#     'trend': {
#         'direction': 1,
#         'direction_name': 'UP',
#         'prob_up': 0.72,
#         'confidence': 0.72,
#         'regime': 'trending_up'
#     },
#     'entry': {
#         'bounce_prob': 0.65,
#         'quality': 'A',
#         'is_pullback': True
#     },
#     'recommendation': 'ENTER LONG (A-grade)'
# }
```

## Configuration

Edit `config.py` to customize:

- **Timeframes**: Which timeframes to analyze
- **Indicators**: EMA periods, RSI/ADX periods, etc.
- **Labels**: Trend detection thresholds, pullback detection
- **Model**: LightGBM hyperparameters

## Project Structure

```
trend_follower/
├── config.py           # Configuration parameters
├── data_loader.py      # Load and preprocess trade CSVs
├── feature_engine.py   # Calculate technical features
├── labels.py           # Generate training labels
├── models.py           # ML model definitions
├── trainer.py          # Training pipeline
├── predictor.py        # Real-time prediction
├── backtest.py         # Strategy backtester
├── run_training.py     # Main entry point
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Model Details

### TrendClassifier

Predicts whether a tradeable trend is starting:
- **Input**: Multi-timeframe features (EMAs, ADX, RSI, volume, microstructure)
- **Output**: Probability of uptrend / downtrend / no-trend
- **Labels**: Based on forward price movement (2+ ATR move with limited drawdown)

### EntryQualityModel

Predicts pullback bounce probability:
- **Input**: Same features + pullback context
- **Output**: Probability of successful bounce, expected R:R
- **Labels**: Based on whether pullbacks to EMA reached target R:R

### RegimeClassifier

Classifies current market regime:
- **Output**: Ranging / Trending Up / Trending Down / Volatile
- **Used to**: Filter trades (e.g., only trade in trending regimes)

## Tips for Best Results

1. **More data is better**: Try to get at least 1M+ trades for training
2. **Match timeframe to style**: Use 5m for day trading, 1h for swing trading
3. **Start with A-grade signals only**: Then relax to B-grade if needed
4. **Monitor regime**: Avoid trading in "volatile" or "ranging" regimes
5. **Backtest thoroughly**: Don't go live until you understand the model's behavior

## Next Steps

After getting this working:
1. Integrate with your exchange connector for live trading
2. Add the TrendFollowerExpert to the orchestrator system
3. Implement proper position sizing and risk management
4. Set up monitoring and logging

## License

MIT F