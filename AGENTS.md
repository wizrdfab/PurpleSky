# AGENTS.md - Instructions for AI Agents

This document provides instructions for AI agents (Claude, GPT, etc.) working on this codebase.

## Mandatory: Update ai-log.txt After Every Change

**CRITICAL**: Every AI agent MUST update `ai-log.txt` after making any code changes.

### Required Information for Each Entry

When updating `ai-log.txt`, include the following information:

1. **Date/Time**: Use ISO 8601 format (YYYY-MM-DD HH:MM)
2. **Agent Name**: Identify yourself (e.g., "Claude", "GPT-4", "Copilot")
3. **Files Modified**: List all files changed with specific line numbers
4. **Original Intent**: What the user requested or what problem was being solved
5. **Changes Made**: Brief description of what was changed and why
6. **Testing Status**: Whether changes were tested and how

### Entry Format Template

```
----------------------------------------------------------------------
v[VERSION] ([Brief Title] - [Month] [Year])
----------------------------------------------------------------------
Agent: [Agent Name]
Date: [YYYY-MM-DD HH:MM]
Files Modified:
- [filename.py]:[line numbers] - [brief description of change]
- [filename2.py]:[line numbers] - [brief description of change]

Original Intent: [What the user asked for]

Changes Made:
1. [First change description]
2. [Second change description]

Testing: [How this was tested, or "Not tested" if applicable]
```

### Example Entry

```
----------------------------------------------------------------------
v1.4 (Probability Calibration Option - Dec 2025)
----------------------------------------------------------------------
Agent: Claude
Date: 2025-12-20 15:30
Files Modified:
- train.py:168-188 - Added --use-calibration and --use-raw-probabilities flags
- train.py:397 - Pass use_calibration to SimpleBacktester
- train.py:436 - Added use_calibration to log_params
- backtest.py:87 - Added use_calibration parameter to __init__
- backtest.py:106 - Store use_calibration instance variable
- backtest.py:244-246 - Log use_calibration in diagnostics
- backtest.py:303 - Pass use_calibration to entry_model.predict()

Original Intent: User requested to reintroduce the probability calibrator
(IsotonicRegression) with an option to enable it during backtest.

Changes Made:
1. Added --use-calibration flag to train.py to enable calibrated probabilities
2. Added --use-raw-probabilities flag for explicit raw probability usage
3. Updated SimpleBacktester to accept and use the use_calibration parameter
4. Updated predict() call to pass the calibration flag

Testing: Not tested (no test data available)
```

## Codebase Overview

### Core Pipeline (train -> backtest -> live)
- **Data ingestion**: `data_loader.py` - Load trade CSVs, create OHLCV bars
- **Features**: `feature_engine.py` - Calculate technical indicators, multi-TF features
- **Labels**: `labels.py` - Generate training labels (EMA touch, bounce outcomes)
- **Training**: `trainer.py` / `train.py` - Model training pipeline
- **Backtest**: `backtest.py` - Strategy evaluation
- **Live trading**: `live_trading.py`, `live_trading_funds.py`

### Key Files
- `config.py` - Configuration parameters
- `models.py` - ML model definitions (TrendClassifier, EntryQualityModel, RegimeClassifier)
- `predictor.py` - Real-time prediction interface
- `hyperopt.py` - Hyperparameter optimization

### Important Patterns
1. **Multi-timeframe analysis**: Uses 1m, 5m, 15m, 1h, 4h timeframes
2. **EMA-based entries**: Primary entry signal is EMA touch detection
3. **Probability calibration**: IsotonicRegression calibrator in EntryQualityModel
4. **Leakage prevention**: Careful handling of HTF data to prevent lookahead

## Environment Notes
- Windows PowerShell environment
- Console encoding: cp1252 (avoid emojis/unicode in console logs)
- Snapshot folders are READ-ONLY (do not modify)

## Testing Guidelines
1. Run backtests to verify changes don't break existing functionality
2. Check for data leakage in feature engineering
3. Verify probability calibration if modifying prediction logic

## Do Not Modify
- Snapshot folders (used as program state "freezes")
- Files in `pybit-master/` (external dependency)

---

## Additional Documentation

- **VERSION_CONTROL.md** - Git workflow, CI/CD pipelines, rollback instructions
- **DATA_SETUP.md** - How to sync large data files to Azure VM
