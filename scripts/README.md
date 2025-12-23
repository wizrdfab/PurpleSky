# Scripts

Utility scripts for Sofia trading bot.

## Historical Data Download

### Setup

```bash
# Install Selenium
pip install selenium

# Make sure Chrome/Chromium is installed
# Chrome will be auto-detected
```

### Download Historical Data from Bybit

```bash
# Download 1 month of BTCUSDT tick data
python scripts/download_bybit_historical.py --symbol BTCUSDT --days 30

# Download specific date range
python scripts/download_bybit_historical.py \
    --symbol BTCUSDT \
    --start 2024-11-01 \
    --end 2024-11-30

# Download in headless mode (no browser window)
python scripts/download_bybit_historical.py \
    --symbol ETHUSDT \
    --days 7 \
    --headless

# Download different data types
python scripts/download_bybit_historical.py \
    --symbol BTCUSDT \
    --days 30 \
    --data-type "Funding Rate"

# Custom output directory
python scripts/download_bybit_historical.py \
    --symbol BTCUSDT \
    --days 30 \
    --output-dir ./my_data
```

**Options:**
- `--symbol`: Trading pair (default: BTCUSDT)
- `--product`: Product type (default: "USDT Perpetual")
- `--data-type`: "Trading", "Funding Rate", "Premium Index", etc.
- `--days`: Number of days to download (backwards from yesterday)
- `--start / --end`: Specific date range (YYYY-MM-DD)
- `--output-dir`: Download directory (default: ./data/bybit_downloads)
- `--headless`: Run without browser window

### Process Downloaded Files

```bash
# Merge all downloaded CSVs into one file
python scripts/process_bybit_downloads.py \
    --input data/bybit_downloads \
    --output data/BTCUSDT_ticks_1month.csv

# Filter by symbol
python scripts/process_bybit_downloads.py \
    -i data/bybit_downloads \
    -o data/BTCUSDT_ticks.csv \
    -s BTCUSDT
```

### Complete Workflow Example

```bash
# 1. Download 30 days of BTCUSDT data
python scripts/download_bybit_historical.py \
    --symbol BTCUSDT \
    --days 30 \
    --headless

# 2. Process and merge files
python scripts/process_bybit_downloads.py \
    --input data/bybit_downloads \
    --output data/BTCUSDT_ticks_30days.csv \
    --symbol BTCUSDT

# 3. Now use the data for training
python train.py --data-dir data --model-dir models
```

## Troubleshooting

### Selenium not working

```bash
# Update Chrome
# On Ubuntu:
sudo apt update
sudo apt install chromium-browser chromium-chromedriver

# On Windows: Download Chrome from google.com/chrome
```

### Downloads failing

- Check your internet connection
- Try reducing `--days` (fewer files at once)
- Run without `--headless` to see what's happening
- Bybit may have changed their website - the script may need updates

### Page elements not found

The Bybit website may have changed. Update the XPath selectors in `download_bybit_historical.py`:

```python
# Look for elements in the browser developer tools
# Update the XPATH queries accordingly
```

## Alternative: Use Klines API

If the website scraper is too fragile, use the klines API instead:

```bash
# This is more reliable and doesn't need Selenium
python scripts/download_klines_api.py --symbol BTCUSDT --days 60
```

(Create this script if needed - it uses the pybit API directly)
