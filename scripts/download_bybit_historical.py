#!/usr/bin/env python3
"""
Download historical data from Bybit website using Selenium

Requirements:
    pip install selenium

Chrome/Chromium browser must be installed
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
import argparse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BybitHistoricalDownloader:
    """Download historical data from Bybit website"""

    URL = "https://www.bybit.com/derivatives/en/history-data"

    def __init__(self, download_dir: str = "./data/bybit_downloads", headless: bool = False):
        self.download_dir = Path(download_dir).absolute()
        self.download_dir.mkdir(parents=True, exist_ok=True)

        # Setup Chrome options
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")

        # Set download directory
        prefs = {
            "download.default_directory": str(self.download_dir),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        }
        chrome_options.add_experimental_option("prefs", prefs)

        # Additional options for stability
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")

        self.driver = webdriver.Chrome(options=chrome_options)
        self.wait = WebDriverWait(self.driver, 20)

        logger.info(f"Download directory: {self.download_dir}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the browser"""
        if self.driver:
            self.driver.quit()

    def wait_for_downloads(self, timeout: int = 60):
        """Wait for all downloads to complete"""
        end_time = time.time() + timeout

        while time.time() < end_time:
            # Check for .crdownload files (Chrome temporary download files)
            downloading = list(self.download_dir.glob("*.crdownload"))
            if not downloading:
                time.sleep(1)  # Extra wait to ensure file is written
                return True

            logger.info(f"Waiting for {len(downloading)} downloads to complete...")
            time.sleep(2)

        logger.warning("Download timeout reached")
        return False

    def download_single_date(
        self,
        product_type: str = "USDT Perpetual",
        data_type: str = "Trading",
        symbol: str = "BTCUSDT",
        date: datetime = None
    ):
        """
        Download data for a single date

        Args:
            product_type: "USDT Perpetual", "Inverse Perpetual", "Spot", etc.
            data_type: "Trading", "Funding Rate", "Premium Index", etc.
            symbol: Trading pair symbol
            date: Date to download (default: yesterday)
        """
        if date is None:
            date = datetime.now() - timedelta(days=1)

        date_str = date.strftime("%Y-%m-%d")
        logger.info(f"Downloading {symbol} {data_type} data for {date_str}")

        try:
            # Load the page
            logger.info("Loading Bybit historical data page...")
            self.driver.get(self.URL)
            time.sleep(3)  # Wait for page to load

            # Step 1: Select Product Type
            logger.info(f"Selecting product type: {product_type}")
            self._select_dropdown("Product Type", product_type)
            time.sleep(1)

            # Step 2: Select Data Type
            logger.info(f"Selecting data type: {data_type}")
            self._select_dropdown("Data Type", data_type)
            time.sleep(1)

            # Step 3: Select Symbol
            logger.info(f"Selecting symbol: {symbol}")
            self._select_dropdown("Symbol", symbol)
            time.sleep(1)

            # Step 4: Select Date
            logger.info(f"Selecting date: {date_str}")
            self._select_date(date)
            time.sleep(1)

            # Step 5: Click Download button
            logger.info("Clicking download button...")
            download_btn = self.wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Download')]"))
            )
            download_btn.click()

            # Wait for download to complete
            logger.info("Waiting for download to complete...")
            if self.wait_for_downloads(timeout=120):
                logger.info(f"✓ Successfully downloaded {symbol} for {date_str}")
                return True
            else:
                logger.error(f"✗ Download failed or timed out for {date_str}")
                return False

        except TimeoutException:
            logger.error(f"Timeout while downloading {date_str}")
            return False
        except Exception as e:
            logger.error(f"Error downloading {date_str}: {e}")
            return False

    def _select_dropdown(self, label: str, value: str):
        """Select a value from a dropdown"""
        try:
            # Find dropdown by label
            dropdown = self.wait.until(
                EC.element_to_be_clickable((
                    By.XPATH,
                    f"//div[contains(@class, 'select') and .//span[contains(text(), '{label}')]]"
                ))
            )
            dropdown.click()
            time.sleep(0.5)

            # Click the option
            option = self.wait.until(
                EC.element_to_be_clickable((
                    By.XPATH,
                    f"//li[contains(text(), '{value}')] | //div[contains(text(), '{value}')]"
                ))
            )
            option.click()

        except Exception as e:
            logger.warning(f"Could not select {label}={value}: {e}")
            # Try alternative method - direct text search
            try:
                option = self.driver.find_element(By.XPATH, f"//*[text()='{value}']")
                option.click()
            except:
                raise

    def _select_date(self, date: datetime):
        """Select a date from the date picker"""
        try:
            # Click date input to open picker
            date_input = self.wait.until(
                EC.element_to_be_clickable((By.XPATH, "//input[@type='text' and @placeholder]"))
            )
            date_input.click()
            time.sleep(0.5)

            # Find and click the specific date
            date_str = str(date.day)
            date_element = self.wait.until(
                EC.element_to_be_clickable((
                    By.XPATH,
                    f"//td[contains(@class, 'available') and .//div[text()='{date_str}']]"
                ))
            )
            date_element.click()

        except Exception as e:
            logger.warning(f"Could not select date: {e}")
            # Try typing the date directly
            try:
                date_input.clear()
                date_input.send_keys(date.strftime("%Y-%m-%d"))
            except:
                raise

    def download_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        product_type: str = "USDT Perpetual",
        data_type: str = "Trading",
        symbol: str = "BTCUSDT"
    ):
        """
        Download data for a date range

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            product_type: Product type to download
            data_type: Data type to download
            symbol: Trading pair symbol
        """
        current = start_date
        success_count = 0
        fail_count = 0

        while current <= end_date:
            try:
                if self.download_single_date(product_type, data_type, symbol, current):
                    success_count += 1
                else:
                    fail_count += 1

                current += timedelta(days=1)

                # Add delay between downloads
                if current <= end_date:
                    logger.info("Waiting 3 seconds before next download...")
                    time.sleep(3)

            except KeyboardInterrupt:
                logger.info("Download interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error on {current}: {e}")
                fail_count += 1
                current += timedelta(days=1)

        logger.info(f"\n{'='*60}")
        logger.info(f"Download Summary:")
        logger.info(f"  Successful: {success_count}")
        logger.info(f"  Failed: {fail_count}")
        logger.info(f"  Total: {success_count + fail_count}")
        logger.info(f"  Files saved to: {self.download_dir}")
        logger.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Download historical data from Bybit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download 1 day of BTCUSDT trading data
    python download_bybit_historical.py --symbol BTCUSDT --days 1

    # Download 30 days of ETHUSDT data
    python download_bybit_historical.py --symbol ETHUSDT --days 30

    # Download specific date range
    python download_bybit_historical.py --symbol BTCUSDT --start 2024-11-01 --end 2024-11-30

    # Download with headless browser (no GUI)
    python download_bybit_historical.py --symbol BTCUSDT --days 7 --headless
        """
    )

    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair symbol")
    parser.add_argument("--product", default="USDT Perpetual", help="Product type")
    parser.add_argument("--data-type", default="Trading", help="Data type (Trading, Funding Rate, etc.)")
    parser.add_argument("--days", type=int, help="Number of days to download (from yesterday backwards)")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", default="./data/bybit_downloads", help="Download directory")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")

    args = parser.parse_args()

    # Determine date range
    if args.start and args.end:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
    elif args.days:
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=args.days - 1)
    else:
        # Default: yesterday only
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date

    logger.info(f"Downloading {args.symbol} from {start_date.date()} to {end_date.date()}")

    # Download
    with BybitHistoricalDownloader(download_dir=args.output_dir, headless=args.headless) as downloader:
        downloader.download_date_range(
            start_date=start_date,
            end_date=end_date,
            product_type=args.product,
            data_type=args.data_type,
            symbol=args.symbol
        )


if __name__ == "__main__":
    main()
