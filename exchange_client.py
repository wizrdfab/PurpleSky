"""
Bybit Exchange Client (Unified Trading).
Handles connectivity, data fetching, and order execution.
"""
from pybit.unified_trading import HTTP
from pybit.exceptions import InvalidRequestError
import pandas as pd
import time
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("ExchangeClient")

class ExchangeClient:
    def __init__(self, api_key: str, api_secret: str, symbol: str, testnet: bool = False):
        self.symbol = symbol
        self.session = HTTP(
            testnet=testnet,
            api_key=api_key,
            api_secret=api_secret,
        )
        print(f"Connected to Bybit {'Testnet' if testnet else 'Mainnet'} for {symbol}")

    def fetch_recent_trades(self, limit: int = 1000) -> pd.DataFrame:
        """
        Fetch recent trades to build bars.
        """
        try:
            resp = self.session.get_public_trade_history(
                category="linear",
                symbol=self.symbol,
                limit=limit
            )
            data = resp.get('result', {}).get('list', [])
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            # Standardize columns to match DataConfig
            # Bybit returns: id, symbol, price, size, side, time, isBlockTrade
            df['timestamp'] = df['time'].astype(float) / 1000.0 # Convert ms to seconds
            df['price'] = df['price'].astype(float)
            df['size'] = df['size'].astype(float)
            df['side'] = df['side'] # 'Buy' / 'Sell'
            
            return df.sort_values('timestamp')
            
        except Exception as e:
            logger.error(f"Error fetching trades: {e}")
            return pd.DataFrame()

    def fetch_kline(self, interval: str = "15", limit: int = 200) -> pd.DataFrame:
        """
        Fetch historical OHLCV klines for bootstrapping.
        Interval: '1', '3', '5', '15', '30', '60', '120', '240', '360', '720', 'D', 'M', 'W'
        """
        try:
            resp = self.session.get_kline(
                category="linear",
                symbol=self.symbol,
                interval=interval,
                limit=limit
            )
            data = resp.get('result', {}).get('list', [])
            # Bybit returns list of lists: [startTime, open, high, low, close, volume, turnover]
            # And it returns them in REVERSE order (newest first).
            if not data:
                return pd.DataFrame()
            
            # Columns: startTime, open, high, low, close, volume, turnover
            df = pd.DataFrame(data, columns=['startTime', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df['timestamp'] = df['startTime'].astype(float) / 1000.0
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            df['turnover'] = df['turnover'].astype(float)
            
            # Sort ascending (Oldest first)
            df = df.sort_values('timestamp').reset_index(drop=True)
            return df
            
        except Exception as e:
            logger.error(f"Error fetching klines: {e}")
            return pd.DataFrame()

    def fetch_orderbook(self, limit: int = 50) -> Dict:
        """
        Fetch current L2 Orderbook snapshot.
        """
        try:
            resp = self.session.get_orderbook(
                category="linear",
                symbol=self.symbol,
                limit=limit
            )
            # Returns {'s': symbol, 'b': [[p, s], ...], 'a': [[p, s], ...], 'ts': timestamp}
            return resp.get('result', {})
        except Exception as e:
            logger.error(f"Error fetching orderbook: {e}")
            return {}

    def _safe_float(self, value, default=0.0) -> float:
        """Safely convert string/None to float."""
        try:
            if value is None or value == "":
                return default
            return float(value)
        except Exception:
            return default

    def get_position(self) -> float:
        """Returns current position size (Signed: + for Long, - for Short)."""
        try:
            resp = self.session.get_positions(
                category="linear",
                symbol=self.symbol
            )
            # Unified account returns list
            positions = resp.get('result', {}).get('list', [])
            for p in positions:
                if p['symbol'] == self.symbol:
                    size = self._safe_float(p.get('size'))
                    side = p.get('side') # 'Buy' or 'Sell'
                    if side == 'Sell':
                        return -size
                    return size
            return 0.0
        except Exception as e:
            logger.error(f"Error getting position: {e}")
            return 0.0

    def get_wallet_balance(self) -> Dict[str, float]:
        """Get Unified Account Balance with Fallback."""
        try:
            resp = self.session.get_wallet_balance(
                accountType="UNIFIED",
                coin="USDT"
            )
            result = resp.get('result', {})
            acct_list = result.get('list', [])
            
            if not acct_list:
                return {'equity': 0.0, 'available': 0.0, 'total_balance': 0.0}
                
            acct = acct_list[0]
            
            # Account Level
            total_equity = self._safe_float(acct.get('totalEquity'))
            total_avail = self._safe_float(acct.get('totalAvailableBalance'))
            
            # Coin Level
            coins = acct.get('coin', [])
            coin_bal = 0.0
            
            if coins:
                c = coins[0]
                # Log raw for debugging (one-time or verbose)
                # logger.info(f"Raw Coin Data: {c}")
                coin_bal = self._safe_float(c.get('walletBalance'))
                
                # Fallback Logic
                if total_avail == 0 and total_equity > 0:
                    # Sometimes availableToWithdraw is hidden deep
                    # Try other fields
                    avail_withdraw = self._safe_float(c.get('availableToWithdraw'))
                    if avail_withdraw > 0:
                        total_avail = avail_withdraw
                    else:
                        # Final Fallback: If no pos, Equity ~= Available
                        # This is a heuristic.
                        total_avail = total_equity

            return {
                'equity': total_equity,
                'available': total_avail,
                'total_balance': coin_bal
            }
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return {'equity': 0.0, 'available': 0.0, 'total_balance': 0.0}

    def get_open_orders(self) -> List[Dict]:
        """Get active limit orders."""
        try:
            resp = self.session.get_open_orders(
                category="linear",
                symbol=self.symbol
            )
            return resp.get('result', {}).get('list', [])
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return []

    def fetch_instrument_info(self) -> Dict:
        """Fetch instrument rules (min qty, step size, tick size)."""
        try:
            resp = self.session.get_instruments_info(
                category="linear",
                symbol=self.symbol
            )
            # result -> list[0] -> lotSizeFilter -> minOrderQty, qtyStep
            data = resp.get('result', {}).get('list', [])
            if not data: return {}
            
            info = data[0]
            lot_filter = info.get('lotSizeFilter', {})
            price_filter = info.get('priceFilter', {})
            
            return {
                'min_qty': float(lot_filter.get('minOrderQty', 0.0)),
                'qty_step': float(lot_filter.get('qtyStep', 0.0)),
                'tick_size': float(price_filter.get('tickSize', 0.0))
            }
        except Exception as e:
            logger.error(f"Error fetching instrument info: {e}")
            return {}

    def place_limit_order(self, side: str, price: float, qty: float, reduce_only: bool = False):
        """Place a Limit order."""
        try:
            resp = self.session.place_order(
                category="linear",
                symbol=self.symbol,
                side=side, # "Buy" or "Sell"
                orderType="Limit",
                qty=str(qty),
                price=str(price),
                timeInForce="PostOnly", # Maker only
                reduceOnly=reduce_only
            )
            return resp
        except Exception as e:
            # Sanitize error message for Windows consoles
            msg = str(e).encode('ascii', 'ignore').decode('ascii')
            logger.error(f"Error placing order: {msg}")
            return None

    def cancel_all_orders(self):
        """Cancel all open orders for symbol."""
        try:
            self.session.cancel_all_orders(
                category="linear",
                symbol=self.symbol
            )
        except Exception as e:
            logger.error(f"Error canceling orders: {e}")

    def market_close(self, side: str, qty: float):
        """Immediately close position at market price."""
        try:
            # If we are Long, we need to Sell Market
            # If we are Short, we need to Buy Market
            side_to_send = "Sell" if side == "Buy" else "Buy"
            self.session.place_order(
                category="linear",
                symbol=self.symbol,
                side=side_to_send,
                orderType="Market",
                qty=str(qty),
                reduceOnly=True
            )
            logger.info(f"Market Close sent: {side_to_send} {qty}")
        except Exception as e:
            logger.error(f"Error market closing: {e}")

    def place_tp_sl(self, side: str, qty: float, tp: float, sl: float):
        """
        Update Position TP/SL or Place Reduce-Only Limit (TP) and Conditional Market (SL).
        For Unified, setting TP/SL on position is easiest.
        """
        try:
            # Set TP/SL for the entire position mode
            self.session.set_trading_stop(
                category="linear",
                symbol=self.symbol,
                takeProfit=str(tp),
                stopLoss=str(sl),
                positionIdx=0 # 0 for One-Way Mode
            )
        except Exception as e:
            logger.error(f"Error setting TP/SL: {e}")