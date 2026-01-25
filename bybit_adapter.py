import logging
import time
import threading
from typing import Dict, List, Optional, Callable
from pybit.unified_trading import HTTP, WebSocket
from interfaces import ExchangeInterface
from local_orderbook import LocalOrderbook

class BybitAdapter(ExchangeInterface):
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.logger = logging.getLogger("BybitAdapter")
        self.session = HTTP(
            testnet=testnet,
            api_key=api_key,
            api_secret=api_secret,
            recv_window=60000, # Increased from 20s to 60s for better stability
        )
        # Note: If pybit version supports it, time_sync=True is even better.
        # But recv_window=60000 is the most compatible way to handle minor drift.
        self.testnet = testnet

    def check_clock_drift(self) -> float:
        """Returns drift in seconds (local - server). Warnings if > 1s."""
        try:
            local_before = time.time() * 1000
            response = self.session.get_server_time()
            local_after = time.time() * 1000
            
            if response['retCode'] == 0:
                server_ms = int(response['result']['timeNano']) / 1_000_000
                # Estimate local time at the moment the server responded
                avg_local_ms = (local_before + local_after) / 2
                drift_ms = avg_local_ms - server_ms
                drift_sec = drift_ms / 1000.0
                
                if abs(drift_sec) > 1.0:
                    self.logger.warning(f"!!! CLOCK DRIFT DETECTED: {drift_sec:.3f}s !!!")
                    self.logger.warning("Please sync your OS clock with NTP for better stability.")
                else:
                    self.logger.info(f"Clock Sync OK (Drift: {drift_sec:.3f}s)")
                return drift_sec
            return 0.0
        except Exception as e:
            self.logger.error(f"Could not check clock drift: {e}")
            return 0.0

    def _map_interval(self, interval: str) -> str:
        mapping = {
            "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
            "1h": "60", "2h": "120", "4h": "240", "6h": "360", "12h": "720",
            "1d": "D", "1w": "W", "1M": "M"
        }
        return mapping.get(interval, interval)

    def _retry_api_call(self, func, *args, **kwargs):
        """Internal helper to retry calls that hit rate limits."""
        max_retries = 3
        for i in range(max_retries):
            try:
                response = func(*args, **kwargs)
                
                # Check if Bybit returned a "Request too frequent" error
                if isinstance(response, dict) and response.get('retCode') == 10002:
                    if i < max_retries - 1:
                        sleep_time = (i + 1) * 2
                        self.logger.warning(f"Rate limit hit (10002). Retrying in {sleep_time}s...")
                        time.sleep(sleep_time)
                        continue
                return response
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "10002" in err_str or "Rate limit" in err_str:
                    if i < max_retries - 1:
                        sleep_time = (i + 1) * 2
                        self.logger.warning(f"Rate limit Exception. Retrying in {sleep_time}s...")
                        time.sleep(sleep_time)
                        continue
                raise e
        return func(*args, **kwargs) # Final attempt

    def get_public_klines(self, symbol: str, interval: str, limit: int = 200, start_time: Optional[int] = None) -> List[Dict]:
        try:
            bybit_interval = self._map_interval(interval)
            all_batches = []
            
            target = limit
            end_ts = None
            
            while sum(len(b) for b in all_batches) < target:
                fetch_limit = min(target - sum(len(b) for b in all_batches), 1000)
                params = {
                    "category": "linear",
                    "symbol": symbol,
                    "interval": bybit_interval,
                    "limit": fetch_limit
                }
                if end_ts:
                    params["end"] = end_ts
                if start_time:
                    params["start"] = start_time
                
                response = self._retry_api_call(self.session.get_kline, **params)
                
                if response['retCode'] != 0 or not response['result']['list']:
                    break
                
                batch = []
                for item in response['result']['list']:
                    batch.append({
                        'timestamp': int(item[0]),
                        'open': float(item[1]),
                        'high': float(item[2]),
                        'low': float(item[3]),
                        'close': float(item[4]),
                        'volume': float(item[5])
                    })
                
                # Batch is newest -> oldest. We store it as is for now.
                all_batches.append(batch)
                # Next request should end just before the oldest bar in this batch
                end_ts = batch[-1]['timestamp'] - 1
                
                if len(batch) < fetch_limit:
                    break # No more data available
            
            # Combine all batches (they are in order: [newest_batch, middle_batch, oldest_batch])
            # Each batch is internally [newest_bar, ..., oldest_bar]
            full_list = []
            for b in all_batches:
                full_list.extend(b)
            
            # Final result needs to be oldest -> newest
            full_list.reverse()
            return full_list
        except Exception as e:
            err_msg = str(e).encode('ascii', 'ignore').decode('ascii')
            self.logger.error(f"Exception fetching klines: {err_msg}")
            return []

    def get_orderbook(self, symbol: str, limit: int = 50) -> Dict:
        try:
            response = self._retry_api_call(
                self.session.get_orderbook,
                category="linear",
                symbol=symbol,
                limit=limit
            )
            if response['retCode'] != 0:
                self.logger.error(f"Bybit Orderbook Error: {response['retMsg']}")
                return {}

            result = response['result']
            return {
                'bids': [[float(i[0]), float(i[1])] for i in result['b']],
                'asks': [[float(i[0]), float(i[1])] for i in result['a']],
                'timestamp': int(result['ts'])
            }
        except Exception as e:
            self.logger.error(f"Exception fetching orderbook: {e}")
            return []

    def get_current_price(self, symbol: str) -> float:
        try:
            response = self._retry_api_call(self.session.get_tickers, category="linear", symbol=symbol)
            if response['retCode'] == 0 and response['result']['list']:
                return float(response['result']['list'][0]['lastPrice'])
            return 0.0
        except Exception as e:
            self.logger.error(f"Exception fetching price: {e}")
            return 0.0

    def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        try:
            kwargs = {'category': 'linear', 'settleCoin': 'USDT'}
            if symbol:
                kwargs['symbol'] = symbol
            
            response = self._retry_api_call(self.session.get_positions, **kwargs)
            if response['retCode'] != 0:
                self.logger.error(f"Bybit Positions Error: {response['retMsg']}")
                return []

            positions = []
            for item in response['result']['list']:
                size = float(item['size'])
                if size > 0:
                    positions.append({
                        'symbol': item['symbol'],
                        'side': item['side'],
                        'size': size,
                        'entry_price': float(item['avgPrice']),
                        'unrealized_pnl': float(item['unrealisedPnl']),
                        'position_idx': int(item['positionIdx']) 
                    })
            return positions
        except Exception as e:
            self.logger.error(f"Exception fetching positions: {e}")
            return []

    def get_wallet_balance(self, coin: str = "USDT") -> float:
        try:
            response = self._retry_api_call(self.session.get_wallet_balance, accountType="UNIFIED", coin=coin)
            if response['retCode'] == 0:
                acct = response['result']['list'][0]
                for c in acct['coin']:
                    if c['coin'] == coin:
                        return float(c['walletBalance'])
            
            response = self._retry_api_call(self.session.get_wallet_balance, accountType="CONTRACT", coin=coin)
            if response['retCode'] == 0 and response['result']['list']:
                 acct = response['result']['list'][0]
                 for c in acct['coin']:
                    if c['coin'] == coin:
                        return float(c['walletBalance'])
            return 0.0
        except Exception as e:
            self.logger.error(f"Exception fetching balance: {e}")
            return 0.0

    def get_open_orders(self, symbol: str) -> List[Dict]:
        try:
            # Fetch active orders (Limit, Market if pending)
            response = self._retry_api_call(self.session.get_open_orders, category="linear", symbol=symbol)
            orders = []
            if response['retCode'] == 0:
                orders.extend(response['result']['list'])
            
            return orders
        except Exception as e:
            self.logger.error(f"Exception fetching open orders: {e}")
            return []

    def get_instrument_info(self, symbol: str) -> Dict:
        try:
            response = self._retry_api_call(self.session.get_instruments_info, category="linear", symbol=symbol)
            if response['retCode'] == 0 and response['result']['list']:
                info = response['result']['list'][0]
                return {
                    'tick_size': float(info['priceFilter']['tickSize']),
                    'qty_step': float(info['lotSizeFilter']['qtyStep']),
                    'min_qty': float(info['lotSizeFilter']['minOrderQty']),
                    'min_notional': float(info['lotSizeFilter'].get('minNotionalValue', 0))
                }
            return {}
        except Exception as e:
            self.logger.error(f"Exception fetching instrument info: {e}")
            return {}

    def place_order(self, symbol: str, side: str, order_type: str, qty: float, price: Optional[float] = None, reduce_only: bool = False, sl: Optional[float] = None, tp: Optional[float] = None, trigger_price: Optional[float] = None, position_idx: Optional[int] = None, trigger_direction: Optional[int] = None, order_link_id: Optional[str] = None) -> Dict:
        try:
            side = side.capitalize()
            order_type = order_type.capitalize()
            
            params = {
                "category": "linear",
                "symbol": symbol,
                "side": side,
                "orderType": order_type,
                "qty": str(qty),
                "reduceOnly": reduce_only
            }
            if price:
                params["price"] = str(price)
            if sl:
                params["stopLoss"] = str(sl)
            if tp:
                params["takeProfit"] = str(tp)
            if trigger_price:
                params["triggerPrice"] = str(trigger_price)
            if position_idx is not None:
                params["positionIdx"] = position_idx
            if trigger_direction:
                params["triggerDirection"] = trigger_direction
            if order_link_id:
                params["orderLinkId"] = order_link_id

            response = self._retry_api_call(self.session.place_order, **params)
            
            if response['retCode'] != 0:
                self.logger.error(f"Bybit Place Order Error: {response['retMsg']}")
                return {'error': response['retMsg']}
                
            return {
                'order_id': response['result']['orderId'],
                'order_link_id': response['result']['orderLinkId']
            }
        except Exception as e:
            # Sanitize error message to avoid UnicodeEncodeError on Windows
            err_msg = str(e).encode('ascii', 'ignore').decode('ascii')
            self.logger.error(f"Exception placing order: {err_msg}")
            return {'error': err_msg}

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        try:
            response = self._retry_api_call(self.session.cancel_order, category="linear", symbol=symbol, orderId=order_id)
            return response['retCode'] == 0
        except Exception as e:
            err_msg = str(e).encode('ascii', 'ignore').decode('ascii')
            self.logger.error(f"Exception cancelling order: {err_msg}")
            return False

    def cancel_all_orders(self, symbol: str) -> bool:
        try:
            response = self._retry_api_call(self.session.cancel_all_orders, category="linear", symbol=symbol)
            return response['retCode'] == 0
        except Exception as e:
            err_msg = str(e).encode('ascii', 'ignore').decode('ascii')
            self.logger.error(f"Exception cancelling all orders: {err_msg}")
            return False

    def set_leverage(self, symbol: str, leverage: float) -> bool:
        try:
            self.session.set_leverage(
                category="linear",
                symbol=symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage)
            )
            return True
        except Exception as e:
            # Code 110043 means leverage not modified (already set)
            if "110043" in str(e):
                return True
            self.logger.error(f"Set Leverage Error: {e}")
            return False

    def set_trading_stop(self, symbol: str, position_idx: int, sl: Optional[float] = None, tp: Optional[float] = None) -> bool:
        try:
            params = {
                "category": "linear",
                "symbol": symbol,
                "positionIdx": position_idx
            }
            if sl is not None: params["stopLoss"] = str(sl)
            if tp is not None: params["takeProfit"] = str(tp)
            
            response = self._retry_api_call(self.session.set_trading_stop, **params)
            return response['retCode'] == 0
        except Exception as e:
            # Code 34040: SL/TP too close or same as current (often happens when clearing 0 -> 0)
            if "34040" in str(e): 
                return True
            self.logger.error(f"Set Trading Stop Error: {e}")
            return False

    def switch_position_mode(self, symbol: str, mode: int = 3) -> bool:
        # mode 0: One-Way, 3: Hedge
        try:
            self.session.switch_position_mode(
                category="linear",
                symbol=symbol,
                mode=mode,
                coin="USDT"
            )
            return True
        except Exception as e:
            if "110025" in str(e): # Position mode not modified
                return True
            self.logger.error(f"Switch Position Mode Error: {e}")
            return False

class BybitWSAdapter:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.logger = logging.getLogger("BybitWSAdapter")
        self.symbol = None
        self.local_book = None
        self.connected = False
        
        try:
            # Public WS (Linear)
            self.ws_public = WebSocket(
                testnet=testnet,
                channel_type="linear"
            )
            
            # Private WS
            self.ws_private = WebSocket(
                testnet=testnet,
                channel_type="private",
                api_key=api_key,
                api_secret=api_secret
            )
            self.connected = True
        except Exception as e:
            self.logger.error(f"WS Connection Failed: {e}")
            self.ws_public = None
            self.ws_private = None
        
        # State containers
        self.latest_kline = None
        self.position_updates = []
        self.execution_updates = []
        
    def subscribe_trades(self, symbol: str, callback: Callable):
        if not self.ws_public: return
        self.ws_public.trade_stream(
            symbol=symbol,
            callback=callback
        )
        self.logger.info(f"Subscribed to Trade Stream for {symbol}")

    def subscribe_orderbook(self, symbol: str, local_book: LocalOrderbook):
        if not self.ws_public: return
        self.symbol = symbol
        self.local_book = local_book
        
        def handle_ob(message):
            try:
                self.local_book.apply_update(message)
            except Exception as e:
                self.logger.error(f"OB Update Error: {e}")

        self.ws_public.orderbook_stream(
            depth=50,
            symbol=symbol,
            callback=handle_ob
        )
        self.logger.info(f"Subscribed to Orderbook for {symbol}")

    def subscribe_kline(self, symbol: str, interval: str):
        if not self.ws_public: return
        # Interval map: 5m -> 5
        map_int = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}
        bybit_int = map_int.get(interval, 5)
        
        def handle_kline(message):
            data = message.get('data', [])
            if data:
                self.latest_kline = data[0]

        self.ws_public.kline_stream(
            interval=bybit_int,
            symbol=symbol,
            callback=handle_kline
        )
        self.logger.info(f"Subscribed to Kline {interval} for {symbol}")

    def subscribe_private(self, on_position=None, on_execution=None):
        if not self.ws_private: return
        def handle_pos(message):
            if on_position: on_position(message)
            
        def handle_exec(message):
            if on_execution: on_execution(message)

        self.ws_private.position_stream(callback=handle_pos)
        self.ws_private.execution_stream(callback=handle_exec)
        self.logger.info("Subscribed to Private Channels")

    def disconnect(self):
        # pybit manages its own threads
        pass
