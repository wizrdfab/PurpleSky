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
            recv_window=20000 
        )
        self.testnet = testnet

    def _map_interval(self, interval: str) -> str:
        mapping = {
            "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
            "1h": "60", "2h": "120", "4h": "240", "6h": "360", "12h": "720",
            "1d": "D", "1w": "W", "1M": "M"
        }
        return mapping.get(interval, interval)

    def get_public_klines(self, symbol: str, interval: str, limit: int = 200) -> List[Dict]:
        try:
            bybit_interval = self._map_interval(interval)
            response = self.session.get_kline(
                category="linear",
                symbol=symbol,
                interval=bybit_interval,
                limit=limit
            )
            
            if response['retCode'] != 0:
                self.logger.error(f"Bybit Kline Error: {response['retMsg']}")
                return []

            klines = []
            for item in response['result']['list']:
                klines.append({
                    'timestamp': int(item[0]),
                    'open': float(item[1]),
                    'high': float(item[2]),
                    'low': float(item[3]),
                    'close': float(item[4]),
                    'volume': float(item[5])
                })
            
            klines.reverse()
            return klines
        except Exception as e:
            self.logger.error(f"Exception fetching klines: {e}")
            return []

    def get_orderbook(self, symbol: str, limit: int = 50) -> Dict:
        try:
            response = self.session.get_orderbook(
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
            return {}

    def get_current_price(self, symbol: str) -> float:
        try:
            response = self.session.get_tickers(category="linear", symbol=symbol)
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
            
            response = self.session.get_positions(**kwargs)
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
            response = self.session.get_wallet_balance(accountType="UNIFIED", coin=coin)
            if response['retCode'] == 0:
                acct = response['result']['list'][0]
                for c in acct['coin']:
                    if c['coin'] == coin:
                        return float(c['walletBalance'])
            
            response = self.session.get_wallet_balance(accountType="CONTRACT", coin=coin)
            if response['retCode'] == 0 and response['result']['list']:
                 acct = response['result']['list'][0]
                 for c in acct['coin']:
                    if c['coin'] == coin:
                        return float(c['walletBalance'])
            return 0.0
        except Exception as e:
            self.logger.error(f"Exception fetching balance: {e}")
            return 0.0

    def place_order(self, symbol: str, side: str, order_type: str, qty: float, price: Optional[float] = None, reduce_only: bool = False, sl: Optional[float] = None, tp: Optional[float] = None, trigger_price: Optional[float] = None) -> Dict:
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

            response = self.session.place_order(**params)
            
            if response['retCode'] != 0:
                self.logger.error(f"Bybit Place Order Error: {response['retMsg']}")
                return {'error': response['retMsg']}
                
            return {
                'order_id': response['result']['orderId'],
                'order_link_id': response['result']['orderLinkId']
            }
        except Exception as e:
            self.logger.error(f"Exception placing order: {e}")
            return {'error': str(e)}

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        try:
            response = self.session.cancel_order(category="linear", symbol=symbol, orderId=order_id)
            return response['retCode'] == 0
        except Exception as e:
            self.logger.error(f"Exception cancelling order: {e}")
            return False

    def cancel_all_orders(self, symbol: str) -> bool:
        try:
            response = self.session.cancel_all_orders(category="linear", symbol=symbol)
            return response['retCode'] == 0
        except Exception as e:
            self.logger.error(f"Exception cancelling all orders: {e}")
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
