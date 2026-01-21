from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

class ExchangeInterface(ABC):
    @abstractmethod
    def get_public_klines(self, symbol: str, interval: str, limit: int = 200) -> List[Dict]:
        """
        Fetch OHLCV data.
        Returns list of dicts: {'timestamp': int (ms), 'open': float, 'high': float, 'low': float, 'close': float, 'volume': float}
        """
        pass

    @abstractmethod
    def get_orderbook(self, symbol: str, limit: int = 50) -> Dict:
        """
        Fetch current orderbook.
        Returns: {'bids': [[price, size], ...], 'asks': [[price, size], ...], 'timestamp': int}
        """
        pass

    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """Fetch latest trade price or mid price."""
        pass

    @abstractmethod
    def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Fetch active positions.
        Returns list of dicts: {'symbol': str, 'side': str (Buy/Sell), 'size': float, 'entry_price': float, 'unrealized_pnl': float}
        """
        pass

    @abstractmethod
    def get_wallet_balance(self, coin: str = "USDT") -> float:
        """Fetch available balance for a specific coin."""
        pass

    @abstractmethod
    def place_order(self, symbol: str, side: str, order_type: str, qty: float, price: Optional[float] = None, reduce_only: bool = False, sl: Optional[float] = None, tp: Optional[float] = None) -> Dict:
        """
        Place a new order.
        side: 'Buy' or 'Sell'
        order_type: 'Market' or 'Limit'
        Returns raw exchange response or simplified dict with 'order_id'.
        """
        pass

    @abstractmethod
    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel a specific order."""
        pass

    @abstractmethod
    def cancel_all_orders(self, symbol: str) -> bool:
        """Cancel all active orders for the symbol."""
        pass
