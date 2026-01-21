from typing import Dict, List, Optional
import logging

class LocalOrderbook:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bids: Dict[float, float] = {} # price -> size
        self.asks: Dict[float, float] = {} # price -> size
        self.timestamp = 0
        self.initialized = False
        self.logger = logging.getLogger(f"LocalBook_{symbol}")

    def apply_update(self, data: Dict):
        """
        Apply Bybit V5 Orderbook update (snapshot or delta).
        """
        msg_type = data.get('type')
        ts = data.get('ts')
        self.timestamp = int(ts) if ts else 0
        
        book_data = data.get('data', {})
        bids = book_data.get('b', [])
        asks = book_data.get('a', [])

        if msg_type == 'snapshot':
            self.bids = {float(p): float(s) for p, s in bids}
            self.asks = {float(p): float(s) for p, s in asks}
            self.initialized = True
            
        elif msg_type == 'delta':
            if not self.initialized:
                # We missed the snapshot? Wait for next re-sub or just process
                # Usually snapshot comes first. If not, this might be partial data.
                return 

            # Update Bids
            for p, s in bids:
                price = float(p)
                size = float(s)
                if size == 0:
                    if price in self.bids:
                        del self.bids[price]
                else:
                    self.bids[price] = size

            # Update Asks
            for p, s in asks:
                price = float(p)
                size = float(s)
                if size == 0:
                    if price in self.asks:
                        del self.asks[price]
                else:
                    self.asks[price] = size

    def get_snapshot(self, limit: int = 50) -> Dict:
        """
        Return sorted Bids/Asks lists [[price, size], ...]
        """
        if not self.initialized:
            return {}

        # Sort Bids (Desc)
        sorted_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)[:limit]
        # Sort Asks (Asc)
        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])[:limit]
        
        # Convert tuples back to lists if needed, keeping float types
        return {
            'bids': [[p, s] for p, s in sorted_bids],
            'asks': [[p, s] for p, s in sorted_asks],
            'timestamp': self.timestamp
        }
