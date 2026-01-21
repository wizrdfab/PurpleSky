import time
import math
from typing import Dict, List, Optional

class OrderbookAggregator:
    def __init__(self, ob_levels: int = 50):
        self.ob_levels = ob_levels
        self.snapshots = []
        self.unique_timestamps = set()
        self.last_ts = 0

    def process_snapshot(self, bids: List[List[float]], asks: List[List[float]], timestamp: int):
        """
        Process a raw snapshot (bids/asks: [[price, size], ...])
        """
        if timestamp in self.unique_timestamps and timestamp != 0:
            return # Skip duplicate work if data hasn't changed
            
        self.last_ts = timestamp
        self.unique_timestamps.add(timestamp)
        
        # Sort and Slice
        # Bids: Descending Price
        bids = sorted(bids, key=lambda x: float(x[0]), reverse=True)[:self.ob_levels]
        # Asks: Ascending Price
        asks = sorted(asks, key=lambda x: float(x[0]))[:self.ob_levels]

        if not bids or not asks:
            return

        bb_p = float(bids[0][0])
        bb_s = float(bids[0][1])
        ba_p = float(asks[0][0])
        ba_s = float(asks[0][1])

        spread = ba_p - bb_p
        mid_price = (ba_p + bb_p) / 2
        micro_price = (ba_p * bb_s + bb_p * ba_s) / (bb_s + ba_s + 1e-9)
        micro_dev = micro_price - mid_price

        bid_depth = sum(float(x[1]) for x in bids)
        ask_depth = sum(float(x[1]) for x in asks)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-9)

        # Slope
        bid_slope = 0.0
        if bid_depth > 0 and len(bids) > 1:
            deepest_bid = float(bids[-1][0])
            bid_slope = (bb_p - deepest_bid) / bid_depth
        
        ask_slope = 0.0
        if ask_depth > 0 and len(asks) > 1:
            deepest_ask = float(asks[-1][0])
            ask_slope = (deepest_ask - ba_p) / ask_depth

        # Integrity (Front-loadedness)
        bid_integrity = bb_s / bid_depth if bid_depth > 0 else 0
        ask_integrity = ba_s / ask_depth if ask_depth > 0 else 0

        self.snapshots.append({
            'spread': spread,
            'micro_dev': micro_dev,
            'imbalance': imbalance,
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'bid_slope': bid_slope,
            'ask_slope': ask_slope,
            'bid_integrity': bid_integrity,
            'ask_integrity': ask_integrity
        })

    def finalize(self) -> Dict[str, float]:
        """
        Aggregate snapshots into bar metrics and reset buffer.
        """
        if not self.snapshots:
            return {}

        n = len(self.snapshots)
        
        # Sums
        sums = {k: 0.0 for k in self.snapshots[0].keys()}
        micro_devs = []
        imbalances = []
        
        for s in self.snapshots:
            for k, v in s.items():
                sums[k] += v
            micro_devs.append(s['micro_dev'])
            imbalances.append(s['imbalance'])
            
        means = {k: v / n for k, v in sums.items()}
        
        # Std Devs
        micro_mean = means['micro_dev']
        micro_var = sum((x - micro_mean) ** 2 for x in micro_devs) / n
        micro_std = math.sqrt(micro_var)

        # Construct result matching FeatureEngine keys
        result = {
            'ob_update_count': len(self.unique_timestamps),
            'ob_spread_mean': means['spread'],
            'ob_micro_dev_mean': means['micro_dev'],
            'ob_micro_dev_std': micro_std,
            'ob_micro_dev_last': micro_devs[-1],
            'ob_imbalance_mean': means['imbalance'],
            'ob_imbalance_last': imbalances[-1],
            'ob_bid_depth_mean': means['bid_depth'],
            'ob_ask_depth_mean': means['ask_depth'],
            'ob_bid_slope_mean': means['bid_slope'],
            'ob_ask_slope_mean': means['ask_slope'],
            'ob_bid_integrity_mean': means['bid_integrity'],
            'ob_ask_integrity_mean': means['ask_integrity']
        }
        
        # Reset
        self.snapshots = []
        self.unique_timestamps = set()
        return result
