import pandas as pd
import numpy as np
import time

# Generate 2 days of 1s data (enough for 15m bars)
dates = pd.date_range(end=pd.Timestamp.now(), periods=86400*2, freq='s')
ts = dates.astype(np.int64) // 10**9

# Random Walk Price
price = 10000 + np.random.randn(len(dates)).cumsum() * 2

# Log-normal size
size = np.random.lognormal(mean=0, sigma=1, size=len(dates))

# Random Side
sides = np.random.choice(['Buy', 'Sell'], size=len(dates))

df = pd.DataFrame({
    'timestamp': ts,
    'price': price,
    'size': size,
    'side': sides
})

df.to_csv('data/BTCUSDT/trades.csv', index=False)
print("Dummy data with Side generated.")