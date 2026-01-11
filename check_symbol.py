
import sys
from binance.um_futures import UMFutures
from binance.error import ClientError

def check_symbol(symbol):
    client = UMFutures()
    try:
        info = client.exchange_info()
        found = False
        for s in info['symbols']:
            if s['symbol'] == symbol:
                found = True
                print(f"Symbol {symbol} FOUND on Binance Futures.")
                break
        if not found:
            print(f"Symbol {symbol} NOT found on Binance Futures.")
    except ClientError as e:
        print(f"Error connecting to Binance: {e}")

if __name__ == "__main__":
    s = sys.argv[1] if len(sys.argv) > 1 else "PIEVERSEUSDT"
    check_symbol(s)
