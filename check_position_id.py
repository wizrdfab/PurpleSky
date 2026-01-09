"""
Check whether Bybit position payload includes a positionId field.

Usage:
  python check_position_id.py --symbol MONUSDT
  python check_position_id.py --symbol MONUSDT --keys-file path/to/keys.json --keys-profile default
  python check_position_id.py --symbol MONUSDT --testnet
"""
import argparse
import json
import os
from typing import Optional, Tuple

from config import CONF
from exchange_client import ExchangeClient


def load_keys(path: Optional[str], profile: str) -> Tuple[Optional[str], Optional[str]]:
    if not path:
        return None, None
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception:
        return None, None

    entry = None
    if isinstance(data, dict):
        if isinstance(data.get("profiles"), dict):
            data = data["profiles"]
        entry = data.get(profile)
        if entry is None and profile != "default":
            entry = data.get("default")
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and item.get("name") == profile:
                entry = item
                break
        if entry is None and profile != "default":
            for item in data:
                if isinstance(item, dict) and item.get("name") == "default":
                    entry = item
                    break
        if entry is None and len(data) == 1 and isinstance(data[0], dict):
            entry = data[0]

    if not isinstance(entry, dict):
        return None, None
    key = entry.get("api_key") or entry.get("key")
    secret = entry.get("api_secret") or entry.get("secret")
    if not key or not secret:
        return None, None
    return str(key), str(secret)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect Bybit position payload for positionId.")
    parser.add_argument("--symbol", type=str, default=CONF.data.symbol)
    parser.add_argument("--testnet", action="store_true")
    parser.add_argument("--keys-file", type=str, default="")
    parser.add_argument("--keys-profile", type=str, default="default")
    parser.add_argument(
        "--dump-raw",
        type=str,
        default="",
        help="Write raw REST payload to JSON file",
    )
    args = parser.parse_args()

    key, secret = load_keys(args.keys_file, args.keys_profile)
    if not key or not secret:
        key = os.getenv("BYBIT_API_KEY")
        secret = os.getenv("BYBIT_API_SECRET")
    if not key or not secret:
        print("Missing API keys. Set BYBIT_API_KEY/BYBIT_API_SECRET or use --keys-file.")
        return 1

    client = ExchangeClient(key, secret, args.symbol, testnet=args.testnet)
    resp = client.session.get_positions(category="linear", symbol=args.symbol)
    if args.dump_raw:
        try:
            with open(args.dump_raw, "w") as f:
                json.dump(resp, f, indent=2)
            print(f"Raw payload saved to: {args.dump_raw}")
        except Exception as exc:
            print(f"Failed to write raw payload: {exc}")
    positions = (resp or {}).get("result", {}).get("list", [])

    print(f"Positions returned: {len(positions)}")
    if not positions:
        return 0

    for idx, pos in enumerate(positions, 1):
        if pos.get("symbol") != args.symbol:
            continue
        print(f"\n--- Position {idx} ({pos.get('symbol')}) ---")
        keys = sorted(pos.keys())
        print(f"Keys: {keys}")
        print(f"positionId: {pos.get('positionId') or pos.get('positionID') or pos.get('position_id')}")
        print(f"createdTime: {pos.get('createdTime')}")
        print(f"updatedTime: {pos.get('updatedTime')}")
        print(f"positionIdx: {pos.get('positionIdx')} side: {pos.get('side')} size: {pos.get('size')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
