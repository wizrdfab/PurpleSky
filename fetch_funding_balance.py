#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

try:
    from pybit.unified_trading import HTTP as BybitHTTP
except Exception as exc:
    print(f"pybit not available: {exc}")
    sys.exit(1)


def load_profile(keys_path: Path, profile: str) -> dict:
    if not keys_path.exists():
        raise FileNotFoundError(f"Keys file not found: {keys_path}")
    payload = json.loads(keys_path.read_text(encoding="utf-8"))
    profiles = payload.get("profiles", {})
    if profile not in profiles:
        raise KeyError(f"Profile not found: {profile}")
    return profiles[profile]


def safe_float(value, default=0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def safe_text(value: object) -> str:
    text = str(value)
    try:
        text.encode("ascii")
        return text
    except UnicodeEncodeError:
        return text.encode("ascii", "backslashreplace").decode("ascii")


def get_member_id(session) -> Optional[str]:
    resp = session.get_api_key_information()
    if not isinstance(resp, dict) or resp.get("retCode") not in (None, 0):
        return None
    result = resp.get("result", {})
    user_id = result.get("userID") or result.get("uid") or result.get("userId")
    if user_id is None:
        return None
    return str(user_id)


def fetch_funding_balance(session, member_id: Optional[str], coin: Optional[str]):
    try:
        params = {"accountType": "FUND"}
        if member_id:
            params["memberId"] = member_id
        if coin:
            params["coin"] = coin
            resp = session.get_coin_balance(**params)
        else:
            resp = session.get_coins_balance(**params)
    except Exception as exc:
        raise RuntimeError(f"Funding balance request failed: {exc}") from exc

    if not isinstance(resp, dict):
        raise RuntimeError("Unexpected response type.")
    if resp.get("retCode") not in (None, 0):
        raise RuntimeError(f"Bybit error: retCode={resp.get('retCode')} retMsg={resp.get('retMsg')}")
    result = resp.get("result", {})
    balances = result.get("balance") or result.get("list") or []
    if isinstance(balances, dict):
        balances = [balances]
    return balances


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch Bybit funding account balance.")
    parser.add_argument("--keys-file", default="key_profiles.json", help="Path to key_profiles.json")
    parser.add_argument("--profile", default="default", help="Profile name in key_profiles.json")
    parser.add_argument("--coin", default=None, help="Optional coin filter (e.g., USDT)")
    parser.add_argument("--testnet", action="store_true", help="Use testnet")
    args = parser.parse_args()

    keys_path = Path(args.keys_file)
    try:
        creds = load_profile(keys_path, args.profile)
    except Exception as exc:
        print(f"Error loading profile: {exc}")
        return 1

    api_key = creds.get("api_key")
    api_secret = creds.get("api_secret")
    if not api_key or not api_secret:
        print("Missing api_key or api_secret in profile.")
        return 1

    session = BybitHTTP(
        testnet=args.testnet,
        api_key=api_key,
        api_secret=api_secret,
    )
    member_id = get_member_id(session)
    try:
        balances = fetch_funding_balance(session, member_id, args.coin)
    except RuntimeError as exc:
        msg = safe_text(exc)
        print(msg)
        if "ErrCode: 10005" in msg or "retCode=10005" in msg or "Permission denied" in msg:
            print("Hint: enable Wallet permissions for this API key (AccountTransfer).")
        return 2
    if not balances:
        print("No funding balances returned.")
        return 0

    stable_total = 0.0
    stable_available = 0.0
    rows = []
    for item in balances:
        coin = str(item.get("coin") or "")
        wallet = safe_float(item.get("walletBalance"))
        transfer = safe_float(item.get("transferBalance"))
        rows.append((coin, wallet, transfer))
        if coin.upper() in {"USDT", "USDC"}:
            stable_total += wallet
            stable_available += transfer

    print(f"Funding account balances for profile '{args.profile}':")
    for coin, wallet, transfer in rows:
        print(f"- {coin}: wallet={wallet:.8f} transferable={transfer:.8f}")
    if stable_total > 0:
        print(f"Stable total (USDT/USDC): {stable_total:.8f} | transferable={stable_available:.8f}")
    else:
        print("No USDT/USDC balances detected. Non-stable coins are listed above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
