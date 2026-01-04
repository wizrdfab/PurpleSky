import argparse
import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

from pybit.unified_trading import HTTP, WebSocket


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def load_keys(path: Path, profile: str) -> Tuple[Optional[str], Optional[str]]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and isinstance(data.get("profiles"), dict):
        data = data["profiles"]
    entry = None
    if isinstance(data, dict):
        entry = data.get(profile) or data.get("default")
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and item.get("name") == profile:
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


def extract_ts(value: object) -> Optional[float]:
    ts = safe_float(value, 0.0)
    if ts <= 0:
        return None
    # If timestamp is in ms
    if ts > 1e11:
        return ts / 1000.0
    return ts


def round_down(value: float, step: float) -> float:
    if step <= 0:
        return value
    return (value // step) * step


def round_up(value: float, step: float) -> float:
    if step <= 0:
        return value
    return ((value + step - 1e-12) // step) * step


def trade_latencies(messages: List[Dict]) -> List[float]:
    latencies: List[float] = []
    for wrapper in messages:
        msg = wrapper.get("message", {})
        recv_ts = safe_float(wrapper.get("recv_ts"), 0.0)
        if recv_ts <= 0:
            continue
        data = msg.get("data", [])
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            continue
        for trade in data:
            ts = None
            for key in ("T", "ts", "time", "timestamp"):
                if key in trade:
                    ts = extract_ts(trade.get(key))
                    if ts:
                        break
            if ts is None:
                continue
            latencies.append((recv_ts - ts) * 1000.0)
    return latencies


def ob_latencies(messages: List[Dict]) -> List[float]:
    latencies: List[float] = []
    for wrapper in messages:
        msg = wrapper.get("message", {})
        recv_ts = safe_float(wrapper.get("recv_ts"), 0.0)
        if recv_ts <= 0:
            continue
        data = msg.get("data")
        if not isinstance(data, dict):
            continue
        ts = extract_ts(data.get("ts") or msg.get("ts"))
        if ts is None:
            continue
        latencies.append((recv_ts - ts) * 1000.0)
    return latencies


def private_latencies(messages: List[Dict], time_keys: Tuple[str, ...]) -> List[float]:
    latencies: List[float] = []
    for wrapper in messages:
        msg = wrapper.get("message", {})
        recv_ts = safe_float(wrapper.get("recv_ts"), 0.0)
        if recv_ts <= 0:
            continue
        data = msg.get("data", [])
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            data = []

        added = False
        for item in data:
            for key in time_keys:
                if key in item:
                    item_ts = extract_ts(item.get(key))
                    if item_ts is not None:
                        latencies.append((recv_ts - item_ts) * 1000.0)
                        added = True
                        break
            if added:
                break

        if not added:
            msg_ts = extract_ts(msg.get("creationTime") or msg.get("ts"))
            if msg_ts is not None:
                latencies.append((recv_ts - msg_ts) * 1000.0)
    return latencies


def stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    values = list(values)
    values.sort()
    count = len(values)
    return {
        "count": count,
        "avg_ms": sum(values) / count,
        "min_ms": values[0],
        "p50_ms": values[count // 2],
        "p95_ms": values[int(count * 0.95) - 1] if count >= 20 else values[-1],
        "max_ms": values[-1],
    }


def extract_message_keys(messages: List[Dict]) -> Tuple[List[str], List[str]]:
    if not messages:
        return [], []
    msg = messages[0].get("message", {})
    msg_keys = sorted(msg.keys())
    data_keys: List[str] = []
    data = msg.get("data")
    if isinstance(data, list) and data:
        data_keys = sorted(data[0].keys())
    elif isinstance(data, dict):
        data_keys = sorted(data.keys())
    return msg_keys, data_keys


def main() -> None:
    parser = argparse.ArgumentParser(description="Bybit WS Probe")
    parser.add_argument("--symbol", default="MONUSDT")
    parser.add_argument("--depth", type=int, default=50)
    parser.add_argument("--duration-sec", type=int, default=20)
    parser.add_argument("--max-messages", type=int, default=200)
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--testnet", action="store_true")
    parser.add_argument("--private", action="store_true", help="Use private streams")
    parser.add_argument("--keys-file", type=str, default="key_profiles.json")
    parser.add_argument("--profile", type=str, default="default")
    parser.add_argument(
        "--topics",
        type=str,
        default="order,execution,position,wallet",
        help="Comma-separated private topics",
    )
    parser.add_argument("--trigger-order", action="store_true", help="Place/cancel a far limit to trigger order stream")
    parser.add_argument("--trigger-offset-pct", type=float, default=0.10)
    parser.add_argument("--trigger-notional-pct", type=float, default=0.03)
    parser.add_argument("--trigger-side", type=str, default="Buy")
    parser.add_argument("--trigger-cancel-delay", type=float, default=2.0)
    parser.add_argument("--trigger-exec", action="store_true", help="Place a small market order to trigger execution stream")
    parser.add_argument("--trigger-exec-close", action="store_true", help="Close execution trigger position immediately")
    parser.add_argument("--trigger-exec-delay", type=float, default=1.0)
    args = parser.parse_args()

    lock = threading.Lock()
    trade_msgs: Deque[Dict] = deque(maxlen=args.max_messages)
    ob_msgs: Deque[Dict] = deque(maxlen=args.max_messages)

    private_msgs: Dict[str, Deque[Dict]] = {}

    def on_trade(msg: Dict) -> None:
        with lock:
            trade_msgs.append({"recv_ts": time.time(), "message": msg})

    def on_ob(msg: Dict) -> None:
        with lock:
            ob_msgs.append({"recv_ts": time.time(), "message": msg})

    ws = None
    trigger_thread = None

    if args.private:
        key, secret = load_keys(Path(args.keys_file), args.profile)
        if not key or not secret:
            raise RuntimeError("API keys required for private streams.")
        ws = WebSocket(
            testnet=args.testnet,
            channel_type="private",
            api_key=key,
            api_secret=secret,
        )

        topics = [t.strip() for t in args.topics.split(",") if t.strip()]
        for topic in topics:
            private_msgs[topic] = deque(maxlen=args.max_messages)

        def make_cb(topic: str):
            def _cb(msg: Dict) -> None:
                with lock:
                    private_msgs[topic].append({"recv_ts": time.time(), "message": msg})
            return _cb

        if "order" in topics:
            ws.order_stream(make_cb("order"))
        if "execution" in topics:
            ws.execution_stream(make_cb("execution"))
        if "fast_execution" in topics:
            ws.fast_execution_stream(make_cb("fast_execution"), categorised_topic="linear")
        if "position" in topics:
            ws.position_stream(make_cb("position"))
        if "wallet" in topics:
            ws.wallet_stream(make_cb("wallet"))

        def trigger_order() -> None:
            session = HTTP(testnet=args.testnet, api_key=key, api_secret=secret)
            info = session.get_instruments_info(category="linear", symbol=args.symbol)
            inst = ((info.get("result") or {}).get("list") or [None])[0] or {}
            lot = inst.get("lotSizeFilter", {})
            price_filter = inst.get("priceFilter", {})
            min_qty = safe_float(lot.get("minOrderQty"))
            qty_step = safe_float(lot.get("qtyStep"))
            min_notional = safe_float(lot.get("minNotionalValue"))
            max_qty = safe_float(lot.get("maxOrderQty"))
            tick = safe_float(price_filter.get("tickSize"))

            ob = session.get_orderbook(category="linear", symbol=args.symbol, limit=1)
            ob_res = ob.get("result") or {}
            bids = ob_res.get("b") or []
            asks = ob_res.get("a") or []
            if bids and asks:
                mid = (safe_float(bids[0][0]) + safe_float(asks[0][0])) / 2.0
            elif bids:
                mid = safe_float(bids[0][0])
            elif asks:
                mid = safe_float(asks[0][0])
            else:
                return

            limit_price = mid * (1 - args.trigger_offset_pct) if args.trigger_side.lower() == "buy" else mid * (1 + args.trigger_offset_pct)
            if tick > 0:
                limit_price = round_down(limit_price, tick)

            bal = session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            acct_list = (bal.get("result") or {}).get("list") or []
            acct = acct_list[0] if acct_list else {}
            equity = safe_float(acct.get("totalEquity"))
            notional = max(min_notional, equity * args.trigger_notional_pct if equity > 0 else min_notional)

            qty = notional / limit_price if limit_price > 0 else 0.0
            min_qty_req = min_qty
            if min_notional > 0 and limit_price > 0:
                min_qty_req = max(min_qty_req, min_notional / limit_price)
            if qty < min_qty_req:
                qty = min_qty_req
            if qty_step > 0:
                qty = round_down(qty, qty_step)
                if qty < min_qty_req:
                    qty = round_up(min_qty_req, qty_step)
            if max_qty > 0 and qty > max_qty:
                qty = max_qty

            resp = session.place_order(
                category="linear",
                symbol=args.symbol,
                side=args.trigger_side,
                orderType="Limit",
                qty=str(qty),
                price=str(limit_price),
                timeInForce="PostOnly",
                reduceOnly=False,
                positionIdx=0,
            )
            order_id = (resp.get("result") or {}).get("orderId")
            if not order_id:
                return
            time.sleep(args.trigger_cancel_delay)
            session.cancel_order(category="linear", symbol=args.symbol, orderId=order_id)

        def trigger_execution() -> None:
            session = HTTP(testnet=args.testnet, api_key=key, api_secret=secret)
            info = session.get_instruments_info(category="linear", symbol=args.symbol)
            inst = ((info.get("result") or {}).get("list") or [None])[0] or {}
            lot = inst.get("lotSizeFilter", {})
            price_filter = inst.get("priceFilter", {})
            min_qty = safe_float(lot.get("minOrderQty"))
            qty_step = safe_float(lot.get("qtyStep"))
            min_notional = safe_float(lot.get("minNotionalValue"))
            max_qty = safe_float(lot.get("maxOrderQty"))
            tick = safe_float(price_filter.get("tickSize"))

            ob = session.get_orderbook(category="linear", symbol=args.symbol, limit=1)
            ob_res = ob.get("result") or {}
            bids = ob_res.get("b") or []
            asks = ob_res.get("a") or []
            if bids and asks:
                mid = (safe_float(bids[0][0]) + safe_float(asks[0][0])) / 2.0
            elif bids:
                mid = safe_float(bids[0][0])
            elif asks:
                mid = safe_float(asks[0][0])
            else:
                return

            bal = session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            acct_list = (bal.get("result") or {}).get("list") or []
            acct = acct_list[0] if acct_list else {}
            equity = safe_float(acct.get("totalEquity"))
            notional = max(min_notional, equity * args.trigger_notional_pct if equity > 0 else min_notional)

            qty = notional / mid if mid > 0 else 0.0
            min_qty_req = min_qty
            if min_notional > 0 and mid > 0:
                min_qty_req = max(min_qty_req, min_notional / mid)
            if qty < min_qty_req:
                qty = min_qty_req
            if qty_step > 0:
                qty = round_down(qty, qty_step)
                if qty < min_qty_req:
                    qty = round_up(min_qty_req, qty_step)
            if max_qty > 0 and qty > max_qty:
                qty = max_qty

            resp = session.place_order(
                category="linear",
                symbol=args.symbol,
                side=args.trigger_side,
                orderType="Market",
                qty=str(qty),
                reduceOnly=False,
                positionIdx=0,
            )
            order_id = (resp.get("result") or {}).get("orderId")
            if not order_id:
                return
            time.sleep(args.trigger_exec_delay)
            if args.trigger_exec_close:
                close_side = "Sell" if args.trigger_side.lower() == "buy" else "Buy"
                session.place_order(
                    category="linear",
                    symbol=args.symbol,
                    side=close_side,
                    orderType="Market",
                    qty=str(qty),
                    reduceOnly=True,
                    positionIdx=0,
                )

        if args.trigger_order:
            trigger_thread = threading.Thread(target=trigger_order, daemon=True)
            trigger_thread.start()
        if args.trigger_exec:
            trigger_thread = threading.Thread(target=trigger_execution, daemon=True)
            trigger_thread.start()

        print(f"Connected PRIVATE WS: symbol={args.symbol} topics={topics} duration={args.duration_sec}s")
        time.sleep(args.duration_sec)
    else:
        ws = WebSocket(testnet=args.testnet, channel_type="linear")
        ws.trade_stream(args.symbol, on_trade)
        ws.orderbook_stream(args.depth, args.symbol, on_ob)
        print(f"Connected WS: symbol={args.symbol} depth={args.depth} duration={args.duration_sec}s")
        time.sleep(args.duration_sec)

    ws.exit()
    time.sleep(0.5)

    with lock:
        trade_list = list(trade_msgs)
        ob_list = list(ob_msgs)
        private_snapshot = {k: list(v) for k, v in private_msgs.items()}

    if args.private:
        topic_defs = {
            "order": ("createdTime", "updatedTime", "ts"),
            "execution": ("execTime", "tradeTime", "ts"),
            "fast_execution": ("execTime", "tradeTime", "ts"),
            "position": ("updatedTime", "createdTime", "ts"),
            "wallet": ("updatedTime", "ts"),
        }
        summaries = {}
        samples = []
        for topic, messages in private_snapshot.items():
            keys, data_keys = extract_message_keys(messages)
            latency = stats(private_latencies(messages, topic_defs.get(topic, ("ts",))))
            summaries[topic] = {
                "count": len(messages),
                "message_keys": keys,
                "data_keys": data_keys,
                "latency_ms": latency,
            }
            for wrapper in messages[: args.max_samples]:
                samples.append({"topic": topic, **wrapper})

        print("Private stream summaries:", summaries)
        out_path = Path(args.out) if args.out else Path(f"ws_private_probe_{args.symbol}.jsonl")
        payload = {
            "symbol": args.symbol,
            "duration_sec": args.duration_sec,
            "topics": list(private_snapshot.keys()),
            "summaries": summaries,
            "samples": samples,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Wrote probe output to {out_path}")
    else:
        trade_stats = stats(trade_latencies(trade_list))
        ob_stats = stats(ob_latencies(ob_list))
        trade_keys, trade_data_keys = extract_message_keys(trade_list)
        ob_keys, ob_data_keys = extract_message_keys(ob_list)

        print("Trade messages:", len(trade_list))
        print("Trade message keys:", trade_keys)
        print("Trade data keys:", trade_data_keys)
        print("Trade latency stats (ms):", trade_stats)
        print("Orderbook messages:", len(ob_list))
        print("Orderbook message keys:", ob_keys)
        print("Orderbook data keys:", ob_data_keys)
        print("Orderbook latency stats (ms):", ob_stats)

        out_path = Path(args.out) if args.out else Path(f"ws_probe_{args.symbol}.jsonl")
        samples = []
        for wrapper in trade_list[: args.max_samples]:
            samples.append({"topic": "trade", **wrapper})
        for wrapper in ob_list[: args.max_samples]:
            samples.append({"topic": "orderbook", **wrapper})
        payload = {
            "symbol": args.symbol,
            "depth": args.depth,
            "duration_sec": args.duration_sec,
            "trade_stats": trade_stats,
            "orderbook_stats": ob_stats,
            "trade_keys": trade_keys,
            "trade_data_keys": trade_data_keys,
            "orderbook_keys": ob_keys,
            "orderbook_data_keys": ob_data_keys,
            "samples": samples,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Wrote probe output to {out_path}")


if __name__ == "__main__":
    main()
