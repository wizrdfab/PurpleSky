
"""
Simulation runner for live_trading_v2 using the local trade/orderbook dataset.
"""

import argparse
import json
import os
import time as time_module
from pathlib import Path
from typing import Optional, Tuple

import exchange_client
from config import CONF
from simulated_exchange import (
    SimulatedExchange,
    TradeStream,
    bar_time_from_ts,
    list_sorted_files,
    resolve_symbol_dir,
)


def timeframe_to_seconds(tf: str) -> int:
    mapping = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "1h": 3600,
        "4h": 14400,
    }
    if tf not in mapping:
        raise ValueError(f"Unsupported timeframe: {tf}")
    return mapping[tf]


def compute_warmup_bars() -> int:
    max_ema = max(CONF.features.ema_periods) if CONF.features.ema_periods else 0
    return max(max_ema, CONF.features.atr_period, CONF.features.rsi_period, 24)


def count_bars(trade_files, tf_seconds: int) -> int:
    stream = TradeStream(trade_files)
    current = None
    count = 0
    while True:
        row = stream.next()
        if row is None:
            break
        bar_time = bar_time_from_ts(row.timestamp, tf_seconds)
        if bar_time != current:
            count += 1
            current = bar_time
    return count


def find_bar_times(trade_files, tf_seconds: int, warmup_idx: int, test_idx: int) -> Tuple[Optional[int], Optional[int]]:
    stream = TradeStream(trade_files)
    current = None
    bar_idx = -1
    warmup_time = None
    test_time = None
    while True:
        row = stream.next()
        if row is None:
            break
        bar_time = bar_time_from_ts(row.timestamp, tf_seconds)
        if bar_time != current:
            bar_idx += 1
            current = bar_time
            if warmup_time is None and bar_idx == warmup_idx:
                warmup_time = bar_time
            if bar_idx == test_idx:
                test_time = bar_time
                break
    return warmup_time, test_time


class SimClock:
    def __init__(self, start_ts: float):
        self.current = float(start_ts)
        self.stop_callback = None

    def time(self) -> float:
        return self.current

    def set(self, ts: float) -> None:
        if ts is None:
            return
        self.current = float(ts)

    def sleep(self, seconds: float) -> None:
        if self.stop_callback and self.stop_callback():
            raise KeyboardInterrupt
        return


def build_live_args(args, symbol_dir: Path):
    return argparse.Namespace(
        model_dir=args.model_dir,
        symbol=args.symbol,
        data_dir=str(symbol_dir),
        window=args.window,
        dry_run=False,
        log_level=args.log_level,
        trade_poll_sec=args.trade_poll_sec,
        ob_poll_sec=args.ob_poll_sec,
        reconcile_sec=args.reconcile_sec,
        heartbeat_sec=args.heartbeat_sec,
        instr_refresh_sec=args.instr_refresh_sec,
        data_lag_warn_sec=args.data_lag_warn_sec,
        data_lag_error_sec=args.data_lag_error_sec,
        max_leverage=args.max_leverage,
        drift_window=args.drift_window,
        drift_z=args.drift_z,
        drift_keys=args.drift_keys,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulated live trading runner")
    parser.add_argument("--symbol", type=str, default=CONF.data.symbol)
    parser.add_argument("--data-dir", type=str, default=str(CONF.data.data_dir))
    parser.add_argument("--model-dir", type=str, default=f"models_v2/{CONF.data.symbol}/rank_1")
    parser.add_argument("--initial-equity", type=float, default=10000.0)
    parser.add_argument("--window", type=int, default=500)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--quiet", action="store_true", help="Reduce simulator console output")
    parser.add_argument("--trade-poll-sec", type=float, default=0.0)
    parser.add_argument("--ob-poll-sec", type=float, default=0.0)
    parser.add_argument("--reconcile-sec", type=float, default=0.0)
    parser.add_argument("--heartbeat-sec", type=float, default=60.0)
    parser.add_argument("--instr-refresh-sec", type=float, default=0.0)
    parser.add_argument("--data-lag-warn-sec", type=float, default=60.0)
    parser.add_argument("--data-lag-error-sec", type=float, default=120.0)
    parser.add_argument("--max-leverage", type=float, default=5.0)
    parser.add_argument("--drift-window", type=int, default=200)
    parser.add_argument("--drift-z", type=float, default=3.0)
    parser.add_argument("--sim-status-every", type=int, default=0, help="Status print cadence in bars (0=auto)")
    parser.add_argument(
        "--state-mode",
        type=str,
        choices=["reset", "keep", "live"],
        default="reset",
        help="State handling: reset/keep uses a sim-specific state file; live uses the live state file.",
    )
    parser.add_argument(
        "--state-suffix",
        type=str,
        default="sim",
        help="Suffix for sim state file when state-mode is reset/keep.",
    )
    parser.add_argument(
        "--drift-keys",
        type=str,
        default="atr,rsi,vol_z,ob_spread_bps,ob_imbalance_mean,taker_buy_ratio,pred_long,pred_short",
    )
    args = parser.parse_args()
    args.drift_keys = [k.strip() for k in args.drift_keys.split(",") if k.strip()]
    return args


def main() -> None:
    args = parse_args()
    symbol_dir = resolve_symbol_dir(Path(args.data_dir), args.symbol)

    trade_files = list_sorted_files(symbol_dir / "Trade", "*.csv")
    if not trade_files:
        raise FileNotFoundError(f"No trade CSVs found under {symbol_dir / 'Trade'}")
    ob_files = list_sorted_files(symbol_dir / "Orderbook", "*.data")
    if not ob_files:
        raise FileNotFoundError(f"No orderbook files found under {symbol_dir / 'Orderbook'}")

    sim_verbose = not args.quiet

    tf_seconds = timeframe_to_seconds(CONF.features.base_timeframe)
    total_bars = count_bars(trade_files, tf_seconds)
    if total_bars == 0:
        raise RuntimeError("No bars found in trade data")

    train_val_ratio = CONF.model.train_ratio + CONF.model.val_ratio
    test_start_idx = int(total_bars * train_val_ratio)
    warmup_bars = compute_warmup_bars()
    warmup_start_idx = max(test_start_idx - warmup_bars, 0)

    warmup_time, test_time = find_bar_times(trade_files, tf_seconds, warmup_start_idx, test_start_idx)
    if warmup_time is None or test_time is None:
        raise RuntimeError("Failed to locate warmup/test start times")

    trade_enable_time = test_time + tf_seconds

    test_total = max(total_bars - test_start_idx, 1)
    status_every = args.sim_status_every if args.sim_status_every > 0 else max(1, test_total // 50)

    if sim_verbose:
        print(f"Symbol: {args.symbol}")
        print(f"Trade files: {len(trade_files)} | Orderbook files: {len(ob_files)}")
        print(f"Base timeframe: {CONF.features.base_timeframe} ({tf_seconds}s)")
        print(f"Total bars: {total_bars}")
        print(f"Warmup bars: {warmup_bars}")
        print(f"Warmup start index: {warmup_start_idx}")
        print(f"Test start index: {test_start_idx}")
        print(f"Warmup start time: {warmup_time}")
        print(f"Test start time: {test_time}")
        print(f"Status cadence: every {status_every} bars")

    state_live_path = Path(f"bot_state_{args.symbol}_v2.json")
    state_sim_path = Path(f"bot_state_{args.symbol}_v2_{args.state_suffix}.json")
    state_path = state_live_path if args.state_mode == "live" else state_sim_path

    if sim_verbose:
        print(f"State mode: {args.state_mode} | State file: {state_path}")

    if args.state_mode in ("keep", "live") and state_path.exists():
        try:
            with open(state_path, "r") as f:
                state_payload = json.load(f)
            last_trade_ts = float(state_payload.get("last_trade_ts", 0.0))
            if last_trade_ts and last_trade_ts > warmup_time:
                print(
                    "[SIM] Warning: state last_trade_ts is ahead of dataset warmup. "
                    "Use --state-mode reset to avoid skipping trades."
                )
        except Exception:
            pass

    if not os.getenv("BYBIT_API_KEY"):
        os.environ["BYBIT_API_KEY"] = "SIM_KEY"
        if sim_verbose:
            print("BYBIT_API_KEY not set. Using dummy key for simulation.")
    if not os.getenv("BYBIT_API_SECRET"):
        os.environ["BYBIT_API_SECRET"] = "SIM_SECRET"
        if sim_verbose:
            print("BYBIT_API_SECRET not set. Using dummy secret for simulation.")

    clock = SimClock(warmup_time)
    sim_exchange = SimulatedExchange(
        symbol=args.symbol,
        data_dir=Path(args.data_dir),
        clock=clock,
        start_time=warmup_time,
        trade_enable_time=trade_enable_time,
        initial_equity=args.initial_equity,
        tf_seconds=tf_seconds,
        start_bar_index=warmup_start_idx,
        test_start_index=test_start_idx,
        total_bars=total_bars,
        verbose=sim_verbose,
        status_every_bars=status_every,
    )

    clock.stop_callback = sim_exchange.should_stop

    time_module.time = clock.time
    time_module.sleep = clock.sleep

    exchange_client.ExchangeClient = lambda api_key, api_secret, symbol: sim_exchange

    import live_trading_v2

    perf_counter = time_module.perf_counter

    def _sim_call(self, name: str, func, *args, **kwargs):
        start = perf_counter()
        try:
            result = func(*args, **kwargs)
            latency_ms = (perf_counter() - start) * 1000.0
            self.latency.update(latency_ms)
            return result
        except Exception as exc:
            latency_ms = (perf_counter() - start) * 1000.0
            self.latency.update(latency_ms)
            self.logger.error(f"API error in {name}: {exc}")
            return None

    class SimStateStore(live_trading_v2.StateStore):
        def __init__(self, path: Path, symbol: str, logger):
            super().__init__(state_path, symbol, logger)

        def load(self) -> None:
            if args.state_mode == "reset":
                return
            super().load()

    live_trading_v2.SafeExchange._call = _sim_call
    live_trading_v2.StateStore = SimStateStore

    live_args = build_live_args(args, symbol_dir)
    trader = live_trading_v2.LiveTradingV2(live_args)

    try:
        trader.run()
    except KeyboardInterrupt:
        pass

    if sim_verbose:
        balance = sim_exchange.get_wallet_balance()
        equity = balance.get("equity", 0.0)
        print("Simulation complete.")
        print(f"Final equity: {equity:.2f}")
        print(f"Realized PnL: {sim_exchange.cum_realized:.2f}")
        print(f"Open orders: {len(sim_exchange.orders)}")
        if sim_exchange.position:
            pos = sim_exchange.position
            print(
                f"Open position: {pos.get('side')} size={pos.get('size', 0):.6f} "
                f"entry={pos.get('entry_price', 0):.6f} unreal={pos.get('unrealized_pnl', 0):.2f}"
            )
        else:
            print("Open position: flat")


if __name__ == "__main__":
    main()
