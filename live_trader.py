"""
Live trading module aligned with the tuned backtest logic.

This implementation mirrors config_tuner/backtest_tuned_config behavior:
- Uses tuned train_config + tuning_summary (required)
- Uses incremental_features for live feature updates
- Applies EMA touch (multi-TF) pullback detection
- Applies EV gating, trend/regime gating, ops cost, and single-position policy
"""
from __future__ import annotations

import argparse
import getpass
import json
import logging
import os
import re
import stat
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import shutil
import textwrap

# Ensure bundled `pybit-master` is importable (repo-local dependency).
_REPO_DIR = Path(__file__).resolve().parent
_PYBIT_DIR = _REPO_DIR / "pybit-master"
if _PYBIT_DIR.exists():
    sys.path.insert(0, str(_PYBIT_DIR))

from pybit.unified_trading import WebSocket  # noqa: E402

from config import TrendFollowerConfig  # noqa: E402
from exchange_client import BybitClient, DEFAULT_RECV_WINDOW_MS  # noqa: E402
from incremental_features import IncrementalBarAggregator, IncrementalFeatureEngine  # noqa: E402
from models import (  # noqa: E402
    CONTEXT_FEATURE_NAMES,
    TrendFollowerModels,
    append_context_features,
)
from predictor import TrendFollowerPredictor  # noqa: E402


DEFAULT_PAPER_CAPITAL = 10000.0
DEFAULT_POSITION_SIZE_PCT = 0.02
DEFAULT_MAX_PENDING_TRADES = 200000
DEFAULT_DROP_LOG_INTERVAL_SECONDS = 30.0
DEFAULT_STALE_DATA_THRESHOLD_SECONDS = 60.0
DEFAULT_HEALTH_CHECK_INTERVAL_SECONDS = 10.0
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 30.0
DEFAULT_MAX_RECONNECT_ATTEMPTS = 10
DEFAULT_RECONNECT_BACKOFF_BASE_SECONDS = 2.0
DEFAULT_RECONNECT_COOLDOWN_SECONDS = 300.0
DEFAULT_LATENCY_WINDOW_SIZE = 300
DEFAULT_ENTRY_PAUSE_LOG_INTERVAL_SECONDS = 60.0
DEFAULT_WARMUP_MAX_ATTEMPTS = 3
DEFAULT_WARMUP_MAX_BARS = 5000
DEFAULT_KEY_POLL_INTERVAL_SECONDS = 0.1
DEFAULT_ACCOUNT_REFRESH_SECONDS = 900.0
DEFAULT_LATENCY_ALERT_SECONDS = 1.0
DEFAULT_LATENCY_UNSTABLE_RATIO = 2.5
DEFAULT_STATUS_LOG_INTERVAL_SECONDS = 300.0
DEFAULT_CONSOLE_FONT_SIZE = 18
DEFAULT_SESSION_LOG_DIR = "./live_trader_logs"
DEFAULT_SESSION_EVENT_BUFFER = 2000

ANSI_RESET = "\x1b[0m"
ANSI_BOLD = "\x1b[1m"
ANSI_RED = "\x1b[31m"
ANSI_YELLOW = "\x1b[38;5;220m"
ANSI_GREEN = "\x1b[38;5;28m"
ANSI_BRIGHT_GREEN = "\x1b[38;5;70m"
ANSI_ORANGE = "\x1b[38;5;208m"
ANSI_GREY = "\x1b[38;5;245m"
ANSI_WHITE = "\x1b[37m"
ANSI_SKY_BLUE = "\x1b[38;5;110m"


class _Cp1252SafeFilter(logging.Filter):
    """Prevent Windows console crashes if a log message contains unsupported chars."""

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover
        try:
            msg = record.getMessage()
        except Exception:
            return True
        safe = msg.encode("cp1252", errors="replace").decode("cp1252", errors="replace")
        record.msg = safe
        record.args = ()
        return True


def _enable_windows_vt() -> bool:
    if os.name != "nt":
        return True
    try:
        import ctypes
    except Exception:
        return False

    kernel32 = ctypes.windll.kernel32
    handles = [-11, -12]  # STD_OUTPUT_HANDLE, STD_ERROR_HANDLE
    for handle_id in handles:
        handle = kernel32.GetStdHandle(handle_id)
        if handle == 0:
            continue
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)) == 0:
            continue
        new_mode = mode.value | 0x0004  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        kernel32.SetConsoleMode(handle, new_mode)
    return True


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    stream = sys.stderr
    if not hasattr(stream, "isatty") or not stream.isatty():
        return False
    if os.name == "nt":
        return _enable_windows_vt()
    return True


class _ConsoleColorFormatter(logging.Formatter):
    def __init__(self, enable_color: bool) -> None:
        super().__init__(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.enable_color = bool(enable_color)

    def format(self, record: logging.LogRecord) -> str:
        if not self.enable_color:
            return super().format(record)

        orig_level = record.levelname
        orig_msg = record.msg
        orig_args = record.args
        try:
            asctime = self.formatTime(record, self.datefmt)
            display_level = orig_level
            message_plain = record.getMessage()
            if message_plain.startswith("ALERT:"):
                display_level = "ALERT"
            level_plain = f"{display_level:<8}"
            prefix_plain = f"{asctime} | {level_plain} | "

            level_color = {
                "INFO": ANSI_GREY,
                "WARNING": ANSI_YELLOW,
                "ERROR": ANSI_RED,
                "CRITICAL": ANSI_RED,
                "ALERT": ANSI_RED,
            }.get(display_level, ANSI_GREY)

            if message_plain.startswith("Health:"):
                message_color = ANSI_GREEN
            elif message_plain.startswith("Heartbeat:"):
                message_color = ANSI_GREY
            elif message_plain.startswith("MODEL:"):
                message_color = ANSI_ORANGE
            elif message_plain.startswith("Status:"):
                message_color = ANSI_BRIGHT_GREEN
            elif message_plain.startswith("OPEN ") or message_plain.startswith("CLOSE "):
                message_color = ANSI_ORANGE
            elif message_plain.startswith("ALERT:"):
                message_color = ANSI_RED
            elif record.levelno >= logging.ERROR:
                message_color = ANSI_RED
            elif record.levelno >= logging.WARNING:
                message_color = ANSI_YELLOW
            else:
                message_color = ANSI_GREY

            term_width = shutil.get_terminal_size((120, 20)).columns
            wrap_width = max(20, term_width - len(prefix_plain))
            indent = " " * len(prefix_plain)
            wrapped_lines = []
            for raw_line in message_plain.splitlines() or [""]:
                wrapped = textwrap.fill(
                    raw_line,
                    width=wrap_width,
                    subsequent_indent=indent,
                )
                wrapped_lines.append(wrapped)
            wrapped_message = "\n".join(wrapped_lines)

            prefix = (
                f"{ANSI_BOLD}{ANSI_GREY}{asctime}{ANSI_RESET}"
                f"{ANSI_BOLD}{ANSI_GREY} | {ANSI_RESET}"
                f"{ANSI_BOLD}{level_color}{level_plain}{ANSI_RESET}"
                f"{ANSI_BOLD}{ANSI_GREY} | {ANSI_RESET}"
            )
            colored_message = f"{ANSI_BOLD}{message_color}{wrapped_message}{ANSI_RESET}"
            return prefix + colored_message
        finally:
            record.levelname = orig_level
            record.msg = orig_msg
            record.args = orig_args


class _KeyListener(threading.Thread):
    def __init__(self, on_key, logger: logging.Logger) -> None:
        super().__init__(daemon=True)
        self._on_key = on_key
        self._logger = logger
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        if os.name == "nt":
            try:
                import msvcrt
            except Exception as exc:
                self._logger.warning("Keyboard controls disabled: %s", exc)
                return
            while not self._stop_event.is_set():
                if msvcrt.kbhit():
                    try:
                        key = msvcrt.getwch()
                    except Exception:
                        key = None
                    if key:
                        self._on_key(key)
                time.sleep(DEFAULT_KEY_POLL_INTERVAL_SECONDS)
            return

        if not sys.stdin.isatty():
            self._logger.warning("Keyboard controls disabled (stdin is not a TTY).")
            return
        try:
            import select
            import termios
            import tty
        except Exception as exc:
            self._logger.warning("Keyboard controls disabled: %s", exc)
            return

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self._stop_event.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], DEFAULT_KEY_POLL_INTERVAL_SECONDS)
                if ready:
                    key = sys.stdin.read(1)
                    if key:
                        self._on_key(key)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _clear_console() -> None:
    try:
        if os.name == "nt":
            os.system("cls")
        else:
            os.system("clear")
    except Exception:
        return


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


class _SessionEventHandler(logging.Handler):
    def __init__(self, on_event) -> None:
        super().__init__()
        self._on_event = on_event

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg)
        if not message:
            return
        if message.startswith("ALERT:") or record.levelno >= logging.ERROR:
            event = {
                "event": "alert",
                "level": record.levelname,
                "message": message,
                "ts": _utc_now().isoformat(),
            }
            self._on_event(event)


def _safe_age_seconds(entry_time: datetime, now: datetime) -> float:
    if entry_time.tzinfo is None:
        return (datetime.now(timezone.utc).replace(tzinfo=None) - entry_time).total_seconds()
    if now.tzinfo is None:
        now = datetime.now(timezone.utc)
    return (now - entry_time).total_seconds()


def _set_console_font_size(size: int, logger: Optional[logging.Logger] = None) -> None:
    if size <= 0:
        return
    if os.name == "nt":
        if os.environ.get("WT_SESSION") and logger:
            logger.warning(
                "ALERT: Windows Terminal often ignores programmatic font sizing; use Settings or Ctrl+Plus."
            )
        try:
            import ctypes
            from ctypes import wintypes

            class COORD(ctypes.Structure):
                _fields_ = [("X", wintypes.SHORT), ("Y", wintypes.SHORT)]

            class CONSOLE_FONT_INFOEX(ctypes.Structure):
                _fields_ = [
                    ("cbSize", wintypes.ULONG),
                    ("nFont", wintypes.DWORD),
                    ("dwFontSize", COORD),
                    ("FontFamily", wintypes.UINT),
                    ("FontWeight", wintypes.UINT),
                    ("FaceName", wintypes.WCHAR * 32),
                ]

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)
            font = CONSOLE_FONT_INFOEX()
            font.cbSize = ctypes.sizeof(CONSOLE_FONT_INFOEX)
            if kernel32.GetCurrentConsoleFontEx(handle, False, ctypes.byref(font)) == 0:
                if logger:
                    logger.warning("ALERT: Failed to read console font settings.")
                return
            current_size = int(font.dwFontSize.Y)
            font.dwFontSize.Y = int(size)
            if kernel32.SetCurrentConsoleFontEx(handle, False, ctypes.byref(font)) == 0:
                if logger:
                    logger.warning("ALERT: Failed to set console font size.")
            else:
                verify = CONSOLE_FONT_INFOEX()
                verify.cbSize = ctypes.sizeof(CONSOLE_FONT_INFOEX)
                if kernel32.GetCurrentConsoleFontEx(handle, False, ctypes.byref(verify)) != 0:
                    applied = int(verify.dwFontSize.Y)
                    if applied != int(size) and logger:
                        logger.warning(
                            "ALERT: Console font size unchanged (requested %s, current %s).",
                            size,
                            applied,
                        )
                elif logger:
                    logger.info(
                        "Console font size requested: %s (current was %s).",
                        size,
                        current_size,
                    )
        except Exception as exc:
            if logger:
                logger.warning("ALERT: Console font adjustment failed: %s", exc)
        return

    try:
        sys.stderr.write(f"\x1b]50;size={int(size)}\x07")
        sys.stderr.flush()
        if logger:
            logger.info("Console font size requested via OSC 50: %s", size)
    except Exception as exc:
        if logger:
            logger.warning("ALERT: Console font adjustment failed: %s", exc)


def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        _ConsoleColorFormatter(enable_color=_supports_color())
    )
    root.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(file_handler)

    log_filter = _Cp1252SafeFilter()
    for handler in root.handlers:
        handler.addFilter(log_filter)

    return logging.getLogger(__name__)


def _load_json_secrets(path: Path) -> dict:
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception as exc:
        raise SystemExit(f"Failed to read secrets JSON: {path} ({exc})")
    if not isinstance(data, dict):
        raise SystemExit(f"Secrets JSON must be an object: {path}")
    return data


def _validate_posix_permissions(path: Path) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise SystemExit(
            f"Secrets file permissions too open: {path} (mode {oct(mode)}). "
            "Run: chmod 600 <secrets.json>"
        )


def _validate_windows_permissions(path: Path) -> None:
    user = getpass.getuser().lower()
    computer = os.environ.get("COMPUTERNAME", "").lower()
    allowed_principals = {
        user,
        f"{computer}\\{user}" if computer else "",
        "nt authority\\system",
        "system",
        "nt authority\\sistema",
        "sistema",
        "builtin\\administrators",
        "administrators",
        "builtin\\administradores",
        "administradores",
        "nt service\\trustedinstaller",
        "trustedinstaller",
    }
    allowed_principals = {p for p in allowed_principals if p}

    try:
        result = subprocess.run(
            ["icacls", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        raise SystemExit(f"Failed to inspect ACLs for secrets file: {exc}")

    if result.returncode != 0:
        raise SystemExit(
            f"Failed to inspect ACLs for secrets file: {result.stderr.strip()}"
        )

    principals: list[str] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        for match in re.findall(r"([^:]+):\\(", line):
            principal = match.strip()
            if principal:
                principals.append(principal)

    invalid = []
    for principal in principals:
        principal_norm = principal.lower().strip()
        if principal_norm in allowed_principals:
            continue
        if principal_norm.endswith(f"\\{user}"):
            continue
        invalid.append(principal)

    if invalid:
        sample = ", ".join(invalid[:5])
        raise SystemExit(
            "Secrets file ACL too permissive. Restrict access to your user, "
            f"SYSTEM, and Administrators. Found: {sample}"
        )


def _validate_secrets_permissions(path: Path) -> None:
    if os.name == "posix":
        _validate_posix_permissions(path)
    elif os.name == "nt":
        _validate_windows_permissions(path)


def _load_keyring_credentials(service: str, logger: logging.Logger) -> tuple[Optional[str], Optional[str]]:
    try:
        import keyring  # type: ignore
    except Exception as exc:
        logger.warning("Keyring unavailable (%s); falling back.", exc)
        return None, None

    try:
        api_key = keyring.get_password(service, "BYBIT_API_KEY")
        api_secret = keyring.get_password(service, "BYBIT_API_SECRET")
    except Exception as exc:
        logger.warning("Keyring lookup failed (%s); falling back.", exc)
        return None, None

    return api_key or None, api_secret or None


def _resolve_api_credentials(
    *,
    api_key: Optional[str],
    api_secret: Optional[str],
    secrets_path: Optional[Path],
    keyring_service: Optional[str],
    logger: logging.Logger,
) -> tuple[str, str, str]:
    if keyring_service:
        k_key, k_secret = _load_keyring_credentials(keyring_service, logger)
        if k_key and k_secret:
            return k_key, k_secret, "keyring"
        if k_key or k_secret:
            logger.warning("Keyring returned incomplete credentials; falling back.")

    if secrets_path is not None:
        if not secrets_path.exists():
            raise SystemExit(f"Secrets file not found: {secrets_path}")
        _validate_secrets_permissions(secrets_path)
        data = _load_json_secrets(secrets_path)
        file_key = data.get("bybit_api_key") or data.get("api_key")
        file_secret = data.get("bybit_api_secret") or data.get("api_secret")
        if file_key and file_secret:
            return str(file_key), str(file_secret), "secrets_file"
        raise SystemExit(
            "Secrets file missing bybit_api_key/api_key or bybit_api_secret/api_secret."
        )

    env_key = os.getenv("BYBIT_API_KEY", "")
    env_secret = os.getenv("BYBIT_API_SECRET", "")
    if env_key and env_secret:
        return env_key, env_secret, "env"

    if api_key and api_secret:
        logger.warning("Using API keys from CLI arguments is insecure; prefer keyring or secrets file.")
        return api_key, api_secret, "cli"

    raise SystemExit(
        "API keys not found. Provide keyring credentials, --secrets-path, or BYBIT_API_KEY/BYBIT_API_SECRET."
    )


def _to_float(value, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return default


def _floor_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    d_value = Decimal(str(value))
    d_step = Decimal(str(step))
    return float((d_value / d_step).to_integral_value(rounding=ROUND_DOWN) * d_step)


def _round_stop_tp(
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    tick_size: float,
    direction: int,
) -> tuple[float, float]:
    if tick_size <= 0:
        return stop_loss, take_profit

    d_tick = Decimal(str(tick_size))

    def _round(value: float, rounding) -> float:
        d_val = Decimal(str(value))
        return float((d_val / d_tick).to_integral_value(rounding=rounding) * d_tick)

    if direction == 1:
        stop = _round(stop_loss, ROUND_UP)
        tp = _round(take_profit, ROUND_DOWN)
        if stop >= entry_price:
            stop = _round(entry_price - tick_size, ROUND_DOWN)
        if tp <= entry_price:
            tp = _round(entry_price + tick_size, ROUND_UP)
        return stop, tp

    stop = _round(stop_loss, ROUND_DOWN)
    tp = _round(take_profit, ROUND_UP)
    if stop <= entry_price:
        stop = _round(entry_price + tick_size, ROUND_UP)
    if tp >= entry_price:
        tp = _round(entry_price - tick_size, ROUND_DOWN)
    return stop, tp


def _apply_config_section(target, data: dict, path_fields: Optional[set] = None) -> None:
    if not isinstance(data, dict):
        return
    path_fields = path_fields or set()
    for key, value in data.items():
        if key in path_fields and value is not None:
            value = Path(value)
        setattr(target, key, value)


def _load_train_config_from_path(config_path: Path) -> tuple[TrendFollowerConfig, dict, dict]:
    if not config_path.exists():
        raise FileNotFoundError(f"Train config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cfg = TrendFollowerConfig()
    if isinstance(data, dict):
        if "data" in data:
            _apply_config_section(cfg.data, data["data"], {"data_dir"})
        if "features" in data:
            _apply_config_section(cfg.features, data["features"])
        if "labels" in data:
            _apply_config_section(cfg.labels, data["labels"])
        if "model" in data:
            _apply_config_section(cfg.model, data["model"], {"model_dir"})
        if "base_timeframe_idx" in data:
            cfg.base_timeframe_idx = int(data["base_timeframe_idx"])
        if "seed" in data:
            cfg.seed = int(data["seed"])

    meta = {}
    if isinstance(data, dict) and "tuning_summary_path" in data:
        meta["tuning_summary_path"] = data["tuning_summary_path"]

    raw = data if isinstance(data, dict) else {}
    return cfg, meta, raw


def _load_tuning_summary(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Tuning summary not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid tuning summary: {path}")
    return data


def _apply_summary_to_config(cfg: TrendFollowerConfig, summary: dict) -> None:
    best_config = summary.get("best_config")
    if not isinstance(best_config, dict):
        return
    if "features" in best_config:
        _apply_config_section(cfg.features, best_config["features"])
    if "labels" in best_config:
        _apply_config_section(cfg.labels, best_config["labels"])
    if "model" in best_config:
        model_cfg = best_config.get("model")
        if isinstance(model_cfg, dict):
            model_cfg = dict(model_cfg)
            model_cfg.pop("model_dir", None)
            _apply_config_section(cfg.model, model_cfg)
    if "base_timeframe_idx" in best_config:
        cfg.base_timeframe_idx = int(best_config["base_timeframe_idx"])
    if "seed" in best_config:
        cfg.seed = int(best_config["seed"])


def _get_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    if isinstance(value, (int, float)):
        return bool(float(value))
    return bool(value)


def _get_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _get_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _extract_tuned_settings(cfg: TrendFollowerConfig, summary: dict) -> dict:
    best_config = summary.get("best_config", {}) if isinstance(summary, dict) else {}
    labels = best_config.get("labels", {}) if isinstance(best_config, dict) else {}
    metrics = summary.get("best_metrics", {}) if isinstance(summary, dict) else {}
    tuner_settings = summary.get("tuner_settings", {}) if isinstance(summary, dict) else {}

    tuned_use_raw = _get_bool(metrics.get("use_raw_probabilities", labels.get("use_raw_probabilities")), False)
    tuned_use_cal = _get_bool(labels.get("use_calibration"), not tuned_use_raw)
    if tuned_use_raw:
        tuned_use_cal = False

    return {
        "best_threshold": _get_float(metrics.get("best_threshold", labels.get("best_threshold", cfg.labels.best_threshold)), cfg.labels.best_threshold),
        "stop_atr_multiple": _get_float(labels.get("stop_atr_multiple", cfg.labels.stop_atr_multiple), cfg.labels.stop_atr_multiple),
        "target_rr": _get_float(labels.get("target_rr", cfg.labels.target_rr), cfg.labels.target_rr),
        "pullback_threshold": _get_float(labels.get("pullback_threshold", cfg.labels.pullback_threshold), cfg.labels.pullback_threshold),
        "entry_forward_window": _get_int(labels.get("entry_forward_window", cfg.labels.entry_forward_window), cfg.labels.entry_forward_window),
        "ev_margin_r": _get_float(labels.get("ev_margin_r", metrics.get("ev_margin_r", cfg.labels.ev_margin_r)), cfg.labels.ev_margin_r),
        "fee_percent": _get_float(labels.get("fee_percent", metrics.get("fee_percent", cfg.labels.fee_percent)), cfg.labels.fee_percent),
        "fee_per_trade_r": metrics.get("fee_per_trade_r", labels.get("fee_per_trade_r", cfg.labels.fee_per_trade_r)),
        "use_expected_rr": _get_bool(labels.get("use_expected_rr", metrics.get("use_expected_rr", cfg.labels.use_expected_rr)), cfg.labels.use_expected_rr),
        "use_ev_gate": _get_bool(labels.get("use_ev_gate", metrics.get("use_ev_gate", cfg.labels.use_ev_gate)), cfg.labels.use_ev_gate),
        "use_trend_gate": _get_bool(labels.get("use_trend_gate", metrics.get("use_trend_gate", cfg.labels.use_trend_gate)), cfg.labels.use_trend_gate),
        "min_trend_prob": _get_float(labels.get("min_trend_prob", metrics.get("min_trend_prob", cfg.labels.min_trend_prob)), cfg.labels.min_trend_prob),
        "use_regime_gate": _get_bool(labels.get("use_regime_gate", metrics.get("use_regime_gate", cfg.labels.use_regime_gate)), cfg.labels.use_regime_gate),
        "min_regime_prob": _get_float(labels.get("min_regime_prob", metrics.get("min_regime_prob", cfg.labels.min_regime_prob)), cfg.labels.min_regime_prob),
        "allow_regime_ranging": _get_bool(labels.get("allow_regime_ranging", cfg.labels.allow_regime_ranging), cfg.labels.allow_regime_ranging),
        "allow_regime_trend_up": _get_bool(labels.get("allow_regime_trend_up", cfg.labels.allow_regime_trend_up), cfg.labels.allow_regime_trend_up),
        "allow_regime_trend_down": _get_bool(labels.get("allow_regime_trend_down", cfg.labels.allow_regime_trend_down), cfg.labels.allow_regime_trend_down),
        "allow_regime_volatile": _get_bool(labels.get("allow_regime_volatile", cfg.labels.allow_regime_volatile), cfg.labels.allow_regime_volatile),
        "regime_align_direction": _get_bool(labels.get("regime_align_direction", cfg.labels.regime_align_direction), cfg.labels.regime_align_direction),
        "use_raw_probabilities": tuned_use_raw,
        "use_calibration": tuned_use_cal,
        "ops_cost_enabled": _get_bool(tuner_settings.get("ops_cost_enabled", metrics.get("ops_cost_enabled", 1.0)), True),
        "ops_cost_target_trades_per_day": _get_float(tuner_settings.get("ops_cost_target_trades_per_day", metrics.get("ops_cost_target_trades_per_day", 30.0)), 30.0),
        "ops_cost_c1": _get_float(tuner_settings.get("ops_cost_c1", metrics.get("ops_cost_c1", 0.01)), 0.01),
        "ops_cost_alpha": _get_float(tuner_settings.get("ops_cost_alpha", metrics.get("ops_cost_alpha", 1.7)), 1.7),
        "single_position": _get_bool(tuner_settings.get("single_position", True), True),
        "opposite_signal_policy": str(tuner_settings.get("opposite_signal_policy", "flip")).strip().lower(),
    }


def _coerce_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.number)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except Exception:
            return None
    return None


def _values_match(a: Any, b: Any, tol: float = 1e-8) -> bool:
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_values_match(x, y, tol=tol) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    a_num = _coerce_float(a)
    b_num = _coerce_float(b)
    if a_num is not None and b_num is not None:
        return abs(a_num - b_num) <= tol
    return str(a) == str(b)


def _summary_value_for_key(section: str, key: str, summary: dict) -> Any:
    best_config = summary.get("best_config", {}) if isinstance(summary, dict) else {}
    labels = best_config.get("labels", {}) if isinstance(best_config, dict) else {}
    metrics = summary.get("best_metrics", {}) if isinstance(summary, dict) else {}
    tuner_settings = summary.get("tuner_settings", {}) if isinstance(summary, dict) else {}

    if section == "labels":
        if key == "use_calibration":
            if "use_raw_probabilities" in metrics:
                return not _get_bool(metrics.get("use_raw_probabilities"), False)
            if "use_raw_probabilities" in labels:
                return not _get_bool(labels.get("use_raw_probabilities"), False)
        if key in metrics:
            return metrics.get(key)
        if key in labels:
            return labels.get(key)
        if key in summary:
            return summary.get(key)
        if key == "calibration_method" and key in tuner_settings:
            return tuner_settings.get(key)
        return None

    if section == "features":
        if isinstance(best_config, dict):
            return best_config.get("features", {}).get(key)
        return None

    if section == "model":
        if isinstance(best_config, dict):
            return best_config.get("model", {}).get(key)
        return None

    return None


def _collect_train_summary_mismatches(train_data: dict, summary: dict) -> List[Dict[str, Any]]:
    mismatches: List[Dict[str, Any]] = []
    if not isinstance(train_data, dict) or not isinstance(summary, dict):
        return mismatches

    best_config = summary.get("best_config", {}) if isinstance(summary, dict) else {}

    for section in ("labels", "features", "model"):
        train_section = train_data.get(section)
        if not isinstance(train_section, dict):
            continue
        for key, train_val in train_section.items():
            if section == "model" and key == "model_dir":
                continue
            summary_val = _summary_value_for_key(section, key, summary)
            if summary_val is None:
                continue
            if not _values_match(train_val, summary_val):
                mismatches.append(
                    {
                        "section": section,
                        "name": key,
                        "train": train_val,
                        "summary": summary_val,
                    }
                )

    for key in ("base_timeframe_idx", "seed"):
        if key in train_data and isinstance(best_config, dict) and key in best_config:
            if not _values_match(train_data.get(key), best_config.get(key)):
                mismatches.append(
                    {
                        "section": "root",
                        "name": key,
                        "train": train_data.get(key),
                        "summary": best_config.get(key),
                    }
                )

    return mismatches


def _grade_signal(bounce_prob: float, trend_aligned: bool, is_pullback: bool) -> str:
    if bounce_prob > 0.6 and trend_aligned and is_pullback:
        return "A"
    if bounce_prob > 0.5 and (trend_aligned or is_pullback):
        return "B"
    return "C"


def _resolve_exit_reason(exit_price: float, stop_loss: float, take_profit: float, direction: int) -> str:
    if direction == 1:
        if exit_price <= stop_loss:
            return "stop_loss"
        if exit_price >= take_profit:
            return "take_profit"
    else:
        if exit_price >= stop_loss:
            return "stop_loss"
        if exit_price <= take_profit:
            return "take_profit"
    return "timeout"


@dataclass
class TradeSignal:
    bar_time: int
    direction: int
    bounce_prob: float
    expected_rr: float
    expected_rr_mean: float
    ev_value: float
    quality: str
    trend_prob: float
    regime_prob: float


@dataclass
class OpenPosition:
    entry_time: datetime
    entry_bar_time: int
    direction: int
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    signal_quality: str
    expected_rr: float
    metadata: dict = field(default_factory=dict)


@dataclass
class ClosedTrade:
    entry_time: datetime
    exit_time: datetime
    direction: int
    entry_price: float
    exit_price: float
    size: float
    stop_loss: float
    take_profit: float
    pnl: float
    pnl_percent: float
    signal_quality: str
    exit_reason: str
    duration_seconds: float
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["entry_time"] = self.entry_time.isoformat()
        data["exit_time"] = self.exit_time.isoformat()
        return data


class TradeBuffer:
    """Thread-safe buffer for accumulating trades with a bounded backlog."""

    def __init__(self, max_pending: int, logger: Optional[logging.Logger] = None):
        self._max_pending = max(0, int(max_pending))
        maxlen = self._max_pending if self._max_pending > 0 else None
        self._pending: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._dropped_total = 0
        self._dropped_since_log = 0
        self._last_drop_log_ts = 0.0
        self._logger = logger or logging.getLogger(__name__)
        self.trade_count = 0

    def add_trades_batch(self, trades: List[dict]) -> None:
        if not trades:
            return
        dropped_to_log = 0
        total_dropped = 0
        with self._lock:
            if self._max_pending > 0:
                overflow = len(self._pending) + len(trades) - self._max_pending
                if overflow > 0:
                    self._dropped_total += overflow
                    self._dropped_since_log += overflow
                    now = time.time()
                    if now - self._last_drop_log_ts >= DEFAULT_DROP_LOG_INTERVAL_SECONDS:
                        dropped_to_log = self._dropped_since_log
                        total_dropped = self._dropped_total
                        self._dropped_since_log = 0
                        self._last_drop_log_ts = now
            self._pending.extend(trades)
            self.trade_count += len(trades)
        if dropped_to_log > 0:
            self._logger.warning(
                "Trade buffer overflow: dropped %s trades (total dropped %s). "
                "Consider increasing --max-pending-trades.",
                dropped_to_log,
                total_dropped,
            )

    def drain_pending(self) -> List[dict]:
        with self._lock:
            if not self._pending:
                return []
            pending = list(self._pending)
            self._pending.clear()
            return pending

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def dropped_total(self) -> int:
        with self._lock:
            return int(self._dropped_total)


class LiveTrader:
    """Live trading engine aligned to tuned backtest logic."""

    def __init__(
        self,
        *,
        train_config_path: Path,
        model_dir_override: Optional[Path],
        symbol: str,
        testnet: bool,
        paper: bool,
        api_key: Optional[str],
        api_secret: Optional[str],
        secrets_path: Optional[Path],
        keyring_service: Optional[str],
        leverage: int,
        balance_asset: str,
        bootstrap_csv: Path,
        log_dir: Path,
        log_file: Optional[str],
        max_pending_trades: int,
        recv_window_ms: int = DEFAULT_RECV_WINDOW_MS,
    ):
        self.logger = setup_logging(log_file)
        _set_console_font_size(DEFAULT_CONSOLE_FONT_SIZE, logger=self.logger)
        self.symbol = symbol
        self.testnet = bool(testnet)
        self.paper = bool(paper)
        self.leverage = max(1, int(leverage))
        self.balance_asset = balance_asset
        self._train_config_path = Path(train_config_path)
        self._tuning_summary_path: Optional[Path] = None
        self._warmup_bars_required: Optional[int] = None
        self._warmup_summary: Optional[str] = None
        self._warmup_base_start: Optional[int] = None
        self._warmup_base_end: Optional[int] = None
        self._bootstrap_readiness_missing: Optional[List[str]] = None
        self._max_pending_trades = max(0, int(max_pending_trades))
        self._recv_window_ms = int(recv_window_ms)

        cfg, meta, train_raw = _load_train_config_from_path(train_config_path)
        tuning_path_raw = meta.get("tuning_summary_path")
        if not tuning_path_raw:
            raise SystemExit("Train config missing tuning_summary_path; cannot proceed.")
        tuning_path = Path(tuning_path_raw)
        if not tuning_path.is_absolute():
            candidates = [
                (train_config_path.parent / tuning_path).resolve(),
                (_REPO_DIR / tuning_path).resolve(),
            ]
            tuning_path = next((path for path in candidates if path.exists()), candidates[0])
        self._tuning_summary_path = tuning_path
        summary = _load_tuning_summary(tuning_path)
        mismatches = _collect_train_summary_mismatches(train_raw, summary)
        if mismatches:
            lines = ["Train config does not match tuning summary (refusing to run):"]
            for mismatch in mismatches:
                section = mismatch.get("section")
                name = mismatch.get("name")
                train_val = mismatch.get("train")
                summary_val = mismatch.get("summary")
                lines.append(f"- {section}.{name}: train={train_val} summary={summary_val}")
            raise SystemExit("\n".join(lines))
        _apply_summary_to_config(cfg, summary)

        train_model_dir = cfg.model.model_dir
        if model_dir_override is not None:
            cfg.model.model_dir = Path(model_dir_override)
        else:
            cfg.model.model_dir = Path(train_model_dir)

        if not cfg.model.model_dir.exists():
            raise SystemExit(f"Model directory not found: {cfg.model.model_dir}")

        self.config = cfg
        self.summary = summary
        self.tuned = _extract_tuned_settings(cfg, summary)

        if not self.tuned.get("single_position", True):
            raise SystemExit("Live trading requires single_position=True to match execution.")

        self.base_tf = self.config.features.timeframe_names[self.config.base_timeframe_idx]
        self.base_tf_seconds = int(self.config.features.timeframes[self.config.base_timeframe_idx])

        self.stop_atr_multiple = float(self.tuned["stop_atr_multiple"])
        self.target_rr = float(self.tuned["target_rr"])
        self.pullback_threshold = float(self.tuned["pullback_threshold"])
        self.entry_forward_window = int(self.tuned["entry_forward_window"])
        self.min_bounce_prob = float(self.tuned["best_threshold"])
        self.max_bounce_prob = 1.0
        self.use_ev_gate = bool(self.tuned["use_ev_gate"])
        self.ev_margin_r = float(self.tuned["ev_margin_r"])
        self.fee_percent = float(self.tuned["fee_percent"])
        self.fee_per_trade_r = self.tuned.get("fee_per_trade_r")
        if self.fee_per_trade_r is not None:
            self.fee_per_trade_r = float(self.fee_per_trade_r)
        self.use_expected_rr = bool(self.tuned["use_expected_rr"])
        self.use_trend_gate = bool(self.tuned["use_trend_gate"])
        self.use_regime_gate = bool(self.tuned["use_regime_gate"])
        self.min_trend_prob = float(self.tuned["min_trend_prob"])
        self.min_regime_prob = float(self.tuned["min_regime_prob"])
        self.allow_regime_ranging = bool(self.tuned["allow_regime_ranging"])
        self.allow_regime_trend_up = bool(self.tuned["allow_regime_trend_up"])
        self.allow_regime_trend_down = bool(self.tuned["allow_regime_trend_down"])
        self.allow_regime_volatile = bool(self.tuned["allow_regime_volatile"])
        self.regime_align_direction = bool(self.tuned["regime_align_direction"])
        self.ops_cost_enabled = bool(self.tuned["ops_cost_enabled"])
        self.ops_cost_target = float(self.tuned["ops_cost_target_trades_per_day"])
        self.ops_cost_c1 = float(self.tuned["ops_cost_c1"])
        self.ops_cost_alpha = float(self.tuned["ops_cost_alpha"])
        self.opposite_signal_policy = str(self.tuned["opposite_signal_policy"] or "ignore").strip().lower()
        if self.opposite_signal_policy not in {"ignore", "close", "flip"}:
            self.opposite_signal_policy = "ignore"

        self.use_raw_probabilities = bool(self.tuned["use_raw_probabilities"])
        self.use_calibration = bool(self.tuned["use_calibration"])
        if self.use_raw_probabilities:
            self.use_calibration = False

        if not bootstrap_csv:
            raise SystemExit("--bootstrap-csv is required for live trading.")
        self.bootstrap_csv = Path(bootstrap_csv)

        self.models = TrendFollowerModels(self.config.model)
        self.models.load_all(self.config.model.model_dir)

        entry_model = getattr(self.models, "entry_model", None)
        if entry_model is None:
            raise SystemExit("Entry model missing; cannot trade.")
        entry_features = getattr(entry_model, "filtered_feature_names", None) or getattr(entry_model, "feature_names", None)
        if not entry_features:
            raise SystemExit("Entry model missing feature names.")
        self.entry_feature_names = list(entry_features)
        self.expected_non_context = [
            name for name in self.entry_feature_names if name not in CONTEXT_FEATURE_NAMES
        ]

        self._require_trend_context = any(name.startswith("trend_prob_") for name in self.entry_feature_names)
        self._require_regime_context = any(name.startswith("regime_prob_") for name in self.entry_feature_names)

        if self._require_trend_context and (self.models.trend_classifier is None or self.models.trend_classifier.model is None):
            raise SystemExit("Entry model expects trend context features, but trend model is missing.")
        if self._require_regime_context and (self.models.regime_classifier is None or self.models.regime_classifier.model is None):
            raise SystemExit("Entry model expects regime context features, but regime model is missing.")

        entry_readiness = train_raw.get("entry_feature_readiness") if isinstance(train_raw, dict) else None
        if not isinstance(entry_readiness, dict):
            raise SystemExit(
                "Train config missing entry_feature_readiness; retrain with readiness snapshot."
            )
        ready_features = entry_readiness.get("ready_features")
        if not isinstance(ready_features, list) or not ready_features:
            raise SystemExit("entry_feature_readiness missing ready_features.")
        ready_non_context = [f for f in ready_features if f not in CONTEXT_FEATURE_NAMES]
        if not ready_non_context:
            raise SystemExit("entry_feature_readiness has no non-context features.")

        unexpected_ready = [f for f in ready_non_context if f not in self.expected_non_context]
        if unexpected_ready:
            bad = ", ".join(sorted(unexpected_ready))
            raise SystemExit(
                "entry_feature_readiness has unexpected features: "
                f"{bad}"
            )

        self.entry_feature_readiness = entry_readiness
        self.entry_ready_features = sorted(set(ready_non_context))
        self.entry_mask_features = [
            name for name in self.expected_non_context if name not in self.entry_ready_features
        ]
        self.entry_readiness_window = entry_readiness.get("window_bars") or entry_readiness.get("window_used")
        if self.use_trend_gate and (self.models.trend_classifier is None or self.models.trend_classifier.model is None):
            raise SystemExit("Trend gate enabled but trend model is missing.")
        if self.use_regime_gate and (self.models.regime_classifier is None or self.models.regime_classifier.model is None):
            raise SystemExit("Regime gate enabled but regime model is missing.")

        self.predictor = self._init_predictor()

        self._bybit: Optional[BybitClient] = None
        self._instrument_info: Optional[dict] = None
        self._qty_step: float = 0.0
        self._min_qty: float = 0.0
        self._tick_size: float = 0.0
        self._min_notional: float = 5.0
        self._api_source: Optional[str] = None

        if not self.paper:
            resolved_key, resolved_secret, source = _resolve_api_credentials(
                api_key=api_key,
                api_secret=api_secret,
                secrets_path=secrets_path,
                keyring_service=keyring_service,
                logger=self.logger,
            )
            self._api_source = source
            self._bybit = BybitClient(
                api_key=resolved_key,
                api_secret=resolved_secret,
                testnet=self.testnet,
                recv_window_ms=self._recv_window_ms,
            )

        self.trade_buffer = TradeBuffer(max_pending_trades, logger=self.logger)
        self.aggregators: Dict[str, IncrementalBarAggregator] = {
            tf_name: IncrementalBarAggregator(tf_seconds)
            for tf_name, tf_seconds in zip(self.config.features.timeframe_names, self.config.features.timeframes)
        }

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        ts = _utc_now().strftime("%Y%m%d_%H%M%S")
        self.trades_file = self.log_dir / f"live_trades_{self.symbol}_{ts}.jsonl"
        self.stats_file = self.log_dir / f"live_stats_{self.symbol}_{ts}.json"
        self._session_start_time = _utc_now()
        self._session_id = self._session_start_time.strftime("%Y%m%d_%H%M%S")
        self._session_log_dir = Path(DEFAULT_SESSION_LOG_DIR) / self.symbol
        self._session_log_dir.mkdir(parents=True, exist_ok=True)
        self._session_log_path = self._session_log_dir / f"{self._session_id}_running.jsonl"
        self._session_log_lock = threading.Lock()
        self._session_event_lock = threading.Lock()
        self._session_events = deque(maxlen=DEFAULT_SESSION_EVENT_BUFFER)
        self._session_event_handler = _SessionEventHandler(self._record_event)
        logging.getLogger().addHandler(self._session_event_handler)
        self._session_start_written = False
        self._session_finalized = False
        self._last_model_diag: Optional[Dict[str, Any]] = None
        self._last_model_inputs: Optional[Dict[str, Any]] = None
        self._top_entry_features = self._compute_entry_top_features(20)

        self.position: Optional[OpenPosition] = None
        self.closed_trades: List[ClosedTrade] = []
        self.paper_capital = float(DEFAULT_PAPER_CAPITAL)

        self._accepted_trades = 0
        self._span_start_bar_time: Optional[int] = None
        self._base_bar_count = 0
        self._close_requested = False
        self._close_request_reason: Optional[str] = None
        self._pending_flip_signal: Optional[TradeSignal] = None
        self._pending_flip_bar: Optional[dict] = None

        self._last_trade_wallclock: Optional[float] = None
        self._last_connect_wallclock: Optional[float] = None
        self._last_health_check: Optional[float] = None
        self._stale_data_threshold_sec = float(
            max(DEFAULT_STALE_DATA_THRESHOLD_SECONDS, self.base_tf_seconds)
        )
        self._health_check_interval_sec = float(DEFAULT_HEALTH_CHECK_INTERVAL_SECONDS)
        self._heartbeat_interval_sec = float(DEFAULT_HEARTBEAT_INTERVAL_SECONDS)
        self._last_heartbeat_time: Optional[float] = None
        self._last_heartbeat_trade_count: int = 0
        self._last_heartbeat_bar_count: int = 0
        self._last_health_reason: Optional[str] = None
        self._reconnect_attempts = 0
        self._reconnect_total = 0
        self._max_reconnect_attempts = int(DEFAULT_MAX_RECONNECT_ATTEMPTS)
        self._reconnect_backoff_base = float(DEFAULT_RECONNECT_BACKOFF_BASE_SECONDS)
        self._reconnect_cooldown_sec = float(DEFAULT_RECONNECT_COOLDOWN_SECONDS)
        self._last_reconnect_time: Optional[float] = None
        self._last_reconnect_block_log: Optional[float] = None
        self._pause_entries = False
        self._pause_reason: Optional[str] = None
        self._pause_since: Optional[float] = None
        self._entry_pause_log_interval_sec = float(DEFAULT_ENTRY_PAUSE_LOG_INTERVAL_SECONDS)
        self._last_pause_log_time: Optional[float] = None
        self._manual_pause = False
        self._manual_override_readiness = False
        self._last_readiness_override_log: Optional[float] = None
        self._key_listener: Optional[_KeyListener] = None
        self._readiness_missing_features: List[str] = []
        self._latency_samples: deque = deque(maxlen=DEFAULT_LATENCY_WINDOW_SIZE)
        self._last_trade_latency: Optional[float] = None
        self._last_trade_price: Optional[float] = None
        self._latency_alert_seconds = float(DEFAULT_LATENCY_ALERT_SECONDS)
        self._latency_unstable_ratio = float(DEFAULT_LATENCY_UNSTABLE_RATIO)
        self._last_position_poll: float = 0.0
        self._position_poll_interval = 5.0
        self._account_refresh_sec = float(DEFAULT_ACCOUNT_REFRESH_SECONDS)
        self._last_balance_update: Optional[float] = None
        self._last_balance: Optional[float] = None
        self._status_log_interval_sec = float(DEFAULT_STATUS_LOG_INTERVAL_SECONDS)
        self._last_status_log_time: Optional[float] = None

        self.running = False
        self.ws = None

        self._bootstrap_warmup()
        self._verify_feature_schema()
        self._log_health_summary()
        self._log_clock_skew()
        self._write_session_start()

        if not self.paper:
            self._sync_existing_exchange_position()

    # ----------------------------------------------------------------- warmup
    def _compute_required_warmup_bars(self) -> int:
        touch_threshold = getattr(self.config.labels, "pullback_threshold", None)
        if touch_threshold is None:
            touch_threshold = getattr(self.config.labels, "touch_threshold_atr", 0.3)
        if touch_threshold is None:
            touch_threshold = 0.3

        min_slope_norm = getattr(self.config.labels, "min_slope_norm", 0.03)
        if min_slope_norm is None:
            min_slope_norm = 0.03

        engine = IncrementalFeatureEngine(
            self.config.features,
            base_tf=self.base_tf,
            pullback_ema=getattr(self.config.labels, "pullback_ema", None),
            touch_threshold_atr=float(touch_threshold),
            min_slope_norm=float(min_slope_norm),
        )
        required = []
        for tf_state in engine.tf_states.values():
            _, req = tf_state.get_warmup_progress()
            required.append(int(req))
        swing_req = int(self.config.features.swing_lookback) * 2 + 1
        required.append(swing_req)
        return max(required) if required else 1

    def _init_predictor(self) -> TrendFollowerPredictor:
        predictor = TrendFollowerPredictor(
            self.config,
            use_incremental=True,
            use_calibration=self.use_calibration,
        )
        if not predictor.use_incremental:
            raise SystemExit("Incremental feature engine is required but unavailable.")

        predictor.models = self.models
        predictor.feature_cols = list(self.entry_feature_names)
        trend_feature_names = []
        if self.models.trend_classifier is not None and self.models.trend_classifier.model is not None:
            trend_feature_names = list(getattr(self.models.trend_classifier, "feature_names", []) or self.expected_non_context)
        predictor.trend_feature_cols = trend_feature_names
        return predictor

    def _bootstrap_warmup(self) -> None:
        if not self.bootstrap_csv.exists():
            raise SystemExit(f"Bootstrap CSV path not found: {self.bootstrap_csv}")
        bootstrap_path = self.bootstrap_csv
        if bootstrap_path.is_file():
            self.config.data.data_dir = bootstrap_path.parent.resolve()
            self.config.data.file_pattern = bootstrap_path.name
        else:
            self.config.data.data_dir = bootstrap_path.resolve()

        try:
            import rust_pipeline_bridge as rust_bridge
        except Exception as exc:
            raise SystemExit(f"Rust pipeline import failed: {exc}")
        if not rust_bridge.is_available():
            raise SystemExit("Rust pipeline module is not available; cannot bootstrap.")

        warmup_bars = self._compute_required_warmup_bars()
        target_bars = int(warmup_bars)
        max_bars = max(int(warmup_bars), int(DEFAULT_WARMUP_MAX_BARS))
        last_counts: Optional[Dict[str, int]] = None

        for attempt in range(1, DEFAULT_WARMUP_MAX_ATTEMPTS + 1):
            self._warmup_bars_required = target_bars
            self.logger.info(
                "Bootstrap warmup using rust pipeline (%s bars per TF).", target_bars
            )
            bars_dict = rust_bridge.build_warmup_bars_from_config(self.config, target_bars)
            if not bars_dict:
                raise SystemExit("Rust pipeline returned no bars for bootstrap.")

            missing = []
            short = []
            counts: Dict[str, int] = {}
            for tf_name in self.config.features.timeframe_names:
                df = bars_dict.get(tf_name)
                if df is None or df.empty:
                    missing.append(tf_name)
                    counts[tf_name] = 0
                    continue
                counts[tf_name] = int(len(df))
                if len(df) < target_bars:
                    short.append((tf_name, len(df)))

            if missing:
                lines = ["Bootstrap data is insufficient for warmup:"]
                lines.append(f"- Missing bars for: {', '.join(missing)}")
                raise SystemExit("\n".join(lines))
            if short:
                short_msg = ", ".join([f"{tf}:{count}" for tf, count in short])
                if not isinstance(self.entry_feature_readiness, dict):
                    raise SystemExit(
                        "Bootstrap data is insufficient for warmup:\n"
                        f"- Not enough bars (<{target_bars}) for: {short_msg}"
                    )
                self.logger.warning(
                    "Bootstrap bars below warmup target (<%s) for: %s. "
                    "Readiness snapshot will enforce required features.",
                    target_bars,
                    short_msg,
                )

            self.predictor = self._init_predictor()
            self.predictor.warm_up_incremental(bars_dict)
            rust_bridge.clear_trades_cache()

            engine = getattr(self.predictor, "incremental_engine", None)
            if engine is not None:
                summary = engine.get_warmup_summary()
                self._warmup_summary = summary
                if not engine.is_warmed_up():
                    if not isinstance(self.entry_feature_readiness, dict):
                        raise SystemExit(f"Incremental features not fully warmed up: {summary}")
                    self.logger.warning(
                        "Incremental features not fully warmed up: %s. "
                        "Proceeding with readiness snapshot.",
                        summary,
                    )

            base_bars = bars_dict.get(self.base_tf)
            if base_bars is not None and not base_bars.empty:
                self._span_start_bar_time = int(base_bars["bar_time"].iloc[0])
                self._base_bar_count = len(base_bars)
                self._warmup_base_start = int(base_bars["bar_time"].iloc[0])
                self._warmup_base_end = int(base_bars["bar_time"].iloc[-1])

            non_finite = self._get_non_finite_ready_features(self.predictor.incremental_features)
            if not non_finite:
                self.logger.info("Bootstrap warmup completed.")
                return

            no_growth = last_counts is not None and all(
                counts.get(tf) == last_counts.get(tf) for tf in counts
            )
            if no_growth or attempt >= DEFAULT_WARMUP_MAX_ATTEMPTS or target_bars >= max_bars:
                non_finite_str = ", ".join(non_finite)
                self._bootstrap_readiness_missing = list(non_finite)
                self.logger.warning(
                    "Bootstrap warmup completed with readiness gaps. "
                    "Non-finite ready features: %s. Entries will remain paused until ready.",
                    non_finite_str,
                )
                return

            next_bars = min(max_bars, int(target_bars * 2))
            self.logger.warning(
                "Required features not finite after warmup (%s bars). Retrying with %s bars.",
                target_bars,
                next_bars,
            )
            last_counts = counts
            target_bars = next_bars

        raise SystemExit("Bootstrap warmup failed; unable to satisfy readiness requirements.")

    def _verify_feature_schema(self) -> None:
        features = self.predictor.incremental_features
        if not features:
            raise SystemExit("Incremental features not initialized after bootstrap.")

        required = self.entry_ready_features or self.expected_non_context
        missing = [name for name in required if name not in features]
        if missing:
            missing_str = ", ".join(missing)
            raise SystemExit(
                "Feature mismatch between live features and entry model. "
                f"Missing ({len(missing)}): {missing_str}"
            )

        self.logger.info(
            "Feature schema OK: %s ready features, %s masked.",
            len(required),
            len(self.entry_mask_features),
        )
        self._update_entry_readiness(features)

    def _get_non_finite_ready_features(self, features: dict) -> List[str]:
        required = self.entry_ready_features or self.expected_non_context
        non_finite = []
        for name in required:
            value = _to_float(features.get(name), None)
            if value is None or not np.isfinite(value):
                non_finite.append(name)
        return non_finite

    def _update_entry_readiness(self, features: dict) -> None:
        if not features:
            return
        missing = self._get_non_finite_ready_features(features)
        missing_sorted = sorted(missing)
        if missing_sorted:
            if self._manual_override_readiness:
                now = time.time()
                if (
                    missing_sorted != self._readiness_missing_features
                    or self._last_readiness_override_log is None
                    or (now - self._last_readiness_override_log) >= self._entry_pause_log_interval_sec
                ):
                    self._readiness_missing_features = missing_sorted
                    self._last_readiness_override_log = now
                    self.logger.warning(
                        "Entry readiness override active; missing features: %s",
                        ", ".join(missing_sorted),
                    )
                return

            if missing_sorted != self._readiness_missing_features:
                self._readiness_missing_features = missing_sorted
                self.logger.warning(
                    "Entry readiness incomplete: non-finite features: %s",
                    ", ".join(missing_sorted),
                )
            self._set_pause_entries("entry_readiness_non_finite")
            return

        self._readiness_missing_features = []
        if self._pause_entries and self._pause_reason == "entry_readiness_non_finite":
            self._clear_pause_entries("entry_readiness_ok")

    def _apply_entry_readiness_mask(self, X_entry: pd.DataFrame) -> pd.DataFrame:
        if not self.entry_mask_features:
            return X_entry
        for col in self.entry_mask_features:
            if col in X_entry.columns:
                X_entry[col] = 0.0
        return X_entry

    def _log_health_summary(self) -> None:
        mode = "paper" if self.paper else "live"
        warmup_info = self._warmup_summary or "unknown"
        warmup_bars = self._warmup_bars_required or 0
        warmup_range = "n/a"
        if self._warmup_base_start is not None and self._warmup_base_end is not None:
            warmup_range = f"{self._warmup_base_start}->{self._warmup_base_end}"

        self.logger.info(
            "Health: Mode = %s | Testnet = %s | Symbol = %s | Leverage = %s | Balance asset = %s",
            mode,
            self.testnet,
            self.symbol,
            self.leverage,
            self.balance_asset,
        )
        if not self.paper:
            self.logger.info(
                "Health: API credentials = %s",
                self._api_source or "unknown",
            )
            self.logger.info("Health: Recv window ms = %s", self._recv_window_ms)
        self.logger.info(
            "Health: Train config = %s | Tuning summary = %s | Model dir = %s",
            self._train_config_path,
            self._tuning_summary_path,
            self.config.model.model_dir,
        )
        self.logger.info(
            "Health: Base TF = %s | Timeframes = %s",
            self.base_tf,
            ",".join(self.config.features.timeframe_names),
        )
        self.logger.info(
            "Health: Pullback EMA = %s | Pullback threshold = %s | Stop ATR = %s | Target RR = %s | Entry forward window = %s",
            getattr(self.config.labels, "pullback_ema", None),
            self.pullback_threshold,
            self.stop_atr_multiple,
            self.target_rr,
            self.entry_forward_window,
        )
        self.logger.info(
            "Health: Use raw = %s | Use calibration = %s | Use EV gate = %s | EV margin R = %s | Use expected RR = %s",
            self.use_raw_probabilities,
            self.use_calibration,
            self.use_ev_gate,
            self.ev_margin_r,
            self.use_expected_rr,
        )
        self.logger.info(
            "Health: Trend gate = %s | Min trend prob = %s | Regime gate = %s | Min regime prob = %s | Regime align = %s",
            self.use_trend_gate,
            self.min_trend_prob,
            self.use_regime_gate,
            self.min_regime_prob,
            self.regime_align_direction,
        )
        self.logger.info(
            "Health: Opposite policy = %s | Single position = %s | Max pending trades = %s",
            self.opposite_signal_policy,
            self.tuned.get("single_position", True),
            self._max_pending_trades,
        )
        self.logger.info(
            "Health: Entry features = %s | Non-context features = %s | Warmup bars = %s | Warmup status = %s | Warmup range = %s",
            len(self.entry_feature_names),
            len(self.expected_non_context),
            warmup_bars,
            warmup_info,
            warmup_range,
        )
        readiness_mode = None
        if isinstance(self.entry_feature_readiness, dict):
            readiness_mode = self.entry_feature_readiness.get("mode")
        self.logger.info(
            "Health: Entry readiness ready = %s | Masked = %s | Window = %s | Mode = %s",
            len(self.entry_ready_features),
            len(self.entry_mask_features),
            self.entry_readiness_window,
            readiness_mode or "unknown",
        )
        if self._readiness_missing_features:
            self.logger.info(
                "Health: Entry readiness status = Pending | Missing = %s",
                ", ".join(self._readiness_missing_features),
            )
        else:
            self.logger.info("Health: Entry readiness status = Ready")
        self.logger.info(
            "Health: WS reconnect stale sec = %s | Check sec = %s | Max attempts = %s | Backoff base = %s | Cooldown sec = %s",
            self._stale_data_threshold_sec,
            self._health_check_interval_sec,
            self._max_reconnect_attempts,
            self._reconnect_backoff_base,
            self._reconnect_cooldown_sec,
        )
        self.logger.info(
            "Health: Heartbeat interval sec = %s", self._heartbeat_interval_sec
        )
        self.logger.warning(
            "Health: Residual gaps = ops-cost trade-rate uses live span since warmup (can be conservative early vs. backtest)."
        )

    def _log_clock_skew(self) -> None:
        if self.paper or self._bybit is None:
            return
        try:
            resp = self._bybit.session.get_server_time()
        except Exception as exc:
            self.logger.warning("Clock skew check failed: %s", exc)
            return
        if not resp or resp.get("retCode", -1) != 0:
            self.logger.warning("Clock skew check failed: %s", resp.get("retMsg") if isinstance(resp, dict) else resp)
            return

        server_ms: Optional[int] = None
        result = resp.get("result", {}) if isinstance(resp, dict) else {}
        if isinstance(result, dict):
            time_nano = result.get("timeNano")
            time_second = result.get("timeSecond")
            try:
                if time_nano is not None:
                    server_ms = int(int(time_nano) / 1_000_000)
                elif time_second is not None:
                    server_ms = int(float(time_second) * 1000)
            except (TypeError, ValueError):
                server_ms = None
        if server_ms is None and isinstance(resp, dict):
            try:
                fallback = resp.get("time")
                if fallback is not None:
                    server_ms = int(float(fallback) * 1000)
            except (TypeError, ValueError):
                server_ms = None

        if server_ms is None:
            self.logger.warning("Clock skew check failed: unable to parse server time.")
            return

        local_ms = int(time.time() * 1000)
        skew_ms = local_ms - server_ms
        self.logger.info(
            "Clock skew: Local ms = %s | Server ms = %s | Skew ms = %s (local - server)",
            local_ms,
            server_ms,
            skew_ms,
        )
        if abs(skew_ms) > 2000:
            self.logger.warning(
                "Clock skew exceeds 2s; sync system time or increase recv_window_ms."
            )

    # --------------------------------------------------------------- exchange
    def _load_instrument_filters(self) -> None:
        if self._instrument_info is not None or self._bybit is None:
            return
        info = self._bybit.get_instrument_info(self.symbol)
        self._instrument_info = info or {}
        lot = (info or {}).get("lotSizeFilter", {}) if isinstance(info, dict) else {}
        price_filter = (info or {}).get("priceFilter", {}) if isinstance(info, dict) else {}

        self._qty_step = _to_float(lot.get("qtyStep"), 0.0) or 0.0
        self._min_qty = _to_float(lot.get("minOrderQty"), 0.0) or 0.0
        self._tick_size = _to_float(price_filter.get("tickSize"), 0.0) or 0.0
        min_notional = _to_float(lot.get("minNotionalValue"), None)
        if min_notional is not None and min_notional > 0:
            self._min_notional = min_notional

    def _round_qty(self, qty: float) -> float:
        self._load_instrument_filters()
        if self._qty_step > 0:
            qty = _floor_to_step(qty, self._qty_step)
        if self._min_qty > 0 and qty < self._min_qty:
            return 0.0
        return float(qty)

    # --------------------------------------------------------------- websocket
    def _connect_websocket(self) -> None:
        self.ws = WebSocket(testnet=self.testnet, channel_type="linear")
        self.ws.trade_stream(symbol=self.symbol, callback=self._handle_trade_message)
        self._last_connect_wallclock = time.time()
        self.logger.info(f"Subscribed to publicTrade.{self.symbol}")

    def _handle_trade_message(self, message: dict) -> None:
        try:
            data = message.get("data")
            if not isinstance(data, list):
                return
            self.trade_buffer.add_trades_batch(data)
            self._last_trade_wallclock = time.time()
            if self._reconnect_attempts > 0:
                self.logger.info("WebSocket connection restored.")
                self._reconnect_attempts = 0
            if self._pause_entries and self._pause_reason in {
                "ws_missing",
                "latency_stale",
                "stale_no_trades",
                "ws_disconnected",
                "stale_hard",
                "stale_soft",
            }:
                self._clear_pause_entries("data_resumed")
        except Exception as exc:
            self.logger.warning(f"Trade message error: {exc}")

    def _set_pause_entries(self, reason: str) -> None:
        if self._manual_pause:
            if not self._pause_entries or self._pause_reason != "manual_pause":
                self._pause_entries = True
                self._pause_reason = "manual_pause"
                self._pause_since = time.time()
            self.logger.warning("Entry paused: Reason = manual_pause")
            return
        if self._pause_entries and self._pause_reason == reason:
            return
        self._pause_entries = True
        self._pause_reason = reason
        self._pause_since = time.time()
        self.logger.warning(f"Entry paused: Reason = {reason}")

    def _clear_pause_entries(self, reason: str) -> None:
        if self._manual_pause:
            return
        if not self._pause_entries:
            return
        self._pause_entries = False
        self._pause_reason = None
        self._pause_since = None
        self.logger.warning(f"Entry resumed: Reason = {reason}")

    def _handle_keypress(self, key: str) -> None:
        if not key:
            return
        key = key.lower()
        if key == "p":
            self._manual_pause = True
            self._manual_override_readiness = False
            self._set_pause_entries("manual_pause")
            self.logger.warning("Manual pause enabled (P).")
            return
        if key == "r":
            self._manual_pause = False
            self._manual_override_readiness = True
            if self._pause_reason in {"manual_pause", "entry_readiness_non_finite"}:
                self._clear_pause_entries("manual_resume")
            self.logger.warning("Manual resume enabled (R); readiness checks overridden.")

    def _start_key_listener(self) -> None:
        if self._key_listener is not None:
            return
        self._key_listener = _KeyListener(self._handle_keypress, self.logger)
        self._key_listener.start()

    def _stop_key_listener(self) -> None:
        if self._key_listener is None:
            return
        self._key_listener.stop()
        self._key_listener.join(timeout=1.0)
        self._key_listener = None

    def _refresh_account_balance(self, now: float) -> None:
        if self.paper:
            if self._last_balance_update is not None:
                if (now - self._last_balance_update) < self._account_refresh_sec:
                    return
            self._last_balance = float(self.paper_capital)
            self._last_balance_update = now
            return
        if self._bybit is None:
            return
        if self._last_balance_update is not None:
            if (now - self._last_balance_update) < self._account_refresh_sec:
                return
        balance = self._bybit.get_available_balance(asset=self.balance_asset, logger=self.logger)
        self._last_balance = float(balance)
        self._last_balance_update = now
        if balance <= 0:
            self.logger.warning(
                "ALERT: balance fetch returned %.4f %s; check API credentials or account.",
                balance,
                self.balance_asset,
            )

    def _format_status_line(self, now: float) -> str:
        balance_str = "n/a"
        if self._last_balance is not None:
            age = None
            if self._last_balance_update is not None:
                age = now - self._last_balance_update
            if age is None:
                balance_str = f"{self._last_balance:.4f} {self.balance_asset}"
            else:
                balance_str = f"{self._last_balance:.4f} {self.balance_asset} (Age = {age:.0f}s)"

        if self.position is None:
            return f"Position = FLAT | Balance = {balance_str}"

        pos = self.position
        side = "LONG" if pos.direction == 1 else "SHORT"
        age_min = _safe_age_seconds(pos.entry_time, _utc_now()) / 60.0
        last_price = self._last_trade_price or pos.entry_price
        pnl = pos.direction * (last_price - pos.entry_price) * pos.size if last_price is not None else 0.0
        return (
            f"Position = {side} | Size = {pos.size:.4f} | Entry = {pos.entry_price:.6f} "
            f"| Stop = {pos.stop_loss:.6f} | TP = {pos.take_profit:.6f} "
            f"| PnL = {pnl:+.2f} | Age = {age_min:.1f}m | Balance = {balance_str}"
        )

    def _format_bar_time(self, bar_time: Optional[int]) -> str:
        if bar_time is None:
            return "n/a"
        try:
            return datetime.fromtimestamp(int(bar_time), timezone.utc).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return str(bar_time)

    def _compute_entry_top_features(self, n: int) -> List[str]:
        if n <= 0:
            return []
        entry_model = getattr(self.models, "entry_model", None)
        if entry_model is None:
            return []

        names = list(
            entry_model.filtered_feature_names
            or entry_model.feature_names
            or self.entry_feature_names
        )
        importances = None
        if entry_model.use_ensemble and entry_model.ensemble_classifiers:
            rows = [
                clf.feature_importances_
                for clf in entry_model.ensemble_classifiers
                if hasattr(clf, "feature_importances_")
            ]
            if rows:
                importances = np.mean(np.asarray(rows, dtype=float), axis=0)
        elif entry_model.classifier is not None and hasattr(entry_model.classifier, "feature_importances_"):
            importances = np.asarray(entry_model.classifier.feature_importances_, dtype=float)

        if importances is None:
            return names[: min(n, len(names))]

        if len(importances) != len(names):
            min_len = min(len(importances), len(names))
            self.logger.warning(
                "ALERT: entry feature importances length mismatch (names=%s importances=%s).",
                len(names),
                len(importances),
            )
            names = names[:min_len]
            importances = importances[:min_len]

        if importances.size == 0:
            return []
        idx = np.argsort(importances)[::-1]
        return [names[i] for i in idx[: min(n, len(idx))]]

    def _extract_top_feature_values(self, X_entry: pd.DataFrame) -> Dict[str, Any]:
        if X_entry is None or X_entry.empty or not self._top_entry_features:
            return {}
        row = X_entry.iloc[0]
        values: Dict[str, Any] = {}
        for name in self._top_entry_features:
            if name not in X_entry.columns:
                continue
            val = row.get(name)
            if isinstance(val, (float, np.floating)) and not np.isfinite(val):
                val = None
            values[name] = _json_safe(val)
        return values

    def _record_event(self, event: Dict[str, Any]) -> None:
        if not event:
            return
        if "ts" not in event:
            event["ts"] = _utc_now().isoformat()
        with self._session_event_lock:
            self._session_events.append(event)

    def _drain_session_events(self) -> List[Dict[str, Any]]:
        with self._session_event_lock:
            if not self._session_events:
                return []
            events = list(self._session_events)
            self._session_events.clear()
            return events

    def _append_session_event(self, event: Dict[str, Any]) -> None:
        if not event:
            return
        event = _json_safe(event)
        event["session_id"] = self._session_id
        try:
            with self._session_log_lock:
                with open(self._session_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event, ensure_ascii=True) + "\n")
        except Exception as exc:
            self.logger.warning(f"Could not write session log: {exc}")

    def _position_snapshot(self) -> Optional[Dict[str, Any]]:
        if self.position is None:
            return None
        pos = self.position
        last_price = self._last_trade_price
        pnl = None
        if last_price is not None:
            pnl = pos.direction * (last_price - pos.entry_price) * pos.size
        return _json_safe(
            {
                "entry_time": pos.entry_time,
                "entry_bar_time": pos.entry_bar_time,
                "direction": pos.direction,
                "entry_price": pos.entry_price,
                "size": pos.size,
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
                "signal_quality": pos.signal_quality,
                "expected_rr": pos.expected_rr,
                "pnl": pnl,
                "metadata": pos.metadata,
            }
        )

    def _account_snapshot(self) -> Dict[str, Any]:
        balance_age = None
        if self._last_balance_update is not None:
            balance_age = time.time() - self._last_balance_update
        return _json_safe(
            {
                "balance": self._last_balance,
                "balance_age_sec": balance_age,
                "balance_asset": self.balance_asset,
                "paper_capital": self.paper_capital if self.paper else None,
            }
        )

    def _write_session_start(self) -> None:
        if self._session_start_written:
            return
        entry = {
            "event": "session_start",
            "ts": self._session_start_time.isoformat(),
            "symbol": self.symbol,
            "mode": "paper" if self.paper else "live",
            "testnet": self.testnet,
            "train_config": str(self._train_config_path),
            "tuning_summary": str(self._tuning_summary_path) if self._tuning_summary_path else None,
            "model_dir": str(self.config.model.model_dir),
            "base_tf": self.base_tf,
            "timeframes": list(self.config.features.timeframe_names),
            "entry_features": len(self.entry_feature_names),
            "non_context_features": len(self.expected_non_context),
            "top_features": list(self._top_entry_features),
            "warmup_bars_required": self._warmup_bars_required,
            "warmup_status": self._warmup_summary,
            "warmup_range": (
                f"{self._warmup_base_start}->{self._warmup_base_end}"
                if self._warmup_base_start is not None and self._warmup_base_end is not None
                else None
            ),
            "entry_readiness": {
                "ready_features": list(self.entry_ready_features),
                "mask_features": list(self.entry_mask_features),
                "window_bars": self.entry_readiness_window,
            },
            "gates": {
                "use_ev_gate": self.use_ev_gate,
                "ev_margin_r": self.ev_margin_r,
                "use_trend_gate": self.use_trend_gate,
                "min_trend_prob": self.min_trend_prob,
                "use_regime_gate": self.use_regime_gate,
                "min_regime_prob": self.min_regime_prob,
                "regime_align": self.regime_align_direction,
                "use_expected_rr": self.use_expected_rr,
            },
            "risk": {
                "position_size_pct": DEFAULT_POSITION_SIZE_PCT,
                "stop_atr_multiple": self.stop_atr_multiple,
                "target_rr": self.target_rr,
                "entry_forward_window": self.entry_forward_window,
            },
        }
        self._append_session_event(entry)
        self._session_start_written = True

    def _write_session_bar(self, bar: dict) -> None:
        bar_time = int(bar.get("bar_time")) if bar.get("bar_time") is not None else None
        if bar_time is None:
            return
        diag = dict(self._last_model_diag) if self._last_model_diag is not None else {
            "bar_time": bar_time,
            "decision": "skip",
            "reason": "no_diag",
            "paused": self._pause_entries,
        }
        diag.pop("top_features", None)
        events = self._drain_session_events()
        entry = {
            "event": "bar",
            "ts": _utc_now().isoformat(),
            "bar_time": bar_time,
            "bar_time_utc": self._format_bar_time(bar_time),
            "bar": _json_safe(
                {
                    "open": bar.get("open"),
                    "high": bar.get("high"),
                    "low": bar.get("low"),
                    "close": bar.get("close"),
                    "volume": bar.get("volume"),
                    "trades": bar.get("trades"),
                }
            ),
            "diagnostics": _json_safe(diag),
            "model_inputs": _json_safe(self._last_model_inputs or {}),
            "position": self._position_snapshot(),
            "account": self._account_snapshot(),
            "events": _json_safe(events),
        }
        self._append_session_event(entry)

    def _finalize_session_log(self) -> None:
        if self._session_finalized:
            return
        self._session_finalized = True
        end_time = _utc_now()
        pending_events = self._drain_session_events()
        summary = {
            "event": "session_end",
            "ts": end_time.isoformat(),
            "trades": len(self.closed_trades),
            "paper_capital": self.paper_capital if self.paper else None,
            "events": _json_safe(pending_events),
        }
        self._append_session_event(summary)
        end_stamp = end_time.strftime("%Y%m%d_%H%M%S")
        final_path = self._session_log_dir / f"{self._session_id}_{end_stamp}.jsonl"
        try:
            if self._session_log_path.exists():
                os.replace(self._session_log_path, final_path)
                self._session_log_path = final_path
        except Exception as exc:
            self.logger.warning(f"Could not finalize session log: {exc}")

    def _log_model_diag(self, diag: Dict[str, Any]) -> None:
        self._last_model_diag = dict(diag)
        self._last_model_inputs = diag.get("top_features")
        parts = []
        bar_time = diag.get("bar_time")
        parts.append(f"Bar = {bar_time}")
        parts.append(f"Time = {self._format_bar_time(bar_time)}")
        close = diag.get("close")
        if close is not None:
            parts.append(f"Close = {close:.6f}")
        if "touch" in diag:
            parts.append(f"Touch = {diag.get('touch')}")
        if "trend_dir" in diag:
            parts.append(f"Trend dir = {diag.get('trend_dir')}")
        if "atr" in diag and diag.get("atr") is not None:
            parts.append(f"ATR = {diag.get('atr'):.4f}")
        if "bounce_prob" in diag and diag.get("bounce_prob") is not None:
            parts.append(f"Prob = {diag.get('bounce_prob'):.4f}")
        if "raw_prob" in diag and diag.get("raw_prob") is not None:
            parts.append(f"Raw = {diag.get('raw_prob'):.4f}")
        if "cal_prob" in diag and diag.get("cal_prob") is not None:
            parts.append(f"Cal = {diag.get('cal_prob'):.4f}")
        if "expected_rr" in diag and diag.get("expected_rr") is not None:
            parts.append(f"RR = {diag.get('expected_rr'):.3f}")
        if "ev" in diag and diag.get("ev") is not None:
            parts.append(f"EV = {diag.get('ev'):.4f}")
        if "trend_prob" in diag and diag.get("trend_prob") is not None:
            parts.append(f"Trend prob = {diag.get('trend_prob'):.3f}")
        if "regime_prob" in diag and diag.get("regime_prob") is not None:
            parts.append(f"Regime prob = {diag.get('regime_prob'):.3f}")
        if "decision" in diag:
            parts.append(f"Decision = {diag.get('decision')}")
        if "reason" in diag:
            parts.append(f"Reason = {diag.get('reason')}")
        if "paused" in diag:
            parts.append(f"Paused = {diag.get('paused')}")
        self.logger.info("MODEL: " + " | ".join(parts))

    def _emit_heartbeat(self, now: float) -> None:
        if self._last_heartbeat_time is None:
            self._last_heartbeat_time = now
            self._last_heartbeat_trade_count = int(self.trade_buffer.trade_count)
            self._last_heartbeat_bar_count = int(self._base_bar_count)
            return

        elapsed = now - self._last_heartbeat_time
        if elapsed <= 0:
            return

        trade_count = int(self.trade_buffer.trade_count)
        bar_count = int(self._base_bar_count)
        trades_delta = trade_count - self._last_heartbeat_trade_count
        bars_delta = bar_count - self._last_heartbeat_bar_count
        trades_per_sec = float(trades_delta) / float(elapsed) if elapsed > 0 else 0.0
        bars_per_hr = float(bars_delta) / float(elapsed) * 3600.0 if elapsed > 0 else 0.0
        pending_count = self.trade_buffer.pending_count()
        dropped_total = self.trade_buffer.dropped_total()

        latency_p50 = None
        latency_p95 = None
        if self._latency_samples:
            samples = list(self._latency_samples)
            samples.sort()
            mid = int(len(samples) * 0.5)
            p95 = int(len(samples) * 0.95)
            latency_p50 = samples[min(mid, len(samples) - 1)]
            latency_p95 = samples[min(p95, len(samples) - 1)]

        age_since_trade = None
        if self._last_trade_wallclock is not None:
            age_since_trade = now - self._last_trade_wallclock

        self._refresh_account_balance(now)

        price_str = f"{self._last_trade_price:.6f}" if self._last_trade_price is not None else "n/a"
        self.logger.info(
            "Heartbeat: Trades = %s | Trades/s = %.3f | Bars = %s | Bars/hr = %.2f | Price = %s | Pending = %s | Dropped = %s | Reconnects = %s | Paused = %s",
            trade_count,
            trades_per_sec,
            bar_count,
            bars_per_hr,
            price_str,
            pending_count,
            dropped_total,
            self._reconnect_total,
            self._pause_entries,
        )
        if latency_p50 is not None and latency_p95 is not None:
            self.logger.info(
                "Heartbeat: Latency p50 = %.3fs | Latency p95 = %.3fs | Last latency = %s",
                latency_p50,
                latency_p95,
                f"{self._last_trade_latency:.3f}s" if self._last_trade_latency is not None else "n/a",
            )
            if self._latency_alert_seconds > 0 and latency_p95 > self._latency_alert_seconds:
                self.logger.warning(
                    "ALERT: Latency p95 = %.3fs exceeds %.3fs threshold.",
                    latency_p95,
                    self._latency_alert_seconds,
                )
            if latency_p50 > 0:
                ratio = latency_p95 / latency_p50
                if ratio >= self._latency_unstable_ratio:
                    self.logger.warning(
                        "ALERT: Latency instability p95/p50 = %.2f (p50 = %.3fs p95 = %.3fs).",
                        ratio,
                        latency_p50,
                        latency_p95,
                    )
        if age_since_trade is not None:
            self.logger.info("Heartbeat: Age since trade = %.1fs", age_since_trade)

        if self._pause_entries and self._pause_reason:
            self.logger.warning("Heartbeat: Entry paused reason = %s", self._pause_reason)

        if (
            self._last_status_log_time is None
            or (now - self._last_status_log_time) >= self._status_log_interval_sec
        ):
            self.logger.info("Status: %s", self._format_status_line(now))
            self._last_status_log_time = now

        self._last_heartbeat_time = now
        self._last_heartbeat_trade_count = trade_count
        self._last_heartbeat_bar_count = bar_count

    def _is_ws_connected(self) -> bool:
        if self.ws is None:
            return False
        if not hasattr(self.ws, "is_connected"):
            return True
        try:
            return bool(self.ws.is_connected())
        except Exception as exc:
            self.logger.warning(f"Error checking WebSocket connection: {exc}")
            return False

    def _check_connection_health(self) -> tuple[bool, str]:
        def _mark_unhealthy(reason: str, message: str, *args) -> tuple[bool, str]:
            if self._last_health_reason != reason:
                self.logger.warning(message, *args)
                self._last_health_reason = reason
            return False, reason

        if self.ws is None:
            return _mark_unhealthy("ws_missing", "WebSocket not initialized.")

        now = time.time()
        stale_hard = float(self._stale_data_threshold_sec) * 2.0

        if self._last_trade_latency is not None and self._last_trade_latency > stale_hard:
            return _mark_unhealthy(
                "latency_stale",
                "Data latency %.1fs exceeds hard threshold %.1fs.",
                self._last_trade_latency,
                stale_hard,
            )

        if self._last_trade_wallclock is None:
            if self._last_connect_wallclock is None:
                self._last_health_reason = None
                return True, "awaiting_data"
            elapsed = now - self._last_connect_wallclock
            if elapsed < self._stale_data_threshold_sec:
                self._last_health_reason = None
                return True, "awaiting_data"
            if not self._is_ws_connected():
                return _mark_unhealthy(
                    "ws_disconnected",
                    "WebSocket not connected after %.0fs without trades.",
                    elapsed,
                )
            return _mark_unhealthy(
                "stale_no_trades",
                "No trades received for %.0fs since connect.",
                elapsed,
            )

        time_since_trade = now - self._last_trade_wallclock
        if time_since_trade < self._stale_data_threshold_sec:
            self._last_health_reason = None
            return True, "ok"

        if not self._is_ws_connected():
            return _mark_unhealthy(
                "ws_disconnected",
                "Connection appears dead: no trades for %.0fs, ws_connected=False.",
                time_since_trade,
            )

        if time_since_trade > stale_hard:
            return _mark_unhealthy(
                "stale_hard",
                "Possible silent disconnect: no trades for %.0fs despite ws_connected=True.",
                time_since_trade,
            )

        return _mark_unhealthy(
            "stale_soft",
            "No trades for %.0fs; attempting reconnect.",
            time_since_trade,
        )

    def _reconnect_websocket(self) -> bool:
        now = time.time()
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            if self._last_reconnect_time is None:
                self._last_reconnect_time = now
            cooldown_elapsed = now - self._last_reconnect_time
            cooldown_remaining = self._reconnect_cooldown_sec - cooldown_elapsed
            if cooldown_remaining > 0:
                if (
                    self._last_reconnect_block_log is None
                    or (now - self._last_reconnect_block_log) >= self._health_check_interval_sec
                ):
                    self.logger.error(
                        "Reconnect cooldown active (%.0fs remaining). Entries paused.",
                        cooldown_remaining,
                    )
                    self._last_reconnect_block_log = now
                return False
            self.logger.error(
                "Reconnect cooldown elapsed; retrying connection attempts.",
            )
            self._reconnect_attempts = 0
            self._last_reconnect_block_log = None

        backoff = min(
            self._reconnect_backoff_base ** self._reconnect_attempts,
            60.0,
        )
        self.logger.info(
            "Reconnection attempt %s/%s (waiting %.1fs)...",
            self._reconnect_attempts + 1,
            self._max_reconnect_attempts,
            backoff,
        )
        time.sleep(backoff)

        try:
            if self.ws is not None:
                self.ws.exit()
        except Exception as exc:
            self.logger.warning(f"Error closing existing WebSocket: {exc}")

        self.ws = None
        self._last_trade_wallclock = None

        try:
            self._connect_websocket()
            self._reconnect_attempts += 1
            self._reconnect_total += 1
            self._last_reconnect_time = time.time()
            self.logger.info("Reconnection initiated; waiting for trades.")
            return True
        except Exception as exc:
            self.logger.error(f"Reconnection failed: {exc}")
            self._reconnect_attempts += 1
            self._reconnect_total += 1
            self._last_reconnect_time = time.time()
            return False

    # -------------------------------------------------------------- main loop
    def start(self) -> None:
        self.running = True
        self._connect_websocket()
        self._last_health_check = time.time()
        self.logger.info("Live trader started.")
        self._start_key_listener()
        self.logger.info("Keyboard controls: R=resume (override readiness), P=pause entries.")

        try:
            while self.running:
                now = time.time()
                if (
                    self._last_health_check is None
                    or (now - self._last_health_check) >= self._health_check_interval_sec
                ):
                    self._last_health_check = now
                    healthy, reason = self._check_connection_health()
                    if not healthy:
                        self._set_pause_entries(reason)
                        self._reconnect_websocket()
                now = time.time()
                if (
                    self._last_heartbeat_time is None
                    or (now - self._last_heartbeat_time) >= self._heartbeat_interval_sec
                ):
                    self._emit_heartbeat(now)
                trades = self.trade_buffer.drain_pending()
                for trade in trades:
                    self._process_trade(trade)
                self._poll_exchange_position()
                time.sleep(0.05)
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user.")
        finally:
            self.stop()

    def stop(self) -> None:
        if not self.running:
            try:
                logging.getLogger().removeHandler(self._session_event_handler)
            except Exception:
                pass
            self._finalize_session_log()
            return
        self.running = False
        self._stop_key_listener()
        if self.ws is not None:
            try:
                self.ws.exit()
            except Exception:
                pass
        self._write_stats()
        try:
            logging.getLogger().removeHandler(self._session_event_handler)
        except Exception:
            pass
        self._finalize_session_log()
        self.logger.info("Live trader stopped.")

    # ------------------------------------------------------------- trade flow
    def _process_trade(self, trade: dict) -> None:
        try:
            timestamp_ms = trade.get("T") or trade.get("timestamp")
            price = float(trade.get("p") or trade.get("price"))
            size = float(trade.get("v") or trade.get("size"))
            side = trade.get("S") or trade.get("side") or "Buy"
            tick_dir = trade.get("L") or trade.get("tickDirection") or "ZeroPlusTick"
        except Exception:
            return

        if timestamp_ms is None:
            return
        timestamp = float(timestamp_ms) / 1000.0 if float(timestamp_ms) > 1e11 else float(timestamp_ms)
        now = time.time()
        latency = now - float(timestamp)
        if latency >= 0 and latency < 3600:
            self._latency_samples.append(latency)
            self._last_trade_latency = latency

        side_num = 1 if str(side).lower() == "buy" else -1
        tick_map = {
            "PlusTick": 1,
            "ZeroPlusTick": 0.5,
            "MinusTick": -1,
            "ZeroMinusTick": -0.5,
        }
        tick_dir_num = tick_map.get(str(tick_dir), 0.0)
        self._last_trade_price = price

        if self.paper and self.position is not None:
            self._check_paper_exit(price)

        completed: List[tuple[str, dict]] = []
        for tf_name, agg in self.aggregators.items():
            bar = agg.add_trade(timestamp, price, size, side_num, tick_dir_num)
            if bar is not None:
                completed.append((tf_name, bar))

        if completed:
            # Update non-base TF bars first
            for tf_name, bar in completed:
                if tf_name != self.base_tf:
                    self.predictor.add_bar(tf_name, bar)
            # Update base TF last
            for tf_name, bar in completed:
                if tf_name == self.base_tf:
                    self.predictor.add_bar(tf_name, bar)
                    self._on_base_bar_close(bar)

    def _on_base_bar_close(self, bar: dict) -> None:
        bar_time = int(bar.get("bar_time")) if bar.get("bar_time") is not None else None
        if bar_time is None:
            return
        self._last_model_diag = None
        self._last_model_inputs = None

        try:
            self._base_bar_count += 1
            if self._span_start_bar_time is None:
                self._span_start_bar_time = bar_time

            self._update_entry_readiness(self.predictor.incremental_features)
            if (
                self._pause_entries
                and self._pause_reason == "entry_readiness_non_finite"
                and not self._manual_override_readiness
            ):
                self._log_model_diag(
                    {
                        "bar_time": bar_time,
                        "close": float(bar["close"]),
                        "touch": bool(self.predictor.incremental_features.get("ema_touch_detected", False)),
                        "trend_dir": int(self.predictor.incremental_features.get("ema_touch_direction", 0) or 0),
                        "atr": _to_float(self.predictor.incremental_features.get(f"{self.base_tf}_atr"), None),
                        "decision": "skipped",
                        "reason": "entry_readiness_non_finite",
                        "paused": True,
                        "gates": {
                            "readiness": {
                                "passed": False,
                                "missing": list(self._readiness_missing_features),
                            }
                        },
                    }
                )
                return

            signal = self._evaluate_signal(bar)

            if self.position is not None:
                if self._close_requested:
                    return
                if signal is not None and signal.direction == -self.position.direction:
                    if self.opposite_signal_policy in {"close", "flip"}:
                        self._request_close(
                            exit_price=float(bar["close"]),
                            exit_reason=self.opposite_signal_policy,
                            flip_signal=signal,
                            flip_bar=bar,
                        )
                        return
                if self.entry_forward_window > 0 and self.position.entry_bar_time is not None:
                    bars_elapsed = int((bar_time - self.position.entry_bar_time) / self.base_tf_seconds)
                    if bars_elapsed >= self.entry_forward_window:
                        self._request_close(exit_price=float(bar["close"]), exit_reason="timeout")
                return

            if signal is None:
                return

            if self._pause_entries:
                now = time.time()
                if (
                    self._last_pause_log_time is None
                    or (now - self._last_pause_log_time) >= self._entry_pause_log_interval_sec
                ):
                    reason = self._pause_reason or "unknown"
                    self.logger.warning("Entry paused; signal skipped. Reason = %s", reason)
                    self._last_pause_log_time = now
                return

            self._open_position(signal, bar)
        finally:
            self._write_session_bar(bar)

    # -------------------------------------------------------------- prediction
    def _evaluate_signal(self, bar: dict) -> Optional[TradeSignal]:
        features = self.predictor.incremental_features
        bar_time = int(bar.get("bar_time")) if bar.get("bar_time") is not None else None
        diag: Dict[str, Any] = {
            "bar_time": bar_time,
            "close": _to_float(features.get("close") if features else None, None),
            "paused": self._pause_entries,
            "model_called": False,
            "gates": {},
        }

        def _log(reason: str, decision: str = "skip") -> None:
            diag["reason"] = reason
            diag["decision"] = decision
            self._log_model_diag(diag)

        if not features:
            _log("no_features")
            return None

        ema_touch_detected = bool(features.get("ema_touch_detected", False))
        diag["touch"] = ema_touch_detected
        if not ema_touch_detected:
            _log("no_touch")
            return None

        touch_dir = int(features.get("ema_touch_direction", 0) or 0)
        diag["gates"]["ema_touch"] = {
            "detected": ema_touch_detected,
            "direction": touch_dir,
        }
        slope_key = f"{self.base_tf}_ema_{self.config.labels.pullback_ema}_slope_norm"
        slope_val = _to_float(features.get(slope_key), 0.0) or 0.0
        slope_dir = int(np.sign(slope_val)) if slope_val != 0 else 0
        trend_dir = touch_dir if touch_dir != 0 else slope_dir
        diag["trend_dir"] = trend_dir
        if trend_dir == 0:
            _log("no_trend_dir")
            return None

        atr_key = f"{self.base_tf}_atr"
        atr_val = _to_float(features.get(atr_key), None)
        if atr_val is None or atr_val <= 0:
            _log("invalid_atr")
            return None
        diag["atr"] = atr_val

        entry_price = _to_float(features.get("close"), None)
        if entry_price is None:
            _log("missing_close")
            return None

        latest = pd.DataFrame([features])

        trend_pred = None
        regime_pred = None
        if self.models.trend_classifier is not None and self.models.trend_classifier.model is not None:
            trend_features = getattr(self.models.trend_classifier, "feature_names", None) or self.expected_non_context
            X_trend = latest.reindex(columns=list(trend_features)).fillna(0)
            trend_pred = self.models.trend_classifier.predict(X_trend)

        if self.models.regime_classifier is not None and self.models.regime_classifier.model is not None:
            regime_features = getattr(self.models.regime_classifier, "feature_names", None) or self.expected_non_context
            X_regime = latest.reindex(columns=list(regime_features)).fillna(0)
            regime_pred = self.models.regime_classifier.predict(X_regime)

        X_base = latest.reindex(columns=self.expected_non_context)
        X_base = X_base.fillna(0)
        X_entry = append_context_features(X_base, trend_pred, regime_pred)
        X_entry = X_entry.reindex(columns=self.entry_feature_names)
        X_entry = self._apply_entry_readiness_mask(X_entry)
        X_entry = X_entry.fillna(0)
        diag["model_called"] = True
        diag["top_features"] = self._extract_top_feature_values(X_entry)

        preds = self.models.entry_model.predict(X_entry, use_calibration=self.use_calibration)
        prob_key = "bounce_prob_raw" if self.use_raw_probabilities else "bounce_prob"
        bounce_prob = float(preds.get(prob_key, preds.get("bounce_prob", [0.0]))[0])
        raw_prob = float(preds.get("bounce_prob_raw", [bounce_prob])[0])
        cal_prob = float(preds.get("bounce_prob", [bounce_prob])[0])
        expected_rr = float(preds.get("expected_rr", [self.target_rr])[0])
        expected_rr_mean = float(preds.get("expected_rr_mean", [expected_rr])[0])
        diag.update(
            {
                "bounce_prob": bounce_prob,
                "raw_prob": raw_prob,
                "cal_prob": cal_prob,
                "expected_rr": expected_rr,
                "expected_rr_mean": expected_rr_mean,
            }
        )
        if bounce_prob < 0.0 or bounce_prob > 1.0:
            self.logger.warning(
                "ALERT: bounce_prob out of bounds (%.4f).", bounce_prob
            )

        trend_prob_dir = 0.0
        if trend_pred is not None:
            prob_up = float(trend_pred.get("prob_up", [0.0])[0])
            prob_down = float(trend_pred.get("prob_down", [0.0])[0])
            prob_neutral = float(trend_pred.get("prob_neutral", [0.0])[0])
            trend_prob_dir = prob_up if trend_dir == 1 else prob_down if trend_dir == -1 else prob_neutral
        diag["trend_prob"] = trend_prob_dir

        trend_gate = {
            "enabled": bool(self.use_trend_gate),
            "min_prob": float(self.min_trend_prob),
            "prob": trend_prob_dir,
            "passed": True,
        }
        diag["gates"]["trend"] = trend_gate
        if self.use_trend_gate:
            if trend_pred is None or trend_prob_dir < float(self.min_trend_prob):
                diag["trend_prob"] = trend_prob_dir
                trend_gate["passed"] = False
                _log("trend_gate")
                return None
            trend_gate["passed"] = True
        if trend_prob_dir < 0.0 or trend_prob_dir > 1.0:
            self.logger.warning("ALERT: trend_prob out of bounds (%.3f).", trend_prob_dir)

        regime_prob_dir = 0.0
        regime_id = None
        if regime_pred is not None:
            regime_id = int(regime_pred.get("regime", [0])[0])
            prob_ranging = float(regime_pred.get("prob_ranging", [0.0])[0])
            prob_trend_up = float(regime_pred.get("prob_trend_up", [0.0])[0])
            prob_trend_down = float(regime_pred.get("prob_trend_down", [0.0])[0])
            prob_volatile = float(regime_pred.get("prob_volatile", [0.0])[0])
            if regime_id == 0:
                regime_prob_dir = prob_ranging
            elif regime_id == 1:
                regime_prob_dir = prob_trend_up
            elif regime_id == 2:
                regime_prob_dir = prob_trend_down
            else:
                regime_prob_dir = prob_volatile
        diag["regime_prob"] = regime_prob_dir

        regime_gate = {
            "enabled": bool(self.use_regime_gate),
            "min_prob": float(self.min_regime_prob),
            "prob": regime_prob_dir,
            "regime_id": regime_id,
            "passed": True,
            "allowed": True,
        }
        diag["gates"]["regime"] = regime_gate
        if self.use_regime_gate:
            if regime_pred is None:
                regime_gate["passed"] = False
                regime_gate["allowed"] = False
                _log("regime_gate_missing")
                return None
            allowed = False
            if regime_id == 0:
                allowed = self.allow_regime_ranging
            elif regime_id == 1:
                allowed = self.allow_regime_trend_up
                if self.regime_align_direction and trend_dir != 1:
                    allowed = False
            elif regime_id == 2:
                allowed = self.allow_regime_trend_down
                if self.regime_align_direction and trend_dir != -1:
                    allowed = False
            else:
                allowed = self.allow_regime_volatile
            regime_gate["allowed"] = bool(allowed)
            if (not allowed) or (regime_prob_dir < float(self.min_regime_prob)):
                regime_gate["passed"] = False
                diag["regime_prob"] = regime_prob_dir
                _log("regime_gate")
                return None
            regime_gate["passed"] = True
        if regime_prob_dir < 0.0 or regime_prob_dir > 1.0:
            self.logger.warning("ALERT: regime_prob out of bounds (%.3f).", regime_prob_dir)

        stop_dist = float(self.stop_atr_multiple) * float(atr_val)
        if stop_dist <= 0:
            self.logger.warning("ALERT: stop_dist non-positive (%.6f).", stop_dist)
            _log("invalid_stop_dist")
            return None

        fee_r = 0.0
        if self.fee_per_trade_r is not None and np.isfinite(float(self.fee_per_trade_r)):
            fee_r = float(self.fee_per_trade_r)
        else:
            fee_r = (self.fee_percent * entry_price) / stop_dist if entry_price > 0 else 0.0

        rr_mean = float(self.target_rr)
        rr_cons = rr_mean
        if self.use_expected_rr:
            rr_mean = float(expected_rr_mean)
            rr_cons = float(expected_rr)

        ops_cost_r = 0.0
        if self.ops_cost_enabled and self.ops_cost_target > 0:
            span_days = self._estimate_span_days(bar_time=int(bar["bar_time"]))
            if span_days > 0:
                trade_count = self._accepted_trades + 1
                trade_rate_day = float(trade_count) / float(span_days)
                if trade_rate_day > float(self.ops_cost_target):
                    excess = trade_rate_day - float(self.ops_cost_target)
                    ops_cost_r = float(self.ops_cost_c1) * (
                        (excess / float(self.ops_cost_target)) ** float(self.ops_cost_alpha)
                    )

        ev_components = self.models.entry_model.compute_expected_rr_components(
            np.asarray([bounce_prob], dtype=float),
            np.asarray([rr_mean], dtype=float),
            rr_conservative=np.asarray([rr_cons], dtype=float),
            cost_r=np.asarray([fee_r + ops_cost_r], dtype=float),
        )
        ev_value = float(ev_components["ev_conservative_r"][0])
        diag["ev"] = ev_value

        ev_gate = {
            "enabled": bool(self.use_ev_gate),
            "margin_r": float(self.ev_margin_r),
            "ev": ev_value,
            "passed": True,
        }
        diag["gates"]["ev"] = ev_gate
        if self.use_ev_gate:
            if ev_value <= float(self.ev_margin_r):
                ev_gate["passed"] = False
                _log("ev_gate")
                return None
            ev_gate["passed"] = True
        else:
            prob_gate = {
                "enabled": True,
                "min_prob": float(self.min_bounce_prob),
                "max_prob": float(self.max_bounce_prob),
                "prob": bounce_prob,
                "passed": True,
            }
            diag["gates"]["prob"] = prob_gate
            if bounce_prob < float(self.min_bounce_prob):
                prob_gate["passed"] = False
                _log("min_prob")
                return None
            if bounce_prob > float(self.max_bounce_prob):
                prob_gate["passed"] = False
                _log("max_prob")
                return None

        slope_dir_val = slope_dir
        trend_aligned = True
        if slope_dir_val != 0:
            trend_aligned = (trend_dir == slope_dir_val)
        quality = _grade_signal(bounce_prob, trend_aligned, True)
        _log("accepted", decision="accept")

        return TradeSignal(
            bar_time=int(bar["bar_time"]),
            direction=trend_dir,
            bounce_prob=bounce_prob,
            expected_rr=rr_cons,
            expected_rr_mean=rr_mean,
            ev_value=ev_value,
            quality=quality,
            trend_prob=trend_prob_dir,
            regime_prob=regime_prob_dir,
        )

    def _estimate_span_days(self, bar_time: int) -> float:
        if self._span_start_bar_time is not None:
            span_sec = float(bar_time - self._span_start_bar_time)
            if span_sec > 0:
                return span_sec / 86400.0
        if self._base_bar_count > 1:
            span_sec = float(self._base_bar_count - 1) * float(self.base_tf_seconds)
            if span_sec > 0:
                return span_sec / 86400.0
        return 0.0

    # ------------------------------------------------------------ positions
    def _open_position(self, signal: TradeSignal, bar: dict) -> None:
        entry_price = float(bar["close"])
        atr_key = f"{self.base_tf}_atr"
        atr_val = _to_float(self.predictor.incremental_features.get(atr_key), None)
        if atr_val is None or atr_val <= 0:
            return

        stop_dist = float(self.stop_atr_multiple) * float(atr_val)
        if stop_dist <= 0:
            return

        stop_loss = entry_price - (signal.direction * stop_dist)
        take_profit = entry_price + (signal.direction * stop_dist * float(signal.expected_rr))

        if not self.paper:
            self._load_instrument_filters()
            stop_loss, take_profit = _round_stop_tp(
                entry_price,
                stop_loss,
                take_profit,
                self._tick_size,
                signal.direction,
            )
        if signal.direction == 1:
            if stop_loss >= entry_price or take_profit <= entry_price:
                self.logger.warning(
                    "ALERT: invalid stop/tp for long (entry=%.6f stop=%.6f tp=%.6f).",
                    entry_price,
                    stop_loss,
                    take_profit,
                )
                return
        else:
            if stop_loss <= entry_price or take_profit >= entry_price:
                self.logger.warning(
                    "ALERT: invalid stop/tp for short (entry=%.6f stop=%.6f tp=%.6f).",
                    entry_price,
                    stop_loss,
                    take_profit,
                )
                return

        if self.paper:
            risk_amount = self.paper_capital * DEFAULT_POSITION_SIZE_PCT
            size = risk_amount / stop_dist if stop_dist > 0 else 0.0
        else:
            if self._bybit is None:
                return
            balance = self._bybit.get_available_balance(asset=self.balance_asset, logger=self.logger)
            if balance <= 0:
                self.logger.warning("ALERT: Available balance is 0; cannot size position.")
                return
            risk_amount = balance * DEFAULT_POSITION_SIZE_PCT
            size = risk_amount / stop_dist if stop_dist > 0 else 0.0
            size = self._round_qty(size)
            if size <= 0:
                self.logger.warning("ALERT: Order size below minimum; skipping entry.")
                return
            notional = size * entry_price
            if notional < self._min_notional:
                self.logger.warning("ALERT: Order notional below exchange minimum; skipping entry.")
                return

        if not self.paper:
            if self._bybit.get_position(self.symbol):
                self.logger.warning("Exchange already has an open position; skipping entry.")
                return
            side = "Buy" if signal.direction == 1 else "Sell"
            order = self._bybit.open_position(
                self.symbol,
                side,
                size,
                stop_loss,
                take_profit,
                leverage=self.leverage,
            )
            if not order.success:
                self.logger.warning(f"ALERT: Order failed: {order.error_message}")
                return

        self.position = OpenPosition(
            entry_time=_utc_now(),
            entry_bar_time=int(bar["bar_time"]),
            direction=signal.direction,
            entry_price=entry_price,
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal_quality=signal.quality,
            expected_rr=signal.expected_rr,
            metadata={
                "bounce_prob": signal.bounce_prob,
                "ev": signal.ev_value,
                "trend_prob": signal.trend_prob,
                "regime_prob": signal.regime_prob,
            },
        )
        self._accepted_trades += 1
        self.logger.info(
            f"OPEN {('LONG' if signal.direction == 1 else 'SHORT')} "
            f"price={entry_price:.6f} stop={stop_loss:.6f} tp={take_profit:.6f} "
            f"quality={signal.quality} ev={signal.ev_value:.4f}"
        )
        diag = self._last_model_diag or {}
        self._record_event(
            {
                "event": "position_open",
                "bar_time": signal.bar_time,
                "direction": signal.direction,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "size": size,
                "expected_rr": signal.expected_rr,
                "signal_quality": signal.quality,
                "model_outputs": {
                    "bounce_prob": signal.bounce_prob,
                    "expected_rr": signal.expected_rr,
                    "expected_rr_mean": diag.get("expected_rr_mean"),
                    "ev": signal.ev_value,
                    "trend_prob": signal.trend_prob,
                    "regime_prob": signal.regime_prob,
                },
                "gates": diag.get("gates"),
                "reason": diag.get("reason", "accepted"),
                "model_inputs": diag.get("top_features"),
            }
        )

    def _request_close(
        self,
        *,
        exit_price: float,
        exit_reason: str,
        flip_signal: Optional[TradeSignal] = None,
        flip_bar: Optional[dict] = None,
    ) -> None:
        if self.position is None:
            return
        self._record_event(
            {
                "event": "close_request",
                "exit_reason": exit_reason,
                "exit_price": exit_price,
                "flip": bool(flip_signal is not None),
            }
        )
        if self.paper:
            self._close_position(exit_price=exit_price, exit_reason=exit_reason)
            if exit_reason == "flip" and flip_signal is not None and flip_bar is not None:
                self._open_position(flip_signal, flip_bar)
            return

        if self._bybit is None:
            return
        if self._close_requested:
            return

        pos = self._bybit.get_position(self.symbol)
        size = float(pos.size) if pos is not None else float(self.position.size)
        if size <= 0:
            return

        side = "Sell" if self.position.direction == 1 else "Buy"
        try:
            resp = self._bybit.session.place_order(
                category=self._bybit.category,
                symbol=self.symbol,
                side=side,
                orderType="Market",
                qty=size,
                timeInForce="IOC",
                reduceOnly=True,
            )
        except Exception as exc:
            self.logger.warning(f"ALERT: Close order failed: {exc}")
            return

        if resp.get("retCode", -1) != 0:
            self.logger.warning(f"ALERT: Close order rejected: {resp.get('retMsg')}")
            return

        self._close_requested = True
        self._close_request_reason = exit_reason
        if exit_reason == "flip" and flip_signal is not None and flip_bar is not None:
            self._pending_flip_signal = flip_signal
            self._pending_flip_bar = dict(flip_bar)

    def _check_paper_exit(self, price: float) -> None:
        if self.position is None:
            return
        pos = self.position
        if pos.direction == 1:
            if price <= pos.stop_loss:
                self._close_position(exit_price=pos.stop_loss, exit_reason="stop_loss")
            elif price >= pos.take_profit:
                self._close_position(exit_price=pos.take_profit, exit_reason="take_profit")
        else:
            if price >= pos.stop_loss:
                self._close_position(exit_price=pos.stop_loss, exit_reason="stop_loss")
            elif price <= pos.take_profit:
                self._close_position(exit_price=pos.take_profit, exit_reason="take_profit")

    def _close_position(self, *, exit_price: float, exit_reason: str) -> None:
        if self.position is None:
            return
        pos = self.position
        now = _utc_now()
        pnl = pos.direction * (exit_price - pos.entry_price) * pos.size
        pnl_percent = pos.direction * (exit_price - pos.entry_price) / pos.entry_price * 100 if pos.entry_price > 0 else 0.0
        duration = (now - pos.entry_time).total_seconds()

        trade = ClosedTrade(
            entry_time=pos.entry_time,
            exit_time=now,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size=pos.size,
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
            pnl=pnl,
            pnl_percent=pnl_percent,
            signal_quality=pos.signal_quality,
            exit_reason=exit_reason,
            duration_seconds=duration,
            metadata=pos.metadata,
        )
        self.closed_trades.append(trade)
        self._save_trade(trade)
        self._record_event(
            {
                "event": "position_close",
                "entry_time": pos.entry_time,
                "exit_time": now,
                "direction": pos.direction,
                "entry_price": pos.entry_price,
                "exit_price": exit_price,
                "size": pos.size,
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
                "pnl": pnl,
                "pnl_percent": pnl_percent,
                "exit_reason": exit_reason,
                "duration_seconds": duration,
                "metadata": pos.metadata,
            }
        )

        if self.paper:
            self.paper_capital += pnl

        self.logger.info(
            f"CLOSE {('LONG' if pos.direction == 1 else 'SHORT')} "
            f"exit={exit_price:.6f} pnl={pnl:+.2f} reason={exit_reason}"
        )
        self.position = None
        self._close_requested = False
        self._close_request_reason = None

    # ------------------------------------------------------- exchange polling
    def _sync_existing_exchange_position(self) -> None:
        if self.paper or self._bybit is None:
            return
        pos = self._bybit.get_position(self.symbol)
        if not pos:
            return
        direction = 1 if pos.side.lower() == "buy" else -1
        self.position = OpenPosition(
            entry_time=_utc_now(),
            entry_bar_time=int(self._span_start_bar_time or 0),
            direction=direction,
            entry_price=float(pos.entry_price),
            size=float(pos.size),
            stop_loss=float(pos.stop_loss) if pos.stop_loss is not None else float(pos.entry_price),
            take_profit=float(pos.take_profit) if pos.take_profit is not None else float(pos.entry_price),
            signal_quality="C",
            expected_rr=float(self.target_rr),
            metadata={"adopted_exchange_position": True},
        )
        self.logger.warning("Adopted existing exchange position on startup; new entries disabled until it closes.")

    def _poll_exchange_position(self) -> None:
        if self.paper or self._bybit is None:
            return
        now_ts = _utc_now().timestamp()
        if now_ts - self._last_position_poll < self._position_poll_interval:
            return
        self._last_position_poll = now_ts

        exchange_pos = self._bybit.get_position(self.symbol)
        if exchange_pos and self.position is None:
            self._sync_existing_exchange_position()
            return

        if not exchange_pos and self.position is not None:
            exit_price = None
            exit_reason = self._close_request_reason or "timeout"
            close_info = None
            try:
                resp = self._bybit.session.get_closed_pnl(category=self._bybit.category, symbol=self.symbol, limit=1)
                if resp.get("retCode", -1) == 0:
                    items = resp.get("result", {}).get("list", [])
                    close_info = items[0] if items else None
            except Exception as exc:
                self.logger.warning(f"Could not fetch closed PnL info: {exc}")

            if isinstance(close_info, dict):
                exit_price = _to_float(
                    close_info.get("avgExitPrice")
                    or close_info.get("exitPrice")
                    or close_info.get("markPrice")
                    or close_info.get("closePrice"),
                    None,
                )
            if exit_price is None:
                exit_price = self.position.entry_price
            if exit_reason is None or exit_reason == "timeout":
                exit_reason = _resolve_exit_reason(exit_price, self.position.stop_loss, self.position.take_profit, self.position.direction)
            self._close_position(exit_price=exit_price, exit_reason=exit_reason)

            if self._pending_flip_signal is not None and self._pending_flip_bar is not None:
                pending_signal = self._pending_flip_signal
                pending_bar = self._pending_flip_bar
                self._pending_flip_signal = None
                self._pending_flip_bar = None
                self._open_position(pending_signal, pending_bar)

    # -------------------------------------------------------------- persistence
    def _save_trade(self, trade: ClosedTrade) -> None:
        try:
            with open(self.trades_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(trade.to_dict()) + "\n")
        except Exception as exc:
            self.logger.warning(f"Could not save trade: {exc}")

    def _write_stats(self) -> None:
        stats = {
            "symbol": self.symbol,
            "paper": self.paper,
            "trades": len(self.closed_trades),
            "capital": self.paper_capital if self.paper else None,
            "parameters": {
                "stop_atr_multiple": self.stop_atr_multiple,
                "target_rr": self.target_rr,
                "pullback_threshold": self.pullback_threshold,
                "entry_forward_window": self.entry_forward_window,
                "use_ev_gate": self.use_ev_gate,
                "ev_margin_r": self.ev_margin_r,
                "fee_percent": self.fee_percent,
                "fee_per_trade_r": self.fee_per_trade_r,
                "use_expected_rr": self.use_expected_rr,
                "use_trend_gate": self.use_trend_gate,
                "use_regime_gate": self.use_regime_gate,
                "opposite_signal_policy": self.opposite_signal_policy,
            },
        }
        try:
            with open(self.stats_file, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
        except Exception as exc:
            self.logger.warning(f"Could not save stats: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live trading aligned to tuned backtest.")
    parser.add_argument("--train-config", type=str, required=True, help="Path to train_config_<trial>.json (required).")
    parser.add_argument("--model-dir", type=str, default=None, help="Optional override model directory.")
    parser.add_argument("--symbol", type=str, required=True, help="Trading symbol (e.g., MONUSDT).")
    parser.add_argument("--testnet", action="store_true", help="Use Bybit testnet.")
    parser.add_argument("--paper", action="store_true", help="Paper trading mode (no real orders).")
    parser.add_argument("--api-key", type=str, default=None, help="Bybit API key (discouraged; use keyring or secrets file).")
    parser.add_argument("--api-secret", type=str, default=None, help="Bybit API secret (discouraged; use keyring or secrets file).")
    parser.add_argument(
        "--recv-window-ms",
        type=int,
        default=DEFAULT_RECV_WINDOW_MS,
        help="Bybit recv_window (ms) for authenticated requests.",
    )
    parser.add_argument(
        "--secrets-path",
        type=str,
        default=None,
        help="Path to JSON secrets file with bybit_api_key/bybit_api_secret.",
    )
    parser.add_argument(
        "--keyring-service",
        type=str,
        default="sofia_bybit",
        help="Keyring service name for stored credentials (set empty to skip).",
    )
    parser.add_argument("--leverage", type=int, default=1, help="Leverage to set on Bybit.")
    parser.add_argument("--balance-asset", type=str, default="USDT", help="Balance asset for sizing.")
    parser.add_argument("--bootstrap-csv", type=str, required=True, help="CSV file or directory for bootstrap trades.")
    parser.add_argument("--log-dir", type=str, default="./live_results", help="Directory for trade logs.")
    parser.add_argument("--log-file", type=str, default=None, help="Optional log file path.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run preflight checks and exit before streaming trades.",
    )
    parser.add_argument(
        "--max-pending-trades",
        type=int,
        default=DEFAULT_MAX_PENDING_TRADES,
        help="Max pending trades to buffer before dropping oldest (prevents memory growth).",
    )

    args = parser.parse_args()
    keyring_service = args.keyring_service
    if keyring_service and keyring_service.strip().lower() in {"none", "off", "disable", "false", "0"}:
        keyring_service = None

    _clear_console()

    trader = LiveTrader(
        train_config_path=Path(args.train_config),
        model_dir_override=Path(args.model_dir) if args.model_dir else None,
        symbol=args.symbol,
        testnet=args.testnet,
        paper=args.paper,
        api_key=args.api_key,
        api_secret=args.api_secret,
        secrets_path=Path(args.secrets_path) if args.secrets_path else None,
        keyring_service=keyring_service,
        leverage=args.leverage,
        balance_asset=args.balance_asset,
        bootstrap_csv=Path(args.bootstrap_csv),
        log_dir=Path(args.log_dir),
        log_file=args.log_file,
        max_pending_trades=args.max_pending_trades,
        recv_window_ms=args.recv_window_ms,
    )

    if args.dry_run:
        trader.logger.info("Dry run complete; exiting before live streaming.")
        trader.stop()
        return

    trader.start()


if __name__ == "__main__":
    main()
