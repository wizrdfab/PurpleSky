"""
Tuning-aligned backtest module.

This backtest mirrors ConfigTuner entry logic (pullback-eligible rows,
EV gating, ops cost, and intrabar realized R) while reporting the same
metrics and artifacts as the original backtest module.
"""
from __future__ import annotations

import json
import sys
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import TrendFollowerConfig
from models import (
    CONTEXT_FEATURE_NAMES,
    EntryQualityModel,
    TrendFollowerModels,
    append_context_features,
    compute_expected_calibration_error,
)


@dataclass
class Trade:
    """Record of a single trade."""
    entry_time: datetime
    exit_time: datetime
    direction: int  # 1 for long, -1 for short
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_percent: float
    signal_quality: str
    exit_reason: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trend_prob: float = 0.0
    bounce_prob: float = 0.0
    is_pullback: bool = False
    trend_aligned: bool = False
    dist_from_ema: float = 0.0
    expected_rr: float = 0.0
    expected_rr_mean: float = 0.0
    ev_value: float = 0.0
    implied_threshold: float = 0.0
    fee_r: float = 0.0
    stop_dist: float = 0.0
    realized_r: float = 0.0
    realized_r_net: float = 0.0


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pnl_percent: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    sharpe_ratio: float = 0.0
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    trades_by_grade: Dict[str, int] = field(default_factory=dict)
    win_rate_by_grade: Dict[str, float] = field(default_factory=dict)
    signal_stats: Dict[str, int] = field(default_factory=dict)
    bounce_prob_stats: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def _max_drawdown(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    equity = np.cumsum(returns)
    peaks = np.maximum.accumulate(equity)
    drawdown = peaks - equity
    if drawdown.size == 0:
        return 0.0
    return float(np.max(drawdown))


def _compute_trade_metrics(returns: np.ndarray) -> Dict[str, float]:
    n_trades = int(returns.size)
    if n_trades == 0:
        return {
            "n_trades": 0.0,
            "total_pnl_r": 0.0,
            "avg_pnl_r": 0.0,
            "win_rate": 0.0,
            "avg_win_r": 0.0,
            "avg_loss_r": 0.0,
            "profit_factor": 0.0,
            "return_std": 0.0,
            "downside_std": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown_r": 0.0,
            "calmar": 0.0,
        }
    total_pnl = float(np.sum(returns))
    avg_pnl = float(np.mean(returns))
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    win_rate = float(wins.size / n_trades)
    avg_win = float(np.mean(wins)) if wins.size > 0 else 0.0
    avg_loss = float(np.mean(losses)) if losses.size > 0 else 0.0
    gross_win = float(np.sum(wins)) if wins.size > 0 else 0.0
    gross_loss = float(np.sum(losses)) if losses.size > 0 else 0.0
    gross_loss_abs = abs(gross_loss)
    if gross_loss_abs > 0:
        profit_factor = float(gross_win / gross_loss_abs)
    else:
        profit_factor = float("inf") if gross_win > 0 else 0.0
    return_std = float(np.std(returns)) if n_trades > 1 else 0.0
    downside_std = float(np.std(losses)) if losses.size > 1 else 0.0
    sharpe = float(avg_pnl / return_std) if return_std > 0 else 0.0
    sortino = float(avg_pnl / downside_std) if downside_std > 0 else 0.0
    max_dd = _max_drawdown(returns)
    calmar = float(total_pnl / max_dd) if max_dd > 0 else total_pnl
    return {
        "n_trades": float(n_trades),
        "total_pnl_r": total_pnl,
        "avg_pnl_r": avg_pnl,
        "win_rate": win_rate,
        "avg_win_r": avg_win,
        "avg_loss_r": avg_loss,
        "profit_factor": profit_factor,
        "return_std": return_std,
        "downside_std": downside_std,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown_r": max_dd,
        "calmar": calmar,
    }


def _median_atr_percent(data: pd.DataFrame, base_tf: str) -> float:
    atr_col = f"{base_tf}_atr"
    if atr_col in data.columns and "close" in data.columns:
        atr_vals = pd.to_numeric(data[atr_col], errors="coerce").to_numpy(dtype=float)
        close_vals = pd.to_numeric(data["close"], errors="coerce").to_numpy(dtype=float)
        mask = (atr_vals > 0) & (close_vals > 0)
        if mask.any():
            median_atr_pct = float(np.nanmedian(atr_vals[mask] / close_vals[mask]))
            if np.isfinite(median_atr_pct) and median_atr_pct > 0:
                return median_atr_pct
    return 0.005


def _estimate_span_days(data: pd.DataFrame, base_tf_seconds: Optional[float]) -> float:
    if data.empty:
        return 0.0
    if "bar_time" in data.columns:
        times = pd.to_numeric(data["bar_time"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(times)
        if mask.any():
            span_sec = float(np.nanmax(times[mask]) - np.nanmin(times[mask]))
            if span_sec > 0:
                return span_sec / 86400.0
    if "datetime" in data.columns:
        dt = pd.to_datetime(data["datetime"], errors="coerce")
        if dt.notna().any():
            span_sec = (dt.max() - dt.min()).total_seconds()
            if span_sec > 0:
                return span_sec / 86400.0
    if base_tf_seconds and base_tf_seconds > 0:
        span_sec = float(max(1, len(data) - 1)) * float(base_tf_seconds)
        return span_sec / 86400.0
    return 0.0


def _get_bar_times(data: pd.DataFrame) -> Optional[np.ndarray]:
    if "bar_time" in data.columns:
        times = pd.to_numeric(data["bar_time"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(times)
        if mask.any():
            return np.where(mask, times.astype(np.int64), -1)
        return None
    if "datetime" in data.columns:
        dt = pd.to_datetime(data["datetime"], errors="coerce")
        if dt.notna().any():
            raw = (dt.view("int64") // 1_000_000_000).astype(np.int64)
            mask = dt.notna().to_numpy()
            return np.where(mask, raw, -1)
    return None


def _compute_fee_r_series(
    data: pd.DataFrame,
    base_tf: str,
    fee_percent: float,
    stop_atr_multiple: float,
    fallback_fee_r: float,
) -> Tuple[np.ndarray, np.ndarray]:
    fee_r = np.full(len(data), fallback_fee_r, dtype=float)
    fallback_mask = np.ones(len(data), dtype=bool)
    atr_col = f"{base_tf}_atr"
    if atr_col not in data.columns or "close" not in data.columns:
        return fee_r, fallback_mask
    atr_vals = pd.to_numeric(data[atr_col], errors="coerce").to_numpy(dtype=float)
    close_vals = pd.to_numeric(data["close"], errors="coerce").to_numpy(dtype=float)
    denom = stop_atr_multiple * atr_vals
    mask = (denom > 0) & (close_vals > 0) & np.isfinite(denom) & np.isfinite(close_vals)
    if mask.any():
        fee_r[mask] = (fee_percent * close_vals[mask]) / denom[mask]
        fallback_mask[mask] = False
    return fee_r, fallback_mask


def _confirm_continue(message: str) -> bool:
    try:
        if not sys.stdin.isatty():
            print(message)
            print("Proceeding without prompt (non-interactive).")
            return True
    except Exception:
        print(message)
        return True
    resp = input(f"{message} Continue? [y/N]: ").strip().lower()
    return resp in ("y", "yes")


def _build_feature_frame(data: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    if not feature_names:
        return pd.DataFrame(index=data.index)
    X = data.reindex(columns=feature_names)
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0.0
    return X.fillna(0)


def _apply_entry_readiness_mask(
    X_entry: pd.DataFrame,
    readiness: Dict[str, Any],
    entry_features: Optional[List[str]] = None,
) -> pd.DataFrame:
    if not isinstance(readiness, dict):
        raise SystemExit("entry_feature_readiness is missing or invalid.")
    ready_features = readiness.get("ready_features")
    if not isinstance(ready_features, list) or not ready_features:
        raise SystemExit("entry_feature_readiness missing ready_features.")

    ready_set = set(ready_features)
    context_cols = set(CONTEXT_FEATURE_NAMES)
    missing_ready = [col for col in ready_set if col not in X_entry.columns]
    if missing_ready:
        missing_str = ", ".join(sorted(missing_ready))
        raise SystemExit(
            "entry_feature_readiness mismatch: missing required features: "
            f"{missing_str}"
        )

    if entry_features:
        entry_set = set(entry_features)
        unexpected = [col for col in ready_set if col not in entry_set and col not in context_cols]
        if unexpected:
            bad = ", ".join(sorted(unexpected))
            raise SystemExit(
                "entry_feature_readiness contains unexpected features: "
                f"{bad}"
            )

    for col in X_entry.columns:
        if col in context_cols:
            continue
        if col not in ready_set:
            X_entry[col] = 0.0
    return X_entry


def _feature_audit(feature_cols: List[str], entry_model: EntryQualityModel) -> Dict[str, Any]:
    expected = getattr(entry_model, "filtered_feature_names", None) or getattr(entry_model, "feature_names", None)
    if not expected:
        return {
            "expected_count": 0,
            "data_count": len(feature_cols),
            "missing": [],
            "extra": [],
        }
    context_cols = set(CONTEXT_FEATURE_NAMES)
    context_expected = [col for col in expected if col in context_cols]
    missing = [col for col in expected if col not in feature_cols and col not in context_cols]
    extra = [col for col in feature_cols if col not in expected and col not in context_cols]
    return {
        "expected_count": len(expected),
        "data_count": len(feature_cols),
        "context_expected": context_expected,
        "missing": missing,
        "extra": extra,
    }


def _slice_pred(pred: Optional[Dict[str, np.ndarray]], mask: pd.Series) -> Optional[Dict[str, np.ndarray]]:
    if pred is None:
        return None
    mask_arr = np.asarray(mask, dtype=bool)
    sliced: Dict[str, np.ndarray] = {}
    for key, values in pred.items():
        arr = np.asarray(values)
        if arr.shape[0] == mask_arr.shape[0]:
            sliced[key] = arr[mask_arr]
    return sliced if sliced else None


def _grade_signal(bounce_prob: float, trend_aligned: bool, is_pullback: bool) -> str:
    if bounce_prob > 0.6 and trend_aligned and is_pullback:
        return "A"
    if bounce_prob > 0.5 and (trend_aligned or is_pullback):
        return "B"
    return "C"


def _resolve_trade_side(trade_side: str, trend_dir: np.ndarray) -> np.ndarray:
    side = (trade_side or "both").strip().lower()
    if side == "long":
        return trend_dir == 1
    if side == "short":
        return trend_dir == -1
    return trend_dir != 0


def _prob_stats(values: np.ndarray, threshold: float) -> Dict[str, float]:
    if values.size == 0:
        return {
            "count": 0.0,
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "max": 0.0,
            "pct_above_threshold": 0.0,
        }
    return {
        "count": float(values.size),
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
        "pct_above_threshold": float(np.mean(values >= threshold)),
    }


def _compute_ev_bin_summary(ev_vals: np.ndarray, returns: np.ndarray) -> str:
    if ev_vals.size == 0 or returns.size == 0:
        return ""
    n = min(ev_vals.size, returns.size)
    ev_vals = ev_vals[:n]
    ret_vals = returns[:n]
    if n <= 0:
        return ""
    n_bins = int(min(10, max(1, n)))
    if n_bins == 1:
        return f"0:{ret_vals.mean():.4f}@{n}"
    edges = np.quantile(ev_vals, np.linspace(0.0, 1.0, n_bins + 1))
    if np.unique(edges).size == 1:
        return f"0:{ret_vals.mean():.4f}@{n}"
    bins = np.digitize(ev_vals, edges[1:-1], right=True)
    parts = []
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            parts.append(f"{b}:{ret_vals[mask].mean():.4f}@{int(mask.sum())}")
    return "|".join(parts)


def _compute_ev_diagnostics(
    labels: np.ndarray,
    probs: np.ndarray,
    ev_vals: np.ndarray,
    gap_vals: np.ndarray,
    rr_mean: np.ndarray,
    rr_cons: np.ndarray,
    realized_r: np.ndarray,
    realized_r_net: np.ndarray,
    fee_r: np.ndarray,
    win_r: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    diag: Dict[str, Any] = {
        "trade_count": int(labels.size),
        "brier": 0.0,
        "logloss": 0.0,
        "ece": 0.0,
        "ev_mean": 0.0,
        "ev_p10": 0.0,
        "threshold_gap_mean": 0.0,
        "threshold_gap_p10": 0.0,
        "expected_rr_mean": 0.0,
        "expected_rr_cons_mean": 0.0,
        "realized_r_mean": 0.0,
        "realized_r_net_mean": 0.0,
        "fee_r_mean": 0.0,
        "expected_rr_bias_ratio": 0.0,
        "expected_rr_mae": 0.0,
        "ev_bin_summary": "",
    }
    if labels.size == 0:
        return diag
    labels_f = labels.astype(float)
    probs_clip = np.clip(probs.astype(float), 1e-6, 1.0 - 1e-6)
    diag["brier"] = float(np.mean((probs_clip - labels_f) ** 2))
    diag["logloss"] = float(-np.mean(labels_f * np.log(probs_clip) + (1.0 - labels_f) * np.log(1.0 - probs_clip)))
    try:
        ece = compute_expected_calibration_error(labels_f.astype(int), probs_clip, n_bins=10)
        diag["ece"] = float(ece.get("ece", 0.0))
    except Exception:
        diag["ece"] = 0.0
    if ev_vals.size > 0:
        diag["ev_mean"] = float(np.mean(ev_vals))
        diag["ev_p10"] = float(np.percentile(ev_vals, 10))
    if gap_vals.size > 0:
        diag["threshold_gap_mean"] = float(np.mean(gap_vals))
        diag["threshold_gap_p10"] = float(np.percentile(gap_vals, 10))
    if rr_mean.size > 0:
        diag["expected_rr_mean"] = float(np.mean(rr_mean))
    if rr_cons.size > 0:
        diag["expected_rr_cons_mean"] = float(np.mean(rr_cons))
    if realized_r.size > 0:
        diag["realized_r_mean"] = float(np.mean(realized_r))
    if realized_r_net.size > 0:
        diag["realized_r_net_mean"] = float(np.mean(realized_r_net))
    if fee_r.size > 0:
        diag["fee_r_mean"] = float(np.mean(fee_r))

    rr_ratio = 0.0
    rr_mae = 0.0
    if win_r is not None and win_r.size == rr_mean.size and labels.size == rr_mean.size:
        win_mask = (labels == 1) & np.isfinite(win_r)
        if win_mask.any():
            realized_win = win_r[win_mask]
            pred_win = rr_mean[win_mask]
            if np.mean(realized_win) > 0:
                rr_ratio = float(np.mean(pred_win) / np.mean(realized_win))
            rr_mae = float(np.mean(np.abs(pred_win - realized_win)))
    diag["expected_rr_bias_ratio"] = float(rr_ratio)
    diag["expected_rr_mae"] = float(rr_mae)
    diag["ev_bin_summary"] = _compute_ev_bin_summary(ev_vals, realized_r_net)
    return diag


def _resolve_exit_reason(exit_type: Optional[str]) -> str:
    if exit_type is None or not isinstance(exit_type, str):
        return "timeout"
    et = exit_type.strip().lower()
    if et == "tp":
        return "take_profit"
    if et == "sl":
        return "stop_loss"
    if et in {"timeout", "time", "expire"}:
        return "timeout"
    return et


def _extract_time(value: Any) -> datetime:
    if value is None:
        return datetime.utcnow()
    if isinstance(value, datetime):
        return value
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 1e11:
            ts /= 1000.0
        try:
            return datetime.utcfromtimestamp(ts)
        except Exception:
            return datetime.utcnow()
    return datetime.utcnow()


def _simulate_single_position_trades(
    data: pd.DataFrame,
    pullback_mask: pd.Series,
    trade_mask: np.ndarray,
    trend_dir: np.ndarray,
    base_tf: str,
    stop_atr_multiple: float,
    entry_forward_window: int,
    realized_r: Optional[np.ndarray],
    fallback_outcomes: np.ndarray,
    opposite_policy: str,
    bars_to_exit: np.ndarray,
    bar_times: Optional[np.ndarray] = None,
    trade_index: Optional[Any] = None,
    use_intrabar: bool = False,
    expected_rr: Optional[np.ndarray] = None,
    use_expected_rr: bool = False,
    target_rr: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    if trade_mask.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float), []
    atr_col = f"{base_tf}_atr" if base_tf else "atr"
    if "close" not in data.columns or atr_col not in data.columns:
        idx = np.where(trade_mask)[0]
        returns = fallback_outcomes[trade_mask]
        exit_pos = np.full(idx.shape, -1, dtype=int)
        reasons = ["timeout"] * int(idx.size)
        return idx, exit_pos, returns, reasons

    pullback_mask_arr = pullback_mask.to_numpy(dtype=bool)
    pullback_pos = np.nonzero(pullback_mask_arr)[0]
    close_vals = pd.to_numeric(data["close"], errors="coerce").to_numpy(dtype=float)
    atr_vals = pd.to_numeric(data[atr_col], errors="coerce").to_numpy(dtype=float)
    if realized_r is None or realized_r.size == 0:
        realized_r = fallback_outcomes
    else:
        realized_r = np.where(np.isfinite(realized_r), realized_r, fallback_outcomes)

    intrabar_enabled = bool(use_intrabar and trade_index is not None and bar_times is not None)
    bar_times_arr = None
    if intrabar_enabled:
        bar_times_arr = np.asarray(bar_times, dtype=np.int64)
        if bar_times_arr.shape[0] != close_vals.shape[0]:
            intrabar_enabled = False
            bar_times_arr = None

    trade_indices: List[int] = []
    exit_positions: List[int] = []
    returns: List[float] = []
    exit_reasons: List[str] = []
    n = trade_mask.size
    i = 0

    while i < n:
        if not trade_mask[i]:
            i += 1
            continue
        direction = int(trend_dir[i])
        if direction == 0:
            i += 1
            continue
        entry_pos = int(pullback_pos[i])
        if entry_pos < 0 or entry_pos >= close_vals.size:
            i += 1
            continue
        current_atr = atr_vals[entry_pos]
        if not np.isfinite(current_atr) or current_atr <= 0:
            i += 1
            continue
        stop_dist = stop_atr_multiple * current_atr
        if not np.isfinite(stop_dist) or stop_dist <= 0:
            i += 1
            continue
        entry_price = close_vals[entry_pos]
        bars_exit = int(max(1, bars_to_exit[i]))
        natural_exit_pos = min(entry_pos + bars_exit, close_vals.size - 1)

        effective_rr = float(target_rr)
        if use_expected_rr and expected_rr is not None:
            if expected_rr.size > i and expected_rr[i] > 0:
                effective_rr = float(expected_rr[i])
        stop_price = entry_price - (direction * stop_dist)
        target_price = entry_price + (direction * stop_dist * effective_rr)

        next_flip_idx = None
        next_flip_pos = None
        flipped = False
        if opposite_policy in {"flip", "close"}:
            j = i + 1
            while j < n and pullback_pos[j] <= natural_exit_pos:
                if trade_mask[j] and int(trend_dir[j]) == -direction:
                    next_flip_idx = j
                    next_flip_pos = int(pullback_pos[j])
                    break
                j += 1

        if intrabar_enabled and bar_times_arr is not None:
            check_end = natural_exit_pos
            if next_flip_pos is not None:
                check_end = min(check_end, next_flip_pos)
            exit_pos = None
            exit_reason = None
            realized = None
            if check_end > entry_pos:
                for bar_pos in range(entry_pos + 1, check_end + 1):
                    bar_time = int(bar_times_arr[bar_pos])
                    if bar_time < 0:
                        continue
                    exit_code, exit_price = trade_index.check_exit(
                        bar_time,
                        int(direction),
                        float(stop_price),
                        float(target_price),
                    )
                    if exit_code == 1 or exit_code == 2:
                        exit_pos = bar_pos
                        exit_reason = "stop_loss" if exit_code == 1 else "take_profit"
                        if direction == 1:
                            realized = (exit_price - entry_price) / stop_dist
                        else:
                            realized = (entry_price - exit_price) / stop_dist
                        break
            if exit_pos is not None:
                returns.append(float(realized))
                trade_indices.append(i)
                exit_positions.append(int(exit_pos))
                exit_reasons.append(str(exit_reason))
                k = i + 1
                while k < n and pullback_pos[k] <= exit_pos:
                    k += 1
                i = k
                continue

        if next_flip_pos is not None and next_flip_idx is not None:
            exit_pos = int(next_flip_pos)
            exit_price = close_vals[exit_pos]
            if direction == 1:
                realized = (exit_price - entry_price) / stop_dist
            else:
                realized = (entry_price - exit_price) / stop_dist
            returns.append(float(realized))
            trade_indices.append(i)
            exit_positions.append(exit_pos)
            exit_reasons.append("flip" if opposite_policy == "flip" else "close")
            if opposite_policy == "flip":
                i = next_flip_idx
            else:
                i = next_flip_idx + 1
            flipped = True

        if flipped:
            continue

        if intrabar_enabled:
            exit_pos = int(natural_exit_pos)
            exit_price = close_vals[exit_pos]
            if direction == 1:
                realized = (exit_price - entry_price) / stop_dist
            else:
                realized = (entry_price - exit_price) / stop_dist
        else:
            realized = float(realized_r[i])
            exit_pos = int(natural_exit_pos)

        returns.append(float(realized))
        trade_indices.append(i)
        exit_positions.append(int(exit_pos))
        exit_reasons.append("timeout")
        k = i + 1
        while k < n and pullback_pos[k] <= exit_pos:
            k += 1
        i = k

    return (
        np.asarray(trade_indices, dtype=int),
        np.asarray(exit_positions, dtype=int),
        np.asarray(returns, dtype=float),
        exit_reasons,
    )


def _compute_equity_curve(
    initial_capital: float,
    pnl_values: List[float],
) -> Tuple[List[float], float, float]:
    equity_curve = []
    equity = float(initial_capital)
    peak = equity
    max_dd = 0.0
    for pnl in pnl_values:
        equity += pnl
        equity_curve.append(equity)
        if equity > peak:
            peak = equity
        drawdown = peak - equity
        if drawdown > max_dd:
            max_dd = drawdown
    max_dd_pct = (max_dd / initial_capital) * 100.0 if initial_capital > 0 else 0.0
    return equity_curve, max_dd, max_dd_pct


def run_tuned_backtest(
    data: pd.DataFrame,
    feature_cols: List[str],
    models: TrendFollowerModels,
    config: TrendFollowerConfig,
    *,
    use_full_data: bool = False,
    trade_side: str = "both",
    use_ev_gate: bool = True,
    ev_margin_r: float = 0.0,
    min_bounce_prob: float = 0.5,
    max_bounce_prob: float = 1.0,
    use_raw_probabilities: bool = False,
    use_calibration: Optional[bool] = None,
    use_expected_rr: bool = True,
    fee_percent: float = 0.0011,
    fee_per_trade_r: Optional[float] = None,
    ops_cost_enabled: bool = True,
    ops_cost_target_trades_per_day: float = 30.0,
    ops_cost_c1: float = 0.01,
    ops_cost_alpha: float = 1.7,
    single_position: bool = True,
    opposite_signal_policy: str = "flip",
    max_holding_bars: Optional[int] = None,
    initial_capital: float = 10000.0,
    position_size_pct: float = 0.02,
    ema_touch_mode: str = "multi",
    use_intrabar: bool = True,
    confirm_missing_models: bool = True,
    entry_feature_readiness: Optional[Dict[str, Any]] = None,
) -> BacktestResult:
    result = BacktestResult()
    if data is None or len(data) == 0:
        return result
    if entry_feature_readiness is None:
        raise SystemExit(
            "entry_feature_readiness missing; train_config must include readiness snapshot."
        )
    base_tf = config.features.timeframe_names[config.base_timeframe_idx]
    tf_seconds_map = dict(zip(config.features.timeframe_names, config.features.timeframes))
    base_tf_seconds = tf_seconds_map.get(base_tf, None)

    if not use_full_data:
        split_idx = int(len(data) * (float(config.model.train_ratio) + float(config.model.val_ratio)))
        split_idx = max(0, min(len(data), split_idx))
        test_df = data.iloc[split_idx:]
    else:
        split_idx = 0
        test_df = data

    if test_df.empty:
        return result

    pullback_mask = ~test_df["pullback_success"].isna()
    pullback_count = int(pullback_mask.sum())
    if pullback_count == 0:
        result.signal_stats = {
            "signals_checked": 0,
            "total_bars": int(len(test_df)),
            "accepted_signals": 0,
        }
        return result

    feature_audit = _feature_audit(feature_cols, models.entry_model)
    if feature_audit.get("missing") or feature_audit.get("extra"):
        missing_list = feature_audit.get("missing") or []
        extra_list = feature_audit.get("extra") or []
        if missing_list:
            print("Missing features:")
            for name in missing_list:
                print(f"  - {name}")
        if extra_list:
            print("Extra features:")
            for name in extra_list:
                print(f"  - {name}")
        missing = ", ".join(missing_list)
        extra = ", ".join(extra_list)
        raise ValueError(
            "Feature mismatch between backtest data and entry model. "
            f"Missing: {len(missing_list)} [{missing}]. "
            f"Extra: {len(extra_list)} [{extra}]. "
            "Rebuild features with the same config used for training."
        )

    X_base = test_df.reindex(columns=feature_cols).fillna(0)
    trend_pred = None
    trend_model = getattr(models, "trend_classifier", None)
    regime_model = getattr(models, "regime_classifier", None)
    missing_models: List[str] = []
    if trend_model is None or getattr(trend_model, "model", None) is None:
        missing_models.append("trend_classifier")
    if regime_model is None or getattr(regime_model, "model", None) is None:
        missing_models.append("regime_classifier")
    if missing_models:
        warning = (
            "WARNING: Missing models for context features: "
            + ", ".join(missing_models)
            + ". Context probabilities will be zeros, which can drift results."
        )
        if confirm_missing_models:
            if not _confirm_continue(warning):
                print("Backtest aborted by user.")
                return result
        else:
            print(warning)

    if trend_model is not None and getattr(trend_model, "model", None) is not None:
        trend_features = getattr(trend_model, "feature_names", None) or feature_cols
        X_trend = _build_feature_frame(test_df, list(trend_features))
        trend_pred = trend_model.predict(X_trend)

    regime_pred = None
    if regime_model is not None and getattr(regime_model, "model", None) is not None:
        regime_features = getattr(regime_model, "feature_names", None) or feature_cols
        X_regime = _build_feature_frame(test_df, list(regime_features))
        regime_pred = regime_model.predict(X_regime)

    X_entry = append_context_features(X_base, trend_pred, regime_pred)
    entry_features = getattr(models.entry_model, "filtered_feature_names", None) or getattr(models.entry_model, "feature_names", None)
    if entry_features:
        X_entry = _build_feature_frame(X_entry, list(entry_features))
    X_entry = _apply_entry_readiness_mask(
        X_entry,
        entry_feature_readiness,
        entry_features=list(entry_features) if entry_features else None,
    )
    X_entry = X_entry.loc[pullback_mask]

    final_use_calibration = bool(config.labels.use_calibration)
    if use_calibration is not None:
        final_use_calibration = bool(use_calibration)
    if use_raw_probabilities:
        final_use_calibration = False

    preds = models.entry_model.predict(X_entry, use_calibration=final_use_calibration)
    prob_key = "bounce_prob_raw" if use_raw_probabilities else "bounce_prob"
    probs = np.asarray(preds.get(prob_key, preds.get("bounce_prob", [])), dtype=float)
    raw_probs = np.asarray(preds.get("bounce_prob_raw", probs), dtype=float)
    cal_probs = np.asarray(preds.get("bounce_prob", raw_probs), dtype=float)
    labels = test_df.loc[pullback_mask, "pullback_success"].astype(int).values
    if probs.size == 0:
        return result

    touch_dir = np.zeros(pullback_count, dtype=int)
    if "ema_touch_direction" in test_df.columns:
        touch_dir = (
            test_df.loc[pullback_mask, "ema_touch_direction"]
            .fillna(0)
            .astype(int)
            .values
        )
    slope_col = f"{base_tf}_ema_{config.labels.pullback_ema}_slope_norm"
    if slope_col in test_df.columns:
        slope_vals = test_df.loc[pullback_mask, slope_col].fillna(0).values.astype(float)
        slope_dir = np.sign(slope_vals).astype(int)
    else:
        slope_dir = np.zeros_like(touch_dir, dtype=int)
    trend_dir = np.where(touch_dir != 0, touch_dir, slope_dir).astype(int)
    direction_mask = trend_dir != 0
    side_mask = _resolve_trade_side(trade_side, trend_dir)
    direction_mask = direction_mask & side_mask

    trend_gate_enabled = bool(getattr(config.labels, "use_trend_gate", False))
    min_trend_prob = float(getattr(config.labels, "min_trend_prob", 0.0))
    regime_gate_enabled = bool(getattr(config.labels, "use_regime_gate", False))
    min_regime_prob = float(getattr(config.labels, "min_regime_prob", 0.0))
    regime_align_direction = bool(getattr(config.labels, "regime_align_direction", True))
    allow_regime_ranging = bool(getattr(config.labels, "allow_regime_ranging", True))
    allow_regime_trend_up = bool(getattr(config.labels, "allow_regime_trend_up", True))
    allow_regime_trend_down = bool(getattr(config.labels, "allow_regime_trend_down", True))
    allow_regime_volatile = bool(getattr(config.labels, "allow_regime_volatile", True))

    trend_prob_dir = np.zeros_like(trend_dir, dtype=float)
    if trend_pred is not None:
        sliced = _slice_pred(trend_pred, pullback_mask)
        if sliced is not None:
            prob_up = np.asarray(sliced.get("prob_up", np.zeros_like(trend_dir, dtype=float)))
            prob_down = np.asarray(sliced.get("prob_down", np.zeros_like(trend_dir, dtype=float)))
            prob_neutral = np.asarray(sliced.get("prob_neutral", np.zeros_like(trend_dir, dtype=float)))
            trend_prob_dir = np.where(trend_dir == 1, prob_up, np.where(trend_dir == -1, prob_down, prob_neutral))

    if trend_gate_enabled and trend_pred is not None:
        trend_gate_mask = (trend_prob_dir >= min_trend_prob) & direction_mask
    else:
        trend_gate_mask = direction_mask

    regime_prob_dir = np.zeros_like(trend_dir, dtype=float)
    regime_gate_mask = np.ones_like(direction_mask, dtype=bool)
    regime_id_counts: Dict[int, int] = {}
    regimes = None
    if regime_pred is not None:
        sliced = _slice_pred(regime_pred, pullback_mask)
        if sliced is not None:
            regimes = np.asarray(sliced.get("regime", np.zeros_like(trend_dir, dtype=int)))
            prob_ranging = np.asarray(sliced.get("prob_ranging", np.zeros_like(trend_dir, dtype=float)))
            prob_trend_up = np.asarray(sliced.get("prob_trend_up", np.zeros_like(trend_dir, dtype=float)))
            prob_trend_down = np.asarray(sliced.get("prob_trend_down", np.zeros_like(trend_dir, dtype=float)))
            prob_volatile = np.asarray(sliced.get("prob_volatile", np.zeros_like(trend_dir, dtype=float)))
            regime_prob_dir = np.where(
                regimes == 0,
                prob_ranging,
                np.where(regimes == 1, prob_trend_up, np.where(regimes == 2, prob_trend_down, prob_volatile)),
            )
            for regime_id in regimes.tolist():
                regime_id_counts[regime_id] = regime_id_counts.get(regime_id, 0) + 1

    if regime_gate_enabled and regime_pred is not None and regimes is not None:
        allowed_mask = np.zeros_like(direction_mask, dtype=bool)
        allowed_mask = np.where(regimes == 0, allow_regime_ranging, allowed_mask)
        allowed_mask = np.where(regimes == 1, allow_regime_trend_up, allowed_mask)
        allowed_mask = np.where(regimes == 2, allow_regime_trend_down, allowed_mask)
        allowed_mask = np.where(regimes == 3, allow_regime_volatile, allowed_mask)
        if regime_align_direction:
            align_mask = np.ones_like(direction_mask, dtype=bool)
            align_mask = np.where(regimes == 1, trend_dir == 1, align_mask)
            align_mask = np.where(regimes == 2, trend_dir == -1, align_mask)
            allowed_mask = allowed_mask & align_mask
        regime_gate_mask = (regime_prob_dir >= min_regime_prob) & allowed_mask & direction_mask

    gate_mask = trend_gate_mask & regime_gate_mask & direction_mask
    gate_rejected_trend = int(np.sum(direction_mask & ~trend_gate_mask))
    gate_rejected_regime = int(np.sum(direction_mask & ~regime_gate_mask))
    gate_rejected_direction = int(np.sum(~direction_mask))

    target_rr = float(config.labels.target_rr)
    outcomes_r = np.where(labels == 1, target_rr, -1.0).astype(float)
    win_r_used = None
    if use_expected_rr and "pullback_win_r" in test_df.columns:
        win_r = pd.to_numeric(
            test_df.loc[pullback_mask, "pullback_win_r"],
            errors="coerce",
        ).to_numpy(dtype=float)
        if win_r.size == 0:
            win_r = np.full_like(labels, target_rr, dtype=float)
        win_r = np.where(np.isfinite(win_r), win_r, target_rr)
        win_r_used = np.where(labels == 1, win_r, np.nan)
        outcomes_r = np.where(labels == 1, win_r, -1.0).astype(float)

    realized_r = None
    if "pullback_realized_r" in test_df.columns:
        realized_r = pd.to_numeric(
            test_df.loc[pullback_mask, "pullback_realized_r"],
            errors="coerce",
        ).to_numpy(dtype=float)

    rr_mean = np.full_like(probs, target_rr, dtype=float)
    rr_cons = rr_mean
    if use_expected_rr:
        rr_mean = np.asarray(preds.get("expected_rr_mean", rr_mean), dtype=float)
        rr_cons = np.asarray(preds.get("expected_rr", rr_mean), dtype=float)

    trade_index = None
    bar_times = None
    intrabar_enabled = False
    if use_intrabar:
        bar_times = _get_bar_times(test_df)
        if bar_times is not None:
            try:
                import rust_pipeline_bridge as rust_bridge  # type: ignore
                if rust_bridge.is_available():
                    trade_index = rust_bridge.build_trade_index(config)
                    intrabar_enabled = True
            except Exception:
                trade_index = None
                intrabar_enabled = False
        if not intrabar_enabled:
            print("WARNING: Intrabar checks unavailable; using labeled outcomes for exits.")

    median_atr_percent = _median_atr_percent(test_df, base_tf)
    fallback_fee_r = float(fee_percent) / (float(config.labels.stop_atr_multiple) * median_atr_percent)
    dynamic_fee = True
    if fee_per_trade_r is not None and np.isfinite(float(fee_per_trade_r)):
        fallback_fee_r = float(fee_per_trade_r)
        dynamic_fee = False

    if dynamic_fee:
        fee_r_all, fee_fallback_all = _compute_fee_r_series(
            test_df,
            base_tf,
            float(fee_percent),
            float(config.labels.stop_atr_multiple),
            fallback_fee_r,
        )
        fee_r_entry = fee_r_all[pullback_mask.to_numpy()]
        fee_fallback_entry = fee_fallback_all[pullback_mask.to_numpy()]
    else:
        fee_r_entry = np.full_like(probs, fallback_fee_r, dtype=float)
        fee_fallback_entry = np.zeros_like(fee_r_entry, dtype=bool)

    ev_components_base = models.entry_model.compute_expected_rr_components(
        probs,
        rr_mean,
        rr_conservative=rr_cons,
        cost_r=fee_r_entry,
    )
    ev_base = ev_components_base["ev_conservative_r"]
    implied_base = np.clip(ev_components_base["implied_threshold"], 0.0, 1.0)
    threshold_gap_base = probs - implied_base

    span_days = _estimate_span_days(test_df, base_tf_seconds)
    if span_days <= 0:
        span_days = 0.0

    ops_cost_r = 0.0
    trade_mask = gate_mask.copy()
    if use_ev_gate:
        trade_mask = gate_mask & (ev_base > float(ev_margin_r))
    else:
        trade_mask = gate_mask & (probs >= float(min_bounce_prob))
        if float(max_bounce_prob) < 1.0:
            trade_mask = trade_mask & (probs <= float(max_bounce_prob))

    trade_count = int(trade_mask.sum())
    coverage = float(trade_count / probs.size) if probs.size > 0 else 0.0
    trade_rate_day = float(trade_count / span_days) if span_days > 0 else 0.0
    if (
        ops_cost_enabled
        and trade_rate_day > float(ops_cost_target_trades_per_day)
        and float(ops_cost_target_trades_per_day) > 0
    ):
        excess = trade_rate_day - float(ops_cost_target_trades_per_day)
        ops_cost_r = float(ops_cost_c1) * (
            (excess / float(ops_cost_target_trades_per_day)) ** float(ops_cost_alpha)
        )

    ev = ev_base
    implied = implied_base
    threshold_gap = threshold_gap_base
    if ops_cost_r > 0.0:
        ev_components_pass2 = models.entry_model.compute_expected_rr_components(
            probs,
            rr_mean,
            rr_conservative=rr_cons,
            cost_r=fee_r_entry + ops_cost_r,
        )
        ev = ev_components_pass2["ev_conservative_r"]
        implied = np.clip(ev_components_pass2["implied_threshold"], 0.0, 1.0)
        threshold_gap = probs - implied
        if use_ev_gate:
            trade_mask = gate_mask & (ev > float(ev_margin_r))
        else:
            trade_mask = gate_mask & (probs >= float(min_bounce_prob))
            if float(max_bounce_prob) < 1.0:
                trade_mask = trade_mask & (probs <= float(max_bounce_prob))
        trade_count = int(trade_mask.sum())
        coverage = float(trade_count / probs.size) if probs.size > 0 else 0.0

    rejected_ev_gate = int(np.sum(gate_mask & ~trade_mask)) if use_ev_gate else 0
    rejected_bounce_prob = 0
    rejected_max_bounce = 0
    if not use_ev_gate:
        rejected_bounce_prob = int(np.sum(gate_mask & (probs < float(min_bounce_prob))))
        if float(max_bounce_prob) < 1.0:
            rejected_max_bounce = int(np.sum(gate_mask & (probs > float(max_bounce_prob))))

    trade_indices = np.where(trade_mask)[0]
    bars_to_exit = None
    if "pullback_bars_to_exit" in test_df.columns:
        bars_to_exit = pd.to_numeric(
            test_df.loc[pullback_mask, "pullback_bars_to_exit"],
            errors="coerce",
        ).to_numpy(dtype=float)
    if bars_to_exit is None or bars_to_exit.size == 0:
        bars_to_exit = np.full_like(trend_dir, int(config.labels.entry_forward_window), dtype=float)
    bars_to_exit = np.where(np.isfinite(bars_to_exit), bars_to_exit, float(config.labels.entry_forward_window))
    if max_holding_bars is not None and max_holding_bars > 0:
        bars_to_exit = np.minimum(bars_to_exit, float(max_holding_bars))

    exit_types = None
    if "pullback_exit_type" in test_df.columns:
        exit_types = (
            test_df.loc[pullback_mask, "pullback_exit_type"]
            .astype(str)
            .values
        )

    if single_position:
        trade_indices, exit_positions, returns_r, exit_reasons = _simulate_single_position_trades(
            test_df,
            pullback_mask,
            trade_mask,
            trend_dir,
            base_tf,
            float(config.labels.stop_atr_multiple),
            int(config.labels.entry_forward_window),
            realized_r,
            outcomes_r,
            opposite_signal_policy,
            bars_to_exit.astype(int),
            bar_times=bar_times,
            trade_index=trade_index,
            use_intrabar=intrabar_enabled,
            expected_rr=rr_cons,
            use_expected_rr=use_expected_rr,
            target_rr=float(target_rr),
        )
    else:
        if realized_r is not None and realized_r.size == probs.size:
            returns_r = realized_r[trade_mask]
        else:
            returns_r = outcomes_r[trade_mask]
        trade_indices = np.where(trade_mask)[0]
        exit_positions = np.full(trade_indices.shape, -1, dtype=int)
        exit_reasons = ["timeout"] * int(trade_indices.size)

    if returns_r.size > 0:
        returns_r_net = returns_r - fee_r_entry[trade_indices]
        if ops_cost_r != 0.0:
            returns_r_net = returns_r_net - float(ops_cost_r)
    else:
        returns_r_net = returns_r

    # Build trades
    trades: List[Trade] = []
    pullback_pos = np.nonzero(pullback_mask.to_numpy(dtype=bool))[0]
    close_vals = pd.to_numeric(test_df["close"], errors="coerce").to_numpy(dtype=float)
    atr_col = f"{base_tf}_atr"
    atr_vals = pd.to_numeric(test_df[atr_col], errors="coerce").to_numpy(dtype=float) if atr_col in test_df.columns else None
    risk_amount = float(initial_capital) * float(position_size_pct)

    for idx, pb_idx in enumerate(trade_indices):
        entry_pos = int(pullback_pos[pb_idx])
        direction = int(trend_dir[pb_idx])
        if entry_pos < 0 or entry_pos >= len(test_df):
            continue
        entry_row = test_df.iloc[entry_pos]
        entry_price = float(close_vals[entry_pos]) if np.isfinite(close_vals[entry_pos]) else 0.0
        current_atr = float(atr_vals[entry_pos]) if atr_vals is not None and np.isfinite(atr_vals[entry_pos]) else 0.0
        stop_dist = float(config.labels.stop_atr_multiple) * current_atr if current_atr > 0 else 0.0
        stop_loss = entry_price - (direction * stop_dist) if stop_dist > 0 else entry_price
        effective_rr = float(target_rr)
        if use_expected_rr and rr_cons.size > pb_idx and rr_cons[pb_idx] > 0:
            effective_rr = float(rr_cons[pb_idx])
        take_profit = entry_price + (direction * stop_dist * effective_rr) if stop_dist > 0 else entry_price

        exit_pos = int(exit_positions[idx]) if exit_positions.size > idx else -1
        if exit_pos >= 0 and exit_pos < len(test_df):
            exit_row = test_df.iloc[exit_pos]
            exit_price = float(close_vals[exit_pos]) if np.isfinite(close_vals[exit_pos]) else entry_price
            exit_time = _extract_time(exit_row.get("datetime", exit_row.get("bar_time", None)))
        else:
            exit_price = entry_price + (direction * stop_dist * float(returns_r[idx])) if stop_dist > 0 else entry_price
            exit_time = _extract_time(entry_row.get("datetime", entry_row.get("bar_time", None)))

        entry_time = _extract_time(entry_row.get("datetime", entry_row.get("bar_time", None)))
        size = (risk_amount / stop_dist) if stop_dist > 0 else 0.0
        pnl = float(returns_r_net[idx]) * risk_amount if returns_r_net.size > idx else 0.0
        pnl_pct = (pnl / initial_capital * 100.0) if initial_capital > 0 else 0.0
        bounce_prob = float(probs[pb_idx]) if probs.size > pb_idx else 0.0
        trend_prob = float(trend_prob_dir[pb_idx]) if trend_prob_dir.size > pb_idx else 0.0
        slope_dir_val = int(slope_dir[pb_idx]) if slope_dir.size > pb_idx else 0
        trend_aligned = True
        if slope_dir_val != 0:
            trend_aligned = (direction == slope_dir_val)
        is_pullback = True
        quality = _grade_signal(bounce_prob, trend_aligned, is_pullback)
        exit_reason = exit_reasons[idx] if idx < len(exit_reasons) else "timeout"
        if exit_reason in {"timeout", "sl", "tp"} and exit_types is not None and exit_types.size > pb_idx:
            exit_reason = _resolve_exit_reason(str(exit_types[pb_idx]))
        elif exit_reason in {"close", "flip"}:
            pass
        else:
            exit_reason = _resolve_exit_reason(exit_reason)

        dist_from_ema = 0.0
        if "ema_touch_dist" in test_df.columns:
            dist_from_ema = float(test_df.loc[pullback_mask, "ema_touch_dist"].fillna(0).values[pb_idx])

        expected_rr = float(rr_cons[pb_idx]) if rr_cons.size > pb_idx else float(target_rr)
        expected_rr_mean = float(rr_mean[pb_idx]) if rr_mean.size > pb_idx else float(target_rr)
        ev_value = float(ev[pb_idx]) if ev.size > pb_idx else 0.0
        implied_threshold = float(implied[pb_idx]) if implied.size > pb_idx else 0.0
        fee_r = float(fee_r_entry[pb_idx]) if fee_r_entry.size > pb_idx else 0.0

        trade = Trade(
            entry_time=entry_time,
            exit_time=exit_time,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            size=size,
            pnl=pnl,
            pnl_percent=pnl_pct,
            signal_quality=quality,
            exit_reason=exit_reason,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trend_prob=trend_prob,
            bounce_prob=bounce_prob,
            is_pullback=is_pullback,
            trend_aligned=trend_aligned,
            dist_from_ema=dist_from_ema,
            expected_rr=expected_rr,
            expected_rr_mean=expected_rr_mean,
            ev_value=ev_value,
            implied_threshold=implied_threshold,
            fee_r=fee_r,
            stop_dist=stop_dist,
            realized_r=float(returns_r[idx]) if returns_r.size > idx else 0.0,
            realized_r_net=float(returns_r_net[idx]) if returns_r_net.size > idx else 0.0,
        )
        trades.append(trade)

    # Summary stats
    total_trades = len(trades)
    pnl_values = [t.pnl for t in trades]
    equity_curve, max_dd, max_dd_pct = _compute_equity_curve(initial_capital, pnl_values)
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    total_pnl = float(sum(pnl_values)) if pnl_values else 0.0
    avg_win = float(np.mean([t.pnl for t in wins])) if wins else 0.0
    avg_loss = float(np.mean([t.pnl for t in losses])) if losses else 0.0
    gross_win = float(np.sum([t.pnl for t in wins])) if wins else 0.0
    gross_loss = abs(float(np.sum([t.pnl for t in losses]))) if losses else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
    win_rate = float(len(wins) / total_trades) if total_trades > 0 else 0.0
    total_pnl_percent = (total_pnl / initial_capital * 100.0) if initial_capital > 0 else 0.0
    return_std = float(np.std(returns_r_net)) if returns_r_net.size > 1 else 0.0
    sharpe = float(np.mean(returns_r_net) / return_std) if return_std > 0 else 0.0

    trades_by_grade: Dict[str, int] = {}
    win_rate_by_grade: Dict[str, float] = {}
    for grade in ["A", "B", "C"]:
        grade_trades = [t for t in trades if t.signal_quality == grade]
        trades_by_grade[grade] = len(grade_trades)
        if grade_trades:
            win_rate_by_grade[grade] = sum(1 for t in grade_trades if t.pnl > 0) / len(grade_trades)

    signal_stats = {
        "signals_checked": int(pullback_count),
        "total_bars": int(len(test_df)),
        "accepted_signals": int(total_trades),
        "ema_touch_raw": int(pullback_count),
        "ema_touch_passed": int(np.sum(direction_mask)),
        "ema_touch_dir_mismatch": int(np.sum(~direction_mask)),
        "rejected_no_ema_touch": 0,
        "rejected_bounce_prob": int(rejected_bounce_prob),
        "rejected_max_bounce_prob": int(rejected_max_bounce),
        "rejected_ev_gate": int(rejected_ev_gate),
        "rejected_trade_side": int(np.sum((trend_dir != 0) & ~side_mask)),
        "trend_up_signals": int(np.sum(trend_dir == 1)),
        "trend_down_signals": int(np.sum(trend_dir == -1)),
        "trend_neutral_signals": int(np.sum(trend_dir == 0)),
    }

    ema_touch_mask = np.ones_like(probs, dtype=bool)
    if ema_touch_mode == "multi" and "ema_touch_detected" in test_df.columns:
        ema_touch_mask = test_df.loc[pullback_mask, "ema_touch_detected"].fillna(False).astype(bool).values

    bounce_prob_stats = {
        "min_bounce_prob": float(min_bounce_prob),
        "all": _prob_stats(probs, float(min_bounce_prob)),
        "ema_touch": _prob_stats(probs[ema_touch_mask], float(min_bounce_prob)) if ema_touch_mask.size else {},
    }

    diag: Dict[str, Any] = {"feature_audit": feature_audit}
    diag["intrabar_used"] = bool(intrabar_enabled)
    if trend_prob_dir.size > 0:
        diag.update({
            "trend_prob_checked_count": float(trend_prob_dir.size),
            "trend_prob_checked_mean": float(np.mean(trend_prob_dir)),
            "trend_prob_checked_p10": float(np.percentile(trend_prob_dir, 10)),
            "trend_prob_checked_p50": float(np.percentile(trend_prob_dir, 50)),
            "trend_prob_checked_p90": float(np.percentile(trend_prob_dir, 90)),
        })
    if regime_prob_dir.size > 0:
        diag.update({
            "regime_prob_checked_count": float(regime_prob_dir.size),
            "regime_prob_checked_mean": float(np.mean(regime_prob_dir)),
            "regime_prob_checked_p10": float(np.percentile(regime_prob_dir, 10)),
            "regime_prob_checked_p50": float(np.percentile(regime_prob_dir, 50)),
            "regime_prob_checked_p90": float(np.percentile(regime_prob_dir, 90)),
        })
        if regime_id_counts:
            diag["regime_id_counts"] = regime_id_counts

    selected_labels = labels[trade_indices] if trade_indices.size > 0 else np.array([], dtype=int)
    selected_probs = probs[trade_indices] if trade_indices.size > 0 else np.array([], dtype=float)
    selected_ev = ev[trade_indices] if trade_indices.size > 0 else np.array([], dtype=float)
    selected_gap = threshold_gap[trade_indices] if trade_indices.size > 0 else np.array([], dtype=float)
    selected_rr_mean = rr_mean[trade_indices] if trade_indices.size > 0 else np.array([], dtype=float)
    selected_rr_cons = rr_cons[trade_indices] if trade_indices.size > 0 else np.array([], dtype=float)
    selected_realized = returns_r if returns_r.size > 0 else np.array([], dtype=float)
    selected_realized_net = returns_r_net if returns_r_net.size > 0 else np.array([], dtype=float)
    selected_fee_r = fee_r_entry[trade_indices] if trade_indices.size > 0 else np.array([], dtype=float)
    selected_win_r = None
    if win_r_used is not None and trade_indices.size > 0:
        selected_win_r = win_r_used[trade_indices]

    diag.update(
        _compute_ev_diagnostics(
            selected_labels,
            selected_probs,
            selected_ev,
            selected_gap,
            selected_rr_mean,
            selected_rr_cons,
            selected_realized,
            selected_realized_net,
            selected_fee_r,
            selected_win_r,
        )
    )

    result.total_trades = total_trades
    result.winning_trades = len(wins)
    result.losing_trades = len(losses)
    result.win_rate = win_rate
    result.total_pnl = total_pnl
    result.total_pnl_percent = total_pnl_percent
    result.avg_win = avg_win
    result.avg_loss = avg_loss
    result.profit_factor = float(profit_factor) if math.isfinite(profit_factor) else 0.0
    result.max_drawdown = max_dd
    result.max_drawdown_percent = max_dd_pct
    result.sharpe_ratio = sharpe
    result.trades = trades
    result.equity_curve = equity_curve
    result.trades_by_grade = trades_by_grade
    result.win_rate_by_grade = win_rate_by_grade
    result.signal_stats = signal_stats
    result.bounce_prob_stats = bounce_prob_stats
    result.diagnostics = diag
    return result


def build_dataset_from_config(
    cfg: TrendFollowerConfig,
    *,
    use_rust_pipeline: bool = True,
    rust_cache_dir: str = "rust_cache",
    rust_write_intermediate: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    if use_rust_pipeline:
        try:
            import rust_pipeline_bridge as rust_bridge  # type: ignore
            if rust_bridge.is_available():
                labeled, feature_cols, _dataset_path = rust_bridge.build_dataset_from_config(
                    cfg,
                    cache_dir=rust_cache_dir,
                    write_intermediate=rust_write_intermediate,
                    force=False,
                )
                return labeled, feature_cols
        except Exception:
            pass
    from data_loader import load_trades, preprocess_trades, create_multi_timeframe_bars
    from feature_engine import calculate_multi_timeframe_features
    from labels import create_training_dataset
    trades = load_trades(cfg.data, verbose=False)
    trades = preprocess_trades(trades, cfg.data)
    bars = create_multi_timeframe_bars(trades, cfg.features.timeframes, cfg.features.timeframe_names, cfg.data)
    base_tf = cfg.features.timeframe_names[cfg.base_timeframe_idx]
    featured = calculate_multi_timeframe_features(bars, base_tf, cfg.features)
    labeled, feature_cols = create_training_dataset(featured, cfg.labels, cfg.features, base_tf)
    return labeled, feature_cols


def print_backtest_results(result: BacktestResult) -> None:
    """Pretty print backtest results."""
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    print(f"\n  Total Trades:     {result.total_trades}")
    print(f"  Winning Trades:   {result.winning_trades}")
    print(f"  Losing Trades:    {result.losing_trades}")
    print(f"  Win Rate:         {result.win_rate:.1%}")

    print(f"\n  Total P&L:        ${result.total_pnl:,.2f}")
    print(f"  Total Return:     {result.total_pnl_percent:.2f}%")
    print(f"  Avg Win:          ${result.avg_win:,.2f}")
    print(f"  Avg Loss:         ${result.avg_loss:,.2f}")
    print(f"  Profit Factor:    {result.profit_factor:.2f}")

    print(f"\n  Max Drawdown:     ${result.max_drawdown:,.2f}")
    print(f"  Max DD %:         {result.max_drawdown_percent:.2f}%")
    print(f"  Sharpe Ratio:     {result.sharpe_ratio:.2f}")
    if result.trades:
        exit_counts: Dict[str, int] = {}
        for trade in result.trades:
            exit_counts[trade.exit_reason] = exit_counts.get(trade.exit_reason, 0) + 1
        parts = [f"{reason}={count}" for reason, count in sorted(exit_counts.items(), key=lambda kv: (-kv[1], kv[0]))]
        print(f"\n  Exit Reasons:     {', '.join(parts)}")

    if result.signal_stats:
        stats = result.signal_stats
        checked = float(stats.get("signals_checked", 0))
        total_bars = float(stats.get("total_bars", 0))
        accepted = stats.get("accepted_signals", 0)
        print("\n  Signal Gate Diagnostics:")
        print(
            "    Bars Checked:   {} / {} ({:.1f}%)".format(
                int(checked),
                int(total_bars),
                (checked / total_bars * 100.0) if total_bars > 0 else 0.0,
            )
        )
        print(f"    Accepted:       {accepted}")
        print(
            "    EMA Touch:      raw={} passed={} dir_mismatch={}".format(
                stats.get("ema_touch_raw", 0),
                stats.get("ema_touch_passed", 0),
                stats.get("ema_touch_dir_mismatch", 0),
            )
        )
        print(
            "    Rejected:       no_ema_touch={} bounce_prob={} max_bounce_prob={} ev_gate={} trade_side={}".format(
                stats.get("rejected_no_ema_touch", 0),
                stats.get("rejected_bounce_prob", 0),
                stats.get("rejected_max_bounce_prob", 0),
                stats.get("rejected_ev_gate", 0),
                stats.get("rejected_trade_side", 0),
            )
        )
        if checked > 0:
            print(
                "    Trend Signals:  up={} down={} neutral={}".format(
                    stats.get("trend_up_signals", 0),
                    stats.get("trend_down_signals", 0),
                    stats.get("trend_neutral_signals", 0),
                )
            )

    if result.diagnostics:
        diag = result.diagnostics
        if "feature_audit" in diag:
            fa = diag.get("feature_audit", {})
            print("\n  Feature Audit:")
            print(
                "    Expected features: {}  Data features: {}".format(
                    fa.get("expected_count", 0),
                    fa.get("data_count", 0),
                )
            )
            missing = fa.get("missing") or []
            extra = fa.get("extra") or []
            if missing:
                print("    Missing features:")
                for name in missing:
                    print(f"      - {name}")
            if extra:
                print("    Extra features:")
                for name in extra:
                    print(f"      - {name}")
            context_expected = fa.get("context_expected") or []
            if context_expected:
                print("    Context features (added later):")
                for name in context_expected:
                    print(f"      - {name}")
        if "trend_prob_checked_count" in diag or "regime_prob_checked_count" in diag:
            print("\n  Trend/Regime Prob Diagnostics (signals checked):")
            if "trend_prob_checked_count" in diag:
                print(
                    "    Trend prob: count={:.0f} mean={:.3f} p10={:.3f} p50={:.3f} p90={:.3f}".format(
                        diag.get("trend_prob_checked_count", 0.0),
                        diag.get("trend_prob_checked_mean", 0.0),
                        diag.get("trend_prob_checked_p10", 0.0),
                        diag.get("trend_prob_checked_p50", 0.0),
                        diag.get("trend_prob_checked_p90", 0.0),
                    )
                )
            if "regime_prob_checked_count" in diag:
                print(
                    "    Regime prob: count={:.0f} mean={:.3f} p10={:.3f} p50={:.3f} p90={:.3f}".format(
                        diag.get("regime_prob_checked_count", 0.0),
                        diag.get("regime_prob_checked_mean", 0.0),
                        diag.get("regime_prob_checked_p10", 0.0),
                        diag.get("regime_prob_checked_p50", 0.0),
                        diag.get("regime_prob_checked_p90", 0.0),
                    )
                )
            if "regime_id_counts" in diag:
                print(f"    Regime id counts: {diag.get('regime_id_counts')}")

    if result.bounce_prob_stats:
        bp_stats = result.bounce_prob_stats
        min_thr = float(bp_stats.get("min_bounce_prob", 0.0))
        all_stats = bp_stats.get("all", {})
        touch_stats = bp_stats.get("ema_touch", {})

        def _print_bp(label: str, s: dict) -> None:
            print(
                "    {}: count={:.0f} mean={:.3f} p50={:.3f} p90={:.3f} max={:.3f} pct>=thr={:.1f}%".format(
                    label,
                    s.get("count", 0.0),
                    s.get("mean", 0.0),
                    s.get("p50", 0.0),
                    s.get("p90", 0.0),
                    s.get("max", 0.0),
                    s.get("pct_above_threshold", 0.0) * 100.0,
                )
            )

        print("\n  Bounce Prob Diagnostics:")
        print(f"    Threshold: {min_thr:.4f}")
        _print_bp("All Signals", all_stats)
        _print_bp("EMA Touch", touch_stats)

    if result.diagnostics:
        diag = result.diagnostics
        print("\n  EV Diagnostics:")
        print(f"    Trades:          {diag.get('trade_count', 0)}")
        print(
            "    Brier:           {brier:.4f}  LogLoss: {logloss:.4f}  ECE: {ece:.4f}".format(
                brier=diag.get("brier", 0.0),
                logloss=diag.get("logloss", 0.0),
                ece=diag.get("ece", 0.0),
            )
        )
        print(
            "    EV mean/p10:     {mean:.4f} / {p10:.4f}".format(
                mean=diag.get("ev_mean", 0.0),
                p10=diag.get("ev_p10", 0.0),
            )
        )
        print(
            "    Gap mean/p10:    {mean:.4f} / {p10:.4f}".format(
                mean=diag.get("threshold_gap_mean", 0.0),
                p10=diag.get("threshold_gap_p10", 0.0),
            )
        )
        print(
            "    RR mean/cons:    {mean:.4f} / {cons:.4f}".format(
                mean=diag.get("expected_rr_mean", 0.0),
                cons=diag.get("expected_rr_cons_mean", 0.0),
            )
        )
        print(
            "    Realized R:      {mean:.4f} (net {net:.4f}) fee_r {fee:.4f}".format(
                mean=diag.get("realized_r_mean", 0.0),
                net=diag.get("realized_r_net_mean", 0.0),
                fee=diag.get("fee_r_mean", 0.0),
            )
        )
        print(
            "    RR bias/mae:     {ratio:.3f} / {mae:.3f}".format(
                ratio=diag.get("expected_rr_bias_ratio", 0.0),
                mae=diag.get("expected_rr_mae", 0.0),
            )
        )
        ev_bins = diag.get("ev_bin_summary", "")
        if ev_bins:
            print(f"    EV bins (Rnet):  {ev_bins}")

    print("\n  Performance by Signal Grade:")
    for grade in ["A", "B", "C"]:
        count = result.trades_by_grade.get(grade, 0)
        wr = result.win_rate_by_grade.get(grade, 0)
        print(f"    Grade {grade}: {count} trades, {wr:.1%} win rate")

    print("=" * 60)


def _trade_to_dict(trade: Trade, *, timeframe_seconds: Optional[int] = None) -> dict:
    """Serialize Trade dataclass into a JSON-friendly dict."""
    def _as_py(v):
        if isinstance(v, np.generic):
            return v.item()
        return v
    def _to_datetime(value):
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (int, float)):
            ts = float(value)
            if ts > 1e11:
                ts /= 1000.0
            return datetime.fromtimestamp(ts)
        return None

    entry_bar_open_time = _to_datetime(trade.entry_time)
    exit_bar_open_time = _to_datetime(trade.exit_time)

    entry_bar_close_time = None
    exit_bar_close_time = None
    if timeframe_seconds is not None:
        if entry_bar_open_time is not None:
            entry_bar_close_time = entry_bar_open_time + timedelta(seconds=int(timeframe_seconds))
        if exit_bar_open_time is not None:
            exit_bar_close_time = exit_bar_open_time + timedelta(seconds=int(timeframe_seconds))

    return {
        "entry_time": (
            entry_bar_close_time.isoformat()
            if entry_bar_close_time
            else (entry_bar_open_time.isoformat() if entry_bar_open_time else None)
        ),
        "entry_bar_open_time": entry_bar_open_time.isoformat() if entry_bar_open_time else None,
        "entry_bar_close_time": entry_bar_close_time.isoformat() if entry_bar_close_time else None,
        "exit_time": exit_bar_open_time.isoformat() if exit_bar_open_time else None,
        "exit_bar_open_time": exit_bar_open_time.isoformat() if exit_bar_open_time else None,
        "exit_bar_close_time": exit_bar_close_time.isoformat() if exit_bar_close_time else None,
        "direction": trade.direction,
        "entry_price": _as_py(trade.entry_price),
        "exit_price": _as_py(trade.exit_price),
        "size": _as_py(trade.size),
        "pnl": _as_py(trade.pnl),
        "pnl_percent": _as_py(trade.pnl_percent),
        "signal_quality": trade.signal_quality,
        "exit_reason": trade.exit_reason,
        "stop_loss": _as_py(trade.stop_loss),
        "take_profit": _as_py(trade.take_profit),
        "trend_prob": _as_py(trade.trend_prob),
        "bounce_prob": _as_py(trade.bounce_prob),
        "is_pullback": _as_py(trade.is_pullback),
        "trend_aligned": _as_py(trade.trend_aligned),
        "dist_from_ema": _as_py(trade.dist_from_ema),
        "expected_rr": _as_py(trade.expected_rr),
        "expected_rr_mean": _as_py(trade.expected_rr_mean),
        "ev_value": _as_py(trade.ev_value),
        "implied_threshold": _as_py(trade.implied_threshold),
        "fee_r": _as_py(trade.fee_r),
        "stop_dist": _as_py(trade.stop_dist),
        "realized_r": _as_py(trade.realized_r),
        "realized_r_net": _as_py(trade.realized_r_net),
    }


def save_backtest_logs(
    result: BacktestResult,
    config: TrendFollowerConfig,
    log_dir: Path,
    *,
    model_dir: Optional[Path] = None,
    driver: Optional[str] = None,
    parameters: Optional[dict] = None,
    extra_metrics: Optional[dict] = None,
) -> Dict[str, Path]:
    """Save a backtest run summary + per-trade JSONL to disk."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    symbol_tag = Path(getattr(config.data, "data_dir", "data")).name if hasattr(config, "data") else "data"
    base_tf = config.features.timeframe_names[config.base_timeframe_idx] if hasattr(config, "features") else "tf"
    prefix = f"backtest_{symbol_tag}_{base_tf}_{run_id}"

    trades_path = log_dir / f"{prefix}_trades.jsonl"
    summary_path = log_dir / f"{prefix}_summary.json"

    base_tf_seconds = None
    if hasattr(config, "features") and hasattr(config.features, "timeframes"):
        base_tf_seconds = int(config.features.timeframes[config.base_timeframe_idx])

    with trades_path.open("w", encoding="utf-8") as f:
        for t in result.trades:
            f.write(json.dumps(_trade_to_dict(t, timeframe_seconds=base_tf_seconds), ensure_ascii=False) + "\n")

    summary = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "driver": driver,
        "model_dir": str(model_dir) if model_dir is not None else None,
        "data_dir": str(getattr(config.data, "data_dir", None)) if hasattr(config, "data") else None,
        "base_timeframe": base_tf,
        "base_timeframe_seconds": base_tf_seconds,
        "parameters": parameters or {},
        "metrics": {
            "total_trades": result.total_trades,
            "winning_trades": result.winning_trades,
            "losing_trades": result.losing_trades,
            "win_rate": result.win_rate,
            "total_pnl": result.total_pnl,
            "total_return_percent": result.total_pnl_percent,
            "avg_win": result.avg_win,
            "avg_loss": result.avg_loss,
            "profit_factor": result.profit_factor,
            "max_drawdown": result.max_drawdown,
            "max_drawdown_percent": result.max_drawdown_percent,
            "sharpe_ratio": result.sharpe_ratio,
            "trades_by_grade": result.trades_by_grade,
            "win_rate_by_grade": result.win_rate_by_grade,
        },
        "signal_stats": result.signal_stats,
        "bounce_prob_stats": result.bounce_prob_stats,
        "diagnostics": result.diagnostics,
        "extra_metrics": extra_metrics or {},
        "files": {
            "trades_jsonl": str(trades_path),
        },
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return {"summary": summary_path, "trades": trades_path}
if __name__ == "__main__":
    print("backtest_tuned_config module loaded")
