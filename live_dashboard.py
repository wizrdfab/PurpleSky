"""
Live Metrics Dashboard (read-only).
Serves a small web UI backed by live_metrics_<symbol>.jsonl.
"""
import argparse
import json
import os
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


def tail_jsonl(path: Path, limit: int = 200, max_bytes: int = 512 * 1024):
    if not path.exists() or path.stat().st_size == 0:
        return []
    size = path.stat().st_size
    block = min(size, max_bytes)
    with open(path, "rb") as f:
        f.seek(-block, os.SEEK_END)
        data = f.read(block)
    lines = data.splitlines()
    if size > block and lines:
        lines = lines[1:]
    records = []
    for line in lines[-limit:]:
        try:
            records.append(json.loads(line.decode("utf-8")))
        except Exception:
            continue
    return records


def read_latest(path: Path):
    records = tail_jsonl(path, limit=1, max_bytes=64 * 1024)
    return records[-1] if records else {}


def safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def safe_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def parse_ts(ts: str):
    if not ts:
        return None
    try:
        return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    except Exception:
        return None


def feature_map(latest: dict):
    vec = latest.get("feature_vector") or {}
    names = vec.get("names") or []
    values = vec.get("values") or []
    mapping = {}
    for i, name in enumerate(names):
        if i >= len(values):
            break
        mapping[name] = values[i]
    return mapping


def compute_trade_stats(metrics_path: Path, limit: int = 400):
    metrics = tail_jsonl(metrics_path, limit=limit)
    if not metrics:
        return {}
    enriched = []
    for item in metrics:
        ts = parse_ts(item.get("ts"))
        enriched.append((ts if ts is not None else 0, item))
    enriched.sort(key=lambda x: x[0])
    ordered = [item for _, item in enriched]

    latest_startup = ordered[-1].get("startup_time")
    if latest_startup:
        ordered = [item for item in ordered if item.get("startup_time") == latest_startup]
        if not ordered:
            ordered = [enriched[-1][1]]

    daily_pnls = [safe_float(item.get("daily_pnl")) for item in ordered]
    trade_pnls = []
    for i in range(1, len(daily_pnls)):
        diff = daily_pnls[i] - daily_pnls[i - 1]
        if abs(diff) <= 1e-9:
            continue
        prev_ts = parse_ts(ordered[i - 1].get("ts"))
        curr_ts = parse_ts(ordered[i].get("ts"))
        if prev_ts and curr_ts:
            prev_date = datetime.fromtimestamp(prev_ts, tz=timezone.utc).date()
            curr_date = datetime.fromtimestamp(curr_ts, tz=timezone.utc).date()
            if curr_date != prev_date and daily_pnls[i] <= 0.0001:
                # Daily reset; ignore this jump.
                continue
        trade_pnls.append(diff)

    equity = [safe_float(item.get("equity")) for item in ordered if safe_float(item.get("equity")) > 0]
    max_dd_pct = None
    if len(equity) >= 2:
        peak = equity[0]
        max_dd = 0.0
        for val in equity:
            peak = max(peak, val)
            if peak > 0:
                dd = (val - peak) / peak
                if dd < max_dd:
                    max_dd = dd
        max_dd_pct = abs(max_dd) * 100.0

    if not trade_pnls:
        return {
            "trades": 0,
            "win_rate": None,
            "sortino": None,
            "profit_factor": None,
            "avg_trade": None,
            "avg_win": None,
            "avg_loss": None,
            "max_drawdown_pct": max_dd_pct,
        }

    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p < 0]
    trade_count = len(trade_pnls)
    win_rate = len(wins) / trade_count if trade_count else None
    avg_trade = sum(trade_pnls) / trade_count if trade_count else None
    avg_win = sum(wins) / len(wins) if wins else None
    avg_loss = sum(losses) / len(losses) if losses else None
    profit_factor = None
    if wins and losses:
        profit_factor = sum(wins) / abs(sum(losses))

    sortino = None
    if losses:
        downside = (sum(p * p for p in losses) / len(losses)) ** 0.5
        if downside > 0:
            sortino = (avg_trade or 0.0) / downside
    else:
        sortino = 10.0

    return {
        "trades": trade_count,
        "win_rate": win_rate,
        "sortino": sortino,
        "profit_factor": profit_factor,
        "avg_trade": avg_trade,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_drawdown_pct": max_dd_pct,
    }


def is_protective_order(order: dict) -> bool:
    order_type = str(order.get("orderType") or "").lower()
    order_filter = str(order.get("orderFilter") or "").lower()
    stop_order_type = str(order.get("stopOrderType") or "").lower()
    create_type = str(order.get("createType") or "").lower()
    if safe_bool(order.get("reduceOnly")) or safe_bool(order.get("closeOnTrigger")):
        return True
    if "tpsl" in order_filter:
        return True
    if order_filter in {"stoporder", "tpslorder"}:
        return True
    protective_types = {
        "takeprofit",
        "stoploss",
        "takeprofitlimit",
        "stoplosslimit",
        "takeprofitmarket",
        "stoplossmarket",
    }
    if order_type in protective_types:
        return True
    if "stop" in create_type or "takeprofit" in create_type or "trailing" in create_type:
        return True
    protective_stop_types = {"takeprofit", "stoploss", "trailingstop"}
    return stop_order_type in protective_stop_types


def read_open_orders_latest(metrics_path: Path, symbol: str):
    orders_path = metrics_path.with_name(f"open_orders_{symbol}.jsonl")
    latest = read_latest(orders_path)
    orders = latest.get("open_orders") if isinstance(latest, dict) else None
    if not isinstance(orders, list):
        return []
    resting = []
    for order in orders:
        if is_protective_order(order):
            continue
        resting.append({
            "order_id": order.get("orderId"),
            "side": order.get("side"),
            "price": safe_float(order.get("price")),
            "qty": safe_float(order.get("qty")),
            "status": order.get("orderStatus"),
            "reduce_only": safe_bool(order.get("reduceOnly")),
            "position_idx": order.get("positionIdx"),
        })
    return resting


def extract_price(latest: dict):
    pos = latest.get("position") or {}
    price = safe_float(pos.get("mark_price"))
    if price > 0:
        return price
    positions = latest.get("positions") or {}
    if isinstance(positions, dict):
        for key in ("long", "short"):
            p = safe_float((positions.get(key) or {}).get("mark_price"))
            if p > 0:
                return p
    fmap = feature_map(latest)
    for key in ("close", "mark_price", "last_price"):
        val = fmap.get(key)
        if val is not None:
            try:
                val_f = float(val)
            except Exception:
                continue
            if val_f > 0:
                return val_f
    return None


def augment_latest(latest: dict, symbol: str, metrics_path: Path):
    if not isinstance(latest, dict):
        return latest
    price = extract_price(latest)
    resting = read_open_orders_latest(metrics_path, symbol)
    latest["dashboard"] = {
        "price": price,
        "resting_orders": resting,
        "resting_count": len(resting),
        "trade_stats": compute_trade_stats(metrics_path),
    }
    return latest


def discover_metrics_files(metrics_dir: Path, symbols=None):
    mapping = {}
    if not metrics_dir.exists():
        return mapping
    for path in metrics_dir.glob("live_metrics_*.jsonl"):
        symbol = path.stem.replace("live_metrics_", "", 1)
        if not symbol:
            continue
        if symbols and symbol not in symbols:
            continue
        mapping[symbol] = path
    return mapping


def summary_from_files(file_map):
    items = []
    for symbol in sorted(file_map.keys()):
        path = file_map[symbol]
        latest = read_latest(path)
        if not latest:
            latest = {"symbol": symbol, "ts": None}
        elif not latest.get("symbol"):
            latest["symbol"] = symbol
        latest = augment_latest(latest, symbol, path)
        items.append(latest)
    return items


EXTRA_SCRIPT = """
<script>
(function() {
  if (window.__dashExtrasInjected) return;
  window.__dashExtrasInjected = true;

  function fmt(v, d) {
    if (v === null || v === undefined) return "--";
    const num = Number(v);
    if (Number.isNaN(num)) return "--";
    return num.toFixed(d);
  }

  function ensureCard() {
    if (document.getElementById("dashMarketCard")) return;
    const grid = document.querySelector(".grid");
    if (!grid) return;
    const card = document.createElement("div");
    card.className = "card";
    card.id = "dashMarketCard";
    card.innerHTML = `
      <h3>Market</h3>
      <div class="value" id="dashPrice">--</div>
      <div class="row"><span>Position Mode</span><span id="dashPosMode">--</span></div>
      <div class="row"><span>Resting Orders</span><span id="dashOrderCount">--</span></div>
      <div class="row"><span>Limits</span><span id="dashOrderList" style="white-space: pre-line; text-align: right;">--</span></div>
    `;
    grid.appendChild(card);
  }

  function ensurePerfCard() {
    if (document.getElementById("dashPerfCard")) return;
    const grid = document.querySelector(".grid");
    if (!grid) return;
    const card = document.createElement("div");
    card.className = "card";
    card.id = "dashPerfCard";
    card.innerHTML = `
      <h3>Performance</h3>
      <div class="value" id="dashWinRate">--</div>
      <div class="row"><span>Trades</span><span id="dashTradeCount">--</span></div>
      <div class="row"><span>Sortino</span><span id="dashSortino">--</span></div>
      <div class="row"><span>Profit Factor</span><span id="dashProfitFactor">--</span></div>
      <div class="row"><span>Avg Trade</span><span id="dashAvgTrade">--</span></div>
      <div class="row"><span>Max DD</span><span id="dashMaxDD">--</span></div>
    `;
    grid.appendChild(card);
  }

  function formatOrders(list) {
    if (!Array.isArray(list) || !list.length) return "--";
    return list.slice(0, 4).map(item => {
      const side = item.side || "?";
      const price = fmt(item.price, 6);
      const qty = fmt(item.qty, 4);
      return `${side} ${price} x${qty}`;
    }).join("\\n");
  }

  function updateExtra(data) {
    if (!data) return;
    ensureCard();
    ensurePerfCard();
    const dash = data.dashboard || {};
    const price = dash.price;
    const priceEl = document.getElementById("dashPrice");
    if (priceEl) priceEl.textContent = price ? fmt(price, 6) : "--";
    const modeEl = document.getElementById("dashPosMode");
    if (modeEl) modeEl.textContent = data.position_mode || "--";
    const countEl = document.getElementById("dashOrderCount");
    if (countEl) countEl.textContent = dash.resting_count ?? (dash.resting_orders ? dash.resting_orders.length : "--");
    const listEl = document.getElementById("dashOrderList");
    if (listEl) listEl.textContent = formatOrders(dash.resting_orders);

    const stats = dash.trade_stats || {};
    const winRate = stats.win_rate;
    const winRateText = (winRate === null || winRate === undefined) ? "--" : `${(winRate * 100).toFixed(1)}%`;
    const winRateEl = document.getElementById("dashWinRate");
    if (winRateEl) winRateEl.textContent = winRateText;
    const tradeCountEl = document.getElementById("dashTradeCount");
    if (tradeCountEl) tradeCountEl.textContent = stats.trades ?? "--";
    const sortinoEl = document.getElementById("dashSortino");
    if (sortinoEl) sortinoEl.textContent = (stats.sortino === null || stats.sortino === undefined) ? "--" : fmt(stats.sortino, 2);
    const pfEl = document.getElementById("dashProfitFactor");
    if (pfEl) pfEl.textContent = (stats.profit_factor === null || stats.profit_factor === undefined) ? "--" : fmt(stats.profit_factor, 2);
    const avgTradeEl = document.getElementById("dashAvgTrade");
    if (avgTradeEl) avgTradeEl.textContent = (stats.avg_trade === null || stats.avg_trade === undefined) ? "--" : fmt(stats.avg_trade, 4);
    const maxDdEl = document.getElementById("dashMaxDD");
    if (maxDdEl) maxDdEl.textContent = (stats.max_drawdown_pct === null || stats.max_drawdown_pct === undefined) ? "--" : `${fmt(stats.max_drawdown_pct, 2)}%`;
  }

  function hook() {
    if (typeof updateDetail === "function") {
      const base = updateDetail;
      window.updateDetail = function(data) {
        base(data);
        updateExtra(data);
      };
      return;
    }
    if (typeof updateLatest === "function") {
      const base = updateLatest;
      window.updateLatest = function(data) {
        base(data);
        updateExtra(data);
      };
      return;
    }
    setTimeout(hook, 250);
  }

  hook();
})();
</script>
"""


def inject_dashboard_html(html: str) -> str:
    if "dashMarketCard" in html:
        return html
    marker = "</body>"
    if marker in html:
        return html.replace(marker, EXTRA_SCRIPT + "\n" + marker)
    return html + EXTRA_SCRIPT


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Live Trading Dashboard</title>
  <style>
    :root {
      --bg1: #0f172a;
      --bg2: #111827;
      --panel: #0b1220;
      --accent: #22c55e;
      --accent2: #f59e0b;
      --danger: #ef4444;
      --muted: #94a3b8;
      --text: #e5e7eb;
      --glow: rgba(34, 197, 94, 0.2);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Space Grotesk", "Bahnschrift", "Trebuchet MS", sans-serif;
      color: var(--text);
      background: radial-gradient(1200px 600px at 10% 10%, #0b1430 0%, transparent 60%),
                  radial-gradient(1200px 600px at 90% 20%, #1b2a3a 0%, transparent 60%),
                  linear-gradient(160deg, var(--bg1), var(--bg2));
      min-height: 100vh;
    }
    header {
      padding: 28px 32px 10px;
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      gap: 16px;
    }
    h1 {
      margin: 0;
      font-size: 28px;
      letter-spacing: 0.5px;
    }
    .sub {
      color: var(--muted);
      font-size: 13px;
      margin-top: 6px;
    }
    .pill {
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.08);
      font-size: 12px;
      color: var(--muted);
    }
    .grid {
      padding: 16px 32px 40px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
    }
    .card {
      background: linear-gradient(180deg, rgba(11,18,32,0.95), rgba(11,18,32,0.85));
      border: 1px solid rgba(148,163,184,0.15);
      border-radius: 16px;
      padding: 16px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.25);
      animation: fadeUp 0.5s ease forwards;
      opacity: 0;
    }
    .card h3 {
      margin: 0 0 10px;
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
    }
    .value {
      font-size: 24px;
      font-weight: 600;
    }
    .row {
      display: flex;
      justify-content: space-between;
      font-size: 13px;
      padding: 4px 0;
      border-bottom: 1px dashed rgba(148,163,184,0.1);
    }
    .row:last-child { border-bottom: none; }
    .ok { color: var(--accent); }
    .warn { color: var(--accent2); }
    .bad { color: var(--danger); }
    .muted { color: var(--muted); }
    canvas {
      width: 100%;
      height: 80px;
      margin-top: 8px;
    }
    @keyframes fadeUp {
      from { transform: translateY(10px); opacity: 0; }
      to { transform: translateY(0); opacity: 1; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1 id="symbol">Live Trading Dashboard</h1>
      <div class="sub" id="subline">Waiting for data...</div>
    </div>
    <div class="pill" id="statusPill">Offline</div>
  </header>
  <section class="grid">
    <div class="card" style="animation-delay: 0.05s">
      <h3>Equity & PnL</h3>
      <div class="value" id="equityVal">--</div>
      <div class="row"><span>Daily PnL</span><span id="dailyPnl">--</span></div>
      <div class="row"><span>Unrealized</span><span id="unrealPnl">--</span></div>
      <canvas id="equityChart"></canvas>
    </div>
    <div class="card" style="animation-delay: 0.1s">
      <h3>Position</h3>
      <div class="value" id="posSide">--</div>
      <div class="row"><span>Size</span><span id="posSize">--</span></div>
      <div class="row"><span>Entry</span><span id="posEntry">--</span></div>
      <div class="row"><span>Mark</span><span id="posMark">--</span></div>
      <div class="row"><span>TP / SL</span><span id="posTPSL">--</span></div>
    </div>
    <div class="card" style="animation-delay: 0.15s">
      <h3>Health</h3>
      <div class="value" id="healthVal">--</div>
      <div class="row"><span>Sentiment</span><span id="sentimentVal">--</span></div>
      <div class="row"><span>Regime</span><span id="regimeVal">--</span></div>
      <div class="row"><span>Trade Enabled</span><span id="tradeEnabled">--</span></div>
      <div class="row"><span>Drift Alerts</span><span id="driftAlerts">--</span></div>
    </div>
    <div class="card" style="animation-delay: 0.2s">
      <h3>Data Quality</h3>
      <div class="value" id="barsVal">--</div>
      <div class="row"><span>Macro %</span><span id="macroPct">--</span></div>
      <div class="row"><span>OB Density</span><span id="obDensity">--</span></div>
      <div class="row"><span>Trade Continuity</span><span id="tradeCont">--</span></div>
      <div class="row"><span>Lag Trade / Bar</span><span id="lagTradeBar">--</span></div>
    </div>
    <div class="card" style="animation-delay: 0.25s">
      <h3>Latency</h3>
      <div class="value" id="restLatency">--</div>
      <div class="row"><span>WS Trade</span><span id="wsTrade">--</span></div>
      <div class="row"><span>WS OB</span><span id="wsOb">--</span></div>
      <div class="row"><span>OB Lag</span><span id="obLag">--</span></div>
      <canvas id="latencyChart"></canvas>
    </div>
    <div class="card" style="animation-delay: 0.3s">
      <h3>Orders</h3>
      <div class="value" id="openOrders">--</div>
      <div class="row"><span>Last Reconcile</span><span id="lastReconcile">--</span></div>
      <div class="row"><span>Source</span><span id="reconcileSource">--</span></div>
      <div class="row"><span>Protective</span><span id="protectiveCount">--</span></div>
    </div>
    <div class="card" style="animation-delay: 0.35s">
      <h3>Errors</h3>
      <div class="value" id="errorCount">--</div>
      <div class="row"><span>Last Runtime</span><span id="lastRuntime">--</span></div>
      <div class="row"><span>Last API</span><span id="lastApi">--</span></div>
    </div>
  </section>
  <script>
    const fmt = (v, d=2) => (v === null || v === undefined) ? "--" : Number(v).toFixed(d);
    const fmtPct = (v) => (v === null || v === undefined) ? "--" : `${Number(v).toFixed(1)}%`;
    const fmtSec = (v) => (v === null || v === undefined) ? "--" : `${Number(v).toFixed(1)}s`;
    const fmtMs = (v) => (v === null || v === undefined) ? "--" : `${Number(v).toFixed(1)}ms`;

    function sparkline(canvas, values, color) {
      const ctx = canvas.getContext("2d");
      const w = canvas.width = canvas.clientWidth;
      const h = canvas.height = canvas.clientHeight;
      ctx.clearRect(0, 0, w, h);
      if (!values.length) return;
      const min = Math.min(...values);
      const max = Math.max(...values);
      const span = (max - min) || 1;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      values.forEach((v, i) => {
        const x = (i / (values.length - 1)) * w;
        const y = h - ((v - min) / span) * h;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    function updateLatest(data) {
      if (!data || !data.symbol) return;
      document.getElementById("symbol").textContent = `Live Trading Dashboard - ${data.symbol}`;
      document.getElementById("subline").textContent = `Startup: ${data.startup_time} - Uptime: ${Math.round(data.uptime_sec)}s`;
      document.getElementById("statusPill").textContent = data.trade_enabled ? "Trading Enabled" : "Trading Paused";
      document.getElementById("statusPill").className = "pill " + (data.trade_enabled ? "ok" : "warn");

      document.getElementById("equityVal").textContent = data.equity ? fmt(data.equity, 4) : "--";
      document.getElementById("dailyPnl").textContent = fmt(data.daily_pnl, 4);
      document.getElementById("unrealPnl").textContent = fmt(data.position.unreal_pnl, 4);

      document.getElementById("posSide").textContent = data.position.side || "--";
      document.getElementById("posSize").textContent = fmt(data.position.size, 4);
      document.getElementById("posEntry").textContent = fmt(data.position.entry_price, 6);
      document.getElementById("posMark").textContent = fmt(data.position.mark_price, 6);
      const tp = fmt(data.position.take_profit, 6);
      const sl = fmt(data.position.stop_loss, 6);
      document.getElementById("posTPSL").textContent = `${tp} / ${sl}`;

      document.getElementById("healthVal").textContent = data.health.status || "--";
      document.getElementById("sentimentVal").textContent = data.health.sentiment || "--";
      document.getElementById("regimeVal").textContent = data.health.regime || "--";
      document.getElementById("tradeEnabled").textContent = data.trade_enabled ? "Yes" : "No";
      const drift = data.drift.alerts && data.drift.alerts.length ? data.drift.alerts.join(", ") : "None";
      document.getElementById("driftAlerts").textContent = drift;

      document.getElementById("barsVal").textContent = data.data_health.bars || 0;
      document.getElementById("macroPct").textContent = fmtPct(data.data_health.macro_pct);
      document.getElementById("obDensity").textContent = fmtPct(data.data_health.ob_density_pct);
      document.getElementById("tradeCont").textContent = data.data_health.trade_cont ?? "--";
      document.getElementById("lagTradeBar").textContent = `${fmtSec(data.latency.lag_trade_sec)} / ${fmtSec(data.latency.lag_bar_sec)}`;

      document.getElementById("restLatency").textContent = fmtMs(data.latency.rest_avg_ms);
      document.getElementById("wsTrade").textContent = fmtMs(data.latency.ws_trade_ms);
      document.getElementById("wsOb").textContent = fmtMs(data.latency.ws_ob_ms);
      document.getElementById("obLag").textContent = fmtSec(data.latency.ob_lag_sec);

      document.getElementById("openOrders").textContent = data.orders.open_orders ?? 0;
      const rec = data.orders.last_reconcile || {};
      document.getElementById("lastReconcile").textContent = rec.ts || "--";
      document.getElementById("reconcileSource").textContent = rec.source || "--";
      document.getElementById("protectiveCount").textContent = rec.protective_orders ?? "--";

      document.getElementById("errorCount").textContent = data.errors.runtime_count + " / " + data.errors.api_count;
      document.getElementById("lastRuntime").textContent = data.errors.last_runtime_error || "--";
      document.getElementById("lastApi").textContent = data.errors.last_api_error || "--";
    }

    async function fetchLatest() {
      const resp = await fetch("/api/latest");
      if (!resp.ok) return;
      const data = await resp.json();
      updateLatest(data);
    }

    async function fetchHistory() {
      const resp = await fetch("/api/metrics?limit=200");
      if (!resp.ok) return;
      const data = await resp.json();
      const equity = data.map(x => Number(x.equity || 0));
      const latency = data.map(x => Number(x.latency?.rest_avg_ms || 0));
      sparkline(document.getElementById("equityChart"), equity, "#22c55e");
      sparkline(document.getElementById("latencyChart"), latency, "#f59e0b");
    }

    fetchLatest();
    fetchHistory();
    setInterval(fetchLatest, 2000);
    setInterval(fetchHistory, 10000);
  </script>
</body>
</html>
"""

HTML_TEMPLATE_PATH = Path(__file__).with_name("live_dashboard.html")
if HTML_TEMPLATE_PATH.exists():
    try:
        HTML_PAGE = HTML_TEMPLATE_PATH.read_text(encoding="utf-8")
    except Exception:
        pass
HTML_PAGE = inject_dashboard_html(HTML_PAGE)


class MetricsHandler(BaseHTTPRequestHandler):
    metrics_dir: Path = Path(".")
    symbols_filter = None
    fixed_files = None
    history_limit: int = 200

    def _send(self, status, payload, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if isinstance(payload, (dict, list)):
            payload = json.dumps(payload, default=str).encode("utf-8")
        elif isinstance(payload, str):
            payload = payload.encode("utf-8")
        self.wfile.write(payload)

    def _resolve_files(self):
        if self.fixed_files:
            return self.fixed_files
        return discover_metrics_files(self.metrics_dir, self.symbols_filter)

    def _choose_symbol(self, file_map, symbol):
        if symbol and symbol in file_map:
            return symbol
        symbols = sorted(file_map.keys())
        return symbols[0] if symbols else None

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self._send(200, HTML_PAGE, "text/html; charset=utf-8")

        file_map = self._resolve_files()
        if parsed.path == "/api/symbols":
            return self._send(200, sorted(file_map.keys()))

        if parsed.path == "/api/summary":
            return self._send(200, summary_from_files(file_map))

        if parsed.path == "/api/latest":
            qs = parse_qs(parsed.query)
            symbol = qs.get("symbol", [""])[0] or None
            symbol = self._choose_symbol(file_map, symbol)
            if not symbol:
                return self._send(200, {})
            latest = read_latest(file_map[symbol])
            if not latest:
                latest = {"symbol": symbol, "ts": None}
            elif not latest.get("symbol"):
                latest["symbol"] = symbol
            latest = augment_latest(latest, symbol, file_map[symbol])
            return self._send(200, latest)

        if parsed.path == "/api/metrics":
            qs = parse_qs(parsed.query)
            symbol = qs.get("symbol", [""])[0] or None
            symbol = self._choose_symbol(file_map, symbol)
            if not symbol:
                return self._send(200, [])
            limit = int(qs.get("limit", [self.history_limit])[0])
            return self._send(200, tail_jsonl(file_map[symbol], limit=limit))

        return self._send(404, {"error": "not found"})


class DiscordNotifier:
    def __init__(
        self,
        webhook_url: str,
        metrics_dir: Path,
        poll_sec: float = 5.0,
        offline_sec: float = 120.0,
        cooldown_sec: float = 60.0,
        drawdown_pct: float = 10.0,
        fixed_files=None,
        symbols_filter=None,
    ):
        self.webhook_url = webhook_url
        self.metrics_dir = metrics_dir
        self.poll_sec = poll_sec
        self.offline_sec = offline_sec
        self.cooldown_sec = cooldown_sec
        self.drawdown_pct = drawdown_pct
        self.fixed_files = fixed_files
        self.symbols_filter = symbols_filter
        self.last_state = {}
        self.last_sent = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        if self.webhook_url:
            self._thread.start()

    def stop(self):
        self._stop.set()

    def _send(self, content: str) -> None:
        if not self.webhook_url:
            return
        payload = json.dumps({"content": content}).encode("utf-8")
        req = Request(
            self.webhook_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "live-dashboard/1.0",
            },
        )
        try:
            with urlopen(req, timeout=10) as resp:
                resp.read()
        except HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            print(f"Discord notify failed: HTTP {exc.code} {exc.reason} {detail}")
        except URLError as exc:
            print(f"Discord notify failed: {exc}")
        except Exception as exc:
            print(f"Discord notify error: {exc}")

    def _cooldown_ok(self, symbol: str, event: str) -> bool:
        key = f"{symbol}:{event}"
        now = time.time()
        last = self.last_sent.get(key, 0.0)
        if now - last < self.cooldown_sec:
            return False
        self.last_sent[key] = now
        return True

    def _notify(self, symbol: str, event: str, message: str) -> None:
        if not self._cooldown_ok(symbol, event):
            return
        self._send(message)

    def _is_order_error(self, text: str) -> bool:
        if not text:
            return False
        lower = text.lower()
        keywords = [
            "place",
            "create",
            "reject",
            "insufficient",
            "order",
            "cancel",
            "close",
            "set_trading_stop",
            "stoploss",
            "takeprofit",
            "position",
        ]
        return any(k in lower for k in keywords)

    def _check_symbol(self, symbol: str, latest: dict) -> None:
        state = self.last_state.setdefault(symbol, {})
        now = time.time()
        ts = parse_ts(latest.get("ts")) if latest else None
        online = ts is not None and (now - ts) <= self.offline_sec
        if state.get("online", True) and not online:
            age = now - ts if ts else None
            age_text = f"{age:.1f}s" if age is not None else "unknown"
            self._notify(symbol, "offline", f"{symbol} offline (last update {age_text} ago)")
        state["online"] = online

        if not latest or not online:
            return

        startup = latest.get("startup_time")
        if startup and startup != state.get("startup_time"):
            self._notify(symbol, "started", f"{symbol} started (startup {startup})")
            state["startup_time"] = startup

        errors = latest.get("errors") or {}
        runtime_count = int(errors.get("runtime_count") or 0)
        api_count = int(errors.get("api_count") or 0)
        if runtime_count > state.get("runtime_count", 0) or api_count > state.get("api_count", 0):
            err_text = errors.get("last_runtime_error") or errors.get("last_api_error") or "unknown error"
            self._notify(symbol, "error", f"{symbol} error: {err_text}")
            if self._is_order_error(err_text):
                self._notify(symbol, "order_error", f"{symbol} order error: {err_text}")
        state["runtime_count"] = runtime_count
        state["api_count"] = api_count

        equity = safe_float(latest.get("equity"), 0.0)
        daily_pnl = safe_float(latest.get("daily_pnl"), 0.0)
        drawdown = equity > 0 and daily_pnl <= -(self.drawdown_pct / 100.0) * equity
        if drawdown and not state.get("drawdown_alerted"):
            pct = (daily_pnl / equity) * 100.0 if equity else 0.0
            self._notify(
                symbol,
                "drawdown",
                f"{symbol} drawdown alert: {daily_pnl:.2f} ({pct:.1f}%)",
            )
            state["drawdown_alerted"] = True
        elif not drawdown:
            state["drawdown_alerted"] = False

    def _run(self) -> None:
        while not self._stop.is_set():
            if self.fixed_files:
                file_map = self.fixed_files
            else:
                file_map = discover_metrics_files(self.metrics_dir, self.symbols_filter)
            for symbol, path in file_map.items():
                latest = read_latest(path)
                if latest and not latest.get("symbol"):
                    latest["symbol"] = symbol
                self._check_symbol(symbol, latest or {})
            time.sleep(self.poll_sec)


def main():
    parser = argparse.ArgumentParser(description="Live Trading Dashboard")
    parser.add_argument("--metrics-dir", type=str, default=".")
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument("--symbol", type=str, default="")
    parser.add_argument("--metrics-file", type=str, default="")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--history-limit", type=int, default=200)
    parser.add_argument("--discord-webhook", type=str, default="")
    parser.add_argument("--notify-offline-sec", type=float, default=120.0)
    parser.add_argument("--notify-poll-sec", type=float, default=5.0)
    parser.add_argument("--notify-cooldown-sec", type=float, default=60.0)
    parser.add_argument("--notify-drawdown-pct", type=float, default=10.0)
    args = parser.parse_args()

    MetricsHandler.metrics_dir = Path(args.metrics_dir)
    symbols_filter = None
    if args.metrics_file:
        file_path = Path(args.metrics_file)
        symbol = args.symbol or file_path.stem.replace("live_metrics_", "", 1)
        MetricsHandler.fixed_files = {symbol: file_path}
    else:
        if args.symbols:
            symbols_filter = {s.strip() for s in args.symbols.split(",") if s.strip()}
        elif args.symbol:
            symbols_filter = {args.symbol}
        MetricsHandler.symbols_filter = symbols_filter

    MetricsHandler.history_limit = args.history_limit

    webhook = args.discord_webhook or os.getenv("DISCORD_WEBHOOK_URL", "")
    notifier = DiscordNotifier(
        webhook_url=webhook,
        metrics_dir=MetricsHandler.metrics_dir,
        poll_sec=args.notify_poll_sec,
        offline_sec=args.notify_offline_sec,
        cooldown_sec=args.notify_cooldown_sec,
        drawdown_pct=args.notify_drawdown_pct,
        fixed_files=MetricsHandler.fixed_files,
        symbols_filter=MetricsHandler.symbols_filter,
    )
    notifier.start()

    server = HTTPServer((args.host, args.port), MetricsHandler)
    print(f"Dashboard running on http://{args.host}:{args.port}")
    if MetricsHandler.fixed_files:
        print(f"Reading metrics: {list(MetricsHandler.fixed_files.values())[0]}")
    else:
        print(f"Reading metrics dir: {MetricsHandler.metrics_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
