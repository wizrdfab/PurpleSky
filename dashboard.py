"""
PurpleSky TUI Dashboard - Professional Trading Interface.
Uses the Rich library for real-time terminal UI.
"""
from datetime import datetime
from rich.console import Console, Group
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.progress import BarColumn, Progress, TextColumn
from rich.align import Align
from rich.theme import Theme

# Custom theme for a "Pro Trader" look
PURPLE_SKY_THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "danger": "bold red",
    "success": "bold green",
    "muted": "bright_black",
    "brand": "bold magenta"
})

class TradingDashboard:
    def __init__(self, symbol: str, mode: str = "LIVE"):
        self.console = Console(theme=PURPLE_SKY_THEME)
        self.symbol = symbol
        self.mode = mode
        self.layout = self._make_layout()
        self.start_time = datetime.now()
        
        # Internal State
        self.data = {
            "price": 0.0,
            "pos_size": 0.0,
            "equity": 0.0,
            "unreal_pnl": 0.0,
            "total_pnl": 0.0,
            "friction": 0.0,
            "health": "CALIBRATING",
            "regime": "CALIBRATING",
            "sentiment": "NEUTRAL",
            "orders": {},
            "signals": {"long": 0.0, "short": 0.0, "thresh": 0.6},
            "recent_logs": []
        }

    def _make_layout(self) -> Layout:
        layout = Layout(name="root")
        layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=14)
        )
        layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=3)
        )
        layout["left"].split_column(
            Layout(name="position", size=8),
            Layout(name="orders", ratio=1)
        )
        layout["right"].split_column(
            Layout(name="market", size=10),
            Layout(name="health", ratio=1)
        )
        return layout

    def update(self, **kwargs):
        self.data.update(kwargs)

    def _get_header(self) -> Panel:
        uptime = str(datetime.now() - self.start_time).split('.')[0]
        mode_color = "success" if self.mode == "LIVE" else "warning"
        
        grid = Table.grid(expand=True)
        grid.add_column(justify="left", ratio=1)
        grid.add_column(justify="center", ratio=1)
        grid.add_column(justify="right", ratio=1)
        
        grid.add_row(
            Text(f" ❖ PURPLESKY // QUANT_CORE_v2.1 ", style="bold cyan"),
            Text(f"⬢ SYMBOL: {self.symbol} ⬢", style="brand"),
            Text(f"UPTIME: {uptime} | {datetime.now().strftime('%H:%M:%S')} ", style="muted")
        )
        return Panel(grid, style="white", border_style="bright_black", subtitle=f"[ {self.mode} MODE ]")

    def _get_position_panel(self) -> Panel:
        size = self.data["pos_size"]
        pnl = self.data["unreal_pnl"]
        pnl_style = "success" if pnl > 0 else "danger" if pnl < 0 else "white"
        
        table = Table.grid(padding=(0, 1))
        table.add_column(style="muted")
        table.add_column(style="bold")
        
        side = "LONG" if size > 0 else "SHORT" if size < 0 else "FLAT"
        side_style = "success" if size > 0 else "danger" if size < 0 else "white"
        
        table.add_row("Position:", Text(side, style=side_style))
        table.add_row("Size:", f"{abs(size):.2f}")
        table.add_row("Unreal PnL:", Text(f"${pnl:+.2f}", style=pnl_style))
        table.add_row("Equity:", f"${self.data['equity']:.2f}")
        
        return Panel(Align.center(table, vertical="middle"), title=" ⚙ POSITION_ENGINE ", border_style="brand")

    def _get_orders_panel(self) -> Panel:
        table = Table(box=None, expand=True, padding=(0, 1))
        table.add_column("ID", style="muted", no_wrap=True)
        table.add_column("Side", justify="center")
        table.add_column("Price", justify="right")
        table.add_column("Qty", justify="right")
        
        for oid, o in self.data["orders"].items():
            style = "success" if o['side'] == "Buy" else "danger"
            table.add_row(str(oid)[:8], Text(o['side'], style=style), f"{o['price']:.4f}", f"{o['qty']:.1f}")
            
        return Panel(table, title=" ☰ ORDER_STACK ", border_style="info")

    def _get_market_panel(self) -> Panel:
        price = self.data["price"]
        sig = self.data["signals"]
        
        # Simple signal gauge
        def make_gauge(val, thresh, color):
            filled = int(val * 10)
            bar = "█" * filled + "░" * (10 - filled)
            return Text(f"[{bar}] {val:.2f}", style=color if val > thresh else "muted")

        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_row(Text(f"LIVE_PRICE: {price:.4f}", style="bold yellow", justify="center"))
        grid.add_row("")
        grid.add_row(Text("ML_PREDICTOR_CONFIDENCE:", style="muted"))
        grid.add_row(Text("BULLISH: ") + make_gauge(sig['long'], sig['thresh'], "success"))
        grid.add_row(Text("BEARISH: ") + make_gauge(sig['short'], sig['thresh'], "danger"))
        
        return Panel(grid, title=" ⚡ ALPHA_FEED ", border_style="cyan")

    def _get_health_panel(self) -> Panel:
        table = Table.grid(padding=(0, 1))
        table.add_column(style="muted")
        table.add_column()
        
        table.add_row("Strategy Health:", self.data["health"])
        table.add_row("Regime Drift:", self.data["regime"])
        table.add_row("Sentiment:", self.data["sentiment"])
        table.add_row("Total Realized:", Text(f"${self.data['total_pnl']:+.2f}", style="success" if self.data['total_pnl'] >= 0 else "danger"))
        table.add_row("Friction Tracker:", Text(f"${self.data['friction']:.2f}", style="warning"))
        
        return Panel(Align.center(table, vertical="middle"), title=" ✚ SYSTEM_VITALS ", border_style="white")

    def _get_footer(self) -> Panel:
        # Filter noise to prevent jumping
        logs = [l for l in self.data["recent_logs"] if "[HEARTBEAT]" not in l]
        logs = logs[-11:] # Show last 11 lines
        
        log_text = Text()
        for log in logs:
            if "!!!" in log: log_text.append(f" {log}\n", style="bold yellow")
            elif "CHAMPION DECISION" in log: log_text.append(f" {log}\n", style="bold cyan")
            elif "[FEATURES]" in log: log_text.append(f" {log}\n", style="muted")
            elif "Verdict" in log: log_text.append(f" {log}\n", style="success")
            else: log_text.append(f" {log}\n", style="muted")
            
        return Panel(log_text, title=" ░ SYSTEM_TERMINAL_OUT ░ ", border_style="bright_black")

    def generate_layout(self) -> Layout:
        self.layout["header"].update(self._get_header())
        self.layout["position"].update(self._get_position_panel())
        self.layout["orders"].update(self._get_orders_panel())
        self.layout["market"].update(self._get_market_panel())
        self.layout["health"].update(self._get_health_panel())
        self.layout["footer"].update(self._get_footer())
        return self.layout

def test_ui():
    import time
    dash = TradingDashboard("RAVEUSDT", "DRY RUN")
    with Live(dash.generate_layout(), console=dash.console, refresh_per_second=4) as live:
        for i in range(100):
            dash.update(
                price=0.4567 + (i * 0.0001),
                unreal_pnl=i * 0.1,
                equity=1000 + i,
                signals={"long": 0.4 + (i/200), "short": 0.1, "thresh": 0.6},
                recent_logs=[f"Processing bar {i}...", "Verdict: Neutral", "Waiting for fill..."]
            )
            live.update(dash.generate_layout())
            time.sleep(0.1)

if __name__ == "__main__":
    test_ui()
