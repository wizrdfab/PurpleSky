
import ctypes
import json
import multiprocessing as mp
import os
import socket
import sys
import threading
import traceback
import time
import webbrowser
from http.client import HTTPConnection
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import tkinter as tk
from tkinter import messagebox, ttk

import customtkinter as ctk

import live_dashboard
import live_trading
from translations import t, get_language, set_language, detect_system_language


def get_resource_root() -> Path:
    override = os.getenv("PURPLESKY_RESOURCE_DIR")
    if override:
        return Path(override).expanduser()
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", "")
        if base:
            return Path(base)
    return Path(__file__).resolve().parent


def get_exe_dir() -> Path:
    override = os.getenv("PURPLESKY_EXE_DIR")
    if override:
        return Path(override).expanduser()
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def get_user_data_dir() -> Path:
    base = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA")
    if not base:
        base = str(Path.home())
    path = Path(base) / "PurpleSky"
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return path


FROZEN = bool(
    getattr(sys, "frozen", False)
    or os.getenv("PURPLESKY_FORCE_FROZEN")
    or os.getenv("PURPLESKY_EXE_DIR")
    or os.getenv("PURPLESKY_RESOURCE_DIR")
)
RESOURCE_ROOT = get_resource_root()
EXE_DIR = get_exe_dir()
DATA_DIR = get_user_data_dir() if FROZEN else EXE_DIR
STATE_PATH = DATA_DIR / "launcher_state.json"
KEYS_PATH = DATA_DIR / "key_profiles.json"
LEGACY_STATE_PATH = EXE_DIR / "launcher_state.json"
LEGACY_KEYS_PATH = EXE_DIR / "key_profiles.json"


def model_root() -> Path:
    if FROZEN:
        exe_models = EXE_DIR / "models"
        if exe_models.is_dir():
            return exe_models
        parent_models = EXE_DIR.parent / "models"
        if parent_models.is_dir():
            return parent_models
        resource_models = RESOURCE_ROOT / "models"
        if resource_models.is_dir():
            return resource_models
    return EXE_DIR / "models"


def resource_path(name: str) -> Path:
    primary = RESOURCE_ROOT / name
    if primary.exists():
        return primary
    fallback = EXE_DIR / name
    if fallback.exists():
        return fallback
    return primary
SOURCE_URL = "https://github.com/wizrdfab/PurpleSky"
DASHBOARD_PORT = 9007
DEBUG_ENABLED = os.getenv("PURPLESKY_DEBUG", "").strip().lower() not in {"", "0", "false", "no"}


@dataclass
class ModelOption:
    symbol: str
    path: Path


@dataclass
class RunItem:
    run_id: str
    option: ModelOption
    profile: str
    metrics_key: str
    status: str = "Idle"
    mode: str = "signal"
    process: Optional[mp.Process] = None

def discover_models() -> List[ModelOption]:
    options: List[ModelOption] = []
    root = model_root()
    if not root.is_dir():
        debug_log(f"No models directory found at {root}")
        return options
    seen = set()
    for model_file in root.rglob("model_long.pkl"):
        model_dir = model_file.parent
        if not (model_dir / "model_short.pkl").exists():
            continue
        if not (model_dir / "features.pkl").exists():
            continue
        key = str(model_dir.resolve())
        if key in seen:
            continue
        seen.add(key)
        symbol = read_params_symbol(model_dir / "params.json") or fallback_symbol(model_dir)
        options.append(ModelOption(symbol=symbol, path=model_dir.resolve()))
    options.sort(key=lambda item: (item.symbol.lower(), str(item.path)))
    debug_log(f"Discovered {len(options)} models under {root}")
    return options


def read_params_symbol(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    symbol = data.get("symbol")
    if not symbol and isinstance(data.get("meta"), dict):
        symbol = data["meta"].get("symbol")
    if not symbol:
        return None
    return str(symbol).strip() or None


def fallback_symbol(model_dir: Path) -> str:
    name = model_dir.name
    if name.lower().startswith("rank_") and model_dir.parent:
        name = model_dir.parent.name
    return name


def display_model_option(option: ModelOption) -> str:
    try:
        display_path = str(option.path.relative_to(Path.cwd()))
    except ValueError:
        display_path = str(option.path)
    return f"{option.symbol} | {display_path}"


def display_model_choice(option: ModelOption, symbol: str) -> str:
    full = display_model_option(option)
    prefix = f"{symbol} | "
    if full.startswith(prefix):
        return full[len(prefix):]
    return full


def display_model_name(option: ModelOption) -> str:
    name = option.path.name
    if name.lower().startswith("rank_") and option.path.parent:
        name = option.path.parent.name
    return name


def state_model_path(path: Path) -> str:
    try:
        return str(path.relative_to(EXE_DIR))
    except ValueError:
        return str(path)


def resolve_state_path(text: str) -> Optional[Path]:
    if not text:
        return None
    try:
        candidate = Path(text).expanduser()
        if not candidate.is_absolute():
            candidate = EXE_DIR / candidate
        return candidate.resolve()
    except Exception:
        return None


def slugify(text: str) -> str:
    out = []
    for ch in text:
        if ch.isascii() and (ch.isalnum() or ch in {"_", "-"}):
            out.append(ch)
        else:
            out.append("_")
    cleaned = "".join(out).strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned


def model_key(option: ModelOption) -> str:
    try:
        rel = option.path.relative_to(model_root())
        rel_text = "__".join(rel.parts)
    except ValueError:
        rel_text = option.path.name
    base = f"{option.symbol}__{rel_text}" if rel_text else option.symbol
    return slugify(base)[:80] or slugify(option.symbol)[:80]


def ensure_data_dir() -> None:
    if DATA_DIR.exists():
        return
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        return


def debug_log(message: str) -> None:
    if not DEBUG_ENABLED:
        return
    ensure_data_dir()
    try:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with (DATA_DIR / "launcher_debug.log").open("a", encoding="utf-8", errors="ignore") as handle:
            handle.write(f"{stamp} {message}\n")
    except Exception:
        pass

def load_state() -> Dict:
    ensure_data_dir()
    if not STATE_PATH.exists() and LEGACY_STATE_PATH.exists():
        try:
            STATE_PATH.write_text(LEGACY_STATE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            pass
    if not STATE_PATH.exists():
        return {}
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def save_state(state: Dict) -> None:
    ensure_data_dir()
    try:
        STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")
    except Exception:
        pass

def load_key_profiles() -> Dict[str, Dict[str, str]]:
    ensure_data_dir()
    if not KEYS_PATH.exists() and LEGACY_KEYS_PATH.exists():
        try:
            KEYS_PATH.write_text(LEGACY_KEYS_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            pass
    if not KEYS_PATH.exists():
        return {}
    try:
        data = json.loads(KEYS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(data, dict):
        profiles = data.get("profiles") if isinstance(data.get("profiles"), dict) else data
        return profiles if isinstance(profiles, dict) else {}
    if isinstance(data, list):
        profiles = {}
        for entry in data:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if not name:
                continue
            profiles[str(name)] = {
                "api_key": str(entry.get("api_key") or entry.get("key") or ""),
                "api_secret": str(entry.get("api_secret") or entry.get("secret") or ""),
            }
        return profiles
    return {}


def save_key_profiles(profiles: Dict[str, Dict[str, str]]) -> None:
    payload = {"profiles": profiles}
    ensure_data_dir()
    try:
        KEYS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    except Exception:
        pass

def build_dashboard_args(notifications: Dict) -> List[str]:
    ensure_data_dir()
    args = ["--metrics-dir", str(DATA_DIR), "--port", str(DASHBOARD_PORT), "--lang", get_language()]
    if not notifications.get("enabled"):
        args.extend(["--no-notify-system", "--no-notify-signals", "--no-notify-direction"])
        return args
    if notifications.get("discord"):
        args.extend(["--discord-webhook", notifications["discord"]])
    if notifications.get("telegram_token") and notifications.get("telegram_chat_id"):
        args.extend(["--telegram-token", notifications["telegram_token"]])
        args.extend(["--telegram-chat-id", notifications["telegram_chat_id"]])
    if not notifications.get("notify_system", True):
        args.append("--no-notify-system")
    if not notifications.get("notify_signals", True):
        args.append("--no-notify-signals")
    if not notifications.get("notify_direction", True):
        args.append("--no-notify-direction")
    return args

def run_dashboard(args: List[str]) -> None:
    ensure_data_dir()
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with (DATA_DIR / "dashboard_status.log").open("a", encoding="utf-8", errors="ignore") as handle:
            handle.write(f"{stamp} dashboard start args={args}\n")
    except Exception:
        pass
    if sys.stdout is None:
        try:
            sys.stdout = (DATA_DIR / "dashboard_stdout.log").open("a", encoding="utf-8", errors="ignore")
        except Exception:
            pass
    if sys.stderr is None:
        try:
            sys.stderr = (DATA_DIR / "dashboard_stdout.log").open("a", encoding="utf-8", errors="ignore")
        except Exception:
            pass
    try:
        live_dashboard.main(args)
    except SystemExit:
        try:
            with (DATA_DIR / "dashboard_status.log").open("a", encoding="utf-8", errors="ignore") as handle:
                handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} dashboard exit (SystemExit)\n")
        except Exception:
            pass
        return
    except Exception:
        ensure_data_dir()
        try:
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with (DATA_DIR / "dashboard_error.log").open("a", encoding="utf-8", errors="ignore") as handle:
                handle.write(f"{stamp} run_dashboard failed\n")
                handle.write(traceback.format_exc())
                handle.write("\n")
            with (DATA_DIR / "dashboard_status.log").open("a", encoding="utf-8", errors="ignore") as handle:
                handle.write(f"{stamp} dashboard exit (error)\n")
        except Exception:
            pass


def run_live_trading(args: List[str]) -> None:
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")
    ensure_data_dir()
    try:
        os.chdir(DATA_DIR)
    except Exception:
        pass
    try:
        cli_args = live_trading.parse_args(args)
        trader = live_trading.LiveTradingV2(cli_args)
        trader.run()
    except Exception:
        ensure_data_dir()
        try:
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with (DATA_DIR / "live_trading_error.log").open("a", encoding="utf-8", errors="ignore") as handle:
                handle.write(f"{stamp} run_live_trading failed\n")
                handle.write(traceback.format_exc())
                handle.write("\n")
        except Exception:
            pass


def open_dashboard(mode: str, host: str = "127.0.0.1", port: int = DASHBOARD_PORT) -> bool:
    view = "traders" if mode == "signal" else "account"
    url = f"http://{host}:{port}/?mode={mode}&view={view}"
    try:
        if webbrowser.open(url, new=2):
            return True
    except Exception:
        pass
    if os.name == "nt":
        try:
            os.startfile(url)
            return True
        except Exception:
            return False
    return False


def center_window(window: tk.Toplevel) -> None:
    window.update_idletasks()
    width = window.winfo_reqwidth()
    height = window.winfo_reqheight()
    screen_w = window.winfo_screenwidth()
    screen_h = window.winfo_screenheight()
    width = min(width, max(screen_w - 40, 200))
    height = min(height, max(screen_h - 40, 200))
    x = max((screen_w - width) // 2, 0)
    y = max((screen_h - height) // 2, 0)
    window.geometry(f"{width}x{height}+{x}+{y}")


def apply_treeview_style() -> None:
    style = ttk.Style()
    style.configure(
        "Modern.Treeview",
        background="#FFFFFF",
        fieldbackground="#FFFFFF",
        foreground="#1F2933",
        rowheight=24,
        bordercolor="#E2E8F0",
        borderwidth=1,
    )
    style.configure(
        "Modern.Treeview.Heading",
        background="#F2F5F9",
        foreground="#1F2933",
        font=("Segoe UI", 10, "bold"),
    )
    style.map(
        "Modern.Treeview",
        background=[("selected", "#DCEBFF")],
        foreground=[("selected", "#1F2933")],
    )


class SoftCheckBox(ctk.CTkCheckBox):
    def _draw(self, no_color_updates: bool = False):
        try:
            self._draw_engine.preferred_drawing_method = "font_shapes"
        except Exception:
            pass
        super()._draw(no_color_updates=no_color_updates)
        self._restyle_checkmark()

    def _restyle_checkmark(self) -> None:
        if not getattr(self, "_canvas", None):
            return
        item_ids = self._canvas.find_withtag("checkmark")
        if not item_ids:
            return
        width = max(1, round(self._checkbox_height / 10))
        for item_id in item_ids:
            try:
                item_type = self._canvas.type(item_id)
                if item_type == "line":
                    self._canvas.itemconfig(item_id, joinstyle=tk.ROUND, capstyle=tk.ROUND, width=width)
                elif item_type == "text":
                    size = max(8, round(self._checkbox_height * 0.55))
                    self._canvas.itemconfig(item_id, font=("CustomTkinter_shapes_font", -size))
            except Exception:
                continue


class SoftRadioButton(ctk.CTkRadioButton):
    def _draw(self, no_color_updates: bool = False):
        try:
            self._draw_engine.preferred_drawing_method = "font_shapes"
        except Exception:
            pass
        super()._draw(no_color_updates=no_color_updates)


def normalize_mode(value: str) -> str:
    value = (value or "").lower()
    return "signal" if value == "signal" else "auto"


def format_run_status(item: RunItem) -> str:
    return item.status

class KeyManagerDialog:
    def __init__(self, parent: tk.Tk, on_save) -> None:
        self.top = ctk.CTkToplevel(parent)
        self.top.title(t("api_keys_title"))
        self.top.resizable(False, False)
        self.top.transient(parent)
        self.top.grab_set()
        self.on_save = on_save

        self.profile_var = tk.StringVar()
        self.api_key_var = tk.StringVar()
        self.api_secret_var = tk.StringVar()

        frame = ctk.CTkFrame(self.top)
        frame.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)

        ctk.CTkLabel(frame, text=t("lbl_profile_name")).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            frame,
            text=t("profile_name_hint"),
            font=ctk.CTkFont(size=10),
            wraplength=320,
            justify="left",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 2))
        self.profile_combo = ctk.CTkComboBox(
            frame,
            variable=self.profile_var,
            values=[],
            width=240,
            command=self._load_selected,
        )
        self.profile_combo.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(2, 8))

        ctk.CTkLabel(frame, text=t("lbl_api_key")).grid(row=3, column=0, sticky="w")
        ctk.CTkEntry(frame, textvariable=self.api_key_var, width=320).grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=(2, 8)
        )

        ctk.CTkLabel(frame, text=t("lbl_api_secret")).grid(row=5, column=0, sticky="w")
        ctk.CTkEntry(frame, textvariable=self.api_secret_var, width=320, show="*").grid(
            row=6, column=0, columnspan=2, sticky="ew", pady=(2, 10)
        )

        button_row = ctk.CTkFrame(frame, fg_color="transparent")
        button_row.grid(row=7, column=0, columnspan=2, sticky="e")
        ctk.CTkButton(button_row, text=t("btn_delete"), command=self._delete_profile).grid(row=0, column=0, sticky="w")
        ctk.CTkButton(button_row, text=t("btn_save"), command=self._save).grid(row=0, column=1, sticky="w", padx=(8, 0))
        ctk.CTkButton(button_row, text=t("btn_close"), command=self.top.destroy).grid(row=0, column=2, sticky="e", padx=(8, 0))

        self.refresh_profiles()
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        center_window(self.top)

    def refresh_profiles(self) -> None:
        profiles = load_key_profiles()
        names = sorted(profiles.keys())
        if "default" not in names:
            names.insert(0, "default")
        self.profile_combo.configure(values=names)
        if names and self.profile_var.get() not in names:
            self.profile_var.set(names[0])
            self.profile_combo.set(names[0])

    def _load_selected(self, _value: str = "") -> None:
        name = self.profile_var.get().strip()
        profiles = load_key_profiles()
        entry = profiles.get(name)
        if not entry:
            self.api_key_var.set("")
            self.api_secret_var.set("")
            return
        self.api_key_var.set(str(entry.get("api_key") or ""))
        self.api_secret_var.set(str(entry.get("api_secret") or ""))

    def _save(self) -> None:
        name = self.profile_var.get().strip()
        api_key = self.api_key_var.get().strip()
        api_secret = self.api_secret_var.get().strip()
        if not name:
            messagebox.showerror(t("api_keys_title"), t("msg_profile_required"), parent=self.top)
            return
        if not api_key or not api_secret:
            messagebox.showerror(t("api_keys_title"), t("msg_keys_required"), parent=self.top)
            return
        profiles = load_key_profiles()
        profiles[name] = {"api_key": api_key, "api_secret": api_secret}
        save_key_profiles(profiles)
        self.refresh_profiles()
        if self.on_save:
            self.on_save(name)
        messagebox.showinfo(t("api_keys_title"), t("msg_profile_saved"), parent=self.top)

    def _delete_profile(self) -> None:
        name = self.profile_var.get().strip()
        if not name:
            messagebox.showerror(t("api_keys_title"), t("msg_select_profile_delete"), parent=self.top)
            return
        profiles = load_key_profiles()
        if name not in profiles:
            messagebox.showinfo(t("api_keys_title"), t("msg_profile_not_found"), parent=self.top)
            return
        if not messagebox.askyesno(t("api_keys_title"), t("msg_delete_profile_confirm", name=name), parent=self.top):
            return
        del profiles[name]
        save_key_profiles(profiles)
        self.refresh_profiles()
        self.api_key_var.set("")
        self.api_secret_var.set("")
        if self.on_save:
            self.on_save("")


class NotificationsDialog:
    def __init__(
        self,
        parent: tk.Tk,
        discord_var: tk.StringVar,
        telegram_token_var: tk.StringVar,
        telegram_chat_id_var: tk.StringVar,
        notify_system_var: tk.BooleanVar,
        notify_signals_var: tk.BooleanVar,
        notify_direction_var: tk.BooleanVar,
    ) -> None:
        self.top = ctk.CTkToplevel(parent)
        self.top.title(t("notifications_title"))
        self.top.resizable(False, False)
        self.top.transient(parent)
        self.top.grab_set()

        frame = ctk.CTkFrame(self.top)
        frame.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)

        ctk.CTkLabel(frame, text=t("lbl_discord_webhook")).grid(row=0, column=0, sticky="w")
        ctk.CTkEntry(frame, textvariable=discord_var, width=360).grid(row=1, column=0, sticky="ew", pady=(2, 6))

        ctk.CTkLabel(frame, text=t("lbl_telegram_token")).grid(row=2, column=0, sticky="w")
        ctk.CTkEntry(frame, textvariable=telegram_token_var, width=360).grid(row=3, column=0, sticky="ew", pady=(2, 6))

        ctk.CTkLabel(frame, text=t("lbl_telegram_chat_id")).grid(row=4, column=0, sticky="w")
        ctk.CTkEntry(frame, textvariable=telegram_chat_id_var, width=360).grid(
            row=5, column=0, sticky="ew", pady=(2, 10)
        )

        notify_types = ctk.CTkFrame(frame, fg_color="transparent")
        notify_types.grid(row=6, column=0, sticky="w")
        SoftCheckBox(
            notify_types,
            text=t("chk_system"),
            variable=notify_system_var,
            checkbox_width=14,
            checkbox_height=14,
            corner_radius=6,
            border_width=1,
        ).grid(row=0, column=0, sticky="w")
        SoftCheckBox(
            notify_types,
            text=t("chk_signals"),
            variable=notify_signals_var,
            checkbox_width=14,
            checkbox_height=14,
            corner_radius=6,
            border_width=1,
        ).grid(row=0, column=1, sticky="w", padx=(10, 0))
        SoftCheckBox(
            notify_types,
            text=t("chk_direction"),
            variable=notify_direction_var,
            checkbox_width=14,
            checkbox_height=14,
            corner_radius=6,
            border_width=1,
        ).grid(row=0, column=2, sticky="w", padx=(10, 0))

        button_frame = ctk.CTkFrame(frame, fg_color="transparent")
        button_frame.grid(row=7, column=0, sticky="e", pady=(12, 0))
        ctk.CTkButton(button_frame, text=t("btn_close"), command=self.top.destroy).grid(row=0, column=0, sticky="e")

        frame.columnconfigure(0, weight=1)
        center_window(self.top)


class ModelConfigDialog:
    def __init__(self, app: "LauncherApp") -> None:
        self.app = app
        self.top = ctk.CTkToplevel(app.root)
        self.top.title(t("configure_models_title"))
        self.top.resizable(False, False)
        self.top.transient(app.root)
        self.top.grab_set()
        self.top.protocol("WM_DELETE_WINDOW", self.close)

        container = ctk.CTkFrame(self.top, fg_color="transparent")
        container.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        title = ctk.CTkLabel(container, text=t("section_models"), font=app.section_font)
        title.grid(row=0, column=0, sticky="w")

        run_frame = ctk.CTkFrame(container, fg_color="transparent")
        run_frame.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ctk.CTkLabel(run_frame, text=t("lbl_run_list")).grid(row=0, column=0, sticky="w")
        app.run_tree = ttk.Treeview(
            run_frame,
            columns=("symbol", "model", "profile"),
            show="headings",
            height=3,
            selectmode="extended",
            style="Modern.Treeview",
        )
        app.run_tree.heading("symbol", text=t("col_symbol"))
        app.run_tree.heading("model", text=t("col_model"))
        app.run_tree.heading("profile", text=t("col_profile"))
        app.run_tree.column("symbol", width=110)
        app.run_tree.column("model", width=260)
        app.run_tree.column("profile", width=120)
        app.run_scroll = ctk.CTkScrollbar(run_frame, orientation="vertical", command=app.run_tree.yview)
        app.run_tree.configure(yscrollcommand=app.run_scroll.set)
        app.run_tree.grid(row=1, column=0, sticky="ew", pady=(2, 4))
        app.run_scroll.grid(row=1, column=1, sticky="ns", pady=(2, 4))

        run_controls = ctk.CTkFrame(run_frame, fg_color="transparent")
        run_controls.grid(row=2, column=0, sticky="ew")
        ctk.CTkButton(run_controls, text=t("btn_remove"), command=app.remove_from_run_list).grid(
            row=0, column=0, sticky="w"
        )
        ctk.CTkButton(run_controls, text=t("btn_add_model"), command=app.open_add_model_dialog).grid(
            row=0, column=1, sticky="e", padx=(8, 0)
        )
        run_controls.columnconfigure(1, weight=1)

        app.profile_frame = ctk.CTkFrame(run_frame, fg_color="transparent")
        app.profile_frame.grid(row=3, column=0, sticky="ew", pady=(4, 0))
        app.profile_line = ctk.CTkFrame(app.profile_frame, fg_color="transparent")
        app.profile_line.grid(row=0, column=0, sticky="w")
        app.profile_label = ctk.CTkLabel(app.profile_line, text=t("lbl_api_profile"))
        app.profile_label.grid(row=0, column=0, sticky="w")
        app.profile_combo = ctk.CTkComboBox(
            app.profile_line,
            variable=app.profile_var,
            values=[],
            width=180,
            state="readonly",
        )
        app.profile_combo.grid(row=0, column=1, sticky="w", padx=(8, 0))
        app.profile_line.columnconfigure(1, weight=1)

        app.profile_buttons = ctk.CTkFrame(app.profile_frame, fg_color="transparent")
        app.profile_buttons.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        app.profile_apply_button = ctk.CTkButton(
            app.profile_buttons, text=t("btn_apply_selected"), command=app.apply_profile, width=140
        )
        app.profile_apply_button.grid(row=0, column=0, sticky="w")
        app.profile_hint = ctk.CTkLabel(
            app.profile_buttons,
            text=t("profile_hint"),
            font=app.desc_font,
            wraplength=320,
            justify="left",
        )
        app.profile_hint.grid(row=0, column=1, sticky="w", padx=(8, 0))
        app.profile_keys_button = ctk.CTkButton(
            app.profile_buttons, text=t("btn_add_manage_profiles"), command=app.open_key_manager, width=200
        )
        app.profile_keys_button.grid(row=0, column=2, sticky="e", padx=(8, 0))
        app.profile_buttons.columnconfigure(1, weight=1)
        app.profile_frame.columnconfigure(0, weight=1)

        run_frame.columnconfigure(0, weight=1)
        run_frame.columnconfigure(1, weight=0)
        run_frame.rowconfigure(1, weight=0)

        footer = ctk.CTkFrame(container, fg_color="transparent")
        footer.grid(row=4, column=0, sticky="e", pady=(8, 0))
        ctk.CTkButton(footer, text=t("btn_close"), command=self.close).grid(row=0, column=0, sticky="e")

        container.columnconfigure(0, weight=1)
        app.refresh_key_profiles()
        app._sync_run_tree()
        app._apply_mode_state()
        app._update_run_tree_height()
        center_window(self.top)

    def close(self) -> None:
        self.app.model_dialog = None
        self.app.run_tree = None
        self.app.run_scroll = None
        self.app.profile_frame = None
        self.app.profile_line = None
        self.app.profile_buttons = None
        self.app.profile_label = None
        self.app.profile_combo = None
        self.app.profile_apply_button = None
        self.app.profile_keys_button = None
        self.app.profile_hint = None
        self.top.destroy()

class LauncherApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.state = load_state()

        # Initialize language from system locale so launcher and dashboard stay aligned.
        set_language(detect_system_language())

        self.root.title(t("window_title"))
        self.root.resizable(False, False)
        self.mp_ctx = mp.get_context("spawn")
        self.models: List[ModelOption] = []
        self.models_by_symbol: Dict[str, List[ModelOption]] = {}
        self.run_items: Dict[str, RunItem] = {}
        self.dashboard_thread: Optional[threading.Thread] = None
        self.dashboard_running = False
        self._dashboard_open_pending = False
        self.key_manager_dialog: Optional[KeyManagerDialog] = None
        self.model_dialog: Optional[ModelConfigDialog] = None
        self.mode_lock: Optional[str] = None
        self._mode_trace_guard = False

        self.mode_var = tk.StringVar(value=normalize_mode(self.state.get("mode", "signal")))
        self.symbol_var = tk.StringVar(value=str(self.state.get("selected_symbol", "")))
        self.model_var = tk.StringVar(value=str(self.state.get("selected_model", "")))
        self.open_browser_var = tk.BooleanVar(
            value=bool(self.state.get("open_dashboard", self.state.get("run_dashboard", True)))
        )
        notify_state = self.state.get("notifications", {}) if isinstance(self.state.get("notifications"), dict) else {}
        self.notify_enabled_var = tk.BooleanVar(value=bool(notify_state.get("enabled", False)))
        self.discord_var = tk.StringVar(value=str(notify_state.get("discord", "")))
        self.telegram_token_var = tk.StringVar(value=str(notify_state.get("telegram_token", "")))
        self.telegram_chat_id_var = tk.StringVar(value=str(notify_state.get("telegram_chat_id", "")))
        self.notify_system_var = tk.BooleanVar(value=bool(notify_state.get("notify_system", True)))
        self.notify_signals_var = tk.BooleanVar(value=bool(notify_state.get("notify_signals", True)))
        self.notify_direction_var = tk.BooleanVar(value=bool(notify_state.get("notify_direction", True)))
        self.testnet_var = tk.BooleanVar(value=bool(self.state.get("testnet", False)))
        self.profile_var = tk.StringVar(value=str(self.state.get("default_profile", "default")))
        self.status_var = tk.StringVar(value=t("status_idle"))
        self.section_font = ctk.CTkFont(size=12, weight="bold")
        self.desc_font = ctk.CTkFont(size=10)
        self._radio_theme = ctk.ThemeManager.theme.get("CTkRadioButton", {})
        self._label_theme = ctk.ThemeManager.theme.get("CTkLabel", {})
        self._mode_disabled_text = self._radio_theme.get("text_color_disabled", ["#9AA4B2", "#6B7280"])
        self._mode_disabled_border = ["#CBD5E1", "#4B5563"]
        self._mode_disabled_fill = ["#E2E8F0", "#3B4456"]
        self._status_default_text = self._label_theme.get("text_color", ["#2B2B2B", "#DCE4EE"])
        self.configure_models_desc: Optional[ctk.CTkLabel] = None

        self._build_ui()
        self.refresh_models(select_saved=True)
        self.refresh_key_profiles()
        self._bind_traces()
        self._apply_mode_state()
        self._load_run_list()
        self._lock_window_size()
        self._update_mode_lock()
        self._update_action_buttons()
        self._poll_processes()
        self._ensure_visible()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self) -> None:
        apply_treeview_style()
        main = ctk.CTkFrame(self.root, fg_color="transparent")
        main.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        def section(row: int, title: str, pady: tuple[int, int]) -> tuple[ctk.CTkFrame, ctk.CTkFrame]:
            frame = ctk.CTkFrame(main, corner_radius=12)
            frame.grid(row=row, column=0, sticky="ew", pady=pady)
            title_label = ctk.CTkLabel(frame, text=title, font=self.section_font)
            title_label.grid(row=0, column=0, sticky="w", padx=10, pady=(4, 2))
            body = ctk.CTkFrame(frame, fg_color="transparent")
            body.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 4))
            frame.columnconfigure(0, weight=1)
            body.columnconfigure(0, weight=1)
            return frame, body

        _, mode_body = section(0, t("section_mode"), (0, 4))
        mode_row = ctk.CTkFrame(mode_body, fg_color="transparent")
        mode_row.grid(row=0, column=0, sticky="ew")
        mode_row.columnconfigure(0, weight=1)
        mode_row.columnconfigure(1, weight=1)
        mode_left = ctk.CTkFrame(mode_row, fg_color="transparent")
        mode_left.grid(row=0, column=0, sticky="w")
        self.signal_label = ctk.CTkLabel(mode_left, text=t("mode_signal_only"))
        self.signal_label.grid(row=0, column=0, sticky="w")
        self.signal_radio = SoftRadioButton(
            mode_left,
            text="",
            variable=self.mode_var,
            value="signal",
            width=26,
            radiobutton_width=26,
            radiobutton_height=26,
            corner_radius=13,
            border_width_unchecked=2,
            border_width_checked=3,
        )
        self.signal_radio.grid(row=0, column=1, sticky="w", padx=(6, 0))
        mode_right = ctk.CTkFrame(mode_row, fg_color="transparent")
        mode_right.grid(row=0, column=1, sticky="e")
        self.auto_radio = SoftRadioButton(
            mode_right,
            text="",
            variable=self.mode_var,
            value="auto",
            width=26,
            radiobutton_width=26,
            radiobutton_height=26,
            corner_radius=13,
            border_width_unchecked=2,
            border_width_checked=3,
        )
        self.auto_radio.grid(row=0, column=0, sticky="w")
        self.auto_label = ctk.CTkLabel(mode_right, text=t("mode_automatic"))
        self.auto_label.grid(row=0, column=1, sticky="w", padx=(6, 0))
        self.signal_label.bind("<Button-1>", lambda _event: self.mode_var.set("signal"))
        self.auto_label.bind("<Button-1>", lambda _event: self.mode_var.set("auto"))

        _, model_body = section(1, t("section_models"), (0, 4))
        model_row = ctk.CTkFrame(model_body, fg_color="transparent")
        model_row.grid(row=0, column=0, sticky="ew")
        self.configure_models_button = ctk.CTkButton(
            model_row, text=t("btn_configure_models"), command=self.open_model_dialog
        )
        self.configure_models_button.grid(row=0, column=0, sticky="w")
        self.configure_models_desc = ctk.CTkLabel(
            model_row,
            text="",
            font=self.desc_font,
            wraplength=360,
            justify="left",
        )
        self.configure_models_desc.grid(row=0, column=1, sticky="w", padx=(10, 0))
        model_row.columnconfigure(1, weight=1)

        _, dash_body = section(2, t("section_dashboard"), (0, 4))
        self.open_browser_check = SoftCheckBox(
            dash_body,
            text=t("chk_start_dashboard"),
            variable=self.open_browser_var,
            checkbox_width=16,
            checkbox_height=16,
            corner_radius=7,
            border_width=1,
        )
        self.open_browser_check.grid(row=0, column=0, sticky="w")
        self.dashboard_desc = ctk.CTkLabel(
            dash_body,
            text=t("dashboard_desc"),
            wraplength=520,
            justify="left",
            font=self.desc_font,
        )
        self.dashboard_desc.grid(row=1, column=0, sticky="w", pady=(2, 0))

        _, notify_body = section(3, t("section_notifications"), (0, 4))
        self.notify_check = SoftCheckBox(
            notify_body,
            text=t("chk_start_notifications"),
            variable=self.notify_enabled_var,
            checkbox_width=16,
            checkbox_height=16,
            corner_radius=7,
            border_width=1,
        )
        self.notify_check.grid(row=0, column=0, sticky="w")
        self.notify_button = ctk.CTkButton(
            notify_body, text=t("btn_configure"), command=self.open_notify_dialog
        )
        self.notify_button.grid(row=0, column=1, sticky="e")
        self.notify_desc = ctk.CTkLabel(
            notify_body,
            text=t("notifications_desc"),
            wraplength=520,
            justify="left",
            font=self.desc_font,
        )
        self.notify_desc.grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 0))

        _, testnet_body = section(4, t("section_trading"), (0, 4))
        self.testnet_frame = ctk.CTkFrame(testnet_body, fg_color="transparent")
        self.testnet_frame.grid(row=0, column=0, sticky="w")
        self.testnet_check = SoftCheckBox(
            self.testnet_frame,
            text=t("chk_testnet"),
            variable=self.testnet_var,
            checkbox_width=16,
            checkbox_height=16,
            corner_radius=7,
            border_width=1,
        )
        self.testnet_check.grid(row=0, column=0, sticky="w")
        self.testnet_desc = ctk.CTkLabel(
            self.testnet_frame,
            text=t("testnet_desc"),
            font=self.desc_font,
            wraplength=520,
            justify="left",
        )
        self.testnet_desc.grid(row=1, column=0, sticky="w", pady=(2, 0))

        bottom_frame = ctk.CTkFrame(main, fg_color="transparent")
        bottom_frame.grid(row=5, column=0, sticky="ew", pady=(4, 0))
        self.requirement_label = ctk.CTkLabel(bottom_frame, text="")
        self.requirement_label.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 2))
        self.start_button = ctk.CTkButton(bottom_frame, text=t("btn_start"), command=self.start_selected)
        self.start_button.grid(row=1, column=0, sticky="w")
        self.stop_button = ctk.CTkButton(bottom_frame, text=t("btn_stop"), command=self.stop_all)
        self.stop_button.grid(row=1, column=1, sticky="w", padx=(8, 0))
        self.status_label = ctk.CTkLabel(bottom_frame, textvariable=self.status_var)
        self.status_label.grid(row=1, column=2, sticky="e", padx=(12, 0))
        self.legal_button = ctk.CTkButton(
            bottom_frame,
            text=t("btn_legal"),
            command=self._show_legal,
            width=70,
            height=26,
        )
        self.legal_button.grid(row=1, column=3, sticky="e", padx=(8, 0))
        bottom_frame.columnconfigure(2, weight=1)

        main.columnconfigure(0, weight=1)

    def _bind_traces(self) -> None:
        self.mode_var.trace_add("write", lambda *_: self._on_mode_change())
        self.open_browser_var.trace_add("write", lambda *_: self._apply_mode_state())
        self.notify_enabled_var.trace_add("write", lambda *_: self._apply_notify_state())
        self.status_var.trace_add("write", lambda *_: self._update_status_style())
        self._update_status_style()

    def _update_status_style(self) -> None:
        if not hasattr(self, "status_label"):
            return
        value = self.status_var.get().strip().lower()
        # Compare with both English and Spanish status values for styling
        running_vals = [t("status_running").lower(), "running", "ejecutando"]
        idle_vals = [t("status_idle").lower(), "idle", "inactivo"]
        stopped_vals = [t("status_stopped").lower(), "stopped", "detenido"]
        if value in running_vals:
            color = ("#1E7A3D", "#49D17D")
        elif value in idle_vals:
            color = ("#B58A00", "#F3C74B")
        elif value in stopped_vals:
            color = ("#7B8794", "#9AA4B2")
        else:
            color = self._status_default_text
        self.status_label.configure(text_color=color)

    def _show_legal(self) -> None:
        license_path = resource_path("LICENSE")
        license_text = str(license_path) if license_path.exists() else "LICENSE not found"
        text = t("legal_text", source_url=SOURCE_URL, license_path=license_text)
        messagebox.showinfo(t("legal_title"), text)

    def _ensure_visible(self) -> None:
        def activate() -> None:
            try:
                self.root.update_idletasks()
            except Exception:
                pass
            try:
                self.root.deiconify()
            except Exception:
                pass
            try:
                self.root.state("normal")
            except Exception:
                pass
            try:
                self.root.attributes("-topmost", True)
            except Exception:
                pass
            try:
                self.root.lift()
            except Exception:
                pass
            try:
                self.root.attributes("-topmost", False)
            except Exception:
                pass
            try:
                self.root.focus_force()
            except Exception:
                pass
            if os.name == "nt":
                try:
                    hwnd = self.root.winfo_id()
                    if hwnd:
                        ctypes.windll.user32.ShowWindow(hwnd, 9)
                        ctypes.windll.user32.SetForegroundWindow(hwnd)
                except Exception:
                    pass

        self.root.after(50, activate)
        self.root.after(300, activate)
        self.root.after(1200, activate)

    def refresh_models(self, select_saved: bool = False) -> None:
        self.models = discover_models()
        self.models_by_symbol = {}
        for option in self.models:
            self.models_by_symbol.setdefault(option.symbol, []).append(option)
        for options in self.models_by_symbol.values():
            options.sort(key=lambda item: str(item.path))
        symbols = sorted(self.models_by_symbol.keys(), key=str.lower)
        if select_saved:
            desired_symbol = str(self.state.get("selected_symbol") or self.symbol_var.get().strip())
        else:
            desired_symbol = self.symbol_var.get().strip()
        if desired_symbol not in symbols:
            desired_symbol = symbols[0] if symbols else ""
        self.symbol_var.set(desired_symbol)
        if select_saved:
            desired_model = str(self.state.get("selected_model") or self.model_var.get().strip())
            self.model_var.set(desired_model)

    def open_model_dialog(self) -> None:
        if self.model_dialog and self.model_dialog.top.winfo_exists():
            self.model_dialog.top.lift()
            self.model_dialog.top.focus_force()
            return
        self.refresh_models(select_saved=True)
        self.model_dialog = ModelConfigDialog(self)

    def open_add_model_dialog(self) -> None:
        if not self.models_by_symbol:
            messagebox.showinfo(t("add_model_title"), t("msg_no_models"), parent=self.root)
            return
        parent = self.model_dialog.top if self.model_dialog and self.model_dialog.top.winfo_exists() else self.root
        dialog = ctk.CTkToplevel(parent)
        dialog.title(t("add_model_title"))
        dialog.resizable(False, False)
        dialog.transient(parent)
        dialog.grab_set()

        container = ctk.CTkFrame(dialog)
        container.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)

        symbol_var = tk.StringVar()
        model_var = tk.StringVar()
        model_map: Dict[str, ModelOption] = {}

        ctk.CTkLabel(container, text=t("lbl_symbol")).grid(row=0, column=0, sticky="w")
        symbol_combo = ctk.CTkComboBox(
            container,
            variable=symbol_var,
            values=[],
            width=180,
            state="readonly",
        )
        symbol_combo.grid(row=0, column=1, sticky="w", padx=(8, 0))

        ctk.CTkLabel(container, text=t("lbl_model")).grid(row=1, column=0, sticky="w", pady=(6, 0))
        model_combo = ctk.CTkComboBox(
            container,
            variable=model_var,
            values=[],
            width=360,
            state="readonly",
        )
        model_combo.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        container.columnconfigure(1, weight=1)

        def update_models() -> None:
            symbol = symbol_var.get().strip()
            options = self.models_by_symbol.get(symbol, [])
            model_map.clear()
            model_values = []
            for option in options:
                display = display_model_choice(option, symbol)
                model_map[display] = option
                model_values.append(display)
            model_combo.configure(values=model_values)
            if not model_values:
                model_var.set("")
                model_combo.set("")
                return
            desired_model = self.model_var.get().strip()
            if desired_model:
                resolved = resolve_state_path(desired_model)
            else:
                resolved = None
            if resolved:
                for display, option in model_map.items():
                    if option.path.resolve() == resolved:
                        model_var.set(display)
                        model_combo.set(display)
                        return
            model_var.set(model_values[0])
            model_combo.set(model_values[0])

        def on_symbol_changed(_value: str = "") -> None:
            update_models()

        def add_selected() -> None:
            display = model_var.get().strip()
            option = model_map.get(display)
            if not option:
                messagebox.showwarning(t("add_model_title"), t("msg_select_model"), parent=dialog)
                return
            profile = self.profile_var.get() or "default"
            self._add_run_item(option, profile)
            self.symbol_var.set(symbol_var.get().strip())
            self.model_var.set(state_model_path(option.path))
            dialog.destroy()

        symbols = sorted(self.models_by_symbol.keys(), key=str.lower)
        symbol_combo.configure(values=symbols)
        desired_symbol = self.symbol_var.get().strip()
        if desired_symbol not in symbols:
            desired_symbol = symbols[0] if symbols else ""
        symbol_var.set(desired_symbol)
        symbol_combo.set(desired_symbol)
        update_models()
        symbol_combo.configure(command=on_symbol_changed)

        buttons = ctk.CTkFrame(container, fg_color="transparent")
        buttons.grid(row=2, column=0, columnspan=2, sticky="e", pady=(10, 0))
        ctk.CTkButton(buttons, text=t("btn_cancel"), command=dialog.destroy).grid(row=0, column=0, sticky="e")
        ctk.CTkButton(buttons, text=t("btn_add"), command=add_selected).grid(row=0, column=1, sticky="e", padx=(8, 0))

        center_window(dialog)
        dialog.wait_window(dialog)

    def open_notify_dialog(self) -> None:
        NotificationsDialog(
            self.root,
            self.discord_var,
            self.telegram_token_var,
            self.telegram_chat_id_var,
            self.notify_system_var,
            self.notify_signals_var,
            self.notify_direction_var,
        )

    def refresh_key_profiles(self) -> None:
        if not self._profile_frame_ready():
            return
        profiles = load_key_profiles()
        names = sorted(profiles.keys())
        if "default" not in names:
            names.insert(0, "default")
        self.profile_combo.configure(values=names)
        if self.profile_var.get() not in names and names:
            self.profile_var.set(names[0])
        if names:
            self.profile_combo.set(self.profile_var.get())

    def _apply_mode_state(self) -> None:
        mode = normalize_mode(self.mode_var.get())
        if self._profile_frame_ready():
            if mode == "signal":
                for child in self.profile_frame.winfo_children():
                    child.grid_remove()
            else:
                self.profile_line.grid(row=0, column=0, sticky="w")
                self.profile_buttons.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        if hasattr(self, "testnet_frame"):
            if mode == "auto":
                self.testnet_check.configure(state="normal")
                self.testnet_desc.configure(text_color=self._label_theme.get("text_color"))
            else:
                self.testnet_check.configure(state="disabled")
                self.testnet_desc.configure(text_color=self._mode_disabled_text)
        if self.configure_models_desc:
            if mode == "auto":
                self.configure_models_desc.configure(text=t("models_desc_auto"))
            else:
                self.configure_models_desc.configure(text=t("models_desc_signal"))
        self._apply_notify_state()

    def _on_mode_change(self) -> None:
        if self._mode_trace_guard:
            return
        requested = normalize_mode(self.mode_var.get())
        if self.mode_lock and requested != self.mode_lock:
            self._mode_trace_guard = True
            self.mode_var.set(self.mode_lock)
            self._mode_trace_guard = False
            messagebox.showinfo(t("launcher_title"), t("msg_stop_first"))
            self._apply_mode_state()
            return
        self._apply_mode_state()
        self._update_requirement_hint()

    def _apply_notify_state(self) -> None:
        enabled = self.notify_enabled_var.get()
        state = "normal" if enabled else "disabled"
        self.notify_button.configure(state=state)
        self._update_requirement_hint()

    def _update_requirement_hint(self) -> None:
        mode = normalize_mode(self.mode_var.get())
        needs_choice = mode == "signal" and not self.open_browser_var.get() and not self.notify_enabled_var.get()
        message = t("msg_select_option") if needs_choice else ""
        self.requirement_label.configure(text=message)

    def _update_run_tree_height(self) -> None:
        if not self._run_tree_ready():
            return
        count = len(self.run_tree.get_children())
        max_visible = 5
        visible = max(3, min(count, max_visible))
        self.run_tree.configure(height=visible)
        if getattr(self, "run_scroll", None) and self.run_scroll.winfo_exists():
            if count > max_visible:
                self.run_scroll.grid()
            else:
                self.run_scroll.grid_remove()

    def _any_running(self) -> bool:
        return any(item.process and item.process.is_alive() for item in self.run_items.values())

    def _update_action_buttons(self) -> None:
        running = self._any_running()
        if hasattr(self, "start_button"):
            self.start_button.configure(state="disabled" if running else "normal")
        if hasattr(self, "stop_button"):
            self.stop_button.configure(state="normal" if running else "disabled")

    def _run_tree_ready(self) -> bool:
        return bool(getattr(self, "run_tree", None) and self.run_tree.winfo_exists())

    def _profile_frame_ready(self) -> bool:
        return bool(getattr(self, "profile_frame", None) and self.profile_frame.winfo_exists())

    def _sync_run_tree(self) -> None:
        if not self._run_tree_ready():
            return
        for item_id in self.run_tree.get_children():
            self.run_tree.delete(item_id)
        for run_id, item in self.run_items.items():
            self.run_tree.insert(
                "",
                "end",
                iid=run_id,
                values=(item.option.symbol, display_model_name(item.option), item.profile),
            )
        self._update_run_tree_height()

    def _lock_window_size(self) -> None:
        current_mode = normalize_mode(self.mode_var.get())

        def measure(mode: str) -> tuple[int, int]:
            self._mode_trace_guard = True
            self.mode_var.set(mode)
            self._mode_trace_guard = False
            self._apply_mode_state()
            self.root.update_idletasks()
            return self.root.winfo_reqwidth(), self.root.winfo_reqheight()

        base_w, base_h = measure(current_mode)
        other_mode = "auto" if current_mode == "signal" else "signal"
        other_w, other_h = measure(other_mode)

        self._mode_trace_guard = True
        self.mode_var.set(current_mode)
        self._mode_trace_guard = False
        self._apply_mode_state()

        width = max(base_w, other_w)
        height = max(base_h, other_h)
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(width, height)

    def _apply_mode_lock_visuals(self) -> None:
        locked = self.mode_lock
        self._style_mode_option(self.signal_radio, self.signal_label, not locked or locked == "signal")
        self._style_mode_option(self.auto_radio, self.auto_label, not locked or locked == "auto")

    def _style_mode_option(self, radio: ctk.CTkRadioButton, label: ctk.CTkLabel, enabled: bool) -> None:
        if enabled:
            radio.configure(
                fg_color=self._radio_theme.get("fg_color"),
                border_color=self._radio_theme.get("border_color"),
                hover_color=self._radio_theme.get("hover_color"),
            )
            label.configure(text_color=self._label_theme.get("text_color"))
            return
        radio.configure(
            fg_color=self._mode_disabled_fill,
            border_color=self._mode_disabled_border,
            hover_color=self._mode_disabled_border,
        )
        label.configure(text_color=self._mode_disabled_text)

    def _update_mode_lock(self) -> None:
        running_modes = {
            item.mode
            for item in self.run_items.values()
            if item.process and item.process.is_alive()
        }
        if not running_modes:
            self.mode_lock = None
        elif len(running_modes) == 1:
            self.mode_lock = next(iter(running_modes))
        else:
            self.mode_lock = normalize_mode(self.mode_var.get())
        if self.mode_lock and self.mode_var.get() != self.mode_lock:
            self._mode_trace_guard = True
            self.mode_var.set(self.mode_lock)
            self._mode_trace_guard = False
            self._apply_mode_state()
        self._apply_mode_lock_visuals()

    def _terminate_item(self, item: RunItem) -> bool:
        if not item.process:
            item.status = "Stopped"
            return True
        if item.process.is_alive():
            item.process.terminate()
            item.process.join(timeout=2)
        if item.process.is_alive() and hasattr(item.process, "kill"):
            try:
                item.process.kill()
            except Exception:
                pass
            item.process.join(timeout=2)
        if item.process.is_alive():
            item.status = "Running"
            return False
        item.process = None
        item.status = "Stopped"
        return True

    def _load_run_list(self) -> None:
        saved = self.state.get("run_list")
        if not isinstance(saved, list):
            return
        for entry in saved:
            if not isinstance(entry, dict):
                continue
            path = resolve_state_path(str(entry.get("model_path", "")))
            if not path:
                continue
            option = next((m for m in self.models if m.path.resolve() == path), None)
            if not option:
                continue
            profile = str(entry.get("profile") or self.profile_var.get() or "default")
            self._add_run_item(option, profile)
        self._sync_run_tree()

    def _add_run_item(self, option: ModelOption, profile: str) -> None:
        run_id = state_model_path(option.path)
        if run_id in self.run_items:
            return
        existing_keys = {item.metrics_key for item in self.run_items.values()}
        base_key = model_key(option)
        metrics_key = base_key
        suffix = 2
        while metrics_key in existing_keys:
            metrics_key = f"{base_key}_{suffix}"
            suffix += 1
        item = RunItem(run_id=run_id, option=option, profile=profile, metrics_key=metrics_key)
        self.run_items[run_id] = item
        if self._run_tree_ready():
            self.run_tree.insert(
                "",
                "end",
                iid=run_id,
                values=(option.symbol, display_model_name(option), profile),
            )
            self._update_run_tree_height()

    def remove_from_run_list(self) -> None:
        if not self._run_tree_ready():
            return
        for run_id in self.run_tree.selection():
            item = self.run_items.get(run_id)
            if not item:
                continue
            if item.process and item.process.is_alive():
                messagebox.showwarning(t("launcher_title"), t("msg_stop_before_remove"))
                return
        for run_id in self.run_tree.selection():
            self.run_tree.delete(run_id)
            if run_id in self.run_items:
                del self.run_items[run_id]
        self._update_run_tree_height()

    def apply_profile(self) -> None:
        profile = self.profile_var.get().strip() or "default"
        if not self._run_tree_ready():
            return
        for run_id in self.run_tree.selection():
            item = self.run_items.get(run_id)
            if not item:
                continue
            item.profile = profile
            self.run_tree.set(run_id, "profile", profile)

    def open_key_manager(self) -> None:
        if self.key_manager_dialog and self.key_manager_dialog.top.winfo_exists():
            self.key_manager_dialog.top.lift()
            self.key_manager_dialog.top.focus_force()
            return
        self.key_manager_dialog = KeyManagerDialog(self.root, self._after_keys_saved)

    def _after_keys_saved(self, profile: str) -> None:
        self.refresh_key_profiles()
        if profile:
            self.profile_var.set(profile)

    def _ensure_dashboard(self, mode: str, notifications: Dict, open_browser: bool) -> None:
        if not self.dashboard_running:
            dash_args = build_dashboard_args(notifications)
            debug_log(f"Dashboard thread start args={dash_args} open_browser={open_browser}")
            self.dashboard_thread = threading.Thread(target=run_dashboard, args=(dash_args,), daemon=True)
            self.dashboard_thread.start()
            self.dashboard_running = True
        if open_browser:
            self._open_dashboard_when_ready(mode)

    def _open_dashboard_when_ready(self, mode: str) -> None:
        if self._dashboard_open_pending:
            return
        self._dashboard_open_pending = True

        def http_ready() -> bool:
            try:
                view = "traders" if mode == "signal" else "account"
                path = f"/?mode={mode}&view={view}"
                conn = HTTPConnection("127.0.0.1", DASHBOARD_PORT, timeout=1.0)
                conn.request("GET", path, headers={"Connection": "close"})
                resp = conn.getresponse()
                body = resp.read()
                conn.close()
                if resp.status != 200:
                    return False
                if not body:
                    return False
                content_type = resp.getheader("Content-Type", "")
                return "text/html" in content_type.lower()
            except Exception:
                return False

        def check(attempts: int) -> None:
            if http_ready():
                debug_log(f"Dashboard ready on port {DASHBOARD_PORT}, opening browser.")
                if not open_dashboard(mode):
                    messagebox.showinfo(
                        t("dashboard_title"),
                        t("msg_dashboard_running", port=DASHBOARD_PORT),
                    )
                self._dashboard_open_pending = False
                return
            if attempts <= 0:
                self._dashboard_open_pending = False
                debug_log(f"Dashboard not ready on port {DASHBOARD_PORT} after retries.")
                messagebox.showwarning(
                    t("dashboard_title"),
                    t("msg_dashboard_failed", port=DASHBOARD_PORT, path=DATA_DIR),
                )
                return
            self.root.after(500, lambda: check(attempts - 1))

        self.root.after(200, lambda: check(12))

    def _start_run_item(self, item: RunItem, history_writer: bool, mode: str) -> None:
        args = ["--model-dir", str(item.option.path), "--symbol", item.option.symbol]
        ensure_data_dir()
        metrics_path = DATA_DIR / f"live_metrics_{item.metrics_key}.jsonl"
        args.extend(["--metrics-log-path", str(metrics_path)])
        if not history_writer:
            args.append("--no-history-writer")
        if mode == "signal":
            args.append("--signal-only")
        else:
            args.extend(["--keys-file", str(KEYS_PATH), "--keys-profile", item.profile or "default"])
            if self.testnet_var.get():
                args.append("--testnet")
        try:
            proc = self.mp_ctx.Process(target=run_live_trading, args=(args,))
            proc.daemon = True
            proc.start()
            item.process = proc
            item.status = "Running"
            item.mode = mode
            debug_log(f"Process started for {item.run_id} pid={proc.pid}")
            if self._run_tree_ready():
                self.run_tree.set(item.run_id, "profile", item.profile)
        except Exception as exc:
            debug_log(f"Failed to start {item.run_id}: {exc}")
            raise

    def start_selected(self) -> None:
        mode = normalize_mode(self.mode_var.get())
        open_browser = bool(self.open_browser_var.get())
        notifications = self._collect_notifications()

        if self.mode_lock and mode != self.mode_lock:
            messagebox.showinfo(t("launcher_title"), t("msg_stop_first"))
            return

        if mode == "signal" and not open_browser and not notifications["enabled"]:
            messagebox.showerror(t("launcher_title"), t("msg_signal_needs_output"))
            return

        if notifications["enabled"]:
            if not notifications.get("discord") and not (
                notifications.get("telegram_token") and notifications.get("telegram_chat_id")
            ):
                if mode == "signal" and not open_browser:
                    messagebox.showerror(t("notifications_title"), t("msg_add_discord_telegram"))
                    return
                messagebox.showwarning(t("notifications_title"), t("msg_no_notification_info"))
                notifications["enabled"] = False
            elif notifications.get("notify_system") and not notifications.get("discord"):
                messagebox.showwarning(t("notifications_title"), t("msg_system_needs_discord"))

        if mode == "auto" and not KEYS_PATH.exists():
            messagebox.showwarning(t("api_keys_title"), t("msg_keys_not_found"))

        selection = list(self.run_tree.selection()) if self._run_tree_ready() else []
        targets = selection if selection else list(self.run_items.keys())
        if not targets:
            messagebox.showerror(t("launcher_title"), t("msg_add_model_first"))
            debug_log("Start blocked: no models in run list.")
            return

        running_symbols = {
            item.option.symbol
            for item in self.run_items.values()
            if item.process and item.process.is_alive()
        }
        started = 0
        for run_id in targets:
            item = self.run_items.get(run_id)
            if not item:
                continue
            if item.process and item.process.is_alive():
                continue
            history_writer = item.option.symbol not in running_symbols
            debug_log(f"Starting {item.run_id} with model {item.option.path}")
            self._start_run_item(item, history_writer, mode)
            running_symbols.add(item.option.symbol)
            started += 1

        has_running = any(
            item.process and item.process.is_alive()
            for item in self.run_items.values()
        )
        if started:
            self.status_var.set(t("status_running"))
            save_state(self.collect_state())
            self.mode_lock = mode
            self._apply_mode_lock_visuals()
        elif has_running:
            self.status_var.set(t("status_running"))
        else:
            messagebox.showerror(t("launcher_title"), t("msg_start_failed", path=DATA_DIR))
            debug_log("No running models after start attempt.")

        if (open_browser or notifications["enabled"]) and (started or has_running):
            self._ensure_dashboard(mode, notifications, open_browser)
        self._update_action_buttons()

    def stop_selected(self) -> None:
        if not self._run_tree_ready():
            return
        stopped = 0
        for run_id in self.run_tree.selection():
            item = self.run_items.get(run_id)
            if not item or not item.process:
                continue
            if self._terminate_item(item):
                if self._run_tree_ready():
                    self.run_tree.set(run_id, "profile", item.profile)
                stopped += 1
        if stopped:
            self.status_var.set(t("status_stopped"))
        self._update_mode_lock()
        self._update_action_buttons()

    def stop_all(self) -> None:
        for item in self.run_items.values():
            if item.process:
                if self._terminate_item(item):
                    if self._run_tree_ready():
                        self.run_tree.set(item.run_id, "profile", item.profile)
        self.status_var.set(t("status_stopped"))
        self._update_mode_lock()
        self._update_action_buttons()

    def _poll_processes(self) -> None:
        changed = False
        for item in self.run_items.values():
            if item.process and item.status == "Running" and not item.process.is_alive():
                item.status = "Exited"
                item.process = None
                if self._run_tree_ready():
                    self.run_tree.set(item.run_id, "profile", item.profile)
                changed = True
        if changed:
            self.status_var.set(t("status_idle"))
        self._update_mode_lock()
        self._update_action_buttons()
        self.root.after(1000, self._poll_processes)

    def _collect_notifications(self) -> Dict:
        return {
            "enabled": bool(self.notify_enabled_var.get()),
            "discord": self.discord_var.get().strip(),
            "telegram_token": self.telegram_token_var.get().strip(),
            "telegram_chat_id": self.telegram_chat_id_var.get().strip(),
            "notify_system": bool(self.notify_system_var.get()),
            "notify_signals": bool(self.notify_signals_var.get()),
            "notify_direction": bool(self.notify_direction_var.get()),
        }

    def collect_state(self) -> Dict:
        run_list = []
        for item in self.run_items.values():
            run_list.append({
                "model_path": state_model_path(item.option.path),
                "profile": item.profile,
            })
        selected_model = self.model_var.get().strip() or None
        return {
            "mode": normalize_mode(self.mode_var.get()),
            "open_dashboard": bool(self.open_browser_var.get()),
            "notifications": self._collect_notifications(),
            "testnet": bool(self.testnet_var.get()),
            "default_profile": self.profile_var.get().strip() or "default",
            "run_list": run_list,
            "selected_symbol": self.symbol_var.get().strip(),
            "selected_model": selected_model,
            "language": get_language(),
        }

    def on_close(self) -> None:
        save_state(self.collect_state())
        self.stop_all()
        self.root.destroy()

def main() -> None:
    ctk.set_appearance_mode("Light")
    theme_path = resource_path("ui_theme.json")
    if theme_path.exists():
        ctk.set_default_color_theme(str(theme_path))
    else:
        ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    LauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    mp.freeze_support()
    main()
