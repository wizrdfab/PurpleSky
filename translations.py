"""
PurpleSky Launcher translations module.
Provides i18n support with auto-detection from OS locale.
"""

import locale
import os
from typing import Dict, Optional

# Translation dictionaries
TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "en": {
        # Window titles
        "window_title": "PurpleSky launcher - Bybit",
        "api_keys_title": "API Keys",
        "notifications_title": "Notifications",
        "configure_models_title": "Configure models",
        "add_model_title": "Add model",
        "legal_title": "Legal",
        "dashboard_title": "Dashboard",
        "launcher_title": "Launcher",

        # Section titles
        "section_mode": "Mode",
        "section_models": "Models",
        "section_dashboard": "Dashboard",
        "section_notifications": "Notifications",
        "section_trading": "Trading",

        # Mode options
        "mode_signal_only": "Signal only",
        "mode_automatic": "Automatic trading",

        # Models section
        "btn_configure_models": "Configure models",
        "models_desc_auto": "Configure models for trading.\nConsider that each model will represent a trader.",
        "models_desc_signal": "Configure models for signals.\nEach model will only provide signals.",
        "lbl_run_list": "Run list",
        "col_symbol": "Symbol",
        "col_model": "Model",
        "col_profile": "Profile",
        "btn_remove": "Remove",
        "btn_add_model": "Add model",
        "lbl_api_profile": "API profile",
        "btn_apply_selected": "Apply to selected",
        "profile_hint": "Assign an API profile to a model. In case two models share symbol, make sure they use profiles from different accounts/sub-accounts.",
        "btn_add_manage_profiles": "Add/Manage profiles",

        # Dashboard section
        "chk_start_dashboard": "Start with dashboard",
        "dashboard_desc": "Dashboard shows the model confidence levels and a chart of the symbol with indicators based on the model predictions.",

        # Notifications section
        "chk_start_notifications": "Start with notifications",
        "btn_configure": "Configure...",
        "notifications_desc": "Notifications for Telegram & Discord",
        "lbl_discord_webhook": "Discord webhook",
        "lbl_telegram_token": "Telegram bot token",
        "lbl_telegram_chat_id": "Telegram chat id",
        "chk_system": "System",
        "chk_signals": "Signals",
        "chk_direction": "Direction",

        # Trading section
        "chk_testnet": "Testnet",
        "testnet_desc": "Launch the models on the testnet. Consider that results from testnet may differ to live trading due to liquidity/price action.",

        # API Keys dialog
        "lbl_profile_name": "Profile name",
        "profile_name_hint": "Type a name to create a new profile",
        "lbl_api_key": "API key",
        "lbl_api_secret": "API secret",
        "btn_delete": "Delete",
        "btn_save": "Save",
        "btn_close": "Close",

        # Add model dialog
        "lbl_symbol": "Symbol",
        "lbl_model": "Model",
        "btn_cancel": "Cancel",
        "btn_add": "Add",

        # Action buttons
        "btn_start": "Start",
        "btn_stop": "Stop",
        "btn_legal": "Legal",

        # Status messages
        "status_idle": "Idle",
        "status_running": "Running",
        "status_stopped": "Stopped",
        "status_exited": "Exited",

        # Error/Warning messages
        "msg_select_option": "Please select at least one option",
        "msg_stop_first": "Please stop all running models first.",
        "msg_profile_required": "Profile name is required.",
        "msg_keys_required": "API key and secret are required.",
        "msg_profile_saved": "Profile saved.",
        "msg_select_profile_delete": "Select a profile to delete.",
        "msg_profile_not_found": "Profile not found.",
        "msg_delete_profile_confirm": "Delete profile '{name}'?",
        "msg_no_models": "No models found. Click Refresh.",
        "msg_select_model": "Select a model to add.",
        "msg_stop_before_remove": "Stop running models before removing them.",
        "msg_signal_needs_output": "Signal mode needs dashboard or notifications. Enable at least one.",
        "msg_add_discord_telegram": "Add Discord/Telegram or enable the dashboard in signal mode.",
        "msg_no_notification_info": "No Discord or Telegram info provided. Notifications will be disabled.",
        "msg_system_needs_discord": "System alerts require Discord. Only signals/direction will be used.",
        "msg_keys_not_found": "key_profiles.json not found. Live trading will switch to dry-run mode.",
        "msg_add_model_first": "Add at least one model to the run list.",
        "msg_start_failed": "Models failed to start. Check {path}\\live_trading_error.log.",
        "msg_dashboard_running": "Dashboard is running at http://127.0.0.1:{port}/",
        "msg_dashboard_failed": "Dashboard did not start on port {port}. Check firewall settings or logs in {path}.",

        # Legal text
        "legal_text": """PurpleSky launcher - Bybit
Copyright (C) 2026 wizrdfab
Licensed under the GNU AGPLv3.
This program is distributed WITHOUT ANY WARRANTY.
Source: {source_url}
License: {license_path}""",

        # Dashboard translations
        "dash_title": "Live Trading Dashboard",
        "dash_waiting": "Waiting for data...",
        "dash_offline": "Offline",
        "dash_warming_up": "WARMING UP",
        "dash_trading_enabled": "Trading Enabled",
        "dash_trading_paused": "Trading Paused",
        "dash_startup": "Startup",
        "dash_uptime": "Uptime",

        # Dashboard cards
        "dash_equity_pnl": "Equity and PnL",
        "dash_daily_pnl": "Daily PnL",
        "dash_unrealized": "Unrealized",
        "dash_account_balance": "Account Balance",
        "dash_available": "Available",
        "dash_profiles": "Profiles",
        "dash_total_return": "Total Return",
        "dash_max_dd": "Max DD",
        "dash_volatility": "Volatility",
        "dash_stability": "Stability",
        "dash_smoothness": "Smoothness",
        "dash_trend_day": "Trend / Day",
        "dash_position": "Position",
        "dash_size": "Size",
        "dash_entry": "Entry",
        "dash_mark": "Mark",
        "dash_tp_sl": "TP / SL",
        "dash_health": "Health",
        "dash_sentiment": "Sentiment",
        "dash_regime": "Regime",
        "dash_trade_enabled": "Trade Enabled",
        "dash_drift_alerts": "Drift Alerts",
        "dash_data_quality": "Data Quality",
        "dash_macro_pct": "Macro %",
        "dash_ob_density": "OB Density",
        "dash_trade_continuity": "Trade Continuity",
        "dash_lag_trade_bar": "Lag Trade / Bar",
        "dash_latency": "Latency",
        "dash_ws_trade": "WS Trade",
        "dash_ws_ob": "WS OB",
        "dash_ob_lag": "OB Lag",
        "dash_orders": "Orders",
        "dash_last_reconcile": "Last Reconcile",
        "dash_source": "Source",
        "dash_protective": "Protective",
        "dash_errors": "Errors",
        "dash_last_runtime": "Last Runtime",
        "dash_last_api": "Last API",

        # Extra cards (injected via JS)
        "dash_market": "Market",
        "dash_position_mode": "Position Mode",
        "dash_resting_orders": "Resting Orders",
        "dash_limits": "Limits",
        "dash_performance": "Performance",
        "dash_trades": "Trades",
        "dash_sortino": "Sortino",
        "dash_profit_factor": "Profit Factor",
        "dash_avg_trade": "Avg Trade",
        "dash_iceberg": "Iceberg",
        "dash_entry_buy": "Entry Buy",
        "dash_entry_sell": "Entry Sell",
        "dash_tp_long": "TP Long",
        "dash_tp_short": "TP Short",
        "dash_controls": "Controls",
        "dash_cancel_buy": "Cancel Buy",
        "dash_cancel_sell": "Cancel Sell",
        "dash_cancel_tp_long": "Cancel TP Long",
        "dash_cancel_tp_short": "Cancel TP Short",
        "dash_iceberg_chart": "Iceberg Chart",
        "dash_chart_failed": "Chart library failed to load.",
        "dash_total_balance": "Total Balance",
        "dash_unified_account": "Unified Account",
        "dash_funding_account": "Funding Account",
        "dash_balance_curve": "Balance Curve (Raw + Flow-Adjusted)",
        "dash_curve_diagnostics": "Curve Diagnostics",
        "dash_model": "Model",
        "dash_signal": "Signal",
        "dash_features": "Features",
        "dash_signal_stream": "Signal Stream",
        "dash_share": "Share",
        "dash_updated": "Updated",
        "dash_flow_events": "Flow Events",
        "dash_raw_return": "Raw Return",
        "dash_points": "Points",
        "dash_pred_long": "Pred Long",
        "dash_pred_short": "Pred Short",
        "dash_dir_long": "Dir Long",
        "dash_dir_short": "Dir Short",
        "dash_threshold": "Threshold",
        "dash_pred_bar": "Pred Bar",
        "dash_dir_threshold": "Dir Threshold",
        "dash_aggressive": "Aggressive",
        "dash_dir_bar": "Dir Bar",
        "dash_model_path": "Model Path",
        "dash_keys_profile": "Keys Profile",
        "dash_side": "Side",
        "dash_entry_price": "Entry Price",
        "dash_tp_price": "TP Price",
        "dash_sl_price": "SL Price",
        "dash_time": "Time",
        "dash_traders": "Traders",
        "dash_active_signals": "Active Signals",
        "dash_signals": "Signals",
        "dash_direction": "Direction",
        "dash_levels": "Levels",
        "dash_sound_on": "Sound On",
        "dash_sound_off": "Sound Off",
        "dash_all": "All",
        "dash_split_scale": "Split Scale",
        "dash_shared_scale": "Shared Scale",
        "dash_raw_curve": "Raw Curve",
        "dash_flow_adjusted": "Flow-Adjusted",
        "dash_save": "Save",
        "dash_reset": "Reset",
        "dash_signal_mode": "Signal Mode",
        "dash_automated_mode": "Automated Mode",
        "dash_show_all": "Show All",
        "dash_no_data_loaded": "No data loaded",
        "dash_flow_note": "Flow-adjusted removes large deposits/withdrawals.",
        "dash_no_series_visible": "No series visible",
        "dash_models_count": "{count} models",
        "dash_active_models": "Active models: {count}",
        "dash_updated_at": "Updated: {time}",
        "dash_selected": "Selected",
        "dash_last": "Last",
        "dash_signal_only": "Signal Only",
        "dash_badge_signal": "SIGNAL",
        "dash_badge_on": "ON",
        "dash_badge_off": "OFF",
        "dash_long": "LONG",
        "dash_short": "SHORT",
        "dash_dual": "DUAL",
        "dash_no_signal": "No Signal",
        "dash_signal_long": "Signal Long",
        "dash_signal_short": "Signal Short",
        "dash_funding_error": "Funding Error",
        "dash_live": "Live",
        "dash_no_data": "No Data",
        "dash_lag_trade": "Lag Trade",
        "dash_last_update": "Last Update",
        "dash_tp": "TP",
        "dash_sl": "SL",
        "dash_pred": "Pred",
        "dash_flow_threshold_note": "Flow threshold: {pct}% (large deposits/withdrawals removed).",
        "dash_total_unified_funding": "Total (Unified + Funding): {total} | Updated: {updated}",
        "dash_funding": "Funding",
        "dash_ok": "OK",
        "dash_raw": "Raw",
        "dash_adjusted": "Adj",
        "dash_direction_short": "Dir",
        "dash_aggressive_short": "Agg",
        "dash_showing": "Showing",
        "dash_signal_label": "signal",
        "dash_price": "Price",
        "dash_volume": "Volume",
        "dash_chart": "Chart",
        "dash_range": "Range",
        "dash_bars": "Bars",
        "dash_interval": "Interval",
        "dash_metrics_feed": "Metrics feed",
        "dash_no_chart_data": "No chart data yet.",
        "dash_no_data_range": "No data in selected range.",
        "dash_buy_clip": "Buy Clip",
        "dash_sell_clip": "Sell Clip",
        "dash_entry_tp": "Entry TP",
        "dash_entry_sl": "Entry SL",
        "dash_tp_clip": "TP Clip",
        "dash_sl_long": "SL Long",
        "dash_sl_short": "SL Short",
        "dash_paused": "Paused",
        "dash_posted": "Posted",
        "dash_ready": "Ready",
        "dash_waiting_state": "Waiting",
        "dash_idle": "Idle",
        "dash_fill": "Fill",
        "dash_target_short": "T",
        "dash_remaining_short": "Rem",
        "dash_clip": "Clip",
        "dash_exchange_short": "exch",
        "dash_na": "n/a",
        "dash_donate_referral": "Donate & Referral link",
        "dash_donate_note": "Contributions support me directly and help improve this tool.",
        "dash_bitcoin": "Bitcoin (BTC)",
        "dash_ethereum": "Ethereum (ETH)",
        "dash_referral_link": "Bybit referral link",
        "dash_close": "Close",
        "dash_flat": "Flat",
        "dash_waiting_balance": "Waiting for balance updates...",
        "dash_yes": "Yes",
        "dash_no": "No",
        "dash_none": "None",
    },

    "es": {
        # Window titles
        "window_title": "PurpleSky lanzador - Bybit",
        "api_keys_title": "Claves API",
        "notifications_title": "Notificaciones",
        "configure_models_title": "Configurar modelos",
        "add_model_title": "Agregar modelo",
        "legal_title": "Legal",
        "dashboard_title": "Panel",
        "launcher_title": "Lanzador",

        # Section titles
        "section_mode": "Modo",
        "section_models": "Modelos",
        "section_dashboard": "Panel",
        "section_notifications": "Notificaciones",
        "section_trading": "Trading",

        # Mode options
        "mode_signal_only": "Solo señales",
        "mode_automatic": "Trading automático",

        # Models section
        "btn_configure_models": "Configurar modelos",
        "models_desc_auto": "Configura modelos para trading.\nCada modelo representará un trader.",
        "models_desc_signal": "Configura modelos para señales.\nCada modelo solo proporcionará señales.",
        "lbl_run_list": "Lista de ejecución",
        "col_symbol": "Símbolo",
        "col_model": "Modelo",
        "col_profile": "Perfil",
        "btn_remove": "Eliminar",
        "btn_add_model": "Agregar modelo",
        "lbl_api_profile": "Perfil API",
        "btn_apply_selected": "Aplicar a selección",
        "profile_hint": "Asigna un perfil API a un modelo. Si dos modelos comparten símbolo, asegúrate de que usen perfiles de diferentes cuentas/subcuentas.",
        "btn_add_manage_profiles": "Agregar/Gestionar perfiles",

        # Dashboard section
        "chk_start_dashboard": "Iniciar con panel",
        "dashboard_desc": "El panel muestra los niveles de confianza del modelo y un gráfico del símbolo con indicadores basados en las predicciones del modelo.",

        # Notifications section
        "chk_start_notifications": "Iniciar con notificaciones",
        "btn_configure": "Configurar...",
        "notifications_desc": "Notificaciones para Telegram y Discord",
        "lbl_discord_webhook": "Webhook de Discord",
        "lbl_telegram_token": "Token del bot de Telegram",
        "lbl_telegram_chat_id": "Chat ID de Telegram",
        "chk_system": "Sistema",
        "chk_signals": "Señales",
        "chk_direction": "Dirección",

        # Trading section
        "chk_testnet": "Testnet",
        "testnet_desc": "Ejecuta los modelos en testnet. Ten en cuenta que los resultados pueden diferir del trading real debido a la liquidez/acción del precio.",

        # API Keys dialog
        "lbl_profile_name": "Nombre del perfil",
        "profile_name_hint": "Escribe un nombre para crear un perfil nuevo",
        "lbl_api_key": "Clave API",
        "lbl_api_secret": "Secreto API",
        "btn_delete": "Eliminar",
        "btn_save": "Guardar",
        "btn_close": "Cerrar",

        # Add model dialog
        "lbl_symbol": "Símbolo",
        "lbl_model": "Modelo",
        "btn_cancel": "Cancelar",
        "btn_add": "Agregar",

        # Action buttons
        "btn_start": "Iniciar",
        "btn_stop": "Detener",
        "btn_legal": "Legal",

        # Status messages
        "status_idle": "Inactivo",
        "status_running": "Ejecutando",
        "status_stopped": "Detenido",
        "status_exited": "Terminado",

        # Error/Warning messages
        "msg_select_option": "Por favor selecciona al menos una opción",
        "msg_stop_first": "Por favor detén todos los modelos primero.",
        "msg_profile_required": "El nombre del perfil es requerido.",
        "msg_keys_required": "La clave API y el secreto son requeridos.",
        "msg_profile_saved": "Perfil guardado.",
        "msg_select_profile_delete": "Selecciona un perfil para eliminar.",
        "msg_profile_not_found": "Perfil no encontrado.",
        "msg_delete_profile_confirm": "¿Eliminar perfil '{name}'?",
        "msg_no_models": "No se encontraron modelos. Haz clic en Actualizar.",
        "msg_select_model": "Selecciona un modelo para agregar.",
        "msg_stop_before_remove": "Detén los modelos en ejecución antes de eliminarlos.",
        "msg_signal_needs_output": "El modo señal necesita panel o notificaciones. Habilita al menos uno.",
        "msg_add_discord_telegram": "Agrega Discord/Telegram o habilita el panel en modo señal.",
        "msg_no_notification_info": "No se proporcionó información de Discord o Telegram. Las notificaciones serán deshabilitadas.",
        "msg_system_needs_discord": "Las alertas del sistema requieren Discord. Solo se usarán señales/dirección.",
        "msg_keys_not_found": "key_profiles.json no encontrado. El trading en vivo cambiará a modo simulado.",
        "msg_add_model_first": "Agrega al menos un modelo a la lista de ejecución.",
        "msg_start_failed": "Los modelos fallaron al iniciar. Revisa {path}\\live_trading_error.log.",
        "msg_dashboard_running": "El panel está ejecutándose en http://127.0.0.1:{port}/",
        "msg_dashboard_failed": "El panel no inició en el puerto {port}. Revisa la configuración del firewall o los logs en {path}.",

        # Legal text
        "legal_text": """PurpleSky lanzador - Bybit
Copyright (C) 2026 wizrdfab
Licenciado bajo GNU AGPLv3.
Este programa se distribuye SIN NINGUNA GARANTÍA.
Código fuente: {source_url}
Licencia: {license_path}""",

        # Dashboard translations
        "dash_title": "Panel de Trading en Vivo",
        "dash_waiting": "Esperando datos...",
        "dash_offline": "Desconectado",
        "dash_warming_up": "CALENTANDO",
        "dash_trading_enabled": "Trading Activo",
        "dash_trading_paused": "Trading Pausado",
        "dash_startup": "Inicio",
        "dash_uptime": "Tiempo activo",

        # Dashboard cards
        "dash_equity_pnl": "Equity y PnL",
        "dash_daily_pnl": "PnL Diario",
        "dash_unrealized": "No Realizado",
        "dash_account_balance": "Balance de Cuenta",
        "dash_available": "Disponible",
        "dash_profiles": "Perfiles",
        "dash_total_return": "Retorno Total",
        "dash_max_dd": "DD Máximo",
        "dash_volatility": "Volatilidad",
        "dash_stability": "Estabilidad",
        "dash_smoothness": "Suavidad",
        "dash_trend_day": "Tendencia / Día",
        "dash_position": "Posición",
        "dash_size": "Tamaño",
        "dash_entry": "Entrada",
        "dash_mark": "Marca",
        "dash_tp_sl": "TP / SL",
        "dash_health": "Salud",
        "dash_sentiment": "Sentimiento",
        "dash_regime": "Régimen",
        "dash_trade_enabled": "Trading Habilitado",
        "dash_drift_alerts": "Alertas de Deriva",
        "dash_data_quality": "Calidad de Datos",
        "dash_macro_pct": "Macro %",
        "dash_ob_density": "Densidad OB",
        "dash_trade_continuity": "Continuidad Trades",
        "dash_lag_trade_bar": "Lag Trade / Barra",
        "dash_latency": "Latencia",
        "dash_ws_trade": "WS Trade",
        "dash_ws_ob": "WS OB",
        "dash_ob_lag": "Lag OB",
        "dash_orders": "Órdenes",
        "dash_last_reconcile": "Última Reconciliación",
        "dash_source": "Fuente",
        "dash_protective": "Protectivas",
        "dash_errors": "Errores",
        "dash_last_runtime": "Último Runtime",
        "dash_last_api": "Última API",

        # Extra cards (injected via JS)
        "dash_market": "Mercado",
        "dash_position_mode": "Modo Posición",
        "dash_resting_orders": "Órdenes en Espera",
        "dash_limits": "Límites",
        "dash_performance": "Rendimiento",
        "dash_trades": "Trades",
        "dash_sortino": "Sortino",
        "dash_profit_factor": "Factor de Beneficio",
        "dash_avg_trade": "Trade Promedio",
        "dash_iceberg": "Iceberg",
        "dash_entry_buy": "Entrada Compra",
        "dash_entry_sell": "Entrada Venta",
        "dash_tp_long": "TP Long",
        "dash_tp_short": "TP Short",
        "dash_controls": "Controles",
        "dash_cancel_buy": "Cancelar Compra",
        "dash_cancel_sell": "Cancelar Venta",
        "dash_cancel_tp_long": "Cancelar TP Long",
        "dash_cancel_tp_short": "Cancelar TP Short",
        "dash_iceberg_chart": "Gráfico Iceberg",
        "dash_chart_failed": "Error al cargar librería de gráficos.",
        "dash_total_balance": "Balance total",
        "dash_unified_account": "Cuenta unificada",
        "dash_funding_account": "Cuenta de financiación",
        "dash_balance_curve": "Curva de balance (bruta + ajustada por flujo)",
        "dash_curve_diagnostics": "Diagnóstico de curva",
        "dash_model": "Modelo",
        "dash_signal": "Señal",
        "dash_features": "Características",
        "dash_signal_stream": "Flujo de señales",
        "dash_share": "Participación",
        "dash_updated": "Actualizado",
        "dash_flow_events": "Eventos de flujo",
        "dash_raw_return": "Retorno bruto",
        "dash_points": "Puntos",
        "dash_pred_long": "Pred Long",
        "dash_pred_short": "Pred Short",
        "dash_dir_long": "Dir Long",
        "dash_dir_short": "Dir Short",
        "dash_threshold": "Umbral",
        "dash_pred_bar": "Barra pred.",
        "dash_dir_threshold": "Umbral dir.",
        "dash_aggressive": "Agresivo",
        "dash_dir_bar": "Barra dir.",
        "dash_model_path": "Ruta del modelo",
        "dash_keys_profile": "Perfil de claves",
        "dash_side": "Lado",
        "dash_entry_price": "Precio de entrada",
        "dash_tp_price": "Precio TP",
        "dash_sl_price": "Precio SL",
        "dash_time": "Hora",
        "dash_traders": "Traders",
        "dash_active_signals": "Señales activas",
        "dash_signals": "Señales",
        "dash_direction": "Dirección",
        "dash_levels": "Niveles",
        "dash_sound_on": "Sonido activado",
        "dash_sound_off": "Sonido desactivado",
        "dash_all": "Todo",
        "dash_split_scale": "Escala separada",
        "dash_shared_scale": "Escala compartida",
        "dash_raw_curve": "Curva bruta",
        "dash_flow_adjusted": "Ajustada por flujo",
        "dash_save": "Guardar",
        "dash_reset": "Restablecer",
        "dash_signal_mode": "Modo señales",
        "dash_automated_mode": "Modo automático",
        "dash_show_all": "Mostrar todo",
        "dash_no_data_loaded": "Sin datos cargados",
        "dash_flow_note": "El ajuste de flujo elimina grandes depósitos/retiros.",
        "dash_no_series_visible": "Sin series visibles",
        "dash_models_count": "{count} modelos",
        "dash_active_models": "Modelos activos: {count}",
        "dash_updated_at": "Actualizado: {time}",
        "dash_selected": "Seleccionado",
        "dash_last": "Último",
        "dash_signal_only": "Solo señales",
        "dash_badge_signal": "SEÑAL",
        "dash_badge_on": "ACT",
        "dash_badge_off": "PAUS",
        "dash_long": "LARGO",
        "dash_short": "CORTO",
        "dash_dual": "DUAL",
        "dash_no_signal": "Sin señal",
        "dash_signal_long": "Señal larga",
        "dash_signal_short": "Señal corta",
        "dash_funding_error": "Error de funding",
        "dash_live": "En vivo",
        "dash_no_data": "Sin datos",
        "dash_lag_trade": "Latencia trade",
        "dash_last_update": "Última actualización",
        "dash_tp": "TP",
        "dash_sl": "SL",
        "dash_pred": "Pred",
        "dash_flow_threshold_note": "Umbral de flujo: {pct}% (se eliminan grandes depósitos/retiros).",
        "dash_total_unified_funding": "Total (Unificada + Funding): {total} | Actualizado: {updated}",
        "dash_funding": "Funding",
        "dash_ok": "OK",
        "dash_raw": "Bruto",
        "dash_adjusted": "Ajust.",
        "dash_direction_short": "Dir",
        "dash_aggressive_short": "Agres",
        "dash_showing": "Mostrando",
        "dash_signal_label": "señal",
        "dash_price": "Precio",
        "dash_volume": "Volumen",
        "dash_chart": "Gr\u00e1fico",
        "dash_range": "Rango",
        "dash_bars": "Velas",
        "dash_interval": "Intervalo",
        "dash_metrics_feed": "Flujo de m\u00e9tricas",
        "dash_no_chart_data": "A\u00fan no hay datos del gr\u00e1fico.",
        "dash_no_data_range": "No hay datos en el rango seleccionado.",
        "dash_buy_clip": "Clip compra",
        "dash_sell_clip": "Clip venta",
        "dash_entry_tp": "Entrada TP",
        "dash_entry_sl": "Entrada SL",
        "dash_tp_clip": "Clip TP",
        "dash_sl_long": "SL Long",
        "dash_sl_short": "SL Short",
        "dash_paused": "Pausado",
        "dash_posted": "Publicado",
        "dash_ready": "Listo",
        "dash_waiting_state": "Esperando",
        "dash_idle": "Inactivo",
        "dash_fill": "Llenado",
        "dash_target_short": "T",
        "dash_remaining_short": "Rest",
        "dash_clip": "Clip",
        "dash_exchange_short": "exch",
        "dash_na": "n/d",
        "dash_donate_referral": "Donar y enlace de referido",
        "dash_donate_note": "Las contribuciones me apoyan directamente y ayudan a mejorar esta herramienta",
        "dash_bitcoin": "Bitcoin (BTC)",
        "dash_ethereum": "Ethereum (ETH)",
        "dash_referral_link": "Enlace de referido de Bybit",
        "dash_close": "Cerrar",
        "dash_flat": "Sin posición",
        "dash_waiting_balance": "Esperando balance...",
        "dash_yes": "Sí",
        "dash_no": "No",
        "dash_none": "Ninguno",
    },
}

# Current language (default: auto-detect)
_current_lang: Optional[str] = None


def detect_system_language() -> str:
    """Detect language from OS locale, returns 'en' or 'es'."""
    try:
        # Check environment variables first
        lang = os.getenv("LANG") or os.getenv("LANGUAGE") or os.getenv("LC_ALL") or ""
        if lang.lower().startswith("es"):
            return "es"

        # Try locale module
        system_locale = locale.getdefaultlocale()[0] or ""
        if system_locale.lower().startswith("es"):
            return "es"

        # Windows-specific check
        if os.name == "nt":
            import ctypes
            windll = ctypes.windll.kernel32
            lang_id = windll.GetUserDefaultUILanguage()
            # Spanish language codes: 0x0a (primary), various sublanguages
            if (lang_id & 0xFF) == 0x0A:
                return "es"
    except Exception:
        pass

    return "en"


def get_language() -> str:
    """Get current language code."""
    global _current_lang
    if _current_lang is None:
        _current_lang = detect_system_language()
    return _current_lang


def set_language(lang: str) -> None:
    """Set current language ('en' or 'es')."""
    global _current_lang
    if lang in TRANSLATIONS:
        _current_lang = lang
    else:
        _current_lang = "en"


def t(key: str, **kwargs) -> str:
    """
    Get translated string by key.
    Supports format placeholders: t("msg_dashboard_running", port=9007)
    Falls back to English if key not found in current language.
    """
    lang = get_language()

    # Try current language first
    text = TRANSLATIONS.get(lang, {}).get(key)

    # Fallback to English
    if text is None:
        text = TRANSLATIONS.get("en", {}).get(key)

    # Return key if not found anywhere
    if text is None:
        return key

    # Apply format arguments
    if kwargs:
        try:
            text = text.format(**kwargs)
        except KeyError:
            pass

    return text


def get_available_languages() -> list:
    """Get list of available language codes."""
    return list(TRANSLATIONS.keys())


def get_dashboard_translations() -> dict:
    """
    Get all dashboard-related translations for the current language.
    Returns a dict that can be JSON-serialized and injected into HTML.
    """
    lang = get_language()
    result = {}

    # Get all keys starting with "dash_"
    lang_dict = TRANSLATIONS.get(lang, {})
    en_dict = TRANSLATIONS.get("en", {})

    for key in en_dict:
        if key.startswith("dash_"):
            # Try current language first, fallback to English
            result[key] = lang_dict.get(key, en_dict.get(key, key))

    return result
