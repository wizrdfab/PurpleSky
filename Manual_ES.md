# Manual de Usuario de PurpleSky

Este manual proporciona instrucciones detalladas sobre cómo operar el **Sistema de Trading Cuantitativo PurpleSky**. Cubre todo el ciclo de vida desde la recolección de datos hasta la ejecución en vivo.

---

## 1. Configuración del Entorno

### 1.1 Prerrequisitos
*   **Sistema Operativo:** Windows, Linux o macOS.
*   **Python:** Versión 3.10 o superior.
*   **RAM:** Mínimo 8GB (16GB recomendado para entrenar datasets grandes).
*   **Almacenamiento:** SSD recomendado para procesamiento de datos más rápido.

### 1.2 Instalación
1.  Clona el repositorio:
    ```bash
    git clone https://github.com/wizrdfab/PurpleSky.git
    cd PurpleSky
    ```
2.  Crea un entorno virtual (opcional pero recomendado):
    ```bash
    python -m venv venv
    # Windows:
    .\venv\Scripts\activate
    # Linux/Mac:
    source venv/bin/activate
    ```
3.  Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```

### 1.3 Claves API
Crea un archivo llamado `key_profiles.json` en el directorio raíz. Este archivo almacena tus credenciales del exchange. **Nunca subas este archivo a Git.**

**Formato:**
```json
{
    "profiles": {
        "default": {
            "api_key": "TU_CLAVE_API_BYBIT",
            "api_secret": "TU_SECRETO_API_BYBIT"
        },
        "paper": {
            "api_key": "TU_CLAVE_TESTNET",
            "api_secret": "TU_SECRETO_TESTNET"
        }
    }
}
```

---

## 2. Adquisición de Datos (El Combustible)

Antes de poder entrenar un modelo, necesitas datos de Tick y Libro de Órdenes de alta calidad. PurpleSky incluye un recolector robusto.

### 2.1 Ejecutar el Recolector
El recolector se conecta al WebSocket de Binance Futures (UM) y guarda los datos en el directorio `data/`.

```bash
# Sintaxis: python data_collector.py [SIMBOLO1] [SIMBOLO2] ...
python data_collector.py FARTCOINUSDT
```

*   **Duración:** Deja esto ejecutándose por al menos **3-7 días** para recopilar suficientes datos para un modelo significativo. Cuanto más, mejor.
*   **Salida:** Los datos se guardan en `data/FARTCOINUSDT_Binance/Trade/` y `data/FARTCOINUSDT_Binance/Orderbook/`.

### 2.2 Estructura de Datos
*   `Trade/*.csv`: Datos de trades tick por tick (Precio, Tamaño, Lado).
*   `Orderbook/*.data`: Snapshots JSONL del Libro de Órdenes Nivel 2 (200 niveles).

---

## 3. Entrenamiento y Optimización (El Motor)

Una vez que tengas datos, usa el pipeline AutoML para construir el "Cerebro" del bot.

### 3.1 El Script `train.py`
Este script realiza Ingeniería de Características, Validación Cruzada y Optimización de Hiperparámetros.

**Comando:**
```bash
python train.py --symbol FARTCOINUSDT --data-dir data/FARTCOINUSDT_Binance --trials 50
```

**Argumentos Clave:**
*   `--symbol`: El nombre del activo (ej. `FARTCOINUSDT`).
*   `--data-dir`: Ruta a los datos recolectados (ej. `data/FARTCOINUSDT_Binance`).
*   `--trials`: Número de pruebas de optimización (Por defecto: 20). Usa 50-100 para mejores resultados.
*   `--timeframe`: Timeframe base para las barras (Por defecto: `5m`).

### 3.2 ¿Qué sucede durante el entrenamiento?
1.  **Generación de Características:** Calcula métricas complejas (Elasticidad de Liquidez, Distancias VWAP).
2.  **K-Fold Purgado:** Divide los datos en chunks, eliminando solapamientos para prevenir trampa.
3.  **Optimización Optuna:** Prueba diferentes combinaciones de:
    *   **Financieros:** Stop Loss, Take Profit, Offset de Límite.
    *   **Lógica:** Umbral Agresivo (Cuándo cambiar a Órdenes de Mercado).
    *   **Modelo:** Profundidad de Árbol, Tasa de Aprendizaje.
4.  **Selección:** Elige la mejor prueba basada en el Ratio Sortino.
5.  **Validación:** Prueba al ganador en el 15% final de datos (Holdout).
6.  **Guardado:** Guarda el modelo en `models/FARTCOINUSDT/...`.

---

## 4. Operaciones en Vivo (Conduciendo)

El script `live_trading_v2.py` es el motor de ejecución.

### 4.1 Prueba Seca (Simulación)
Siempre comienza con una Prueba Seca para verificar conexiones y lógica sin arriesgar fondos.

```bash
python live_trading_v2.py --symbol FARTCOINUSDT --keys-file key_profiles.json --dry-run
```
*   **Comportamiento:** Simula órdenes localmente. No se envían llamadas API al exchange para colocación.
*   **Logs:** Revisa los logs para ver mensajes "Placing Buy Limit...".

### 4.2 Trading en Vivo (Dinero Real)
Cuando estés seguro, elimina la bandera `--dry-run`.

```bash
python live_trading_v2.py --symbol FARTCOINUSDT --keys-file key_profiles.json
```

**Argumentos Opcionales:**
*   `--log-features`: Registra el vector de características para cada barra (útil para depuración).
*   `--testnet`: Se conecta al Testnet de Bybit en lugar del Mainnet.

### 4.3 Entendiendo la Estrategia Híbrida en Acción
El bot operará en dos modos automáticamente:
1.  **Modo Francotirador (Órdenes Límite):** Verás que coloca órdenes *por debajo* del precio actual (para Compras). Espera una caída. Si la caída no ocurre, cancela y mueve la orden.
2.  **Modo Agresor (Órdenes de Mercado):** Si la señal de tendencia sube (ej. > 0.87), verás entradas inmediatas de Mercado con un Take Profit mucho más amplio (10x).

---

## 5. Monitoreo y Panel

### 5.1 El Panel
PurpleSky incluye un panel web local ligero.

1.  Ejecuta en una terminal separada:
    ```bash
    python live_dashboard.py
    ```
2.  Abre tu navegador en: `http://localhost:9007`

**Características:**
*   **Gráfico en Vivo:** Gráfico de velas con marcadores de Compra/Venta.
*   **Confianza del Modelo:** Medidor en tiempo real de las probabilidades de los modelos de Ejecución y Dirección.
*   **Curva de PnL:** Rastrea el rendimiento de tu sesión.

### 5.2 Archivos de Log
*   `live_trading_v2.log`: Logs operacionales generales (Heartbeats, Errores).
*   `signals_SYMBOL.jsonl`: Log estructurado de cada señal generada por el modelo.
*   `open_orders_SYMBOL.jsonl`: Snapshot crudo de órdenes abiertas (para depurar reconciliación).

---

## 6. Solución de Problemas

**P: "History insufficient / bootstrapping..."**
*   **Causa:** El bot necesita datos históricos para calcular características (ej. VWAP de 24h).
*   **Solución:** El bot intenta descargar esto del exchange automáticamente. Si falla, asegúrate de que tu conexión a internet sea estable o deja el `data_collector.py` ejecutándose más tiempo antes de iniciar el trading en vivo.

**P: "Daily drawdown limit exceeded"**
*   **Causa:** El fusible de seguridad saltó. Perdiste demasiado dinero hoy (Por defecto: 3% del equity).
*   **Solución:** El bot pausa el trading. Revisa tu estrategia. Para reiniciar, reinicia el script (la línea base del equity se reinicia).

**P: "Order not confirmed resting"**
*   **Causa:** Lag del API. El bot colocó una orden pero el exchange no la reportó inmediatamente vía WebSocket.
*   **Solución:** El bot reintentará la confirmación vía REST API. Usualmente se resuelve solo.

**P: "Drift alerts"**
*   **Causa:** El comportamiento actual del mercado (volatilidad, spread) es muy diferente de los Datos de Entrenamiento.
*   **Acción:** Si esto persiste, el modelo puede estar obsoleto. Considera recolectar nuevos datos y reentrenar (`train.py`).

---

**¡Feliz Trading!**
*El Equipo PurpleSky*
