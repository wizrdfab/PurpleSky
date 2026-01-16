![Estado](https://img.shields.io/badge/Estado-Pre--Alpha-orange)

PurpleSky - Sistema Avanzado de Trading Cuantitativo
====================================================

PurpleSky es un motor de trading algorítmico de alto rendimiento, basado en eventos, diseñado para mercados de derivados de criptomonedas (Bybit/Binance). Implementa una Estrategia Híbrida novedosa que cambia dinámicamente entre scalping conservador de Reversión a la Media y seguimiento agresivo de Tendencia basado en una arquitectura de machine learning de doble capa.

--------------------------------------------------------------------------------

Filosofía Central
-----------------

PurpleSky resuelve el "Dilema del Scalper" (ganar poco frecuentemente pero perder el gran movimiento) separando la Probabilidad de Ejecución del Sesgo Direccional:

1. Modelo de Ejecución (El Francotirador):
   * Objetivo: Predice la probabilidad de que una orden límite se ejecute y alcance un objetivo de ganancia antes de un stop-loss.
   * Comportamiento: Contrario. Busca desequilibrios de liquidez y sobreextensiones (Reversión a la Media).
   * Salida: Señales de entrada "seguras" para mercados laterales/de rango.

2. Modelo de Dirección (El Filtro de Tendencia):
   * Objetivo: Predice puramente dónde estará el precio en el futuro (ej. t+30m), ignorando la mecánica de entrada.
   * Comportamiento: Basado en momentum. Identifica rupturas y flujo direccional fuerte.
   * Salida: Una puntuación de "Confianza de Tendencia" (0.0 a 1.0).

La Lógica Híbrida:
El sistema fusiona estas dos señales en una estrategia de ejecución dinámica:

Régimen          | Confianza Tend. | Estrategia       | Tipo de Ejecución   | Objetivo TP
-----------------|-----------------|------------------|---------------------|----------------------
Neutral / Lateral| Moderada (0.5-0.8)| Reversión Media | Orden Límite (Maker)| Estándar (0.8 ATR)
Ruptura          | Extrema (> 0.87)| Seguir Tendencia | Orden Mercado (Taker)| Agresivo (10x Std)

Esto permite a PurpleSky "escalpear el ruido" para ingresos consistentes mientras "aprovecha la ruptura" para ganancias extraordinarias.

--------------------------------------------------------------------------------

Arquitectura del Sistema
------------------------

1. Pipeline de Datos e Ingeniería de Características (feature_engine.py)
PurpleSky no depende de indicadores OHLCV simples. Ingiere datos de Tick y Libro de Órdenes de alta frecuencia para generar Alpha de Microestructura:
* Elasticidad del Libro: Mide qué tan "blandos" son los muros de liquidez (pendiente de la profundidad bid/ask).
* Dominancia de Liquidez: Ratio de profundidad pasiva vs volumen activo de takers.
* Z-Scores de Régimen: Normaliza volatilidad y volumen contra una línea base de 30 días para detectar eventos de "Despertar".
* Desviación del Micro-Precio: Detecta presión de compra/venta oculta dentro del spread.

2. Núcleo de Machine Learning (models.py, train.py)
* Algoritmo: LightGBM (Gradient Boosting) con DART (Dropout) para prevenir sobreajuste.
* Validación: Usa Validación Cruzada K-Fold Combinatoria Purgada (CPCV).
    * Purgado: Elimina puntos de datos donde las etiquetas de entrenamiento se solapan con datos de prueba para prevenir sesgo de anticipación (Fuga de datos).
    * Embargo: Añade buffers de seguridad entre divisiones.
* Optimización: Usa Optuna para optimizar no solo hiperparámetros del modelo (profundidad, hojas) sino también parámetros financieros (Stop Loss ATR, Take Profit ATR, Umbral Agresivo).

3. Backtester Basado en Eventos (backtest.py)
Un motor de simulación estricto y realista:
* Simulación de Latencia: Considera el tiempo de procesamiento.
* Estructura de Comisiones: Modela comisiones Maker vs Taker con precisión.
* Timeouts: Cancela órdenes límite si no se ejecutan en N barras (previniendo "fills tóxicos" posteriores).
* Mark-to-Market: Rastrea la curva de equity tick por tick.

4. Motor de Producción (live_trading_v2.py)
El script de ejecución probado en batalla:
* WebSocket Primero: Escucha streams públicos (Trade/Book) y privados (Order/Position) en tiempo real.
* Recuperación de Estado: Mantiene una base de datos JSON local para sobrevivir caídas o reinicios sin perder rastro de órdenes.
* Monitoreo de Deriva: Alerta si la distribución de datos en vivo diverge significativamente de los datos de entrenamiento (Deriva Conceptual).
* Gestión Multi-Posición: Puede gestionar hasta 3 posiciones superpuestas para escalar en tendencias.

--------------------------------------------------------------------------------

Instalación y Configuración
---------------------------

Prerrequisitos:
* Python 3.10+
* Cuenta de Bybit o Binance Futures (Actualmente, solo hay implementación de trading automático para Bybit, se requieren claves API para trading en vivo)

Instalación:
```bash
git clone https://github.com/wizrdfab/PurpleSky.git
cd PurpleSky
pip install -r requirements.txt
```

Configuración:
Edita config.py para ajustar configuraciones globales, o usa argumentos de línea de comandos.

--------------------------------------------------------------------------------

Flujo de Trabajo de Uso
-----------------------

1. Recolección de Datos
Recolecta datos crudos de tick y profundidad para tu activo objetivo. Puedes hacerlo directamente desde Bybit https://www.bybit.com/derivatives/en/history-data (datos de trades crudos y datos de libro de órdenes) o puedes descargar los datos de Binance usando un recolector (construye historial, así que puede tomar un tiempo antes de tener suficiente para entrenar los modelos):

```bash
python data_collector.py FARTCOINUSDT SOLUSDT ADAUSDT ZECUSDT
```
Almacena datos en data/FARTCOINUSDT_Binance/ data/SOLUSDT_Binance ...

2. Entrenamiento del Modelo (AutoML)
Ejecuta el pipeline de optimización. Esto:
1. Cargará datos y generará características.
2. Ejecutará pruebas de Optuna para encontrar los mejores parámetros de Estrategia Híbrida.
3. Validará el modelo campeón en un conjunto de prueba reservado.
4. Guardará el modelo en models/.
```bash
python train.py --symbol FARTCOINUSDT --data-dir data/FARTCOINUSDT_Binance --trials 50
```

3. Simulación / Prueba Seca
Prueba el modelo en vivo sin usar dinero real.
```bash

python live_trading_v2.py --symbol FARTCOINUSDT --keys-file key_profiles.json --dry-run
python live_trading_v2.py --symbol FARTCOINUSDT --keys-file key_profiles.json --testnet
```

4. Trading en Vivo (Dinero Real)
Advertencia: Asegúrate de entender los riesgos.
```bash
python live_trading_v2.py --symbol FARTCOINUSDT --keys-file key_profiles.json
```

5. Panel de Monitoreo
Lanza el panel web local para ver señales en vivo y PnL. El panel tiene un gráfico donde las señales se pueden ver claramente.
```bash
python live_dashboard.py
```
Accede en http://localhost:9007

El panel soporta notificaciones de Discord/Telegram. Puedes pasarlas como:

python live_dashboard.py --discord-webhook {URL} --telegram-token {TOKEN} --telegram-chat-id {CHATID}

Esto enviará notificaciones de drawdown máximo, errores, inicios y señales de alta confianza (tendencia agresiva o scalping).

--------------------------------------------------------------------------------

Métricas de Rendimiento (Ejemplo)
---------------------------------

Objetivo: FARTCOINUSDT (Altcoin de Alta Volatilidad)
Período: Holdout Fuera de Muestra (15% del dataset, ~4.5 días de trading)

Métrica          | Valor         | Notas
-----------------|---------------|----------------------------------------------
Retorno Total    | > 15.8%       | En ventana de validación corta
Tasa de Éxito    | ~81%          | Alta consistencia debido a entrada selectiva
Ratio Sortino    | 0.23          | Mejorado por lógica de "Home Run"
Drawdown Máximo  | 6.9%          | Gestionado vía SL estricto

Los resultados varían según el régimen del mercado. El rendimiento pasado no es indicativo de resultados futuros.

--------------------------------------------------------------------------------

Se sugiere encarecidamente que entrenes el modelo para ciertas monedas con datos de Binance (usando el data_collector por ejemplo) porque en algunas monedas Binance lidera desde el libro de órdenes/volumen y el modelo usa eso para hacer predicciones. Sin embargo, puedes usar datos de Binance para obtener señales para operar en Bybit manualmente o usar esos datos para operar automáticamente en Bybit simplemente entrenando el modelo con datos de Binance y ejecutándolo conectado a Bybit. Usar datos de Bybit para FARTCOINUSDT está bien porque Bybit lidera esta moneda, por lo que los resultados son más confiables. Pronto podría agregar un conector a la API de Binance para trading automático.

Contacto y Soporte
------------------

* Reportes de Bugs: Si encuentras un bug, por favor abre un Issue aquí: ../../issues
* Solicitudes de Funciones: ¿Tienes una idea? Inicia una discusión en la pestaña de Issues.
* Consultas de Negocio: contact@purple-sky.online

Comunidad
---------

¡No dudes en contactarme si necesitas ayuda o simplemente quieres charlar!

* Twitter/X: @Fabb_998 (https://twitter.com/Fabb_998)
* Telegram: Únete al Canal (https://t.me/PurpleSkymm)
* Discord: Únete al Servidor (https://discord.gg/JjSC23Cv)

--------------------------------------------------------------------------------

Soporte y Donaciones
--------------------

Si encuentras este proyecto útil, considera hacer una donación o registrarte con mi código de referido en Bybit:

* Bitcoin (BTC): 1PucNiXsUCzfrMqUGCPfwgdyE3BL8Xnrrp
* Ethereum (ETH): 0x58ef00f47d6e94dfc486a2ed9b3dd3cfaf3c9714

Crea una cuenta de Bybit usando mi enlace de referido:
https://www.bybit.com/invite?ref=14VP14Z

--------------------------------------------------------------------------------

Descargo de Responsabilidad
---------------------------

Este software es solo para propósitos educativos. No arriesgues dinero que tengas miedo de perder. USA EL SOFTWARE BAJO TU PROPIO RIESGO. LOS AUTORES Y TODOS LOS AFILIADOS NO ASUMEN RESPONSABILIDAD POR TUS RESULTADOS DE TRADING.

Recomendamos encarecidamente que pruebes el software en el Testnet de Bybit antes de usar dinero real.

Nota Legal:
Este software se proporciona "tal cual", sin garantía de ningún tipo, expresa o implícita, incluyendo pero no limitado a las garantías de comerciabilidad, idoneidad para un propósito particular y no infracción. En ningún caso los autores o titulares de derechos de autor serán responsables de cualquier reclamo, daños u otra responsabilidad, ya sea en una acción de contrato, agravio o de otro modo, que surja de, fuera de o en conexión con el software o el uso u otros tratos en el software.
