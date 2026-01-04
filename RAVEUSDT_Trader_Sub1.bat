@echo off
title PurpleSky LIQUIDITY BOT - WATCHDOG (RAVEUSDT)
color 0A

:: Load API Keys safely
if exist secrets.bat call secrets.bat

if "%BYBIT_API_KEY%"=="" (
    echo [WARNING] API Key NOT found. Bot will run in DRY MODE.
) else (
    echo [INFO] API Key detected. Live Trading Authorized.
)

:loop
cls
echo ========================================================
echo        STARTING SOFIA LIVE TRADER - RAVEUSDT
echo ========================================================
echo Timestamp: %date% %time%
echo.

if "%BYBIT_API_KEY%"=="" (
    echo [WARNING] API Key NOT found. Bot will run in DRY MODE.
) else (
    echo [INFO] API Key detected. Live Trading Authorized.
)
echo.

:: Run the Python Bot using the newly trained v4 model
python -W "ignore" live_trader.py --symbol RAVEUSDT --model-root models_v4/RAVEUSDT

:: If we get here, the bot crashed or closed.
color 4F
echo.
echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
echo        WARNING: BOT CRASHED OR STOPPED
echo        RESTARTING IN 5 SECONDS...
echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
echo.
timeout /t 5 >nul
color 0A
goto loop