@echo off
setlocal EnableDelayedExpansion
title AI Fitness Trainer — Setup

echo.
echo  ════════════════════════════════════════════════════════
echo   AI Fitness Trainer — Windows Setup
echo  ════════════════════════════════════════════════════════
echo.

python --version 2>nul | findstr /C:"3.10" /C:"3.11" /C:"3.12" >nul
if errorlevel 1 (
    echo  ERROR: Python 3.10+ required.
    echo  Download: https://www.python.org/downloads/
    pause & exit /b 1
)

if not exist .venv (
    python -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo.
echo  Setup complete!
echo  To start: python main.py
echo  API server: uvicorn server.main_api:app --reload
echo.
pause
