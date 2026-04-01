@echo off
title ALPR Gate Management System
color 0B
cls

echo.
echo  ============================================================
echo   ALPR GATE MANAGEMENT SYSTEM
echo   Starting server...
echo  ============================================================
echo.

:: Change to script directory
cd /d "%~dp0"

:: Check if venv exists
if not exist "..\venv\Scripts\activate.bat" (
    echo  ERROR: Virtual environment not found!
    echo  Expected: %~dp0..\venv\Scripts\activate.bat
    echo.
    pause
    exit /b 1
)

:: Activate venv
call ..\venv\Scripts\activate.bat

:: Check if main.py exists
if not exist "main.py" (
    echo  ERROR: main.py not found in %~dp0
    pause
    exit /b 1
)

:: Check if dashboard.html exists
if not exist "dashboard.html" (
    echo  ERROR: dashboard.html not found in %~dp0
    pause
    exit /b 1
)

echo  Activating virtual environment... OK
echo  Starting Flask server on http://localhost:5000
echo.

:: Wait 2 seconds then open browser automatically
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:5000"

:: Start the server (keeps this window open)
python main.py

echo.
echo  Server stopped. Press any key to exit.
pause
