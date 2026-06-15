@echo off
REM ===================================================================
REM  Permit-to-Proof  -  one-click launcher for Windows
REM  Creates a virtual environment, installs dependencies (only when
REM  missing or changed), copies .env on first run, then opens the app
REM  in your browser. No terminal interaction needed after this.
REM ===================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

REM --- 1. Find a Python 3 interpreter (prefer 3.13) -----------------
set "PYEXE="
py -3.13 --version >nul 2>&1 && set "PYEXE=py -3.13"
if not defined PYEXE ( py -3 --version >nul 2>&1 && set "PYEXE=py -3" )
if not defined PYEXE ( python --version >nul 2>&1 && set "PYEXE=python" )
if not defined PYEXE (
    echo.
    echo   ERROR: Python 3 was not found on this machine.
    echo   Install it from https://www.python.org/downloads/ ^(check
    echo   "Add Python to PATH" during setup^), then run this file again.
    echo.
    pause
    exit /b 1
)

REM --- 2. Create the virtual environment if needed -----------------
if not exist ".venv" (
    echo Creating virtual environment...
    %PYEXE% -m venv .venv
    if errorlevel 1 ( echo Failed to create venv. & pause & exit /b 1 )
)
call ".venv\Scripts\activate.bat"

REM --- 3. Install / update dependencies only when changed ----------
fc /b requirements.txt ".venv\requirements.lock" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies ^(first run or requirements changed^)...
    python -m pip install --upgrade pip -q
    python -m pip install -q -r requirements.txt
    if errorlevel 1 ( echo Dependency install failed. & pause & exit /b 1 )
    copy /y requirements.txt ".venv\requirements.lock" >nul
)

REM --- 4. First-run secrets file -----------------------------------
if not exist ".env" if exist ".env.example" copy ".env.example" ".env" >nul

REM --- 5. Open the browser shortly after the server starts ---------
start "" /b cmd /c "timeout /t 4 /nobreak >nul & start http://localhost:8501"

REM --- 6. Launch the app -------------------------------------------
echo.
echo   Permit-to-Proof is starting at http://localhost:8501
echo   Leave this window open while you use the app. Close it to stop.
echo.
streamlit run app\main.py

endlocal
