@echo off
setlocal
cd /d "%~dp0"

REM ---- Pick python via py launcher (avoids Windows Store alias) ----
set "PYLAUNCH=py -3.11"
%PYLAUNCH% -c "import sys" >nul 2>&1
if errorlevel 1 (
    set "PYLAUNCH=py"
)

REM ---- Create venv if missing ----
if not exist ".venv\Scripts\python.exe" (
    %PYLAUNCH% -m venv .venv
)

REM ---- Always use venv python directly ----
set "PYEXE=%cd%\.venv\Scripts\python.exe"

REM ---- Install deps (quiet). If machine has no internet, this will fail. ----
"%PYEXE%" -m pip install --upgrade pip
"%PYEXE%" -m pip install streamlit opencv-python torch torchvision pillow playsound3

REM ---- Open browser ----
start "" "http://localhost:8501"


endlocal
