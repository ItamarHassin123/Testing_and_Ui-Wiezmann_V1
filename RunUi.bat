@echo off
setlocal
cd /d "%~dp0"
set "PYLAUNCH=py -3.11"
%PYLAUNCH% -c "import sys" >nul 2>&1
if errorlevel 1 set "PYLAUNCH=py"
if not exist ".venv\Scripts\python.exe" (
    %PYLAUNCH% -m venv .venv
)
set "PYEXE=%cd%\.venv\Scripts\python.exe"
"%PYEXE%" -m pip install --upgrade pip
"%PYEXE%" -m pip install streamlit opencv-python torch torchvision pillow playsound3
REM ---- Actually run the app (this blocks and serves) ----
"%PYEXE%" -m streamlit run app.py
pause
endlocal