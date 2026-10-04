@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo Set up the app first:
    echo python -m venv .venv
    echo .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -c "import tkinter, httpx, elevenlabs, PIL, sounddevice, pydantic, imageio_ffmpeg, dotenv" >nul 2>&1
if errorlevel 1 (
    echo App dependencies are missing. Run:
    echo .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)
start "Repair assistant" ".venv\Scripts\pythonw.exe" -m src.main %*
