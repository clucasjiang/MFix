@echo off
rem Voice repair assistant pointing with the Fixpoint laser rig.
rem Extra options pass through, e.g.  assistant.cmd --rig-port COM7 --search off
cd /d "%~dp0LLM-pthysical-fix-helper-main"
if not exist ".venv\Scripts\python.exe" (
    echo Set up the assistant first, from LLM-pthysical-fix-helper-main:
    echo   python -m venv --system-site-packages .venv
    echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m src.main --rig %*
