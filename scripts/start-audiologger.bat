@echo off
REM Alternative launcher: shows a brief CMD flash on start, then exits.
REM Prefer start-audiologger.vbs for a completely silent launch.

setlocal
cd /d "%~dp0.."

if not exist ".venv\Scripts\pythonw.exe" (
    echo AudioLogger venv not found at .venv\Scripts\pythonw.exe
    echo Run 'uv venv' and 'uv pip install -e ".[gpu,dev]"' first.
    pause
    exit /b 1
)

start "" ".venv\Scripts\pythonw.exe" -m audiologger
endlocal
