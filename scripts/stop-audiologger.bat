@echo off
REM Kill any running AudioLogger tray + worker processes -- only those, not
REM every Python process started from the project's venv (pytest, a REPL...).
REM Useful when the tray icon hangs as a ghost or Quit didn't shut it down cleanly.

powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='pythonw.exe' OR Name='python.exe'\" | Where-Object { $_.CommandLine -match '-m\s+audiologger(\.transcribe_worker)?(\s|$)' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force; Write-Host ('Killed PID ' + $_.ProcessId) }"

echo.
echo Done. If a tray icon ghost remains, hover over it to make Windows refresh.
pause
