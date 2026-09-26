@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Chyba: chyba existujuce .venv\Scripts\python.exe so SpinQLabLink.
  exit /b 2
)
".venv\Scripts\python.exe" "spinq_live_suite.py" --config "live_suite.example.toml"
exit /b %errorlevel%
