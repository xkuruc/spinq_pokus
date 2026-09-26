@echo off
setlocal
cd /d "%~dp0"
if "%~1"=="" (
  echo Pouzitie: reanalyze_windows.cmd .\results\RUN_ID
  exit /b 2
)
if not exist ".benchmark-venv\Scripts\python.exe" (
  echo Chyba: chyba existujuce .benchmark-venv\Scripts\python.exe.
  exit /b 2
)
".benchmark-venv\Scripts\python.exe" "reanalyze_saved.py" "%~1" --upload
exit /b %errorlevel%
