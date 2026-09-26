@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Chyba: chyba existujuce .venv\Scripts\python.exe so SpinQLabLink.
  exit /b 2
)
".venv\Scripts\python.exe" -c "import importlib.metadata,sys; sys.exit(0 if importlib.metadata.version('spinqlablink') == '1.0.2' else 2)"
if errorlevel 1 (
  echo Chyba: povodne funkcne prostredie nema SpinQLabLink 1.0.2. Nic sa nemeralo.
  exit /b 2
)
if not exist ".benchmark-venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m venv ".benchmark-venv"
  if errorlevel 1 exit /b 2
)
set PIP_NO_INPUT=1
".benchmark-venv\Scripts\python.exe" -m pip --disable-pip-version-check install --only-binary=:all: -r "requirements-benchmark.txt"
if errorlevel 1 (
  echo Instalacia izolovanych benchmark zavislosti zlyhala. Povodna .venv sa nezmenila.
  exit /b 2
)
".benchmark-venv\Scripts\python.exe" "benchmark_windows.py" %*
exit /b %errorlevel%
