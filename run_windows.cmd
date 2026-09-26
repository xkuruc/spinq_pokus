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
".benchmark-venv\Scripts\python.exe" -m pip --disable-pip-version-check install --only-binary=:all: -r "requirements-local-core.txt"
if errorlevel 1 (
  echo Instalacia izolovanych zakladnych zavislosti zlyhala. Povodna .venv sa nezmenila.
  exit /b 2
)
".benchmark-venv\Scripts\python.exe" -m pip show torch >nul 2>&1
if errorlevel 1 (
  echo Instalujem volitelny CPU PyTorch z oficialneho PyTorch indexu.
  ".benchmark-venv\Scripts\python.exe" -m pip --disable-pip-version-check install --only-binary=:all: torch --index-url https://download.pytorch.org/whl/cpu
  if errorlevel 1 echo PyTorch nie je dostupny; neurónové metody budu preskocene s dovodom.
)
".benchmark-venv\Scripts\python.exe" -m spinq_local.preflight --output ".benchmark-preflight.json"
if errorlevel 1 (
  echo Predstartova kontrola SDK alebo numeriky zlyhala. Ziadny RF experiment sa nespustil.
  exit /b 2
)
".benchmark-venv\Scripts\python.exe" "local_benchmark_windows.py" --preflight ".benchmark-preflight.json" %*
exit /b %errorlevel%
