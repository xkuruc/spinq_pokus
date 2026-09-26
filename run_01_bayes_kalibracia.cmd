@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Chyba: chyba povodne funkcne .venv so SpinQLabLink. Ziadne meranie sa nespustilo.
  exit /b 2
)
".venv\Scripts\python.exe" -c "import importlib.metadata,sys; sys.exit(0 if importlib.metadata.version('spinqlablink') == '1.0.2' else 2)"
if errorlevel 1 (
  echo Chyba: povodne prostredie nema SpinQLabLink 1.0.2. Ziadne meranie sa nespustilo.
  exit /b 2
)

if not exist ".bayes-venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m venv ".bayes-venv"
  if errorlevel 1 exit /b 2
)
".bayes-venv\Scripts\python.exe" -c "import spinqlablink,numpy,scipy,sklearn,matplotlib"
if errorlevel 1 (
  set PIP_NO_INPUT=1
  ".bayes-venv\Scripts\python.exe" -m pip --disable-pip-version-check install --only-binary=:all: -r "requirements-01-bayes.txt"
  if errorlevel 1 (
    echo Chyba: izolovane zavislosti sa nenainstalovali. Povodna .venv sa nezmenila.
    exit /b 2
  )
)

".bayes-venv\Scripts\python.exe" "bayes_01_windows.py" --config "config-01-bayes.json" %*
exit /b %errorlevel%
