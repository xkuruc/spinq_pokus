@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo 01 ONLINE ERROR: Missing original working .venv with SpinQLabLink. No hardware command sent.
  exit /b 2
)
".venv\Scripts\python.exe" -c "import importlib.metadata,sys; sys.exit(0 if importlib.metadata.version('spinqlablink') == '1.0.2' else 2)"
if errorlevel 1 (
  echo 01 ONLINE ERROR: Original .venv does not contain tested SpinQLabLink 1.0.2. No hardware command sent.
  exit /b 2
)

if not exist ".bayes-online-venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m venv ".bayes-online-venv"
  if errorlevel 1 exit /b 2
)
".bayes-online-venv\Scripts\python.exe" -c "import spinqlablink,numpy,scipy"
if errorlevel 1 (
  set PIP_NO_INPUT=1
  ".bayes-online-venv\Scripts\python.exe" -m pip --disable-pip-version-check install --only-binary=:all: -r "requirements-bayes-online.txt"
  if errorlevel 1 (
    echo 01 ONLINE ERROR: Isolated dependencies unavailable. Original .venv remains unchanged.
    exit /b 2
  )
)

".bayes-online-venv\Scripts\python.exe" "bayes_online_windows.py" %*
exit /b %errorlevel%
