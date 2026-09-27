#!/bin/sh
set -eu
cd "$(dirname "$0")"

if [ ! -x ".bench-test/bin/python" ]; then
  echo "01 ONLINE ERROR: Missing local Python environment .bench-test; no device command sent." >&2
  exit 2
fi

if [ "${1-}" = "--full" ]; then
  shift
  exec ".bench-test/bin/python" "bayes_online_windows.py" --skip-upload "$@"
fi

exec ".bench-test/bin/python" "bayes_online_windows.py" --quick --skip-upload "$@"
