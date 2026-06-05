#!/bin/bash
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

if command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON="$(command -v python)"
else
  echo "Python 3.10 or newer is required."
  echo "Install Python from https://www.python.org/downloads/"
  read -r -p "Press Enter to close..."
  exit 1
fi

"$PYTHON" "$SCRIPT_DIR/bootstrap.py"
STATUS=$?

if [ "$STATUS" -ne 0 ]; then
  echo
  read -r -p "Press Enter to close..."
fi

exit "$STATUS"
