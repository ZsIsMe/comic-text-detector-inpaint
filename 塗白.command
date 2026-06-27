#!/bin/bash
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

wait_for_error() {
  echo
  read -r -p "按 Enter 關閉..."
}

close_current_terminal_window() {
  local current_tty
  current_tty="$(tty)"
  osascript -e "delay 0.4" \
    -e "tell application \"Terminal\"" \
    -e "repeat with w in windows" \
    -e "repeat with t in tabs of w" \
    -e "if tty of t is \"$current_tty\" then" \
    -e "close w" \
    -e "return" \
    -e "end if" \
    -e "end repeat" \
    -e "end repeat" \
    -e "end tell" >/dev/null 2>&1 </dev/null &
}

if command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON="$(command -v python)"
else
  echo "Python 3.10 or newer is required."
  echo "Install Python from https://www.python.org/downloads/"
  wait_for_error
  exit 1
fi

"$PYTHON" "$SCRIPT_DIR/bootstrap.py"
STATUS=$?

if [ "$STATUS" -ne 0 ]; then
  wait_for_error
else
  close_current_terminal_window
fi

exit "$STATUS"
