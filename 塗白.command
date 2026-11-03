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

python_is_usable() {
  [ -n "$1" ] && [ -x "$1" ] && "$1" -c 'import sys; sys.exit(sys.version_info < (3, 10))' >/dev/null 2>&1
}

select_python() {
  local candidate version

  candidate="$SCRIPT_DIR/.venv/bin/python"
  if python_is_usable "$candidate"; then
    printf '%s\n' "$candidate"
    return 0
  fi

  # Finder may not include Homebrew in PATH. Prefer the tested Python versions.
  for version in 3.10 3.11 3.12; do
    for candidate in "$(command -v "python$version" 2>/dev/null || true)" \
                     "/opt/homebrew/bin/python$version" "/usr/local/bin/python$version"; do
      if python_is_usable "$candidate"; then
        printf '%s\n' "$candidate"
        return 0
      fi
    done
  done

  for candidate in /opt/homebrew/bin/python3.* /usr/local/bin/python3.* \
                   /opt/homebrew/bin/python3 /usr/local/bin/python3 \
                   "$(command -v python3 2>/dev/null || true)" \
                   "$(command -v python 2>/dev/null || true)"; do
    if python_is_usable "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

if ! PYTHON="$(select_python)"; then
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
