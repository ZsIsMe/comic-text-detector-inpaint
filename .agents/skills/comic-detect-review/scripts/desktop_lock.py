"""Cooperative POSIX desktop-input lock for Solid Inpaint Computer Use.

Run in an interactive exec/TTY session before physical mouse or keyboard input.
The OS releases flock when this process exits, including after a crash. MCP
calls and image reading do not require this lock.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from pathlib import Path

import coordination


DEFAULT_LOCK = coordination.ROOT / 'desktop.lock'


def acquire(path: Path, *, wait: bool = False):
    coordination.directory(path.parent)
    handle = path.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
    except BaseException:
        handle.close()
        raise
    return handle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-file", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--wait", action="store_true", help="Wait until the desktop lock is free")
    args = parser.parse_args(argv)
    try:
        with acquire(args.lock_file, wait=args.wait):
            print(json.dumps({"locked": True, "pid": os.getpid(),
                              "lock_file": str(args.lock_file)}), flush=True)
            # Keep the process alive until the agent sends a newline or EOF.
            sys.stdin.readline()
        print(json.dumps({"locked": False}), flush=True)
        return 0
    except BlockingIOError:
        print(json.dumps({"locked": False, "error": "desktop lock held by another session"}),
              file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
