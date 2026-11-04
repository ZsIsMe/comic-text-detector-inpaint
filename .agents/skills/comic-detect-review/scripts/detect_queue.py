"""Run a detection command with a per-user POSIX GPU queue lock.

Only GPU detection is queued. CPU detection can use --device cpu. The lock
descriptor is inherited by the child, so a killed wrapper cannot free it
while that child is still running.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

import coordination


def run(command: list[str], device: str, lock_file: Path) -> int:
    if not command:
        raise ValueError('Provide a detection command after --')
    if device == 'cpu':
        return subprocess.call(command)
    interrupted = [None]
    with coordination.file_lock(lock_file) as handle:
        child = subprocess.Popen(command, pass_fds=(handle.fileno(),), start_new_session=True)
        def forward(signum, _frame):
            interrupted[0] = signum
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass
        old_int = signal.signal(signal.SIGINT, forward)
        old_term = signal.signal(signal.SIGTERM, forward)
        try:
            code = child.wait()
        finally:
            signal.signal(signal.SIGINT, old_int)
            signal.signal(signal.SIGTERM, old_term)
    if interrupted[0] is not None:
        return 128 + interrupted[0]
    return 128 - code if code < 0 else code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('gpu', 'cpu'), default='gpu')
    parser.add_argument('--lock-file', type=Path, default=coordination.ROOT / 'gpu-detection.lock')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    try:
        return run(command, args.device, args.lock_file)
    except (OSError, ValueError) as exc:
        print(f'detect_queue: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
