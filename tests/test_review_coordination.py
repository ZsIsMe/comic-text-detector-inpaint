"""POSIX process checks for the review and GPU detection coordination layer."""

import json
import os
import select
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / '.agents/skills/comic-detect-review/scripts'


class CoordinationProcessTests(unittest.TestCase):
    def test_desktop_different_tmpdirs_compete_for_same_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = root / 'tmp-a', root / 'tmp-b'
            a.mkdir()
            b.mkdir()
            cmd = [sys.executable, str(SCRIPTS / 'desktop_lock.py'), '--lock-file', str(root / 'desktop.lock')]
            holder = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, text=True, env={**os.environ, 'TMPDIR': str(a)})
            try:
                self.assertTrue(json.loads(holder.stdout.readline())['locked'])
                contender = subprocess.run(cmd, input='\n', capture_output=True, text=True,
                                           timeout=5, env={**os.environ, 'TMPDIR': str(b)})
                self.assertEqual(contender.returncode, 2)
                holder.stdin.write('\n')
                holder.stdin.flush()
                self.assertEqual(holder.wait(timeout=5), 0)
                acquired = subprocess.run(cmd, input='\n', capture_output=True, text=True,
                                          timeout=5, env={**os.environ, 'TMPDIR': str(b)})
                self.assertEqual(acquired.returncode, 0)
            finally:
                if holder.poll() is None:
                    holder.kill()
                    holder.wait(timeout=5)
                holder.stdin.close()
                holder.stdout.close()
                holder.stderr.close()

    def test_instance_transaction_locks_serialize_only_matching_gui(self):
        code = ("import sys,coordination; from pathlib import Path; "
                "p=coordination.instance_lock_path(sys.argv[1],Path(sys.argv[2])); "
                "\nwith coordination.file_lock(p):\n print('acquired',flush=True); sys.stdin.readline()")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def start(ident):
                return subprocess.Popen([sys.executable, '-c', code, ident, str(root / 'coord')],
                                        cwd=str(SCRIPTS), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True)
            first = start('a' * 32)
            second = None
            third = None
            try:
                self.assertEqual(first.stdout.readline().strip(), 'acquired')
                second = start('a' * 32)
                self.assertFalse(select.select([second.stdout], [], [], .15)[0])
                third = start('b' * 32)
                self.assertTrue(select.select([third.stdout], [], [], 2)[0])
                self.assertEqual(third.stdout.readline().strip(), 'acquired')
                first.stdin.write('\n')
                first.stdin.flush()
                self.assertTrue(select.select([second.stdout], [], [], 2)[0])
                self.assertEqual(second.stdout.readline().strip(), 'acquired')
            finally:
                for proc in (first, second, third):
                    if proc is None:
                        continue
                    if proc.poll() is None:
                        proc.stdin.write('\n')
                        proc.stdin.flush()
                        try:
                            proc.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait(timeout=3)
                    proc.stdin.close()
                    proc.stdout.close()
                    proc.stderr.close()

    def test_gpu_queue_serializes_detection_and_forwards_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / 'events.txt'
            lock = root / 'gpu.lock'
            worker = ("import sys,time; from pathlib import Path; p=Path(sys.argv[1]); "
                      "name=sys.argv[2]; f=p.open('a'); f.write('start '+name+'\\n'); f.flush(); "
                      "time.sleep(.2); f.write('end '+name+'\\n'); f.close()")
            def cmd(name):
                return [sys.executable, str(SCRIPTS / 'detect_queue.py'), '--lock-file', str(lock),
                        '--', sys.executable, '-c', worker, str(log), name]
            a = root / 'tmp-a'
            b = root / 'tmp-b'
            a.mkdir()
            b.mkdir()
            first = subprocess.Popen(cmd('a'), env={**os.environ, 'TMPDIR': str(a)})
            second = subprocess.Popen(cmd('b'), env={**os.environ, 'TMPDIR': str(b)})
            self.assertEqual(first.wait(timeout=10), 0)
            self.assertEqual(second.wait(timeout=10), 0)
            lines = log.read_text().splitlines()
            self.assertEqual(len(lines), 4)
            self.assertEqual([line.split()[0] for line in lines], ['start', 'end', 'start', 'end'])
            failed = subprocess.run([sys.executable, str(SCRIPTS / 'detect_queue.py'),
                                     '--lock-file', str(lock), '--', sys.executable,
                                     '-c', 'raise SystemExit(7)'], timeout=5)
            self.assertEqual(failed.returncode, 7)
            recovered = subprocess.run(cmd('c'), timeout=5)
            self.assertEqual(recovered.returncode, 0)

    def test_gpu_queue_interrupt_releases_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock = root / 'gpu.lock'
            started = root / 'started'
            wrapper = [sys.executable, str(SCRIPTS / 'detect_queue.py'),
                       '--lock-file', str(lock), '--']
            running = subprocess.Popen(wrapper + [sys.executable, '-c',
                'import pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text("ready"); time.sleep(30)',
                str(started)], stderr=subprocess.DEVNULL)
            try:
                for _ in range(100):
                    if started.exists():
                        break
                    running.poll()
                    if running.returncode is not None:
                        self.fail('Detection worker exited before startup')
                    time.sleep(.02)
                self.assertTrue(started.exists())
                running.send_signal(signal.SIGINT)
                self.assertEqual(running.wait(timeout=5), 130)
                following = subprocess.run(wrapper + [sys.executable, '-c', 'pass'], timeout=5)
                self.assertEqual(following.returncode, 0)
            finally:
                if running.poll() is None:
                    running.kill()
                    running.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
