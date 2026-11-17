"""Small POSIX flock coordination shared across TMPDIR values for one user."""

from __future__ import annotations

import fcntl
import json
import os
import stat
import uuid
from contextlib import contextmanager
from pathlib import Path


ROOT = Path('/tmp') / f'solid-inpaint-{os.getuid()}' / 'coordination'


def _private_dir(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError(f'Unsafe coordination directory: {path}')


def directory(root: Path = ROOT) -> Path:
    root = Path(root)
    if root == ROOT:
        _private_dir(ROOT.parent)
    _private_dir(root)
    return root


@contextmanager
def file_lock(path: Path, *, wait: bool = True):
    directory(path.parent)
    handle = path.open('a+')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        yield handle
    finally:
        handle.close()


def instance_lock_path(instance_id: str, root: Path = ROOT) -> Path:
    if len(instance_id) != 32 or any(ch not in '0123456789abcdef' for ch in instance_id):
        raise ValueError('Invalid GUI instance ID')
    return directory(root) / f'{instance_id}.lock'


def claim_path(instance_id: str, root: Path = ROOT) -> Path:
    return directory(root) / f'{instance_id}.claim.json'


def read_claim(instance_id: str, root: Path = ROOT) -> dict | None:
    path = claim_path(instance_id, root)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def atomic_json(path: Path, value: dict) -> None:
    directory(path.parent)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def process_alive(pid: int) -> bool:
    if pid < 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def verify_claim(binding: dict, binding_path: Path, root: Path = ROOT) -> None:
    token = binding.get('claim_token')
    if not token:
        raise RuntimeError('Legacy binding has no GUI claim; run bind again explicitly')
    claim = read_claim(binding['instance_id'], root)
    if not claim or any(claim.get(key) != value for key, value in (
        ('instance_id', binding['instance_id']), ('pid', binding['pid']),
        ('session', binding['session']), ('binding', str(binding_path.resolve())),
        ('claim_token', token))):
        raise RuntimeError('GUI claim was released or belongs to another session; bind again')


def create_claim(binding: dict, binding_path: Path, root: Path = ROOT) -> dict:
    """Caller holds this instance's file lock and has checked live GUI status."""
    previous = read_claim(binding['instance_id'], root)
    identity = {key: binding[key] for key in ('instance_id', 'pid', 'session')}
    identity['binding'] = str(binding_path.resolve())
    if previous:
        same = all(previous.get(key) == value for key, value in identity.items())
        if same and binding_path.exists():
            existing = json.loads(binding_path.read_text(encoding='utf-8'))
            if existing.get('claim_token') == previous.get('claim_token'):
                return previous
        if process_alive(int(previous.get('pid', 0))):
            raise RuntimeError('GUI instance is already claimed by another session or binding')
        claim_path(binding['instance_id'], root).unlink(missing_ok=True)
    claim = {**identity, 'claim_token': uuid.uuid4().hex}
    atomic_json(claim_path(binding['instance_id'], root), claim)
    return claim
