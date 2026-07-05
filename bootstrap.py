#!/usr/bin/env python3
"""Bootstrap launcher for Solid Inpaint."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import urllib.request
import venv
from pathlib import Path


APP_NAME = '塗白'
MODEL_URL = 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.2.1/comictextdetector.pt'
MODEL_PATH = Path(__file__).resolve().parent / 'models' / 'comictextdetector.pt'
ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / '.venv'
REQUIREMENTS = ROOT / 'requirements.txt'


def info(message: str) -> None:
    print(f'[{APP_NAME}] {message}', flush=True)


def fail(message: str, code: int = 1) -> None:
    print(f'[{APP_NAME}] ERROR: {message}', file=sys.stderr, flush=True)
    raise SystemExit(code)


def ensure_python_version() -> None:
    if sys.version_info < (3, 10):
        fail('Python 3.10 or newer is required.')


def venv_python() -> Path:
    if os.name == 'nt':
        return VENV_DIR / 'Scripts' / 'python.exe'
    return VENV_DIR / 'bin' / 'python'


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    info('Running: ' + ' '.join(cmd))
    try:
        subprocess.check_call(cmd, cwd=str(cwd or ROOT))
    except subprocess.CalledProcessError as exc:
        fail(f'Command failed with exit code {exc.returncode}: {" ".join(cmd)}', exc.returncode)


def ensure_venv() -> Path:
    python = venv_python()
    if python.exists():
        return python

    info('Creating virtual environment...')
    try:
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    except Exception as exc:
        fail(f'Could not create virtual environment: {exc}')

    if not python.exists():
        fail(f'Virtual environment Python was not found: {python}')
    return python


def ensure_dependencies(python: Path) -> None:
    stamp = VENV_DIR / '.solid_inpaint_requirements_installed'
    req_mtime = REQUIREMENTS.stat().st_mtime if REQUIREMENTS.exists() else 0
    if stamp.exists():
        try:
            if float(stamp.read_text(encoding='utf-8')) >= req_mtime:
                return
        except ValueError:
            pass

    info('Installing Python dependencies. This may take a while on first launch...')
    run([str(python), '-m', 'pip', 'install', '-U', 'pip'])
    run([str(python), '-m', 'pip', 'install', '-r', str(REQUIREMENTS)])
    stamp.write_text(str(req_mtime), encoding='utf-8')


def download_model() -> None:
    if MODEL_PATH.exists():
        return

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = MODEL_PATH.with_suffix('.pt.download')
    info('Model file is missing.')
    info('Downloading model from the original public release:')
    info(MODEL_URL)
    info('Model license and ownership belong to the original project.')

    try:
        with urllib.request.urlopen(MODEL_URL, timeout=30) as response:
            total = int(response.headers.get('Content-Length') or 0)
            downloaded = 0
            hasher = hashlib.sha256()
            with tmp_path.open('wb') as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    hasher.update(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 / total
                        print(f'\r[{APP_NAME}] Downloading model: {pct:5.1f}%', end='', flush=True)
            if total:
                print()
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        fail(
            'Could not download the model. You can manually place it at '
            f'{MODEL_PATH}. Details: {exc}'
        )

    if tmp_path.stat().st_size < 1024 * 1024:
        tmp_path.unlink()
        fail('Downloaded model is unexpectedly small.')

    tmp_path.replace(MODEL_PATH)
    info('Model downloaded successfully.')


def launch_app(python: Path) -> None:
    info('Starting UI...')
    run([str(python), str(ROOT / 'solid_inpaint_ui.py')])


def main() -> None:
    ensure_python_version()
    python = ensure_venv()
    ensure_dependencies(python)
    download_model()
    launch_app(python)


if __name__ == '__main__':
    main()
