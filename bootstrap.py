#!/usr/bin/env python3
"""Bootstrap launcher for Solid Inpaint."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
import venv
from pathlib import Path


APP_NAME = '塗白'
MODEL_URL = 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.2.1/comictextdetector.pt'
MODEL_PATH = Path(__file__).resolve().parent / 'models' / 'comictextdetector.pt'
CTBD_MODEL_URL = 'https://huggingface.co/ogkalu/comic-text-and-bubble-detector/resolve/main/detector.onnx'
CTBD_MODEL_PATH = Path(__file__).resolve().parent / 'models' / 'comic-text-and-bubble-detector.onnx'
CTBD_MODEL_SHA256 = '065744e91c0594ad8663aa8b870ce3fb27222942eded5a3cc388ce23421bd195'
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


def is_venv_healthy(python: Path) -> bool:
    try:
        subprocess.check_call(
            [str(python), '--version'],
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def ensure_venv() -> Path:
    python = venv_python()
    if python.exists():
        if is_venv_healthy(python):
            return python
        info('Virtual environment is broken. Recreating it...')
        try:
            shutil.rmtree(VENV_DIR)
        except Exception as exc:
            fail(f'Could not remove broken virtual environment: {exc}')

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


def download_model(url: str, model_path: Path, expected_sha256: str | None = None) -> None:
    if model_path.exists():
        if expected_sha256:
            hasher = hashlib.sha256()
            with model_path.open('rb') as source:
                while chunk := source.read(1024 * 1024):
                    hasher.update(chunk)
            if hasher.hexdigest() != expected_sha256:
                fail(f'Model checksum does not match: {model_path}')
        return

    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = model_path.with_suffix(model_path.suffix + '.download')
    info('Model file is missing.')
    info('Downloading model from the original public release:')
    info(url)
    info('Model license and ownership belong to the original project.')

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
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
            f'{model_path}. Details: {exc}'
        )

    if tmp_path.stat().st_size < 1024 * 1024:
        tmp_path.unlink()
        fail('Downloaded model is unexpectedly small.')

    if expected_sha256 and hasher.hexdigest() != expected_sha256:
        tmp_path.unlink()
        fail(f'Downloaded model checksum does not match: {model_path}')

    tmp_path.replace(model_path)
    info('Model downloaded successfully.')


def launch_app(python: Path) -> None:
    info('Starting UI...')
    run([str(python), str(ROOT / 'solid_inpaint_ui.py')])


def main() -> None:
    ensure_python_version()
    python = ensure_venv()
    ensure_dependencies(python)
    # Download both choices so selecting either detector never starts an
    # unexpected network operation inside the UI worker thread.
    download_model(CTBD_MODEL_URL, CTBD_MODEL_PATH, CTBD_MODEL_SHA256)
    download_model(MODEL_URL, MODEL_PATH)
    launch_app(python)


if __name__ == '__main__':
    main()
