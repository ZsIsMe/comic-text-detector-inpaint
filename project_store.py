"""Version 2 project storage: authoritative page decisions and disposable caches.

Page archives contain a coloured solid layer, an exclusive OTHER mask and an
internal edit lock (including erasures). No detector output is an editable layer.
All writes replace complete files atomically; cache updates preserve other keys.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading
import zipfile

import numpy as np

FORMAT_VERSION = 2
_lock = threading.RLock()


def _atomic(path: Path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_archive(path):
    path = Path(path)
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def write_archive(path, data):
    with _lock:
        _atomic(Path(path), lambda handle: np.savez_compressed(handle, **data))


def update_cache_file(path, **values):
    with _lock:
        try:
            data = read_archive(path)
        except (OSError, ValueError, EOFError, zipfile.BadZipFile):
            data = {}
        data.update(values)
        write_archive(path, data)


def read_cache_file(path):
    try:
        return read_archive(path)
    except (OSError, ValueError, EOFError, zipfile.BadZipFile):
        return {}


def project_path(raw):
    return Path(raw) / 'project.json'


def load_project(raw):
    path = project_path(raw)
    if not path.exists():
        return {'version': FORMAT_VERSION, 'pages': {}, 'settings': {}}
    value = json.loads(path.read_text(encoding='utf-8'))
    if value.get('version') != FORMAT_VERSION:
        raise ValueError('不支援的專案格式；請使用新的輸出資料夾。')
    return value


def update_project(raw, **values):
    with _lock:
        project = load_project(raw)
        project.update(values)
        pages = project.get('pages', {})
        project['summary'] = {
            'total': max(len(project.get('images', [])), len(pages)),
            'processed': sum(bool(p.get('processed')) and 'error' not in p for p in pages.values()),
            'failed': sum('error' in p for p in pages.values()),
            'with_other_mask': sum(p.get('other_pixels', 0) > 0 for p in pages.values()),
        }
        payload = json.dumps(project, ensure_ascii=False, indent=2).encode('utf-8')
        _atomic(project_path(raw), lambda handle: handle.write(payload))
        return project


def merge_report(raw, report):
    """Merge worker metadata without replacing newer per-page saves."""
    with _lock:
        pages = dict(load_project(raw).get('pages', {}))
        for name, info in report.get('pages', {}).items():
            if info.get('skipped') and not info.get('processed'):
                continue
            if 'error' in info or name not in pages:
                pages[name] = info
        return update_project(raw, **dict(report, pages=pages))


def page_path(paths, image):
    return Path(paths['raw']) / 'pages' / (Path(image).name + '.npz')


def cache_path(paths, image):
    return Path(paths['raw']) / 'cache' / (Path(image).name + '.npz')


def source_signature(image):
    stat = Path(image).stat()
    return f'{stat.st_size}:{stat.st_mtime_ns}'


def has_page(paths, image):
    entry = load_project(paths['raw']).get('pages', {}).get(Path(image).name, {})
    return page_path(paths, image).exists() or entry.get('processed', False)


def empty_page(shape):
    return {'overlay': np.zeros((*shape, 4), np.uint8),
            'other': np.zeros(shape, np.uint8), 'edited': np.zeros(shape, np.uint8)}


def load_page(paths, image, shape):
    path = page_path(paths, image)
    if not path.exists():
        entry = load_project(paths['raw']).get('pages', {}).get(Path(image).name, {})
        if entry.get('source_signature', source_signature(image)) != source_signature(image):
            raise ValueError('原圖已變更，請重新偵測。')
        if entry.get('has_data'):
            raise ValueError(f'正式頁面資料遺失：{path}')
        return empty_page(shape)
    data = read_archive(path)
    if int(data.get('version', -1)) != FORMAT_VERSION:
        raise ValueError(f'不支援的頁面格式：{path}')
    if str(data.get('source_signature', '')) != source_signature(image):
        raise ValueError('原圖已變更，請重新偵測。')
    if (data['overlay'].shape != (*shape, 4)
            or data['other'].shape != shape or data['edited'].shape != shape):
        raise ValueError(f'頁面資料尺寸與原圖不一致：{path}')
    if np.any((data['overlay'][:, :, 3] > 0) & (data['other'] > 0)):
        raise ValueError(f'頁面資料的兩類選區重疊：{path}')
    return {key: data[key] for key in ('overlay', 'other', 'edited')}


def page_summary(page):
    return {'processed': True, 'filled_pixels': int(np.count_nonzero(page['overlay'][:, :, 3])),
            'other_pixels': int(np.count_nonzero(page['other'])),
            'edited_pixels': int(np.count_nonzero(page['edited']))}


def save_page(paths, image, page, summary=None):
    overlay = np.asarray(page['overlay'], dtype=np.uint8).copy()
    other = np.where(page['other'] > 0, 255, 0).astype(np.uint8)
    edited = np.where(page['edited'] > 0, 255, 0).astype(np.uint8)
    if overlay.shape != (*other.shape, 4) or edited.shape != other.shape:
        raise ValueError('頁面資料尺寸不一致。')
    if np.any((overlay[:, :, 3] > 0) & (other > 0)):
        raise ValueError('純色填充和圖像修補不能重疊。')
    overlay[overlay[:, :, 3] == 0] = 0
    state = {'overlay': overlay, 'other': other, 'edited': edited}
    has_data = any(np.any(value) for value in state.values())
    signature = source_signature(image)
    with _lock:
        path = page_path(paths, image)
        # Empty processed pages need only a compact project entry. Erased pages
        # retain their edit locks, even when both visible classes are empty.
        if has_data:
            write_archive(path, dict(state, version=np.array(FORMAT_VERSION), source_signature=np.array(signature)))
        elif path.exists():
            path.unlink()
        project = load_project(paths['raw'])
        pages = project.get('pages', {})
        info = dict(summary or {})
        info.update(page_summary(state), source_signature=signature, has_data=has_data)
        # Per-block diagnostics belong to disposable cache, not project.json.
        info = {key: value for key, value in info.items()
                if key not in ('blocks_debug', 'bubbles_debug', 'solid_fill_settings')}
        info.pop('error', None)
        pages[Path(image).name] = info
        update_project(paths['raw'], pages=pages)
    return info
