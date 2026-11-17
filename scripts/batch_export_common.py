"""Read-only project inspection shared by serial export command-line tools."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import unicodedata

import cv2
import numpy as np

import project_store as store
from detect_solid_inpaint_folder import image_files_in_folder


def natural_key(value):
    return [int(x) if x.isdigit() else x.casefold() for x in re.split(r'(\d+)', str(value))]


def paths_for(folder):
    output = Path(folder) / 'ctd_inpainted'
    return {'output': str(output), 'raw': str(output / 'raw'),
            'export_pair': str(output / 'export_pair')}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def raw_hashes(folder):
    raw = Path(paths_for(folder)['raw'])
    files = [raw / 'project.json'] + sorted((raw / 'pages').glob('*.npz'))
    return {str(p.relative_to(raw)): sha256(p) for p in files}


def preflight_project(folder):
    folder = Path(folder)
    paths = paths_for(folder)
    project = store.load_project(paths['raw'])
    images = image_files_in_folder(str(folder))
    names = {Path(image).name for image in images}
    if not images:
        raise ValueError('Project contains no source images')
    if set(project.get('images', [])) != names:
        raise ValueError('Manifest images and source image filenames differ')
    if set(project.get('pages', {})) != names:
        raise ValueError('Manifest page entries and source images differ')
    stems = [unicodedata.normalize('NFC', Path(image).stem).casefold() for image in images]
    if len(set(stems)) != len(stems):
        raise ValueError('Source filenames have duplicate PNG stems')
    errors = []
    for image in images:
        entry = project['pages'][Path(image).name]
        signature = store.source_signature(image)
        archive = store.page_path(paths, image)
        if not entry.get('processed') or entry.get('error'):
            errors.append(f'{Path(image).name}: not successfully processed')
        if entry.get('source_signature', signature) != signature:
            errors.append(f'{Path(image).name}: manifest source signature mismatch')
        if entry.get('has_data') and not archive.exists():
            errors.append(f'{Path(image).name}: authoritative page archive missing')
        if archive.exists():
            try:
                with np.load(archive, allow_pickle=False) as data:
                    if int(data.get('version', -1)) != store.FORMAT_VERSION:
                        errors.append(f'{Path(image).name}: unsupported page archive version')
                    if str(data.get('source_signature', '')) != signature:
                        errors.append(f'{Path(image).name}: archive source signature mismatch')
            except Exception as exc:
                errors.append(f'{Path(image).name}: unreadable archive: {exc}')
    if errors:
        raise ValueError('\n'.join(errors))
    return images


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f'Image cannot be decoded: {path}')
    return image


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def discover_projects(root):
    root = Path(root)
    return sorted({p.parents[2] for p in root.rglob('project.json')
                   if p.parent.name == 'raw' and p.parent.parent.name == 'ctd_inpainted'},
                  key=lambda p: natural_key(p.relative_to(root)))
