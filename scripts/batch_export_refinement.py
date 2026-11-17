"""Serial, resumable refinement export from saved v2 projects; never edits raw data."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor

from detect_solid_inpaint_folder import (
    read_page_state, _compose_overlay_preview,
)
from solid_inpaint_ui import _overlay_mask_on_bgr, DEFAULT_OTHER_MASK_DISPLAY_ALPHA
from scripts.batch_export_common import (
    natural_key, paths_for, sha256, raw_hashes, preflight_project, read_image, save_json,
    discover_projects,
)


def display_settings():
    settings = QSettings('ComicTextDetector', 'SolidInpaintUI')
    color = QColor(str(settings.value('other_mask_display_color', '#FF6EA5')))
    if not color.isValid():
        color = QColor(255, 110, 165)
    try:
        percent = int(settings.value('other_mask_display_alpha_percent',
                                     round(DEFAULT_OTHER_MASK_DISPLAY_ALPHA * 100)))
    except (ValueError, TypeError):
        percent = round(DEFAULT_OTHER_MASK_DISPLAY_ALPHA * 100)
    return (color.blue(), color.green(), color.red()), max(0, min(100, percent)) / 100


def write_verified_png(path, expected):
    ok, encoded = cv2.imencode('.png', expected)
    if not ok:
        raise IOError(f'PNG encoding failed: {path}')
    encoded.tofile(str(path))
    actual = read_image(path)
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        raise ValueError(f'Export verification failed: {path}')
    return sha256(path)


def verified_resume(folder, previous, color, alpha):
    if not previous or previous.get('status') != 'success':
        return False
    if previous.get('color_bgr') != list(color) or previous.get('alpha') != alpha:
        return False
    if raw_hashes(folder) != previous.get('raw_hashes'):
        return False
    output = Path(paths_for(folder)['export_pair'])
    expected = previous.get('export_hashes', {})
    actual_files = {str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()}
    if not expected or actual_files != set(expected):
        return False
    return all(sha256(output / name) == digest for name, digest in expected.items())


def export_project(folder, images, color, alpha):
    folder = Path(folder)
    paths = paths_for(folder)
    final = Path(paths['export_pair'])
    if final.exists():
        raise FileExistsError(f'Existing export is preserved: {final}')
    before = raw_hashes(folder)
    staging = final.with_name(f'export_pair.batch-staging-{uuid.uuid4().hex[:12]}')
    (staging / 'other_mask').mkdir(parents=True)
    (staging / 'colored').mkdir()
    hashes = {}
    try:
        for index, image in enumerate(images, 1):
            base = read_image(image)
            state = read_page_state(paths, image, base.shape[:2])
            clean = _compose_overlay_preview(base, state['overlay'])
            other = state['other']
            colored = _overlay_mask_on_bgr(clean, other, alpha, color)
            filename = Path(image).stem + '.png'
            for rel, value in ((filename, clean), ('other_mask/' + filename, other),
                               ('colored/' + filename, colored)):
                target = staging / rel
                hashes[str(target.relative_to(staging))] = write_verified_png(target, value)
            print(f'  PAGE {index}/{len(images)} {Path(image).name}', flush=True)
        if len(hashes) != len(images) * 3:
            raise ValueError('Unexpected export file count')
        after = raw_hashes(folder)
        if after != before:
            raise RuntimeError('Authoritative raw files changed during export; staging not published')
        # All output has been decoded and compared pixel-for-pixel before publishing.
        if final.exists():
            raise FileExistsError(f'Export appeared during processing: {final}')
        os.replace(staging, final)
        return {'status': 'success', 'pages': len(images), 'png_files': len(hashes),
                'raw_hashes': before, 'export_hashes': hashes,
                'color_bgr': list(color), 'alpha': alpha, 'output': str(final)}
    except Exception:
        print(f'  Unpublished staging retained: {staging}', flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / 'report.json'
    previous = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
    folders = discover_projects(root)
    if not folders:
        raise ValueError('No v2 projects found')
    color, alpha = display_settings()
    report = {'root': str(root), 'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
              'preflight_only': args.preflight_only, 'projects': {}}
    queue = []
    for index, folder in enumerate(folders, 1):
        key = str(folder.relative_to(root))
        print(f'PREFLIGHT {index}/{len(folders)} {key}', flush=True)
        try:
            images = preflight_project(folder)
            queue.append((folder, images))
            report['projects'][key] = {'status': 'ready', 'pages': len(images)}
        except Exception as exc:
            report['projects'][key] = {'status': 'preflight_failed', 'error': str(exc)}
            print(f'  FAILED: {exc}', flush=True)
    save_json(report_dir / 'preflight.json', report)
    if args.preflight_only:
        return 1 if any(v['status'] != 'ready' for v in report['projects'].values()) else 0
    with (report_dir / 'progress.jsonl').open('a', encoding='utf-8') as log:
        for index, (folder, images) in enumerate(queue, 1):
            key = str(folder.relative_to(root))
            print(f'PROJECT {index}/{len(queue)} {key}', flush=True)
            started = time.monotonic()
            try:
                prior = previous.get('projects', {}).get(key)
                if verified_resume(folder, prior, color, alpha):
                    record = dict(prior, resumed=True)
                    print('  VERIFIED PREVIOUS SUCCESS', flush=True)
                else:
                    record = export_project(folder, images, color, alpha)
            except Exception as exc:
                record = {'status': 'failed', 'error': str(exc), 'pages': len(images)}
                print(f'  FAILED: {exc}', flush=True)
            record['elapsed_seconds'] = round(time.monotonic() - started, 3)
            report['projects'][key] = record
            log.write(json.dumps({'project': key, **record}, ensure_ascii=False) + '\n')
            log.flush()
            os.fsync(log.fileno())
            save_json(report_path, report)
            print(f"  RESULT {record['status']} {record['elapsed_seconds']}s", flush=True)
    report['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    report['summary'] = {status: sum(x['status'] == status for x in report['projects'].values())
                         for status in ('success', 'failed', 'preflight_failed', 'ready')}
    save_json(report_path, report)
    print(json.dumps(report['summary']), flush=True)
    return 0 if all(x['status'] == 'success' for x in report['projects'].values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
