"""Serial PSD material/LaMa export from saved projects, preserving existing PSDs.

Exit 0: all PSDs are verified current. Exit 1: errors or preserved PSDs whose
freshness is stale/unknown. Exit 2: invalid arguments or startup failure.
PSD report filenames are prefixed with "psd-" and can share a report directory
with the refinement exporter.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

import project_store as store
import psd_export
from lama_config import MODEL_NAME
from scripts.batch_export_common import (
    discover_projects, paths_for, preflight_project, raw_hashes, read_image, save_json, sha256,
)

REPAIR_POLICY = 'nonempty-other-v1'


def verify_outputs(folder, images, created):
    """Validate current PNGs and read PSDs without imposing layers on hand edits."""
    from psd_tools import PSDImage

    paths = paths_for(folder)
    root = Path(paths['output']) / 'psd_assets'
    created = {Path(p).resolve() for p in created}
    hashes, details = {}, {}
    for index, source in enumerate(images, 1):
        source = Path(source)
        base = read_image(source)
        page = store.load_page(paths, source, base.shape[:2])
        pngs = {name: root / name / f'{source.stem}.png'
                for name in ('solid', 'other_mask')}
        solid, other = (read_image(pngs[name]) for name in pngs)
        if not np.array_equal(solid, page['overlay']) or not np.array_equal(other, page['other']):
            raise ValueError(f'Saved page and exported PNGs differ: {source.name}')
        needs_repair = bool(np.any(other))
        repair_path = root / MODEL_NAME / f'{source.stem}.png'
        if needs_repair:
            pngs[MODEL_NAME] = repair_path
            repair = read_image(repair_path)
            if repair.ndim != 3 or repair.shape[2] != 4:
                raise ValueError(f'Repair PNG is not BGRA: {source.name}')
            repair_rgba = cv2.cvtColor(repair, cv2.COLOR_BGRA2RGBA)
            psd_export._validate_repair(repair_rgba, other)
        destination = psd_export._existing(root / 'psds' / f'{source.stem}.psd')
        if destination is None or not destination.is_file():
            raise ValueError(f'PSD output is missing: {source.name}')
        psd = PSDImage.open(destination)
        # Decode document pixels as well as its header (also for preserved PSDs).
        if psd.numpy() is None:
            raise ValueError(f'PSD composite cannot be read: {destination}')
        is_new = destination.resolve() in created
        if is_new:
            expected_names = ['bg']
            if np.any(solid[:, :, 3]):
                expected_names.append('overlay-manual')
            if needs_repair:
                expected_names.append(MODEL_NAME)
            depth = 16 if base.dtype == np.uint16 else 8
            if (psd.size != (base.shape[1], base.shape[0]) or psd.depth != depth
                    or psd.color_mode.name != 'RGB' or psd.channels != 3
                    or [layer.name for layer in psd] != expected_names
                    or not all(layer.visible for layer in psd)):
                raise ValueError(f'New PSD structure differs from source: {destination}')
            for layer in psd:
                if layer.numpy() is None:
                    raise ValueError(f'New PSD layer cannot be decoded: {layer.name}')
            if needs_repair:
                maximum = (1 << depth) - 1
                repair_pixels = np.rint(psd[-1].numpy() * maximum).astype(
                    np.uint16 if depth == 16 else np.uint8)
                expected_repair = (repair_rgba.astype(np.uint16) * 257 if depth == 16 else repair_rgba)
                if not np.array_equal(repair_pixels, expected_repair):
                    raise ValueError(f'PSD repair layer differs from repair PNG: {destination}')
        for file in (*pngs.values(), destination):
            hashes[str(file.relative_to(root))] = sha256(file)
        unused_repair = None if needs_repair else psd_export._existing(repair_path)
        details[str(destination.relative_to(root))] = {
            'created': is_new, 'size': list(psd.size), 'depth': psd.depth,
            'layers': [layer.name for layer in psd],
            'materials': [str(p.relative_to(root)) for p in pngs.values()],
            'repair_required': needs_repair,
            'repair_policy_matches': needs_repair or not any(layer.name == MODEL_NAME for layer in psd),
            'omitted_repair_png': None if needs_repair else str(repair_path.relative_to(root)),
            'unused_repair_png': (str(unused_repair.relative_to(root))
                                  if unused_repair is not None and unused_repair.is_file() else None),
        }
        print(f'  VERIFY {index}/{len(images)} {source.name}', flush=True)
    if len(hashes) != len(images) * 3 + sum(d['repair_required'] for d in details.values()):
        raise ValueError('Unexpected number of current PNG/PSD outputs')
    return hashes, details


def verified_resume(folder, previous):
    """Missing/corrupt exports invalidate resume; hand-edited PSDs are preserved."""
    if not previous or previous.get('status') not in ('success', 'preserved'):
        return False
    if previous.get('repair_policy') != REPAIR_POLICY:
        return False
    if raw_hashes(folder) != previous.get('raw_hashes'):
        return False
    expected = previous.get('export_hashes', {})
    repair_pages = previous.get('repair_pages')
    if (not isinstance(repair_pages, int) or not 0 <= repair_pages <= previous.get('pages', 0)
            or not expected or len(expected) != previous.get('pages', 0) * 3 + repair_pages):
        return False
    root = Path(paths_for(folder)['output']) / 'psd_assets'
    try:
        if any(psd_export._existing(root / name) is not None
               for name in previous.get('omitted_repair_pngs', [])):
            return False
        if any(not (root / name).is_file() or sha256(root / name) != digest
               for name, digest in previous.get('unused_repair_hashes', {}).items()):
            return False
        return all((root / name).is_file() and sha256(root / name) == digest
                   for name, digest in expected.items())
    except OSError:
        return False


def export_project(folder, images, previous=None):
    before = raw_hashes(folder)
    paths = paths_for(folder)

    def progress(stage, page, pages, filename, completed, total):
        print(f'  {stage} {page}/{pages} {filename} [{completed}/{total}]', flush=True)

    try:
        print(f'  MATERIAL PNGs {len(images)} pages', flush=True)
        result = psd_export.export_project_assets(images, paths, overwrite=False, progress=progress)
        hashes, details = verify_outputs(folder, images, result['created'])
    finally:
        if raw_hashes(folder) != before:
            raise RuntimeError('Authoritative raw data changed during PSD export')
    prior_raw = (previous or {}).get('raw_hashes')
    prior_hashes = (previous or {}).get('export_hashes', {})
    prior_freshness = (previous or {}).get('psd_freshness', {})
    freshness = {}
    for name, detail in details.items():
        if detail['created']:
            freshness[name] = 'current'
        elif not detail['repair_policy_matches']:
            freshness[name] = 'stale'
        elif prior_raw and prior_raw != before:
            freshness[name] = 'stale'
        elif (prior_raw == before and prior_hashes.get(name) == hashes[name]
              and prior_freshness.get(name) == 'current'):
            freshness[name] = ('current' if all(prior_hashes.get(p) == hashes[p]
                                               for p in detail['materials']) else 'stale')
        else:
            # A previously diagnosed stale PSD stays stale until explicitly replaced.
            freshness[name] = ('stale' if prior_freshness.get(name) == 'stale' else 'unknown')
    current = all(value == 'current' for value in freshness.values())
    repair_pages = sum(d['repair_required'] for d in details.values())
    output = Path(paths['output']) / 'psd_assets'
    return {
        'status': 'success' if current else 'preserved', 'pages': len(images),
        'repair_policy': REPAIR_POLICY, 'repair_pages': repair_pages,
        'omitted_repair_pngs': [d['omitted_repair_png'] for d in details.values()
                               if d['omitted_repair_png'] is not None and d['unused_repair_png'] is None],
        'unused_repair_hashes': {d['unused_repair_png']: sha256(output / d['unused_repair_png'])
                                for d in details.values() if d['unused_repair_png'] is not None},
        'png_files': len(images) * 2 + repair_pages,
        'created': result['created'], 'skipped': result['skipped'],
        'raw_hashes': before, 'export_hashes': hashes, 'psd_details': details,
        'psd_freshness': freshness, 'materials_current': True, 'psds_current': current,
        'output': str(Path(paths['output']) / 'psd_assets'),
    }


def run(root, report_dir, *, preflight_only=False):
    root = Path(root).resolve(strict=True)
    report_dir = Path(report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / 'psd-report.json'
    previous = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
    if previous and previous.get('root') != str(root):
        raise ValueError('Report directory belongs to a different source root')
    folders = discover_projects(root)
    if not folders:
        raise ValueError('No v2 projects found')
    report = {'root': str(root), 'kind': 'psd',
              'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
              'preflight_only': preflight_only, 'projects': {}}
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
    save_json(report_dir / 'psd-preflight.json', report)
    if preflight_only:
        return 1 if any(v['status'] != 'ready' for v in report['projects'].values()) else 0
    with (report_dir / 'psd-progress.jsonl').open('a', encoding='utf-8') as log:
        for index, (folder, images) in enumerate(queue, 1):
            key = str(folder.relative_to(root))
            print(f'PROJECT {index}/{len(queue)} {key}', flush=True)
            started = time.monotonic()
            prior = previous.get('projects', {}).get(key)
            try:
                if verified_resume(folder, prior):
                    record = dict(prior, resumed=True, created_this_run=0, skipped_this_run=len(images))
                    print('  VERIFIED PREVIOUS OUTPUTS', flush=True)
                else:
                    record = export_project(folder, images, prior)
                    record['created_this_run'] = len(record['created'])
                    record['skipped_this_run'] = len(record['skipped'])
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
                         for status in ('success', 'preserved', 'failed', 'preflight_failed', 'ready')}
    report['summary']['created_psds'] = sum(x.get('created_this_run', 0) for x in report['projects'].values())
    report['summary']['skipped_psds'] = sum(x.get('skipped_this_run', 0) for x in report['projects'].values())
    save_json(report_path, report)
    print(json.dumps(report['summary']), flush=True)
    return 0 if all(x['status'] == 'success' for x in report['projects'].values()) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args(argv)
    try:
        return run(args.root, args.report_dir, preflight_only=args.preflight_only)
    except (OSError, ValueError) as exc:
        print(f'STARTUP FAILED: {exc}', file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
