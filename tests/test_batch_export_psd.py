"""Small saved-project exports using deterministic LaMa pixels and real PSD I/O."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from psd_tools import PSDImage

import project_store as store
import psd_export
from image_io import imwrite
from scripts import batch_export_psd as batch
from scripts.batch_export_common import paths_for, preflight_project, raw_hashes


def repair(image, mask):
    pixels = np.zeros((*mask.shape, 4), np.uint8)
    pixels[mask > 0] = (21, 88, 155, 255)
    return pixels


class BatchExportPsdTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.folder, self.image = self.project('2')

    def project(self, name, *, empty=False, depth=8):
        folder = self.root / name
        folder.mkdir()
        image = folder / '01.png'
        base = np.full((8, 10, 3), 140, np.uint8)
        if depth == 16:
            base = base.astype(np.uint16) * 257
        imwrite(image, base)
        paths = paths_for(folder)
        store.update_project(paths['raw'], images=['01.png'])
        page = store.empty_page(base.shape[:2])
        if not empty:
            page['overlay'][0:2, 0:3] = (255, 255, 255, 255)
            page['other'][4:6, 5:7] = 255
        store.save_page(paths, image, page)
        return folder, image

    def export(self, folder=None, previous=None):
        folder = folder or self.folder
        with patch.object(psd_export, 'inpaint_other', side_effect=repair):
            return batch.export_project(folder, preflight_project(folder), previous)

    def test_pngs_psd_layers_and_raw_preservation_with_verified_resume(self):
        before = raw_hashes(self.folder)
        with patch.object(psd_export, 'inpaint_other', side_effect=repair) as infer:
            record = batch.export_project(self.folder, preflight_project(self.folder))
        infer.assert_called_once()
        self.assertEqual(record['status'], 'success')
        self.assertEqual(record['png_files'], 3)
        self.assertEqual(len(record['created']), 1)
        self.assertEqual(raw_hashes(self.folder), before)
        self.assertTrue(batch.verified_resume(self.folder, record))
        self.assertEqual(set(record['psd_freshness'].values()), {'current'})
        self.assertEqual(record['psd_details'][str(Path('psds') / '01.psd')]['layers'],
                         ['bg', 'overlay-manual', 'lama_large_512px'])

    def test_empty_page_without_npz_does_not_load_model_and_16bit_is_preserved(self):
        folder, image = self.project('empty16', empty=True, depth=16)
        self.assertFalse(store.page_path(paths_for(folder), image).exists())
        with patch('lama_inpainter.LamaLargeInpainter', side_effect=AssertionError('no model')):
            record = batch.export_project(folder, preflight_project(folder))
        detail = next(iter(record['psd_details'].values()))
        self.assertEqual(detail['depth'], 16)
        self.assertEqual(detail['layers'], ['bg', 'lama_large_512px'])
        self.assertEqual(record['status'], 'success')

    def test_existing_hand_edited_psd_is_preserved_without_layer_requirements(self):
        record = self.export()
        psd_path = Path(record['created'][0])
        psd = PSDImage.open(psd_path)
        psd[-1].name = 'my retouch'
        psd[-1].visible = False
        psd.save(psd_path)
        before = psd_path.read_bytes()
        preserved = self.export()
        self.assertEqual(psd_path.read_bytes(), before)
        self.assertEqual(preserved['created'], [])
        self.assertEqual(len(preserved['skipped']), 1)
        self.assertEqual(preserved['status'], 'preserved')
        self.assertEqual(set(preserved['psd_freshness'].values()), {'unknown'})
        self.assertTrue(preserved['materials_current'])
        self.assertFalse(preserved['psds_current'])

    def test_changed_raw_keeps_old_psd_and_reports_stale_across_resume(self):
        record = self.export()
        path = Path(record['created'][0])
        before = path.read_bytes()
        paths = paths_for(self.folder)
        page = store.load_page(paths, self.image, (8, 10))
        page['overlay'][0, 0] = (10, 20, 30, 255)
        store.save_page(paths, self.image, page)
        self.assertFalse(batch.verified_resume(self.folder, record))
        stale = self.export(previous=record)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(stale['status'], 'preserved')
        self.assertEqual(set(stale['psd_freshness'].values()), {'stale'})
        again = self.export(previous=stale)
        self.assertEqual(set(again['psd_freshness'].values()), {'stale'})

    def test_missing_and_corrupt_outputs_invalidate_resume_without_overwriting_psd(self):
        record = self.export()
        output = Path(record['output'])
        png = output / 'solid' / '01.png'
        png.unlink()
        self.assertFalse(batch.verified_resume(self.folder, record))
        restored = self.export(previous=record)
        self.assertEqual(restored['status'], 'success')
        self.assertTrue(png.exists())
        psd = output / 'psds' / '01.psd'
        psd.unlink()
        self.assertFalse(batch.verified_resume(self.folder, restored))
        regenerated = self.export(previous=restored)
        self.assertEqual(len(regenerated['created']), 1)
        psd.write_bytes(b'corrupt hand edits')
        self.assertFalse(batch.verified_resume(self.folder, regenerated))
        with self.assertRaises(Exception):
            self.export(previous=regenerated)
        self.assertEqual(psd.read_bytes(), b'corrupt hand edits')

    def test_changed_repair_pixels_do_not_mark_preserved_psd_current(self):
        record = self.export()

        def different_repair(image, mask):
            pixels = repair(image, mask)
            pixels[mask > 0, 0] = 200
            return pixels

        with patch.object(psd_export, 'inpaint_other', side_effect=different_repair):
            changed = batch.export_project(self.folder, preflight_project(self.folder), record)
        self.assertEqual(changed['status'], 'preserved')
        self.assertEqual(set(changed['psd_freshness'].values()), {'stale'})

    def test_concurrent_raw_change_is_never_reported_as_success(self):
        def concurrent_edit(image, mask):
            store.update_project(paths_for(self.folder)['raw'], settings={'edited': True})
            return repair(image, mask)

        with patch.object(psd_export, 'inpaint_other', side_effect=concurrent_edit):
            with self.assertRaisesRegex(RuntimeError, 'raw data changed'):
                batch.export_project(self.folder, preflight_project(self.folder))

    def test_bad_signature_preflight_fails_without_creating_assets(self):
        stat = self.image.stat()
        os.utime(self.image, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000000))
        with self.assertRaisesRegex(ValueError, 'signature mismatch'):
            preflight_project(self.folder)
        report = self.root / 'reports'
        with patch.object(psd_export, 'export_project_assets', side_effect=AssertionError('no export')):
            self.assertEqual(batch.run(self.root, report, preflight_only=True), 1)
        self.assertFalse((self.folder / 'ctd_inpainted' / 'psd_assets').exists())
        self.assertEqual(json.loads((report / 'psd-preflight.json').read_text())['projects']['2']['status'],
                         'preflight_failed')

    def test_project_failure_continues_natural_order_and_completed_projects_resume(self):
        self.project('10')
        report_dir = self.root / 'reports'
        original = psd_export.export_project_assets
        calls = []

        def export(images, paths, **kwargs):
            calls.append(Path(paths['output']).parent.name)
            if calls[-1] == '2':
                raise RuntimeError('test export failure')
            return original(images, paths, **kwargs)

        with patch.object(psd_export, 'export_project_assets', side_effect=export), \
                patch.object(psd_export, 'inpaint_other', side_effect=repair):
            self.assertEqual(batch.run(self.root, report_dir), 1)
        self.assertEqual(calls, ['2', '10'])
        report = json.loads((report_dir / 'psd-report.json').read_text())
        self.assertEqual(report['projects']['2']['status'], 'failed')
        self.assertEqual(report['projects']['10']['status'], 'success')
        with patch.object(psd_export, 'export_project_assets', wraps=original) as run_export, \
                patch.object(psd_export, 'inpaint_other', side_effect=repair):
            self.assertEqual(batch.run(self.root, report_dir), 0)
        self.assertEqual(run_export.call_count, 1)
        report = json.loads((report_dir / 'psd-report.json').read_text())
        self.assertTrue(report['projects']['10']['resumed'])


if __name__ == '__main__':
    unittest.main()
