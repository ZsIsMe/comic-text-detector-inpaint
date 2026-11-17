import os
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

import project_store as store
from scripts import batch_export_refinement as batch


class BatchExportRefinementTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name) / 'chapter'
        self.folder.mkdir()
        self.image = self.folder / '01.png'
        self.base = np.full((4, 5, 3), 100, np.uint8)
        cv2.imencode('.png', self.base)[1].tofile(str(self.image))
        self.paths = batch.paths_for(self.folder)
        store.update_project(self.paths['raw'], images=['01.png'])
        page = store.empty_page(self.base.shape[:2])
        page['overlay'][0, 0] = (255, 255, 255, 255)
        page['other'][1, 1] = 255
        store.save_page(self.paths, str(self.image), page)

    def test_export_preserves_authoritative_data_and_matches_gui_pixels(self):
        images = batch.preflight_project(self.folder)
        before = batch.raw_hashes(self.folder)
        color, alpha = (165, 110, 255), .38
        record = batch.export_project(self.folder, images, color, alpha)
        self.assertEqual(record['png_files'], 3)
        self.assertEqual(batch.raw_hashes(self.folder), before)
        output = Path(self.paths['export_pair'])
        clean = batch.read_image(output / '01.png')
        self.assertEqual(clean[0, 0].tolist(), [255, 255, 255])
        self.assertEqual(clean[1, 1].tolist(), [100, 100, 100])
        mask = batch.read_image(output / 'other_mask' / '01.png')
        self.assertEqual(int(mask[1, 1]), 255)
        colored = batch.read_image(output / 'colored' / '01.png')
        expected = (np.array([100, 100, 100]) * .62 + np.array(color) * .38).astype(np.uint8)
        np.testing.assert_array_equal(colored[1, 1], expected)
        self.assertTrue(batch.verified_resume(self.folder, record, color, alpha))
        (output / '01.png').write_bytes(b'broken')
        self.assertFalse(batch.verified_resume(self.folder, record, color, alpha))

    def test_changed_source_signature_is_rejected_before_export(self):
        stat = self.image.stat()
        os.utime(self.image, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000000))
        with self.assertRaisesRegex(ValueError, 'signature mismatch'):
            batch.preflight_project(self.folder)
        self.assertFalse(Path(self.paths['export_pair']).exists())

    def test_processed_empty_page_without_archive_exports_blank_mask(self):
        store.save_page(self.paths, str(self.image), store.empty_page(self.base.shape[:2]))
        self.assertFalse(store.page_path(self.paths, str(self.image)).exists())
        images = batch.preflight_project(self.folder)
        batch.export_project(self.folder, images, (165, 110, 255), .38)
        output = Path(self.paths['export_pair'])
        np.testing.assert_array_equal(batch.read_image(output / '01.png'), self.base)
        self.assertFalse(np.any(batch.read_image(output / 'other_mask' / '01.png')))


if __name__ == '__main__':
    unittest.main()
