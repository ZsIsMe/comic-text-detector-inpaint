from pathlib import Path
import tempfile
import unittest

import numpy as np

import project_store as store


class ReviewStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.image = root / '01.jpg'
        self.image.write_bytes(b'original image')
        self.paths = {'raw': str(root / 'ctd_inpainted' / 'raw')}
        self.page = store.empty_page((8, 12))

    def info(self):
        return store.load_project(self.paths['raw'])['pages'][self.image.name]

    def test_review_survives_identical_save_and_reopen(self):
        self.page['overlay'][1:3, 2:5] = (50, 60, 70, 255)
        first = store.save_page(self.paths, self.image, self.page)
        self.assertFalse(store.is_page_reviewed(first))
        marked = store.set_page_reviewed(self.paths, self.image)
        self.assertTrue(store.is_page_reviewed(marked))
        self.assertTrue(store.is_page_reviewed(store.save_page(self.paths, self.image, self.page)))
        self.assertTrue(store.is_page_reviewed(self.info()))
        self.assertEqual(first['state_digest'], self.info()['reviewed_digest'])

    def test_each_saved_component_invalidates_review_and_undo_restores_it(self):
        self.page['overlay'][1:3, 2:5] = (50, 60, 70, 255)
        store.save_page(self.paths, self.image, self.page)
        store.set_page_reviewed(self.paths, self.image)
        baseline = {key: value.copy() for key, value in self.page.items()}
        for key, index, value in (
            ('overlay', (1, 2, 0), 51),
            ('other', (5, 6), 255),
            ('edited', (5, 6), 255),
        ):
            with self.subTest(key=key):
                changed = {name: array.copy() for name, array in baseline.items()}
                changed[key][index] = value
                self.assertFalse(store.is_page_reviewed(store.save_page(
                    self.paths, self.image, changed)))
                self.assertTrue(store.is_page_reviewed(store.save_page(
                    self.paths, self.image, baseline)))

    def test_empty_saved_page_can_be_reviewed_and_source_change_blocks_marking(self):
        store.save_page(self.paths, self.image, self.page)
        self.assertFalse(store.page_path(self.paths, self.image).exists())
        self.assertTrue(store.is_page_reviewed(store.set_page_reviewed(
            self.paths, self.image)))
        self.assertFalse(store.is_page_reviewed(store.set_page_reviewed(
            self.paths, self.image, False)))
        self.image.write_bytes(b'changed original image')
        with self.assertRaisesRegex(ValueError, '原圖已變更'):
            store.set_page_reviewed(self.paths, self.image)

    def test_report_merge_keeps_review_and_missing_archive_cannot_be_marked(self):
        self.page['other'][2:4, 3:6] = 255
        store.save_page(self.paths, self.image, self.page)
        store.set_page_reviewed(self.paths, self.image)
        old = dict(self.info())
        store.merge_report(self.paths['raw'], {'pages': {self.image.name: old}})
        self.assertTrue(store.is_page_reviewed(self.info()))
        store.page_path(self.paths, self.image).unlink()
        with self.assertRaisesRegex(ValueError, '資料遺失'):
            store.set_page_reviewed(self.paths, self.image)

    def test_stale_worker_summary_cannot_restore_removed_review(self):
        store.save_page(self.paths, self.image, self.page)
        old = store.set_page_reviewed(self.paths, self.image)
        store.set_page_reviewed(self.paths, self.image, False)
        saved = store.save_page(self.paths, self.image, self.page, old)
        self.assertFalse(store.is_page_reviewed(saved))
        self.assertNotIn('reviewed_digest', saved)

    def test_existing_project_with_page_archive_can_be_reviewed_without_redetection(self):
        self.page['other'][2:4, 3:6] = 255
        store.save_page(self.paths, self.image, self.page)
        archive = store.page_path(self.paths, self.image)
        before = archive.read_bytes()
        project = store.load_project(self.paths['raw'])
        project['pages'][self.image.name].pop('state_digest')
        store.update_project(self.paths['raw'], pages=project['pages'])

        marked = store.set_page_reviewed(self.paths, self.image)
        self.assertTrue(store.is_page_reviewed(marked))
        self.assertEqual(archive.read_bytes(), before)

    def test_existing_empty_page_can_be_reviewed_without_creating_archive(self):
        store.save_page(self.paths, self.image, self.page)
        project = store.load_project(self.paths['raw'])
        project['pages'][self.image.name].pop('state_digest')
        store.update_project(self.paths['raw'], pages=project['pages'])

        with self.assertRaisesRegex(ValueError, '請先載入'):
            store.set_page_reviewed(self.paths, self.image)
        marked = store.set_page_reviewed(
            self.paths, self.image, page_state=self.page)
        self.assertTrue(store.is_page_reviewed(marked))
        self.assertFalse(store.page_path(self.paths, self.image).exists())


if __name__ == '__main__':
    unittest.main()
