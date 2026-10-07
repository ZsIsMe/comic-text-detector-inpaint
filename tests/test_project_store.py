from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import project_store as store
from detect_solid_inpaint_folder import (
    _ensure_dirs, read_page_state, save_page_edits, regenerate_image_from_mask,
    regenerate_image_from_imported_mask, _write_preview_pdf, load_report,
    export_psd_assets,
)


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.image = np.full((180, 220, 3), 240, np.uint8)
        self.mask = np.zeros(self.image.shape[:2], np.uint8)
        self.mask[60:100, 80:100] = 255
        self.image[self.mask > 0] = 0
        self.path = str(self.root/'01.png')
        cv2.imwrite(self.path, self.image)
        self.paths = _ensure_dirs(str(self.root))

    def tearDown(self):
        self.tmp.cleanup()

    def classify(self):
        with patch('detect_solid_inpaint_folder.detect_bubbles', return_value=([], 'detected')):
            return regenerate_image_from_mask(self.path, self.paths, self.mask)

    def test_page_is_authoritative_after_cache_deleted_and_no_duplicate_pngs(self):
        self.classify()
        before = read_page_state(self.paths, self.path)
        shutil.rmtree(Path(self.paths['raw'])/'cache')
        with patch('detect_solid_inpaint_folder.create_detector', side_effect=AssertionError('should not detect')):
            regenerate_image_from_mask(self.path, self.paths)
            after = read_page_state(self.paths, self.path)
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])
        self.assertEqual({p.name for p in Path(self.paths['raw']).iterdir()}, {'pages', 'project.json'})
        self.assertEqual(len(list((Path(self.paths['raw'])/'pages').iterdir())), 1)
        self.assertFalse(list(Path(self.paths['raw']).rglob('*.png')))
        self.assertFalse((Path(self.paths['output'])/'preview_report.pdf').exists())

    def test_erasures_and_manual_other_survive_reclassification(self):
        self.classify()
        state = read_page_state(self.paths, self.path)
        solid = state['overlay'][:, :, 3].copy()
        other = state['other'].copy()
        solid[65:75, 83:88] = 0
        solid[80:90, 90:95] = 0
        other[80:90, 90:95] = 255
        save_page_edits(self.path, self.paths, solid, other)
        self.classify()
        state = read_page_state(self.paths, self.path)
        self.assertFalse(np.any(state['overlay'][65:75, 83:88, 3]))
        self.assertFalse(np.any(state['other'][65:75, 83:88]))
        self.assertTrue(np.all(state['other'][80:90, 90:95] == 255))
        self.assertTrue(np.all(state['edited'][65:75, 83:88] == 255))

    def test_different_colors_survive_edits_and_reopen(self):
        state = store.empty_page(self.mask.shape)
        state['overlay'][30:50, 30:50] = (200, 210, 220, 255)
        state['overlay'][30:50, 50:70] = (90, 100, 110, 255)
        store.save_page(self.paths, self.path, state)
        other = np.zeros_like(self.mask)
        other[120:130, 140:150] = 255
        save_page_edits(self.path, self.paths, state['overlay'][:, :, 3], other)
        after = read_page_state(self.paths, self.path)
        np.testing.assert_array_equal(after['overlay'], state['overlay'])

    def test_extending_saved_fills_uses_nearest_color_without_flattening_neighbors(self):
        state = store.empty_page(self.mask.shape)
        state['overlay'][40:80, 40:60] = (200, 210, 220, 255)
        state['overlay'][40:80, 70:90] = (90, 100, 110, 255)
        store.save_page(self.paths, self.path, state)
        solid = state['overlay'][:, :, 3].copy();solid[40:80, 60:70] = 255
        page, _ = save_page_edits(self.path, self.paths, solid, state['other'])
        np.testing.assert_array_equal(page['overlay'][50, 60], [200, 210, 220, 255])
        np.testing.assert_array_equal(page['overlay'][50, 69], [90, 100, 110, 255])
        np.testing.assert_array_equal(page['overlay'][40:80, 40:60], state['overlay'][40:80, 40:60])

    def test_invalid_overlap_and_unavailable_color_do_not_overwrite_page(self):
        self.classify()
        file = store.page_path(self.paths, self.path)
        before = file.read_bytes()
        with self.assertRaisesRegex(ValueError, '重疊'):
            save_page_edits(self.path, self.paths, self.mask, self.mask)
        self.assertEqual(before, file.read_bytes())
        with self.assertRaisesRegex(ValueError, '取色'):
            save_page_edits(self.path, self.paths, np.full_like(self.mask, 255), np.zeros_like(self.mask), state=store.empty_page(self.mask.shape))
        self.assertEqual(before, file.read_bytes())
        page, _ = save_page_edits(self.path, self.paths, np.full_like(self.mask, 255),
                                  np.zeros_like(self.mask), fill_color=(80, 90, 100))
        np.testing.assert_array_equal(page['overlay'][0, 0], [80, 90, 100, 255])

    def test_empty_page_has_no_archives_but_erase_lock_is_saved(self):
        with patch('detect_solid_inpaint_folder.detect_bubbles') as detect:
            regenerate_image_from_mask(self.path, self.paths, self.mask*0)
        detect.assert_not_called()
        self.assertTrue(store.has_page(self.paths, self.path))
        self.assertFalse(list(Path(self.paths['raw']).rglob('*.npz')))
        self.classify()
        save_page_edits(self.path, self.paths, self.mask*0, self.mask*0)
        self.assertTrue(store.page_path(self.paths, self.path).exists())
        self.classify()
        self.assertFalse(np.any(read_page_state(self.paths, self.path)['overlay']))

    def test_cache_updates_preserve_other_payloads_and_recover_from_bad_cache(self):
        file = store.cache_path(self.paths, self.path)
        store.write_archive(file, {'text_mask': self.mask})  # legacy cache
        store.update_cache_file(file, bubble_json=np.array('{}'))
        store.update_cache_file(file, sample=self.mask)
        self.assertEqual(set(store.read_cache_file(file)), {'bubble_json', 'sample'})
        file.write_bytes(b'broken')
        self.assertEqual(store.read_cache_file(file), {})
        store.update_cache_file(file, sample=self.mask)
        np.testing.assert_array_equal(store.read_cache_file(file)['sample'], self.mask)

    def test_full_filenames_do_not_collide(self):
        other_path = str(self.root/'01.jpg')
        cv2.imwrite(other_path, self.image)
        self.assertNotEqual(store.page_path(self.paths, self.path), store.page_path(self.paths, other_path))

    def test_changed_source_and_missing_authoritative_file_are_errors(self):
        self.classify()
        file = store.page_path(self.paths, self.path)
        file.unlink()
        with self.assertRaisesRegex(ValueError, '遺失'):
            read_page_state(self.paths, self.path)
        self.classify()
        cv2.imwrite(self.path, np.full_like(self.image, 10))
        with self.assertRaisesRegex(ValueError, '原圖已變更'):
            read_page_state(self.paths, self.path)

    def test_failed_atomic_write_keeps_prior_page(self):
        self.classify()
        before = store.page_path(self.paths, self.path).read_bytes()
        state = read_page_state(self.paths, self.path)
        with patch('project_store.os.replace', side_effect=OSError('disk error')):
            with self.assertRaises(OSError):
                store.save_page(self.paths, self.path, state)
        self.assertEqual(before, store.page_path(self.paths, self.path).read_bytes())

    def test_import_intersection_keeps_colors_and_locks_erased_pixels(self):
        self.classify()
        before = read_page_state(self.paths, self.path)
        imported = self.root/'imported';imported.mkdir()
        limit = np.full_like(self.mask, 255);limit[:80] = 0
        cv2.imwrite(str(imported/'01.png'), limit)
        regenerate_image_from_imported_mask(self.path, self.paths, str(imported), 'intersect')
        after = read_page_state(self.paths, self.path)
        self.assertFalse(np.any(after['overlay'][:80]))
        np.testing.assert_array_equal(after['overlay'][80:], before['overlay'][80:])
        self.classify()
        self.assertFalse(np.any(read_page_state(self.paths, self.path)['overlay'][:80]))

    def test_pdf_is_explicit_export_outside_raw(self):
        self.classify()
        path = _write_preview_pdf([self.path], self.paths, load_report(self.paths))
        self.assertEqual(Path(path).parent, Path(self.paths['output']))
        self.assertTrue(Path(path).read_bytes().startswith(b'%PDF'))
        self.assertFalse(list(Path(self.paths['raw']).rglob('*.png')))

    def test_psd_assets_are_explicit_exports_of_saved_colors(self):
        self.classify()
        root = Path(export_psd_assets([self.path], self.paths))
        page = read_page_state(self.paths, self.path)
        np.testing.assert_array_equal(cv2.imread(str(root/'solid/01.png'), -1), page['overlay'])
        np.testing.assert_array_equal(cv2.imread(str(root/'other_mask/01.png'), 0), page['other'])
        self.assertFalse(list(Path(self.paths['raw']).rglob('*.png')))

    def test_missing_cache_rebuild_uses_saved_detector_and_keeps_edits(self):
        self.classify()
        page = read_page_state(self.paths, self.path)
        solid = page['overlay'][:, :, 3].copy();solid[70:80, 82:90] = 0
        save_page_edits(self.path, self.paths, solid, page['other'])
        shutil.rmtree(Path(self.paths['raw'])/'cache')
        store.update_project(self.paths['raw'], detector='rfdetr', detector_params={'device': 'cpu'})
        with patch('detect_solid_inpaint_folder.create_detector') as factory, \
                patch('detect_solid_inpaint_folder.detect_bubbles', return_value=([], 'detected')):
            factory.return_value.return_value = (None, self.mask, None)
            regenerate_image_from_mask(self.path, self.paths, reclassify=True)
            factory.assert_called_once_with('rfdetr', {'device': 'cpu'})
        self.assertFalse(np.any(read_page_state(self.paths, self.path)['overlay'][70:80, 82:90]))
        self.assertNotIn('text_mask', store.read_cache_file(store.cache_path(self.paths, self.path)))

    def test_reclassify_ignores_legacy_text_mask_cache(self):
        self.classify()
        cache = store.cache_path(self.paths, self.path)
        legacy = store.read_cache_file(cache)
        legacy['text_mask'] = np.zeros_like(self.mask)
        store.write_archive(cache, legacy)
        with patch('detect_solid_inpaint_folder.create_detector') as factory, \
                patch('detect_solid_inpaint_folder.detect_bubbles', return_value=([], 'detected')):
            factory.return_value.return_value = (None, self.mask, None)
            regenerate_image_from_mask(self.path, self.paths, reclassify=True)
            factory.assert_called_once()
        self.assertNotIn('text_mask', store.read_cache_file(cache))

    def test_legacy_project_is_not_silently_overwritten(self):
        legacy = self.root/'legacy/ctd_inpainted/raw/mask'
        legacy.mkdir(parents=True)
        file = legacy/'01.png';file.write_bytes(b'original')
        with self.assertRaisesRegex(ValueError, '舊格式'):
            _ensure_dirs(str(self.root/'legacy'))
        self.assertEqual(file.read_bytes(), b'original')

    def test_missing_import_does_not_mark_unprocessed_page_complete(self):
        report = regenerate_image_from_imported_mask(self.path, self.paths, str(self.root/'missing'), 'replace')
        self.assertFalse(report['processed'])
        self.assertFalse(store.has_page(self.paths, self.path))
        self.assertFalse(store.page_path(self.paths, self.path).exists())


class EditorTests(unittest.TestCase):
    def setUp(self):
        ProjectTests.setUp(self)
        self.classify = lambda: ProjectTests.classify(self)
        self.classify()
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import QSettings
        from solid_inpaint_ui import MainWindow
        self.app = QApplication.instance() or QApplication([])
        settings = QSettings(str(self.root/'ui.ini'), QSettings.Format.IniFormat)
        self.settings_patch = patch('solid_inpaint_ui.QSettings', return_value=settings)
        self.sample_patch = patch.object(MainWindow, 'queue_background_sample')
        self.settings_patch.start();self.sample_patch.start()
        self.window = MainWindow()
        self.window.load_folder(str(self.root))

    def tearDown(self):
        self.window.close()
        self.sample_patch.stop();self.settings_patch.stop()
        ProjectTests.tearDown(self)

    def test_rect_local_intersection_extracts_dark_components_and_skips_dots(self):
        from PySide6.QtCore import Qt
        view = self.window.mask_view
        source = np.full((24, 32, 3), 220, np.uint8)
        source[6:13, 5:7] = 80
        source[6:13, 11:13] = 150
        source[7, 16] = 0
        source[10, 18] = 0
        old = np.zeros(source.shape[:2], np.uint8)
        old[2:18, 2:21] = 255
        old[2:6, 25:29] = 255
        selection = np.zeros_like(old, bool)
        selection[4:16, 4:20] = True
        view.set_source_image(source)
        view.set_tool('rect')
        view.set_selection_combine_mode('local_intersect')
        view.set_mask(old, old.shape)
        view._apply_selection(selection, Qt.MouseButton.LeftButton)
        expected = np.zeros_like(old)
        expected[4:16, 4:20] = 255
        expected[2:6, 25:29] = 255
        np.testing.assert_array_equal(view.mask, expected)

        view.set_local_intersect_dark_refine(True)
        view.set_local_intersect_min_area(5)
        view.set_local_intersect_dark_threshold(100)
        view.set_mask(old, old.shape)
        view._apply_selection(selection, Qt.MouseButton.LeftButton)
        expected[:] = 0
        expected[6:13, 5:7] = 255
        expected[2:6, 25:29] = 255
        np.testing.assert_array_equal(view.mask, expected)

        view.set_local_intersect_dark_threshold(160)
        view.set_mask(old, old.shape)
        view._apply_selection(selection, Qt.MouseButton.LeftButton)
        expected[6:13, 11:13] = 255
        np.testing.assert_array_equal(view.mask, expected)

        # No qualifying stroke leaves the old component alone.
        source[:] = 220
        source[7, 16] = 0
        view.set_source_image(source)
        view.set_mask(old, old.shape)
        view._apply_selection(selection, Qt.MouseButton.LeftButton)
        np.testing.assert_array_equal(view.mask, old)

    def test_dark_refine_uses_original_pixels_for_clipped_roi_and_rect_only(self):
        from PySide6.QtCore import Qt
        view = self.window.mask_view
        source = np.full((30, 30, 3), 220, np.uint8)
        source[13:17, 14:16] = 70
        old = np.zeros(source.shape[:2], np.uint8)
        old[8:22, 8:22] = 255
        selection = np.zeros_like(old, bool)
        selection[10:20, 10:20] = True
        view.set_source_image(source)
        view.set_mask(old, old.shape)
        view.set_selection_combine_mode('local_intersect')
        view.set_local_intersect_dark_refine(True)
        view.set_local_intersect_dark_threshold(100)
        view.set_local_intersect_min_area(5)
        view.set_edit_clip_rect((10, 10, 20, 20))
        view.set_tool('rect')
        view._apply_selection(selection, Qt.MouseButton.LeftButton)
        expected = old.copy()
        expected[10:20, 10:20] = 0
        expected[13:17, 14:16] = 255
        np.testing.assert_array_equal(view.mask, expected)

        view.set_edit_clip_rect(None)
        view.set_mask(old, old.shape)
        view.set_local_intersect_offset(2)
        view._apply_selection(selection, Qt.MouseButton.LeftButton)
        self.assertFalse(np.any(view.mask[~selection]))
        self.assertFalse(np.any(view.mask[selection & (old == 0)]))
        self.assertGreater(np.count_nonzero(view.mask), 8)

        view.set_local_intersect_offset(0)
        for tool in ('magic', 'lasso'):
            view.set_tool(tool)
            view.set_mask(old, old.shape)
            view._apply_selection(selection, Qt.MouseButton.LeftButton)
            np.testing.assert_array_equal(view.mask > 0, selection)

    def test_dark_refine_rect_saves_and_undo_restores_page(self):
        from PySide6.QtCore import Qt
        w = self.window
        source = np.full(self.image.shape, 220, np.uint8)
        source[70:80, 85:90] = 50
        w.mask_view.set_source_image(source)
        w.set_edit_tool('rect')
        w.set_selection_combine_mode('local_intersect')
        w.local_intersect_dark_checkbox.setChecked(True)
        w.local_intersect_dark_threshold_spinbox.setValue(100)
        w.local_intersect_min_area_spinbox.setValue(5)
        before = read_page_state(self.paths, self.path)
        self.assertTrue(w.mask_view._apply_rect((80, 60), (99, 99), Qt.MouseButton.LeftButton))
        w.mask_view.editStarted.emit()
        w.mask_view.maskEdited.emit(w.mask_view.mask.copy())
        saved = read_page_state(self.paths, self.path)
        self.assertEqual(np.count_nonzero(saved['overlay'][:, :, 3]), 50)
        self.assertTrue(np.all(saved['overlay'][70:80, 85:90, 3] == 255))
        w.undo_mask()
        undone = read_page_state(self.paths, self.path)
        np.testing.assert_array_equal(undone['overlay'], before['overlay'])
        w.redo_mask()
        redone = read_page_state(self.paths, self.path)
        np.testing.assert_array_equal(redone['overlay'], saved['overlay'])

    def test_magic_drag_samples_path_once_and_saves_one_undo(self):
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtTest import QTest
        w = self.window
        w.set_edit_mode('manual_other')
        w.set_edit_tool('magic')
        w.set_selection_combine_mode('add')
        source = np.full(self.image.shape, 220, np.uint8)
        source[68:73, 84:87] = 80
        source[68:73, 94:97] = 80
        w.mask_view.set_source_image(source)
        w.magic_scope_spinbox.setValue(32)
        w.magic_dwell_spinbox.setValue(0)
        w.magic_tolerance_slider.setValue(0)
        w.show()
        view = w.mask_view
        start = view.mapFromScene(QPointF(85, 70))
        end = view.mapFromScene(QPointF(95, 70))
        self.assertEqual(view.image_point_from_view(start), (85, 70))
        before_undo = len(w.undo_stack)
        viewport = view.viewport()
        QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(viewport, end)
        self.assertEqual(len(w.undo_stack), before_undo)
        self.assertFalse(np.any(w.current_manual_other))
        QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=end)
        self.assertEqual(len(w.undo_stack), before_undo + 1)
        saved = read_page_state(self.paths, self.path)['other']
        self.assertEqual(np.count_nonzero(saved), 30)
        w.undo_mask()
        self.assertFalse(np.any(read_page_state(self.paths, self.path)['other']))
        w.redo_mask()
        np.testing.assert_array_equal(read_page_state(self.paths, self.path)['other'], saved)

    def test_magic_drag_to_outside_roi_still_samples_up_to_boundary(self):
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtTest import QTest
        w = self.window
        w.set_edit_mode('manual_other')
        w.set_edit_tool('magic')
        source = np.full(self.image.shape, 220, np.uint8)
        source[68:73, 84:87] = 80
        source[68:73, 97:100] = 80
        source[68:73, 100:104] = 80
        view = w.mask_view
        view.set_source_image(source)
        view.set_edit_clip_rect((80, 60, 100, 90))
        view.set_magic_scope_px(32)
        view.set_magic_dwell_ms(0)
        view.set_magic_tolerance(0)
        w.show()
        start = view.mapFromScene(QPointF(85, 70))
        outside = view.mapFromScene(QPointF(105, 70))
        viewport = view.viewport()
        QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(viewport, outside)
        QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=outside)
        saved = read_page_state(self.paths, self.path)['other']
        self.assertTrue(np.all(saved[68:73, 84:87] == 255))
        self.assertTrue(np.all(saved[68:73, 97:100] == 255))
        self.assertFalse(np.any(saved[:, 100:]))

    def test_magic_fixed_color_scope_and_preview_are_crop_bound(self):
        from solid_inpaint_ui import MaskEditorView
        view = MaskEditorView()
        source = np.full((100, 100, 3), 220, np.uint8)
        source[49:52, 49:52] = 100
        source[49:52, 58:61] = 100
        source[49:52, 63:65] = 100
        source[49:52, 65:68] = 115
        old = np.full((100, 100), 255, np.uint8)
        view.set_source_image(source)
        view.set_mask(old, old.shape)
        view.set_tool('magic')
        view.set_selection_combine_mode('subtract')
        view.set_magic_tolerance(10)
        view.set_magic_scope_px(32)
        view.set_magic_dwell_ms(0)
        rect = view._magic_scope_rect((50, 50))
        self.assertEqual(rect, (34, 34, 66, 66))
        with patch('solid_inpaint_ui.cv2.floodFill', wraps=cv2.floodFill) as flood:
            view._magic_preview_pending_point = (50, 50)
            view._refresh_magic_preview()
        self.assertEqual(flood.call_args.args[0].shape[:2], (32, 32))
        self.assertLessEqual(view._magic_preview_item.pixmap().width(), 32)

        view._start_magic_stroke((50, 50))
        view._advance_magic_stroke((60, 50))
        view._advance_magic_stroke((66, 50))  # Cross the square boundary.
        view._finish_magic_stroke()
        self.assertFalse(np.any(view.mask[49:52, 49:52]))
        self.assertFalse(np.any(view.mask[49:52, 58:61]))
        self.assertFalse(np.any(view.mask[49:52, 63:65]))
        self.assertTrue(np.all(view.mask[49:52, 65:68] == 255))
        self.assertTrue(np.all(view.mask[:34] == 255))
        self.assertTrue(np.all(view.mask[:, 66:] == 255))

        view.set_mask(old, old.shape)
        view.set_edit_clip_rect((40, 40, 56, 56))
        self.assertEqual(view._magic_scope_rect((50, 50)), (40, 40, 56, 56))
        selection = view._magic_selection_at((50, 50))
        self.assertFalse(np.any(selection[:40]))
        self.assertFalse(np.any(selection[56:]))
        self.assertFalse(np.any(selection[:, :40]))
        self.assertFalse(np.any(selection[:, 56:]))

        # A dark path outside the ROI cannot join two regions inside it.
        u_shape = np.full((10, 10, 3), 220, np.uint8)
        u_shape[2, 3:8] = 80
        u_shape[3:8, 3] = 80
        u_shape[3:8, 7] = 80
        view.set_source_image(u_shape)
        view.set_mask(np.zeros((10, 10), np.uint8), (10, 10))
        view.set_edit_clip_rect((3, 3, 8, 8))
        view.set_magic_tolerance(0)
        clipped = view._magic_selection_at((3, 5))
        self.assertTrue(np.all(clipped[3:8, 3]))
        self.assertFalse(np.any(clipped[3:8, 7]))

    def test_magic_stroke_cancel_emit_modes_and_intersection(self):
        from PySide6.QtCore import Qt
        from solid_inpaint_ui import MaskEditorView
        source = np.full((80, 80, 3), 220, np.uint8)
        source[38:42, 38:42] = 80
        source[38:42, 48:52] = 80
        old = np.zeros((80, 80), np.uint8)
        old[35:45, 35:45] = 255
        old[35:45, 45:55] = 255
        old[5:10, 5:10] = 255
        view = MaskEditorView()
        view.set_source_image(source)
        view.set_mask(old, old.shape)
        view.set_tool('magic')
        view.set_magic_scope_px(32)
        view.set_magic_dwell_ms(0)
        view.set_magic_tolerance(0)
        view.set_selection_combine_mode('local_intersect')
        rect = view._magic_scope_rect((39, 39))
        local = view._magic_local_selection((39, 39), rect)
        additions, removals, fallback = view._magic_preview_masks(local, rect)
        self.assertEqual(additions.shape, (32, 32))
        self.assertEqual(removals.shape, (32, 32))
        self.assertFalse(np.any(fallback))
        np.testing.assert_array_equal(view.mask, old)
        view._start_magic_stroke((39, 39))
        view._advance_magic_stroke((49, 39))
        view.cancel_magic_stroke()
        np.testing.assert_array_equal(view.mask, old)
        view._start_magic_stroke((39, 39))
        view._advance_magic_stroke((49, 39))
        view._finish_magic_stroke()
        self.assertTrue(np.all(view.mask[5:10, 5:10] == 255))
        self.assertFalse(np.any(view.mask[35:45, 35:55] & (source[35:45, 35:55, 0] == 220)))
        self.assertTrue(np.all(view.mask[38:42, 38:42] == 255))
        self.assertTrue(np.all(view.mask[38:42, 48:52] == 255))

        view.set_mask(old, old.shape)
        view.set_selection_combine_mode('selection_inner')
        ring = np.ones((32, 32), dtype=bool)
        ring[10:14, 10:14] = False
        view._apply_magic_selection_local(ring, rect, Qt.MouseButton.LeftButton)
        self.assertTrue(np.all(view.mask[5:10, 5:10] == 255))
        self.assertTrue(np.all(view.mask[rect[1] + 10:rect[1] + 14,
                                          rect[0] + 10:rect[0] + 14] == 255))

        selections = []
        view.selectionCreated.connect(lambda selection: selections.append(selection))
        for mode in ('transfer_from_other', 'ctd_detect_selection',
                     'local_edit_selection'):
            view.set_selection_combine_mode(mode)
            view._start_magic_stroke((39, 39))
            view._advance_magic_stroke((49, 39))
            view._finish_magic_stroke()
            self.assertEqual(len(selections), 1)
            self.assertEqual(np.count_nonzero(selections.pop()), 32)

    def test_magic_scope_setting_restores_and_local_dialog_syncs(self):
        from PySide6.QtWidgets import QDialog
        from solid_inpaint_ui import LocalEditDialog, MainWindow
        w = self.window
        w.magic_scope_spinbox.setValue(768)
        w.magic_dwell_spinbox.setValue(225)
        self.assertEqual(w.mask_view.magic_scope_px, 768)
        self.assertEqual(w.mask_view.magic_dwell_ms, 225)
        self.assertEqual(int(w.settings.value('magic_scope_px')), 768)
        self.assertEqual(int(w.settings.value('magic_dwell_ms')), 225)
        another = MainWindow()
        try:
            self.assertEqual(another.magic_scope_spinbox.value(), 768)
            self.assertEqual(another.mask_view.magic_scope_px, 768)
            self.assertEqual(another.magic_dwell_spinbox.value(), 225)
        finally:
            another.close()
        selection = np.zeros_like(self.mask, bool)
        selection[60:100, 80:100] = True

        def inspect_dialog(dialog):
            self.assertEqual(dialog.magic_scope_spinbox.value(), 768)
            self.assertEqual(dialog.magic_dwell_spinbox.value(), 225)
            dialog.magic_scope_spinbox.setValue(512)
            dialog.magic_dwell_spinbox.setValue(75)
            self.assertEqual(dialog.view.magic_scope_px, 512)
            self.assertEqual(dialog.view.magic_dwell_ms, 75)
            self.assertEqual(w.magic_scope_spinbox.value(), 512)
            self.assertEqual(w.magic_dwell_spinbox.value(), 75)
            return QDialog.DialogCode.Rejected

        with patch.object(LocalEditDialog, 'exec', inspect_dialog):
            w.open_local_edit_dialog(selection)
        self.assertEqual(int(w.settings.value('magic_scope_px')), 512)
        self.assertEqual(int(w.settings.value('magic_dwell_ms')), 75)

    def test_magic_dwell_waits_for_endpoint_and_scope_follows_cursor(self):
        from PySide6.QtTest import QTest
        from solid_inpaint_ui import MaskEditorView
        source = np.full((80, 100, 3), 220, np.uint8)
        source[38:42, 19:23] = 80
        source[38:42, 39:43] = 80
        source[38:42, 59:64] = 80
        blank = np.zeros(source.shape[:2], np.uint8)
        view = MaskEditorView()
        view.set_source_image(source)
        view.set_mask(blank, blank.shape)
        view.set_tool('magic')
        view.set_magic_scope_px(32)
        view.set_magic_tolerance(0)
        view.set_magic_dwell_ms(50)
        view._start_magic_stroke((20, 40))
        self.assertTrue(np.all(view._magic_stroke_covered[38:42, 19:23]))
        view._advance_magic_stroke((60, 40))
        self.assertEqual(view._magic_stroke_rect, (44, 24, 76, 56))
        self.assertTrue(view._magic_candidate_item.isVisible())
        self.assertFalse(np.any(view._magic_stroke_covered[38:42, 59:64]))
        QTest.qWait(75)
        self.assertTrue(np.all(view._magic_stroke_covered[38:42, 59:64]))
        self.assertFalse(view._magic_candidate_item.isVisible())
        view._finish_magic_stroke()
        self.assertTrue(np.all(view.mask[38:42, 19:23]))
        self.assertTrue(np.all(view.mask[38:42, 59:64]))
        self.assertFalse(np.any(view.mask[38:42, 39:43]))

        # With a long connected stroke, moving the scope extends confirmed pixels.
        source[:] = 220
        source[39:41, 20:81] = 80
        view.set_source_image(source)
        view.set_mask(blank, blank.shape)
        view._start_magic_stroke((20, 40))
        view._advance_magic_stroke((40, 40))
        view._advance_magic_stroke((60, 40))
        view._finish_magic_stroke()
        self.assertTrue(np.all(view.mask[39:41, 20:76]))
        self.assertFalse(np.any(view.mask[39:41, 76:81]))

    def test_magic_dwell_quick_pass_release_and_candidate_reset(self):
        from PySide6.QtTest import QTest
        from solid_inpaint_ui import MaskEditorView
        source = np.full((80, 100, 3), 220, np.uint8)
        source[38:42, 19:23] = 80
        source[38:42, 40:51] = 80
        blank = np.zeros(source.shape[:2], np.uint8)
        view = MaskEditorView()
        view.set_source_image(source)
        view.set_mask(blank, blank.shape)
        view.set_tool('magic')
        view.set_magic_scope_px(32)
        view.set_magic_tolerance(0)
        view.set_magic_dwell_ms(50)
        view._start_magic_stroke((20, 40))
        view._advance_magic_stroke((42, 40))
        QTest.qWait(20)
        view._advance_magic_stroke((48, 40))  # Same component, new scope.
        QTest.qWait(40)
        self.assertTrue(np.all(view._magic_stroke_covered[38:42, 40:51]))
        view._finish_magic_stroke()

        view.set_mask(blank, blank.shape)
        view._start_magic_stroke((20, 40))
        view._advance_magic_stroke((42, 40))
        QTest.qWait(20)
        view._advance_magic_stroke((70, 40))  # White cancels the candidate.
        QTest.qWait(40)
        self.assertFalse(np.any(view._magic_stroke_covered[38:42, 40:51]))
        view._advance_magic_stroke((42, 40))
        QTest.qWait(25)
        self.assertFalse(np.any(view._magic_stroke_covered[38:42, 40:51]))
        view._finish_magic_stroke()  # Early release never adds pending.
        self.assertFalse(np.any(view.mask[38:42, 40:51]))

    def test_magic_dwell_qtest_stationary_release_saves_once(self):
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtTest import QTest
        w = self.window
        w.set_edit_mode('manual_other')
        w.set_edit_tool('magic')
        w.magic_scope_spinbox.setValue(32)
        w.magic_dwell_spinbox.setValue(50)
        w.magic_tolerance_slider.setValue(0)
        source = np.full(self.image.shape, 220, np.uint8)
        source[68:73, 84:87] = 80
        source[68:73, 94:97] = 80
        view = w.mask_view
        view.set_source_image(source)
        w.show()
        start = view.mapFromScene(QPointF(85, 70))
        second = view.mapFromScene(QPointF(95, 70))
        viewport = view.viewport()
        before_undo = len(w.undo_stack)
        QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(viewport, second)
        self.assertEqual(len(w.undo_stack), before_undo)
        QTest.qWait(75)
        QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=second)
        saved = read_page_state(self.paths, self.path)['other']
        self.assertEqual(np.count_nonzero(saved), 30)
        self.assertEqual(len(w.undo_stack), before_undo + 1)
        w.undo_mask()
        self.assertFalse(np.any(read_page_state(self.paths, self.path)['other']))
        w.redo_mask()
        np.testing.assert_array_equal(read_page_state(self.paths, self.path)['other'], saved)

    def test_magic_dwell_leave_and_lifecycle_cancel_timer(self):
        from PySide6.QtTest import QTest
        from solid_inpaint_ui import MaskEditorView
        source = np.full((80, 100, 3), 220, np.uint8)
        source[38:42, 19:23] = 80
        source[38:42, 59:63] = 80
        blank = np.zeros(source.shape[:2], np.uint8)
        view = MaskEditorView()
        view.set_source_image(source)
        view.set_mask(blank, blank.shape)
        view.set_tool('magic')
        view.set_magic_dwell_ms(50)
        view.set_magic_scope_px(32)
        view.set_magic_tolerance(0)
        view._start_magic_stroke((20, 40))
        view._advance_magic_stroke((60, 40))
        QTest.qWait(25)
        view._advance_magic_stroke(None)  # Leave viewport.
        QTest.qWait(40)
        view._advance_magic_stroke((60, 40))
        QTest.qWait(25)
        self.assertFalse(np.any(view._magic_stroke_covered[38:42, 59:63]))
        QTest.qWait(40)
        self.assertTrue(np.all(view._magic_stroke_covered[38:42, 59:63]))
        view.cancel_magic_stroke()

        for cancel in (
            lambda: view.set_tool('rect'),
            lambda: view.set_source_image(source),
            lambda: view.set_edit_clip_rect((0, 0, 80, 80)),
            lambda: view.set_mask(blank, blank.shape),
        ):
            view.set_tool('magic')
            view._start_magic_stroke((20, 40))
            view._advance_magic_stroke((60, 40))
            self.assertTrue(view._magic_dwell_timer.isActive())
            cancel()
            QTest.qWait(65)
            self.assertIsNone(view._magic_stroke_rect)
            self.assertFalse(view._magic_dwell_timer.isActive())
            self.assertFalse(view._magic_candidate_item.isVisible())
            view.set_edit_clip_rect(None)

    def test_magic_dwell_release_confirms_elapsed_candidate_before_timer_dispatch(self):
        import time
        from solid_inpaint_ui import MaskEditorView
        source = np.full((80, 100, 3), 220, np.uint8)
        source[38:42, 19:23] = 80
        source[38:42, 59:63] = 80
        blank = np.zeros(source.shape[:2], np.uint8)
        view = MaskEditorView()
        view.set_source_image(source)
        view.set_mask(blank, blank.shape)
        view.set_tool('magic')
        view.set_magic_dwell_ms(30)
        view._start_magic_stroke((20, 40))
        view._advance_magic_stroke((60, 40))
        time.sleep(0.045)  # No Qt event dispatch during sleep.
        self.assertTrue(view._magic_dwell_timer.isActive())
        view._finish_magic_stroke()
        self.assertTrue(np.all(view.mask[38:42, 59:63] == 255))

    def test_magic_escape_and_setting_change_cancel_active_stroke(self):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        w = self.window
        w.set_edit_tool('magic')
        w.show()
        view = w.mask_view
        original = view.mask.copy()
        view._start_magic_stroke((85, 70))
        self.assertTrue(w.is_mask_stroke_active)
        QTest.keyClick(view, Qt.Key.Key_Escape)
        self.assertIsNone(view._magic_stroke_rect)
        self.assertFalse(w.is_mask_stroke_active)
        np.testing.assert_array_equal(view.mask, original)

        view._start_magic_stroke((85, 70))
        w.magic_tolerance_slider.setValue(w.magic_tolerance_slider.value() + 1)
        self.assertIsNone(view._magic_stroke_rect)
        np.testing.assert_array_equal(view.mask, original)

    def test_solid_bubble_transfer_moves_whole_selection_and_undo_restores_fill(self):
        w = self.window
        state = read_page_state(self.paths, self.path)
        state['overlay'][25:150, 30:180] = [240,240,240,255]
        store.save_page(self.paths, self.path, state)
        w.current_manual_solid = state['overlay'][:,:,3].copy()
        w.set_edit_mode('manual_other')
        before = w.current_manual_solid.copy()
        self.assertTrue(w.transfer_selection_from_other_masks(np.ones(self.mask.shape, bool)))
        self.assertFalse(np.any(w.current_manual_solid))
        self.assertTrue(np.all(w.current_manual_other[self.mask > 0] == 255))
        self.assertEqual(w.current_manual_other[30, 35], 255)
        self.assertTrue(w.save_all_edit_masks())
        w.undo_mask()
        np.testing.assert_array_equal(w.current_manual_solid, before)
        self.assertFalse(np.any(w.current_manual_other))

    def test_batch_other_conversion_moves_whole_bubble_and_preserves_manual_areas(self):
        from solid_inpaint_ui import _convert_image_edit_masks
        state = read_page_state(self.paths, self.path)
        state['overlay'][25:150,30:180] = [240,240,240,255]
        state['edited'][30:35,35:40] = 255
        state['other'][155:160,35:40] = 255
        store.save_page(self.paths, self.path, state)
        _convert_image_edit_masks(self.paths, self.path, 'manual_other')
        result = read_page_state(self.paths, self.path)
        self.assertFalse(np.any(result['overlay']))
        self.assertTrue(np.all(result['other'][self.mask > 0] == 255))
        self.assertEqual(result['other'][40,50], 255)
        self.assertTrue(np.all(result['other'][30:35,35:40] == 255))
        self.assertTrue(np.all(result['other'][155:160,35:40] == 255))

    def test_two_classes_mutually_exclusive_and_undo_restores_original_color(self):
        from solid_inpaint_ui import EDIT_MODE_LABELS, ConvertMasksDialog, AddDetectionDialog
        self.assertEqual(set(EDIT_MODE_LABELS), {'manual_solid', 'manual_other'})
        self.assertEqual(ConvertMasksDialog().selected_target_mode(), 'manual_solid')
        self.assertEqual(AddDetectionDialog(self.window.settings).selected_target_mode(), 'manual_other')
        w = self.window
        original = read_page_state(self.paths, self.path)['overlay'].copy()
        w.set_edit_mode('manual_other')
        w.push_undo_snapshot()
        other = w.current_manual_other.copy();other[70:80, 83:90] = 255
        w.set_current_edit_mask(other)
        self.assertFalse(np.any(w.current_manual_solid[70:80, 83:90]))
        self.assertTrue(w.save_all_edit_masks())
        w.queue_auto_render()
        self.assertTrue(np.all(read_page_state(self.paths, self.path)['other'][70:80, 83:90] == 255))
        w.set_edit_mode('manual_solid')
        w.undo_mask()
        np.testing.assert_array_equal(read_page_state(self.paths, self.path)['overlay'], original)
        w.redo_mask()
        self.assertTrue(np.all(read_page_state(self.paths, self.path)['other'][70:80, 83:90] == 255))

    def test_erase_all_does_not_modify_cache_or_reappear_on_reload(self):
        w = self.window
        before = store.read_cache_file(store.cache_path(self.paths, self.path))
        selection = np.zeros_like(self.mask, bool);selection[65:95, 82:98] = True
        w.on_erase_all_masks_requested(selection)
        w.reload_current()
        self.assertFalse(np.any(w.current_manual_solid[selection]))
        self.assertFalse(np.any(w.current_manual_other[selection]))
        after = store.read_cache_file(store.cache_path(self.paths, self.path))
        self.assertEqual(set(after), set(before))
        for key in before:
            np.testing.assert_array_equal(after[key], before[key])

    def test_export_uses_authoritative_page_without_png_intermediates(self):
        with patch('solid_inpaint_ui.QMessageBox.information'):
            self.window.export_refinement_package()
        exported = Path(self.paths['export_pair'])/'01.png'
        self.assertTrue(exported.exists())
        page = read_page_state(self.paths, self.path)
        from detect_solid_inpaint_folder import _compose_overlay_preview
        np.testing.assert_array_equal(cv2.imread(str(exported)), _compose_overlay_preview(self.image, page['overlay']))
        self.assertFalse(list(Path(self.paths['raw']).rglob('*.png')))

    def test_cancel_color_picker_preserves_saved_page_and_editor(self):
        from PySide6.QtGui import QColor
        w = self.window
        store.save_page(self.paths, self.path, store.empty_page(self.mask.shape))
        w.reload_current()
        before = read_page_state(self.paths, self.path)['overlay'].copy()
        w.set_current_edit_mask(np.full_like(self.mask, 255))
        with patch('solid_inpaint_ui.QColorDialog.getColor', return_value=QColor()):
            self.assertFalse(w.save_all_edit_masks())
        np.testing.assert_array_equal(read_page_state(self.paths, self.path)['overlay'], before)
        np.testing.assert_array_equal(w.current_manual_solid, before[:, :, 3])

    def test_mask_original_slider_changes_canvas_and_live_brush_without_changing_page(self):
        from solid_inpaint_ui import EDIT_MODE_COLORS
        w = self.window
        before = read_page_state(self.paths, self.path)
        w.current_manual_other[10:15, 10:15] = 255
        w.current_manual_solid[10:15, 10:15] = 0
        w.current_background_sample = np.zeros_like(self.mask)
        w.current_background_sample[0:5, 0:5] = 255
        w.show_background_sample = False

        def canvas_bgr():
            from PySide6.QtGui import QImage
            image = w.mask_view.pixmap_item.pixmap().toImage().convertToFormat(QImage.Format.Format_RGB888)
            rgb = np.frombuffer(image.bits(), np.uint8).reshape(image.height(), image.bytesPerLine())
            return rgb[:, :image.width()*3].reshape(image.height(), image.width(), 3)[:, :, ::-1].copy()

        w.alpha_slider.setValue(0)
        np.testing.assert_array_equal(canvas_bgr(), w.current_base)
        w.alpha_slider.setValue(100)
        mask_canvas = canvas_bgr()
        np.testing.assert_array_equal(mask_canvas[20, 20], [0, 0, 0])
        np.testing.assert_array_equal(mask_canvas[12, 12], EDIT_MODE_COLORS['manual_other'])
        self.assertEqual(w.mask_color_combo.currentText(), '白色')
        np.testing.assert_array_equal(mask_canvas[70, 90], EDIT_MODE_COLORS['manual_solid'])
        from solid_inpaint_ui import MASK_DISPLAY_COLORS
        w.mask_color_combo.setCurrentText('青色')
        self.assertFalse(np.array_equal(canvas_bgr()[70, 90], mask_canvas[70, 90]))
        np.testing.assert_array_equal(canvas_bgr()[12, 12], EDIT_MODE_COLORS['manual_other'])
        self.assertEqual(w.current_edit_color(), MASK_DISPLAY_COLORS['青色'])
        for value in (0, 50, 100):
            w.alpha_slider.setValue(value)
            expected = canvas_bgr()
            w.set_edit_tool('brush')
            w.prepare_brush_live_preview()
            live = w.render_brush_live_patch(0, 0, 30, 30)
            np.testing.assert_array_equal(live[:, :, :3], expected[:30, :30])
            self.assertTrue(np.all(live[:, :, 3] == 255))
            w.mask_view.stop_live_mask_preview()
        after = read_page_state(self.paths, self.path)
        for key in before:
            np.testing.assert_array_equal(after[key], before[key])

    def test_preview_shows_complete_fill_and_samples_only_outside_selection(self):
        from solid_inpaint_ui import _editor_mask_preview
        base = np.full((8, 8, 3), 200, np.uint8)
        base[3, 3] = 0
        solid = np.zeros((8, 8), np.uint8); solid[1:7, 1:7] = 255
        sample = np.full_like(solid, 255)
        for alpha in (0, 0.5, 0.8, 1):
            actual = _editor_mask_preview(base, solid, None, alpha, sample)
            expected_fill = (base[2, 2] * (1-alpha) + 255*alpha).astype(np.uint8)
            np.testing.assert_array_equal(actual[2, 2], expected_fill)
            expected_sample = (np.array([200, 200, 200]) * (1-alpha) * .72
                               + np.array([70, 235, 255]) * .28).astype(np.uint8)
            np.testing.assert_array_equal(actual[0, 0], expected_sample)
        pure = _editor_mask_preview(base, solid, None, 1, sample)
        np.testing.assert_array_equal(pure[3, 3], [255, 255, 255])
        np.testing.assert_array_equal(pure[2, 2], [255, 255, 255])
        self.assertEqual(self.window.alpha_slider.value(), 70)
        self.window.alpha_slider.setValue(63)
        self.assertEqual(self.window._load_mask_alpha_percent(), 63)

    def test_f1_preview_shows_whole_fill_and_keeps_sample_outside_it(self):
        from solid_inpaint_ui import _editor_mask_preview
        base = np.full((8, 8, 3), 200, np.uint8)
        solid = np.zeros((8, 8), np.uint8); solid[1:7, 1:7] = 255
        other = np.zeros_like(solid); other[0, 7] = 255
        sample = np.full_like(solid, 255)
        preview = _editor_mask_preview(
            base, solid, other, 1, sample,
        )
        np.testing.assert_array_equal(preview[3, 3], [255, 255, 255])
        np.testing.assert_array_equal(preview[2, 2], [255, 255, 255])
        np.testing.assert_array_equal(preview[0, 0], [19, 65, 71])
        np.testing.assert_array_equal(preview[0, 7], [165, 110, 255])

    def test_mode_switch_preserves_white_fill_preview_and_uses_cached_sample(self):
        w = self.window
        self.assertIsNotNone(w.current_background_sample)
        self.assertFalse(hasattr(w, 'detected_text_mask'))
        w.show_background_sample = False
        w.alpha_slider.setValue(100)
        w.set_edit_mode('manual_solid')
        first = w.mask_view.pixmap_item.pixmap().toImage()
        w.set_edit_mode('manual_other')
        second = w.mask_view.pixmap_item.pixmap().toImage()
        self.assertEqual(first, second)
        self.assertGreater(np.count_nonzero(w.current_manual_solid), np.count_nonzero(self.mask))

    def test_missing_sample_cache_leaves_preview_usable(self):
        w = self.window
        store.cache_path(self.paths, self.path).unlink()
        w.reload_current()
        self.assertIsNone(w.current_background_sample)
        w.set_edit_mode('manual_other')
        self.assertFalse(w.mask_view.pixmap_item.pixmap().isNull())

    def test_manual_addition_and_erasure_change_display(self):
        from solid_inpaint_ui import _editor_mask_preview, LocalEditDialog
        w = self.window
        w.show_background_sample = False
        w.alpha_slider.setValue(100)
        solid = w.current_manual_solid.copy()
        solid[20:25, 20:25] = 255
        solid[70:75, 85:90] = 0
        w.set_current_edit_mask(solid)
        expected = _editor_mask_preview(
            w.current_base, solid, w.current_manual_other, 1,
        )
        np.testing.assert_array_equal(expected[22, 22], [255]*3)
        np.testing.assert_array_equal(expected[72, 87], [0]*3)
        w.mask_view.set_mask(solid, solid.shape)
        np.testing.assert_array_equal(w.render_brush_live_patch(0, 0, 100, 100)[:, :, :3], expected[:100, :100])
        dialog = LocalEditDialog(w.current_base, solid, (0, 0, 100, 100), (255, 255, 255), 1,
            preview_context={'mode': 'manual_solid', 'solid': solid.copy(), 'other': w.current_manual_other.copy(),
                             'solid_color': (255, 255, 255), 'sample': None})
        np.testing.assert_array_equal(dialog.render_preview(solid), expected)
        np.testing.assert_array_equal(dialog.render_live_patch(0, 0, 100, 100)[:, :, :3], expected[:100, :100])
        dialog.close()

    def test_local_edit_dialog_accepts_grayscale_source(self):
        from solid_inpaint_ui import LocalEditDialog
        grayscale = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        dialog = LocalEditDialog(
            grayscale, self.mask, (0, 0, 100, 100), (255, 255, 255), 0,
        )
        self.assertEqual(dialog.page_bgr.shape, (*grayscale.shape, 3))
        np.testing.assert_array_equal(dialog.page_bgr[:, :, 0], grayscale)
        np.testing.assert_array_equal(dialog.render_preview(self.mask)[:, :, 0], grayscale)
        dialog.close()

    def test_local_edit_switches_classes_and_undo_redo_restores_both(self):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        from solid_inpaint_ui import LocalEditDialog
        source = np.full((12, 12, 3), 180, np.uint8)
        solid = np.zeros((12, 12), np.uint8); solid[3:5, 3:5] = 255
        other = np.zeros_like(solid); other[6:8, 6:8] = 255
        dialog = LocalEditDialog(source, solid, (2, 2, 9, 9), (255, 255, 255), 1,
            preview_context={'mode': 'manual_solid', 'solid': solid, 'other': other,
                             'solid_color': (255, 255, 255), 'sample': None})
        try:
            dialog.show()
            self.app.processEvents()
            dialog.magic_tolerance_slider.setFocus()
            QTest.keyClick(dialog.magic_tolerance_slider, Qt.Key.Key_F2)
            self.assertEqual(dialog.edit_mode, 'manual_other')
            dialog.on_edit_started()
            edited = dialog.current_mask.copy(); edited[3:5, 3:5] = 255
            dialog.on_mask_edited(edited)
            self.assertTrue(np.all(dialog.current_other[3:5, 3:5] == 255))
            self.assertFalse(np.any(dialog.current_solid[3:5, 3:5]))
            QTest.keyClick(dialog.magic_tolerance_slider, Qt.Key.Key_F1)
            self.assertEqual(dialog.edit_mode, 'manual_solid')
            dialog.on_edit_started()
            edited = dialog.current_mask.copy(); edited[6:8, 6:8] = 255
            dialog.on_mask_edited(edited)
            self.assertTrue(np.all(dialog.current_solid[6:8, 6:8] == 255))
            self.assertFalse(np.any(dialog.current_other[6:8, 6:8]))
            dialog.undo(); dialog.undo()
            result_solid, result_other = dialog.result_masks()
            np.testing.assert_array_equal(result_solid, solid[2:9, 2:9])
            np.testing.assert_array_equal(result_other, other[2:9, 2:9])
            dialog.redo(); dialog.redo()
            self.assertFalse(np.any(dialog.current_solid[3:5, 3:5]))
            self.assertFalse(np.any(dialog.current_other[6:8, 6:8]))
            self.assertFalse(np.any(solid[6:8, 6:8]))
            self.assertTrue(np.all(other[6:8, 6:8] == 255))
        finally:
            dialog.close()

    def test_local_magic_sliders_change_selection_within_roi(self):
        from solid_inpaint_ui import LocalEditDialog
        source = np.full((12, 12, 3), 130, np.uint8)
        source[5:7, 5:7] = 100
        blank = np.zeros((12, 12), np.uint8)
        dialog = LocalEditDialog(source, blank, (3, 3, 9, 9), (255, 255, 255), 1,
                                 magic_tolerance=0, magic_expand_px=0)
        try:
            dialog.set_tool('magic')
            exact = dialog.view._magic_selection_at((5, 5))
            self.assertEqual(int(np.count_nonzero(exact)), 4)
            dialog.magic_tolerance_slider.setValue(40)
            self.assertEqual(dialog.magic_tolerance_label.text(), '容差 40')
            self.assertEqual(int(np.count_nonzero(dialog.view._magic_selection_at((5, 5)))), 36)
            dialog.magic_tolerance_slider.setValue(0)
            dialog.magic_expand_slider.setValue(2)
            self.assertEqual(dialog.magic_expand_label.text(), '擴展 2px')
            expanded = dialog.view._magic_selection_at((5, 5))
            self.assertGreater(int(np.count_nonzero(expanded)), 4)
            self.assertFalse(np.any(expanded[:3]))
            self.assertFalse(np.any(expanded[9:]))
            self.assertFalse(np.any(expanded[:, :3]))
            self.assertFalse(np.any(expanded[:, 9:]))
            self.assertIsNone(dialog.view._magic_selection_at((2, 5)))
        finally:
            dialog.close()

    def test_local_brush_stroke_blocks_mode_switch_until_committed(self):
        from PySide6.QtCore import QRectF, Qt
        from PySide6.QtTest import QTest
        from solid_inpaint_ui import LocalEditDialog
        source = np.full((12, 12, 3), 180, np.uint8)
        blank = np.zeros((12, 12), np.uint8)
        dialog = LocalEditDialog(source, blank, (2, 2, 9, 9), (255, 255, 255), 1)
        try:
            dialog.show()
            self.app.processEvents()
            dialog.view._set_brush_stroke_active(True)
            dialog.on_edit_started()
            dialog.view.mask[5, 5] = 255
            dialog.on_brush_preview_changed(QRectF(5, 5, 1, 1))
            dialog.magic_tolerance_slider.setFocus()
            QTest.keyClick(dialog.magic_tolerance_slider, Qt.Key.Key_F2)
            self.assertEqual(dialog.edit_mode, 'manual_solid')
            self.assertTrue(dialog.solid_btn.isChecked())
            dialog.view._set_brush_stroke_active(False)
            dialog.on_mask_edited(dialog.view.mask)
            QTest.keyClick(dialog.magic_tolerance_slider, Qt.Key.Key_F2)
            self.assertEqual(dialog.edit_mode, 'manual_other')
            self.assertEqual(dialog.current_solid[5, 5], 255)
            self.assertEqual(dialog.current_other[5, 5], 0)
        finally:
            dialog.close()

    def test_local_mode_and_magic_settings_sync_to_main_without_saving_masks(self):
        from PySide6.QtWidgets import QDialog
        from solid_inpaint_ui import LocalEditDialog
        w = self.window
        selection = np.zeros_like(self.mask, bool); selection[60:100, 80:100] = True
        before = read_page_state(self.paths, self.path)
        for result in (QDialog.DialogCode.Rejected, QDialog.DialogCode.Accepted):
            with self.subTest(result=result):
                w.set_edit_mode('manual_solid')
                w.set_edit_tool('rect')
                w.set_selection_combine_mode('local_edit_selection')
                w.magic_tolerance_slider.setValue(18)
                w.magic_expand_slider.setValue(3)

                def change_settings(dialog):
                    self.assertEqual(dialog.magic_tolerance_slider.value(), 18)
                    self.assertEqual(dialog.magic_expand_slider.value(), 3)
                    dialog.set_edit_mode('manual_other')
                    self.assertEqual(w.edit_mode, 'manual_other')
                    self.assertTrue(w.edit_manual_other_btn.isChecked())
                    np.testing.assert_array_equal(w.mask_view.mask, w.current_manual_other)
                    self.assertEqual(w.mask_view.tool, 'rect')
                    self.assertEqual(w.selection_combine_mode, 'local_edit_selection')
                    dialog.magic_tolerance_slider.setValue(44)
                    dialog.magic_expand_slider.setValue(9)
                    self.assertEqual(w.magic_tolerance_slider.value(), 44)
                    self.assertEqual(w.magic_expand_slider.value(), 9)
                    self.assertEqual(w.magic_tolerance_label.text(), '容差 44')
                    self.assertEqual(w.magic_expand_label.text(), '擴展 9px')
                    self.assertEqual(w.mask_view.magic_tolerance, 44)
                    self.assertEqual(w.mask_view.magic_expand_px, 9)
                    return result

                with patch.object(LocalEditDialog, 'exec', change_settings):
                    w.open_local_edit_dialog(selection)
                self.assertEqual(w.edit_mode, 'manual_other')
                self.assertEqual(w.mask_view.tool, 'rect')
                self.assertEqual(w.selection_combine_mode, 'add')
                self.assertEqual(w.magic_tolerance_slider.value(), 44)
                self.assertEqual(w.magic_expand_slider.value(), 9)
                np.testing.assert_array_equal(w.mask_view.mask, w.current_manual_other)
                after = read_page_state(self.paths, self.path)
                for key in before:
                    np.testing.assert_array_equal(after[key], before[key])

    def test_local_edit_apply_and_cancel_keep_both_classes_and_main_history(self):
        from PySide6.QtWidgets import QDialog
        from solid_inpaint_ui import LocalEditDialog
        w = self.window
        state = read_page_state(self.paths, self.path)
        state['overlay'][62:66, 82:86] = 0
        state['other'][62:66, 82:86] = 255
        store.save_page(self.paths, self.path, state)
        w.reload_current()
        before = read_page_state(self.paths, self.path)
        selection = np.zeros_like(self.mask, bool); selection[60:100, 80:100] = True

        def edit_dialog(dialog):
            self.assertEqual(dialog.magic_tolerance_slider.value(), w.mask_view.magic_tolerance)
            self.assertEqual(dialog.magic_expand_slider.value(), w.mask_view.magic_expand_px)
            dialog.set_edit_mode('manual_other')
            dialog.on_edit_started()
            other = dialog.current_mask.copy(); other[70:74, 88:92] = 255
            dialog.on_mask_edited(other)
            dialog.set_edit_mode('manual_solid')
            dialog.on_edit_started()
            solid = dialog.current_mask.copy(); solid[62:66, 82:86] = 255
            dialog.on_mask_edited(solid)
            return QDialog.DialogCode.Rejected

        with patch.object(LocalEditDialog, 'exec', edit_dialog):
            w.open_local_edit_dialog(selection)
        cancelled = read_page_state(self.paths, self.path)
        for key in before:
            np.testing.assert_array_equal(cancelled[key], before[key])

        def apply_dialog(dialog):
            edit_dialog(dialog)
            return QDialog.DialogCode.Accepted

        initial_undo = len(w.undo_stack)
        with patch.object(LocalEditDialog, 'exec', apply_dialog):
            w.open_local_edit_dialog(selection)
        after = read_page_state(self.paths, self.path)
        self.assertEqual(len(w.undo_stack), initial_undo + 1)
        self.assertTrue(np.all(after['other'][70:74, 88:92] == 255))
        self.assertFalse(np.any(after['overlay'][70:74, 88:92, 3]))
        self.assertTrue(np.all(after['overlay'][62:66, 82:86, 3] == 255))
        self.assertFalse(np.any(after['other'][62:66, 82:86]))
        self.assertTrue(np.all(after['edited'][62:66, 82:86] == 255))
        w.undo_mask()
        undone = read_page_state(self.paths, self.path)
        np.testing.assert_array_equal(undone['overlay'], before['overlay'])
        np.testing.assert_array_equal(undone['other'], before['other'])
        w.redo_mask()
        redone = read_page_state(self.paths, self.path)
        np.testing.assert_array_equal(redone['overlay'], after['overlay'])
        np.testing.assert_array_equal(redone['other'], after['other'])
