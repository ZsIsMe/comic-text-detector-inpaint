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
