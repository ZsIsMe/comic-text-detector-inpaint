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
        store.update_cache_file(file, text_mask=self.mask)
        store.update_cache_file(file, bubble_json=np.array('{}'))
        store.update_cache_file(file, sample=self.mask)
        self.assertEqual(set(store.read_cache_file(file)), {'text_mask', 'bubble_json', 'sample'})
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
        np.testing.assert_array_equal(store.read_cache_file(store.cache_path(self.paths, self.path))['text_mask'], self.mask)

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

    def test_erase_all_does_not_modify_detector_cache_or_reappear_on_reload(self):
        w = self.window
        before = store.read_cache_file(store.cache_path(self.paths, self.path))['text_mask'].copy()
        selection = np.zeros_like(self.mask, bool);selection[65:95, 82:98] = True
        w.on_erase_all_masks_requested(selection)
        w.reload_current()
        self.assertFalse(np.any(w.current_manual_solid[selection]))
        self.assertFalse(np.any(w.current_manual_other[selection]))
        np.testing.assert_array_equal(store.read_cache_file(store.cache_path(self.paths, self.path))['text_mask'], before)

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
        self.assertEqual(w.mask_color_combo.currentText(), '淡黃色')
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

    def test_mask_mix_endpoints_midpoint_and_readable_default(self):
        from solid_inpaint_ui import _editor_mask_preview, EDIT_MODE_COLORS
        base = np.full((4, 4, 3), 255, np.uint8)
        base[1, 1] = 0
        solid = np.zeros((4, 4), np.uint8)
        solid[:3, :3] = 255
        sample = solid.copy()
        for alpha in (0.0, 0.28, 0.5, 0.8, 1.0):
            plain = _editor_mask_preview(base, solid, None, alpha)
            overlapping = _editor_mask_preview(base, solid, None, alpha, sample)
            np.testing.assert_array_equal(plain, overlapping)
        mask = _editor_mask_preview(base, solid, None, 1.0)
        np.testing.assert_array_equal(mask[0, 0], EDIT_MODE_COLORS['manual_solid'])
        np.testing.assert_array_equal(mask[1, 1], mask[0, 0])
        np.testing.assert_array_equal(mask[3, 3], [0, 0, 0])
        np.testing.assert_array_equal(_editor_mask_preview(base, solid, None, 0, sample), base)
        midpoint = _editor_mask_preview(base, solid, None, 0.5)
        np.testing.assert_array_equal(midpoint, ((base.astype(float) + mask) / 2).astype(np.uint8))
        self.assertEqual(self.window.alpha_slider.value(), 28)
        preview = _editor_mask_preview(base, solid, None, 0.28)
        np.testing.assert_array_equal(preview[0, 0], [203, 249, 255])
        self.assertTrue(np.all(preview[0, 0].astype(int) - preview[1, 1] >= 180))
        self.window.alpha_slider.setValue(63)
        self.assertEqual(self.window._load_mask_alpha_percent(), 63)
