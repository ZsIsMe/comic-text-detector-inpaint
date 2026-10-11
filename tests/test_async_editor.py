"""Editing must stay responsive while persistence is deliberately blocked."""
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QCloseEvent, QImage, QPainter
from PySide6.QtWidgets import QMessageBox

import project_store as store
import solid_inpaint_ui as ui
import test_project_store as fixtures


class AsyncEditorTests(unittest.TestCase):
    def setUp(self):
        fixtures.EditorTests.setUp(self)
        self.window.set_edit_mode('manual_other')

    def tearDown(self):
        fixtures.EditorTests.tearDown(self)

    def emit_edit(self, mask):
        view = self.window.mask_view
        view.editStarted.emit()
        view.mask = mask.copy()
        view.maskEdited.emit(mask.copy())

    def test_continuous_edits_clear_and_undo_do_not_wait_for_old_save(self):
        w = self.window
        started, release, returned = threading.Event(), threading.Event(), threading.Event()
        real_save = store.save_page
        written = []

        def blocked_save(paths, image, page, *args, **kwargs):
            if not written:
                started.set()
                if not release.wait(10):
                    raise RuntimeError('test did not release the save worker')
            written.append(page['other'].copy())
            result = real_save(paths, image, page, *args, **kwargs)
            returned.set()
            return result

        with patch.object(store, 'save_page', side_effect=blocked_save):
            try:
                first = w.current_manual_other.copy()
                first[10:15, 10:15] = 255
                self.emit_edit(first)
                self.assertTrue(started.wait(3))
                self.assertFalse(returned.is_set())
                np.testing.assert_array_equal(w.current_manual_other, first)

                second = first.copy()
                second[20:25, 20:25] = 255
                self.emit_edit(second)
                selection = np.zeros_like(second, bool)
                selection[10:15, 10:15] = True
                w.on_erase_all_masks_requested(selection)
                final = second.copy()
                final[selection] = 0
                np.testing.assert_array_equal(w.current_manual_other, final)
                np.testing.assert_array_equal(w.mask_view.mask, final)
                w.undo_mask()
                np.testing.assert_array_equal(w.current_manual_other, second)
                w.redo_mask()
                np.testing.assert_array_equal(w.current_manual_other, final)
                self.assertFalse(returned.is_set())

                release.set()
                self.assertTrue(w.flush_pending_edits())
                self.app.processEvents()  # Deliver both old and latest result signals.
                actual = fixtures.read_page_state(self.paths, self.path)
                np.testing.assert_array_equal(actual['other'], final)
                np.testing.assert_array_equal(w.current_manual_other, final)
                np.testing.assert_array_equal(w.mask_view.mask, final)
                self.assertTrue(np.all(actual['edited'][selection] == 255))
                self.assertLessEqual(len(written), 2)
            finally:
                release.set()
                w.flush_pending_edits()

    def test_left_patch_matches_final_mask_before_right_preview_completes(self):
        w = self.window
        started, release = threading.Event(), threading.Event()
        real_save = store.save_page
        right_before = w.preview_view.pixmap_item.pixmap().toImage()

        def blocked_save(*args, **kwargs):
            started.set()
            if not release.wait(10):
                raise RuntimeError('test did not release the save worker')
            return real_save(*args, **kwargs)

        with patch.object(store, 'save_page', side_effect=blocked_save):
            try:
                mask = w.current_manual_other.copy()
                mask[10:20, 10:20] = 255
                self.emit_edit(mask)
                self.assertTrue(started.wait(3))
                item = w.mask_view._committed_mask_overlay
                self.assertTrue(item.isVisible())
                patch_bgra = item.renderer(0, 0, 32, 32)
                expected = ui._editor_mask_preview(
                    w.current_base[:32, :32], w.current_manual_solid[:32, :32],
                    w.current_manual_other[:32, :32], w.alpha,
                    None if not w.show_background_sample or w.current_background_sample is None
                    else w.current_background_sample[:32, :32],
                    solid_color=w.mask_display_color,
                )
                np.testing.assert_array_equal(patch_bgra[:, :, :3], expected)
                self.assertTrue(np.all(patch_bgra[:, :, 3] == 255))
                self.assertEqual(w.preview_view.pixmap_item.pixmap().toImage(), right_before)
            finally:
                release.set()
                self.assertTrue(w.flush_pending_edits())

    def test_save_failure_keeps_in_memory_edit_and_can_be_retried(self):
        w = self.window
        before = store.page_path(self.paths, self.path).read_bytes()
        mask = w.current_manual_other.copy()
        mask[10:15, 10:15] = 255
        with patch.object(store, 'save_page', side_effect=OSError('test disk failure')), \
                patch.object(QMessageBox, 'warning'):
            self.emit_edit(mask)
            self.assertFalse(w.flush_pending_edits())
            self.app.processEvents()
            np.testing.assert_array_equal(w.current_manual_other, mask)
            np.testing.assert_array_equal(w.mask_view.mask, mask)
            self.assertEqual(store.page_path(self.paths, self.path).read_bytes(), before)
        self.assertTrue(w.save_all_edit_masks())
        actual = fixtures.read_page_state(self.paths, self.path)
        np.testing.assert_array_equal(actual['other'], mask)

    def test_magic_release_reuses_selection_and_shows_actual_operation_while_save_waits(self):
        w = self.window
        view = w.mask_view
        real_save = store.save_page
        source = np.full(self.image.shape, 220, np.uint8)
        source[14:17, 14:17] = 80
        for mode in ('manual_solid', 'manual_other'):
            for operation in ('add', 'subtract', 'local_intersect'):
                with self.subTest(mode=mode, operation=operation):
                    w.set_edit_mode(mode)
                    initial = np.zeros(self.image.shape[:2], np.uint8)
                    if operation != 'add':
                        initial[12:19, 12:19] = 255
                        initial[24:27, 24:27] = 255
                    self.emit_edit(initial)
                    self.assertTrue(w.flush_pending_edits())
                    w.set_edit_tool('magic')
                    w.set_selection_combine_mode(operation)
                    view.set_source_image(source)
                    view.set_magic_scope_px(32)
                    view.set_magic_dwell_ms(0)
                    view.set_magic_tolerance(0)
                    view.set_magic_expand_px(0)
                    view.set_local_intersect_offset(0)
                    view._start_magic_stroke((15, 15))
                    expected = initial.copy()
                    if operation == 'local_intersect':
                        expected[12:19, 12:19] = 0
                    expected[14:17, 14:17] = 0 if operation == 'subtract' else 255
                    started, release = threading.Event(), threading.Event()

                    def blocked_save(*args, **kwargs):
                        started.set()
                        if not release.wait(10):
                            raise RuntimeError('test did not release the save worker')
                        return real_save(*args, **kwargs)

                    with patch.object(store, 'save_page', side_effect=blocked_save):
                        try:
                            with patch.object(view, '_magic_component_at', side_effect=AssertionError('recomputed selection')), \
                                    patch.object(ui.cv2, 'floodFill', side_effect=AssertionError('repeated flood fill')):
                                view._finish_magic_stroke()
                            self.assertTrue(started.wait(3))
                            np.testing.assert_array_equal(w.current_edit_mask(), expected)
                            np.testing.assert_array_equal(view.mask, expected)
                            actual = view._committed_mask_overlay.renderer(0, 0, 32, 32)
                            expected_preview = ui._editor_mask_preview(
                                w.current_base[:32, :32], w.current_manual_solid[:32, :32],
                                w.current_manual_other[:32, :32], w.alpha,
                                w.current_background_sample[:32, :32]
                                if w.show_background_sample and w.current_background_sample is not None else None,
                                solid_color=w.mask_display_color,
                            )
                            np.testing.assert_array_equal(actual[:, :, :3], expected_preview)
                        finally:
                            release.set()
                            self.assertTrue(w.flush_pending_edits())

    def check_flush_boundary(self, action):
        w = self.window
        started, release = threading.Event(), threading.Event()
        real_save = store.save_page
        original_flush = w.flush_pending_edits

        def blocked_save(*args, **kwargs):
            started.set()
            if not release.wait(10):
                raise RuntimeError('test did not release the save worker')
            return real_save(*args, **kwargs)

        def flush():
            release.set()
            return original_flush()

        mask = w.current_manual_other.copy()
        mask[10:15, 10:15] = 255
        with patch.object(store, 'save_page', side_effect=blocked_save):
            try:
                self.emit_edit(mask)
                self.assertTrue(started.wait(3))
                with patch.object(w, 'flush_pending_edits', side_effect=flush) as barrier:
                    action()
                    self.assertTrue(barrier.called)
                np.testing.assert_array_equal(
                    fixtures.read_page_state(self.paths, self.path)['other'], mask)
            finally:
                release.set()
                original_flush()

    def test_explicit_reload_flushes_instead_of_losing_pending_edit(self):
        self.check_flush_boundary(self.window.reload_current)
        self.assertTrue(np.all(self.window.current_manual_other[10:15, 10:15] == 255))

    def test_switch_folder_flushes_old_page_before_changing_project(self):
        with tempfile.TemporaryDirectory() as folder:
            self.check_flush_boundary(lambda: self.window.load_folder(folder))
            self.assertEqual(self.window.folder, folder)
            self.app.processEvents()
            self.assertEqual(self.window.folder, folder)

    def test_close_flushes_latest_page_before_accepting_close(self):
        event = QCloseEvent()
        self.check_flush_boundary(lambda: self.window.closeEvent(event))
        self.assertTrue(event.isAccepted())

    def test_failed_flush_prevents_folder_switch_and_close(self):
        w = self.window
        mask = w.current_manual_other.copy()
        mask[10:15, 10:15] = 255
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(store, 'save_page', side_effect=OSError('test disk failure')), \
                patch.object(QMessageBox, 'warning'):
            self.emit_edit(mask)
            original_folder = w.folder
            w.load_folder(folder)
            self.assertEqual(w.folder, original_folder)
            event = QCloseEvent()
            w.closeEvent(event)
            self.assertFalse(event.isAccepted())
            np.testing.assert_array_equal(w.current_manual_other, mask)
        self.assertTrue(w.save_all_edit_masks())

    def test_saved_summary_updates_existing_list_item_after_edit_clear_and_undo(self):
        w = self.window
        item = w.list_widget.currentItem()
        mask = np.zeros_like(w.current_manual_other)
        mask[10:15, 10:15] = 255
        self.emit_edit(mask)
        self.assertTrue(w.flush_pending_edits())
        self.assertIs(w.list_widget.currentItem(), item)
        self.assertIn(ui.STATUS_OTHER, item.text())
        self.assertEqual(w.report['pages']['01.png']['other_pixels'], 25)
        self.assertIn('OTHER 1', w.summary_label.text())
        w.on_erase_all_masks_requested(mask > 0)
        self.assertTrue(w.flush_pending_edits())
        self.assertIs(w.list_widget.currentItem(), item)
        self.assertIn(ui.STATUS_OK, item.text())
        self.assertEqual(w.report['pages']['01.png']['other_pixels'], 0)
        self.assertIn('OTHER 0', w.summary_label.text())
        w.undo_mask()
        self.assertTrue(w.flush_pending_edits())
        self.assertIn(ui.STATUS_OTHER, item.text())
        self.assertEqual(w.report['pages']['01.png']['other_pixels'], 25)

    def test_drag_rectangle_stays_visible_over_previously_committed_tiles(self):
        w = self.window
        view = w.mask_view
        w.set_edit_tool('rect')
        mask = w.current_manual_other.copy()
        mask[10:20, 10:20] = 255
        self.emit_edit(mask)
        self.assertTrue(w.flush_pending_edits())
        self.assertTrue(view._committed_mask_overlay.isVisible())

        def render():
            image = QImage(220, 180, QImage.Format.Format_ARGB32)
            image.fill(Qt.GlobalColor.transparent)
            painter = QPainter(image)
            view.scene().render(painter, QRectF(0, 0, 220, 180), QRectF(0, 0, 220, 180))
            painter.end()
            return image

        # The left border crosses the updated opaque tile; the right border
        # crosses only the original pixmap, reproducing the partial disappearance.
        for button in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            for start, end in (((40, 40), (180, 140)), ((180, 140), (40, 40))):
                with self.subTest(button=button, start=start):
                    before = render()
                    view._update_rubber_band(start, end, button)
                    after = render()
                    self.assertNotEqual(after.copy(37, 50, 7, 70), before.copy(37, 50, 7, 70))
                    self.assertNotEqual(after.copy(177, 50, 7, 70), before.copy(177, 50, 7, 70))
                    view._clear_rubber_band()


if __name__ == '__main__':
    unittest.main()
