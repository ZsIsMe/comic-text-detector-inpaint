"""Cmd/Ctrl-right rectangle swaps existing solid and OTHER pixels only."""
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import solid_inpaint_ui as ui
import test_project_store as fixtures


class TypeSwapViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_mouse_gesture_routes_across_tools_and_latches_press_modifier(self):
        for tool in ('brush', 'rect', 'magic', 'lasso', 'roi_bounds'):
            with self.subTest(tool=tool):
                view = ui.MaskEditorView()
                image = np.full((40, 50, 3), 240, np.uint8)
                view.set_qimage(ui._qimage_from_bgr(image))
                view.set_mask(np.zeros((40, 50), np.uint8), (40, 50))
                view.set_tool(tool)
                view.show()
                self.app.processEvents()
                swap, erase, edits = [], [], []
                view.swapClassesRequested.connect(swap.append)
                view.eraseAllMasksRequested.connect(erase.append)
                view.maskEdited.connect(edits.append)
                start = view.mapFromScene(QPointF(5, 5))
                end = view.mapFromScene(QPointF(20, 20))
                self.assertEqual(view.image_point_from_view(start), (5, 5))
                viewport = view.viewport()
                QTest.mousePress(viewport, Qt.MouseButton.RightButton,
                                 Qt.KeyboardModifier.ControlModifier, pos=start)
                QTest.mouseMove(viewport, end)
                QTest.mouseRelease(viewport, Qt.MouseButton.RightButton,
                                   Qt.KeyboardModifier.NoModifier, pos=end)
                self.assertEqual(len(swap), 1)
                self.assertEqual(len(erase), 0)
                self.assertEqual(len(edits), 0)
                self.assertTrue(swap[0][5:21, 5:21].all())
                view.close()

    def test_ordinary_right_stays_erase_when_modifier_is_added_mid_drag(self):
        view = ui.MaskEditorView()
        view.set_qimage(ui._qimage_from_bgr(np.full((40, 50, 3), 240, np.uint8)))
        view.set_mask(np.zeros((40, 50), np.uint8), (40, 50))
        view.show(); self.app.processEvents()
        swap, erase = [], []
        view.swapClassesRequested.connect(swap.append)
        view.eraseAllMasksRequested.connect(erase.append)
        start = view.mapFromScene(QPointF(5, 5)); end = view.mapFromScene(QPointF(20, 20))
        QTest.mousePress(view.viewport(), Qt.MouseButton.RightButton, pos=start)
        QTest.mouseMove(view.viewport(), end)
        QTest.mouseRelease(view.viewport(), Qt.MouseButton.RightButton,
                           Qt.KeyboardModifier.ControlModifier, pos=end)
        self.assertEqual(len(erase), 1)
        self.assertFalse(swap)
        view.close()

    def test_platform_command_mapping_and_escape_cancel(self):
        view = ui.MaskEditorView()
        attribute = Qt.ApplicationAttribute.AA_MacDontSwapCtrlAndMeta
        original = QApplication.testAttribute(attribute)
        try:
            with patch.object(ui.sys, 'platform', 'darwin'):
                QApplication.setAttribute(attribute, False)
                self.assertTrue(view._is_class_swap_modifier(Qt.KeyboardModifier.ControlModifier))
                self.assertFalse(view._is_class_swap_modifier(Qt.KeyboardModifier.MetaModifier))
                QApplication.setAttribute(attribute, True)
                self.assertTrue(view._is_class_swap_modifier(Qt.KeyboardModifier.MetaModifier))
                self.assertFalse(view._is_class_swap_modifier(Qt.KeyboardModifier.ControlModifier))
            with patch.object(ui.sys, 'platform', 'linux'):
                self.assertTrue(view._is_class_swap_modifier(Qt.KeyboardModifier.ControlModifier))
                self.assertFalse(view._is_class_swap_modifier(Qt.KeyboardModifier.MetaModifier))
        finally:
            QApplication.setAttribute(attribute, original)
        view.set_qimage(ui._qimage_from_bgr(np.full((40, 50, 3), 240, np.uint8)))
        view.set_mask(np.zeros((40, 50), np.uint8), (40, 50))
        view.show(); self.app.processEvents()
        swap = []
        view.swapClassesRequested.connect(swap.append)
        start = view.mapFromScene(QPointF(5, 5)); end = view.mapFromScene(QPointF(20, 20))
        QTest.mousePress(view.viewport(), Qt.MouseButton.RightButton,
                         Qt.KeyboardModifier.ControlModifier, pos=start)
        QTest.keyClick(view, Qt.Key.Key_Escape)
        QTest.mouseRelease(view.viewport(), Qt.MouseButton.RightButton, pos=end)
        self.assertFalse(swap)
        self.assertFalse(view._swap_classes_drag)
        view.close()


class TypeSwapWindowTests(unittest.TestCase):
    def setUp(self):
        fixtures.EditorTests.setUp(self)

    def tearDown(self):
        fixtures.EditorTests.tearDown(self)

    def test_closed_window_cancels_pending_recent_folder_restore(self):
        w = self.window
        w.folder = ''
        w.recent_folders = ['/missing/recent/project']
        with patch.object(ui.QMessageBox, 'warning', side_effect=AssertionError('late restore')):
            w.close()
            self.app.processEvents()
        self.assertFalse(w.startup_restore_timer.isActive())

    def test_mixed_swap_single_undo_redo_and_saved_page(self):
        w = self.window
        w.current_manual_other[65:75, 115:125] = 255
        self.assertTrue(w._commit_current_masks())
        self.assertTrue(w.flush_pending_edits())
        before = fixtures.read_page_state(self.paths, self.path)
        selection = np.zeros(self.mask.shape, bool)
        selection[55:105, 75:130] = True
        undo_count = len(w.undo_stack)
        w.set_edit_mode('manual_other')
        w.set_edit_tool('magic')
        w.set_selection_combine_mode('local_intersect')
        w.on_swap_classes_requested(selection)
        self.assertTrue(w.flush_pending_edits())
        changed = fixtures.read_page_state(self.paths, self.path)
        self.assertEqual(len(w.undo_stack), undo_count + 1)
        self.assertTrue(np.all(changed['other'][self.mask > 0] == 255))
        self.assertTrue(np.all(changed['overlay'][65:75, 115:125, 3] == 255))
        self.assertFalse(np.any(changed['other'][65:75, 115:125]))
        self.assertFalse(np.any(changed['overlay'][~selection, 3] != before['overlay'][~selection, 3]))
        np.testing.assert_array_equal(changed['overlay'][~selection, :3], before['overlay'][~selection, :3])
        np.testing.assert_array_equal(changed['edited'][~selection], before['edited'][~selection])
        self.assertFalse(np.any((changed['overlay'][:, :, 3] > 0) & (changed['other'] > 0)))
        w.undo_mask(); self.assertTrue(w.flush_pending_edits())
        undone = fixtures.read_page_state(self.paths, self.path)
        for key in before:
            np.testing.assert_array_equal(undone[key], before[key])
        w.redo_mask(); self.assertTrue(w.flush_pending_edits())
        redone = fixtures.read_page_state(self.paths, self.path)
        for key in changed:
            np.testing.assert_array_equal(redone[key], changed[key])
        w.reload_current()
        np.testing.assert_array_equal(w.current_page['overlay'], changed['overlay'])

    def test_no_hit_does_not_save_or_clear_redo(self):
        w = self.window
        selection = np.zeros(self.mask.shape, bool)
        selection[:10, :10] = True
        before_undo, before_redo = w.undo_stack.copy(), w.redo_stack.copy()
        with patch.object(w, '_commit_current_masks', side_effect=AssertionError('unneeded commit')), \
                patch.object(w, 'push_undo_snapshot', side_effect=AssertionError('unneeded undo')):
            w.on_swap_classes_requested(selection)
        self.assertEqual(w.undo_stack, before_undo)
        self.assertEqual(w.redo_stack, before_redo)

    def test_cancel_color_fallback_restores_both_masks_and_histories(self):
        w = self.window
        w.current_manual_other[65:75, 115:125] = 255
        self.assertTrue(w._commit_current_masks()); self.assertTrue(w.flush_pending_edits())
        before = w.current_masks_snapshot()
        undo_before, redo_before = w.undo_stack.copy(), w.redo_stack.copy()
        selection = np.zeros(self.mask.shape, bool); selection[65:75, 115:125] = True
        with patch.object(ui, 'prepare_page_edits', side_effect=ValueError('取色失敗')), \
                patch.object(ui.QColorDialog, 'getColor', return_value=QColor()):
            w.on_swap_classes_requested(selection)
        np.testing.assert_array_equal(w.current_manual_solid, before['manual_solid'])
        np.testing.assert_array_equal(w.current_manual_other, before['manual_other'])
        np.testing.assert_array_equal(w.current_page['overlay'], before['overlay'])
        np.testing.assert_array_equal(w.current_page['edited'], before['edited'])
        self.assertEqual(w.undo_stack, undo_before)
        self.assertEqual(w.redo_stack, redo_before)


class TypeSwapLocalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_roi_clips_swap_and_undo_redo_before_apply(self):
        source = np.full((20, 20, 3), 240, np.uint8)
        solid = np.zeros((20, 20), np.uint8); solid[3:8, 3:8] = 255
        other = np.zeros_like(solid); other[10:15, 10:15] = 255
        original_solid, original_other = solid.copy(), other.copy()
        dialog = ui.LocalEditDialog(source, solid, (5, 5, 13, 13), (255, 255, 255), 1,
            preview_context={'mode': 'manual_solid', 'solid': solid, 'other': other,
                             'solid_color': (255, 255, 255), 'sample': None})
        try:
            dialog.set_tool('roi_bounds')
            selection = np.ones((20, 20), bool)
            dialog.on_swap_classes_requested(selection)
            self.assertEqual(len(dialog.undo_stack), 1)
            self.assertFalse(np.any(dialog.current_solid[5:8, 5:8]))
            self.assertTrue(np.all(dialog.current_other[5:8, 5:8] == 255))
            self.assertTrue(np.all(dialog.current_solid[10:13, 10:13] == 255))
            np.testing.assert_array_equal(dialog.current_solid[3:5, 3:8], original_solid[3:5, 3:8])
            np.testing.assert_array_equal(dialog.current_other[13:15, 10:15], original_other[13:15, 10:15])
            dialog.undo()
            np.testing.assert_array_equal(dialog.current_solid, original_solid)
            np.testing.assert_array_equal(dialog.current_other, original_other)
            dialog.redo()
            self.assertFalse(np.any(dialog.current_solid[5:8, 5:8]))
            dialog.accept()
            solid_crop, other_crop = dialog.result_masks()
            self.assertTrue(np.all(other_crop[0:3, 0:3] == 255))
            self.assertTrue(np.all(solid_crop[5:8, 5:8] == 255))
            np.testing.assert_array_equal(solid, original_solid)
            np.testing.assert_array_equal(other, original_other)
        finally:
            dialog.close()

    def test_local_view_gesture_is_private_until_apply_and_cancel_discards_it(self):
        source = np.full((30, 30, 3), 240, np.uint8)
        solid = np.zeros((30, 30), np.uint8); solid[8:13, 8:13] = 255
        other = np.zeros_like(solid); other[15:20, 15:20] = 255
        dialog = ui.LocalEditDialog(source, other, (5, 5, 23, 23), (255, 255, 255), 1,
            preview_context={'mode': 'manual_other', 'solid': solid, 'other': other,
                             'solid_color': (255, 255, 255), 'sample': None})
        try:
            dialog.set_tool('roi_bounds')
            dialog.show(); self.app.processEvents()
            start = dialog.view.mapFromScene(QPointF(6, 6))
            end = dialog.view.mapFromScene(QPointF(21, 21))
            QTest.mousePress(dialog.view.viewport(), Qt.MouseButton.RightButton,
                             Qt.KeyboardModifier.ControlModifier, pos=start)
            QTest.mouseMove(dialog.view.viewport(), end)
            QTest.mouseRelease(dialog.view.viewport(), Qt.MouseButton.RightButton, pos=end)
            self.assertEqual(len(dialog.undo_stack), 1)
            self.assertTrue(np.all(dialog.current_other[8:13, 8:13] == 255))
            self.assertTrue(np.all(dialog.current_solid[15:20, 15:20] == 255))
            np.testing.assert_array_equal(solid[8:13, 8:13], np.full((5, 5), 255, np.uint8))
            np.testing.assert_array_equal(other[15:20, 15:20], np.full((5, 5), 255, np.uint8))
            dialog.reject()
            np.testing.assert_array_equal(solid[8:13, 8:13], np.full((5, 5), 255, np.uint8))
            np.testing.assert_array_equal(other[15:20, 15:20], np.full((5, 5), 255, np.uint8))
        finally:
            dialog.close()


if __name__ == '__main__':
    unittest.main()
