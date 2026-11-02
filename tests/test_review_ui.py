"""Review controls must keep page, save and edit state visible to GUI users."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMessageBox

import project_store as store
import solid_inpaint_ui as ui
import test_project_store as fixtures


class ReviewUiTests(unittest.TestCase):
    def setUp(self):
        fixtures.EditorTests.setUp(self)

    def tearDown(self):
        fixtures.EditorTests.tearDown(self)

    def test_review_requires_saved_page_and_persists_across_navigation(self):
        w = self.window
        second = str(self.root / '02.png')
        cv2.imwrite(second, np.full_like(self.image, 240))
        w.load_folder(str(self.root))
        self.assertIn('第 1/2 頁 · 01.png · 待回看', w.current_page_label.text())
        self.assertEqual(w.current_page_label.accessibleName(), w.current_page_label.text())
        self.assertEqual(w.save_state_label.accessibleName(), w.save_state_label.text())
        self.assertIn('已校對 0', w.summary_label.text())
        self.assertTrue(w.set_current_reviewed(True))
        self.assertIn('已校對 1', w.summary_label.text())
        self.assertIn('01.png · 已校對', w.current_page_label.text())
        self.assertEqual(w.current_page_label.accessibleName(), w.current_page_label.text())
        self.assertTrue(store.is_page_reviewed(store.load_project(self.paths['raw'])['pages']['01.png']))
        w.page_jump_input.setText('2')
        w.jump_to_page()
        self.assertEqual(w.current_img_path, second)
        self.assertIn('第 2/2 頁 · 02.png', w.current_page_label.text())
        self.assertIn('尚未偵測', w.save_state_label.text())
        self.assertFalse(w.set_current_reviewed(True))
        w.page_jump_input.setText('01.png')
        w.jump_to_page()
        self.assertEqual(w.current_img_path, self.path)
        self.assertIn('01.png · 已校對', w.current_page_label.text())

    def test_review_invalidates_on_edit_and_complete_moves_next(self):
        w = self.window
        second = str(self.root / '02.png')
        cv2.imwrite(second, np.full_like(self.image, 240))
        w.load_folder(str(self.root))
        self.assertTrue(w.set_current_reviewed(True))
        w.set_edit_mode('manual_other')
        changed = w.current_manual_other.copy()
        changed[10:15, 10:15] = 255
        w.on_mask_edited(changed)
        self.assertIn('保存中', w.save_state_label.text())
        self.assertEqual(w.save_state_label.accessibleName(), w.save_state_label.text())
        self.assertIn('待回看', w.current_page_label.text())
        self.assertTrue(w.flush_pending_edits())
        self.assertIn('已保存', w.save_state_label.text())
        self.assertEqual(w.save_state_label.accessibleName(), w.save_state_label.text())
        self.assertIn('已校對 0', w.summary_label.text())
        w.complete_page_and_next()
        self.assertEqual(w.current_img_path, second)
        self.assertIn('已校對 1', w.summary_label.text())

    def test_restore_original_clears_both_classes_and_undo_restores(self):
        w = self.window
        state = store.load_page(self.paths, self.path, self.mask.shape)
        state['overlay'][65:70, 84:89] = (255, 255, 255, 255)
        state['overlay'][75:80, 84:89] = 0
        state['other'][75:80, 84:89] = 255
        store.save_page(self.paths, self.path, state)
        w.reload_current()
        original_solid = w.current_manual_solid.copy()
        original_other = w.current_manual_other.copy()
        w.set_edit_tool('rect')
        w.set_selection_combine_mode('restore_original')
        w.mask_view._apply_rect((83, 64), (90, 81), Qt.MouseButton.LeftButton)
        self.assertTrue(w.flush_pending_edits())
        self.assertFalse(np.any(w.current_manual_solid[64:82, 83:91]))
        self.assertFalse(np.any(w.current_manual_other[64:82, 83:91]))
        np.testing.assert_array_equal(
            w.current_manual_solid[80:90, 100:105], original_solid[80:90, 100:105]
        )
        w.undo_mask()
        self.assertTrue(w.flush_pending_edits())
        np.testing.assert_array_equal(w.current_manual_solid, original_solid)
        np.testing.assert_array_equal(w.current_manual_other, original_other)

    def test_tool_switch_keeps_compatible_subtract_and_rejects_restore_on_brush(self):
        w = self.window
        w.set_selection_combine_mode('subtract')
        for tool in ('brush', 'magic', 'lasso', 'rect'):
            w.set_edit_tool(tool)
            self.assertEqual(w.selection_combine_mode, 'subtract')
            self.assertIn('減去', w.edit_state_label.text())
            self.assertEqual(w.edit_state_label.accessibleName(), w.edit_state_label.text())
        w.set_selection_combine_mode('restore_original')
        self.assertIn('恢復原圖', w.edit_state_label.text())
        w.set_edit_tool('lasso')
        self.assertEqual(w.selection_combine_mode, 'restore_original')
        w.set_edit_tool('brush')
        self.assertEqual(w.selection_combine_mode, 'add')

    def test_text_fields_keep_letters_and_arrows_without_navigation(self):
        w = self.window
        second = str(self.root / '02.png')
        cv2.imwrite(second, np.full_like(self.image, 240))
        w.load_folder(str(self.root))
        w.show()
        self.app.processEvents()
        w.folder_path_input.clear()
        w.folder_path_input.setFocus()
        QTest.keyClicks(w.folder_path_input, 'a/d')
        QTest.keyClick(w.folder_path_input, Qt.Key.Key_Left)
        QTest.keyClick(w.folder_path_input, Qt.Key.Key_Right)
        self.assertEqual(w.folder_path_input.text(), 'a/d')
        self.assertEqual(w.list_widget.currentRow(), 0)
        w.page_jump_input.setFocus()
        QTest.keyClicks(w.page_jump_input, 'a.jpg')
        self.assertEqual(w.page_jump_input.text(), 'a.jpg')
        self.assertEqual(w.list_widget.currentRow(), 0)
        QTest.keyClick(w.page_jump_input, Qt.Key.Key_F10)
        self.assertEqual(w.selection_combine_mode, 'subtract')
        spin = w.other_mask_expand_spinbox
        spin.setFocus()
        before = spin.value()
        QTest.keyClick(spin, Qt.Key.Key_Up)
        self.assertEqual(spin.value(), before + 1)
        self.assertEqual(w.list_widget.currentRow(), 0)
        w.mask_view.setFocus()
        QTest.keyClick(w.mask_view, Qt.Key.Key_F3)
        self.assertEqual(w.selection_combine_mode, 'restore_original')
        QTest.keyClick(w.mask_view, Qt.Key.Key_D)
        self.assertEqual(w.list_widget.currentRow(), 1)
        QTest.keyClick(w.mask_view, Qt.Key.Key_A)
        self.assertEqual(w.list_widget.currentRow(), 0)

    def test_return_jump_restores_canvas_focus_and_function_keys(self):
        w = self.window
        second = str(self.root / '02.png')
        cv2.imwrite(second, np.full_like(self.image, 240))
        w.load_folder(str(self.root))
        w.show()
        self.app.processEvents()
        w.page_jump_input.setFocus()
        QTest.keyClicks(w.page_jump_input, '2')
        QTest.keyClick(w.page_jump_input, Qt.Key.Key_Return)
        self.assertEqual(w.current_img_path, second)
        self.assertIs(w.focusWidget(), w.mask_view)
        for key, tool in ((Qt.Key.Key_F10, 'rect'),
                          (Qt.Key.Key_F7, 'magic'),
                          (Qt.Key.Key_F5, 'rect')):
            QTest.keyClick(w.focusWidget(), key)
            self.assertEqual(w.mask_view.tool, tool)
            self.assertEqual(w.selection_combine_mode, 'subtract')
        w.folder_path_input.setFocus()
        QTest.keyClick(w.folder_path_input, Qt.Key.Key_Return)
        self.assertEqual(w.folder, str(self.root))
        self.assertIs(w.focusWidget(), w.mask_view)

    def test_save_failure_cannot_mark_page_reviewed(self):
        w = self.window
        w.set_edit_mode('manual_other')
        changed = w.current_manual_other.copy()
        changed[10:15, 10:15] = 255
        with patch.object(store, 'save_page', side_effect=OSError('disk failure')), \
                patch.object(QMessageBox, 'warning'):
            w.on_mask_edited(changed)
            self.assertFalse(w.set_current_reviewed(True))
            self.assertIn('失敗', w.save_state_label.text())
        self.assertFalse(store.is_page_reviewed(
            store.load_project(self.paths['raw'])['pages']['01.png']))
        self.assertTrue(w.flush_pending_edits())

    def test_control_open_failure_returns_error_without_modal_dialog(self):
        w = self.window
        missing = str(self.root / 'missing-folder')
        with patch.object(QMessageBox, 'warning', side_effect=AssertionError('modal dialog')):
            self.assertFalse(w.load_folder(missing, silent=True))
        self.assertIn('文件夾不存在', w._last_control_error)
        self.assertEqual(w.current_img_path, self.path)

    def test_local_roi_accessibility_tracks_bounds_and_mode(self):
        w = self.window
        dialog = ui.LocalEditDialog(
            w.current_base, w.current_manual_solid, (5, 10, 45, 55),
            w.mask_display_color, w.alpha,
        )
        self.addCleanup(dialog.close)
        dialog.set_roi_box((7, 9, 43, 52))
        self.assertIn('ROI X 7 Y 9 · 36 × 43 px', dialog.roi_state_label.text())
        self.assertEqual(dialog.roi_state_label.accessibleName(), dialog.roi_state_label.text())
        dialog.set_edit_mode('manual_other')
        self.assertIn('圖像修補', dialog.roi_state_label.accessibleName())


if __name__ == '__main__':
    unittest.main()
