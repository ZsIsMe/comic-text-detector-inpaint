"""The explicit review layout works across logical screen geometries."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtCore import QPoint, QRect, QSettings, Qt
from PySide6.QtWidgets import QApplication, QScrollArea, QToolBar
from PySide6.QtTest import QTest

import solid_inpaint_ui as ui


class ReviewWindowTests(unittest.TestCase):
    def test_geometry_centers_and_clamps_on_each_screen(self):
        cases = (
            (QRect(0, 0, 1920, 1080), QRect(210, 40, 1500, 1000)),
            (QRect(-1600, 200, 1280, 720), QRect(-1560, 240, 1200, 640)),
            (QRect(100, 100, 3000, 1600), QRect(850, 400, 1500, 1000)),
        )
        for screen, expected in cases:
            with self.subTest(screen=screen):
                self.assertEqual(ui.review_window_geometry(screen), expected)

    def test_button_arranges_current_screen_and_defers_canvas_fit(self):
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as temp:
            settings = QSettings(str(Path(temp) / 'ui.ini'), QSettings.Format.IniFormat)
            with patch.object(ui, 'QSettings', return_value=settings), \
                    patch('solid_inpaint_control.start_control', return_value=Mock()):
                window = ui.MainWindow()
                window.startup_restore_timer.stop()
                try:
                    action = window.arrange_review_action
                    self.assertEqual(action.text(), '整理監修視窗')
                    screen = SimpleNamespace(availableGeometry=lambda: QRect(-1600, 200, 1280, 720))
                    with patch.object(window, 'screen', return_value=screen), \
                            patch.object(window, 'showNormal') as show_normal, \
                            patch.object(window, 'setGeometry') as set_geometry:
                        action.trigger()
                    show_normal.assert_called_once_with()
                    set_geometry.assert_called_once_with(QRect(-1560, 240, 1200, 640))
                    self.assertTrue(window.resize_fit_timer.isActive())
                    self.assertIn('已整理', window.status.currentMessage())

                    window.resize_fit_timer.stop()
                    window.is_mask_stroke_active = True
                    with patch.object(window, 'showNormal') as show_normal, \
                            patch.object(window, 'setGeometry') as set_geometry:
                        action.trigger()
                    show_normal.assert_not_called()
                    set_geometry.assert_not_called()
                    self.assertFalse(window.resize_fit_timer.isActive())
                    self.assertIn('正在編輯', window.status.currentMessage())
                finally:
                    window.close()
        del app

    def test_review_canvas_uses_space_released_by_hidden_controls(self):
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as temp:
            settings = QSettings(str(Path(temp) / 'ui.ini'), QSettings.Format.IniFormat)
            with patch.object(ui, 'QSettings', return_value=settings), \
                    patch('solid_inpaint_control.start_control', return_value=Mock()):
                window = ui.MainWindow()
                window.startup_restore_timer.stop()
                window.show()
                try:
                    self.assertFalse(window.folder_label.isVisible())
                    self.assertFalse(window.progress_row.isVisible())
                    self.assertFalse(window.project_path_row.isVisible())
                    self.assertFalse(window.view_options_panel.isVisible())
                    self.assertFalse(window.advanced_selection_panel.isVisible())
                    for width, height, min_canvas_height in ((1200, 640, 280), (1500, 1000, 640)):
                        with self.subTest(size=(width, height)):
                            window.resize(width, height)
                            window.set_edit_tool('rect')
                            app.processEvents()
                            self.assertFalse(window.magic_scope_controls.isVisible())
                            self.assertFalse(window.local_intersect_dark_controls.isVisible())
                            self.assertGreaterEqual(window.mask_view.height(), min_canvas_height)
                            self.assertEqual(window.mask_view.height(), window.preview_view.height())

                            rect_height = window.mask_view.height()
                            window.set_edit_tool('brush')
                            app.processEvents()
                            self.assertLess(window.mask_view.height(), rect_height)
                            window.set_edit_tool('magic')
                            app.processEvents()
                            self.assertTrue(window.magic_scope_controls.isVisible())
                            self.assertFalse(window.local_intersect_dark_controls.isVisible())
                            self.assertLess(window.mask_view.height(), rect_height)
                            window.set_edit_tool('rect')
                            window.set_selection_combine_mode('local_intersect')
                            app.processEvents()
                            self.assertTrue(window.local_intersect_dark_controls.isVisible())
                            window.set_selection_combine_mode('add')
                            window.advanced_selection_toggle.setChecked(False)
                            app.processEvents()
                            self.assertFalse(window.local_intersect_dark_controls.isVisible())
                            self.assertEqual(window.mask_view.height(), rect_height)

                    window.resize(800, 900)
                    window.page_list_toggle.setChecked(True)
                    window.view_options_toggle.setChecked(True)
                    window.advanced_selection_toggle.setChecked(True)
                    window.review_tools_action.setChecked(True)
                    app.processEvents()
                    self.assertLessEqual(window.width(), 800)
                    self.assertTrue(window.project_path_row.isVisible())
                    self.assertFalse(window.workspace_panel.findChildren(QScrollArea))
                    for tool, mode in (('rect', 'local_intersect'), ('magic', 'add'), ('brush', 'add')):
                        window.set_edit_tool(tool)
                        window.set_selection_combine_mode(mode)
                        for _ in range(3):
                            app.processEvents()
                        panels = (window.edit_controls, window.advanced_selection_panel,
                                  window.view_options_panel, window.local_intersect_dark_controls,
                                  window.magic_scope_controls, window.magic_controls)
                        for panel in panels:
                            if panel.isVisible():
                                for control in panel.findChildren(ui.QWidget):
                                    if control.isVisible() and not control.parent().inherits('QAbstractSpinBox'):
                                        origin = control.mapTo(panel, QPoint())
                                        self.assertGreaterEqual(origin.x(), 0)
                                        self.assertLessEqual(origin.x() + control.width(), panel.width())
                    toolbar = window.findChild(QToolBar)
                    for button in (window.open_button, window.more_button, window.detect_button):
                        self.assertTrue(button.isVisible())
                        self.assertTrue(toolbar.rect().contains(button.mapTo(toolbar, QPoint())))

                    self.assertFalse(window.progress_row.isVisible())
                    with patch.object(window, 'save_all_edit_masks', return_value=True), \
                            patch.object(ui, 'QThread') as thread, \
                            patch.object(ui, 'FolderWorker'):
                        self.assertTrue(window.start_worker('detect', []))
                        thread.return_value.start.assert_called_once_with()
                        self.assertTrue(window.progress_row.isVisible())
                        window.on_worker_progress(1, 2, '1.jpg')
                        app.processEvents()
                        self.assertEqual(window.progress.value(), 50)
                        window.cleanup_worker()
                        app.processEvents()
                        self.assertFalse(window.progress_row.isVisible())
                finally:
                    window.close()
        del app

    def test_compact_menus_share_button_and_shortcut_state_and_follow_sidebar(self):
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as temp:
            settings = QSettings(str(Path(temp) / 'ui.ini'), QSettings.Format.IniFormat)
            with patch.object(ui, 'QSettings', return_value=settings), \
                    patch('solid_inpaint_control.start_control', return_value=Mock()):
                window = ui.MainWindow()
            window.startup_restore_timer.stop()
            window.resize(1250, 900)
            window.show()
            try:
                def settle():
                    for _ in range(5):
                        app.processEvents()

                def choose(button):
                    action = next(action for action, candidate in window.compact_edit_actions
                                  if candidate is button)
                    self.assertTrue(action.isEnabled())
                    action.trigger()
                    settle()
                    self.assertTrue(action.isChecked())
                    self.assertTrue(button.isChecked())

                window.page_list_toggle.setChecked(False)
                settle()
                self.assertTrue(window.edit_shape_controls.isVisible())
                self.assertFalse(window.compact_edit_controls.isVisible())
                window.page_list_toggle.setChecked(True)
                settle()
                self.assertFalse(window.edit_shape_controls.isVisible())
                self.assertTrue(window.compact_edit_controls.isVisible())
                choose(window.lasso_btn)
                self.assertEqual(window.mask_view.tool, 'lasso')
                self.assertEqual(window.compact_edit_buttons['tool'].text(), '工具：套索')
                choose(window.selection_subtract_btn)
                self.assertEqual(window.selection_combine_mode, 'subtract')
                self.assertEqual(window.compact_edit_buttons['operation'].text(), '操作：減去')
                QTest.keyClick(window, Qt.Key.Key_F5)
                settle()
                self.assertEqual(window.mask_view.tool, 'rect')
                self.assertEqual(window.compact_edit_buttons['tool'].text(), '工具：矩形')
                choose(window.selection_transfer_btn)
                choose(window.brush_btn)
                self.assertEqual(window.selection_combine_mode, 'add')
                self.assertEqual(window.compact_edit_buttons['operation'].text(), '操作：添加')
                transfer = next(action for action, button in window.compact_edit_actions
                                if button is window.selection_transfer_btn)
                self.assertFalse(transfer.isEnabled())
                choose(window.edit_manual_other_btn)
                self.assertEqual(window.edit_mode, 'manual_other')
                self.assertEqual(window.compact_edit_buttons['target'].text(), '類型：圖像修補')
                window.resize(1500, 900)
                settle()
                self.assertTrue(window.edit_shape_controls.isVisible())
                self.assertTrue(window.brush_btn.isChecked())
                self.assertTrue(window.selection_add_btn.isChecked())
                self.assertTrue(window.edit_manual_other_btn.isChecked())
                self.assertFalse(window.compact_edit_controls.isVisible())
            finally:
                window.close()


if __name__ == '__main__':
    unittest.main()
