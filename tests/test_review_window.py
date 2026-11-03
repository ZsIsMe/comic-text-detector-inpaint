"""The explicit review layout works across logical screen geometries."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtCore import QRect, QSettings
from PySide6.QtWidgets import QApplication

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
                    button = window.arrange_review_button
                    self.assertEqual(button.accessibleName(), '整理監修視窗')
                    screen = SimpleNamespace(availableGeometry=lambda: QRect(-1600, 200, 1280, 720))
                    with patch.object(window, 'screen', return_value=screen), \
                            patch.object(window, 'showNormal') as show_normal, \
                            patch.object(window, 'setGeometry') as set_geometry:
                        button.click()
                    show_normal.assert_called_once_with()
                    set_geometry.assert_called_once_with(QRect(-1560, 240, 1200, 640))
                    self.assertTrue(window.resize_fit_timer.isActive())
                    self.assertIn('已整理', window.status.currentMessage())

                    window.resize_fit_timer.stop()
                    window.is_mask_stroke_active = True
                    with patch.object(window, 'showNormal') as show_normal, \
                            patch.object(window, 'setGeometry') as set_geometry:
                        button.click()
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
                            self.assertEqual(window.mask_view.height(), rect_height)
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
                            app.processEvents()
                            self.assertFalse(window.local_intersect_dark_controls.isVisible())
                            self.assertEqual(window.mask_view.height(), rect_height)

                    window.resize(1200, 640)
                    app.processEvents()
                    for scroll in (window.edit_toolbar_scroll, window.view_options_scroll):
                        with self.subTest(scroll=scroll):
                            bar = scroll.horizontalScrollBar()
                            self.assertGreater(bar.maximum(), 0)
                            self.assertGreaterEqual(scroll.viewport().height(), scroll.widget().sizeHint().height())
                            bar.setValue(bar.maximum())
                            self.assertEqual(bar.value(), bar.maximum())

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


if __name__ == '__main__':
    unittest.main()
