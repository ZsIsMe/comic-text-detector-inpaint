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


if __name__ == '__main__':
    unittest.main()
