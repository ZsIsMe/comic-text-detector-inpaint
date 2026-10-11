"""Dispatch real viewport wheel events, including mixed native delta formats."""
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication

import solid_inpaint_ui as ui
import test_project_store as fixtures


class ViewWheelTests(unittest.TestCase):
    def setUp(self):
        fixtures.EditorTests.setUp(self)
        self.keyboard_patch = patch.object(QApplication, 'queryKeyboardModifiers',
                                           return_value=Qt.KeyboardModifier.NoModifier)
        self.keyboard_query = self.keyboard_patch.start()

    def tearDown(self):
        self.keyboard_patch.stop()
        fixtures.EditorTests.tearDown(self)

    def wheel(self, pixel, angle, modifiers=Qt.KeyboardModifier.AltModifier):
        view = self.window.mask_view
        point = QPointF(view.viewport().rect().center())
        event = QWheelEvent(point, point, QPoint(*pixel), QPoint(*angle),
                            Qt.MouseButton.NoButton, modifiers,
                            Qt.ScrollPhase.NoScrollPhase, False)
        QApplication.sendEvent(view.viewport(), event)
        self.assertTrue(event.isAccepted())

    def test_alt_zoom_handles_vertical_mixed_and_horizontal_wheel_deltas(self):
        w = self.window
        for pixel, angle, direction in (
            ((0, 0), (0, 120), 1),
            ((0, 0), (0, -120), -1),
            ((0, 18), (0, 0), 1),
            ((1, 0), (0, 120), 1),  # A horizontal pixel must not hide vertical angle.
            ((1, 0), (0, -120), -1),  # Vertical delta has priority over x.
            ((18, 0), (0, 0), 1),
            ((0, 0), (-120, 0), -1),
        ):
            with self.subTest(pixel=pixel, angle=angle):
                w.mask_view.actual_size()
                self.wheel(pixel, angle)
                expected = ui.VIEW_ZOOM_STEP if direction > 0 else 1 / ui.VIEW_ZOOM_STEP
                self.assertAlmostEqual(w.mask_view.transform().m11(), expected)
                self.assertAlmostEqual(w.preview_view.transform().m11(), expected)
        self.keyboard_query.assert_not_called()

    def test_real_time_alt_state_fills_missing_wheel_modifier(self):
        view = self.window.mask_view
        view.actual_size()
        self.keyboard_query.return_value = Qt.KeyboardModifier.AltModifier
        self.wheel((0, 0), (0, 120), Qt.KeyboardModifier.NoModifier)
        self.assertAlmostEqual(view.transform().m11(), ui.VIEW_ZOOM_STEP)
        self.keyboard_query.assert_called_once()

    def test_real_time_control_does_not_replace_event_modifiers(self):
        view = self.window.mask_view
        view.actual_size()
        self.keyboard_query.return_value = Qt.KeyboardModifier.ControlModifier
        with patch.object(view, 'pan_by') as pan:
            self.wheel((0, 0), (0, 120), Qt.KeyboardModifier.NoModifier)
        pan.assert_called_once_with(0, -120)
        self.assertAlmostEqual(view.transform().m11(), 1)

    def test_plain_and_control_wheels_keep_pan_behavior(self):
        view = self.window.mask_view
        view.actual_size()
        for modifiers, expected in (
            (Qt.KeyboardModifier.NoModifier, (0, -120)),
            (Qt.KeyboardModifier.ControlModifier, (-120, 0)),
        ):
            with self.subTest(modifiers=modifiers), patch.object(view, 'pan_by') as pan:
                self.wheel((0, 0), (0, 120), modifiers)
                pan.assert_called_once_with(*expected)
                self.assertAlmostEqual(view.transform().m11(), 1)

    def test_alt_wheel_changes_no_masks_or_history_in_every_edit_tool(self):
        w = self.window
        before = {key: value.copy() for key, value in w.current_page.items()}
        undo_count, redo_count = len(w.undo_stack), len(w.redo_stack)
        for tool in ('rect', 'brush', 'lasso', 'magic'):
            with self.subTest(tool=tool):
                w.set_edit_tool(tool)
                w.mask_view.actual_size()
                self.wheel((0, 0), (0, 120))
                self.assertAlmostEqual(w.mask_view.transform().m11(), ui.VIEW_ZOOM_STEP)
                for key in before:
                    np.testing.assert_array_equal(w.current_page[key], before[key])
                self.assertEqual(len(w.undo_stack), undo_count)
                self.assertEqual(len(w.redo_stack), redo_count)

    def test_zero_delta_does_not_change_scale(self):
        view = self.window.mask_view
        view.actual_size()
        self.wheel((0, 0), (0, 0))
        self.assertAlmostEqual(view.transform().m11(), 1)


if __name__ == '__main__':
    unittest.main()
