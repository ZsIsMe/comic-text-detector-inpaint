"""Brush input should update only its dirty region and retain the existing canvas."""
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QImage, QPainter

import solid_inpaint_ui as ui
import test_project_store as fixtures


class BrushInputTests(unittest.TestCase):
    def setUp(self):
        fixtures.EditorTests.setUp(self)
        self.window.set_edit_tool('brush')
        self.window.mask_view.set_brush_radius(3)

    def tearDown(self):
        fixtures.EditorTests.tearDown(self)

    def start_stroke(self, point):
        view = self.window.mask_view
        view._edit_started = False
        view._set_brush_stroke_active(True)
        view._begin_edit_once()
        view._paint_brush(point, Qt.MouseButton.LeftButton)
        view.brushPreviewChanged.emit(view._brush_dirty_rect(point, point))

    def finish_stroke(self):
        view = self.window.mask_view
        view._set_brush_stroke_active(False)
        view.maskEdited.emit(view.mask.copy())
        view._edit_started = False
        self.assertTrue(self.window.flush_pending_edits())

    def render_scene(self):
        height, width = self.window.current_base.shape[:2]
        image = QImage(width, height, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        bounds = QRectF(0, 0, width, height)
        self.window.mask_view.scene().render(painter, bounds, bounds)
        painter.end()
        return image

    def assert_canvas_matches_masks(self):
        w = self.window
        expected = ui._editor_mask_preview(
            w.current_base, w.current_manual_solid, w.current_manual_other,
            w.alpha, w.current_background_sample if w.show_background_sample else None,
            solid_color=w.mask_display_color,
        )
        self.assertEqual(self.render_scene().convertToFormat(QImage.Format.Format_RGB32),
                         ui._qimage_from_bgr(expected).convertToFormat(QImage.Format.Format_RGB32))

    def test_first_press_keeps_pixmap_and_renders_only_touched_tiles(self):
        w = self.window
        w.set_edit_mode('manual_other')
        before = w.mask_view.pixmap_item.pixmap().cacheKey()
        undo_count = len(w.undo_stack)
        with patch.object(w.mask_view, 'set_qimage', side_effect=AssertionError('replaced full canvas')), \
                patch.object(ui, '_editor_mask_preview', wraps=ui._editor_mask_preview) as compose:
            self.start_stroke((15, 15))
            self.render_scene()
        self.assertEqual(w.mask_view.pixmap_item.pixmap().cacheKey(), before)
        self.assertEqual(len(w.undo_stack), undo_count + 1)
        self.assertTrue(compose.called)
        for call in compose.call_args_list:
            self.assertLessEqual(call.args[0].shape[0], ui.LiveMaskOverlayItem.TILE_SIZE)
            self.assertLessEqual(call.args[0].shape[1], ui.LiveMaskOverlayItem.TILE_SIZE)
        self.finish_stroke()

    def test_drag_updates_only_dirty_region_and_preserves_other_pixels_in_both_modes(self):
        w = self.window
        view = w.mask_view
        for mode in ('manual_solid', 'manual_other'):
            with self.subTest(mode=mode):
                opposite = 'manual_other' if mode == 'manual_solid' else 'manual_solid'
                w.set_edit_mode(opposite)
                seed = w.current_edit_mask().copy()
                seed[10:20, 10:20] = 255
                w.on_mask_edited(seed)
                self.assertTrue(w.flush_pending_edits())
                w.set_edit_mode(mode)
                before = w.current_edit_mask().copy()
                opposite_before = w.mask_for_mode(opposite).copy()
                # A difference outside the dirty rectangle must not be pulled
                # into current state by this update.
                view.mask[150, 200] = 255 if before[150, 200] == 0 else 0
                view._paint_line((12, 12), (17, 16), Qt.MouseButton.LeftButton)
                dirty = view._brush_dirty_rect((12, 12), (17, 16))
                rect = dirty.toAlignedRect()
                roi = np.s_[rect.top():rect.bottom() + 1, rect.left():rect.right() + 1]
                expected = before.copy()
                expected[roi] = view.mask[roi]
                expected_other = opposite_before.copy()
                expected_other[roi][expected[roi] > 0] = 0
                with patch.object(w, 'set_current_edit_mask', side_effect=AssertionError('whole-page synchronization')):
                    w.update_brush_live_preview(dirty)
                np.testing.assert_array_equal(w.current_edit_mask(), expected)
                np.testing.assert_array_equal(w.mask_for_mode(opposite), expected_other)
                view.mask[150, 200] = before[150, 200]
                self.assertTrue(w.flush_pending_edits())

    def test_consecutive_strokes_display_and_undo_redo_are_exact(self):
        w = self.window
        w.set_edit_mode('manual_other')
        original = w.current_manual_other.copy()
        self.start_stroke((15, 15))
        self.assert_canvas_matches_masks()
        self.finish_stroke()
        first = w.current_manual_other.copy()
        self.start_stroke((150, 130))
        self.assert_canvas_matches_masks()
        self.finish_stroke()
        final = w.current_manual_other.copy()
        self.assert_canvas_matches_masks()
        w.undo_mask()
        np.testing.assert_array_equal(w.current_manual_other, first)
        self.assert_canvas_matches_masks()
        w.undo_mask()
        np.testing.assert_array_equal(w.current_manual_other, original)
        self.assert_canvas_matches_masks()
        w.redo_mask()
        w.redo_mask()
        np.testing.assert_array_equal(w.current_manual_other, final)
        self.assert_canvas_matches_masks()

    def test_local_first_press_also_keeps_existing_canvas(self):
        w = self.window
        dialog = ui.LocalEditDialog(
            w.current_base, w.current_edit_mask(), (5, 5, 35, 35),
            w.current_edit_color(), w.alpha, w,
        )
        dialog.view.set_tool('brush')
        before = dialog.view.pixmap_item.pixmap().cacheKey()
        with patch.object(dialog.view, 'set_qimage', side_effect=AssertionError('replaced full local canvas')):
            dialog.on_edit_started()
        self.assertEqual(dialog.view.pixmap_item.pixmap().cacheKey(), before)
        self.assertEqual(len(dialog.undo_stack), 1)
        dialog.reject()

    def test_subtract_at_page_edge_and_clipped_strokes_preserve_outside_masks(self):
        w = self.window
        view = w.mask_view
        w.set_edit_mode('manual_other')
        seed = np.zeros_like(w.current_manual_other)
        seed[:8, :8] = 255
        w.on_mask_edited(seed)
        self.assertTrue(w.flush_pending_edits())
        w.set_selection_combine_mode('subtract')
        self.start_stroke((0, 0))
        self.assertEqual(w.current_manual_other[0, 0], 0)
        self.assertEqual(w.current_manual_other[7, 7], 255)
        self.assert_canvas_matches_masks()
        self.finish_stroke()
        before = w.current_manual_other.copy()
        w.set_selection_combine_mode('add')
        view.set_edit_clip_rect((10, 10, 20, 20))
        self.start_stroke((10, 10))
        view._paint_line((10, 10), (19, 19), Qt.MouseButton.LeftButton)
        view.brushPreviewChanged.emit(view._brush_dirty_rect((10, 10), (19, 19)))
        outside = np.ones_like(before, bool)
        outside[10:20, 10:20] = False
        np.testing.assert_array_equal(w.current_manual_other[outside], before[outside])
        self.assertTrue(np.all(w.current_manual_other[range(10, 20), range(10, 20)] == 255))
        self.assert_canvas_matches_masks()
        self.finish_stroke()

    def test_shift_line_release_commits_without_drag_preview(self):
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QMouseEvent
        w = self.window
        view = w.mask_view
        w.set_edit_mode('manual_other')
        start, end = (15, 15), (35, 25)
        view._active_button = Qt.MouseButton.LeftButton
        view._brush_line_start = start
        view._edit_started = False
        view._set_brush_stroke_active(True)
        point = QPointF(view.mapFromScene(QPointF(*end)))
        event = QMouseEvent(QMouseEvent.Type.MouseButtonRelease, point, point,
                            Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
                            Qt.KeyboardModifier.ShiftModifier)
        view.mouseReleaseEvent(event)
        self.assertTrue(w.flush_pending_edits())
        self.assertEqual(w.current_manual_other[start[1], start[0]], 255)
        self.assertEqual(w.current_manual_other[end[1], end[0]], 255)
        self.assert_canvas_matches_masks()
        self.assertEqual(len(w.undo_stack), 1)


if __name__ == '__main__':
    unittest.main()
