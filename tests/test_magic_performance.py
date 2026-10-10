import unittest

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication

from solid_inpaint_ui import MaskEditorView


class MagicComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_seeded_component_matches_all_labels_including_background_and_roi(self):
        source = np.full((128, 160, 3), 220, np.uint8)
        source[20:50, 20:50] = 80
        source[50, 50] = 80  # Diagonal connectivity must remain eight-way.
        source[70:90, 60:100] = 85
        source[5:15, 100:150] = 100  # Outside the fixed initial color range.
        view = MaskEditorView()
        view.set_mask(np.zeros(source.shape[:2], np.uint8), source.shape[:2])
        view.set_source_image(source)
        view.set_magic_scope_px(96)
        for clip in (None, (25, 25, 110, 100)):
            view.set_edit_clip_rect(clip)
            view._magic_stroke_lower = (70, 70, 70)
            view._magic_stroke_upper = (90, 90, 90)
            for point in ((30, 30), (50, 50), (80, 80), (60, 30), (105, 90)):
                with self.subTest(clip=clip, point=point):
                    expected = view._magic_component_at(point)
                    actual = view._magic_component_at(point, seeded=True)
                    self.assertEqual(actual[0], expected[0])
                    np.testing.assert_array_equal(actual[1] == actual[2], expected[1] == expected[2])
        view.cancel_magic_stroke()

    def test_dense_selection_bounds_and_allowed_holes_preserve_release(self):
        view = MaskEditorView()
        shape = (1100, 1300)
        old = np.full(shape, 255, np.uint8)
        view.set_mask(old, shape)
        view.set_selection_combine_mode('local_intersect')
        view.set_local_intersect_offset(0)
        allowed = np.zeros(shape, bool)
        allowed[100:1000, 150:1200] = True
        allowed[400:500, 500:700] = False
        selection = np.zeros(shape, bool)
        selection[200:800, 250:1100] = True
        expected = old.copy()
        expected[allowed & ~selection] = 0
        self.assertTrue(view._apply_magic_stroke_selection(selection, allowed))
        np.testing.assert_array_equal(view.mask, expected)
        self.assertFalse(view._apply_magic_stroke_selection(selection, np.zeros(shape, bool)))

    def test_expand_dense_and_empty_matches_original_morphology(self):
        view = MaskEditorView()
        for radius in (1, 8, 20):
            view.set_magic_expand_px(radius)
            selection = np.zeros((240, 300), bool)
            selection[20:200, 0:260] = True
            if radius <= 16:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)
                expected = cv2.dilate(selection.astype(np.uint8), kernel) > 0
            else:
                expected = cv2.distanceTransform((~selection).astype(np.uint8), cv2.DIST_L2,
                                                 cv2.DIST_MASK_PRECISE) <= radius
            np.testing.assert_array_equal(view._expand_magic_selection(selection), expected)
            empty = np.zeros_like(selection)
            np.testing.assert_array_equal(view._expand_magic_selection(empty), empty)


if __name__ == '__main__':
    unittest.main()
