"""Local processing must preserve full-page sampling and classification results."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import detect_solid_inpaint_folder as fill


def _full_page_block(text_mask, box, padding):
    return np.s_[:, :], fill._mask_for_box(text_mask, box), (0, 0)


class FillRoiTests(unittest.TestCase):
    def test_prepare_dirty_roi_matches_full_page_at_edges_and_saved_color_contacts(self):
        import project_store as store
        rng = np.random.default_rng(29)
        color = rng.integers(190, 251, (180, 240, 3), dtype=np.uint8)
        for source in (color, cv2.cvtColor(color, cv2.COLOR_BGR2GRAY),
                       cv2.cvtColor(color, cv2.COLOR_BGR2BGRA).astype(np.uint16) * 256 + 119):
            for box in ((48, 40, 58, 60), (0, 0, 9, 7), (226, 170, 240, 180), (110, 90, 125, 110)):
                with self.subTest(source=source.shape, box=box):
                    state = store.empty_page(source.shape[:2])
                    state['overlay'][40:60, 40:48] = [160, 210, 250, 255]
                    state['overlay'][40:60, 58:66] = [90, 100, 150, 255]
                    state['other'][88:110, 126:129] = 255
                    solid = state['overlay'][:, :, 3].copy()
                    other = state['other'].copy()
                    x1, y1, x2, y2 = box
                    solid[y1:y2, x1:x2] = 255
                    before = {key: array.copy() for key, array in state.items()}
                    with patch.object(fill, 'imread', side_effect=AssertionError('prepare read disk')), \
                            patch.object(store, 'save_page', side_effect=AssertionError('prepare wrote disk')):
                        expected = fill.prepare_page_edits(source, solid, other, state=state)
                        actual = fill.prepare_page_edits(source, solid, other, state=state, dirty_bbox=box)
                    for key in actual:
                        np.testing.assert_array_equal(actual[key], expected[key])
                        np.testing.assert_array_equal(state[key], before[key])
                    if x1 == 48:
                        np.testing.assert_array_equal(actual['overlay'][50, 48], [160, 210, 250, 255])
                        np.testing.assert_array_equal(actual['overlay'][50, 57], [90, 100, 150, 255])

    def test_prepare_dirty_clear_keeps_unaffected_colors_and_edit_locks(self):
        import project_store as store
        image = np.full((70, 90, 3), 230, np.uint8)
        state = store.empty_page(image.shape[:2])
        state['overlay'][10:20, 12:25] = [80, 150, 200, 255]
        state['overlay'][50:60, 50:65] = [90, 100, 110, 127]
        state['other'][25:35, 18:28] = 255
        state['edited'][55:60, 10:15] = 255
        solid, other = state['overlay'][:, :, 3].copy(), state['other'].copy()
        solid[15:31, 20:26] = 0
        other[15:31, 20:26] = 0
        expected = fill.prepare_page_edits(image, solid, other, state=state)
        actual = fill.prepare_page_edits(image, solid, other, state=state, dirty_bbox=(20, 15, 26, 31))
        for key in actual:
            np.testing.assert_array_equal(actual[key], expected[key])
        np.testing.assert_array_equal(actual['overlay'][50:60, 50:65], state['overlay'][50:60, 50:65])
        self.assertTrue(np.all(actual['edited'][55:60, 10:15] == 255))

    def test_manual_fill_matches_full_page_at_edges_and_with_excluded_neighbors(self):
        rng = np.random.default_rng(18)
        image = rng.integers(190, 251, (120, 160, 3), dtype=np.uint8)
        solid = np.zeros(image.shape[:2], np.uint8)
        for x, y in ((0, 0), (18, 14), (78, 45), (150, 115)):
            solid[y:y+7, x:x+6] = 255
        exclude = solid.copy()
        exclude[44:57, 84:90] = 255
        image[exclude > 0] = 0
        # Exclude the inner ring so the expanded fallback ring is required.
        exclude[:11, :11] = 255
        with patch.object(fill, '_block_roi', _full_page_block):
            expected = fill._manual_solid_overlay(image, solid, exclude)
        actual = fill._manual_solid_overlay(image, solid, exclude)
        np.testing.assert_array_equal(expected[0], actual[0])
        self.assertEqual(expected[1:], actual[1:])

    def test_manual_fill_morphology_only_uses_padded_local_regions(self):
        image = np.full((1400, 1600, 3), 210, np.uint8)
        solid = np.zeros(image.shape[:2], np.uint8)
        solid[600:624, 700:718] = 255
        with patch.object(fill.cv2, 'dilate', wraps=cv2.dilate) as dilate:
            overlay, regions, pixels = fill._manual_solid_overlay(image, solid, solid)
        self.assertEqual((regions, pixels), (1, 24 * 18))
        self.assertTrue(dilate.called)
        for call in dilate.call_args_list:
            padding = fill.BLOCK_PADDING_PX * 2 + fill.SAMPLE_RING_PX * 4
            self.assertLessEqual(call.args[0].shape[0], 24 + padding)
            self.assertLessEqual(call.args[0].shape[1], 18 + padding)
        self.assertTrue(np.all(overlay[600:624, 700:718] == [210, 210, 210, 255]))

    def test_wand_matches_full_page_at_edges_and_global_offsets(self):
        for x, y in [(0, 0), (28, 26), (600, 500), (910, 710)]:
            with self.subTest(x=x, y=y):
                image = np.full((800, 1000, 3), 120, np.uint8)
                image[max(0, y-24):min(800, y+60), max(0, x-22):min(1000, x+62)] = 255
                text = np.zeros(image.shape[:2], np.uint8)
                text[y:y+31, x:x+29] = 255
                image[text > 0] = 0
                repair, ring = fill._sample_ring_for_local_text(
                    text, text, fill._kernel(fill.REPAIR_EXPAND_PX), fill._kernel(fill.SAMPLE_RING_PX))
                expected = fill._background_wand_sample_local(image, text, repair, ring)
                actual = fill._background_wand_sample(image, text, repair, ring)
                np.testing.assert_array_equal(expected, actual)
                if x == 600:
                    self.assertGreater(np.count_nonzero(actual), np.count_nonzero(ring))

    def test_disconnected_fallback_ring_is_preserved(self):
        image = np.full((280, 350, 3), 20, np.uint8)
        text = np.zeros(image.shape[:2], np.uint8)
        text[120:140, 160:180] = 255
        repair = cv2.dilate(text, fill._kernel(3))
        ring = np.zeros_like(text)
        ring[5:8, 8:30] = 255
        ring[270:274, 300:325] = 255
        expected = fill._background_wand_sample_local(image, text, repair, ring)
        np.testing.assert_array_equal(expected, fill._background_wand_sample(image, text, repair, ring))

    def test_classification_matches_full_page_with_edges_and_manual_protection(self):
        image = np.full((330, 440, 3), 245, np.uint8)
        text = np.zeros(image.shape[:2], np.uint8)
        for x, y in [(0, 6), (112, 138), (379, 265)]:
            text[y:y+27, x:x+18] = 255
        # Text-like noise within block padding retains the original ownership.
        text[135, 109] = 255
        image[text > 0] = 0
        image[255:318:3, 366:428] = 180
        polygon = np.array([[70, 95], [205, 95], [205, 208], [70, 208]], np.float32)
        protected = np.zeros_like(text)
        protected[142:147, 118:123] = 255
        for polygons, protection in [(None, None), ([polygon], None), ([polygon], protected)]:
            with self.subTest(bubbles=bool(polygons), protected=protection is not None):
                with patch.object(fill, '_block_roi', _full_page_block), patch.object(
                        fill, '_background_wand_sample', fill._background_wand_sample_local):
                    expected = fill._solid_overlay_from_mask(image, text, polygons, protected=protection)
                actual = fill._solid_overlay_from_mask(image, text, polygons, protected=protection)
                for before, after in zip(expected[:3], actual[:3]):
                    np.testing.assert_array_equal(before, after)
                self.assertEqual(expected[3], actual[3])

    def test_empty_text_stays_empty(self):
        image = np.full((200, 250, 3), 255, np.uint8)
        text = np.zeros(image.shape[:2], np.uint8)
        overlay, other, sample, report = fill._solid_overlay_from_mask(image, text)
        self.assertFalse(np.any(overlay))
        self.assertFalse(np.any(other))
        self.assertFalse(np.any(sample))
        self.assertEqual(report['blocks'], 0)


if __name__ == '__main__':
    unittest.main()
