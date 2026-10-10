"""Local processing must preserve full-page sampling and classification results."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import detect_solid_inpaint_folder as fill


def _full_page_block(text_mask, box, padding):
    return np.s_[:, :], fill._mask_for_box(text_mask, box), (0, 0)


class FillRoiTests(unittest.TestCase):
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
