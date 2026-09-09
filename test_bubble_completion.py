"""Small geometric regression tests; no detector or external layout module."""
import unittest

import cv2
import numpy as np

from bubble_completion import complete_region, symmetry_error, reflect_mask


class CompletionTests(unittest.TestCase):
    def partition(self, body, cut, seeds):
        working = body.copy()
        cv2.line(working, tuple(cut['a']), tuple(cut['b']), 0, 3)
        _, labels = cv2.connectedComponents(working, connectivity=4)
        return [(labels == labels[y, x]).astype(np.uint8)*255 for x, y in seeds]

    def test_overlapping_rectangles(self):
        left = np.zeros((260, 250), np.uint8)
        right = np.zeros_like(left)
        cv2.rectangle(left, (20, 80), (130, 230), 255, -1)
        cv2.rectangle(right, (95, 25), (215, 180), 255, -1)
        body = left | right
        cut = {'a': [95, 80], 'b': [130, 180]}
        regions = self.partition(body, cut, [(65, 155), (165, 100)])
        completed = []
        snapshot = body.copy()
        for region, expected in zip(regions, (left, right)):
            original = region.copy()
            result, outline, debug = complete_region(region, body, [cut])
            self.assertEqual(debug['status'], 'completed')
            self.assertEqual(debug['shape'], 'rounded_rectangle')
            self.assertTrue(np.array_equal(original, region))
            self.assertTrue(np.all(result[region > 0] > 0))
            self.assertFalse(np.any((result > 0) & (body == 0)))
            self.assertGreater(np.count_nonzero(result & expected)/np.count_nonzero(expected), .98)
            self.assertTrue(outline.any())
            completed.append(result)
        self.assertTrue(np.any(completed[0] & completed[1]))
        self.assertTrue(np.array_equal(snapshot, body))

    def test_curved_outline(self):
        left = np.zeros((280, 300), np.uint8)
        right = np.zeros_like(left)
        cv2.ellipse(left, (100, 140), (65, 95), 0, 0, 360, 255, -1)
        cv2.ellipse(right, (190, 140), (65, 95), 0, 0, 360, 255, -1)
        body = left | right
        cut = {'a': [145, 0], 'b': [145, 279]}
        regions = self.partition(body, cut, [(100, 140), (190, 140)])
        for region in regions:
            result, outline, debug = complete_region(region, body, [cut])
            self.assertEqual(debug['status'], 'completed')
            self.assertEqual(debug['shape'], 'mirrored_outline')
            self.assertGreater(np.count_nonzero(result), np.count_nonzero(region))
            self.assertFalse(np.any((result > 0) & (body == 0)))
            self.assertLess(debug['symmetry_error_after'], .02)
            self.assertLess(debug['symmetry_error_after'], debug['symmetry_error_before'])

    def test_nonelliptical_outline_and_horizontal_cut(self):
        # Egg-shaped, left/right symmetric, but not top/bottom symmetric.
        left = np.zeros((300, 300), np.uint8)
        for y in range(35, 256):
            t = (y-145)/110
            radius = int(65*np.sqrt(max(0, 1-t*t))*(1+.22*t))
            left[y, 100-radius:101+radius] = 255
        right = cv2.warpAffine(left, np.float32([[1, 0, 90], [0, 1, 0]]), (300, 300))
        for transpose in (False, True):
            body = left | right
            cut = {'a': [145, 0], 'b': [145, 299]}
            seed = (100, 145)
            expected = left
            if transpose:
                body = body.T.copy()
                cut = {key: point[::-1] for key, point in cut.items()}
                seed = seed[::-1]
                expected = left.T
            region = self.partition(body, cut, [seed])[0]
            result, _, debug = complete_region(region, body, [cut])
            self.assertEqual(debug['status'], 'completed')
            self.assertEqual(debug['shape'], 'mirrored_outline')
            self.assertLess(symmetry_error(result, debug['axis']), .02)
            iou = np.count_nonzero(result & expected)/np.count_nonzero(result | expected)
            self.assertGreater(iou, .98)

    def test_tilted_axis_reflection(self):
        mask = np.zeros((180, 180), np.uint8)
        cv2.ellipse(mask, (90, 90), (25, 55), -10, 0, 360, 255, -1)
        axis = {'point': [90, 90], 'direction': [np.sin(np.deg2rad(10)), np.cos(np.deg2rad(10))]}
        self.assertLess(symmetry_error(mask, axis), .04)
        twice = reflect_mask(reflect_mask(mask, axis), axis)
        self.assertLess(np.count_nonzero(mask != twice)/np.count_nonzero(mask), .04)

    def test_missing_axis_evidence_is_not_guessed(self):
        body = np.zeros((150, 150), np.uint8)
        cv2.circle(body, (75, 75), 55, 255, -1)
        cut = {'a': [75, 0], 'b': [75, 149]}
        region = self.partition(body, cut, [(100, 75)])[0]
        result, outline, debug = complete_region(region, body, [cut])
        self.assertEqual(debug['status'], 'insufficient_symmetry_evidence')
        self.assertTrue(np.array_equal(result, region))
        self.assertFalse(outline.any())

    def test_no_cut_no_completion(self):
        mask = np.zeros((100, 100), np.uint8)
        cv2.circle(mask, (50, 50), 30, 255, -1)
        result, outline, debug = complete_region(mask, mask, [])
        self.assertTrue(np.array_equal(result, mask))
        self.assertFalse(outline.any())
        self.assertEqual(debug['status'], 'unchanged')

    def test_empty_region(self):
        mask = np.zeros((50, 50), np.uint8)
        result, outline, debug = complete_region(mask, mask, [{'a': [10, 0], 'b': [10, 49]}])
        self.assertEqual(debug['status'], 'empty_region')
        self.assertFalse(result.any() or outline.any())


if __name__ == '__main__':
    unittest.main()
