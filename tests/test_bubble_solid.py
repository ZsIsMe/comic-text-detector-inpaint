from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

import bubble_solid as bubble
import project_store as store
from detect_solid_inpaint_folder import (
    _solid_overlay_from_mask, _quality_from_sample, _ensure_dirs,
    regenerate_image_from_mask, read_page_state, save_page_edits,
)


def scene(color=(255, 255, 255)):
    image = np.full((180, 200, 3), color, np.uint8)
    polygon = np.array([[20, 20], [179, 20], [179, 159], [20, 159]], np.float32)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[70:105, 80:95] = 255
    image[mask > 0] = 0
    return image, mask, [polygon]


def solid_seeds(mask, color=(255, 255, 255)):
    seeds = np.zeros((*mask.shape, 4), np.uint8)
    seeds[mask > 0, :3] = color
    seeds[mask > 0, 3] = 255
    return seeds


class SolidBubbleTests(unittest.TestCase):
    def test_solid_text_expands_to_exact_inset_without_changing_source(self):
        image, mask, polys = scene()
        before = image.copy()
        seeds = solid_seeds(mask, (170, 200, 230))
        overlay, records = bubble.fill_bubbles(mask, seeds, polys)
        region = bubble._polygon_mask(polys[0], mask.shape)
        expected = cv2.erode(region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
        np.testing.assert_array_equal(overlay[:, :, 3] > 0, expected)
        self.assertTrue(np.all(overlay[expected, :3] == [170, 200, 230]))
        np.testing.assert_array_equal(image, before)
        self.assertEqual(records[0]['inset_px'], 3)
        self.assertEqual(records[0]['reason'], 'solid_text_seed')

    def test_no_solid_seed_keeps_bubble_unfilled(self):
        _, mask, polys = scene()
        overlay, records = bubble.fill_bubbles(mask, np.zeros((*mask.shape, 4), np.uint8), polys)
        self.assertFalse(np.any(overlay))
        self.assertEqual(records[0]['reason'], 'no_solid_text_seed')
        self.assertFalse(records[0]['accepted'])

    def test_b_only_does_not_trigger_expansion(self):
        from detect_solid_inpaint_folder import SolidQuality
        image, mask, polys = scene()
        failed = SolidQuality(False, 0, (0, 0, 0), 100, 0, 0, 100, 0, 100, 100, 'full')
        with patch('detect_solid_inpaint_folder._best_quality', return_value=failed):
            output, other, _, report = _solid_overlay_from_mask(image, mask, polys)
        self.assertEqual(report['solid_bubbles'], 0)
        self.assertFalse(report['blocks_debug'][0]['local_is_solid'])
        self.assertFalse(np.any(output))
        self.assertTrue(np.all(other[mask > 0] == 255))

    def test_a_and_b_in_same_bubble_fills_inset_and_clears_auto_b(self):
        from detect_solid_inpaint_folder import SolidQuality
        image = np.full((180, 200, 3), 255, np.uint8)
        mask = np.zeros((180, 200), np.uint8)
        mask[45:65, 50:70] = 255
        mask[95:115, 125:145] = 255
        image[mask > 0] = 0
        polygon = np.array([[20, 20], [179, 20], [179, 159], [20, 159]], np.float32)
        solid = SolidQuality(True, 0, (191, 207, 223), 0, 1, 1, 0, 1, 0, 100, 'full')
        failed = SolidQuality(False, 0, (0, 0, 0), 100, 0, 0, 100, 0, 100, 100, 'full')
        with patch('detect_solid_inpaint_folder._best_quality', side_effect=[solid, failed]):
            output, other, _, report = _solid_overlay_from_mask(image, mask, [polygon])
        self.assertEqual(sum(b['local_is_solid'] for b in report['blocks_debug']), 1)
        self.assertEqual(report['solid_bubbles'], 1)
        self.assertEqual(report['auto_blocks'], 2)
        self.assertFalse(np.any(other))
        np.testing.assert_array_equal(output[105, 135], [191, 207, 223, 255])
        np.testing.assert_array_equal(output[80, 100], [191, 207, 223, 255])

    def test_expansion_uses_a_color_even_when_source_bubble_has_other_color(self):
        from detect_solid_inpaint_folder import SolidQuality
        image, mask, polys = scene((80, 90, 100))
        chosen = SolidQuality(True, 0, (20, 40, 60), 0, 1, 1, 0, 1, 0, 100, 'full')
        with patch('detect_solid_inpaint_folder._best_quality', return_value=chosen):
            output, _, _, report = _solid_overlay_from_mask(image, mask, polys)
        self.assertEqual(report['solid_bubbles'], 1)
        np.testing.assert_array_equal(output[40, 40], [20, 40, 60, 255])
        self.assertEqual(report['bubbles_debug'][0]['fill_bgr'], [20, 40, 60])

    def test_gradient_noise_and_art_do_not_reclassify_existing_a(self):
        from detect_solid_inpaint_folder import SolidQuality
        image, mask, polys = scene()
        chosen = SolidQuality(True, 0, (200, 200, 200), 0, 1, 1, 0, 1, 0, 100, 'full')
        for variation in ('gradient', 'halftone', 'art'):
            with self.subTest(variation=variation):
                source = image.copy()
                if variation == 'gradient':
                    source[:] = np.linspace(100, 250, 200).astype(np.uint8)[None, :, None]
                elif variation == 'halftone':
                    source[30:150:5, 30:170:5] = 0
                else:
                    cv2.circle(source, (145, 45), 15, (0, 0, 0), -1)
                source[mask > 0] = 0
                with patch('detect_solid_inpaint_folder._best_quality', return_value=chosen):
                    output, _, _, report = _solid_overlay_from_mask(source, mask, polys)
                self.assertEqual(report['solid_bubbles'], 1)
                np.testing.assert_array_equal(output[40, 40], [200, 200, 200, 255])

    def test_outside_and_crossing_text_do_not_trigger_expansion(self):
        _, mask, polys = scene()
        outside = np.zeros_like(mask)
        outside[70:100, 2:12] = 255
        overlay, records = bubble.fill_bubbles(outside, solid_seeds(outside), polys)
        self.assertFalse(np.any(overlay))
        self.assertEqual(records[0]['reason'], 'no_solid_text_seed')
        crossing = np.zeros_like(mask)
        crossing[60:90, 5:35] = 255
        overlay, records = bubble.fill_bubbles(crossing, solid_seeds(crossing), polys)
        self.assertFalse(np.any(overlay))
        self.assertEqual(records[0]['reason'], 'no_solid_text_seed')

    def test_tiny_alpha_edge_without_text_seed_cannot_trigger_expansion(self):
        _, mask, polys = scene()
        seeds = np.zeros((*mask.shape, 4), np.uint8)
        seeds[80, 100] = [255, 255, 255, 255]
        overlay, records = bubble.fill_bubbles(mask, seeds, polys)
        self.assertFalse(np.any(overlay))
        self.assertEqual(records[0]['seed_pixels'], 0)

    def test_small_verified_a_still_triggers_expansion(self):
        _, mask, polys = scene()
        tiny = np.zeros_like(mask)
        tiny[80:82, 85:87] = 255
        overlay, records = bubble.fill_bubbles(tiny, solid_seeds(tiny), polys)
        self.assertTrue(records[0]['accepted'])
        self.assertEqual(records[0]['seed_pixels'], 4)
        self.assertEqual(overlay[40, 40, 3], 255)

    def test_multiple_a_colors_choose_largest_seed_deterministically(self):
        _, mask, polys = scene()
        mask[:] = 0
        mask[55:80, 45:65] = 255
        mask[95:110, 120:140] = 255
        seeds = solid_seeds(mask, (10, 20, 30))
        seeds[95:110, 120:140, :3] = [90, 100, 110]
        output, records = bubble.fill_bubbles(mask, seeds, polys)
        np.testing.assert_array_equal(output[40, 40], [10, 20, 30, 255])
        self.assertEqual(records[0]['seed_colors'], 2)
        self.assertEqual(records[0]['seed_color_pixels'], 500)
        # A tie uses lexicographically first BGR independent of detector order.
        seeds[55:80, 45:65, 3] = 0
        seeds[55:70, 45:65] = [10, 20, 30, 255]
        output, _ = bubble.fill_bubbles(mask, seeds, polys)
        np.testing.assert_array_equal(output[40, 40], [10, 20, 30, 255])

    def test_irregular_polygon_does_not_claim_seed_in_bounding_box(self):
        shape = (180, 200)
        mask = np.zeros(shape, np.uint8)
        mask[35:50, 35:50] = 255
        mask[90:105, 120:135] = 255
        seeds = solid_seeds(mask, (200, 200, 200))
        seeds[90:105, 120:135, :3] = [80, 100, 120]
        polygons = [np.array([[20, 20], [175, 20], [175, 65], [70, 65], [70, 155], [20, 155]], np.float32),
                    np.array([[100, 80], [165, 80], [165, 135], [100, 135]], np.float32)]
        output, records = bubble.fill_bubbles(mask, seeds, polygons)
        self.assertEqual(sum(r['accepted'] for r in records), 2)
        np.testing.assert_array_equal(output[40, 40], [200, 200, 200, 255])
        np.testing.assert_array_equal(output[95, 125], [80, 100, 120, 255])
        self.assertEqual(output[70, 120, 3], 0)

    def test_overlapping_polygons_preserve_seam_and_each_color(self):
        shape = (180, 200)
        mask = np.zeros(shape, np.uint8)
        mask[70:95, 45:65] = 255
        mask[70:95, 135:155] = 255
        seeds = solid_seeds(mask, (200, 200, 200))
        seeds[70:95, 135:155, :3] = [80, 100, 120]
        polygons = [np.array([[20, 20], [112, 20], [112, 159], [20, 159]], np.float32),
                    np.array([[87, 20], [179, 20], [179, 159], [87, 159]], np.float32)]
        first, records = bubble.fill_bubbles(mask, seeds, polygons)
        second, _ = bubble.fill_bubbles(mask, seeds, list(reversed(polygons)))
        np.testing.assert_array_equal(first, second)
        self.assertEqual(sum(r['accepted'] for r in records), 2)
        np.testing.assert_array_equal(first[80, 50], [200, 200, 200, 255])
        np.testing.assert_array_equal(first[80, 145], [80, 100, 120, 255])
        self.assertFalse(np.any(first[80, 85:115, 3]))

    def test_solid_text_in_overlap_still_triggers_each_owning_bubble(self):
        mask = np.zeros((180, 200), np.uint8)
        mask[70:95, 95:105] = 255
        polygons = [np.array([[20, 20], [112, 20], [112, 159], [20, 159]], np.float32),
                    np.array([[87, 20], [179, 20], [179, 159], [87, 159]], np.float32)]
        output, records = bubble.fill_bubbles(mask, solid_seeds(mask), polygons)
        self.assertEqual(sum(record['accepted'] for record in records), 2)
        self.assertEqual(output[40, 40, 3], 255)
        self.assertEqual(output[40, 160, 3], 255)
        self.assertFalse(np.any(output[80, 85:115, 3]))

    def test_protected_pixels_are_excluded_without_vetoing_other_inset(self):
        _, mask, polys = scene()
        protected = np.zeros_like(mask)
        protected[40:45, 145:150] = 255
        output, records = bubble.fill_bubbles(mask, solid_seeds(mask), polys, protected=protected)
        self.assertTrue(records[0]['accepted'])
        self.assertFalse(np.any(output[protected > 0, 3]))
        self.assertEqual(output[40, 40, 3], 255)

    def test_no_text_never_triggers_fill(self):
        _, mask, polys = scene()
        empty = mask * 0
        output, records = bubble.fill_bubbles(empty, solid_seeds(empty), polys)
        self.assertFalse(np.any(output))
        self.assertFalse(records[0]['accepted'])

    def test_spatial_spread_remains_available_to_local_classifier(self):
        image = np.full((40, 40, 3), 200, np.uint8)
        sample = np.ones((40, 40), bool)
        spread, cells = bubble.spatial_spread(image, sample)
        self.assertEqual(spread, 0)
        self.assertEqual(cells, 16)

    def test_sparse_flat_external_sample_is_not_an_unknown_gradient(self):
        image = np.full((30, 30, 3), 200, np.uint8)
        sample = np.zeros((30, 30), np.uint8)
        sample[10, 5:17] = 255
        quality = _quality_from_sample(image, sample, 'full')
        self.assertEqual(quality.sampled_cells, 0)
        self.assertTrue(quality.is_solid)


class BubbleIntegrationTests(unittest.TestCase):
    def test_settings_persist_and_invalid_values_are_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(bubble.load_settings(directory), {'enabled': True, 'shrink_percent': 2.0})
            bubble.save_settings(directory, False, 3.5)
            self.assertEqual(bubble.load_settings(directory), {'enabled': False, 'shrink_percent': 3.5})
            store.update_project(directory, settings={'solid_fill': {'enabled':'false', 'shrink_percent':'NaN'}})
            self.assertEqual(bubble.load_settings(directory), {'enabled': True, 'shrink_percent': 2.0})

    def test_regeneration_omits_text_mask_and_restores_manual_other_and_erasure(self):
        with tempfile.TemporaryDirectory() as directory:
            image, mask, polys = scene()
            image[80:84, 111:115] = 0
            path = str(Path(directory)/'page.png')
            cv2.imwrite(path, image)
            paths = _ensure_dirs(directory)
            with patch('detect_solid_inpaint_folder.detect_bubbles', return_value=(polys, 'detected')):
                report = regenerate_image_from_mask(path, paths, mask)
                self.assertEqual(report['solid_bubbles'], 1)
                self.assertNotIn('text_mask', store.read_cache_file(store.cache_path(paths, path)))
                protected = np.zeros_like(mask)
                protected[80:84, 111:115] = 255
                state = read_page_state(paths, path)
                solid = state['overlay'][:, :, 3].copy()
                solid[protected > 0] = 0
                erased = np.zeros_like(mask)
                erased[45:48, 45:48] = 255
                solid[erased > 0] = 0
                save_page_edits(path, paths, solid, protected)
                report = regenerate_image_from_mask(path, paths, mask)
                self.assertEqual(report['solid_bubbles'], 1)
                result = read_page_state(paths, path)
                self.assertFalse(np.any(result['overlay'][protected > 0, 3]))
                self.assertTrue(np.all(result['other'][protected > 0] == 255))
                self.assertFalse(np.any(result['overlay'][erased > 0, 3]))
                self.assertFalse(np.any(result['other'][erased > 0]))
                self.assertEqual(result['overlay'][40, 40, 3], 255)

    def test_disabled_or_missing_model_falls_back_and_reports_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            image, mask, _ = scene()
            path = str(Path(directory)/'page.png')
            cv2.imwrite(path, image)
            paths = _ensure_dirs(directory)
            with patch('detect_solid_inpaint_folder.detect_bubbles', side_effect=FileNotFoundError('missing')) as detect:
                report = regenerate_image_from_mask(path, paths, mask)
                self.assertEqual(report['bubble_detection']['status'], 'unavailable')
                bubble.save_settings(paths['raw'], False, 2)
                report = regenerate_image_from_mask(path, paths, mask)
                self.assertEqual(report['bubble_detection']['status'], 'disabled')
                self.assertEqual(detect.call_count, 1)

    def test_detection_prefers_cuda_then_mps_and_cache_skips_inference(self):
        for available, mps_available, expected_device in (
                (True, True, 'cuda:0'), (True, False, 'cuda:0'),
                (False, True, 'mps'), (False, False, 'cpu')):
            with self.subTest(cuda=available, mps=mps_available), tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory)/'model.pt'
                checkpoint.write_bytes(b'model')
                cache = Path(directory)/'cache.npz'
                image = np.full((64, 64, 3), 255, np.uint8)
                polygon = np.array([[5, 5], [50, 5], [50, 50], [5, 50]], np.float32)
                import torch
                result = SimpleNamespace(masks=SimpleNamespace(xy=[polygon]),
                    boxes=SimpleNamespace(cls=torch.tensor([0])), names={0:'balloon'})
                with patch.object(bubble, 'MODEL_PATH', checkpoint), patch.object(bubble, '_model', None), \
                        patch.object(bubble, '_model_signature', None), patch('ultralytics.YOLO') as factory, \
                        patch('torch.cuda.is_available', return_value=available) as cuda_available, \
                        patch('torch.backends.mps.is_available', return_value=mps_available) as mps_probe:
                    predict = factory.return_value.predict
                    predict.return_value = [result]
                    polygons, status = bubble.detect_bubbles(image, cache)
                    self.assertEqual(status, 'detected')
                    np.testing.assert_array_equal(polygons[0], polygon)
                    self.assertEqual(predict.call_args.kwargs['device'], expected_device)
                    self.assertEqual(predict.call_args.kwargs['imgsz'], 1600)
                    self.assertTrue(predict.call_args.kwargs['retina_masks'])
                    if available:
                        mps_probe.assert_not_called()
                    else:
                        mps_probe.assert_called_once()
                    cuda_available.reset_mock()
                    mps_probe.reset_mock()
                    cuda_available.return_value = not available
                    cached, status = bubble.detect_bubbles(image, cache)
                    self.assertEqual(status, 'cache')
                    np.testing.assert_array_equal(cached[0], polygon)
                    cuda_available.assert_not_called()
                    mps_probe.assert_not_called()
                    predict.assert_called_once()
                    factory.assert_called_once()

    def test_polygon_cache_invalidates_on_image_and_model_change(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)/'model.pt'
            checkpoint.write_bytes(b'model')
            cache = Path(directory)/'cache.npz'
            image = np.full((64, 64, 3), 255, np.uint8)
            polygon = np.array([[5, 5], [50, 5], [50, 50], [5, 50]], np.float32)
            import torch
            result = SimpleNamespace(masks=SimpleNamespace(xy=[polygon]),
                boxes=SimpleNamespace(cls=torch.tensor([0])), names={0:'balloon'})
            with patch.object(bubble, 'MODEL_PATH', checkpoint), patch.object(bubble, '_model', None), \
                    patch.object(bubble, '_model_signature', None), patch('ultralytics.YOLO') as factory:
                factory.return_value.predict.return_value = [result]
                self.assertEqual(bubble.detect_bubbles(image, cache)[1], 'detected')
                self.assertEqual(bubble.detect_bubbles(image, cache)[1], 'cache')
                image[0, 0] = 0
                self.assertEqual(bubble.detect_bubbles(image, cache)[1], 'detected')
                checkpoint.write_bytes(b'different model')
                self.assertEqual(bubble.detect_bubbles(image, cache)[1], 'detected')
                self.assertEqual(factory.call_count, 2)
                result.masks = None
                image[0, 1] = 0
                self.assertEqual(bubble.detect_bubbles(image, cache)[0], [])

    def test_long_page_tiles_cover_both_orientations(self):
        for shape in ((1000, 100), (100, 1000)):
            covered = np.zeros(shape, bool)
            for x1, y1, x2, y2 in bubble._tiles(shape):
                covered[y1:y2, x1:x2] = True
                self.assertLessEqual(max(x2-x1, y2-y1), 200)
            self.assertTrue(covered.all())


if __name__ == '__main__':
    unittest.main()
