from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

import bubble_solid as bubble
from detect_solid_inpaint_folder import (
    _solid_overlay_from_mask, _quality_from_sample, _ensure_dirs,
    regenerate_image_from_mask, _output_path, _manual_other_path,
)


def scene(color=(255, 255, 255)):
    image = np.full((180, 200, 3), color, np.uint8)
    polygon = np.array([[20, 20], [179, 20], [179, 159], [20, 159]], np.float32)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[70:105, 80:95] = 255
    image[mask > 0] = 0
    return image, mask, [polygon]


class SolidBubbleTests(unittest.TestCase):
    def test_missed_dot_is_filled_and_inset_preserved(self):
        image, mask, polys = scene()
        image[80:84, 111:115] = 0
        old, _, _, _ = _solid_overlay_from_mask(image, mask)
        new, other, _, report = _solid_overlay_from_mask(image, mask, polys)
        self.assertEqual(np.count_nonzero(old[80:84, 111:115, 3]), 0)
        self.assertTrue(np.all(new[80:84, 111:115] == 255))
        self.assertFalse(np.any(new[20:23, :, 3]))
        self.assertFalse(np.any(new[:20, :, 3]))
        self.assertEqual(report['solid_bubbles'], 1)
        self.assertFalse(np.any(other))
        self.assertEqual(image[81, 112, 0], 0)

    def test_gray_and_colored_compression_noise_pass(self):
        rng = np.random.default_rng(42)
        for color in ((200, 200, 200), (180, 205, 230), (35, 40, 55)):
            with self.subTest(color=color):
                image, mask, polys = scene(color)
                image = np.clip(image.astype(int) + rng.integers(-2, 3, image.shape), 0, 255).astype(np.uint8)
                output, _, _, records = bubble.fill_bubbles(image, mask, polys)
                self.assertTrue(records[0]['accepted'], records)
                np.testing.assert_allclose(output[80, 85, :3], color, atol=1)
                sample = np.zeros(mask.shape, np.uint8)
                sample[30:60, 30:160] = 255
                self.assertTrue(_quality_from_sample(image, sample, 'full').is_solid)

    def test_many_antialiased_fragments_are_not_mistaken_for_texture(self):
        image, mask, polys = scene()
        for y in range(68, 108, 4):
            image[y:y+2, 109:112] = 240
            image[y, 110] = 0
        output, _, _, records = bubble.fill_bubbles(image, mask, polys)
        self.assertTrue(records[0]['accepted'], records)
        self.assertEqual(output[68, 110, 3], 255)

    def test_gray_fringe_next_to_mask_does_not_require_unmasked_dark_core(self):
        image, mask, polys = scene()
        image[85:95, 97] = 241
        output, _, _, records = bubble.fill_bubbles(image, mask, polys)
        self.assertTrue(records[0]['accepted'], records)
        self.assertTrue(np.all(output[85:95, 97] == 255))

    def test_sparse_flat_external_sample_is_not_an_unknown_gradient(self):
        image = np.full((30, 30, 3), 200, np.uint8)
        sample = np.zeros((30, 30), np.uint8)
        sample[10, 5:17] = 255
        quality = _quality_from_sample(image, sample, 'full')
        self.assertEqual(quality.sampled_cells, 0)
        self.assertTrue(quality.is_solid)

    def test_small_outline_overshoot_is_preserved_without_vetoing_bubble(self):
        image, mask, polys = scene()
        image[45:80, 23:25] = 0
        output, _, _, records = bubble.fill_bubbles(image, mask, polys)
        self.assertTrue(records[0]['accepted'], records)
        self.assertGreater(records[0]['protected_outline_pixels'], 0)
        self.assertFalse(np.any(output[45:80, 23:25, 3]))
        self.assertTrue(np.all(output[mask > 0, 3] == 255))

    def test_small_bubble_perimeter_does_not_consume_residual_budget(self):
        image = np.full((80, 80, 3), 255, np.uint8)
        polygon = np.array([[10, 10], [69, 10], [69, 69], [10, 69]], np.float32)
        mask = np.zeros((80, 80), np.uint8)
        mask[30:45, 35:45] = 255
        image[mask > 0] = 0
        image[12:65, 11:13] = 0
        output, _, _, records = bubble.fill_bubbles(image, mask, [polygon])
        self.assertTrue(records[0]['accepted'], records)
        self.assertFalse(np.any(output[12:65, 11:13, 3]))
        self.assertTrue(np.all(output[mask > 0, 3] == 255))

    def test_near_edge_text_recovers_inset_without_erasing_frame(self):
        image, mask, polys = scene()
        image[:] = 255
        mask[:] = 0
        image[20:160, 20:22] = 0
        mask[70:100, 31:47] = 255
        image[mask > 0] = 0
        output, _, _, records = bubble.fill_bubbles(image, mask, polys, shrink_ratio=0.1)
        self.assertTrue(records[0]['accepted'], records)
        self.assertGreater(records[0]['recovered_rim_pixels'], 0)
        self.assertTrue(np.all(output[mask > 0, 3] == 255))
        self.assertFalse(np.any(output[20:160, 20:22, 3]))

    def test_rim_recovery_does_not_erase_strokes_connected_to_frame(self):
        image, mask, polys = scene()
        image[:] = 255
        mask[:] = 0
        image[20:160, 20:22] = 0
        mask[70:100, 31:47] = 255
        image[mask > 0] = 0
        image[80, 20:40] = 0
        output, _, _, records = bubble.fill_bubbles(image, mask, polys, shrink_ratio=0.1)
        self.assertTrue(records[0]['accepted'], records)
        self.assertFalse(np.any(output[80, 20:34, 3]))

    def test_tiny_boundary_island_is_ignored_without_mutating_mask_or_art(self):
        image, mask, polys = scene()
        mask[60:120, 70:130] = 255
        mask[18:22, 90:94] = 255
        image[mask > 0] = 0
        before = mask.copy()
        output, other, _, report = _solid_overlay_from_mask(image, mask, polys)
        self.assertEqual(report['solid_bubbles'], 1)
        self.assertEqual(report['bubbles_debug'][0]['ignored_boundary_text_pixels'], 16)
        self.assertFalse(np.any(output[18:22, 90:94, 3]))
        self.assertFalse(np.any(other))
        np.testing.assert_array_equal(mask, before)
        protected = np.zeros_like(mask)
        protected[18:22, 90:94] = 255
        output, other, _, _ = _solid_overlay_from_mask(image, mask, polys, protected=protected)
        self.assertTrue(np.all(other[protected > 0] == 255))
        self.assertFalse(np.any(output[protected > 0, 3]))

    def test_many_boundary_islands_are_not_ignored_as_one_tiny_error(self):
        image, mask, polys = scene()
        mask[60:120, 70:130] = 255
        for x in range(40, 150, 10):
            mask[18:22, x:x+4] = 255
        image[mask > 0] = 0
        _, _, _, records = bubble.fill_bubbles(image, mask, polys)
        self.assertEqual(records[0]['reason'], 'text_crosses_boundary')

    def test_tiny_other_beside_fill_is_preserved_without_warning_unless_manual(self):
        image, mask, polys = scene()
        overlay = np.full((*mask.shape, 4), 255, np.uint8)
        overlay[80, 79] = 0
        veto = np.full(mask.shape, 255, np.uint8)
        sample = np.zeros_like(mask)
        result = (overlay, veto, sample, [{'accepted': True}])
        with patch('detect_solid_inpaint_folder.fill_bubbles', return_value=result):
            output, other, _, report = _solid_overlay_from_mask(image, mask, polys)
            self.assertEqual(other[80, 79], 0)
            self.assertEqual(output[80, 79, 3], 0)
            self.assertEqual(report['ignored_boundary_pixels'], 1)
            protected = np.zeros_like(mask)
            protected[80, 79] = 255
            _, other, _, _ = _solid_overlay_from_mask(image, mask, polys, protected=protected)
            self.assertEqual(other[80, 79], 255)
        overlay[80:90, 79] = 0
        with patch('detect_solid_inpaint_folder.fill_bubbles', return_value=result):
            _, other, _, _ = _solid_overlay_from_mask(image, mask, polys)
            self.assertTrue(np.all(other[80:90, 79] == 255))

    def test_uncertain_artwork_keeps_safe_local_fill_without_erasing_art(self):
        image, mask, polys = scene()
        image[40:44, 145:149] = 0
        output, _, _, report = _solid_overlay_from_mask(image, mask, polys)
        self.assertEqual(report['solid_bubbles'], 0)
        self.assertTrue(report['bubbles_debug'][0]['local_fallback'])
        self.assertTrue(np.all(output[mask > 0, 3] == 255))
        self.assertFalse(np.any(output[38:46, 143:151, 3]))

    def test_gradient_rejects_and_cannot_fall_back_to_flat_patch(self):
        image, mask, polys = scene()
        image[:] = np.linspace(220, 250, 200).astype(np.uint8)[None, :, None]
        image[mask > 0] = 0
        output, other, _, report = _solid_overlay_from_mask(image, mask, polys)
        self.assertEqual(report['solid_bubbles'], 0)
        self.assertFalse(np.any(output[mask > 0, 3]))
        self.assertTrue(np.all(other[mask > 0] == 255))

    def test_artwork_far_from_text_vetoes_fill(self):
        image, mask, polys = scene()
        image[40:44, 145:149] = 0
        output, _, _, records = bubble.fill_bubbles(image, mask, polys)
        self.assertFalse(np.any(output))
        self.assertEqual(records[0]['reason'], 'unexplained_marks')

    def test_long_emphasis_line_vetoes_fill(self):
        image, mask, polys = scene()
        image[55:110, 110] = 0
        _, _, _, records = bubble.fill_bubbles(image, mask, polys)
        self.assertFalse(records[0]['accepted'])

    def test_sparse_light_compression_speckles_are_tolerated(self):
        image, mask, polys = scene()
        for x in range(35, 170, 12):
            image[35, x] = 233
        output, _, _, records = bubble.fill_bubbles(image, mask, polys)
        self.assertTrue(records[0]['accepted'], records)
        self.assertTrue(np.all(output[35, 35, :3] == 255))

    def test_halftone_pattern_is_not_treated_as_compression_noise(self):
        image, mask, polys = scene()
        image[30:150:5, 30:170:5] = 230
        _, _, _, records = bubble.fill_bubbles(image, mask, polys)
        self.assertFalse(records[0]['accepted'])

    def test_insufficient_background_is_rejected(self):
        image, mask, polys = scene()
        mask[23:157, 23:177] = 255
        image[mask > 0] = 0
        _, _, _, records = bubble.fill_bubbles(image, mask, polys)
        self.assertEqual(records[0]['reason'], 'insufficient_background')

    def test_overlap_without_image_boundary_allows_only_local_fill(self):
        image, mask, polys = scene()
        polys.append(polys[0] + [10, 0])
        output, _, _, records = bubble.fill_bubbles(image, mask, polys)
        self.assertFalse(np.any(output))
        self.assertTrue(all(r['reason'] == 'overlap_text_only' for r in records))
        output, _, _, report = _solid_overlay_from_mask(image, mask, polys)
        self.assertTrue(np.all(output[mask > 0, 3] == 255))
        self.assertEqual(report['solid_bubbles'], 0)

    def test_small_overlap_keeps_each_color_and_protects_divider(self):
        image = np.full((180, 200, 3), 200, np.uint8)
        image[:, 101:] = 245
        image[20:160, 100] = 0
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[70:105, 50:65] = 255
        mask[70:105, 140:155] = 255
        image[mask > 0] = 0
        polygons = [np.array([[20,20],[100,20],[100,159],[20,159]], np.float32),
                    np.array([[100,20],[179,20],[179,159],[100,159]], np.float32)]
        output, _, _, records = bubble.fill_bubbles(image, mask, polygons)
        self.assertEqual(sum(r['accepted'] for r in records), 2)
        self.assertTrue(all(r['overlap_resolution'] == 'edge_trim' for r in records))
        self.assertTrue(np.all(output[70:105, 50:65, 0] == 200))
        self.assertTrue(np.all(output[70:105, 140:155, 0] == 245))
        self.assertFalse(np.any(output[:, 98:103, 3]))

    def test_large_overlap_splits_at_actual_frame_and_validates_both_sides(self):
        image, mask, polygons = scene((200,200,200))
        image[:] = 200
        image[:, 101:] = 245
        image[20:160, 99:102] = 0
        mask[:] = 0
        mask[70:105, 50:65] = 255
        mask[70:105, 140:155] = 255
        image[mask > 0] = 0
        polygons.append(np.array([[95,20],[179,20],[179,159],[95,159]], np.float32))
        output, _, _, records = bubble.fill_bubbles(image, mask, polygons)
        self.assertEqual(sum(r['accepted'] for r in records), 2, records)
        self.assertTrue(all(r['overlap_resolution'] == 'image_boundary_split' for r in records))
        self.assertEqual(output[80, 55, 0], 200)
        self.assertEqual(output[80, 145, 0], 245)
        self.assertFalse(np.any(output[20:160, 99:102, 3]))

    def test_partition_restores_enclosed_art_before_color_validation(self):
        image, mask, polygons = scene()
        image[:] = 255
        image[20:160, 99:102] = 0
        mask[:] = 0
        mask[100:120, 50:65] = 255
        mask[100:120, 140:155] = 255
        image[mask > 0] = 0
        # Closed drawing must not turn into a hole omitted from the sample.
        cv2.circle(image, (60, 50), 15, (0,0,0), -1)
        polygons.append(np.array([[95,20],[179,20],[179,159],[95,159]], np.float32))
        output, _, _, records = bubble.fill_bubbles(image, mask, polygons)
        self.assertEqual(sum(r['accepted'] for r in records), 1, records)
        self.assertFalse(np.any(output[35:65, 45:75, 3]))
        self.assertEqual(output[110, 145, 3], 255)

    def test_ambiguous_overlap_cannot_bypass_gradient_rejection(self):
        image, mask, polygons = scene()
        image[:] = np.linspace(220, 250, 200).astype(np.uint8)[None, :, None]
        image[mask > 0] = 0
        polygons.append(polygons[0] + [10, 0])
        output, other, _, report = _solid_overlay_from_mask(image, mask, polygons)
        self.assertEqual(report['solid_bubbles'], 0)
        self.assertFalse(np.any(output[mask > 0, 3]))
        self.assertTrue(np.all(other[mask > 0] == 255))

    def test_irregular_regions_do_not_claim_neighbor_text_in_their_bounds(self):
        image, mask, _ = scene()
        mask[:] = 0
        mask[35:50, 35:50] = 255
        mask[90:105, 120:135] = 255
        image[:] = 255
        image[mask > 0] = 0
        # Concave L and a box occupy disjoint pixels but share bounding boxes.
        polygons = [np.array([[20,20],[175,20],[175,65],[70,65],[70,155],[20,155]],np.float32),
                    np.array([[100,80],[165,80],[165,135],[100,135]],np.float32)]
        output, _, _, records = bubble.fill_bubbles(image, mask, polygons)
        self.assertEqual(sum(r['accepted'] for r in records), 2, records)
        self.assertTrue(np.all(output[mask > 0, 3] == 255))

    def test_separate_bubbles_keep_their_own_color(self):
        image, mask, polys = scene((200, 200, 200))
        joined = np.concatenate((image, np.full_like(image, 245)), axis=1)
        masks = np.concatenate((mask, mask), axis=1)
        joined[:, 200:][mask > 0] = 0
        output, _, _, records = bubble.fill_bubbles(joined, masks, polys + [polys[0]+[200, 0]])
        self.assertEqual(sum(r['accepted'] for r in records), 2)
        self.assertEqual(output[80, 85, 0], 200)
        self.assertEqual(output[80, 285, 0], 245)

    def test_manual_other_protects_entire_bubble(self):
        image, mask, polys = scene()
        protected = np.zeros_like(mask)
        protected[80:84, 111:115] = 255
        output, _, _, records = bubble.fill_bubbles(image, mask, polys, protected=protected)
        self.assertFalse(np.any(output))
        self.assertEqual(records[0]['reason'], 'manual_other')

    def test_no_text_never_triggers_whole_bubble_fill(self):
        image, mask, polys = scene()
        output, _, _, records = bubble.fill_bubbles(image, mask*0, polys)
        self.assertFalse(np.any(output))
        self.assertEqual(records, [])

    def test_crossing_text_preserves_bubble_border(self):
        image, mask, polys = scene()
        mask[60:70, 10:30] = 255
        image[mask > 0] = 0
        output, _, _, report = _solid_overlay_from_mask(image, mask, polys)
        self.assertEqual(report['solid_bubbles'], 0)
        self.assertFalse(np.any(output[60:70, 17:25, 3]))


class BubbleIntegrationTests(unittest.TestCase):
    def test_settings_persist_and_invalid_values_are_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(bubble.load_settings(directory), {'enabled': True, 'shrink_percent': 2.0})
            bubble.save_settings(directory, False, 3.5)
            self.assertEqual(bubble.load_settings(directory), {'enabled': False, 'shrink_percent': 3.5})
            (Path(directory)/bubble.SETTINGS_FILE).write_text('{"enabled":"false","shrink_percent":"NaN"}')
            self.assertEqual(bubble.load_settings(directory), {'enabled': True, 'shrink_percent': 2.0})

    def test_regeneration_keeps_text_mask_and_protects_manual_other(self):
        with tempfile.TemporaryDirectory() as directory:
            image, mask, polys = scene()
            image[80:84, 111:115] = 0
            path = str(Path(directory)/'page.png')
            cv2.imwrite(path, image)
            paths = _ensure_dirs(directory)
            with patch('detect_solid_inpaint_folder.detect_bubbles', return_value=(polys, 'detected')):
                report = regenerate_image_from_mask(path, paths, mask)
                self.assertEqual(report['solid_bubbles'], 1)
                np.testing.assert_array_equal(cv2.imread(str(Path(paths['mask'])/'page.png'), 0), mask)
                protected = np.zeros_like(mask)
                protected[80:84, 111:115] = 255
                cv2.imwrite(_manual_other_path(paths, path), protected)
                report = regenerate_image_from_mask(path, paths, mask)
                self.assertEqual(report['solid_bubbles'], 0)
                result = cv2.imread(_output_path(paths, path), -1)
                self.assertFalse(np.any(result[80:84, 111:115, 3]))

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

    def test_polygon_cache_invalidates_on_image_and_model_change(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)/'model.pt'
            checkpoint.write_bytes(b'model')
            cache = Path(directory)/'cache.json'
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
