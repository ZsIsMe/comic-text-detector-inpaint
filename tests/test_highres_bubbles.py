"""HighRes instance geometry and the explicit per-call text/bubble handoff."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

import bubble_solid as bubble
import detect_solid_inpaint_folder as pipeline
import project_store as store
from rfdetr_detector import (
    BubblePrediction, RfDetrSegDetector, bubble_polygons, model_signature, source_key,
)


class HighResBubbleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.weights = self.root / 'model.safetensors'
        self.weights.write_bytes(b'mocked weights')
        self.paths = pipeline._ensure_dirs(str(self.root))
        self.path = str(self.root / 'page.png')
        self.image = np.full((120, 160, 3), (170, 190, 210), np.uint8)
        cv2.imwrite(self.path, self.image)
        self.polygon = np.array(
            [[15, 15], [144, 15], [144, 90], [100, 90], [100, 105], [15, 105]], np.float32)
        self.bubble_mask = bubble._polygon_mask(self.polygon, self.image.shape[:2]) > 0
        self.addCleanup(patch.stopall)
        patch.object(bubble, 'MODEL_PATH', self.weights).start()
        patch.object(bubble, '_model', None).start()
        patch.object(bubble, '_model_signature', None).start()

    def detections(self, *, empty_text=False):
        masks = np.zeros((5, *self.image.shape[:2]), bool)
        if not empty_text:
            masks[0, 40:45, 50:55] = True
            masks[1, 65:70, 105:110] = True
        masks[2] = self.bubble_mask
        masks[3, 1:10, 1:10] = True  # Low-confidence second bubble.
        masks[4] = True  # Panel must never be used as a bubble.
        return SimpleNamespace(class_id=np.array([0, 1, 2, 2, 3]),
                               confidence=np.array([.90, .90, .50, .49, .90]), mask=masks)

    def detector(self, mode='text_onomatopoeia', *, empty_text=False, **kwargs):
        detector = RfDetrSegDetector(self.weights, device='cpu', mask_mode=mode, **kwargs)
        detector._model = MagicMock()
        detector._model.predict.return_value = self.detections(empty_text=empty_text)
        return detector

    def test_normal_rf_page_shares_one_prediction_independent_of_text_mask_mode(self):
        for mode in ('text_onomatopoeia', 'text', 'onomatopoeia', 'all'):
            with self.subTest(mode=mode):
                store.cache_path(self.paths, self.path).unlink(missing_ok=True)
                detector = self.detector(mode, inpaint_mask_dilate=11,
                                         mask_unification_method='rectangle')
                with patch.object(bubble, 'RfDetrSegDetector', side_effect=AssertionError('second model')):
                    report = pipeline.process_image_with_detector(self.path, self.paths, detector)
                detector._model.predict.assert_called_once()
                self.assertEqual(report['bubble_detection'], {'status': 'shared', 'instances': 1})
                cached = json.loads(str(store.read_cache_file(
                    store.cache_path(self.paths, self.path))['bubble_json']))
                actual = bubble._polygon_mask(np.asarray(cached['polygons'][0]), self.image.shape[:2]) > 0
                np.testing.assert_array_equal(actual, self.bubble_mask)
                self.assertFalse(actual[100, 130])  # Not its rectangular bounding box.

    def test_empty_text_can_direct_white_fill_bubble_from_same_prediction(self):
        detector = self.detector('text', empty_text=True)
        report = pipeline.process_image_with_detector(self.path, self.paths, detector)
        detector._model.predict.assert_called_once()
        self.assertEqual(report['bubble_detection']['status'], 'shared')
        self.assertEqual(report['solid_bubbles'], 1)
        state = pipeline.read_page_state(self.paths, self.path)
        np.testing.assert_array_equal(state['overlay'][50, 50], [255, 255, 255, 255])
        self.assertFalse(np.any(state['overlay'][100, 130]))

    def test_disabled_uses_original_three_value_api_without_bubble_cache_or_model(self):
        detector = self.detector()
        bubble.save_settings(self.paths['raw'], False, 2)
        with patch.object(detector, 'detect_with_bubbles', side_effect=AssertionError('disabled')) as shared, \
                patch.object(pipeline, 'detect_bubbles', side_effect=AssertionError('disabled')) as detect:
            report = pipeline.process_image_with_detector(self.path, self.paths, detector)
        shared.assert_not_called()
        detect.assert_not_called()
        detector._model.predict.assert_called_once()
        self.assertEqual(report['bubble_detection']['status'], 'disabled')

    def test_bubble_failure_preserves_valid_text_result(self):
        detector = self.detector()
        with patch('rfdetr_detector.bubble_polygons', side_effect=RuntimeError('bad geometry')):
            report = pipeline.process_image_with_detector(self.path, self.paths, detector)
        self.assertEqual(report['bubble_detection']['status'], 'unavailable')
        self.assertIn('bad geometry', report['bubble_detection']['error'])
        self.assertTrue(report['processed'])
        state = pipeline.read_page_state(self.paths, self.path)
        self.assertTrue(np.any(state['overlay'][:, :, 3]) or np.any(state['other']))

    def test_bubbles_keep_official_threshold_despite_text_overrides(self):
        detector = self.detector('text', class_thresholds={name: .95 for name in
                                 ('text', 'onomatopoeia', 'bubble', 'panel')})
        result, shared = detector.detect_with_bubbles(self.image)
        self.assertFalse(np.any(result[1]))
        self.assertEqual(len(shared.polygons), 1)
        self.assertEqual(detector._model.predict.call_args.kwargs['threshold'], .50)
        np.testing.assert_array_equal(
            bubble._polygon_mask(shared.polygons[0], self.image.shape[:2]) > 0, self.bubble_mask)

    def test_each_instance_uses_its_mask_and_missing_masks_never_fall_back_to_boxes(self):
        second = np.zeros_like(self.bubble_mask)
        second[108:117, 145:156] = True
        detections = SimpleNamespace(
            class_id=[2, 2, 2, 2, 3], confidence=[.5, .6, .49, .9, .9],
            mask=[self.bubble_mask, second, second, None, np.ones_like(second)])
        polygons = bubble_polygons(detections, self.image.shape[:2])
        self.assertEqual(len(polygons), 2)
        for polygon, expected in zip(polygons, (self.bubble_mask, second)):
            np.testing.assert_array_equal(
                bubble._polygon_mask(polygon, self.image.shape[:2]) > 0, expected)
        detections.mask = None
        self.assertEqual(bubble_polygons(detections, self.image.shape[:2]), [])

    def test_changed_source_pixels_force_fresh_inference_without_loading_second_model(self):
        detector = self.detector()
        _, shared = detector.detect_with_bubbles(self.image)
        changed = self.image.copy()
        changed[0, 0] = 0
        cache = store.cache_path(self.paths, self.path)
        with patch.object(bubble, 'RfDetrSegDetector', side_effect=AssertionError('second model')):
            _, status = bubble.detect_bubbles(changed, cache, shared_prediction=shared)
        self.assertEqual(status, 'detected')
        self.assertEqual(detector._model.predict.call_count, 2)
        np.testing.assert_array_equal(detector._model.predict.call_args.args[0], changed[:, :, ::-1])

    def test_pipeline_compares_actual_pixels_for_gray_bgra_and_16bit_sources(self):
        for kind, source, count in (
                ('gray', self.image[:, :, 0], 1),
                ('bgra', np.dstack((self.image, np.full(self.image.shape[:2], 85, np.uint8))), 1),
                ('16bit', self.image.astype(np.uint16) * 257, 2)):
            with self.subTest(kind=kind):
                path = str(self.root / f'{kind}.png')
                cv2.imwrite(path, source)
                detector = self.detector()
                with patch.object(bubble, 'RfDetrSegDetector', side_effect=AssertionError('second model')):
                    report = pipeline.process_image_with_detector(path, self.paths, detector)
                self.assertEqual(detector._model.predict.call_count, count)
                expected = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                np.testing.assert_array_equal(detector._model.predict.call_args.args[0], expected)
                self.assertEqual(report['bubble_detection']['status'], 'shared' if count == 1 else 'detected')

    def test_old_cache_and_model_or_inference_changes_invalidate_polygons(self):
        cache = store.cache_path(self.paths, self.path)
        store.update_cache_file(cache, bubble_json=np.array(json.dumps(
            {'key': [1, list(self.image.shape), 'old MangaLens'], 'polygons': []})))
        with patch.object(bubble, 'RfDetrSegDetector') as constructor:
            constructor.return_value.predict_bubbles.return_value = [self.polygon]
            self.assertEqual(bubble.detect_bubbles(self.image, cache)[1], 'detected')
            self.assertEqual(bubble.detect_bubbles(self.image, cache)[1], 'cache')
            for name, value in (('MODEL_SHA256', 'changed'), ('RESOLUTION', 2048), ('TILE_VERSION', 2)):
                with self.subTest(setting=name), patch.object(bubble, name, value):
                    self.assertEqual(bubble.detect_bubbles(self.image, cache)[1], 'detected')
            self.assertEqual(constructor.return_value.predict_bubbles.call_count, 4)

    def test_long_page_reuses_loaded_detector_rejects_seams_and_deduplicates(self):
        for horizontal in (False, True):
            with self.subTest(horizontal=horizontal):
                image = np.zeros((600, 100, 3), np.uint8)
                rect = lambda top, bottom: np.array([[20, top], [80, top],
                                                    [80, bottom], [20, bottom]], np.float32)
                results = [[rect(180, 199)], [rect(30, 80)], [rect(120, 170)], [rect(20, 70)]]
                expected = [rect(180, 230), rect(420, 470)]
                if horizontal:
                    image = image.transpose(1, 0, 2).copy()
                    results = [[p[:, ::-1] for p in group] for group in results]
                    expected = [p[:, ::-1] for p in expected]
                predictor = MagicMock()
                predictor.predict_bubbles.side_effect = results
                shared = BubblePrediction(source_key(image), model_signature(self.weights),
                                          [self.polygon], predictor=predictor)
                cache = self.root / f'long-{horizontal}.npz'
                with patch.object(bubble, 'RfDetrSegDetector', side_effect=AssertionError('second model')):
                    polygons, status = bubble.detect_bubbles(image, cache, shared_prediction=shared)
                self.assertEqual(status, 'detected')
                self.assertEqual(predictor.predict_bubbles.call_count, 4)
                self.assertEqual(len(polygons), 2)
                actual = sorted((cv2.boundingRect(p) for p in polygons))
                self.assertEqual(actual, sorted(cv2.boundingRect(p) for p in expected))

    def test_model_predict_is_serialized_for_shared_and_crop_inference(self):
        detector = self.detector()
        entered, release = threading.Event(), threading.Event()
        first = True

        def predict(*args, **kwargs):
            nonlocal first
            if first:
                first = False
                entered.set()
                if not release.wait(3):
                    raise RuntimeError('test did not release first prediction')
            return self.detections()

        detector._model.predict.side_effect = predict
        with ThreadPoolExecutor(max_workers=2) as executor:
            text = executor.submit(detector.detect_with_bubbles, self.image)
            try:
                self.assertTrue(entered.wait(2))
                crop = executor.submit(detector.predict_bubbles, self.image)
                with self.assertRaises(TimeoutError):
                    crop.result(timeout=.05)
            finally:
                release.set()
            self.assertEqual(len(text.result(timeout=2)[1].polygons), 1)
            self.assertEqual(len(crop.result(timeout=2)), 1)


if __name__ == '__main__':
    unittest.main()
