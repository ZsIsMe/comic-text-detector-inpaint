"""HighRes download, loader and mask contracts without loading real weights."""

from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import bootstrap
import detect_solid_inpaint_folder as pipeline
import load_model
import min_infer_rfdetr
import rfdetr_config as config
import lama_config
from rfdetr_detector import RfDetrSegDetector


class HighResTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.weights = Path(self.tmp.name) / 'model.safetensors'
        self.weights.write_bytes(b'test weights; model loading is mocked')

    def test_bootstrap_can_import_before_model_dependencies_are_installed(self):
        # -S excludes site-packages, as on a first launch before venv setup.
        result = subprocess.run(
            [sys.executable, '-S', '-c',
             'import bootstrap; assert "torch" not in __import__("sys").modules'],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_bootstrap_downloads_highres_and_factory_uses_the_same_weights(self):
        self.assertEqual(config.MODEL_REPO, 'mayocream/koharu-layout-highres')
        self.assertEqual(config.MODEL_FILE.parent.name, 'koharu-layout-highres')
        self.assertEqual(config.MODEL_SHA256,
                         '03de2f77281974f5084cdd8652ace2b55dc15ac0378aed66654e62537cdda255')
        with patch.object(bootstrap, 'ensure_venv', return_value=Path('/test/python')), \
                patch.object(bootstrap, 'ensure_dependencies'), \
                patch.object(bootstrap, 'download_model') as download, \
                patch.object(bootstrap, 'launch_app'):
            bootstrap.main()
        download.assert_any_call(config.MODEL_URL, config.MODEL_FILE, config.MODEL_SHA256)
        download.assert_any_call(lama_config.MODEL_URL, lama_config.MODEL_PATH, lama_config.MODEL_SHA256)
        self.assertEqual(download.call_count, 2)
        self.assertEqual(pipeline.detector_model_path('rfdetr'), config.MODEL_FILE)
        with patch.object(pipeline.osp, 'isfile', return_value=True), \
                patch('rfdetr_detector.RfDetrSegDetector') as constructor:
            pipeline.create_detector('rfdetr', {'device': 'cpu', 'mask_mode': 'text'})
        constructor.assert_called_once_with(str(config.MODEL_FILE), device='cpu', mask_mode='text')

    def test_loader_verifies_highres_and_constructs_1920_before_strict_loading(self):
        model = MagicMock()
        constructor = MagicMock(return_value=model)
        fake_rfdetr = types.ModuleType('rfdetr')
        fake_rfdetr.RFDETRSeg2XLarge = constructor
        state = {'mock_weight': object()}
        with patch.dict(sys.modules, {'rfdetr': fake_rfdetr}), \
                patch.object(load_model, 'verify_model_sha256', return_value=True) as verify, \
                patch.object(load_model, 'load_file', return_value=state) as load_file:
            self.assertIs(load_model.load_rfdetr_model(self.weights, device='mps'), model)
        verify.assert_called_once_with(self.weights)
        constructor.assert_called_once_with(
            pretrain_weights=None, resolution=1920, num_select=160, num_classes=4)
        load_file.assert_called_once_with(str(self.weights), device='cpu')
        model.model.model.load_state_dict.assert_called_once_with(state, strict=True)
        self.assertEqual(model.model.device, 'mps')
        self.assertEqual(load_model.MODEL_FILE, config.MODEL_FILE)
        self.assertEqual(load_model.verify_model_sha256.__defaults__, (config.MODEL_SHA256,))

    def test_loader_rejects_unverified_weights_before_constructing_model(self):
        constructor = MagicMock()
        fake_rfdetr = types.ModuleType('rfdetr')
        fake_rfdetr.RFDETRSeg2XLarge = constructor
        with patch.dict(sys.modules, {'rfdetr': fake_rfdetr}), \
                patch.object(load_model, 'load_file') as read_weights:
            with self.assertRaisesRegex(ValueError, 'SHA-256 不符'):
                load_model.load_rfdetr_model(self.weights)
        constructor.assert_not_called()
        read_weights.assert_not_called()

    def detections(self):
        # For each class, one instance just below and one at its threshold.
        masks = np.zeros((8, 7, 13), dtype=bool)
        for index in range(8):
            masks[index, 2, index] = True
        detections = MagicMock()
        detections.class_id = np.repeat(np.arange(4), 2)
        detections.confidence = np.array([.24, .25, .29, .30, .49, .50, .39, .40])
        detections.mask = masks
        detections.data = {}
        detections.__len__.return_value = len(masks)
        return detections

    def test_adapter_filters_each_class_and_returns_original_size_mask_unions(self):
        image = np.full((7, 13, 3), (10, 20, 30), dtype=np.uint8)
        expected_columns = {
            'text_onomatopoeia': (1, 3), 'text': (1,),
            'onomatopoeia': (3,), 'all': (1, 3, 5, 7),
        }
        for mode, columns in expected_columns.items():
            with self.subTest(mode=mode):
                detector = RfDetrSegDetector(
                    self.weights, device='cpu', inpaint_mask_dilate=0, mask_mode=mode)
                detector._model = MagicMock()
                detector._model.predict.return_value = self.detections()
                raw, refined, blocks = detector(image)
                expected = np.zeros(image.shape[:2], dtype=np.uint8)
                expected[2, columns] = 255
                np.testing.assert_array_equal(raw, expected)
                np.testing.assert_array_equal(refined, expected)
                self.assertEqual(blocks, [])
                args, kwargs = detector._model.predict.call_args
                np.testing.assert_array_equal(args[0], image[:, :, ::-1])
                self.assertEqual(kwargs, {
                    'threshold': .25, 'shape': (1920, 1920), 'include_source_image': False})

    def test_minimal_inference_uses_shared_size_and_class_thresholds(self):
        model = MagicMock()
        model.predict.return_value = self.detections()
        image = np.zeros((7, 13, 3), dtype=np.uint8)
        with patch.object(min_infer_rfdetr, '_save_mask') as save_mask:
            result = min_infer_rfdetr.run_device(model, image, 'cpu', Path(self.tmp.name))
        self.assertEqual(result['counts'], {name: 1 for name in config.CLASS_NAMES})
        self.assertEqual(save_mask.call_count, 4)
        self.assertEqual(model.predict.call_args.kwargs, {
            'threshold': .25, 'shape': (1920, 1920), 'include_source_image': False})


if __name__ == '__main__':
    unittest.main()
