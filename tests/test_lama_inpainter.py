"""LaMa checkpoint and pixel contracts using a tiny deterministic model."""

import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import cv2
import torch

import lama_inpainter as lama


class FlatModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, image, mask):
        self.inputs.append((image.clone(), mask.clone()))
        return torch.tensor([.1, .5, .9], device=image.device).view(1, 3, 1, 1).expand_as(image)


class LamaInpainterTests(unittest.TestCase):
    def test_loader_verifies_hash_and_uses_weights_only_and_strict_state(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'weights.ckpt'
            checkpoint.write_bytes(b'checked model')
            state = {'weight': torch.ones(1)}
            model = MagicMock()
            model.eval.return_value = model
            model.requires_grad_.return_value = model
            model.to.return_value = model
            with patch.object(lama, 'MODEL_SHA256', hashlib.sha256(checkpoint.read_bytes()).hexdigest()), \
                    patch('torch.load', return_value={'gen_state_dict': state}) as load, \
                    patch('vendor.lama_large.LamaLargeGenerator', return_value=model):
                self.assertIs(lama.load_generator(checkpoint, device='cpu'), model)
            load.assert_called_once_with(checkpoint, map_location='cpu', weights_only=True)
            model.load_state_dict.assert_called_once_with(state, strict=True)
            model.requires_grad_.assert_called_once_with(False)
            with patch('torch.load') as load:
                with self.assertRaisesRegex(ValueError, 'SHA-256'):
                    lama.load_generator(checkpoint)
                load.assert_not_called()

    def test_preprocessing_preserves_thin_masks_and_output_has_exact_original_mask_alpha(self):
        image = np.full((101, 2051, 3), (20, 40, 80), np.uint8)
        before = image.copy()
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[49, 97] = 255
        model = FlatModel()
        inpainter = lama.LamaLargeInpainter(device='cpu')
        with patch.object(lama, 'load_generator', return_value=model) as load:
            repaired = inpainter(image, mask)
            second = inpainter(image, mask)
        self.assertEqual(load.call_count, 1)
        np.testing.assert_array_equal(repaired, second)
        self.assertTrue(np.all(repaired[mask == 0] == 0))
        np.testing.assert_array_equal(repaired[mask > 0], [[26, 128, 230, 255]])
        rgb, small_mask = model.inputs[0]
        self.assertEqual(rgb.shape[1], 3)
        self.assertEqual(rgb.shape[-2] % 64, 0)
        self.assertEqual(tuple(rgb.shape[-2:]), (64, 64))
        self.assertGreater(int(small_mask.sum()), 0)
        self.assertEqual(set(small_mask.unique().tolist()), {0, 1})
        np.testing.assert_allclose(rgb[0, :, 0, 0], np.array([20, 40, 80]) / 255, atol=1e-7)
        np.testing.assert_array_equal(image, before)

    def test_context_window_uses_upstream_area_ratio_and_clipped_minimum(self):
        self.assertEqual(lama._context_window((100, 80, 200, 130), 400, 300),
                         (90, 70, 210, 140))
        self.assertEqual(lama._context_window((0, 0, 1, 1), 500, 400), (0, 0, 32, 32))
        self.assertEqual(lama._context_window((499, 399, 500, 400), 500, 400),
                         (468, 368, 500, 400))
        self.assertEqual(lama._context_window((3, 2, 4, 3), 9, 7), (0, 0, 9, 7))

    def test_grouping_merges_16px_gaps_and_bridges_without_dropping_tiny_regions(self):
        boxes = [(10, 10, 11, 11), (27, 10, 28, 11), (44, 10, 45, 11),
                 (100, 100, 101, 101)]
        self.assertEqual(lama._merge_boxes(boxes, gap=16),
                         [(10, 10, 45, 11), (100, 100, 101, 101)])
        self.assertEqual(lama._merge_boxes([(0, 0, 1, 1), (18, 0, 19, 1)], gap=16),
                         [(0, 0, 1, 1), (18, 0, 19, 1)])
        mask = np.zeros((160, 200), np.uint8)
        mask[30, 30] = mask[30, 60] = mask[120, 120] = 255
        self.assertEqual(lama._repair_windows(mask),
                         [(15, 15, 77, 47), (105, 105, 137, 137)])

    def test_square_reflect_padding_preserves_unscaled_small_crop(self):
        image = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)
        mask = np.zeros((3, 5), np.uint8)
        mask[1, 2] = 255
        rgb, binary, small = lama._prepare(image, mask, 1536)
        self.assertEqual(small, (3, 5))
        self.assertEqual(rgb.shape, (64, 64, 3))
        np.testing.assert_array_equal(rgb[:3, :5], image.astype(np.float32) / 255)
        np.testing.assert_array_equal(rgb, cv2.copyMakeBorder(
            image.astype(np.float32) / 255, 0, 61, 0, 59, cv2.BORDER_REFLECT))
        np.testing.assert_array_equal(binary, cv2.copyMakeBorder(
            (mask > 0).astype(np.float32), 0, 61, 0, 59, cv2.BORDER_REFLECT))

    def test_large_crop_is_linear_downscaled_to_1536_and_keeps_one_pixel_masks(self):
        self.assertEqual(lama.MAX_INPAINT_SIZE, 1536)
        image = np.zeros((1800, 2000, 3), np.uint8)
        image[:, ::2] = 255
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[1, 1] = mask[1799, 1999] = 255
        rgb, binary, small = lama._prepare(image, mask, lama.MAX_INPAINT_SIZE)
        self.assertEqual(small, (1382, 1536))
        self.assertEqual(rgb.shape, (1536, 1536, 3))
        expected = cv2.resize(image.astype(np.float32) / 255, (1536, 1382),
                              interpolation=cv2.INTER_LINEAR)
        np.testing.assert_array_equal(rgb[:1382], expected)
        self.assertGreater(float(binary[0:3, 0:3].sum()), 0)
        self.assertGreater(float(binary[1379:1382, 1533:1536].sum()), 0)

    def test_multiple_local_predictions_map_back_to_original_coordinates(self):
        rows, columns = np.indices((160, 260))
        image = np.stack((rows, columns % 256, (rows + columns) % 256), axis=-1).astype(np.uint8)
        before = image.copy()
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[2, 5] = 255
        mask[100:105, 230:236] = 255
        mask_before = mask.copy()

        class EchoModel(FlatModel):
            def forward(self, image, mask):
                self.inputs.append((image.clone(), mask.clone()))
                return image

        model = EchoModel()
        with patch.object(lama, 'load_generator', return_value=model) as load:
            inpainter = lama.LamaLargeInpainter(device='cpu')
            output = inpainter(image, mask)
        self.assertEqual(load.call_count, 1)
        self.assertEqual(len(model.inputs), 2)
        np.testing.assert_array_equal(output[mask > 0, :3], image[mask > 0])
        self.assertTrue(np.all(output[mask > 0, 3] == 255))
        self.assertFalse(np.any(output[mask == 0]))
        for (x1, y1, x2, y2), (tensor, _) in zip(lama._repair_windows(mask), model.inputs):
            np.testing.assert_allclose(tensor[0, :, :y2-y1, :x2-x1].numpy().transpose(1, 2, 0),
                                       image[y1:y2, x1:x2].astype(np.float32) / 255)
        np.testing.assert_array_equal(image, before)
        np.testing.assert_array_equal(mask, mask_before)

    def test_overlapping_context_windows_are_inferred_once_and_cover_every_mask_pixel(self):
        image = np.zeros((160, 200, 3), np.uint8)
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[30, 30] = mask[30, 60] = 255
        model = FlatModel()
        with patch.object(lama, 'load_generator', return_value=model):
            output = lama.LamaLargeInpainter(device='cpu')(image, mask)
        self.assertEqual(len(model.inputs), 1)
        np.testing.assert_array_equal(output[:, :, 3], mask)

    def test_empty_mask_never_loads_or_constructs_a_model(self):
        image = np.zeros((9, 7, 3), np.uint8)
        empty = np.zeros((9, 7), np.uint8)
        with patch.object(lama, 'LamaLargeInpainter', side_effect=AssertionError('must stay lazy')):
            self.assertFalse(np.any(lama.inpaint_other(image, empty)))
        instance = lama.LamaLargeInpainter()
        with patch.object(lama, 'load_generator', side_effect=AssertionError('must stay lazy')):
            self.assertFalse(np.any(instance(image, empty)))

    def test_16bit_context_is_reduced_without_changing_source_or_output_geometry(self):
        image = np.full((1, 9, 4), (1024, 30000, 65535, 12345), np.uint16)
        before = image.copy()
        mask = np.ones((1, 9), np.uint8) * 255
        model = FlatModel()
        with patch.object(lama, 'load_generator', return_value=model):
            result = lama.LamaLargeInpainter(device='cpu')(image, mask)
        self.assertEqual(result.shape, (1, 9, 4))
        np.testing.assert_allclose(model.inputs[0][0][0, :, 0, 0],
                                   np.array([4, 117, 255]) / 255, atol=1e-7)
        np.testing.assert_array_equal(image, before)

    def test_mps_model_transfer_failure_retries_entire_loader_on_cpu(self):
        model = FlatModel()
        inpainter = lama.LamaLargeInpainter()
        image = np.zeros((16, 16, 3), np.uint8)
        mask = np.ones((16, 16), np.uint8)
        with patch.object(lama, '_device', return_value='mps'), \
                patch.object(lama, 'load_generator', side_effect=[RuntimeError('MPS unavailable'), model]) as load:
            result = inpainter(image, mask)
        self.assertEqual(inpainter.device, 'cpu')
        self.assertEqual([call.kwargs['device'] for call in load.call_args_list], ['mps', 'cpu'])
        self.assertTrue(np.all(result[:, :, 3] == 255))

    def test_unavailable_mps_fft_chooses_cpu(self):
        with patch('torch.cuda.is_available', return_value=False), \
                patch('torch.backends.mps.is_available', return_value=True), \
                patch('torch.arange', side_effect=NotImplementedError('no complex FFT')):
            self.assertEqual(lama._device(), 'cpu')

    def test_mps_prediction_fft_failure_retries_on_cpu(self):
        class FailOnceModel(FlatModel):
            def __init__(self):
                super().__init__()
                self.attempts = 0

            def forward(self, image, mask):
                self.attempts += 1
                if self.attempts == 1:
                    raise NotImplementedError('MPS complex FFT unavailable')
                return super().forward(image, mask)

        original_to = torch.Tensor.to

        def to(tensor, *args, **kwargs):
            return tensor if args and args[0] == 'mps' else original_to(tensor, *args, **kwargs)

        model = FailOnceModel()
        inpainter = lama.LamaLargeInpainter(device='cpu')
        inpainter.model, inpainter.device = model, 'mps'
        with patch.object(torch.Tensor, 'to', to):
            output = inpainter(np.zeros((20, 20, 3), np.uint8), np.ones((20, 20), np.uint8))
        self.assertEqual(inpainter.device, 'cpu')
        self.assertEqual(model.attempts, 2)
        self.assertTrue(np.all(output[:, :, 3] == 255))

    def test_import_does_not_load_torch_psd_tools_or_weights(self):
        code = ('import sys; import lama_inpainter; import psd_export; '
                'assert "torch" not in sys.modules; assert "psd_tools" not in sys.modules; '
                'assert lama_inpainter._model is None')
        result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
