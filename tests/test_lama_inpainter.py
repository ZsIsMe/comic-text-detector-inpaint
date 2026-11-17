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
            model.to.assert_called_once_with(device='cpu', dtype=torch.float32)
            with patch('torch.load') as load:
                with self.assertRaisesRegex(ValueError, 'SHA-256'):
                    lama.load_generator(checkpoint)
                load.assert_not_called()

    def test_defaults_and_single_pixel_alpha_are_preserved_without_lowlevel_grow(self):
        from lama_config import MERGE_DISTANCE, CONTEXT_RATIO, MASK_EXPAND, REPAIR_POLICY
        self.assertEqual((lama.MAX_INPAINT_SIZE, MERGE_DISTANCE, CONTEXT_RATIO, MASK_EXPAND),
                         (768, 24, 1.7, 8))
        self.assertEqual(REPAIR_POLICY, 'manga-local-crop768-grow8-v3')
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
        self.assertFalse(np.any(repaired[mask == 0]))
        np.testing.assert_array_equal(repaired[mask > 0], [[26, 128, 230, 255]])
        rgb, small_mask = model.inputs[0]
        self.assertEqual(tuple(rgb.shape), (1, 3, 64, 64))
        self.assertEqual(rgb.dtype, torch.float32)
        self.assertEqual(small_mask.dtype, torch.float32)
        self.assertEqual(set(small_mask.unique().tolist()), {1})
        np.testing.assert_array_equal(rgb[0, :, 0, 0],
                                      (torch.tensor([20, 40, 80], dtype=torch.float32) / 255).numpy())
        np.testing.assert_array_equal(image, before)

    def test_region_geometry_dilates_25_square_then_uses_original_mask_bounds(self):
        mask = np.zeros((300, 400), bool)
        mask[80:130, 100:200] = True
        labels, regions = lama.region_crops(mask)
        self.assertEqual(regions, [(90, 70, 210, 140, 1)])
        self.assertEqual(int((labels > 0).sum()), (100 + 24) * (50 + 24))
        # A one-pixel crop stays one pixel: no minimum context is injected.
        tiny = np.zeros((10, 12), bool)
        tiny[0, 0] = True
        self.assertEqual(lama.region_crops(tiny)[1], [(0, 0, 1, 1, 1)])
        # The square dilation joins a 24-pixel gap, but not a 25-pixel gap.
        close = np.zeros((100, 120), bool)
        close[30, 30] = close[30, 55] = True
        self.assertEqual(lama.region_crops(close)[1], [(30, 30, 56, 31, 1)])
        far = np.zeros_like(close)
        far[30, 30] = far[30, 56] = True
        self.assertEqual(lama.region_crops(far)[1],
                         [(30, 30, 31, 31, 1), (56, 30, 57, 31, 2)])

    def test_overlapping_contexts_stay_separate_and_only_their_label_mask_is_written(self):
        image = np.full((700, 1100, 3), 100, np.uint8)
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[100:500, 100:500] = mask[100:500, 530:930] = 255
        before, mask_before = image.copy(), mask.copy()
        labels, regions = lama.region_crops(mask > 0)
        self.assertEqual(regions, [(39, 39, 561, 561, 1), (469, 39, 991, 561, 2)])
        calls = []

        def predict(context, local_mask):
            calls.append((context.copy(), local_mask.copy()))
            return np.full_like(context, 10 * len(calls))

        instance = lama.LamaLargeInpainter(device='cpu')
        with patch.object(instance, '_predict_crop', side_effect=predict):
            output = instance(image, mask)
        self.assertEqual(len(calls), 2)
        for (_, local_mask), (x1, y1, x2, y2, label) in zip(calls, regions):
            np.testing.assert_array_equal(local_mask,
                (mask[y1:y2, x1:x2] > 0) & (labels[y1:y2, x1:x2] == label))
            self.assertEqual(int(local_mask.sum()), 400 * 400)
        self.assertTrue(np.all(output[100:500, 100:500, :3] == 10))
        self.assertTrue(np.all(output[100:500, 530:930, :3] == 20))
        # The second overlapping context sees the first group's completed repair.
        np.testing.assert_array_equal(calls[1][0][150 - 39, 480 - 469], [10, 10, 10])
        np.testing.assert_array_equal(output[:, :, 3], mask)
        self.assertFalse(np.any(output[mask == 0]))
        np.testing.assert_array_equal(image, before)
        np.testing.assert_array_equal(mask, mask_before)

    def test_prepare_matches_uint8_linear_ceil64_then_square_reflect_reference(self):
        image = np.arange(65 * 139 * 3, dtype=np.uint8).reshape(65, 139, 3)
        mask = np.zeros((65, 139), bool)
        mask[10:14, 17:23] = True
        rgb, binary, small = lama._prepare(image, mask, 768)
        self.assertEqual(small, (128, 192))
        resized = cv2.resize(image, (192, 128), interpolation=cv2.INTER_LINEAR)
        resized_mask = cv2.resize(mask.astype(np.uint8) * 255, (192, 128),
                                  interpolation=cv2.INTER_LINEAR)
        np.testing.assert_array_equal(rgb, cv2.copyMakeBorder(resized, 0, 64, 0, 0, cv2.BORDER_REFLECT))
        np.testing.assert_array_equal(binary, cv2.copyMakeBorder(resized_mask, 0, 64, 0, 0,
                                                               cv2.BORDER_REFLECT) >= 128)
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertEqual(binary.dtype, np.bool_)

    def test_prepare_768_limit_and_reference_mask_threshold_without_area_coverage(self):
        image = np.zeros((1800, 2000, 3), np.uint8)
        image[:, ::2] = 255
        mask = np.zeros(image.shape[:2], bool)
        mask[0, 0] = True
        rgb, binary, small = lama._prepare(image, mask, 768)
        self.assertEqual(small, (704, 768))
        self.assertEqual(rgb.shape, (768, 768, 3))
        np.testing.assert_array_equal(rgb[:704], cv2.resize(image, (768, 704), interpolation=cv2.INTER_LINEAR))
        resized_mask = cv2.resize(mask.astype(np.uint8) * 255, (768, 704), interpolation=cv2.INTER_LINEAR)
        np.testing.assert_array_equal(binary[:704], resized_mask >= 128)
        self.assertFalse(np.any(binary))  # Deliberately follows reference threshold, not old thin-mask coverage.
        skinny = np.zeros((1, 3000, 3), np.uint8)
        rgb, _, small = lama._prepare(skinny, np.ones((1, 3000), bool), 768)
        self.assertEqual(small, (64, 768))  # max(1, round(...)) avoids a zero-sized axis.
        self.assertEqual(rgb.shape, (768, 768, 3))

    def test_prediction_rounds_uint8_before_unpadding_and_restoring_crop_size(self):
        class PatternModel(torch.nn.Module):
            def forward(self, image, mask):
                self.inputs = image.clone(), mask.clone()
                h, w = image.shape[-2:]
                grid = torch.arange(h * w, dtype=torch.float32).reshape(h, w) / (h * w - 1)
                self.prediction = torch.stack((grid, grid * .731 + .001, grid * .333)).unsqueeze(0)
                return self.prediction

        image = np.arange(73 * 139 * 3, dtype=np.uint8).reshape(73, 139, 3)
        mask = np.zeros((73, 139), bool)
        mask[12:50, 20:110] = True
        instance = lama.LamaLargeInpainter(device='cpu')
        instance.model, instance.device = PatternModel(), 'cpu'
        actual = instance._predict_crop(image, mask)
        input_rgb, input_mask = instance.model.inputs
        reference_wrapped = input_mask * instance.model.prediction + (1 - input_mask) * input_rgb
        predicted = reference_wrapped[0].permute(1, 2, 0).numpy()
        expected = np.clip(np.round(predicted * 255), 0, 255).astype(np.uint8)[:128, :192]
        expected = cv2.resize(expected, (139, 73), interpolation=cv2.INTER_LINEAR)
        np.testing.assert_array_equal(actual, expected)
        old_order = cv2.resize(predicted[:128, :192], (139, 73), interpolation=cv2.INTER_LINEAR)
        old_order = np.clip(np.round(old_order * 255), 0, 255).astype(np.uint8)
        self.assertTrue(np.any(actual != old_order))
        prepared = lama._prepare(image, mask, 768)[0]
        reference_tensor = torch.from_numpy(prepared).permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32) / 255
        torch.testing.assert_close(instance.model.inputs[0], reference_tensor, rtol=0, atol=0)
        self.assertEqual(instance.model.inputs[0].stride(), reference_tensor.stride())

    def test_network_resolution_masked_compositing_precedes_boundary_resampling(self):
        image = np.full((73, 139, 3), 100, np.uint8)
        mask = np.zeros((73, 139), bool)
        mask[20:50, 40:90] = True
        model = FlatModel()
        instance = lama.LamaLargeInpainter(device='cpu')
        instance.model, instance.device = model, 'cpu'
        actual = instance._predict_crop(image, mask)
        # Independent reference-node preparation and Spandrel wrapper formula.
        resized_image = cv2.resize(image, (192, 128), interpolation=cv2.INTER_LINEAR)
        resized_mask = cv2.resize(mask.astype(np.uint8) * 255, (192, 128),
                                  interpolation=cv2.INTER_LINEAR)
        padded_image = cv2.copyMakeBorder(resized_image, 0, 64, 0, 0, cv2.BORDER_REFLECT)
        padded_mask = cv2.copyMakeBorder(resized_mask, 0, 64, 0, 0, cv2.BORDER_REFLECT)
        rgb = torch.from_numpy(padded_image).permute(2, 0, 1).unsqueeze(0).float() / 255
        selected = torch.from_numpy(padded_mask >= 128).unsqueeze(0).unsqueeze(0).float()
        raw = torch.tensor([.1, .5, .9]).view(1, 3, 1, 1).expand_as(rgb)
        wrapped = selected * raw + (1 - selected) * rgb
        pixels = wrapped[0].permute(1, 2, 0).numpy()
        pixels = np.clip(np.round(pixels * 255), 0, 255).astype(np.uint8)[:128, :192]
        expected = cv2.resize(pixels, (139, 73), interpolation=cv2.INTER_LINEAR)
        np.testing.assert_array_equal(actual, expected)
        # Missing the wrapper incorrectly leaves the generator's mask-exterior
        # colors available to interpolation back into selected boundary pixels.
        self.assertTrue(np.any(actual[mask] != np.array([26, 128, 230], np.uint8)))
        np.testing.assert_array_equal(actual[0, 0], [100, 100, 100])

    def test_masks_and_images_are_validated_even_when_empty_without_constructing_model(self):
        image = np.zeros((9, 7, 3), np.uint8)
        cases = [np.zeros((8, 7)), np.full((9, 7), np.nan),
                 np.full((9, 7), -1), np.full((9, 7), np.inf),
                 np.zeros((9, 7), np.complex64), np.full((9, 7), 'bad')]
        with patch.object(lama, 'LamaLargeInpainter', side_effect=AssertionError('must stay lazy')):
            for mask in cases:
                with self.subTest(dtype=mask.dtype, shape=mask.shape):
                    with self.assertRaises(ValueError):
                        lama.inpaint_other(image, mask)
            with self.assertRaises(ValueError):
                lama.inpaint_other(image.astype(np.float32), np.zeros((9, 7)))

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
            if kwargs.get('device') == 'mps':
                kwargs = dict(kwargs, device='cpu')
            return original_to(tensor, *args, **kwargs)

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
