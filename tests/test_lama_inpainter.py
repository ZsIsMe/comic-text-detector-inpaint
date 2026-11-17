"""LaMa checkpoint and pixel contracts using a tiny deterministic model."""

import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
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
        self.assertEqual(rgb.shape[-1], 512)
        self.assertGreater(int(small_mask.sum()), 0)
        self.assertEqual(set(small_mask.unique().tolist()), {0, 1})
        np.testing.assert_allclose(rgb[0, :, 0, 0], np.array([20, 40, 80]) / 255, atol=1e-7)
        np.testing.assert_array_equal(image, before)

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

    def test_import_does_not_load_torch_psd_tools_or_weights(self):
        code = ('import sys; import lama_inpainter; import psd_export; '
                'assert "torch" not in sys.modules; assert "psd_tools" not in sys.modules; '
                'assert lama_inpainter._model is None')
        result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
