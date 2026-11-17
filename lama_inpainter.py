"""Lazy, serialized lama_large_512px inference for PSD repair layers only."""

from __future__ import annotations

import hashlib
from pathlib import Path
import threading

import cv2
import numpy as np

from lama_config import MODEL_PATH, MODEL_SHA256, MAX_INPAINT_SIZE, PAD_MODULO


def _device(requested='auto'):
    import torch

    if requested not in ('auto', 'cpu', 'cuda', 'mps'):
        raise ValueError(f'不支援的 LaMa 裝置：{requested}')
    if requested == 'cuda' or (requested == 'auto' and torch.cuda.is_available()):
        return 'cuda'
    if requested == 'mps' or (requested == 'auto' and torch.backends.mps.is_available()):
        try:
            sample = torch.arange(64, dtype=torch.float32, device='mps').reshape(1, 1, 8, 8) / 64
            restored = torch.fft.irfftn(torch.fft.rfftn(sample, dim=(-2, -1), norm='ortho'),
                                       s=(8, 8), dim=(-2, -1), norm='ortho')
            if torch.allclose(restored.cpu(), sample.cpu(), atol=1e-5, rtol=1e-5):
                return 'mps'
        except (RuntimeError, NotImplementedError):
            pass
    return 'cpu'


def load_generator(path=MODEL_PATH, *, device='cpu'):
    import torch
    from vendor.lama_large import LamaLargeGenerator

    path = Path(path)
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    if hasher.hexdigest() != MODEL_SHA256:
        raise ValueError(f'LaMa 模型 SHA-256 不符：{path}')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('gen_state_dict'), dict):
        raise ValueError('LaMa 模型缺少 gen_state_dict。')
    model = LamaLargeGenerator()
    model.load_state_dict(checkpoint['gen_state_dict'], strict=True)
    return model.eval().requires_grad_(False).to(device)


def _prepare(image, mask, max_size):
    height, width = image.shape[:2]
    scale = min(1.0, max_size / max(height, width))
    target = max(1, round(width * scale)), max(1, round(height * scale))
    rgb = image.astype(np.float32) / 255.0
    binary = (mask > 0).astype(np.float32)
    if target != (width, height):
        rgb = cv2.resize(rgb, target, interpolation=cv2.INTER_AREA)
        # Resize float coverage, not uint8/nearest: even a one-pixel mask must survive.
        binary = (cv2.resize(binary, target, interpolation=cv2.INTER_AREA) > 0).astype(np.float32)
    small_h, small_w = binary.shape
    bottom, right = (-small_h) % PAD_MODULO, (-small_w) % PAD_MODULO
    rgb = cv2.copyMakeBorder(rgb, 0, bottom, 0, right, cv2.BORDER_REFLECT_101)
    binary = cv2.copyMakeBorder(binary, 0, bottom, 0, right, cv2.BORDER_REFLECT_101)
    return rgb, binary, (small_h, small_w)


class LamaLargeInpainter:
    def __init__(self, model_path=MODEL_PATH, *, device='auto', max_size=MAX_INPAINT_SIZE):
        if type(max_size) is not int or max_size < 1:
            raise ValueError('LaMa 尺寸必須為正整數。')
        self.model_path = Path(model_path)
        self.requested_device = device
        self.max_size = max_size
        self.device = None
        self.model = None
        self._lock = threading.Lock()

    def __call__(self, image_rgb, other_mask):
        image = np.asarray(image_rgb)
        mask = np.asarray(other_mask)
        if image.ndim != 3 or image.shape[2] not in (3, 4) or image.dtype not in (np.uint8, np.uint16):
            raise ValueError('LaMa 需要 RGB/RGBA uint8 或 uint16 圖像。')
        if mask.shape != image.shape[:2]:
            raise ValueError('LaMa Mask 尺寸與原圖不一致。')
        output = np.zeros((*mask.shape, 4), np.uint8)
        active = mask > 0
        if not np.any(active):
            return output
        context = image[:, :, :3]
        if context.dtype == np.uint16:
            context = (context >> 8).astype(np.uint8)
        rgb, binary, small_shape = _prepare(context, mask, self.max_size)
        import torch

        with self._lock:
            if self.model is None:
                self.device = _device(self.requested_device)
                try:
                    self.model = load_generator(self.model_path, device=self.device)
                except (RuntimeError, NotImplementedError):
                    if self.device != 'mps':
                        raise
                    self.device = 'cpu'
                    self.model = load_generator(self.model_path, device='cpu')
            image_tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0)
            mask_tensor = torch.from_numpy(binary).unsqueeze(0).unsqueeze(0)
            with torch.inference_mode():
                try:
                    predicted = self.model(image_tensor.to(self.device), mask_tensor.to(self.device))
                except (RuntimeError, NotImplementedError):
                    if self.device != 'mps':
                        raise
                    # Retry the entire model on CPU; never discard FFT imaginary components.
                    self.model = self.model.to('cpu')
                    self.device = 'cpu'
                    predicted = self.model(image_tensor, mask_tensor)
            prediction = predicted.detach().float().cpu().numpy()
        if prediction.shape != (1, 3, *binary.shape) or not np.isfinite(prediction).all():
            raise ValueError('LaMa 回傳無效的修補圖像。')
        small_h, small_w = small_shape
        repaired = prediction[0, :, :small_h, :small_w].transpose(1, 2, 0)
        repaired = cv2.resize(repaired, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        repaired = np.rint(np.clip(repaired, 0, 1) * 255).astype(np.uint8)
        output[active, :3] = repaired[active]
        output[active, 3] = 255
        return output


_model = None
_model_lock = threading.Lock()


def inpaint_other(image_rgb, other_mask):
    """Return an RGBA repair layer; an empty mask does not initialize the model."""
    if not np.any(other_mask):
        return np.zeros((*np.asarray(other_mask).shape, 4), np.uint8)
    global _model
    with _model_lock:
        if _model is None:
            _model = LamaLargeInpainter()
        return _model(image_rgb, other_mask)
