"""Lazy, serialized lama_large_512px inference for PSD repair layers only."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import threading

import cv2
import numpy as np

from lama_config import MODEL_PATH, MODEL_SHA256, MAX_INPAINT_SIZE, PAD_MODULO


def _merge_boxes(boxes, gap=0):
    """Coalesce touching/near exclusive xyxy rectangles, including bridges."""
    merged = []
    for box in sorted(boxes):
        box = tuple(int(value) for value in box)
        pending = list(merged)
        merged = []
        while pending:
            other = pending.pop()
            if (box[0] <= other[2] + gap and other[0] <= box[2] + gap
                    and box[1] <= other[3] + gap and other[1] <= box[3] + gap):
                box = (min(box[0], other[0]), min(box[1], other[1]),
                       max(box[2], other[2]), max(box[3], other[3]))
                # The enlarged union may now bridge a rectangle already visited.
                pending.extend(merged)
                merged = []
            else:
                merged.append(other)
        merged.append(box)
    return sorted(merged, key=lambda box: (box[1], box[0], box[3], box[2]))


def _context_window(box, width, height, ratio=1.7, minimum=32):
    """BallonsTranslator enlarge_window (aspect_ratio=1), with tiny-mask context.

    Upstream utils/imgproc_utils.py solves a quadratic for added *area*, rather
    than multiplying each side by ratio. Our minimum 32px per axis adapts this
    to authoritative masks, which can contain individual pixels instead of text
    block rectangles. The minimum window shifts at image edges without padding.
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    delta = int(round((math.sqrt((w + h) ** 2 + 4 * (ratio - 1) * w * h)
                       - (w + h)) / 4))
    dx, dy = min(x1, width - x2, delta), min(y1, height - y2, delta)
    x1, y1, x2, y2 = x1 - dx, y1 - dy, x2 + dx, y2 + dy
    target_w, target_h = min(width, max(minimum, x2 - x1)), min(height, max(minimum, y2 - y1))
    x1 = min(max(0, x1 - (target_w - (x2 - x1)) // 2), width - target_w)
    y1 = min(max(0, y1 - (target_h - (y2 - y1)) // 2), height - target_h)
    return x1, y1, x1 + target_w, y1 + target_h


def _repair_windows(mask):
    """Use only saved OTHER pixels, never disposable detection caches."""
    active = np.ascontiguousarray(mask > 0, dtype=np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(active, connectivity=8)
    boxes = [(x, y, x + w, y + h) for x, y, w, h, _ in stats[1:count]]
    grouped = _merge_boxes(boxes, gap=16)
    height, width = mask.shape
    return _merge_boxes([_context_window(box, width, height) for box in grouped])


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
        rgb = cv2.resize(rgb, target, interpolation=cv2.INTER_LINEAR)
        # Resize float coverage, not uint8/nearest: even a one-pixel mask must survive.
        binary = (cv2.resize(binary, target, interpolation=cv2.INTER_AREA) > 0).astype(np.float32)
    small_h, small_w = binary.shape
    side = math.ceil(max(small_h, small_w) / PAD_MODULO) * PAD_MODULO
    bottom, right = side - small_h, side - small_w
    rgb = cv2.copyMakeBorder(rgb, 0, bottom, 0, right, cv2.BORDER_REFLECT)
    binary = cv2.copyMakeBorder(binary, 0, bottom, 0, right, cv2.BORDER_REFLECT)
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
        context = image[:, :, :3].copy()
        if context.dtype == np.uint16:
            context = (context >> 8).astype(np.uint8)
        with self._lock:
            remaining = active.copy()
            for x1, y1, x2, y2 in _repair_windows(mask):
                local_mask = remaining[y1:y2, x1:x2]
                if not np.any(local_mask):
                    continue
                repaired = self._predict_crop(context[y1:y2, x1:x2], local_mask)
                context[y1:y2, x1:x2][local_mask] = repaired[local_mask]
                local_output = output[y1:y2, x1:x2]
                local_output[local_mask, :3] = repaired[local_mask]
                local_output[local_mask, 3] = 255
                local_mask[:] = False
        if np.any(remaining):
            raise ValueError('LaMa 局部修補未覆蓋全部 OTHER 選區。')
        return output

    def _predict_crop(self, context, mask):
        """Predict one local window; caller holds the model lock for the page."""
        import torch

        rgb, binary, small_shape = _prepare(context, mask, self.max_size)
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
        return repaired


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
