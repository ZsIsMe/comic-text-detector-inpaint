"""Lazy, serialized lama_large_512px inference for PSD repair layers only."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import threading

import cv2
import numpy as np

from lama_config import (
    MODEL_PATH, MODEL_SHA256, MAX_INPAINT_SIZE, PAD_MODULO, MERGE_DISTANCE, CONTEXT_RATIO,
)


def region_crops(mask, merge_distance=MERGE_DISTANCE, context_ratio=CONTEXT_RATIO):
    """LaMa-Manga-Local-Crop grouping: dilated labels, original-mask bounds.

    Mirrors ComfyUI-MangaLaMa region_crops. Contexts may overlap; their selected
    label masks stay separate. No bbox-distance merge or minimum window is used.
    """
    grouping = mask.astype(np.uint8)
    if merge_distance > 0:
        grouping = cv2.dilate(grouping, np.ones((merge_distance + 1, merge_distance + 1), np.uint8))
    count, labels = cv2.connectedComponents(grouping, connectivity=8)
    height, width = mask.shape
    regions = []
    for label in range(1, count):
        selected = mask & (labels == label)
        ys, xs = np.nonzero(selected)
        if not len(xs):
            continue
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        w, h = x2 - x1, y2 - y1
        margin = round((math.sqrt((w + h) ** 2 + 4 * (context_ratio - 1) * w * h) - w - h) / 4)
        dx, dy = min(margin, x1, width - x2), min(margin, y1, height - y2)
        regions.append((x1 - dx, y1 - dy, x2 + dx, y2 + dy, label))
    regions.sort(key=lambda item: (item[1], item[0]))
    return labels, regions


def _validated_inputs(image_rgb, other_mask):
    image, mask = np.asarray(image_rgb), np.asarray(other_mask)
    if (image.ndim != 3 or image.shape[2] not in (3, 4)
            or image.dtype not in (np.uint8, np.uint16) or not all(image.shape[:2])):
        raise ValueError('LaMa 需要非空 RGB/RGBA uint8 或 uint16 圖像。')
    if mask.shape != image.shape[:2]:
        raise ValueError('LaMa Mask 尺寸與原圖不一致。')
    if mask.dtype.kind not in 'buif' or not np.isfinite(mask).all() or np.any(mask < 0):
        raise ValueError('LaMa Mask 必須為有限、非負的數值選區。')
    return image, mask > 0


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
    return model.eval().requires_grad_(False).to(device=device, dtype=torch.float32)


def _prepare(image, mask, max_size):
    """Match the reference node's uint8 resize, ceil64 and square REFLECT order."""
    height, width = image.shape[:2]
    scale = min(1.0, max_size / max(height, width))
    # The reference rounds to zero for an extremely thin crop. Clamp to one
    # before ceil64 solely to make that geometry valid; all other math matches.
    resized_w = math.ceil(max(1, round(width * scale)) / PAD_MODULO) * PAD_MODULO
    resized_h = math.ceil(max(1, round(height * scale)) / PAD_MODULO) * PAD_MODULO
    rgb = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    binary = cv2.resize(mask.astype(np.uint8) * 255, (resized_w, resized_h),
                        interpolation=cv2.INTER_LINEAR)
    side = max(resized_w, resized_h)
    rgb = cv2.copyMakeBorder(rgb, 0, side - resized_h, 0, side - resized_w, cv2.BORDER_REFLECT)
    binary = cv2.copyMakeBorder(binary, 0, side - resized_h, 0, side - resized_w, cv2.BORDER_REFLECT)
    return rgb, binary >= 128, (resized_h, resized_w)


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
        image, active = _validated_inputs(image_rgb, other_mask)
        output = np.zeros((*active.shape, 4), np.uint8)
        if not np.any(active):
            return output
        context = image[:, :, :3].copy()
        if context.dtype == np.uint16:
            context = (context >> 8).astype(np.uint8)
        with self._lock:
            labels, regions = region_crops(active)
            for x1, y1, x2, y2, label in regions:
                local_mask = active[y1:y2, x1:x2] & (labels[y1:y2, x1:x2] == label)
                repaired = self._predict_crop(context[y1:y2, x1:x2], local_mask)
                context[y1:y2, x1:x2][local_mask] = repaired[local_mask]
                local_output = output[y1:y2, x1:x2]
                local_output[local_mask, :3] = repaired[local_mask]
                local_output[local_mask, 3] = 255
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
        image_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        mask_tensor = torch.from_numpy(binary).unsqueeze(0).unsqueeze(0)

        def predict(device):
            image_float = image_tensor.to(device=device, dtype=torch.float32) / 255.0
            mask_float = mask_tensor.to(device=device, dtype=torch.float32)
            prediction = self.model(image_float, mask_float)
            # Spandrel's LaMa wrapper composites at network resolution before
            # quantization/resizing. The generator already masks its own input.
            return mask_float * prediction + (1 - mask_float) * image_float

        with torch.inference_mode():
            try:
                predicted = predict(self.device)
            except (RuntimeError, NotImplementedError):
                if self.device != 'mps':
                    raise
                # Retry the entire model on CPU; never discard FFT imaginary components.
                self.model = self.model.to('cpu')
                self.device = 'cpu'
                predicted = predict('cpu')
        prediction = predicted.detach().float().cpu().numpy()
        if prediction.shape != (1, 3, *binary.shape) or not np.isfinite(prediction).all():
            raise ValueError('LaMa 回傳無效的修補圖像。')
        small_h, small_w = small_shape
        repaired = prediction[0].transpose(1, 2, 0)
        repaired = np.clip(np.round(repaired * 255), 0, 255).astype(np.uint8)
        repaired = repaired[:small_h, :small_w]
        if (small_w, small_h) != (mask.shape[1], mask.shape[0]):
            repaired = cv2.resize(repaired, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        return repaired


_model = None
_model_lock = threading.Lock()


def inpaint_other(image_rgb, other_mask):
    """Return an RGBA repair layer; an empty mask does not initialize the model."""
    image, active = _validated_inputs(image_rgb, other_mask)
    if not np.any(active):
        return np.zeros((*active.shape, 4), np.uint8)
    global _model
    with _model_lock:
        if _model is None:
            _model = LamaLargeInpainter()
        return _model(image, active)
