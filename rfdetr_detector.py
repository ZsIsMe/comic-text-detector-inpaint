"""RF-DETR 分割模型 adapter。

輸出專案統一的偵測介面：
``__call__(image, ...) -> (raw_mask, refined_mask, blocks)``，其中 mask 是與原圖
同尺寸的 0/255 二值圖，內容為 text + onomatopoeia 兩個類別的 mask 聯集
（即需要被塗白的文字/狀聲詞區域）。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import threading

import cv2
import numpy as np

from mask_utils import DEFAULT_MASK_DILATE, MASK_UNIFICATION_METHODS, unify_mask as _unify_mask
from rfdetr_config import CLASS_NAMES, CLASS_THRESHOLDS, RESOLUTION


MIN_THRESHOLD = min(CLASS_THRESHOLDS.values())
MASK_MODES = {
    'text_onomatopoeia': ('text', 'onomatopoeia'),
    'text': ('text',),
    'onomatopoeia': ('onomatopoeia',),
    'all': CLASS_NAMES,
}


def source_key(image: np.ndarray) -> tuple:
    """Exact input identity: shape alone cannot prove a reused mask is current."""
    return (image.dtype.str, tuple(image.shape),
            hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest())


def model_signature(path: Path) -> tuple:
    stat = path.stat()
    return (str(path.resolve()), stat.st_size, stat.st_mtime_ns)


@dataclass
class BubblePrediction:
    """One call's unexpanded instance polygons, tied to its source and weights."""

    source: tuple
    model: tuple
    polygons: list[np.ndarray]
    error: str | None = None
    predictor: RfDetrSegDetector | None = None


def bubble_polygons(detections, shape) -> list[np.ndarray]:
    """Keep each class-2 instance's outer contour; never replace it with a box."""
    polygons = []
    instance_masks = detections.mask
    if instance_masks is None:
        return polygons
    for class_id, confidence, instance_mask in zip(
            detections.class_id, detections.confidence, instance_masks):
        if int(class_id) != 2 or float(confidence) < CLASS_THRESHOLDS['bubble']:
            continue
        if instance_mask is None:
            continue
        mask = np.asarray(instance_mask)
        if mask.shape != tuple(shape) or not np.isfinite(mask).all():
            continue
        contours, _ = cv2.findContours(
            (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            if len(contour) >= 3 and cv2.contourArea(contour) > 0:
                polygons.append(contour[:, 0, :].astype(np.float32))
    return polygons


def _auto_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return 'cuda'
        if torch.backends.mps.is_available():
            return 'mps'
    except Exception:
        pass
    return 'cpu'


class RfDetrSegDetector:
    """以 RF-DETR 分割權重偵測文字並輸出二值 mask 的 detector。"""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = 'auto',
        inpaint_mask_dilate: int = DEFAULT_MASK_DILATE,
        mask_unification_method: str = 'none',
        mask_mode: str = 'text_onomatopoeia',
        class_thresholds: dict[str, float] | None = None,
    ) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f'找不到 RF-DETR 模型檔：{path}')
        mask_unification_method = str(mask_unification_method)
        if mask_unification_method not in MASK_UNIFICATION_METHODS:
            raise ValueError(f'不支援的 Mask 合併方式：{mask_unification_method}')
        self.device = _auto_device() if str(device) == 'auto' else str(device)
        if self.device not in ('cpu', 'mps', 'cuda'):
            raise ValueError(f'不支援的裝置：{device}')
        mask_mode = str(mask_mode)
        if mask_mode not in MASK_MODES:
            raise ValueError(f'不支援的塗白範圍：{mask_mode}，可用 {tuple(MASK_MODES)}')
        self.model_path = path
        self.inpaint_mask_dilate = max(0, int(inpaint_mask_dilate))
        self.mask_unification_method = mask_unification_method
        self.mask_mode = mask_mode
        self.class_thresholds = dict(CLASS_THRESHOLDS)
        self.class_thresholds.update(class_thresholds or {})
        self._model = None
        self._predict_lock = threading.Lock()

    def _get_model(self):
        if self._model is None:
            try:
                from load_model import load_rfdetr_model
            except ImportError as exc:
                raise RuntimeError(
                    'RF-DETR 需要 rfdetr 與相關依賴，請先完成依賴安裝（參考 HANDOFF_RFDETR.md）。'
                ) from exc
            self._model = load_rfdetr_model(self.model_path, device=self.device)
        return self._model

    @staticmethod
    def _rgb(image):
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)

    def _predict(self, image, *, with_bubbles=False):
        threshold = min(self.class_thresholds.values())
        if with_bubbles:
            threshold = min(threshold, CLASS_THRESHOLDS['bubble'])
        # The loader and model must not be used concurrently by crop inference.
        with self._predict_lock:
            return self._get_model().predict(
                self._rgb(image), threshold=threshold,
                shape=(RESOLUTION, RESOLUTION), include_source_image=False)

    def predict_bubbles(self, image):
        return bubble_polygons(self._predict(image, with_bubbles=True), image.shape[:2])

    def _masks_from_detections(self, detections, shape):
        height, width = shape
        masks: dict[str, np.ndarray] = {
            name: np.zeros((height, width), dtype=bool) for name in CLASS_NAMES
        }
        if detections.mask is None:
            return masks
        for class_id, confidence, instance_mask in zip(
            detections.class_id,
            detections.confidence,
            detections.mask,
        ):
            name = (
                CLASS_NAMES[int(class_id)]
                if 0 <= int(class_id) < len(CLASS_NAMES)
                else 'text'
            )
            if float(confidence) < self.class_thresholds[name] or instance_mask is None:
                continue
            instance_mask = np.asarray(instance_mask, dtype=bool)
            if instance_mask.shape == (height, width):
                masks[name] |= instance_mask
        return masks

    def _result(self, detections, shape):
        masks = self._masks_from_detections(detections, shape)
        mask = np.zeros(shape, dtype=bool)
        for name in MASK_MODES[self.mask_mode]:
            mask |= masks[name]
        mask = mask.astype(np.uint8) * 255
        if self.inpaint_mask_dilate > 0 and np.any(mask):
            kernel = np.ones(
                (self.inpaint_mask_dilate, self.inpaint_mask_dilate), dtype=np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        mask = _unify_mask(mask, self.mask_unification_method)
        return mask, mask.copy(), []

    def detect_with_bubbles(self, image, **kwargs):
        """Return the normal text result and this call's separate bubble geometry."""
        detections = self._predict(image, with_bubbles=True)
        result = self._result(detections, image.shape[:2])
        try:
            polygons = bubble_polygons(detections, image.shape[:2])
            error = None
        except Exception as exc:
            # Bad bubble geometry must not discard otherwise valid text masks.
            polygons, error = [], str(exc)
        return result, BubblePrediction(
            source_key(image), model_signature(self.model_path), polygons, error, self)

    def __call__(
        self,
        image: np.ndarray,
        refine_mode: int | None = None,
        keep_undetected_mask: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, list]:
        del refine_mode, keep_undetected_mask
        return self._result(self._predict(image), image.shape[:2])
