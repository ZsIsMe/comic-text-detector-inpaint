"""RF-DETR 分割模型 adapter。

輸出與專案現有 ``ComicTextAndBubbleDetector`` 相同的介面：
``__call__(image, ...) -> (raw_mask, refined_mask, blocks)``，其中 mask 是與原圖
同尺寸的 0/255 二值圖，內容為 text + onomatopoeia 兩個類別的 mask 聯集
（即需要被塗白的文字/狀聲詞區域）。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ctbd_detector import CTBD_DEFAULT_MASK_DILATE, CTBD_MASK_UNIFICATION_METHODS, _unify_mask


CLASS_NAMES = ('text', 'onomatopoeia', 'bubble', 'panel')
CLASS_THRESHOLDS = {'text': 0.25, 'onomatopoeia': 0.20, 'bubble': 0.50, 'panel': 0.50}
RESOLUTION = 1152
MIN_THRESHOLD = min(CLASS_THRESHOLDS.values())
MASK_MODES = {
    'text_onomatopoeia': ('text', 'onomatopoeia'),
    'text': ('text',),
    'onomatopoeia': ('onomatopoeia',),
    'all': CLASS_NAMES,
}


def _auto_device() -> str:
    try:
        import torch

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
        inpaint_mask_dilate: int = CTBD_DEFAULT_MASK_DILATE,
        mask_unification_method: str = 'none',
        mask_mode: str = 'text_onomatopoeia',
        class_thresholds: dict[str, float] | None = None,
    ) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f'找不到 RF-DETR 模型檔：{path}')
        mask_unification_method = str(mask_unification_method)
        if mask_unification_method not in CTBD_MASK_UNIFICATION_METHODS:
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

    def _detect_masks(self, rgb: np.ndarray) -> dict[str, np.ndarray]:
        model = self._get_model()
        detections = model.predict(
            rgb,
            threshold=min(self.class_thresholds.values()),
            shape=(RESOLUTION, RESOLUTION),
            include_source_image=False,
        )
        height, width = rgb.shape[:2]
        masks: dict[str, np.ndarray] = {
            name: np.zeros((height, width), dtype=bool) for name in CLASS_NAMES
        }
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
            if float(confidence) < self.class_thresholds[name]:
                continue
            masks[name] |= np.asarray(instance_mask, dtype=bool)
        return masks

    def __call__(
        self,
        image: np.ndarray,
        refine_mode: int | None = None,
        keep_undetected_mask: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, list]:
        del refine_mode, keep_undetected_mask
        if image.ndim == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks = self._detect_masks(rgb)
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
        for name in MASK_MODES[self.mask_mode]:
            mask |= masks[name]
        mask = mask.astype(np.uint8) * 255
        if self.inpaint_mask_dilate > 0 and np.any(mask):
            kernel = np.ones(
                (self.inpaint_mask_dilate, self.inpaint_mask_dilate),
                dtype=np.uint8,
            )
            mask = cv2.dilate(mask, kernel, iterations=1)
        mask = _unify_mask(mask, self.mask_unification_method)
        return mask, mask.copy(), []
