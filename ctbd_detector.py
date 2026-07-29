"""ONNX adapter for ogkalu/comic-text-and-bubble-detector."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


CTBD_INPUT_SIZE = 640
CTBD_CONFIDENCE_THRESHOLD = 0.3
CTBD_DEFAULT_MASK_DILATE = 4
CTBD_MASK_UNIFICATION_METHODS = {'none', 'rectangle', 'hull'}
CTBD_TEXT_REGION_FILTERS = {'all', 'text_bubble', 'text_free'}


def _calculate_iou(rect1: np.ndarray | list, rect2: np.ndarray | list) -> float:
    x1, y1, x2, y2 = rect1
    px1, py1, px2, py2 = rect2
    ix1, iy1 = max(x1, px1), max(y1, py1)
    ix2, iy2 = min(x2, px2), min(y2, py2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area1 = max(0, x2 - x1) * max(0, y2 - y1)
    area2 = max(0, px2 - px1) * max(0, py2 - py1)
    union = area1 + area2 - intersection
    return float(intersection / union) if union > 0 else 0.0


def _rectangle_fits(bigger: np.ndarray | list, smaller: np.ndarray | list) -> bool:
    x1, y1, x2, y2 = bigger
    sx1, sy1, sx2, sy2 = smaller
    return x1 <= sx1 and y1 <= sy1 and x2 >= sx2 and y2 >= sy2


def _merge_duplicate_boxes(boxes: np.ndarray, iou_threshold: float = 0.7) -> np.ndarray:
    if boxes.size == 0 or len(boxes) < 2:
        return boxes

    remaining = [box.tolist() for box in boxes]
    merged: list[list[int]] = []
    while remaining:
        component = [remaining.pop(0)]
        changed = True
        while changed:
            changed = False
            for candidate in remaining[:]:
                if any(_calculate_iou(candidate, member) >= iou_threshold for member in component):
                    component.append(candidate)
                    remaining.remove(candidate)
                    changed = True
        values = np.asarray(component)
        merged.append([
            int(values[:, 0].min()),
            int(values[:, 1].min()),
            int(values[:, 2].max()),
            int(values[:, 3].max()),
        ])
    return np.asarray(merged, dtype=np.int32)


def _remove_contained_boxes(boxes: np.ndarray, threshold: float = 0.8) -> np.ndarray:
    if boxes.size == 0 or len(boxes) < 2:
        return boxes

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    ordered = boxes[np.argsort(areas)[::-1]]
    kept: list[np.ndarray] = []
    for box in ordered:
        area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
        if area <= 0:
            continue
        contained = False
        for larger in kept:
            ix1, iy1 = max(box[0], larger[0]), max(box[1], larger[1])
            ix2, iy2 = min(box[2], larger[2]), min(box[3], larger[3])
            intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if intersection / area >= threshold:
                contained = True
                break
        if not contained:
            kept.append(box)
    return np.asarray(kept, dtype=np.int32) if kept else np.empty((0, 4), dtype=np.int32)


def _content_boxes(image_crop: np.ndarray) -> list[tuple[int, int, int, int]]:
    if image_crop.size == 0:
        return []
    gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    boxes: list[tuple[int, int, int, int]] = []
    for threshold_type in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            threshold_type,
            11,
            2,
        )
        count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for label in range(1, count):
            if int(stats[label, cv2.CC_STAT_AREA]) <= 10:
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            # Components touching the crop edge are normally the background.
            if x > 0 and y > 0 and x + w < width and y + h < height:
                boxes.append((x, y, x + w, y + h))
    return boxes


def _unify_mask(mask: np.ndarray, method: str) -> np.ndarray:
    if method == 'none':
        return mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask)
    points = np.concatenate(contours, axis=0)
    unified = np.zeros_like(mask)
    if method == 'rectangle':
        x, y, width, height = cv2.boundingRect(points)
        cv2.rectangle(unified, (x, y), (x + width, y + height), 255, -1)
    else:
        hull = cv2.convexHull(points)
        cv2.drawContours(unified, [hull], 0, 255, -1)
    return unified


class _ImageSlicer:
    """Split very tall comic pages and merge detections back into page coordinates."""

    def __init__(self) -> None:
        self.trigger_ratio = 3.5
        self.target_ratio = 3.0
        self.overlap_ratio = 0.2
        self.minimum_last_slice_ratio = 0.7

    def process(
        self,
        image: np.ndarray,
        detect: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = image.shape[:2]
        if width <= 0 or height / width <= self.trigger_ratio:
            return detect(image)

        slice_height = max(1, int(width * self.target_ratio))
        step = max(1, int(slice_height * (1 - self.overlap_ratio)))
        slice_count = max(1, math.ceil(height / step))
        if (
            slice_count > 1
            and (height - (slice_count - 1) * step) / slice_height
            < self.minimum_last_slice_ratio
        ):
            slice_count -= 1

        collected_bubbles: list[np.ndarray] = []
        collected_text: list[np.ndarray] = []
        for index in range(slice_count):
            start_y = min(index * step, height - 1)
            end_y = height if index == slice_count - 1 else min(start_y + slice_height, height)
            bubble_boxes, text_boxes = detect(image[start_y:end_y].copy())
            for boxes, destination in (
                (bubble_boxes, collected_bubbles),
                (text_boxes, collected_text),
            ):
                if boxes.size:
                    adjusted = boxes.copy()
                    adjusted[:, [1, 3]] += start_y
                    destination.append(adjusted)

        def merge(collected: list[np.ndarray]) -> np.ndarray:
            if not collected:
                return np.empty((0, 4), dtype=np.int32)
            return _remove_contained_boxes(
                _merge_duplicate_boxes(np.vstack(collected), 0.5),
                0.85,
            )

        return merge(collected_bubbles), merge(collected_text)


class ComicTextAndBubbleDetector:
    """Return a CTD-compatible binary text mask from the RT-DETR-V2 ONNX model.

    The callable result intentionally matches the legacy detector's
    ``(raw_mask, refined_mask, blocks)`` shape so the existing processing and
    manual-edit workflows can use either model without branching.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        inpaint_mask_dilate: int = CTBD_DEFAULT_MASK_DILATE,
        mask_unification_method: str = 'none',
        text_region_filter: str = 'all',
    ) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f'找不到 CTBD 模型檔：{path}')
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError('CTBD 需要 onnxruntime，請重新執行 bootstrap.py 安裝依賴。') from exc

        mask_unification_method = str(mask_unification_method)
        text_region_filter = str(text_region_filter)
        if mask_unification_method not in CTBD_MASK_UNIFICATION_METHODS:
            raise ValueError(f'不支援的 Mask 合併方式：{mask_unification_method}')
        if text_region_filter not in CTBD_TEXT_REGION_FILTERS:
            raise ValueError(f'不支援的文字區域篩選值：{text_region_filter}')

        self.model_path = path
        self.inpaint_mask_dilate = max(0, int(inpaint_mask_dilate))
        self.mask_unification_method = mask_unification_method
        self.text_region_filter = text_region_filter
        self.session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
        self.slicer = _ImageSlicer()

    def _detect_boxes(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        original_height, original_width = image.shape[:2]
        resized = cv2.resize(image, (CTBD_INPUT_SIZE, CTBD_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        target_sizes = np.asarray([[original_width, original_height]], dtype=np.int64)
        labels, boxes, scores = self.session.run(
            None,
            {'images': tensor, 'orig_target_sizes': target_sizes},
        )

        bubble_boxes: list[list[int]] = []
        text_boxes: list[list[int]] = []
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            if float(score) < CTBD_CONFIDENCE_THRESHOLD:
                continue
            x1, y1, x2, y2 = (int(value) for value in box)
            x1 = max(0, min(original_width, x1))
            x2 = max(0, min(original_width, x2))
            y1 = max(0, min(original_height, y1))
            y2 = max(0, min(original_height, y2))
            if x2 - x1 > 5 and y2 - y1 > 5:
                if int(label) == 0:
                    bubble_boxes.append([x1, y1, x2, y2])
                elif int(label) in (1, 2):
                    text_boxes.append([x1, y1, x2, y2])

        def as_array(items: list[list[int]]) -> np.ndarray:
            return np.asarray(items, dtype=np.int32) if items else np.empty((0, 4), dtype=np.int32)

        return as_array(bubble_boxes), as_array(text_boxes)

    def _filter_text_boxes(self, text_boxes: np.ndarray, bubble_boxes: np.ndarray) -> np.ndarray:
        if self.text_region_filter == 'all' or text_boxes.size == 0:
            return text_boxes
        bubbles = bubble_boxes.tolist() if bubble_boxes.size else []
        kept: list[np.ndarray] = []
        for text_box in text_boxes:
            inside_bubble = any(
                _rectangle_fits(bubble, text_box)
                or _calculate_iou(bubble, text_box) >= 0.2
                for bubble in bubbles
            )
            text_class = 'text_bubble' if inside_bubble else 'text_free'
            if text_class == self.text_region_filter:
                kept.append(text_box)
        return np.asarray(kept, dtype=np.int32) if kept else np.empty((0, 4), dtype=np.int32)

    def _mask_from_boxes(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        for x1, y1, x2, y2 in boxes:
            crop_y1 = max(0, int(y1) - 10)
            crop_y2 = min(height, int(y2) + 10)
            crop_x1 = max(0, int(x1))
            crop_x2 = min(width, int(x2))
            if crop_x1 >= crop_x2 or crop_y1 >= crop_y2:
                continue
            crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
            local_mask = np.zeros((height, width), dtype=np.uint8)
            for bx1, by1, bx2, by2 in _content_boxes(crop):
                cv2.rectangle(
                    local_mask,
                    (crop_x1 + bx1, crop_y1 + by1),
                    (crop_x1 + bx2, crop_y1 + by2),
                    255,
                    -1,
                )
            if self.inpaint_mask_dilate > 0 and np.any(local_mask):
                kernel = np.ones(
                    (self.inpaint_mask_dilate, self.inpaint_mask_dilate),
                    dtype=np.uint8,
                )
                local_mask = cv2.dilate(local_mask, kernel, iterations=1)
            local_mask = _unify_mask(local_mask, self.mask_unification_method)
            mask = cv2.bitwise_or(mask, local_mask)
        return mask

    def __call__(
        self,
        image: np.ndarray,
        refine_mode: int | None = None,
        keep_undetected_mask: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, list]:
        del refine_mode, keep_undetected_mask
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        bubble_boxes, text_boxes = self.slicer.process(image, self._detect_boxes)
        bubble_boxes = _remove_contained_boxes(_merge_duplicate_boxes(bubble_boxes), 0.8)
        text_boxes = _remove_contained_boxes(_merge_duplicate_boxes(text_boxes), 0.8)
        filtered_text_boxes = self._filter_text_boxes(text_boxes, bubble_boxes)
        mask = self._mask_from_boxes(image, filtered_text_boxes)
        return mask, mask.copy(), []
