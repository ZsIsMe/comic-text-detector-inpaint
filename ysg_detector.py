"""YSGYOLO adapter, ported from BallonsTranslator's detector_ysg.py (GPL-3.0).

See vendor/LICENSE-BallonsTranslator. Model weights remain external to git.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


# Model class names are stable keys; descriptions supplied by the user.
YSG_LABEL_DESCRIPTIONS = {
    'balloon': '氣泡外的文字',
    'qipao': '氣泡內的文字',
    'shuqing': '豎斜：豎著的氣泡內和氣泡外的傾斜文字',
    'changfangtiao': '長方條：全部橫向文字，不區分氣泡或矩形框內外',
    'hengxie': '橫斜：長方條的上位版，所有橫著的傾斜文字',
    'other': '框體：氣泡，以及任意包含文字的垂直、水平框體',
}
YSG_DEFAULT_LABELS = tuple(name for name in YSG_LABEL_DESCRIPTIONS if name != 'other')
# This exact YOLO26 checkpoint names classes 2/4 fangkuai/kuangwai. Per the
# user's choice, retain BT's six selectors without guessing class aliases.
YSG_UNSUPPORTED_LABELS = frozenset({'shuqing', 'hengxie'})


def round_mask_components(mask: np.ndarray, radius: int) -> np.ndarray:
    """Trim connected-component corners as in BT's utils/imgproc_utils.py.

    Intersecting with a rounded rectangle only removes pixels, preserving holes
    and gaps. Nearby detections that already touch are treated as one component.

    >>> mask = np.zeros((8, 8), dtype=np.uint8)
    >>> mask[1:7, 1:7] = 255
    >>> rounded = round_mask_components(mask, 2)
    >>> bool(rounded[1, 1] == 0 and rounded[3, 3] == 255)
    True
    """
    if radius <= 0:
        return mask.copy()
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    rounded = np.zeros_like(mask)
    for label in range(1, count):
        x, y, width, height, _ = (int(value) for value in stats[label])
        component = (labels[y:y + height, x:x + width] == label).astype(np.uint8) * 255
        corner_radius = min(int(radius), width // 2, height // 2)
        if corner_radius > 0:
            rounded_rect = np.zeros((height, width), dtype=np.uint8)
            # At the maximum radius one central rectangle may have no width.
            if corner_radius <= width - corner_radius - 1:
                cv2.rectangle(rounded_rect, (corner_radius, 0),
                              (width - corner_radius - 1, height - 1), 255, -1)
            if corner_radius <= height - corner_radius - 1:
                cv2.rectangle(rounded_rect, (0, corner_radius),
                              (width - 1, height - corner_radius - 1), 255, -1)
            for cx, cy in (
                (corner_radius, corner_radius),
                (width - corner_radius - 1, corner_radius),
                (corner_radius, height - corner_radius - 1),
                (width - corner_radius - 1, height - corner_radius - 1),
            ):
                cv2.circle(rounded_rect, (cx, cy), corner_radius, 255, -1)
            component = cv2.bitwise_and(component, rounded_rect)
        region = rounded[y:y + height, x:x + width]
        # OR preserves other components that share this bounding rectangle.
        region[:] = np.bitwise_or(region, component.astype(mask.dtype))
    return rounded


def _text_blocks(polygons: list[np.ndarray], width: int, height: int,
                 merge: bool, vertical: bool) -> list:
    """Use the upstream grouping algorithm with the app's existing TextBlock.

    The vertical hint affects block direction and reading order, never mask area.
    """
    from utils.textblock import TextBlock
    from vendor.ysg_textlines_merge import Quadrilateral, merge_bboxes_text_region

    lines = [Quadrilateral(points, '', 1.0) for points in polygons]
    if vertical:
        for line in lines:
            line.direction = 'v'
    groups = (
        (group for group, _, _ in merge_bboxes_text_region(lines, width, height))
        if merge else ([line] for line in lines)
    )
    blocks = []
    for group in groups:
        points = [line.pts.tolist() for line in group]
        is_vertical = vertical or sum(line.direction == 'v' for line in group) > len(group) / 2
        angle = float(np.rad2deg(np.mean([line.angle for line in group])) - 90)
        block = TextBlock(
            [0, 0, 0, 0], lines=points, vertical=is_vertical,
            font_size=int(min(line.font_size for line in group)),
            angle=0 if abs(angle) < 3 else angle,
        )
        block.adjust_bbox()
        blocks.append(block)

    # BallonsTranslator's sort_regions: page rows first, then reading direction.
    right_to_left = any(block.vertical for block in blocks)
    ordered = []
    for block in sorted(blocks, key=lambda item: item.center()[1]):
        x, y = block.center()
        for index, previous in enumerate(ordered):
            if y > previous.xyxy[3]:
                continue
            if y < previous.xyxy[1]:
                ordered.insert(index + 1, block)
                break
            if (right_to_left and x > previous.center()[0]) or (
                not right_to_left and x < previous.center()[0]
            ):
                ordered.insert(index, block)
                break
        else:
            ordered.append(block)
    return ordered


class YSGYoloDetector:
    """Produce the (raw_mask, refined_mask, blocks) used by all app workflows."""

    def __init__(
        self, model_path: str | Path, *, device: str = 'auto',
        labels: list[str] | tuple[str, ...] | None = None,
        merge_text_lines: bool = True, source_text_vertical: bool = False,
        mask_dilate_size: int = 0, mask_corner_radius: int = 0,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f'找不到 YSGYOLO 模型檔：{self.model_path}')
        self.labels = tuple(YSG_DEFAULT_LABELS if labels is None else labels)
        unknown = set(self.labels) - YSG_LABEL_DESCRIPTIONS.keys()
        if unknown:
            raise ValueError(f'不支援的 YSGYOLO 標籤：{sorted(unknown)}')
        if device not in {'auto', 'cpu', 'mps', 'cuda'}:
            raise ValueError(f'不支援的 YSGYOLO 裝置：{device}')
        self.merge_text_lines = bool(merge_text_lines)
        self.source_text_vertical = bool(source_text_vertical)
        self.mask_dilate_size = max(0, int(mask_dilate_size))
        self.mask_corner_radius = max(0, int(mask_corner_radius))
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                'YSGYOLO 需要 ultralytics>=8.4.14，請重新執行 bootstrap.py 安裝依賴。'
            ) from exc
        mps_available = torch.backends.mps.is_available()
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else ('mps' if mps_available else 'cpu')
        if device == 'mps' and not mps_available:
            raise RuntimeError('目前環境無法使用 MPS，請改選 CPU 或自動。')
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('目前環境無法使用 CUDA，請改選 CPU 或自動。')
        self.device = device
        self.model = YOLO(str(self.model_path)).to(device=device)

    def _mask_and_blocks(self, result: object, shape: tuple[int, int]) -> tuple[np.ndarray, list]:
        height, width = shape
        mask = np.zeros(shape, dtype=np.uint8)
        polygons = []
        valid_ids = {index for index, name in result.names.items() if name in self.labels}
        # Standard YOLO26 boxes; OBB is retained from the upstream adapter.
        for detections, oriented in ((result.boxes, False), (result.obb, True)):
            if detections is None:
                continue
            classes = detections.cls.cpu().numpy().astype(int)
            coordinates = (detections.xyxyxyxy if oriented else detections.xyxy).cpu().numpy()
            for class_id, coords in zip(classes, coordinates):
                if class_id not in valid_ids or not np.isfinite(coords).all():
                    continue
                if oriented:
                    points = coords.astype(np.int32)
                else:
                    x1, y1, x2, y2 = coords.astype(int)
                    points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                points[:, 0] = np.clip(points[:, 0], 0, width - 1)
                points[:, 1] = np.clip(points[:, 1], 0, height - 1)
                if cv2.contourArea(points) <= 0:
                    continue
                cv2.fillPoly(mask, [points], 255)
                polygons.append(points)
        # Merge only text-block metadata, exactly as in the source detector.
        blocks = _text_blocks(polygons, width, height, self.merge_text_lines, self.source_text_vertical)
        radius = self.mask_dilate_size
        if radius > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            mask = cv2.dilate(mask, kernel)
        # Round the final detected mask, before it is added to existing layers.
        if self.mask_corner_radius > 0:
            mask = round_mask_components(mask, self.mask_corner_radius)
        return mask, blocks

    def __call__(self, image: np.ndarray, refine_mode: int | None = None,
                 keep_undetected_mask: bool = False) -> tuple[np.ndarray, np.ndarray, list]:
        del refine_mode, keep_undetected_mask
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        if not self.labels:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            return mask, mask.copy(), []
        result = self.model.predict(
            source=image, save=False, show=False, verbose=False,
            conf=0.3, iou=0.5, imgsz=1024, agnostic_nms=True,
        )[0]
        mask, blocks = self._mask_and_blocks(result, image.shape[:2])
        return mask, mask.copy(), blocks
