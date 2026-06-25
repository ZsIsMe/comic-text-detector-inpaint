#!/usr/bin/env python3
"""Detect text masks and create solid-background inpaint overlays."""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
VENDOR_DIR = SCRIPT_DIR / 'vendor'
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

from inference import TextDetector
from utils.io_utils import find_all_imgs, imread, imwrite
from utils.textmask import REFINEMASK_ANNOTATION


OUTPUT_DIR = 'ctd_inpainted'
MASK_DIR = 'mask'
OTHER_MASK_DIR = 'other_mask'
INPAINTED_DIR = 'inpainted'
BACKGROUND_SAMPLE_CACHE_DIR = 'background_sample_cache'
REPORT_JSON = 'solid_inpaint_report.json'
PREVIEW_PDF = 'preview_report.pdf'
MODEL_PATH = Path(__file__).resolve().parent / 'models' / 'comictextdetector.pt'

REPAIR_EXPAND_PX = 3
SAMPLE_RING_PX = 3
GROUP_MERGE_PX = 16
BLOCK_PADDING_PX = 8
MIN_COMPONENT_AREA = 4
MIN_BOX_SIZE_PX = 2
MIN_SAMPLE_PIXELS = 12
MIN_DIRECTIONAL_SAMPLE_PIXELS = 24
BACKGROUND_WAND_TOLERANCE = 24
BACKGROUND_WAND_SEED_MIN_DISTANCE_PX = 4
BACKGROUND_WAND_SEED_SEARCH_PX = 36
BACKGROUND_WAND_ROI_PADDING_PX = 72
BACKGROUND_WAND_EDGE_MARGIN_PX = 6
BACKGROUND_WAND_MAX_CANDIDATES = 16
BACKGROUND_WAND_MAX_REJECTS = 5
BACKGROUND_WAND_MAX_AREA_RATIO = 0.45
BACKGROUND_WAND_DARK_SEED_MIN = 90
BACKGROUND_WAND_GRADIENT_MAX = 24
SOLID_P90_P10_MAX = 12
SOLID_PEAK_RATIO_MIN = 0.62
SOLID_CLOSE_DELTA_MAX = 10
SOLID_CLOSE_RATIO_MIN = 0.72
SOLID_P95_DELTA_MAX = 16
DIRECTIONAL_SOLID_P90_P10_MAX = 8
DIRECTIONAL_SOLID_CLOSE_RATIO_MIN = 0.82
DIRECTIONAL_SOLID_P95_DELTA_MAX = 12
DIRECTIONAL_FALLBACK_MAX_FULL_SPREAD = 28
DIRECTIONAL_FALLBACK_MIN_FULL_CLOSE_RATIO = 0.45
DIRECTIONAL_FILL_AGREEMENT_MAX = 10
MIN_DIRECTIONAL_AGREEMENT_COUNT = 2
WHITE_DOMINANT_MIN = 242
WHITE_PEAK_RATIO_MIN = 0.68
WHITE_FULL_PEAK_RATIO_MIN = 0.18
WHITE_CLOSE_DELTA_MAX = 10
WHITE_CLOSE_RATIO_MIN = 0.78
DIRECTIONAL_WHITE_CLOSE_RATIO_MIN = 0.84
WHITE_P95_DELTA_MAX = 18
PREVIEW_PAGE_WIDTH = 2400
PREVIEW_MARGIN = 36
PREVIEW_GAP = 24
PREVIEW_HEADER_HEIGHT = 72
PREVIEW_LABEL_HEIGHT = 34


@dataclass
class SolidQuality:
    is_solid: bool
    score: float
    fill_bgr: tuple[int, int, int]
    max_spread: int
    min_peak_ratio: float
    close_ratio: float
    p95_delta: int
    white_close_ratio: float
    white_p95_delta: int
    sample_pixels: int
    mode: str


def _ensure_dirs(img_dir: str) -> dict[str, str]:
    out_dir = osp.join(img_dir, OUTPUT_DIR)
    paths = {
        'output': out_dir,
        'mask': osp.join(out_dir, MASK_DIR),
        'other_mask': osp.join(out_dir, OTHER_MASK_DIR),
        'inpainted': osp.join(out_dir, INPAINTED_DIR),
        'background_sample_cache': osp.join(out_dir, BACKGROUND_SAMPLE_CACHE_DIR),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def _output_path(paths: dict[str, str], img_path: str) -> str:
    return osp.join(paths['inpainted'], f'{Path(img_path).stem}.png')


def _mask_path(paths: dict[str, str], img_path: str) -> str:
    return osp.join(paths['mask'], f'{Path(img_path).stem}.png')


def _other_mask_path(paths: dict[str, str], img_path: str) -> str:
    return osp.join(paths['other_mask'], f'{Path(img_path).stem}.png')


def _mask_hash(mask: np.ndarray | None) -> int:
    if mask is None:
        return 0
    mask_bin = np.where(mask > 0, 255, 0).astype(np.uint8)
    shape_hash = zlib.crc32(str(mask_bin.shape).encode('ascii'))
    return zlib.crc32(mask_bin.tobytes(), shape_hash)


def _background_sample_cache_path(paths: dict[str, str], img_path: str) -> str:
    cache_dir = paths.get('background_sample_cache') or osp.join(paths['output'], BACKGROUND_SAMPLE_CACHE_DIR)
    return osp.join(cache_dir, f'{Path(img_path).stem}.npz')


def _save_background_sample_cache(
    paths: dict[str, str],
    img_path: str,
    mask_hash: int,
    sample: np.ndarray,
) -> None:
    cache_path = _background_sample_cache_path(paths, img_path)
    os.makedirs(osp.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        mask_hash=np.array(mask_hash, dtype=np.uint32),
        sample=np.where(sample > 0, 255, 0).astype(np.uint8),
    )


def _clip_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(width, int(x1))),
        max(0, min(height, int(y1))),
        max(0, min(width, int(x2))),
        max(0, min(height, int(y2))),
    )


def _expand_box(
    box: tuple[int, int, int, int],
    padding: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return _clip_box((x1 - padding, y1 - padding, x2 + padding, y2 + padding), width, height)


def _boxes_near(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    padding: int,
) -> bool:
    return not (
        a[2] + padding < b[0]
        or b[2] + padding < a[0]
        or a[3] + padding < b[1]
        or b[3] + padding < a[1]
    )


def _union_box(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    )


def _component_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    text_mask = (mask > 0).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(text_mask, connectivity=8)
    boxes: list[tuple[int, int, int, int]] = []
    for label in range(1, count):
        x, y, w, h, area = [int(value) for value in stats[label]]
        if area < MIN_COMPONENT_AREA or w < MIN_BOX_SIZE_PX or h < MIN_BOX_SIZE_PX:
            continue
        boxes.append((x, y, x + w, y + h))
    return boxes


def _merge_boxes(
    boxes: list[tuple[int, int, int, int]],
    padding: int,
    width: int,
    height: int,
) -> list[tuple[int, int, int, int]]:
    merged = [_clip_box(box, width, height) for box in boxes]
    changed = True
    while changed:
        changed = False
        for a in range(len(merged)):
            for b in range(a + 1, len(merged)):
                if _boxes_near(merged[a], merged[b], padding):
                    merged[a] = _union_box(merged[a], merged[b])
                    del merged[b]
                    changed = True
                    break
            if changed:
                break
    return [
        _expand_box(box, BLOCK_PADDING_PX, width, height)
        for box in merged
    ]


def _kernel(radius: int) -> np.ndarray:
    size = radius * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _mask_for_box(text_mask: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    h, w = text_mask.shape[:2]
    x1, y1, x2, y2 = _clip_box(box, w, h)
    mask = np.zeros((h, w), dtype=np.uint8)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = text_mask[y1:y2, x1:x2]
    return mask


def _direction_mask(shape: tuple[int, int], repair_area: np.ndarray, direction: str) -> np.ndarray:
    ys, xs = np.where(repair_area > 0)
    mask = np.zeros(shape, dtype=np.uint8)
    if xs.size == 0:
        return mask
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    if direction == 'top':
        mask[:top, left:right] = 255
    elif direction == 'bottom':
        mask[bottom:, left:right] = 255
    elif direction == 'left':
        mask[top:bottom, :left] = 255
    elif direction == 'right':
        mask[top:bottom, right:] = 255
    return mask


def sample_ring_from_mask(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape[:2]
    text_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    boxes = _merge_boxes(_component_boxes(text_mask), GROUP_MERGE_PX, width, height)
    repair_kernel = _kernel(REPAIR_EXPAND_PX)
    ring_kernel = _kernel(SAMPLE_RING_PX)
    sample_mask = np.zeros((height, width), dtype=np.uint8)

    for box in boxes:
        local_text = _mask_for_box(text_mask, box)
        if not np.any(local_text):
            continue
        repair_area = cv2.dilate(local_text, repair_kernel, iterations=1)
        expanded = cv2.dilate(repair_area, ring_kernel, iterations=1)
        sample_ring = cv2.bitwise_and(expanded, cv2.bitwise_not(repair_area))
        sample_ring = cv2.bitwise_and(sample_ring, cv2.bitwise_not(text_mask))
        sample_mask = cv2.bitwise_or(sample_mask, sample_ring)
    return sample_mask


def _hist_channel(values: np.ndarray) -> tuple[int, float, int]:
    if values.size == 0:
        return 255, 0.0, 0
    hist = np.bincount(values.astype(np.uint8), minlength=256)
    total = int(hist.sum())
    peak_value = int(hist.argmax())
    peak_ratio = float(hist[peak_value] / total) if total else 0.0
    cdf = np.cumsum(hist)
    p10 = int(np.searchsorted(cdf, total * 0.10, side='left'))
    p90 = int(np.searchsorted(cdf, total * 0.90, side='left'))
    return p90 - p10, peak_ratio, peak_value


def _dominant_channel(values: np.ndarray) -> int:
    if values.size == 0:
        return 0
    hist = np.bincount(values.astype(np.uint8), minlength=256)
    return int(hist.argmax())


def _sample_ring_for_local_text(
    text_mask: np.ndarray,
    local_text: np.ndarray,
    repair_kernel: np.ndarray,
    ring_kernel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    repair_area = cv2.dilate(local_text, repair_kernel, iterations=1)
    expanded = cv2.dilate(repair_area, ring_kernel, iterations=1)
    sample_ring = cv2.bitwise_and(expanded, cv2.bitwise_not(repair_area))
    sample_ring = cv2.bitwise_and(sample_ring, cv2.bitwise_not(text_mask))
    return repair_area, sample_ring


def _dominant_bgr(samples: np.ndarray) -> tuple[int, int, int]:
    return (
        _dominant_channel(samples[:, 0]),
        _dominant_channel(samples[:, 1]),
        _dominant_channel(samples[:, 2]),
    )


def _seed_search_mask(repair_area: np.ndarray, text_mask: np.ndarray) -> np.ndarray:
    outer = cv2.dilate(repair_area, _kernel(BACKGROUND_WAND_SEED_SEARCH_PX), iterations=1)
    inner = cv2.dilate(repair_area, _kernel(BACKGROUND_WAND_SEED_MIN_DISTANCE_PX), iterations=1)
    search = cv2.bitwise_and(outer, cv2.bitwise_not(inner))
    search = cv2.bitwise_and(search, cv2.bitwise_not(text_mask))
    return search


def _candidate_seed_points(
    color_img: np.ndarray,
    text_mask: np.ndarray,
    repair_area: np.ndarray,
    sample_ring: np.ndarray,
) -> list[tuple[int, int]]:
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    gradient = np.maximum(np.abs(grad_x), np.abs(grad_y)).astype(np.uint16)

    search = _seed_search_mask(repair_area, text_mask) > 0
    clean = (
        search
        & (text_mask == 0)
        & (repair_area == 0)
        & (gray >= BACKGROUND_WAND_DARK_SEED_MIN)
        & (gradient <= BACKGROUND_WAND_GRADIENT_MAX)
    )
    if int(np.count_nonzero(clean)) < MIN_SAMPLE_PIXELS:
        clean = search & (text_mask == 0) & (repair_area == 0) & (gray >= BACKGROUND_WAND_DARK_SEED_MIN)
    if int(np.count_nonzero(clean)) < MIN_SAMPLE_PIXELS:
        clean = (sample_ring > 0) & (text_mask == 0) & (repair_area == 0)
    if int(np.count_nonzero(clean)) == 0:
        return []

    seed_pixels = color_img[clean]
    dominant = np.array(_dominant_bgr(seed_pixels), dtype=np.int16)
    ys, xs = np.where(clean)
    colors = color_img[ys, xs].astype(np.int16)
    color_delta = np.max(np.abs(colors - dominant), axis=1)
    center_y, center_x = np.mean(np.where(repair_area > 0), axis=1)
    distances = np.hypot(xs.astype(np.float32) - float(center_x), ys.astype(np.float32) - float(center_y))
    scores = color_delta.astype(np.float32) + distances * 0.03 + gradient[ys, xs].astype(np.float32) * 0.3
    order = np.argsort(scores)

    points: list[tuple[int, int]] = []
    used = np.zeros(clean.shape, dtype=np.uint8)
    suppress_kernel = _kernel(5)
    for idx in order:
        x, y = int(xs[idx]), int(ys[idx])
        if used[y, x] > 0:
            continue
        points.append((x, y))
        marker = np.zeros(clean.shape, dtype=np.uint8)
        marker[y, x] = 255
        used = cv2.bitwise_or(used, cv2.dilate(marker, suppress_kernel, iterations=1))
        if len(points) >= BACKGROUND_WAND_MAX_CANDIDATES:
            break
    return points


def _wand_selection_from_seed(
    color_img: np.ndarray,
    seed: tuple[int, int],
    roi_box: tuple[int, int, int, int],
) -> np.ndarray:
    h, w = color_img.shape[:2]
    x1, y1, x2, y2 = roi_box
    roi = color_img[y1:y2, x1:x2]
    seed_x, seed_y = seed[0] - x1, seed[1] - y1
    if seed_x < 0 or seed_y < 0 or seed_x >= roi.shape[1] or seed_y >= roi.shape[0]:
        return np.zeros((h, w), dtype=np.uint8)
    flood_mask = np.zeros((roi.shape[0] + 2, roi.shape[1] + 2), dtype=np.uint8)
    diff = (BACKGROUND_WAND_TOLERANCE,) * 3
    flags = 8 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    cv2.floodFill(roi.copy(), flood_mask, (seed_x, seed_y), (0, 0, 0), diff, diff, flags)
    selection = np.zeros((h, w), dtype=np.uint8)
    selection[y1:y2, x1:x2] = np.where(
        flood_mask[1:roi.shape[0] + 1, 1:roi.shape[1] + 1] > 0,
        255,
        0,
    ).astype(np.uint8)
    return selection


def _touches_roi_edge(selection: np.ndarray, roi_box: tuple[int, int, int, int], margin: int) -> bool:
    x1, y1, x2, y2 = roi_box
    roi_selection = selection[y1:y2, x1:x2] > 0
    if not np.any(roi_selection):
        return False
    margin = max(1, min(margin, roi_selection.shape[0], roi_selection.shape[1]))
    return bool(
        np.any(roi_selection[:margin, :])
        or np.any(roi_selection[-margin:, :])
        or np.any(roi_selection[:, :margin])
        or np.any(roi_selection[:, -margin:])
    )


def _background_wand_sample(
    color_img: np.ndarray,
    text_mask: np.ndarray,
    repair_area: np.ndarray,
    sample_ring: np.ndarray,
) -> np.ndarray:
    seeds = _candidate_seed_points(color_img, text_mask, repair_area, sample_ring)
    if not seeds:
        return sample_ring

    h, w = sample_ring.shape[:2]
    ys, xs = np.where(repair_area > 0)
    if xs.size == 0:
        return sample_ring
    roi_box = _expand_box(
        (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
        BACKGROUND_WAND_ROI_PADDING_PX,
        w,
        h,
    )
    roi_area = max(1, (roi_box[2] - roi_box[0]) * (roi_box[3] - roi_box[1]))
    max_area = int(roi_area * BACKGROUND_WAND_MAX_AREA_RATIO)
    x1, y1, x2, y2 = roi_box
    contact_kernel = _kernel(3)
    contact_area = cv2.dilate(repair_area, contact_kernel, iterations=1)
    best_selection: np.ndarray | None = None
    best_score = -1.0
    reject_count = 0

    for seed in seeds:
        if (
            seed[0] <= x1 + BACKGROUND_WAND_EDGE_MARGIN_PX
            or seed[0] >= x2 - BACKGROUND_WAND_EDGE_MARGIN_PX - 1
            or seed[1] <= y1 + BACKGROUND_WAND_EDGE_MARGIN_PX
            or seed[1] >= y2 - BACKGROUND_WAND_EDGE_MARGIN_PX - 1
        ):
            continue
        selection = _wand_selection_from_seed(color_img, seed, roi_box)
        selection = cv2.bitwise_and(selection, cv2.bitwise_not(text_mask))
        selection = cv2.bitwise_and(selection, cv2.bitwise_not(repair_area))
        if _touches_roi_edge(selection, roi_box, BACKGROUND_WAND_EDGE_MARGIN_PX):
            reject_count += 1
            if reject_count >= BACKGROUND_WAND_MAX_REJECTS and best_selection is None:
                break
            continue
        area = int(np.count_nonzero(selection))
        if area < MIN_SAMPLE_PIXELS or area > max_area:
            if area > max_area:
                reject_count += 1
                if reject_count >= BACKGROUND_WAND_MAX_REJECTS and best_selection is None:
                    break
            continue
        contact = int(np.count_nonzero((selection > 0) & (contact_area > 0)))
        ring_contact = int(np.count_nonzero((selection > 0) & (sample_ring > 0)))
        if contact == 0 and ring_contact < MIN_SAMPLE_PIXELS:
            continue
        score = area + ring_contact * 8 + contact * 4
        if score > best_score:
            best_score = score
            best_selection = selection

    if best_selection is not None:
        return best_selection
    return sample_ring


def background_sample_from_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape[:2]
    sample_mask = np.zeros((height, width), dtype=np.uint8)
    for background in iter_background_samples_from_mask(img, mask):
        sample_mask = cv2.bitwise_or(sample_mask, background)
    return sample_mask


def iter_background_samples_from_mask(img: np.ndarray, mask: np.ndarray):
    color_img = img[:, :, :3].copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    height, width = mask.shape[:2]
    text_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    boxes = _merge_boxes(_component_boxes(text_mask), GROUP_MERGE_PX, width, height)
    repair_kernel = _kernel(REPAIR_EXPAND_PX)
    ring_kernel = _kernel(SAMPLE_RING_PX)

    for box in boxes:
        local_text = _mask_for_box(text_mask, box)
        if not np.any(local_text):
            continue
        repair_area, sample_ring = _sample_ring_for_local_text(text_mask, local_text, repair_kernel, ring_kernel)
        try:
            yield _background_wand_sample(color_img, text_mask, repair_area, sample_ring)
        except Exception:
            yield sample_ring


def _quality_from_sample(color_img: np.ndarray, sample_mask: np.ndarray, mode: str) -> SolidQuality:
    active = sample_mask > 0
    sample_pixels = int(np.count_nonzero(active))
    if sample_pixels < MIN_SAMPLE_PIXELS:
        return SolidQuality(False, -1.0, (0, 0, 0), 255, 0.0, 0.0, 255, 0.0, 255, sample_pixels, mode)

    samples = color_img[active]
    b_values = samples[:, 0]
    g_values = samples[:, 1]
    r_values = samples[:, 2]
    b_spread, b_peak_ratio, b_peak = _hist_channel(b_values)
    g_spread, g_peak_ratio, g_peak = _hist_channel(g_values)
    r_spread, r_peak_ratio, r_peak = _hist_channel(r_values)
    max_spread = int(max(b_spread, g_spread, r_spread))
    min_peak_ratio = float(min(b_peak_ratio, g_peak_ratio, r_peak_ratio))
    fill_bgr = (
        _dominant_channel(b_values),
        _dominant_channel(g_values),
        _dominant_channel(r_values),
    )
    deltas = np.max(
        np.abs(samples.astype(np.int16) - np.array(fill_bgr, dtype=np.int16)),
        axis=1,
    )
    close_ratio = float(np.count_nonzero(deltas <= SOLID_CLOSE_DELTA_MAX) / sample_pixels)
    p95_delta = int(np.percentile(deltas, 95))
    white_deltas = np.max(255 - samples.astype(np.int16), axis=1)
    white_close_ratio = float(np.count_nonzero(white_deltas <= WHITE_CLOSE_DELTA_MAX) / sample_pixels)
    white_p95_delta = int(np.percentile(white_deltas, 95))
    spread_limit = SOLID_P90_P10_MAX
    close_ratio_limit = SOLID_CLOSE_RATIO_MIN
    p95_delta_limit = SOLID_P95_DELTA_MAX
    white_close_ratio_limit = WHITE_CLOSE_RATIO_MIN
    white_peak_ratio_limit = WHITE_FULL_PEAK_RATIO_MIN
    if mode != 'full':
        spread_limit = DIRECTIONAL_SOLID_P90_P10_MAX
        close_ratio_limit = DIRECTIONAL_SOLID_CLOSE_RATIO_MIN
        p95_delta_limit = DIRECTIONAL_SOLID_P95_DELTA_MAX
        white_close_ratio_limit = DIRECTIONAL_WHITE_CLOSE_RATIO_MIN
        white_peak_ratio_limit = WHITE_PEAK_RATIO_MIN
    strict_solid = (
        sample_pixels >= (MIN_DIRECTIONAL_SAMPLE_PIXELS if mode != 'full' else MIN_SAMPLE_PIXELS)
        and max_spread <= spread_limit
        and min_peak_ratio >= SOLID_PEAK_RATIO_MIN
        and close_ratio >= close_ratio_limit
        and p95_delta <= p95_delta_limit
    )
    white_dominant = (
        sample_pixels >= (MIN_DIRECTIONAL_SAMPLE_PIXELS if mode != 'full' else MIN_SAMPLE_PIXELS)
        and b_peak >= WHITE_DOMINANT_MIN
        and g_peak >= WHITE_DOMINANT_MIN
        and r_peak >= WHITE_DOMINANT_MIN
        and min_peak_ratio >= white_peak_ratio_limit
        and white_close_ratio >= white_close_ratio_limit
        and white_p95_delta <= WHITE_P95_DELTA_MAX
    )
    is_solid = strict_solid or white_dominant
    score = (
        min_peak_ratio * 1000.0
        + close_ratio * 400.0
        + white_close_ratio * 120.0
        - max_spread * 4.0
        - p95_delta * 6.0
        + min(sample_pixels, 1000) * 0.001
    )
    if white_dominant:
        score += 50.0
    if strict_solid:
        score += 100.0
    return SolidQuality(
        is_solid,
        score,
        fill_bgr,
        max_spread,
        min_peak_ratio,
        close_ratio,
        p95_delta,
        white_close_ratio,
        white_p95_delta,
        sample_pixels,
        mode,
    )


def _best_quality(color_img: np.ndarray, repair_area: np.ndarray, sample_ring: np.ndarray) -> SolidQuality:
    full = _quality_from_sample(color_img, sample_ring, 'full')
    if full.is_solid:
        return full
    if (
        full.max_spread > DIRECTIONAL_FALLBACK_MAX_FULL_SPREAD
        and full.close_ratio < DIRECTIONAL_FALLBACK_MIN_FULL_CLOSE_RATIO
    ):
        return full

    directionals = []
    for direction in ('top', 'bottom', 'left', 'right'):
        directional = cv2.bitwise_and(sample_ring, _direction_mask(sample_ring.shape, repair_area, direction))
        directionals.append(_quality_from_sample(color_img, directional, direction))

    solid_directionals = [
        item for item in directionals
        if item.is_solid and item.sample_pixels >= MIN_DIRECTIONAL_SAMPLE_PIXELS
    ]
    if len(solid_directionals) < MIN_DIRECTIONAL_AGREEMENT_COUNT:
        return full
    fill_values = np.array([item.fill_bgr for item in solid_directionals], dtype=np.int16)
    fill_disagreement = int(np.max(fill_values.max(axis=0) - fill_values.min(axis=0)))
    if fill_disagreement > DIRECTIONAL_FILL_AGREEMENT_MAX:
        return full
    return max(solid_directionals, key=lambda item: item.score)


def _solid_overlay_from_mask(
    img: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if len(img.shape) == 2:
        color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        color_img = img[:, :, :3].copy()
    height, width = mask.shape[:2]
    text_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    boxes = _merge_boxes(_component_boxes(text_mask), GROUP_MERGE_PX, width, height)
    repair_kernel = _kernel(REPAIR_EXPAND_PX)
    ring_kernel = _kernel(SAMPLE_RING_PX)
    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    other_mask = np.zeros((height, width), dtype=np.uint8)
    background_sample_mask = np.zeros((height, width), dtype=np.uint8)
    debug_blocks = []

    for box in boxes:
        local_text = _mask_for_box(text_mask, box)
        if not np.any(local_text):
            continue
        repair_area, sample_ring = _sample_ring_for_local_text(text_mask, local_text, repair_kernel, ring_kernel)
        background_sample = _background_wand_sample(color_img, text_mask, repair_area, sample_ring)
        background_sample_mask = cv2.bitwise_or(background_sample_mask, background_sample)
        quality = _best_quality(color_img, repair_area, background_sample)

        if quality.is_solid:
            active = repair_area > 0
            overlay[active, 0] = quality.fill_bgr[0]
            overlay[active, 1] = quality.fill_bgr[1]
            overlay[active, 2] = quality.fill_bgr[2]
            overlay[active, 3] = 255
        else:
            other_mask = cv2.bitwise_or(other_mask, repair_area)

        block_debug = asdict(quality)
        block_debug['box'] = [int(value) for value in box]
        debug_blocks.append(block_debug)

    summary = {
        'blocks': len(boxes),
        'auto_blocks': sum(1 for item in debug_blocks if item['is_solid']),
        'other_blocks': sum(1 for item in debug_blocks if not item['is_solid']),
        'other_pixels': int(np.count_nonzero(other_mask)),
        'blocks_debug': debug_blocks,
    }
    return overlay, other_mask, background_sample_mask, summary


def _bgr_to_pil_rgb(img: np.ndarray) -> Image.Image:
    if len(img.shape) == 2:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    return Image.fromarray(cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB))


def _mask_to_pil_rgb(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))


def _compose_overlay_preview(base_img: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    if len(base_img.shape) == 2:
        base = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    else:
        base = base_img[:, :, :3].copy()
    if overlay is None or len(overlay.shape) != 3 or overlay.shape[2] < 4:
        return base
    alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
    return (
        base.astype(np.float32) * (1.0 - alpha)
        + overlay[:, :, :3].astype(np.float32) * alpha
    ).astype(np.uint8)


def _fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    fitted = image.copy()
    fitted.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (width, height), 'white')
    x = (width - fitted.width) // 2
    y = (height - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = box[0] + (box[2] - box[0] - text_w) // 2
    y = box[1] + (box[3] - box[1] - text_h) // 2
    draw.text((x, y), text, font=font, fill=fill)


def _preview_page(
    img_path: str,
    paths: dict[str, str],
    page_summary: dict,
    page_width: int = PREVIEW_PAGE_WIDTH,
) -> Image.Image | None:
    base_img = imread(img_path, cv2.IMREAD_UNCHANGED)
    overlay = imread(_output_path(paths, img_path), cv2.IMREAD_UNCHANGED)
    mask = imread(_mask_path(paths, img_path), cv2.IMREAD_GRAYSCALE)
    other_mask = imread(_other_mask_path(paths, img_path), cv2.IMREAD_GRAYSCALE)
    if base_img is None or overlay is None or mask is None or other_mask is None:
        return None

    column_width = (page_width - PREVIEW_MARGIN * 2 - PREVIEW_GAP * 3) // 4
    image_area_height = int(column_width * base_img.shape[0] / max(1, base_img.shape[1]))
    page_height = PREVIEW_MARGIN * 2 + PREVIEW_HEADER_HEIGHT + PREVIEW_LABEL_HEIGHT + image_area_height
    page = Image.new('RGB', (page_width, page_height), 'white')
    draw = ImageDraw.Draw(page)
    font = ImageFont.load_default()

    title = (
        f'{osp.basename(img_path)}    '
        f'blocks={page_summary.get("blocks", 0)}    '
        f'auto={page_summary.get("auto_blocks", 0)}    '
        f'other={page_summary.get("other_blocks", 0)}    '
        f'other_pixels={page_summary.get("other_pixels", 0)}'
    )
    draw.text((PREVIEW_MARGIN, PREVIEW_MARGIN), title, font=font, fill=(0, 0, 0))
    draw.line(
        (PREVIEW_MARGIN, PREVIEW_MARGIN + PREVIEW_HEADER_HEIGHT - 18, page_width - PREVIEW_MARGIN, PREVIEW_MARGIN + PREVIEW_HEADER_HEIGHT - 18),
        fill=(210, 210, 210),
        width=2,
    )

    preview = _compose_overlay_preview(base_img, overlay)
    panels = [
        ('original', _bgr_to_pil_rgb(base_img)),
        ('preview', _bgr_to_pil_rgb(preview)),
        ('mask', _mask_to_pil_rgb(mask)),
        ('other_mask', _mask_to_pil_rgb(other_mask)),
    ]
    top = PREVIEW_MARGIN + PREVIEW_HEADER_HEIGHT
    for index, (label, panel) in enumerate(panels):
        x = PREVIEW_MARGIN + index * (column_width + PREVIEW_GAP)
        label_box = (x, top, x + column_width, top + PREVIEW_LABEL_HEIGHT)
        _draw_centered_text(draw, label_box, label, font)
        fitted = _fit_image(panel, column_width, image_area_height)
        page.paste(fitted, (x, top + PREVIEW_LABEL_HEIGHT))
        draw.rectangle(
            (x, top + PREVIEW_LABEL_HEIGHT, x + column_width, top + PREVIEW_LABEL_HEIGHT + image_area_height),
            outline=(210, 210, 210),
            width=1,
        )
    return page


def _write_preview_pdf(
    imglist: list[str],
    paths: dict[str, str],
    report: dict,
) -> str | None:
    pages: list[Image.Image] = []
    for img_path in tqdm(imglist, desc='preview pdf'):
        page_summary = report.get('pages', {}).get(osp.basename(img_path), {})
        if 'error' in page_summary:
            continue
        page = _preview_page(img_path, paths, page_summary)
        if page is not None:
            pages.append(page)
    if not pages:
        return None

    pdf_path = osp.join(paths['output'], PREVIEW_PDF)
    first, rest = pages[0], pages[1:]
    first.save(pdf_path, 'PDF', resolution=150.0, save_all=True, append_images=rest)
    return pdf_path


def image_files_in_folder(img_dir: str) -> list[str]:
    imglist = find_all_imgs(img_dir, abs_path=True)
    imglist = [path for path in imglist if not osp.basename(path).startswith('mask-')]
    imglist.sort(key=lambda path: osp.basename(path).lower())
    return imglist


def create_detector() -> TextDetector:
    model_path = osp.abspath(str(MODEL_PATH))
    if not osp.isfile(model_path):
        raise FileNotFoundError(f'找不到模型檔：{model_path}')
    return TextDetector(model_path=model_path, input_size=1024, device='cpu', act='leaky')


def process_image_with_detector(
    img_path: str,
    paths: dict[str, str],
    detector: TextDetector,
) -> dict:
    img = imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f'無法讀取原圖：{img_path}')
    detect_img = img[:, :, :3] if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    _, mask_refined, _ = detector(
        detect_img,
        refine_mode=REFINEMASK_ANNOTATION,
        keep_undetected_mask=True,
    )
    mask_refined = np.where(mask_refined > 0, 255, 0).astype(np.uint8)
    return regenerate_image_from_mask(img_path, paths, mask_refined)


def regenerate_image_from_mask(
    img_path: str,
    paths: dict[str, str],
    mask: np.ndarray | None = None,
) -> dict:
    img = imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f'無法讀取原圖：{img_path}')
    detect_img = img[:, :, :3] if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if mask is None:
        mask_path = _mask_path(paths, img_path)
        if not osp.isfile(mask_path):
            raise FileNotFoundError(f'找不到 mask：{mask_path}')
        mask = imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f'無法讀取 mask：{mask_path}')
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)

    overlay, other_mask, background_sample, page_summary = _solid_overlay_from_mask(detect_img, mask)
    imwrite(_mask_path(paths, img_path), mask)
    imwrite(_other_mask_path(paths, img_path), other_mask)
    imwrite(_output_path(paths, img_path), overlay)
    _save_background_sample_cache(paths, img_path, _mask_hash(mask), background_sample)
    return page_summary


def regenerate_image_from_ysgyolo_mask(
    img_path: str,
    paths: dict[str, str],
    ysgyolo_dir: str,
) -> dict:
    mask_path = _mask_path(paths, img_path)
    if not osp.isfile(mask_path):
        raise FileNotFoundError(f'找不到 mask：{mask_path}')
    mask = imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f'無法讀取 mask：{mask_path}')

    ysgyolo_path = osp.join(ysgyolo_dir, f'{Path(img_path).stem}.png')
    if not osp.isfile(ysgyolo_path):
        return regenerate_image_from_mask(img_path, paths, mask)
    ysgyolo_mask = imread(ysgyolo_path, cv2.IMREAD_GRAYSCALE)
    if ysgyolo_mask is None:
        raise FileNotFoundError(f'無法讀取 ysgyolo mask：{ysgyolo_path}')
    if ysgyolo_mask.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f'ysgyolo mask 尺寸不一致：{ysgyolo_path} '
            f'{ysgyolo_mask.shape[1]}x{ysgyolo_mask.shape[0]}，'
            f'原 mask {mask.shape[1]}x{mask.shape[0]}'
        )

    mask_bin = np.where(mask > 0, 255, 0).astype(np.uint8)
    ysgyolo_bin = np.where(ysgyolo_mask > 0, 255, 0).astype(np.uint8)
    merged_mask = cv2.bitwise_and(mask_bin, ysgyolo_bin)
    return regenerate_image_from_mask(img_path, paths, merged_mask)


def build_report(
    img_dir: str,
    paths: dict[str, str],
    imglist: list[str],
    pages: dict,
) -> dict:
    summary = {
        'total': len(imglist),
        'processed': 0,
        'with_other_mask': 0,
        'failed': 0,
    }
    for info in pages.values():
        if 'error' in info:
            summary['failed'] += 1
            continue
        summary['processed'] += 1
        if int(info.get('other_pixels', 0)) > 0:
            summary['with_other_mask'] += 1
    return {
        'image_dir': osp.abspath(img_dir),
        'output_dir': paths['output'],
        'model': osp.abspath(str(MODEL_PATH)),
        'device': 'cpu',
        'pages': pages,
        'summary': summary,
    }


def write_report(paths: dict[str, str], report: dict) -> str:
    report_path = osp.join(paths['output'], REPORT_JSON)
    with open(report_path, 'w', encoding='utf8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report_path


def load_report(paths: dict[str, str]) -> dict:
    report_path = osp.join(paths['output'], REPORT_JSON)
    if not osp.isfile(report_path):
        return {}
    with open(report_path, 'r', encoding='utf8') as f:
        return json.load(f)


def run(img_dir: str) -> int:
    img_dir = osp.abspath(img_dir)
    if not osp.isdir(img_dir):
        raise FileNotFoundError(f'找不到資料夾：{img_dir}')
    model_path = osp.abspath(str(MODEL_PATH))
    if not osp.isfile(model_path):
        raise FileNotFoundError(f'找不到模型檔：{model_path}')
    device = 'cpu'

    paths = _ensure_dirs(img_dir)
    imglist = image_files_in_folder(img_dir)
    if not imglist:
        print(f'資料夾內沒有可處理的圖片：{img_dir}')
        return 0

    print(f'資料夾：{img_dir}')
    print(f'輸出：{paths["output"]}')
    print(f'模型：{model_path}')
    print(f'裝置：{device}')
    print(f'圖片數量：{len(imglist)}')

    detector = create_detector()
    pages = {}

    for img_path in tqdm(imglist, desc='solid inpaint'):
        img_name = osp.basename(img_path)
        try:
            pages[img_name] = process_image_with_detector(img_path, paths, detector)
        except Exception as exc:
            pages[img_name] = {'error': str(exc)}

    report = build_report(img_dir, paths, imglist, pages)
    report_path = write_report(paths, report)
    preview_pdf_path = _write_preview_pdf(imglist, paths, report)

    print('完成。輸出：')
    print(f'  - {paths["mask"]}/<檔名>.png')
    print(f'  - {paths["other_mask"]}/<檔名>.png')
    print(f'  - {paths["inpainted"]}/<檔名>.png')
    print(f'  - {report_path}')
    if preview_pdf_path is not None:
        print(f'  - {preview_pdf_path}')
    print(f'含 other_mask 頁數：{report["summary"]["with_other_mask"]}')
    if report['summary']['failed']:
        print(f'失敗頁數：{report["summary"]["failed"]}')
    return int(report['summary']['processed'])


def main() -> None:
    parser = argparse.ArgumentParser(
        description='偵測文字 mask，生成純色背景 inpainted overlay 和 other_mask。',
    )
    parser.add_argument('img_dir', help='輸入圖片資料夾路徑')
    args = parser.parse_args()
    run(args.img_dir)


if __name__ == '__main__':
    main()
