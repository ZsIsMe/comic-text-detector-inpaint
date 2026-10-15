"""MangaLens bubble geometry for expanding independently verified solid text.

The polygon/2% inset approach follows manga-translator-ui's ballon_fill.py and
mangalens_detector.py (GPL-3.0; see vendor/LICENSE-BallonsTranslator).
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import cv2
import numpy as np
import project_store as store

MODEL_PATH = Path(__file__).resolve().parent / 'models' / 'mangalens.pt'
MODEL_URL = 'https://www.modelscope.cn/models/hgmzhn/manga-translator-ui/resolve/master/mangalens.pt'
MODEL_SHA256 = '4028152940f7c910f40192f46ede3b3f6c7129e5c76849c324d3564f8ac50198'
CACHE_VERSION = 1
_model = None
_model_signature = None
_model_lock = threading.Lock()


def load_settings(raw_dir: str | Path) -> dict:
    defaults = {'enabled': True, 'shrink_percent': 2.0}
    try:
        data = store.load_project(raw_dir).get('settings', {}).get('solid_fill', {})
        enabled = data.get('enabled', True)
        defaults['enabled'] = enabled if isinstance(enabled, bool) else True
        percent = float(data.get('shrink_percent', 2.0))
        if np.isfinite(percent):
            defaults['shrink_percent'] = max(0.0, min(percent, 10.0))
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return defaults


def save_settings(raw_dir: str | Path, enabled: bool, shrink_percent: float) -> None:
    if not np.isfinite(shrink_percent) or not 0 <= shrink_percent <= 10:
        raise ValueError('氣泡內縮比例必須介於 0–10%。')
    settings = store.load_project(raw_dir).get('settings', {})
    settings['solid_fill'] = {'enabled': bool(enabled), 'shrink_percent': float(shrink_percent)}
    store.update_project(raw_dir, settings=settings)


def _tiles(shape):
    """Overlap long-page windows so small bubbles are not lost in a page resize."""
    h, w = shape[:2]
    vertical = h >= w
    long_side, short_side = max(h, w), min(h, w)
    if long_side <= short_side * 2.5:
        yield 0, 0, w, h
        return
    length = min(long_side, short_side * 2)
    step = max(1, length - short_side // 2)
    starts = list(range(0, long_side - length + 1, step))
    if starts[-1] != long_side - length:
        starts.append(long_side - length)
    for start in starts:
        yield (0, start, w, start + length) if vertical else (start, 0, start + length, h)


def detect_bubbles(image: np.ndarray, cache_path: Path) -> tuple[list[np.ndarray], str]:
    """Cache per-instance polygons by source pixels and model version, never text mask.

    No detection-box fallback: a box is not a safe whole-bubble fill boundary.
    Inference prefers CUDA, then MPS, and is serialized because workers share the model.
    """
    global _model, _model_signature
    stat = MODEL_PATH.stat()
    signature = (str(MODEL_PATH), stat.st_size, stat.st_mtime_ns)
    digest = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
    key = [CACHE_VERSION, list(image.shape), digest, list(signature)]
    try:
        cached = json.loads(str(store.read_cache_file(cache_path)['bubble_json']))
        if cached['key'] == key:
            polygons = [np.asarray(p, dtype=np.float32) for p in cached['polygons']]
            if all(p.ndim == 2 and p.shape[1] == 2 and len(p) >= 3
                   and np.isfinite(p).all() for p in polygons):
                return polygons, 'cache'
    except (OSError, ValueError, KeyError, TypeError):
        pass
    polygons = []
    with _model_lock:
        import torch

        if torch.cuda.is_available():
            device = 'cuda:0'
        elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        if _model is None or _model_signature != signature:
            from ultralytics import YOLO
            _model = YOLO(str(MODEL_PATH), task='segment')
            _model_signature = signature
        for x1, y1, x2, y2 in _tiles(image.shape):
            crop = np.ascontiguousarray(image[y1:y2, x1:x2])
            result = _model.predict(crop, imgsz=1600, conf=0.25, iou=0.7,
                                    device=device, retina_masks=True, verbose=False)[0]
            if result.masks is None:
                continue
            for polygon, class_id in zip(result.masks.xy, result.boxes.cls.cpu().numpy()):
                if result.names[int(class_id)] != 'balloon':
                    continue
                pts = np.asarray(polygon, dtype=np.float32)
                if len(pts) < 3 or not np.isfinite(pts).all():
                    continue
                # Reject bubbles cut by internal tile seams; overlapping windows
                # can contribute the complete instance instead.
                if ((x1 > 0 and pts[:, 0].min() <= 2)
                        or (y1 > 0 and pts[:, 1].min() <= 2)
                        or (x2 < image.shape[1] and pts[:, 0].max() >= x2-x1-3)
                        or (y2 < image.shape[0] and pts[:, 1].max() >= y2-y1-3)):
                    continue
                polygons.append(pts + np.array([x1, y1], dtype=np.float32))
    polygons = _deduplicate(polygons, image.shape[:2])
    store.update_cache_file(cache_path, bubble_json=np.array(json.dumps(
        {'key': key, 'polygons': [p.tolist() for p in polygons]})))
    return polygons, 'detected'


def _polygon_mask(polygon, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    points = np.rint(polygon).astype(np.int32)
    points[:, 0] = np.clip(points[:, 0], 0, shape[1]-1)
    points[:, 1] = np.clip(points[:, 1], 0, shape[0]-1)
    cv2.fillPoly(mask, [points], 255)
    return mask


def _deduplicate(polygons, shape):
    kept = []
    # Largest complete instance wins when overlapping long-page windows agree.
    for p in sorted(polygons, key=lambda a: abs(cv2.contourArea(a)), reverse=True):
        x, y, w, h = cv2.boundingRect(p)
        duplicate = False
        for q in kept:
            qx, qy, qw, qh = cv2.boundingRect(q)
            if min(x+w, qx+qw) <= max(x, qx) or min(y+h, qy+qh) <= max(y, qy):
                continue
            ox, oy = min(x, qx), min(y, qy)
            roi_shape = (max(y+h, qy+qh)-oy, max(x+w, qx+qw)-ox)
            a = _polygon_mask(p - [ox, oy], roi_shape) > 0
            b = _polygon_mask(q - [ox, oy], roi_shape) > 0
            if np.count_nonzero(a & b) / max(1, np.count_nonzero(a | b)) > 0.7:
                duplicate = True
                break
        if not duplicate:
            kept.append(p)
    return kept


def spatial_spread(image: np.ndarray, sample: np.ndarray) -> tuple[float, int]:
    """Compare 4x4 spatial medians; histogram alone cannot reject gradients."""
    ys, xs = np.where(sample)
    if not len(xs):
        return 255.0, 0
    colors = []
    for yy in np.array_split(np.arange(ys.min(), ys.max()+1), 4):
        for xx in np.array_split(np.arange(xs.min(), xs.max()+1), 4):
            if not len(yy) or not len(xx):
                continue
            crop = sample[yy[0]:yy[-1]+1, xx[0]:xx[-1]+1]
            if np.count_nonzero(crop) >= 8:
                colors.append(np.median(image[yy[0]:yy[-1]+1, xx[0]:xx[-1]+1][crop], axis=0))
    if not colors:
        return 255.0, 0
    return float(np.ptp(np.asarray(colors), axis=0).max()), len(colors)


def fill_bubbles(text_mask: np.ndarray, solid_seeds: np.ndarray,
                 polygons: list[np.ndarray], shrink_ratio: float = 0.02,
                 protected: np.ndarray | None = None) -> tuple[np.ndarray, list[dict]]:
    """Expand verified solid text inside MangaLens polygons using its chosen BGR.

    solid_seeds contains only text pixels already classified as solid, with the
    local classifier's colour in BGRA. Bubble pixels are never sampled or
    classified again. Overlap seams and protected manual edits stay untouched.
    """
    shape = text_mask.shape
    if solid_seeds.shape != (*shape, 4):
        raise ValueError('Solid seed overlay must match the text mask shape.')
    protected_area = np.zeros(shape, bool) if protected is None else protected > 0
    seeds = (solid_seeds[:, :, 3] > 0) & (text_mask > 0) & ~protected_area
    _, text_labels, text_stats, _ = cv2.connectedComponentsWithStats(
        (text_mask > 0).astype(np.uint8), connectivity=8)
    regions = []
    for polygon in polygons:
        points = np.asarray(polygon, np.float32)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
            continue
        region = _polygon_mask(points, shape)
        if np.any(region):
            regions.append(region)
    # Sorting does not affect ownership, but keeps diagnostics stable when the
    # detector returns the same instances in a different order.
    regions.sort(key=lambda region: (cv2.boundingRect(region)[1], cv2.boundingRect(region)[0],
                                     -int(np.count_nonzero(region))))
    coverage = np.zeros(shape, np.uint16)
    for region in regions:
        coverage += (region > 0).astype(np.uint16)
    seam = coverage > 1
    seam_guard = cv2.dilate(seam.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    overlay = np.zeros((*shape, 4), np.uint8)
    records = []
    for region in regions:
        x, y, w, h = cv2.boundingRect(region)
        roi = np.s_[y:y+h, x:x+w]
        local_region = region[roi] > 0
        local_seed = seeds[roi] & local_region
        qualified = np.zeros((h, w), bool)
        labels, counts = np.unique(text_labels[roi][local_seed], return_counts=True)
        for label, count in zip(labels, counts):
            if label > 0:
                total = text_stats[label, cv2.CC_STAT_AREA]
                inside = np.count_nonzero((text_labels[roi] == label) & local_region)
                if inside >= total * 0.9:
                    qualified |= local_seed & (text_labels[roi] == label)
        seed_pixels = int(np.count_nonzero(qualified))
        inset = max(1, round(min(w, h) * shrink_ratio)) if shrink_ratio else 0
        record = {'box': [x, y, x+w, y+h], 'inset_px': inset,
                  'seed_pixels': seed_pixels,
                  'overlap_pixels': int(np.count_nonzero(seam[roi] & local_region)),
                  'accepted': False}
        if seed_pixels == 0:
            record['reason'] = 'no_solid_text_seed'
            records.append(record)
            continue
        safe = cv2.erode(region[roi],
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*inset+1,)*2),
                         borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0 if inset else local_region
        safe &= ~seam_guard[roi] & ~protected_area[roi]
        if not np.any(safe):
            record['reason'] = 'empty_inset'
            records.append(record)
            continue
        colors, counts = np.unique(solid_seeds[roi][qualified, :3], axis=0, return_counts=True)
        # np.unique sorts colours lexicographically; argmax makes ties stable.
        color_index = int(np.argmax(counts))
        color = colors[color_index]
        view = overlay[roi]
        view[safe, :3] = color
        view[safe, 3] = 255
        record.update(accepted=True, reason='solid_text_seed',
                      fill_bgr=color.astype(int).tolist(),
                      seed_color_pixels=int(counts[color_index]),
                      seed_colors=int(len(colors)),
                      filled_pixels=int(np.count_nonzero(safe)))
        records.append(record)
    return overlay, records
