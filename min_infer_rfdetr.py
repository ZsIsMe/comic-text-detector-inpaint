"""RF-DETR 最小推理驗證：CPU 與 MPS 各跑一次，輸出逐類別二值 mask。

用法：
    .venv/bin/python min_infer_rfdetr.py [圖片路徑] [輸出目錄]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from load_model import CLASS_NAMES, CLASS_THRESHOLDS, load_rfdetr_model


def _save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def run_device(model, image: np.ndarray, device: str, out_dir: Path) -> dict:
    model.model.device = device
    t0 = time.time()
    detections = model.predict(
        image,
        threshold=0.20,
        shape=(1152, 1152),
        include_source_image=False,
    )
    elapsed = time.time() - t0

    height, width = image.shape[:2]
    masks: dict[str, np.ndarray] = {
        name: np.zeros((height, width), dtype=bool) for name in CLASS_NAMES
    }
    counts: dict[str, int] = {name: 0 for name in CLASS_NAMES}
    scores: dict[str, list[float]] = {name: [] for name in CLASS_NAMES}

    class_names = detections.data.get('class_name', [None] * len(detections))
    for class_name, class_id, confidence, instance_mask in zip(
        class_names,
        detections.class_id,
        detections.confidence,
        detections.mask,
    ):
        name = str(class_name) if class_name is not None else CLASS_NAMES[int(class_id)]
        if name not in masks:
            name = CLASS_NAMES[int(class_id)] if int(class_id) < len(CLASS_NAMES) else 'text'
        threshold = CLASS_THRESHOLDS[name]
        if float(confidence) < threshold:
            continue
        counts[name] += 1
        scores[name].append(float(confidence))
        masks[name] |= np.asarray(instance_mask, dtype=bool)

    for name in CLASS_NAMES:
        _save_mask(masks[name], out_dir / f'mask_{device}_{name}.png')

    return {
        'device': device,
        'elapsed': elapsed,
        'counts': counts,
        'score_ranges': {
            name: (min(scores[name]), max(scores[name])) if scores[name] else None
            for name in CLASS_NAMES
        },
    }


def main() -> None:
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/Users/zhongsheng/Downloads/87/14.jpg')
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('/tmp/rfdetr-infer-out')
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch

    print(f'torch {torch.__version__} | mps_available={torch.backends.mps.is_available()}')
    image = np.asarray(Image.open(image_path).convert('RGB'))
    print(f'image {image_path} | shape={image.shape}')

    model = load_rfdetr_model(verify=True)
    print(f'weights loaded: {model.model.model.__class__.__name__}')

    for device in ('cpu', 'mps'):
        if device == 'mps' and not torch.backends.mps.is_available():
            print('MPS 不可用，跳過')
            continue
        try:
            result = run_device(model, image, device, out_dir)
        except Exception as exc:  # noqa: BLE001 - 測試腳本需要完整錯誤資訊
            print(f'[{device}] FAILED: {type(exc).__name__}: {exc}')
            continue
        print(
            f"[{result['device']}] elapsed={result['elapsed']:.2f}s "
            f"counts={result['counts']} ranges={result['score_ranges']}"
        )


if __name__ == '__main__':
    main()
