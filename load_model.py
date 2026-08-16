"""RF-DETR 模型載入工具（koharu-layout-rfdetr-seg-2xl-1152）。

載入邏輯對應 HANDOFF_RFDETR.md 中的官方 loader：

    RFDETRSeg2XLarge(pretrain_weights=None, resolution=1152,
                     num_select=160, num_classes=4)
    model.model.model.load_state_dict(load_file(...), strict=True)
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from safetensors.torch import load_file


MODEL_DIR = Path(__file__).parent / 'models' / 'koharu-layout-rfdetr-seg-2xl-1152'
MODEL_FILE = MODEL_DIR / 'model.safetensors'
MODEL_SHA256 = '9bf6d2cbd7793c956d8c857bb1672a396eb7f100eb0682f86830d05e31168efb'

CLASS_NAMES = ('text', 'onomatopoeia', 'bubble', 'panel')
CLASS_THRESHOLDS = {'text': 0.25, 'onomatopoeia': 0.20, 'bubble': 0.50, 'panel': 0.50}
RESOLUTION = 1152
NUM_SELECT = 160


def verify_model_sha256(model_path: Path, expected: str = MODEL_SHA256) -> bool:
    """回傳權重檔 SHA-256 是否與預期一致；不一致時拋出例外。"""
    digest = hashlib.sha256()
    with model_path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(
            f'SHA-256 不符：expect {expected}，actual {actual}（{model_path}）'
        )
    return True


def load_rfdetr_model(
    model_path: str | Path = MODEL_FILE,
    *,
    device: str = 'cpu',
    verify: bool = True,
):
    """載入 RF-DETR 分割模型並回傳 RFDETRSeg2XLarge 實例。

    device 可為 'cpu' / 'mps' / 'cuda'；權重一律先載入 CPU，
    由 predict() 第一次呼叫時自動搬到目標裝置。
    """
    from rfdetr import RFDETRSeg2XLarge

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f'找不到 RF-DETR 權重檔：{path}')
    if verify:
        verify_model_sha256(path)

    model = RFDETRSeg2XLarge(
        pretrain_weights=None,
        resolution=RESOLUTION,
        num_select=NUM_SELECT,
        num_classes=len(CLASS_NAMES),
    )
    model.model.model.load_state_dict(load_file(str(path), device='cpu'), strict=True)
    model.model.device = device
    return model
