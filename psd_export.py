"""Atomic layered PSD exports from saved pages or existing PNG materials.

The PSD dependency and LaMa weights are loaded only on an explicit export.
Existing PSDs are skipped by default to protect subsequent Photoshop edits.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
import tempfile
import unicodedata

import cv2
import numpy as np
from PIL import Image, ImageCms

from image_io import imread, imwrite
from lama_config import MODEL_NAME
from lama_inpainter import inpaint_other


def _api():
    try:
        from psd_tools import PSDImage
        from psd_tools.constants import Resource
        from psd_tools.psd.image_resources import ImageResource
    except ImportError as exc:
        raise ValueError('PSD 導出需要 psd-tools==1.26.1；請重新執行 bootstrap.py 安裝依賴。') from exc
    return PSDImage, Resource, ImageResource


def validate_stems(images):
    images = [Path(image) for image in images]
    keys = [unicodedata.normalize('NFC', image.stem).casefold() for image in images]
    if len(keys) != len(set(keys)):
        raise ValueError('原圖有去除副檔名後同名的檔案（不分大小寫），無法安全導出 PSD／PNG。')
    return images


def _existing(destination):
    destination = Path(destination)
    if destination.exists():
        return destination
    if destination.parent.exists():
        key = unicodedata.normalize('NFC', destination.name).casefold()
        for path in destination.parent.iterdir():
            if unicodedata.normalize('NFC', path.name).casefold() == key:
                return path
    return None


def _source(source_path):
    image = imread(source_path, cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype not in (np.uint8, np.uint16):
        raise ValueError(f'原圖格式無法導出 PSD：{source_path}')
    with Image.open(source_path) as original:
        profile = original.info.get('icc_profile')
        mode = original.mode
        cms = ImageCms.ImageCmsProfile(io.BytesIO(profile)) if profile else None
        space = cms.profile.xcolor_space.strip() if cms else None
        if mode == 'CMYK' or space == 'GRAY':
            if image.dtype == np.uint16:
                raise ValueError('16-bit 灰階 ICC 不能經 8-bit 色彩轉換導出；請先提供 RGB 色彩版本。')
            target = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB'))
            try:
                converted = (ImageCms.profileToProfile(original, cms, target, outputMode='RGB')
                             if cms else original.convert('RGB'))
            except ImageCms.PyCMSError as exc:
                raise ValueError(f'原圖 ICC 色彩轉換失敗：{source_path}：{exc}') from exc
            return np.asarray(converted).copy(), target.tobytes()
        if space not in (None, 'RGB'):
            raise ValueError(f'原圖 ICC 色彩空間 {space} 不能套用到 RGB PSD。')
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    elif image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError('PSD 原圖必須為灰階、BGR 或 BGRA。')
    return image, profile


def _at_depth(array, depth):
    if array.dtype == np.uint16:
        return array if depth == 16 else (array >> 8).astype(np.uint8)
    if array.dtype != np.uint8:
        raise ValueError('PSD 素材必須為 uint8 或 uint16。')
    return array.astype(np.uint16) * 257 if depth == 16 else array


def _verify(path, layers, size, depth, profile):
    PSDImage, Resource, _ = _api()
    psd = PSDImage.open(path)
    if (psd.size != size or psd.depth != depth or len(psd) != len(layers)
            or psd.color_mode.name != 'RGB' or psd.channels != 3):
        raise ValueError('PSD 回讀尺寸、深度、RGB 通道或圖層數不符。')
    maximum = (1 << depth) - 1
    for actual, (name, pixels, visible) in zip(psd, layers):
        if (actual.name != name or actual.visible != visible or actual.kind != 'pixel'
                or actual.bbox != (0, 0, *size) or actual.has_mask()):
            raise ValueError(f'PSD 圖層結構回讀不符：{name}')
        colors = np.rint(actual.numpy('color') * maximum).astype(pixels.dtype)
        alpha = np.rint(actual.numpy('shape')[:, :, 0] * maximum).astype(pixels.dtype)
        expected_alpha = (pixels[:, :, 3] if pixels.shape[2] == 4
                          else np.full(pixels.shape[:2], maximum, pixels.dtype))
        if not np.array_equal(colors, pixels[:, :, :3]) or not np.array_equal(alpha, expected_alpha):
            raise ValueError(f'PSD 圖層顏色或透明度回讀不符：{name}')
    if psd.image_resources.get_data(Resource.ICC_PROFILE) != profile:
        raise ValueError('PSD ICC 色彩描述檔回讀不符。')


def _validate_repair(repair, mask):
    if repair.shape != (*mask.shape, 4) or repair.dtype != np.uint8:
        raise ValueError('LaMa 回傳的透明圖層格式不符。')
    if np.any(repair[mask <= 0]) or not np.all(repair[:, :, 3][mask > 0] == 255):
        raise ValueError('LaMa 圖層透明範圍與 OTHER 不一致。')


def export_repair_png(source_path, other_mask, destination):
    """Generate and atomically save one original-size RGBA LaMa repair PNG.

    This helper does not import psd-tools or write source materials/project data.
    Its PNG can be passed directly to export_page_psd(repair_path=...).
    """
    background, _ = _source(source_path)
    mask = np.asarray(other_mask)
    if mask.shape != background.shape[:2]:
        raise ValueError(f'LaMa Mask 尺寸與原圖不一致：{source_path}')
    repair = inpaint_other(background, mask)
    _validate_repair(repair, mask)
    bgra = cv2.cvtColor(repair, cv2.COLOR_RGBA2BGRA)
    success, encoded = cv2.imencode('.png', bgra)
    if not success:
        raise OSError('無法編碼 LaMa 透明修補 PNG。')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{destination.stem}-', suffix='.png.tmp',
                                     dir=destination.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(encoded.tobytes())
            stream.flush()
            os.fsync(stream.fileno())
        actual = imread(temporary, cv2.IMREAD_UNCHANGED)
        if not np.array_equal(actual, bgra):
            raise ValueError('LaMa PNG 顏色或透明度回讀不符。')
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return str(destination)


def export_page_psd(source_path, overlay_bgra, other_mask, destination, *,
                    overwrite=False, lama_visible=True, repair_rgba=None, repair_path=None):
    """Create one PSD; return False when preserving an existing output.

    Input overlay uses the project's BGRA convention. LaMa receives RGB pixels,
    and its generated layer is restricted exactly to the original OTHER mask.
    """
    destination = Path(destination)
    existing = _existing(destination)
    if existing is not None:
        if not overwrite:
            return False
        destination = existing
    PSDImage, Resource, ImageResource = _api()
    background, profile = _source(source_path)
    height, width = background.shape[:2]
    overlay = (np.zeros((height, width, 4), np.uint8) if overlay_bgra is None
               else np.asarray(overlay_bgra))
    mask = np.asarray(other_mask)
    if overlay.shape != (height, width, 4) or mask.shape != (height, width):
        raise ValueError(f'PSD 素材尺寸與原圖不一致：{source_path}')
    depth = 16 if background.dtype == np.uint16 else 8
    if repair_rgba is not None and repair_path is not None:
        raise ValueError('repair_rgba 與 repair_path 只能指定一項。')
    if repair_path is not None:
        repair = imread(repair_path, cv2.IMREAD_UNCHANGED)
        if repair is None or repair.ndim != 3 or repair.shape[2] != 4:
            raise ValueError(f'LaMa PNG 必須是 RGBA：{repair_path}')
        repair = cv2.cvtColor(repair, cv2.COLOR_BGRA2RGBA)
    elif repair_rgba is not None:
        repair = np.asarray(repair_rgba)
    else:
        repair = inpaint_other(background, mask)
    _validate_repair(repair, mask)
    layers = [('bg', background, True)]
    if np.any(overlay[:, :, 3]):
        layers.append(('overlay-manual', _at_depth(overlay[:, :, [2, 1, 0, 3]], depth), True))
    layers.append((MODEL_NAME, _at_depth(repair, depth), bool(lama_visible)))
    # Three document channels avoid a separate composite "Alpha 1" channel.
    # RGBA layer arrays still retain their own native -1 transparency channels.
    psd = PSDImage.new('RGB', (width, height), depth=depth)
    for name, pixels, visible in layers:
        layer = psd.create_pixel_layer_from_array(pixels, name=name, top=0, left=0)
        layer.visible = visible
    if profile:
        psd.image_resources[Resource.ICC_PROFILE] = ImageResource(key=Resource.ICC_PROFILE, data=profile)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{destination.stem}-', suffix='.psd.tmp',
                                     dir=destination.parent)
    os.close(fd)
    try:
        psd.save(temporary)
        _verify(temporary, layers, (width, height), depth, profile)
        if not overwrite and _existing(destination) is not None:
            return False
        os.replace(temporary, destination)
    except Exception as exc:
        raise OSError(f'PSD 導出失敗：{destination}：{exc}') from exc
    finally:
        Path(temporary).unlink(missing_ok=True)
    return True


def export_psds_from_assets(images, assets_dir, *, overwrite=False, lama_visible=True, progress=None):
    """Generate all repair PNGs first, then build PSDs from those exact pixels.

    Existing solid/other_mask PNGs and project raw are read-only. Explicit export
    updates repair PNGs even when preserving pre-existing, hand-edited PSDs.
    progress(stage, page, pages, filename, completed_units, total_units) reports
    page starts and completions across the two real processing stages.
    """
    images = validate_stems(images)
    root = Path(assets_dir)
    result = {'assets_dir': str(root), 'psds_dir': str(root / 'psds'),
              'repair_dir': str(root / MODEL_NAME), 'repair_pngs': [], 'created': [], 'skipped': []}
    total = len(images)
    for index, source in enumerate(images, 1):
        if progress:
            progress('LaMa 修補', index, total, source.name, index - 1, total * 2)
        other = imread(root / 'other_mask' / f'{source.stem}.png', cv2.IMREAD_GRAYSCALE)
        repair_path = root / MODEL_NAME / f'{source.stem}.png'
        export_repair_png(source, other, repair_path)
        result['repair_pngs'].append(str(repair_path))
        if progress:
            progress('LaMa 修補', index, total, source.name, index, total * 2)
    for index, source in enumerate(images, 1):
        if progress:
            progress('PSD 組裝', index, total, source.name, total + index - 1, total * 2)
        destination = root / 'psds' / f'{source.stem}.psd'
        if not overwrite and _existing(destination) is not None:
            result['skipped'].append(str(destination))
        else:
            solid_path = root / 'solid' / f'{source.stem}.png'
            solid = imread(solid_path, cv2.IMREAD_UNCHANGED) if solid_path.exists() else None
            other = imread(root / 'other_mask' / f'{source.stem}.png', cv2.IMREAD_GRAYSCALE)
            created = export_page_psd(source, solid, other, destination,
                                      overwrite=overwrite, lama_visible=lama_visible,
                                      repair_path=root / MODEL_NAME / f'{source.stem}.png')
            result['created' if created else 'skipped'].append(str(destination))
        if progress:
            progress('PSD 組裝', index, total, source.name, total + index, total * 2)
    return result


def export_project_assets(images, paths, *, overwrite=False, progress=None):
    """Export saved PNG materials and layered PSDs; never classify or modify raw."""
    import project_store as store

    images = validate_stems(images)
    root = Path(paths['output']) / 'psd_assets'
    for name in ('solid', 'other_mask'):
        (root / name).mkdir(parents=True, exist_ok=True)
    for source in images:
        image = imread(source, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f'無法讀取原圖：{source}')
        page = store.load_page(paths, source, image.shape[:2])
        imwrite(root / 'solid' / f'{source.stem}.png', page['overlay'])
        imwrite(root / 'other_mask' / f'{source.stem}.png', page['other'])
    return export_psds_from_assets(images, root, overwrite=overwrite, progress=progress)
