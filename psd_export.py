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
from lama_config import MODEL_NAME, MASK_EXPAND
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


def _binary_other(other_mask):
    """Apply the workflow's >0.5 threshold to supported image-mask encodings."""
    mask = np.asarray(other_mask)
    if mask.dtype == np.bool_:
        return mask.copy()
    if mask.dtype == np.uint8:
        return mask >= 128
    if mask.dtype == np.uint16:
        return mask >= 32768
    if np.issubdtype(mask.dtype, np.floating):
        if not np.isfinite(mask).all() or np.any(mask < 0) or np.any(mask > 1):
            raise ValueError('LaMa 浮點 Mask 必須為有限的 0 到 1 數值。')
        return mask > 0.5
    raise ValueError('LaMa Mask 必須為 uint8、uint16、bool 或 0 到 1 浮點格式。')


def prepare_repair_inputs(background_rgb, overlay_bgra, other_mask, edited=None):
    """Build the workflow base and its expanded, edit-protected repair mask.

    OTHER is a saved binary classification. GrowMask(8, tapered_corners=True)
    uses eight iterations of a 3x3 cross, not an elliptical kernel. Expansion is
    only a derived export selection; solid pixels and manually erased decisions
    remain protected, and none of the supplied arrays are changed.
    """
    background = np.asarray(background_rgb)
    other = np.asarray(other_mask)
    if (background.ndim != 3 or background.shape[2] not in (3, 4)
            or background.dtype not in (np.uint8, np.uint16)
            or other.shape != background.shape[:2]):
        raise ValueError('LaMa 原圖或 Mask 尺寸、格式不符。')
    shape = other.shape
    overlay = (np.zeros((*shape, 4), np.uint8) if overlay_bgra is None
               else np.asarray(overlay_bgra))
    if overlay.shape != (*shape, 4) or overlay.dtype not in (np.uint8, np.uint16):
        raise ValueError('LaMa 填色素材尺寸或格式不符。')
    if edited is not None and np.asarray(edited).shape != shape:
        raise ValueError('LaMa 人工修改保護範圍尺寸不符。')
    active = _binary_other(other)
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    expanded = cv2.dilate(active.astype(np.uint8), cross, iterations=MASK_EXPAND) > 0
    expanded[overlay[:, :, 3] > 0] = False
    if edited is not None:
        expanded[(np.asarray(edited) > 0) & ~active] = False
    depth = 16 if background.dtype == np.uint16 else 8
    colors = _at_depth(overlay[:, :, [2, 1, 0]], depth)
    composite = background.copy()
    if background.dtype == np.uint8 and overlay.dtype == np.uint8:
        # Match export_pair's compositor exactly: opaque replacement and a
        # float32 blend truncated to uint8 for partially transparent pixels.
        opacity = overlay[:, :, 3]
        opaque = opacity == 255
        composite[opaque, :3] = colors[opaque]
        partial = (opacity > 0) & ~opaque
        if np.any(partial):
            alpha = opacity[partial, None].astype(np.float32) / 255.0
            composite[partial, :3] = (
                background[partial, :3].astype(np.float32) * (1.0 - alpha)
                + colors[partial].astype(np.float32) * alpha).astype(np.uint8)
    else:
        alpha = overlay[:, :, 3:4].astype(np.float64) / np.iinfo(overlay.dtype).max
        composite[:, :, :3] = np.rint(background[:, :, :3].astype(np.float64) * (1 - alpha)
                                     + colors.astype(np.float64) * alpha).astype(background.dtype)
    return composite, expanded.astype(np.uint8) * 255


def _validate_repair(repair, mask):
    if repair.shape != (*mask.shape, 4) or repair.dtype != np.uint8:
        raise ValueError('LaMa 回傳的透明圖層格式不符。')
    if np.any(repair[mask <= 0]) or not np.all(repair[:, :, 3][mask > 0] == 255):
        raise ValueError('LaMa 圖層透明範圍與實際修補 Mask 不一致。')


def _write_png_atomic(destination, pixels):
    success, encoded = cv2.imencode('.png', pixels)
    if not success:
        raise OSError('無法編碼 LaMa PNG。')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{destination.stem}-', suffix='.png.tmp',
                                     dir=destination.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(encoded.tobytes())
            stream.flush()
            os.fsync(stream.fileno())
        if not np.array_equal(imread(temporary, cv2.IMREAD_UNCHANGED), pixels):
            raise ValueError('LaMa PNG 顏色或透明度回讀不符。')
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return str(destination)


def export_repair_png(source_path, other_mask, destination, *, overlay_bgra=None,
                      edited=None, mask_destination=None) -> str | None:
    """Generate and atomically save one original-size RGBA LaMa repair PNG.

    This helper does not import psd-tools or write source materials/project data.
    Its PNG and optional actual-mask PNG can be passed to export_page_psd.
    Empty effective selection returns None without inference or creating a
    repair-image directory; a requested mask PNG is still saved. Only an old,
    matching, entirely zero BGRA repair PNG is removed in that case.
    """
    background, _ = _source(source_path)
    context, mask = prepare_repair_inputs(background, overlay_bgra, other_mask, edited)
    if mask_destination is not None:
        _write_png_atomic(mask_destination, mask)
    if not np.any(mask > 0):
        existing = _existing(destination)
        if existing is not None and existing.is_file():
            try:
                old = imread(existing, cv2.IMREAD_UNCHANGED)
            except (OSError, ValueError, cv2.error):
                old = None
            if old is not None and old.shape == (*mask.shape, 4) and not np.any(old):
                existing.unlink()
        return None
    repair = inpaint_other(context, mask)
    _validate_repair(repair, mask)
    bgra = cv2.cvtColor(repair, cv2.COLOR_RGBA2BGRA)
    return _write_png_atomic(destination, bgra)


def export_page_psd(source_path, overlay_bgra, other_mask, destination, *,
                    overwrite=False, lama_visible=True, repair_rgba=None, repair_path=None,
                    edited=None, repair_mask=None):
    """Create one PSD; return False when preserving an existing output.

    Input overlay uses the project's BGRA convention. LaMa receives RGB pixels,
    and its generated layer is restricted to the expanded, protected repair mask.
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
    context, allowed_mask = prepare_repair_inputs(background, overlay, mask, edited)
    if repair_mask is None:
        mask = allowed_mask
    else:
        supplied = np.asarray(repair_mask)
        required = _binary_other(mask) & (overlay[:, :, 3] == 0)
        if (supplied.shape != mask.shape or supplied.dtype != np.uint8
                or not np.all((supplied == 0) | (supplied == 255))):
            raise ValueError('實際修補 Mask 尺寸或格式不符。')
        if np.any((supplied > 0) & (allowed_mask == 0)) or np.any(required & (supplied == 0)):
            raise ValueError('實際修補 Mask 超出允許範圍或缺少原 OTHER。')
        mask = supplied
    depth = 16 if background.dtype == np.uint16 else 8
    repair = None
    if np.any(mask > 0):
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
            repair = inpaint_other(context, mask)
        _validate_repair(repair, mask)
    layers = [('bg', background, True)]
    if np.any(overlay[:, :, 3]):
        layers.append(('overlay-manual', _at_depth(overlay[:, :, [2, 1, 0, 3]], depth), True))
    if repair is not None:
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


def export_psds_from_assets(images, assets_dir, *, overwrite=False, lama_visible=True, progress=None,
                            edited_loader=None):
    """Generate all repair PNGs first, then build PSDs from those exact pixels.

    Existing solid/other_mask PNGs and project raw are read-only. Explicit export
    updates repair PNGs even when preserving pre-existing, hand-edited PSDs.
    Each actual repair mask is saved before inference and read again by stage
    two. A matching raw project or edited_loader preserves manual erasures;
    assets without either can protect solid pixels but cannot recover erasures.
    progress(stage, page, pages, filename, completed_units, total_units) reports
    page starts and completions across the two real processing stages.
    """
    images = validate_stems(images)
    root = Path(assets_dir)
    if edited_loader is None and (root.parent / 'raw' / 'project.json').is_file():
        project_folder = root.parent.parent.resolve()
        if all(source.parent.resolve() == project_folder for source in images):
            import project_store as store
            raw_paths = {'raw': str(root.parent / 'raw')}
            edited_loader = lambda source, shape: store.load_page(raw_paths, source, shape)['edited']
    result = {'assets_dir': str(root), 'psds_dir': str(root / 'psds'),
              'repair_dir': str(root / MODEL_NAME), 'repair_mask_dir': str(root / 'repair_mask'),
              'repair_pngs': [], 'created': [], 'skipped': []}
    total = len(images)
    for index, source in enumerate(images, 1):
        if progress:
            progress('LaMa 修補', index, total, source.name, index - 1, total * 2)
        other = imread(root / 'other_mask' / f'{source.stem}.png', cv2.IMREAD_GRAYSCALE)
        solid_path = root / 'solid' / f'{source.stem}.png'
        solid = imread(solid_path, cv2.IMREAD_UNCHANGED) if solid_path.exists() else None
        edited = edited_loader(source, other.shape) if edited_loader is not None else None
        repair_path = root / MODEL_NAME / f'{source.stem}.png'
        exported = export_repair_png(source, other, repair_path, overlay_bgra=solid, edited=edited,
                                     mask_destination=root / 'repair_mask' / f'{source.stem}.png')
        if exported is not None:
            result['repair_pngs'].append(exported)
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
            actual_mask = imread(root / 'repair_mask' / f'{source.stem}.png', cv2.IMREAD_GRAYSCALE)
            repair_options = ({'repair_path': root / MODEL_NAME / f'{source.stem}.png'}
                              if np.any(actual_mask > 0) else {})
            created = export_page_psd(source, solid, other, destination,
                                      overwrite=overwrite, lama_visible=lama_visible,
                                      repair_mask=actual_mask,
                                      **repair_options)
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
    return export_psds_from_assets(images, root, overwrite=overwrite, progress=progress,
        edited_loader=lambda source, shape: store.load_page(paths, source, shape)['edited'])
