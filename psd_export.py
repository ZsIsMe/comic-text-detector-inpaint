"""Atomic layered PSD exports from saved pages or existing PNG materials.

The PSD dependency and LaMa weights are loaded only on an explicit export.
Existing PSDs are skipped by default to protect subsequent Photoshop edits.
"""

from __future__ import annotations

import io
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
import unicodedata
import uuid

import cv2
import numpy as np
from PIL import Image, ImageCms

from image_io import imread, imwrite
from lama_config import MODEL_NAME, MASK_EXPAND
from lama_inpainter import inpaint_other

REPAIR_WORKFLOW_ROOT = Path(__file__).resolve().parent / 'repair_workflows'


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
                    edited=None, repair_mask=None, repair_layers=()):
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
    names = {layer[0] for layer in layers}
    for entry in repair_layers:
        name, pixels, visible = _additional_repair_layer(entry, (height, width))
        if name in names:
            raise ValueError(f'修補圖層名稱重複：{name}')
        names.add(name)
        if pixels is not None:
            protected = overlay[:, :, 3] > 0
            if edited is not None:
                protected |= (np.asarray(edited) > 0) & ~_binary_other(other_mask)
            if np.any((pixels[:, :, 3] > 0) & protected):
                raise ValueError(f'修補圖層超出人工保護範圍：{name}')
            layers.append((name, _at_depth(pixels, depth), visible))
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
                            edited_loader=None, extra_methods=(), comfy_url='http://127.0.0.1:8188',
                            model_map=None, export_name=None):
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
    if edited_loader is None:
        outputs = [root.parent]
        if root.parent.name == 'ai_exports' and len(root.resolve().parents) > 2:
            outputs.append(root.resolve().parents[2])
        for output in outputs:
            if ((output / 'raw' / 'project.json').is_file()
                    and all(source.parent.resolve() == output.parent.resolve() for source in images)):
                import project_store as store
                raw_paths = {'raw': str(output / 'raw')}
                edited_loader = lambda source, shape: store.load_page(raw_paths, source, shape)['edited']
                break
    if extra_methods or export_name is not None:
        output, name = _multi_export_root(root, export_name)
        return _export_multi_assets(images, root, output, export_name=name,
            extra_methods=extra_methods, comfy_url=comfy_url, model_map=model_map,
            edited_loader=edited_loader, overwrite=overwrite, lama_visible=lama_visible,
            progress=progress)
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


def export_project_assets(images, paths, *, overwrite=False, progress=None, extra_methods=(),
                          comfy_url='http://127.0.0.1:8188', model_map=None, export_name=None):
    """Export saved PNG materials and layered PSDs; never classify or modify raw."""
    import project_store as store

    images = validate_stems(images)
    root = Path(paths['output']) / 'psd_assets'
    enhanced = bool(extra_methods) or export_name is not None
    if enhanced:
        root, export_name = _multi_export_root(root, export_name)
        _read_multi_manifest(root, export_signature(extra_methods, comfy_url, model_map))
    for name in ('solid', 'other_mask'):
        (root / name).mkdir(parents=True, exist_ok=True)
    for source in images:
        image = imread(source, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f'無法讀取原圖：{source}')
        page = store.load_page(paths, source, image.shape[:2])
        imwrite(root / 'solid' / f'{source.stem}.png', page['overlay'])
        imwrite(root / 'other_mask' / f'{source.stem}.png', page['other'])
    edited_loader = lambda source, shape: store.load_page(paths, source, shape)['edited']
    if enhanced:
        return _export_multi_assets(images, root, root, export_name=export_name,
            extra_methods=extra_methods, comfy_url=comfy_url, model_map=model_map,
            edited_loader=edited_loader, overwrite=overwrite, progress=progress)
    return export_psds_from_assets(images, root, overwrite=overwrite, progress=progress,
        edited_loader=edited_loader)


def _additional_repair_layer(entry, shape):
    """Read an independent difference layer; its mask is not limited to OTHER."""
    if not isinstance(entry, dict):
        raise ValueError('附加修補圖層必須為結構化資料。')
    name = entry.get('name') or entry.get('method_id')
    if not isinstance(name, str) or not name or name in ('bg', 'overlay-manual', MODEL_NAME):
        raise ValueError('附加修補圖層名稱不符。')
    path = entry.get('path') or entry.get('rgba_path')
    if path is None:
        if not entry.get('empty'):
            raise ValueError(f'缺少修補 PNG：{name}')
        return name, None, bool(entry.get('visible', False))
    mask_path = entry.get('mask_path')
    if not mask_path:
        raise ValueError(f'缺少修補差分 Mask：{name}')
    pixels = imread(path, cv2.IMREAD_UNCHANGED)
    mask = imread(mask_path, cv2.IMREAD_UNCHANGED)
    if (mask.shape != shape or mask.dtype != np.uint8
            or not np.all((mask == 0) | (mask == 255))):
        raise ValueError(f'修補差分 Mask 格式不符：{name}')
    if pixels.ndim != 3 or pixels.shape[2] != 4:
        raise ValueError(f'修補 PNG 必須為 RGBA：{name}')
    rgba = cv2.cvtColor(pixels, cv2.COLOR_BGRA2RGBA)
    _validate_repair(rgba, mask)
    return name, rgba if np.any(mask) else None, bool(entry.get('visible', False))


def _multi_export_root(assets_dir, export_name):
    assets_dir = Path(assets_dir)
    if export_name is None:
        export_name = time.strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:8]
    if (not isinstance(export_name, str)
            or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', export_name) is None):
        raise ValueError('export_name 必須為 1 到 80 字元的單一名稱，只能包含英數字、-、_。')
    # Accept a returned run directory as an assets entry without nesting runs.
    if assets_dir.parent.name == 'ai_exports':
        assets_dir = assets_dir.parent.parent
    return assets_dir / 'ai_exports' / export_name, export_name


def export_signature(extra_methods=(), comfy_url='http://127.0.0.1:8188', model_map=None):
    """Canonical configuration shared by enhanced exports and CLI resume."""
    from lama_config import REPAIR_POLICY
    if extra_methods:
        from comfy_repair import validate_methods
        methods = tuple(validate_methods(extra_methods))
    else:
        methods = ()
    workflow_assets = {}
    if methods:
        setup_hash = _hash_file(REPAIR_WORKFLOW_ROOT / 'setup_manifest.json')
        for method in methods:
            folder = REPAIR_WORKFLOW_ROOT / method
            workflow_assets[method] = {
                'workflow_sha256': _hash_file(folder / 'workflow_api.json'),
                'method_manifest_sha256': _hash_file(folder / 'manifest.json'),
                'dependency_setup_manifest_sha256': setup_hash,
            }
    if model_map is not None and not isinstance(model_map, dict):
        raise ValueError('model_map 必須為 JSON 物件。')
    try:
        model_map = json.loads(json.dumps(model_map or {}, sort_keys=True, ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise ValueError('model_map 必須為可序列化的 JSON 物件。') from exc
    return {'extra_methods': list(methods), 'comfy_url': comfy_url,
            'model_map': model_map, 'lama_policy': REPAIR_POLICY,
            'workflow_assets': workflow_assets, 'format': 2}


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.stem}-', suffix='.json.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _input_digest(source, solid, other, edited):
    digest = hashlib.sha256()
    digest.update(_hash_file(source).encode('ascii'))
    for array in (solid, other, edited):
        if array is None:
            digest.update(b'none\0')
        else:
            value = np.ascontiguousarray(array)
            digest.update(str(value.dtype).encode('ascii'))
            digest.update(str(value.shape).encode('ascii'))
            digest.update(value.tobytes())
    return digest.hexdigest()


def _read_multi_manifest(root, signature):
    manifest_path = Path(root) / 'export_manifest.json'
    previous = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    if previous and previous.get('config') != signature:
        raise ValueError('本次 export_name 已用於另一組模型或參數；請使用新名稱。')
    return previous


def _relative_artifacts(root, paths):
    root = Path(root).resolve()
    hashes = {}
    for value in paths:
        if value is None:
            continue
        path = Path(value).resolve()
        try:
            name = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f'修補輸出不在本次目錄內：{path}') from exc
        if not path.is_file():
            raise ValueError(f'修補輸出遺失：{path}')
        hashes[name] = _hash_file(path)
    return hashes


def _export_multi_assets(images, materials, root, *, export_name, extra_methods=(),
                          comfy_url='http://127.0.0.1:8188', model_map=None,
                          edited_loader=None, overwrite=False, lama_visible=True, progress=None):
    """Finish all selected methods before assembling complete per-page PSDs."""
    signature = export_signature(extra_methods, comfy_url, model_map)
    methods = tuple(signature['extra_methods'])
    root, materials = Path(root), Path(materials)
    manifest_path = root / 'export_manifest.json'
    previous = _read_multi_manifest(root, signature)
    old_pages = previous.get('pages', {})
    manifest = {'config': signature, 'export_name': export_name, 'pages': {}, 'complete': False}
    result = {'assets_dir': str(root), 'psds_dir': str(root / 'psds'),
              'repair_dir': str(root / MODEL_NAME), 'repair_mask_dir': str(root / 'repair_mask'),
              'repair_pngs': [], 'created': [], 'skipped': [], 'preserved_stale': [],
              'failed_pages': [], 'errors': {}, 'extra_methods': list(methods),
              'export_name': export_name, 'manifest_path': str(manifest_path), 'complete': False}
    total = len(images)
    units, completed = total * (2 + len(methods)), 0
    prepared = {}
    client = None

    def notify(stage, index, source):
        if progress:
            progress(stage, index, total, source.name, completed, units)

    def fail(source, exc):
        name = source.name
        if name not in result['failed_pages']:
            result['failed_pages'].append(name)
        result['errors'][name] = str(exc)
        manifest['pages'][name] = {'status': 'failed', 'error': str(exc)}
        _json_atomic(manifest_path, manifest)

    # Every page's selected materials finish before the first PSD assembly.
    for index, source in enumerate(images, 1):
        source = Path(source)
        try:
            notify('LaMa 修補', index, source)
            solid_path = materials / 'solid' / f'{source.stem}.png'
            solid = imread(solid_path, cv2.IMREAD_UNCHANGED) if solid_path.is_file() else None
            other_path = materials / 'other_mask' / f'{source.stem}.png'
            other = imread(other_path, cv2.IMREAD_GRAYSCALE)
            edited = edited_loader(source, other.shape) if edited_loader is not None else None
            background, _ = _source(source)
            context, effective = prepare_repair_inputs(background, solid, other, edited)
            protected = (solid[:, :, 3] > 0 if solid is not None else np.zeros(other.shape, bool))
            original_other = _binary_other(other)
            if edited is not None:
                protected |= (np.asarray(edited) > 0) & ~original_other
            input_digest = _input_digest(source, solid, other, edited)
            outputs = []
            for name, pixels in (('solid', solid), ('other_mask', other)):
                if pixels is not None:
                    destination = root / name / f'{source.stem}.png'
                    if destination.resolve() != (materials / name / destination.name).resolve():
                        _write_png_atomic(destination, pixels)
                    outputs.append(destination)
            mask_path = root / 'repair_mask' / f'{source.stem}.png'
            lama_path = root / MODEL_NAME / f'{source.stem}.png'
            exported = export_repair_png(source, other, lama_path, overlay_bgra=solid,
                                          edited=edited, mask_destination=mask_path)
            outputs.append(mask_path)
            if exported:
                outputs.append(exported)
                result['repair_pngs'].append(exported)
            completed += 1
            notify('LaMa 修補', index, source)
            extras = []
            for method in methods:
                notify(method, index, source)
                if np.any(original_other):
                    if client is None:
                        from comfy_repair import ComfyRepairClient
                        candidate = ComfyRepairClient(comfy_url, model_map=model_map)
                        readiness = candidate.check(methods)
                        if readiness is False or (isinstance(readiness, dict) and readiness.get('ready') is False):
                            raise ValueError(f'ComfyUI 尚未就緒：{readiness}')
                        client = candidate
                    rgb8 = context[:, :, :3]
                    if rgb8.dtype == np.uint16:
                        rgb8 = (rgb8 >> 8).astype(np.uint8)
                    response = client.run(method, rgb8.copy(), original_other.astype(np.uint8) * 255,
                        root, key=source.stem, protected_mask=protected.copy())
                else:
                    response = {'method_id': method, 'empty': True, 'rgba_path': None,
                                'mask_path': None, 'result_path': None, 'manifest_path': None}
                entry = {'method_id': method, 'name': method, 'path': response.get('rgba_path'),
                         'mask_path': response.get('mask_path'), 'empty': bool(response.get('empty')),
                         'visible': False}
                _, pixels, _ = _additional_repair_layer(entry, other.shape)
                if pixels is not None and np.any((pixels[:, :, 3] > 0) & protected):
                    raise ValueError(f'{method} 的修補差分覆蓋人工保護範圍。')
                extras.append(entry)
                for key in ('rgba_path', 'mask_path', 'result_path', 'manifest_path'):
                    if response.get(key):
                        outputs.append(response[key])
                completed += 1
                notify(method, index, source)
            hashes = _relative_artifacts(root, outputs)
            record = {'status': 'materials_ready', 'input_digest': input_digest,
                      'material_hashes': hashes, 'extra_layers': extras}
            manifest['pages'][source.name] = record
            prepared[source.name] = {'source': source, 'extras': extras, 'record': record}
            _json_atomic(manifest_path, manifest)
        except Exception as exc:
            fail(source, exc)
            completed = index * (1 + len(methods))

    for index, source in enumerate(images, 1):
        source = Path(source)
        notify('PSD 組裝', index, source)
        page = prepared.get(source.name)
        if page is not None:
            destination = root / 'psds' / f'{source.stem}.psd'
            record = page['record']
            try:
                solid_path = root / 'solid' / f'{source.stem}.png'
                solid = imread(solid_path, cv2.IMREAD_UNCHANGED) if solid_path.is_file() else None
                other = imread(root / 'other_mask' / f'{source.stem}.png', cv2.IMREAD_GRAYSCALE)
                edited = edited_loader(source, other.shape) if edited_loader is not None else None
                effective = imread(root / 'repair_mask' / f'{source.stem}.png', cv2.IMREAD_GRAYSCALE)
                if _input_digest(source, solid, other, edited) != record['input_digest']:
                    raise ValueError('原圖或頁面素材在兩個階段之間改變。')
                artifact_paths = [root / name for name in record['material_hashes']]
                if _relative_artifacts(root, artifact_paths) != record['material_hashes']:
                    raise ValueError('修補素材在兩個階段之間改變。')
                repair_options = {'repair_path': root / MODEL_NAME / f'{source.stem}.png'} if np.any(effective) else {}
                created = export_page_psd(source, solid, other, destination,
                    overwrite=overwrite, lama_visible=lama_visible, edited=edited,
                    repair_mask=effective, repair_layers=page['extras'], **repair_options)
                result['created' if created else 'skipped'].append(str(destination))
                actual = _existing(destination)
                digest = _hash_file(actual)
                old = old_pages.get(source.name, {})
                current = created or (old.get('status') == 'current'
                    and old.get('input_digest') == record['input_digest']
                    and old.get('material_hashes') == record['material_hashes']
                    and old.get('psd_hash') == digest)
                # Decode all layers even when retaining a previously certified PSD.
                PSDImage, _, _ = _api()
                saved = PSDImage.open(actual)
                if saved.numpy() is None or any(layer.numpy() is None for layer in saved):
                    raise ValueError('PSD 合成或圖層無法讀取。')
                if (_relative_artifacts(root, artifact_paths) != record['material_hashes']
                        or _input_digest(source, solid, other, edited) != record['input_digest']):
                    raise ValueError('輸入或修補素材在 PSD 組裝期間改變。')
                record.update(status='current' if current else 'preserved_stale',
                              psd_path=str(actual), psd_hash=digest)
                if not current:
                    result['preserved_stale'].append(str(actual))
                manifest['pages'][source.name] = record
            except Exception as exc:
                fail(source, exc)
        completed += 1
        notify('PSD 組裝', index, source)
        _json_atomic(manifest_path, manifest)
    result['complete'] = not result['failed_pages'] and not result['preserved_stale']
    manifest['complete'] = result['complete']
    _json_atomic(manifest_path, manifest)
    result['artifact_hashes'] = _relative_artifacts(root,
        [path for path in root.rglob('*') if path.is_file() and not path.name.endswith('.tmp')])
    return result
