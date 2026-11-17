"""Real PSD roundtrips with deterministic, bounded repair pixels."""

import contextlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, ImageCms
from psd_tools import PSDImage
from psd_tools.constants import ChannelID, ColorMode, Resource

import psd_export as export
import project_store as store
from image_io import imread, imwrite


def repair(image, mask):
    result = np.zeros((*mask.shape, 4), np.uint8)
    result[mask > 0] = (21, 88, 155, 255)
    return result


def read_pixels(layer, depth, channel=None):
    return np.rint(layer.numpy(channel) * ((1 << depth) - 1)).astype(
        np.uint16 if depth == 16 else np.uint8)


def files(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob('*') if path.is_file()}


class PsdExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / '原圖.png'
        self.image = np.full((12, 16, 3), (120, 170, 220), np.uint8)
        imwrite(self.source, self.image)
        self.overlay = np.zeros((12, 16, 4), np.uint8)
        self.overlay[2:5, 3:7] = (70, 140, 210, 128)
        self.other = np.zeros((12, 16), np.uint8)
        self.other[7:10, 9:13] = 255
        self.destination = self.root / 'psds' / '原圖.psd'

    def create(self, source=None, overlay=None, other=None, destination=None, **kwargs):
        with patch.object(export, 'inpaint_other', side_effect=repair):
            return export.export_page_psd(
                source or self.source, self.overlay if overlay is None else overlay,
                self.other if other is None else other, destination or self.destination, **kwargs)

    def effective(self, background=None, overlay=None, other=None, edited=None):
        return export.prepare_repair_inputs(
            self.image[:, :, ::-1] if background is None else background,
            self.overlay if overlay is None else overlay,
            self.other if other is None else other, edited)[1]

    def test_layer_order_exact_pixels_alpha_visible_repair_and_original_size(self):
        self.assertTrue(self.create())
        psd = PSDImage.open(self.destination)
        self.assertEqual(psd.size, (16, 12))
        self.assertEqual(psd.depth, 8)
        self.assertEqual(psd.color_mode, ColorMode.RGB)
        self.assertEqual(psd.channels, 3)
        self.assertEqual([layer.name for layer in psd], ['bg', 'overlay-manual', 'lama_large_512px'])
        self.assertEqual([layer.name for layer in reversed(list(psd))],
                         ['lama_large_512px', 'overlay-manual', 'bg'])
        for layer in psd:
            self.assertEqual(layer.kind, 'pixel')
            self.assertEqual(layer.bbox, (0, 0, 16, 12))
            self.assertFalse(layer.has_mask())
            self.assertIn(ChannelID.TRANSPARENCY_MASK, [channel.id for channel in layer._record.channel_info])
        np.testing.assert_array_equal(read_pixels(psd[0], 8, 'color'), self.image[:, :, ::-1])
        np.testing.assert_array_equal(read_pixels(psd[1], 8), self.overlay[:, :, [2, 1, 0, 3]])
        np.testing.assert_array_equal(read_pixels(psd[2], 8), repair(self.image, self.effective()))
        self.assertTrue(psd[2].visible)
        self.assertEqual(psd.composite(apply_icc=False).convert('RGB').getpixel((10, 8)), (21, 88, 155))
        psd[2].visible = False
        self.assertEqual(psd.composite(apply_icc=False).convert('RGB').getpixel((10, 8)), (220, 170, 120))
        hidden_destination = self.root / 'explicitly-hidden.psd'
        self.assertTrue(self.create(destination=hidden_destination, lama_visible=False))
        self.assertFalse(PSDImage.open(hidden_destination)[2].visible)

    def test_no_solid_omits_overlay_and_empty_other_does_not_load_lama(self):
        empty = np.zeros_like(self.other)
        with patch.object(export, 'inpaint_other', side_effect=AssertionError('no inference')):
            self.assertTrue(export.export_page_psd(self.source, None, empty, self.destination))
        psd = PSDImage.open(self.destination)
        self.assertEqual([layer.name for layer in psd], ['bg'])
        np.testing.assert_array_equal(read_pixels(psd[0], 8, 'color'), self.image[:, :, ::-1])

    def test_8_and_16bit_gray_and_bgra_preserve_background_precision_and_alpha(self):
        gray = np.arange(12 * 16, dtype=np.uint16).reshape(12, 16) * 311
        bgra = np.dstack((gray, gray // 2, 65535 - gray, np.full_like(gray, 30001)))
        for name, image, depth in (('gray', gray, 16), ('bgra', bgra, 16),
                                   ('bgra8', (bgra >> 8).astype(np.uint8), 8)):
            with self.subTest(mode=name):
                source = self.root / f'{name}.png'
                imwrite(source, image)
                destination = self.root / f'{name}.psd'
                self.assertTrue(self.create(source=source, destination=destination))
                psd = PSDImage.open(destination)
                self.assertEqual(psd.depth, depth)
                self.assertEqual(psd.color_mode, ColorMode.RGB)
                self.assertEqual(psd.channels, 3)
                for layer in psd:
                    self.assertFalse(layer.has_mask())
                    self.assertIn(ChannelID.TRANSPARENCY_MASK,
                                  [channel.id for channel in layer._record.channel_info])
                expected = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB if image.ndim == 2 else cv2.COLOR_BGRA2RGBA)
                actual = read_pixels(psd[0], depth)
                np.testing.assert_array_equal(actual[:, :, :3], expected[:, :, :3])
                np.testing.assert_array_equal(actual[:, :, 3],
                    np.full(gray.shape, 65535, np.uint16) if image.ndim == 2 else image[:, :, 3])
                expected_overlay = self.overlay[:, :, [2, 1, 0, 3]]
                expected_repair = repair(expected, self.effective(background=expected))
                if depth == 16:
                    expected_overlay = expected_overlay.astype(np.uint16) * 257
                    expected_repair = expected_repair.astype(np.uint16) * 257
                np.testing.assert_array_equal(read_pixels(psd[1], depth), expected_overlay)
                np.testing.assert_array_equal(read_pixels(psd[2], depth), expected_repair)

    def test_rgb_icc_is_preserved_and_untagged_cmyk_is_converted_to_rgb(self):
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
        source = self.root / 'icc.jpg'
        Image.fromarray(self.image[:, :, ::-1]).save(source, icc_profile=profile)
        self.create(source=source)
        psd = PSDImage.open(self.destination)
        self.assertEqual(psd.image_resources.get_data(Resource.ICC_PROFILE), profile)
        np.testing.assert_array_equal(read_pixels(psd[0], 8, 'color'), imread(source)[:, :, ::-1])
        cmyk_source = self.root / 'cmyk.jpg'
        Image.new('CMYK', (16, 12), (50, 70, 90, 20)).save(cmyk_source)
        self.create(source=cmyk_source, destination=self.root / 'cmyk.psd')
        cmyk_psd = PSDImage.open(self.root / 'cmyk.psd')
        profile = ImageCms.ImageCmsProfile(__import__('io').BytesIO(
            cmyk_psd.image_resources.get_data(Resource.ICC_PROFILE)))
        self.assertEqual(profile.profile.xcolor_space.strip(), 'RGB')
        with Image.open(cmyk_source) as original:
            np.testing.assert_array_equal(read_pixels(cmyk_psd[0], 8, 'color'),
                                           np.asarray(original.convert('RGB')))

    def test_existing_psd_is_preserved_including_case_variants_and_explicit_overwrite_works(self):
        self.destination.parent.mkdir()
        existing = self.destination.with_name('原圖.PSD')
        existing.write_bytes(b'user hand edited PSD')
        with patch.object(export, 'inpaint_other', side_effect=AssertionError('skip existing')):
            self.assertFalse(export.export_page_psd(self.source, self.overlay, self.other, self.destination))
        self.assertEqual(existing.read_bytes(), b'user hand edited PSD')
        self.assertEqual(len(list(self.destination.parent.iterdir())), 1)
        self.assertTrue(self.create(overwrite=True))
        self.assertEqual(PSDImage.open(existing)[0].name, 'bg')

    def test_atomic_save_or_readback_failure_cleans_temporary_file_and_preserves_existing(self):
        self.destination.parent.mkdir()
        self.destination.write_bytes(b'existing hand edits')

        def failed_save(psd, path, *args, **kwargs):
            Path(path).write_bytes(b'partial')
            raise OSError('disk full')

        for target, failure in (('save', failed_save), ('verify', ValueError('bad roundtrip'))):
            with self.subTest(stage=target), contextlib.ExitStack() as stack:
                if target == 'save':
                    stack.enter_context(patch.object(PSDImage, 'save', failed_save))
                else:
                    stack.enter_context(patch.object(export, '_verify', side_effect=failure))
                with self.assertRaisesRegex(OSError, 'PSD 導出失敗'):
                    self.create(overwrite=True)
            self.assertEqual(self.destination.read_bytes(), b'existing hand edits')
            self.assertEqual([path.name for path in self.destination.parent.iterdir()], [self.destination.name])

    def test_assets_api_reads_existing_pngs_without_changing_materials_or_raw(self):
        assets = self.root / '素材'
        (assets / 'solid').mkdir(parents=True)
        (assets / 'other_mask').mkdir()
        imwrite(assets / 'solid' / '原圖.png', self.overlay)
        imwrite(assets / 'other_mask' / '原圖.png', self.other)
        raw = self.root / 'raw'
        raw.mkdir()
        (raw / 'project.json').write_bytes(b'untouched project')
        before = files(assets), files(raw), self.source.read_bytes()
        with patch.object(export, 'inpaint_other', side_effect=repair):
            result = export.export_psds_from_assets([self.source], assets)
        self.assertEqual(len(result['created']), 1)
        self.assertEqual(len(result['skipped']), 0)
        self.assertEqual({name: data for name, data in files(assets).items()
                          if not name.startswith(('psds/', 'lama_large_512px/', 'repair_mask/'))}, before[0])
        self.assertEqual(files(raw), before[1])
        self.assertEqual(self.source.read_bytes(), before[2])
        with patch.object(export, 'inpaint_other', side_effect=repair) as generate:
            repeat = export.export_psds_from_assets([self.source], assets)
        self.assertEqual(len(repeat['skipped']), 1)
        generate.assert_called_once()  # Explicit export refreshes PNGs while preserving the PSD.

    def test_missing_solid_material_is_optional_but_other_mask_is_required(self):
        assets = self.root / '素材'
        (assets / 'other_mask').mkdir(parents=True)
        imwrite(assets / 'other_mask' / '原圖.png', self.other)
        with patch.object(export, 'inpaint_other', side_effect=repair):
            export.export_psds_from_assets([self.source], assets)
        psd = PSDImage.open(assets / 'psds' / '原圖.psd')
        self.assertEqual([layer.name for layer in psd], ['bg', 'lama_large_512px'])
        (assets / 'other_mask' / '原圖.png').unlink()
        with self.assertRaises(OSError):
            export.export_psds_from_assets([self.source], assets, overwrite=True)

    def test_project_export_preserves_raw_and_returns_legacy_root_string(self):
        from detect_solid_inpaint_folder import _ensure_dirs, export_psd_assets
        paths = _ensure_dirs(str(self.root))
        page = {'overlay': self.overlay, 'other': self.other, 'edited': np.zeros_like(self.other)}
        store.save_page(paths, self.source, page)
        raw = Path(paths['raw'])
        before = files(raw)
        with patch.object(export, 'inpaint_other', side_effect=repair):
            result = export_psd_assets([str(self.source)], paths)
        self.assertIsInstance(result, str)
        np.testing.assert_array_equal(imread(Path(result) / 'solid' / '原圖.png', cv2.IMREAD_UNCHANGED), self.overlay)
        np.testing.assert_array_equal(imread(Path(result) / 'other_mask' / '原圖.png', cv2.IMREAD_GRAYSCALE), self.other)
        self.assertEqual(files(raw), before)

    def test_duplicate_casefold_stems_fail_before_writing_and_missing_dependency_is_clear(self):
        assets = self.root / 'not-created'
        with self.assertRaisesRegex(ValueError, '同名'):
            export.export_psds_from_assets([self.root / 'Page.jpg', self.root / 'page.PNG'], assets)
        self.assertFalse(assets.exists())
        with patch.dict('sys.modules', {'psd_tools': None}):
            with self.assertRaisesRegex(ValueError, 'psd-tools==1.26.1'):
                export.export_page_psd(self.source, self.overlay, self.other, self.destination)
        self.assertFalse(self.destination.exists())

    def test_all_repair_pngs_finish_before_any_psd_and_are_the_only_layer_pixel_source(self):
        assets = self.root / '素材'
        (assets / 'solid').mkdir(parents=True)
        (assets / 'other_mask').mkdir()
        second = self.root / '第二頁.png'
        imwrite(second, self.image)
        images = [self.source, second]
        for source in images:
            imwrite(assets / 'solid' / f'{source.stem}.png', self.overlay)
            imwrite(assets / 'other_mask' / f'{source.stem}.png', self.other)
        calls, stages = [], []
        page_export = export.export_page_psd

        def generate(image, mask):
            self.assertFalse((assets / 'psds').exists())
            calls.append('repair')
            result = repair(image, mask)
            result[mask > 0, 0] = len(calls) * 33
            return result

        def build(*args, **kwargs):
            self.assertEqual(len(list((assets / 'lama_large_512px').glob('*.png'))), 2)
            self.assertIn('repair_path', kwargs)
            return page_export(*args, **kwargs)

        with patch.object(export, 'inpaint_other', side_effect=generate) as inference, \
                patch.object(export, 'export_page_psd', side_effect=build):
            result = export.export_psds_from_assets(images, assets,
                                                   progress=lambda *values: stages.append(values))
        self.assertEqual(inference.call_count, 2)
        self.assertEqual(len(result['repair_pngs']), 2)
        self.assertEqual([event[0] for event in stages], ['LaMa 修補'] * 4 + ['PSD 組裝'] * 4)
        self.assertEqual([event[4] for event in stages], [0, 1, 1, 2, 2, 3, 3, 4])
        self.assertTrue(all(event[5] == 4 for event in stages))
        for source in images:
            png = imread(assets / 'lama_large_512px' / f'{source.stem}.png', cv2.IMREAD_UNCHANGED)
            psd = PSDImage.open(assets / 'psds' / f'{source.stem}.psd')
            np.testing.assert_array_equal(read_pixels(psd[-1], 8), png[:, :, [2, 1, 0, 3]])

    def test_atomic_repair_png_failure_preserves_existing_and_cleans_temporary(self):
        destination = self.root / 'lama_large_512px' / '原圖.png'
        destination.parent.mkdir()
        destination.write_bytes(b'previous generated repair')
        with patch.object(export, 'inpaint_other', side_effect=repair), \
                patch.object(export.os, 'replace', side_effect=OSError('replace failed')):
            with self.assertRaisesRegex(OSError, 'replace failed'):
                export.export_repair_png(self.source, self.other, destination)
        self.assertEqual(destination.read_bytes(), b'previous generated repair')
        self.assertEqual([path.name for path in destination.parent.iterdir()], [destination.name])

    def test_empty_other_does_not_export_repair_png_or_create_directory(self):
        destination = self.root / 'lama_large_512px' / '原圖.png'
        empty = np.zeros_like(self.other)
        with patch.object(export, 'inpaint_other', side_effect=AssertionError('empty')):
            self.assertIsNone(export.export_repair_png(self.source, empty, destination))
        self.assertFalse(destination.exists())
        self.assertFalse(destination.parent.exists())
        with self.assertRaisesRegex(ValueError, '尺寸'):
            export.export_repair_png(self.source, np.zeros((1, 1), np.uint8), destination)

    def test_empty_other_removes_only_matching_entirely_zero_bgra_png(self):
        destination = self.root / 'lama_large_512px' / '原圖.png'
        destination.parent.mkdir()
        empty = np.zeros_like(self.other)
        imwrite(destination, np.zeros((*empty.shape, 4), np.uint8))
        with patch.object(export, 'inpaint_other', side_effect=AssertionError('empty')):
            self.assertIsNone(export.export_repair_png(self.source, empty, destination))
        self.assertFalse(destination.exists())
        # A transparent file with hidden RGB is not an old all-zero generated file.
        for pixels in (np.full((*empty.shape, 4), 50, np.uint8),
                       np.dstack((np.ones((*empty.shape, 3), np.uint8), empty)),
                       np.zeros((1, 1, 4), np.uint8),
                       np.zeros((*empty.shape, 3), np.uint8)):
            with self.subTest(shape=pixels.shape):
                imwrite(destination, pixels)
                before = destination.read_bytes()
                with patch.object(export, 'inpaint_other', side_effect=AssertionError('empty')):
                    self.assertIsNone(export.export_repair_png(self.source, empty, destination))
                self.assertEqual(destination.read_bytes(), before)
        destination.write_bytes(b'bad PNG; preserve it')
        before = destination.read_bytes()
        self.assertIsNone(export.export_repair_png(self.source, empty, destination))
        self.assertEqual(destination.read_bytes(), before)

    def test_empty_other_ignores_repair_inputs_and_keeps_nonempty_solid_layer(self):
        empty = np.zeros_like(self.other)
        with patch.object(export, 'inpaint_other', side_effect=AssertionError('empty')):
            self.assertTrue(export.export_page_psd(
                self.source, self.overlay, empty, self.destination,
                repair_path=self.root / 'missing.png', repair_rgba=np.ones((1, 1))))
        psd = PSDImage.open(self.destination)
        self.assertEqual([layer.name for layer in psd], ['bg', 'overlay-manual'])
        np.testing.assert_array_equal(read_pixels(psd[1], 8), self.overlay[:, :, [2, 1, 0, 3]])

    def test_mixed_empty_and_nonempty_project_keeps_two_phases_and_raw_unchanged(self):
        from detect_solid_inpaint_folder import _ensure_dirs
        second = self.root / '空頁.png'
        imwrite(second, self.image)
        paths = _ensure_dirs(str(self.root))
        store.save_page(paths, self.source, {'overlay': self.overlay, 'other': self.other,
                                           'edited': np.zeros_like(self.other)})
        store.save_page(paths, second, store.empty_page(self.other.shape))
        raw = Path(paths['raw'])
        raw_before = files(raw)
        sources_before = {p.name: p.read_bytes() for p in (self.source, second)}
        assets = Path(paths['output']) / 'psd_assets'
        stages = []
        original_build = export.export_page_psd

        def build(source, *args, **kwargs):
            self.assertTrue((assets / 'lama_large_512px' / '原圖.png').exists())
            self.assertFalse((assets / 'lama_large_512px' / '空頁.png').exists())
            if Path(source) == second:
                self.assertNotIn('repair_path', kwargs)
            return original_build(source, *args, **kwargs)

        with patch.object(export, 'inpaint_other', side_effect=repair) as inference, \
                patch.object(export, 'export_page_psd', side_effect=build):
            result = export.export_project_assets([self.source, second], paths,
                                                  progress=lambda *v: stages.append(v))
        inference.assert_called_once()
        self.assertEqual(len(result['repair_pngs']), 1)
        self.assertEqual(len(result['created']), 2)
        self.assertEqual([event[0] for event in stages], ['LaMa 修補'] * 4 + ['PSD 組裝'] * 4)
        self.assertEqual([event[4] for event in stages], [0, 1, 1, 2, 2, 3, 3, 4])
        self.assertTrue(all(event[5] == 4 for event in stages))
        self.assertEqual([p.name for p in PSDImage.open(assets / 'psds' / '空頁.psd')], ['bg'])
        self.assertEqual(files(raw), raw_before)
        self.assertEqual({p.name: p.read_bytes() for p in (self.source, second)}, sources_before)

    def test_workflow_grow_is_eight_pixel_diamond_and_preserves_manual_decisions(self):
        background = np.full((35, 35, 3), (20, 40, 80), np.uint8)
        overlay = np.zeros((35, 35, 4), np.uint8)
        overlay[17, 20] = (240, 100, 10, 255)
        other = np.zeros((35, 35), np.uint8)
        other[17, 17] = 255
        edited = np.zeros_like(other)
        edited[17, 17] = edited[20, 17] = 255
        before = tuple(a.copy() for a in (background, overlay, other, edited))
        composite, actual = export.prepare_repair_inputs(background, overlay, other, edited)
        rows, columns = np.indices(other.shape)
        expected = (abs(rows - 17) + abs(columns - 17) <= 8).astype(np.uint8) * 255
        expected[17, 20] = expected[20, 17] = 0
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual[17, 17], 255)  # Manually assigned OTHER remains repairable.
        np.testing.assert_array_equal(composite[17, 20], [10, 100, 240])
        for source, original in zip((background, overlay, other, edited), before):
            np.testing.assert_array_equal(source, original)

    def test_repair_context_composites_bgra_at_source_depth_and_preserves_source_alpha(self):
        overlay = np.array([[[240, 100, 10, 128], [40, 80, 120, 255]]], np.uint8)
        mask = np.zeros((1, 2), np.uint8)
        for dtype, maximum, factor in ((np.uint8, 255, 1), (np.uint16, 65535, 257)):
            with self.subTest(dtype=dtype):
                background = np.full((1, 2, 4), (20 * factor, 40 * factor, 80 * factor,
                                                maximum // 3), dtype)
                composite, actual = export.prepare_repair_inputs(background, overlay, mask)
                if dtype == np.uint8:
                    alpha = np.float32(128) / np.float32(255)
                    expected = (np.array([20, 40, 80], np.float32) * (np.float32(1) - alpha)
                                + np.array([10, 100, 240], np.float32) * alpha).astype(dtype)
                else:
                    expected = np.rint(np.array([20, 40, 80]) * factor * (127 / 255)
                                       + np.array([10, 100, 240]) * factor * (128 / 255)).astype(dtype)
                np.testing.assert_array_equal(composite[0, 0, :3], expected)
                np.testing.assert_array_equal(composite[0, 1, :3], np.array([120, 80, 40]) * factor)
                np.testing.assert_array_equal(composite[:, :, 3], background[:, :, 3])
                self.assertEqual(composite.dtype, dtype)
                self.assertFalse(np.any(actual))

    def test_8bit_context_matches_actual_export_pair_compositor_for_partial_alpha(self):
        from detect_solid_inpaint_folder import _compose_overlay_preview
        random = np.random.default_rng(73)
        base_bgr = random.integers(0, 256, (9, 13, 3), dtype=np.uint8)
        overlay = random.integers(0, 256, (9, 13, 4), dtype=np.uint8)
        overlay[0, :6, 3] = [0, 1, 127, 128, 254, 255]
        expected_bgr = _compose_overlay_preview(base_bgr, overlay)
        background_rgba = np.dstack((base_bgr[:, :, ::-1], np.full((9, 13), 91, np.uint8)))
        actual, _ = export.prepare_repair_inputs(background_rgba, overlay, np.zeros((9, 13), np.uint8))
        np.testing.assert_array_equal(actual[:, :, :3], expected_bgr[:, :, ::-1])
        np.testing.assert_array_equal(actual[:, :, 3], background_rgba[:, :, 3])

    def test_actual_mask_png_matches_repair_alpha_and_model_sees_composited_base(self):
        edited = np.zeros_like(self.other)
        edited[6, 9] = 255  # Protected erasure beside OTHER, inside the expansion.
        expected_context, expected_mask = export.prepare_repair_inputs(
            self.image[:, :, ::-1], self.overlay, self.other, edited)
        destination = self.root / 'repair.png'
        mask_destination = self.root / 'mask.png'
        source_before = self.source.read_bytes()
        with patch.object(export, 'inpaint_other', side_effect=repair) as infer:
            export.export_repair_png(self.source, self.other, destination,
                overlay_bgra=self.overlay, edited=edited, mask_destination=mask_destination)
        np.testing.assert_array_equal(infer.call_args.args[0], expected_context)
        np.testing.assert_array_equal(infer.call_args.args[1], expected_mask)
        np.testing.assert_array_equal(imread(mask_destination, cv2.IMREAD_GRAYSCALE), expected_mask)
        pixels = imread(destination, cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(pixels[:, :, 3], expected_mask)
        self.assertFalse(np.any(pixels[expected_mask == 0]))
        self.assertEqual(self.source.read_bytes(), source_before)

    def test_project_and_assets_reexport_reuse_raw_erase_protection_without_changing_it(self):
        from detect_solid_inpaint_folder import _ensure_dirs
        paths = _ensure_dirs(str(self.root))
        edited = np.zeros_like(self.other)
        edited[6, 9] = 255
        store.save_page(paths, self.source, {'overlay': self.overlay, 'other': self.other,
                                             'edited': edited})
        raw = Path(paths['raw'])
        before = files(raw)
        expected_mask = self.effective(edited=edited)
        with patch.object(export, 'inpaint_other', side_effect=repair):
            result = export.export_project_assets([self.source], paths)
        assets = Path(result['assets_dir'])
        np.testing.assert_array_equal(imread(assets / 'repair_mask' / '原圖.png',
                                            cv2.IMREAD_GRAYSCALE), expected_mask)
        psd_path = assets / 'psds' / '原圖.psd'
        psd_before = psd_path.read_bytes()
        np.testing.assert_array_equal(read_pixels(PSDImage.open(psd_path)[-1], 8)[:, :, 3],
                                      expected_mask)
        with patch.object(export, 'inpaint_other', side_effect=repair) as infer:
            repeated = export.export_psds_from_assets([self.source], assets)
        infer.assert_called_once()
        np.testing.assert_array_equal(infer.call_args.args[1], expected_mask)
        self.assertEqual(repeated['skipped'], [str(psd_path)])
        self.assertEqual(psd_path.read_bytes(), psd_before)
        self.assertEqual(files(raw), before)

    def test_assets_reads_edited_once_and_phase_two_uses_persisted_actual_mask(self):
        assets = self.root / 'isolated-materials'
        (assets / 'solid').mkdir(parents=True)
        (assets / 'other_mask').mkdir()
        imwrite(assets / 'solid' / '原圖.png', self.overlay)
        imwrite(assets / 'other_mask' / '原圖.png', self.other)
        edited = np.zeros_like(self.other)
        edited[6, 9] = 255
        reads = []

        def load(source, shape):
            reads.append((source, shape))
            return edited

        with patch.object(export, 'inpaint_other', side_effect=repair) as infer:
            result = export.export_psds_from_assets([self.source], assets, edited_loader=load)
        self.assertEqual(len(reads), 1)
        infer.assert_called_once()
        expected_mask = self.effective(edited=edited)
        actual = read_pixels(PSDImage.open(result['created'][0])[-1], 8)
        np.testing.assert_array_equal(actual[:, :, 3], expected_mask)

    def test_actual_mask_override_cannot_escape_expansion_or_omit_original_other(self):
        image = np.zeros((35, 35, 3), np.uint8)
        source = self.root / 'larger.png'
        imwrite(source, image)
        other = np.zeros((35, 35), np.uint8)
        other[17, 17] = 255
        _, allowed = export.prepare_repair_inputs(image, None, other)
        outside = allowed.copy()
        outside[0, 0] = 255
        missing = allowed.copy()
        missing[17, 17] = 0
        nonbinary = allowed.copy()
        nonbinary[17, 17] = 1
        for invalid in (outside, missing, nonbinary):
            with self.subTest(invalid=invalid[17, 17]), patch.object(
                    export, 'inpaint_other', side_effect=AssertionError('must reject before inference')):
                with self.assertRaisesRegex(ValueError, 'Mask'):
                    export.export_page_psd(source, None, other, self.destination, repair_mask=invalid)

    def test_empty_assets_have_black_actual_mask_but_no_repair_png_or_layer(self):
        assets = self.root / 'empty-materials'
        (assets / 'other_mask').mkdir(parents=True)
        imwrite(assets / 'other_mask' / '原圖.png', np.zeros_like(self.other))
        with patch.object(export, 'inpaint_other', side_effect=AssertionError('empty')):
            result = export.export_psds_from_assets([self.source], assets)
        self.assertFalse(np.any(imread(assets / 'repair_mask' / '原圖.png', cv2.IMREAD_GRAYSCALE)))
        self.assertFalse((assets / 'lama_large_512px').exists())
        self.assertEqual([p.name for p in PSDImage.open(result['created'][0])], ['bg'])

    def test_workflow_threshold_matches_uint8_uint16_bool_and_normalized_float_masks(self):
        np.testing.assert_array_equal(export._binary_other(np.array([0, 1, 127, 128, 255], np.uint8)),
                                      [False, False, False, True, True])
        np.testing.assert_array_equal(export._binary_other(np.array([0, 32767, 32768, 65535], np.uint16)),
                                      [False, False, True, True])
        np.testing.assert_array_equal(export._binary_other(np.array([False, True])), [False, True])
        values = np.array([0.0, 0.49, 0.5, np.nextafter(0.5, 1), 1.0], np.float64)
        np.testing.assert_array_equal(export._binary_other(values), [False, False, False, True, True])
        for invalid in (np.array([np.nan]), np.array([np.inf]), np.array([-0.01]),
                        np.array([1.01]), np.array([0, 1], np.int32)):
            with self.subTest(mask=invalid):
                with self.assertRaisesRegex(ValueError, 'Mask'):
                    export._binary_other(invalid)

    def test_low_gray_or_half_float_mask_skips_inference_and_accepts_empty_actual_mask(self):
        empty = np.zeros_like(self.other)
        for label, mask in (('gray', np.full_like(self.other, 127)),
                            ('half-float', np.full(self.other.shape, 0.5, np.float32))):
            with self.subTest(label=label), patch.object(
                    export, 'inpaint_other', side_effect=AssertionError('below threshold')):
                png = self.root / f'{label}-repair.png'
                actual = self.root / f'{label}-mask.png'
                self.assertIsNone(export.export_repair_png(self.source, mask, png,
                                                          mask_destination=actual))
                self.assertFalse(png.exists())
                np.testing.assert_array_equal(imread(actual, cv2.IMREAD_GRAYSCALE), empty)
                psd = self.root / f'{label}.psd'
                self.assertTrue(export.export_page_psd(self.source, self.overlay, mask, psd,
                                                       repair_mask=empty))
                self.assertEqual([p.name for p in PSDImage.open(psd)], ['bg', 'overlay-manual'])


if __name__ == '__main__':
    unittest.main()
