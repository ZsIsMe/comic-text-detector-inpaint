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
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob('*') if path.is_file()}


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

    def test_layer_order_exact_pixels_alpha_hidden_repair_and_original_size(self):
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
        np.testing.assert_array_equal(read_pixels(psd[2], 8), repair(self.image, self.other))
        self.assertFalse(psd[2].visible)
        # Hidden repair pixels do not alter the default composite.
        self.assertEqual(psd.composite(apply_icc=False).getpixel((10, 8)), (220, 170, 120))
        psd[2].visible = True
        self.assertEqual(psd.composite(apply_icc=False).convert('RGB').getpixel((10, 8)), (21, 88, 155))

    def test_no_solid_omits_overlay_and_empty_other_does_not_load_lama(self):
        empty = np.zeros_like(self.other)
        with patch('lama_inpainter.LamaLargeInpainter', side_effect=AssertionError('no weights')):
            self.assertTrue(export.export_page_psd(self.source, None, empty, self.destination))
        psd = PSDImage.open(self.destination)
        self.assertEqual([layer.name for layer in psd], ['bg', 'lama_large_512px'])
        self.assertFalse(psd[1].visible)
        self.assertFalse(np.any(read_pixels(psd[1], 8)))

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
                expected_repair = repair(expected, self.other)
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
                          if not name.startswith(('psds/', 'lama_large_512px/'))}, before[0])
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

    def test_empty_other_exports_transparent_repair_png_without_loading_model(self):
        destination = self.root / 'repair.png'
        empty = np.zeros_like(self.other)
        with patch('lama_inpainter.LamaLargeInpainter', side_effect=AssertionError('empty')):
            export.export_repair_png(self.source, empty, destination)
        actual = imread(destination, cv2.IMREAD_UNCHANGED)
        self.assertEqual(actual.shape, (12, 16, 4))
        self.assertFalse(np.any(actual))
        with patch.object(export, 'inpaint_other', side_effect=AssertionError('use on-disk PNG')):
            export.export_page_psd(self.source, None, empty, self.destination, repair_path=destination)


if __name__ == '__main__':
    unittest.main()
