"""The app's one-model entry points and extracted image/mask utilities."""

import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import cv2
import numpy as np
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

import detect_solid_inpaint_folder as pipeline
from image_io import find_all_imgs, imread, imwrite
from mask_utils import DEFAULT_MASK_DILATE, unify_mask
import solid_inpaint_ui as ui


class HighResOnlyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_old_preferences_offer_only_highres_and_do_not_reuse_old_model_parameters(self):
        self.app = QApplication.instance() or QApplication([])
        settings = QSettings(str(self.root / 'settings.ini'), QSettings.Format.IniFormat)
        for old_model in ('legacy_ctd', 'ctbd', 'ysgyolo'):
            with self.subTest(model=old_model):
                settings.clear()
                settings.setValue('detector/last_name', old_model)
                settings.setValue(f'detector/{old_model}/device', 'cuda')
                settings.setValue(f'detector/{old_model}/inpaint_mask_dilate', 63)
                widget = ui.DetectorSelectorWidget(settings)
                try:
                    self.assertEqual(list(widget.buttons), ['rfdetr'])
                    self.assertTrue(widget.buttons['rfdetr'].isChecked())
                    self.assertEqual(widget.selected_detector(), 'rfdetr')
                    self.assertEqual(widget.selected_detector_params(), {
                        'device': 'auto', 'inpaint_mask_dilate': 2, 'mask_mode': 'text_onomatopoeia'})
                    self.assertFalse(hasattr(widget, 'ctbd_config_frame'))
                    self.assertFalse(hasattr(widget, 'ysg_config_frame'))
                    widget.save_settings()
                    self.assertEqual(settings.value('detector/last_name'), 'rfdetr')
                finally:
                    widget.deleteLater()

    def test_factory_rejects_removed_models_instead_of_silently_switching_explicit_choices(self):
        self.assertEqual(pipeline.DEFAULT_DETECTOR, 'rfdetr')
        for old_model in ('legacy_ctd', 'ctbd', 'ysgyolo'):
            with self.subTest(model=old_model), self.assertRaisesRegex(ValueError, '不支援'):
                pipeline.create_detector(old_model)

    def test_unicode_image_io_preserves_16bit_and_alpha_and_exports_png(self):
        folder = self.root / '漫畫.jpg目錄'
        folder.mkdir()
        gray = np.arange(120, dtype=np.uint16).reshape(10, 12) * 509
        source = folder / '第１頁.jpg'
        imwrite(source, gray)
        output = source.with_suffix('.png')
        self.assertTrue(output.exists())
        self.assertFalse(source.exists())
        self.assertTrue(output.read_bytes().startswith(b'\x89PNG'))
        decoded = imread(output, cv2.IMREAD_UNCHANGED)
        self.assertEqual(decoded.dtype, np.uint16)
        np.testing.assert_array_equal(decoded, gray)
        np.testing.assert_array_equal(imread(output), cv2.cvtColor((gray >> 8).astype(np.uint8), cv2.COLOR_GRAY2BGR))
        rgba = np.zeros((10, 12, 4), np.uint8)
        rgba[:, :, :] = (11, 22, 33, 77)
        alpha_path = folder / '第２頁.png'
        imwrite(alpha_path, rgba)
        np.testing.assert_array_equal(imread(alpha_path, cv2.IMREAD_UNCHANGED), rgba)
        np.testing.assert_array_equal(imread(alpha_path), rgba[:, :, :3])

    def test_folder_discovery_keeps_symlink_page_name_and_ignores_nested_assets(self):
        assets = self.root / 'assets'
        assets.mkdir()
        target = assets / 'actual.png'
        imwrite(target, np.full((3, 3, 3), 255, np.uint8))
        link = self.root / '第２頁.PNG'
        try:
            link.symlink_to(target)
        except OSError as exc:
            self.skipTest(f'Symlinks are unavailable: {exc}')
        self.assertEqual(find_all_imgs(self.root), [link.name])
        self.assertEqual(find_all_imgs(self.root, abs_path=True), [str(link.absolute())])
        np.testing.assert_array_equal(imread(link), imread(target))

    def test_mask_unification_retains_none_rectangle_and_hull_behavior(self):
        mask = np.zeros((30, 40), np.uint8)
        mask[5:10, 5:10] = 255
        mask[15:20, 25:30] = 255
        original = mask.copy()
        self.assertIs(unify_mask(mask, 'none'), mask)
        self.assertEqual(DEFAULT_MASK_DILATE, 4)
        rectangle = unify_mask(mask, 'rectangle')
        hull = unify_mask(mask, 'hull')
        self.assertTrue(np.all(rectangle[5:20, 5:30] == 255))
        self.assertTrue(np.all(hull[mask > 0] == 255))
        self.assertEqual(hull[12, 17], 255)
        self.assertEqual(hull[18, 6], 0)
        self.assertGreater(np.count_nonzero(rectangle), np.count_nonzero(hull))
        for method in ('rectangle', 'hull'):
            self.assertFalse(np.any(unify_mask(np.zeros_like(mask), method)))
        np.testing.assert_array_equal(mask, original)


if __name__ == '__main__':
    unittest.main()
