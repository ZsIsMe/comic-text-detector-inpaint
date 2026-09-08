from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import cv2
import numpy as np
import torch
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from detect_solid_inpaint_folder import (
    _ensure_dirs, _mask_path, add_detection_to_mask, create_detector, DETECTOR_YSGYOLO,
)
from solid_inpaint_ui import AddDetectionDialog, DetectorSelectionDialog, DetectorSelectorWidget
from ysg_detector import YSGYoloDetector, YSG_DEFAULT_LABELS, YSG_LABEL_DESCRIPTIONS, round_mask_components


def result_with_boxes(boxes: list, classes: list[int], names: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        names=names or dict(enumerate(YSG_LABEL_DESCRIPTIONS)),
        boxes=SimpleNamespace(cls=torch.tensor(classes), xyxy=torch.tensor(boxes)),
        obb=None,
    )


class RoundMaskTests(unittest.TestCase):
    def test_rounding_trims_each_component_without_filling_holes_or_gaps(self) -> None:
        mask = np.zeros((70, 100), np.uint8)
        mask[5:35, 5:40] = 255
        mask[10:15, 15:20] = 0
        mask[40:65, 55:90] = 255
        rounded = round_mask_components(mask, 8)
        self.assertFalse(np.any(rounded[mask == 0]))
        self.assertEqual(rounded[5, 5], 0)
        self.assertEqual(rounded[40, 55], 0)
        self.assertEqual(rounded[20, 20], 255)
        self.assertEqual(rounded[50, 70], 255)
        self.assertEqual(mask[5, 5], 255)

    def test_overlapping_component_bounds_do_not_erase_other_components(self) -> None:
        mask = np.zeros((50, 50), np.uint8)
        cv2.rectangle(mask, (2, 2), (47, 47), 255, 2)
        mask[18:30, 18:30] = 255
        rounded = round_mask_components(mask, 5)
        self.assertEqual(rounded[23, 23], 255)
        self.assertEqual(rounded[25, 2], 255)
        self.assertFalse(np.any(rounded[mask == 0]))

    def test_large_radius_is_clamped_and_thin_components_survive(self) -> None:
        mask = np.zeros((20, 40), np.uint8)
        mask[2:10, 2:10] = 255
        mask[2:10, 15] = 255
        mask[15, 20] = 255
        oversized = round_mask_components(mask, 1000)
        np.testing.assert_array_equal(oversized, round_mask_components(mask, 4))
        np.testing.assert_array_equal(oversized[:, 15:], mask[:, 15:])

    def test_disabled_and_empty_masks(self) -> None:
        mask = np.ones((10, 10), np.uint8) * 255
        unchanged = round_mask_components(mask, 0)
        np.testing.assert_array_equal(mask, unchanged)
        self.assertIsNot(mask, unchanged)
        self.assertFalse(round_mask_components(np.zeros_like(mask), 12).any())


class YSGDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model_path = Path(self.tmp.name) / 'model.pt'
        self.model_path.touch()
        loader = patch('ultralytics.YOLO')
        self.yolo = loader.start()
        self.addCleanup(loader.stop)
        self.detector = YSGYoloDetector(self.model_path, device='cpu')
        self.model = self.yolo.return_value.to.return_value

    def infer(self, result: SimpleNamespace, image: np.ndarray | None = None) -> tuple:
        self.model.predict.return_value = [result]
        return self.detector(np.zeros((100, 200, 3), np.uint8) if image is None else image)

    def test_default_filter_excludes_frames_and_follows_names_not_ids(self) -> None:
        mask, refined, blocks = self.infer(result_with_boxes(
            [[10, 10, 20, 40], [30, 10, 50, 40], [70, 10, 90, 40]],
            [5, 0, 2], {0: 'other', 2: 'hengxie', 5: 'balloon'},
        ))
        self.assertEqual(mask[20, 15], 255)
        self.assertEqual(mask[20, 35], 0)
        self.assertEqual(mask[20, 80], 255)
        np.testing.assert_array_equal(mask, refined)
        self.assertIsNot(mask, refined)
        self.assertTrue(blocks)

    def test_each_label_can_be_selected_alone(self) -> None:
        names = dict(enumerate(YSG_LABEL_DESCRIPTIONS))
        boxes = [[10 + i * 25, 10, 20 + i * 25, 30] for i in names]
        for class_id, name in names.items():
            self.detector.labels = (name,)
            mask, _, _ = self.infer(result_with_boxes(boxes, list(names), names))
            for other_id in names:
                self.assertEqual(mask[20, 15 + other_id * 25], 255 if class_id == other_id else 0)

    def test_actual_checkpoint_classes_are_not_aliased_to_absent_labels(self) -> None:
        names = {0: 'balloon', 1: 'qipao', 2: 'fangkuai', 3: 'changfangtiao', 4: 'kuangwai', 5: 'other'}
        boxes = [[10 + i * 25, 10, 20 + i * 25, 30] for i in names]
        result = result_with_boxes(boxes, list(names), names)
        mask, _, _ = self.infer(result)
        for class_id in names:
            self.assertEqual(mask[20, 15 + class_id * 25], 255 if class_id in (0, 1, 3) else 0)
        self.detector.labels = ('shuqing', 'hengxie')
        mask, _, blocks = self.infer(result)
        self.assertFalse(mask.any())
        self.assertEqual(blocks, [])

    def test_merge_groups_lines_without_filling_gaps(self) -> None:
        result = result_with_boxes([[10, 10, 90, 20], [10, 24, 90, 34]], [0, 0])
        merged_mask, _, merged = self.infer(result)
        self.detector.merge_text_lines = False
        split_mask, _, split = self.infer(result)
        self.assertEqual(len(merged), 1)
        self.assertEqual(len(split), 2)
        self.assertEqual(merged_mask[22, 50], 0)
        np.testing.assert_array_equal(merged_mask, split_mask)

    def test_vertical_hint_changes_line_order_without_changing_mask(self) -> None:
        result = result_with_boxes([[10, 10, 90, 20], [14, 24, 94, 34]], [0, 0])
        mask, _, blocks = self.infer(result)
        self.assertFalse(blocks[0].vertical)
        self.detector.source_text_vertical = True
        vertical_mask, _, vertical_blocks = self.infer(result)
        self.assertTrue(vertical_blocks[0].vertical)
        self.assertEqual(vertical_blocks[0].lines[0][0][0], 14)
        np.testing.assert_array_equal(mask, vertical_mask)

    def test_dilation_matches_upstream_elliptical_radius(self) -> None:
        result = result_with_boxes([[10, 10, 20, 40]], [0])
        mask, _, _ = self.infer(result)
        self.detector.mask_dilate_size = 2
        expanded, _, _ = self.infer(result)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        np.testing.assert_array_equal(expanded, cv2.dilate(mask, kernel))

    def test_empty_selection_skips_prediction_and_returns_blank_mask(self) -> None:
        self.detector.labels = ()
        mask, _, blocks = self.detector(np.zeros((50, 80), np.uint8))
        self.model.predict.assert_not_called()
        self.assertEqual(mask.shape, (50, 80))
        self.assertFalse(mask.any())
        self.assertEqual(blocks, [])

    def test_rounding_runs_after_dilation_and_leaves_blocks_unchanged(self) -> None:
        result = result_with_boxes([[10, 10, 60, 50]], [0])
        self.detector.mask_dilate_size = 2
        expanded, _, blocks = self.infer(result)
        self.detector.mask_corner_radius = 12
        rounded, refined, rounded_blocks = self.infer(result)
        np.testing.assert_array_equal(rounded, round_mask_components(expanded, 12))
        np.testing.assert_array_equal(rounded, refined)
        self.assertLess(np.count_nonzero(rounded), np.count_nonzero(expanded))
        self.assertEqual(rounded_blocks[0].xyxy, blocks[0].xyxy)

    def test_add_detection_keeps_existing_corner_pixels(self) -> None:
        folder = self.tmp.name
        page = str(Path(folder) / 'page.png')
        cv2.imwrite(page, np.full((100, 200, 3), 255, np.uint8))
        paths = _ensure_dirs(folder)
        current = np.zeros((100, 200), np.uint8)
        current[10, 10] = 255  # This corner would be trimmed if the old mask were rounded.
        current[80:85, 150:155] = 255
        cv2.imwrite(_mask_path(paths, page), current)
        self.detector.mask_corner_radius = 12
        result = result_with_boxes([[10, 10, 60, 50]], [0])
        rounded, _, _ = self.infer(result)
        add_detection_to_mask(page, paths, self.detector)
        merged = cv2.imread(_mask_path(paths, page), cv2.IMREAD_GRAYSCALE)
        np.testing.assert_array_equal(merged, cv2.bitwise_or(current, rounded))
        self.assertEqual(merged[10, 10], 255)

    def test_oriented_boxes_clipping_and_degenerate_boxes(self) -> None:
        result = SimpleNamespace(names={0: 'balloon'}, boxes=None, obb=SimpleNamespace(
            cls=torch.tensor([0, 0]), xyxyxyxy=torch.tensor([
                [[-5, 10], [20, 0], [30, 20], [0, 30]],
                [[10, 10], [10, 10], [10, 10], [10, 10]],
            ]),
        ))
        mask, _, blocks = self.infer(result, np.zeros((100, 200, 4), np.uint8))
        self.assertEqual(mask[15, 15], 255)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(self.model.predict.call_args.kwargs['source'].shape[2], 3)

    def test_no_detections(self) -> None:
        mask, _, blocks = self.infer(result_with_boxes([], []))
        self.assertFalse(mask.any())
        self.assertEqual(blocks, [])

    def test_factory_passes_settings_and_rejects_unknown_labels(self) -> None:
        with patch('detect_solid_inpaint_folder.YSG_MODEL_PATH', self.model_path):
            detector = create_detector(DETECTOR_YSGYOLO, {
                'device': 'cpu', 'labels': ['other'], 'mask_corner_radius': 12,
            })
        self.assertEqual(detector.labels, ('other',))
        self.assertEqual(detector.mask_corner_radius, 12)
        with self.assertRaises(ValueError):
            YSGYoloDetector(self.model_path, labels=['typo'])


class YSGSelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_defaults_descriptions_and_persistence_in_both_workflows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'settings.ini')
            settings = QSettings(path, QSettings.Format.IniFormat)
            widget = DetectorSelectorWidget(settings)
            widget.buttons[DETECTOR_YSGYOLO].setChecked(True)
            params = widget.selected_detector_params()
            self.assertEqual(params['labels'], list(YSG_DEFAULT_LABELS))
            self.assertTrue(params['merge_text_lines'])
            self.assertFalse(params['source_text_vertical'])
            self.assertEqual(params['mask_dilate_size'], 0)
            self.assertEqual(params['mask_corner_radius'], 0)
            for key, checkbox in widget.ysg_label_checks.items():
                self.assertTrue(checkbox.toolTip().startswith(YSG_LABEL_DESCRIPTIONS[key]))
                if key in ('shuqing', 'hengxie'):
                    self.assertIn('此模型無效', checkbox.text())
                checkbox.setChecked(False)
            widget.ysg_device_combo.setCurrentIndex(widget.ysg_device_combo.findData('cpu'))
            widget.ysg_merge_check.setChecked(False)
            widget.ysg_vertical_check.setChecked(True)
            widget.ysg_dilate_spin.setValue(3)
            widget.ysg_corner_spin.setValue(12)
            expected = widget.selected_detector_params()
            widget.save_settings()
            for dialog_type in (DetectorSelectionDialog, AddDetectionDialog):
                dialog = dialog_type(QSettings(path, QSettings.Format.IniFormat))
                self.assertEqual(dialog.selected_detector(), DETECTOR_YSGYOLO)
                self.assertEqual(dialog.selected_detector_params(), expected)
                dialog.deleteLater()
            widget.deleteLater()

    def test_invalid_saved_radius_falls_back_or_clamps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = QSettings(str(Path(directory) / 'settings.ini'), QSettings.Format.IniFormat)
            for value, expected in (('invalid', 0), (-10, 0), (1001, 1000)):
                settings.setValue('detector/ysgyolo/mask_corner_radius', value)
                widget = DetectorSelectorWidget(settings)
                self.assertEqual(widget.ysg_corner_spin.value(), expected)
                widget.deleteLater()


if __name__ == '__main__':
    unittest.main()
