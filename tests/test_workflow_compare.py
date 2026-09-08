from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from workflow_compare_ui import (
    build_region_labels,
    calculate_difference_mask,
    compose_result,
    discover_export_pair,
    expand_mask,
)


class WorkflowCompareTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / 'export_pair'
        for name in ('alpha', 'beta'):
            (self.root / 'inpaint_workflows' / name).mkdir(parents=True)

        self.base = np.zeros((20, 24, 3), dtype=np.uint8)
        self.mask = np.zeros((20, 24), dtype=np.uint8)
        self.mask[2:8, 3:10] = 255
        self.mask[12:18, 14:22] = 255
        alpha = np.full_like(self.base, (0, 0, 255))
        beta = np.full_like(self.base, (0, 255, 0))
        cv2.imwrite(str(self.root / '01.png'), self.base)
        cv2.imwrite(str(self.root / 'inpaint_workflows' / 'alpha' / '01.png'), alpha)
        cv2.imwrite(str(self.root / 'inpaint_workflows' / 'beta' / '01.png'), beta)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_discovers_workflows_and_mask_regions(self) -> None:
        project = discover_export_pair(self.root)
        labels, count = build_region_labels(self.mask)
        self.assertEqual(project.page_stems, ['01'])
        self.assertEqual(list(project.workflows), ['alpha', 'beta'])
        self.assertEqual(count, 2)
        self.assertNotEqual(int(labels[4, 5]), int(labels[14, 16]))

    def test_calculates_difference_mask_and_filters_tiny_noise(self) -> None:
        result = self.base.copy()
        result[3:9, 4:11] = (30, 40, 50)
        result[15, 20] = (255, 255, 255)

        difference_mask = calculate_difference_mask(
            self.base, result, threshold=12, min_area=8
        )

        self.assertEqual(int(difference_mask[5, 6]), 255)
        self.assertEqual(int(difference_mask[15, 20]), 0)
        self.assertEqual(int(difference_mask[0, 0]), 0)

    def test_difference_threshold_ignores_small_color_change(self) -> None:
        result = self.base.copy()
        result[2:12, 2:12] = (8, 8, 8)

        difference_mask = calculate_difference_mask(
            self.base, result, threshold=12, min_area=1
        )

        self.assertFalse(np.any(difference_mask))

    def test_composes_different_workflows_inside_mask_only(self) -> None:
        project = discover_export_pair(self.root)
        labels, _ = build_region_labels(self.mask)
        assignment = np.zeros(self.mask.shape, dtype=np.uint16)
        assignment[labels == labels[4, 5]] = 2
        assignment[labels == labels[14, 16]] = 3

        output, missing = compose_result(
            project,
            '01',
            assignment,
            {'alpha': 2, 'beta': 3},
            feather_px=0,
        )

        self.assertEqual(missing, [])
        np.testing.assert_array_equal(output[4, 5], (0, 0, 255))
        np.testing.assert_array_equal(output[14, 16], (0, 255, 0))
        np.testing.assert_array_equal(output[0, 0], self.base[0, 0])

    def test_expands_mask_by_requested_radius(self) -> None:
        point = np.zeros((21, 21), dtype=np.uint8)
        point[10, 10] = 255
        expanded = expand_mask(point, 5)

        self.assertEqual(int(expanded[10, 10]), 255)
        self.assertEqual(int(expanded[10, 15]), 255)
        self.assertEqual(int(expanded[10, 16]), 0)
        np.testing.assert_array_equal(expand_mask(point, 0), point)


if __name__ == '__main__':
    unittest.main()
