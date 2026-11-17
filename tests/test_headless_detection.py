"""Headless folder detection must keep existing project decisions intact."""

import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import detect_solid_inpaint_folder as cli
import project_store as store


def classify(image, mask, *args, **kwargs):
    overlay = np.zeros((*mask.shape, 4), np.uint8)
    other = np.where(mask > 0, np.uint8(255), np.uint8(0))
    return overlay, other, np.zeros(mask.shape, np.uint8), {}


class FakeDetector:
    device = 'cpu'

    def __init__(self, *, fail_markers=(), mark=0):
        self.fail_markers = set(fail_markers)
        self.mark = mark
        self.calls = []

    def __call__(self, image, **kwargs):
        marker = int(image[0, 0, 0])
        self.calls.append(marker)
        if marker in self.fail_markers:
            raise RuntimeError('synthetic detector failure')
        mask = np.zeros(image.shape[:2], np.uint8)
        if marker != 1:
            mask[4 + self.mark:8 + self.mark, 4:8] = 255
        return None, mask, None


class HeadlessDetectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        for name, marker in (('1.png', 1), ('2.png', 2), ('10.png', 10)):
            image = np.full((24, 24, 3), marker, np.uint8)
            self.assertTrue(cv2.imwrite(str(self.folder / name), image))
        self.paths = {'raw': str(self.folder / 'ctd_inpainted' / 'raw')}

    def execute(self, detector=None, *, model=None, arguments=()):
        detector = detector or FakeDetector()
        with patch.object(cli, 'detector_model_path', return_value=model or Path(__file__)), \
                patch.object(cli, 'create_detector', return_value=detector) as create, \
                patch.object(cli, 'bubble_context', return_value=([], {
                    'shrink_percent': 2.0, 'direct_white': True}, {'status': 'disabled'})), \
                patch.object(cli, '_solid_overlay_from_mask', side_effect=classify), \
                contextlib.redirect_stdout(io.StringIO()) as stdout, \
                contextlib.redirect_stderr(io.StringIO()) as stderr:
            status = cli.main([str(self.folder), *arguments])
        return status, detector, create, stdout.getvalue(), stderr.getvalue()

    def report(self):
        return store.load_project(self.paths['raw'])

    def test_natural_order_range_and_repeat_skip_empty_page_without_model(self):
        status, detector, create, output, _ = self.execute(arguments=('--start-page', '1',
                                                                     '--end-page', '2'))
        self.assertEqual(status, 0)
        self.assertEqual(detector.calls, [1, 2])
        self.assertIn('本次頁面：1–2', output)
        self.assertEqual(self.report()['images'], ['1.png', '2.png', '10.png'])
        self.assertTrue(self.report()['pages']['1.png']['processed'])
        self.assertFalse(store.page_path(self.paths, '1.png').exists())
        self.assertNotIn('10.png', self.report()['pages'])
        store.set_page_reviewed(self.paths, self.folder / '1.png')
        project_path = store.project_path(self.paths['raw'])
        before = project_path.read_bytes()
        status, _, create, output, _ = self.execute(model=Path('/nonexistent/model.pt'),
                                                   arguments=('--start-page', '1', '--end-page', '2'))
        self.assertEqual(status, 0)
        create.assert_not_called()
        self.assertIn('跳過已有結果 2 頁', output)
        self.assertEqual(project_path.read_bytes(), before)
        self.assertTrue(store.is_page_reviewed(self.report()['pages']['1.png']))

    def test_redetect_selected_page_preserves_manual_pixels_and_other_pages(self):
        self.assertEqual(self.execute()[0], 0)
        image = self.folder / '2.png'
        page = store.load_page(self.paths, image, (24, 24))
        page['other'][15, 15] = 255
        page['edited'][15, 15] = 255
        store.save_page(self.paths, image, page)
        store.set_page_reviewed(self.paths, image)
        before = store.page_path(self.paths, '10.png').read_bytes()
        status, detector, _, _, _ = self.execute(FakeDetector(mark=2), arguments=(
            '--start-page', '2', '--end-page', '2', '--redetect'))
        self.assertEqual(status, 0)
        self.assertEqual(detector.calls, [2])
        after = store.load_page(self.paths, image, (24, 24))
        self.assertEqual(int(after['other'][15, 15]), 255)
        self.assertEqual(int(after['edited'][15, 15]), 255)
        self.assertEqual(int(after['other'][4, 5]), 0)
        self.assertEqual(int(after['other'][8, 5]), 255)
        self.assertFalse(store.is_page_reviewed(self.report()['pages']['2.png']))
        self.assertEqual(store.page_path(self.paths, '10.png').read_bytes(), before)

    def test_changed_source_and_missing_page_archive_do_not_redetect(self):
        self.assertEqual(self.execute()[0], 0)
        project_before = store.project_path(self.paths['raw']).read_bytes()
        changed = self.folder / '1.png'
        cv2.imwrite(str(changed), np.full((24, 24, 3), 77, np.uint8))
        store.page_path(self.paths, '2.png').unlink()
        status, detector, create, _, error = self.execute(arguments=('--start-page', '1',
                                                                    '--end-page', '2'))
        self.assertEqual(status, 1)
        self.assertIn('1.png', error)
        self.assertIn('2.png', error)
        create.assert_not_called()
        self.assertEqual(detector.calls, [])
        self.assertEqual(store.project_path(self.paths['raw']).read_bytes(), project_before)

    def test_page_archive_content_must_match_project_digest(self):
        self.assertEqual(self.execute()[0], 0)
        archive_path = store.page_path(self.paths, '2.png')
        archive = store.read_archive(archive_path)
        archive['other'][0, 0] = 255
        store.write_archive(archive_path, archive)
        project_path = store.project_path(self.paths['raw'])
        before = project_path.read_bytes()
        status, detector, create, _, error = self.execute(arguments=('--start-page', '2',
                                                                    '--end-page', '2'))
        self.assertEqual(status, 1)
        self.assertIn('內容與專案記錄不一致', error)
        create.assert_not_called()
        self.assertEqual(detector.calls, [])
        self.assertEqual(project_path.read_bytes(), before)

    def test_failed_page_exit_code_and_retry_leave_other_reviews(self):
        status, detector, _, output, error = self.execute(FakeDetector(fail_markers={2}))
        self.assertEqual(status, 1)
        self.assertEqual(detector.calls, [1, 2, 10])
        self.assertIn('失敗 1 頁', output)
        self.assertIn('synthetic detector failure', error)
        self.assertEqual(self.report()['summary']['failed'], 1)
        store.set_page_reviewed(self.paths, self.folder / '10.png')
        status, detector, _, output, _ = self.execute()
        self.assertEqual(status, 0)
        self.assertEqual(detector.calls, [2])
        self.assertIn('跳過已有結果 2 頁', output)
        self.assertEqual(self.report()['summary']['processed'], 3)
        self.assertEqual(self.report()['summary']['failed'], 0)
        self.assertTrue(store.is_page_reviewed(self.report()['pages']['10.png']))

    def test_bad_range_missing_model_and_unsupported_device_fail_before_writes(self):
        for args in (('--start-page', '0'), ('--end-page', '4'),
                     ('--start-page', '3', '--end-page', '2'), ('--device', 'mps')):
            with self.subTest(args=args):
                status, _, create, _, _ = self.execute(arguments=args)
                self.assertEqual(status, 2)
                create.assert_not_called()
                self.assertFalse((self.folder / 'ctd_inpainted').exists())
        status, _, create, _, error = self.execute(model=Path('/nonexistent/model.pt'))
        self.assertEqual(status, 2)
        self.assertIn('找不到模型檔', error)
        create.assert_not_called()
        self.assertFalse((self.folder / 'ctd_inpainted').exists())
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main([str(self.folder / 'missing')]), 2)

    def test_model_parameters_and_bubble_preference_follow_project(self):
        status, _, create, _, _ = self.execute(arguments=(
            '--start-page', '1', '--end-page', '1', '--detector', 'ysgyolo',
            '--device', 'mps', '--no-bubble'))
        self.assertEqual(status, 0)
        create.assert_called_once_with('ysgyolo', {'device': 'mps'})
        self.assertEqual(self.report()['settings']['solid_fill']['enabled'], False)
        self.assertEqual(self.report()['detector_params'], {'device': 'mps'})
        status, _, create, _, _ = self.execute(arguments=('--start-page', '2', '--end-page', '2'))
        self.assertEqual(status, 0)
        create.assert_called_once_with('ysgyolo', {'device': 'mps'})
        self.assertEqual(self.report()['settings']['solid_fill']['enabled'], False)
        status, _, create, _, _ = self.execute(arguments=(
            '--start-page', '3', '--end-page', '3', '--detector', 'ctbd', '--device', 'auto',
            '--bubble'))
        self.assertEqual(status, 0)
        create.assert_called_once_with('ctbd', {})
        self.assertEqual(self.report()['settings']['solid_fill']['enabled'], True)
        self.assertEqual(self.report()['detector_params'], {})

    def test_importing_cli_does_not_start_qt_gui(self):
        source = ('import sys; import detect_solid_inpaint_folder; '
                  'assert "PySide6.QtWidgets" not in sys.modules; '
                  'assert "solid_inpaint_ui" not in sys.modules')
        child = subprocess.run([sys.executable, '-c', source], cwd=str(Path(cli.__file__).parent),
                               capture_output=True, text=True)
        self.assertEqual(child.returncode, 0, child.stderr)


if __name__ == '__main__':
    unittest.main()
