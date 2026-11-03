"""Contract checks for the real local socket with a lightweight GUI stand-in."""

import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import uuid

import cv2
import numpy as np
from PySide6.QtCore import QObject, QSettings, Signal
from PySide6.QtWidgets import QApplication, QDialog, QListWidget, QWidget

from solid_inpaint_control import ControlError, request_control, start_control


class DummyWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(dict)
    failed = Signal(str)


class DummyButton:
    def __init__(self, checked=False):
        self.checked = checked

    def setChecked(self, value):
        self.checked = value


class DetectorSelectorWidget(QWidget):
    def __init__(self, settings):
        super().__init__()
        self.buttons = {name: DummyButton(name == settings.value('detector/last_name', 'legacy_ctd'))
                        for name in ('legacy_ctd', 'ctbd', 'rfdetr', 'ysgyolo')}

    def selected_detector(self):
        return next(name for name, button in self.buttons.items() if button.checked)

    def selected_detector_params(self):
        return {'device': 'cpu', 'inpaint_mask_dilate': 0, 'mask_mode': 'text_onomatopoeia'}


class DummyTimer:
    def isActive(self):
        return False


class DummyWindow(QWidget):
    def __init__(self, root):
        super().__init__()
        self.folder = ''
        self.paths = {}
        self.imglist = []
        self.report = {'pages': {}}
        self.current_img_path = ''
        self.current_base = None
        self.current_page = None
        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self._selected)
        self.worker_thread = None
        self.page_worker_thread = None
        self.ctd_selection_worker_thread = None
        self.render_timer = DummyTimer()
        self.is_local_edit_active = False
        self.is_mask_stroke_active = False
        self._pending_edit_future = None
        self._last_edit_error = None
        self._last_control_error = ''
        self._edit_revision = 1
        self._saved_edit_revision = 1
        self.fail_save = False
        self.worker = None
        self.settings = QSettings(str(root / 'settings.ini'), QSettings.Format.IniFormat)
        self.settings.setValue('detector/last_name', 'rfdetr')

    def _selected(self, row):
        if row >= 0:
            self.current_img_path = self.imglist[row]
            self.current_base = np.full((20, 30, 3), 128, np.uint8)
            self.current_page = {'overlay': np.zeros((20, 30, 4), np.uint8),
                                 'other': np.zeros((20, 30), np.uint8),
                                 'edited': np.zeros((20, 30), np.uint8)}

    def load_folder(self, folder, *, silent=False):
        assert silent
        self.folder = folder
        self.imglist = [str(Path(folder) / '1.jpg'), str(Path(folder) / '2.jpg')]
        self.list_widget.clear()
        self.list_widget.addItems(['1.jpg', '2.jpg'])
        self.list_widget.setCurrentRow(0)
        return True

    def save_all_edit_masks(self, *, silent=False):
        assert silent
        if self.fail_save:
            self._last_control_error = 'mock disk full'
            return False
        return True

    def has_existing_masks(self, images):
        return bool(self.report['pages'].get(Path(images[0]).name))

    def start_worker(self, mode, images, **kwargs):
        assert kwargs.get('silent') is True
        self.worker = DummyWorker()
        self.worker_thread = object()
        kwargs['on_worker_ready'](self.worker)
        return True


class McpControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        name_patch = patch('solid_inpaint_control.server_name',
                           return_value='/tmp/si-t-' + uuid.uuid4().hex[:12])
        name_patch.start()
        self.addCleanup(name_patch.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ('1.jpg', '2.jpg'):
            (self.root / name).write_bytes(b'image placeholder')
        self.window = DummyWindow(self.root)
        self.server = start_control(self.window)
        self.addCleanup(self.server.close)
        self.addCleanup(self.window.close)

    def call(self, method, params=None):
        # The transport is a real Qt socket. The client waits on another thread
        # while the test pumps the GUI event loop to process the request.
        with ThreadPoolExecutor(max_workers=1) as executor:
            task = executor.submit(request_control, method, params, timeout_ms=5000)
            deadline = time.monotonic() + 5
            while not task.done() and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(.005)
            self.app.processEvents()
            return task.result(timeout=1)

    def test_open_and_jump_and_save_failure_cannot_leave_page(self):
        self.assertEqual(self.call('open_project', {'path': str(self.root)})['page_count'], 2)
        self.assertEqual(self.call('go_to_page', {'name': '2.jpg'})['current_page'], '2.jpg')
        self.window.fail_save = True
        with self.assertRaisesRegex(ControlError, 'mock disk full'):
            self.call('go_to_page', {'index': 1})
        self.assertEqual(self.window.current_img_path, str(self.root / '2.jpg'))
        with self.assertRaisesRegex(ControlError, 'mock disk full'):
            self.call('save_project')

    def test_detection_requires_explicit_redetect_and_reports_progress(self):
        self.call('open_project', {'path': str(self.root)})
        self.window.report['pages']['1.jpg'] = {'processed': True}
        with self.assertRaises(ControlError) as raised:
            self.call('start_detection', {'start_page': 1, 'end_page': 1})
        self.assertEqual(raised.exception.code, 'confirmation_required')
        started = self.call('start_detection', {'start_page': 1, 'end_page': 1,
                                                'allow_redetect': True})
        job = started['job_id']
        self.window.worker.progress.emit(1, 1, '1.jpg')
        self.app.processEvents()
        self.assertEqual(self.call('get_job_status', {'job_id': job})['name'], '1.jpg')
        self.window.worker.finished.emit({'pages': {'1.jpg': {'processed': True}}})
        self.app.processEvents()
        self.assertEqual(self.call('get_job_status', {'job_id': job})['state'], 'completed')

    def test_detection_defaults_to_all_project_pages_and_reports_range_gap(self):
        self.call('open_project', {'path': str(self.root)})
        self.window.imglist = [str(self.root / f'{index}.jpg') for index in range(1, 31)]
        all_pages = self.call('start_detection')
        self.assertEqual(all_pages['project_page_count'], 30)
        self.assertEqual(all_pages['selected_page_count'], 30)
        self.assertEqual(all_pages['coverage_missing_pages'], [])
        self.window.worker_thread = None
        partial = self.call('start_detection', {'end_page': 28})
        self.assertEqual(partial['selected_page_count'], 28)
        self.assertEqual(partial['coverage_missing_pages'], ['29.jpg', '30.jpg'])
        self.window.worker.finished.emit({'pages': {}})
        self.app.processEvents()
        final = self.call('get_job_status', {'job_id': partial['job_id']})
        self.assertEqual(final['state'], 'completed')
        self.assertEqual(final['project_page_count'], 30)
        self.assertEqual(final['coverage_missing_pages'], ['29.jpg', '30.jpg'])

    def test_image_matches_current_page_and_busy_guard(self):
        self.call('open_project', {'path': str(self.root)})
        result = self.call('get_page_view', {'kind': 'original', 'max_side': 512})
        import base64
        image = cv2.imdecode(np.frombuffer(base64.b64decode(result['image_png_base64']), np.uint8),
                             cv2.IMREAD_COLOR)
        self.assertEqual(result['name'], '1.jpg')
        self.assertEqual((result['width'], result['height']), (30, 20))
        self.assertEqual(image.shape[:2], (20, 30))
        crop = self.call('get_page_view', {'kind': 'mask', 'crop': [4, 5, 12, 10]})
        self.assertEqual(crop['origin'], [4, 5])
        self.assertEqual((crop['image_width'], crop['image_height']), (48, 40))
        self.assertEqual(crop['scale'], 4.0)
        self.window.is_mask_stroke_active = True
        with self.assertRaises(ControlError) as raised:
            self.call('go_to_page', {'index': 2})
        self.assertEqual(raised.exception.code, 'busy')

    def test_validation_does_not_touch_gui(self):
        with self.assertRaises(ControlError) as no_project:
            self.call('save_project')
        self.assertEqual(no_project.exception.code, 'no_project')
        with self.assertRaises(ControlError):
            self.call('open_project', {'path': 'relative/path'})
        self.assertFalse(self.window.folder)
        self.call('open_project', {'path': str(self.root)})
        for params in ({'start_page': 0}, {'end_page': 3},
                       {'bubble': {'shrink_percent': 50}},
                       {'bubble': {'shrink_percent': True}},
                       {'detector_params': {'bad_key': 1}}):
            with self.subTest(params=params), self.assertRaises(ControlError):
                self.call('start_detection', params)
        self.assertIsNone(self.window.worker)

    def test_unloaded_page_does_not_claim_saved(self):
        self.call('open_project', {'path': str(self.root)})
        self.window.current_page = None
        self.assertFalse(self.call('get_app_status')['page_loaded'])
        with self.assertRaises(ControlError) as unavailable:
            self.call('save_project')
        self.assertEqual(unavailable.exception.code, 'page_unavailable')

    def test_unsaved_revision_and_modal_dialog_are_reported(self):
        self.call('open_project', {'path': str(self.root)})
        self.window._edit_revision += 1
        self.assertTrue(self.call('get_app_status')['save_pending'])
        dialog = QDialog(self.window)
        dialog.setModal(True)
        dialog.show()
        self.app.processEvents()
        try:
            with self.assertRaises(ControlError) as raised:
                self.call('go_to_page', {'index': 2})
            self.assertEqual(raised.exception.code, 'busy')
        finally:
            dialog.close()

    def test_child_process_with_different_tmpdir_connects(self):
        alternate = self.root / 'other-temp'
        alternate.mkdir()
        child_code = (
            'import json,sys; import solid_inpaint_control as c; '
            'c.server_name=lambda:sys.argv[1]; '
            'print(json.dumps(c.request_control("get_app_status")))'
        )
        child = subprocess.Popen(
            [sys.executable, '-c', child_code, self.server.name],
            env={**os.environ, 'TMPDIR': str(alternate)},
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        try:
            deadline = time.monotonic() + 8
            while child.poll() is None and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(.005)
            stdout, stderr = child.communicate(timeout=1)
            self.assertEqual(child.returncode, 0, stderr)
            self.assertEqual(json.loads(stdout)['page_count'], 0)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()

    def test_second_window_failure_cannot_remove_active_endpoint(self):
        second = DummyWindow(self.root)
        try:
            with self.assertRaises(RuntimeError):
                start_control(second)
        finally:
            second.close()
        import gc
        gc.collect()
        self.assertEqual(self.call('get_app_status')['page_count'], 0)
        self.assertTrue(Path(self.server.name).exists())


class RealWindowControlTests(unittest.TestCase):
    def test_mcp_opens_real_window_without_dialog_and_reads_page(self):
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cv2.imwrite(str(root / '1.jpg'), np.full((32, 24, 3), 210, np.uint8))
            cv2.imwrite(str(root / '2.jpg'), np.full((36, 28, 3), 120, np.uint8))
            isolated_name = '/tmp/si-r-' + uuid.uuid4().hex[:12]
            with patch('solid_inpaint_control.server_name', return_value=isolated_name):
                from PySide6.QtCore import QSettings as RealSettings
                import solid_inpaint_ui
                with patch.object(solid_inpaint_ui, 'QSettings',
                                  side_effect=lambda *a: RealSettings(str(root / 'settings.ini'),
                                                                       RealSettings.Format.IniFormat)):
                    window = solid_inpaint_ui.MainWindow()
                try:
                    window.startup_restore_timer.stop()
                    self.assertIsNotNone(window.control_server, window._last_control_error)

                    def call(method, params=None):
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            task = executor.submit(request_control, method, params, timeout_ms=8000)
                            deadline = time.monotonic() + 8
                            while not task.done() and time.monotonic() < deadline:
                                app.processEvents()
                                time.sleep(.005)
                            return task.result(timeout=1)

                    opened = call('open_project', {'path': str(root)})
                    self.assertEqual(opened['page_count'], 2)
                    self.assertEqual(opened['current_page'], '1.jpg')
                    self.assertEqual(call('go_to_page', {'index': 2})['current_page'], '2.jpg')
                    view = call('get_page_view', {'kind': 'original', 'crop': [2, 3, 10, 12]})
                    self.assertEqual(view['name'], '2.jpg')
                    self.assertEqual(view['origin'], [2, 3])
                    self.assertEqual((view['width'], view['height']), (28, 36))
                    self.assertEqual(call('save_project')['saved'], True)
                    self.assertIsNone(QApplication.activeModalWidget())
                finally:
                    window.close()


if __name__ == '__main__':
    unittest.main()
