"""PSD dispatch follows GUI saving and asynchronous job contracts.

Transport is exercised separately by test_mcp_control; keep this fixture on
the GUI thread so Windows' blocking named-pipe waits cannot stall Qt signals.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtCore import QObject, Signal

import solid_inpaint_control as control
import test_mcp_control as fixtures


class PsdWorker(QObject):
    progress = Signal(str, int, int, str, int, int)
    finished = Signal(dict)
    failed = Signal(str)


class McpPsdExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.McpControlTests.setUpClass()

    def setUp(self):
        self.registry = tempfile.TemporaryDirectory()
        self.addCleanup(self.registry.cleanup)
        self.patch = patch.object(control, '_registry_dir', return_value=Path(self.registry.name))
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.fixture = fixtures.McpControlTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.window = self.fixture.window
        self.call = lambda method, params=None: self.fixture.server.dispatch(method, params or {})
        self.call('open_project', {'path': str(self.fixture.root)})
        self.window.paths = {'raw': str(self.fixture.root / 'ctd_inpainted' / 'raw')}
        self.started = []
        self.save_calls = []

        def save(*, silent=False):
            self.save_calls.append(silent)
            return not self.window.fail_save

        def start(images, *, options, silent, on_worker_ready):
            self.started.append((list(images), options, silent))
            self.window.worker = PsdWorker()
            self.window.worker_thread = object()
            on_worker_ready(self.window.worker)
            return True

        self.window.save_all_edit_masks = save
        self.window.start_psd_export = start

    def test_selected_methods_and_range_save_before_start_and_report_real_stages(self):
        methods = ['klein9b_fp8_lanpaint', 'qwen21_int8_convrot']
        job = self.call('start_psd_export', {'extra_methods': methods, 'start_page': 2})
        self.assertEqual(self.save_calls, [True])
        images, options, silent = self.started[0]
        self.assertEqual(images, [self.window.imglist[1]])
        self.assertEqual(options['extra_methods'], methods)
        self.assertEqual(options['export_name'], job['job_id'])
        self.assertTrue(silent)
        self.assertEqual(job['kind'], 'psd_export')
        self.window.worker.progress.emit('Qwen 修補', 1, 1, '2.jpg', 2, 4)
        status = self.call('get_job_status', {'job_id': job['job_id']})
        self.assertEqual(status['stage'], 'Qwen 修補')
        self.assertEqual(status['completed_units'], 2)
        result = {'complete': True, 'failed_pages': [], 'created': ['2.psd'],
                  'assets_dir': '/materials', 'extra_methods': methods}
        self.window.worker.finished.emit(result)
        status = self.call('get_job_status', {'job_id': job['job_id']})
        self.assertEqual(status['state'], 'completed')
        self.assertEqual(status['result'], result)

    def test_save_failure_and_busy_state_never_start_export(self):
        self.window.fail_save = True
        with self.assertRaises(control.ControlError) as error:
            self.call('start_psd_export')
        self.assertEqual(error.exception.code, 'save_failed')
        self.assertEqual(self.started, [])
        self.window.fail_save = False
        self.window.is_local_edit_active = True
        with self.assertRaises(control.ControlError) as error:
            self.call('start_psd_export')
        self.assertEqual(error.exception.code, 'busy')
        self.assertEqual(self.started, [])

    def test_bad_parameters_are_rejected_before_saving_or_starting(self):
        invalid = [
            {'extra_methods': 'qwen21_int8_convrot'},
            {'extra_methods': ['unverified-model']},
            {'extra_methods': ['qwen21_int8_convrot'] * 2},
            {'start_page': True}, {'end_page': 3},
            {'comfy_url': 'file:///tmp'}, {'comfy_url': None},
            {'export_name': '../existing'},
            {'export_name': '_invalid_start'},
            {'model_map': {'qwen21_int8_convrot': {'clip': 'encoder'}}},
            {'extra_methods': ['qwen21_int8_convrot'],
             'model_map': {'qwen21_int8_convrot': {'unexpected': 'value'}}},
        ]
        for params in invalid:
            with self.subTest(params=params):
                with self.assertRaises(control.ControlError) as error:
                    self.call('start_psd_export', params)
                self.assertEqual(error.exception.code, 'bad_request')
        self.assertEqual(self.save_calls, [])
        self.assertEqual(self.started, [])

    def test_lama_default_needs_no_comfy_and_partial_results_are_not_completed(self):
        job = self.call('start_psd_export')
        self.assertEqual(self.started[0][1]['extra_methods'], [])
        self.assertIsNone(job['export_name'])
        self.window.worker.finished.emit({'complete': False, 'failed_pages': ['2.jpg'],
                                           'errors': {'2.jpg': 'Qwen failed'}})
        status = self.call('get_job_status')
        self.assertEqual(status['state'], 'incomplete')
        self.assertEqual(status['failed_pages'], ['2.jpg'])

    def test_resume_name_and_model_map_survive_worker_failure(self):
        job = self.call('start_psd_export', {
            'extra_methods': ['qwen21_int8_convrot'], 'export_name': 'trial_001',
            'comfy_url': 'http://localhost:8189',
            'model_map': {'qwen21_int8_convrot': {'unet': 'subfolder/model.safetensors'}},
        })
        self.window.worker.failed.emit('connection lost; prompt_id retained')
        status = self.call('get_job_status', {'job_id': job['job_id']})
        self.assertEqual(status['state'], 'failed')
        self.assertEqual(status['export_name'], 'trial_001')
        self.assertEqual(self.started[0][1]['model_map']['qwen21_int8_convrot']['unet'],
                         'subfolder/model.safetensors')


if __name__ == '__main__':
    unittest.main()
