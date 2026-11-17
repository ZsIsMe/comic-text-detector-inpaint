"""PSD export saves first, runs off the GUI thread, and reports real outcomes."""

import unittest
from unittest.mock import patch

from PySide6.QtGui import QCloseEvent

import solid_inpaint_ui as ui
import test_project_store as fixtures


class PsdExportUiTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.EditorTests()
        self.fixture.setUp()
        self.window = self.fixture.window

    def tearDown(self):
        if isinstance(self.window.worker, ui.PsdExportWorker):
            self.window.cleanup_worker()
        self.fixture.tearDown()

    def test_export_saves_first_and_schedules_background_worker_without_running_inference(self):
        w = self.window
        with patch.object(w, 'save_all_edit_masks', return_value=True) as save, \
                patch.object(ui.QThread, 'start') as start, \
                patch('psd_export.export_project_assets', side_effect=AssertionError('not on GUI thread')) as export:
            w.export_psd_materials()
        save.assert_called_once_with()
        start.assert_called_once_with()
        export.assert_not_called()
        self.assertIsInstance(w.worker, ui.PsdExportWorker)
        self.assertFalse(w.mask_view.isEnabled())
        self.assertEqual(w.worker.images, tuple(w.imglist))
        self.assertIsNot(w.worker.paths, w.paths)
        event = QCloseEvent()
        w.closeEvent(event)
        self.assertFalse(event.isAccepted())
        self.assertIn('PSD 正在導出', w.status.currentMessage())

    def test_failed_save_cannot_start_export(self):
        with patch.object(self.window, 'save_all_edit_masks', return_value=False), \
                patch.object(ui, 'PsdExportWorker') as worker:
            self.window.export_psd_materials()
        worker.assert_not_called()
        self.assertIsNone(self.window.worker_thread)

    def test_constructor_failure_restores_idle_state_and_allows_retry(self):
        w = self.window
        with patch.object(w, 'save_all_edit_masks', return_value=True), \
                patch.object(ui, 'PsdExportWorker', side_effect=RuntimeError('cannot construct')), \
                patch.object(ui.QThread, 'start') as start:
            self.assertFalse(w.start_psd_export(silent=True))
        start.assert_not_called()
        self.assertIsNone(w.worker)
        self.assertIsNone(w.worker_thread)
        self.assertTrue(w.mask_view.isEnabled())
        self.assertFalse(w.progress_row.isVisible())
        self.assertFalse(w._control_worker_silent)
        self.assertIn('cannot construct', w._last_control_error)
        with patch.object(w, 'save_all_edit_masks', return_value=True), patch.object(ui.QThread, 'start'):
            self.assertTrue(w.start_psd_export(silent=True))

    def test_observer_attach_failure_cleans_worker_without_starting_export(self):
        w = self.window

        def failed_attach(worker):
            raise RuntimeError('cannot attach observer')

        with patch.object(w, 'save_all_edit_masks', return_value=True), \
                patch.object(ui.QThread, 'start') as start:
            self.assertFalse(w.start_psd_export(silent=True, on_worker_ready=failed_attach))
        start.assert_not_called()
        self.assertIsNone(w.worker)
        self.assertIsNone(w.worker_thread)
        self.assertTrue(w.mask_view.isEnabled())
        self.assertFalse(w.progress_row.isVisible())
        self.assertIn('cannot attach observer', w._last_control_error)
        with patch.object(w, 'save_all_edit_masks', return_value=True), patch.object(ui.QThread, 'start'):
            self.assertTrue(w.start_psd_export(silent=True))

    def test_thread_start_failure_cleans_busy_state_and_keeps_saved_data(self):
        w = self.window
        with patch.object(w, 'save_all_edit_masks', return_value=True) as save, \
                patch.object(ui.QThread, 'start', side_effect=RuntimeError('cannot start thread')):
            self.assertFalse(w.start_psd_export(silent=True))
        save.assert_called_once_with(silent=True)
        self.assertIsNone(w.worker)
        self.assertIsNone(w.worker_thread)
        self.assertTrue(w.mask_view.isEnabled())
        self.assertFalse(w.progress_row.isVisible())
        self.assertIn('cannot start thread', w._last_control_error)

    def test_ai_options_are_copied_and_forwarded_without_new_ui_controls(self):
        options = {'extra_methods': ['qwen21_int8_convrot'], 'export_name': 'trial_01',
                   'model_map': {'qwen21_int8_convrot': {'unet': 'model.safetensors'}}}
        worker = ui.PsdExportWorker(['page.png'], {'raw': '/raw', 'output': '/output'}, **options)
        options['model_map']['qwen21_int8_convrot']['unet'] = 'changed.safetensors'
        with patch('psd_export.export_project_assets', return_value={}) as export:
            worker.run()
        self.assertEqual(export.call_args.kwargs['extra_methods'], ['qwen21_int8_convrot'])
        self.assertEqual(export.call_args.kwargs['model_map']['qwen21_int8_convrot']['unet'],
                         'model.safetensors')

    def test_worker_reports_progress_created_and_skipped_outputs_and_catches_failure(self):
        worker = ui.PsdExportWorker(['page.png'], {'raw': '/raw', 'output': '/output'})
        progress, finished, failed = [], [], []
        worker.progress.connect(lambda *value: progress.append(value))
        worker.finished.connect(finished.append)
        worker.failed.connect(failed.append)
        result = {'assets_dir': '/output/psd_assets', 'psds_dir': '/output/psd_assets/psds',
                  'repair_dir': '/output/psd_assets/lama_large_512px',
                  'created': ['page.psd'], 'skipped': ['edited.psd']}

        def export(images, paths, *, progress):
            progress('LaMa 修補', 1, 1, 'page.png', 0, 2)
            progress('LaMa 修補', 1, 1, 'page.png', 1, 2)
            progress('PSD 組裝', 1, 1, 'page.png', 1, 2)
            progress('PSD 組裝', 1, 1, 'page.png', 2, 2)
            return result

        with patch('psd_export.export_project_assets', side_effect=export):
            worker.run()
        self.assertEqual([value[4] for value in progress], [0, 1, 1, 2])
        self.assertEqual(finished, [result])
        self.assertEqual(failed, [])
        self.window.on_psd_export_finished(result)
        message = self.window.status.currentMessage()
        self.assertIn('/psds', message)
        self.assertIn('保留已有 1', message)
        self.assertIn('預設顯示', message)
        self.window.on_psd_export_progress('LaMa 修補', 1, 2, 'first.png', 0, 4)
        self.assertEqual(self.window.progress.value(), 0)
        self.assertIn('LaMa 修補 1/2', self.window.progress_label.text())
        self.window.on_psd_export_progress('PSD 組裝', 1, 2, 'first.png', 2, 4)
        self.assertEqual(self.window.progress.value(), 50)
        self.assertIn('PSD 組裝 1/2', self.window.progress_label.text())
        self.window.on_psd_export_progress('PSD 組裝', 2, 2, 'last.png', 4, 4)
        self.assertEqual(self.window.progress.value(), 100)
        with patch('psd_export.export_project_assets', side_effect=OSError('disk full')):
            worker.run()
        self.assertEqual(failed, ['disk full'])
        self.assertEqual(len(finished), 1)


if __name__ == '__main__':
    unittest.main()
