"""Unified detection options apply only when the batch starts."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from PySide6.QtWidgets import QDialog, QMessageBox
import solid_inpaint_ui as ui
import bubble_solid as bubble
import project_store as store
import detect_solid_inpaint_folder as fill
import test_project_store as fixtures


class UnifiedDetectionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.EditorTests()
        self.fixture.setUp()
        self.w = self.fixture.window

    def tearDown(self):
        self.fixture.tearDown()

    def dialog(self, accepted=True):
        dialog = MagicMock()
        dialog.exec.return_value = QDialog.DialogCode.Accepted if accepted else QDialog.DialogCode.Rejected
        dialog.selected_detector.return_value = ui.DETECTOR_YSGYOLO
        dialog.selected_detector_params.return_value = {'device': 'cpu'}
        dialog.selected_solid_settings.return_value = {'enabled': False, 'shrink_percent': 4.5}
        return dialog

    def test_inline_options_follow_project_and_are_shared_by_all_models(self):
        from PySide6.QtGui import QAction
        dialog = ui.DetectorSelectionDialog(self.w.settings, self.w,
            solid_settings={'enabled': False, 'shrink_percent': 6.0})
        self.assertFalse(dialog.bubble_shrink.isEnabled())
        for button in dialog.buttons.values():
            button.setChecked(True)
            self.assertEqual(dialog.selected_solid_settings(), {'enabled': False, 'shrink_percent': 6.0})
        dialog.bubble_enabled.setChecked(True)
        self.assertTrue(dialog.bubble_shrink.isEnabled())
        self.assertFalse(any(action.text() == '純色填充設定' for action in self.w.findChildren(QAction)))
        dialog.close()

    def test_cancel_dialog_does_not_save_or_start(self):
        before = store.load_project(self.w.paths['raw'])
        dialog = self.dialog(False)
        with patch.object(ui, 'DetectorSelectionDialog', return_value=dialog), patch.object(self.w, 'start_worker') as start:
            self.w.run_or_load()
        dialog.save_settings.assert_not_called()
        start.assert_not_called()
        self.assertEqual(before, store.load_project(self.w.paths['raw']))

    def test_cancel_redetection_does_not_save_options(self):
        before = store.load_project(self.w.paths['raw'])
        dialog = self.dialog()
        with patch.object(ui, 'DetectorSelectionDialog', return_value=dialog), patch.object(self.w, 'start_worker') as start, patch.object(ui.QMessageBox, 'warning', return_value=QMessageBox.StandardButton.Cancel):
            self.w.run_or_load()
        start.assert_not_called()
        dialog.save_settings.assert_not_called()
        self.assertEqual(before, store.load_project(self.w.paths['raw']))

    def test_accepted_options_are_passed_to_detection(self):
        dialog = self.dialog()
        with patch.object(ui, 'DetectorSelectionDialog', return_value=dialog), patch.object(self.w, 'start_worker', return_value=True) as start, patch.object(ui.QMessageBox, 'warning', return_value=QMessageBox.StandardButton.Yes):
            self.w.run_or_load()
        self.assertEqual(start.call_args.args[0], 'detect')
        self.assertEqual(start.call_args.kwargs['solid_settings'], {'enabled': False, 'shrink_percent': 4.5})
        dialog.save_settings.assert_called_once()

    def test_batch_persists_disabled_and_bypasses_cached_bubbles(self):
        options = {'enabled': False, 'shrink_percent': 4.5}
        try:
            with patch.object(ui, 'QThread') as thread, patch.object(ui, 'FolderWorker'):
                self.assertTrue(self.w.start_worker('detect', self.w.imglist, solid_settings=options))
                self.assertEqual(bubble.load_settings(self.w.paths['raw']), options)
                thread.return_value.start.assert_called_once()
        finally:
            self.w.worker_thread = None
            self.w.worker = None
        # Even a populated old bubble cache must never trigger checking when off.
        cache = store.cache_path(self.w.paths, self.fixture.path)
        store.update_cache_file(cache, bubble_json=np.array('{"polygons": [[[0,0],[10,0],[10,10]]]}'))
        with patch.object(fill, 'detect_bubbles', side_effect=AssertionError('must bypass cache and model')):
            result = fill.regenerate_image_from_mask(self.fixture.path, self.w.paths, self.fixture.mask)
        self.assertEqual(result['bubble_detection']['status'], 'disabled')
        self.assertEqual(result['solid_bubbles'], 0)

    def test_failed_edit_save_does_not_change_options(self):
        before = bubble.load_settings(self.w.paths['raw'])
        with patch.object(self.w, 'save_all_edit_masks', return_value=False), patch.object(ui, 'QThread') as thread:
            self.assertFalse(self.w.start_worker('detect', self.w.imglist, solid_settings={'enabled': False, 'shrink_percent': 4.5}))
            thread.assert_not_called()
        self.assertEqual(bubble.load_settings(self.w.paths['raw']), before)
