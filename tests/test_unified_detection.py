"""Unified detection options apply only when the batch starts."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QMessageBox
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
        dialog.selected_solid_settings.return_value = {'enabled': False, 'shrink_percent': 4.5,
                                                        'direct_white': False}
        dialog.selected_page_range.return_value = (1, len(self.w.imglist))
        return dialog

    def add_pages(self, count=4):
        for index in range(2, count + 1):
            cv2.imwrite(str(self.fixture.root / f'{index:02d}.png'), self.fixture.image)
        self.w.load_folder(str(self.fixture.root))

    def test_inline_options_are_shared_by_all_models(self):
        from PySide6.QtGui import QAction
        dialog = ui.DetectorSelectionDialog(self.w.settings, self.w,
            solid_settings={'enabled': False, 'shrink_percent': 6.0}, image_paths=self.w.imglist)
        self.assertFalse(dialog.bubble_shrink.isEnabled())
        self.assertFalse(dialog.bubble_direct_white.isEnabled())
        self.assertTrue(dialog.bubble_direct_white.isChecked())
        for button in dialog.buttons.values():
            button.setChecked(True)
            self.assertEqual(dialog.selected_solid_settings(),
                             {'enabled': False, 'shrink_percent': 6.0, 'direct_white': True})
        dialog.bubble_enabled.setChecked(True)
        self.assertTrue(dialog.bubble_shrink.isEnabled())
        self.assertTrue(dialog.bubble_direct_white.isEnabled())
        dialog.bubble_direct_white.setChecked(False)
        dialog.bubble_enabled.setChecked(False)
        self.assertFalse(dialog.bubble_direct_white.isEnabled())
        self.assertFalse(dialog.selected_solid_settings()['direct_white'])
        dialog.bubble_enabled.setChecked(True)
        self.assertFalse(dialog.bubble_direct_white.isChecked())
        self.assertFalse(any(action.text() == '純色填充設定' for action in self.w.findChildren(QAction)))
        dialog.close()

    def test_bubble_options_follow_global_settings_across_projects_and_restart(self):
        first = ui.DetectorSelectionDialog(self.w.settings, image_paths=self.w.imglist)
        first.bubble_enabled.setChecked(False)
        first.bubble_direct_white.setChecked(False)
        first.bubble_shrink.setValue(5.5)
        first.save_settings()
        first.close()

        other_folder = self.fixture.root / 'other_project'
        other_folder.mkdir()
        cv2.imwrite(str(other_folder / '01.png'), self.fixture.image)
        self.w.load_folder(str(other_folder))
        store.update_project(self.w.paths['raw'], settings={'solid_fill': {
            'enabled': True, 'shrink_percent': 1.0, 'direct_white': True}})
        reopened_settings = QSettings(str(self.fixture.root / 'ui.ini'), QSettings.Format.IniFormat)
        reopened = ui.DetectorSelectionDialog(reopened_settings, image_paths=self.w.imglist)
        self.assertEqual(reopened.selected_solid_settings(), {
            'enabled': False, 'shrink_percent': 5.5, 'direct_white': False})
        reopened.close()

    def test_bubble_options_save_only_after_detection_starts(self):
        store.update_project(self.w.paths['raw'], settings={'solid_fill': {
            'enabled': False, 'shrink_percent': 7.0, 'direct_white': False}})

        def cancel_options(dialog):
            dialog.bubble_enabled.setChecked(False)
            dialog.bubble_direct_white.setChecked(False)
            return QDialog.DialogCode.Rejected

        with patch.object(ui.DetectorSelectionDialog, 'exec', cancel_options), \
                patch.object(self.w, 'start_worker') as start:
            self.w.run_or_load()
        start.assert_not_called()
        self.assertFalse(self.w.settings.contains('bubble/enabled'))

        def choose_options(dialog):
            self.assertEqual(dialog.selected_solid_settings(), {
                'enabled': True, 'shrink_percent': 2.0, 'direct_white': True})
            dialog.bubble_enabled.setChecked(False)
            dialog.bubble_direct_white.setChecked(False)
            dialog.bubble_shrink.setValue(4.5)
            return QDialog.DialogCode.Accepted

        with patch.object(ui.DetectorSelectionDialog, 'exec', choose_options), \
                patch.object(self.w, 'start_worker', return_value=False), \
                patch.object(ui.QMessageBox, 'warning', return_value=QMessageBox.StandardButton.Yes):
            self.w.run_or_load()
        self.assertFalse(self.w.settings.contains('bubble/enabled'))

        with patch.object(ui.DetectorSelectionDialog, 'exec', choose_options), \
                patch.object(self.w, 'start_worker', return_value=True), \
                patch.object(ui.QMessageBox, 'warning', return_value=QMessageBox.StandardButton.Yes):
            self.w.run_or_load()
        self.assertEqual(ui.DetectorSelectionDialog(self.w.settings).selected_solid_settings(), {
            'enabled': False, 'shrink_percent': 4.5, 'direct_white': False})

    def test_page_range_defaults_to_all_and_tracks_file_names(self):
        paths = ['01.png', '03.png', '10.jpg', '20.jpg']
        dialog = ui.DetectorSelectionDialog(self.w.settings, image_paths=paths)
        self.assertEqual(dialog.selected_page_range(), (1, 4))
        self.assertIn('01.png → 20.jpg', dialog.page_range_hint.text())
        dialog.start_page_spin.setValue(3)
        self.assertEqual(dialog.selected_page_range(), (3, 4))
        self.assertIn('10.jpg → 20.jpg', dialog.page_range_hint.text())
        dialog.end_page_spin.setValue(2)
        self.assertEqual(dialog.selected_page_range(), (2, 2))
        self.assertIn('共 1 頁', dialog.page_range_hint.text())
        dialog.start_page_spin.setValue(99)
        self.assertEqual(dialog.selected_page_range(), (4, 4))
        dialog.close()
        reopened = ui.DetectorSelectionDialog(self.w.settings, image_paths=paths)
        self.assertEqual(reopened.selected_page_range(), (1, 4))
        reopened.close()

    def test_empty_folder_cannot_start_from_dialog(self):
        dialog = ui.DetectorSelectionDialog(self.w.settings, image_paths=[])
        buttons = dialog.findChild(QDialogButtonBox)
        self.assertFalse(buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled())
        self.assertFalse(dialog.start_page_spin.isEnabled())
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
        self.assertEqual(start.call_args.kwargs['solid_settings'],
                         {'enabled': False, 'shrink_percent': 4.5, 'direct_white': False})
        dialog.save_settings.assert_called_once()

    def test_selected_inclusive_range_reaches_worker_and_confirmation(self):
        self.add_pages()
        dialog = self.dialog()
        dialog.selected_page_range.return_value = (2, 3)
        with patch.object(ui, 'DetectorSelectionDialog', return_value=dialog), \
                patch.object(self.w, 'start_worker', return_value=True) as start, \
                patch.object(self.w, 'has_existing_masks', return_value=True) as existing, \
                patch.object(ui.QMessageBox, 'warning', return_value=QMessageBox.StandardButton.Yes) as warning:
            self.w.run_or_load()
        selected = self.w.imglist[1:3]
        existing.assert_called_once_with(selected)
        self.assertEqual(start.call_args.args[:2], ('detect', selected))
        self.assertIn('第 2–3 頁', warning.call_args.args[2])
        self.assertIn('關閉', warning.call_args.args[2])

    def test_confirmation_names_saved_bubble_mode(self):
        dialog = self.dialog()
        dialog.selected_solid_settings.return_value = {
            'enabled': True, 'shrink_percent': 2.0, 'direct_white': False}
        with patch.object(ui, 'DetectorSelectionDialog', return_value=dialog), \
                patch.object(self.w, 'start_worker', return_value=True), \
                patch.object(self.w, 'has_existing_masks', return_value=True), \
                patch.object(ui.QMessageBox, 'warning', return_value=QMessageBox.StandardButton.Yes) as warning:
            self.w.run_or_load()
        self.assertIn('依純色文字底色填充', warning.call_args.args[2])
        dialog.selected_solid_settings.return_value['direct_white'] = True
        with patch.object(ui, 'DetectorSelectionDialog', return_value=dialog), \
                patch.object(self.w, 'start_worker', return_value=True), \
                patch.object(self.w, 'has_existing_masks', return_value=True), \
                patch.object(ui.QMessageBox, 'warning', return_value=QMessageBox.StandardButton.Yes) as warning:
            self.w.run_or_load()
        self.assertIn('直接對氣泡塗白', warning.call_args.args[2])

    def test_unprocessed_subset_does_not_warn_about_other_saved_pages(self):
        self.add_pages()
        dialog = self.dialog()
        dialog.selected_page_range.return_value = (2, 2)
        with patch.object(ui, 'DetectorSelectionDialog', return_value=dialog), \
                patch.object(self.w, 'start_worker', return_value=True) as start, \
                patch.object(ui.QMessageBox, 'warning') as warning:
            self.w.run_or_load()
        warning.assert_not_called()
        self.assertEqual(start.call_args.args[1], [self.w.imglist[1]])

    def test_invalid_range_never_starts_detection(self):
        dialog = self.dialog()
        dialog.selected_page_range.return_value = (0, 0)
        with patch.object(ui, 'DetectorSelectionDialog', return_value=dialog), \
                patch.object(self.w, 'start_worker') as start, \
                patch.object(ui.QMessageBox, 'warning'):
            self.w.run_or_load()
        start.assert_not_called()
        dialog.save_settings.assert_not_called()

    def test_subset_detection_preserves_unselected_pages_and_caches(self):
        self.add_pages()
        for index, path in enumerate(self.w.imglist, 1):
            page = store.empty_page(self.fixture.mask.shape)
            page['overlay'][20:25, 20:25] = (index, index, index, 255)
            store.save_page(self.w.paths, path, page)
            store.update_cache_file(store.cache_path(self.w.paths, path), sentinel=np.array(index))
        before_report = store.load_project(self.w.paths['raw'])
        untouched = [path for index, path in enumerate(self.w.imglist) if index != 1]
        before_files = {p: p.read_bytes() for path in untouched for p in (
            store.page_path(self.w.paths, path), store.cache_path(self.w.paths, path))}
        detector = MagicMock(device='cpu')
        detector.return_value = (None, self.fixture.mask, None)
        worker = ui.FolderWorker(self.w.folder, 'detect', [self.w.imglist[1]])
        failures, progress = [], []
        worker.failed.connect(failures.append)
        worker.progress.connect(lambda current, total, name: progress.append((current, total, name)))
        with patch.object(ui, 'create_detector', return_value=detector), \
                patch.object(fill, 'detect_bubbles', return_value=([], 'detected')):
            worker.run()
        self.assertFalse(failures)
        self.assertEqual(progress, [(1, 1, '02.png')])
        self.assertEqual(detector.call_count, 1)
        for path, contents in before_files.items():
            self.assertEqual(path.read_bytes(), contents)
        after_report = store.load_project(self.w.paths['raw'])
        for path in untouched:
            name = Path(path).name
            self.assertEqual(after_report['pages'][name], before_report['pages'][name])
        self.assertEqual(after_report['summary']['total'], 4)
        self.assertEqual(after_report['summary']['processed'], 4)

    def test_empty_worker_selection_does_not_fall_back_to_all_pages(self):
        worker = ui.FolderWorker(self.w.folder, 'detect', [])
        failures = []
        worker.failed.connect(failures.append)
        with patch.object(ui, 'create_detector', return_value=MagicMock(device='cpu')), \
                patch.object(ui, 'process_image_with_detector') as process:
            worker.run()
        self.assertFalse(failures)
        process.assert_not_called()

    def test_batch_persists_disabled_and_bypasses_cached_bubbles(self):
        options = {'enabled': False, 'shrink_percent': 4.5, 'direct_white': False}
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
