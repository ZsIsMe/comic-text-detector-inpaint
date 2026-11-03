"""MCP page editing against the real GUI state and authoritative project store."""

import base64
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import cv2
import numpy as np
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QApplication

import project_store as store
import solid_inpaint_ui as ui
from solid_inpaint_control import ControlError, ControlServer
from detect_solid_inpaint_folder import prepare_page_edits


def _decode(view):
    return cv2.imdecode(np.frombuffer(base64.b64decode(view['image_png_base64']), np.uint8),
                        cv2.IMREAD_COLOR)


class McpReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ('1.png', '2.png'):
            self.assertTrue(cv2.imwrite(str(self.root / name),
                                        np.full((40, 60, 3), 240, np.uint8)))
        settings = QSettings(str(self.root / 'settings.ini'), QSettings.Format.IniFormat)
        with patch.object(ui, 'QSettings', return_value=settings), \
                patch('solid_inpaint_control.start_control', return_value=Mock()):
            self.window = ui.MainWindow()
        self.addCleanup(self.window.close)
        self.window.startup_restore_timer.stop()
        self.assertTrue(self.window.load_folder(str(self.root), silent=True))
        # Direct dispatch exercises the same GUI-thread endpoint without
        # binding an extra local socket next to the user's running application.
        self.control = ControlServer.__new__(ControlServer)
        self.control.window = self.window
        self.control._review_tokens = {}
        self.control._last_batch_stamp = None
        self.control.latest_job_id = ''
        self.control.jobs = {}

    def bundle(self, **kwargs):
        return self.control.dispatch('get_review_bundle', kwargs)

    def apply(self, token, operations, **kwargs):
        return self.control.dispatch('apply_page_edits',
                                     {'page_token': token, 'operations': operations, **kwargs})

    def seed_page(self):
        w = self.window
        state = store.empty_page((40, 60))
        state['overlay'][2:8, 2:8] = (5, 6, 7, 255)
        state['other'][10:14, 20:25] = 255
        summary = store.save_page(w.paths, w.current_img_path, state)
        w.report.setdefault('pages', {})[Path(w.current_img_path).name] = summary
        w.reload_current()
        return {key: value.copy() for key, value in w.current_page.items()}

    def test_bundle_has_same_saved_snapshot_and_original_pixel_crop(self):
        self.seed_page()
        bundle = self.bundle(crop=[1, 2, 12, 10], max_side=1024, include_mask=True)
        self.assertEqual(list(bundle['images']), ['original', 'composite', 'mask'])
        self.assertTrue(bundle['saved'])
        self.assertEqual((bundle['width'], bundle['height']), (60, 40))
        self.assertEqual(bundle['origin'], [1, 2])
        self.assertEqual(bundle['scale'], 4)
        originals = _decode(bundle['images']['original'])
        composites = _decode(bundle['images']['composite'])
        self.assertEqual(originals.shape, composites.shape)
        self.assertEqual(originals.shape[:2], (40, 48))
        np.testing.assert_array_equal(originals[0, 0], [240, 240, 240])
        self.assertFalse(np.array_equal(originals, composites))
        self.assertEqual(bundle['images']['original']['revision'],
                         bundle['images']['composite']['revision'])

    def test_rect_polygon_transfer_and_single_undo_match_gui_core(self):
        before = self.seed_page()
        w = self.window
        token = self.bundle()['page_token']
        operations = [
            {'kind': 'restore', 'shape': {'rect': [3, 3, 2, 2]}},
            {'kind': 'add', 'target': 'solid', 'color_rgb': [200, 100, 50],
             'shape': {'rect': [30, 20, 3, 4]}},
            {'kind': 'transfer', 'target': 'solid', 'color_rgb': [1, 2, 3],
             'shape': {'polygon': [[20, 10], [24, 10], [24, 13], [20, 13]]}},
            {'kind': 'add', 'target': 'other', 'shape': {'rect': [2, 2, 2, 2]}},
        ]
        # GUI's authoritative preparation algorithm is the pixel oracle.
        expected = {key: value.copy() for key, value in before.items()}
        solid, other = before['overlay'][:, :, 3].copy(), before['other'].copy()
        solid[3:5, 3:5] = 0; other[3:5, 3:5] = 0
        expected = prepare_page_edits(w.current_base, solid, other, state=expected,
                                      dirty_bbox=(3, 3, 5, 5))
        solid[20:24, 30:33] = 255; other[20:24, 30:33] = 0
        expected = prepare_page_edits(w.current_base, solid, other, state=expected,
                                      dirty_bbox=(30, 20, 33, 24), fill_color=(50, 100, 200))
        selection = np.zeros((40, 60), np.uint8)
        cv2.fillPoly(selection, [np.array([[20, 10], [24, 10], [24, 13], [20, 13]])],
                     1, lineType=cv2.LINE_8)
        moved = (selection > 0) & (other > 0)
        other[moved] = 0; solid[moved] = 255
        expected = prepare_page_edits(w.current_base, solid, other, state=expected,
                                      dirty_bbox=(20, 10, 25, 14), fill_color=(3, 2, 1))
        other[2:4, 2:4] = 255; solid[2:4, 2:4] = 0
        expected = prepare_page_edits(w.current_base, solid, other, state=expected,
                                      dirty_bbox=(2, 2, 4, 4))
        prior_undo = len(w.undo_stack)
        result = self.apply(token, operations)
        self.assertTrue(result['saved'])
        self.assertEqual(len(w.undo_stack), prior_undo + 1)
        for key in expected:
            np.testing.assert_array_equal(w.current_page[key], expected[key])
        self.assertFalse(np.any((w.current_page['overlay'][:, :, 3] > 0)
                                & (w.current_page['other'] > 0)))
        stored = store.load_page(w.paths, w.current_img_path, (40, 60))
        for key in expected:
            np.testing.assert_array_equal(stored[key], expected[key])
        self.assertFalse(result['reviewed'])
        undone = self.control.dispatch('undo_page_edit', {'page_token': result['page_token']})
        self.assertTrue(undone['saved'])
        for key in before:
            np.testing.assert_array_equal(w.current_page[key], before[key])
            np.testing.assert_array_equal(store.load_page(w.paths, w.current_img_path, (40, 60))[key], before[key])

    def test_mcp_rect_and_polygon_match_mask_editor_view_signals(self):
        before = self.seed_page()
        w = self.window
        view = w.mask_view
        w.set_edit_mode('manual_other')
        w.set_edit_tool('rect')
        w.set_selection_combine_mode('restore_original')
        view._apply_rect((3, 3), (4, 4), Qt.MouseButton.LeftButton)

        w.set_selection_combine_mode('add')
        view.editStarted.emit()
        self.assertTrue(view._apply_rect((30, 20), (32, 23), Qt.MouseButton.LeftButton))
        view.maskEdited.emit(view.mask.copy())

        w.set_edit_mode('manual_other')
        w.set_edit_tool('lasso')
        w.set_selection_combine_mode('transfer_from_other')
        polygon = [[2, 2], [7, 2], [7, 7], [2, 7]]
        for point in polygon:
            view._add_lasso_point(tuple(point))
        self.assertTrue(view.finish_lasso())
        w.render_timer.stop()
        self.assertTrue(w.save_all_edit_masks(silent=True))
        gui_page = {key: value.copy() for key, value in w.current_page.items()}

        summary = store.save_page(w.paths, w.current_img_path, before)
        w.report['pages']['1.png'] = summary
        w.reload_current()
        w.render_timer.stop()
        token = self.bundle()['page_token']
        mcp_page = self.apply(token, [
            {'kind': 'restore', 'shape': {'rect': [3, 3, 2, 2]}},
            {'kind': 'add', 'target': 'other', 'shape': {'rect': [30, 20, 3, 4]}},
            {'kind': 'transfer', 'target': 'other', 'shape': {'polygon': polygon}},
        ])
        self.assertTrue(mcp_page['saved'])
        for key in ('overlay', 'other', 'edited'):
            np.testing.assert_array_equal(w.current_page[key], gui_page[key])

    def test_invalid_later_operation_and_stale_tokens_cannot_mutate(self):
        before = self.seed_page()
        token = self.bundle()['page_token']
        self.assertNotEqual(self.bundle(crop=[1, 1, 4, 4])['page_token'], token)
        good = {'kind': 'add', 'target': 'other', 'shape': {'rect': [1, 1, 3, 3]}}
        bad = {'kind': 'restore', 'shape': {'rect': [59, 39, 2, 2]}}
        undo_count = len(self.window.undo_stack)
        with self.assertRaises(ControlError) as error:
            self.apply(token, [good, bad])
        self.assertEqual(error.exception.code, 'bad_request')
        self.assertEqual(len(self.window.undo_stack), undo_count)
        for key in before:
            np.testing.assert_array_equal(self.window.current_page[key], before[key])
        with self.assertRaises(ControlError):
            self.apply(token, [good], crop=[59, 39, 2, 2])
        self.assertEqual(len(self.window.undo_stack), undo_count)
        with self.assertRaises(ControlError) as missing_color:
            self.apply(token, [{'kind': 'add', 'target': 'solid',
                                'shape': {'rect': [1, 1, 3, 3]}}])
        self.assertEqual(missing_color.exception.code, 'bad_request')
        self.window.is_mask_stroke_active = True
        try:
            with self.assertRaises(ControlError) as busy:
                self.apply(token, [good])
            self.assertEqual(busy.exception.code, 'busy')
        finally:
            self.window.is_mask_stroke_active = False
        changed = self.apply(token, [good])
        with self.assertRaises(ControlError) as old_after_edit:
            self.apply(token, [good])
        self.assertEqual(old_after_edit.exception.code, 'stale_page')
        self.assertTrue(changed['saved'])
        self.control.dispatch('go_to_page', {'index': 2})
        with self.assertRaises(ControlError) as error:
            self.apply(token, [good])
        self.assertEqual(error.exception.code, 'stale_page')

    def test_failed_save_is_reported_and_gui_undo_can_restore(self):
        before = self.seed_page()
        token = self.bundle()['page_token']
        with patch.object(store, 'save_page', side_effect=OSError('mock disk full')):
            with self.assertRaises(ControlError) as failed:
                self.apply(token, [{'kind': 'add', 'target': 'other',
                                    'shape': {'rect': [1, 1, 3, 3]}}])
        self.assertEqual(failed.exception.code, 'save_failed')
        self.assertIn('GUI 撤銷', str(failed.exception))
        self.assertFalse(np.array_equal(self.window.current_page['other'], before['other']))
        self.assertEqual(self.window.current_img_path, str(self.root / '1.png'))
        self.window.undo_mask()
        self.assertTrue(self.window.save_all_edit_masks(silent=True))
        for key in before:
            np.testing.assert_array_equal(self.window.current_page[key], before[key])
        self.assertFalse(store.is_page_reviewed(self.window.report['pages']['1.png']))

    def test_source_change_rejected_and_review_mark_requires_saved_processed_page(self):
        token = self.bundle()['page_token']
        with self.assertRaises(ControlError):
            self.control.dispatch('mark_page_reviewed', {'page_token': token})
        self.seed_page()
        token = self.bundle()['page_token']
        (self.root / '1.png').write_bytes(b'changed source')
        with self.assertRaises(ControlError) as error:
            self.control.dispatch('mark_page_reviewed', {'page_token': token})
        self.assertEqual(error.exception.code, 'stale_page')
        with self.assertRaises(ControlError):
            self.bundle()

    def test_external_page_archive_change_rejects_new_bundle(self):
        self.seed_page()
        token = self.bundle()['page_token']
        archive = store.page_path(self.window.paths, self.window.current_img_path)
        archive.write_bytes(archive.read_bytes() + b'external edit')
        with self.assertRaises(ControlError) as stale:
            self.control.dispatch('mark_page_reviewed', {'page_token': token})
        self.assertEqual(stale.exception.code, 'stale_page')
        with self.assertRaises(ControlError):
            self.bundle()

    def test_external_change_blocks_flush_of_pending_gui_masks(self):
        self.seed_page()
        token = self.bundle()['page_token']
        w = self.window
        w.current_manual_other[0, 0] = 255
        (self.root / '1.png').write_bytes(b'changed source')
        try:
            with patch.object(w, 'save_all_edit_masks', side_effect=AssertionError('must not save')):
                for method, params in (
                    ('get_review_bundle', {}),
                    ('apply_page_edits', {'page_token': token, 'operations': [
                        {'kind': 'restore', 'shape': {'rect': [1, 1, 1, 1]}}]}),
                    ('mark_page_reviewed', {'page_token': token}),
                ):
                    with self.subTest(method=method), self.assertRaises(ControlError) as raised:
                        self.control.dispatch(method, params)
                    self.assertEqual(raised.exception.code, 'stale_page')
        finally:
            w.current_manual_other[0, 0] = 0

    def test_external_archive_change_blocks_flush_of_pending_gui_masks(self):
        self.seed_page()
        w = self.window
        token = self.bundle()['page_token']
        w.current_manual_other[0, 0] = 255
        archive = store.page_path(w.paths, w.current_img_path)
        archive.write_bytes(archive.read_bytes() + b'external edit')
        try:
            with patch.object(w, 'save_all_edit_masks', side_effect=AssertionError('must not save')):
                with self.assertRaises(ControlError) as raised:
                    self.control.dispatch('apply_page_edits', {'page_token': token,
                        'operations': [{'kind': 'restore', 'shape': {'rect': [1, 1, 1, 1]}}]})
                self.assertEqual(raised.exception.code, 'stale_page')
        finally:
            w.current_manual_other[0, 0] = 0

    def test_review_advance_returns_next_full_page_and_edit_never_auto_reviews(self):
        self.seed_page()
        token = self.bundle(crop=[50, 30, 10, 10])['page_token']
        edited = self.apply(token, [{'kind': 'add', 'target': 'other',
                                     'shape': {'rect': [1, 1, 2, 2]}}])
        self.assertFalse(edited['reviewed'])
        result = self.control.dispatch('mark_page_reviewed',
                                       {'page_token': edited['page_token'], 'advance': True,
                                        'crop': [50, 30, 10, 10]})
        self.assertTrue(result['advanced'])
        self.assertTrue(result['marked_reviewed'])
        self.assertEqual(result['name'], '2.png')
        self.assertEqual(result['origin'], [0, 0])
        self.assertEqual(result['crop_width'], 60)

    def test_review_mark_reports_saved_even_if_next_image_cannot_load(self):
        self.seed_page()
        token = self.bundle()['page_token']
        (self.root / '2.png').write_bytes(b'broken image')
        result = self.control.dispatch('mark_page_reviewed',
                                       {'page_token': token, 'advance': True})
        self.assertTrue(result['saved'])
        self.assertTrue(result['marked_reviewed'])
        self.assertTrue(result['advanced'])
        self.assertTrue(result['next_page_unavailable'])
        self.assertIsNone(self.window.current_base)
        self.assertTrue(self.window.mask_view.pixmap_item.pixmap().isNull())
        self.assertTrue(self.window.preview_view.pixmap_item.pixmap().isNull())
        self.assertFalse(np.any(self.window.mask_view.mask))
        self.assertTrue(store.is_page_reviewed(self.window.report['pages']['1.png']))


if __name__ == '__main__':
    unittest.main()
