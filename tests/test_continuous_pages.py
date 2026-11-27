"""Exercise two simultaneously visible pages through the real editor viewport."""
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QSettings, Qt
from PySide6.QtGui import QColor, QImage, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox, QToolBar

import project_store as store
import solid_inpaint_ui as ui


class ContinuousPagesTests(unittest.TestCase):
    def setUp(self):
        self.app = QApplication.instance() or QApplication([])
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.files = []
        for number, (width, height, gray) in enumerate(
                [(360, 720, 190), (280, 660, 110), (320, 580, 210)], start=1):
            path = root / f'{number}.png'
            image = QImage(width, height, QImage.Format.Format_RGB32)
            image.fill(QColor(gray, gray, gray))
            self.assertTrue(image.save(str(path)))
            self.files.append(str(path))
        # A saved colored band distinguishes the edited neighbor from its raw source.
        state = store.empty_page((660, 280))
        state['overlay'][65:80, :, :] = (45, 95, 200, 255)
        store.save_page(ui._ensure_dirs(str(root)), self.files[1], state)
        settings = QSettings(str(root / 'ui.ini'), QSettings.Format.IniFormat)
        with patch.object(ui, 'QSettings', return_value=settings), \
                patch('solid_inpaint_control.start_control', return_value=None):
            self.window = ui.MainWindow()
        self.addCleanup(self.window.close)
        self.window.startup_restore_timer.stop()
        self.window.resize(1400, 890)
        self.window.show()
        self.app.processEvents()
        self.assertTrue(self.window.load_folder(str(root)))
        self.window.set_edit_mode('manual_other')
        self.window.set_edit_tool('brush')
        self.view = self.window.mask_view
        self.view.set_brush_radius(3)
        # Allow the initial window resize's delayed fit before simulating input.
        QTest.qWait(self.window.resize_fit_timer.interval() + 50)
        self.view.actual_size()
        self.app.processEvents()

    def wheel_to_center(self, scene_point):
        view = self.view
        center = view.viewport().rect().center()
        delta = center.y() - view.mapFromScene(scene_point).y()
        event = QWheelEvent(QPointF(center), QPointF(view.viewport().mapToGlobal(center)),
                            QPoint(0, delta), QPoint(), Qt.MouseButton.NoButton,
                            Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.ScrollUpdate, False)
        QApplication.sendEvent(view.viewport(), event)
        self.assertTrue(event.isAccepted())

    def scroll_to_seam(self):
        view = self.view
        self.wheel_to_center(QPointF(180, 720))
        self.assertEqual(self.window.current_img_path, self.files[0])
        edge = view.mapFromScene(180, 720)
        self.assertAlmostEqual(edge.y(), view.viewport().rect().center().y(), delta=2)
        return edge

    def expected_rgb(self, view, gray, band=False):
        if view is self.window.preview_view:
            return (200, 95, 45) if band else (gray, gray, gray)
        solid = np.full((1, 1), 255 if band else 0, np.uint8)
        bgr = ui._editor_mask_preview(np.full((1, 1, 3), gray, np.uint8), solid,
                                      np.zeros_like(solid), self.window.alpha,
                                      solid_color=self.window.mask_display_color)
        return tuple(map(int, bgr[0, 0, ::-1]))

    def second_top(self, view, x):
        # Inspect what the user sees, including delayed neighbor rendering.
        rgb = self.expected_rgb(view, 110)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            self.app.processEvents()
            pixmap = view.viewport().grab()
            image = pixmap.toImage()
            ratio = pixmap.devicePixelRatio()
            for y in range(view.viewport().height()):
                if image.pixelColor(round(x * ratio), round(y * ratio)).getRgb()[:3] == rgb:
                    return y
            QTest.qWait(10)
        self.fail('Second page did not appear on the visible canvas')

    def click(self, point):
        QTest.mouseClick(self.view.viewport(), Qt.MouseButton.LeftButton, pos=point)

    def read_mask(self, row):
        return ui.read_page_state(self.window.paths, self.files[row])['other']

    def browse_page(self, row):
        active = self.files.index(self.window.current_img_path)
        strip = self.view.page_strip
        self.wheel_to_center(strip.rect(row).center() - strip.rect(active).topLeft())
        self.app.processEvents()
        listing = self.window.list_widget
        self.assertEqual([listing.row(item) for item in listing.selectedItems()], [row])
        self.assertEqual(self.window.current_img_path, self.files[active])

    def scroll_away_from(self, target):
        """Scroll until the active and target pages are both completely offscreen."""
        active = self.files.index(self.window.current_img_path)
        other = next(row for row in range(len(self.files)) if row not in (active, target))
        self.view.actual_size()
        self.view.zoom_by(1.5, keep_center=True)
        self.browse_page(other)
        visible = self.view.mapToScene(self.view.viewport().rect()).boundingRect()
        self.assertEqual(self.view.page_strip.visible_rows(active, visible), [other])

    def assert_page_shown(self, row):
        self.assertEqual(self.window.current_img_path, self.files[row])
        self.assertEqual(self.window.list_widget.currentRow(), row)
        for view in (self.view, self.window.preview_view):
            center = view.viewport().rect().center()
            self.assertTrue(view.visible_image_rect().contains(view.mapToScene(center)))
            pixmap = view.viewport().grab()
            pos = center * pixmap.devicePixelRatio()
            color = pixmap.toImage().pixelColor(pos)
            self.assertEqual(color.getRgb()[:3], self.expected_rgb(view, (190, 110, 210)[row]))

    def click_list_row(self, row):
        listing = self.window.list_widget
        pos = listing.visualItemRect(listing.item(row)).center()
        QTest.mouseClick(listing.viewport(), Qt.MouseButton.LeftButton, pos=pos)

    def submit_jump(self, text):
        field = self.window.page_jump_input
        field.setFocus()
        field.selectAll()
        QTest.keyClicks(field, text)
        QTest.keyClick(field, Qt.Key.Key_Return)

    def test_list_and_return_jump_reveal_different_or_already_selected_page(self):
        for navigate, argument, target in (
            (self.click_list_row, 1, 1), (self.click_list_row, 1, 1),
            (self.submit_jump, '3', 2), (self.submit_jump, '3', 2),
            (self.submit_jump, '1.png', 0), (self.submit_jump, '1.png', 0),
        ):
            with self.subTest(action=navigate.__name__, argument=argument):
                self.scroll_away_from(target)
                navigate(argument)
                self.assert_page_shown(target)

    def test_previous_and_next_keys_reveal_their_target(self):
        for key, browse, target in ((Qt.Key.Key_D, 1, 2), (Qt.Key.Key_PageDown, 0, 1),
                                    (Qt.Key.Key_A, 2, 1), (Qt.Key.Key_PageUp, 1, 0)):
            with self.subTest(key=key):
                self.browse_page(browse)
                self.view.setFocus()
                QTest.keyClick(self.view, key)
                self.assert_page_shown(target)

    def test_highlight_crosses_seam_before_one_click_starts_editing_new_page(self):
        w = self.window
        strip = self.view.page_strip
        for row, y in ((0, strip.rect(0).bottom() - 20), (1, strip.rect(1).top() + 20)):
            self.wheel_to_center(QPointF(strip.rect(row).center().x(), y))
            self.app.processEvents()
            visible = self.view.mapToScene(self.view.viewport().rect()).boundingRect()
            self.assertEqual(strip.visible_rows(0, visible), [0, 1])
            self.assertEqual(w.list_widget.selectedItems(), [w.list_widget.item(row)])
            self.assertEqual(w.current_img_path, self.files[0])
        point = self.view.viewport().rect().center()
        self.click(point)
        self.assertEqual(w.current_img_path, self.files[1])
        x, y = self.view.image_point_from_view(point)
        self.assertEqual(w.current_manual_other[y, x], 255)
        self.assertFalse(self.read_mask(0).any())

    def test_toolbar_and_complete_page_controls_reveal_their_target(self):
        toolbar = self.window.findChild(QToolBar)
        actions = {action.text(): action for action in toolbar.actions()}
        for name, browse, target in (('下一頁', 1, 2), ('上一頁', 1, 0), ('下一頁', 0, 1)):
            with self.subTest(action=name, target=target):
                self.browse_page(browse)
                QTest.mouseClick(toolbar.widgetForAction(actions[name]), Qt.MouseButton.LeftButton)
                self.assert_page_shown(target)
        self.browse_page(0)
        QTest.mouseClick(self.window.review_next_btn, Qt.MouseButton.LeftButton)
        self.assert_page_shown(2)
        pages = store.load_project(self.window.paths['raw'])['pages']
        self.assertTrue(store.is_page_reviewed(pages['2.png']))

    def test_long_book_sidebar_follows_while_collapsed_and_restores_custom_width(self):
        w = self.window
        root = Path(self.files[0]).parent
        data = Path(self.files[0]).read_bytes()
        for number in range(4, 75):
            path = root / f'{number}.png'
            path.write_bytes(data)
            self.files.append(str(path))
        self.assertTrue(w.load_folder(str(root)))
        self.assertGreater(w.list_widget.verticalScrollBar().maximum(), 0)
        sizes = w.page_splitter.sizes()
        w.page_splitter.setSizes([sizes[0] + 77, sizes[1] - 77])
        self.app.processEvents()
        width = w.page_list_panel.width()
        canvas_width = self.view.viewport().width()
        self.view.actual_size()
        self.view.zoom_by(1.4, keep_center=True)
        scale = self.view.transform().m11()

        def assert_list_row_visible(row):
            rect = w.list_widget.visualItemRect(w.list_widget.item(row))
            self.assertTrue(w.list_widget.viewport().rect().contains(rect.center()))

        self.browse_page(65)
        self.assertGreater(w.list_widget.verticalScrollBar().value(), 0)
        assert_list_row_visible(65)
        QTest.mouseClick(w.page_list_toggle, Qt.MouseButton.LeftButton)
        self.app.processEvents()
        self.assertFalse(w.page_list_panel.isVisible())
        self.assertTrue(w.page_list_toggle.isVisible())
        self.assertGreater(self.view.viewport().width(), canvas_width)
        self.assertAlmostEqual(self.view.transform().m11(), scale)
        self.browse_page(7)
        center_before = self.view.mapToScene(self.view.viewport().rect().center())
        QTest.mouseClick(w.page_list_toggle, Qt.MouseButton.LeftButton)
        self.app.processEvents()
        self.assertTrue(w.page_list_panel.isVisible())
        self.assertAlmostEqual(w.page_list_panel.width(), width, delta=2)
        self.assertAlmostEqual(self.view.transform().m11(), scale)
        center_after = self.view.mapToScene(self.view.viewport().rect().center())
        self.assertAlmostEqual(center_after.y(), center_before.y(), delta=2)
        self.assertEqual(w.list_widget.selectedItems(), [w.list_widget.item(7)])
        assert_list_row_visible(7)
        self.assertEqual(w.current_img_path, self.files[0])
        self.click_list_row(7)
        self.assertEqual(w.current_img_path, self.files[7])

    def test_wheel_shows_adjacent_pages_in_both_panes_without_switching(self):
        edge = self.scroll_to_seam()
        tops = []
        for view in (self.view, self.window.preview_view):
            x = view.mapFromScene(180, 700).x()
            tops.append(self.second_top(view, x))
            snapshot = view.viewport().grab()
            ratio = snapshot.devicePixelRatio()
            image = snapshot.toImage()
            sample = image.pixelColor(round(x * ratio), round((edge.y() - 12) * ratio))
            self.assertEqual(sample.getRgb()[:3], self.expected_rgb(view, 190))
            band = image.pixelColor(round(x * ratio), round((tops[-1] + 72) * ratio))
            self.assertEqual(band.getRgb()[:3], self.expected_rgb(view, 110, band=True))
        self.assertAlmostEqual(tops[0], tops[1], delta=1)
        self.assertGreaterEqual(tops[0], edge.y())
        self.assertEqual(self.window.list_widget.currentRow(), 0)
        self.assertEqual(len(self.window.undo_stack), 0)
        # Explicit page jumps still reveal the selected page even if it was offscreen.
        self.window.list_widget.setCurrentRow(2)
        self.assertEqual(self.window.current_img_path, self.files[2])
        self.assertTrue(self.view.visible_image_rect().contains(QPointF(160, 290)))

    def test_new_press_switches_page_without_moving_view_and_saves_both_strokes(self):
        w = self.window
        self.view.zoom_by(1.25, keep_center=True)
        edge = self.scroll_to_seam()
        top = self.second_top(self.view, edge.x())
        zoom = self.view.transform().m11()
        first_dot = edge - QPoint(0, 25)
        second_dot = QPoint(edge.x(), top + 24)
        first_point = self.view.image_point_from_view(first_dot)
        self.click(first_dot)
        self.click(second_dot)
        self.assertEqual(w.current_img_path, self.files[1])
        self.assertEqual(w.list_widget.currentRow(), 1)
        self.assertAlmostEqual(self.view.transform().m11(), zoom)
        self.assertAlmostEqual(self.second_top(self.view, edge.x() - 14), top, delta=1)
        x, y = self.view.image_point_from_view(second_dot)
        self.assertAlmostEqual(y, 24 / zoom, delta=1)
        self.assertEqual(w.current_manual_other[y, x], 255)
        self.assertEqual(self.read_mask(0)[first_point[1], first_point[0]], 255)
        second_mask = w.current_manual_other.copy()
        self.click(first_dot + QPoint(20, 0))
        self.assertEqual(w.current_img_path, self.files[0])
        np.testing.assert_array_equal(self.read_mask(1), second_mask)
        self.assertEqual(w.current_manual_other[first_point[1], first_point[0]], 255)

    def test_drag_over_seam_is_clipped_and_never_draws_along_the_edge(self):
        edge = self.scroll_to_seam()
        top = self.second_top(self.view, edge.x())
        start = edge - QPoint(0, 12)
        outside = QPoint(edge.x(), top + 20)
        finish = outside + QPoint(45, 0)
        QTest.mousePress(self.view.viewport(), Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(self.view.viewport(), outside)
        QTest.mouseMove(self.view.viewport(), finish)
        QTest.mouseRelease(self.view.viewport(), Qt.MouseButton.LeftButton, pos=finish)
        self.assertEqual(self.window.current_img_path, self.files[0])
        self.assertTrue(self.window.flush_pending_edits())
        mask = self.read_mask(0)
        self.assertEqual(mask[-1, 180], 255)
        self.assertFalse(mask[-1, 190:].any())
        self.assertFalse(self.read_mask(1).any())

    def test_gap_and_preview_clicks_do_not_paint_or_continue_old_lasso(self):
        w = self.window
        edge = self.scroll_to_seam()
        top = self.second_top(self.view, edge.x())
        if top > edge.y() + 1:
            self.click(QPoint(edge.x(), (top + edge.y()) // 2))
            self.assertFalse(w.current_manual_other.any())
            self.assertEqual(w.list_widget.currentRow(), 0)
        w.set_edit_tool('lasso')
        self.click(edge - QPoint(20, 20))
        self.click(edge - QPoint(10, 40))
        self.assertEqual(len(self.view._lasso_points), 2)
        preview = w.preview_view
        x = preview.mapFromScene(180, 700).x()
        top = self.second_top(preview, x)
        QTest.mouseClick(preview.viewport(), Qt.MouseButton.LeftButton, pos=QPoint(x, top + 22))
        self.assertEqual(w.list_widget.currentRow(), 1)
        self.assertFalse(self.view._lasso_points)
        self.assertFalse(w.current_manual_other.any())
        self.assertFalse(self.read_mask(0).any())

    def test_failed_save_keeps_active_page_and_rejects_neighbor_press(self):
        w = self.window
        edge = self.scroll_to_seam()
        top = self.second_top(self.view, edge.x())
        with patch.object(store, 'save_page', side_effect=OSError('test disk failure')), \
                patch.object(QMessageBox, 'warning'):
            self.click(edge - QPoint(0, 20))
            before = w.current_manual_other.copy()
            self.assertTrue(before.any())
            self.click(QPoint(edge.x(), top + 20))
            self.assertEqual(w.current_img_path, self.files[0])
            self.assertEqual(w.list_widget.currentRow(), 0)
            np.testing.assert_array_equal(w.current_manual_other, before)
            self.assertFalse(self.read_mask(1).any())
        self.assertTrue(w.flush_pending_edits())


if __name__ == '__main__':
    unittest.main()
