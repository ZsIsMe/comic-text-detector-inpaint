#!/usr/bin/env python3
"""PySide6 UI for Solid Inpaint."""

from __future__ import annotations

import os
import os.path as osp
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, QPoint, QRectF, QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QCursor, QIcon, QImage, QKeySequence, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QCheckBox,
    QComboBox,
    QProgressBar,
    QRadioButton,
    QSlider,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QButtonGroup,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from detect_solid_inpaint_folder import (
    _compose_overlay_preview,
    _ensure_dirs,
    _background_sample_cache_path,
    _mask_hash,
    _manual_other_path,
    _manual_solid_path,
    _mask_path,
    _other_mask_path,
    _output_path,
    _save_background_sample_cache,
    _write_preview_pdf,
    build_report,
    create_detector,
    image_files_in_folder,
    iter_background_samples_from_mask,
    load_report,
    process_image_with_detector,
    REFINEMASK_ANNOTATION,
    regenerate_image_from_mask,
    regenerate_image_from_imported_mask,
    write_report,
)
from utils.io_utils import imread, imwrite


STATUS_OK = ''
STATUS_OTHER = '有 OTHER'
STATUS_FAILED = '失敗'
STATUS_TODO = '未處理'
MAX_RECENT_FOLDERS = 12
MAX_FOLDER_PROGRESS = 200
MAX_UNDO_STEPS = 30
DEFAULT_MASK_ALPHA_PERCENT = 80
DEFAULT_OTHER_MASK_PREVIEW_EXPAND_PX = 5
MAX_OTHER_MASK_PREVIEW_EXPAND_PX = 80
OTHER_MASK_DISPLAY_ALPHA = 0.38
OTHER_MASK_PREVIEW_RING_ALPHA = 0.16
DEFAULT_BRUSH_RADIUS = 24
MIN_BRUSH_RADIUS = 2
MAX_BRUSH_RADIUS = 160
DEFAULT_MAGIC_TOLERANCE = 28
MIN_MAGIC_TOLERANCE = 0
MAX_MAGIC_TOLERANCE = 100
DEFAULT_LOCAL_INTERSECT_OFFSET_PX = 0
MIN_LOCAL_INTERSECT_OFFSET_PX = -80
MAX_LOCAL_INTERSECT_OFFSET_PX = 80
CTD_SELECTION_PADDING_PX = 16
MASK_DISPLAY_COLORS: dict[str, tuple[int, int, int]] = {
    '白色': (255, 255, 255),
    '紅色': (60, 80, 255),
    '青色': (255, 245, 70),
    '黃色': (60, 220, 255),
    '綠色': (100, 230, 120),
}
EDIT_MODE_LABELS = {
    'mask': '自動',
    'manual_solid': '強制純色',
    'manual_other': '需要修改',
}
SELECTION_COMBINE_LABELS = {
    'add': '添加',
    'subtract': '減去',
    'local_intersect': '局部交集',
    'selection_inner': '選區內部',
    'transfer_from_other': '從其他轉入',
    'ctd_detect_selection': '添加CTD檢測選區',
}
EDIT_MODE_COLORS = {
    'mask': (255, 255, 255),
    'manual_solid': (100, 230, 120),
    'manual_other': (165, 110, 255),
}
VIEW_ZOOM_STEP = 1.15
VIEW_KEY_PAN_STEP = 80
APP_VERSION = '0.2.0'
APP_ICON_PATH = SCRIPT_DIR / 'icons' / 'tubai_icon_1024.png'
SAMPLE_RING_DISPLAY_ALPHA = 0.28
SAMPLE_RING_DISPLAY_COLOR_BGR = (70, 235, 255)


def _qimage_from_bgr(img: np.ndarray) -> QImage:
    if len(img.shape) == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()


def _qimage_from_rgba(img: np.ndarray) -> QImage:
    rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    h, w = rgba.shape[:2]
    return QImage(rgba.data, w, h, rgba.strides[0], QImage.Format.Format_RGBA8888).copy()


def _optional_imread(path: str, flags: int) -> np.ndarray | None:
    if not osp.isfile(path):
        return None
    try:
        return imread(path, flags)
    except Exception:
        return None


def _mask_from_optional_image(img: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if img is None or img.shape[:2] != shape:
        return np.zeros(shape, dtype=np.uint8)
    if len(img.shape) == 3 and img.shape[2] >= 4:
        img = img[:, :, 3]
    elif len(img.shape) == 3:
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    return np.where(img > 0, 255, 0).astype(np.uint8)


def _imwrite_transparent_mask(path: str, mask: np.ndarray, color_bgr: tuple[int, int, int]) -> None:
    mask_bin = np.where(mask > 0, 255, 0).astype(np.uint8)
    overlay = np.zeros((mask_bin.shape[0], mask_bin.shape[1], 4), dtype=np.uint8)
    active = mask_bin > 0
    overlay[active, 0] = color_bgr[0]
    overlay[active, 1] = color_bgr[1]
    overlay[active, 2] = color_bgr[2]
    overlay[active, 3] = 255
    imwrite(path, overlay)


def _write_edit_mask(paths: dict[str, str], img_path: str, mode: str, mask: np.ndarray) -> None:
    if mode == 'manual_solid':
        _imwrite_transparent_mask(_manual_solid_path(paths, img_path), mask, EDIT_MODE_COLORS['manual_solid'])
    elif mode == 'manual_other':
        _imwrite_transparent_mask(_manual_other_path(paths, img_path), mask, EDIT_MODE_COLORS['manual_other'])
    else:
        imwrite(_mask_path(paths, img_path), np.where(mask > 0, 255, 0).astype(np.uint8))


def _read_edit_masks_for_image(paths: dict[str, str], img_path: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    base = _optional_imread(img_path, cv2.IMREAD_UNCHANGED)
    if base is None:
        raise FileNotFoundError(f'無法讀取原圖：{img_path}')
    shape = base.shape[:2]
    masks = {
        'mask': _mask_from_optional_image(_optional_imread(_mask_path(paths, img_path), cv2.IMREAD_GRAYSCALE), shape),
        'manual_solid': _mask_from_optional_image(
            _optional_imread(_manual_solid_path(paths, img_path), cv2.IMREAD_UNCHANGED),
            shape,
        ),
        'manual_other': _mask_from_optional_image(
            _optional_imread(_manual_other_path(paths, img_path), cv2.IMREAD_UNCHANGED),
            shape,
        ),
    }
    return shape, masks


def _convert_image_edit_masks(paths: dict[str, str], img_path: str, target_mode: str) -> np.ndarray:
    shape, masks = _read_edit_masks_for_image(paths, img_path)
    combined = np.zeros(shape, dtype=np.uint8)
    for mask in masks.values():
        combined = cv2.bitwise_or(combined, np.where(mask > 0, 255, 0).astype(np.uint8))
    empty = np.zeros(shape, dtype=np.uint8)
    for mode in EDIT_MODE_LABELS:
        _write_edit_mask(paths, img_path, mode, combined if mode == target_mode else empty)
    return combined if target_mode == 'mask' else empty


def _load_background_sample_cache(paths: dict[str, str], img_path: str, mask: np.ndarray) -> np.ndarray | None:
    cache_path = _background_sample_cache_path(paths, img_path)
    if not osp.isfile(cache_path):
        return None
    try:
        data = np.load(cache_path)
        if int(data['mask_hash']) != _mask_hash(mask):
            return None
        sample = np.asarray(data['sample'], dtype=np.uint8)
        if sample.shape != mask.shape:
            return None
        return np.where(sample > 0, 255, 0).astype(np.uint8)
    except Exception:
        return None


def _mask_overlay_image(
    base: np.ndarray,
    mask: np.ndarray | None,
    alpha: float,
    color_bgr: tuple[int, int, int],
) -> np.ndarray:
    base_bgr = base[:, :, :3].copy() if len(base.shape) == 3 else cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    if mask is None:
        return base_bgr
    mask_active = mask > 0
    dimmed = (base_bgr.astype(np.float32) * max(0.0, 1.0 - alpha)).astype(np.uint8)
    color = np.zeros_like(base_bgr)
    color[:, :, 0] = color_bgr[0]
    color[:, :, 1] = color_bgr[1]
    color[:, :, 2] = color_bgr[2]
    blended = dimmed.copy()
    blended[mask_active] = (
        base_bgr[mask_active].astype(np.float32) * (1.0 - alpha)
        + color[mask_active].astype(np.float32) * alpha
    ).astype(np.uint8)
    return blended


def _overlay_mask_on_bgr(
    base_bgr: np.ndarray,
    mask: np.ndarray | None,
    alpha: float,
    color_bgr: tuple[int, int, int],
) -> np.ndarray:
    if mask is None:
        return base_bgr
    active = mask > 0
    if not np.any(active):
        return base_bgr
    color = np.zeros_like(base_bgr)
    color[:, :, 0] = color_bgr[0]
    color[:, :, 1] = color_bgr[1]
    color[:, :, 2] = color_bgr[2]
    output = base_bgr.copy()
    output[active] = (
        output[active].astype(np.float32) * (1.0 - alpha)
        + color[active].astype(np.float32) * alpha
    ).astype(np.uint8)
    return output


def _expanded_mask_ring(mask: np.ndarray | None, radius: int) -> np.ndarray | None:
    if mask is None or radius <= 0:
        return None
    mask_bin = np.where(mask > 0, 255, 0).astype(np.uint8)
    if not np.any(mask_bin):
        return None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    expanded = cv2.dilate(mask_bin, kernel, iterations=1)
    return cv2.bitwise_and(expanded, cv2.bitwise_not(mask_bin))


def _overlay_transparent_mask_on_bgr(
    base_bgr: np.ndarray,
    mask: np.ndarray | None,
    alpha: float,
    color_bgr: tuple[int, int, int],
) -> np.ndarray:
    if mask is None:
        return base_bgr
    active = mask > 0
    if not np.any(active):
        return base_bgr
    color = np.zeros_like(base_bgr)
    color[:, :, 0] = color_bgr[0]
    color[:, :, 1] = color_bgr[1]
    color[:, :, 2] = color_bgr[2]
    output = base_bgr.copy()
    output[active] = (
        output[active].astype(np.float32) * (1.0 - alpha)
        + color[active].astype(np.float32) * alpha
    ).astype(np.uint8)
    return output


def _make_magic_cursor() -> QCursor:
    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(QColor(8, 12, 16, 210), 4))
    painter.drawLine(8, 25, 21, 12)
    painter.setPen(QPen(QColor('#e9fffb'), 2))
    painter.drawLine(8, 25, 21, 12)
    painter.setPen(QPen(QColor(8, 12, 16, 230), 3))
    painter.drawLine(22, 4, 22, 9)
    painter.drawLine(22, 15, 22, 20)
    painter.drawLine(14, 12, 19, 12)
    painter.drawLine(25, 12, 30, 12)
    painter.setPen(QPen(QColor('#f7d95c'), 1))
    painter.drawLine(22, 4, 22, 9)
    painter.drawLine(22, 15, 22, 20)
    painter.drawLine(14, 12, 19, 12)
    painter.drawLine(25, 12, 30, 12)
    painter.end()
    return QCursor(pixmap, 22, 12)


class ImageView(QGraphicsView):
    viewChanged = Signal()
    viewportResized = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene().addItem(self.pixmap_item)
        self.setBackgroundBrush(QColor('#0b0d10'))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._zoom = 1.0
        self._suppress_view_changed = False
        self.horizontalScrollBar().valueChanged.connect(self._emit_view_changed)
        self.verticalScrollBar().valueChanged.connect(self._emit_view_changed)

    def _emit_view_changed(self) -> None:
        if not self._suppress_view_changed:
            self.viewChanged.emit()

    def set_qimage(self, image: QImage | None, keep_view: bool = False) -> None:
        if image is None:
            self.pixmap_item.setPixmap(QPixmap())
            self.scene().setSceneRect(0, 0, 1, 1)
            self._emit_view_changed()
            return
        old_transform = self.transform()
        old_zoom = self._zoom
        old_horizontal_scroll = self.horizontalScrollBar().value()
        old_vertical_scroll = self.verticalScrollBar().value()
        pixmap = QPixmap.fromImage(image)
        self.pixmap_item.setPixmap(pixmap)
        self.scene().setSceneRect(pixmap.rect())
        if keep_view:
            self.setTransform(old_transform)
            self._zoom = old_zoom
            self.horizontalScrollBar().setValue(old_horizontal_scroll)
            self.verticalScrollBar().setValue(old_vertical_scroll)
        else:
            self.fit()
        self._emit_view_changed()

    def fit(self) -> None:
        pixmap = self.pixmap_item.pixmap()
        if pixmap.isNull():
            return
        self.resetTransform()
        self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom = self.transform().m11()
        self._emit_view_changed()

    def actual_size(self) -> None:
        self.resetTransform()
        self._zoom = 1.0
        self._emit_view_changed()

    def zoom_by(
        self,
        factor: float,
        keep_center: bool = False,
        anchor_pos: QPoint | None = None,
    ) -> None:
        if self.pixmap_item.pixmap().isNull():
            return
        old_center = self.mapToScene(self.viewport().rect().center()) if keep_center else None
        old_anchor_scene = self.mapToScene(anchor_pos) if anchor_pos is not None else None
        self._zoom *= factor
        self.scale(factor, factor)
        if old_anchor_scene is not None and anchor_pos is not None:
            new_anchor_view = self.mapFromScene(old_anchor_scene)
            delta = new_anchor_view - anchor_pos
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() + delta.y())
        if old_center is not None:
            self.centerOn(old_center)
        self._emit_view_changed()

    def pan_by(self, dx: int, dy: int) -> None:
        if self.pixmap_item.pixmap().isNull():
            return
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + dx)
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() + dy)
        self._emit_view_changed()

    def copy_view_from(self, source: 'ImageView') -> None:
        if self.pixmap_item.pixmap().isNull() or source.pixmap_item.pixmap().isNull():
            return
        center = source.mapToScene(source.viewport().rect().center())
        self._suppress_view_changed = True
        try:
            self.setTransform(source.transform())
            self._zoom = source._zoom
            self.centerOn(center)
            self.horizontalScrollBar().setValue(source.horizontalScrollBar().value())
            self.verticalScrollBar().setValue(source.verticalScrollBar().value())
        finally:
            self._suppress_view_changed = False

    def visible_image_rect(self) -> QRectF:
        pixmap = self.pixmap_item.pixmap()
        if pixmap.isNull():
            return QRectF()
        visible = self.mapToScene(self.viewport().rect()).boundingRect()
        image_rect = QRectF(pixmap.rect())
        return visible.intersected(image_rect)

    def center_on_image_point(self, x: float, y: float) -> None:
        if self.pixmap_item.pixmap().isNull():
            return
        self.centerOn(x, y)
        self._emit_view_changed()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.viewportResized.emit()
        self._emit_view_changed()

    def wheelEvent(self, event) -> None:
        if self.pixmap_item.pixmap().isNull():
            return
        pixel_delta = event.pixelDelta()
        angle_delta = event.angleDelta()
        modifiers = event.modifiers()
        if modifiers & Qt.KeyboardModifier.AltModifier:
            zoom_delta = pixel_delta.y() if not pixel_delta.isNull() else angle_delta.y()
            if zoom_delta != 0:
                factor = VIEW_ZOOM_STEP if zoom_delta > 0 else 1 / VIEW_ZOOM_STEP
                self.zoom_by(factor, anchor_pos=event.position().toPoint())
            event.accept()
            return
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            horizontal_delta = pixel_delta.x() if not pixel_delta.isNull() and pixel_delta.x() else angle_delta.x()
            vertical_delta = pixel_delta.y() if not pixel_delta.isNull() and pixel_delta.y() else angle_delta.y()
            pan_delta = horizontal_delta if horizontal_delta else vertical_delta
            if pan_delta:
                self.pan_by(-pan_delta, 0)
            event.accept()
            return
        if not pixel_delta.isNull():
            self.pan_by(-pixel_delta.x(), -pixel_delta.y())
            event.accept()
            return
        if not angle_delta.isNull():
            self.pan_by(-angle_delta.x(), -angle_delta.y())
        event.accept()


class PassivePreviewView(ImageView):
    def __init__(self) -> None:
        super().__init__()
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setInteractive(False)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.viewport().setCursor(Qt.CursorShape.ArrowCursor)

    def mousePressEvent(self, event) -> None:
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        event.accept()

    def wheelEvent(self, event) -> None:
        event.accept()

    def keyPressEvent(self, event) -> None:
        event.accept()

    def keyReleaseEvent(self, event) -> None:
        event.accept()


class RubberBandRectItem(QGraphicsRectItem):
    def __init__(self) -> None:
        super().__init__()
        self.erase_all_style = False

    def set_erase_all_style(self, enabled: bool) -> None:
        if self.erase_all_style == enabled:
            return
        self.erase_all_style = enabled
        self.update()

    def boundingRect(self) -> QRectF:
        rect = super().boundingRect()
        if not self.erase_all_style:
            return rect
        return rect.adjusted(-8, -8, 8, 8)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        if not self.erase_all_style:
            super().paint(painter, option, widget)
            return

        rect = self.rect()
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        glow_pen = QPen(QColor(115, 135, 150, 95), 8, Qt.PenStyle.SolidLine)
        glow_pen.setCosmetic(True)
        painter.setPen(glow_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect.adjusted(-2, -2, 2, 2))

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(154, 167, 179, 62))
        painter.drawRect(rect)

        painter.save()
        painter.setClipRect(rect)
        hatch_pen = QPen(QColor(182, 194, 204, 145), 1)
        hatch_pen.setCosmetic(True)
        painter.setPen(hatch_pen)
        step = 10
        left = int(rect.left())
        right = int(rect.right())
        top = int(rect.top())
        bottom = int(rect.bottom())
        height = max(1, bottom - top)
        for x in range(left - height - step, right + step, step):
            painter.drawLine(x, bottom, x + height, top)
        painter.restore()

        border_pen = QPen(QColor('#9aa7b3'), 3, Qt.PenStyle.DashLine)
        border_pen.setCosmetic(True)
        painter.setPen(border_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect)
        painter.restore()


class MaskEditorView(ImageView):
    editStarted = Signal()
    maskEdited = Signal(object)
    selectionCreated = Signal(object)
    eraseAllMasksRequested = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.tool = 'brush'
        self.selection_combine_mode = 'add'
        self.brush_radius = DEFAULT_BRUSH_RADIUS
        self.magic_tolerance = DEFAULT_MAGIC_TOLERANCE
        self.local_intersect_offset_px = DEFAULT_LOCAL_INTERSECT_OFFSET_PX
        self.mask: np.ndarray | None = None
        self.source_bgr: np.ndarray | None = None
        self.image_shape: tuple[int, int] | None = None
        self._active_button: Qt.MouseButton | None = None
        self._drag_start: tuple[int, int] | None = None
        self._last_brush_point: tuple[int, int] | None = None
        self._brush_line_start: tuple[int, int] | None = None
        self._brush_line_preview: QGraphicsLineItem | None = None
        self._erase_all_drag = False
        self._panning = False
        self._pan_last_pos: QPoint | None = None
        self._rubber_band: RubberBandRectItem | None = None
        self._brush_cursor: QGraphicsEllipseItem | None = None
        self._edit_started = False
        self._magic_cursor = _make_magic_cursor()
        self._rect_pen_add = QPen(QColor('#e9fffb'), 2, Qt.PenStyle.DashLine)
        self._rect_pen_remove = QPen(QColor('#ff8f8f'), 2, Qt.PenStyle.DashLine)
        self._rect_pen_intersect = QPen(QColor('#ffd86f'), 2, Qt.PenStyle.DashLine)
        self._rect_pen_detect = QPen(QColor('#70bdff'), 2, Qt.PenStyle.DashLine)
        for pen in (
            self._rect_pen_add,
            self._rect_pen_remove,
            self._rect_pen_intersect,
            self._rect_pen_detect,
        ):
            pen.setCosmetic(True)
        self._rect_brush_default = QBrush(QColor(255, 255, 255, 30))
        self._brush_line_pen_add = QPen(QColor('#e9fffb'), 2, Qt.PenStyle.DashLine)
        self._brush_line_pen_remove = QPen(QColor('#ff8f8f'), 2, Qt.PenStyle.DashLine)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)

    def set_mask(
        self,
        mask: np.ndarray | None,
        shape: tuple[int, int],
        reset_brush_line: bool = False,
    ) -> None:
        self.image_shape = shape
        if reset_brush_line:
            self._brush_line_start = None
            self._clear_brush_line_preview()
        if mask is None:
            self.mask = np.zeros(shape, dtype=np.uint8)
        else:
            self.mask = np.where(mask > 0, 255, 0).astype(np.uint8)

    def set_source_image(self, image: np.ndarray | None) -> None:
        if image is None:
            self.source_bgr = None
            return
        if len(image.shape) == 2:
            self.source_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            self.source_bgr = image[:, :, :3].copy()

    def set_tool(self, tool: str) -> None:
        if tool != 'brush':
            self._brush_line_start = None
            self._clear_brush_line_preview()
        self.tool = tool
        self._clear_rubber_band()
        self._update_brush_cursor_visibility()
        self._update_tool_cursor()

    def set_selection_combine_mode(self, mode: str) -> None:
        if mode in SELECTION_COMBINE_LABELS:
            self.selection_combine_mode = mode

    def set_brush_radius(self, radius: int) -> None:
        self.brush_radius = max(MIN_BRUSH_RADIUS, min(MAX_BRUSH_RADIUS, int(radius)))
        self._move_brush_cursor_to_last_position()

    def set_magic_tolerance(self, tolerance: int) -> None:
        self.magic_tolerance = max(MIN_MAGIC_TOLERANCE, min(MAX_MAGIC_TOLERANCE, int(tolerance)))

    def set_local_intersect_offset(self, offset_px: int) -> None:
        self.local_intersect_offset_px = max(
            MIN_LOCAL_INTERSECT_OFFSET_PX,
            min(MAX_LOCAL_INTERSECT_OFFSET_PX, int(offset_px)),
        )

    def image_point_from_view(self, pos: QPoint, clamp: bool = False) -> tuple[int, int] | None:
        if self.image_shape is None:
            return None
        pixmap = self.pixmap_item.pixmap()
        if pixmap.isNull():
            return None
        scene_pos = self.mapToScene(pos)
        item_pos = self.pixmap_item.mapFromScene(scene_pos)
        x = int(round(item_pos.x()))
        y = int(round(item_pos.y()))
        height, width = self.image_shape
        if clamp:
            return max(0, min(width - 1, x)), max(0, min(height - 1, y))
        if x < 0 or y < 0 or x >= width or y >= height:
            return None
        return x, y

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._is_pan_modifier(event.modifiers()):
            self._panning = True
            self._pan_last_pos = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() not in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            super().mousePressEvent(event)
            return
        point = self.image_point_from_view(event.position().toPoint())
        if point is None or self.mask is None:
            super().mousePressEvent(event)
            return
        if event.button() == Qt.MouseButton.RightButton:
            self._active_button = event.button()
            self._drag_start = point
            self._last_brush_point = point
            self._brush_line_start = None
            self._erase_all_drag = True
            self._edit_started = False
            self._clear_brush_line_preview()
            self._update_rubber_band(point, point, event.button())
            event.accept()
            return
        if self.tool == 'magic':
            if self._apply_magic_wand(point, event.button()):
                self._begin_edit_once()
                if self.mask is not None:
                    self.maskEdited.emit(self.mask.copy())
            self._edit_started = False
            event.accept()
            return
        if self.tool == 'brush':
            self._active_button = event.button()
            self._last_brush_point = point
            self._edit_started = False
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self._brush_line_start = point
                self._update_brush_line_preview(point, point, event.button())
            else:
                self._begin_edit_once()
                self._paint_brush(point, event.button())
                self.maskEdited.emit(self.mask.copy())
            event.accept()
            return
        self._active_button = event.button()
        self._drag_start = point
        self._last_brush_point = point
        self._edit_started = False
        self._update_rubber_band(point, point, event.button())
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._panning:
            current = event.position().toPoint()
            if self._pan_last_pos is not None:
                delta = current - self._pan_last_pos
                self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
                self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self._pan_last_pos = current
            event.accept()
            return
        hover_point = self.image_point_from_view(event.position().toPoint())
        self._update_brush_cursor(hover_point)
        if self._active_button is None:
            event.accept()
            return
        point = self.image_point_from_view(event.position().toPoint(), clamp=True)
        if point is None or self.mask is None:
            event.accept()
            return
        if self._erase_all_drag:
            self._update_rubber_band(self._drag_start or point, point, self._active_button)
            event.accept()
            return
        if self.tool == 'brush':
            if self._brush_line_start is not None:
                self._update_brush_line_preview(self._brush_line_start, point, self._active_button)
            else:
                self._begin_edit_once()
                self._paint_line(self._last_brush_point or point, point, self._active_button)
                self._last_brush_point = point
                self.maskEdited.emit(self.mask.copy())
        else:
            self._update_rubber_band(self._drag_start or point, point, self._active_button)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if self._panning and event.button() == Qt.MouseButton.LeftButton:
            self._panning = False
            self._pan_last_pos = None
            self._update_tool_cursor()
            event.accept()
            return
        if event.button() != self._active_button:
            super().mouseReleaseEvent(event)
            return
        point = self.image_point_from_view(event.position().toPoint(), clamp=True)
        if self._erase_all_drag and self.mask is not None and self._drag_start is not None and point is not None:
            selection = self._selection_from_rect(self._drag_start, point)
            if selection is not None:
                self.eraseAllMasksRequested.emit(selection)
        elif self.tool == 'brush' and self.mask is not None and self._brush_line_start is not None and point is not None:
            self._begin_edit_once()
            self._paint_line(self._brush_line_start, point, self._active_button)
            self.maskEdited.emit(self.mask.copy())
        if not self._erase_all_drag and self.tool == 'rect' and self.mask is not None and self._drag_start is not None and point is not None:
            if self._apply_rect(self._drag_start, point, self._active_button):
                self._begin_edit_once()
                if self.mask is not None:
                    self.maskEdited.emit(self.mask.copy())
        self._active_button = None
        self._drag_start = None
        self._last_brush_point = None
        self._brush_line_start = None
        self._erase_all_drag = False
        self._edit_started = False
        self._clear_brush_line_preview()
        self._clear_rubber_band()
        event.accept()

    def leaveEvent(self, event) -> None:
        self._update_brush_cursor(None)
        super().leaveEvent(event)

    def _is_pan_modifier(self, modifiers: Qt.KeyboardModifier) -> bool:
        return bool(
            modifiers & Qt.KeyboardModifier.MetaModifier
            or modifiers & Qt.KeyboardModifier.ControlModifier
        )

    def _begin_edit_once(self) -> None:
        if self._edit_started:
            return
        self._edit_started = True
        self.editStarted.emit()

    def _paint_brush(self, point: tuple[int, int], button: Qt.MouseButton) -> None:
        if self.mask is None:
            return
        color = self._brush_color_for_button(button)
        cv2.circle(self.mask, point, self.brush_radius, color, thickness=-1, lineType=cv2.LINE_8)

    def _brush_color_for_button(self, button: Qt.MouseButton) -> int:
        operation = self._selection_operation_for_button(button)
        return 0 if operation == 'subtract' else 255

    def _paint_line(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        button: Qt.MouseButton,
    ) -> None:
        if self.mask is None:
            return
        color = self._brush_color_for_button(button)
        cv2.line(self.mask, start, end, color, thickness=self.brush_radius * 2, lineType=cv2.LINE_8)
        cv2.circle(self.mask, end, self.brush_radius, color, thickness=-1, lineType=cv2.LINE_8)

    def _apply_rect(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        button: Qt.MouseButton,
    ) -> bool:
        if self.mask is None:
            return False
        selection = self._selection_from_rect(start, end)
        if selection is None:
            return False
        old_mask = self.mask.copy()
        if self._should_emit_selection(button):
            self.selectionCreated.emit(selection.copy())
            return False
        self._apply_selection(selection, button)
        return not np.array_equal(old_mask, self.mask)

    def _selection_from_rect(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> np.ndarray | None:
        if self.mask is None:
            return None
        x1, x2 = sorted((start[0], end[0]))
        y1, y2 = sorted((start[1], end[1]))
        if x2 < x1 or y2 < y1:
            return None
        selection = np.zeros(self.mask.shape[:2], dtype=bool)
        selection[y1:y2 + 1, x1:x2 + 1] = True
        return selection

    def _apply_magic_wand(
        self,
        point: tuple[int, int],
        button: Qt.MouseButton,
    ) -> bool:
        if self.mask is None or self.source_bgr is None:
            return False
        height, width = self.mask.shape[:2]
        if not (0 <= point[0] < width and 0 <= point[1] < height):
            return False
        flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
        tolerance = int(self.magic_tolerance)
        diff = (tolerance, tolerance, tolerance)
        flags = (
            8
            | cv2.FLOODFILL_FIXED_RANGE
            | cv2.FLOODFILL_MASK_ONLY
            | (255 << 8)
        )
        cv2.floodFill(
            self.source_bgr.copy(),
            flood_mask,
            point,
            (0, 0, 0),
            diff,
            diff,
            flags,
        )
        selection = flood_mask[1:height + 1, 1:width + 1] > 0
        if not np.any(selection):
            return False
        if self._should_emit_selection(button):
            self.selectionCreated.emit(selection.copy())
            return False
        old_mask = self.mask.copy()
        self._apply_selection(selection, button)
        return not np.array_equal(old_mask, self.mask)

    def _apply_selection(self, selection: np.ndarray, button: Qt.MouseButton) -> None:
        if self.mask is None:
            return
        operation = self._selection_operation_for_button(button)
        if operation == 'add':
            self.mask[selection] = 255
            return
        if operation == 'subtract':
            self.mask[selection] = 0
            return
        if operation == 'local_intersect':
            self._apply_local_intersection(selection)
            return
        if operation == 'selection_inner' and self.tool == 'magic':
            self._apply_selection_inner(selection)

    def _selection_operation_for_button(self, button: Qt.MouseButton) -> str:
        if button == Qt.MouseButton.RightButton:
            return 'subtract'
        return self.selection_combine_mode

    def _should_emit_selection(self, button: Qt.MouseButton) -> bool:
        return (
            button == Qt.MouseButton.LeftButton
            and self.selection_combine_mode in ('transfer_from_other', 'ctd_detect_selection')
        )

    def _apply_local_intersection(self, selection: np.ndarray) -> None:
        if self.mask is None:
            return
        current = self.mask > 0
        overlap = current & selection
        if not np.any(overlap):
            return
        _, labels = cv2.connectedComponents(current.astype(np.uint8), connectivity=8)
        hit_labels = np.unique(labels[overlap])
        hit_labels = hit_labels[hit_labels != 0]
        if hit_labels.size == 0:
            return
        touched_components = np.isin(labels, hit_labels)
        local_result = touched_components & selection
        local_result = self._offset_local_intersection(local_result)
        next_mask = current.copy()
        next_mask[touched_components] = False
        next_mask |= local_result
        self.mask[:, :] = np.where(next_mask, 255, 0).astype(np.uint8)

    def _apply_selection_inner(self, selection: np.ndarray) -> None:
        if self.mask is None:
            return
        holes = self._selection_holes(selection)
        if np.any(holes):
            self.mask[holes] = 255

    def _selection_holes(self, selection: np.ndarray) -> np.ndarray:
        selection = np.asarray(selection, dtype=bool)
        if not np.any(selection):
            return np.zeros(selection.shape, dtype=bool)
        inverse = ~selection
        if not np.any(inverse):
            return np.zeros(selection.shape, dtype=bool)
        _, labels = cv2.connectedComponents(inverse.astype(np.uint8), connectivity=8)
        border_labels = np.unique(
            np.concatenate((
                labels[0, :],
                labels[-1, :],
                labels[:, 0],
                labels[:, -1],
            ))
        )
        return inverse & ~np.isin(labels, border_labels)

    def _offset_local_intersection(self, local_result: np.ndarray) -> np.ndarray:
        offset_px = int(self.local_intersect_offset_px)
        if offset_px == 0 or not np.any(local_result):
            return local_result
        kernel_size = abs(offset_px) * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        src = local_result.astype(np.uint8)
        if offset_px > 0:
            return cv2.dilate(src, kernel, iterations=1) > 0
        return cv2.erode(src, kernel, iterations=1) > 0

    def _update_rubber_band(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        button: Qt.MouseButton,
    ) -> None:
        x1, x2 = sorted((start[0], end[0]))
        y1, y2 = sorted((start[1], end[1]))
        if self._rubber_band is None:
            self._rubber_band = RubberBandRectItem()
            self.scene().addItem(self._rubber_band)
        if button == Qt.MouseButton.RightButton or self._erase_all_drag:
            self._rubber_band.set_erase_all_style(True)
            self._rubber_band.setRect(x1, y1, max(1, x2 - x1), max(1, y2 - y1))
            return
        self._rubber_band.set_erase_all_style(False)
        self._rubber_band.setBrush(self._rect_brush_default)
        operation = self._selection_operation_for_button(button)
        if operation == 'subtract':
            self._rubber_band.setPen(self._rect_pen_remove)
        elif operation == 'ctd_detect_selection':
            self._rubber_band.setPen(self._rect_pen_detect)
        elif operation in ('local_intersect', 'selection_inner', 'transfer_from_other'):
            self._rubber_band.setPen(self._rect_pen_intersect)
        else:
            self._rubber_band.setPen(self._rect_pen_add)
        self._rubber_band.setRect(x1, y1, max(1, x2 - x1), max(1, y2 - y1))

    def _clear_rubber_band(self) -> None:
        if self._rubber_band is not None:
            self.scene().removeItem(self._rubber_band)
            self._rubber_band = None

    def _update_brush_line_preview(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        button: Qt.MouseButton,
    ) -> None:
        if self._brush_line_preview is None:
            self._brush_line_preview = QGraphicsLineItem()
            self._brush_line_preview.setZValue(11)
            self.scene().addItem(self._brush_line_preview)
        pen = self._brush_line_pen_remove if self._brush_color_for_button(button) == 0 else self._brush_line_pen_add
        self._brush_line_preview.setPen(pen)
        self._brush_line_preview.setLine(start[0], start[1], end[0], end[1])

    def _clear_brush_line_preview(self) -> None:
        if self._brush_line_preview is not None:
            self.scene().removeItem(self._brush_line_preview)
            self._brush_line_preview = None

    def _update_brush_cursor(self, point: tuple[int, int] | None) -> None:
        if self.tool != 'brush' or point is None:
            if self._brush_cursor is not None:
                self._brush_cursor.setVisible(False)
            return
        if self._brush_cursor is None:
            self._brush_cursor = QGraphicsEllipseItem()
            self._brush_cursor.setBrush(QColor(233, 255, 251, 28))
            self._brush_cursor.setPen(QPen(Qt.PenStyle.NoPen))
            self._brush_cursor.setZValue(10)
            self.scene().addItem(self._brush_cursor)
        radius = self.brush_radius
        self._brush_cursor.setRect(point[0] - radius, point[1] - radius, radius * 2, radius * 2)
        self._brush_cursor.setVisible(True)

    def _update_brush_cursor_visibility(self) -> None:
        if self.tool != 'brush' and self._brush_cursor is not None:
            self._brush_cursor.setVisible(False)

    def _move_brush_cursor_to_last_position(self) -> None:
        if self._brush_cursor is None or not self._brush_cursor.isVisible():
            return
        rect = self._brush_cursor.rect()
        center = rect.center()
        self._update_brush_cursor((int(center.x()), int(center.y())))

    def _update_tool_cursor(self) -> None:
        if self._panning:
            return
        if self.tool == 'magic':
            self.setCursor(self._magic_cursor)
        else:
            self.unsetCursor()


class FolderWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(
        self,
        folder: str,
        mode: str,
        image_paths: list[str] | None = None,
        imported_mask_dir: str = '',
        imported_mask_mode: str = '',
    ) -> None:
        super().__init__()
        self.folder = folder
        self.mode = mode
        self.image_paths = image_paths
        self.imported_mask_dir = imported_mask_dir
        self.imported_mask_mode = imported_mask_mode

    def run(self) -> None:
        try:
            paths = _ensure_dirs(self.folder)
            imglist = self.image_paths or image_files_in_folder(self.folder)
            detector = create_detector() if self.mode == 'detect' else None
            existing = load_report(paths)
            pages = dict(existing.get('pages', {}))
            total = len(imglist)
            for index, img_path in enumerate(imglist, start=1):
                name = osp.basename(img_path)
                self.progress.emit(index, total, name)
                try:
                    if self.mode == 'detect':
                        pages[name] = process_image_with_detector(img_path, paths, detector)
                    elif self.mode == 'imported_mask':
                        pages[name] = regenerate_image_from_imported_mask(
                            img_path,
                            paths,
                            self.imported_mask_dir,
                            self.imported_mask_mode,
                        )
                    else:
                        pages[name] = regenerate_image_from_mask(img_path, paths)
                except Exception as exc:
                    pages[name] = {'error': str(exc)}
            report = build_report(self.folder, paths, image_files_in_folder(self.folder), pages)
            write_report(paths, report)
            self.finished.emit(report)
        except Exception as exc:
            self.failed.emit(str(exc))


class ConvertMasksWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, folder: str, paths: dict[str, str], imglist: list[str], target_mode: str) -> None:
        super().__init__()
        self.folder = folder
        self.paths = dict(paths)
        self.imglist = list(imglist)
        self.target_mode = target_mode

    def run(self) -> None:
        try:
            existing = load_report(self.paths)
            pages = dict(existing.get('pages', {}))
            total = len(self.imglist)
            for index, img_path in enumerate(self.imglist, start=1):
                name = osp.basename(img_path)
                self.progress.emit(index, total, name)
                try:
                    auto_mask = _convert_image_edit_masks(self.paths, img_path, self.target_mode)
                    pages[name] = regenerate_image_from_mask(img_path, self.paths, auto_mask)
                except Exception as exc:
                    pages[name] = {'error': str(exc)}
            report = build_report(self.folder, self.paths, self.imglist, pages)
            write_report(self.paths, report)
            self.finished.emit(report)
        except Exception as exc:
            self.failed.emit(str(exc))


class PageRegenerateWorker(QObject):
    finished = Signal(str, dict)
    failed = Signal(str, str)

    def __init__(self, folder: str, paths: dict[str, str], imglist: list[str], img_path: str, mask: np.ndarray) -> None:
        super().__init__()
        self.folder = folder
        self.paths = paths
        self.imglist = imglist
        self.img_path = img_path
        self.mask = np.where(mask > 0, 255, 0).astype(np.uint8)

    def run(self) -> None:
        try:
            existing = load_report(self.paths)
            pages = dict(existing.get('pages', {}))
            summary = regenerate_image_from_mask(self.img_path, self.paths, self.mask)
            pages[osp.basename(self.img_path)] = summary
            report = build_report(self.folder, self.paths, self.imglist, pages)
            write_report(self.paths, report)
            self.finished.emit(self.img_path, report)
        except Exception as exc:
            self.failed.emit(self.img_path, str(exc))


class ConvertMasksDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle('轉換選區')
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        target_label = QLabel('將全部選區轉為')
        layout.addWidget(target_label)
        target_layout = QHBoxLayout()
        self.target_group = QButtonGroup(self)
        self.target_buttons: dict[str, QRadioButton] = {}
        for mode, label in EDIT_MODE_LABELS.items():
            button = QRadioButton(label)
            if mode == 'mask':
                button.setChecked(True)
            self.target_group.addButton(button)
            self.target_buttons[mode] = button
            target_layout.addWidget(button)
        target_layout.addStretch()
        layout.addLayout(target_layout)

        scope_label = QLabel('作用範圍')
        layout.addWidget(scope_label)
        scope_layout = QHBoxLayout()
        self.current_page_radio = QRadioButton('當前頁面')
        self.current_page_radio.setChecked(True)
        self.all_pages_radio = QRadioButton('全部頁面')
        self.scope_group = QButtonGroup(self)
        self.scope_group.addButton(self.current_page_radio)
        self.scope_group.addButton(self.all_pages_radio)
        scope_layout.addWidget(self.current_page_radio)
        scope_layout.addWidget(self.all_pages_radio)
        scope_layout.addStretch()
        layout.addLayout(scope_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText('確認')
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText('取消')
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_target_mode(self) -> str:
        for mode, button in self.target_buttons.items():
            if button.isChecked():
                return mode
        return 'mask'

    def selected_scope(self) -> str:
        return 'all' if self.all_pages_radio.isChecked() else 'current'


class BackgroundSampleWorker(QObject):
    partial = Signal(int, str, object, object)
    finished = Signal(int, str, object, object)
    failed = Signal(int, str, object, str)

    def __init__(self, request_id: int, img_path: str, image: np.ndarray, mask: np.ndarray) -> None:
        super().__init__()
        self.request_id = request_id
        self.img_path = img_path
        self.image = image.copy()
        self.mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        self.mask_hash = _mask_hash(self.mask)
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    def run(self) -> None:
        try:
            sample = np.zeros(self.mask.shape, dtype=np.uint8)
            for block_sample in iter_background_samples_from_mask(self.image, self.mask):
                if self.cancelled:
                    break
                sample = cv2.bitwise_or(sample, block_sample)
                self.partial.emit(self.request_id, self.img_path, self.mask_hash, sample.copy())
                if self.cancelled:
                    break
            self.finished.emit(self.request_id, self.img_path, self.mask_hash, sample)
        except Exception as exc:
            self.failed.emit(self.request_id, self.img_path, self.mask_hash, str(exc))


class CtdSelectionWorker(QObject):
    finished = Signal(int, str, object, int)
    failed = Signal(int, str, str)

    def __init__(self, request_id: int, img_path: str, image: np.ndarray, selection: np.ndarray) -> None:
        super().__init__()
        self.request_id = request_id
        self.img_path = img_path
        self.image = image.copy()
        self.selection = np.asarray(selection, dtype=bool).copy()

    def run(self) -> None:
        try:
            if self.image.shape[:2] != self.selection.shape[:2]:
                raise ValueError('選區尺寸和當前圖片不一致。')
            if not np.any(self.selection):
                raise ValueError('選區是空的。')

            ys, xs = np.where(self.selection)
            height, width = self.selection.shape[:2]
            x1 = max(0, int(xs.min()) - CTD_SELECTION_PADDING_PX)
            x2 = min(width - 1, int(xs.max()) + CTD_SELECTION_PADDING_PX)
            y1 = max(0, int(ys.min()) - CTD_SELECTION_PADDING_PX)
            y2 = min(height - 1, int(ys.max()) + CTD_SELECTION_PADDING_PX)

            crop = self.image[y1:y2 + 1, x1:x2 + 1]
            detect_img = crop[:, :, :3] if len(crop.shape) == 3 else cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            detector = create_detector()
            _, mask_refined, _ = detector(
                detect_img,
                refine_mode=REFINEMASK_ANNOTATION,
                keep_undetected_mask=True,
            )
            detected_crop = np.where(mask_refined > 0, 255, 0).astype(np.uint8)

            detected = np.zeros(self.selection.shape[:2], dtype=np.uint8)
            crop_selection = self.selection[y1:y2 + 1, x1:x2 + 1]
            detected_roi = detected[y1:y2 + 1, x1:x2 + 1]
            detected_roi[crop_selection & (detected_crop > 0)] = 255
            self.finished.emit(self.request_id, self.img_path, detected, int(np.count_nonzero(detected)))
        except Exception as exc:
            self.failed.emit(self.request_id, self.img_path, str(exc))


class PdfWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, imglist: list[str], paths: dict[str, str], report: dict) -> None:
        super().__init__()
        self.imglist = list(imglist)
        self.paths = dict(paths)
        self.report = dict(report)

    def run(self) -> None:
        try:
            pdf_path = _write_preview_pdf(self.imglist, self.paths, self.report)
            self.finished.emit(pdf_path)
        except Exception as exc:
            self.failed.emit(str(exc))


class FloatingNavigator(QWidget):
    viewportCenterRequested = Signal(float, float)
    positionChanged = Signal(int, int)
    closeRequested = Signal()

    TITLE_HEIGHT = 26
    PANEL_MARGIN = 10

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(210, 250)
        self.setMouseTracking(True)
        self._pixmap = QPixmap()
        self._image_size = QRectF()
        self._viewport_rect = QRectF()
        self._drag_mode = ''
        self._panel_drag_offset = QPoint()
        self._viewport_drag_offset = QPoint()
        self._last_mouse_pos = QPoint()
        self.setToolTip('拖動標題列移動小地圖；拖動框口或點擊縮圖移動畫布')

    def set_qimage(self, image: QImage | None) -> None:
        if image is None or image.isNull():
            self._pixmap = QPixmap()
            self._image_size = QRectF()
        else:
            self._pixmap = QPixmap.fromImage(image)
            self._image_size = QRectF(0, 0, image.width(), image.height())
        self.update()

    def set_viewport_rect(self, rect: QRectF) -> None:
        self._viewport_rect = QRectF(rect)
        self.update()

    def _title_rect(self) -> QRectF:
        return QRectF(0, 0, self.width(), self.TITLE_HEIGHT)

    def _close_rect(self) -> QRectF:
        return QRectF(self.width() - 26, 4, 18, 18)

    def _thumbnail_bounds(self) -> QRectF:
        return QRectF(
            self.PANEL_MARGIN,
            self.TITLE_HEIGHT + self.PANEL_MARGIN,
            self.width() - self.PANEL_MARGIN * 2,
            self.height() - self.TITLE_HEIGHT - self.PANEL_MARGIN * 2,
        )

    def _thumbnail_rect(self) -> QRectF:
        bounds = self._thumbnail_bounds()
        if self._image_size.isEmpty():
            return QRectF()
        scale = min(
            bounds.width() / self._image_size.width(),
            bounds.height() / self._image_size.height(),
        )
        width = self._image_size.width() * scale
        height = self._image_size.height() * scale
        return QRectF(
            bounds.left() + (bounds.width() - width) / 2,
            bounds.top() + (bounds.height() - height) / 2,
            width,
            height,
        )

    def _image_point_from_widget(self, point: QPoint) -> tuple[float, float] | None:
        thumb = self._thumbnail_rect()
        if thumb.isEmpty():
            return None
        x = (point.x() - thumb.left()) / thumb.width() * self._image_size.width()
        y = (point.y() - thumb.top()) / thumb.height() * self._image_size.height()
        return (
            max(0.0, min(self._image_size.width(), x)),
            max(0.0, min(self._image_size.height(), y)),
        )

    def _widget_rect_from_image_rect(self, rect: QRectF) -> QRectF:
        thumb = self._thumbnail_rect()
        if thumb.isEmpty() or self._image_size.isEmpty() or rect.isEmpty():
            return QRectF()
        x_scale = thumb.width() / self._image_size.width()
        y_scale = thumb.height() / self._image_size.height()
        return QRectF(
            thumb.left() + rect.left() * x_scale,
            thumb.top() + rect.top() * y_scale,
            max(4.0, rect.width() * x_scale),
            max(4.0, rect.height() * y_scale),
        )

    def _move_panel(self, pos: QPoint) -> None:
        parent = self.parentWidget()
        if parent is None:
            self.move(pos)
            self.positionChanged.emit(self.x(), self.y())
            return
        max_x = max(0, parent.width() - self.width())
        max_y = max(0, parent.height() - self.height())
        clamped = QPoint(max(0, min(max_x, pos.x())), max(0, min(max_y, pos.y())))
        self.move(clamped)
        self.positionChanged.emit(self.x(), self.y())

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        panel = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        painter.setPen(QPen(QColor('#41505e'), 1))
        painter.setBrush(QColor(22, 27, 32, 236))
        painter.drawRoundedRect(panel, 6, 6)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(35, 43, 51, 238))
        painter.drawRoundedRect(QRectF(1, 1, self.width() - 2, self.TITLE_HEIGHT), 5, 5)
        painter.setPen(QColor('#e6ebef'))
        painter.drawText(QRectF(10, 0, self.width() - 42, self.TITLE_HEIGHT), Qt.AlignmentFlag.AlignVCenter, 'Navigator')

        close_rect = self._close_rect()
        painter.setPen(QPen(QColor('#aab5bf'), 2))
        painter.drawLine(close_rect.left() + 5, close_rect.top() + 5, close_rect.right() - 5, close_rect.bottom() - 5)
        painter.drawLine(close_rect.right() - 5, close_rect.top() + 5, close_rect.left() + 5, close_rect.bottom() - 5)

        thumb_bounds = self._thumbnail_bounds()
        painter.setPen(QPen(QColor('#303944'), 1))
        painter.setBrush(QColor('#0b0d10'))
        painter.drawRoundedRect(thumb_bounds, 4, 4)

        thumb = self._thumbnail_rect()
        if self._pixmap.isNull() or thumb.isEmpty():
            painter.setPen(QColor('#74808b'))
            painter.drawText(thumb_bounds, Qt.AlignmentFlag.AlignCenter, '無圖片')
            return

        painter.drawPixmap(thumb.toRect(), self._pixmap)

        viewport = self._widget_rect_from_image_rect(self._viewport_rect).intersected(thumb)
        if not viewport.isEmpty():
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(233, 255, 251, 35))
            painter.drawRect(viewport)
            painter.setPen(QPen(QColor('#e9fffb'), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(viewport)

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            event.accept()
            return
        pos = event.position().toPoint()
        self._last_mouse_pos = pos
        if self._close_rect().contains(pos):
            self.closeRequested.emit()
            event.accept()
            return
        if self._title_rect().contains(pos):
            self._drag_mode = 'panel'
            self._panel_drag_offset = pos
            event.accept()
            return
        thumb = self._thumbnail_rect()
        if thumb.contains(pos):
            viewport = self._widget_rect_from_image_rect(self._viewport_rect)
            self._drag_mode = 'viewport'
            if viewport.contains(pos):
                self._viewport_drag_offset = pos - viewport.center().toPoint()
            else:
                self._viewport_drag_offset = QPoint()
                self._request_center_from_widget_pos(pos)
            event.accept()
            return
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        pos = event.position().toPoint()
        self._last_mouse_pos = pos
        if self._drag_mode == 'panel':
            self._move_panel(self.pos() + pos - self._panel_drag_offset)
        elif self._drag_mode == 'viewport':
            self._request_center_from_widget_pos(pos - self._viewport_drag_offset)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        self._drag_mode = ''
        self.positionChanged.emit(self.x(), self.y())
        event.accept()

    def _request_center_from_widget_pos(self, pos: QPoint) -> None:
        image_point = self._image_point_from_widget(pos)
        if image_point is None:
            return
        self.viewportCenterRequested.emit(image_point[0], image_point[1])


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('塗白')
        if APP_ICON_PATH.is_file():
            self.setWindowIcon(QIcon(str(APP_ICON_PATH)))
        self.resize(1500, 900)
        self.folder = ''
        self.paths: dict[str, str] = {}
        self.imglist: list[str] = []
        self.report: dict = {}
        self.current_img_path = ''
        self.current_base: np.ndarray | None = None
        self.current_mask: np.ndarray | None = None
        self.current_manual_solid: np.ndarray | None = None
        self.current_manual_other: np.ndarray | None = None
        self.current_background_sample: np.ndarray | None = None
        self.edit_mode = 'mask'
        self.selection_combine_mode = 'add'
        self.settings = QSettings('ComicTextDetector', 'SolidInpaintUI')
        self.mask_alpha_percent = self._load_mask_alpha_percent()
        self.other_mask_preview_expand_px = self._load_other_mask_preview_expand_px()
        self.navigator_visible = self._load_navigator_visible()
        self.navigator_position = self._load_navigator_position()
        self.alpha = self.mask_alpha_percent / 100.0
        self.mask_display_color = MASK_DISPLAY_COLORS['白色']
        self.show_other_mask = True
        self.show_background_sample = True
        self.undo_stack: list[dict[str, np.ndarray]] = []
        self.redo_stack: list[dict[str, np.ndarray]] = []
        self.worker_thread: QThread | None = None
        self.worker: FolderWorker | ConvertMasksWorker | None = None
        self.page_worker_thread: QThread | None = None
        self.page_worker: PageRegenerateWorker | None = None
        self.background_worker_thread: QThread | None = None
        self.background_worker: BackgroundSampleWorker | None = None
        self.background_sample_request_id = 0
        self.pending_background_img_path = ''
        self.pending_background_mask: np.ndarray | None = None
        self.ctd_selection_worker_thread: QThread | None = None
        self.ctd_selection_worker: CtdSelectionWorker | None = None
        self.ctd_selection_request_id = 0
        self.ctd_selection_edit_mode = ''
        self.pdf_worker_thread: QThread | None = None
        self.pdf_worker: PdfWorker | None = None
        self.pending_render_img_path = ''
        self.pending_render_mask: np.ndarray | None = None
        self.suppress_list_selection = False
        self._syncing_preview_view = False
        self.render_timer = QTimer(self)
        self.render_timer.setSingleShot(True)
        self.render_timer.timeout.connect(self.start_pending_render)
        self.background_sample_timer = QTimer(self)
        self.background_sample_timer.setSingleShot(True)
        self.background_sample_timer.timeout.connect(self.start_background_sample_worker)
        self.resize_fit_timer = QTimer(self)
        self.resize_fit_timer.setSingleShot(True)
        self.resize_fit_timer.timeout.connect(self.fit_both_views)
        self.auto_fit_on_resize = False
        self.recent_folders = self._load_recent_folders()
        self.folder_progress = self._load_folder_progress()

        self._build_ui()
        self._apply_style()
        QTimer.singleShot(0, self.load_latest_recent_folder)

    def _build_ui(self) -> None:
        toolbar = QToolBar('工具')
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        choose_action = QAction('選擇文件夾', self)
        choose_action.triggered.connect(self.choose_folder)
        toolbar.addAction(choose_action)

        self.recent_menu = QMenu(self)
        self.recent_button = QToolButton()
        self.recent_button.setText('打開最近列表')
        self.recent_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.recent_button.setMenu(self.recent_menu)
        toolbar.addWidget(self.recent_button)
        self.update_recent_menu()

        prev_action = QAction('上一頁', self)
        prev_action.setShortcuts([QKeySequence('A')])
        prev_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        prev_action.triggered.connect(self.previous_image)
        toolbar.addAction(prev_action)

        next_action = QAction('下一頁', self)
        next_action.setShortcuts([QKeySequence('D')])
        next_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        next_action.triggered.connect(self.next_image)
        toolbar.addAction(next_action)

        zoom_in_action = QAction('放大', self)
        zoom_in_action.setShortcuts([
            QKeySequence(Qt.KeyboardModifier.ControlModifier | Qt.Key.Key_Plus),
            QKeySequence(Qt.KeyboardModifier.MetaModifier | Qt.Key.Key_Plus),
            QKeySequence(Qt.KeyboardModifier.ControlModifier | Qt.Key.Key_Equal),
            QKeySequence(Qt.KeyboardModifier.MetaModifier | Qt.Key.Key_Equal),
        ])
        zoom_in_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        zoom_in_action.triggered.connect(lambda: self.mask_view.zoom_by(VIEW_ZOOM_STEP, keep_center=True))
        self.addAction(zoom_in_action)

        zoom_out_action = QAction('縮小', self)
        zoom_out_action.setShortcuts([
            QKeySequence(Qt.KeyboardModifier.ControlModifier | Qt.Key.Key_Minus),
            QKeySequence(Qt.KeyboardModifier.MetaModifier | Qt.Key.Key_Minus),
        ])
        zoom_out_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        zoom_out_action.triggered.connect(lambda: self.mask_view.zoom_by(1 / VIEW_ZOOM_STEP, keep_center=True))
        self.addAction(zoom_out_action)

        brush_down_action = QAction('縮小筆刷', self)
        brush_down_action.setShortcut(QKeySequence('['))
        brush_down_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        brush_down_action.triggered.connect(lambda: self.change_brush_radius(-4))
        self.addAction(brush_down_action)

        brush_up_action = QAction('放大筆刷', self)
        brush_up_action.setShortcut(QKeySequence(']'))
        brush_up_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        brush_up_action.triggered.connect(lambda: self.change_brush_radius(4))
        self.addAction(brush_up_action)

        pan_left_action = QAction('左移畫布', self)
        pan_left_action.setShortcut(QKeySequence('Left'))
        pan_left_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        pan_left_action.triggered.connect(lambda: self.mask_view.pan_by(-VIEW_KEY_PAN_STEP, 0))
        self.addAction(pan_left_action)

        pan_right_action = QAction('右移畫布', self)
        pan_right_action.setShortcut(QKeySequence('Right'))
        pan_right_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        pan_right_action.triggered.connect(lambda: self.mask_view.pan_by(VIEW_KEY_PAN_STEP, 0))
        self.addAction(pan_right_action)

        pan_up_action = QAction('上移畫布', self)
        pan_up_action.setShortcut(QKeySequence('Up'))
        pan_up_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        pan_up_action.triggered.connect(lambda: self.mask_view.pan_by(0, -VIEW_KEY_PAN_STEP))
        self.addAction(pan_up_action)

        pan_down_action = QAction('下移畫布', self)
        pan_down_action.setShortcut(QKeySequence('Down'))
        pan_down_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        pan_down_action.triggered.connect(lambda: self.mask_view.pan_by(0, VIEW_KEY_PAN_STEP))
        self.addAction(pan_down_action)

        open_output_action = QAction('打開輸出', self)
        open_output_action.triggered.connect(self.open_output)
        toolbar.addAction(open_output_action)

        generate_pdf_action = QAction('生成 PDF', self)
        generate_pdf_action.triggered.connect(self.generate_pdf)
        toolbar.addAction(generate_pdf_action)

        open_pdf_action = QAction('打開 PDF', self)
        open_pdf_action.triggered.connect(self.open_pdf)
        toolbar.addAction(open_pdf_action)

        help_action = QAction('說明', self)
        help_action.triggered.connect(self.show_help)
        toolbar.addAction(help_action)

        self.navigator_action = QAction('小地圖', self)
        self.navigator_action.setCheckable(True)
        self.navigator_action.setChecked(self.navigator_visible)
        self.navigator_action.triggered.connect(self.set_navigator_visible)
        toolbar.addAction(self.navigator_action)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        self.detect_button = QToolButton()
        self.detect_button.setText('偵測並生成')
        self.detect_button.setProperty('primary', True)
        self.detect_button.clicked.connect(self.run_or_load)
        toolbar.addWidget(self.detect_button)

        self.imported_mask_button = QToolButton()
        self.imported_mask_button.setText('使用傳入 Mask 運行')
        self.imported_mask_button.setToolTip('選擇以傳入 Mask 取代目前 Mask，或取兩者交集後運行')
        self.imported_mask_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.imported_mask_menu = QMenu(self.imported_mask_button)
        replace_mask_action = self.imported_mask_menu.addAction('取代目前 Mask')
        replace_mask_action.triggered.connect(lambda: self.run_with_imported_masks('replace'))
        intersect_masks_action = self.imported_mask_menu.addAction('取兩者交集')
        intersect_masks_action.triggered.connect(lambda: self.run_with_imported_masks('intersect'))
        self.imported_mask_button.setMenu(self.imported_mask_menu)
        toolbar.addWidget(self.imported_mask_button)

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(10, 8, 10, 10)
        root_layout.setSpacing(8)

        top_row = QHBoxLayout()
        self.folder_label = QLabel('未選擇文件夾')
        self.progress_label = QLabel('')
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        top_row.addWidget(self.folder_label, 4)
        top_row.addWidget(self.progress_label, 2)
        top_row.addWidget(self.progress, 3)
        root_layout.addLayout(top_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter, 1)

        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.addWidget(QLabel('圖片列表'))
        self.summary_label = QLabel('共 0 張')
        left_layout.addWidget(self.summary_label)
        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self.on_image_selected)
        left_layout.addWidget(self.list_widget, 1)
        splitter.addWidget(left_panel)

        workspace_panel = QFrame()
        self.workspace_panel = workspace_panel
        workspace_layout = QVBoxLayout(workspace_panel)
        workspace_layout.setContentsMargins(10, 10, 10, 10)
        workspace_layout.setSpacing(8)

        mode_toolbar = QHBoxLayout()
        self.edit_mask_btn = QPushButton('F1 自動')
        self.edit_mask_btn.setCheckable(True)
        self.edit_mask_btn.setChecked(True)
        self.edit_mask_btn.clicked.connect(lambda: self.set_edit_mode('mask'))
        self.edit_manual_solid_btn = QPushButton('F2 強制純色')
        self.edit_manual_solid_btn.setCheckable(True)
        self.edit_manual_solid_btn.clicked.connect(lambda: self.set_edit_mode('manual_solid'))
        self.edit_manual_other_btn = QPushButton('F3 需要修改')
        self.edit_manual_other_btn.setCheckable(True)
        self.edit_manual_other_btn.clicked.connect(lambda: self.set_edit_mode('manual_other'))
        self.convert_masks_btn = QPushButton('批處理')
        self.convert_masks_btn.setToolTip('將全部選區轉為指定類型；此操作不能回撤')
        self.convert_masks_btn.clicked.connect(self.show_convert_masks_dialog)
        mode_toolbar.addWidget(self.edit_mask_btn)
        mode_toolbar.addWidget(self.edit_manual_solid_btn)
        mode_toolbar.addWidget(self.edit_manual_other_btn)
        mode_toolbar.addWidget(self.convert_masks_btn)
        mode_toolbar.addStretch()
        workspace_layout.addLayout(mode_toolbar)

        edit_toolbar = QHBoxLayout()
        self.rect_btn = QPushButton('F5 矩形')
        self.rect_btn.setCheckable(True)
        self.rect_btn.clicked.connect(lambda: self.set_edit_tool('rect'))
        self.magic_btn = QPushButton('F7 魔法棒')
        self.magic_btn.setCheckable(True)
        self.magic_btn.clicked.connect(lambda: self.set_edit_tool('magic'))
        self.brush_btn = QPushButton('F6 筆刷')
        self.brush_btn.setCheckable(True)
        self.brush_btn.clicked.connect(lambda: self.set_edit_tool('brush'))
        self.selection_add_btn = QPushButton('F9 添加')
        self.selection_add_btn.setCheckable(True)
        self.selection_add_btn.setChecked(True)
        self.selection_add_btn.setToolTip('筆刷、矩形和魔法棒左鍵添加到目前 mask')
        self.selection_subtract_btn = QPushButton('F10 減去')
        self.selection_subtract_btn.setCheckable(True)
        self.selection_subtract_btn.setToolTip('筆刷、矩形和魔法棒左鍵從目前 mask 減去；右鍵矩形會清除所有 mask')
        self.selection_intersect_btn = QPushButton('F11 局部交集')
        self.selection_intersect_btn.setCheckable(True)
        self.selection_intersect_btn.setToolTip('只裁切本次選區碰到的既有 mask 區塊，不影響其他區塊')
        self.selection_inner_btn = QPushButton('選區內部')
        self.selection_inner_btn.setCheckable(True)
        self.selection_inner_btn.setToolTip('魔法棒專用：提取本次選區包圍住的內部孔洞，例如點氣泡空白後取得文字')
        self.selection_transfer_btn = QPushButton('F12 從其他轉入')
        self.selection_transfer_btn.setCheckable(True)
        self.selection_transfer_btn.setToolTip('把本次選區內其他 mask 的重疊部分移到目前 mask，並從原 mask 移除')
        self.selection_ctd_btn = QPushButton('添加CTD檢測選區')
        self.selection_ctd_btn.setCheckable(True)
        self.selection_ctd_btn.setToolTip('只對本次矩形選區跑 CTD，並把檢測結果添加到目前 mask')
        self.selection_combine_group = QButtonGroup(self)
        self.selection_combine_group.setExclusive(True)
        self.selection_combine_group.addButton(self.selection_add_btn)
        self.selection_combine_group.addButton(self.selection_subtract_btn)
        self.selection_combine_group.addButton(self.selection_intersect_btn)
        self.selection_combine_group.addButton(self.selection_inner_btn)
        self.selection_combine_group.addButton(self.selection_transfer_btn)
        self.selection_combine_group.addButton(self.selection_ctd_btn)
        self.selection_add_btn.clicked.connect(lambda: self.set_selection_combine_mode('add'))
        self.selection_subtract_btn.clicked.connect(lambda: self.set_selection_combine_mode('subtract'))
        self.selection_intersect_btn.clicked.connect(lambda: self.set_selection_combine_mode('local_intersect'))
        self.selection_inner_btn.clicked.connect(lambda: self.set_selection_combine_mode('selection_inner'))
        self.selection_transfer_btn.clicked.connect(lambda: self.set_selection_combine_mode('transfer_from_other'))
        self.selection_ctd_btn.clicked.connect(lambda: self.set_selection_combine_mode('ctd_detect_selection'))
        self.undo_btn = QPushButton('撤銷')
        self.undo_btn.clicked.connect(self.undo_mask)
        self.redo_btn = QPushButton('重做')
        self.redo_btn.clicked.connect(self.redo_mask)
        self.brush_down_btn = QPushButton('-')
        self.brush_down_btn.clicked.connect(lambda: self.change_brush_radius(-4))
        self.brush_up_btn = QPushButton('+')
        self.brush_up_btn.clicked.connect(lambda: self.change_brush_radius(4))
        self.brush_label = QLabel(f'筆刷 {DEFAULT_BRUSH_RADIUS}px')
        self.magic_tolerance_label = QLabel(f'容差 {DEFAULT_MAGIC_TOLERANCE}')
        self.magic_tolerance_slider = QSlider(Qt.Orientation.Horizontal)
        self.magic_tolerance_slider.setRange(MIN_MAGIC_TOLERANCE, MAX_MAGIC_TOLERANCE)
        self.magic_tolerance_slider.setValue(DEFAULT_MAGIC_TOLERANCE)
        self.magic_tolerance_slider.setFixedWidth(130)
        self.magic_tolerance_slider.valueChanged.connect(self.on_magic_tolerance_changed)
        self.local_intersect_controls = QWidget()
        local_intersect_layout = QHBoxLayout(self.local_intersect_controls)
        local_intersect_layout.setContentsMargins(0, 0, 0, 0)
        local_intersect_layout.setSpacing(8)
        local_intersect_layout.addWidget(QLabel('交集偏移'))
        self.local_intersect_offset_spinbox = QSpinBox()
        self.local_intersect_offset_spinbox.setRange(
            MIN_LOCAL_INTERSECT_OFFSET_PX,
            MAX_LOCAL_INTERSECT_OFFSET_PX,
        )
        self.local_intersect_offset_spinbox.setValue(DEFAULT_LOCAL_INTERSECT_OFFSET_PX)
        self.local_intersect_offset_spinbox.setSuffix(' px')
        self.local_intersect_offset_spinbox.setToolTip('局部交集結果的擴展/收縮；正數擴展，負數收縮')
        self.local_intersect_offset_spinbox.valueChanged.connect(self.on_local_intersect_offset_changed)
        local_intersect_layout.addWidget(self.local_intersect_offset_spinbox)
        edit_toolbar.addWidget(self.rect_btn)
        edit_toolbar.addWidget(self.brush_btn)
        edit_toolbar.addWidget(self.magic_btn)
        edit_toolbar.addSpacing(10)
        self.selection_combine_controls = QWidget()
        selection_combine_layout = QHBoxLayout(self.selection_combine_controls)
        selection_combine_layout.setContentsMargins(0, 0, 0, 0)
        selection_combine_layout.setSpacing(8)
        selection_combine_layout.addWidget(self.selection_add_btn)
        selection_combine_layout.addWidget(self.selection_subtract_btn)
        selection_combine_layout.addWidget(self.selection_intersect_btn)
        selection_combine_layout.addWidget(self.selection_inner_btn)
        selection_combine_layout.addWidget(self.selection_transfer_btn)
        selection_combine_layout.addWidget(self.selection_ctd_btn)
        edit_toolbar.addWidget(self.selection_combine_controls)
        edit_toolbar.addSpacing(10)
        edit_toolbar.addWidget(self.local_intersect_controls)
        edit_toolbar.addSpacing(10)
        self.brush_controls = QWidget()
        brush_controls_layout = QHBoxLayout(self.brush_controls)
        brush_controls_layout.setContentsMargins(0, 0, 0, 0)
        brush_controls_layout.setSpacing(8)
        brush_controls_layout.addWidget(self.brush_down_btn)
        brush_controls_layout.addWidget(self.brush_label)
        brush_controls_layout.addWidget(self.brush_up_btn)
        edit_toolbar.addWidget(self.brush_controls)
        edit_toolbar.addSpacing(10)
        self.magic_controls = QWidget()
        magic_controls_layout = QHBoxLayout(self.magic_controls)
        magic_controls_layout.setContentsMargins(0, 0, 0, 0)
        magic_controls_layout.setSpacing(8)
        magic_controls_layout.addWidget(self.magic_tolerance_label)
        magic_controls_layout.addWidget(self.magic_tolerance_slider)
        edit_toolbar.addWidget(self.magic_controls)
        edit_toolbar.addSpacing(10)
        edit_toolbar.addWidget(self.undo_btn)
        edit_toolbar.addWidget(self.redo_btn)
        edit_toolbar.addStretch()
        workspace_layout.addLayout(edit_toolbar)

        view_options = QHBoxLayout()
        view_options.addWidget(QLabel('Mask 顯示'))
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setInvertedAppearance(True)
        self.alpha_slider.setValue(self.mask_alpha_percent)
        self.alpha_slider.valueChanged.connect(self.on_alpha_changed)
        self.alpha_label = QLabel(f'目前 {self.mask_alpha_percent}%')
        view_options.addWidget(QLabel('100%'))
        view_options.addWidget(self.alpha_slider, 1)
        view_options.addWidget(QLabel('0%'))
        view_options.addWidget(self.alpha_label)
        view_options.addWidget(QLabel('Mask 顏色'))
        self.mask_color_combo = QComboBox()
        for name in MASK_DISPLAY_COLORS:
            self.mask_color_combo.addItem(name)
        self.mask_color_combo.setCurrentText('白色')
        self.mask_color_combo.currentTextChanged.connect(self.on_mask_display_color_changed)
        view_options.addWidget(self.mask_color_combo)
        self.background_sample_checkbox = QCheckBox('顯示背景選區')
        self.background_sample_checkbox.setChecked(True)
        self.background_sample_checkbox.stateChanged.connect(self.on_show_background_sample_changed)
        view_options.addWidget(self.background_sample_checkbox)
        self.background_sample_status = QLabel('')
        self.background_sample_status.setFixedWidth(120)
        view_options.addWidget(self.background_sample_status)
        view_options.addSpacing(12)

        self.other_mask_checkbox = QCheckBox('顯示 other_mask')
        self.other_mask_checkbox.setChecked(True)
        self.other_mask_checkbox.stateChanged.connect(self.on_show_other_mask_changed)
        view_options.addWidget(self.other_mask_checkbox)
        view_options.addSpacing(8)
        view_options.addWidget(QLabel('PS 擴展預覽'))
        self.other_mask_expand_spinbox = QSpinBox()
        self.other_mask_expand_spinbox.setRange(0, MAX_OTHER_MASK_PREVIEW_EXPAND_PX)
        self.other_mask_expand_spinbox.setValue(self.other_mask_preview_expand_px)
        self.other_mask_expand_spinbox.setSuffix(' px')
        self.other_mask_expand_spinbox.setToolTip('只影響右側淡紫色預覽，不改變輸出的 OTHER_CHANNEL')
        self.other_mask_expand_spinbox.valueChanged.connect(self.on_other_mask_preview_expand_changed)
        view_options.addWidget(self.other_mask_expand_spinbox)
        view_options.addSpacing(12)
        fit_btn = QPushButton('F4')
        fit_btn.setToolTip('兩張圖同時適應窗口')
        fit_btn.clicked.connect(self.fit_both_views)
        view_options.addWidget(fit_btn)
        view_options.addStretch()
        workspace_layout.addLayout(view_options)

        views_grid = QGridLayout()
        views_grid.setContentsMargins(0, 0, 0, 0)
        views_grid.setSpacing(8)
        views_grid.setColumnStretch(0, 1)
        views_grid.setColumnStretch(1, 1)
        views_grid.setRowStretch(1, 1)
        views_grid.addWidget(QLabel('Mask / 原圖'), 0, 0)
        views_grid.addWidget(QLabel('Inpainted 預覽'), 0, 1)

        self.mask_view = MaskEditorView()
        self.mask_view.editStarted.connect(self.push_undo_snapshot)
        self.mask_view.maskEdited.connect(self.on_mask_edited)
        self.mask_view.selectionCreated.connect(self.on_selection_created)
        self.mask_view.eraseAllMasksRequested.connect(self.on_erase_all_masks_requested)
        self.mask_view.viewChanged.connect(self.sync_preview_view)
        self.mask_view.viewChanged.connect(self.update_navigator_viewport)
        self.mask_view.viewportResized.connect(self.sync_preview_view)
        self.mask_view.viewportResized.connect(self.update_navigator_viewport)
        views_grid.addWidget(self.mask_view, 1, 0)

        self.preview_view = PassivePreviewView()
        views_grid.addWidget(self.preview_view, 1, 1)

        workspace_layout.addLayout(views_grid, 1)
        splitter.addWidget(workspace_panel)

        self.navigator = FloatingNavigator(workspace_panel)
        self.navigator.viewportCenterRequested.connect(self.center_mask_view_from_navigator)
        self.navigator.positionChanged.connect(self.save_navigator_position)
        self.navigator.closeRequested.connect(lambda: self.navigator_action.setChecked(False))
        self.navigator.closeRequested.connect(lambda: self.set_navigator_visible(False))
        self.navigator.setVisible(self.navigator_visible)
        QTimer.singleShot(0, self.restore_navigator_position)

        for button in (
            self.edit_mask_btn,
            self.edit_manual_solid_btn,
            self.edit_manual_other_btn,
            self.convert_masks_btn,
            self.brush_btn,
            self.rect_btn,
            self.magic_btn,
            self.selection_add_btn,
            self.selection_subtract_btn,
            self.selection_intersect_btn,
            self.selection_inner_btn,
            self.selection_transfer_btn,
            self.selection_ctd_btn,
            self.undo_btn,
            self.redo_btn,
            self.brush_down_btn,
            self.brush_up_btn,
            fit_btn,
        ):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        splitter.setSizes([250, 1300])
        self.setCentralWidget(root)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage('可編輯 text mask、強制純色和需要修改 mask，編輯後自動生成預覽。')
        self.set_selection_combine_mode('add')
        self.set_edit_tool('rect')
        self.auto_fit_on_resize = True

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #111418; color: #e6ebef; }
            QToolBar { background: #171b20; border: 0; spacing: 8px; padding: 8px; }
            QToolButton, QPushButton {
                background: #242b33; color: #e6ebef; border: 1px solid #3a4551;
                border-radius: 5px; padding: 6px 12px;
            }
            QToolButton:hover, QPushButton:hover { background: #2b3440; }
            QPushButton:checked {
                background: #e9fffb; color: #10161a; border-color: #e9fffb;
                font-weight: 600;
            }
            QToolButton[primary="true"] { color: #dffcf0; background: #1f5d45; border-color: #38a96d; }
            QToolButton[primary="true"]:hover { background: #277252; border-color: #57c989; }
            QToolButton::menu-indicator { image: none; width: 0; }
            QFrame { background: #1c2128; border: 1px solid #2e3640; border-radius: 6px; }
            QLabel { color: #dfe5ea; border: 0; }
            QMenu { background: #1c2128; color: #e6ebef; border: 1px solid #3a4551; padding: 4px; }
            QMenu::item { padding: 7px 22px 7px 10px; border-radius: 4px; }
            QMenu::item:selected { background: #2b3440; }
            QMenu::item:disabled { color: #7c8792; }
            QComboBox {
                background: #242b33; color: #e6ebef; border: 1px solid #3a4551;
                border-radius: 5px; padding: 5px 8px;
            }
            QComboBox:hover { background: #2b3440; }
            QComboBox::drop-down { border: 0; width: 18px; }
            QSpinBox {
                background: #242b33; color: #e6ebef; border: 1px solid #3a4551;
                border-radius: 5px; padding: 5px 8px;
            }
            QSpinBox:hover { background: #2b3440; }
            QListWidget { background: #1c2128; border: 0; color: #dfe5ea; outline: none; }
            QListWidget::item { padding: 8px 6px; border-bottom: 1px solid #27303a; }
            QListWidget::item:selected { background: #243039; color: #ffffff; }
            QProgressBar { background: #27303a; border: 0; border-radius: 4px; height: 8px; text-align: center; }
            QProgressBar::chunk { background: #38a996; border-radius: 4px; }
            QSlider::groove:horizontal { background: #303944; height: 6px; border-radius: 3px; }
            QSlider::handle:horizontal { background: #e9fffb; border: 2px solid #38a996; width: 14px; margin: -5px 0; border-radius: 7px; }
            """
        )

    def choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, '選擇圖片文件夾', self.folder or str(Path.home()))
        if not folder:
            return
        self.load_folder(folder)

    def load_latest_recent_folder(self) -> None:
        if self.folder or not self.recent_folders:
            return
        self.load_folder(self.recent_folders[0])

    def load_folder(self, folder: str) -> None:
        if not osp.isdir(folder):
            QMessageBox.warning(self, '文件夾不存在', folder)
            self.remove_recent_folder(folder)
            return
        self.folder = folder
        self.paths = _ensure_dirs(folder)
        self.imglist = image_files_in_folder(folder)
        self.report = load_report(self.paths)
        self.folder_label.setText(folder)
        self.add_recent_folder(folder)
        self.refresh_list()
        if self.imglist:
            self.list_widget.setCurrentRow(self.saved_progress_row(folder))

    def _load_recent_folders(self) -> list[str]:
        value = self.settings.value('recent_folders', [])
        if isinstance(value, str):
            folders = [value]
        elif isinstance(value, (list, tuple)):
            folders = [str(item) for item in value]
        else:
            folders = []
        seen = set()
        result = []
        for folder in folders:
            normalized = osp.abspath(osp.expanduser(folder))
            if normalized in seen or not osp.isdir(normalized):
                continue
            seen.add(normalized)
            result.append(normalized)
        return result[:MAX_RECENT_FOLDERS]

    def _load_mask_alpha_percent(self) -> int:
        value = self.settings.value('mask_alpha_percent', DEFAULT_MASK_ALPHA_PERCENT)
        try:
            percent = int(value)
        except (TypeError, ValueError):
            percent = DEFAULT_MASK_ALPHA_PERCENT
        return max(0, min(100, percent))

    def save_mask_alpha_percent(self) -> None:
        self.settings.setValue('mask_alpha_percent', self.mask_alpha_percent)

    def _load_other_mask_preview_expand_px(self) -> int:
        value = self.settings.value('other_mask_preview_expand_px', DEFAULT_OTHER_MASK_PREVIEW_EXPAND_PX)
        try:
            expand_px = int(value)
        except (TypeError, ValueError):
            expand_px = DEFAULT_OTHER_MASK_PREVIEW_EXPAND_PX
        return max(0, min(MAX_OTHER_MASK_PREVIEW_EXPAND_PX, expand_px))

    def save_other_mask_preview_expand_px(self) -> None:
        self.settings.setValue('other_mask_preview_expand_px', self.other_mask_preview_expand_px)

    def _load_navigator_visible(self) -> bool:
        value = self.settings.value('navigator_visible', True)
        if isinstance(value, bool):
            return value
        return str(value).lower() not in ('0', 'false', 'no')

    def _load_navigator_position(self) -> QPoint | None:
        value = self.settings.value('navigator_position', '')
        if not value:
            return None
        if isinstance(value, QPoint):
            return value
        try:
            x_text, y_text = str(value).split(',', 1)
            return QPoint(int(x_text), int(y_text))
        except (TypeError, ValueError):
            return None

    def set_navigator_visible(self, visible: bool) -> None:
        self.navigator_visible = bool(visible)
        self.settings.setValue('navigator_visible', self.navigator_visible)
        if getattr(self, 'navigator_action', None) is not None:
            self.navigator_action.setChecked(self.navigator_visible)
        if getattr(self, 'navigator', None) is not None:
            self.navigator.setVisible(self.navigator_visible)
            if self.navigator_visible:
                self.restore_navigator_position()
                self.navigator.raise_()
                self.update_navigator_viewport()

    def restore_navigator_position(self) -> None:
        if getattr(self, 'navigator', None) is None or getattr(self, 'workspace_panel', None) is None:
            return
        parent = self.workspace_panel
        if self.navigator_position is None:
            margin = 18
            pos = QPoint(
                max(0, parent.width() - self.navigator.width() - margin),
                max(0, parent.height() - self.navigator.height() - margin),
            )
        else:
            max_x = max(0, parent.width() - self.navigator.width())
            max_y = max(0, parent.height() - self.navigator.height())
            pos = QPoint(
                max(0, min(max_x, self.navigator_position.x())),
                max(0, min(max_y, self.navigator_position.y())),
            )
        self.navigator.move(pos)
        self.navigator.raise_()

    def save_navigator_position(self, x: int, y: int) -> None:
        self.navigator_position = QPoint(int(x), int(y))
        self.settings.setValue('navigator_position', f'{int(x)},{int(y)}')

    def save_recent_folders(self) -> None:
        self.settings.setValue('recent_folders', self.recent_folders)

    def _load_folder_progress(self) -> dict[str, str]:
        value = self.settings.value('folder_progress', {})
        if not isinstance(value, dict):
            return {}
        progress: dict[str, str] = {}
        for folder, image_name in value.items():
            folder_key = osp.abspath(osp.expanduser(str(folder)))
            image_base = osp.basename(str(image_name))
            if folder_key and image_base:
                progress[folder_key] = image_base
        return progress

    def save_folder_progress(self) -> None:
        if len(self.folder_progress) > MAX_FOLDER_PROGRESS:
            recent_keys = {osp.abspath(osp.expanduser(folder)) for folder in self.recent_folders}
            filtered = {
                folder: image_name
                for folder, image_name in self.folder_progress.items()
                if folder in recent_keys
            }
            self.folder_progress = dict(list(filtered.items())[-MAX_FOLDER_PROGRESS:])
        self.settings.setValue('folder_progress', self.folder_progress)

    def remember_folder_progress(self, img_path: str) -> None:
        if not self.folder or not img_path:
            return
        folder_key = osp.abspath(osp.expanduser(self.folder))
        image_base = osp.basename(img_path)
        if not image_base:
            return
        self.folder_progress[folder_key] = image_base
        self.save_folder_progress()

    def saved_progress_row(self, folder: str) -> int:
        folder_key = osp.abspath(osp.expanduser(folder))
        image_base = self.folder_progress.get(folder_key, '')
        if not image_base:
            return 0
        for index, img_path in enumerate(self.imglist):
            if osp.basename(img_path) == image_base:
                return index
        return 0

    def add_recent_folder(self, folder: str) -> None:
        normalized = osp.abspath(osp.expanduser(folder))
        self.recent_folders = [item for item in self.recent_folders if item != normalized]
        self.recent_folders.insert(0, normalized)
        self.recent_folders = self.recent_folders[:MAX_RECENT_FOLDERS]
        self.save_recent_folders()
        self.update_recent_menu()

    def remove_recent_folder(self, folder: str) -> None:
        normalized = osp.abspath(osp.expanduser(folder))
        self.recent_folders = [item for item in self.recent_folders if item != normalized]
        self.save_recent_folders()
        self.update_recent_menu()

    def clear_recent_folders(self) -> None:
        self.recent_folders = []
        self.save_recent_folders()
        self.update_recent_menu()

    def show_convert_masks_dialog(self) -> None:
        if not self.folder or not self.paths or not self.imglist:
            QMessageBox.information(self, '沒有圖片', '請先選擇圖片文件夾。')
            return
        if self.worker_thread is not None:
            QMessageBox.information(self, '正在執行', '已有任務在執行中。')
            return
        if self.page_worker_thread is not None or self.render_timer.isActive():
            QMessageBox.information(self, '正在生成預覽', '請等待當前頁預覽生成完成。')
            return
        dialog = ConvertMasksDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.convert_masks(dialog.selected_target_mode(), dialog.selected_scope())

    def convert_masks(self, target_mode: str, scope: str) -> None:
        self.set_edit_mode(target_mode)
        if scope == 'all':
            self.save_all_edit_masks()
            self.undo_stack = []
            self.redo_stack = []
            self.update_edit_buttons()
            self.start_convert_worker(self.imglist, target_mode)
            self.status.showMessage('正在轉換全部頁面選區...')
            return
        self.convert_current_page_masks(target_mode)

    def convert_current_page_masks(self, target_mode: str) -> None:
        if not self.current_img_path:
            return
        self.save_all_edit_masks()
        auto_mask = _convert_image_edit_masks(self.paths, self.current_img_path, target_mode)
        shape, masks = _read_edit_masks_for_image(self.paths, self.current_img_path)
        self.current_mask = masks['mask']
        self.current_manual_solid = masks['manual_solid']
        self.current_manual_other = masks['manual_other']
        self.undo_stack = []
        self.redo_stack = []
        self.update_edit_buttons()
        if self.current_base is not None:
            self.mask_view.set_mask(self.current_edit_mask(), shape, reset_brush_line=True)
        self.queue_background_sample()
        self.refresh_mask_preview(keep_view=True)
        self.pending_render_img_path = self.current_img_path
        self.pending_render_mask = auto_mask.copy()
        self.start_pending_render()
        self.status.showMessage('正在轉換當前頁面選區...')

    def update_recent_menu(self) -> None:
        self.recent_menu.clear()
        if self.recent_folders:
            for folder in self.recent_folders:
                action = QAction(folder, self)
                action.triggered.connect(lambda checked=False, path=folder: self.load_folder(path))
                self.recent_menu.addAction(action)
        else:
            empty_action = QAction('沒有最近文件夾', self)
            empty_action.setEnabled(False)
            self.recent_menu.addAction(empty_action)

        self.recent_menu.addSeparator()
        clear_action = QAction('清除', self)
        clear_action.setEnabled(bool(self.recent_folders))
        clear_action.triggered.connect(self.clear_recent_folders)
        self.recent_menu.addAction(clear_action)

    def run_or_load(self) -> None:
        if not self.folder:
            self.choose_folder()
            if not self.folder:
                return
        if self.page_worker_thread is not None or self.render_timer.isActive():
            QMessageBox.information(self, '正在生成預覽', '請等待當前頁預覽生成完成後再重新偵測。')
            return
        if self.has_existing_masks():
            reply = QMessageBox.warning(
                self,
                '確認重新偵測',
                '當前文件夾已存在 mask。\n\n'
                '「偵測並生成」會重新跑 detector，並覆蓋已有的 mask、other_mask 和 inpainted 輸出。\n'
                '如果你已經手動修改過 mask，這些修改會丟失。\n\n'
                '確定要繼續嗎？',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        self.start_worker('detect', self.imglist)

    def run_with_imported_masks(self, mode: str) -> None:
        if not self.folder:
            self.choose_folder()
            if not self.folder:
                return
        if not self.imglist:
            QMessageBox.information(self, '沒有圖片', '當前文件夾內沒有可處理的圖片。')
            return
        if self.worker_thread is not None:
            QMessageBox.information(self, '正在執行', '已有任務在執行中。')
            return
        if self.page_worker_thread is not None or self.render_timer.isActive():
            QMessageBox.information(self, '正在生成預覽', '請等待當前頁預覽生成完成後再處理傳入 Mask。')
            return

        if mode not in {'replace', 'intersect'}:
            raise ValueError(f'不支援的傳入 Mask 處理方式：{mode}')

        missing_masks = []
        if mode == 'intersect':
            missing_masks = [
                osp.basename(img_path)
                for img_path in self.imglist
                if not osp.isfile(_mask_path(self.paths, img_path))
            ]
        if mode == 'intersect' and len(missing_masks) == len(self.imglist):
            QMessageBox.warning(
                self,
                '找不到目前 Mask',
                '當前文件夾沒有可用的 mask。\n\n'
                '請先放入 ctd_inpainted/mask/<檔名>.png，或先執行「偵測並生成」。',
            )
            return

        dialog_title = (
            '選擇用來取代目前 Mask 的文件夾'
            if mode == 'replace'
            else '選擇用來取兩者交集的 Mask 文件夾'
        )
        imported_mask_dir = QFileDialog.getExistingDirectory(self, dialog_title, self.folder)
        if not imported_mask_dir:
            return

        matched_count = sum(
            1
            for img_path in self.imglist
            if osp.isfile(osp.join(imported_mask_dir, f'{Path(img_path).stem}.png'))
        )
        if mode == 'replace':
            title = '取代目前 Mask'
            message = (
                '傳入 Mask 將取代目前 Mask，然後重新運行。\n'
                '沒有同名傳入 Mask 的頁面會保留目前 Mask。'
            )
        else:
            title = '取兩者交集'
            message = (
                '傳入 Mask 會與目前 Mask 取交集，然後重新運行。\n'
                '沒有同名傳入 Mask 的頁面會保留目前 Mask。'
            )
        message += f'\n\n找到同名傳入 Mask：{matched_count} / {len(self.imglist)}'
        if missing_masks:
            preview = '\n'.join(missing_masks[:8])
            more = len(missing_masks) - 8
            if more > 0:
                preview += f'\n...以及 {more} 個文件'
            message += f'\n\n有 {len(missing_masks)} 張圖片缺少現有 mask，這些頁會標記為失敗：\n{preview}'

        reply = QMessageBox.question(
            self,
            title,
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.start_worker(
            'imported_mask',
            self.imglist,
            imported_mask_dir=imported_mask_dir,
            imported_mask_mode=mode,
        )

    def has_existing_masks(self) -> bool:
        if not self.paths:
            return False
        mask_dir = self.paths.get('mask')
        if not mask_dir or not osp.isdir(mask_dir):
            return False
        return any(
            osp.isfile(_mask_path(self.paths, img_path))
            for img_path in self.imglist
        )

    def start_worker(
        self,
        mode: str,
        image_paths: list[str],
        imported_mask_dir: str = '',
        imported_mask_mode: str = '',
    ) -> None:
        if self.worker_thread is not None:
            QMessageBox.information(self, '正在執行', '已有任務在執行中。')
            return
        if self.page_worker_thread is not None:
            QMessageBox.information(self, '正在生成預覽', '請等待當前頁預覽生成完成。')
            return
        self.progress.setValue(0)
        self.worker_thread = QThread()
        self.worker = FolderWorker(self.folder, mode, image_paths, imported_mask_dir, imported_mask_mode)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_worker_progress)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.failed.connect(self.on_worker_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.cleanup_worker)
        self.worker_thread.start()

    def start_convert_worker(self, image_paths: list[str], target_mode: str) -> None:
        if self.worker_thread is not None:
            QMessageBox.information(self, '正在執行', '已有任務在執行中。')
            return
        if self.page_worker_thread is not None or self.render_timer.isActive():
            QMessageBox.information(self, '正在生成預覽', '請等待當前頁預覽生成完成。')
            return
        self.progress.setValue(0)
        self.worker_thread = QThread()
        self.worker = ConvertMasksWorker(self.folder, self.paths, image_paths, target_mode)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_worker_progress)
        self.worker.finished.connect(self.on_convert_worker_finished)
        self.worker.failed.connect(self.on_worker_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.cleanup_worker)
        self.worker_thread.start()

    def on_worker_progress(self, current: int, total: int, name: str) -> None:
        percent = int(current * 100 / max(1, total))
        self.progress.setValue(percent)
        self.progress_label.setText(f'{current} / {total}  {name}')

    def on_worker_finished(self, report: dict) -> None:
        self.report = report
        self.progress.setValue(100)
        self.refresh_list()
        self.reload_current()
        self.status.showMessage('任務完成。')

    def on_convert_worker_finished(self, report: dict) -> None:
        self.report = report
        self.progress.setValue(100)
        self.undo_stack = []
        self.redo_stack = []
        self.refresh_list()
        self.reload_current()
        self.status.showMessage('選區轉換完成。')

    def on_worker_failed(self, message: str) -> None:
        QMessageBox.critical(self, '執行失敗', message)

    def cleanup_worker(self) -> None:
        self.worker = None
        self.worker_thread = None

    def refresh_list(self, keep_scroll: bool = True) -> None:
        previous_path = self.current_img_path
        vertical_scroll = self.list_widget.verticalScrollBar().value()
        horizontal_scroll = self.list_widget.horizontalScrollBar().value()
        self.suppress_list_selection = True
        self.list_widget.clear()
        pages = self.report.get('pages', {})
        other_count = 0
        failed_count = 0
        selected_row = -1
        for img_path in self.imglist:
            name = osp.basename(img_path)
            info = pages.get(name, {})
            status = STATUS_TODO
            if 'error' in info:
                status = STATUS_FAILED
                failed_count += 1
            elif info:
                if int(info.get('other_pixels', 0)) > 0:
                    status = STATUS_OTHER
                    other_count += 1
                else:
                    status = STATUS_OK
            item = QListWidgetItem(f'{name}    {status}')
            item.setData(Qt.ItemDataRole.UserRole, img_path)
            if status == STATUS_OTHER:
                item.setForeground(QColor('#d59a45'))
            elif status == STATUS_FAILED:
                item.setForeground(QColor('#e06767'))
            elif status == STATUS_OK:
                item.setForeground(QColor('#57b66f'))
            self.list_widget.addItem(item)
            if img_path == previous_path:
                selected_row = self.list_widget.count() - 1
        self.summary_label.setText(f'共 {len(self.imglist)} 張    OTHER {other_count}    失敗 {failed_count}')
        if selected_row >= 0:
            self.list_widget.setCurrentRow(selected_row)
        self.suppress_list_selection = False
        if keep_scroll:
            self.restore_list_scroll(vertical_scroll, horizontal_scroll)

    def restore_list_scroll(self, vertical_scroll: int, horizontal_scroll: int) -> None:
        vertical_bar = self.list_widget.verticalScrollBar()
        horizontal_bar = self.list_widget.horizontalScrollBar()
        vertical_bar.setValue(max(vertical_bar.minimum(), min(vertical_bar.maximum(), vertical_scroll)))
        horizontal_bar.setValue(max(horizontal_bar.minimum(), min(horizontal_bar.maximum(), horizontal_scroll)))

    def on_image_selected(self, row: int) -> None:
        if self.suppress_list_selection:
            return
        if row < 0:
            return
        item = self.list_widget.item(row)
        if item is None:
            return
        self.current_img_path = item.data(Qt.ItemDataRole.UserRole)
        self.remember_folder_progress(self.current_img_path)
        self.undo_stack = []
        self.redo_stack = []
        self.reload_current()

    def previous_image(self) -> None:
        if not self.imglist:
            return
        row = self.list_widget.currentRow()
        if row <= 0:
            return
        self.list_widget.setCurrentRow(row - 1)

    def next_image(self) -> None:
        if not self.imglist:
            return
        row = self.list_widget.currentRow()
        if row < 0:
            self.list_widget.setCurrentRow(0)
            return
        if row < self.list_widget.count() - 1:
            self.list_widget.setCurrentRow(row + 1)

    def reload_current(self, keep_view: bool = False) -> None:
        if not self.current_img_path:
            return
        base = _optional_imread(self.current_img_path, cv2.IMREAD_UNCHANGED)
        mask = _optional_imread(_mask_path(self.paths, self.current_img_path), cv2.IMREAD_GRAYSCALE)
        manual_solid = _optional_imread(_manual_solid_path(self.paths, self.current_img_path), cv2.IMREAD_UNCHANGED)
        manual_other = _optional_imread(_manual_other_path(self.paths, self.current_img_path), cv2.IMREAD_UNCHANGED)
        other_mask = _optional_imread(_other_mask_path(self.paths, self.current_img_path), cv2.IMREAD_GRAYSCALE)
        overlay = _optional_imread(_output_path(self.paths, self.current_img_path), cv2.IMREAD_UNCHANGED)
        if base is None:
            return
        self.current_base = base
        shape = base.shape[:2]
        self.current_mask = np.where(mask > 0, 255, 0).astype(np.uint8) if mask is not None else np.zeros(shape, dtype=np.uint8)
        self.current_manual_solid = _mask_from_optional_image(manual_solid, shape)
        self.current_manual_other = _mask_from_optional_image(manual_other, shape)
        self.current_background_sample = _load_background_sample_cache(self.paths, self.current_img_path, self.current_mask)
        if self.current_background_sample is None:
            self.queue_background_sample()
        else:
            self.background_sample_status.setText('')
        self.mask_view.set_mask(self.current_edit_mask(), shape, reset_brush_line=True)
        self.mask_view.set_source_image(self.current_base)
        self.refresh_mask_preview(keep_view=keep_view)

        if overlay is not None:
            preview = _compose_overlay_preview(base, overlay)
        else:
            preview = base[:, :, :3].copy() if len(base.shape) == 3 else cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        preview = self.apply_other_mask_preview(preview, other_mask)
        self.preview_view.set_qimage(_qimage_from_bgr(preview), keep_view=keep_view)
        self.sync_preview_view()
        QTimer.singleShot(0, self.sync_preview_view)

        self.update_edit_buttons()

    def apply_other_mask_preview(self, preview: np.ndarray, other_mask: np.ndarray | None) -> np.ndarray:
        if not self.show_other_mask:
            return preview
        ring = _expanded_mask_ring(other_mask, self.other_mask_preview_expand_px)
        preview = _overlay_mask_on_bgr(
            preview,
            ring,
            OTHER_MASK_PREVIEW_RING_ALPHA,
            EDIT_MODE_COLORS['manual_other'],
        )
        return _overlay_mask_on_bgr(
            preview,
            other_mask,
            OTHER_MASK_DISPLAY_ALPHA,
            EDIT_MODE_COLORS['manual_other'],
        )

    def refresh_mask_preview(self, keep_view: bool = True) -> None:
        if self.current_base is None:
            self.mask_view.set_qimage(None)
            if getattr(self, 'navigator', None) is not None:
                self.navigator.set_qimage(None)
            return
        mask_preview = _mask_overlay_image(
            self.current_base,
            self.current_mask,
            self.alpha,
            self.mask_display_color,
        )
        mask_preview = _overlay_transparent_mask_on_bgr(
            mask_preview,
            self.current_manual_solid,
            self.alpha,
            EDIT_MODE_COLORS['manual_solid'],
        )
        mask_preview = _overlay_transparent_mask_on_bgr(
            mask_preview,
            self.current_manual_other,
            self.alpha,
            EDIT_MODE_COLORS['manual_other'],
        )
        if self.show_background_sample and self.current_background_sample is not None:
            mask_preview = _overlay_mask_on_bgr(
                mask_preview,
                self.current_background_sample,
                SAMPLE_RING_DISPLAY_ALPHA,
                SAMPLE_RING_DISPLAY_COLOR_BGR,
            )
        mask_qimage = _qimage_from_bgr(mask_preview)
        self.mask_view.set_qimage(mask_qimage, keep_view=keep_view)
        if getattr(self, 'navigator', None) is not None:
            self.navigator.set_qimage(mask_qimage)
        current_mask = self.current_edit_mask()
        if current_mask is not None:
            self.mask_view.set_mask(current_mask, self.current_base.shape[:2])
            self.mask_view.set_source_image(self.current_base)
        self.update_navigator_viewport()

    def on_alpha_changed(self, value: int) -> None:
        self.mask_alpha_percent = max(0, min(100, int(value)))
        self.alpha = value / 100.0
        self.alpha_label.setText(f'目前 {value}%')
        self.save_mask_alpha_percent()
        self.refresh_mask_preview(keep_view=True)

    def on_mask_display_color_changed(self, color_name: str) -> None:
        self.mask_display_color = MASK_DISPLAY_COLORS.get(color_name, MASK_DISPLAY_COLORS['白色'])
        self.refresh_mask_preview(keep_view=True)

    def on_show_background_sample_changed(self, state: int) -> None:
        self.show_background_sample = state == Qt.CheckState.Checked.value
        if self.show_background_sample and self.current_background_sample is None:
            self.queue_background_sample()
        self.refresh_mask_preview(keep_view=True)

    def on_show_other_mask_changed(self, state: int) -> None:
        self.show_other_mask = state == Qt.CheckState.Checked.value
        self.reload_current()

    def on_other_mask_preview_expand_changed(self, value: int) -> None:
        self.other_mask_preview_expand_px = max(0, min(MAX_OTHER_MASK_PREVIEW_EXPAND_PX, int(value)))
        self.save_other_mask_preview_expand_px()
        self.reload_current(keep_view=True)

    def current_edit_mask(self) -> np.ndarray | None:
        if self.edit_mode == 'manual_solid':
            return self.current_manual_solid
        if self.edit_mode == 'manual_other':
            return self.current_manual_other
        return self.current_mask

    def current_masks_snapshot(self) -> dict[str, np.ndarray]:
        return {
            'mask': self.current_mask.copy() if self.current_mask is not None else np.zeros((1, 1), dtype=np.uint8),
            'manual_solid': (
                self.current_manual_solid.copy()
                if self.current_manual_solid is not None
                else np.zeros_like(self.current_mask)
                if self.current_mask is not None
                else np.zeros((1, 1), dtype=np.uint8)
            ),
            'manual_other': (
                self.current_manual_other.copy()
                if self.current_manual_other is not None
                else np.zeros_like(self.current_mask)
                if self.current_mask is not None
                else np.zeros((1, 1), dtype=np.uint8)
            ),
        }

    def restore_masks_snapshot(self, snapshot: dict[str, np.ndarray]) -> None:
        self.current_mask = np.where(snapshot['mask'] > 0, 255, 0).astype(np.uint8)
        self.current_manual_solid = np.where(snapshot['manual_solid'] > 0, 255, 0).astype(np.uint8)
        self.current_manual_other = np.where(snapshot['manual_other'] > 0, 255, 0).astype(np.uint8)

    def mask_for_mode(self, mode: str) -> np.ndarray | None:
        if mode == 'manual_solid':
            return self.current_manual_solid
        if mode == 'manual_other':
            return self.current_manual_other
        if mode == 'mask':
            return self.current_mask
        return None

    def set_current_edit_mask(self, mask: np.ndarray) -> None:
        mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        if self.edit_mode == 'manual_solid':
            self.current_manual_solid = mask
        elif self.edit_mode == 'manual_other':
            self.current_manual_other = mask
        else:
            self.current_mask = mask

    def current_edit_color(self) -> tuple[int, int, int]:
        if self.edit_mode == 'mask':
            return self.mask_display_color
        return EDIT_MODE_COLORS[self.edit_mode]

    def set_edit_mode(self, mode: str, reset_lower: bool = True) -> None:
        if mode not in EDIT_MODE_LABELS:
            return
        self.edit_mode = mode
        self.undo_stack = []
        self.redo_stack = []
        self.edit_mask_btn.setChecked(mode == 'mask')
        self.edit_manual_solid_btn.setChecked(mode == 'manual_solid')
        self.edit_manual_other_btn.setChecked(mode == 'manual_other')
        self.background_sample_checkbox.setEnabled(True)
        if self.show_background_sample and self.current_background_sample is None:
            self.queue_background_sample()
        edit_mask = self.current_edit_mask()
        if edit_mask is not None and self.current_base is not None:
            self.mask_view.set_mask(edit_mask, self.current_base.shape[:2], reset_brush_line=True)
        self.refresh_mask_preview(keep_view=True)
        self.update_edit_buttons()
        self.status.showMessage(f'正在編輯：{EDIT_MODE_LABELS[mode]}')
        if reset_lower:
            self.set_edit_tool('rect', reset_selection=True)

    def set_edit_tool(self, tool: str, reset_selection: bool = True) -> None:
        self.mask_view.set_tool(tool)
        self.brush_btn.setChecked(tool == 'brush')
        self.rect_btn.setChecked(tool == 'rect')
        self.magic_btn.setChecked(tool == 'magic')
        if (
            (reset_selection and self.selection_combine_mode != 'add')
            or not self.selection_mode_allowed_for_tool(self.selection_combine_mode, tool)
        ):
            self.set_selection_combine_mode('add')
        self.selection_combine_controls.setVisible(tool in ('rect', 'magic', 'brush'))
        self.brush_controls.setVisible(tool == 'brush')
        self.magic_controls.setVisible(tool == 'magic')
        self.update_selection_combine_controls_visibility()
        self.update_local_intersect_controls_visibility()
        self.update_selection_combine_status()

    def set_selection_combine_mode(self, mode: str) -> None:
        if mode not in SELECTION_COMBINE_LABELS:
            return
        if not self.selection_mode_allowed_for_tool(mode, self.mask_view.tool):
            return
        self.selection_combine_mode = mode
        self.mask_view.set_selection_combine_mode(mode)
        self.selection_add_btn.setChecked(mode == 'add')
        self.selection_subtract_btn.setChecked(mode == 'subtract')
        self.selection_intersect_btn.setChecked(mode == 'local_intersect')
        self.selection_inner_btn.setChecked(mode == 'selection_inner')
        self.selection_transfer_btn.setChecked(mode == 'transfer_from_other')
        self.selection_ctd_btn.setChecked(mode == 'ctd_detect_selection')
        self.update_selection_combine_controls_visibility()
        self.update_local_intersect_controls_visibility()
        self.update_selection_combine_status()

    def selection_mode_allowed_for_tool(self, mode: str, tool: str) -> bool:
        if tool == 'brush':
            return mode in ('add', 'subtract')
        if tool == 'rect':
            return mode != 'selection_inner'
        if tool == 'magic':
            return mode in SELECTION_COMBINE_LABELS
        return False

    def update_selection_combine_controls_visibility(self) -> None:
        if getattr(self, 'selection_inner_btn', None) is None:
            return
        tool = self.mask_view.tool
        rect_or_magic = tool in ('rect', 'magic')
        self.selection_add_btn.setVisible(tool in ('rect', 'magic', 'brush'))
        self.selection_subtract_btn.setVisible(tool in ('rect', 'magic', 'brush'))
        self.selection_intersect_btn.setVisible(rect_or_magic)
        self.selection_inner_btn.setVisible(tool == 'magic')
        self.selection_transfer_btn.setVisible(rect_or_magic)
        self.selection_ctd_btn.setVisible(rect_or_magic)

    def update_local_intersect_controls_visibility(self) -> None:
        if getattr(self, 'local_intersect_controls', None) is None:
            return
        visible = (
            self.mask_view.tool in ('rect', 'magic')
            and self.selection_combine_mode == 'local_intersect'
        )
        self.local_intersect_controls.setVisible(visible)

    def update_selection_combine_status(self) -> None:
        if getattr(self, 'mask_view', None) is None:
            return
        if self.mask_view.tool == 'brush':
            label = SELECTION_COMBINE_LABELS.get(self.selection_combine_mode, '添加')
            self.status.showMessage(f'筆刷模式：{label}；右鍵拖矩形清除所有 mask')
            return
        label = SELECTION_COMBINE_LABELS.get(self.selection_combine_mode, '添加')
        self.status.showMessage(f'選區模式：{label}；右鍵拖矩形清除所有 mask')

    def change_brush_radius(self, delta: int) -> None:
        self.mask_view.set_brush_radius(self.mask_view.brush_radius + delta)
        self.brush_label.setText(f'筆刷 {self.mask_view.brush_radius}px')

    def on_magic_tolerance_changed(self, value: int) -> None:
        self.mask_view.set_magic_tolerance(value)
        self.magic_tolerance_label.setText(f'容差 {value}')

    def on_local_intersect_offset_changed(self, value: int) -> None:
        self.mask_view.set_local_intersect_offset(value)

    def fit_both_views(self) -> None:
        self.mask_view.fit()
        self.sync_preview_view()

    def sync_preview_view(self) -> None:
        if self._syncing_preview_view:
            return
        if getattr(self, 'mask_view', None) is None or getattr(self, 'preview_view', None) is None:
            return
        self._syncing_preview_view = True
        try:
            self.preview_view.copy_view_from(self.mask_view)
        finally:
            self._syncing_preview_view = False

    def update_navigator_viewport(self) -> None:
        if getattr(self, 'navigator', None) is None or getattr(self, 'mask_view', None) is None:
            return
        self.navigator.set_viewport_rect(self.mask_view.visible_image_rect())

    def center_mask_view_from_navigator(self, x: float, y: float) -> None:
        if getattr(self, 'mask_view', None) is None:
            return
        self.mask_view.center_on_image_point(x, y)
        self.sync_preview_view()
        self.update_navigator_viewport()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if getattr(self, 'navigator', None) is not None:
            self.restore_navigator_position()
            self.update_navigator_viewport()
        if getattr(self, 'auto_fit_on_resize', False):
            self.resize_fit_timer.start(120)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_F1:
            self.set_edit_mode('mask')
            return
        if event.key() == Qt.Key.Key_F2:
            self.set_edit_mode('manual_solid')
            return
        if event.key() == Qt.Key.Key_F3:
            self.set_edit_mode('manual_other')
            return
        if event.key() == Qt.Key.Key_F4:
            self.fit_both_views()
            return
        if event.key() == Qt.Key.Key_F5:
            self.set_edit_tool('rect')
            return
        if event.key() == Qt.Key.Key_F6:
            self.set_edit_tool('brush')
            return
        if event.key() == Qt.Key.Key_F7:
            self.set_edit_tool('magic')
            return
        if event.key() == Qt.Key.Key_F9:
            self.set_selection_combine_mode('add')
            return
        if event.key() == Qt.Key.Key_F10:
            self.set_selection_combine_mode('subtract')
            return
        if event.key() == Qt.Key.Key_F11:
            self.set_selection_combine_mode('local_intersect')
            return
        if event.key() == Qt.Key.Key_F12:
            self.set_selection_combine_mode('transfer_from_other')
            return
        if event.key() == Qt.Key.Key_B:
            self.set_edit_tool('brush')
            return
        if event.key() == Qt.Key.Key_R:
            self.set_edit_tool('rect')
            return
        if event.key() == Qt.Key.Key_W:
            self.set_edit_tool('magic')
            return
        if event.key() == Qt.Key.Key_BracketLeft:
            self.change_brush_radius(-4)
            return
        if event.key() == Qt.Key.Key_BracketRight:
            self.change_brush_radius(4)
            return
        if event.matches(QKeySequence.StandardKey.Undo):
            self.undo_mask()
            return
        if event.matches(QKeySequence.StandardKey.Redo):
            self.redo_mask()
            return
        super().keyPressEvent(event)

    def push_undo_snapshot(self) -> None:
        if self.current_mask is None:
            return
        self.undo_stack.append(self.current_masks_snapshot())
        self.undo_stack = self.undo_stack[-MAX_UNDO_STEPS:]
        self.redo_stack = []
        self.update_edit_buttons()

    def on_mask_edited(self, mask: object) -> None:
        self.set_current_edit_mask(np.asarray(mask))
        if self.edit_mode == 'mask':
            self.queue_background_sample()
        self.save_current_edit_mask()
        self.refresh_mask_preview(keep_view=True)
        self.queue_auto_render()
        self.update_edit_buttons()

    def on_selection_created(self, selection: object) -> None:
        selection_mask = np.asarray(selection, dtype=bool)
        if self.selection_combine_mode == 'ctd_detect_selection':
            self.start_ctd_selection_detection(selection_mask)
            return
        if self.selection_combine_mode != 'transfer_from_other':
            return
        if self.transfer_selection_from_other_masks(selection_mask):
            self.queue_background_sample()
            self.save_all_edit_masks()
            if self.current_base is not None:
                self.mask_view.set_mask(self.current_edit_mask(), self.current_base.shape[:2])
            self.refresh_mask_preview(keep_view=True)
            self.queue_auto_render()
            self.update_edit_buttons()
            self.status.showMessage('已從其他 mask 轉入選區重疊部分。')
        else:
            self.status.showMessage('選區沒有碰到其他 mask。')

    def on_erase_all_masks_requested(self, selection: object) -> None:
        selection_mask = np.asarray(selection, dtype=bool)
        if self.current_mask is None or selection_mask.shape[:2] != self.current_mask.shape[:2]:
            self.status.showMessage('右鍵清除選區尺寸不一致。')
            return
        masks = [
            self.current_mask,
            self.current_manual_solid,
            self.current_manual_other,
        ]
        changed = any(mask is not None and np.any(mask[selection_mask] > 0) for mask in masks)
        if not changed:
            self.status.showMessage('右鍵矩形內沒有可清除的 mask。')
            return
        self.push_undo_snapshot()
        for mask in masks:
            if mask is not None:
                mask[selection_mask] = 0
        self.queue_background_sample()
        self.save_all_edit_masks()
        if self.current_base is not None:
            self.mask_view.set_mask(self.current_edit_mask(), self.current_base.shape[:2])
        self.refresh_mask_preview(keep_view=True)
        self.queue_auto_render()
        self.update_edit_buttons()
        self.status.showMessage('已清除右鍵矩形內所有 mask。')

    def start_ctd_selection_detection(self, selection: np.ndarray) -> None:
        if self.current_base is None or not self.current_img_path or self.current_edit_mask() is None:
            self.status.showMessage('沒有可檢測的當前圖片。')
            return
        if self.ctd_selection_worker_thread is not None:
            QMessageBox.information(self, '正在檢測', 'CTD 正在檢測上一個選區。')
            return
        if self.worker_thread is not None:
            QMessageBox.information(self, '正在執行', '已有批量任務在執行中。')
            return
        self.ctd_selection_request_id += 1
        request_id = self.ctd_selection_request_id
        img_path = self.current_img_path
        self.ctd_selection_edit_mode = self.edit_mode
        self.status.showMessage('CTD 正在檢測選區...')
        self.ctd_selection_worker_thread = QThread()
        self.ctd_selection_worker = CtdSelectionWorker(request_id, img_path, self.current_base, selection)
        self.ctd_selection_worker.moveToThread(self.ctd_selection_worker_thread)
        self.ctd_selection_worker_thread.started.connect(self.ctd_selection_worker.run)
        self.ctd_selection_worker.finished.connect(self.on_ctd_selection_finished)
        self.ctd_selection_worker.failed.connect(self.on_ctd_selection_failed)
        self.ctd_selection_worker.finished.connect(self.ctd_selection_worker_thread.quit)
        self.ctd_selection_worker.failed.connect(self.ctd_selection_worker_thread.quit)
        self.ctd_selection_worker_thread.finished.connect(self.cleanup_ctd_selection_worker)
        self.ctd_selection_worker_thread.start()

    def on_ctd_selection_finished(self, request_id: int, img_path: str, detected: object, detected_pixels: int) -> None:
        if request_id != self.ctd_selection_request_id or img_path != self.current_img_path:
            return
        if self.edit_mode != self.ctd_selection_edit_mode:
            self.status.showMessage('已切換編輯類型，CTD 選區結果已忽略。')
            return
        current = self.current_edit_mask()
        if current is None:
            return
        detected_mask = np.asarray(detected, dtype=np.uint8)
        if detected_mask.shape[:2] != current.shape[:2]:
            self.status.showMessage('CTD 選區結果尺寸不一致。')
            return
        detected_bool = detected_mask > 0
        if detected_pixels <= 0 or not np.any(detected_bool):
            self.status.showMessage('CTD 選區內沒有檢測到文字。')
            return
        add_pixels = detected_bool & (current == 0)
        if not np.any(add_pixels):
            self.status.showMessage('CTD 檢測結果已在目前 mask 中。')
            return
        self.push_undo_snapshot()
        current[detected_bool] = 255
        self.set_current_edit_mask(current)
        if self.edit_mode == 'mask':
            self.queue_background_sample()
        self.save_current_edit_mask()
        if self.current_base is not None:
            self.mask_view.set_mask(self.current_edit_mask(), self.current_base.shape[:2])
        self.refresh_mask_preview(keep_view=True)
        self.queue_auto_render()
        self.update_edit_buttons()
        self.status.showMessage(f'CTD 已添加選區檢測結果：{int(np.count_nonzero(add_pixels))} px。')

    def on_ctd_selection_failed(self, request_id: int, img_path: str, message: str) -> None:
        if request_id == self.ctd_selection_request_id and img_path == self.current_img_path:
            self.status.showMessage(f'CTD 選區檢測失敗：{message}')

    def cleanup_ctd_selection_worker(self) -> None:
        self.ctd_selection_worker = None
        self.ctd_selection_worker_thread = None

    def transfer_selection_from_other_masks(self, selection: np.ndarray) -> bool:
        current = self.mask_for_mode(self.edit_mode)
        if current is None:
            return False
        transfer = np.zeros(current.shape, dtype=bool)
        for mode in EDIT_MODE_LABELS:
            if mode == self.edit_mode:
                continue
            other = self.mask_for_mode(mode)
            if other is None:
                continue
            transfer |= (other > 0) & selection
        if not np.any(transfer):
            return False
        self.push_undo_snapshot()
        current[transfer] = 255
        for mode in EDIT_MODE_LABELS:
            if mode == self.edit_mode:
                continue
            other = self.mask_for_mode(mode)
            if other is not None:
                other[transfer] = 0
        return True

    def undo_mask(self) -> None:
        if self.current_mask is None or not self.undo_stack:
            return
        self.redo_stack.append(self.current_masks_snapshot())
        self.restore_masks_snapshot(self.undo_stack.pop())
        if self.current_base is not None:
            self.mask_view.set_mask(self.current_edit_mask(), self.current_base.shape[:2])
        self.queue_background_sample()
        self.save_all_edit_masks()
        self.refresh_mask_preview(keep_view=True)
        self.queue_auto_render()
        self.update_edit_buttons()

    def redo_mask(self) -> None:
        if self.current_mask is None or not self.redo_stack:
            return
        self.undo_stack.append(self.current_masks_snapshot())
        self.restore_masks_snapshot(self.redo_stack.pop())
        if self.current_base is not None:
            self.mask_view.set_mask(self.current_edit_mask(), self.current_base.shape[:2])
        self.queue_background_sample()
        self.save_all_edit_masks()
        self.refresh_mask_preview(keep_view=True)
        self.queue_auto_render()
        self.update_edit_buttons()

    def update_edit_buttons(self) -> None:
        self.undo_btn.setEnabled(bool(self.undo_stack))
        self.redo_btn.setEnabled(bool(self.redo_stack))

    def save_current_mask(self) -> None:
        if not self.current_img_path or self.current_mask is None:
            return
        imwrite(_mask_path(self.paths, self.current_img_path), self.current_mask)

    def save_current_edit_mask(self) -> None:
        if not self.current_img_path:
            return
        if self.edit_mode == 'manual_solid':
            if self.current_manual_solid is not None:
                _imwrite_transparent_mask(
                    _manual_solid_path(self.paths, self.current_img_path),
                    self.current_manual_solid,
                    EDIT_MODE_COLORS['manual_solid'],
                )
            return
        if self.edit_mode == 'manual_other':
            if self.current_manual_other is not None:
                _imwrite_transparent_mask(
                    _manual_other_path(self.paths, self.current_img_path),
                    self.current_manual_other,
                    EDIT_MODE_COLORS['manual_other'],
                )
            return
        self.save_current_mask()

    def save_all_edit_masks(self) -> None:
        if not self.current_img_path:
            return
        if self.current_mask is not None:
            self.save_current_mask()
        if self.current_manual_solid is not None:
            _imwrite_transparent_mask(
                _manual_solid_path(self.paths, self.current_img_path),
                self.current_manual_solid,
                EDIT_MODE_COLORS['manual_solid'],
            )
        if self.current_manual_other is not None:
            _imwrite_transparent_mask(
                _manual_other_path(self.paths, self.current_img_path),
                self.current_manual_other,
                EDIT_MODE_COLORS['manual_other'],
            )

    def queue_background_sample(self) -> None:
        self.background_sample_request_id += 1
        if (
            not self.show_background_sample
            or not self.current_img_path
            or self.current_base is None
            or self.current_mask is None
        ):
            self.pending_background_img_path = ''
            self.pending_background_mask = None
            self.background_sample_timer.stop()
            if self.background_worker is not None:
                self.background_worker.cancel()
            if hasattr(self, 'background_sample_status'):
                self.background_sample_status.setText('')
            return
        if self.background_worker is not None:
            self.background_worker.cancel()
        self.pending_background_img_path = self.current_img_path
        self.pending_background_mask = self.current_mask.copy()
        self.background_sample_status.setText('背景選區等待中...')
        self.background_sample_timer.start(450)

    def start_background_sample_worker(self) -> None:
        if (
            not self.pending_background_img_path
            or self.pending_background_mask is None
            or self.pending_background_img_path != self.current_img_path
            or self.current_base is None
        ):
            return
        if self.background_worker_thread is not None:
            self.background_sample_timer.start(450)
            return
        request_id = self.background_sample_request_id
        img_path = self.pending_background_img_path
        mask = self.pending_background_mask.copy()
        base = self.current_base.copy()
        self.pending_background_img_path = ''
        self.pending_background_mask = None
        self.background_sample_status.setText('背景選區計算中...')

        self.background_worker_thread = QThread()
        self.background_worker = BackgroundSampleWorker(request_id, img_path, base, mask)
        self.background_worker.moveToThread(self.background_worker_thread)
        self.background_worker_thread.started.connect(self.background_worker.run)
        self.background_worker.partial.connect(self.on_background_sample_partial)
        self.background_worker.finished.connect(self.on_background_sample_finished)
        self.background_worker.failed.connect(self.on_background_sample_failed)
        self.background_worker.finished.connect(self.background_worker_thread.quit)
        self.background_worker.failed.connect(self.background_worker_thread.quit)
        self.background_worker_thread.finished.connect(self.cleanup_background_worker)
        self.background_worker_thread.start()

    def on_background_sample_partial(
        self,
        request_id: int,
        img_path: str,
        mask_hash: int,
        sample: object,
    ) -> None:
        if (
            request_id != self.background_sample_request_id
            or img_path != self.current_img_path
            or mask_hash != _mask_hash(self.current_mask)
        ):
            return
        self.current_background_sample = np.asarray(sample, dtype=np.uint8)
        self.refresh_mask_preview(keep_view=True)

    def on_background_sample_finished(
        self,
        request_id: int,
        img_path: str,
        mask_hash: int,
        sample: object,
    ) -> None:
        if (
            request_id != self.background_sample_request_id
            or img_path != self.current_img_path
            or mask_hash != _mask_hash(self.current_mask)
        ):
            return
        self.current_background_sample = np.asarray(sample, dtype=np.uint8)
        _save_background_sample_cache(self.paths, img_path, mask_hash, self.current_background_sample)
        self.background_sample_status.setText('')
        self.refresh_mask_preview(keep_view=True)

    def on_background_sample_failed(self, request_id: int, img_path: str, mask_hash: int, message: str) -> None:
        if (
            request_id == self.background_sample_request_id
            and img_path == self.current_img_path
            and mask_hash == _mask_hash(self.current_mask)
        ):
            self.background_sample_status.setText('背景選區失敗')
        self.status.showMessage(f'背景選區計算失敗：{message}')

    def cleanup_background_worker(self) -> None:
        self.background_worker = None
        self.background_worker_thread = None
        if self.pending_background_img_path:
            self.background_sample_timer.start(50)

    def queue_auto_render(self) -> None:
        if not self.current_img_path or self.current_mask is None:
            return
        self.pending_render_img_path = self.current_img_path
        self.pending_render_mask = self.current_mask.copy()
        self.status.showMessage(f'{osp.basename(self.current_img_path)} 正在生成預覽...')
        self.render_timer.start(450)

    def start_pending_render(self) -> None:
        if not self.pending_render_img_path or self.pending_render_mask is None:
            return
        if self.worker_thread is not None:
            self.render_timer.start(450)
            return
        if self.page_worker_thread is not None:
            return
        img_path = self.pending_render_img_path
        mask = self.pending_render_mask.copy()
        self.pending_render_img_path = ''
        self.pending_render_mask = None
        self.page_worker_thread = QThread()
        self.page_worker = PageRegenerateWorker(self.folder, self.paths, self.imglist, img_path, mask)
        self.page_worker.moveToThread(self.page_worker_thread)
        self.page_worker_thread.started.connect(self.page_worker.run)
        self.page_worker.finished.connect(self.on_page_render_finished)
        self.page_worker.failed.connect(self.on_page_render_failed)
        self.page_worker.finished.connect(self.page_worker_thread.quit)
        self.page_worker.failed.connect(self.page_worker_thread.quit)
        self.page_worker_thread.finished.connect(self.cleanup_page_worker)
        self.page_worker_thread.start()

    def on_page_render_finished(self, img_path: str, report: dict) -> None:
        self.report = report
        if self.pending_render_img_path == img_path and self.pending_render_mask is not None:
            imwrite(_mask_path(self.paths, img_path), self.pending_render_mask)
        self.refresh_list()
        if img_path == self.current_img_path:
            if self.pending_render_img_path:
                self.status.showMessage('預覽已更新，等待最新編輯。')
            else:
                self.reload_current(keep_view=True)
                self.status.showMessage('預覽已更新。')
        if self.pending_render_img_path:
            self.render_timer.start(50)

    def on_page_render_failed(self, img_path: str, message: str) -> None:
        pages = dict(self.report.get('pages', {}))
        pages[osp.basename(img_path)] = {'error': message}
        self.report = build_report(self.folder, self.paths, self.imglist, pages)
        write_report(self.paths, self.report)
        self.refresh_list()
        self.status.showMessage(f'{osp.basename(img_path)} 預覽生成失敗：{message}')
        if self.pending_render_img_path:
            self.render_timer.start(50)

    def cleanup_page_worker(self) -> None:
        self.page_worker = None
        self.page_worker_thread = None
        if self.pending_render_img_path:
            self.render_timer.start(50)

    def generate_pdf(self) -> None:
        if not self.paths or not self.imglist:
            return
        if self.pdf_worker_thread is not None:
            QMessageBox.information(self, '正在生成 PDF', 'PDF 已在生成中。')
            return
        if self.page_worker_thread is not None or self.render_timer.isActive():
            QMessageBox.information(self, '正在生成預覽', '請等待當前頁預覽生成完成後再生成 PDF。')
            return
        report = self.report or load_report(self.paths)
        if not report:
            QMessageBox.information(self, '沒有報告', '請先偵測並生成，或載入已有輸出。')
            return
        self.status.showMessage('正在生成 PDF...')
        self.pdf_worker_thread = QThread()
        self.pdf_worker = PdfWorker(self.imglist, self.paths, report)
        self.pdf_worker.moveToThread(self.pdf_worker_thread)
        self.pdf_worker_thread.started.connect(self.pdf_worker.run)
        self.pdf_worker.finished.connect(self.on_pdf_finished)
        self.pdf_worker.failed.connect(self.on_pdf_failed)
        self.pdf_worker.finished.connect(self.pdf_worker_thread.quit)
        self.pdf_worker.failed.connect(self.pdf_worker_thread.quit)
        self.pdf_worker_thread.finished.connect(self.cleanup_pdf_worker)
        self.pdf_worker_thread.start()

    def on_pdf_finished(self, pdf_path: object) -> None:
        if pdf_path:
            self.status.showMessage(f'PDF 已生成：{pdf_path}')
        else:
            self.status.showMessage('沒有可生成 PDF 的頁面。')

    def on_pdf_failed(self, message: str) -> None:
        QMessageBox.critical(self, 'PDF 生成失敗', message)
        self.status.showMessage('PDF 生成失敗。')

    def cleanup_pdf_worker(self) -> None:
        self.pdf_worker = None
        self.pdf_worker_thread = None

    def show_help(self) -> None:
        QMessageBox.information(
            self,
            '塗白 說明',
            f'塗白 UI\n'
            f'版本：{APP_VERSION}\n\n'
            '滑鼠操作：\n'
            '筆刷左鍵：按「添加 / 減去」處理目前 mask\n'
            '右鍵拖拽：不分工具，矩形清除自動 / 強制純色 / 需要修改 mask\n'
            '筆刷 Shift + 按下拖到鬆開：連接按下和鬆開位置\n'
            '矩形 / 魔法棒左鍵：按「添加 / 減去 / 局部交集 / 選區內部 / 從其他轉入 / 添加CTD檢測選區」處理目前 mask\n'
            '局部交集：只裁切本次選區碰到的既有 mask 區塊\n'
            '交集偏移：局部交集結果正數擴展，負數收縮，0 保持原大小\n'
            '選區內部：魔法棒專用，提取本次選區包圍住的內部孔洞\n'
            '從其他轉入：把選區內其他 mask 的重疊部分移到目前 mask\n'
            '添加CTD檢測選區：只對本次選區跑 CTD，將檢測結果添加到目前 mask\n'
            '魔法棒：點擊相近的連續區域\n'
            'Command/Ctrl + 左鍵拖拽：平移畫布\n'
            '觸摸板雙指：移動畫布\n'
            '鼠標滾輪：移動畫布\n'
            'Command + 滾輪：左右移動畫布\n'
            'Option + 滾輪：以鼠標位置為中心縮放\n'
            '小地圖：拖標題列移動面板；拖框口或點擊縮圖移動畫布\n\n'
            '快捷鍵：\n'
            'Command/Ctrl + +：放大\n'
            'Command/Ctrl + -：縮小（保持頁面中心點）\n'
            '方向鍵：移動畫布\n'
            'A：上一頁\n'
            'D：下一頁\n'
            'F1：自動 mask\n'
            'F2：強制純色 mask\n'
            'F3：需要修改 mask\n'
            'F4：兩張圖同時適應窗口\n'
            'F5：矩形工具\n'
            'F6：筆刷工具\n'
            'F7：魔法棒工具\n'
            'F9：添加\n'
            'F10：減去\n'
            'F11：局部交集\n'
            'F12：從其他轉入\n'
            '[：縮小筆刷\n'
            ']：放大筆刷\n'
            'Ctrl+Z：撤銷\n'
            'Ctrl+Shift+Z：重做\n\n'
            '注意：綠色「偵測並生成」會重新跑 detector，可能覆蓋已有 mask。\n'
            '「使用傳入 Mask 運行」提供兩種方式：\n'
            '「取代目前 Mask」會選擇傳入 Mask 文件夾，用同名 PNG 覆蓋目前 mask 後重生成輸出。\n'
            '「取兩者交集」會選擇傳入 Mask 文件夾，用同名 PNG 與目前 mask 取交集後重生成輸出。',
        )

    def open_output(self) -> None:
        if not self.paths:
            return
        self._open_path(self.paths['output'])

    def open_pdf(self) -> None:
        if not self.paths:
            return
        pdf_path = osp.join(self.paths['output'], 'preview_report.pdf')
        if osp.isfile(pdf_path):
            self._open_path(pdf_path)

    def _open_path(self, path: str) -> None:
        if sys.platform == 'darwin':
            subprocess.Popen(['open', path])
        elif os.name == 'nt':
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(['xdg-open', path])


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName('塗白')
    app.setOrganizationName('ComicTextDetector')
    if APP_ICON_PATH.is_file():
        app.setWindowIcon(QIcon(str(APP_ICON_PATH)))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
