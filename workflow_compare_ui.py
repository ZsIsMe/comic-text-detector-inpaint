#!/usr/bin/env python3
"""Independent workflow comparison and masked compositing window."""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QImage, QKeySequence, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGraphicsLineItem,
    QGraphicsEllipseItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)


IMAGE_EXTENSIONS = {'.bmp', '.jpg', '.jpeg', '.png'}
BASE_CODE = 1
FIRST_WORKFLOW_CODE = 2
MAX_RECENT_FOLDERS = 12
MAX_UNDO_STEPS = 30
STATE_DIR_NAME = '.workflow_compare'
STATE_FILE_NAME = 'selection.json'
ASSIGNMENT_DIR_NAME = 'assignments'
RESULT_DIR_NAME = 'result'

WORKFLOW_COLORS = [
    (48, 196, 255),
    (82, 216, 144),
    (255, 139, 92),
    (232, 105, 184),
    (172, 128, 255),
    (255, 205, 76),
]


def _natural_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r'(\d+)', value)]


def _image_map(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        return {}
    result: dict[str, Path] = {}
    for path in sorted(folder.iterdir(), key=lambda item: _natural_key(item.name)):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            result.setdefault(path.stem, path)
    return result


def _read_image(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except (OSError, cv2.error):
        return None


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise OSError(f'無法編碼圖片：{path.name}')
    encoded.tofile(str(path))


def _as_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image[:, :, :3].copy()


def _qimage_from_bgr(image: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(_as_bgr(image), cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    return QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888).copy()


def _qimage_from_bgra(image: np.ndarray) -> QImage:
    rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    height, width = rgba.shape[:2]
    return QImage(rgba.data, width, height, rgba.strides[0], QImage.Format.Format_RGBA8888).copy()


@dataclass(frozen=True)
class WorkflowProject:
    root: Path
    pages: dict[str, Path]
    masks: dict[str, Path]
    workflows: dict[str, dict[str, Path]]

    @property
    def page_stems(self) -> list[str]:
        return sorted(self.pages, key=_natural_key)


def discover_export_pair(folder: str | Path) -> WorkflowProject:
    root = Path(folder).expanduser().resolve()
    pages = _image_map(root)
    masks = _image_map(root / 'other_mask')
    workflows_root = root / 'inpaint_workflows'
    workflows: dict[str, dict[str, Path]] = {}
    if workflows_root.is_dir():
        for child in sorted(workflows_root.iterdir(), key=lambda item: _natural_key(item.name)):
            if child.is_dir() and not child.name.startswith('.'):
                images = _image_map(child)
                if images:
                    workflows[child.name] = images
    if not pages:
        raise ValueError('所選文件夾根目錄沒有圖片。')
    if not (root / 'other_mask').is_dir():
        raise ValueError('所選文件夾內缺少 other_mask。')
    if not workflows:
        raise ValueError('inpaint_workflows 內沒有可用的工作流圖片文件夾。')
    return WorkflowProject(root, pages, masks, workflows)


def read_page_mask(project: WorkflowProject, stem: str, shape: tuple[int, int]) -> np.ndarray:
    mask_path = project.masks.get(stem)
    if mask_path is None:
        return np.zeros(shape, dtype=np.uint8)
    mask = _read_image(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != shape:
        return np.zeros(shape, dtype=np.uint8)
    return np.where(mask > 127, 255, 0).astype(np.uint8)


def expand_mask(mask: np.ndarray, expand_px: int) -> np.ndarray:
    radius = max(0, int(expand_px))
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    if radius == 0 or not np.any(binary):
        return binary
    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(binary, kernel, iterations=1)


def build_region_labels(mask: np.ndarray) -> tuple[np.ndarray, int]:
    count, labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    return labels.astype(np.int32), max(0, count - 1)


def compose_result(
    project: WorkflowProject,
    stem: str,
    assignment: np.ndarray,
    workflow_codes: dict[str, int],
    feather_px: int = 1,
) -> tuple[np.ndarray, list[str]]:
    base_raw = _read_image(project.pages[stem])
    if base_raw is None:
        raise OSError(f'無法讀取底圖：{project.pages[stem].name}')
    base = _as_bgr(base_raw)
    if assignment.shape != base.shape[:2]:
        raise ValueError(f'{project.pages[stem].name} 的選擇尺寸與底圖不一致。')

    sources: list[tuple[np.ndarray, np.ndarray]] = []
    missing: list[str] = []
    for workflow_name, code in workflow_codes.items():
        selected = assignment == code
        if not np.any(selected):
            continue
        source_path = project.workflows.get(workflow_name, {}).get(stem)
        source_raw = _read_image(source_path) if source_path is not None else None
        if source_raw is None:
            missing.append(workflow_name)
            continue
        source = _as_bgr(source_raw)
        if source.shape != base.shape:
            missing.append(f'{workflow_name}（尺寸不同）')
            continue
        sources.append((selected, source))

    if feather_px <= 0:
        output = base.copy()
        for selected, source in sources:
            output[selected] = source[selected]
        return output, missing

    base_float = base.astype(np.float32)
    weighted_sources = np.zeros_like(base_float)
    alpha_sum = np.zeros(base.shape[:2], dtype=np.float32)
    sigma = max(0.35, float(feather_px) * 0.65)
    allowed = assignment > BASE_CODE
    for selected, source in sources:
        alpha = cv2.GaussianBlur(selected.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
        alpha *= allowed.astype(np.float32)
        weighted_sources += source.astype(np.float32) * alpha[:, :, None]
        alpha_sum += alpha
    mix = np.clip(alpha_sum, 0.0, 1.0)
    normalized = weighted_sources / np.maximum(alpha_sum[:, :, None], 1e-6)
    output = base_float * (1.0 - mix[:, :, None]) + normalized * mix[:, :, None]
    output[alpha_sum <= 1e-6] = base_float[alpha_sum <= 1e-6]
    return np.clip(output, 0, 255).astype(np.uint8), missing


class ClippedPixmapItem(QGraphicsPixmapItem):
    def __init__(self) -> None:
        super().__init__()
        self.fraction = 0.0

    def set_fraction(self, fraction: float) -> None:
        self.fraction = max(0.0, min(1.0, float(fraction)))
        self.update()

    def paint(self, painter: QPainter, option, widget=None) -> None:
        pixmap = self.pixmap()
        if pixmap.isNull() or self.fraction <= 0:
            return
        painter.save()
        painter.setClipRect(QRectF(0, 0, pixmap.width() * self.fraction, pixmap.height()))
        super().paint(painter, option, widget)
        painter.restore()


class WorkflowCompareView(QGraphicsView):
    compareChanged = Signal(float)
    assignStarted = Signal(str)
    assignPoint = Signal(str, int, int)
    assignRect = Signal(str, int, int, int, int, bool)
    assignFinished = Signal(str)
    viewChanged = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.result_item = QGraphicsPixmapItem()
        self.original_item = ClippedPixmapItem()
        self.overlay_item = QGraphicsPixmapItem()
        self.divider_item = QGraphicsLineItem()
        self.selection_rect_item = QGraphicsRectItem()
        self.brush_cursor_item = QGraphicsEllipseItem()
        self.result_item.setZValue(0)
        self.original_item.setZValue(1)
        self.overlay_item.setZValue(2)
        self.divider_item.setZValue(3)
        self.selection_rect_item.setZValue(4)
        self.brush_cursor_item.setZValue(4)
        self.scene().addItem(self.result_item)
        self.scene().addItem(self.original_item)
        self.scene().addItem(self.overlay_item)
        self.scene().addItem(self.divider_item)
        self.scene().addItem(self.selection_rect_item)
        self.scene().addItem(self.brush_cursor_item)
        self.divider_item.setPen(QPen(QColor('#f7f3e8'), 2))
        selection_pen = QPen(QColor('#62d7c5'), 2, Qt.PenStyle.DashLine)
        selection_pen.setCosmetic(True)
        self.selection_rect_item.setPen(selection_pen)
        self.selection_rect_item.setBrush(QColor(72, 204, 183, 40))
        self.selection_rect_item.hide()
        brush_pen = QPen(QColor('#62d7c5'), 1)
        brush_pen.setCosmetic(True)
        self.brush_cursor_item.setPen(brush_pen)
        self.brush_cursor_item.setBrush(Qt.BrushStyle.NoBrush)
        self.brush_cursor_item.hide()
        self.setBackgroundBrush(QColor('#0b0f14'))
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setMouseTracking(True)
        self.workflow_name = ''
        self.brush_radius = 32
        self.compare_fraction = 0.0
        self.interaction_mode = 'rect'
        self._dragging_divider = False
        self._assigning = False
        self._panning = False
        self._last_mouse_pos = QPoint()
        self._rect_start: QPointF | None = None
        self._rect_use_base = False
        self._assign_button = Qt.MouseButton.NoButton
        self._suppress_view_changed = False
        self.horizontalScrollBar().valueChanged.connect(self._emit_view_changed)
        self.verticalScrollBar().valueChanged.connect(self._emit_view_changed)

    def _emit_view_changed(self) -> None:
        if not self._suppress_view_changed:
            self.viewChanged.emit()

    def set_images(self, original: QImage | None, result: QImage | None, keep_view: bool = False) -> None:
        old_transform = self.transform()
        old_center = self.mapToScene(self.viewport().rect().center())
        original_pixmap = QPixmap.fromImage(original) if original is not None else QPixmap()
        result_pixmap = QPixmap.fromImage(result) if result is not None else original_pixmap
        self.result_item.setPixmap(result_pixmap)
        self.original_item.setPixmap(original_pixmap)
        rect = result_pixmap.rect() if not result_pixmap.isNull() else original_pixmap.rect()
        self.scene().setSceneRect(QRectF(rect))
        self._update_divider()
        if keep_view and not original_pixmap.isNull():
            self.setTransform(old_transform)
            self.centerOn(old_center)
        else:
            self.fit_image()

    def set_overlay(self, overlay: QImage | None) -> None:
        self.overlay_item.setPixmap(QPixmap.fromImage(overlay) if overlay is not None else QPixmap())

    def set_compare_fraction(self, fraction: float, emit: bool = False) -> None:
        self.compare_fraction = max(0.0, min(1.0, float(fraction)))
        self.original_item.set_fraction(self.compare_fraction)
        self._update_divider()
        if emit:
            self.compareChanged.emit(self.compare_fraction)

    def _update_divider(self) -> None:
        height = self.sceneRect().height()
        x = self.sceneRect().width() * self.compare_fraction
        self.divider_item.setLine(x, 0, x, height)

    def fit_image(self) -> None:
        if self.sceneRect().isEmpty():
            return
        self.resetTransform()
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._emit_view_changed()

    def copy_view_from(self, source: 'WorkflowCompareView') -> None:
        if self.sceneRect().isEmpty() or source.sceneRect().isEmpty():
            return
        center = source.mapToScene(source.viewport().rect().center())
        self._suppress_view_changed = True
        try:
            self.setTransform(source.transform())
            self.centerOn(center)
        finally:
            self._suppress_view_changed = False

    def _scene_image_point(self, pos: QPoint) -> tuple[int, int] | None:
        point = self.mapToScene(pos)
        rect = self.sceneRect()
        if not rect.contains(point):
            return None
        return int(point.x()), int(point.y())

    def _divider_is_near(self, pos: QPoint) -> bool:
        x = self.sceneRect().width() * self.compare_fraction
        divider_view_x = self.mapFromScene(QPointF(x, 0)).x()
        return abs(pos.x() - divider_view_x) <= 12

    def _set_compare_from_position(self, pos: QPoint) -> None:
        point = self.mapToScene(pos)
        width = max(1.0, self.sceneRect().width())
        self.set_compare_fraction(point.x() / width, emit=True)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._divider_is_near(event.position().toPoint()):
            self._dragging_divider = True
            self._set_compare_from_position(event.position().toPoint())
            event.accept()
            return
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            point = self._scene_image_point(event.position().toPoint())
            if point is not None:
                self._assigning = True
                self._assign_button = event.button()
                self.assignStarted.emit(self.workflow_name)
                if self.interaction_mode == 'brush' and event.button() == Qt.MouseButton.LeftButton:
                    self.assignPoint.emit(self.workflow_name, point[0], point[1])
                else:
                    self._rect_use_base = event.button() == Qt.MouseButton.RightButton
                    rect_color = QColor('#f3d88a') if self._rect_use_base else QColor('#62d7c5')
                    rect_pen = QPen(rect_color, 2, Qt.PenStyle.DashLine)
                    rect_pen.setCosmetic(True)
                    self.selection_rect_item.setPen(rect_pen)
                    self.selection_rect_item.setBrush(
                        QColor(243, 216, 138, 36)
                        if self._rect_use_base
                        else QColor(72, 204, 183, 40)
                    )
                    self._rect_start = QPointF(point[0], point[1])
                    self.selection_rect_item.setRect(QRectF(self._rect_start, self._rect_start))
                    self.selection_rect_item.show()
                event.accept()
                return
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._last_mouse_pos = event.position().toPoint()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        pos = event.position().toPoint()
        if self._dragging_divider:
            self._set_compare_from_position(pos)
            event.accept()
            return
        if self._assigning and self.interaction_mode == 'brush' and self._rect_start is None:
            point = self._scene_image_point(pos)
            if point is not None:
                self.assignPoint.emit(self.workflow_name, point[0], point[1])
            event.accept()
            return
        if self._assigning and self._rect_start is not None:
            point = self.mapToScene(pos)
            image_rect = self.sceneRect()
            point.setX(max(image_rect.left(), min(image_rect.right(), point.x())))
            point.setY(max(image_rect.top(), min(image_rect.bottom(), point.y())))
            self.selection_rect_item.setRect(QRectF(self._rect_start, point).normalized())
            event.accept()
            return
        if self._panning:
            delta = pos - self._last_mouse_pos
            self._last_mouse_pos = pos
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self._emit_view_changed()
            event.accept()
            return
        if self.interaction_mode == 'brush':
            point = self.mapToScene(pos)
            if self.sceneRect().contains(point):
                radius = self.brush_radius
                self.brush_cursor_item.setRect(
                    point.x() - radius,
                    point.y() - radius,
                    radius * 2,
                    radius * 2,
                )
                self.brush_cursor_item.show()
            else:
                self.brush_cursor_item.hide()
        else:
            self.brush_cursor_item.hide()
        self.viewport().setCursor(
            Qt.CursorShape.SplitHCursor if self._divider_is_near(pos) else Qt.CursorShape.CrossCursor
        )
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._dragging_divider and event.button() == Qt.MouseButton.LeftButton:
            self._dragging_divider = False
            event.accept()
            return
        if self._assigning and event.button() == self._assign_button:
            self._assigning = False
            if self._rect_start is not None:
                rect = self.selection_rect_item.rect().normalized().intersected(self.sceneRect())
                self.selection_rect_item.hide()
                self._rect_start = None
                if rect.width() >= 1 and rect.height() >= 1:
                    self.assignRect.emit(
                        self.workflow_name,
                        max(0, int(rect.left())),
                        max(0, int(rect.top())),
                        max(1, int(np.ceil(rect.width()))),
                        max(1, int(np.ceil(rect.height()))),
                        self._rect_use_base,
                    )
            self._rect_use_base = False
            self._assign_button = Qt.MouseButton.NoButton
            self.assignFinished.emit(self.workflow_name)
            event.accept()
            return
        if self._panning and event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self.viewport().unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        if self.sceneRect().isEmpty():
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        anchor = self.mapToScene(event.position().toPoint())
        factor = 1.15 if delta > 0 else 1 / 1.15
        self.scale(factor, factor)
        self.centerOn(anchor)
        self._emit_view_changed()
        event.accept()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._emit_view_changed()

    def leaveEvent(self, event) -> None:
        self.brush_cursor_item.hide()
        super().leaveEvent(event)


class WorkflowPanel(QFrame):
    workflowChanged = Signal(object)
    applyWholeRequested = Signal(str)
    compareChanged = Signal(float)

    def __init__(self) -> None:
        super().__init__()
        self.setProperty('panel', True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(7)
        header = QHBoxLayout()
        self.color_swatch = QLabel()
        self.color_swatch.setFixedSize(18, 18)
        self.color_swatch.setToolTip('此工作流在選區中的顏色')
        header.addWidget(self.color_swatch)
        self.workflow_combo = QComboBox()
        self.workflow_combo.currentIndexChanged.connect(self._workflow_index_changed)
        header.addWidget(self.workflow_combo, 1)
        self.availability_label = QLabel('')
        self.availability_label.setProperty('muted', True)
        header.addWidget(self.availability_label)
        layout.addLayout(header)
        self.view = WorkflowCompareView()
        layout.addWidget(self.view, 1)
        self.compare_slider = QSlider(Qt.Orientation.Horizontal, self.view.viewport())
        self.compare_slider.setRange(0, 1000)
        self.compare_slider.setValue(0)
        self.compare_slider.setToolTip('原圖顯示範圍；所有圖片同步')
        self.compare_slider.setStyleSheet(
            """
            QSlider { background: rgba(13, 18, 23, 185); border-radius: 10px; }
            QSlider::groove:horizontal { background: #4a5661; height: 5px; border-radius: 2px; }
            QSlider::sub-page:horizontal { background: #e1bd62; border-radius: 2px; }
            QSlider::handle:horizontal { background: #fff4cf; border: 2px solid #b9943d; width: 13px; margin: -5px 0; border-radius: 7px; }
            """
        )
        self.compare_slider.valueChanged.connect(
            lambda value: self.compareChanged.emit(value / 1000.0)
        )
        self.view.viewChanged.connect(self.update_compare_slider_geometry)
        self.apply_whole_button = QPushButton('整個 Mask 採用這組')
        self.apply_whole_button.clicked.connect(self._request_apply_whole)
        layout.addWidget(self.apply_whole_button)
        self._workflow_names: list[str] = []
        QTimer.singleShot(0, self.update_compare_slider_geometry)

    def _workflow_index_changed(self) -> None:
        self._update_workflow_color()
        self.workflowChanged.emit(self)

    def _update_workflow_color(self) -> None:
        index = self.workflow_combo.currentIndex()
        if index < 0:
            self.color_swatch.setStyleSheet('background: transparent; border: 1px solid #4b5660; border-radius: 3px;')
            return
        blue, green, red = WORKFLOW_COLORS[index % len(WORKFLOW_COLORS)]
        color = f'#{red:02x}{green:02x}{blue:02x}'
        self.color_swatch.setStyleSheet(
            f'background: {color}; border: 1px solid rgba(255,255,255,110); border-radius: 3px;'
        )

    def _request_apply_whole(self) -> None:
        workflow = self.current_workflow()
        if workflow:
            self.applyWholeRequested.emit(workflow)

    def current_workflow(self) -> str:
        return str(self.workflow_combo.currentData() or '')

    def set_workflows(self, names: list[str], selected_index: int) -> None:
        self._workflow_names = list(names)
        self.workflow_combo.blockSignals(True)
        self.workflow_combo.clear()
        for name in names:
            self.workflow_combo.addItem(name, name)
        if names:
            self.workflow_combo.setCurrentIndex(min(selected_index, len(names) - 1))
        self.workflow_combo.blockSignals(False)
        self.view.workflow_name = self.current_workflow()
        self._update_workflow_color()

    def set_compare_fraction(self, fraction: float) -> None:
        self.compare_slider.blockSignals(True)
        self.compare_slider.setValue(round(max(0.0, min(1.0, fraction)) * 1000))
        self.compare_slider.blockSignals(False)
        self.view.set_compare_fraction(fraction)

    def update_compare_slider_geometry(self) -> None:
        if self.view.sceneRect().isEmpty():
            self.compare_slider.hide()
            return
        image_rect = self.view.mapFromScene(self.view.sceneRect()).boundingRect()
        visible_rect = image_rect.intersected(self.view.viewport().rect())
        if visible_rect.width() < 40 or visible_rect.height() < 24:
            self.compare_slider.hide()
            return
        slider_height = 22
        slider_y = max(0, visible_rect.top())
        self.compare_slider.setGeometry(
            visible_rect.left(), slider_y, visible_rect.width(), slider_height
        )
        self.compare_slider.show()
        self.compare_slider.raise_()


class ResultPreviewView(QGraphicsView):
    viewChanged = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene().addItem(self.pixmap_item)
        self.setBackgroundBrush(QColor('#0b0f14'))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._panning = False
        self._last_mouse_pos = QPoint()
        self._suppress_view_changed = False
        self.horizontalScrollBar().valueChanged.connect(self._emit_view_changed)
        self.verticalScrollBar().valueChanged.connect(self._emit_view_changed)

    def _emit_view_changed(self) -> None:
        if not self._suppress_view_changed:
            self.viewChanged.emit()

    def set_image(self, image: QImage | None) -> None:
        old_size = self.pixmap_item.pixmap().size()
        old_transform = self.transform()
        old_center = self.mapToScene(self.viewport().rect().center())
        pixmap = QPixmap.fromImage(image) if image is not None else QPixmap()
        self.pixmap_item.setPixmap(pixmap)
        self.scene().setSceneRect(QRectF(pixmap.rect()))
        if not pixmap.isNull() and pixmap.size() == old_size:
            self.setTransform(old_transform)
            self.centerOn(old_center)
        else:
            self.fit_image()

    def fit_image(self) -> None:
        if self.pixmap_item.pixmap().isNull():
            return
        self.resetTransform()
        self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._emit_view_changed()

    def copy_view_from(self, source: QGraphicsView) -> None:
        if self.sceneRect().isEmpty() or source.sceneRect().isEmpty():
            return
        center = source.mapToScene(source.viewport().rect().center())
        self._suppress_view_changed = True
        try:
            self.setTransform(source.transform())
            self.centerOn(center)
        finally:
            self._suppress_view_changed = False

    def mousePressEvent(self, event) -> None:
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            self._panning = True
            self._last_mouse_pos = event.position().toPoint()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._panning:
            pos = event.position().toPoint()
            delta = pos - self._last_mouse_pos
            self._last_mouse_pos = pos
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self._emit_view_changed()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._panning:
            self._panning = False
            self.viewport().unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta == 0 or self.pixmap_item.pixmap().isNull():
            return
        anchor = self.mapToScene(event.position().toPoint())
        self.scale(1.15 if delta > 0 else 1 / 1.15, 1.15 if delta > 0 else 1 / 1.15)
        self.centerOn(anchor)
        self._emit_view_changed()
        event.accept()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._emit_view_changed()


class ResultPreviewPanel(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setProperty('panel', True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(7)
        header = QHBoxLayout()
        title = QLabel('當頁實際效果')
        title.setProperty('resultTitle', True)
        header.addWidget(title)
        header.addStretch(1)
        self.page_label = QLabel('')
        self.page_label.setProperty('muted', True)
        header.addWidget(self.page_label)
        layout.addLayout(header)
        self.view = ResultPreviewView()
        layout.addWidget(self.view, 1)
        footer = QLabel('最終合成')
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setProperty('resultFooter', True)
        layout.addWidget(footer)

    def set_result(self, page_name: str, image: np.ndarray | None, warning: str = '') -> None:
        self.page_label.setText(warning or page_name)
        self.page_label.setToolTip(page_name + (f'\n{warning}' if warning else ''))
        self.view.set_image(_qimage_from_bgr(image) if image is not None else None)


class WorkflowCompareWindow(QMainWindow):
    """Compare any number of workflow folders and composite by mask area."""

    def __init__(self, initial_folder: str = '', parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle('工作流比較與合成')
        self.resize(1580, 920)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.settings = QSettings('ComicTextDetector', 'WorkflowCompareUI')
        try:
            saved_mask_expand = int(self.settings.value('mask_expand_px', 5))
        except (TypeError, ValueError):
            saved_mask_expand = 5
        self.mask_expand_px = max(0, min(80, saved_mask_expand))
        self.project: WorkflowProject | None = None
        self.current_stem = ''
        self.current_base: np.ndarray | None = None
        self.current_mask: np.ndarray | None = None
        self.assignment: np.ndarray | None = None
        self.workflow_codes: dict[str, int] = {}
        self.undo_stack: list[np.ndarray] = []
        self.redo_stack: list[np.ndarray] = []
        self.compare_fraction = 0.0
        self._syncing_views = False
        self._stroke_changed = False
        self.initialized_pages: set[str] = set()
        self.recent_folders = self._load_recent_folders()
        self._build_ui()
        self._apply_style()
        if initial_folder:
            self.load_folder(initial_folder)
        elif self.recent_folders:
            self.load_folder(self.recent_folders[0], quiet=True)

    def _build_ui(self) -> None:
        toolbar = QToolBar('工作流比較')
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        choose_action = QAction('選擇 export_pair', self)
        choose_action.triggered.connect(self.choose_folder)
        toolbar.addAction(choose_action)

        fit_action = QAction('適合視窗', self)
        fit_action.triggered.connect(self.fit_views)
        toolbar.addAction(fit_action)

        self.undo_action = QAction('復原', self)
        self.undo_action.setShortcut('Ctrl+Z')
        self.undo_action.triggered.connect(self.undo)
        toolbar.addAction(self.undo_action)
        self.redo_action = QAction('重做', self)
        self.redo_action.setShortcut('Ctrl+Shift+Z')
        self.redo_action.triggered.connect(self.redo)
        toolbar.addAction(self.redo_action)

        toolbar.addSeparator()
        self.show_regions_checkbox = QCheckBox('M 顯示選區')
        self.show_regions_checkbox.setChecked(False)
        self.show_regions_checkbox.toggled.connect(self.refresh_overlays)
        toolbar.addWidget(self.show_regions_checkbox)

        toggle_regions_action = QAction('顯示／隱藏選區', self)
        toggle_regions_action.setShortcut(QKeySequence('M'))
        toggle_regions_action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        toggle_regions_action.triggered.connect(self.show_regions_checkbox.toggle)
        self.addAction(toggle_regions_action)

        brush_down_action = QAction('縮小筆刷', self)
        brush_down_action.setShortcut(QKeySequence('['))
        brush_down_action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        brush_down_action.triggered.connect(lambda: self._change_brush_size(-4))
        self.addAction(brush_down_action)
        brush_up_action = QAction('放大筆刷', self)
        brush_up_action.setShortcut(QKeySequence(']'))
        brush_up_action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        brush_up_action.triggered.connect(lambda: self._change_brush_size(4))
        self.addAction(brush_up_action)

        toolbar.addSeparator()
        output_action = QAction('輸出結果', self)
        output_action.triggered.connect(self.export_results)
        toolbar.addAction(output_action)

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(10, 8, 10, 10)
        root_layout.setSpacing(8)

        folder_row = QHBoxLayout()
        self.folder_label = QLabel('尚未選擇 export_pair')
        self.folder_label.setProperty('folder', True)
        folder_row.addWidget(self.folder_label, 1)
        self.progress_label = QLabel('')
        self.progress_label.setProperty('muted', True)
        folder_row.addWidget(self.progress_label)
        root_layout.addLayout(folder_row)

        controls = QFrame()
        controls.setProperty('controlBar', True)
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(10, 7, 10, 7)
        controls_layout.addWidget(QLabel('套用方式'))
        self.rectangle_button = QPushButton('矩形')
        self.rectangle_button.setCheckable(True)
        self.rectangle_button.setChecked(True)
        self.brush_button = QPushButton('筆刷')
        self.brush_button.setCheckable(True)
        mode_group = QButtonGroup(self)
        mode_group.setExclusive(True)
        mode_group.addButton(self.rectangle_button)
        mode_group.addButton(self.brush_button)
        self.rectangle_button.toggled.connect(self._update_interaction_mode)
        self.brush_button.toggled.connect(self._update_interaction_mode)
        controls_layout.addWidget(self.rectangle_button)
        controls_layout.addWidget(self.brush_button)
        controls_layout.addSpacing(8)
        controls_layout.addWidget(QLabel('筆刷'))
        self.brush_size = QSpinBox()
        self.brush_size.setRange(2, 160)
        self.brush_size.setValue(32)
        self.brush_size.setSuffix(' px')
        self.brush_size.valueChanged.connect(self._brush_size_changed)
        controls_layout.addWidget(self.brush_size)
        controls_layout.addSpacing(8)
        controls_layout.addWidget(QLabel('Mask 擴大'))
        self.mask_expand_size = QSpinBox()
        self.mask_expand_size.setRange(0, 80)
        self.mask_expand_size.setValue(self.mask_expand_px)
        self.mask_expand_size.setSuffix(' px')
        self.mask_expand_size.setToolTip('向外擴大可採用工作流結果的 Mask 範圍')
        self.mask_expand_size.valueChanged.connect(self._mask_expand_changed)
        controls_layout.addWidget(self.mask_expand_size)
        controls_layout.addSpacing(8)
        controls_layout.addWidget(QLabel('邊緣羽化'))
        self.feather_size = QSpinBox()
        self.feather_size.setRange(0, 8)
        try:
            saved_feather = int(self.settings.value('feather_px', 1))
        except (TypeError, ValueError):
            saved_feather = 1
        self.feather_size.setValue(max(0, min(8, saved_feather)))
        self.feather_size.setSuffix(' px')
        self.feather_size.valueChanged.connect(self._feather_changed)
        controls_layout.addWidget(self.feather_size)
        controls_layout.addStretch(1)
        self.clear_button = QPushButton('重設為第一組')
        self.clear_button.clicked.connect(self.reset_page_assignment)
        controls_layout.addWidget(self.clear_button)
        self.base_whole_button = QPushButton('整個 Mask 保留底圖')
        self.base_whole_button.clicked.connect(lambda: self.apply_whole(''))
        controls_layout.addWidget(self.base_whole_button)
        root_layout.addWidget(controls)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QFrame()
        left.setProperty('sidebar', True)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.addWidget(QLabel('圖片'))
        self.page_list = QListWidget()
        self.page_list.currentRowChanged.connect(self._page_row_changed)
        left_layout.addWidget(self.page_list, 1)
        splitter.addWidget(left)

        workspace = QWidget()
        workspace_layout = QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(8)

        panel_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.result_panel = ResultPreviewPanel()
        self.result_panel.view.viewChanged.connect(
            lambda source=self.result_panel.view: self._sync_views_from(source)
        )
        panel_splitter.addWidget(self.result_panel)
        self.panels: list[WorkflowPanel] = []
        for _ in range(3):
            panel = WorkflowPanel()
            panel.workflowChanged.connect(self._panel_workflow_changed)
            panel.applyWholeRequested.connect(self.apply_whole)
            panel.compareChanged.connect(self._view_compare_changed)
            panel.view.compareChanged.connect(self._view_compare_changed)
            panel.view.assignStarted.connect(self._assignment_started)
            panel.view.assignPoint.connect(self._assign_point)
            panel.view.assignRect.connect(self._assign_rect)
            panel.view.assignFinished.connect(self._assignment_finished)
            panel.view.viewChanged.connect(lambda source=panel.view: self._sync_views_from(source))
            self.panels.append(panel)
            panel_splitter.addWidget(panel)
        panel_splitter.setSizes([1, 1, 1, 1])
        workspace_layout.addWidget(panel_splitter, 1)

        self.hint_label = QLabel(
            '左鍵矩形採用工作流；右鍵矩形保留原圖；中鍵拖動畫面；M 顯示選區。'
        )
        self.hint_label.setProperty('muted', True)
        workspace_layout.addWidget(self.hint_label)
        splitter.addWidget(workspace)
        splitter.setSizes([210, 1370])
        root_layout.addWidget(splitter, 1)
        self.setCentralWidget(root)
        self.statusBar().showMessage('請選擇 export_pair 文件夾。')
        self._update_interaction_mode()
        self._brush_size_changed(self.brush_size.value())
        self._update_actions()

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #14191f; color: #e9edf1; }
            QToolBar { background: #1b222a; border: 0; border-bottom: 1px solid #303944; spacing: 5px; padding: 5px; }
            QToolBar QToolButton { background: #27313b; border: 1px solid #35414d; border-radius: 5px; padding: 6px 10px; }
            QToolBar QToolButton:hover { background: #32404c; }
            QLabel[folder="true"] { color: #f5d889; font-weight: 600; }
            QLabel[muted="true"] { color: #95a2ad; }
            QLabel[resultTitle="true"] { color: #77dfc6; font-weight: 700; }
            QLabel[resultFooter="true"] { background: #22433f; color: #aef0df; border: 1px solid #38665f; border-radius: 5px; padding: 7px; font-weight: 600; }
            QFrame[sidebar="true"], QFrame[panel="true"], QFrame[controlBar="true"] { background: #1b222a; border: 1px solid #2b3540; border-radius: 7px; }
            QListWidget { background: #11161b; border: 0; border-radius: 5px; padding: 4px; }
            QListWidget::item { padding: 7px 6px; border-radius: 4px; }
            QListWidget::item:selected { background: #315c62; color: white; }
            QPushButton, QComboBox, QSpinBox { background: #27313b; border: 1px solid #3a4753; border-radius: 5px; padding: 6px 9px; }
            QPushButton:hover, QComboBox:hover { border-color: #55b8aa; background: #2d3944; }
            QPushButton:checked { background: #2d7169; border-color: #67cbbb; }
            QPushButton:disabled { color: #66727c; background: #20272e; }
            QSlider::groove:horizontal { background: #303944; height: 6px; border-radius: 3px; }
            QSlider::handle:horizontal { background: #f5d889; border: 2px solid #bd9b43; width: 14px; margin: -5px 0; border-radius: 7px; }
            QStatusBar { background: #101419; color: #aab4bd; }
            """
        )

    def choose_folder(self) -> None:
        start = self.recent_folders[0] if self.recent_folders else str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, '選擇 export_pair 文件夾', start)
        if folder:
            self.load_folder(folder)

    def load_folder(self, folder: str, quiet: bool = False) -> None:
        self.save_current_assignment()
        try:
            project = discover_export_pair(folder)
        except ValueError as exc:
            if not quiet:
                QMessageBox.warning(self, '無法載入', str(exc))
            return
        self.project = project
        self.current_stem = ''
        self.current_base = None
        self.current_mask = None
        self.assignment = None
        saved_state = self._load_state()
        saved_expand = saved_state.get('mask_expand_px')
        if isinstance(saved_expand, int):
            self.mask_expand_px = max(0, min(80, saved_expand))
            self.mask_expand_size.blockSignals(True)
            self.mask_expand_size.setValue(self.mask_expand_px)
            self.mask_expand_size.blockSignals(False)
        self.workflow_codes = self._load_workflow_codes(project)
        initialized = saved_state.get('initialized_pages', [])
        self.initialized_pages = {
            str(stem) for stem in initialized if isinstance(stem, str) and stem in project.pages
        } if isinstance(initialized, list) else set()
        if int(saved_state.get('version', 0) or 0) < 2:
            for stem in project.page_stems:
                legacy_path = self._assignment_path(stem)
                legacy = _read_image(legacy_path, cv2.IMREAD_UNCHANGED) if legacy_path.is_file() else None
                if legacy is not None and np.any(legacy):
                    self.initialized_pages.add(stem)
        self.folder_label.setText(str(project.root))
        self._remember_folder(project.root)
        names = list(project.workflows)
        for index, panel in enumerate(self.panels):
            panel.setVisible(index < min(3, len(names)))
            panel.set_workflows(names, index)
        self._populate_pages()
        if self.page_list.count():
            saved_stem = str(self._load_state().get('current_page', ''))
            target_row = project.page_stems.index(saved_stem) if saved_stem in project.pages else 0
            self.page_list.setCurrentRow(target_row)
        self.statusBar().showMessage(f'已載入 {len(project.pages)} 張圖片、{len(project.workflows)} 組工作流。')

    def _state_dir(self) -> Path | None:
        return self.project.root / STATE_DIR_NAME if self.project is not None else None

    def _load_state(self) -> dict:
        state_dir = self._state_dir()
        if state_dir is None:
            return {}
        path = state_dir / STATE_FILE_NAME
        try:
            value = json.loads(path.read_text(encoding='utf-8'))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_state(self) -> None:
        if self.project is None:
            return
        try:
            state_dir = self.project.root / STATE_DIR_NAME
            state_dir.mkdir(parents=True, exist_ok=True)
            state = {
                'version': 2,
                'current_page': self.current_stem,
                'workflow_codes': self.workflow_codes,
                'initialized_pages': sorted(self.initialized_pages, key=_natural_key),
                'mask_expand_px': self.mask_expand_px,
                'feather_px': self.feather_size.value(),
            }
            temporary = state_dir / f'{STATE_FILE_NAME}.tmp'
            temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
            os.replace(temporary, state_dir / STATE_FILE_NAME)
        except OSError as exc:
            self.statusBar().showMessage(f'無法保存工作流設定：{exc}')

    def _load_workflow_codes(self, project: WorkflowProject) -> dict[str, int]:
        saved = self._load_state().get('workflow_codes', {})
        codes: dict[str, int] = {}
        used = {BASE_CODE}
        if isinstance(saved, dict):
            for name in project.workflows:
                value = saved.get(name)
                if isinstance(value, int) and value >= FIRST_WORKFLOW_CODE and value not in used:
                    codes[name] = value
                    used.add(value)
        next_code = FIRST_WORKFLOW_CODE
        for name in project.workflows:
            if name in codes:
                continue
            while next_code in used:
                next_code += 1
            codes[name] = next_code
            used.add(next_code)
        return codes

    def _assignment_path(self, stem: str) -> Path:
        assert self.project is not None
        return self.project.root / STATE_DIR_NAME / ASSIGNMENT_DIR_NAME / f'{stem}.png'

    def _load_assignment(self, stem: str, shape: tuple[int, int]) -> np.ndarray:
        path = self._assignment_path(stem)
        saved = _read_image(path, cv2.IMREAD_UNCHANGED) if path.is_file() else None
        if saved is not None and saved.ndim == 3:
            saved = saved[:, :, 0]
        if saved is not None and saved.shape == shape and stem in self.initialized_pages:
            assignment = saved.astype(np.uint16)
        else:
            assignment = np.zeros(shape, dtype=np.uint16)
        if self.project is None:
            return assignment
        mask = self._page_effective_mask(stem, shape) > 0
        default_code = self._default_workflow_code(stem)
        assignment[~mask] = 0
        assignment[mask & (assignment == 0)] = default_code
        return assignment

    def _page_effective_mask(self, stem: str, shape: tuple[int, int]) -> np.ndarray:
        assert self.project is not None
        return expand_mask(read_page_mask(self.project, stem, shape), self.mask_expand_px)

    def _default_workflow_code(self, stem: str) -> int:
        if self.project is None:
            return BASE_CODE
        for workflow_name, pages in self.project.workflows.items():
            if stem in pages:
                return self.workflow_codes[workflow_name]
        return BASE_CODE

    def save_current_assignment(self) -> None:
        if self.project is None or not self.current_stem or self.assignment is None:
            return
        try:
            _write_png(self._assignment_path(self.current_stem), self.assignment)
        except OSError as exc:
            self.statusBar().showMessage(f'無法保存 {self.current_stem} 的選擇：{exc}')
            return
        self.initialized_pages.add(self.current_stem)
        self._save_state()
        self._update_page_item(self.current_stem)

    def _populate_pages(self) -> None:
        self.page_list.blockSignals(True)
        self.page_list.clear()
        assert self.project is not None
        for stem in self.project.page_stems:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, stem)
            self.page_list.addItem(item)
            self._update_page_item(stem)
        self.page_list.blockSignals(False)
        self._update_progress()

    def _assignment_progress(self, stem: str) -> tuple[int, int]:
        assert self.project is not None
        base = _read_image(self.project.pages[stem])
        if base is None:
            return 0, 0
        shape = base.shape[:2]
        mask = self._page_effective_mask(stem, shape) > 0
        total = int(np.count_nonzero(mask))
        if total == 0:
            return 0, 0
        if stem == self.current_stem and self.assignment is not None:
            assignment = self.assignment
        else:
            assignment = self._load_assignment(stem, shape)
        assigned = int(np.count_nonzero((assignment > 0) & mask))
        return assigned, total

    def _update_page_item(self, stem: str) -> None:
        if self.project is None:
            return
        try:
            row = self.project.page_stems.index(stem)
        except ValueError:
            return
        item = self.page_list.item(row)
        assigned, total = self._assignment_progress(stem)
        source_name = self.project.pages[stem].name
        if total == 0:
            item.setText(f'○  {source_name}  無 Mask')
            item.setForeground(QColor('#8d99a3'))
        elif assigned >= total:
            item.setText(f'●  {source_name}  完成')
            item.setForeground(QColor('#76d7b4'))
        elif assigned:
            percent = round(assigned * 100 / total)
            item.setText(f'◐  {source_name}  {percent}%')
            item.setForeground(QColor('#f0cf78'))
        else:
            item.setText(f'○  {source_name}  未選擇')
            item.setForeground(QColor('#c1c8ce'))

    def _update_progress(self) -> None:
        if self.project is None:
            self.progress_label.setText('')
            return
        completed = 0
        for stem in self.project.page_stems:
            assigned, total = self._assignment_progress(stem)
            if total == 0 or assigned >= total:
                completed += 1
        self.progress_label.setText(f'完成 {completed}/{len(self.project.pages)}')

    def _page_row_changed(self, row: int) -> None:
        if self.project is None or row < 0:
            return
        self.save_current_assignment()
        item = self.page_list.item(row)
        stem = str(item.data(Qt.ItemDataRole.UserRole))
        base_raw = _read_image(self.project.pages[stem])
        if base_raw is None:
            QMessageBox.warning(self, '讀取失敗', self.project.pages[stem].name)
            return
        self.current_stem = stem
        self.current_base = _as_bgr(base_raw)
        self.current_mask = self._page_effective_mask(stem, self.current_base.shape[:2])
        self.assignment = self._load_assignment(stem, self.current_base.shape[:2])
        valid_codes = {0, BASE_CODE, *self.workflow_codes.values()}
        self.assignment[~np.isin(self.assignment, list(valid_codes))] = 0
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._reload_all_panels(keep_view=False)
        self.refresh_overlays()
        self._update_actions()
        self.update_result_preview()
        self.statusBar().showMessage(
            f'{self.project.pages[stem].name}：Mask 範圍 {int(np.count_nonzero(self.current_mask))} 像素。'
        )

    def _panel_workflow_changed(self, panel: WorkflowPanel) -> None:
        panel.view.workflow_name = panel.current_workflow()
        self._reload_panel(panel, keep_view=True)

    def _reload_all_panels(self, keep_view: bool) -> None:
        for panel in self.panels:
            if not panel.isHidden():
                self._reload_panel(panel, keep_view)
        self._set_compare_fraction(self.compare_fraction)

    def _reload_panel(self, panel: WorkflowPanel, keep_view: bool) -> None:
        if self.project is None or self.current_base is None or not self.current_stem:
            panel.view.set_images(None, None)
            return
        workflow = panel.current_workflow()
        panel.view.workflow_name = workflow
        source_path = self.project.workflows.get(workflow, {}).get(self.current_stem)
        result_raw = _read_image(source_path) if source_path is not None else None
        valid = result_raw is not None and _as_bgr(result_raw).shape == self.current_base.shape
        result = _as_bgr(result_raw) if valid and result_raw is not None else self.current_base
        panel.availability_label.setText('' if valid else '此頁缺少結果')
        panel.apply_whole_button.setEnabled(valid)
        panel.view.setEnabled(valid)
        panel.compare_slider.setEnabled(valid)
        panel.view.set_images(_qimage_from_bgr(self.current_base), _qimage_from_bgr(result), keep_view=keep_view)
        panel.set_compare_fraction(self.compare_fraction)
        QTimer.singleShot(0, panel.update_compare_slider_geometry)

    def _view_compare_changed(self, fraction: float) -> None:
        self._set_compare_fraction(fraction)

    def _set_compare_fraction(self, fraction: float) -> None:
        self.compare_fraction = max(0.0, min(1.0, fraction))
        for panel in self.panels:
            panel.set_compare_fraction(self.compare_fraction)

    def _sync_views_from(self, source: QGraphicsView) -> None:
        if self._syncing_views:
            return
        self._syncing_views = True
        try:
            if self.result_panel.view is not source:
                self.result_panel.view.copy_view_from(source)
            for panel in self.panels:
                if not panel.isHidden() and panel.view is not source:
                    panel.view.copy_view_from(source)
        finally:
            self._syncing_views = False

    def fit_views(self) -> None:
        if not self.result_panel.view.pixmap_item.pixmap().isNull():
            self.result_panel.view.fit_image()
            return
        for panel in self.panels:
            if not panel.isHidden():
                panel.view.fit_image()
                return

    def _update_interaction_mode(self) -> None:
        mode = 'brush' if self.brush_button.isChecked() else 'rect'
        for panel in self.panels:
            panel.view.interaction_mode = mode
            if mode != 'brush':
                panel.view.brush_cursor_item.hide()
        self.hint_label.setText(
            '左鍵筆刷採用工作流；右鍵矩形保留原圖；[ / ] 調整大小；中鍵平移。'
            if mode == 'brush'
            else '左鍵矩形採用工作流；右鍵矩形保留原圖；中鍵平移；M 顯示選區。'
        )

    def _change_brush_size(self, delta: int) -> None:
        self.brush_size.setValue(self.brush_size.value() + delta)
        self.statusBar().showMessage(f'筆刷大小：{self.brush_size.value()} px')

    def _brush_size_changed(self, value: int) -> None:
        for panel in self.panels:
            panel.view.brush_radius = value

    def _mask_expand_changed(self, value: int) -> None:
        self.mask_expand_px = max(0, min(80, int(value)))
        self.settings.setValue('mask_expand_px', self.mask_expand_px)
        if self.project is None:
            return
        if self.current_stem and self.current_base is not None and self.assignment is not None:
            self.current_mask = self._page_effective_mask(
                self.current_stem,
                self.current_base.shape[:2],
            )
            effective = self.current_mask > 0
            self.assignment[~effective] = 0
            self.assignment[effective & (self.assignment == 0)] = self._default_workflow_code(
                self.current_stem
            )
            self.refresh_overlays()
            self.save_current_assignment()
            self.update_result_preview()
        for stem in self.project.page_stems:
            self._update_page_item(stem)
        self._update_progress()
        self._save_state()
        self.statusBar().showMessage(f'Mask 擴大：{self.mask_expand_px} px')

    def _feather_changed(self, value: int) -> None:
        self.settings.setValue('feather_px', value)
        self._save_state()
        self.update_result_preview()

    def _effective_code(self, workflow: str, use_base: bool = False) -> int | None:
        return BASE_CODE if use_base else self.workflow_codes.get(workflow)

    def _push_undo(self) -> None:
        if self.assignment is None:
            return
        self.undo_stack.append(self.assignment.copy())
        self.undo_stack = self.undo_stack[-MAX_UNDO_STEPS:]
        self.redo_stack.clear()
        self._update_actions()

    def _assignment_started(self, workflow: str) -> None:
        self._stroke_changed = False
        self._push_undo()

    def _assign_point(self, workflow: str, x: int, y: int) -> None:
        if self.assignment is None or self.current_mask is None:
            return
        height, width = self.assignment.shape
        if not (0 <= x < width and 0 <= y < height) or self.current_mask[y, x] == 0:
            return
        code = self._effective_code(workflow)
        if code is None:
            return
        brush_mask = np.zeros_like(self.current_mask)
        cv2.circle(brush_mask, (x, y), self.brush_size.value(), 255, -1, lineType=cv2.LINE_AA)
        selected = (brush_mask > 0) & (self.current_mask > 0)
        if np.any(selected):
            self.assignment[selected] = code
            self._stroke_changed = True
            self.refresh_overlays()

    def _assign_rect(
        self,
        workflow: str,
        x: int,
        y: int,
        width: int,
        height: int,
        use_base: bool = False,
    ) -> None:
        if self.assignment is None or self.current_mask is None:
            return
        code = self._effective_code(workflow, use_base=use_base)
        if code is None:
            return
        image_height, image_width = self.assignment.shape
        left = max(0, min(image_width, x))
        top = max(0, min(image_height, y))
        right = max(left, min(image_width, x + width))
        bottom = max(top, min(image_height, y + height))
        selected = self.current_mask[top:bottom, left:right] > 0
        if not np.any(selected):
            self.statusBar().showMessage('矩形沒有覆蓋任何 Mask。')
            return
        region = self.assignment[top:bottom, left:right]
        region[selected] = code
        self._stroke_changed = True
        if code == BASE_CODE:
            self.statusBar().showMessage('右鍵矩形內的 Mask 已取消工作流選擇，保留原圖。')
        else:
            self.statusBar().showMessage(f'矩形內的 Mask 已指定為 {workflow}。')

    def _assignment_finished(self, workflow: str) -> None:
        if self._stroke_changed:
            self._finish_assignment_change()
        elif self.undo_stack:
            self.undo_stack.pop()
            self._update_actions()

    def _finish_assignment_change(self) -> None:
        self.refresh_overlays()
        self.save_current_assignment()
        self._update_progress()
        self._update_actions()
        self.update_result_preview()

    def update_result_preview(self) -> None:
        if self.project is None or self.assignment is None or not self.current_stem:
            self.result_panel.set_result('', None)
            return
        try:
            output, missing = compose_result(
                self.project,
                self.current_stem,
                self.assignment,
                self.workflow_codes,
                feather_px=self.feather_size.value(),
            )
            warning = f'缺少：{"、".join(missing)}' if missing else ''
            self.result_panel.set_result(
                self.project.pages[self.current_stem].name,
                output,
                warning,
            )
        except (OSError, ValueError) as exc:
            self.result_panel.set_result(
                self.project.pages[self.current_stem].name,
                None,
                str(exc),
            )

    def apply_whole(self, workflow: str) -> None:
        if self.assignment is None or self.current_mask is None:
            return
        code = BASE_CODE if not workflow else self.workflow_codes.get(workflow)
        if code is None:
            return
        self._push_undo()
        self.assignment[self.current_mask > 0] = code
        self._finish_assignment_change()
        source = '底圖' if code == BASE_CODE else workflow
        self.statusBar().showMessage(f'整個 Mask 已指定為 {source}。')

    def reset_page_assignment(self) -> None:
        if self.assignment is None or self.current_mask is None or not self.current_stem:
            return
        default_code = self._default_workflow_code(self.current_stem)
        reset = np.zeros_like(self.assignment)
        reset[self.current_mask > 0] = default_code
        if np.array_equal(reset, self.assignment):
            return
        self._push_undo()
        self.assignment = reset
        self._finish_assignment_change()
        default_name = next(
            (name for name, code in self.workflow_codes.items() if code == default_code),
            '底圖',
        )
        self.statusBar().showMessage(f'已將此頁重設為 {default_name}。')

    def undo(self) -> None:
        if self.assignment is None or not self.undo_stack:
            return
        self.redo_stack.append(self.assignment.copy())
        self.assignment = self.undo_stack.pop()
        self._finish_assignment_change()

    def redo(self) -> None:
        if self.assignment is None or not self.redo_stack:
            return
        self.undo_stack.append(self.assignment.copy())
        self.assignment = self.redo_stack.pop()
        self._finish_assignment_change()

    def _update_actions(self) -> None:
        self.undo_action.setEnabled(bool(self.undo_stack))
        self.redo_action.setEnabled(bool(self.redo_stack))
        enabled = self.assignment is not None
        self.clear_button.setEnabled(enabled)
        self.base_whole_button.setEnabled(enabled)

    def _assignment_overlay(self) -> QImage | None:
        if (
            not self.show_regions_checkbox.isChecked()
            or self.assignment is None
            or self.current_mask is None
        ):
            return None
        height, width = self.assignment.shape
        overlay = np.zeros((height, width, 4), dtype=np.uint8)
        unassigned = (self.current_mask > 0) & (self.assignment == 0)
        base_selected = self.assignment == BASE_CODE
        overlay[base_selected] = (135, 145, 155, 72)
        for index, (name, code) in enumerate(self.workflow_codes.items()):
            color = WORKFLOW_COLORS[index % len(WORKFLOW_COLORS)]
            selected = self.assignment == code
            overlay[selected] = (*color, 82)
        contours, _ = cv2.findContours(
            (self.current_mask > 0).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (255, 229, 130, 210), 1, cv2.LINE_AA)
        if np.any(unassigned):
            overlay[unassigned, 3] = np.maximum(overlay[unassigned, 3], 22)
        return _qimage_from_bgra(overlay)

    def refresh_overlays(self) -> None:
        overlay = self._assignment_overlay()
        for panel in self.panels:
            panel.view.set_overlay(overlay)

    def export_results(self) -> None:
        if self.project is None:
            QMessageBox.information(self, '尚未載入', '請先選擇 export_pair 文件夾。')
            return
        self.save_current_assignment()
        incomplete: list[str] = []
        missing_masks: list[str] = []
        for stem in self.project.page_stems:
            base = _read_image(self.project.pages[stem])
            if base is None:
                continue
            shape = base.shape[:2]
            mask = self._page_effective_mask(stem, shape) > 0
            if not np.any(mask):
                missing_masks.append(self.project.pages[stem].name)
                continue
            assignment = self._load_assignment(stem, shape)
            if np.any((assignment == 0) & mask):
                incomplete.append(self.project.pages[stem].name)
        if incomplete:
            preview = '、'.join(incomplete[:8])
            if len(incomplete) > 8:
                preview += f' 等 {len(incomplete)} 張'
            answer = QMessageBox.question(
                self,
                '仍有未選擇區域',
                f'{preview} 尚有 Mask 區域未指定來源。\n未指定部分將保留底圖，是否繼續輸出？',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        result_dir = self.project.root / RESULT_DIR_NAME
        errors: list[str] = [f'{name}：沒有可用 Mask，已輸出底圖' for name in missing_masks]
        manifest_pages: dict[str, dict[str, object]] = {}
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result_dir.mkdir(parents=True, exist_ok=True)
            for stem in self.project.page_stems:
                base = _read_image(self.project.pages[stem])
                if base is None:
                    errors.append(f'{stem}：底圖讀取失敗')
                    continue
                assignment = self._load_assignment(stem, base.shape[:2])
                try:
                    output, missing = compose_result(
                        self.project,
                        stem,
                        assignment,
                        self.workflow_codes,
                        feather_px=self.feather_size.value(),
                    )
                    _write_png(result_dir / f'{stem}.png', output)
                    mask_for_counts = self._page_effective_mask(stem, assignment.shape) > 0
                    counts: dict[str, int] = {
                        '未指定': int(np.count_nonzero((assignment == 0) & mask_for_counts)),
                        '底圖': int(np.count_nonzero((assignment == BASE_CODE) & mask_for_counts)),
                    }
                    for workflow_name, code in self.workflow_codes.items():
                        count = int(np.count_nonzero((assignment == code) & mask_for_counts))
                        if count:
                            counts[workflow_name] = count
                    manifest_pages[f'{stem}.png'] = {
                        'source': self.project.pages[stem].name,
                        'pixel_counts': counts,
                        'warnings': missing,
                    }
                    if missing:
                        errors.append(f'{stem}：缺少 {"、".join(missing)}')
                except (OSError, ValueError) as exc:
                    errors.append(str(exc))
            manifest = {
                'version': 1,
                'export_pair': str(self.project.root),
                'mask_expand_px': self.mask_expand_px,
                'feather_px': self.feather_size.value(),
                'workflow_codes': self.workflow_codes,
                'pages': manifest_pages,
            }
            manifest_tmp = result_dir / f'{STATE_FILE_NAME}.tmp'
            manifest_tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
            os.replace(manifest_tmp, result_dir / STATE_FILE_NAME)
        finally:
            QApplication.restoreOverrideCursor()
        if errors:
            QMessageBox.warning(
                self,
                '輸出完成，但有警告',
                f'結果已輸出到：\n{result_dir}\n\n' + '\n'.join(errors[:12]),
            )
        else:
            QMessageBox.information(self, '輸出完成', f'結果已輸出到：\n{result_dir}')
        self.statusBar().showMessage(f'輸出完成：{result_dir}')

    def _load_recent_folders(self) -> list[str]:
        value = self.settings.value('recent_export_pair_folders', [])
        values = [value] if isinstance(value, str) else list(value or [])
        result: list[str] = []
        for item in values:
            path = str(Path(str(item)).expanduser())
            if path not in result and Path(path).is_dir():
                result.append(path)
        return result[:MAX_RECENT_FOLDERS]

    def _remember_folder(self, folder: Path) -> None:
        value = str(folder)
        self.recent_folders = [value] + [item for item in self.recent_folders if item != value]
        self.recent_folders = self.recent_folders[:MAX_RECENT_FOLDERS]
        self.settings.setValue('recent_export_pair_folders', self.recent_folders)

    def closeEvent(self, event) -> None:
        self.save_current_assignment()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName('工作流比較與合成')
    app.setOrganizationName('ComicTextDetector')
    window = WorkflowCompareWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
