"""Page geometry shared by the editing canvas and its synchronized preview."""
from bisect import bisect_left, bisect_right

from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QImageReader


class PageStrip:
    GAP = 12

    def __init__(self, paths):
        sizes = []
        for path in paths:
            reader = QImageReader(str(path))
            reader.setDecideFormatFromContent(True)
            size = reader.size()
            sizes.append((size.width(), size.height()) if not size.isEmpty() else None)
        # Keep unreadable pages in list order. Selecting one still follows the
        # editor's usual load-error path, and subsequent pages remain reachable.
        fallback = next((size for size in sizes if size is not None), (720, 1024))
        width = max((size[0] for size in sizes if size is not None), default=fallback[0])
        self._rects = []
        self._tops = []
        self._bottoms = []
        top = 0
        for size in sizes:
            w, h = size or fallback
            self._rects.append(QRectF((width - w) // 2, top, w, h))
            self._tops.append(top)
            self._bottoms.append(top + h)
            top += h + self.GAP
        self._bounds = QRectF(0, 0, width, max(1, top - self.GAP))

    def rect(self, row: int) -> QRectF:
        return self._rects[row]

    def scene_bounds(self, active_row: int) -> QRectF:
        return self._bounds.translated(-self.rect(active_row).topLeft())

    def row_nearest_y(self, document_y: float) -> int:
        # In the gap, hand over halfway between one page and the next.
        return max(0, bisect_right(self._tops, document_y + self.GAP / 2) - 1)

    def visible_rows(self, active_row: int, scene_rect: QRectF) -> list[int]:
        if active_row < 0 or scene_rect.isEmpty():
            return []
        visible = scene_rect.translated(self.rect(active_row).topLeft())
        start = bisect_right(self._bottoms, visible.top())
        stop = bisect_left(self._tops, visible.bottom())
        return [row for row in range(start, stop) if self.rect(row).intersects(visible)]

    def row_at(self, active_row: int, scene_pos: QPointF) -> int | None:
        point = scene_pos + self.rect(active_row).topLeft()
        row = bisect_right(self._tops, point.y()) - 1
        if row < 0:
            return None
        rect = self.rect(row)
        if rect.left() <= point.x() < rect.right() and point.y() < rect.bottom():
            return row
        return None
