"""QGraphicsItem subclasses for overlay elements."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetricsF,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import QGraphicsItem, QStyleOptionGraphicsItem, QWidget

from ..model import CalibrationLine, Polygon

# -- Color constants (migrated from overlay.py) --

COLOR_CALIBRATION = QColor("#00FF00")
COLOR_POLYGON_OPEN = QColor("#FFAA00")
COLOR_POLYGON_CLOSED = QColor("#00AAFF")
COLOR_POLYGON_FILL = QColor(0, 170, 255, 38)
COLOR_POLYGON_SELECTED = QColor("#FFFF00")
COLOR_POLYGON_SELECTED_FILL = QColor(255, 255, 0, 25)
COLOR_VERTEX = QColor("#FF0000")
COLOR_VERTEX_SNAPPED = QColor("#00FF00")
COLOR_ACTIVE_VERTEX = QColor("#FFFF00")
COLOR_HOVER_VERTEX = QColor("#FF8800")
COLOR_ROI = QColor("#FF00FF")
COLOR_GRID = QColor("#FF00FF")
COLOR_TEXT_BG = QColor(0, 0, 0, 178)
COLOR_RUBBERBAND = QColor(255, 255, 255, 178)
COLOR_SNAP_MARKER = QColor("#00FF00")

_LABEL_FONT = QFont("monospace", 9)
_AREA_FONT = QFont("sans-serif", 10, QFont.Weight.Bold)


class CalibrationLineItem(QGraphicsItem):
    """Draws a calibration reference line with endpoints and optional label."""

    def __init__(
        self,
        line: CalibrationLine,
        show_label: bool = True,
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self._line = line
        self._show_label = show_label
        self._label = f"{line.known_length:.3g} {line.unit.value}"

    def boundingRect(self) -> QRectF:
        margin = 20.0
        xs = [self._line.p0.x, self._line.p1.x]
        ys = [self._line.p0.y, self._line.p1.y]
        return QRectF(
            min(xs) - margin,
            min(ys) - margin,
            max(xs) - min(xs) + 2 * margin,
            max(ys) - min(ys) + 2 * margin,
        )

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        p0 = QPointF(self._line.p0.x, self._line.p0.y)
        p1 = QPointF(self._line.p1.x, self._line.p1.y)

        # Line segment
        pen = QPen(COLOR_CALIBRATION, 2.0)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawLine(p0, p1)

        # Endpoints
        painter.setBrush(COLOR_CALIBRATION)
        r = 3.0
        painter.drawEllipse(p0, r, r)
        painter.drawEllipse(p1, r, r)

        # Label at midpoint
        if self._show_label:
            mid = (p0 + p1) / 2.0
            painter.setFont(_LABEL_FONT)
            fm = QFontMetricsF(_LABEL_FONT)
            tw = fm.horizontalAdvance(self._label)
            th = fm.height()
            bg = QRectF(mid.x() - tw / 2 - 3, mid.y() - th - 3, tw + 6, th + 4)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(COLOR_TEXT_BG)
            painter.drawRoundedRect(bg, 3, 3)
            painter.setPen(COLOR_CALIBRATION)
            painter.drawText(bg, Qt.AlignmentFlag.AlignCenter, self._label)


class PolygonOverlayItem(QGraphicsItem):
    """Draws a polygon with edges, fill, vertices, and optional area label."""

    def __init__(
        self,
        polygon: Polygon,
        active_vertex_idx: Optional[int] = None,
        hover_vertex_idx: Optional[int] = None,
        area_text: Optional[str] = None,
        selected: bool = False,
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self._polygon = polygon
        self._active_vi = active_vertex_idx
        self._hover_vi = hover_vertex_idx
        self._area_text = area_text
        self._selected = selected

    def boundingRect(self) -> QRectF:
        if self._polygon.n_vertices == 0:
            return QRectF()
        margin = 15.0
        xs = [v.x for v in self._polygon.vertices]
        ys = [v.y for v in self._polygon.vertices]
        return QRectF(
            min(xs) - margin,
            min(ys) - margin,
            max(xs) - min(xs) + 2 * margin,
            max(ys) - min(ys) + 2 * margin,
        )

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        poly = self._polygon
        if poly.n_vertices == 0:
            return

        points = [QPointF(v.x, v.y) for v in poly.vertices]

        # Edges
        if poly.n_vertices >= 2:
            if self._selected:
                color = COLOR_POLYGON_SELECTED
            elif poly.closed:
                color = COLOR_POLYGON_CLOSED
            else:
                color = COLOR_POLYGON_OPEN
            lw = 2.5 if self._selected else 1.5
            pen = QPen(color, lw)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            path = QPainterPath()
            path.moveTo(points[0])
            for pt in points[1:]:
                path.lineTo(pt)
            if poly.closed:
                path.closeSubpath()
            painter.drawPath(path)

            # Fill closed polygon
            if poly.closed and poly.n_vertices >= 3:
                fill = (
                    COLOR_POLYGON_SELECTED_FILL
                    if self._selected
                    else COLOR_POLYGON_FILL
                )
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(fill))
                fill_path = QPainterPath()
                fill_path.moveTo(points[0])
                for pt in points[1:]:
                    fill_path.lineTo(pt)
                fill_path.closeSubpath()
                painter.fillPath(fill_path, QBrush(fill))

        # Vertices
        for i, v in enumerate(poly.vertices):
            if i == self._active_vi:
                c = COLOR_ACTIVE_VERTEX
                r = 4.0
            elif i == self._hover_vi:
                c = COLOR_HOVER_VERTEX
                r = 3.5
            elif v.snapped:
                c = COLOR_VERTEX_SNAPPED
                r = 2.5
            else:
                c = COLOR_VERTEX
                r = 2.5
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(c)
            painter.drawEllipse(QPointF(v.x, v.y), r, r)

        # Area label at centroid
        if self._area_text and poly.n_vertices >= 3:
            cx = sum(v.x for v in poly.vertices) / poly.n_vertices
            cy = sum(v.y for v in poly.vertices) / poly.n_vertices
            painter.setFont(_AREA_FONT)
            fm = QFontMetricsF(_AREA_FONT)
            tw = fm.horizontalAdvance(self._area_text)
            th = fm.height()
            bg = QRectF(cx - tw / 2 - 4, cy - th / 2 - 2, tw + 8, th + 4)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(COLOR_TEXT_BG)
            painter.drawRoundedRect(bg, 4, 4)
            painter.setPen(QColor("white"))
            painter.drawText(bg, Qt.AlignmentFlag.AlignCenter, self._area_text)


class PointMarkerItem(QGraphicsItem):
    """Draws transient click points (cross markers) during calibration."""

    def __init__(
        self,
        points: List[Tuple[float, float]],
        color: QColor = COLOR_VERTEX,
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self._points = points
        self._color = color

    def boundingRect(self) -> QRectF:
        if not self._points:
            return QRectF()
        margin = 10.0
        xs = [p[0] for p in self._points]
        ys = [p[1] for p in self._points]
        return QRectF(
            min(xs) - margin,
            min(ys) - margin,
            max(xs) - min(xs) + 2 * margin,
            max(ys) - min(ys) + 2 * margin,
        )

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        pen = QPen(self._color, 2.0)
        pen.setCosmetic(True)
        painter.setPen(pen)
        arm = 6.0
        for x, y in self._points:
            painter.drawLine(QPointF(x - arm, y), QPointF(x + arm, y))
            painter.drawLine(QPointF(x, y - arm), QPointF(x, y + arm))


class RubberbandLineItem(QGraphicsItem):
    """Draws a white dashed preview line."""

    def __init__(
        self,
        from_pt: Tuple[float, float],
        to_pt: Tuple[float, float],
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self._from = QPointF(from_pt[0], from_pt[1])
        self._to = QPointF(to_pt[0], to_pt[1])

    def boundingRect(self) -> QRectF:
        margin = 2.0
        x0 = min(self._from.x(), self._to.x()) - margin
        y0 = min(self._from.y(), self._to.y()) - margin
        x1 = max(self._from.x(), self._to.x()) + margin
        y1 = max(self._from.y(), self._to.y()) + margin
        return QRectF(x0, y0, x1 - x0, y1 - y0)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        pen = QPen(COLOR_RUBBERBAND, 1.0, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawLine(self._from, self._to)


class SnapMarkerItem(QGraphicsItem):
    """Draws a green diamond snap indicator."""

    def __init__(
        self,
        pos: Tuple[float, float],
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self._pos = QPointF(pos[0], pos[1])

    def boundingRect(self) -> QRectF:
        r = 6.0
        return QRectF(self._pos.x() - r, self._pos.y() - r, 2 * r, 2 * r)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        pen = QPen(COLOR_SNAP_MARKER, 1.5)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        r = 5.0
        path = QPainterPath()
        c = self._pos
        path.moveTo(c.x(), c.y() - r)
        path.lineTo(c.x() + r, c.y())
        path.lineTo(c.x(), c.y() + r)
        path.lineTo(c.x() - r, c.y())
        path.closeSubpath()
        painter.drawPath(path)


class RoiRectItem(QGraphicsItem):
    """Draws a magenta dashed ROI rectangle."""

    def __init__(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self._start = start
        self._end = end

    def boundingRect(self) -> QRectF:
        margin = 2.0
        x0 = min(self._start[0], self._end[0]) - margin
        y0 = min(self._start[1], self._end[1]) - margin
        x1 = max(self._start[0], self._end[0]) + margin
        y1 = max(self._start[1], self._end[1]) + margin
        return QRectF(x0, y0, x1 - x0, y1 - y0)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        pen = QPen(COLOR_ROI, 1.5, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        x0, y0 = self._start
        x1, y1 = self._end
        painter.drawRect(QRectF(
            min(x0, x1), min(y0, y1),
            abs(x1 - x0), abs(y1 - y0),
        ))


class GridOverlayItem(QGraphicsItem):
    """Draws detected grid lines over the image."""

    def __init__(
        self,
        origin: Tuple[float, float],
        pitch_x: float,
        pitch_y: float,
        angle_deg: float,
        extent: Tuple[float, float],
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self._origin = origin
        self._pitch_x = pitch_x
        self._pitch_y = pitch_y
        self._angle_deg = angle_deg
        self._extent = extent

    def boundingRect(self) -> QRectF:
        ox, oy = self._origin
        w, h = self._extent
        margin = max(w, h) * 0.1
        return QRectF(
            ox - margin, oy - margin,
            w + 2 * margin, h + 2 * margin,
        )

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        ox, oy = self._origin
        w, h = self._extent
        cos_a = np.cos(np.radians(self._angle_deg))
        sin_a = np.sin(np.radians(self._angle_deg))

        pen = QPen(COLOR_GRID, 0.8)
        pen.setCosmetic(True)
        grid_color = QColor(COLOR_GRID)
        grid_color.setAlpha(102)  # ~0.4 * 255
        pen.setColor(grid_color)
        painter.setPen(pen)

        n_x = int(w / self._pitch_x) + 2 if self._pitch_x > 0 else 0
        n_y = int(h / self._pitch_y) + 2 if self._pitch_y > 0 else 0

        for i in range(-1, n_x):
            offset = i * self._pitch_x
            x0 = ox + offset * cos_a
            y0 = oy + offset * sin_a
            x1 = x0 - h * sin_a
            y1 = y0 + h * cos_a
            painter.drawLine(QPointF(x0, y0), QPointF(x1, y1))

        for j in range(-1, n_y):
            offset = j * self._pitch_y
            x0 = ox - offset * sin_a
            y0 = oy + offset * cos_a
            x1 = x0 + w * cos_a
            y1 = y0 + w * sin_a
            painter.drawLine(QPointF(x0, y0), QPointF(x1, y1))
