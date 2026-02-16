"""QGraphicsView-based image canvas with zoom/pan and overlay management."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from PySide6.QtCore import QPoint, QPointF, Qt, Signal
from PySide6.QtGui import QImage, QMouseEvent, QPainter, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
)

if TYPE_CHECKING:
    from .handlers import CanvasEventHandler

# Scroll zoom factor per step
_ZOOM_FACTOR = 1.2


class ImageCanvas(QGraphicsView):
    """Displays an image with overlays, zoom, and pan support."""

    cursor_moved = Signal(float, float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._overlay_items: list[QGraphicsItem] = []
        self._handler: Optional[CanvasEventHandler] = None

        # Middle-button panning state
        self._panning = False
        self._pan_start: Optional[QPoint] = None

        # View settings
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMouseTracking(True)
        self.setStyleSheet("background-color: #1a1a1a; border: none;")

    # -- Public API --

    def set_image_rgb(self, rgb_array: np.ndarray) -> None:
        """Display an RGB numpy array as the background image."""
        h, w, ch = rgb_array.shape
        bytes_per_line = ch * w
        qimg = QImage(
            rgb_array.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        )
        pixmap = QPixmap.fromImage(qimg)

        if self._pixmap_item is not None:
            self._pixmap_item.setPixmap(pixmap)
        else:
            self._pixmap_item = self._scene.addPixmap(pixmap)
            self._pixmap_item.setZValue(-1000)

        self._scene.setSceneRect(0, 0, w, h)

    def set_event_handler(self, handler: CanvasEventHandler) -> None:
        self._handler = handler

    def clear_overlays(self) -> None:
        """Remove all overlay items from the scene."""
        for item in self._overlay_items:
            self._scene.removeItem(item)
        self._overlay_items.clear()

    def add_overlay(self, item: QGraphicsItem) -> None:
        """Add an overlay item to the scene."""
        self._scene.addItem(item)
        self._overlay_items.append(item)

    def fit_image(self) -> None:
        """Fit the entire image in the viewport."""
        if self._pixmap_item is not None:
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def image_size(self) -> tuple[int, int]:
        """Return (width, height) of the current image, or (0, 0)."""
        if self._pixmap_item is None:
            return (0, 0)
        pm = self._pixmap_item.pixmap()
        return (pm.width(), pm.height())

    # -- Event handling --

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Zoom centered on cursor position."""
        angle = event.angleDelta().y()
        if angle == 0:
            return

        factor = 1.0 / _ZOOM_FACTOR if angle > 0 else _ZOOM_FACTOR
        # Zoom around cursor position
        cursor_scene = self.mapToScene(event.position().toPoint())
        self.scale(1.0 / factor, 1.0 / factor)
        # Adjust so cursor stays under the same scene point
        new_pos = self.mapFromScene(cursor_scene)
        delta = new_pos - event.position().toPoint()
        self.horizontalScrollBar().setValue(
            self.horizontalScrollBar().value() + delta.x()
        )
        self.verticalScrollBar().setValue(
            self.verticalScrollBar().value() + delta.y()
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        # Middle button: start panning
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        # Delegate to handler
        if self._handler is not None:
            scene_pos = self.mapToScene(event.position().toPoint())
            self._handler.on_press(event.button(), scene_pos, event)
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        # End panning
        if event.button() == Qt.MouseButton.MiddleButton and self._panning:
            self._panning = False
            self._pan_start = None
            self.unsetCursor()
            event.accept()
            return

        if self._handler is not None:
            scene_pos = self.mapToScene(event.position().toPoint())
            self._handler.on_release(event.button(), scene_pos)
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        # Panning
        if self._panning and self._pan_start is not None:
            delta = event.position().toPoint() - self._pan_start
            self._pan_start = event.position().toPoint()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            event.accept()
            return

        scene_pos = self.mapToScene(event.position().toPoint())
        self.cursor_moved.emit(scene_pos.x(), scene_pos.y())

        if self._handler is not None:
            self._handler.on_move(scene_pos)
        event.accept()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if self._handler is not None:
            scene_pos = self.mapToScene(event.position().toPoint())
            self._handler.on_double_click(event.button(), scene_pos)
        event.accept()

    def contextMenuEvent(self, event) -> None:
        # Handled via right-click in handler; suppress default
        event.accept()
