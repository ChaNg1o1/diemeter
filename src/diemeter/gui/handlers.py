"""Qt event dispatch for the measurement GUI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QMenu

from .i18n import tr
from .modes import Mode

if TYPE_CHECKING:
    from .app import MeasurementApp


class CanvasEventHandler:
    """Routes Qt mouse events from the canvas to the appropriate mode handler."""

    def __init__(self, app: MeasurementApp) -> None:
        self.app = app

    # -- Main dispatchers --

    def on_press(
        self, button: Qt.MouseButton, scene_pos: QPointF, event: QMouseEvent
    ) -> None:
        x, y = scene_pos.x(), scene_pos.y()

        # Right button: show context menu
        if button == Qt.MouseButton.RightButton:
            self._show_context_menu(x, y, event.globalPosition().toPoint())
            return

        # Left button only from here
        if button != Qt.MouseButton.LeftButton:
            return

        # Snap to edge if enabled
        x, y, snapped, confidence = self.app.maybe_snap(x, y)

        mode = self.app.state.mode

        if mode == Mode.VIEW:
            self._handle_view_click(x, y)
        elif mode == Mode.CALIBRATE_LINE:
            self._handle_calibrate_line_click(x, y)
        elif mode == Mode.CALIBRATE_RECT:
            self._handle_calibrate_rect_click(x, y)
        elif mode == Mode.POLYGON_DRAW:
            self._handle_polygon_draw_click(x, y, snapped, confidence)
        elif mode == Mode.POLYGON_EDIT:
            self._handle_polygon_edit_click(x, y)
        elif mode == Mode.GRID_DETECT:
            self._handle_grid_detect_click(x, y)
        elif mode == Mode.CALIBRATE_PAD_PITCH:
            self._handle_pad_pitch_click(x, y)

    def on_release(self, button: Qt.MouseButton, scene_pos: QPointF) -> None:
        state = self.app.state
        mode = state.mode

        if mode == Mode.POLYGON_EDIT:
            if button == Qt.MouseButton.LeftButton:
                state.dragging_vertex_idx = None

        elif mode == Mode.GRID_DETECT:
            if (
                button == Qt.MouseButton.LeftButton
                and state.roi_start is not None
            ):
                state.roi_end = (scene_pos.x(), scene_pos.y())
                self.app.run_grid_detect()

        elif mode == Mode.CALIBRATE_PAD_PITCH:
            if (
                button == Qt.MouseButton.LeftButton
                and state.roi_start is not None
            ):
                state.roi_end = (scene_pos.x(), scene_pos.y())
                self.app.run_pad_pitch_calibrate()

    def on_move(self, scene_pos: QPointF) -> None:
        x, y = scene_pos.x(), scene_pos.y()
        state = self.app.state

        # Update cursor position for rubber-band / info
        state.cursor_pos = (x, y)

        mode = state.mode

        # Drag vertex in edit mode
        if mode == Mode.POLYGON_EDIT:
            if state.dragging_vertex_idx is not None:
                self.app.drag_vertex(x, y)
            else:
                old_hover = state.hover_vertex_idx
                vi = self.app.find_nearest_vertex(x, y)
                state.hover_vertex_idx = vi
                if vi != old_hover:
                    if vi is not None:
                        self.app.canvas.setCursor(Qt.CursorShape.PointingHandCursor)
                    else:
                        self.app.canvas.unsetCursor()
                    self.app.redraw()

        # Rubber-band preview for drawing modes
        elif mode in (
            Mode.POLYGON_DRAW, Mode.CALIBRATE_LINE, Mode.CALIBRATE_RECT,
        ):
            if state.points or (
                mode == Mode.POLYGON_DRAW
                and self.app._current_polygon is not None
                and self.app._current_polygon.n_vertices > 0
            ):
                self.app.redraw()

        # ROI rectangle preview
        elif mode == Mode.GRID_DETECT and state.roi_start is not None:
            state.roi_end = (x, y)
            self.app.redraw()

        elif mode == Mode.CALIBRATE_PAD_PITCH and state.roi_start is not None:
            state.roi_end = (x, y)
            self.app.redraw()

    def on_double_click(
        self, button: Qt.MouseButton, scene_pos: QPointF
    ) -> None:
        if button != Qt.MouseButton.LeftButton:
            return
        mode = self.app.state.mode
        if mode == Mode.POLYGON_DRAW:
            poly = self.app._current_polygon
            if poly is not None and poly.n_vertices >= 3:
                self.app.close_current_polygon()

    # -- Context menu --

    def _show_context_menu(
        self, x: float, y: float, global_pos
    ) -> None:
        """Build and show a QMenu based on current mode and click position."""
        app = self.app
        mode = app.state.mode
        menu = QMenu(app.canvas)

        if mode == Mode.VIEW:
            poly_idx = app.find_nearest_polygon(x, y)
            if poly_idx is not None:
                menu.addAction(
                    tr("ctx.edit_polygon"),
                    lambda idx=poly_idx: self._ctx_edit_polygon(idx),
                )
                menu.addAction(
                    tr("ctx.delete_polygon"),
                    lambda idx=poly_idx: app.delete_polygon(idx),
                )

        elif mode == Mode.POLYGON_DRAW:
            poly = app._current_polygon
            if poly is not None and poly.n_vertices >= 3:
                menu.addAction(
                    tr("ctx.close_polygon"), app.close_current_polygon
                )
            menu.addAction(
                tr("ctx.cancel"), lambda: app.switch_mode(Mode.VIEW)
            )

        elif mode == Mode.POLYGON_EDIT:
            if app.state.hover_vertex_idx is not None:
                menu.addAction(
                    tr("ctx.delete_vertex"), app.delete_nearest_vertex
                )
            menu.addAction(
                tr("ctx.insert_vertex"), app.insert_vertex_on_edge
            )
            menu.addAction(
                tr("ctx.done_editing"), lambda: app.switch_mode(Mode.VIEW)
            )

        elif mode in (Mode.CALIBRATE_LINE, Mode.CALIBRATE_RECT):
            menu.addAction(
                tr("ctx.cancel"), lambda: app.switch_mode(Mode.VIEW)
            )

        elif mode in (Mode.GRID_DETECT, Mode.CALIBRATE_PAD_PITCH):
            menu.addAction(
                tr("ctx.cancel"), lambda: app.switch_mode(Mode.VIEW)
            )

        if not menu.isEmpty():
            menu.exec(global_pos)

    def _ctx_edit_polygon(self, idx: int) -> None:
        self.app.state.active_polygon_idx = idx
        self.app.switch_mode(Mode.POLYGON_EDIT)

    # -- Mode-specific handlers --

    def _handle_view_click(self, x: float, y: float) -> None:
        poly_idx = self.app.find_nearest_polygon(x, y)
        self.app.state.active_polygon_idx = poly_idx
        self.app.redraw()

    def _handle_calibrate_line_click(self, x: float, y: float) -> None:
        self.app.state.add_point(x, y)
        if len(self.app.state.points) == 2:
            self.app.finish_calibrate_line()
        else:
            self.app.redraw()

    def _handle_calibrate_rect_click(self, x: float, y: float) -> None:
        self.app.state.add_point(x, y)
        if len(self.app.state.points) == 4:
            self.app.finish_calibrate_rect()
        else:
            self.app.redraw()

    def _handle_polygon_draw_click(
        self,
        x: float,
        y: float,
        snapped: bool,
        confidence: float,
    ) -> None:
        self.app.add_polygon_vertex(
            x,
            y,
            snapped=snapped,
            confidence=confidence,
        )

    def _handle_polygon_edit_click(self, x: float, y: float) -> None:
        vi = self.app.find_nearest_vertex(x, y)
        if vi is not None:
            self.app.state.dragging_vertex_idx = vi
        else:
            poly_idx = self.app.find_nearest_polygon(x, y)
            if (
                poly_idx is not None
                and poly_idx != self.app.state.active_polygon_idx
            ):
                self.app.state.active_polygon_idx = poly_idx
                self.app.redraw()

    def _handle_grid_detect_click(self, x: float, y: float) -> None:
        self.app.state.roi_start = (x, y)
        self.app.state.roi_end = None

    def _handle_pad_pitch_click(self, x: float, y: float) -> None:
        self.app.state.roi_start = (x, y)
        self.app.state.roi_end = None
