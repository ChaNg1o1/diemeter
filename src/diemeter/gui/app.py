"""Main measurement application using PySide6."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QInputDialog,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QStatusBar,
)

from ..calibration import (
    DEFAULT_LINE_ENDPOINT_SIGMA_PX,
    DEFAULT_LINE_HUBER_K,
    DEFAULT_LINE_IRLS_MAX_ITERS,
    DEFAULT_LINE_IRLS_TOL,
    calibrate_from_line,
    calibrate_from_pad_pitch,
    calibrate_from_rect,
    chi_squared_consistency,
    update_calibration_lines,
    update_calibration_rect,
)
from ..edge import EdgeDetector
from ..export import export_results
from ..grid import detect_grid
from ..measurement import measure_polygon, polygon_area_pixels
from ..model import (
    Calibration,
    GridResult,
    Polygon,
    Session,
    Unit,
    Vertex,
    convert_area,
    convert_length,
    unit_from_str,
)
from ..session import load_session, save_session
from .canvas import ImageCanvas
from .handlers import CanvasEventHandler
from .i18n import get_language, set_language, tr
from .items import (
    CalibrationLineItem,
    GridOverlayItem,
    PointMarkerItem,
    PolygonOverlayItem,
    RoiRectItem,
    RubberbandLineItem,
    SnapMarkerItem,
)
from .modes import Mode, ModeState
from .info_panel import InfoPanel
from .toolbar import ToolbarManager

_LINE_CAL_CONFIDENCE_LEVEL = 0.95
_LINE_CAL_WARN_REL_UNC_PERCENT = 1.0
_LINE_CAL_MIN_RECOMMENDED_PIXELS = 180.0


class MeasurementApp(QMainWindow):
    """Interactive photogrammetric measurement application."""

    def __init__(
        self,
        image_path: str,
        session_path: Optional[str] = None,
    ) -> None:
        # Ensure QApplication exists
        self._qapp = QApplication.instance() or QApplication(sys.argv)

        super().__init__()
        self.image_path = image_path
        self.session_path = session_path

        # Load image
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        self.display_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        self._warped_image: Optional[np.ndarray] = None

        # Session state
        if session_path and Path(session_path).exists():
            self.session = load_session(session_path)
        else:
            self.session = Session(image_path=image_path)
        if self.session.display_unit == Unit.PIXEL:
            self.session.display_unit = (
                self.session.calibration.unit
                if self.session.calibration.unit != Unit.PIXEL
                else Unit.MILLIMETER
            )

        # Interactive state
        self.state = ModeState()
        self._current_polygon: Optional[Polygon] = None
        self._polygon_counter = 0

        # Edge detector (lazy init)
        self._edge_detector: Optional[EdgeDetector] = None

        # Undo stack: list of (action_type, data) tuples
        self._undo_stack: List[Tuple[str, object]] = []
        self._last_line_input = "10 mm"
        self._last_rect_input = "10 x 8 mm"

        # -- UI setup --
        self.setWindowTitle(tr("window.title"))
        self.resize(1200, 900)
        self.setStyleSheet("QMainWindow { background: #1a1a1a; }")

        # Canvas
        self.canvas = ImageCanvas(self)
        self.canvas.set_image_rgb(self.display_image)
        self.setCentralWidget(self.canvas)

        # Toolbar
        self._toolbar_mgr = ToolbarManager(self)

        # Info panel (right dock)
        self._info_panel = InfoPanel(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._info_panel)

        # Status bar
        self._status_bar = QStatusBar(self)
        self._status_bar.setStyleSheet(
            "QStatusBar { background: #2a2a2a; color: #e0e0e0;"
            "  font-family: monospace; font-size: 11px; }"
        )
        self.setStatusBar(self._status_bar)

        # Event handler
        self._handler = CanvasEventHandler(self)
        self.canvas.set_event_handler(self._handler)

        # Cursor position → status bar
        self.canvas.cursor_moved.connect(self._on_cursor_moved)

        # Keyboard shortcuts
        self._setup_shortcuts()

        # Ensure loaded/new results are aligned to the selected display unit.
        self._remeasure_all()

        # Initial draw + fit
        self.redraw()
        self.canvas.fit_image()

    def _setup_shortcuts(self) -> None:
        """Register keyboard shortcuts via QShortcut."""
        shortcuts = {
            "C": lambda: self.switch_mode(Mode.CALIBRATE_LINE),
            "R": lambda: self.switch_mode(Mode.CALIBRATE_RECT),
            "P": lambda: self.switch_mode(Mode.POLYGON_DRAW),
            "E": lambda: self.switch_mode(Mode.POLYGON_EDIT),
            "G": lambda: self.switch_mode(Mode.GRID_DETECT),
            "B": lambda: self.switch_mode(Mode.CALIBRATE_PAD_PITCH),
            "Escape": lambda: self.switch_mode(Mode.VIEW),
            "S": lambda: self.toggle_edge_snap(),
            "[": lambda: self.adjust_snap_radius(-1),
            "]": lambda: self.adjust_snap_radius(+1),
            "D": lambda: self.delete_nearest_vertex(),
            "I": lambda: self.insert_vertex_on_edge(),
            "Return": self._on_enter_key,
            "Ctrl+S": lambda: self.save_session(),
            "Ctrl+E": lambda: self.export_results(),
            "Ctrl+Z": lambda: self.undo(),
        }
        for key, callback in shortcuts.items():
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(callback)

    def _on_enter_key(self) -> None:
        """Handle Enter key based on current mode."""
        mode = self.state.mode
        if mode == Mode.POLYGON_DRAW:
            self.close_current_polygon()
        elif mode == Mode.POLYGON_EDIT:
            self.switch_mode(Mode.VIEW)

    def _on_cursor_moved(self, x: float, y: float) -> None:
        """Update status bar with cursor position info."""
        self._refresh_status_bar(cursor=(x, y))

    # -- Public API for handlers --

    def switch_mode(self, mode: Mode) -> None:
        """Switch to a new interaction mode."""
        # Save any in-progress polygon
        if (
            self.state.mode == Mode.POLYGON_DRAW
            and self._current_polygon is not None
            and self._current_polygon.n_vertices >= 3
        ):
            self.close_current_polygon()

        self.state.switch_to(mode)

        if mode == Mode.POLYGON_DRAW:
            self._polygon_counter += 1
            self._current_polygon = Polygon(
                label=f"poly_{self._polygon_counter}"
            )

        if mode == Mode.POLYGON_EDIT:
            if self.session.polygons:
                self.state.active_polygon_idx = len(self.session.polygons) - 1

        self.redraw()

    def maybe_snap(self, x: float, y: float) -> Tuple[float, float, bool, float]:
        """Snap coordinates to nearest edge if edge snap is enabled."""
        if not self.session.edge_snap_enabled:
            return x, y, False, 1.0
        if self._edge_detector is None:
            self._edge_detector = EdgeDetector(self.original_image)
            self._edge_detector.compute_subpixel()
        result = self._edge_detector.snap_to_edge(
            x, y, radius=float(self.session.edge_snap_radius)
        )
        if result is not None:
            return result[0], result[1], True, result[2]
        return x, y, False, 1.0

    def toggle_edge_snap(self) -> None:
        """Toggle edge snap on/off."""
        self.set_edge_snap_enabled(not self.session.edge_snap_enabled)

    def set_edge_snap_enabled(self, enabled: bool) -> None:
        """Enable/disable edge snap and refresh the UI."""
        enabled = bool(enabled)
        if enabled and self._edge_detector is None:
            self._edge_detector = EdgeDetector(self.original_image)
            self._edge_detector.compute_subpixel()
        if self.session.edge_snap_enabled == enabled:
            return
        self.session.edge_snap_enabled = enabled
        self.redraw()

    def set_snap_radius(self, radius: int) -> None:
        """Set edge snap query radius (pixels)."""
        radius = int(radius)
        if radius <= 0:
            raise ValueError("snap radius must be positive")
        if self.session.edge_snap_radius == radius:
            return
        self.session.edge_snap_radius = radius
        self._toolbar_mgr.set_snap_radius(radius)
        self._toolbar_mgr.update_highlight(
            self.state.mode,
            self.session.edge_snap_enabled,
            radius,
        )
        self._refresh_status_bar()
        if self.session.edge_snap_enabled:
            self.redraw()

    def adjust_snap_radius(self, delta: int) -> None:
        """Nudge snap radius by +/- delta via keyboard shortcuts."""
        self.set_snap_radius(max(1, self.session.edge_snap_radius + int(delta)))

    def set_display_unit(self, unit: Unit) -> None:
        """Set display/output unit for measured physical quantities."""
        if unit == Unit.PIXEL:
            return
        if self.session.display_unit == unit:
            return
        self.session.display_unit = unit
        self._toolbar_mgr.set_display_unit(unit)
        self._remeasure_all()
        self.redraw()

    # -- Calibration (synchronous modal dialogs) --

    def finish_calibrate_line(self) -> None:
        """Finish a two-point line calibration."""
        pts = self.state.points
        if len(pts) != 2:
            return

        parsed = self._prompt_line_calibration_input()
        if parsed is None:
            self.state.points = []
            self.redraw()
            return
        length, unit = parsed

        line = calibrate_from_line(pts[0], pts[1], length, unit)
        self._undo_stack.append(
            ("add_cal_line", len(self.session.calibration.lines))
        )
        self.session.calibration.lines.append(line)
        self.session.calibration.unit = unit
        update_calibration_lines(
            self.session.calibration,
            sigma_per_pixel=DEFAULT_LINE_ENDPOINT_SIGMA_PX,
            robust=True,
            huber_k=DEFAULT_LINE_HUBER_K,
            max_irls_iters=DEFAULT_LINE_IRLS_MAX_ITERS,
            irls_tol=DEFAULT_LINE_IRLS_TOL,
        )
        self._show_line_calibration_quality_warning()

        self.state.switch_to(Mode.VIEW)
        self._remeasure_all()
        self.redraw()

    def finish_calibrate_rect(self) -> None:
        """Finish a four-corner rectangle calibration."""
        pts = self.state.points
        if len(pts) != 4:
            return

        parsed = self._prompt_rect_calibration_input()
        if parsed is None:
            self.state.points = []
            self.redraw()
            return
        w, h, unit = parsed

        rect = calibrate_from_rect(pts, w, h, unit)
        self.session.calibration.rect = rect

        answer = QMessageBox.question(
            self,
            tr("dialog.perspective_title"),
            tr("dialog.perspective_warp"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )

        if answer == QMessageBox.StandardButton.Yes:
            cal, warped = update_calibration_rect(
                self.session.calibration,
                image=self.original_image,
                dst_scale=max(
                    10.0, self.session.calibration.pixels_per_unit
                ),
            )
            self.session.calibration = cal
            if warped is not None:
                self._warped_image = warped
                self.display_image = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
                self.canvas.set_image_rgb(self.display_image)
        else:
            update_calibration_rect(
                self.session.calibration, dst_scale=10.0
            )

        self.state.switch_to(Mode.VIEW)
        self._remeasure_all()
        self.redraw()

    # -- Polygon drawing --

    def add_polygon_vertex(
        self,
        x: float,
        y: float,
        snapped: bool,
        confidence: float,
    ) -> None:
        """Add a vertex to the current polygon being drawn."""
        if self._current_polygon is None:
            return
        self._current_polygon.vertices.append(
            Vertex(
                x=x,
                y=y,
                snapped=bool(snapped),
                confidence=float(confidence) if snapped else 1.0,
            )
        )
        self.redraw()

    def close_current_polygon(self) -> None:
        """Close and finalize the current polygon."""
        poly = self._current_polygon
        if poly is None or poly.n_vertices < 3:
            return

        # Remove redundant last vertex if very close to previous (double-click)
        if poly.n_vertices >= 4:
            last = poly.vertices[-1]
            prev = poly.vertices[-2]
            dist = ((last.x - prev.x) ** 2 + (last.y - prev.y) ** 2) ** 0.5
            if dist < 3.0:
                poly.vertices.pop()

        poly.closed = True
        self._undo_stack.append(("add_polygon", len(self.session.polygons)))
        self.session.polygons.append(poly)
        self._current_polygon = None

        if self.session.calibration.is_calibrated:
            mg = self._get_max_grad()
            result = measure_polygon(
                poly,
                self.session.calibration,
                target_unit=self.session.display_unit,
                max_grad=mg,
            )
            self.session.results.append(result)

        self.state.active_polygon_idx = len(self.session.polygons) - 1
        self.state.switch_to(Mode.VIEW)
        self.redraw()

    # -- Polygon editing --

    def find_nearest_vertex(
        self, x: float, y: float, threshold: float = 15.0
    ) -> Optional[int]:
        """Find the nearest vertex index in the active polygon."""
        idx = self.state.active_polygon_idx
        if idx is None or idx >= len(self.session.polygons):
            return None
        poly = self.session.polygons[idx]
        best_dist = threshold
        best_i = None
        for i, v in enumerate(poly.vertices):
            d = ((v.x - x) ** 2 + (v.y - y) ** 2) ** 0.5
            if d < best_dist:
                best_dist = d
                best_i = i
        return best_i

    def drag_vertex(self, x: float, y: float) -> None:
        """Move the dragged vertex to new position."""
        idx = self.state.active_polygon_idx
        vi = self.state.dragging_vertex_idx
        if idx is None or vi is None:
            return
        poly = self.session.polygons[idx]
        if vi < poly.n_vertices:
            sx, sy, snapped, confidence = self.maybe_snap(x, y)
            poly.vertices[vi].x = sx
            poly.vertices[vi].y = sy
            poly.vertices[vi].snapped = snapped
            poly.vertices[vi].confidence = float(confidence) if snapped else 1.0
            self._remeasure_polygon(idx)
            self.redraw()

    def delete_nearest_vertex(self) -> None:
        """Delete the nearest vertex from the active polygon."""
        idx = self.state.active_polygon_idx
        if idx is None:
            return
        poly = self.session.polygons[idx]
        if poly.n_vertices <= 3:
            return
        vi = (
            self.state.hover_vertex_idx
            if self.state.hover_vertex_idx is not None
            else self.state.dragging_vertex_idx
        )
        if vi is not None and vi < poly.n_vertices:
            poly.vertices.pop(vi)
            self.state.dragging_vertex_idx = None
            self.state.hover_vertex_idx = None
            self._remeasure_polygon(idx)
            self.redraw()

    def insert_vertex_on_edge(self) -> None:
        """Insert a vertex at the midpoint of the nearest edge."""
        idx = self.state.active_polygon_idx
        if idx is None:
            return
        poly = self.session.polygons[idx]
        if poly.n_vertices < 2:
            return
        vi = (
            self.state.hover_vertex_idx
            if self.state.hover_vertex_idx is not None
            else self.state.dragging_vertex_idx
        )
        if vi is None:
            vi = poly.n_vertices - 1
        j = (vi + 1) % poly.n_vertices
        mx = (poly.vertices[vi].x + poly.vertices[j].x) / 2
        my = (poly.vertices[vi].y + poly.vertices[j].y) / 2
        poly.vertices.insert(j, Vertex(x=mx, y=my))
        self._remeasure_polygon(idx)
        self.redraw()

    # -- Click-to-select polygon --

    def find_nearest_polygon(
        self, x: float, y: float, threshold: float = 15.0
    ) -> Optional[int]:
        """Find the nearest closed polygon by point-to-edge distance."""
        best_dist = threshold
        best_idx = None
        for i, poly in enumerate(self.session.polygons):
            if not poly.closed or poly.n_vertices < 3:
                continue
            for k in range(poly.n_vertices):
                j = (k + 1) % poly.n_vertices
                d = _point_to_segment_dist(
                    x, y,
                    poly.vertices[k].x, poly.vertices[k].y,
                    poly.vertices[j].x, poly.vertices[j].y,
                )
                if d < best_dist:
                    best_dist = d
                    best_idx = i
        return best_idx

    def delete_polygon(self, idx: int) -> None:
        """Delete a polygon by index."""
        if 0 <= idx < len(self.session.polygons):
            label = self.session.polygons[idx].label
            self.session.polygons.pop(idx)
            self.session.results = [
                r for r in self.session.results if r.polygon_label != label
            ]
            if self.state.active_polygon_idx == idx:
                self.state.active_polygon_idx = None
            elif (
                self.state.active_polygon_idx is not None
                and self.state.active_polygon_idx > idx
            ):
                self.state.active_polygon_idx -= 1
            self.redraw()

    # -- Grid detection --

    def run_grid_detect(self) -> None:
        """Run grid detection on the selected ROI."""
        start = self.state.roi_start
        end = self.state.roi_end
        if start is None or end is None:
            return

        x0 = int(min(start[0], end[0]))
        y0 = int(min(start[1], end[1]))
        x1 = int(max(start[0], end[0]))
        y1 = int(max(start[1], end[1]))
        roi = (x0, y0, x1 - x0, y1 - y0)

        if roi[2] < 20 or roi[3] < 20:
            return

        img = (
            self._warped_image
            if self._warped_image is not None
            else self.original_image
        )
        result = detect_grid(
            img, roi=roi, calibration=self.session.calibration
        )
        if result is not None:
            result.origin_x = float(x0)
            result.origin_y = float(y0)
            self.session.grid_results.append(result)

        self.state.switch_to(Mode.VIEW)
        self.redraw()

    # -- Pad pitch calibration --

    def run_pad_pitch_calibrate(self) -> None:
        """Run FFT pad pitch detection on the selected ROI and calibrate."""
        start = self.state.roi_start
        end = self.state.roi_end
        if start is None or end is None:
            return

        x0 = int(min(start[0], end[0]))
        y0 = int(min(start[1], end[1]))
        x1 = int(max(start[0], end[0]))
        y1 = int(max(start[1], end[1]))
        roi = (x0, y0, x1 - x0, y1 - y0)

        if roi[2] < 20 or roi[3] < 20:
            return

        img = (
            self._warped_image
            if self._warped_image is not None
            else self.original_image
        )

        # Detect pitch (without known length yet — we just need pixels).
        from ..grid import detect_grid

        grid_result = detect_grid(img, roi=roi)
        if grid_result is None:
            QMessageBox.information(
                self,
                tr("dialog.calibrate_pad_pitch_title"),
                tr("dialog.pad_pitch_failed"),
            )
            self.state.switch_to(Mode.VIEW)
            self.redraw()
            return

        pitch_px = float(np.hypot(grid_result.pitch_x, grid_result.pitch_y))

        # Prompt for known pitch.
        parsed = self._prompt_pad_pitch_input(pitch_px, grid_result.confidence)
        if parsed is None:
            self.state.switch_to(Mode.VIEW)
            self.redraw()
            return
        known_pitch, unit = parsed

        result = calibrate_from_pad_pitch(
            img, roi=roi, known_pitch=known_pitch, unit=unit,
        )
        if result is None:
            QMessageBox.information(
                self,
                tr("dialog.calibrate_pad_pitch_title"),
                tr("dialog.pad_pitch_failed"),
            )
            self.state.switch_to(Mode.VIEW)
            self.redraw()
            return

        line, grid_res = result
        self._undo_stack.append(
            ("add_cal_line", len(self.session.calibration.lines))
        )
        self.session.calibration.lines.append(line)
        self.session.calibration.unit = unit
        update_calibration_lines(
            self.session.calibration,
            sigma_per_pixel=DEFAULT_LINE_ENDPOINT_SIGMA_PX,
            robust=True,
            huber_k=DEFAULT_LINE_HUBER_K,
            max_irls_iters=DEFAULT_LINE_IRLS_MAX_ITERS,
            irls_tol=DEFAULT_LINE_IRLS_TOL,
        )
        self._show_line_calibration_quality_warning()

        # Store grid result for overlay.
        self.session.grid_results.append(grid_res)

        self.state.switch_to(Mode.VIEW)
        self._remeasure_all()
        self.redraw()

    def _prompt_pad_pitch_input(
        self,
        pitch_px: float,
        confidence: float,
    ) -> Optional[Tuple[float, Unit]]:
        """Prompt until valid pad pitch input is provided or cancelled."""
        default_text = "80 um"
        while True:
            text, ok = QInputDialog.getText(
                self,
                tr("dialog.calibrate_pad_pitch_title"),
                tr(
                    "dialog.enter_pad_pitch",
                    pitch_px=pitch_px,
                    conf=confidence,
                ),
                QLineEdit.EchoMode.Normal,
                default_text,
            )
            if not ok:
                return None

            raw = text.strip()
            length, unit = self._parse_length_input(raw)
            if length is not None:
                return length, unit

            QMessageBox.warning(
                self,
                tr("dialog.input_error_title"),
                tr("dialog.invalid_length"),
            )
            default_text = raw

    # -- Undo --

    def undo(self) -> None:
        """Undo the last action."""
        if not self._undo_stack:
            return
        action, data = self._undo_stack.pop()
        if action == "add_polygon":
            idx = data
            if idx < len(self.session.polygons):
                label = self.session.polygons[idx].label
                self.session.polygons.pop(idx)
                self.session.results = [
                    r
                    for r in self.session.results
                    if r.polygon_label != label
                ]
        elif action == "add_cal_line":
            idx = data
            if idx < len(self.session.calibration.lines):
                self.session.calibration.lines.pop(idx)
                if self.session.calibration.lines:
                    update_calibration_lines(
                        self.session.calibration,
                        sigma_per_pixel=DEFAULT_LINE_ENDPOINT_SIGMA_PX,
                        robust=True,
                        huber_k=DEFAULT_LINE_HUBER_K,
                        max_irls_iters=DEFAULT_LINE_IRLS_MAX_ITERS,
                        irls_tol=DEFAULT_LINE_IRLS_TOL,
                    )
                else:
                    self.session.calibration.pixels_per_unit = 0.0
                    self.session.calibration.ppu_uncertainty = 0.0
                self._remeasure_all()
        self.redraw()

    # -- Save / Export --

    def save_session(self) -> None:
        """Save the current session."""
        path = self.session_path or self.image_path + ".diemeter.json"
        save_session(self.session, path)
        self._status_bar.showMessage(f"Session saved to {path}", 3000)

    def export_results(self) -> None:
        """Export measurement results via file dialog."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            tr("dialog.export_title"),
            "",
            "CSV (*.csv);;JSON (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            export_results(
                path,
                self.session.results,
                self._grid_results_for_current_unit() or None,
            )
            self._status_bar.showMessage(f"Exported to {path}", 3000)
        except Exception as e:
            QMessageBox.warning(self, "Export Error", str(e))

    # -- Redraw --

    def redraw(self) -> None:
        """Redraw all overlays."""
        self.canvas.clear_overlays()

        # Calibration lines
        for line in self.session.calibration.lines:
            self.canvas.add_overlay(CalibrationLineItem(line))

        # Transient click points
        if self.state.points:
            self.canvas.add_overlay(PointMarkerItem(self.state.points))

        # Grid overlays
        for g in self.session.grid_results:
            if g.pitch_x > 0 and g.pitch_y > 0:
                ext_w = g.pitch_x * max(20, int(500 / max(g.pitch_x, 1)))
                ext_h = g.pitch_y * max(20, int(500 / max(g.pitch_y, 1)))
                self.canvas.add_overlay(
                    GridOverlayItem(
                        origin=(g.origin_x, g.origin_y),
                        pitch_x=g.pitch_x,
                        pitch_y=g.pitch_y,
                        angle_deg=g.angle_deg,
                        extent=(ext_w, ext_h),
                    )
                )

        # Stored polygons
        for i, poly in enumerate(self.session.polygons):
            area_text = None
            if poly.closed and self.session.calibration.is_calibrated:
                for r in self.session.results:
                    if r.polygon_label == poly.label:
                        rel_err = (
                            (r.area_uncertainty / r.area_physical * 100)
                            if r.area_physical > 0 else 0.0
                        )
                        area_text = (
                            f"{r.area_physical:.4g} {r.unit.value}\u00b2"
                            f" \u00b1{rel_err:.1f}%"
                        )
                        break
            active_vi = None
            hover_vi = None
            is_selected = i == self.state.active_polygon_idx
            if is_selected:
                active_vi = self.state.dragging_vertex_idx
                hover_vi = self.state.hover_vertex_idx
            self.canvas.add_overlay(
                PolygonOverlayItem(
                    poly,
                    active_vertex_idx=active_vi,
                    hover_vertex_idx=hover_vi,
                    area_text=area_text,
                    selected=is_selected,
                )
            )

        # Current in-progress polygon
        if (
            self._current_polygon is not None
            and self._current_polygon.n_vertices > 0
        ):
            self.canvas.add_overlay(
                PolygonOverlayItem(self._current_polygon)
            )

        # Rubber-band preview line
        cursor = self.state.cursor_pos
        if cursor is not None:
            mode = self.state.mode
            if mode == Mode.POLYGON_DRAW and self._current_polygon is not None:
                n = self._current_polygon.n_vertices
                if n > 0:
                    last_v = self._current_polygon.vertices[-1]
                    self.canvas.add_overlay(
                        RubberbandLineItem(
                            (last_v.x, last_v.y), cursor
                        )
                    )
            elif mode in (Mode.CALIBRATE_LINE, Mode.CALIBRATE_RECT):
                if self.state.points:
                    last_pt = self.state.points[-1]
                    self.canvas.add_overlay(
                        RubberbandLineItem(last_pt, cursor)
                    )

            # Snap marker preview
            if self.session.edge_snap_enabled and mode in (
                Mode.POLYGON_DRAW,
                Mode.CALIBRATE_LINE,
                Mode.CALIBRATE_RECT,
            ):
                sx, sy, snapped, _ = self.maybe_snap(cursor[0], cursor[1])
                if snapped:
                    self.canvas.add_overlay(SnapMarkerItem((sx, sy)))

        # ROI rectangle
        if self.state.roi_start and self.state.roi_end:
            self.canvas.add_overlay(
                RoiRectItem(self.state.roi_start, self.state.roi_end)
            )

        # Info panel
        self._info_panel.refresh(
            self.session,
            selected_polygon_idx=self.state.active_polygon_idx,
            display_unit=self.session.display_unit,
        )

        # Status bar + toolbar
        self._refresh_status_bar()
        self._toolbar_mgr.update_highlight(
            self.state.mode,
            self.session.edge_snap_enabled,
            self.session.edge_snap_radius,
        )

    # -- i18n --

    def toggle_language(self) -> None:
        """Toggle between English and Chinese."""
        set_language("zh" if get_language() == "en" else "en")
        self._retranslate_ui()

    def _retranslate_ui(self) -> None:
        """Refresh all translatable UI text."""
        self._toolbar_mgr.retranslate()
        self._refresh_status_bar()
        self.redraw()

    # -- Status bar --

    def _refresh_status_bar(
        self, cursor: Optional[Tuple[float, float]] = None
    ) -> None:
        """Update the status bar text."""
        parts = []
        display_unit = self.session.display_unit

        # Mode + help
        mode_key = f"help.{self.state.mode.value}"
        parts.append(f"[{tr(f'mode.{self.state.mode.value}')}]  {tr(mode_key)}")

        # Calibration
        cal = self.session.calibration
        if cal.is_calibrated:
            ppu = cal.pixels_per_unit
            ppu_unc = cal.ppu_uncertainty
            ppu_unit = cal.unit
            if display_unit != cal.unit:
                try:
                    # Convert px/(cal unit) -> px/(display unit).
                    scale = convert_length(1.0, display_unit, cal.unit)
                    ppu *= scale
                    ppu_unc *= scale
                    ppu_unit = display_unit
                except ValueError:
                    pass
            parts.append(
                tr("status.scale", ppu=ppu, unit=ppu_unit.value, unc=ppu_unc)
            )
        else:
            parts.append(tr("status.not_calibrated"))

        # Snap
        parts.append(
            tr("status.snap_on", radius=self.session.edge_snap_radius)
            if self.session.edge_snap_enabled
            else tr("status.snap_off", radius=self.session.edge_snap_radius)
        )

        # Cursor
        if cursor:
            cx, cy = cursor
            pos = tr("status.cursor", x=cx, y=cy)
            if cal.is_calibrated:
                px = cx / cal.pixels_per_unit
                py = cy / cal.pixels_per_unit
                phys_unit = cal.unit
                if display_unit != cal.unit:
                    try:
                        px = convert_length(px, cal.unit, display_unit)
                        py = convert_length(py, cal.unit, display_unit)
                        phys_unit = display_unit
                    except ValueError:
                        pass
                pos += tr("status.cursor_phys",
                          px=px, py=py, unit=phys_unit.value)
            parts.append(pos)

        # Current polygon area
        poly = self._current_polygon
        if poly is not None and poly.n_vertices >= 3:
            area_px = polygon_area_pixels(poly)
            area_str = tr("status.current_area_px", area=area_px)
            if cal.is_calibrated:
                area_phys = area_px / (cal.pixels_per_unit ** 2)
                area_unit = cal.unit
                if display_unit != cal.unit:
                    try:
                        area_phys = convert_area(area_phys, cal.unit, display_unit)
                        area_unit = display_unit
                    except ValueError:
                        pass
                area_str += tr("status.current_area_phys",
                               area=area_phys, unit=area_unit.value)
            parts.append(area_str)

        # Grid results
        for g in self._grid_results_for_current_unit():
            parts.append(
                tr("status.grid", px=g.pitch_x_physical,
                   py=g.pitch_y_physical, unit=g.unit.value,
                   conf=g.confidence)
            )

        self._status_bar.showMessage("  |  ".join(parts))

    # -- Internal helpers --

    def _get_max_grad(self) -> float:
        """Get max gradient from edge detector, or default."""
        if self._edge_detector is not None and self._edge_detector.max_grad > 0:
            return self._edge_detector.max_grad
        return 255.0

    def _line_calibration_quality_warning_text(self) -> Optional[str]:
        """Build a quantitative warning message when line calibration quality is weak."""
        cal = self.session.calibration
        if not cal.lines or not cal.is_calibrated:
            return None

        rel_unc_pct = (
            100.0 * cal.ppu_uncertainty / cal.pixels_per_unit
            if cal.pixels_per_unit > 0
            else 0.0
        )

        if len(cal.lines) == 1:
            px_len = cal.lines[0].pixel_length
            if px_len < _LINE_CAL_MIN_RECOMMENDED_PIXELS:
                return tr(
                    "warn.cal.single_line_short",
                    px=px_len,
                    min_px=_LINE_CAL_MIN_RECOMMENDED_PIXELS,
                    sigma=DEFAULT_LINE_ENDPOINT_SIGMA_PX,
                    rel=rel_unc_pct,
                )
            return None

        chi2_value, p_value, consistent = chi_squared_consistency(
            cal.lines,
            sigma_per_pixel=DEFAULT_LINE_ENDPOINT_SIGMA_PX,
            confidence_level=_LINE_CAL_CONFIDENCE_LEVEL,
        )
        if not consistent:
            return tr(
                "warn.cal.inconsistent",
                n=len(cal.lines),
                chi2=chi2_value,
                p=p_value,
                conf=100.0 * _LINE_CAL_CONFIDENCE_LEVEL,
                rel=rel_unc_pct,
            )

        if rel_unc_pct > _LINE_CAL_WARN_REL_UNC_PERCENT:
            return tr(
                "warn.cal.high_unc",
                n=len(cal.lines),
                rel=rel_unc_pct,
                threshold=_LINE_CAL_WARN_REL_UNC_PERCENT,
            )

        return None

    def _show_line_calibration_quality_warning(self) -> None:
        message = self._line_calibration_quality_warning_text()
        if message:
            QMessageBox.warning(
                self,
                tr("dialog.calibration_quality_title"),
                message,
            )

    def _grid_results_for_current_unit(self) -> List[GridResult]:
        """Return grid results converted to the current display unit."""
        out: List[GridResult] = []
        target = self.session.display_unit
        for g in self.session.grid_results:
            if g.unit == Unit.PIXEL or g.unit == target:
                out.append(g)
                continue
            try:
                out.append(
                    GridResult(
                        pitch_x=g.pitch_x,
                        pitch_y=g.pitch_y,
                        pitch_x_physical=convert_length(
                            g.pitch_x_physical, g.unit, target
                        ),
                        pitch_y_physical=convert_length(
                            g.pitch_y_physical, g.unit, target
                        ),
                        angle_deg=g.angle_deg,
                        confidence=g.confidence,
                        unit=target,
                        origin_x=g.origin_x,
                        origin_y=g.origin_y,
                    )
                )
            except ValueError:
                out.append(g)
        return out

    def _remeasure_all(self) -> None:
        """Re-measure all polygons with current calibration."""
        self.session.results.clear()
        if not self.session.calibration.is_calibrated:
            return
        mg = self._get_max_grad()
        for poly in self.session.polygons:
            if poly.closed and poly.n_vertices >= 3:
                result = measure_polygon(
                    poly,
                    self.session.calibration,
                    target_unit=self.session.display_unit,
                    max_grad=mg,
                )
                self.session.results.append(result)

    def _remeasure_polygon(self, idx: int) -> None:
        """Re-measure a single polygon."""
        if not self.session.calibration.is_calibrated:
            return
        poly = self.session.polygons[idx]
        mg = self._get_max_grad()
        result = measure_polygon(
            poly,
            self.session.calibration,
            target_unit=self.session.display_unit,
            max_grad=mg,
        )
        for i, r in enumerate(self.session.results):
            if r.polygon_label == poly.label:
                self.session.results[i] = result
                return
        self.session.results.append(result)

    def _prompt_line_calibration_input(self) -> Optional[Tuple[float, Unit]]:
        """Prompt until valid line calibration input is provided or cancelled."""
        while True:
            text, ok = QInputDialog.getText(
                self,
                tr("dialog.calibrate_line_title"),
                tr("dialog.enter_distance"),
                QLineEdit.EchoMode.Normal,
                self._last_line_input,
            )
            if not ok:
                return None

            raw = text.strip()
            length, unit = self._parse_length_input(raw)
            if length is not None:
                self._last_line_input = raw
                return length, unit

            QMessageBox.warning(
                self,
                tr("dialog.input_error_title"),
                tr("dialog.invalid_length"),
            )

    def _prompt_rect_calibration_input(self) -> Optional[Tuple[float, float, Unit]]:
        """Prompt until valid rectangle dimensions are provided or cancelled."""
        while True:
            text, ok = QInputDialog.getText(
                self,
                tr("dialog.calibrate_rect_title"),
                tr("dialog.enter_rect_dim"),
                QLineEdit.EchoMode.Normal,
                self._last_rect_input,
            )
            if not ok:
                return None

            raw = text.strip()
            w, h, unit = self._parse_rect_input(raw)
            if w is not None and h is not None:
                self._last_rect_input = raw
                return w, h, unit

            QMessageBox.warning(
                self,
                tr("dialog.input_error_title"),
                tr("dialog.invalid_rect"),
            )

    @staticmethod
    def _parse_length_input(s: str) -> Tuple[Optional[float], Unit]:
        """Parse '14.5 mm' style input into (value, unit)."""
        s = s.strip()
        parts = s.split()
        if len(parts) == 1:
            try:
                val = float(parts[0])
                if val <= 0:
                    return None, Unit.MILLIMETER
                return val, Unit.MILLIMETER
            except ValueError:
                return None, Unit.MILLIMETER
        elif len(parts) == 2:
            try:
                val = float(parts[0])
                if val <= 0:
                    return None, Unit.MILLIMETER
                unit = unit_from_str(parts[1])
                return val, unit
            except (ValueError, KeyError):
                return None, Unit.MILLIMETER
        return None, Unit.MILLIMETER

    @staticmethod
    def _parse_rect_input(
        s: str,
    ) -> Tuple[Optional[float], Optional[float], Unit]:
        """Parse '10 x 8 mm' or '10x8mm' style input."""
        s = s.strip().lower()
        s = s.replace(" x ", "x").replace("\u00d7 ", "x").replace(" \u00d7", "x")

        unit = Unit.MILLIMETER
        for suffix in ["mm", "cm", "um", "in", "mil", "m"]:
            if s.endswith(suffix):
                unit = unit_from_str(suffix)
                s = s[: -len(suffix)].strip()
                break

        parts = s.split("x")
        if len(parts) != 2:
            return None, None, unit
        try:
            w = float(parts[0].strip())
            h = float(parts[1].strip())
            if w <= 0 or h <= 0:
                return None, None, unit
            return w, h, unit
        except ValueError:
            return None, None, unit

    def show(self) -> None:
        """Display the interactive window and enter event loop."""
        super().show()
        self._qapp.exec()


def _point_to_segment_dist(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
) -> float:
    """Compute the minimum distance from point (px,py) to segment (ax,ay)-(bx,by)."""
    dx = bx - ax
    dy = by - ay
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return ((px - proj_x) ** 2 + (py - proj_y) ** 2) ** 0.5
