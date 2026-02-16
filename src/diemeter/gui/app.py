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
    QMainWindow,
    QMessageBox,
    QStatusBar,
)

from ..calibration import (
    calibrate_from_line,
    calibrate_from_rect,
    update_calibration_lines,
    update_calibration_rect,
)
from ..edge import EdgeDetector
from ..export import export_results
from ..grid import detect_grid
from ..measurement import measure_polygon, polygon_area_pixels
from ..model import (
    Calibration,
    Polygon,
    Session,
    Unit,
    Vertex,
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
from .toolbar import ToolbarManager


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

        # Interactive state
        self.state = ModeState()
        self._current_polygon: Optional[Polygon] = None
        self._polygon_counter = 0

        # Edge detector (lazy init)
        self._edge_detector: Optional[EdgeDetector] = None

        # Undo stack: list of (action_type, data) tuples
        self._undo_stack: List[Tuple[str, object]] = []

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
            "Escape": lambda: self.switch_mode(Mode.VIEW),
            "S": lambda: self.toggle_edge_snap(),
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

    def maybe_snap(self, x: float, y: float) -> Tuple[float, float]:
        """Snap coordinates to nearest edge if edge snap is enabled."""
        if not self.session.edge_snap_enabled:
            return x, y
        if self._edge_detector is None:
            self._edge_detector = EdgeDetector(self.original_image)
            self._edge_detector.compute_subpixel()
        result = self._edge_detector.snap_to_edge(
            x, y, radius=float(self.session.edge_snap_radius)
        )
        if result is not None:
            return result[0], result[1]
        return x, y

    def toggle_edge_snap(self) -> None:
        """Toggle edge snap on/off."""
        self.session.edge_snap_enabled = not self.session.edge_snap_enabled
        if self.session.edge_snap_enabled and self._edge_detector is None:
            self._edge_detector = EdgeDetector(self.original_image)
            self._edge_detector.compute_subpixel()
        self.redraw()

    # -- Calibration (synchronous modal dialogs) --

    def finish_calibrate_line(self) -> None:
        """Finish a two-point line calibration."""
        pts = self.state.points
        if len(pts) != 2:
            return

        text, ok = QInputDialog.getText(
            self,
            tr("dialog.calibrate_line_title"),
            tr("dialog.enter_distance"),
        )
        if not ok or not text.strip():
            self.state.points = []
            self.redraw()
            return

        length, unit = self._parse_length_input(text.strip())
        if length is None:
            self.state.points = []
            self.redraw()
            return

        line = calibrate_from_line(pts[0], pts[1], length, unit)
        self._undo_stack.append(
            ("add_cal_line", len(self.session.calibration.lines))
        )
        self.session.calibration.lines.append(line)
        self.session.calibration.unit = unit
        update_calibration_lines(self.session.calibration)

        self.state.switch_to(Mode.VIEW)
        self._remeasure_all()
        self.redraw()

    def finish_calibrate_rect(self) -> None:
        """Finish a four-corner rectangle calibration."""
        pts = self.state.points
        if len(pts) != 4:
            return

        text, ok = QInputDialog.getText(
            self,
            tr("dialog.calibrate_rect_title"),
            tr("dialog.enter_rect_dim"),
        )
        if not ok or not text.strip():
            self.state.points = []
            self.redraw()
            return

        w, h, unit = self._parse_rect_input(text.strip())
        if w is None:
            self.state.points = []
            self.redraw()
            return

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

    def add_polygon_vertex(self, x: float, y: float) -> None:
        """Add a vertex to the current polygon being drawn."""
        if self._current_polygon is None:
            return
        snapped = self.session.edge_snap_enabled
        self._current_polygon.vertices.append(
            Vertex(x=x, y=y, snapped=snapped)
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
            result = measure_polygon(poly, self.session.calibration)
            self.session.results.append(result)

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
            x, y = self.maybe_snap(x, y)
            poly.vertices[vi].x = x
            poly.vertices[vi].y = y
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
                    update_calibration_lines(self.session.calibration)
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
                self.session.grid_results or None,
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
                        area_text = (
                            f"{r.area_physical:.4g} {r.unit.value}\u00b2"
                            f" \u00b1 {r.area_uncertainty:.2g}"
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
                sx, sy = self.maybe_snap(cursor[0], cursor[1])
                if (sx, sy) != (cursor[0], cursor[1]):
                    self.canvas.add_overlay(SnapMarkerItem((sx, sy)))

        # ROI rectangle
        if self.state.roi_start and self.state.roi_end:
            self.canvas.add_overlay(
                RoiRectItem(self.state.roi_start, self.state.roi_end)
            )

        # Status bar + toolbar
        self._refresh_status_bar()
        self._toolbar_mgr.update_highlight(
            self.state.mode, self.session.edge_snap_enabled
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

        # Mode + help
        mode_key = f"help.{self.state.mode.value}"
        parts.append(f"[{tr(f'mode.{self.state.mode.value}')}]  {tr(mode_key)}")

        # Calibration
        cal = self.session.calibration
        if cal.is_calibrated:
            parts.append(
                tr("status.scale", ppu=cal.pixels_per_unit,
                   unit=cal.unit.value, unc=cal.ppu_uncertainty)
            )
        else:
            parts.append(tr("status.not_calibrated"))

        # Snap
        parts.append(
            tr("status.snap_on") if self.session.edge_snap_enabled
            else tr("status.snap_off")
        )

        # Cursor
        if cursor:
            cx, cy = cursor
            pos = tr("status.cursor", x=cx, y=cy)
            if cal.is_calibrated:
                px = cx / cal.pixels_per_unit
                py = cy / cal.pixels_per_unit
                pos += tr("status.cursor_phys",
                          px=px, py=py, unit=cal.unit.value)
            parts.append(pos)

        # Current polygon area
        poly = self._current_polygon
        if poly is not None and poly.n_vertices >= 3:
            area_px = polygon_area_pixels(poly)
            area_str = tr("status.current_area_px", area=area_px)
            if cal.is_calibrated:
                area_phys = area_px / (cal.pixels_per_unit ** 2)
                area_str += tr("status.current_area_phys",
                               area=area_phys, unit=cal.unit.value)
            parts.append(area_str)

        # Grid results
        for g in self.session.grid_results:
            parts.append(
                tr("status.grid", px=g.pitch_x_physical,
                   py=g.pitch_y_physical, unit=g.unit.value,
                   conf=g.confidence)
            )

        self._status_bar.showMessage("  |  ".join(parts))

    # -- Internal helpers --

    def _remeasure_all(self) -> None:
        """Re-measure all polygons with current calibration."""
        self.session.results.clear()
        if not self.session.calibration.is_calibrated:
            return
        for poly in self.session.polygons:
            if poly.closed and poly.n_vertices >= 3:
                result = measure_polygon(poly, self.session.calibration)
                self.session.results.append(result)

    def _remeasure_polygon(self, idx: int) -> None:
        """Re-measure a single polygon."""
        if not self.session.calibration.is_calibrated:
            return
        poly = self.session.polygons[idx]
        result = measure_polygon(poly, self.session.calibration)
        for i, r in enumerate(self.session.results):
            if r.polygon_label == poly.label:
                self.session.results[i] = result
                return
        self.session.results.append(result)

    @staticmethod
    def _parse_length_input(s: str) -> Tuple[Optional[float], Unit]:
        """Parse '14.5 mm' style input into (value, unit)."""
        s = s.strip()
        parts = s.split()
        if len(parts) == 1:
            try:
                return float(parts[0]), Unit.MILLIMETER
            except ValueError:
                return None, Unit.MILLIMETER
        elif len(parts) == 2:
            try:
                val = float(parts[0])
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
