"""Right-side info panel for calibration details, polygon measurements, and grid info."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDockWidget,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..model import Calibration, MeasurementResult, Polygon, Session
from .i18n import tr

if TYPE_CHECKING:
    from .app import MeasurementApp


class InfoPanel(QDockWidget):
    """Dockable info panel showing calibration, polygon, and grid details."""

    def __init__(self, app: MeasurementApp) -> None:
        super().__init__("Info", app)
        self._app = app
        self.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        container = QWidget()
        self._layout = QVBoxLayout(container)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setSpacing(12)

        # Calibration section
        self._cal_label = QLabel()
        self._cal_label.setWordWrap(True)
        self._cal_label.setStyleSheet("color: #e0e0e0; font-size: 11px;")
        self._layout.addWidget(self._cal_label)

        # Polygon measurement section
        self._poly_label = QLabel()
        self._poly_label.setWordWrap(True)
        self._poly_label.setStyleSheet("color: #e0e0e0; font-size: 11px;")
        self._layout.addWidget(self._poly_label)

        # Vertex table
        self._vtx_table = QTableWidget()
        self._vtx_table.setColumnCount(7)
        self._vtx_table.setHorizontalHeaderLabels([
            "#", "x(px)", "y(px)", "x(phys)", "y(phys)", "snap", "\u03c3"
        ])
        self._vtx_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._vtx_table.setMaximumHeight(200)
        self._vtx_table.setStyleSheet(
            "QTableWidget { background: #2a2a2a; color: #e0e0e0; font-size: 10px; }"
            "QHeaderView::section { background: #333; color: #ccc; font-size: 10px; }"
        )
        self._layout.addWidget(self._vtx_table)

        # Grid section
        self._grid_label = QLabel()
        self._grid_label.setWordWrap(True)
        self._grid_label.setStyleSheet("color: #e0e0e0; font-size: 11px;")
        self._layout.addWidget(self._grid_label)

        self._layout.addStretch()

        container.setStyleSheet("background: #1e1e1e;")
        self.setWidget(container)
        self.setMinimumWidth(280)

    def refresh(
        self,
        session: Session,
        selected_polygon_idx: Optional[int] = None,
    ) -> None:
        """Update all sections from current session state."""
        self._update_calibration(session.calibration)
        self._update_polygon(session, selected_polygon_idx)
        self._update_grid(session)

    def _update_calibration(self, cal: Calibration) -> None:
        if not cal.is_calibrated:
            self._cal_label.setText(
                "<b>Calibration</b><br>Not calibrated"
            )
            return

        lines = [
            "<b>Calibration</b>",
            f"Scale: {cal.pixels_per_unit:.4f} px/{cal.unit.value} "
            f"\u00b1 {cal.ppu_uncertainty:.4f}",
            f"Method: {'line' if cal.lines else 'rect'}"
            f"{f' ({len(cal.lines)} lines)' if cal.lines else ''}",
        ]

        if cal.homography:
            lines.append("Perspective: corrected")

        self._cal_label.setText("<br>".join(lines))

    def _update_polygon(
        self, session: Session, idx: Optional[int]
    ) -> None:
        if idx is None or idx >= len(session.polygons):
            self._poly_label.setText("<b>Polygon</b><br>None selected")
            self._vtx_table.setRowCount(0)
            return

        poly = session.polygons[idx]
        result: Optional[MeasurementResult] = None
        for r in session.results:
            if r.polygon_label == poly.label:
                result = r
                break

        lines = [f"<b>Polygon: {poly.label}</b>"]

        if result is not None:
            rel_err = (
                (result.area_uncertainty / result.area_physical * 100)
                if result.area_physical > 0 else 0.0
            )
            u = result.unit.value
            lines.extend([
                f"Area: {result.area_physical:.6g} {u}\u00b2 "
                f"\u00b1 {result.area_uncertainty:.4g} ({rel_err:.2f}%)",
                f"  vertex \u03c3: {result.area_uncertainty_vertex:.4g} {u}\u00b2"
                f"  |  ppu \u03c3: {result.area_uncertainty_ppu:.4g} {u}\u00b2",
                f"Perimeter: {result.perimeter_physical:.6g} {u} "
                f"\u00b1 {result.perimeter_uncertainty:.4g}",
                f"Vertices: {result.n_vertices}  |  "
                f"Area(px): {result.area_pixels:.2f}",
            ])
        else:
            lines.append("Not measured (no calibration)")

        self._poly_label.setText("<br>".join(lines))

        # Vertex table
        cal = session.calibration
        self._vtx_table.setRowCount(poly.n_vertices)
        for i, v in enumerate(poly.vertices):
            self._vtx_table.setItem(i, 0, QTableWidgetItem(str(i)))
            self._vtx_table.setItem(i, 1, QTableWidgetItem(f"{v.x:.2f}"))
            self._vtx_table.setItem(i, 2, QTableWidgetItem(f"{v.y:.2f}"))
            if cal.is_calibrated:
                px = v.x / cal.pixels_per_unit
                py = v.y / cal.pixels_per_unit
                self._vtx_table.setItem(i, 3, QTableWidgetItem(f"{px:.4f}"))
                self._vtx_table.setItem(i, 4, QTableWidgetItem(f"{py:.4f}"))
            else:
                self._vtx_table.setItem(i, 3, QTableWidgetItem("-"))
                self._vtx_table.setItem(i, 4, QTableWidgetItem("-"))
            self._vtx_table.setItem(
                i, 5, QTableWidgetItem("\u2713" if v.snapped else "")
            )
            sigma_str = f"{v.confidence:.1f}" if v.snapped else "manual"
            self._vtx_table.setItem(i, 6, QTableWidgetItem(sigma_str))

    def _update_grid(self, session: Session) -> None:
        if not session.grid_results:
            self._grid_label.setText("<b>Grid</b><br>No detections")
            return

        lines = ["<b>Grid Detections</b>"]
        for i, g in enumerate(session.grid_results):
            u = g.unit.value
            lines.append(
                f"#{i+1}: pitch=({g.pitch_x_physical:.3g}, "
                f"{g.pitch_y_physical:.3g}) {u}  "
                f"angle={g.angle_deg:.1f}\u00b0  "
                f"conf={g.confidence:.3f}"
            )
        self._grid_label.setText("<br>".join(lines))
