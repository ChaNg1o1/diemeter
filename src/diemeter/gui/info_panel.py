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

from ..calibration import (
    DEFAULT_LINE_ENDPOINT_SIGMA_PX,
    chi_squared_consistency,
    line_calibration_diagnostics,
)
from ..model import (
    Calibration,
    MeasurementResult,
    Session,
    Unit,
    convert_area,
    convert_length,
)
from ..uncertainty import confidence_to_sigma
from .i18n import tr

if TYPE_CHECKING:
    from .app import MeasurementApp

_LINE_CAL_CONFIDENCE_LEVEL = 0.95


class InfoPanel(QDockWidget):
    """Dockable info panel showing calibration, polygon, and grid details."""

    def __init__(self, app: MeasurementApp) -> None:
        super().__init__(tr("info.title"), app)
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

        # Per-line calibration diagnostics table
        self._cal_table = QTableWidget()
        self._cal_table.setColumnCount(7)
        self._cal_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._cal_table.setMaximumHeight(140)
        self._cal_table.setStyleSheet(
            "QTableWidget { background: #2a2a2a; color: #e0e0e0; font-size: 10px; }"
            "QHeaderView::section { background: #333; color: #ccc; font-size: 10px; }"
        )
        self._layout.addWidget(self._cal_table)

        # Polygon measurement section
        self._poly_label = QLabel()
        self._poly_label.setWordWrap(True)
        self._poly_label.setStyleSheet("color: #e0e0e0; font-size: 11px;")
        self._layout.addWidget(self._poly_label)

        # Vertex table
        self._vtx_table = QTableWidget()
        self._vtx_table.setColumnCount(8)
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
        display_unit: Unit = Unit.MILLIMETER,
    ) -> None:
        """Update all sections from current session state."""
        self.setWindowTitle(tr("info.title"))
        self._vtx_table.setHorizontalHeaderLabels([
            "#",
            "x(px)",
            "y(px)",
            f"x({display_unit.value})",
            f"y({display_unit.value})",
            tr("info.vtx_snap"),
            tr("info.vtx_conf"),
            tr("info.vtx_sigma"),
        ])
        self._update_calibration(session.calibration, display_unit)
        self._update_polygon(session, selected_polygon_idx, display_unit)
        self._update_grid(session, display_unit)

    def _update_calibration(self, cal: Calibration, display_unit: Unit) -> None:
        if not cal.is_calibrated:
            self._cal_label.setText(
                f"<b>{tr('info.calibration')}</b><br>{tr('info.not_calibrated')}"
            )
            self._cal_table.setRowCount(0)
            self._cal_table.setVisible(False)
            return

        ppu = cal.pixels_per_unit
        ppu_unc = cal.ppu_uncertainty
        unit = cal.unit
        if display_unit != cal.unit:
            try:
                # ppu unit: px / (display unit)
                scale = convert_length(1.0, display_unit, cal.unit)
                ppu *= scale
                ppu_unc *= scale
                unit = display_unit
            except ValueError:
                pass

        lines = [
            f"<b>{tr('info.calibration')}</b>",
            tr("info.scale", ppu=ppu, unit=unit.value, unc=ppu_unc),
        ]

        if cal.lines:
            lines.append(tr("info.method_line", n=len(cal.lines)))
            rel_unc_pct = (
                100.0 * cal.ppu_uncertainty / cal.pixels_per_unit
                if cal.pixels_per_unit > 0
                else 0.0
            )
            lines.append(tr("info.cal_rel_unc", rel=rel_unc_pct))
            if len(cal.lines) >= 2:
                chi2_value, p_value, is_consistent = chi_squared_consistency(
                    cal.lines,
                    sigma_per_pixel=DEFAULT_LINE_ENDPOINT_SIGMA_PX,
                    confidence_level=_LINE_CAL_CONFIDENCE_LEVEL,
                )
                status_key = (
                    "info.cal_consistent"
                    if is_consistent
                    else "info.cal_inconsistent"
                )
                lines.append(
                    tr(
                        "info.cal_consistency",
                        chi2=chi2_value,
                        p=p_value,
                        conf=100.0 * _LINE_CAL_CONFIDENCE_LEVEL,
                        status=tr(status_key),
                    )
                )
            else:
                lines.append(
                    tr("info.cal_sigma_assumption", sigma=DEFAULT_LINE_ENDPOINT_SIGMA_PX)
                )

            diagnostics = line_calibration_diagnostics(
                cal.lines,
                sigma_per_pixel=DEFAULT_LINE_ENDPOINT_SIGMA_PX,
                robust=True,
            )
            self._cal_table.setVisible(True)
            self._cal_table.setHorizontalHeaderLabels([
                "#",
                tr("info.cal_line_px"),
                f"{tr('info.cal_line_known')} ({cal.unit.value})",
                f"PPU (px/{cal.unit.value})",
                tr("info.cal_line_sigma"),
                tr("info.cal_line_delta_pct"),
                tr("info.cal_line_source"),
            ])
            self._cal_table.setRowCount(len(diagnostics))
            for row, d in enumerate(diagnostics):
                self._cal_table.setItem(row, 0, QTableWidgetItem(str(d.index + 1)))
                self._cal_table.setItem(row, 1, QTableWidgetItem(f"{d.pixel_length:.2f}"))
                self._cal_table.setItem(row, 2, QTableWidgetItem(f"{d.known_length:.4g}"))
                self._cal_table.setItem(row, 3, QTableWidgetItem(f"{d.ppu:.4f}"))
                self._cal_table.setItem(row, 4, QTableWidgetItem(f"{d.ppu_sigma:.4f}"))
                self._cal_table.setItem(row, 5, QTableWidgetItem(f"{d.deviation_percent:+.3f}%"))
                source = getattr(cal.lines[d.index], "source", "manual") if d.index < len(cal.lines) else "manual"
                self._cal_table.setItem(row, 6, QTableWidgetItem(source))
        else:
            lines.append(tr("info.method_rect"))
            self._cal_table.setRowCount(0)
            self._cal_table.setVisible(False)

        if cal.homography:
            lines.append(tr("info.perspective"))

        self._cal_label.setText("<br>".join(lines))

    def _update_polygon(
        self,
        session: Session,
        idx: Optional[int],
        display_unit: Unit,
    ) -> None:
        if idx is None or idx >= len(session.polygons):
            self._poly_label.setText(
                f"<b>{tr('info.polygon')}</b><br>{tr('info.none_selected')}"
            )
            self._vtx_table.setRowCount(0)
            return

        poly = session.polygons[idx]
        result: Optional[MeasurementResult] = None
        for r in session.results:
            if r.polygon_label == poly.label:
                result = r
                break

        lines = [f"<b>{tr('info.polygon')}: {poly.label}</b>"]

        if result is not None:
            area = result.area_physical
            area_unc = result.area_uncertainty
            area_unc_v = result.area_uncertainty_vertex
            area_unc_ppu = result.area_uncertainty_ppu
            perim = result.perimeter_physical
            perim_unc = result.perimeter_uncertainty
            bbox_w = result.bbox_width_physical
            bbox_h = result.bbox_height_physical
            bbox_w_unc = result.bbox_width_uncertainty
            bbox_h_unc = result.bbox_height_uncertainty
            out_unit = result.unit

            if result.unit != display_unit:
                try:
                    area = convert_area(area, result.unit, display_unit)
                    area_unc = convert_area(area_unc, result.unit, display_unit)
                    area_unc_v = convert_area(area_unc_v, result.unit, display_unit)
                    area_unc_ppu = convert_area(area_unc_ppu, result.unit, display_unit)
                    perim = convert_length(perim, result.unit, display_unit)
                    perim_unc = convert_length(perim_unc, result.unit, display_unit)
                    bbox_w = convert_length(bbox_w, result.unit, display_unit)
                    bbox_h = convert_length(bbox_h, result.unit, display_unit)
                    bbox_w_unc = convert_length(bbox_w_unc, result.unit, display_unit)
                    bbox_h_unc = convert_length(bbox_h_unc, result.unit, display_unit)
                    out_unit = display_unit
                except ValueError:
                    pass

            rel_err = (area_unc / area * 100) if area > 0 else 0.0
            u = out_unit.value
            lines.extend([
                tr("info.area", area=area, unit=u, unc=area_unc, rel=rel_err),
                tr("info.unc_decomp", v=area_unc_v, p=area_unc_ppu, unit=u),
                tr("info.perimeter", perim=perim, unit=u, unc=perim_unc),
                tr(
                    "info.bbox",
                    w=bbox_w,
                    h=bbox_h,
                    unit=u,
                    wu=bbox_w_unc,
                    hu=bbox_h_unc,
                ),
                tr("info.bbox_angle", angle=result.bbox_angle_deg),
                tr("info.vertices_summary", n=result.n_vertices,
                   apx=result.area_pixels),
            ])
        else:
            lines.append(tr("info.not_measured"))

        self._poly_label.setText("<br>".join(lines))

        # Vertex table
        cal = session.calibration
        max_grad = self._app._get_max_grad()
        self._vtx_table.setRowCount(poly.n_vertices)
        for i, v in enumerate(poly.vertices):
            self._vtx_table.setItem(i, 0, QTableWidgetItem(str(i)))
            self._vtx_table.setItem(i, 1, QTableWidgetItem(f"{v.x:.2f}"))
            self._vtx_table.setItem(i, 2, QTableWidgetItem(f"{v.y:.2f}"))
            if cal.is_calibrated:
                px = v.x / cal.pixels_per_unit
                py = v.y / cal.pixels_per_unit
                try:
                    px = convert_length(px, cal.unit, display_unit)
                    py = convert_length(py, cal.unit, display_unit)
                except ValueError:
                    pass
                self._vtx_table.setItem(i, 3, QTableWidgetItem(f"{px:.4f}"))
                self._vtx_table.setItem(i, 4, QTableWidgetItem(f"{py:.4f}"))
            else:
                self._vtx_table.setItem(i, 3, QTableWidgetItem("-"))
                self._vtx_table.setItem(i, 4, QTableWidgetItem("-"))
            self._vtx_table.setItem(
                i, 5, QTableWidgetItem("\u2713" if v.snapped else "")
            )
            conf_str = f"{v.confidence:.1f}" if v.snapped else "-"
            self._vtx_table.setItem(i, 6, QTableWidgetItem(conf_str))
            sigma = confidence_to_sigma(v.confidence, max_grad, v.snapped)
            self._vtx_table.setItem(i, 7, QTableWidgetItem(f"{sigma:.3f}"))

    def _update_grid(self, session: Session, display_unit: Unit) -> None:
        if not session.grid_results:
            self._grid_label.setText(
                f"<b>{tr('info.grid')}</b><br>{tr('info.no_detections')}"
            )
            return

        lines = [f"<b>{tr('info.grid_detections')}</b>"]
        for i, g in enumerate(session.grid_results):
            pitch_x = g.pitch_x_physical
            pitch_y = g.pitch_y_physical
            out_unit = g.unit
            if g.unit != Unit.PIXEL and g.unit != display_unit:
                try:
                    pitch_x = convert_length(g.pitch_x_physical, g.unit, display_unit)
                    pitch_y = convert_length(g.pitch_y_physical, g.unit, display_unit)
                    out_unit = display_unit
                except ValueError:
                    pass
            u = out_unit.value
            lines.append(
                f"#{i+1}: pitch=({pitch_x:.3g}, {pitch_y:.3g}) {u}  "
                f"angle={g.angle_deg:.1f}\u00b0  "
                f"conf={g.confidence:.3f}"
            )
        self._grid_label.setText("<br>".join(lines))
