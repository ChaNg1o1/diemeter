"""JSON session save/load for diemeter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .model import (
    Calibration,
    CalibrationLine,
    CalibrationRect,
    GridResult,
    MeasurementResult,
    Polygon,
    Session,
    Unit,
    Vertex,
)


def _vertex_to_dict(v: Vertex) -> Dict[str, Any]:
    return {
        "x": v.x,
        "y": v.y,
        "snapped": v.snapped,
        "confidence": v.confidence,
    }


def _vertex_from_dict(d: Dict[str, Any]) -> Vertex:
    return Vertex(
        x=d["x"],
        y=d["y"],
        snapped=d.get("snapped", False),
        confidence=d.get("confidence", 1.0),
    )


def _cal_line_to_dict(line: CalibrationLine) -> Dict[str, Any]:
    return {
        "p0": _vertex_to_dict(line.p0),
        "p1": _vertex_to_dict(line.p1),
        "known_length": line.known_length,
        "unit": line.unit.value,
    }


def _cal_line_from_dict(d: Dict[str, Any]) -> CalibrationLine:
    return CalibrationLine(
        p0=_vertex_from_dict(d["p0"]),
        p1=_vertex_from_dict(d["p1"]),
        known_length=d["known_length"],
        unit=Unit(d["unit"]),
    )


def _cal_rect_to_dict(rect: CalibrationRect) -> Dict[str, Any]:
    return {
        "corners": [_vertex_to_dict(c) for c in rect.corners],
        "known_width": rect.known_width,
        "known_height": rect.known_height,
        "unit": rect.unit.value,
    }


def _cal_rect_from_dict(d: Dict[str, Any]) -> CalibrationRect:
    return CalibrationRect(
        corners=[_vertex_from_dict(c) for c in d["corners"]],
        known_width=d["known_width"],
        known_height=d["known_height"],
        unit=Unit(d["unit"]),
    )


def _calibration_to_dict(cal: Calibration) -> Dict[str, Any]:
    return {
        "lines": [_cal_line_to_dict(l) for l in cal.lines],
        "rect": _cal_rect_to_dict(cal.rect) if cal.rect else None,
        "unit": cal.unit.value,
        "pixels_per_unit": cal.pixels_per_unit,
        "ppu_uncertainty": cal.ppu_uncertainty,
        "homography": cal.homography,
    }


def _calibration_from_dict(d: Dict[str, Any]) -> Calibration:
    return Calibration(
        lines=[_cal_line_from_dict(l) for l in d.get("lines", [])],
        rect=_cal_rect_from_dict(d["rect"]) if d.get("rect") else None,
        unit=Unit(d.get("unit", "mm")),
        pixels_per_unit=d.get("pixels_per_unit", 0.0),
        ppu_uncertainty=d.get("ppu_uncertainty", 0.0),
        homography=d.get("homography"),
    )


def _polygon_to_dict(poly: Polygon) -> Dict[str, Any]:
    return {
        "vertices": [_vertex_to_dict(v) for v in poly.vertices],
        "label": poly.label,
        "closed": poly.closed,
    }


def _polygon_from_dict(d: Dict[str, Any]) -> Polygon:
    return Polygon(
        vertices=[_vertex_from_dict(v) for v in d["vertices"]],
        label=d.get("label", ""),
        closed=d.get("closed", False),
    )


def _result_to_dict(r: MeasurementResult) -> Dict[str, Any]:
    return {
        "polygon_label": r.polygon_label,
        "area_pixels": r.area_pixels,
        "area_physical": r.area_physical,
        "area_uncertainty": r.area_uncertainty,
        "area_uncertainty_vertex": r.area_uncertainty_vertex,
        "area_uncertainty_ppu": r.area_uncertainty_ppu,
        "perimeter_physical": r.perimeter_physical,
        "perimeter_uncertainty": r.perimeter_uncertainty,
        "unit": r.unit.value,
        "n_vertices": r.n_vertices,
    }


def _result_from_dict(d: Dict[str, Any]) -> MeasurementResult:
    return MeasurementResult(
        polygon_label=d["polygon_label"],
        area_pixels=d["area_pixels"],
        area_physical=d["area_physical"],
        area_uncertainty=d["area_uncertainty"],
        area_uncertainty_vertex=d.get("area_uncertainty_vertex", 0.0),
        area_uncertainty_ppu=d.get("area_uncertainty_ppu", 0.0),
        perimeter_physical=d.get("perimeter_physical", 0.0),
        perimeter_uncertainty=d.get("perimeter_uncertainty", 0.0),
        unit=Unit(d["unit"]),
        n_vertices=d["n_vertices"],
    )


def _grid_result_to_dict(g: GridResult) -> Dict[str, Any]:
    return {
        "pitch_x": g.pitch_x,
        "pitch_y": g.pitch_y,
        "pitch_x_physical": g.pitch_x_physical,
        "pitch_y_physical": g.pitch_y_physical,
        "angle_deg": g.angle_deg,
        "confidence": g.confidence,
        "unit": g.unit.value,
        "origin_x": g.origin_x,
        "origin_y": g.origin_y,
    }


def _grid_result_from_dict(d: Dict[str, Any]) -> GridResult:
    return GridResult(
        pitch_x=d["pitch_x"],
        pitch_y=d["pitch_y"],
        pitch_x_physical=d["pitch_x_physical"],
        pitch_y_physical=d["pitch_y_physical"],
        angle_deg=d["angle_deg"],
        confidence=d["confidence"],
        unit=Unit(d.get("unit", "mm")),
        origin_x=d.get("origin_x", 0.0),
        origin_y=d.get("origin_y", 0.0),
    )


def session_to_dict(session: Session) -> Dict[str, Any]:
    """Serialize a Session to a JSON-compatible dict."""
    return {
        "version": "0.1.0",
        "image_path": session.image_path,
        "calibration": _calibration_to_dict(session.calibration),
        "polygons": [_polygon_to_dict(p) for p in session.polygons],
        "results": [_result_to_dict(r) for r in session.results],
        "grid_results": [_grid_result_to_dict(g) for g in session.grid_results],
        "edge_snap_enabled": session.edge_snap_enabled,
        "edge_snap_radius": session.edge_snap_radius,
    }


def session_from_dict(d: Dict[str, Any]) -> Session:
    """Deserialize a Session from a dict."""
    return Session(
        image_path=d.get("image_path", ""),
        calibration=_calibration_from_dict(d.get("calibration", {})),
        polygons=[_polygon_from_dict(p) for p in d.get("polygons", [])],
        results=[_result_from_dict(r) for r in d.get("results", [])],
        grid_results=[_grid_result_from_dict(g) for g in d.get("grid_results", [])],
        edge_snap_enabled=d.get("edge_snap_enabled", False),
        edge_snap_radius=d.get("edge_snap_radius", 15),
    )


def save_session(session: Session, path: str | Path) -> None:
    """Save session to a JSON file."""
    path = Path(path)
    data = session_to_dict(session)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_session(path: str | Path) -> Session:
    """Load session from a JSON file."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return session_from_dict(data)
