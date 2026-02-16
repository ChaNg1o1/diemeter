"""Tests for session.py — serialization roundtrips."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from diemeter.model import (
    Calibration,
    CalibrationLine,
    MeasurementResult,
    Polygon,
    Session,
    Unit,
    Vertex,
)
from diemeter.session import (
    load_session,
    save_session,
    session_from_dict,
    session_to_dict,
)


class TestSessionRoundtrip:
    def test_empty_session(self):
        s = Session()
        d = session_to_dict(s)
        s2 = session_from_dict(d)
        assert s2.image_path == ""
        assert not s2.calibration.is_calibrated
        assert s2.polygons == []

    def test_full_session(self):
        s = Session(
            image_path="test.png",
            calibration=Calibration(
                lines=[CalibrationLine(
                    Vertex(0, 0), Vertex(100, 0), 10.0, Unit.MILLIMETER
                )],
                pixels_per_unit=10.0,
                ppu_uncertainty=0.1,
                unit=Unit.MILLIMETER,
            ),
            polygons=[Polygon(
                vertices=[Vertex(0, 0), Vertex(10, 0), Vertex(10, 10), Vertex(0, 10)],
                label="poly_1",
                closed=True,
            )],
            results=[MeasurementResult(
                polygon_label="poly_1",
                area_pixels=100.0,
                area_physical=1.0,
                area_uncertainty=0.1,
                area_uncertainty_vertex=0.05,
                area_uncertainty_ppu=0.087,
                perimeter_physical=4.0,
                perimeter_uncertainty=0.02,
                unit=Unit.MILLIMETER,
                n_vertices=4,
            )],
        )
        d = session_to_dict(s)
        s2 = session_from_dict(d)
        assert s2.image_path == "test.png"
        assert s2.calibration.pixels_per_unit == 10.0
        assert s2.results[0].area_uncertainty_vertex == 0.05
        assert s2.results[0].perimeter_physical == 4.0

    def test_save_load_file(self):
        s = Session(image_path="test.png")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        save_session(s, path)
        s2 = load_session(path)
        assert s2.image_path == "test.png"
        Path(path).unlink()

    def test_backward_compatible_load(self):
        """Old session without new fields should load with defaults."""
        old_data = {
            "version": "0.1.0",
            "image_path": "test.png",
            "calibration": {"unit": "mm", "pixels_per_unit": 10.0, "ppu_uncertainty": 0.1},
            "polygons": [],
            "results": [{
                "polygon_label": "poly_1",
                "area_pixels": 100.0,
                "area_physical": 1.0,
                "area_uncertainty": 0.1,
                "unit": "mm",
                "n_vertices": 4,
            }],
            "grid_results": [],
        }
        s = session_from_dict(old_data)
        assert s.results[0].area_uncertainty_vertex == 0.0
        assert s.results[0].perimeter_physical == 0.0
