"""Tests for model.py — domain types and unit conversion."""

from __future__ import annotations

import math

import pytest

from diemeter.model import (
    Calibration,
    MeasurementResult,
    Polygon,
    Unit,
    Vertex,
    convert_area,
    convert_length,
    unit_from_str,
)


class TestMeasurementResultExtended:
    def test_has_uncertainty_decomposition(self):
        r = MeasurementResult(
            polygon_label="test",
            area_pixels=10000.0,
            area_physical=100.0,
            area_uncertainty=5.0,
            area_uncertainty_vertex=3.0,
            area_uncertainty_ppu=4.0,
            perimeter_physical=40.0,
            perimeter_uncertainty=0.5,
            unit=Unit.MILLIMETER,
            n_vertices=4,
        )
        assert r.area_uncertainty_vertex == 3.0
        assert r.area_uncertainty_ppu == 4.0
        assert r.perimeter_physical == 40.0
        assert r.perimeter_uncertainty == 0.5

    def test_defaults_backward_compatible(self):
        r = MeasurementResult(
            polygon_label="test",
            area_pixels=100.0,
            area_physical=10.0,
            area_uncertainty=1.0,
            unit=Unit.MILLIMETER,
            n_vertices=3,
        )
        assert r.area_uncertainty_vertex == 0.0
        assert r.area_uncertainty_ppu == 0.0
        assert r.perimeter_physical == 0.0
        assert r.perimeter_uncertainty == 0.0
