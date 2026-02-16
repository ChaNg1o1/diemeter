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


class TestUnitFromStr:
    def test_basic_units(self):
        assert unit_from_str("mm") == Unit.MILLIMETER
        assert unit_from_str("um") == Unit.MICROMETER
        assert unit_from_str("\u03bcm") == Unit.MICROMETER
        assert unit_from_str("cm") == Unit.CENTIMETER
        assert unit_from_str("m") == Unit.METER
        assert unit_from_str("in") == Unit.INCH
        assert unit_from_str("mil") == Unit.MIL
        assert unit_from_str("px") == Unit.PIXEL

    def test_case_insensitive(self):
        assert unit_from_str("MM") == Unit.MILLIMETER
        assert unit_from_str("Mm") == Unit.MILLIMETER

    def test_with_whitespace(self):
        assert unit_from_str("  mm  ") == Unit.MILLIMETER

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown unit"):
            unit_from_str("furlongs")


class TestConvertLength:
    def test_identity(self):
        assert convert_length(5.0, Unit.MILLIMETER, Unit.MILLIMETER) == 5.0

    def test_mm_to_cm(self):
        assert convert_length(10.0, Unit.MILLIMETER, Unit.CENTIMETER) == pytest.approx(1.0)

    def test_inch_to_mm(self):
        assert convert_length(1.0, Unit.INCH, Unit.MILLIMETER) == pytest.approx(25.4)

    def test_pixel_raises(self):
        with pytest.raises(ValueError, match="pixels"):
            convert_length(1.0, Unit.PIXEL, Unit.MILLIMETER)


class TestConvertArea:
    def test_identity(self):
        assert convert_area(5.0, Unit.MILLIMETER, Unit.MILLIMETER) == 5.0

    def test_mm2_to_cm2(self):
        assert convert_area(100.0, Unit.MILLIMETER, Unit.CENTIMETER) == pytest.approx(1.0)

    def test_pixel_raises(self):
        with pytest.raises(ValueError, match="pixels"):
            convert_area(1.0, Unit.PIXEL, Unit.MILLIMETER)
