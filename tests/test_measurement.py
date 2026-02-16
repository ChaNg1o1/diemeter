"""Tests for measurement.py — Shoelace area and measurement pipeline."""

from __future__ import annotations

import math

import pytest

from diemeter.measurement import (
    measure_polygon,
    polygon_area_pixels,
    polygon_perimeter_pixels,
    shoelace_area,
)
from diemeter.model import Calibration, Polygon, Unit, Vertex


class TestShoelaceArea:
    def test_unit_square(self):
        verts = [(0, 0), (1, 0), (1, 1), (0, 1)]
        assert shoelace_area(verts) == pytest.approx(1.0)

    def test_right_triangle(self):
        # 3-4-5 triangle, area = 6
        verts = [(0, 0), (3, 0), (0, 4)]
        assert shoelace_area(verts) == pytest.approx(6.0)

    def test_large_square(self):
        verts = [(50, 50), (150, 50), (150, 150), (50, 150)]
        assert shoelace_area(verts) == pytest.approx(10000.0)

    def test_clockwise_same_as_ccw(self):
        ccw = [(0, 0), (4, 0), (4, 3), (0, 3)]
        cw = list(reversed(ccw))
        assert shoelace_area(ccw) == pytest.approx(shoelace_area(cw))

    def test_degenerate_line(self):
        verts = [(0, 0), (5, 5)]
        assert shoelace_area(verts) == 0.0

    def test_single_point(self):
        assert shoelace_area([(1, 2)]) == 0.0

    def test_empty(self):
        assert shoelace_area([]) == 0.0

    def test_regular_hexagon(self):
        # Regular hexagon with circumradius R
        R = 10.0
        verts = [
            (R * math.cos(math.radians(60 * i)), R * math.sin(math.radians(60 * i)))
            for i in range(6)
        ]
        expected = 1.5 * math.sqrt(3) * R * R
        assert shoelace_area(verts) == pytest.approx(expected, rel=1e-10)


class TestPolygonAreaPixels:
    def test_basic(self):
        poly = Polygon(
            vertices=[
                Vertex(x=0, y=0),
                Vertex(x=100, y=0),
                Vertex(x=100, y=100),
                Vertex(x=0, y=100),
            ],
            closed=True,
        )
        assert polygon_area_pixels(poly) == pytest.approx(10000.0)

    def test_fewer_than_3(self):
        poly = Polygon(vertices=[Vertex(x=0, y=0), Vertex(x=1, y=1)])
        assert polygon_area_pixels(poly) == 0.0


class TestPolygonPerimeter:
    def test_unit_square(self):
        poly = Polygon(
            vertices=[
                Vertex(x=0, y=0),
                Vertex(x=1, y=0),
                Vertex(x=1, y=1),
                Vertex(x=0, y=1),
            ],
            closed=True,
        )
        assert polygon_perimeter_pixels(poly) == pytest.approx(4.0)


class TestMeasurePolygon:
    def test_with_calibration(self):
        poly = Polygon(
            vertices=[
                Vertex(x=0, y=0),
                Vertex(x=100, y=0),
                Vertex(x=100, y=100),
                Vertex(x=0, y=100),
            ],
            label="test",
            closed=True,
        )
        cal = Calibration(
            pixels_per_unit=10.0,  # 10 px/mm
            unit=Unit.MILLIMETER,
        )
        result = measure_polygon(poly, cal)
        # 10000 px^2 / (10 px/mm)^2 = 100 mm^2
        assert result.area_pixels == pytest.approx(10000.0)
        assert result.area_physical == pytest.approx(100.0)
        assert result.unit == Unit.MILLIMETER

    def test_unit_conversion(self):
        poly = Polygon(
            vertices=[
                Vertex(x=0, y=0),
                Vertex(x=100, y=0),
                Vertex(x=100, y=100),
                Vertex(x=0, y=100),
            ],
            label="test",
            closed=True,
        )
        cal = Calibration(
            pixels_per_unit=10.0,
            unit=Unit.MILLIMETER,
        )
        # Convert to cm: 100 mm^2 = 1 cm^2
        result = measure_polygon(poly, cal, target_unit=Unit.CENTIMETER)
        assert result.area_physical == pytest.approx(1.0)
        assert result.unit == Unit.CENTIMETER

    def test_uncalibrated_raises(self):
        poly = Polygon(
            vertices=[Vertex(x=0, y=0), Vertex(x=1, y=0), Vertex(x=0, y=1)],
            closed=True,
        )
        cal = Calibration()
        with pytest.raises(ValueError, match="not been performed"):
            measure_polygon(poly, cal)
