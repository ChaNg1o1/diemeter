"""Integration tests -- end-to-end measurement with known ground truth."""

from __future__ import annotations

import math

import numpy as np
import pytest

from diemeter.calibration import calibrate_from_line, update_calibration_lines
from diemeter.measurement import measure_polygon
from diemeter.model import Calibration, Polygon, Unit, Vertex


class TestEndToEndMeasurement:
    def test_known_square_area(self):
        """Calibrate with known line, measure known square, verify area."""
        # Setup: 100px line = 10mm -> ppu = 10 px/mm
        cal = Calibration(unit=Unit.MILLIMETER)
        line = calibrate_from_line((0, 0), (100, 0), 10.0, Unit.MILLIMETER)
        cal.lines.append(line)
        update_calibration_lines(cal)

        assert cal.pixels_per_unit == pytest.approx(10.0)

        # Measure a 200x200 px square -> should be 20x20 mm = 400 mm^2
        poly = Polygon(
            vertices=[
                Vertex(x=0, y=0),
                Vertex(x=200, y=0),
                Vertex(x=200, y=200),
                Vertex(x=0, y=200),
            ],
            label="square",
            closed=True,
        )
        result = measure_polygon(poly, cal)
        assert result.area_physical == pytest.approx(400.0)
        assert result.perimeter_physical == pytest.approx(80.0)
        assert result.unit == Unit.MILLIMETER

    def test_uncertainty_increases_with_vertex_noise(self):
        """Snapped vertices (high confidence) should have lower uncertainty than manual."""
        cal = Calibration(
            pixels_per_unit=10.0,
            ppu_uncertainty=0.1,
            unit=Unit.MILLIMETER,
        )

        # Manual vertices (no snap)
        poly_manual = Polygon(
            vertices=[
                Vertex(x=0, y=0, snapped=False),
                Vertex(x=200, y=0, snapped=False),
                Vertex(x=200, y=200, snapped=False),
                Vertex(x=0, y=200, snapped=False),
            ],
            label="manual",
            closed=True,
        )

        # Snapped vertices (high confidence)
        poly_snapped = Polygon(
            vertices=[
                Vertex(x=0, y=0, snapped=True, confidence=200.0),
                Vertex(x=200, y=0, snapped=True, confidence=200.0),
                Vertex(x=200, y=200, snapped=True, confidence=200.0),
                Vertex(x=0, y=200, snapped=True, confidence=200.0),
            ],
            label="snapped",
            closed=True,
        )

        r_manual = measure_polygon(poly_manual, cal)
        r_snapped = measure_polygon(poly_snapped, cal)

        # Same area
        assert r_manual.area_physical == pytest.approx(r_snapped.area_physical)
        # Snapped should have lower vertex uncertainty
        assert r_snapped.area_uncertainty_vertex < r_manual.area_uncertainty_vertex

    def test_multi_line_calibration_consistency(self):
        """Multiple calibration lines with consistent ratios should yield
        a tight PPU estimate with near-zero uncertainty.

        Note: The plan originally used chi_squared_consistency which does not
        exist in the current codebase. Instead we verify that the weighted
        average PPU is accurate and the standard deviation (ppu_uncertainty)
        is effectively zero when all lines agree perfectly.
        """
        cal = Calibration(unit=Unit.MILLIMETER)
        # Three lines all giving exactly 10 px/mm
        for length_px, length_mm in [(100, 10.0), (200, 20.0), (150, 15.0)]:
            line = calibrate_from_line(
                (0, 0), (length_px, 0), length_mm, Unit.MILLIMETER
            )
            cal.lines.append(line)
        update_calibration_lines(cal)

        assert cal.pixels_per_unit == pytest.approx(10.0, rel=0.01)
        # All lines agree perfectly, so weighted std should be ~0
        assert cal.ppu_uncertainty == pytest.approx(0.0, abs=1e-10)
