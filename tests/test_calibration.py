"""Tests for calibration.py — line/rect calibration and homography."""

from __future__ import annotations

import math

import numpy as np
import pytest

from diemeter.calibration import (
    calibrate_from_line,
    calibrate_from_rect,
    chi_squared_consistency,
    compute_homography,
    reprojection_error,
    update_calibration_lines,
    update_calibration_rect,
    warp_point,
    weighted_average_ppu,
)
from diemeter.model import Calibration, CalibrationLine, Unit, Vertex


class TestCalibrateFromLine:
    def test_horizontal(self):
        line = calibrate_from_line((0, 0), (100, 0), 10.0, Unit.MILLIMETER)
        assert line.pixel_length == pytest.approx(100.0)
        assert line.pixels_per_unit == pytest.approx(10.0)

    def test_diagonal(self):
        line = calibrate_from_line((0, 0), (30, 40), 5.0)
        assert line.pixel_length == pytest.approx(50.0)
        assert line.pixels_per_unit == pytest.approx(10.0)

    def test_zero_length_raises(self):
        with pytest.raises(ValueError, match="positive"):
            calibrate_from_line((0, 0), (10, 0), 0.0)


class TestWeightedAveragePPU:
    def test_single_line(self):
        line = calibrate_from_line((0, 0), (200, 0), 20.0)
        ppu, std = weighted_average_ppu([line])
        assert ppu == pytest.approx(10.0)
        assert std == 0.0

    def test_consistent_lines(self):
        l1 = calibrate_from_line((0, 0), (100, 0), 10.0)  # ppu=10
        l2 = calibrate_from_line((0, 0), (200, 0), 20.0)  # ppu=10
        ppu, std = weighted_average_ppu([l1, l2])
        assert ppu == pytest.approx(10.0)
        assert std == pytest.approx(0.0, abs=1e-10)

    def test_inconsistent_lines(self):
        l1 = calibrate_from_line((0, 0), (100, 0), 10.0)   # ppu=10
        l2 = calibrate_from_line((0, 0), (200, 0), 18.0)   # ppu=11.11
        ppu, std = weighted_average_ppu([l1, l2])
        # Weighted average with weights 100 and 200
        # ppu = (100*10 + 200*11.11) / 300 = 10.74
        assert ppu > 10.0
        assert ppu < 11.11
        assert std > 0.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            weighted_average_ppu([])

    def test_longer_line_weighted_more(self):
        # Short line: 10px for 1mm => ppu=10
        l_short = calibrate_from_line((0, 0), (10, 0), 1.0)
        # Long line: 210px for 21mm => ppu=10 (same)
        l_long = calibrate_from_line((0, 0), (210, 0), 21.0)
        ppu, _ = weighted_average_ppu([l_short, l_long])
        assert ppu == pytest.approx(10.0)


class TestUpdateCalibrationLines:
    def test_updates_ppu(self):
        cal = Calibration(unit=Unit.MILLIMETER)
        cal.lines.append(calibrate_from_line((0, 0), (100, 0), 10.0))
        cal.lines.append(calibrate_from_line((0, 0), (50, 0), 5.0))
        update_calibration_lines(cal)
        assert cal.is_calibrated
        assert cal.pixels_per_unit == pytest.approx(10.0)


class TestRectCalibration:
    def test_basic_rect(self):
        rect = calibrate_from_rect(
            corners=[(0, 0), (100, 0), (100, 80), (0, 80)],
            known_width=10.0,
            known_height=8.0,
        )
        assert len(rect.corners) == 4
        assert rect.known_width == 10.0
        assert rect.known_height == 8.0

    def test_wrong_corner_count(self):
        with pytest.raises(ValueError, match="4 corners"):
            calibrate_from_rect(
                corners=[(0, 0), (100, 0), (100, 80)],
                known_width=10.0,
                known_height=8.0,
            )

    def test_homography_preserves_corners(self):
        rect = calibrate_from_rect(
            corners=[(10, 10), (110, 15), (105, 85), (5, 80)],
            known_width=10.0,
            known_height=8.0,
        )
        H, (dst_w, dst_h), ppu = compute_homography(rect, dst_scale=10.0)
        assert dst_w == 100
        assert dst_h == 80
        assert ppu == pytest.approx(10.0)

        # Warp corners and check they land at expected positions
        tl = warp_point((10, 10), H)
        tr = warp_point((110, 15), H)
        assert tl[0] == pytest.approx(0.0, abs=0.5)
        assert tl[1] == pytest.approx(0.0, abs=0.5)
        assert tr[0] == pytest.approx(100.0, abs=0.5)
        assert tr[1] == pytest.approx(0.0, abs=0.5)

    def test_reprojection_error_small(self):
        # Non-perspective rect -> should have near-zero reproj error
        rect = calibrate_from_rect(
            corners=[(0, 0), (100, 0), (100, 80), (0, 80)],
            known_width=10.0,
            known_height=8.0,
        )
        H, _, _ = compute_homography(rect, dst_scale=10.0)
        err = reprojection_error(rect, H, dst_scale=10.0)
        assert err < 1.0


class TestUpdateCalibrationRect:
    def test_updates_ppu_without_image(self):
        cal = Calibration(unit=Unit.MILLIMETER)
        cal.rect = calibrate_from_rect(
            corners=[(0, 0), (100, 0), (100, 80), (0, 80)],
            known_width=10.0,
            known_height=8.0,
        )
        cal, warped = update_calibration_rect(cal, image=None, dst_scale=10.0)
        assert cal.is_calibrated
        assert cal.pixels_per_unit == pytest.approx(10.0)
        assert warped is None


class TestChiSquaredConsistency:
    def test_consistent_lines(self):
        """Lines with similar ppu should pass."""
        lines = [
            CalibrationLine(Vertex(0, 0), Vertex(100, 0), 10.0, Unit.MILLIMETER),
            CalibrationLine(Vertex(0, 0), Vertex(0, 101), 10.1, Unit.MILLIMETER),
        ]
        chi2, p_value, is_consistent = chi_squared_consistency(lines)
        assert chi2 >= 0
        assert 0 <= p_value <= 1
        assert is_consistent is True

    def test_inconsistent_lines(self):
        """Lines with very different ppu should fail."""
        lines = [
            CalibrationLine(Vertex(0, 0), Vertex(100, 0), 10.0, Unit.MILLIMETER),
            CalibrationLine(Vertex(0, 0), Vertex(100, 0), 20.0, Unit.MILLIMETER),
        ]
        chi2, p_value, is_consistent = chi_squared_consistency(lines)
        assert is_consistent is False

    def test_single_line(self):
        """Single line should always be consistent."""
        lines = [
            CalibrationLine(Vertex(0, 0), Vertex(100, 0), 10.0, Unit.MILLIMETER),
        ]
        chi2, p_value, is_consistent = chi_squared_consistency(lines)
        assert chi2 == 0.0
        assert is_consistent is True


class TestRectCalibrationUncertainty:
    def test_uncertainty_from_corner_localization(self):
        """Rect calibration uncertainty should be based on corner sigma, not reprojection error."""
        corners = [(80, 60), (220, 70), (210, 230), (90, 220)]
        rect = calibrate_from_rect(corners, 10.0, 8.0, Unit.MILLIMETER)
        cal = Calibration(rect=rect, unit=Unit.MILLIMETER)
        cal, _ = update_calibration_rect(cal, dst_scale=10.0)
        # With sigma_corner=0.5 default, uncertainty should be positive but small
        assert cal.ppu_uncertainty > 0
        # Should be proportional to corner uncertainty, not zero-ish reprojection error
        assert cal.ppu_uncertainty > 0.001
