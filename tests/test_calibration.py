"""Tests for calibration.py — line/rect calibration and homography."""

from __future__ import annotations

import math

import numpy as np
import pytest

from diemeter.calibration import (
    calibrate_from_line,
    calibrate_from_pad_pitch,
    calibrate_from_rect,
    chi_squared_consistency,
    compute_homography,
    line_calibration_diagnostics,
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

    def test_zero_pixel_length_raises(self):
        with pytest.raises(ValueError, match="pixel length"):
            calibrate_from_line((10, 10), (10, 10), 5.0)


class TestWeightedAveragePPU:
    def test_single_line(self):
        line = calibrate_from_line((0, 0), (200, 0), 20.0)
        ppu, std = weighted_average_ppu([line])
        assert ppu == pytest.approx(10.0)
        # Single-line uncertainty from endpoint localization model:
        # sigma_ppu ~= ppu * sqrt(2) * sigma_px / L_px
        expected_std = 10.0 * math.sqrt(2.0) * 0.5 / 200.0
        assert std == pytest.approx(expected_std)

    def test_consistent_lines(self):
        l1 = calibrate_from_line((0, 0), (100, 0), 10.0)  # ppu=10
        l2 = calibrate_from_line((0, 0), (200, 0), 20.0)  # ppu=10
        ppu, std = weighted_average_ppu([l1, l2])
        assert ppu == pytest.approx(10.0)
        # Even perfectly consistent lines retain finite uncertainty from endpoint noise.
        assert std > 0.0
        # Inverse-variance mean uncertainty (chi2 inflation = 1 for perfect consistency):
        # sigmas: [0.0707107, 0.0353553], sem = sqrt(1/sum(1/sigma_i^2)).
        expected_sem = math.sqrt(1.0 / (200.0 + 800.0))
        assert std == pytest.approx(expected_sem)

    def test_inconsistent_lines(self):
        l1 = calibrate_from_line((0, 0), (100, 0), 10.0)   # ppu=10
        l2 = calibrate_from_line((0, 0), (200, 0), 18.0)   # ppu=11.11
        ppu, std = weighted_average_ppu([l1, l2])
        # Weighted mean should remain between the two line estimates.
        assert ppu > 10.0
        assert ppu < 11.11
        assert std > 0.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            weighted_average_ppu([])

    def test_invalid_zero_pixel_line_raises(self):
        bad = CalibrationLine(Vertex(1, 1), Vertex(1, 1), 10.0, Unit.MILLIMETER)
        with pytest.raises(ValueError, match="pixel length"):
            weighted_average_ppu([bad, calibrate_from_line((0, 0), (100, 0), 10.0)])

    def test_longer_line_weighted_more(self):
        # Short line: 10px for 1mm => ppu=10
        l_short = calibrate_from_line((0, 0), (10, 0), 1.0)
        # Long line: 210px for 21mm => ppu=10 (same)
        l_long = calibrate_from_line((0, 0), (210, 0), 21.0)
        ppu, _ = weighted_average_ppu([l_short, l_long])
        assert ppu == pytest.approx(10.0)

    def test_robust_weighting_reduces_outlier_impact(self):
        l1 = calibrate_from_line((0, 0), (100, 0), 10.0)   # ppu=10
        l2 = calibrate_from_line((0, 0), (200, 0), 20.0)   # ppu=10
        l3 = calibrate_from_line((0, 0), (150, 0), 10.0)   # ppu=15 outlier
        ppu_nonrob, _ = weighted_average_ppu([l1, l2, l3], robust=False)
        ppu_rob, _ = weighted_average_ppu([l1, l2, l3], robust=True)
        assert abs(ppu_rob - 10.0) < abs(ppu_nonrob - 10.0)

    def test_invalid_sigma_per_pixel_raises(self):
        l1 = calibrate_from_line((0, 0), (100, 0), 10.0)
        with pytest.raises(ValueError, match="sigma_per_pixel"):
            weighted_average_ppu([l1], sigma_per_pixel=0.0)


class TestUpdateCalibrationLines:
    def test_updates_ppu(self):
        cal = Calibration(unit=Unit.MILLIMETER)
        cal.lines.append(calibrate_from_line((0, 0), (100, 0), 10.0))
        cal.lines.append(calibrate_from_line((0, 0), (50, 0), 5.0))
        update_calibration_lines(cal)
        assert cal.is_calibrated
        assert cal.pixels_per_unit == pytest.approx(10.0)
        assert cal.ppu_uncertainty > 0.0


class TestLineCalibrationDiagnostics:
    def test_empty_returns_empty_list(self):
        assert line_calibration_diagnostics([]) == []

    def test_consistent_lines_have_small_deviation(self):
        lines = [
            calibrate_from_line((0, 0), (100, 0), 10.0),
            calibrate_from_line((0, 0), (200, 0), 20.0),
        ]
        diagnostics = line_calibration_diagnostics(lines)
        assert len(diagnostics) == 2
        for d in diagnostics:
            assert d.pixel_length > 0
            assert d.ppu > 0
            assert d.ppu_sigma > 0
            assert abs(d.deviation_percent) < 1e-6

    def test_outlier_line_shows_larger_deviation(self):
        lines = [
            calibrate_from_line((0, 0), (100, 0), 10.0),  # ppu=10
            calibrate_from_line((0, 0), (200, 0), 20.0),  # ppu=10
            calibrate_from_line((0, 0), (150, 0), 10.0),  # ppu=15 (outlier)
        ]
        diagnostics = line_calibration_diagnostics(lines, robust=True)
        deviations = [abs(d.deviation_percent) for d in diagnostics]
        assert deviations[2] == pytest.approx(max(deviations))

    def test_invalid_sigma_raises(self):
        lines = [calibrate_from_line((0, 0), (100, 0), 10.0)]
        with pytest.raises(ValueError, match="sigma_per_pixel"):
            line_calibration_diagnostics(lines, sigma_per_pixel=0.0)


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

    def test_invalid_dst_scale_raises(self):
        rect = calibrate_from_rect(
            corners=[(0, 0), (100, 0), (100, 80), (0, 80)],
            known_width=10.0,
            known_height=8.0,
        )
        with pytest.raises(ValueError, match="dst_scale"):
            compute_homography(rect, dst_scale=0.0)


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

    def test_invalid_sigma_corner_raises(self):
        cal = Calibration(unit=Unit.MILLIMETER)
        cal.rect = calibrate_from_rect(
            corners=[(0, 0), (100, 0), (100, 80), (0, 80)],
            known_width=10.0,
            known_height=8.0,
        )
        with pytest.raises(ValueError, match="sigma_corner"):
            update_calibration_rect(cal, sigma_corner=-0.1)

    def test_invalid_mc_samples_raises(self):
        cal = Calibration(unit=Unit.MILLIMETER)
        cal.rect = calibrate_from_rect(
            corners=[(0, 0), (100, 0), (100, 80), (0, 80)],
            known_width=10.0,
            known_height=8.0,
        )
        with pytest.raises(ValueError, match="n_mc_samples"):
            update_calibration_rect(cal, n_mc_samples=-1)


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

    def test_invalid_zero_pixel_line_raises(self):
        lines = [
            CalibrationLine(Vertex(0, 0), Vertex(0, 0), 10.0, Unit.MILLIMETER),
            CalibrationLine(Vertex(0, 0), Vertex(100, 0), 10.0, Unit.MILLIMETER),
        ]
        with pytest.raises(ValueError, match="pixel length"):
            chi_squared_consistency(lines)

    def test_invalid_sigma_per_pixel_raises(self):
        lines = [
            CalibrationLine(Vertex(0, 0), Vertex(100, 0), 10.0, Unit.MILLIMETER),
            CalibrationLine(Vertex(0, 0), Vertex(0, 100), 10.0, Unit.MILLIMETER),
        ]
        with pytest.raises(ValueError, match="sigma_per_pixel"):
            chi_squared_consistency(lines, sigma_per_pixel=0.0)


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


class TestPerLineSigma:
    """Test that per-line endpoint_sigma is respected in weighted averaging."""

    def test_pad_pitch_line_gets_lower_sigma(self):
        """A pad_pitch line with lower endpoint_sigma should get smaller ppu uncertainty."""
        manual_line = calibrate_from_line((0, 0), (100, 0), 10.0)
        assert manual_line.endpoint_sigma == 0.5

        pad_line = CalibrationLine(
            Vertex(0, 0), Vertex(100, 0), 10.0, Unit.MILLIMETER,
            endpoint_sigma=0.2, source="pad_pitch",
        )
        _, std_manual = weighted_average_ppu([manual_line])
        _, std_pad = weighted_average_ppu([pad_line])
        # Lower endpoint sigma should yield lower ppu uncertainty.
        assert std_pad < std_manual

    def test_mixed_lines_weighted_correctly(self):
        """A pad_pitch line (sigma=0.2) mixed with a manual line (sigma=0.5)
        should weight the pad_pitch line more heavily."""
        manual = CalibrationLine(
            Vertex(0, 0), Vertex(100, 0), 10.0, Unit.MILLIMETER,
            endpoint_sigma=0.5, source="manual",
        )  # ppu=10
        pad = CalibrationLine(
            Vertex(0, 0), Vertex(110, 0), 10.0, Unit.MILLIMETER,
            endpoint_sigma=0.2, source="pad_pitch",
        )  # ppu=11

        ppu, _ = weighted_average_ppu([manual, pad])
        # Weighted mean should be pulled toward pad line due to its lower sigma.
        assert ppu > 10.5

    def test_update_calibration_lines_with_mixed_sources(self):
        """update_calibration_lines should work with mixed manual/pad_pitch lines."""
        cal = Calibration(unit=Unit.MILLIMETER)
        cal.lines.append(CalibrationLine(
            Vertex(0, 0), Vertex(100, 0), 10.0, Unit.MILLIMETER,
            endpoint_sigma=0.5, source="manual",
        ))
        cal.lines.append(CalibrationLine(
            Vertex(0, 0), Vertex(100, 0), 10.0, Unit.MILLIMETER,
            endpoint_sigma=0.2, source="pad_pitch",
        ))
        update_calibration_lines(cal)
        assert cal.is_calibrated
        assert cal.pixels_per_unit == pytest.approx(10.0)


class TestCalibrateFromPadPitch:
    """Test calibrate_from_pad_pitch with synthetic periodic images."""

    def _make_periodic_image(self, pitch_px: int = 50, size: int = 400) -> np.ndarray:
        """Create a synthetic image with vertical stripes at a given pitch."""
        img = np.zeros((size, size), dtype=np.uint8)
        for col in range(size):
            if (col % pitch_px) < (pitch_px // 4):
                img[:, col] = 255
        return img

    def test_detects_pitch_and_creates_line(self):
        pitch_px = 50
        img = self._make_periodic_image(pitch_px=pitch_px)
        roi = (0, 0, 400, 400)
        result = calibrate_from_pad_pitch(img, roi, known_pitch=80.0, unit=Unit.MICROMETER)
        assert result is not None
        line, grid_res = result

        assert line.source == "pad_pitch"
        assert line.endpoint_sigma == pytest.approx(0.2)
        assert line.unit == Unit.MICROMETER
        assert line.known_length == pytest.approx(80.0)
        # pixel_length should be close to pitch_px
        assert line.pixel_length == pytest.approx(pitch_px, abs=2.0)
        # GridResult should have physical units filled in
        assert grid_res.unit == Unit.MICROMETER

    def test_returns_none_for_no_pattern(self):
        # Uniform gray image should produce no pattern.
        img = np.full((200, 200), 128, dtype=np.uint8)
        roi = (0, 0, 200, 200)
        result = calibrate_from_pad_pitch(img, roi, known_pitch=80.0, unit=Unit.MICROMETER)
        assert result is None

    def test_invalid_known_pitch_raises(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError, match="known_pitch"):
            calibrate_from_pad_pitch(img, (0, 0, 100, 100), known_pitch=0.0)

    def test_invalid_endpoint_sigma_raises(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError, match="endpoint_sigma"):
            calibrate_from_pad_pitch(
                img, (0, 0, 100, 100), known_pitch=80.0, endpoint_sigma=0.0,
            )
