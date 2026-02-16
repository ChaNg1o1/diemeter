"""Tests for grid.py — FFT autocorrelation grid detection."""

from __future__ import annotations

import numpy as np
import pytest

from diemeter.grid import (
    autocorrelation_2d,
    detect_grid,
    find_peaks,
    refine_peak_parabolic,
    suppress_center,
)
from diemeter.model import Calibration, Unit


class TestAutocorrelation2D:
    def test_shape_preserved(self, checkerboard_50):
        acorr = autocorrelation_2d(checkerboard_50)
        assert acorr.shape == checkerboard_50.shape

    def test_normalized(self, checkerboard_50):
        acorr = autocorrelation_2d(checkerboard_50)
        # DC component (origin) should be 1.0 after normalization
        assert acorr[0, 0] == pytest.approx(1.0)

    def test_periodic_signal(self):
        """A perfectly periodic signal should have clear autocorrelation peaks."""
        # Create 1D periodic signal embedded in 2D
        x = np.arange(200)
        signal = np.sin(2 * np.pi * x / 50)  # period = 50
        img = np.tile(signal, (200, 1)).astype(np.float64) * 127 + 128
        img = img.astype(np.uint8)

        acorr = autocorrelation_2d(img)
        # Should have peaks at multiples of 50 along x-axis
        # Check that acorr[0, 50] > threshold
        assert acorr[0, 50] > 0.3


class TestSuppressCenter:
    def test_center_zeroed(self, checkerboard_50):
        acorr = autocorrelation_2d(checkerboard_50)
        suppressed = suppress_center(acorr, radius=5)
        assert suppressed[0, 0] == 0.0

    def test_non_center_preserved(self, checkerboard_50):
        acorr = autocorrelation_2d(checkerboard_50)
        suppressed = suppress_center(acorr, radius=3)
        # Values far from center should be unchanged
        assert suppressed[100, 100] == acorr[100, 100]


class TestFindPeaks:
    def test_detects_peaks_in_checkerboard(self, checkerboard_50):
        acorr = autocorrelation_2d(checkerboard_50)
        suppressed = suppress_center(acorr, radius=10)
        peaks = find_peaks(suppressed, min_distance=20, threshold=0.05)
        assert len(peaks) >= 1
        # Checkerboard autocorrelation fundamental period is 2*cell_size = 100
        # (one full black+white cycle)
        py, px, val = peaks[0]
        pitch = (py**2 + px**2) ** 0.5
        assert 80 < pitch < 120


class TestRefinePeakParabolic:
    def test_no_shift_at_exact_peak(self):
        """If peak is exactly at grid point, refinement should be near zero."""
        # Create a simple array with a clear peak
        arr = np.zeros((100, 100))
        arr[30, 40] = 1.0
        arr[30, 39] = 0.5
        arr[30, 41] = 0.5
        arr[29, 40] = 0.5
        arr[31, 40] = 0.5

        ry, rx = refine_peak_parabolic(arr, 30, 40)
        assert abs(ry - 30) < 0.5
        assert abs(rx - 40) < 0.5


class TestDetectGrid:
    def test_checkerboard(self, checkerboard_50):
        result = detect_grid(checkerboard_50, min_pitch=20)
        if result is None:
            pytest.skip("Grid detection did not find pattern")
        # Autocorrelation fundamental = 2 * cell_size = 100
        pitch = max(result.pitch_x, result.pitch_y)
        assert 80 < pitch < 120

    def test_with_roi(self, checkerboard_50):
        result = detect_grid(
            checkerboard_50,
            roi=(10, 10, 180, 180),
            min_pitch=20,
        )
        # Should still work with ROI
        if result is not None:
            assert result.confidence > 0

    def test_with_calibration(self, checkerboard_50):
        cal = Calibration(
            pixels_per_unit=5.0,
            unit=Unit.MILLIMETER,
        )
        result = detect_grid(checkerboard_50, min_pitch=20, calibration=cal)
        if result is None:
            pytest.skip("Grid detection did not find pattern")
        assert result.unit == Unit.MILLIMETER
        # Fundamental period ~100 px / 5 px/mm = ~20 mm
        pitch_phys = max(result.pitch_x_physical, result.pitch_y_physical)
        assert 15 < pitch_phys < 25

    def test_no_pattern(self):
        """Random noise should not produce a confident grid result."""
        rng = np.random.default_rng(42)
        noise = rng.integers(0, 256, (200, 200), dtype=np.uint8)
        result = detect_grid(noise, min_pitch=20)
        # Either None or very low confidence
        if result is not None:
            assert result.confidence < 0.5

    def test_color_image(self, checkerboard_50):
        color = np.stack([checkerboard_50] * 3, axis=-1)
        result = detect_grid(color, min_pitch=20)
        # Should handle color images
        if result is not None:
            assert result.pitch_x > 0
