"""FFT-based autocorrelation for repeating pattern (die grid) detection."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage

from .model import Calibration, GridResult


def autocorrelation_2d(image: np.ndarray) -> np.ndarray:
    """Compute 2D autocorrelation via FFT.

    acorr = ifft2(|fft2(img)|^2), normalized to [0, 1].
    """
    if len(image.shape) == 3:
        # Convert to grayscale via luminance
        gray = np.dot(image[..., :3], [0.114, 0.587, 0.299])
    else:
        gray = image.astype(np.float64)

    # Subtract mean to remove DC component
    gray = gray - gray.mean()

    f = np.fft.fft2(gray)
    power = np.abs(f) ** 2
    acorr = np.real(np.fft.ifft2(power))

    # Normalize
    acorr_max = acorr[0, 0]
    if acorr_max > 0:
        acorr /= acorr_max

    return acorr


def suppress_center(acorr: np.ndarray, radius: int = 10) -> np.ndarray:
    """Zero out the center peak of an autocorrelation map."""
    h, w = acorr.shape
    result = acorr.copy()
    cy, cx = 0, 0  # DC is at (0, 0) in fftshift-free layout
    # Suppress corners (due to FFT wrapping, peaks near edges are also center-equivalent)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                y = (cy + dy) % h
                x = (cx + dx) % w
                result[y, x] = 0.0
    return result


def find_peaks(
    acorr: np.ndarray,
    min_distance: int = 20,
    threshold: float = 0.1,
    max_peaks: int = 10,
) -> List[Tuple[int, int, float]]:
    """Find peaks in the autocorrelation map.

    Returns list of (y, x, value) sorted by value descending.
    """
    h, w = acorr.shape

    # Use scipy maximum_filter for local maxima detection
    neighborhood_size = max(3, min_distance)
    local_max = ndimage.maximum_filter(acorr, size=neighborhood_size)
    detected = (acorr == local_max) & (acorr > threshold)

    ys, xs = np.nonzero(detected)
    values = acorr[ys, xs]

    # Sort by value descending
    order = np.argsort(values)[::-1]
    ys, xs, values = ys[order], xs[order], values[order]

    # Non-maximum suppression by min_distance
    peaks: List[Tuple[int, int, float]] = []
    for y, x, v in zip(ys, xs, values):
        # Map to centered coordinates (handle FFT wrapping)
        cy = y if y < h // 2 else y - h
        cx = x if x < w // 2 else x - w
        # Skip origin
        if cy == 0 and cx == 0:
            continue
        too_close = False
        for py, px, _ in peaks:
            if (cy - py) ** 2 + (cx - px) ** 2 < min_distance**2:
                too_close = True
                break
        if not too_close:
            peaks.append((cy, cx, float(v)))
        if len(peaks) >= max_peaks:
            break

    return peaks


def refine_peak_parabolic(
    acorr: np.ndarray,
    y: int,
    x: int,
) -> Tuple[float, float]:
    """Refine peak position using 2D parabolic interpolation.

    Fits a parabola to the 3 points along each axis through the peak.
    """
    h, w = acorr.shape
    # Map centered coords back to array indices
    yi = y % h
    xi = x % w

    # Parabolic fit along x
    xm = acorr[yi, (xi - 1) % w]
    xc = acorr[yi, xi]
    xp = acorr[yi, (xi + 1) % w]
    denom_x = 2.0 * (xm - 2.0 * xc + xp)
    dx = (xm - xp) / denom_x if abs(denom_x) > 1e-10 else 0.0

    # Parabolic fit along y
    ym = acorr[(yi - 1) % h, xi]
    yc = acorr[yi, xi]
    yp = acorr[(yi + 1) % h, xi]
    denom_y = 2.0 * (ym - 2.0 * yc + yp)
    dy = (ym - yp) / denom_y if abs(denom_y) > 1e-10 else 0.0

    return float(y + dy), float(x + dx)


def detect_grid(
    image: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]] = None,
    min_pitch: int = 20,
    calibration: Optional[Calibration] = None,
) -> Optional[GridResult]:
    """Detect repeating pattern pitch using 2D FFT autocorrelation.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    roi : (x, y, w, h), optional
        Region of interest. If None, use full image.
    min_pitch : int
        Minimum expected pitch in pixels.
    calibration : Calibration, optional
        For converting to physical units.

    Returns
    -------
    GridResult or None
        Detected grid parameters, or None if no clear pattern found.
    """
    if roi is not None:
        rx, ry, rw, rh = roi
        crop = image[ry : ry + rh, rx : rx + rw]
    else:
        crop = image

    acorr = autocorrelation_2d(crop)
    acorr_suppressed = suppress_center(acorr, radius=max(5, min_pitch // 2))

    peaks = find_peaks(
        acorr_suppressed,
        min_distance=min_pitch,
        threshold=0.05,
        max_peaks=6,
    )

    if len(peaks) < 1:
        return None

    # The first peak corresponds to the primary pitch
    py, px, confidence = peaks[0]

    # Refine with parabolic interpolation
    py_ref, px_ref = refine_peak_parabolic(acorr_suppressed, py, px)

    pitch_x = abs(px_ref)
    pitch_y = abs(py_ref)
    angle = np.degrees(np.arctan2(py_ref, px_ref))

    # Convert to physical units if calibrated
    if calibration is not None and calibration.is_calibrated:
        ppu = calibration.pixels_per_unit
        pitch_x_phys = pitch_x / ppu
        pitch_y_phys = pitch_y / ppu
        unit = calibration.unit
    else:
        pitch_x_phys = pitch_x
        pitch_y_phys = pitch_y
        from .model import Unit
        unit = Unit.PIXEL

    return GridResult(
        pitch_x=pitch_x,
        pitch_y=pitch_y,
        pitch_x_physical=pitch_x_phys,
        pitch_y_physical=pitch_y_phys,
        angle_deg=float(angle),
        confidence=confidence,
        unit=unit,
    )
