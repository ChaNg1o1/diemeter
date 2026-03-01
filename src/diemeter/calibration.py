"""Scale calibration and perspective correction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.stats import chi2 as chi2_dist

from .model import (
    Calibration,
    CalibrationLine,
    CalibrationRect,
    GridResult,
    Unit,
    Vertex,
)

DEFAULT_LINE_ENDPOINT_SIGMA_PX = 0.5
DEFAULT_LINE_HUBER_K = 1.5
DEFAULT_LINE_IRLS_MAX_ITERS = 25
DEFAULT_LINE_IRLS_TOL = 1e-9


@dataclass(frozen=True)
class LineCalibrationDiagnostic:
    """Per-reference diagnostics for multi-line scale calibration."""

    index: int
    pixel_length: float
    known_length: float
    unit: Unit
    ppu: float
    ppu_sigma: float
    deviation_percent: float


def calibrate_from_line(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    known_length: float,
    unit: Unit = Unit.MILLIMETER,
) -> CalibrationLine:
    """Create a CalibrationLine from two pixel coordinates and a known length."""
    try:
        known_length = float(known_length)
    except (TypeError, ValueError):
        raise ValueError("known_length must be positive") from None
    if not np.isfinite(known_length) or known_length <= 0:
        raise ValueError("known_length must be positive")
    try:
        x0 = float(p0[0])
        y0 = float(p0[1])
        x1 = float(p1[0])
        y1 = float(p1[1])
    except (TypeError, IndexError, ValueError):
        raise ValueError("p0 and p1 must be 2D numeric coordinates") from None

    pixel_length = float(np.hypot(x1 - x0, y1 - y0))
    if not np.isfinite(pixel_length) or pixel_length <= 0:
        raise ValueError("Calibration line pixel length must be positive")

    return CalibrationLine(
        p0=Vertex(x=x0, y=y0),
        p1=Vertex(x=x1, y=y1),
        known_length=known_length,
        unit=unit,
    )


def _validated_ppu_inputs(
    lines: List[CalibrationLine],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (pixel_lengths, ppus) after validating calibration lines."""
    pixel_lengths = []
    ppus = []
    for idx, line in enumerate(lines):
        known_length = float(line.known_length)
        if not np.isfinite(known_length) or known_length <= 0:
            raise ValueError(
                f"Calibration line {idx} has non-positive known_length: {line.known_length!r}"
            )

        dx = float(line.p1.x - line.p0.x)
        dy = float(line.p1.y - line.p0.y)
        pixel_length = float(np.hypot(dx, dy))
        if not np.isfinite(pixel_length) or pixel_length <= 0:
            raise ValueError(
                f"Calibration line {idx} has non-positive pixel length"
            )

        ppu = pixel_length / known_length
        if not np.isfinite(ppu) or ppu <= 0:
            raise ValueError(
                f"Calibration line {idx} produced invalid pixels-per-unit value"
            )

        pixel_lengths.append(pixel_length)
        ppus.append(ppu)

    return (
        np.asarray(pixel_lengths, dtype=np.float64),
        np.asarray(ppus, dtype=np.float64),
    )


def _line_ppu_sigmas(
    pixel_lengths: np.ndarray,
    ppus: np.ndarray,
    lines: List[CalibrationLine],
    sigma_per_pixel: float,
) -> np.ndarray:
    """Return per-line ppu sigma from endpoint localization uncertainty.

    Uses each line's own endpoint_sigma when available, falling back to
    the global sigma_per_pixel parameter.
    """
    # ppu = L_px / L_phys, with sigma(L_px) ~= sqrt(2) * sigma_endpoint.
    # Therefore sigma(ppu) ~= ppu * sqrt(2) * sigma_endpoint / L_px.
    sigmas_ep = np.array(
        [getattr(line, "endpoint_sigma", sigma_per_pixel) for line in lines],
        dtype=np.float64,
    )
    return ppus * np.sqrt(2.0) * sigmas_ep / pixel_lengths


def weighted_average_ppu(
    lines: List[CalibrationLine],
    sigma_per_pixel: float = DEFAULT_LINE_ENDPOINT_SIGMA_PX,
    robust: bool = True,
    huber_k: float = DEFAULT_LINE_HUBER_K,
    max_irls_iters: int = DEFAULT_LINE_IRLS_MAX_ITERS,
    irls_tol: float = DEFAULT_LINE_IRLS_TOL,
) -> Tuple[float, float]:
    """Compute weighted-average pixels-per-unit from multiple calibration lines.

    The estimator uses inverse-variance weighting where each line's uncertainty
    is derived from endpoint localization uncertainty in pixels.
    Optionally applies robust Huber IRLS weighting to reduce outlier impact.

    Returns
    -------
    (ppu, ppu_uncertainty) : tuple of float
        The weighted mean ppu and estimated uncertainty (1 sigma).
    """
    try:
        sigma_per_pixel = float(sigma_per_pixel)
        huber_k = float(huber_k)
        max_irls_iters_float = float(max_irls_iters)
        irls_tol = float(irls_tol)
    except (TypeError, ValueError):
        raise ValueError(
            "sigma_per_pixel, huber_k, max_irls_iters, and irls_tol must be numeric"
        ) from None
    if not np.isfinite(sigma_per_pixel) or sigma_per_pixel <= 0:
        raise ValueError("sigma_per_pixel must be positive")
    if not np.isfinite(huber_k) or huber_k <= 0:
        raise ValueError("huber_k must be positive")
    if not np.isfinite(max_irls_iters_float) or not max_irls_iters_float.is_integer():
        raise ValueError("max_irls_iters must be a positive integer")
    max_irls_iters = int(max_irls_iters_float)
    if max_irls_iters <= 0:
        raise ValueError("max_irls_iters must be a positive integer")
    if not np.isfinite(irls_tol) or irls_tol <= 0:
        raise ValueError("irls_tol must be positive")

    if not lines:
        raise ValueError("No calibration lines provided")

    pixel_lengths, ppus = _validated_ppu_inputs(lines)
    sigmas = _line_ppu_sigmas(pixel_lengths, ppus, lines, sigma_per_pixel)
    if not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0):
        raise ValueError("Calibration lines produced invalid uncertainty values")

    # Single-line calibration still has non-zero uncertainty from endpoint noise.
    if len(lines) == 1:
        return float(ppus[0]), float(sigmas[0])

    inv_var = 1.0 / (sigmas**2)
    if not np.all(np.isfinite(inv_var)) or np.any(inv_var <= 0):
        raise ValueError("Calibration lines produced invalid inverse-variance weights")

    w_sum = float(np.sum(inv_var))
    if not np.isfinite(w_sum) or w_sum <= 0:
        raise ValueError("Calibration lines produced invalid total weight")
    ppu_mean = float(np.dot(inv_var, ppus) / w_sum)

    eff_weights = inv_var.copy()
    if robust:
        for _ in range(max_irls_iters):
            residual_norm = (ppus - ppu_mean) / sigmas
            huber = np.ones_like(residual_norm)
            idx = np.abs(residual_norm) > huber_k
            huber[idx] = huber_k / np.abs(residual_norm[idx])
            eff_weights = inv_var * huber
            eff_sum = float(np.sum(eff_weights))
            if not np.isfinite(eff_sum) or eff_sum <= 0:
                raise ValueError("Calibration lines produced invalid robust weights")
            new_mean = float(np.dot(eff_weights, ppus) / eff_sum)
            if abs(new_mean - ppu_mean) <= irls_tol * max(1.0, abs(ppu_mean)):
                ppu_mean = new_mean
                break
            ppu_mean = new_mean

    # Standard error of weighted mean with variance inflation when residuals
    # exceed expected measurement noise (reduced chi-square > 1).
    eff_sum = float(np.sum(eff_weights))
    sem = float(np.sqrt(1.0 / eff_sum))
    chi2_value = float(np.sum(((ppus - ppu_mean) / sigmas) ** 2))
    dof = len(lines) - 1
    inflation = float(np.sqrt(max(1.0, chi2_value / dof))) if dof > 0 else 1.0
    ppu_std = sem * inflation

    if not np.isfinite(ppu_mean):
        raise ValueError("Failed to compute a finite weighted pixels-per-unit")
    if not np.isfinite(ppu_std) or ppu_std < 0:
        raise ValueError("Failed to compute a finite pixels-per-unit uncertainty")
    return ppu_mean, ppu_std


def line_calibration_diagnostics(
    lines: List[CalibrationLine],
    sigma_per_pixel: float = DEFAULT_LINE_ENDPOINT_SIGMA_PX,
    robust: bool = True,
    huber_k: float = DEFAULT_LINE_HUBER_K,
    max_irls_iters: int = DEFAULT_LINE_IRLS_MAX_ITERS,
    irls_tol: float = DEFAULT_LINE_IRLS_TOL,
) -> List[LineCalibrationDiagnostic]:
    """Return per-line diagnostics relative to the aggregated PPU estimate."""
    if not lines:
        return []
    try:
        sigma_per_pixel = float(sigma_per_pixel)
    except (TypeError, ValueError):
        raise ValueError("sigma_per_pixel must be positive") from None
    if not np.isfinite(sigma_per_pixel) or sigma_per_pixel <= 0:
        raise ValueError("sigma_per_pixel must be positive")

    pixel_lengths, ppus = _validated_ppu_inputs(lines)
    sigmas = _line_ppu_sigmas(pixel_lengths, ppus, lines, sigma_per_pixel)
    ppu_mean, _ = weighted_average_ppu(
        lines,
        sigma_per_pixel=sigma_per_pixel,
        robust=robust,
        huber_k=huber_k,
        max_irls_iters=max_irls_iters,
        irls_tol=irls_tol,
    )

    if ppu_mean <= 0:
        deviations = np.zeros_like(ppus)
    else:
        deviations = 100.0 * (ppus - ppu_mean) / ppu_mean

    out: List[LineCalibrationDiagnostic] = []
    for i, line in enumerate(lines):
        out.append(
            LineCalibrationDiagnostic(
                index=i,
                pixel_length=float(pixel_lengths[i]),
                known_length=float(line.known_length),
                unit=line.unit,
                ppu=float(ppus[i]),
                ppu_sigma=float(sigmas[i]),
                deviation_percent=float(deviations[i]),
            )
        )
    return out


def chi_squared_consistency(
    lines: List[CalibrationLine],
    sigma_per_pixel: float = DEFAULT_LINE_ENDPOINT_SIGMA_PX,
    confidence_level: float = 0.95,
) -> Tuple[float, float, bool]:
    """Test whether multiple calibration lines give consistent ppu values.

    Uses chi-squared test: each line's ppu is compared against the weighted
    mean, with uncertainty estimated from endpoint localization error.

    Parameters
    ----------
    lines : list of CalibrationLine
        At least 1 calibration line.
    sigma_per_pixel : float
        Assumed endpoint localization uncertainty in pixels.
    confidence_level : float
        Threshold for consistency (default 0.95).

    Returns
    -------
    (chi2_value, p_value, is_consistent) : tuple
    """
    try:
        sigma_per_pixel = float(sigma_per_pixel)
        confidence_level = float(confidence_level)
    except (TypeError, ValueError):
        raise ValueError("sigma_per_pixel and confidence_level must be numeric") from None

    if not np.isfinite(sigma_per_pixel) or sigma_per_pixel <= 0:
        raise ValueError("sigma_per_pixel must be positive")
    if not np.isfinite(confidence_level) or not (0.0 < confidence_level < 1.0):
        raise ValueError("confidence_level must be in (0, 1)")

    if len(lines) <= 1:
        if len(lines) == 1:
            _validated_ppu_inputs(lines)
        return 0.0, 1.0, True

    pixel_lengths, ppus = _validated_ppu_inputs(lines)
    sigmas = _line_ppu_sigmas(pixel_lengths, ppus, lines, sigma_per_pixel)
    if not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0):
        raise ValueError("Calibration lines produced invalid sigma values")

    weights = 1.0 / (sigmas ** 2)
    weights_sum = np.sum(weights)
    if not np.isfinite(weights_sum) or weights_sum <= 0:
        raise ValueError("Calibration lines produced invalid chi-squared weights")
    ppu_mean = np.sum(weights * ppus) / weights_sum

    chi2_value = float(np.sum(((ppus - ppu_mean) / sigmas) ** 2))
    dof = len(lines) - 1
    p_value = float(1.0 - chi2_dist.cdf(chi2_value, dof))
    p_value = float(np.clip(p_value, 0.0, 1.0))
    is_consistent = p_value > (1.0 - confidence_level)

    return chi2_value, p_value, is_consistent


def update_calibration_lines(
    calibration: Calibration,
    sigma_per_pixel: float = DEFAULT_LINE_ENDPOINT_SIGMA_PX,
    robust: bool = True,
    huber_k: float = DEFAULT_LINE_HUBER_K,
    max_irls_iters: int = DEFAULT_LINE_IRLS_MAX_ITERS,
    irls_tol: float = DEFAULT_LINE_IRLS_TOL,
) -> Calibration:
    """Recompute ppu and uncertainty from all calibration lines.

    Mutates and returns the calibration object.
    """
    if not calibration.lines:
        return calibration
    # Ensure all lines use the same unit
    base_unit = calibration.unit
    for line in calibration.lines:
        if line.unit != base_unit:
            raise ValueError(
                f"Mixed units in calibration lines: {line.unit} vs {base_unit}"
            )
    ppu, ppu_std = weighted_average_ppu(
        calibration.lines,
        sigma_per_pixel=sigma_per_pixel,
        robust=robust,
        huber_k=huber_k,
        max_irls_iters=max_irls_iters,
        irls_tol=irls_tol,
    )
    calibration.pixels_per_unit = ppu
    calibration.ppu_uncertainty = ppu_std
    return calibration


DEFAULT_PAD_PITCH_ENDPOINT_SIGMA_PX = 0.2


def calibrate_from_pad_pitch(
    image: np.ndarray,
    roi: Tuple[int, int, int, int],
    known_pitch: float,
    unit: Unit = Unit.MILLIMETER,
    min_pitch: int = 20,
    endpoint_sigma: float = DEFAULT_PAD_PITCH_ENDPOINT_SIGMA_PX,
) -> Optional[Tuple[CalibrationLine, "GridResult"]]:
    """Create a CalibrationLine from FFT-detected bond pad pitch.

    Parameters
    ----------
    image : np.ndarray
        Input image (BGR or grayscale).
    roi : (x, y, w, h)
        Region of interest covering a row of bond pads.
    known_pitch : float
        Known physical pad-to-pad pitch in ``unit``.
    unit : Unit
        Physical unit for ``known_pitch``.
    min_pitch : int
        Minimum expected pitch in pixels (passed to ``detect_grid``).
    endpoint_sigma : float
        FFT sub-pixel localization uncertainty in pixels per endpoint.

    Returns
    -------
    (CalibrationLine, GridResult) or None
        A synthetic calibration line whose pixel length equals the detected
        pitch magnitude, and the underlying GridResult.  Returns *None* if
        no periodic pattern was found.
    """
    from .grid import detect_grid

    if not np.isfinite(known_pitch) or known_pitch <= 0:
        raise ValueError("known_pitch must be positive")
    if not np.isfinite(endpoint_sigma) or endpoint_sigma <= 0:
        raise ValueError("endpoint_sigma must be positive")

    result = detect_grid(image, roi=roi, min_pitch=min_pitch)
    if result is None:
        return None

    result.origin_x = float(roi[0])
    result.origin_y = float(roi[1])

    # Pitch magnitude in pixels
    pitch_px = float(np.hypot(result.pitch_x, result.pitch_y))
    if pitch_px <= 0:
        return None

    # Construct a synthetic horizontal CalibrationLine whose pixel length
    # equals the detected pitch.  The two endpoints are arbitrary; only
    # their distance matters.
    cx = float(roi[0] + roi[2] / 2.0)
    cy = float(roi[1] + roi[3] / 2.0)
    line = CalibrationLine(
        p0=Vertex(x=cx - pitch_px / 2.0, y=cy),
        p1=Vertex(x=cx + pitch_px / 2.0, y=cy),
        known_length=known_pitch,
        unit=unit,
        endpoint_sigma=endpoint_sigma,
        source="pad_pitch",
    )

    # Annotate the GridResult with physical units.
    if known_pitch > 0 and pitch_px > 0:
        ppu = pitch_px / known_pitch
        result.pitch_x_physical = result.pitch_x / ppu
        result.pitch_y_physical = result.pitch_y / ppu
        result.unit = unit

    return line, result


def calibrate_from_rect(
    corners: List[Tuple[float, float]],
    known_width: float,
    known_height: float,
    unit: Unit = Unit.MILLIMETER,
) -> CalibrationRect:
    """Create a CalibrationRect from four pixel corners.

    corners order: top-left, top-right, bottom-right, bottom-left.
    """
    if len(corners) != 4:
        raise ValueError("Exactly 4 corners required")
    if known_width <= 0 or known_height <= 0:
        raise ValueError("known_width and known_height must be positive")
    verts = [Vertex(x=c[0], y=c[1]) for c in corners]
    return CalibrationRect(
        corners=verts,
        known_width=known_width,
        known_height=known_height,
        unit=unit,
    )


def compute_homography(
    rect: CalibrationRect,
    dst_scale: float = 10.0,
) -> Tuple[np.ndarray, Tuple[int, int], float]:
    """Compute homography matrix for perspective correction.

    Parameters
    ----------
    rect : CalibrationRect
        Source corners + known physical dimensions.
    dst_scale : float
        Pixels per physical unit in the rectified image.

    Returns
    -------
    (H, (dst_w, dst_h), ppu) : tuple
        H: 3x3 homography matrix
        (dst_w, dst_h): destination image size in pixels
        ppu: exact pixels_per_unit in the rectified image
    """
    try:
        dst_scale = float(dst_scale)
    except (TypeError, ValueError):
        raise ValueError("dst_scale must be positive") from None
    if not np.isfinite(dst_scale) or dst_scale <= 0:
        raise ValueError("dst_scale must be positive")

    src_pts = np.array(
        [[c.x, c.y] for c in rect.corners], dtype=np.float64
    )

    dst_w_px = rect.known_width * dst_scale
    dst_h_px = rect.known_height * dst_scale

    dst_pts = np.array(
        [
            [0.0, 0.0],
            [dst_w_px, 0.0],
            [dst_w_px, dst_h_px],
            [0.0, dst_h_px],
        ],
        dtype=np.float64,
    )

    H, mask = cv2.findHomography(src_pts, dst_pts)
    if H is None:
        raise ValueError("Failed to compute homography — corners may be degenerate")

    ppu = dst_w_px / rect.known_width
    return H, (int(round(dst_w_px)), int(round(dst_h_px))), ppu


def warp_image(
    image: np.ndarray,
    H: np.ndarray,
    dst_size: Tuple[int, int],
) -> np.ndarray:
    """Apply perspective warp to an image.

    Parameters
    ----------
    image : np.ndarray
        Source image (BGR or grayscale).
    H : np.ndarray
        3x3 homography matrix.
    dst_size : (width, height)
        Output image size.

    Returns
    -------
    np.ndarray
        Warped image.
    """
    return cv2.warpPerspective(
        image,
        H,
        dst_size,
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def warp_point(
    point: Tuple[float, float],
    H: np.ndarray,
) -> Tuple[float, float]:
    """Transform a single point through a homography."""
    pt = np.array([point[0], point[1], 1.0], dtype=np.float64)
    dst = H @ pt
    if abs(dst[2]) < 1e-12:
        raise ValueError("Homography produced an invalid homogeneous coordinate")
    dst /= dst[2]
    return float(dst[0]), float(dst[1])


def reprojection_error(
    rect: CalibrationRect,
    H: np.ndarray,
    dst_scale: float = 10.0,
) -> float:
    """Compute mean reprojection error for the calibration rectangle.

    Returns error in pixels (source image coordinates).
    """
    src_pts = np.array(
        [[c.x, c.y] for c in rect.corners], dtype=np.float64
    )
    dst_pts = np.array(
        [
            [0.0, 0.0],
            [rect.known_width * dst_scale, 0.0],
            [rect.known_width * dst_scale, rect.known_height * dst_scale],
            [0.0, rect.known_height * dst_scale],
        ],
        dtype=np.float64,
    )

    H_inv = np.linalg.inv(H)
    errors = []
    for i in range(4):
        pt = np.array([dst_pts[i][0], dst_pts[i][1], 1.0])
        reproj = H_inv @ pt
        reproj /= reproj[2]
        err = np.sqrt(
            (reproj[0] - src_pts[i][0]) ** 2
            + (reproj[1] - src_pts[i][1]) ** 2
        )
        errors.append(err)
    return float(np.mean(errors))


def update_calibration_rect(
    calibration: Calibration,
    image: Optional[np.ndarray] = None,
    dst_scale: float = 10.0,
    sigma_corner: float = 0.5,
    n_mc_samples: int = 200,
) -> Tuple[Calibration, Optional[np.ndarray]]:
    """Perform rectangle-based calibration with optional perspective warp.

    Parameters
    ----------
    calibration : Calibration
        Must have .rect set.
    image : np.ndarray, optional
        If provided, will be warped.
    dst_scale : float
        Pixels per unit in rectified image.
    sigma_corner : float
        Corner localization uncertainty in pixels.
    n_mc_samples : int
        Number of Monte Carlo samples for ppu uncertainty estimation.

    Returns
    -------
    (calibration, warped_image) : tuple
        Updated calibration and optionally the warped image.
    """
    rect = calibration.rect
    if rect is None:
        raise ValueError("No calibration rectangle set")
    try:
        sigma_corner = float(sigma_corner)
    except (TypeError, ValueError):
        raise ValueError("sigma_corner must be non-negative") from None
    try:
        n_mc_samples_float = float(n_mc_samples)
    except (TypeError, ValueError):
        raise ValueError("n_mc_samples must be non-negative") from None
    if not np.isfinite(n_mc_samples_float) or not n_mc_samples_float.is_integer():
        raise ValueError("n_mc_samples must be a non-negative integer")
    n_mc_samples = int(n_mc_samples_float)
    if not np.isfinite(sigma_corner) or sigma_corner < 0:
        raise ValueError("sigma_corner must be non-negative")
    if n_mc_samples < 0:
        raise ValueError("n_mc_samples must be non-negative")

    H, dst_size, ppu = compute_homography(rect, dst_scale)

    calibration.pixels_per_unit = ppu
    calibration.unit = rect.unit
    calibration.homography = H.flatten().tolist()

    # Monte Carlo estimation of ppu uncertainty from corner localization error.
    # compute_homography returns ppu = dst_scale always (rectified-image ppu),
    # so we instead estimate effective source-image ppu by averaging the pixel
    # lengths of the four edges divided by their known physical lengths.
    rng = np.random.default_rng(0)
    ppus_mc = []
    base_corners = np.array([(c.x, c.y) for c in rect.corners], dtype=np.float64)
    known_edge_lengths = np.array([
        rect.known_width,   # top:    TL -> TR
        rect.known_height,  # right:  TR -> BR
        rect.known_width,   # bottom: BR -> BL
        rect.known_height,  # left:   BL -> TL
    ])
    for _ in range(n_mc_samples):
        noise = rng.normal(0, sigma_corner, size=(4, 2))
        noisy = base_corners + noise
        # Compute pixel edge lengths from the noisy corners
        edge_pixels = np.array([
            np.linalg.norm(noisy[(i + 1) % 4] - noisy[i])
            for i in range(4)
        ])
        # Effective ppu for each edge, then average
        edge_ppus = edge_pixels / known_edge_lengths
        ppus_mc.append(float(np.mean(edge_ppus)))

    if ppus_mc:
        calibration.ppu_uncertainty = float(np.std(ppus_mc))
    else:
        calibration.ppu_uncertainty = 0.0

    warped = None
    if image is not None:
        warped = warp_image(image, H, dst_size)

    return calibration, warped
