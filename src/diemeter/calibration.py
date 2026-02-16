"""Scale calibration and perspective correction."""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .model import (
    Calibration,
    CalibrationLine,
    CalibrationRect,
    Unit,
    Vertex,
)


def calibrate_from_line(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    known_length: float,
    unit: Unit = Unit.MILLIMETER,
) -> CalibrationLine:
    """Create a CalibrationLine from two pixel coordinates and a known length."""
    if known_length <= 0:
        raise ValueError("known_length must be positive")
    return CalibrationLine(
        p0=Vertex(x=p0[0], y=p0[1]),
        p1=Vertex(x=p1[0], y=p1[1]),
        known_length=known_length,
        unit=unit,
    )


def weighted_average_ppu(lines: List[CalibrationLine]) -> Tuple[float, float]:
    """Compute weighted-average pixels-per-unit from multiple calibration lines.

    Weight = pixel length of each line (longer lines have lower relative
    quantization error).

    Returns
    -------
    (ppu, ppu_uncertainty) : tuple of float
        The weighted mean ppu and its standard deviation estimate.
    """
    if not lines:
        raise ValueError("No calibration lines provided")
    if len(lines) == 1:
        return lines[0].pixels_per_unit, 0.0

    weights = np.array([line.pixel_length for line in lines])
    ppus = np.array([line.pixels_per_unit for line in lines])
    w_sum = weights.sum()
    ppu_mean = float(np.dot(weights, ppus) / w_sum)

    # Weighted standard deviation
    if len(lines) > 1:
        variance = float(
            np.dot(weights, (ppus - ppu_mean) ** 2) / w_sum
        )
        ppu_std = variance ** 0.5
    else:
        ppu_std = 0.0

    return ppu_mean, ppu_std


def update_calibration_lines(calibration: Calibration) -> Calibration:
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
    ppu, ppu_std = weighted_average_ppu(calibration.lines)
    calibration.pixels_per_unit = ppu
    calibration.ppu_uncertainty = ppu_std
    return calibration


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

    Returns
    -------
    (calibration, warped_image) : tuple
        Updated calibration and optionally the warped image.
    """
    rect = calibration.rect
    if rect is None:
        raise ValueError("No calibration rectangle set")

    H, dst_size, ppu = compute_homography(rect, dst_scale)

    calibration.pixels_per_unit = ppu
    calibration.unit = rect.unit
    calibration.homography = H.flatten().tolist()

    # Compute reprojection error as a quality metric
    reproj_err = reprojection_error(rect, H, dst_scale)
    # Use reprojection error to estimate ppu uncertainty
    # Rough: uncertainty ~ reproj_err / known_length * ppu
    avg_dim = (rect.known_width + rect.known_height) / 2
    calibration.ppu_uncertainty = reproj_err / avg_dim if avg_dim > 0 else 0.0

    warped = None
    if image is not None:
        warped = warp_image(image, H, dst_size)

    return calibration, warped
