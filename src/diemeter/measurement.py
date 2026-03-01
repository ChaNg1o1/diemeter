"""Polygon area measurement using the Shoelace formula."""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from .model import (
    Calibration,
    MeasurementResult,
    Polygon,
    Unit,
    convert_area,
    convert_length,
)
from .uncertainty import (
    build_vertex_sigmas,
    perimeter_uncertainty,
    total_area_uncertainty,
)


def shoelace_area(vertices: List[Tuple[float, float]]) -> float:
    """Compute the signed area of a polygon using the Shoelace formula.

    Parameters
    ----------
    vertices : list of (x, y) tuples
        Polygon vertices in order (clockwise or counter-clockwise).

    Returns
    -------
    float
        Unsigned area in squared pixel units.
    """
    n = len(vertices)
    if n < 3:
        return 0.0
    xs = np.array([v[0] for v in vertices])
    ys = np.array([v[1] for v in vertices])
    # Shoelace: A = 0.5 * |sum(x_i * y_{i+1} - x_{i+1} * y_i)|
    area = 0.5 * abs(
        np.dot(xs, np.roll(ys, -1)) - np.dot(np.roll(xs, -1), ys)
    )
    return float(area)


def polygon_area_pixels(polygon: Polygon) -> float:
    """Compute pixel area of a Polygon dataclass."""
    if polygon.n_vertices < 3:
        return 0.0
    verts = [(v.x, v.y) for v in polygon.vertices]
    return shoelace_area(verts)


def polygon_perimeter_pixels(polygon: Polygon) -> float:
    """Compute pixel perimeter of a closed polygon."""
    if polygon.n_vertices < 2:
        return 0.0
    total = 0.0
    verts = polygon.vertices
    n = len(verts)
    for i in range(n):
        j = (i + 1) % n
        dx = verts[j].x - verts[i].x
        dy = verts[j].y - verts[i].y
        total += (dx * dx + dy * dy) ** 0.5
    return total


def min_area_bbox_pixels(
    polygon: Polygon,
) -> Tuple[float, float, float]:
    """Compute the minimum-area bounding rectangle of a polygon.

    Returns
    -------
    (width_px, height_px, angle_deg) : tuple
        Width (shorter side), height (longer side), and rotation angle
        in degrees of the minimum-area enclosing rectangle.
        For fewer than 3 vertices returns (0, 0, 0).
    """
    if polygon.n_vertices < 3:
        return 0.0, 0.0, 0.0
    pts = np.array(
        [(v.x, v.y) for v in polygon.vertices], dtype=np.float32
    )
    (_cx, _cy), (w, h), angle = cv2.minAreaRect(pts)
    # Ensure width <= height (width = shorter side)
    if w > h:
        w, h = h, w
        angle = (angle + 90.0) % 180.0
    return float(w), float(h), float(angle)


def bbox_uncertainty_mc(
    polygon: Polygon,
    vertex_sigmas: np.ndarray,
    n_samples: int = 200,
) -> Tuple[float, float]:
    """Estimate bounding-box dimension uncertainty via Monte Carlo.

    Returns
    -------
    (sigma_width, sigma_height) : tuple of float
        Standard deviations of width and height in pixels.
    """
    if polygon.n_vertices < 3 or n_samples <= 0:
        return 0.0, 0.0
    base = np.array([(v.x, v.y) for v in polygon.vertices], dtype=np.float64)
    rng = np.random.default_rng(42)
    widths = []
    heights = []
    for _ in range(n_samples):
        noise = rng.normal(0, 1, size=base.shape) * vertex_sigmas[:, None]
        noisy = (base + noise).astype(np.float32)
        (_cx, _cy), (w, h), _angle = cv2.minAreaRect(noisy)
        if w > h:
            w, h = h, w
        widths.append(w)
        heights.append(h)
    return float(np.std(widths)), float(np.std(heights))


def measure_polygon(
    polygon: Polygon,
    calibration: Calibration,
    target_unit: Unit | None = None,
    max_grad: float = 255.0,
) -> MeasurementResult:
    """Measure a polygon's area and perimeter with full uncertainty propagation.

    Parameters
    ----------
    polygon : Polygon
        The polygon to measure.
    calibration : Calibration
        Must be calibrated (pixels_per_unit > 0).
    target_unit : Unit, optional
        Convert result to this unit. Defaults to calibration unit.
    max_grad : float
        Maximum gradient magnitude for confidence-to-sigma mapping.

    Returns
    -------
    MeasurementResult
    """
    if not calibration.is_calibrated:
        raise ValueError("Calibration has not been performed yet")

    area_px = polygon_area_pixels(polygon)
    perim_px = polygon_perimeter_pixels(polygon)
    ppu = calibration.pixels_per_unit
    out_unit = target_unit or calibration.unit

    # Physical values in calibration unit
    area_cal = area_px / (ppu * ppu)
    perim_cal = perim_px / ppu

    # Minimum-area bounding rectangle
    bbox_w_px, bbox_h_px, bbox_angle = min_area_bbox_pixels(polygon)
    bbox_w_cal = bbox_w_px / ppu
    bbox_h_cal = bbox_h_px / ppu

    # Full uncertainty propagation
    vertex_sigmas = build_vertex_sigmas(polygon, max_grad)
    total_unc, vert_unc, ppu_unc = total_area_uncertainty(
        polygon, calibration, vertex_sigma=vertex_sigmas
    )

    # Perimeter uncertainty
    verts = [(v.x, v.y) for v in polygon.vertices]
    perim_unc_px = perimeter_uncertainty(verts, sigma_xy=vertex_sigmas)
    perim_unc_cal = perim_unc_px / ppu

    # Bbox uncertainty (MC in pixels, then convert)
    bbox_w_unc_px, bbox_h_unc_px = bbox_uncertainty_mc(
        polygon, vertex_sigmas
    )
    # Combine vertex and ppu contributions in quadrature
    bbox_w_unc_cal = float(np.sqrt(
        (bbox_w_unc_px / ppu) ** 2
        + (bbox_w_cal * calibration.ppu_uncertainty / ppu) ** 2
    ))
    bbox_h_unc_cal = float(np.sqrt(
        (bbox_h_unc_px / ppu) ** 2
        + (bbox_h_cal * calibration.ppu_uncertainty / ppu) ** 2
    ))

    # Unit conversion if needed
    if out_unit != calibration.unit:
        area_physical = convert_area(area_cal, calibration.unit, out_unit)
        # Scale uncertainties by the same conversion factor
        scale_area = area_physical / area_cal if area_cal > 0 else 0.0
        total_unc *= scale_area
        vert_unc *= scale_area
        ppu_unc *= scale_area

        perim_physical = convert_length(perim_cal, calibration.unit, out_unit)
        scale_len = perim_physical / perim_cal if perim_cal > 0 else 0.0
        perim_unc_cal *= scale_len

        bbox_w_physical = convert_length(bbox_w_cal, calibration.unit, out_unit)
        bbox_h_physical = convert_length(bbox_h_cal, calibration.unit, out_unit)
        bbox_w_unc_cal = convert_length(bbox_w_unc_cal, calibration.unit, out_unit)
        bbox_h_unc_cal = convert_length(bbox_h_unc_cal, calibration.unit, out_unit)
    else:
        area_physical = area_cal
        perim_physical = perim_cal
        bbox_w_physical = bbox_w_cal
        bbox_h_physical = bbox_h_cal

    return MeasurementResult(
        polygon_label=polygon.label,
        area_pixels=area_px,
        area_physical=area_physical,
        area_uncertainty=total_unc,
        area_uncertainty_vertex=vert_unc,
        area_uncertainty_ppu=ppu_unc,
        perimeter_physical=perim_physical,
        perimeter_uncertainty=perim_unc_cal,
        unit=out_unit,
        n_vertices=polygon.n_vertices,
        bbox_width_physical=bbox_w_physical,
        bbox_height_physical=bbox_h_physical,
        bbox_angle_deg=bbox_angle,
        bbox_width_uncertainty=bbox_w_unc_cal,
        bbox_height_uncertainty=bbox_h_unc_cal,
    )
