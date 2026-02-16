"""Polygon area measurement using the Shoelace formula."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .model import (
    Calibration,
    MeasurementResult,
    Polygon,
    Unit,
    convert_area,
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


def measure_polygon(
    polygon: Polygon,
    calibration: Calibration,
    target_unit: Unit | None = None,
) -> MeasurementResult:
    """Measure a polygon's area given a calibration.

    Parameters
    ----------
    polygon : Polygon
        The polygon to measure.
    calibration : Calibration
        Must be calibrated (pixels_per_unit > 0).
    target_unit : Unit, optional
        Convert result to this unit. Defaults to calibration unit.

    Returns
    -------
    MeasurementResult
    """
    if not calibration.is_calibrated:
        raise ValueError("Calibration has not been performed yet")

    area_px = polygon_area_pixels(polygon)
    ppu = calibration.pixels_per_unit
    area_cal = area_px / (ppu * ppu)
    out_unit = target_unit or calibration.unit

    if out_unit != calibration.unit:
        area_physical = convert_area(area_cal, calibration.unit, out_unit)
    else:
        area_physical = area_cal

    # Uncertainty propagation (simplified: ppu uncertainty only, no vertex uncertainty)
    # Full propagation is in uncertainty.py; here we provide a quick estimate.
    if calibration.ppu_uncertainty > 0 and ppu > 0:
        # dA/d(ppu) = -2 * A_px / ppu^3
        # sigma_A = |dA/d(ppu)| * sigma_ppu
        sigma_a_cal = 2.0 * area_cal * (calibration.ppu_uncertainty / ppu)
        if out_unit != calibration.unit:
            sigma_a = convert_area(sigma_a_cal, calibration.unit, out_unit)
        else:
            sigma_a = sigma_a_cal
    else:
        sigma_a = 0.0

    return MeasurementResult(
        polygon_label=polygon.label,
        area_pixels=area_px,
        area_physical=area_physical,
        area_uncertainty=sigma_a,
        unit=out_unit,
        n_vertices=polygon.n_vertices,
    )
