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

    # Full uncertainty propagation
    vertex_sigmas = build_vertex_sigmas(polygon, max_grad)
    total_unc, vert_unc, ppu_unc = total_area_uncertainty(
        polygon, calibration, vertex_sigma=vertex_sigmas
    )

    # Perimeter uncertainty
    verts = [(v.x, v.y) for v in polygon.vertices]
    perim_unc_px = perimeter_uncertainty(verts, sigma_xy=vertex_sigmas)
    perim_unc_cal = perim_unc_px / ppu

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
    else:
        area_physical = area_cal
        perim_physical = perim_cal

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
    )
