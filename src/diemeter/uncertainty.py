"""Analytic uncertainty propagation for polygon area measurements."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .model import Calibration, Polygon


def shoelace_partials(
    vertices: List[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute partial derivatives of Shoelace area w.r.t. each vertex.

    For the Shoelace formula:
        A = 0.5 * |sum(x_i * y_{i+1} - x_{i+1} * y_i)|

    The partials are:
        dA/dx_i = 0.5 * (y_{i+1} - y_{i-1})
        dA/dy_i = 0.5 * (x_{i-1} - x_{i+1})

    Parameters
    ----------
    vertices : list of (x, y)
        Polygon vertices in order.

    Returns
    -------
    (dA_dx, dA_dy) : tuple of np.ndarray
        Partial derivatives, each of shape (n,).
    """
    n = len(vertices)
    if n < 3:
        return np.zeros(0), np.zeros(0)

    xs = np.array([v[0] for v in vertices])
    ys = np.array([v[1] for v in vertices])

    # Compute signed area to determine sign convention
    signed_area = 0.5 * (
        np.dot(xs, np.roll(ys, -1)) - np.dot(np.roll(xs, -1), ys)
    )
    sign = 1.0 if signed_area >= 0 else -1.0

    y_next = np.roll(ys, -1)
    y_prev = np.roll(ys, 1)
    x_next = np.roll(xs, -1)
    x_prev = np.roll(xs, 1)

    dA_dx = sign * 0.5 * (y_next - y_prev)
    dA_dy = sign * 0.5 * (x_prev - x_next)

    return dA_dx, dA_dy


def area_uncertainty_vertices(
    vertices: List[Tuple[float, float]],
    sigma_xy: float | np.ndarray = 0.5,
) -> float:
    """Compute area uncertainty from vertex position uncertainty alone.

    Parameters
    ----------
    vertices : list of (x, y)
        Polygon vertices.
    sigma_xy : float or np.ndarray
        Position uncertainty per vertex (isotropic). If array, shape (n,).

    Returns
    -------
    float
        Standard deviation of pixel area due to vertex position uncertainty.
    """
    dA_dx, dA_dy = shoelace_partials(vertices)
    if len(dA_dx) == 0:
        return 0.0

    sigma = np.atleast_1d(np.asarray(sigma_xy, dtype=np.float64))
    if sigma.shape == (1,):
        sigma = np.broadcast_to(sigma, dA_dx.shape)

    # sigma_A^2 = sum( (dA/dx_i)^2 * sigma_i^2 + (dA/dy_i)^2 * sigma_i^2 )
    variance = np.sum((dA_dx**2 + dA_dy**2) * sigma**2)
    return float(np.sqrt(variance))


def area_uncertainty_ppu(
    area_pixels: float,
    ppu: float,
    sigma_ppu: float,
) -> float:
    """Compute physical area uncertainty from ppu uncertainty alone.

    A_phys = A_px / ppu^2
    dA_phys/d(ppu) = -2 * A_px / ppu^3
    sigma_A_phys = |dA_phys/d(ppu)| * sigma_ppu
                 = 2 * A_phys * (sigma_ppu / ppu)
    """
    if ppu <= 0:
        return 0.0
    area_phys = area_pixels / (ppu * ppu)
    return 2.0 * area_phys * (sigma_ppu / ppu)


def total_area_uncertainty(
    polygon: Polygon,
    calibration: Calibration,
    vertex_sigma: float | np.ndarray = 0.5,
) -> Tuple[float, float, float]:
    """Compute total physical area uncertainty combining both sources.

    Parameters
    ----------
    polygon : Polygon
        The measured polygon.
    calibration : Calibration
        Calibration with ppu and ppu_uncertainty.
    vertex_sigma : float or np.ndarray
        Per-vertex position uncertainty in pixels.

    Returns
    -------
    (total_sigma, vertex_contrib, ppu_contrib) : tuple of float
        Total physical area uncertainty and its two components,
        all in calibration.unit squared.
    """
    if not calibration.is_calibrated:
        raise ValueError("Calibration not performed")

    verts = [(v.x, v.y) for v in polygon.vertices]
    ppu = calibration.pixels_per_unit

    # Pixel-space area
    n = len(verts)
    if n < 3:
        return 0.0, 0.0, 0.0
    xs = np.array([v[0] for v in verts])
    ys = np.array([v[1] for v in verts])
    area_px = 0.5 * abs(
        np.dot(xs, np.roll(ys, -1)) - np.dot(np.roll(xs, -1), ys)
    )

    # Vertex position contribution (in pixel area), then convert to physical
    sigma_a_px = area_uncertainty_vertices(verts, vertex_sigma)
    vertex_contrib = sigma_a_px / (ppu * ppu)

    # PPU contribution (directly in physical units)
    ppu_contrib = area_uncertainty_ppu(area_px, ppu, calibration.ppu_uncertainty)

    # Combine in quadrature (independent sources)
    total = float(np.sqrt(vertex_contrib**2 + ppu_contrib**2))

    return total, vertex_contrib, ppu_contrib


def confidence_to_sigma(
    confidence: float,
    max_grad: float,
    snapped: bool,
) -> float:
    """Map Devernay edge confidence to vertex position uncertainty (sigma in pixels).

    Parameters
    ----------
    confidence : float
        Gradient magnitude at the snapped edge point.
    max_grad : float
        Maximum gradient magnitude across all edge points (normalization).
    snapped : bool
        Whether this vertex was snapped to an edge.

    Returns
    -------
    float
        Estimated position uncertainty in pixels.
    """
    if not snapped:
        return 1.0
    if max_grad <= 0:
        return 0.5
    ratio = min(confidence / max_grad, 1.0)
    return max(0.1, 1.0 - 0.9 * ratio)


def build_vertex_sigmas(polygon: Polygon, max_grad: float) -> np.ndarray:
    """Build per-vertex sigma array from polygon vertex metadata.

    Parameters
    ----------
    polygon : Polygon
        Polygon with vertex snap/confidence info.
    max_grad : float
        Maximum gradient magnitude for normalization.

    Returns
    -------
    np.ndarray
        Array of shape (n_vertices,) with per-vertex sigma values.
    """
    return np.array([
        confidence_to_sigma(v.confidence, max_grad, v.snapped)
        for v in polygon.vertices
    ])


def perimeter_uncertainty(
    vertices: List[Tuple[float, float]],
    sigma_xy: float | np.ndarray = 0.5,
) -> float:
    """Compute perimeter uncertainty from vertex position uncertainty.

    For each edge (v_i, v_{i+1}), length L_k = |v_{i+1} - v_i|.
    Perimeter P = sum(L_k).
    dL_k/dx_i = -(x_{i+1} - x_i) / L_k,  dL_k/dy_i = -(y_{i+1} - y_i) / L_k
    dL_k/dx_{i+1} = (x_{i+1} - x_i) / L_k,  etc.

    Parameters
    ----------
    vertices : list of (x, y)
        Closed polygon vertices.
    sigma_xy : float or np.ndarray
        Per-vertex position uncertainty (isotropic).

    Returns
    -------
    float
        Standard deviation of perimeter in pixels.
    """
    n = len(vertices)
    if n < 2:
        return 0.0

    sigma = np.atleast_1d(np.asarray(sigma_xy, dtype=np.float64))
    if sigma.shape == (1,):
        sigma = np.broadcast_to(sigma, (n,))

    xs = np.array([v[0] for v in vertices])
    ys = np.array([v[1] for v in vertices])

    # dP/dx_i and dP/dy_i accumulate contributions from edges (i-1,i) and (i,i+1)
    dP_dx = np.zeros(n)
    dP_dy = np.zeros(n)

    for k in range(n):
        j = (k + 1) % n
        dx = xs[j] - xs[k]
        dy = ys[j] - ys[k]
        length = np.sqrt(dx * dx + dy * dy)
        if length < 1e-12:
            continue
        # Edge k: from vertex k to vertex j
        # dL_k/dx_k = -dx/L, dL_k/dy_k = -dy/L
        # dL_k/dx_j = +dx/L, dL_k/dy_j = +dy/L
        dP_dx[k] += -dx / length
        dP_dy[k] += -dy / length
        dP_dx[j] += dx / length
        dP_dy[j] += dy / length

    variance = np.sum((dP_dx**2 + dP_dy**2) * sigma**2)
    return float(np.sqrt(variance))
