"""Canny + Devernay sub-pixel edge detection and snap-to-edge."""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


def compute_edges(
    image: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
    blur_ksize: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run Canny edge detection and compute gradient magnitude/direction.

    Parameters
    ----------
    image : np.ndarray
        Input image (BGR or grayscale).
    low_threshold, high_threshold : int
        Canny thresholds.
    blur_ksize : int
        Gaussian blur kernel size (must be odd).

    Returns
    -------
    (edges, grad_mag, grad_dir) : tuple
        edges: binary edge map
        grad_mag: gradient magnitude
        grad_dir: gradient direction in radians
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # Compute gradients for sub-pixel refinement
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_dir = np.arctan2(grad_y, grad_x)

    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    return edges, grad_mag, grad_dir


def devernay_subpixel(
    edges: np.ndarray,
    grad_mag: np.ndarray,
    grad_dir: np.ndarray,
) -> List[Tuple[float, float, float]]:
    """Apply Devernay sub-pixel refinement to Canny edge points.

    For each edge pixel, fit a parabola along the gradient direction
    to find the sub-pixel position of the edge.

    Parameters
    ----------
    edges : np.ndarray
        Binary Canny edge map.
    grad_mag : np.ndarray
        Gradient magnitude.
    grad_dir : np.ndarray
        Gradient direction in radians.

    Returns
    -------
    list of (sub_x, sub_y, confidence)
        Sub-pixel edge points with confidence (gradient magnitude at edge).
    """
    h, w = edges.shape
    edge_points = []

    ys, xs = np.nonzero(edges)

    for y, x in zip(ys, xs):
        dx = np.cos(grad_dir[y, x])
        dy = np.sin(grad_dir[y, x])

        # Sample gradient magnitude at (x-dx, y-dy), (x, y), (x+dx, y+dy)
        # using bilinear interpolation
        m_center = grad_mag[y, x]

        xm, ym = x - dx, y - dy
        xp, yp = x + dx, y + dy

        m_minus = _bilinear_sample(grad_mag, xm, ym)
        m_plus = _bilinear_sample(grad_mag, xp, yp)

        if m_minus is None or m_plus is None:
            edge_points.append((float(x), float(y), m_center))
            continue

        # Parabolic interpolation
        denom = 2.0 * (m_minus - 2.0 * m_center + m_plus)
        if abs(denom) < 1e-10:
            edge_points.append((float(x), float(y), m_center))
            continue

        t = (m_minus - m_plus) / denom
        # Clamp t to [-0.5, 0.5] for stability
        t = max(-0.5, min(0.5, t))

        sub_x = x + t * dx
        sub_y = y + t * dy
        confidence = m_center + 0.5 * t * (m_minus - m_plus)

        edge_points.append((float(sub_x), float(sub_y), float(confidence)))

    return edge_points


def _bilinear_sample(
    img: np.ndarray,
    x: float,
    y: float,
) -> Optional[float]:
    """Bilinear interpolation at sub-pixel coordinates."""
    h, w = img.shape[:2]
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = x0 + 1, y0 + 1

    if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h:
        return None

    fx, fy = x - x0, y - y0
    val = (
        img[y0, x0] * (1 - fx) * (1 - fy)
        + img[y0, x1] * fx * (1 - fy)
        + img[y1, x0] * (1 - fx) * fy
        + img[y1, x1] * fx * fy
    )
    return float(val)


class EdgeDetector:
    """Manages edge detection state and snap-to-edge queries."""

    def __init__(
        self,
        image: np.ndarray,
        low_threshold: int = 50,
        high_threshold: int = 150,
    ) -> None:
        self._image = image
        self._edges, self._grad_mag, self._grad_dir = compute_edges(
            image, low_threshold, high_threshold
        )
        self._subpixel_points: Optional[List[Tuple[float, float, float]]] = None
        # Spatial index for fast nearest-neighbor queries
        self._points_array: Optional[np.ndarray] = None
        self._confidences: Optional[np.ndarray] = None

    @property
    def edges(self) -> np.ndarray:
        return self._edges

    def compute_subpixel(self) -> None:
        """Compute sub-pixel edge points (may be slow for large images)."""
        self._subpixel_points = devernay_subpixel(
            self._edges, self._grad_mag, self._grad_dir
        )
        if self._subpixel_points:
            pts = np.array([(p[0], p[1]) for p in self._subpixel_points])
            conf = np.array([p[2] for p in self._subpixel_points])
            self._points_array = pts
            self._confidences = conf

    def snap_to_edge(
        self,
        x: float,
        y: float,
        radius: float = 15.0,
    ) -> Optional[Tuple[float, float, float]]:
        """Find nearest sub-pixel edge point within radius.

        Parameters
        ----------
        x, y : float
            Click position in pixels.
        radius : float
            Search radius in pixels.

        Returns
        -------
        (sub_x, sub_y, confidence) or None
            Nearest sub-pixel edge point, or None if none within radius.
        """
        if self._points_array is None:
            self.compute_subpixel()

        if self._points_array is None or len(self._points_array) == 0:
            return None

        # Euclidean distance to all edge points
        dists = np.sqrt(
            (self._points_array[:, 0] - x) ** 2
            + (self._points_array[:, 1] - y) ** 2
        )

        within = dists <= radius
        if not np.any(within):
            return None

        # Among points within radius, pick the one with highest confidence
        candidates = np.where(within)[0]
        best_idx = candidates[np.argmax(self._confidences[candidates])]

        pt = self._points_array[best_idx]
        conf = self._confidences[best_idx]
        return float(pt[0]), float(pt[1]), float(conf)
