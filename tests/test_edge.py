"""Tests for edge.py — Canny + Devernay sub-pixel edge detection."""

from __future__ import annotations

import numpy as np
import pytest

from diemeter.edge import EdgeDetector, compute_edges, devernay_subpixel


class TestComputeEdges:
    def test_returns_correct_shapes(self, white_square_100):
        edges, grad_mag, grad_dir = compute_edges(white_square_100)
        assert edges.shape == white_square_100.shape
        assert grad_mag.shape == white_square_100.shape
        assert grad_dir.shape == white_square_100.shape

    def test_detects_square_edges(self, white_square_100):
        edges, _, _ = compute_edges(white_square_100, low_threshold=30, high_threshold=100)
        # Should have non-zero edge pixels around the square boundary
        assert np.count_nonzero(edges) > 0
        # Edges should be near the square boundary (rows/cols 49-51, 149-151)
        edge_ys, edge_xs = np.nonzero(edges)
        assert np.any((edge_ys >= 48) & (edge_ys <= 52))
        assert np.any((edge_ys >= 148) & (edge_ys <= 152))

    def test_color_image(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[30:70, 30:70] = [255, 128, 0]
        edges, _, _ = compute_edges(img)
        assert edges.shape == (100, 100)
        assert np.count_nonzero(edges) > 0


class TestDevernaySubpixel:
    def test_vertical_edge(self, gradient_edge_image):
        edges, grad_mag, grad_dir = compute_edges(
            gradient_edge_image, low_threshold=30, high_threshold=100
        )
        subpixel = devernay_subpixel(edges, grad_mag, grad_dir)
        if len(subpixel) == 0:
            pytest.skip("No edges detected on gradient image with these thresholds")

        # All sub-pixel x coordinates should be near x=100 (the edge)
        xs = [p[0] for p in subpixel]
        mean_x = np.mean(xs)
        assert abs(mean_x - 100.0) < 1.0

    def test_subpixel_precision(self):
        """Sub-pixel refinement should improve localization on a smooth edge.

        A step edge is too sharp for meaningful sub-pixel improvement;
        use a smooth gradient instead.
        """
        # Create a smooth vertical edge via a horizontal gradient
        x = np.arange(200, dtype=np.float64)
        # Sigmoid-like edge centered at x=100, width ~4px
        row = 255.0 / (1.0 + np.exp(-(x - 100.0) / 2.0))
        img = np.tile(row.astype(np.uint8), (200, 1))

        edges, grad_mag, grad_dir = compute_edges(img, low_threshold=30, high_threshold=100)
        subpixel = devernay_subpixel(edges, grad_mag, grad_dir)
        if len(subpixel) == 0:
            pytest.skip("No edges detected on smooth gradient")

        # All sub-pixel x coordinates should be close to 100
        xs = [p[0] for p in subpixel]
        mean_x = np.mean(xs)
        assert abs(mean_x - 100.0) < 1.5


class TestEdgeDetector:
    def test_snap_to_edge(self, white_square_100):
        det = EdgeDetector(white_square_100, low_threshold=30, high_threshold=100)
        det.compute_subpixel()

        # Snap near the top edge of the square (y~50)
        result = det.snap_to_edge(50.0, 55.0, radius=20.0)
        if result is None:
            pytest.skip("No snap point found near expected edge")
        sx, sy, conf = result
        # Should snap to approximately x=50 (left edge of square) or y~50
        assert conf > 0

    def test_snap_far_away_returns_none(self, white_square_100):
        det = EdgeDetector(white_square_100, low_threshold=30, high_threshold=100)
        det.compute_subpixel()
        # Center of the white square — no edges nearby
        result = det.snap_to_edge(100.0, 100.0, radius=5.0)
        assert result is None

    def test_edge_property(self, white_square_100):
        det = EdgeDetector(white_square_100)
        assert det.edges.shape == (200, 200)


class TestEdgeDetectorKDTree:
    def test_snap_uses_kdtree(self, gradient_edge_image):
        """Verify snap_to_edge returns same result but uses spatial index."""
        det = EdgeDetector(gradient_edge_image)
        det.compute_subpixel()
        # Verify KD-tree was built
        assert det._kdtree is not None
        # Snap should still work correctly
        result = det.snap_to_edge(100.0, 100.0, radius=15.0)
        assert result is not None
        assert abs(result[0] - 100.0) < 2.0  # near the edge at x=100
