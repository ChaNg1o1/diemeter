"""Tests for uncertainty.py — analytic uncertainty propagation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from diemeter.model import Calibration, Polygon, Unit, Vertex
from diemeter.uncertainty import (
    area_uncertainty_ppu,
    area_uncertainty_vertices,
    shoelace_partials,
    total_area_uncertainty,
)


class TestShoelacePartials:
    def test_unit_square(self):
        verts = [(0, 0), (1, 0), (1, 1), (0, 1)]
        dA_dx, dA_dy = shoelace_partials(verts)
        assert len(dA_dx) == 4
        assert len(dA_dy) == 4
        # For a CCW unit square, partials should be well-defined
        # dA/dx_0 = 0.5 * (y_1 - y_3) = 0.5 * (0 - 1) = -0.5
        assert dA_dx[0] == pytest.approx(-0.5)
        # dA/dy_0 = 0.5 * (x_3 - x_1) = 0.5 * (0 - 1) = -0.5
        assert dA_dy[0] == pytest.approx(-0.5)

    def test_fewer_than_3(self):
        dA_dx, dA_dy = shoelace_partials([(0, 0)])
        assert len(dA_dx) == 0


class TestAreaUncertaintyVertices:
    def test_zero_sigma(self):
        verts = [(0, 0), (100, 0), (100, 100), (0, 100)]
        assert area_uncertainty_vertices(verts, sigma_xy=0.0) == 0.0

    def test_positive_sigma(self):
        verts = [(0, 0), (100, 0), (100, 100), (0, 100)]
        sigma = area_uncertainty_vertices(verts, sigma_xy=0.5)
        assert sigma > 0.0
        # For a 100x100 square with 0.5px uncertainty, sigma should be
        # on the order of tens of px^2
        assert sigma < 1000  # sanity upper bound

    def test_larger_polygon_more_uncertain(self):
        small = [(0, 0), (10, 0), (10, 10), (0, 10)]
        large = [(0, 0), (100, 0), (100, 100), (0, 100)]
        s_small = area_uncertainty_vertices(small, sigma_xy=0.5)
        s_large = area_uncertainty_vertices(large, sigma_xy=0.5)
        assert s_large > s_small

    def test_per_vertex_sigma(self):
        verts = [(0, 0), (100, 0), (100, 100), (0, 100)]
        # One vertex much more uncertain
        sigmas = np.array([0.1, 0.1, 0.1, 5.0])
        s = area_uncertainty_vertices(verts, sigma_xy=sigmas)
        assert s > 0.0

    def test_monte_carlo_cross_validation(self):
        """Cross-validate analytic uncertainty with Monte Carlo simulation."""
        verts = [(0, 0), (100, 0), (100, 50), (0, 50)]
        sigma = 1.0
        analytic = area_uncertainty_vertices(verts, sigma_xy=sigma)

        # Monte Carlo
        rng = np.random.default_rng(42)
        n_samples = 50000
        areas = []
        verts_arr = np.array(verts, dtype=np.float64)
        for _ in range(n_samples):
            perturbed = verts_arr + rng.normal(0, sigma, verts_arr.shape)
            xs, ys = perturbed[:, 0], perturbed[:, 1]
            a = 0.5 * abs(
                np.dot(xs, np.roll(ys, -1)) - np.dot(np.roll(xs, -1), ys)
            )
            areas.append(a)
        mc_std = np.std(areas)

        # Should agree within ~10%
        assert analytic == pytest.approx(mc_std, rel=0.15)


class TestAreaUncertaintyPPU:
    def test_zero_sigma(self):
        assert area_uncertainty_ppu(10000, 10.0, 0.0) == 0.0

    def test_basic(self):
        # A_phys = 10000 / 100 = 100
        # sigma = 2 * 100 * (0.5 / 10) = 10
        sigma = area_uncertainty_ppu(10000, 10.0, 0.5)
        assert sigma == pytest.approx(10.0)


class TestTotalAreaUncertainty:
    def test_both_sources(self):
        poly = Polygon(
            vertices=[
                Vertex(x=0, y=0),
                Vertex(x=100, y=0),
                Vertex(x=100, y=100),
                Vertex(x=0, y=100),
            ],
            closed=True,
        )
        cal = Calibration(
            pixels_per_unit=10.0,
            ppu_uncertainty=0.2,
            unit=Unit.MILLIMETER,
        )
        total, vert_part, ppu_part = total_area_uncertainty(poly, cal, vertex_sigma=0.5)
        assert total > 0
        assert vert_part > 0
        assert ppu_part > 0
        # total should be sqrt(vert^2 + ppu^2)
        expected = math.sqrt(vert_part**2 + ppu_part**2)
        assert total == pytest.approx(expected)

    def test_uncalibrated_raises(self):
        poly = Polygon(
            vertices=[Vertex(x=0, y=0), Vertex(x=1, y=0), Vertex(x=0, y=1)],
            closed=True,
        )
        with pytest.raises(ValueError):
            total_area_uncertainty(poly, Calibration())


from diemeter.uncertainty import confidence_to_sigma, build_vertex_sigmas


class TestConfidenceToSigma:
    def test_manual_click_no_snap(self):
        """Unsnapped vertex should have sigma = 1.0."""
        assert confidence_to_sigma(confidence=1.0, max_grad=255.0, snapped=False) == 1.0

    def test_high_confidence_snap(self):
        """High-confidence snap should approach 0.1."""
        sigma = confidence_to_sigma(confidence=255.0, max_grad=255.0, snapped=True)
        assert sigma == pytest.approx(0.1, abs=0.01)

    def test_low_confidence_snap(self):
        """Low-confidence snap should yield high sigma (close to 1.0)."""
        sigma = confidence_to_sigma(confidence=25.5, max_grad=255.0, snapped=True)
        assert 0.8 < sigma < 1.0

    def test_clamp_minimum(self):
        """Sigma should never go below 0.1."""
        sigma = confidence_to_sigma(confidence=1000.0, max_grad=100.0, snapped=True)
        assert sigma >= 0.1


class TestBuildVertexSigmas:
    def test_mixed_vertices(self):
        """Build sigma array from mix of snapped and unsnapped vertices."""
        poly = Polygon(
            vertices=[
                Vertex(x=0, y=0, snapped=False, confidence=1.0),
                Vertex(x=100, y=0, snapped=True, confidence=200.0),
                Vertex(x=100, y=100, snapped=True, confidence=50.0),
                Vertex(x=0, y=100, snapped=False, confidence=1.0),
            ],
            closed=True,
        )
        sigmas = build_vertex_sigmas(poly, max_grad=255.0)
        assert len(sigmas) == 4
        # Unsnapped vertices have sigma=1.0
        assert sigmas[0] == 1.0
        assert sigmas[3] == 1.0
        # Snapped vertices have sigma < 1.0
        assert sigmas[1] < 1.0
        assert sigmas[2] < 1.0
        # Higher confidence → lower sigma
        assert sigmas[1] < sigmas[2]
