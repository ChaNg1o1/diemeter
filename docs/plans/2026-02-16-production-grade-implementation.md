# Production-Grade Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade diemeter from prototype to production-grade: connect full uncertainty propagation, add numerical detail panel, optimize interaction/performance, harden tests.

**Architecture:** Incremental refactoring across 4 parallel work streams. Agent 1 (measurement precision) ships model changes first; Agents 2-4 can then run in parallel.

**Tech Stack:** Python 3.9+, PySide6, OpenCV, NumPy, SciPy, pytest

---

## Stream 1: Measurement Precision (Agent 1)

### Task 1: Extend MeasurementResult with uncertainty decomposition and perimeter

**Files:**
- Modify: `src/diemeter/model.py:152-162`
- Test: `tests/test_model.py` (new)

**Step 1: Write the failing test**

Create `tests/test_model.py`:

```python
"""Tests for model.py — domain types and unit conversion."""

from __future__ import annotations

import math

import pytest

from diemeter.model import (
    Calibration,
    MeasurementResult,
    Polygon,
    Unit,
    Vertex,
    convert_area,
    convert_length,
    unit_from_str,
)


class TestMeasurementResultExtended:
    def test_has_uncertainty_decomposition(self):
        r = MeasurementResult(
            polygon_label="test",
            area_pixels=10000.0,
            area_physical=100.0,
            area_uncertainty=5.0,
            area_uncertainty_vertex=3.0,
            area_uncertainty_ppu=4.0,
            perimeter_physical=40.0,
            perimeter_uncertainty=0.5,
            unit=Unit.MILLIMETER,
            n_vertices=4,
        )
        assert r.area_uncertainty_vertex == 3.0
        assert r.area_uncertainty_ppu == 4.0
        assert r.perimeter_physical == 40.0
        assert r.perimeter_uncertainty == 0.5

    def test_defaults_backward_compatible(self):
        r = MeasurementResult(
            polygon_label="test",
            area_pixels=100.0,
            area_physical=10.0,
            area_uncertainty=1.0,
            unit=Unit.MILLIMETER,
            n_vertices=3,
        )
        assert r.area_uncertainty_vertex == 0.0
        assert r.area_uncertainty_ppu == 0.0
        assert r.perimeter_physical == 0.0
        assert r.perimeter_uncertainty == 0.0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_model.py::TestMeasurementResultExtended -v`
Expected: FAIL — `MeasurementResult.__init__() got unexpected keyword argument 'area_uncertainty_vertex'`

**Step 3: Implement — extend MeasurementResult**

In `src/diemeter/model.py`, change the `MeasurementResult` dataclass (lines 152-162):

```python
@dataclass
class MeasurementResult:
    """Result of a polygon area measurement."""

    polygon_label: str
    area_pixels: float
    area_physical: float
    area_uncertainty: float
    unit: Unit
    n_vertices: int
    # Uncertainty decomposition
    area_uncertainty_vertex: float = 0.0
    area_uncertainty_ppu: float = 0.0
    # Perimeter
    perimeter_physical: float = 0.0
    perimeter_uncertainty: float = 0.0
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_model.py::TestMeasurementResultExtended -v`
Expected: PASS

**Step 5: Run all existing tests to verify no regression**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS (new fields have defaults, so existing code is unaffected)

**Step 6: Commit**

```bash
git add src/diemeter/model.py tests/test_model.py
git commit -m "feat(model): extend MeasurementResult with uncertainty decomposition and perimeter"
```

---

### Task 2: Add confidence_to_sigma mapping in uncertainty.py

**Files:**
- Modify: `src/diemeter/uncertainty.py`
- Test: `tests/test_uncertainty.py`

**Step 1: Write the failing test**

Append to `tests/test_uncertainty.py`:

```python
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
        """Low-confidence snap should be around 0.5."""
        sigma = confidence_to_sigma(confidence=25.5, max_grad=255.0, snapped=True)
        assert 0.4 < sigma < 0.7

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
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_uncertainty.py::TestConfidenceToSigma -v`
Expected: FAIL — ImportError

**Step 3: Implement**

Add to `src/diemeter/uncertainty.py` (after the existing imports, add `from .model import Polygon` if not already there — it's already imported via `from .model import Calibration, Polygon`):

```python
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
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_uncertainty.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/diemeter/uncertainty.py tests/test_uncertainty.py
git commit -m "feat(uncertainty): add confidence_to_sigma and build_vertex_sigmas"
```

---

### Task 3: Add perimeter_uncertainty to uncertainty.py

**Files:**
- Modify: `src/diemeter/uncertainty.py`
- Test: `tests/test_uncertainty.py`

**Step 1: Write the failing test**

Append to `tests/test_uncertainty.py`:

```python
from diemeter.uncertainty import perimeter_uncertainty


class TestPerimeterUncertainty:
    def test_unit_square(self):
        """Perimeter of unit square with known vertex uncertainty."""
        verts = [(0, 0), (1, 0), (1, 1), (0, 1)]
        sigma = perimeter_uncertainty(verts, sigma_xy=0.1)
        assert sigma > 0.0
        # Each edge contributes, so total should be non-trivial
        assert sigma < 1.0  # sanity: small sigma on small polygon

    def test_zero_sigma(self):
        verts = [(0, 0), (100, 0), (100, 100), (0, 100)]
        assert perimeter_uncertainty(verts, sigma_xy=0.0) == 0.0

    def test_per_vertex_sigma(self):
        verts = [(0, 0), (100, 0), (100, 100), (0, 100)]
        sigmas = np.array([0.1, 0.1, 0.1, 5.0])
        sigma = perimeter_uncertainty(verts, sigma_xy=sigmas)
        assert sigma > 0.0

    def test_monte_carlo_cross_validation(self):
        """Cross-validate perimeter uncertainty with Monte Carlo."""
        verts = [(0, 0), (100, 0), (100, 50), (0, 50)]
        sigma_xy = 1.0
        analytic = perimeter_uncertainty(verts, sigma_xy=sigma_xy)

        rng = np.random.default_rng(42)
        n_samples = 50000
        perimeters = []
        verts_arr = np.array(verts, dtype=np.float64)
        for _ in range(n_samples):
            perturbed = verts_arr + rng.normal(0, sigma_xy, verts_arr.shape)
            n = len(perturbed)
            perim = sum(
                np.sqrt((perturbed[(i+1) % n][0] - perturbed[i][0])**2 +
                        (perturbed[(i+1) % n][1] - perturbed[i][1])**2)
                for i in range(n)
            )
            perimeters.append(perim)
        mc_std = np.std(perimeters)
        assert analytic == pytest.approx(mc_std, rel=0.15)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_uncertainty.py::TestPerimeterUncertainty -v`
Expected: FAIL — ImportError

**Step 3: Implement**

Add to `src/diemeter/uncertainty.py`:

```python
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
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_uncertainty.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/diemeter/uncertainty.py tests/test_uncertainty.py
git commit -m "feat(uncertainty): add perimeter_uncertainty with analytic propagation"
```

---

### Task 4: Connect full uncertainty propagation in measure_polygon

**Files:**
- Modify: `src/diemeter/measurement.py:66-119`
- Modify: `tests/test_measurement.py`

**Step 1: Write the failing test**

Add to `tests/test_measurement.py`:

```python
class TestMeasurePolygonFullUncertainty:
    def test_uncertainty_decomposition(self):
        poly = Polygon(
            vertices=[
                Vertex(x=0, y=0, snapped=True, confidence=200.0),
                Vertex(x=100, y=0, snapped=True, confidence=200.0),
                Vertex(x=100, y=100, snapped=False, confidence=1.0),
                Vertex(x=0, y=100, snapped=False, confidence=1.0),
            ],
            label="test",
            closed=True,
        )
        cal = Calibration(
            pixels_per_unit=10.0,
            ppu_uncertainty=0.2,
            unit=Unit.MILLIMETER,
        )
        result = measure_polygon(poly, cal)
        assert result.area_uncertainty_vertex > 0.0
        assert result.area_uncertainty_ppu > 0.0
        # Total should be quadrature combination
        import math
        expected_total = math.sqrt(
            result.area_uncertainty_vertex**2 + result.area_uncertainty_ppu**2
        )
        assert result.area_uncertainty == pytest.approx(expected_total, rel=1e-6)

    def test_perimeter_in_result(self):
        poly = Polygon(
            vertices=[
                Vertex(x=0, y=0),
                Vertex(x=100, y=0),
                Vertex(x=100, y=100),
                Vertex(x=0, y=100),
            ],
            label="test",
            closed=True,
        )
        cal = Calibration(
            pixels_per_unit=10.0,
            unit=Unit.MILLIMETER,
        )
        result = measure_polygon(poly, cal)
        # Perimeter = 400 px / 10 px/mm = 40 mm
        assert result.perimeter_physical == pytest.approx(40.0)

    def test_max_grad_passed_through(self):
        """Ensure max_grad parameter affects vertex uncertainty."""
        poly = Polygon(
            vertices=[
                Vertex(x=0, y=0, snapped=True, confidence=100.0),
                Vertex(x=100, y=0, snapped=True, confidence=100.0),
                Vertex(x=100, y=100, snapped=True, confidence=100.0),
                Vertex(x=0, y=100, snapped=True, confidence=100.0),
            ],
            label="test",
            closed=True,
        )
        cal = Calibration(
            pixels_per_unit=10.0,
            ppu_uncertainty=0.2,
            unit=Unit.MILLIMETER,
        )
        r1 = measure_polygon(poly, cal, max_grad=100.0)
        r2 = measure_polygon(poly, cal, max_grad=200.0)
        # Higher max_grad means same confidence is lower relative → higher sigma
        assert r2.area_uncertainty_vertex > r1.area_uncertainty_vertex
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_measurement.py::TestMeasurePolygonFullUncertainty -v`
Expected: FAIL

**Step 3: Implement — rewrite measure_polygon**

Replace `measure_polygon` in `src/diemeter/measurement.py`:

```python
from .uncertainty import (
    build_vertex_sigmas,
    perimeter_uncertainty,
    total_area_uncertainty,
)


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

        from .model import convert_length
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
```

Also update the imports at the top of `measurement.py` to add `convert_length`:

```python
from .model import (
    Calibration,
    MeasurementResult,
    Polygon,
    Unit,
    convert_area,
    convert_length,
)
```

**Step 4: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/diemeter/measurement.py tests/test_measurement.py
git commit -m "feat(measurement): connect full uncertainty propagation with perimeter"
```

---

### Task 5: Add chi-squared consistency check for multi-line calibration

**Files:**
- Modify: `src/diemeter/calibration.py`
- Modify: `tests/test_calibration.py`

**Step 1: Write the failing test**

Add to `tests/test_calibration.py`:

```python
from diemeter.calibration import chi_squared_consistency


class TestChiSquaredConsistency:
    def test_consistent_lines(self):
        """Lines with similar ppu should pass."""
        lines = [
            CalibrationLine(Vertex(0, 0), Vertex(100, 0), 10.0, Unit.MILLIMETER),
            CalibrationLine(Vertex(0, 0), Vertex(0, 101), 10.1, Unit.MILLIMETER),
        ]
        chi2, p_value, is_consistent = chi_squared_consistency(lines)
        assert chi2 >= 0
        assert 0 <= p_value <= 1
        assert is_consistent is True

    def test_inconsistent_lines(self):
        """Lines with very different ppu should fail."""
        lines = [
            CalibrationLine(Vertex(0, 0), Vertex(100, 0), 10.0, Unit.MILLIMETER),
            CalibrationLine(Vertex(0, 0), Vertex(100, 0), 20.0, Unit.MILLIMETER),
        ]
        chi2, p_value, is_consistent = chi_squared_consistency(lines)
        assert is_consistent is False

    def test_single_line(self):
        """Single line should always be consistent."""
        lines = [
            CalibrationLine(Vertex(0, 0), Vertex(100, 0), 10.0, Unit.MILLIMETER),
        ]
        chi2, p_value, is_consistent = chi_squared_consistency(lines)
        assert chi2 == 0.0
        assert is_consistent is True
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_calibration.py::TestChiSquaredConsistency -v`
Expected: FAIL — ImportError

**Step 3: Implement**

Add to `src/diemeter/calibration.py`:

```python
from scipy.stats import chi2 as chi2_dist


def chi_squared_consistency(
    lines: List[CalibrationLine],
    sigma_per_pixel: float = 0.5,
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
    if len(lines) <= 1:
        return 0.0, 1.0, True

    ppus = np.array([line.pixels_per_unit for line in lines])
    # Uncertainty of each ppu: sigma_ppu ≈ ppu * sqrt(2) * sigma / pixel_length
    sigmas = np.array([
        line.pixels_per_unit * np.sqrt(2) * sigma_per_pixel / line.pixel_length
        for line in lines
    ])

    weights = 1.0 / (sigmas ** 2)
    ppu_mean = np.sum(weights * ppus) / np.sum(weights)

    chi2_value = float(np.sum(((ppus - ppu_mean) / sigmas) ** 2))
    dof = len(lines) - 1
    p_value = float(1.0 - chi2_dist.cdf(chi2_value, dof))
    is_consistent = p_value > (1.0 - confidence_level)

    return chi2_value, p_value, is_consistent
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_calibration.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/diemeter/calibration.py tests/test_calibration.py
git commit -m "feat(calibration): add chi-squared consistency check for multi-line calibration"
```

---

### Task 6: Improve rectangle calibration uncertainty estimation

**Files:**
- Modify: `src/diemeter/calibration.py:235-277`
- Modify: `tests/test_calibration.py`

**Step 1: Write the failing test**

Add to `tests/test_calibration.py`:

```python
class TestRectCalibrationUncertainty:
    def test_uncertainty_from_corner_localization(self):
        """Rect calibration uncertainty should be based on corner sigma, not reprojection error."""
        corners = [(80, 60), (220, 70), (210, 230), (90, 220)]
        rect = calibrate_from_rect(corners, 10.0, 8.0, Unit.MILLIMETER)
        cal = Calibration(rect=rect, unit=Unit.MILLIMETER)
        cal, _ = update_calibration_rect(cal, dst_scale=10.0)
        # With sigma_corner=0.5 default, uncertainty should be positive but small
        assert cal.ppu_uncertainty > 0
        # Should be proportional to corner uncertainty, not zero-ish reprojection error
        assert cal.ppu_uncertainty > 0.001
```

**Step 2: Run test — might pass or fail depending on current behavior**

Run: `python -m pytest tests/test_calibration.py::TestRectCalibrationUncertainty -v`

**Step 3: Implement — replace heuristic with Monte Carlo corner propagation**

Modify `update_calibration_rect` in `src/diemeter/calibration.py`:

```python
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

    H, dst_size, ppu = compute_homography(rect, dst_scale)

    calibration.pixels_per_unit = ppu
    calibration.unit = rect.unit
    calibration.homography = H.flatten().tolist()

    # Monte Carlo estimation of ppu uncertainty from corner localization error
    rng = np.random.default_rng(0)
    ppus = []
    base_corners = [(c.x, c.y) for c in rect.corners]
    for _ in range(n_mc_samples):
        noisy_corners = [
            (cx + rng.normal(0, sigma_corner), cy + rng.normal(0, sigma_corner))
            for cx, cy in base_corners
        ]
        try:
            noisy_rect = CalibrationRect(
                corners=[Vertex(x=c[0], y=c[1]) for c in noisy_corners],
                known_width=rect.known_width,
                known_height=rect.known_height,
                unit=rect.unit,
            )
            _, _, mc_ppu = compute_homography(noisy_rect, dst_scale)
            ppus.append(mc_ppu)
        except (ValueError, cv2.error):
            continue

    if ppus:
        calibration.ppu_uncertainty = float(np.std(ppus))
    else:
        # Fallback: rough estimate
        avg_dim = (rect.known_width + rect.known_height) / 2
        reproj_err = reprojection_error(rect, H, dst_scale)
        calibration.ppu_uncertainty = reproj_err / avg_dim if avg_dim > 0 else 0.0

    warped = None
    if image is not None:
        warped = warp_image(image, H, dst_size)

    return calibration, warped
```

**Step 4: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/diemeter/calibration.py tests/test_calibration.py
git commit -m "feat(calibration): Monte Carlo ppu uncertainty for rect calibration"
```

---

## Stream 2: Performance (Agent 2)

### Task 7: Vectorize Devernay sub-pixel edge detection

**Files:**
- Modify: `src/diemeter/edge.py:52-115`
- Test: `tests/test_edge.py` (existing tests cover correctness)

**Step 1: Run existing edge tests as baseline**

Run: `python -m pytest tests/test_edge.py -v`
Expected: ALL PASS

**Step 2: Implement vectorized devernay_subpixel**

Replace `devernay_subpixel` in `src/diemeter/edge.py`:

```python
def devernay_subpixel(
    edges: np.ndarray,
    grad_mag: np.ndarray,
    grad_dir: np.ndarray,
) -> List[Tuple[float, float, float]]:
    """Apply Devernay sub-pixel refinement to Canny edge points (vectorized).

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
    ys, xs = np.nonzero(edges)
    n = len(xs)
    if n == 0:
        return []

    xs_f = xs.astype(np.float64)
    ys_f = ys.astype(np.float64)

    dx = np.cos(grad_dir[ys, xs])
    dy = np.sin(grad_dir[ys, xs])
    m_center = grad_mag[ys, xs].astype(np.float64)

    # Minus and plus sample positions
    xm, ym = xs_f - dx, ys_f - dy
    xp, yp = xs_f + dx, ys_f + dy

    # Vectorized bilinear sampling
    m_minus = _bilinear_sample_batch(grad_mag, xm, ym)
    m_plus = _bilinear_sample_batch(grad_mag, xp, yp)

    # Parabolic interpolation
    denom = 2.0 * (m_minus - 2.0 * m_center + m_plus)

    # Where interpolation is valid
    valid = (m_minus >= 0) & (m_plus >= 0) & (np.abs(denom) >= 1e-10)

    t = np.zeros(n)
    t[valid] = (m_minus[valid] - m_plus[valid]) / denom[valid]
    t = np.clip(t, -0.5, 0.5)

    sub_x = xs_f + t * dx
    sub_y = ys_f + t * dy
    confidence = m_center + 0.5 * t * (m_minus - m_plus)
    # For invalid points, use original coordinates
    confidence = np.where(valid, confidence, m_center)

    return list(zip(sub_x.tolist(), sub_y.tolist(), confidence.tolist()))


def _bilinear_sample_batch(
    img: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Vectorized bilinear interpolation. Returns -1.0 for out-of-bounds."""
    h, w = img.shape[:2]
    x0 = np.floor(x).astype(np.intp)
    y0 = np.floor(y).astype(np.intp)
    x1 = x0 + 1
    y1 = y0 + 1

    # Bounds check
    valid = (x0 >= 0) & (y0 >= 0) & (x1 < w) & (y1 < h)
    result = np.full_like(x, -1.0, dtype=np.float64)

    if not np.any(valid):
        return result

    x0v, y0v = x0[valid], y0[valid]
    x1v, y1v = x1[valid], y1[valid]
    fx = x[valid] - x0v
    fy = y[valid] - y0v

    result[valid] = (
        img[y0v, x0v] * (1 - fx) * (1 - fy)
        + img[y0v, x1v] * fx * (1 - fy)
        + img[y1v, x0v] * (1 - fx) * fy
        + img[y1v, x1v] * fx * fy
    )
    return result
```

**Step 3: Run tests to verify correctness preserved**

Run: `python -m pytest tests/test_edge.py -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add src/diemeter/edge.py
git commit -m "perf(edge): vectorize Devernay sub-pixel detection with NumPy batch ops"
```

---

### Task 8: Add KD-tree spatial index for edge snap queries

**Files:**
- Modify: `src/diemeter/edge.py` (EdgeDetector class)
- Test: `tests/test_edge.py`

**Step 1: Write the failing test**

Add to `tests/test_edge.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_edge.py::TestEdgeDetectorKDTree -v`
Expected: FAIL — `_kdtree` attribute doesn't exist

**Step 3: Implement**

Modify `EdgeDetector` in `src/diemeter/edge.py`:

```python
from scipy.spatial import cKDTree


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
        self._points_array: Optional[np.ndarray] = None
        self._confidences: Optional[np.ndarray] = None
        self._kdtree: Optional[cKDTree] = None
        self._max_grad: float = 0.0

    @property
    def edges(self) -> np.ndarray:
        return self._edges

    @property
    def max_grad(self) -> float:
        """Maximum gradient magnitude across all sub-pixel edge points."""
        return self._max_grad

    def compute_subpixel(self) -> None:
        """Compute sub-pixel edge points and build spatial index."""
        self._subpixel_points = devernay_subpixel(
            self._edges, self._grad_mag, self._grad_dir
        )
        if self._subpixel_points:
            pts = np.array([(p[0], p[1]) for p in self._subpixel_points])
            conf = np.array([p[2] for p in self._subpixel_points])
            self._points_array = pts
            self._confidences = conf
            self._kdtree = cKDTree(pts)
            self._max_grad = float(conf.max()) if len(conf) > 0 else 0.0

    def snap_to_edge(
        self,
        x: float,
        y: float,
        radius: float = 15.0,
    ) -> Optional[Tuple[float, float, float]]:
        """Find nearest sub-pixel edge point within radius using KD-tree.

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

        if self._kdtree is None or self._points_array is None:
            return None

        # KD-tree ball query
        indices = self._kdtree.query_ball_point([x, y], radius)
        if not indices:
            return None

        # Among candidates, pick highest confidence
        candidates = np.array(indices)
        best_idx = candidates[np.argmax(self._confidences[candidates])]

        pt = self._points_array[best_idx]
        conf = self._confidences[best_idx]
        return float(pt[0]), float(pt[1]), float(conf)
```

**Step 4: Run all edge tests**

Run: `python -m pytest tests/test_edge.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/diemeter/edge.py tests/test_edge.py
git commit -m "perf(edge): replace O(n) snap with KD-tree spatial index"
```

---

### Task 9: Async edge detection with QThread

**Files:**
- Modify: `src/diemeter/gui/app.py`

**Step 1: Implement EdgeWorker and integrate**

Add to `src/diemeter/gui/app.py` before the `MeasurementApp` class:

```python
from PySide6.QtCore import QThread, Signal as QtSignal


class _EdgeWorker(QThread):
    """Background worker for sub-pixel edge computation."""

    finished = QtSignal()

    def __init__(self, detector) -> None:
        super().__init__()
        self._detector = detector

    def run(self) -> None:
        self._detector.compute_subpixel()
        self.finished.emit()
```

Modify `MeasurementApp.__init__` to add:

```python
self._edge_worker: Optional[_EdgeWorker] = None
```

Modify `MeasurementApp.toggle_edge_snap`:

```python
def toggle_edge_snap(self) -> None:
    """Toggle edge snap on/off."""
    self.session.edge_snap_enabled = not self.session.edge_snap_enabled
    if self.session.edge_snap_enabled and self._edge_detector is None:
        self._edge_detector = EdgeDetector(self.original_image)
        self._edge_worker = _EdgeWorker(self._edge_detector)
        self._edge_worker.finished.connect(self._on_edge_detection_done)
        self._edge_worker.start()
    self.redraw()

def _on_edge_detection_done(self) -> None:
    """Called when background edge detection finishes."""
    self._edge_worker = None
    self.redraw()
```

Modify `MeasurementApp.maybe_snap` to check if ready:

```python
def maybe_snap(self, x: float, y: float) -> Tuple[float, float]:
    """Snap coordinates to nearest edge if edge snap is enabled and ready."""
    if not self.session.edge_snap_enabled:
        return x, y
    if self._edge_detector is None or self._edge_detector._kdtree is None:
        return x, y  # Still computing
    result = self._edge_detector.snap_to_edge(
        x, y, radius=float(self.session.edge_snap_radius)
    )
    if result is not None:
        return result[0], result[1]
    return x, y
```

**Step 2: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS (no GUI tests exist yet, core logic unchanged)

**Step 3: Commit**

```bash
git add src/diemeter/gui/app.py
git commit -m "perf(gui): async edge detection in QThread to avoid UI blocking"
```

---

## Stream 3: Interaction & Display (Agent 3)

### Task 10: Update session serialization for new MeasurementResult fields

**Files:**
- Modify: `src/diemeter/session.py:114-133`
- Test: `tests/test_session.py` (new)

**Step 1: Write the failing test**

Create `tests/test_session.py`:

```python
"""Tests for session.py — serialization roundtrips."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from diemeter.model import (
    Calibration,
    CalibrationLine,
    MeasurementResult,
    Polygon,
    Session,
    Unit,
    Vertex,
)
from diemeter.session import (
    load_session,
    save_session,
    session_from_dict,
    session_to_dict,
)


class TestSessionRoundtrip:
    def test_empty_session(self):
        s = Session()
        d = session_to_dict(s)
        s2 = session_from_dict(d)
        assert s2.image_path == ""
        assert not s2.calibration.is_calibrated
        assert s2.polygons == []

    def test_full_session(self):
        s = Session(
            image_path="test.png",
            calibration=Calibration(
                lines=[CalibrationLine(
                    Vertex(0, 0), Vertex(100, 0), 10.0, Unit.MILLIMETER
                )],
                pixels_per_unit=10.0,
                ppu_uncertainty=0.1,
                unit=Unit.MILLIMETER,
            ),
            polygons=[Polygon(
                vertices=[Vertex(0, 0), Vertex(10, 0), Vertex(10, 10), Vertex(0, 10)],
                label="poly_1",
                closed=True,
            )],
            results=[MeasurementResult(
                polygon_label="poly_1",
                area_pixels=100.0,
                area_physical=1.0,
                area_uncertainty=0.1,
                area_uncertainty_vertex=0.05,
                area_uncertainty_ppu=0.087,
                perimeter_physical=4.0,
                perimeter_uncertainty=0.02,
                unit=Unit.MILLIMETER,
                n_vertices=4,
            )],
        )
        d = session_to_dict(s)
        s2 = session_from_dict(d)
        assert s2.image_path == "test.png"
        assert s2.calibration.pixels_per_unit == 10.0
        assert s2.results[0].area_uncertainty_vertex == 0.05
        assert s2.results[0].perimeter_physical == 4.0

    def test_save_load_file(self):
        s = Session(image_path="test.png")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        save_session(s, path)
        s2 = load_session(path)
        assert s2.image_path == "test.png"
        Path(path).unlink()

    def test_backward_compatible_load(self):
        """Old session without new fields should load with defaults."""
        old_data = {
            "version": "0.1.0",
            "image_path": "test.png",
            "calibration": {"unit": "mm", "pixels_per_unit": 10.0, "ppu_uncertainty": 0.1},
            "polygons": [],
            "results": [{
                "polygon_label": "poly_1",
                "area_pixels": 100.0,
                "area_physical": 1.0,
                "area_uncertainty": 0.1,
                "unit": "mm",
                "n_vertices": 4,
            }],
            "grid_results": [],
        }
        s = session_from_dict(old_data)
        assert s.results[0].area_uncertainty_vertex == 0.0
        assert s.results[0].perimeter_physical == 0.0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_session.py -v`
Expected: FAIL — new fields not serialized/deserialized

**Step 3: Implement**

Update `_result_to_dict` and `_result_from_dict` in `src/diemeter/session.py`:

```python
def _result_to_dict(r: MeasurementResult) -> Dict[str, Any]:
    return {
        "polygon_label": r.polygon_label,
        "area_pixels": r.area_pixels,
        "area_physical": r.area_physical,
        "area_uncertainty": r.area_uncertainty,
        "area_uncertainty_vertex": r.area_uncertainty_vertex,
        "area_uncertainty_ppu": r.area_uncertainty_ppu,
        "perimeter_physical": r.perimeter_physical,
        "perimeter_uncertainty": r.perimeter_uncertainty,
        "unit": r.unit.value,
        "n_vertices": r.n_vertices,
    }


def _result_from_dict(d: Dict[str, Any]) -> MeasurementResult:
    return MeasurementResult(
        polygon_label=d["polygon_label"],
        area_pixels=d["area_pixels"],
        area_physical=d["area_physical"],
        area_uncertainty=d["area_uncertainty"],
        area_uncertainty_vertex=d.get("area_uncertainty_vertex", 0.0),
        area_uncertainty_ppu=d.get("area_uncertainty_ppu", 0.0),
        perimeter_physical=d.get("perimeter_physical", 0.0),
        perimeter_uncertainty=d.get("perimeter_uncertainty", 0.0),
        unit=Unit(d["unit"]),
        n_vertices=d["n_vertices"],
    )
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_session.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/diemeter/session.py tests/test_session.py
git commit -m "feat(session): serialize new MeasurementResult fields with backward compat"
```

---

### Task 11: Update export with new fields

**Files:**
- Modify: `src/diemeter/export.py`
- Test: `tests/test_export.py` (new)

**Step 1: Write the failing test**

Create `tests/test_export.py`:

```python
"""Tests for export.py — CSV and JSON export."""

from __future__ import annotations

import csv
import json
from io import StringIO

import pytest

from diemeter.export import (
    export_results,
    results_to_csv,
    results_to_json,
)
from diemeter.model import MeasurementResult, Unit


def _make_result():
    return MeasurementResult(
        polygon_label="poly_1",
        area_pixels=10000.0,
        area_physical=100.0,
        area_uncertainty=5.0,
        area_uncertainty_vertex=3.0,
        area_uncertainty_ppu=4.0,
        perimeter_physical=40.0,
        perimeter_uncertainty=0.5,
        unit=Unit.MILLIMETER,
        n_vertices=4,
    )


class TestResultsToCSV:
    def test_header_includes_new_fields(self):
        csv_str = results_to_csv([_make_result()])
        reader = csv.reader(StringIO(csv_str))
        header = next(reader)
        assert "area_uncertainty_vertex" in header
        assert "area_uncertainty_ppu" in header
        assert "perimeter_physical" in header
        assert "perimeter_uncertainty" in header

    def test_values_correct(self):
        csv_str = results_to_csv([_make_result()])
        reader = csv.reader(StringIO(csv_str))
        next(reader)  # skip header
        row = next(reader)
        # Find index of perimeter_physical
        header = results_to_csv([_make_result()]).split("\n")[0].split(",")
        idx = [h.strip() for h in header].index("perimeter_physical")
        assert float(row[idx]) == pytest.approx(40.0, abs=0.01)

    def test_empty_results(self):
        assert results_to_csv([]) == ""


class TestResultsToJSON:
    def test_new_fields_present(self):
        json_str = results_to_json([_make_result()])
        data = json.loads(json_str)
        assert data[0]["area_uncertainty_vertex"] == 3.0
        assert data[0]["perimeter_physical"] == 40.0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_export.py -v`
Expected: FAIL — CSV header doesn't include new fields

**Step 3: Implement**

Update `results_to_csv` in `src/diemeter/export.py`:

```python
def results_to_csv(results: List[MeasurementResult]) -> str:
    """Serialize measurement results to CSV string."""
    if not results:
        return ""
    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "label",
        "area_pixels",
        "area_physical",
        "area_uncertainty",
        "area_uncertainty_vertex",
        "area_uncertainty_ppu",
        "perimeter_physical",
        "perimeter_uncertainty",
        "unit",
        "n_vertices",
    ])
    for r in results:
        writer.writerow([
            r.polygon_label,
            f"{r.area_pixels:.4f}",
            f"{r.area_physical:.6f}",
            f"{r.area_uncertainty:.6f}",
            f"{r.area_uncertainty_vertex:.6f}",
            f"{r.area_uncertainty_ppu:.6f}",
            f"{r.perimeter_physical:.6f}",
            f"{r.perimeter_uncertainty:.6f}",
            r.unit.value,
            r.n_vertices,
        ])
    return buf.getvalue()
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_export.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/diemeter/export.py tests/test_export.py
git commit -m "feat(export): include uncertainty decomposition and perimeter in CSV/JSON"
```

---

### Task 12: Create info panel (QDockWidget)

**Files:**
- Create: `src/diemeter/gui/info_panel.py`
- Modify: `src/diemeter/gui/app.py`

**Step 1: Implement InfoPanel**

Create `src/diemeter/gui/info_panel.py`:

```python
"""Right-side info panel for calibration details, polygon measurements, and grid info."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDockWidget,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..model import Calibration, MeasurementResult, Polygon, Session
from .i18n import tr

if TYPE_CHECKING:
    from .app import MeasurementApp


class InfoPanel(QDockWidget):
    """Dockable info panel showing calibration, polygon, and grid details."""

    def __init__(self, app: MeasurementApp) -> None:
        super().__init__("Info", app)
        self._app = app
        self.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        container = QWidget()
        self._layout = QVBoxLayout(container)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setSpacing(12)

        # Calibration section
        self._cal_label = QLabel()
        self._cal_label.setWordWrap(True)
        self._cal_label.setStyleSheet("color: #e0e0e0; font-size: 11px;")
        self._layout.addWidget(self._cal_label)

        # Polygon measurement section
        self._poly_label = QLabel()
        self._poly_label.setWordWrap(True)
        self._poly_label.setStyleSheet("color: #e0e0e0; font-size: 11px;")
        self._layout.addWidget(self._poly_label)

        # Vertex table
        self._vtx_table = QTableWidget()
        self._vtx_table.setColumnCount(7)
        self._vtx_table.setHorizontalHeaderLabels([
            "#", "x(px)", "y(px)", "x(phys)", "y(phys)", "snap", "\u03c3"
        ])
        self._vtx_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._vtx_table.setMaximumHeight(200)
        self._vtx_table.setStyleSheet(
            "QTableWidget { background: #2a2a2a; color: #e0e0e0; font-size: 10px; }"
            "QHeaderView::section { background: #333; color: #ccc; font-size: 10px; }"
        )
        self._layout.addWidget(self._vtx_table)

        # Grid section
        self._grid_label = QLabel()
        self._grid_label.setWordWrap(True)
        self._grid_label.setStyleSheet("color: #e0e0e0; font-size: 11px;")
        self._layout.addWidget(self._grid_label)

        self._layout.addStretch()

        container.setStyleSheet("background: #1e1e1e;")
        self.setWidget(container)
        self.setMinimumWidth(280)

    def refresh(
        self,
        session: Session,
        selected_polygon_idx: Optional[int] = None,
    ) -> None:
        """Update all sections from current session state."""
        self._update_calibration(session.calibration)
        self._update_polygon(session, selected_polygon_idx)
        self._update_grid(session)

    def _update_calibration(self, cal: Calibration) -> None:
        if not cal.is_calibrated:
            self._cal_label.setText(
                "<b>Calibration</b><br>Not calibrated"
            )
            return

        lines = [
            "<b>Calibration</b>",
            f"Scale: {cal.pixels_per_unit:.4f} px/{cal.unit.value} "
            f"\u00b1 {cal.ppu_uncertainty:.4f}",
            f"Method: {'line' if cal.lines else 'rect'}"
            f"{f' ({len(cal.lines)} lines)' if cal.lines else ''}",
        ]

        if cal.homography:
            lines.append("Perspective: corrected")

        self._cal_label.setText("<br>".join(lines))

    def _update_polygon(
        self, session: Session, idx: Optional[int]
    ) -> None:
        if idx is None or idx >= len(session.polygons):
            self._poly_label.setText("<b>Polygon</b><br>None selected")
            self._vtx_table.setRowCount(0)
            return

        poly = session.polygons[idx]
        result: Optional[MeasurementResult] = None
        for r in session.results:
            if r.polygon_label == poly.label:
                result = r
                break

        lines = [f"<b>Polygon: {poly.label}</b>"]

        if result is not None:
            rel_err = (
                (result.area_uncertainty / result.area_physical * 100)
                if result.area_physical > 0 else 0.0
            )
            u = result.unit.value
            lines.extend([
                f"Area: {result.area_physical:.6g} {u}\u00b2 "
                f"\u00b1 {result.area_uncertainty:.4g} ({rel_err:.2f}%)",
                f"  vertex \u03c3: {result.area_uncertainty_vertex:.4g} {u}\u00b2"
                f"  |  ppu \u03c3: {result.area_uncertainty_ppu:.4g} {u}\u00b2",
                f"Perimeter: {result.perimeter_physical:.6g} {u} "
                f"\u00b1 {result.perimeter_uncertainty:.4g}",
                f"Vertices: {result.n_vertices}  |  "
                f"Area(px): {result.area_pixels:.2f}",
            ])
        else:
            lines.append("Not measured (no calibration)")

        self._poly_label.setText("<br>".join(lines))

        # Vertex table
        cal = session.calibration
        self._vtx_table.setRowCount(poly.n_vertices)
        for i, v in enumerate(poly.vertices):
            self._vtx_table.setItem(i, 0, QTableWidgetItem(str(i)))
            self._vtx_table.setItem(i, 1, QTableWidgetItem(f"{v.x:.2f}"))
            self._vtx_table.setItem(i, 2, QTableWidgetItem(f"{v.y:.2f}"))
            if cal.is_calibrated:
                px = v.x / cal.pixels_per_unit
                py = v.y / cal.pixels_per_unit
                self._vtx_table.setItem(i, 3, QTableWidgetItem(f"{px:.4f}"))
                self._vtx_table.setItem(i, 4, QTableWidgetItem(f"{py:.4f}"))
            else:
                self._vtx_table.setItem(i, 3, QTableWidgetItem("-"))
                self._vtx_table.setItem(i, 4, QTableWidgetItem("-"))
            self._vtx_table.setItem(
                i, 5, QTableWidgetItem("\u2713" if v.snapped else "")
            )
            sigma_str = f"{v.confidence:.1f}" if v.snapped else "manual"
            self._vtx_table.setItem(i, 6, QTableWidgetItem(sigma_str))

    def _update_grid(self, session: Session) -> None:
        if not session.grid_results:
            self._grid_label.setText("<b>Grid</b><br>No detections")
            return

        lines = ["<b>Grid Detections</b>"]
        for i, g in enumerate(session.grid_results):
            u = g.unit.value
            lines.append(
                f"#{i+1}: pitch=({g.pitch_x_physical:.3g}, "
                f"{g.pitch_y_physical:.3g}) {u}  "
                f"angle={g.angle_deg:.1f}\u00b0  "
                f"conf={g.confidence:.3f}"
            )
        self._grid_label.setText("<br>".join(lines))
```

**Step 2: Integrate into app.py**

In `src/diemeter/gui/app.py`, add to `__init__` after the toolbar setup:

```python
from .info_panel import InfoPanel

# (inside __init__, after toolbar)
self._info_panel = InfoPanel(self)
self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._info_panel)
```

Add to the end of `redraw()`:

```python
self._info_panel.refresh(
    self.session,
    selected_polygon_idx=self.state.active_polygon_idx,
)
```

**Step 3: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add src/diemeter/gui/info_panel.py src/diemeter/gui/app.py
git commit -m "feat(gui): add dockable info panel with calibration, polygon, and grid details"
```

---

### Task 13: Update canvas polygon area label format

**Files:**
- Modify: `src/diemeter/gui/app.py:577-602`

**Step 1: Implement**

In `app.py`'s `redraw()`, update the polygon area_text formatting (around line 580):

```python
# Stored polygons
for i, poly in enumerate(self.session.polygons):
    area_text = None
    if poly.closed and self.session.calibration.is_calibrated:
        for r in self.session.results:
            if r.polygon_label == poly.label:
                rel_err = (
                    (r.area_uncertainty / r.area_physical * 100)
                    if r.area_physical > 0 else 0.0
                )
                area_text = (
                    f"{r.area_physical:.4g} {r.unit.value}\u00b2"
                    f" \u00b1{rel_err:.1f}%"
                )
                break
```

**Step 2: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add src/diemeter/gui/app.py
git commit -m "feat(gui): show relative error percentage in polygon canvas label"
```

---

### Task 14: Pass max_grad through to measure_polygon from app

**Files:**
- Modify: `src/diemeter/gui/app.py`

**Step 1: Implement**

Update `_remeasure_all` and `_remeasure_polygon` in `app.py` to pass `max_grad`:

```python
def _get_max_grad(self) -> float:
    """Get max gradient from edge detector, or default."""
    if self._edge_detector is not None and self._edge_detector.max_grad > 0:
        return self._edge_detector.max_grad
    return 255.0

def _remeasure_all(self) -> None:
    """Re-measure all polygons with current calibration."""
    self.session.results.clear()
    if not self.session.calibration.is_calibrated:
        return
    mg = self._get_max_grad()
    for poly in self.session.polygons:
        if poly.closed and poly.n_vertices >= 3:
            result = measure_polygon(poly, self.session.calibration, max_grad=mg)
            self.session.results.append(result)

def _remeasure_polygon(self, idx: int) -> None:
    """Re-measure a single polygon."""
    if not self.session.calibration.is_calibrated:
        return
    poly = self.session.polygons[idx]
    mg = self._get_max_grad()
    result = measure_polygon(poly, self.session.calibration, max_grad=mg)
    for i, r in enumerate(self.session.results):
        if r.polygon_label == poly.label:
            self.session.results[i] = result
            return
    self.session.results.append(result)
```

Also update `close_current_polygon`:

```python
if self.session.calibration.is_calibrated:
    mg = self._get_max_grad()
    result = measure_polygon(poly, self.session.calibration, max_grad=mg)
    self.session.results.append(result)
```

**Step 2: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add src/diemeter/gui/app.py
git commit -m "feat(gui): pass edge detector max_grad through measurement pipeline"
```

---

## Stream 4: Test Hardening (Agent 4)

### Task 15: Add model.py unit tests

**Files:**
- Create or extend: `tests/test_model.py`

**Step 1: Write tests**

Add to `tests/test_model.py` (already created in Task 1, extend):

```python
class TestUnitFromStr:
    def test_basic_units(self):
        assert unit_from_str("mm") == Unit.MILLIMETER
        assert unit_from_str("um") == Unit.MICROMETER
        assert unit_from_str("\u03bcm") == Unit.MICROMETER
        assert unit_from_str("cm") == Unit.CENTIMETER
        assert unit_from_str("m") == Unit.METER
        assert unit_from_str("in") == Unit.INCH
        assert unit_from_str("mil") == Unit.MIL
        assert unit_from_str("px") == Unit.PIXEL

    def test_case_insensitive(self):
        assert unit_from_str("MM") == Unit.MILLIMETER
        assert unit_from_str("Mm") == Unit.MILLIMETER

    def test_with_whitespace(self):
        assert unit_from_str("  mm  ") == Unit.MILLIMETER

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown unit"):
            unit_from_str("furlongs")


class TestConvertLength:
    def test_identity(self):
        assert convert_length(5.0, Unit.MILLIMETER, Unit.MILLIMETER) == 5.0

    def test_mm_to_cm(self):
        assert convert_length(10.0, Unit.MILLIMETER, Unit.CENTIMETER) == pytest.approx(1.0)

    def test_inch_to_mm(self):
        assert convert_length(1.0, Unit.INCH, Unit.MILLIMETER) == pytest.approx(25.4)

    def test_pixel_raises(self):
        with pytest.raises(ValueError, match="pixels"):
            convert_length(1.0, Unit.PIXEL, Unit.MILLIMETER)


class TestConvertArea:
    def test_identity(self):
        assert convert_area(5.0, Unit.MILLIMETER, Unit.MILLIMETER) == 5.0

    def test_mm2_to_cm2(self):
        assert convert_area(100.0, Unit.MILLIMETER, Unit.CENTIMETER) == pytest.approx(1.0)

    def test_pixel_raises(self):
        with pytest.raises(ValueError, match="pixels"):
            convert_area(1.0, Unit.PIXEL, Unit.MILLIMETER)
```

**Step 2: Run tests**

Run: `python -m pytest tests/test_model.py -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add tests/test_model.py
git commit -m "test(model): add unit_from_str, convert_length, convert_area tests"
```

---

### Task 16: Add CLI tests

**Files:**
- Create: `tests/test_cli.py`

**Step 1: Write tests**

```python
"""Tests for cli.py — argument parsing."""

from __future__ import annotations

import pytest

from diemeter.cli import build_parser, main


class TestBuildParser:
    def test_image_required(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_image_positional(self):
        parser = build_parser()
        args = parser.parse_args(["test.png"])
        assert args.image == "test.png"
        assert args.session is None
        assert args.edge_snap is False
        assert args.snap_radius == 15

    def test_all_options(self):
        parser = build_parser()
        args = parser.parse_args([
            "test.png", "-s", "session.json",
            "--edge-snap", "--snap-radius", "25",
        ])
        assert args.image == "test.png"
        assert args.session == "session.json"
        assert args.edge_snap is True
        assert args.snap_radius == 25


class TestMain:
    def test_missing_image(self, capsys):
        ret = main(["nonexistent_image_12345.png"])
        assert ret == 1
        captured = capsys.readouterr()
        assert "not found" in captured.err
```

**Step 2: Run tests**

Run: `python -m pytest tests/test_cli.py -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add tests/test_cli.py
git commit -m "test(cli): add argument parsing and missing image tests"
```

---

### Task 17: Add integration test with synthetic ground truth

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write test**

```python
"""Integration tests — end-to-end measurement with known ground truth."""

from __future__ import annotations

import math

import numpy as np
import pytest

from diemeter.calibration import calibrate_from_line, update_calibration_lines
from diemeter.measurement import measure_polygon
from diemeter.model import Calibration, Polygon, Unit, Vertex


class TestEndToEndMeasurement:
    def test_known_square_area(self):
        """Calibrate with known line, measure known square, verify area."""
        # Setup: 100px line = 10mm → ppu = 10 px/mm
        cal = Calibration(unit=Unit.MILLIMETER)
        line = calibrate_from_line((0, 0), (100, 0), 10.0, Unit.MILLIMETER)
        cal.lines.append(line)
        update_calibration_lines(cal)

        assert cal.pixels_per_unit == pytest.approx(10.0)

        # Measure a 200x200 px square → should be 20x20 mm = 400 mm²
        poly = Polygon(
            vertices=[
                Vertex(x=0, y=0),
                Vertex(x=200, y=0),
                Vertex(x=200, y=200),
                Vertex(x=0, y=200),
            ],
            label="square",
            closed=True,
        )
        result = measure_polygon(poly, cal)
        assert result.area_physical == pytest.approx(400.0)
        assert result.perimeter_physical == pytest.approx(80.0)
        assert result.unit == Unit.MILLIMETER

    def test_uncertainty_increases_with_vertex_noise(self):
        """Snapped vertices (high confidence) should have lower uncertainty than manual."""
        cal = Calibration(
            pixels_per_unit=10.0,
            ppu_uncertainty=0.1,
            unit=Unit.MILLIMETER,
        )

        # Manual vertices (no snap)
        poly_manual = Polygon(
            vertices=[
                Vertex(x=0, y=0, snapped=False),
                Vertex(x=200, y=0, snapped=False),
                Vertex(x=200, y=200, snapped=False),
                Vertex(x=0, y=200, snapped=False),
            ],
            label="manual",
            closed=True,
        )

        # Snapped vertices (high confidence)
        poly_snapped = Polygon(
            vertices=[
                Vertex(x=0, y=0, snapped=True, confidence=200.0),
                Vertex(x=200, y=0, snapped=True, confidence=200.0),
                Vertex(x=200, y=200, snapped=True, confidence=200.0),
                Vertex(x=0, y=200, snapped=True, confidence=200.0),
            ],
            label="snapped",
            closed=True,
        )

        r_manual = measure_polygon(poly_manual, cal)
        r_snapped = measure_polygon(poly_snapped, cal)

        # Same area
        assert r_manual.area_physical == pytest.approx(r_snapped.area_physical)
        # Snapped should have lower vertex uncertainty
        assert r_snapped.area_uncertainty_vertex < r_manual.area_uncertainty_vertex

    def test_multi_line_calibration_consistency(self):
        """Multiple calibration lines should improve estimate."""
        from diemeter.calibration import chi_squared_consistency

        cal = Calibration(unit=Unit.MILLIMETER)
        # Three lines all giving ~10 px/mm
        for length_px, length_mm in [(100, 10.0), (200, 20.0), (150, 15.0)]:
            line = calibrate_from_line((0, 0), (length_px, 0), length_mm, Unit.MILLIMETER)
            cal.lines.append(line)
        update_calibration_lines(cal)

        assert cal.pixels_per_unit == pytest.approx(10.0, rel=0.01)
        chi2, p_value, consistent = chi_squared_consistency(cal.lines)
        assert consistent is True
```

**Step 2: Run tests**

Run: `python -m pytest tests/test_integration.py -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add end-to-end integration tests with ground truth verification"
```

---

## Execution Order Summary

**Phase 1 (Sequential — model changes):**
- Task 1: Extend MeasurementResult
- Task 2: confidence_to_sigma
- Task 3: perimeter_uncertainty
- Task 4: Connect full uncertainty in measure_polygon

**Phase 2 (Parallel — after Phase 1):**
- Agent A: Task 5, Task 6 (calibration improvements)
- Agent B: Task 7, Task 8, Task 9 (performance)
- Agent C: Task 10, Task 11, Task 12, Task 13, Task 14 (display/interaction)
- Agent D: Task 15, Task 16, Task 17 (tests)

**Total: 17 tasks, ~34 commits**
