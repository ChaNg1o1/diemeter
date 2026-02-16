# Production-Grade Optimization Design

**Date:** 2026-02-16
**Status:** Approved
**Approach:** Incremental refactoring with 4 parallel agents

## Context

diemeter v0.1.0 has complete core algorithms (Shoelace area, Devernay sub-pixel edge detection, FFT grid detection, analytic uncertainty propagation) but critical gaps prevent production use:

1. **Uncertainty propagation disconnected** -- `uncertainty.py` implements full vertex+PPU propagation but `measure_polygon()` only uses PPU-only inline formula. User-facing uncertainty is systematically underestimated.
2. **Insufficient numerical detail** -- Perimeter, vertex coordinates, uncertainty decomposition, calibration diagnostics are all computed but never displayed.
3. **Rough interaction** -- Modal calibration dialogs, no input validation feedback, incomplete undo, full-scene redraw on every event.
4. **Test coverage gaps** -- session, export, CLI, model, GUI have zero tests.

## Design

### Section 1: Measurement Precision

#### 1.1 Connect Full Uncertainty Propagation

- `measure_polygon()` calls `total_area_uncertainty(polygon, calibration, vertex_sigma)` instead of inline PPU-only formula.
- `MeasurementResult` extended with: `area_uncertainty_vertex`, `area_uncertainty_ppu`, `perimeter_physical`, `perimeter_uncertainty`.
- New `perimeter_uncertainty()` function: partial derivatives of edge lengths w.r.t. vertex positions, propagated in quadrature.

#### 1.2 Devernay Confidence to Vertex Sigma

- New `confidence_to_sigma(confidence, max_grad) -> float`:
  - Manual click (no snap): sigma = 1.0 px
  - Snap + low confidence: sigma ~ 0.5 px
  - Snap + high confidence: sigma ~ 0.1 px
  - Formula: `sigma = max(0.1, 1.0 - 0.9 * (confidence / max_grad))`
- `total_area_uncertainty()` receives per-vertex sigma array constructed from each vertex's snap state.

#### 1.3 Calibration Uncertainty Improvement

- Rectangle calibration: replace `reproj_err / avg_dim` heuristic with corner localization uncertainty propagation (±sigma_corner per corner, Monte Carlo or analytic propagation to ppu).
- Line calibration: add chi-squared consistency check across multiple lines. Warn user when ppu dispersion exceeds expected range.

### Section 2: Numerical Display

#### 2.1 Info Panel (New QDockWidget)

Right-side dockable/collapsible panel with three sections:

**Calibration info:**
- Scale: `{ppu:.4f} px/{unit} +/- {unc:.4f}`
- Method: line/rect, line count
- Multi-line consistency: chi-squared value + pass/warn
- Rect calibration reprojection error

**Selected polygon details:**
- Area: `{area:.6g} {unit}^2 +/- {total_unc:.4g}`
- Uncertainty decomposition: vertex contribution vs PPU contribution
- Perimeter: `{perim:.6g} {unit} +/- {perim_unc:.4g}`
- Relative error: `+/-{rel_err:.2f}%`
- Vertex table: index | x(px) | y(px) | x(phys) | y(phys) | snap | confidence | sigma

**Grid detection:**
- pitch_x / pitch_y (physical + pixels)
- Angle, confidence, autocorrelation peak strength

#### 2.2 Canvas Annotation Enhancement

- Polygon label format: `{area:.4g} {unit}^2 +/-{rel_err:.1f}%`
- Optional display: vertex numbers, edge length labels
- Calibration line residuals (per-line ppu deviation %)

#### 2.3 Export Enhancement

- CSV/JSON: add perimeter, uncertainty decomposition, vertex list, calibration metadata
- PDF report export (future iteration)

### Section 3: Interaction Optimization

#### 3.1 Calibration Flow Redesign

- Replace modal QInputDialog with inline input bar (QWidget in status area)
- Flow: draw line/rect -> input bar appears -> enter length + select unit -> Enter confirm / Esc cancel
- Real-time input validation: red border + tooltip on invalid input
- Multi-line: real-time ppu refresh and consistency check on each new line

#### 3.2 Complete Undo System

- Command pattern: each operation as `(do, undo)` closure pair
- Covers: vertex move, vertex add/delete, polygon delete, calibration changes, grid detection
- Redo support (Ctrl+Shift+Z)

#### 3.3 Incremental Redraw

- Each polygon/calibration/grid maintains its own QGraphicsItem reference
- State changes update only affected items (`update()` not delete+recreate)
- Rubber-band / snap marker as persistent items controlled via `setVisible()` / `setPos()`
- Redraw cost: O(changed_items) instead of O(all_items)

#### 3.4 Performance

**Edge snap -- KD-tree:**
- Build `scipy.spatial.cKDTree` after `compute_subpixel()`
- `snap_to_edge()` via `tree.query_ball_point(xy, radius)` + confidence sort
- O(log n) per query instead of O(n)

**Devernay sub-pixel -- vectorization:**
- Replace Python loop with NumPy vectorized operations
- `np.where` for edge coordinates, batch bilinear sampling, batch parabolic fitting
- Expected 10-50x speedup

**Edge detection async:**
- `EdgeDetector.compute_subpixel()` in QThread / concurrent.futures worker
- Background precomputation on image load; snap disabled with progress indicator until ready
- UI thread never blocked

### Section 4: Test Hardening

#### 4.1 New Tests

| Module | Coverage |
|--------|----------|
| `model.py` | `convert_length/area` full unit matrix, `unit_from_str` edge cases, `Vertex/Polygon` construction |
| `session.py` | Roundtrip (save -> load -> compare), missing field tolerance, version migration |
| `export.py` | CSV/JSON output content verification, empty results |
| `cli.py` | Argument parsing, file-not-found error |
| `measurement.py` | Full uncertainty propagation (new logic), perimeter uncertainty |
| `uncertainty.py` | `confidence_to_sigma` mapping, per-vertex sigma array |
| `calibration.py` | Chi-squared consistency check, rect calibration corner uncertainty |

#### 4.2 Integration Tests

- End-to-end headless: load image -> calibrate -> draw polygon -> measure -> export -> verify values
- Ground truth validation with synthetic images (checkerboard, known-size rectangles)

## Agent Parallelization Strategy

```
Agent 1 (Measurement Precision)    Agent 2 (Numerical Display)
  measurement.py                     gui/info_panel.py (new)
  uncertainty.py                     gui/items.py (annotation)
  calibration.py                     export.py
  edge.py (confidence->sigma)        model.py (Result extension)

Agent 3 (Interaction Optimization)  Agent 4 (Test Hardening)
  gui/app.py (undo/redraw)           tests/test_session.py (new)
  gui/handlers.py                    tests/test_export.py (new)
  gui/toolbar.py                     tests/test_cli.py (new)
  gui/canvas.py                      tests/test_model.py (new)
  edge.py (KD-tree/vectorize)        tests/ (update existing)
```

**Dependencies:** Agent 1 goes first (model extensions). Agent 2 and 3 run in parallel after Agent 1's model changes land. Agent 4 runs fully in parallel throughout.
