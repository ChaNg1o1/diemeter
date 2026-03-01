"""Tests for headless diemeter engine controller."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from diemeter.engine.controller import EngineController
from diemeter.gui.modes import Mode
from diemeter.model import Polygon, Vertex


def test_line_calibration_prompt_roundtrip() -> None:
    c = EngineController()

    c.switch_mode(Mode.CALIBRATE_LINE)
    c.pointer_event("press", "left", 0.0, 0.0)
    c.pointer_event("press", "left", 100.0, 0.0)

    assert c.snapshot()["pending_prompt"]["type"] == "line_distance"

    c.submit_prompt("line_distance", "10 mm")

    assert c.session.calibration.is_calibrated
    assert c.session.calibration.pixels_per_unit == pytest.approx(10.0)
    assert c.snapshot()["pending_prompt"] is None
    assert c.state.mode == Mode.VIEW


def test_context_actions_for_selected_polygon() -> None:
    c = EngineController()
    poly = Polygon(
        vertices=[
            Vertex(x=0.0, y=0.0),
            Vertex(x=100.0, y=0.0),
            Vertex(x=100.0, y=80.0),
            Vertex(x=0.0, y=80.0),
        ],
        label="poly_1",
        closed=True,
    )
    c.session.polygons.append(poly)
    c.switch_mode(Mode.VIEW)

    actions = c.context_actions(50.0, 0.5)
    action_ids = {a["id"] for a in actions}

    assert "edit_polygon" in action_ids
    assert "delete_polygon" in action_ids


def test_unknown_rpc_method_raises() -> None:
    c = EngineController()
    with pytest.raises(ValueError, match="Unknown method"):
        c.handle("unknown.method", {})


def _make_periodic_image(pitch_px: int = 50, size: int = 400) -> np.ndarray:
    """Create a synthetic image with vertical stripes at a given pitch."""
    img = np.zeros((size, size), dtype=np.uint8)
    for col in range(size):
        if (col % pitch_px) < (pitch_px // 4):
            img[:, col] = 255
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def test_pad_pitch_calibration_full_flow() -> None:
    """Integration test: open image → pad pitch mode → ROI → prompt → calibrated."""
    pitch_px = 50
    img = _make_periodic_image(pitch_px=pitch_px, size=400)

    # Write to temp file so engine can open it.
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        cv2.imwrite(f.name, img)
        img_path = f.name

    try:
        c = EngineController()
        c.open_image(img_path)

        # Switch to pad pitch mode.
        c.switch_mode(Mode.CALIBRATE_PAD_PITCH)
        assert c.state.mode == Mode.CALIBRATE_PAD_PITCH

        # Drag ROI covering the full image.
        c.pointer_event("press", "left", 0.0, 0.0)
        c.pointer_event("move", "left", 400.0, 400.0)
        c.pointer_event("release", "left", 400.0, 400.0)

        # Should have a pending pad_pitch prompt.
        snap = c.snapshot()
        assert snap["pending_prompt"] is not None
        assert snap["pending_prompt"]["type"] == "pad_pitch"

        # Submit known pitch: e.g. 80 um per pad.
        c.submit_prompt("pad_pitch", "80 um")

        # Verify calibration result.
        assert c.session.calibration.is_calibrated
        ppu = c.session.calibration.pixels_per_unit
        expected_ppu = pitch_px / 80.0  # px/um
        assert ppu == pytest.approx(expected_ppu, rel=0.05)

        # Verify the line source is pad_pitch.
        assert len(c.session.calibration.lines) == 1
        assert c.session.calibration.lines[0].source == "pad_pitch"
        assert c.session.calibration.lines[0].endpoint_sigma == pytest.approx(0.2)

        # Mode should return to VIEW.
        assert c.state.mode == Mode.VIEW

        # A grid result should have been stored.
        assert len(c.session.grid_results) >= 1
    finally:
        Path(img_path).unlink(missing_ok=True)


def test_pad_pitch_no_pattern_returns_no_prompt() -> None:
    """If no periodic pattern is detected, no prompt should be set."""
    img = np.full((200, 200, 3), 128, dtype=np.uint8)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        cv2.imwrite(f.name, img)
        img_path = f.name

    try:
        c = EngineController()
        c.open_image(img_path)

        c.switch_mode(Mode.CALIBRATE_PAD_PITCH)
        c.pointer_event("press", "left", 0.0, 0.0)
        c.pointer_event("release", "left", 200.0, 200.0)

        # No periodic pattern → no prompt, mode returns to VIEW.
        assert c.snapshot()["pending_prompt"] is None
        assert c.state.mode == Mode.VIEW
    finally:
        Path(img_path).unlink(missing_ok=True)


def test_pad_pitch_context_actions_include_cancel() -> None:
    """CALIBRATE_PAD_PITCH mode should offer a cancel context action."""
    c = EngineController()
    c.switch_mode(Mode.CALIBRATE_PAD_PITCH)
    actions = c.context_actions(50.0, 50.0)
    assert any(a["id"] == "cancel" for a in actions)
