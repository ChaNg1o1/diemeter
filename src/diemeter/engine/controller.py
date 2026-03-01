"""Headless controller that exposes diemeter behavior without Qt."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..calibration import (
    DEFAULT_LINE_ENDPOINT_SIGMA_PX,
    DEFAULT_LINE_HUBER_K,
    DEFAULT_LINE_IRLS_MAX_ITERS,
    DEFAULT_LINE_IRLS_TOL,
    calibrate_from_line,
    calibrate_from_pad_pitch,
    calibrate_from_rect,
    chi_squared_consistency,
    update_calibration_lines,
    update_calibration_rect,
)
from ..edge import EdgeDetector
from ..export import export_results
from ..grid import detect_grid
from ..measurement import measure_polygon, polygon_area_pixels
from ..model import (
    Calibration,
    GridResult,
    Polygon,
    Session,
    Unit,
    Vertex,
    convert_area,
    convert_length,
    unit_from_str,
)
from ..session import load_session, save_session
from ..gui.i18n import get_language, set_language, tr
from ..gui.modes import Mode, ModeState

_LINE_CAL_CONFIDENCE_LEVEL = 0.95
_LINE_CAL_WARN_REL_UNC_PERCENT = 1.0
_LINE_CAL_MIN_RECOMMENDED_PIXELS = 180.0


class EngineController:
    """Headless application controller for non-Qt clients."""

    def __init__(self) -> None:
        self.image_path: str = ""
        self.original_image: Optional[np.ndarray] = None
        self.display_image: Optional[np.ndarray] = None
        self._display_image_path: str = ""
        self._warped_image: Optional[np.ndarray] = None

        self.session = Session()
        self.state = ModeState()
        self._current_polygon: Optional[Polygon] = None
        self._polygon_counter = 0

        self._edge_detector: Optional[EdgeDetector] = None
        self._undo_stack: List[Tuple[str, Any]] = []
        self._pending_prompt: Optional[Dict[str, Any]] = None
        self._last_warning: Optional[str] = None

    # ---------------------------------------------------------------------
    # RPC entry
    # ---------------------------------------------------------------------

    def handle(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = params or {}

        if method == "engine.ping":
            return self._with_snapshot({"ok": True})

        if method == "engine.initialize":
            self.initialize(
                language=params.get("language"),
                display_unit=params.get("display_unit"),
                edge_snap_enabled=params.get("edge_snap_enabled"),
                snap_radius=params.get("snap_radius"),
            )
            return self._with_snapshot({"ok": True})

        if method == "image.open":
            path = params.get("path")
            if not path:
                raise ValueError("image.open requires 'path'")
            self.open_image(path)
            return self._with_snapshot({"ok": True})

        if method == "session.new":
            self.new_session()
            return self._with_snapshot({"ok": True})

        if method == "session.load":
            path = params.get("path")
            if not path:
                raise ValueError("session.load requires 'path'")
            self.load_session_path(path)
            return self._with_snapshot({"ok": True})

        if method == "session.save":
            path = params.get("path")
            if not path:
                raise ValueError("session.save requires 'path'")
            self.save_session_path(path)
            return self._with_snapshot({"ok": True, "path": str(path)})

        if method == "session.export":
            path = params.get("path")
            if not path:
                raise ValueError("session.export requires 'path'")
            self.export_results_path(path)
            return self._with_snapshot({"ok": True, "path": str(path)})

        if method == "mode.set":
            mode = params.get("mode")
            if not mode:
                raise ValueError("mode.set requires 'mode'")
            self.switch_mode(Mode(mode))
            return self._with_snapshot({"ok": True})

        if method == "settings.set":
            self.set_settings(
                edge_snap_enabled=params.get("edge_snap_enabled"),
                snap_radius=params.get("snap_radius"),
                display_unit=params.get("display_unit"),
                language=params.get("language"),
            )
            return self._with_snapshot({"ok": True})

        if method == "input.pointer":
            self.pointer_event(
                phase=params.get("phase"),
                button=params.get("button"),
                x=params.get("x"),
                y=params.get("y"),
            )
            return self._with_snapshot({"ok": True})

        if method == "input.key":
            action = params.get("action")
            if not action:
                raise ValueError("input.key requires 'action'")
            self.key_action(action)
            return self._with_snapshot({"ok": True})

        if method == "input.context_actions":
            x = params.get("x")
            y = params.get("y")
            if x is None or y is None:
                raise ValueError("input.context_actions requires x/y")
            actions = self.context_actions(float(x), float(y))
            return self._with_snapshot({"actions": actions})

        if method == "action.invoke":
            action = params.get("action")
            if not action:
                raise ValueError("action.invoke requires 'action'")
            self.invoke_action(
                action=str(action),
                polygon_index=params.get("polygon_index"),
            )
            return self._with_snapshot({"ok": True})

        if method == "prompt.submit":
            prompt_type = params.get("type")
            value = params.get("value")
            if not prompt_type:
                raise ValueError("prompt.submit requires 'type'")
            self.submit_prompt(prompt_type=str(prompt_type), value=value)
            return self._with_snapshot({"ok": True})

        if method == "snapshot.get":
            return {"snapshot": self.snapshot()}

        raise ValueError(f"Unknown method: {method}")

    # ---------------------------------------------------------------------
    # Session/image
    # ---------------------------------------------------------------------

    def initialize(
        self,
        language: Optional[str] = None,
        display_unit: Optional[str] = None,
        edge_snap_enabled: Optional[bool] = None,
        snap_radius: Optional[int] = None,
    ) -> None:
        if language is not None:
            set_language(str(language))

        if display_unit:
            self.set_display_unit(unit_from_str(display_unit))

        if edge_snap_enabled is not None:
            self.session.edge_snap_enabled = bool(edge_snap_enabled)

        if snap_radius is not None:
            self.session.edge_snap_radius = int(snap_radius)

    def open_image(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {path}")

        self.image_path = str(p)
        self.original_image = img
        self.display_image = img
        self._display_image_path = str(p)
        self._warped_image = None

        self.session = Session(image_path=str(p))
        self._ensure_valid_display_unit()

        self.state = ModeState()
        self._current_polygon = None
        self._polygon_counter = 0
        self._edge_detector = None
        self._undo_stack = []
        self._pending_prompt = None
        self._last_warning = None

    def new_session(self) -> None:
        if not self.image_path or self.original_image is None:
            self.session = Session()
            self._current_polygon = None
            self.state = ModeState()
            self._pending_prompt = None
            return

        prev_unit = self.session.display_unit
        prev_snap = self.session.edge_snap_enabled
        prev_radius = self.session.edge_snap_radius

        self.session = Session(
            image_path=self.image_path,
            display_unit=prev_unit,
            edge_snap_enabled=prev_snap,
            edge_snap_radius=prev_radius,
        )
        self._ensure_valid_display_unit()
        self.state = ModeState()
        self._current_polygon = None
        self._polygon_counter = 0
        self._undo_stack = []
        self._pending_prompt = None
        self.display_image = self.original_image
        self._display_image_path = self.image_path
        self._warped_image = None

    def load_session_path(self, path: str) -> None:
        session = load_session(path)
        if not session.image_path:
            raise ValueError("Session does not contain image_path")

        self.open_image(session.image_path)
        self.session = session
        self._ensure_valid_display_unit()

        # Keep polygon counter monotonic.
        self._polygon_counter = max(
            [self._extract_polygon_counter(p.label) for p in self.session.polygons] + [0]
        )

        # Recreate warped display if rect calibration is present.
        if self.session.calibration.rect is not None and self.original_image is not None:
            try:
                cal, warped = update_calibration_rect(
                    self.session.calibration,
                    image=self.original_image,
                    dst_scale=max(10.0, self.session.calibration.pixels_per_unit or 10.0),
                )
                self.session.calibration = cal
                if warped is not None:
                    self._set_warped_display(warped)
            except Exception:
                # Preserve load robustness. Keep original image if warp fails.
                self.display_image = self.original_image
                self._display_image_path = self.image_path
                self._warped_image = None

        self._edge_detector = None
        self._pending_prompt = None
        self.state = ModeState()
        self._remeasure_all()

    def save_session_path(self, path: str) -> None:
        save_session(self.session, path)

    def export_results_path(self, path: str) -> None:
        export_results(
            path,
            self.session.results,
            self._grid_results_for_current_unit() or None,
        )

    # ---------------------------------------------------------------------
    # Settings/mode
    # ---------------------------------------------------------------------

    def set_settings(
        self,
        edge_snap_enabled: Optional[bool] = None,
        snap_radius: Optional[int] = None,
        display_unit: Optional[str] = None,
        language: Optional[str] = None,
    ) -> None:
        if edge_snap_enabled is not None:
            self.session.edge_snap_enabled = bool(edge_snap_enabled)
            if self.session.edge_snap_enabled:
                self._ensure_edge_detector()

        if snap_radius is not None:
            radius = int(snap_radius)
            if radius <= 0:
                raise ValueError("snap_radius must be positive")
            self.session.edge_snap_radius = radius

        if display_unit is not None:
            self.set_display_unit(unit_from_str(display_unit))

        if language is not None:
            set_language(language)

    def switch_mode(self, mode: Mode) -> None:
        if (
            self.state.mode == Mode.POLYGON_DRAW
            and mode != Mode.POLYGON_DRAW
            and self._current_polygon is not None
            and self._current_polygon.n_vertices >= 3
        ):
            self.close_current_polygon()

        self.state.switch_to(mode)

        if mode == Mode.POLYGON_DRAW:
            self._polygon_counter += 1
            self._current_polygon = Polygon(label=f"poly_{self._polygon_counter}")
        elif mode == Mode.POLYGON_EDIT:
            if self.session.polygons:
                self.state.active_polygon_idx = len(self.session.polygons) - 1
        else:
            if mode != Mode.POLYGON_DRAW:
                self._current_polygon = None

    def set_display_unit(self, unit: Unit) -> None:
        if unit == Unit.PIXEL:
            return
        if self.session.display_unit == unit:
            return
        self.session.display_unit = unit
        self._remeasure_all()

    # ---------------------------------------------------------------------
    # Inputs
    # ---------------------------------------------------------------------

    def pointer_event(
        self,
        phase: Optional[str],
        button: Optional[str],
        x: Optional[float],
        y: Optional[float],
    ) -> None:
        if phase is None:
            raise ValueError("input.pointer requires phase")
        if x is None or y is None:
            raise ValueError("input.pointer requires x/y")

        phase = str(phase)
        button = str(button or "left")
        px = float(x)
        py = float(y)

        if phase == "move":
            self._on_move(px, py)
            return

        if phase == "release":
            self._on_release(button, px, py)
            return

        if phase == "double_click":
            self._on_double_click(button, px, py)
            return

        if phase != "press":
            raise ValueError(f"Unknown pointer phase: {phase}")

        if button == "right":
            # Right-click actions are queried via input.context_actions.
            self.state.cursor_pos = (px, py)
            return

        if button != "left":
            self.state.cursor_pos = (px, py)
            return

        snapped_x, snapped_y, snapped, conf = self.maybe_snap(px, py)
        self.state.cursor_pos = (px, py)

        mode = self.state.mode
        if mode == Mode.VIEW:
            self.state.active_polygon_idx = self.find_nearest_polygon(snapped_x, snapped_y)
            return

        if mode == Mode.CALIBRATE_LINE:
            self.state.add_point(snapped_x, snapped_y)
            if len(self.state.points) == 2:
                self._pending_prompt = {
                    "type": "line_distance",
                    "title": tr("dialog.calibrate_line_title"),
                    "message": tr("dialog.enter_distance"),
                }
            return

        if mode == Mode.CALIBRATE_RECT:
            self.state.add_point(snapped_x, snapped_y)
            if len(self.state.points) == 4:
                self._pending_prompt = {
                    "type": "rect_size",
                    "title": tr("dialog.calibrate_rect_title"),
                    "message": tr("dialog.enter_rect_dim"),
                }
            return

        if mode == Mode.POLYGON_DRAW:
            self.add_polygon_vertex(snapped_x, snapped_y, snapped=snapped, confidence=conf)
            return

        if mode == Mode.POLYGON_EDIT:
            vi = self.find_nearest_vertex(snapped_x, snapped_y)
            if vi is not None:
                self.state.dragging_vertex_idx = vi
            else:
                poly_idx = self.find_nearest_polygon(snapped_x, snapped_y)
                if poly_idx is not None and poly_idx != self.state.active_polygon_idx:
                    self.state.active_polygon_idx = poly_idx
            return

        if mode == Mode.GRID_DETECT:
            self.state.roi_start = (snapped_x, snapped_y)
            self.state.roi_end = None

        if mode == Mode.CALIBRATE_PAD_PITCH:
            self.state.roi_start = (snapped_x, snapped_y)
            self.state.roi_end = None

    def key_action(self, action: str) -> None:
        action = str(action)
        if action == "undo":
            self.undo()
            return
        if action == "delete_vertex":
            self.delete_nearest_vertex()
            return
        if action == "insert_vertex":
            self.insert_vertex_on_edge()
            return
        if action == "enter":
            if self.state.mode == Mode.POLYGON_DRAW:
                self.close_current_polygon()
            elif self.state.mode == Mode.POLYGON_EDIT:
                self.switch_mode(Mode.VIEW)
            return
        if action == "escape":
            self.switch_mode(Mode.VIEW)
            return
        raise ValueError(f"Unknown key action: {action}")

    def context_actions(self, x: float, y: float) -> List[Dict[str, Any]]:
        mode = self.state.mode
        actions: List[Dict[str, Any]] = []

        if mode == Mode.VIEW:
            poly_idx = self.find_nearest_polygon(x, y)
            if poly_idx is not None:
                actions.append({
                    "id": "edit_polygon",
                    "label": tr("ctx.edit_polygon"),
                    "polygon_index": poly_idx,
                })
                actions.append({
                    "id": "delete_polygon",
                    "label": tr("ctx.delete_polygon"),
                    "polygon_index": poly_idx,
                })

        elif mode == Mode.POLYGON_DRAW:
            poly = self._current_polygon
            if poly is not None and poly.n_vertices >= 3:
                actions.append({"id": "close_polygon", "label": tr("ctx.close_polygon")})
            actions.append({"id": "cancel", "label": tr("ctx.cancel")})

        elif mode == Mode.POLYGON_EDIT:
            if self.state.hover_vertex_idx is not None:
                actions.append({"id": "delete_vertex", "label": tr("ctx.delete_vertex")})
            actions.append({"id": "insert_vertex", "label": tr("ctx.insert_vertex")})
            actions.append({"id": "done_editing", "label": tr("ctx.done_editing")})

        elif mode in (Mode.CALIBRATE_LINE, Mode.CALIBRATE_RECT, Mode.GRID_DETECT, Mode.CALIBRATE_PAD_PITCH):
            actions.append({"id": "cancel", "label": tr("ctx.cancel")})

        return actions

    def invoke_action(self, action: str, polygon_index: Optional[int] = None) -> None:
        if action == "edit_polygon":
            if polygon_index is None:
                raise ValueError("edit_polygon requires polygon_index")
            self.state.active_polygon_idx = int(polygon_index)
            self.switch_mode(Mode.POLYGON_EDIT)
            return
        if action == "delete_polygon":
            if polygon_index is None:
                raise ValueError("delete_polygon requires polygon_index")
            self.delete_polygon(int(polygon_index))
            return
        if action == "close_polygon":
            self.close_current_polygon()
            return
        if action == "cancel":
            self.switch_mode(Mode.VIEW)
            return
        if action == "delete_vertex":
            self.delete_nearest_vertex()
            return
        if action == "insert_vertex":
            self.insert_vertex_on_edge()
            return
        if action == "done_editing":
            self.switch_mode(Mode.VIEW)
            return
        raise ValueError(f"Unknown action: {action}")

    def submit_prompt(self, prompt_type: str, value: Any) -> None:
        pending = self._pending_prompt
        if pending is None:
            raise ValueError("No pending prompt")

        expected = pending.get("type")
        if expected != prompt_type:
            raise ValueError(f"Unexpected prompt type: {prompt_type!r}, expected {expected!r}")

        if prompt_type == "line_distance":
            text = str(value or "").strip()
            length, unit = self._parse_length_input(text)
            if length is None:
                raise ValueError("Invalid line distance. Example: '14.5 mm'")

            pts = self.state.points
            if len(pts) != 2:
                raise ValueError("Line calibration requires exactly 2 points")

            line = calibrate_from_line(pts[0], pts[1], length, unit)
            self._undo_stack.append(("add_cal_line", len(self.session.calibration.lines)))
            self.session.calibration.lines.append(line)
            self.session.calibration.unit = unit
            update_calibration_lines(
                self.session.calibration,
                sigma_per_pixel=DEFAULT_LINE_ENDPOINT_SIGMA_PX,
                robust=True,
                huber_k=DEFAULT_LINE_HUBER_K,
                max_irls_iters=DEFAULT_LINE_IRLS_MAX_ITERS,
                irls_tol=DEFAULT_LINE_IRLS_TOL,
            )
            self._last_warning = self._line_calibration_quality_warning_text()

            self.state.switch_to(Mode.VIEW)
            self._pending_prompt = None
            self._remeasure_all()
            return

        if prompt_type == "rect_size":
            text = str(value or "").strip()
            w, h, unit = self._parse_rect_input(text)
            if w is None or h is None:
                raise ValueError("Invalid rectangle size. Example: '10 x 8 mm'")

            pts = self.state.points
            if len(pts) != 4:
                raise ValueError("Rectangle calibration requires exactly 4 points")

            rect = calibrate_from_rect(pts, w, h, unit)
            self.session.calibration.rect = rect
            self.session.calibration.unit = unit

            self._pending_prompt = {
                "type": "perspective_confirm",
                "title": tr("dialog.perspective_title"),
                "message": tr("dialog.perspective_warp"),
                "default": True,
            }
            return

        if prompt_type == "perspective_confirm":
            apply_warp = bool(value)
            if apply_warp:
                if self.original_image is None:
                    raise ValueError("No image loaded")
                cal, warped = update_calibration_rect(
                    self.session.calibration,
                    image=self.original_image,
                    dst_scale=max(10.0, self.session.calibration.pixels_per_unit or 10.0),
                )
                self.session.calibration = cal
                if warped is not None:
                    self._set_warped_display(warped)
            else:
                update_calibration_rect(self.session.calibration, dst_scale=10.0)
                # Keep the original image when user rejects warp.
                self.display_image = self.original_image
                self._display_image_path = self.image_path
                self._warped_image = None
            self._last_warning = None

            self.state.switch_to(Mode.VIEW)
            self._pending_prompt = None
            self._remeasure_all()
            return

        if prompt_type == "pad_pitch":
            text = str(value or "").strip()
            length, unit = self._parse_length_input(text)
            if length is None:
                raise ValueError("Invalid pad pitch. Example: '80 um'")

            roi = pending.get("_roi")
            if roi is None:
                raise ValueError("No ROI stored for pad pitch calibration")

            image = self._warped_image if self._warped_image is not None else self.original_image
            if image is None:
                raise ValueError("No image loaded")

            result = calibrate_from_pad_pitch(image, roi=tuple(roi), known_pitch=length, unit=unit)
            if result is None:
                raise ValueError("FFT detection failed on re-run")

            line, grid_res = result
            self._undo_stack.append(("add_cal_line", len(self.session.calibration.lines)))
            self.session.calibration.lines.append(line)
            self.session.calibration.unit = unit
            update_calibration_lines(
                self.session.calibration,
                sigma_per_pixel=DEFAULT_LINE_ENDPOINT_SIGMA_PX,
                robust=True,
                huber_k=DEFAULT_LINE_HUBER_K,
                max_irls_iters=DEFAULT_LINE_IRLS_MAX_ITERS,
                irls_tol=DEFAULT_LINE_IRLS_TOL,
            )
            self._last_warning = self._line_calibration_quality_warning_text()

            self.session.grid_results.append(grid_res)
            self.state.switch_to(Mode.VIEW)
            self._pending_prompt = None
            self._remeasure_all()
            return

        raise ValueError(f"Unsupported prompt type: {prompt_type}")

    # ---------------------------------------------------------------------
    # Interaction internals
    # ---------------------------------------------------------------------

    def _on_move(self, x: float, y: float) -> None:
        state = self.state
        state.cursor_pos = (x, y)
        mode = state.mode

        if mode == Mode.POLYGON_EDIT:
            if state.dragging_vertex_idx is not None:
                self.drag_vertex(x, y)
            else:
                state.hover_vertex_idx = self.find_nearest_vertex(x, y)

        elif mode == Mode.GRID_DETECT and state.roi_start is not None:
            state.roi_end = (x, y)

        elif mode == Mode.CALIBRATE_PAD_PITCH and state.roi_start is not None:
            state.roi_end = (x, y)

    def _on_release(self, button: str, x: float, y: float) -> None:
        state = self.state
        mode = state.mode

        if button != "left":
            return

        if mode == Mode.POLYGON_EDIT:
            state.dragging_vertex_idx = None

        elif mode == Mode.GRID_DETECT and state.roi_start is not None:
            state.roi_end = (x, y)
            self.run_grid_detect()

        elif mode == Mode.CALIBRATE_PAD_PITCH and state.roi_start is not None:
            state.roi_end = (x, y)
            self.run_pad_pitch_detect()

    def _on_double_click(self, button: str, x: float, y: float) -> None:
        if button != "left":
            return
        if self.state.mode == Mode.POLYGON_DRAW:
            poly = self._current_polygon
            if poly is not None and poly.n_vertices >= 3:
                self.close_current_polygon()

    def maybe_snap(self, x: float, y: float) -> Tuple[float, float, bool, float]:
        if not self.session.edge_snap_enabled:
            return x, y, False, 1.0

        self._ensure_edge_detector()
        if self._edge_detector is None:
            return x, y, False, 1.0

        result = self._edge_detector.snap_to_edge(
            x,
            y,
            radius=float(self.session.edge_snap_radius),
        )
        if result is None:
            return x, y, False, 1.0
        return result[0], result[1], True, result[2]

    def _ensure_edge_detector(self) -> None:
        if self._edge_detector is not None:
            return
        if self.original_image is None:
            return
        self._edge_detector = EdgeDetector(self.original_image)
        self._edge_detector.compute_subpixel()

    def add_polygon_vertex(self, x: float, y: float, snapped: bool, confidence: float) -> None:
        if self._current_polygon is None:
            return
        self._current_polygon.vertices.append(
            Vertex(x=x, y=y, snapped=snapped, confidence=float(confidence))
        )

    def close_current_polygon(self) -> None:
        poly = self._current_polygon
        if poly is None or poly.n_vertices < 3:
            return

        if poly.n_vertices >= 4:
            last = poly.vertices[-1]
            prev = poly.vertices[-2]
            dist = ((last.x - prev.x) ** 2 + (last.y - prev.y) ** 2) ** 0.5
            if dist < 3.0:
                poly.vertices.pop()

        poly.closed = True
        self._undo_stack.append(("add_polygon", len(self.session.polygons)))
        self.session.polygons.append(poly)
        self._current_polygon = None

        if self.session.calibration.is_calibrated:
            mg = self._get_max_grad()
            result = measure_polygon(
                poly,
                self.session.calibration,
                target_unit=self.session.display_unit,
                max_grad=mg,
            )
            self.session.results.append(result)

        self.state.active_polygon_idx = len(self.session.polygons) - 1
        self.state.switch_to(Mode.VIEW)

    def find_nearest_vertex(self, x: float, y: float, threshold: float = 15.0) -> Optional[int]:
        idx = self.state.active_polygon_idx
        if idx is None or idx >= len(self.session.polygons):
            return None
        poly = self.session.polygons[idx]
        best_dist = threshold
        best_i = None
        for i, v in enumerate(poly.vertices):
            d = ((v.x - x) ** 2 + (v.y - y) ** 2) ** 0.5
            if d < best_dist:
                best_dist = d
                best_i = i
        return best_i

    def drag_vertex(self, x: float, y: float) -> None:
        idx = self.state.active_polygon_idx
        vi = self.state.dragging_vertex_idx
        if idx is None or vi is None:
            return
        if idx >= len(self.session.polygons):
            return
        poly = self.session.polygons[idx]
        if vi >= poly.n_vertices:
            return

        sx, sy, snapped, conf = self.maybe_snap(x, y)
        poly.vertices[vi].x = sx
        poly.vertices[vi].y = sy
        poly.vertices[vi].snapped = snapped
        poly.vertices[vi].confidence = conf if snapped else 1.0

        self._remeasure_polygon(idx)

    def delete_nearest_vertex(self) -> None:
        idx = self.state.active_polygon_idx
        if idx is None or idx >= len(self.session.polygons):
            return
        poly = self.session.polygons[idx]
        if poly.n_vertices <= 3:
            return

        vi = (
            self.state.hover_vertex_idx
            if self.state.hover_vertex_idx is not None
            else self.state.dragging_vertex_idx
        )
        if vi is None or vi >= poly.n_vertices:
            return

        poly.vertices.pop(vi)
        self.state.dragging_vertex_idx = None
        self.state.hover_vertex_idx = None
        self._remeasure_polygon(idx)

    def insert_vertex_on_edge(self) -> None:
        idx = self.state.active_polygon_idx
        if idx is None or idx >= len(self.session.polygons):
            return
        poly = self.session.polygons[idx]
        if poly.n_vertices < 2:
            return

        vi = (
            self.state.hover_vertex_idx
            if self.state.hover_vertex_idx is not None
            else self.state.dragging_vertex_idx
        )
        if vi is None:
            vi = poly.n_vertices - 1

        j = (vi + 1) % poly.n_vertices
        mx = (poly.vertices[vi].x + poly.vertices[j].x) / 2
        my = (poly.vertices[vi].y + poly.vertices[j].y) / 2
        poly.vertices.insert(j, Vertex(x=mx, y=my))
        self._remeasure_polygon(idx)

    def find_nearest_polygon(self, x: float, y: float, threshold: float = 15.0) -> Optional[int]:
        best_dist = threshold
        best_idx = None
        for i, poly in enumerate(self.session.polygons):
            if not poly.closed or poly.n_vertices < 3:
                continue
            for k in range(poly.n_vertices):
                j = (k + 1) % poly.n_vertices
                d = _point_to_segment_dist(
                    x,
                    y,
                    poly.vertices[k].x,
                    poly.vertices[k].y,
                    poly.vertices[j].x,
                    poly.vertices[j].y,
                )
                if d < best_dist:
                    best_dist = d
                    best_idx = i
        return best_idx

    def delete_polygon(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.session.polygons):
            return
        label = self.session.polygons[idx].label
        self.session.polygons.pop(idx)
        self.session.results = [r for r in self.session.results if r.polygon_label != label]

        if self.state.active_polygon_idx == idx:
            self.state.active_polygon_idx = None
        elif self.state.active_polygon_idx is not None and self.state.active_polygon_idx > idx:
            self.state.active_polygon_idx -= 1

    def run_grid_detect(self) -> None:
        start = self.state.roi_start
        end = self.state.roi_end
        if start is None or end is None:
            return

        x0 = int(min(start[0], end[0]))
        y0 = int(min(start[1], end[1]))
        x1 = int(max(start[0], end[0]))
        y1 = int(max(start[1], end[1]))
        roi = (x0, y0, x1 - x0, y1 - y0)

        if roi[2] < 20 or roi[3] < 20:
            return

        image = self._warped_image if self._warped_image is not None else self.original_image
        if image is None:
            return

        result = detect_grid(image, roi=roi, calibration=self.session.calibration)
        if result is not None:
            result.origin_x = float(x0)
            result.origin_y = float(y0)
            self.session.grid_results.append(result)

        self.state.switch_to(Mode.VIEW)

    def run_pad_pitch_detect(self) -> None:
        """Run FFT pitch detection on ROI, then set pending prompt for user input."""
        start = self.state.roi_start
        end = self.state.roi_end
        if start is None or end is None:
            return

        x0 = int(min(start[0], end[0]))
        y0 = int(min(start[1], end[1]))
        x1 = int(max(start[0], end[0]))
        y1 = int(max(start[1], end[1]))
        roi = (x0, y0, x1 - x0, y1 - y0)

        if roi[2] < 20 or roi[3] < 20:
            return

        image = self._warped_image if self._warped_image is not None else self.original_image
        if image is None:
            return

        grid_result = detect_grid(image, roi=roi)
        if grid_result is None:
            self._last_warning = tr("dialog.pad_pitch_failed")
            self.state.switch_to(Mode.VIEW)
            return

        pitch_px = float(np.hypot(grid_result.pitch_x, grid_result.pitch_y))

        # Store the ROI and detection result for use when prompt is submitted.
        self._pending_prompt = {
            "type": "pad_pitch",
            "title": tr("dialog.calibrate_pad_pitch_title"),
            "message": tr(
                "dialog.enter_pad_pitch",
                pitch_px=pitch_px,
                conf=grid_result.confidence,
            ),
            "_roi": roi,
            "_pitch_px": pitch_px,
        }

    def undo(self) -> None:
        if not self._undo_stack:
            return

        action, data = self._undo_stack.pop()
        if action == "add_polygon":
            idx = int(data)
            if idx < len(self.session.polygons):
                label = self.session.polygons[idx].label
                self.session.polygons.pop(idx)
                self.session.results = [
                    r for r in self.session.results if r.polygon_label != label
                ]

        elif action == "add_cal_line":
            idx = int(data)
            if idx < len(self.session.calibration.lines):
                self.session.calibration.lines.pop(idx)
                if self.session.calibration.lines:
                    update_calibration_lines(
                        self.session.calibration,
                        sigma_per_pixel=DEFAULT_LINE_ENDPOINT_SIGMA_PX,
                        robust=True,
                        huber_k=DEFAULT_LINE_HUBER_K,
                        max_irls_iters=DEFAULT_LINE_IRLS_MAX_ITERS,
                        irls_tol=DEFAULT_LINE_IRLS_TOL,
                    )
                    self._last_warning = self._line_calibration_quality_warning_text()
                else:
                    self.session.calibration.pixels_per_unit = 0.0
                    self.session.calibration.ppu_uncertainty = 0.0
                    self._last_warning = None
                self._remeasure_all()

    # ---------------------------------------------------------------------
    # Snapshot
    # ---------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        width = int(self.display_image.shape[1]) if self.display_image is not None else 0
        height = int(self.display_image.shape[0]) if self.display_image is not None else 0

        return {
            "language": get_language(),
            "mode": self.state.mode.value,
            "image": {
                "path": self._display_image_path,
                "original_path": self.image_path,
                "width": width,
                "height": height,
                "warped": self._warped_image is not None,
            },
            "session": {
                "display_unit": self.session.display_unit.value,
                "edge_snap_enabled": self.session.edge_snap_enabled,
                "edge_snap_radius": self.session.edge_snap_radius,
                "is_calibrated": self.session.calibration.is_calibrated,
            },
            "pending_prompt": self._pending_prompt,
            "warnings": [self._last_warning] if self._last_warning else [],
            "overlays": self._overlays_snapshot(),
            "info": self._info_snapshot(),
            "status": self._status_snapshot(),
        }

    def _overlays_snapshot(self) -> Dict[str, Any]:
        mode = self.state.mode
        cursor = self.state.cursor_pos

        polygons = []
        for idx, poly in enumerate(self.session.polygons):
            area_text = None
            if poly.closed and self.session.calibration.is_calibrated:
                result = self._result_for_label(poly.label)
                if result is not None and result.area_physical > 0:
                    rel_err = result.area_uncertainty / result.area_physical * 100
                    area_text = f"{result.area_physical:.4g} {result.unit.value}² ±{rel_err:.1f}%"

            selected = idx == self.state.active_polygon_idx
            polygons.append({
                "label": poly.label,
                "closed": poly.closed,
                "selected": selected,
                "area_text": area_text,
                "active_vertex_idx": self.state.dragging_vertex_idx if selected else None,
                "hover_vertex_idx": self.state.hover_vertex_idx if selected else None,
                "vertices": [
                    {
                        "x": v.x,
                        "y": v.y,
                        "snapped": v.snapped,
                        "confidence": v.confidence,
                    }
                    for v in poly.vertices
                ],
            })

        current_poly = None
        if self._current_polygon is not None and self._current_polygon.n_vertices > 0:
            current_poly = {
                "label": self._current_polygon.label,
                "closed": self._current_polygon.closed,
                "vertices": [
                    {
                        "x": v.x,
                        "y": v.y,
                        "snapped": v.snapped,
                        "confidence": v.confidence,
                    }
                    for v in self._current_polygon.vertices
                ],
            }

        rubberband = None
        if cursor is not None:
            if mode == Mode.POLYGON_DRAW and self._current_polygon is not None and self._current_polygon.n_vertices > 0:
                last = self._current_polygon.vertices[-1]
                rubberband = {
                    "from": {"x": last.x, "y": last.y},
                    "to": {"x": cursor[0], "y": cursor[1]},
                }
            elif mode in (Mode.CALIBRATE_LINE, Mode.CALIBRATE_RECT) and self.state.points:
                last = self.state.points[-1]
                rubberband = {
                    "from": {"x": last[0], "y": last[1]},
                    "to": {"x": cursor[0], "y": cursor[1]},
                }

        snap_marker = None
        if cursor is not None and self.session.edge_snap_enabled and mode in (
            Mode.POLYGON_DRAW,
            Mode.CALIBRATE_LINE,
            Mode.CALIBRATE_RECT,
        ):
            sx, sy, snapped, _ = self.maybe_snap(cursor[0], cursor[1])
            if snapped:
                snap_marker = {"x": sx, "y": sy}

        roi = None
        if self.state.roi_start is not None and self.state.roi_end is not None:
            roi = {
                "x0": self.state.roi_start[0],
                "y0": self.state.roi_start[1],
                "x1": self.state.roi_end[0],
                "y1": self.state.roi_end[1],
            }

        grids = []
        for g in self.session.grid_results:
            if g.pitch_x <= 0 or g.pitch_y <= 0:
                continue
            ext_w = g.pitch_x * max(20, int(500 / max(g.pitch_x, 1)))
            ext_h = g.pitch_y * max(20, int(500 / max(g.pitch_y, 1)))
            grids.append({
                "origin": {"x": g.origin_x, "y": g.origin_y},
                "pitch_x": g.pitch_x,
                "pitch_y": g.pitch_y,
                "angle_deg": g.angle_deg,
                "extent_w": ext_w,
                "extent_h": ext_h,
            })

        return {
            "calibration_lines": [
                {
                    "x0": line.p0.x,
                    "y0": line.p0.y,
                    "x1": line.p1.x,
                    "y1": line.p1.y,
                    "label": f"{line.known_length:.3g} {line.unit.value}",
                }
                for line in self.session.calibration.lines
            ],
            "points": [{"x": p[0], "y": p[1]} for p in self.state.points],
            "polygons": polygons,
            "current_polygon": current_poly,
            "rubberband": rubberband,
            "snap_marker": snap_marker,
            "roi": roi,
            "grids": grids,
        }

    def _info_snapshot(self) -> Dict[str, Any]:
        return {
            "calibration": self._calibration_info(),
            "polygon": self._polygon_info(),
            "grid": self._grid_info(),
        }

    def _status_snapshot(self) -> Dict[str, Any]:
        parts: List[str] = []
        display_unit = self.session.display_unit

        mode_key = f"help.{self.state.mode.value}"
        parts.append(f"[{tr(f'mode.{self.state.mode.value}')}] {tr(mode_key)}")

        cal = self.session.calibration
        if cal.is_calibrated:
            ppu = cal.pixels_per_unit
            ppu_unc = cal.ppu_uncertainty
            ppu_unit = cal.unit
            if display_unit != cal.unit:
                try:
                    scale = convert_length(1.0, display_unit, cal.unit)
                    ppu *= scale
                    ppu_unc *= scale
                    ppu_unit = display_unit
                except ValueError:
                    pass
            parts.append(tr("status.scale", ppu=ppu, unit=ppu_unit.value, unc=ppu_unc))
        else:
            parts.append(tr("status.not_calibrated"))

        parts.append(
            tr("status.snap_on", radius=self.session.edge_snap_radius)
            if self.session.edge_snap_enabled
            else tr("status.snap_off", radius=self.session.edge_snap_radius)
        )

        if self.state.cursor_pos is not None:
            cx, cy = self.state.cursor_pos
            pos = tr("status.cursor", x=cx, y=cy)
            if cal.is_calibrated:
                px = cx / cal.pixels_per_unit
                py = cy / cal.pixels_per_unit
                phys_unit = cal.unit
                if display_unit != cal.unit:
                    try:
                        px = convert_length(px, cal.unit, display_unit)
                        py = convert_length(py, cal.unit, display_unit)
                        phys_unit = display_unit
                    except ValueError:
                        pass
                pos += tr("status.cursor_phys", px=px, py=py, unit=phys_unit.value)
            parts.append(pos)

        poly = self._current_polygon
        if poly is not None and poly.n_vertices >= 3:
            area_px = polygon_area_pixels(poly)
            area_str = tr("status.current_area_px", area=area_px)
            if cal.is_calibrated:
                area_phys = area_px / (cal.pixels_per_unit ** 2)
                area_unit = cal.unit
                if display_unit != cal.unit:
                    try:
                        area_phys = convert_area(area_phys, cal.unit, display_unit)
                        area_unit = display_unit
                    except ValueError:
                        pass
                area_str += tr("status.current_area_phys", area=area_phys, unit=area_unit.value)
            parts.append(area_str)

        for g in self._grid_results_for_current_unit():
            parts.append(
                tr(
                    "status.grid",
                    px=g.pitch_x_physical,
                    py=g.pitch_y_physical,
                    unit=g.unit.value,
                    conf=g.confidence,
                )
            )

        return {
            "parts": parts,
            "message": "  |  ".join(parts),
        }

    def _calibration_info(self) -> Dict[str, Any]:
        cal = self.session.calibration
        display_unit = self.session.display_unit

        if not cal.is_calibrated:
            return {
                "calibrated": False,
                "title": tr("info.calibration"),
                "summary": tr("info.not_calibrated"),
            }

        ppu = cal.pixels_per_unit
        ppu_unc = cal.ppu_uncertainty
        unit = cal.unit
        if display_unit != cal.unit:
            try:
                scale = convert_length(1.0, display_unit, cal.unit)
                ppu *= scale
                ppu_unc *= scale
                unit = display_unit
            except ValueError:
                pass

        chi2_value, p_value, consistent = (
            chi_squared_consistency(
                cal.lines,
                sigma_per_pixel=DEFAULT_LINE_ENDPOINT_SIGMA_PX,
                confidence_level=_LINE_CAL_CONFIDENCE_LEVEL,
            )
            if len(cal.lines) >= 2
            else (0.0, 1.0, True)
        )
        rel_unc_pct = (
            100.0 * cal.ppu_uncertainty / cal.pixels_per_unit
            if cal.pixels_per_unit > 0
            else 0.0
        )

        return {
            "calibrated": True,
            "title": tr("info.calibration"),
            "scale_ppu": ppu,
            "scale_uncertainty": ppu_unc,
            "scale_relative_uncertainty_percent": rel_unc_pct,
            "unit": unit.value,
            "method": "line" if cal.lines else "rect",
            "line_count": len(cal.lines),
            "endpoint_sigma_px": DEFAULT_LINE_ENDPOINT_SIGMA_PX,
            "perspective_corrected": bool(cal.homography),
            "chi_squared": {
                "value": chi2_value,
                "p_value": p_value,
                "consistent": consistent,
                "confidence_level": _LINE_CAL_CONFIDENCE_LEVEL,
            },
        }

    def _line_calibration_quality_warning_text(self) -> Optional[str]:
        """Return a quantitative warning if line calibration quality is weak."""
        cal = self.session.calibration
        if not cal.lines or not cal.is_calibrated:
            return None

        rel_unc_pct = (
            100.0 * cal.ppu_uncertainty / cal.pixels_per_unit
            if cal.pixels_per_unit > 0
            else 0.0
        )

        if len(cal.lines) == 1:
            px_len = cal.lines[0].pixel_length
            if px_len < _LINE_CAL_MIN_RECOMMENDED_PIXELS:
                return tr(
                    "warn.cal.single_line_short",
                    px=px_len,
                    min_px=_LINE_CAL_MIN_RECOMMENDED_PIXELS,
                    sigma=DEFAULT_LINE_ENDPOINT_SIGMA_PX,
                    rel=rel_unc_pct,
                )
            return None

        chi2_value, p_value, consistent = chi_squared_consistency(
            cal.lines,
            sigma_per_pixel=DEFAULT_LINE_ENDPOINT_SIGMA_PX,
            confidence_level=_LINE_CAL_CONFIDENCE_LEVEL,
        )
        if not consistent:
            return tr(
                "warn.cal.inconsistent",
                n=len(cal.lines),
                chi2=chi2_value,
                p=p_value,
                conf=100.0 * _LINE_CAL_CONFIDENCE_LEVEL,
                rel=rel_unc_pct,
            )

        if rel_unc_pct > _LINE_CAL_WARN_REL_UNC_PERCENT:
            return tr(
                "warn.cal.high_unc",
                n=len(cal.lines),
                rel=rel_unc_pct,
                threshold=_LINE_CAL_WARN_REL_UNC_PERCENT,
            )

        return None

    def _polygon_info(self) -> Dict[str, Any]:
        idx = self.state.active_polygon_idx
        if idx is None or idx >= len(self.session.polygons):
            return {
                "selected": False,
                "title": tr("info.polygon"),
                "summary": tr("info.none_selected"),
            }

        poly = self.session.polygons[idx]
        result = self._result_for_label(poly.label)

        if result is None:
            return {
                "selected": True,
                "label": poly.label,
                "measured": False,
                "summary": tr("info.not_measured"),
                "vertex_rows": self._vertex_rows(poly),
            }

        area = result.area_physical
        area_unc = result.area_uncertainty
        area_unc_v = result.area_uncertainty_vertex
        area_unc_ppu = result.area_uncertainty_ppu
        perim = result.perimeter_physical
        perim_unc = result.perimeter_uncertainty
        bbox_w = result.bbox_width_physical
        bbox_h = result.bbox_height_physical
        bbox_w_unc = result.bbox_width_uncertainty
        bbox_h_unc = result.bbox_height_uncertainty
        out_unit = result.unit

        if result.unit != self.session.display_unit:
            try:
                area = convert_area(area, result.unit, self.session.display_unit)
                area_unc = convert_area(area_unc, result.unit, self.session.display_unit)
                area_unc_v = convert_area(area_unc_v, result.unit, self.session.display_unit)
                area_unc_ppu = convert_area(area_unc_ppu, result.unit, self.session.display_unit)
                perim = convert_length(perim, result.unit, self.session.display_unit)
                perim_unc = convert_length(perim_unc, result.unit, self.session.display_unit)
                bbox_w = convert_length(bbox_w, result.unit, self.session.display_unit)
                bbox_h = convert_length(bbox_h, result.unit, self.session.display_unit)
                bbox_w_unc = convert_length(bbox_w_unc, result.unit, self.session.display_unit)
                bbox_h_unc = convert_length(bbox_h_unc, result.unit, self.session.display_unit)
                out_unit = self.session.display_unit
            except ValueError:
                pass

        rel_err = (area_unc / area * 100.0) if area > 0 else 0.0

        return {
            "selected": True,
            "label": poly.label,
            "measured": True,
            "unit": out_unit.value,
            "area": area,
            "area_uncertainty": area_unc,
            "area_uncertainty_vertex": area_unc_v,
            "area_uncertainty_ppu": area_unc_ppu,
            "relative_error_percent": rel_err,
            "perimeter": perim,
            "perimeter_uncertainty": perim_unc,
            "bbox_width": bbox_w,
            "bbox_height": bbox_h,
            "bbox_angle_deg": result.bbox_angle_deg,
            "bbox_width_uncertainty": bbox_w_unc,
            "bbox_height_uncertainty": bbox_h_unc,
            "n_vertices": result.n_vertices,
            "area_pixels": result.area_pixels,
            "vertex_rows": self._vertex_rows(poly),
        }

    def _grid_info(self) -> Dict[str, Any]:
        detections = []
        for g in self._grid_results_for_current_unit():
            detections.append({
                "pitch_x": g.pitch_x_physical,
                "pitch_y": g.pitch_y_physical,
                "unit": g.unit.value,
                "angle_deg": g.angle_deg,
                "confidence": g.confidence,
            })
        return {
            "count": len(detections),
            "detections": detections,
        }

    def _vertex_rows(self, poly: Polygon) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        cal = self.session.calibration
        display_unit = self.session.display_unit

        for i, v in enumerate(poly.vertices):
            if cal.is_calibrated:
                px = v.x / cal.pixels_per_unit
                py = v.y / cal.pixels_per_unit
                phys_unit = cal.unit
                try:
                    if display_unit != cal.unit:
                        px = convert_length(px, cal.unit, display_unit)
                        py = convert_length(py, cal.unit, display_unit)
                        phys_unit = display_unit
                except ValueError:
                    phys_unit = cal.unit
            else:
                px = None
                py = None
                phys_unit = display_unit

            rows.append({
                "index": i,
                "x_px": v.x,
                "y_px": v.y,
                "x_phys": px,
                "y_phys": py,
                "unit": phys_unit.value,
                "snapped": v.snapped,
                "confidence": v.confidence,
            })
        return rows

    # ---------------------------------------------------------------------
    # helpers
    # ---------------------------------------------------------------------

    def _ensure_valid_display_unit(self) -> None:
        if self.session.display_unit == Unit.PIXEL:
            self.session.display_unit = (
                self.session.calibration.unit
                if self.session.calibration.unit != Unit.PIXEL
                else Unit.MILLIMETER
            )

    def _result_for_label(self, label: str):
        for result in self.session.results:
            if result.polygon_label == label:
                return result
        return None

    def _grid_results_for_current_unit(self) -> List[GridResult]:
        out: List[GridResult] = []
        target = self.session.display_unit
        for g in self.session.grid_results:
            if g.unit == Unit.PIXEL or g.unit == target:
                out.append(g)
                continue
            try:
                out.append(
                    GridResult(
                        pitch_x=g.pitch_x,
                        pitch_y=g.pitch_y,
                        pitch_x_physical=convert_length(g.pitch_x_physical, g.unit, target),
                        pitch_y_physical=convert_length(g.pitch_y_physical, g.unit, target),
                        angle_deg=g.angle_deg,
                        confidence=g.confidence,
                        unit=target,
                        origin_x=g.origin_x,
                        origin_y=g.origin_y,
                    )
                )
            except ValueError:
                out.append(g)
        return out

    def _get_max_grad(self) -> float:
        if self._edge_detector is not None and self._edge_detector.max_grad > 0:
            return self._edge_detector.max_grad
        return 255.0

    def _remeasure_all(self) -> None:
        self.session.results.clear()
        if not self.session.calibration.is_calibrated:
            return
        mg = self._get_max_grad()
        for poly in self.session.polygons:
            if poly.closed and poly.n_vertices >= 3:
                result = measure_polygon(
                    poly,
                    self.session.calibration,
                    target_unit=self.session.display_unit,
                    max_grad=mg,
                )
                self.session.results.append(result)

    def _remeasure_polygon(self, idx: int) -> None:
        if not self.session.calibration.is_calibrated:
            return
        if idx < 0 or idx >= len(self.session.polygons):
            return

        poly = self.session.polygons[idx]
        mg = self._get_max_grad()
        result = measure_polygon(
            poly,
            self.session.calibration,
            target_unit=self.session.display_unit,
            max_grad=mg,
        )
        for i, existing in enumerate(self.session.results):
            if existing.polygon_label == poly.label:
                self.session.results[i] = result
                return
        self.session.results.append(result)

    def _set_warped_display(self, warped: np.ndarray) -> None:
        self._warped_image = warped
        self.display_image = warped

        # Keep a stable on-disk image path so Swift can reload lazily.
        tmp = tempfile.NamedTemporaryFile(prefix="diemeter-warped-", suffix=".png", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()
        cv2.imwrite(str(tmp_path), warped)
        self._display_image_path = str(tmp_path)

    @staticmethod
    def _parse_length_input(s: str) -> Tuple[Optional[float], Unit]:
        s = s.strip()
        parts = s.split()
        if len(parts) == 1:
            try:
                return float(parts[0]), Unit.MILLIMETER
            except ValueError:
                return None, Unit.MILLIMETER
        if len(parts) == 2:
            try:
                val = float(parts[0])
                unit = unit_from_str(parts[1])
                return val, unit
            except (ValueError, KeyError):
                return None, Unit.MILLIMETER
        return None, Unit.MILLIMETER

    @staticmethod
    def _parse_rect_input(s: str) -> Tuple[Optional[float], Optional[float], Unit]:
        s = s.strip().lower()
        s = s.replace(" x ", "x").replace("× ", "x").replace(" ×", "x")

        unit = Unit.MILLIMETER
        for suffix in ["mm", "cm", "um", "in", "mil", "m"]:
            if s.endswith(suffix):
                unit = unit_from_str(suffix)
                s = s[: -len(suffix)].strip()
                break

        parts = s.split("x")
        if len(parts) != 2:
            return None, None, unit
        try:
            w = float(parts[0].strip())
            h = float(parts[1].strip())
            return w, h, unit
        except ValueError:
            return None, None, unit

    @staticmethod
    def _extract_polygon_counter(label: str) -> int:
        if not label.startswith("poly_"):
            return 0
        try:
            return int(label.split("_", 1)[1])
        except Exception:
            return 0

    def _with_snapshot(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(payload)
        out["snapshot"] = self.snapshot()
        return out


def _point_to_segment_dist(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    dx = bx - ax
    dy = by - ay
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return ((px - proj_x) ** 2 + (py - proj_y) ** 2) ** 0.5
