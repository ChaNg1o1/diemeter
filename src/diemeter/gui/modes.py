"""Mode enum and state machine for the GUI."""

from __future__ import annotations

import enum
from typing import Optional, Tuple


class Mode(enum.Enum):
    """Interactive modes for the measurement GUI."""

    VIEW = "view"
    CALIBRATE_LINE = "calibrate_line"
    CALIBRATE_RECT = "calibrate_rect"
    CALIBRATE_PAD_PITCH = "calibrate_pad_pitch"
    POLYGON_DRAW = "polygon_draw"
    POLYGON_EDIT = "polygon_edit"
    GRID_DETECT = "grid_detect"


class ModeState:
    """Tracks current mode and transient interaction state."""

    def __init__(self) -> None:
        self.mode: Mode = Mode.VIEW
        # Accumulated click points for current interaction
        self.points: list = []
        # Index of polygon/vertex being edited
        self.active_polygon_idx: Optional[int] = None
        self.dragging_vertex_idx: Optional[int] = None
        # ROI drag state for grid detect
        self.roi_start: Optional[Tuple[float, float]] = None
        self.roi_end: Optional[Tuple[float, float]] = None
        # Cursor position (updated on mouse move)
        self.cursor_pos: Optional[Tuple[float, float]] = None
        # Hover state: index of vertex the cursor is near (for visual feedback)
        self.hover_vertex_idx: Optional[int] = None

    def switch_to(self, mode: Mode) -> None:
        """Switch mode and reset transient state."""
        self.mode = mode
        self.points = []
        self.dragging_vertex_idx = None
        self.hover_vertex_idx = None
        self.roi_start = None
        self.roi_end = None

    def add_point(self, x: float, y: float) -> None:
        self.points.append((x, y))
