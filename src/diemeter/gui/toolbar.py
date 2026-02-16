"""QToolBar-based toolbar with mode switching and actions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import QToolBar

from .i18n import tr
from .modes import Mode

if TYPE_CHECKING:
    from .app import MeasurementApp

# Mode key → i18n key mapping
_MODE_TR_KEYS = {
    Mode.VIEW: "mode.view",
    Mode.CALIBRATE_LINE: "mode.calibrate_line",
    Mode.CALIBRATE_RECT: "mode.calibrate_rect",
    Mode.POLYGON_DRAW: "mode.polygon_draw",
    Mode.POLYGON_EDIT: "mode.polygon_edit",
    Mode.GRID_DETECT: "mode.grid_detect",
}


class ToolbarManager:
    """Manages the main QToolBar with mode actions and utility buttons."""

    def __init__(self, app: MeasurementApp) -> None:
        self._app = app
        self._toolbar = QToolBar("Main", app)
        self._toolbar.setMovable(False)
        self._toolbar.setStyleSheet(
            "QToolBar { background: #2a2a2a; spacing: 4px; padding: 2px; }"
            "QToolButton { color: #e0e0e0; background: #2a2a2a; border: none;"
            "  padding: 4px 10px; border-radius: 3px; font-size: 11px; }"
            "QToolButton:hover { background: #444444; }"
            "QToolButton:checked { background: #1a5fb4; color: white; }"
        )
        app.addToolBar(self._toolbar)

        # Mode actions in an exclusive group
        self._mode_group = QActionGroup(app)
        self._mode_group.setExclusive(True)
        self._mode_actions: dict[Mode, QAction] = {}

        for mode in (
            Mode.VIEW,
            Mode.CALIBRATE_LINE,
            Mode.CALIBRATE_RECT,
            Mode.POLYGON_DRAW,
            Mode.POLYGON_EDIT,
            Mode.GRID_DETECT,
        ):
            action = QAction(tr(_MODE_TR_KEYS[mode]), app)
            action.setCheckable(True)
            self._mode_group.addAction(action)
            self._toolbar.addAction(action)
            self._mode_actions[mode] = action
            # Closure to capture mode
            action.triggered.connect(
                lambda checked, m=mode: app.switch_mode(m)
            )

        # Default: VIEW checked
        self._mode_actions[Mode.VIEW].setChecked(True)

        self._toolbar.addSeparator()

        # Snap toggle (independent of mode group)
        self._snap_action = QAction(tr("toolbar.snap"), app)
        self._snap_action.setCheckable(True)
        self._snap_action.triggered.connect(lambda: app.toggle_edge_snap())
        self._toolbar.addAction(self._snap_action)

        # Undo button
        self._undo_action = QAction(tr("toolbar.undo"), app)
        self._undo_action.triggered.connect(lambda: app.undo())
        self._toolbar.addAction(self._undo_action)

        self._toolbar.addSeparator()

        # Language toggle
        self._lang_action = QAction(tr("toolbar.lang"), app)
        self._lang_action.triggered.connect(lambda: app.toggle_language())
        self._toolbar.addAction(self._lang_action)

    def update_highlight(self, mode: Mode, snap_on: bool) -> None:
        """Sync toolbar state with current mode and snap state."""
        action = self._mode_actions.get(mode)
        if action is not None:
            action.setChecked(True)
        self._snap_action.setChecked(snap_on)
        # Update snap button style
        if snap_on:
            self._snap_action.setToolTip("Edge snap: ON")
        else:
            self._snap_action.setToolTip("Edge snap: OFF")

    def retranslate(self) -> None:
        """Update all action texts after language change."""
        for mode, action in self._mode_actions.items():
            action.setText(tr(_MODE_TR_KEYS[mode]))
        self._snap_action.setText(tr("toolbar.snap"))
        self._undo_action.setText(tr("toolbar.undo"))
        self._lang_action.setText(tr("toolbar.lang"))
