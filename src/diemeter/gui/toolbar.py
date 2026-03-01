"""QToolBar-based toolbar with mode switching and actions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import QComboBox, QLabel, QSpinBox, QToolBar

from ..model import Unit
from .i18n import tr
from .modes import Mode

if TYPE_CHECKING:
    from .app import MeasurementApp

# Mode key → i18n key mapping
_MODE_TR_KEYS = {
    Mode.VIEW: "mode.view",
    Mode.CALIBRATE_LINE: "mode.calibrate_line",
    Mode.CALIBRATE_RECT: "mode.calibrate_rect",
    Mode.CALIBRATE_PAD_PITCH: "mode.calibrate_pad_pitch",
    Mode.POLYGON_DRAW: "mode.polygon_draw",
    Mode.POLYGON_EDIT: "mode.polygon_edit",
    Mode.GRID_DETECT: "mode.grid_detect",
}

_DISPLAY_UNITS = [
    Unit.MICROMETER,
    Unit.MILLIMETER,
    Unit.CENTIMETER,
    Unit.METER,
    Unit.INCH,
    Unit.MIL,
]


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
            Mode.CALIBRATE_PAD_PITCH,
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

        self._snap_radius_label = QLabel(tr("toolbar.snap_radius"), app)
        self._snap_radius_label.setStyleSheet(
            "color: #cfcfcf; font-size: 11px; padding: 0 4px;"
        )
        self._toolbar.addWidget(self._snap_radius_label)

        self._snap_radius_spin = QSpinBox(app)
        self._snap_radius_spin.setRange(1, 200)
        self._snap_radius_spin.setSingleStep(1)
        self._snap_radius_spin.setFixedWidth(64)
        self._snap_radius_spin.setStyleSheet(
            "QSpinBox { color: #e0e0e0; background: #2a2a2a;"
            "  border: 1px solid #555; border-radius: 3px; padding: 2px 4px; }"
            "QSpinBox::up-button, QSpinBox::down-button { width: 14px; }"
        )
        self._snap_radius_spin.valueChanged.connect(self._on_snap_radius_changed)
        self._toolbar.addWidget(self._snap_radius_spin)

        # Undo button
        self._undo_action = QAction(tr("toolbar.undo"), app)
        self._undo_action.triggered.connect(lambda: app.undo())
        self._toolbar.addAction(self._undo_action)

        self._toolbar.addSeparator()

        # Save / Export
        self._save_action = QAction(tr("toolbar.save"), app)
        self._save_action.triggered.connect(lambda: app.save_session())
        self._toolbar.addAction(self._save_action)

        self._export_action = QAction(tr("toolbar.export"), app)
        self._export_action.triggered.connect(lambda: app.export_results())
        self._toolbar.addAction(self._export_action)

        self._toolbar.addSeparator()

        # Display unit selector
        self._unit_label = QLabel(tr("toolbar.unit"), app)
        self._unit_label.setStyleSheet("color: #cfcfcf; font-size: 11px; padding: 0 4px;")
        self._toolbar.addWidget(self._unit_label)

        self._unit_combo = QComboBox(app)
        self._unit_combo.setMinimumWidth(88)
        self._unit_combo.setStyleSheet(
            "QComboBox { color: #e0e0e0; background: #2a2a2a;"
            "  border: 1px solid #555; border-radius: 3px; padding: 2px 6px; }"
            "QComboBox QAbstractItemView { background: #2a2a2a; color: #e0e0e0; }"
        )
        for unit in _DISPLAY_UNITS:
            self._unit_combo.addItem(unit.value, unit)
        self._unit_combo.currentIndexChanged.connect(self._on_unit_changed)
        self._toolbar.addWidget(self._unit_combo)

        # Initialize selector to current app state.
        self.set_display_unit(app.session.display_unit)
        self.set_snap_radius(app.session.edge_snap_radius)

        self._toolbar.addSeparator()

        # Language toggle
        self._lang_action = QAction(tr("toolbar.lang"), app)
        self._lang_action.triggered.connect(lambda: app.toggle_language())
        self._toolbar.addAction(self._lang_action)

    def update_highlight(self, mode: Mode, snap_on: bool, snap_radius: int) -> None:
        """Sync toolbar state with current mode and snap state."""
        action = self._mode_actions.get(mode)
        if action is not None:
            action.setChecked(True)
        self._snap_action.setChecked(snap_on)
        self.set_snap_radius(snap_radius)
        # Update snap button style
        if snap_on:
            self._snap_action.setToolTip(f"Edge snap: ON (r={snap_radius}px)")
        else:
            self._snap_action.setToolTip(f"Edge snap: OFF (r={snap_radius}px)")

    def retranslate(self) -> None:
        """Update all action texts after language change."""
        for mode, action in self._mode_actions.items():
            action.setText(tr(_MODE_TR_KEYS[mode]))
        self._snap_action.setText(tr("toolbar.snap"))
        self._undo_action.setText(tr("toolbar.undo"))
        self._save_action.setText(tr("toolbar.save"))
        self._export_action.setText(tr("toolbar.export"))
        self._unit_label.setText(tr("toolbar.unit"))
        self._snap_radius_label.setText(tr("toolbar.snap_radius"))
        self._lang_action.setText(tr("toolbar.lang"))

    def set_display_unit(self, unit: Unit) -> None:
        """Sync the unit combo from external app state."""
        idx = self._unit_combo.findData(unit, Qt.ItemDataRole.UserRole)
        if idx >= 0 and idx != self._unit_combo.currentIndex():
            self._unit_combo.blockSignals(True)
            self._unit_combo.setCurrentIndex(idx)
            self._unit_combo.blockSignals(False)

    def set_snap_radius(self, radius: int) -> None:
        """Sync the snap radius spinbox from external app state."""
        if radius != self._snap_radius_spin.value():
            self._snap_radius_spin.blockSignals(True)
            self._snap_radius_spin.setValue(int(radius))
            self._snap_radius_spin.blockSignals(False)

    def _on_unit_changed(self, index: int) -> None:
        """Handle user selecting a new display unit."""
        unit = self._unit_combo.itemData(index, Qt.ItemDataRole.UserRole)
        if isinstance(unit, Unit):
            self._app.set_display_unit(unit)

    def _on_snap_radius_changed(self, value: int) -> None:
        """Handle user changing edge snap radius."""
        self._app.set_snap_radius(int(value))
