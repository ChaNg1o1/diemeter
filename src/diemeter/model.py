"""Domain types for diemeter."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


class Unit(enum.Enum):
    """Physical unit for measurements."""

    PIXEL = "px"
    MICROMETER = "um"
    MILLIMETER = "mm"
    CENTIMETER = "cm"
    METER = "m"
    INCH = "in"
    MIL = "mil"  # thousandths of an inch


# Conversion factors to meters.
_TO_METERS = {
    Unit.MICROMETER: 1e-6,
    Unit.MILLIMETER: 1e-3,
    Unit.CENTIMETER: 1e-2,
    Unit.METER: 1.0,
    Unit.INCH: 0.0254,
    Unit.MIL: 0.0254e-3,
}


def unit_from_str(s: str) -> Unit:
    """Parse a unit string (case-insensitive) into a Unit enum."""
    mapping = {
        "px": Unit.PIXEL,
        "pixel": Unit.PIXEL,
        "um": Unit.MICROMETER,
        "μm": Unit.MICROMETER,
        "micrometer": Unit.MICROMETER,
        "mm": Unit.MILLIMETER,
        "millimeter": Unit.MILLIMETER,
        "cm": Unit.CENTIMETER,
        "centimeter": Unit.CENTIMETER,
        "m": Unit.METER,
        "meter": Unit.METER,
        "in": Unit.INCH,
        "inch": Unit.INCH,
        "mil": Unit.MIL,
    }
    key = s.strip().lower()
    if key not in mapping:
        raise ValueError(f"Unknown unit: {s!r}. Valid: {list(mapping.keys())}")
    return mapping[key]


def convert_length(value: float, from_unit: Unit, to_unit: Unit) -> float:
    """Convert a length value between physical units."""
    if from_unit == to_unit:
        return value
    if from_unit == Unit.PIXEL or to_unit == Unit.PIXEL:
        raise ValueError("Cannot convert to/from pixels without calibration")
    meters = value * _TO_METERS[from_unit]
    return meters / _TO_METERS[to_unit]


def convert_area(value: float, from_unit: Unit, to_unit: Unit) -> float:
    """Convert an area value between physical units (applies squared factor)."""
    if from_unit == to_unit:
        return value
    if from_unit == Unit.PIXEL or to_unit == Unit.PIXEL:
        raise ValueError("Cannot convert to/from pixels without calibration")
    meters_sq = value * _TO_METERS[from_unit] ** 2
    return meters_sq / _TO_METERS[to_unit] ** 2


@dataclass
class Vertex:
    """A 2D point, optionally with sub-pixel refinement info."""

    x: float
    y: float
    snapped: bool = False
    confidence: float = 1.0  # 1.0 = user click, higher = edge-snapped


@dataclass
class CalibrationLine:
    """A two-point calibration reference."""

    p0: Vertex
    p1: Vertex
    known_length: float  # in `unit`
    unit: Unit = Unit.MILLIMETER

    @property
    def pixel_length(self) -> float:
        dx = self.p1.x - self.p0.x
        dy = self.p1.y - self.p0.y
        return (dx * dx + dy * dy) ** 0.5

    @property
    def pixels_per_unit(self) -> float:
        return self.pixel_length / self.known_length


@dataclass
class CalibrationRect:
    """A four-corner calibration reference for perspective correction."""

    corners: List[Vertex]  # exactly 4, in order: TL, TR, BR, BL
    known_width: float
    known_height: float
    unit: Unit = Unit.MILLIMETER

    def __post_init__(self) -> None:
        if len(self.corners) != 4:
            raise ValueError("CalibrationRect requires exactly 4 corners")


@dataclass
class Calibration:
    """Combined calibration state."""

    lines: List[CalibrationLine] = field(default_factory=list)
    rect: Optional[CalibrationRect] = None
    unit: Unit = Unit.MILLIMETER
    # Computed after calibration
    pixels_per_unit: float = 0.0
    ppu_uncertainty: float = 0.0
    # Homography matrix (3x3), stored as flat list for JSON
    homography: Optional[List[float]] = None

    @property
    def is_calibrated(self) -> bool:
        return self.pixels_per_unit > 0.0


@dataclass
class Polygon:
    """A closed polygon for area measurement."""

    vertices: List[Vertex] = field(default_factory=list)
    label: str = ""
    closed: bool = False

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)


@dataclass
class MeasurementResult:
    """Result of a polygon area measurement."""

    polygon_label: str
    area_pixels: float
    area_physical: float
    area_uncertainty: float
    unit: Unit
    n_vertices: int


@dataclass
class GridResult:
    """Result of repeating pattern (die grid) detection."""

    pitch_x: float  # pixels
    pitch_y: float  # pixels
    pitch_x_physical: float
    pitch_y_physical: float
    angle_deg: float  # rotation of grid axes
    confidence: float  # peak SNR
    unit: Unit = Unit.MILLIMETER
    origin_x: float = 0.0  # ROI origin in image coordinates
    origin_y: float = 0.0


@dataclass
class Session:
    """Persistent session state."""

    image_path: str = ""
    calibration: Calibration = field(default_factory=Calibration)
    polygons: List[Polygon] = field(default_factory=list)
    results: List[MeasurementResult] = field(default_factory=list)
    grid_results: List[GridResult] = field(default_factory=list)
    edge_snap_enabled: bool = False
    edge_snap_radius: int = 15
