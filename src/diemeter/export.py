"""CSV and JSON export for measurement results."""

from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path
from typing import List

from .model import GridResult, MeasurementResult
from .session import _grid_result_to_dict, _result_to_dict


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
        "unit",
        "n_vertices",
    ])
    for r in results:
        writer.writerow([
            r.polygon_label,
            f"{r.area_pixels:.4f}",
            f"{r.area_physical:.6f}",
            f"{r.area_uncertainty:.6f}",
            r.unit.value,
            r.n_vertices,
        ])
    return buf.getvalue()


def results_to_json(results: List[MeasurementResult]) -> str:
    """Serialize measurement results to JSON string."""
    data = [_result_to_dict(r) for r in results]
    return json.dumps(data, indent=2)


def grid_results_to_csv(results: List[GridResult]) -> str:
    """Serialize grid detection results to CSV string."""
    if not results:
        return ""
    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "pitch_x_px",
        "pitch_y_px",
        "pitch_x_physical",
        "pitch_y_physical",
        "angle_deg",
        "confidence",
        "unit",
    ])
    for g in results:
        writer.writerow([
            f"{g.pitch_x:.4f}",
            f"{g.pitch_y:.4f}",
            f"{g.pitch_x_physical:.6f}",
            f"{g.pitch_y_physical:.6f}",
            f"{g.angle_deg:.2f}",
            f"{g.confidence:.4f}",
            g.unit.value,
        ])
    return buf.getvalue()


def grid_results_to_json(results: List[GridResult]) -> str:
    """Serialize grid detection results to JSON string."""
    data = [_grid_result_to_dict(g) for g in results]
    return json.dumps(data, indent=2)


def export_results(
    path: str | Path,
    results: List[MeasurementResult],
    grid_results: List[GridResult] | None = None,
) -> None:
    """Export results to file. Format determined by extension (.csv or .json)."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        content = results_to_csv(results)
        if grid_results:
            content += "\n" + grid_results_to_csv(grid_results)
    elif suffix == ".json":
        data = {
            "measurements": [_result_to_dict(r) for r in results],
        }
        if grid_results:
            data["grid_detections"] = [_grid_result_to_dict(g) for g in grid_results]
        content = json.dumps(data, indent=2)
    else:
        raise ValueError(f"Unsupported export format: {suffix}. Use .csv or .json")

    path.write_text(content, encoding="utf-8")
