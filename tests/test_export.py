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
from diemeter.model import GridResult, MeasurementResult, Unit


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


class TestExportResults:
    def test_csv_grid_only_no_leading_blank_line(self, tmp_path):
        path = tmp_path / "out.csv"
        grid = GridResult(
            pitch_x=100.0,
            pitch_y=100.0,
            pitch_x_physical=10.0,
            pitch_y_physical=10.0,
            angle_deg=0.0,
            confidence=0.8,
            unit=Unit.MILLIMETER,
        )
        export_results(path, results=[], grid_results=[grid])
        content = path.read_text(encoding="utf-8")
        assert content.startswith("pitch_x_px,")
