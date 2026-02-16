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
