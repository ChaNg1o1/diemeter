"""Command-line interface for diemeter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="diemeter",
        description="High-precision photogrammetric area estimation tool",
    )
    parser.add_argument(
        "image",
        help="Path to the image file to measure",
    )
    parser.add_argument(
        "-s", "--session",
        help="Path to session file (.json) to load/save",
        default=None,
    )
    parser.add_argument(
        "--edge-snap",
        action="store_true",
        help="Enable edge snap on startup",
    )
    parser.add_argument(
        "--snap-radius",
        type=int,
        default=15,
        help="Edge snap search radius in pixels (default: 15)",
    )
    parser.add_argument(
        "-V", "--version",
        action="version",
        version=f"%(prog)s {_get_version()}",
    )
    return parser


def _get_version() -> str:
    try:
        from . import __version__
        return __version__
    except ImportError:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    image_path = args.image
    if not Path(image_path).exists():
        print(f"Error: image not found: {image_path}", file=sys.stderr)
        return 1
    if args.snap_radius <= 0:
        print("Error: --snap-radius must be positive", file=sys.stderr)
        return 1

    # Import here to avoid slow startup for --help / --version
    from .gui.app import MeasurementApp

    try:
        app = MeasurementApp(
            image_path=image_path,
            session_path=args.session,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.snap_radius != 15:
        app.set_snap_radius(args.snap_radius)
    if args.edge_snap:
        app.set_edge_snap_enabled(True)

    app.show()
    return 0
