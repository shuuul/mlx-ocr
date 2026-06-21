"""Shared CLI helpers for benchmark runners."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.common.types import VARIANTS


def add_runner_args(parser: argparse.ArgumentParser) -> None:
    """Register common runner CLI arguments."""
    parser.add_argument(
        "--variant",
        choices=VARIANTS,
        required=True,
        help="PP-OCRv6 model variant to benchmark.",
    )
    parser.add_argument(
        "--images",
        nargs="*",
        type=Path,
        default=None,
        help="Image paths. Defaults to examples/images/*.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warmup iterations per image.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Timed benchmark iterations per image.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write JSON benchmark records.",
    )
