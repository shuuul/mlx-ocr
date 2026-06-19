#!/usr/bin/env python3
"""Backward-compatible wrapper for the mlx-ocr benchmark runner."""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"


def main() -> None:
    """Run all mlx-ocr variants via the shared benchmark runner."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    for variant in ("tiny", "small", "medium"):
        output = RESULTS_DIR / f"mlx_{variant}_{stamp}.json"
        command = [
            sys.executable,
            "-m",
            "benchmarks.runners.mlx_ocr",
            "--variant",
            variant,
            "--output",
            str(output),
        ]
        subprocess.run(command, check=True, cwd=REPO_ROOT)
        print(output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
