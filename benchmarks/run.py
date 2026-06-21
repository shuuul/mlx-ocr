#!/usr/bin/env python3
"""Orchestrate PP-OCRv6 backend benchmarks in isolated subprocesses."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from benchmarks.common.images import resolve_example_images
from benchmarks.common.types import (
    BACKENDS,
    VARIANTS,
    Backend,
    Variant,
    records_from_json,
    records_to_json,
)
from benchmarks.compare import print_comparison_table

REPO_ROOT = Path(__file__).resolve().parents[1]

RUNNER_MODULES: dict[Backend, str] = {
    "mlx": "benchmarks.runners.mlx_ocr",
    "paddle_cpu": "benchmarks.runners.paddle_cpu",
    "paddle_onnx": "benchmarks.runners.paddle_onnx",
    "mineru_pipeline": "benchmarks.runners.mineru_pipeline",
}
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
PADDLE_PYTHON = os.environ.get("PADDLE_BENCHMARK_PYTHON", sys.executable)
MINERU_PYTHON = os.environ.get("MINERU_BENCHMARK_PYTHON", sys.executable)


def parse_backends(value: str) -> list[Backend]:
    """Parse a comma-separated backend list."""
    raw_backends = [item.strip() for item in value.split(",") if item.strip()]
    if not raw_backends:
        raise argparse.ArgumentTypeError("at least one backend is required")
    backends: list[Backend] = []
    for backend in raw_backends:
        if backend not in BACKENDS:
            raise argparse.ArgumentTypeError(f"unknown backend: {backend}")
        backends.append(backend)
    return backends


def parse_variants(values: list[str] | None, variant: str | None) -> list[Variant]:
    """Resolve variant CLI arguments."""
    if values:
        variants: list[Variant] = []
        for item in values:
            if item not in VARIANTS:
                raise SystemExit(f"unknown variant: {item}")
            variants.append(item)
        return variants
    if variant is not None:
        if variant not in VARIANTS:
            raise SystemExit(f"unknown variant: {variant}")
        return [variant]
    return list(VARIANTS)


def run_backend(
    backend: Backend,
    *,
    variant: Variant,
    images: list[Path],
    warmup: int,
    runs: int,
) -> Path:
    """Run one backend benchmark subprocess and return its JSON output path."""
    runner_module = RUNNER_MODULES[backend]
    output_path = RESULTS_DIR / f"{backend}_{variant}.json"
    if backend == "mlx":
        python = sys.executable
    elif backend == "mineru_pipeline":
        python = MINERU_PYTHON
    else:
        python = PADDLE_PYTHON
    command = [
        python,
        "-m",
        runner_module,
        "--variant",
        variant,
        "--warmup",
        str(warmup),
        "--runs",
        str(runs),
        "--output",
        str(output_path),
    ]
    for image in images:
        command.extend(["--images", str(image)])

    print(f"running {' '.join(command)}", flush=True)
    subprocess.run(command, check=True, cwd=REPO_ROOT)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    """Build the benchmark orchestrator argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends",
        type=parse_backends,
        default=list(BACKENDS),
        help=f"Comma-separated backends. Choices: {', '.join(BACKENDS)}.",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        choices=VARIANTS,
        help="Model variants to benchmark. Defaults to all variants.",
    )
    parser.add_argument(
        "--variant",
        choices=VARIANTS,
        help="Shortcut for benchmarking a single variant.",
    )
    parser.add_argument(
        "--images",
        nargs="*",
        type=Path,
        default=None,
        help="Image paths. Defaults to examples/images/*.",
    )
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations.")
    parser.add_argument("--runs", type=int, default=5, help="Timed iterations.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional merged JSON output path.",
    )
    return parser


def main() -> None:
    """Run selected backend benchmarks and print a comparison table."""
    parser = build_parser()
    args = parser.parse_args()

    if args.warmup < 0 or args.runs < 1:
        raise SystemExit("--warmup must be >= 0 and --runs must be >= 1")

    variants = parse_variants(args.variants, args.variant)
    images = resolve_example_images(tuple(args.images) if args.images else None)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    result_paths: list[Path] = []
    for backend in args.backends:
        for variant in variants:
            result_paths.append(
                run_backend(
                    backend,
                    variant=variant,
                    images=images,
                    warmup=args.warmup,
                    runs=args.runs,
                )
            )

    if args.output is None:
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        args.output = RESULTS_DIR / f"run_{stamp}.json"

    merged_records = []
    for path in result_paths:
        merged_records.extend(records_from_json(path.read_text(encoding="utf-8")))
    args.output.write_text(records_to_json(merged_records), encoding="utf-8")
    print(f"saved merged results to {args.output}", flush=True)
    print_comparison_table(result_paths)


if __name__ == "__main__":
    main()
