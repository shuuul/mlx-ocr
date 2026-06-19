#!/usr/bin/env python3
"""Benchmark PaddleOCR PP-OCRv6 with the Paddle CPU engine."""

from __future__ import annotations

import argparse

from benchmarks.common.runner_cli import add_runner_args, resolve_runner_images
from benchmarks.common.types import records_to_json
from benchmarks.runners._paddle import benchmark_variant


def main() -> None:
    """Run the PaddleOCR CPU benchmark runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_runner_args(parser)
    args = parser.parse_args()

    if args.warmup < 0 or args.runs < 1:
        raise SystemExit("--warmup must be >= 0 and --runs must be >= 1")

    images = resolve_runner_images(args.images)
    print(
        f"benchmarking backend=paddle_cpu variant={args.variant} "
        f"images={[path.name for path in images]}...",
        flush=True,
    )
    records = benchmark_variant(
        images,
        backend="paddle_cpu",
        variant=args.variant,
        engine="paddle",
        device="cpu",
        warmup_runs=args.warmup,
        bench_runs=args.runs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(records_to_json(records), encoding="utf-8")


if __name__ == "__main__":
    main()
