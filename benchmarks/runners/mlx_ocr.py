#!/usr/bin/env python3
"""Benchmark mlx4ocr PP_OCRv6 pipeline."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import mlx.core as mx
import numpy as np

from benchmarks.common.images import resolve_example_images
from benchmarks.common.memory import memory_snapshot
from benchmarks.common.runner_cli import add_runner_args
from benchmarks.common.types import BenchmarkRecord, Variant, records_to_json
from mlx_ocr import PP_OCRv6


def benchmark_variant(
    image_paths: list[Path],
    variant: Variant,
    *,
    warmup_runs: int,
    bench_runs: int,
) -> list[BenchmarkRecord]:
    """Benchmark all images for one PP-OCRv6 variant."""
    mx.reset_peak_memory()
    load_start = time.perf_counter()
    pipeline = PP_OCRv6.from_hub(variant)
    load_s = time.perf_counter() - load_start
    memory_after_load = memory_snapshot(include_mlx=True)

    results: list[BenchmarkRecord] = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to read image: {image_path}")

        for _ in range(warmup_runs):
            pipeline(image)
            mx.eval(mx.array(0))
        warmup_start = time.perf_counter()
        pipeline(image)
        mx.eval(mx.array(0))
        warmup_s = time.perf_counter() - warmup_start

        timings: list[float] = []
        last_result = pipeline(image)
        for _ in range(bench_runs):
            start = time.perf_counter()
            last_result = pipeline(image)
            mx.eval(mx.array(0))
            timings.append(time.perf_counter() - start)

        memory_after_infer = memory_snapshot(include_mlx=True)
        timing_arr = np.asarray(timings, dtype=np.float64)
        texts = tuple(recognition.text for recognition in last_result.recognitions)
        results.append(
            BenchmarkRecord(
                backend="mlx",
                image=image_path.name,
                image_shape=(image.shape[0], image.shape[1]),
                variant=variant,
                load_s=load_s,
                warmup_s=warmup_s,
                infer_mean_s=float(timing_arr.mean()),
                infer_std_s=float(timing_arr.std()),
                detections=len(last_result.detections),
                texts=texts,
                memory_after_load=memory_after_load,
                memory_after_infer=memory_after_infer,
            )
        )
    return results


def main() -> None:
    """Run the mlx4ocr benchmark runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_runner_args(parser)
    args = parser.parse_args()

    if args.warmup < 0 or args.runs < 1:
        raise SystemExit("--warmup must be >= 0 and --runs must be >= 1")

    images = resolve_example_images(tuple(args.images) if args.images else None)
    print(
        f"benchmarking backend=mlx variant={args.variant} "
        f"images={[path.name for path in images]}...",
        flush=True,
    )
    records = benchmark_variant(
        images,
        args.variant,
        warmup_runs=args.warmup,
        bench_runs=args.runs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(records_to_json(records), encoding="utf-8")


if __name__ == "__main__":
    main()
