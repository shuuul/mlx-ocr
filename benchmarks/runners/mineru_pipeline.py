#!/usr/bin/env python3
"""Benchmark MinerU pipeline OCR (PytorchPaddleOCR on MPS/CUDA)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from mineru.model.ocr.pytorch_paddle import PytorchPaddleOCR
from mineru.utils.config_reader import get_device

from benchmarks.common.memory import memory_snapshot
from benchmarks.common.runner_cli import add_runner_args, resolve_runner_images
from benchmarks.common.types import BenchmarkRecord, Variant, records_to_json

MINERU_LANG_BY_VARIANT: dict[Variant, str] = {
    # MinerU 3.4 maps English/Latin to ``ch`` (PP-OCRv6 small det + small rec).
    "tiny": "ch",
    "small": "ch",
    # ``ch_server`` uses PP-OCRv6 small det + medium rec (closest to mlx medium rec).
    "medium": "ch_server",
}


def extract_ocr_output(result: list[list[list[object]]] | None) -> tuple[int, tuple[str, ...]]:
    """Extract detection count and texts from a MinerU OCR result."""
    if not result or result[0] is None:
        return 0, ()
    page = result[0]
    texts = tuple(str(item[1][0]) for item in page)
    return len(page), texts


def benchmark_variant(
    image_paths: list[Path],
    variant: Variant,
    *,
    warmup_runs: int,
    bench_runs: int,
) -> list[BenchmarkRecord]:
    """Benchmark all images for one MinerU pipeline OCR configuration."""
    lang = MINERU_LANG_BY_VARIANT[variant]

    load_start = time.perf_counter()
    pipeline = PytorchPaddleOCR(lang=lang)
    load_s = time.perf_counter() - load_start
    memory_after_load = memory_snapshot()

    results: list[BenchmarkRecord] = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to read image: {image_path}")

        for _ in range(warmup_runs):
            pipeline.ocr(image, det=True, rec=True)

        warmup_start = time.perf_counter()
        pipeline.ocr(image, det=True, rec=True)
        warmup_s = time.perf_counter() - warmup_start

        timings: list[float] = []
        last_result = pipeline.ocr(image, det=True, rec=True)
        for _ in range(bench_runs):
            start = time.perf_counter()
            last_result = pipeline.ocr(image, det=True, rec=True)
            timings.append(time.perf_counter() - start)

        memory_after_infer = memory_snapshot()
        timing_arr = np.asarray(timings, dtype=np.float64)
        detections, texts = extract_ocr_output(last_result)
        results.append(
            BenchmarkRecord(
                backend="mineru_pipeline",
                image=image_path.name,
                image_shape=(image.shape[0], image.shape[1]),
                variant=variant,
                load_s=load_s,
                warmup_s=warmup_s,
                infer_mean_s=float(timing_arr.mean()),
                infer_std_s=float(timing_arr.std()),
                detections=detections,
                texts=texts,
                memory_after_load=memory_after_load,
                memory_after_infer=memory_after_infer,
            )
        )
    return results


def main() -> None:
    """Run the MinerU pipeline OCR benchmark runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_runner_args(parser)
    args = parser.parse_args()

    if args.warmup < 0 or args.runs < 1:
        raise SystemExit("--warmup must be >= 0 and --runs must be >= 1")

    images = resolve_runner_images(args.images)
    lang = MINERU_LANG_BY_VARIANT[args.variant]
    device = get_device()
    print(
        f"benchmarking backend=mineru_pipeline variant={args.variant} "
        f"lang={lang} device={device} images={[path.name for path in images]}...",
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
