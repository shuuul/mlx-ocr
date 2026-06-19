#!/usr/bin/env python3
"""Benchmark PP_OCRv6 pipeline on PaddleOCR example images."""

from __future__ import annotations

import json
import resource
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import mlx.core as mx
import numpy as np

from mlx_ocr import PP_OCRv6

REPO_ROOT = Path(__file__).resolve().parents[2]
PADDLE_OCR_ROOT = REPO_ROOT.parent / "PaddleOCR"
EXAMPLE_CACHE = REPO_ROOT / ".cache" / "paddle_examples"
VARIANTS = ("tiny", "small", "medium")
WARMUP_RUNS = 2
BENCH_RUNS = 5


@dataclass(frozen=True)
class MemorySnapshot:
    """Process and MLX memory readings."""

    rss_mb: float
    mlx_active_mb: float
    mlx_peak_mb: float
    mlx_cache_mb: float


@dataclass(frozen=True)
class ImageBenchmark:
    """Benchmark result for one image and variant."""

    image: str
    image_shape: tuple[int, int]
    variant: str
    load_s: float
    warmup_s: float
    infer_mean_s: float
    infer_std_s: float
    detections: int
    texts: tuple[str, ...]
    memory_after_load: MemorySnapshot
    memory_after_infer: MemorySnapshot


def rss_mb() -> float:
    """Return peak process RSS in megabytes."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    return usage / 1024


def mlx_memory_snapshot() -> tuple[float, float, float]:
    """Return MLX active, peak, and cache memory in megabytes."""
    return (
        mx.get_active_memory() / (1024 * 1024),
        mx.get_peak_memory() / (1024 * 1024),
        mx.get_cache_memory() / (1024 * 1024),
    )


def memory_snapshot() -> MemorySnapshot:
    """Capture process RSS and MLX memory."""
    active, peak, cache = mlx_memory_snapshot()
    return MemorySnapshot(
        rss_mb=rss_mb(),
        mlx_active_mb=active,
        mlx_peak_mb=peak,
        mlx_cache_mb=cache,
    )


def resolve_example_images() -> list[Path]:
    """Resolve PaddleOCR example images referenced by PP-OCRv6 configs."""
    examples_dir = REPO_ROOT / "examples" / "images"
    candidates = [
        examples_dir / "img_10.jpg",
        examples_dir / "word_1.jpg",
        examples_dir / "sample_doc.jpg",
        EXAMPLE_CACHE / "img_10.jpg",
        PADDLE_OCR_ROOT / "doc" / "imgs_en" / "img_10.jpg",
    ]
    images: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        images.append(path)
    if not images:
        raise FileNotFoundError("no PaddleOCR example images found")
    return images


def benchmark_variant(
    image_paths: list[Path],
    variant: str,
) -> list[ImageBenchmark]:
    """Benchmark all images for one PP-OCRv6 variant."""
    mx.reset_peak_memory()
    load_start = time.perf_counter()
    pipeline = PP_OCRv6.from_hub(variant)
    load_s = time.perf_counter() - load_start
    memory_after_load = memory_snapshot()

    results: list[ImageBenchmark] = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to read image: {image_path}")

        for _ in range(WARMUP_RUNS):
            pipeline(image)
            mx.eval(mx.array(0))
        warmup_start = time.perf_counter()
        pipeline(image)
        mx.eval(mx.array(0))
        warmup_s = time.perf_counter() - warmup_start

        timings: list[float] = []
        last_result = pipeline(image)
        for _ in range(BENCH_RUNS):
            start = time.perf_counter()
            last_result = pipeline(image)
            mx.eval(mx.array(0))
            timings.append(time.perf_counter() - start)

        memory_after_infer = memory_snapshot()
        timing_arr = np.asarray(timings, dtype=np.float64)
        texts = tuple(recognition.text for recognition in last_result.recognitions)
        results.append(
            ImageBenchmark(
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
    images = resolve_example_images()
    results: list[ImageBenchmark] = []
    for variant in VARIANTS:
        print(f"benchmarking variant={variant} images={[p.name for p in images]}...", flush=True)
        results.extend(benchmark_variant(images, variant))

    payload = [asdict(result) for result in results]
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
