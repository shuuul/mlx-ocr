"""Shared PaddleOCR benchmark helpers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from paddleocr import PaddleOCR

from benchmarks.common.memory import memory_snapshot
from benchmarks.common.types import Backend, BenchmarkRecord, Variant

PaddleEngine = Literal["paddle", "onnxruntime"]


def build_paddle_pipeline(
    *,
    variant: Variant,
    engine: PaddleEngine,
    device: str = "cpu",
) -> PaddleOCR:
    """Construct a PaddleOCR pipeline aligned with mlx-ocr det+rec scope."""
    return PaddleOCR(
        ocr_version="PP-OCRv6",
        text_detection_model_name=f"PP-OCRv6_{variant}_det",
        text_recognition_model_name=f"PP-OCRv6_{variant}_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device=device,
        engine=engine,
    )


def extract_ocr_output(result: list[object]) -> tuple[int, tuple[str, ...]]:
    """Extract detection count and texts from a PaddleOCR predict result."""
    if not result:
        return 0, ()
    page = result[0]
    if not isinstance(page, dict):
        raise TypeError("expected PaddleOCR predict result page to be a dict")
    texts = tuple(str(text) for text in page["rec_texts"])
    return len(page["dt_polys"]), texts


def benchmark_variant(
    image_paths: list[Path],
    *,
    backend: Backend,
    variant: Variant,
    engine: PaddleEngine,
    device: str,
    warmup_runs: int,
    bench_runs: int,
) -> list[BenchmarkRecord]:
    """Benchmark all images for one PaddleOCR backend and variant."""
    load_start = time.perf_counter()
    pipeline = build_paddle_pipeline(variant=variant, engine=engine, device=device)
    load_s = time.perf_counter() - load_start
    memory_after_load = memory_snapshot()

    results: list[BenchmarkRecord] = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to read image: {image_path}")

        for _ in range(warmup_runs):
            pipeline.predict(str(image_path))

        warmup_start = time.perf_counter()
        pipeline.predict(str(image_path))
        warmup_s = time.perf_counter() - warmup_start

        timings: list[float] = []
        last_result = pipeline.predict(str(image_path))
        for _ in range(bench_runs):
            start = time.perf_counter()
            last_result = pipeline.predict(str(image_path))
            timings.append(time.perf_counter() - start)

        memory_after_infer = memory_snapshot()
        timing_arr = np.asarray(timings, dtype=np.float64)
        detections, texts = extract_ocr_output(last_result)
        results.append(
            BenchmarkRecord(
                backend=backend,
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
