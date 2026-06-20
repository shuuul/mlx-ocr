"""Benchmark result types and JSON serialization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Literal

Backend = Literal["mlx", "paddle_cpu", "paddle_onnx", "mineru_pipeline"]
Variant = Literal["tiny", "small", "medium"]

BACKENDS: tuple[Backend, ...] = (
    "mlx",
    "paddle_cpu",
    "paddle_onnx",
    "mineru_pipeline",
)
VARIANTS: tuple[Variant, ...] = ("tiny", "small", "medium")


@dataclass(frozen=True)
class MemorySnapshot:
    """Process RSS and optional MLX allocator readings in megabytes."""

    rss_mb: float
    mlx_active_mb: float = 0.0
    mlx_peak_mb: float = 0.0
    mlx_cache_mb: float = 0.0


@dataclass(frozen=True)
class BenchmarkRecord:
    """Benchmark result for one backend, variant, and image."""

    backend: Backend
    image: str
    image_shape: tuple[int, int]
    variant: Variant
    load_s: float
    warmup_s: float
    infer_mean_s: float
    infer_std_s: float
    detections: int
    texts: tuple[str, ...]
    memory_after_load: MemorySnapshot
    memory_after_infer: MemorySnapshot


def records_to_json(records: list[BenchmarkRecord]) -> str:
    """Serialize benchmark records to indented JSON."""
    return json.dumps([asdict(record) for record in records], indent=2)


def records_from_json(payload: str) -> list[BenchmarkRecord]:
    """Deserialize benchmark records from JSON."""
    raw = json.loads(payload)
    if not isinstance(raw, list):
        raise ValueError("benchmark JSON must be a list of records")

    records: list[BenchmarkRecord] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("each benchmark record must be a JSON object")
        records.append(
            BenchmarkRecord(
                backend=item["backend"],
                image=item["image"],
                image_shape=(int(item["image_shape"][0]), int(item["image_shape"][1])),
                variant=item["variant"],
                load_s=float(item["load_s"]),
                warmup_s=float(item["warmup_s"]),
                infer_mean_s=float(item["infer_mean_s"]),
                infer_std_s=float(item["infer_std_s"]),
                detections=int(item["detections"]),
                texts=tuple(item["texts"]),
                memory_after_load=MemorySnapshot(**item["memory_after_load"]),
                memory_after_infer=MemorySnapshot(**item["memory_after_infer"]),
            )
        )
    return records
