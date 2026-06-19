"""Unit tests for benchmark common helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.common.images import REPO_ROOT, resolve_example_images
from benchmarks.common.memory import memory_snapshot, rss_mb
from benchmarks.common.types import (
    BenchmarkRecord,
    MemorySnapshot,
    records_from_json,
    records_to_json,
)


def test_rss_mb_positive() -> None:
    assert rss_mb() > 0.0


def test_memory_snapshot_without_mlx() -> None:
    snapshot = memory_snapshot(include_mlx=False)
    assert snapshot.rss_mb > 0.0
    assert snapshot.mlx_active_mb == 0.0
    assert snapshot.mlx_peak_mb == 0.0
    assert snapshot.mlx_cache_mb == 0.0


def test_resolve_example_images_finds_repo_images() -> None:
    images = resolve_example_images()
    assert images
    assert all(path.is_file() for path in images)
    assert images[0].is_relative_to(REPO_ROOT / "examples" / "images")


def test_resolve_example_images_explicit_path() -> None:
    image = REPO_ROOT / "examples" / "images" / "img_10.jpg"
    if not image.is_file():
        pytest.skip("example image missing")
    resolved = resolve_example_images((image,))
    assert resolved == [image]


def test_resolve_example_images_missing_explicit_path() -> None:
    with pytest.raises(ValueError, match="image not found"):
        resolve_example_images((Path("missing-image.jpg"),))


def test_benchmark_record_json_round_trip() -> None:
    record = BenchmarkRecord(
        backend="mlx",
        image="img_10.jpg",
        image_shape=(480, 640),
        variant="medium",
        load_s=1.25,
        warmup_s=0.4,
        infer_mean_s=0.12,
        infer_std_s=0.01,
        detections=3,
        texts=("hello", "world"),
        memory_after_load=MemorySnapshot(rss_mb=100.0, mlx_peak_mb=50.0),
        memory_after_infer=MemorySnapshot(rss_mb=120.0, mlx_peak_mb=55.0),
    )
    payload = records_to_json([record])
    restored = records_from_json(payload)
    assert restored == [record]
