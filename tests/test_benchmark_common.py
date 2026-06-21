"""Unit tests for benchmark common helpers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

import benchmarks.run as benchmark_run
from benchmarks.common.images import REPO_ROOT, resolve_example_images
from benchmarks.common.memory import memory_snapshot, rss_mb
from benchmarks.common.types import (
    BenchmarkRecord,
    MemorySnapshot,
    records_from_json,
    records_to_json,
)
from benchmarks.compare import format_csv, format_markdown_table, record_row


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


def test_parse_backends_accepts_comma_separated_values() -> None:
    assert benchmark_run.parse_backends("mlx,paddle_cpu") == ["mlx", "paddle_cpu"]


def test_parse_backends_rejects_unknown_backend() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="unknown backend"):
        benchmark_run.parse_backends("mlx,unknown")


def test_build_parser_defaults_to_mlx_backend() -> None:
    args = benchmark_run.build_parser().parse_args([])

    assert args.backends == ["mlx"]


def test_parse_variants_resolves_single_and_multiple_values() -> None:
    assert benchmark_run.parse_variants(None, "medium") == ["medium"]
    assert benchmark_run.parse_variants(["tiny", "small"], None) == ["tiny", "small"]


def test_compare_formats_markdown_and_csv_rows() -> None:
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
        texts=("hello",),
        memory_after_load=MemorySnapshot(rss_mb=100.0, mlx_peak_mb=50.0),
        memory_after_infer=MemorySnapshot(rss_mb=120.0, mlx_peak_mb=55.0),
    )
    row = record_row(record)

    assert "| backend | variant | image |" in format_markdown_table([row])
    assert "mlx,medium,img_10.jpg" in format_csv([row])


def test_run_main_creates_custom_output_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"fake image")
    output_path = tmp_path / "nested" / "results" / "merged.json"

    def fake_run_backend(
        backend: benchmark_run.Backend,
        *,
        variant: benchmark_run.Variant,
        images: list[Path],
        warmup: int,
        runs: int,
    ) -> Path:
        assert backend == "mlx"
        assert variant == "medium"
        assert images == [image_path]
        assert warmup == 0
        assert runs == 1
        result_path = tmp_path / "runner.json"
        record = BenchmarkRecord(
            backend="mlx",
            image=image_path.name,
            image_shape=(1, 1),
            variant="medium",
            load_s=0.0,
            warmup_s=0.0,
            infer_mean_s=0.0,
            infer_std_s=0.0,
            detections=0,
            texts=(),
            memory_after_load=MemorySnapshot(rss_mb=1.0),
            memory_after_infer=MemorySnapshot(rss_mb=1.0),
        )
        result_path.write_text(records_to_json([record]), encoding="utf-8")
        return result_path

    monkeypatch.setattr(benchmark_run, "run_backend", fake_run_backend)
    monkeypatch.setattr(benchmark_run, "print_comparison_table", lambda paths: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmarks.run",
            "--variant",
            "medium",
            "--images",
            str(image_path),
            "--warmup",
            "0",
            "--runs",
            "1",
            "--output",
            str(output_path),
        ],
    )

    benchmark_run.main()

    assert output_path.is_file()
    assert records_from_json(output_path.read_text(encoding="utf-8"))[0].image == "input.png"
