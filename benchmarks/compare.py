#!/usr/bin/env python3
"""Print comparison tables from benchmark JSON results."""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

from benchmarks.common.types import BenchmarkRecord, records_from_json

TABLE_COLUMNS: tuple[str, ...] = (
    "backend",
    "variant",
    "image",
    "infer_mean_ms",
    "infer_std_ms",
    "load_s",
    "rss_after_load_mb",
    "rss_after_infer_mb",
    "mlx_peak_mb",
    "detections",
)


def record_row(record: BenchmarkRecord) -> dict[str, str]:
    """Convert one benchmark record into a printable table row."""
    return {
        "backend": record.backend,
        "variant": record.variant,
        "image": record.image,
        "infer_mean_ms": f"{record.infer_mean_s * 1000:.1f}",
        "infer_std_ms": f"{record.infer_std_s * 1000:.1f}",
        "load_s": f"{record.load_s:.3f}",
        "rss_after_load_mb": f"{record.memory_after_load.rss_mb:.1f}",
        "rss_after_infer_mb": f"{record.memory_after_infer.rss_mb:.1f}",
        "mlx_peak_mb": f"{record.memory_after_infer.mlx_peak_mb:.1f}",
        "detections": str(record.detections),
    }


def format_markdown_table(rows: list[dict[str, str]]) -> str:
    """Format rows as a markdown table."""
    if not rows:
        return "no benchmark records"

    header = "| " + " | ".join(TABLE_COLUMNS) + " |"
    separator = "| " + " | ".join("---" for _ in TABLE_COLUMNS) + " |"
    body = ["| " + " | ".join(row[column] for column in TABLE_COLUMNS) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def format_csv(rows: list[dict[str, str]]) -> str:
    """Format rows as CSV."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=TABLE_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().rstrip()


def load_records(paths: list[Path]) -> list[BenchmarkRecord]:
    """Load benchmark records from one or more JSON files."""
    records: list[BenchmarkRecord] = []
    for path in paths:
        records.extend(records_from_json(path.read_text(encoding="utf-8")))
    return records


def print_comparison_table(paths: list[Path], *, fmt: str = "markdown") -> None:
    """Print a comparison table for the given benchmark JSON files."""
    rows = [record_row(record) for record in load_records(paths)]
    rows.sort(key=lambda row: (row["variant"], row["image"], row["backend"]))
    if fmt == "csv":
        print(format_csv(rows))
        return
    print(format_markdown_table(rows))


def main() -> None:
    """CLI entrypoint for benchmark comparison tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path, help="Benchmark JSON files.")
    parser.add_argument(
        "--format",
        choices=("markdown", "csv"),
        default="markdown",
        help="Output table format.",
    )
    args = parser.parse_args()
    print_comparison_table(args.results, fmt=args.format)


if __name__ == "__main__":
    main()
