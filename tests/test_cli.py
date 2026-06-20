"""Tests for CLI argument helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer

from mlx_ocr.cli import expand_image_paths, resolve_formats


def test_resolve_formats_defaults_to_all_outputs() -> None:
    assert resolve_formats(None, quiet=False) == ("text", "system", "json", "markdown")


def test_resolve_formats_quiet_default_skips_text() -> None:
    assert resolve_formats(None, quiet=True) == ("system", "json", "markdown")


def test_resolve_formats_preserves_requested_order_without_duplicates() -> None:
    assert resolve_formats(["markdown", "json", "markdown"], quiet=False) == ("markdown", "json")


def test_expand_image_paths_accepts_files_and_directories(tmp_path: Path) -> None:
    image_a = tmp_path / "a.jpg"
    image_b = tmp_path / "b.png"
    text_file = tmp_path / "notes.txt"
    image_a.write_bytes(b"")
    image_b.write_bytes(b"")
    text_file.write_text("skip", encoding="utf-8")

    assert expand_image_paths((image_a, tmp_path)) == (image_a, image_a, image_b)


def test_expand_image_paths_rejects_missing_input() -> None:
    with pytest.raises(typer.BadParameter, match="missing input path"):
        expand_image_paths((Path("missing.jpg"),))
