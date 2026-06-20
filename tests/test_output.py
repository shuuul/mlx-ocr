"""Tests for upstream-aligned OCR output serialization."""

from __future__ import annotations

import json
from pathlib import Path

from mlx_ocr.output import to_markdown, to_system_results_line
from mlx_ocr.types import BoundingBox, OCRResult, TextDetection, TextRecognition


def test_system_results_line_matches_predict_system_shape() -> None:
    result = OCRResult(
        detections=(
            TextDetection(
                box=BoundingBox(points=((10.0, 20.0), (100.0, 20.0), (100.0, 40.0), (10.0, 40.0))),
                score=0.9,
            ),
        ),
        recognitions=(TextRecognition(text="Hello", score=0.95),),
    )
    line = to_system_results_line(result, "img_10.jpg")
    basename, payload_text = line.rstrip("\n").split("\t", maxsplit=1)
    assert basename == "img_10.jpg"
    payload = json.loads(payload_text)
    assert payload == [
        {
            "transcription": "Hello",
            "points": [[10, 20], [100, 20], [100, 40], [10, 40]],
        }
    ]


def test_save_to_json_writes_paddlex_layout(tmp_path: Path) -> None:
    result = OCRResult(
        detections=(
            TextDetection(
                box=BoundingBox(points=((1.0, 2.0), (3.0, 2.0), (3.0, 4.0), (1.0, 4.0))),
                score=0.8,
            ),
        ),
        recognitions=(TextRecognition(text="A", score=0.7),),
    )
    path = result.save_to_json(tmp_path, input_path="examples/images/img_10.jpg")
    assert path.name == "img_10_res.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["res"]["rec_texts"] == ["A"]
    assert data["res"]["rec_scores"] == [0.7]
    assert data["res"]["input_path"] == "examples/images/img_10.jpg"


def test_to_markdown_includes_text_scores_and_points() -> None:
    result = OCRResult(
        detections=(
            TextDetection(
                box=BoundingBox(points=((1.0, 2.0), (3.0, 2.0), (3.0, 4.0), (1.0, 4.0))),
                score=0.8,
            ),
        ),
        recognitions=(TextRecognition(text="A|B", score=0.7),),
    )

    markdown = to_markdown(result, title="img_10", input_path="examples/images/img_10.jpg")

    assert markdown.startswith("# img_10\n")
    assert "Source: `examples/images/img_10.jpg`" in markdown
    assert "A|B" in markdown
    assert "| 1 | A\\|B | 0.700000 | `[[1, 2], [3, 2], [3, 4], [1, 4]]` |" in markdown


def test_save_to_markdown_uses_input_stem(tmp_path: Path) -> None:
    result = OCRResult(
        detections=(),
        recognitions=(),
    )

    path = result.save_to_markdown(tmp_path, input_path="examples/images/img_10.jpg")

    assert path == tmp_path / "img_10.md"
    assert "No text detected." in path.read_text(encoding="utf-8")
