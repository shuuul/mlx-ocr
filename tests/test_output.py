"""Tests for upstream-aligned OCR output serialization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlx_ocr.output import save_to_json, save_to_markdown, to_markdown, to_system_results_line
from mlx_ocr.types import BoundingBox, OCRResult, OCRTextBlock, TextDetection, TextRecognition


def test_ocr_result_from_ppocrv6_builds_blocks() -> None:
    box_a = BoundingBox(points=((1.0, 2.0), (3.0, 2.0), (3.0, 4.0), (1.0, 4.0)))
    box_b = BoundingBox(points=((5.0, 6.0), (7.0, 6.0), (7.0, 8.0), (5.0, 8.0)))

    result = OCRResult.from_ppocrv6(
        (
            TextDetection(box=box_a, score=0.8),
            TextDetection(box=box_b, score=0.6),
        ),
        (
            TextRecognition(text="Hello", score=0.95),
            TextRecognition(text="World", score=0.85),
        ),
        model="medium",
    )

    assert result.engine == "ppocrv6"
    assert result.model == "medium"
    assert result.text == "Hello\nWorld"
    assert result.blocks == (
        OCRTextBlock(text="Hello", box=box_a, detection_score=0.8, recognition_score=0.95),
        OCRTextBlock(text="World", box=box_b, detection_score=0.6, recognition_score=0.85),
    )


def test_ocr_result_from_ppocrv6_rejects_mismatched_lengths() -> None:
    box = BoundingBox(points=((1.0, 2.0), (3.0, 2.0), (3.0, 4.0), (1.0, 4.0)))

    with pytest.raises(ValueError):
        OCRResult.from_ppocrv6(
            (TextDetection(box=box, score=0.8),),
            (),
        )


def test_system_results_line_matches_predict_system_shape() -> None:
    result = OCRResult(
        blocks=(
            OCRTextBlock(
                text="Hello",
                box=BoundingBox(points=((10.0, 20.0), (100.0, 20.0), (100.0, 40.0), (10.0, 40.0))),
                detection_score=0.9,
                recognition_score=0.95,
            ),
        ),
        text="Hello",
        engine="ppocrv6",
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
        blocks=(
            OCRTextBlock(
                text="A",
                box=BoundingBox(points=((1.0, 2.0), (3.0, 2.0), (3.0, 4.0), (1.0, 4.0))),
                detection_score=0.8,
                recognition_score=0.7,
            ),
        ),
        text="A",
        engine="ppocrv6",
    )
    path = save_to_json(result, tmp_path, input_path="examples/images/img_10.jpg")
    assert path.name == "img_10_res.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["res"]["rec_texts"] == ["A"]
    assert data["res"]["rec_scores"] == [0.7]
    assert data["res"]["input_path"] == "examples/images/img_10.jpg"


def test_save_to_json_requires_geometry_for_paddlex_layout(tmp_path: Path) -> None:
    result = OCRResult(
        blocks=(OCRTextBlock(text="generated text"),),
        text="generated text",
        engine="glm-ocr",
    )

    with pytest.raises(ValueError, match="missing geometry"):
        save_to_json(result, tmp_path, input_path="examples/images/img_10.jpg")


def test_save_to_json_requires_detection_score_for_paddlex_layout(tmp_path: Path) -> None:
    result = OCRResult(
        blocks=(
            OCRTextBlock(
                text="A",
                box=BoundingBox(points=((1.0, 2.0), (3.0, 2.0), (3.0, 4.0), (1.0, 4.0))),
                recognition_score=0.7,
            ),
        ),
        text="A",
        engine="ppocrv6",
    )

    with pytest.raises(ValueError, match="missing detection score"):
        save_to_json(result, tmp_path, input_path="examples/images/img_10.jpg")


def test_save_to_json_requires_recognition_score_for_paddlex_layout(tmp_path: Path) -> None:
    result = OCRResult(
        blocks=(
            OCRTextBlock(
                text="A",
                box=BoundingBox(points=((1.0, 2.0), (3.0, 2.0), (3.0, 4.0), (1.0, 4.0))),
                detection_score=0.8,
            ),
        ),
        text="A",
        engine="ppocrv6",
    )

    with pytest.raises(ValueError, match="missing recognition score"):
        save_to_json(result, tmp_path, input_path="examples/images/img_10.jpg")


def test_to_markdown_returns_text_body_only() -> None:
    result = OCRResult(
        blocks=(
            OCRTextBlock(
                text="A|B",
                box=BoundingBox(points=((1.0, 2.0), (3.0, 2.0), (3.0, 4.0), (1.0, 4.0))),
                detection_score=0.8,
                recognition_score=0.7,
            ),
        ),
        text="A|B",
        engine="ppocrv6",
    )

    markdown = to_markdown(result)

    assert markdown == "A|B\n"


def test_save_to_markdown_uses_input_stem(tmp_path: Path) -> None:
    result = OCRResult(
        blocks=(),
        text="",
        engine="ppocrv6",
    )

    path = save_to_markdown(result, tmp_path, input_path="examples/images/img_10.jpg")

    assert path == tmp_path / "img_10.md"
    assert path.read_text(encoding="utf-8") == ""
