"""Tests for CLI argument helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import typer

from mlx_ocr.cli import (
    InputDocument,
    PageOCRResult,
    RenderedPage,
    collect_input_documents,
    format_json_document,
    format_markdown_document,
    format_txt_document,
    resolve_format,
)
from mlx_ocr.types import BoundingBox, OCRResult, TextDetection, TextRecognition


def make_result(text: str) -> OCRResult:
    """Build a minimal OCR result for CLI formatting tests."""
    return OCRResult(
        detections=(
            TextDetection(
                box=BoundingBox(points=((1.0, 2.0), (3.0, 2.0), (3.0, 4.0), (1.0, 4.0))),
                score=0.8,
            ),
        ),
        recognitions=(TextRecognition(text=text, score=0.7),),
    )


def make_page(text: str, page_index: int | None) -> PageOCRResult:
    """Build a rendered page OCR result for CLI formatting tests."""
    name = "img_10.jpg" if page_index is None else f"doc_page_{page_index + 1:04d}"
    return PageOCRResult(
        rendered=RenderedPage(
            image=np.zeros((1, 1, 3), dtype=np.uint8),
            input_path="/tmp/input",
            output_name=name,
            page_index=page_index,
        ),
        result=make_result(text),
    )


def test_resolve_format_accepts_supported_values() -> None:
    assert resolve_format("markdown") == "markdown"
    assert resolve_format("txt") == "txt"
    assert resolve_format("json") == "json"


def test_resolve_format_rejects_unsupported_value() -> None:
    with pytest.raises(typer.BadParameter, match="format must be one of"):
        resolve_format("system")


def test_format_markdown_document_adds_pdf_page_headings() -> None:
    markdown = format_markdown_document((make_page("A", 0), make_page("B", 1)))

    assert markdown == "## Page 1\n\nA\n\n## Page 2\n\nB\n"


def test_format_txt_document_adds_pdf_page_headings() -> None:
    text = format_txt_document((make_page("A", 0), make_page("B", 1)))

    assert text == "# Page 1\nA\n\n# Page 2\nB\n"


def test_format_json_document_keeps_page_metadata(tmp_path: Path) -> None:
    document_path = tmp_path / "doc.pdf"
    document_path.write_bytes(b"%PDF")
    document = InputDocument(path=document_path, stem="doc", is_pdf=True)

    data = json.loads(format_json_document(document, (make_page("A", 0), make_page("B", 1))))

    assert data["input_path"] == str(document_path.resolve())
    assert [page["page_index"] for page in data["pages"]] == [0, 1]
    assert [page["res"]["rec_texts"] for page in data["pages"]] == [["A"], ["B"]]


def test_collect_input_documents_accepts_files_and_directories(tmp_path: Path) -> None:
    image_a = tmp_path / "a.jpg"
    image_b = tmp_path / "b.png"
    pdf = tmp_path / "doc.pdf"
    text_file = tmp_path / "notes.txt"
    image_a.write_bytes(b"")
    image_b.write_bytes(b"")
    pdf.write_bytes(b"")
    text_file.write_text("skip", encoding="utf-8")

    documents = collect_input_documents((image_a, tmp_path))
    assert tuple(document.path for document in documents) == (image_a, image_a, image_b, pdf)


def test_collect_input_documents_marks_pdf_inputs(tmp_path: Path) -> None:
    image = tmp_path / "a.jpg"
    pdf = tmp_path / "doc.pdf"
    image.write_bytes(b"")
    pdf.write_bytes(b"")

    documents = collect_input_documents((image, pdf))

    assert [document.stem for document in documents] == ["a", "doc"]
    assert [document.is_pdf for document in documents] == [False, True]


def test_collect_input_documents_rejects_missing_input() -> None:
    with pytest.raises(typer.BadParameter, match="missing input path"):
        collect_input_documents((Path("missing.jpg"),))
