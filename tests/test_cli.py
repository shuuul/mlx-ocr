"""Tests for CLI argument helpers."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
import typer
from typer.testing import CliRunner

import mlx_ocr.cli as cli
from mlx_ocr.cli import (
    InputDocument,
    PageOCRResult,
    RenderedPage,
    collect_input_documents,
    format_json_document,
    format_markdown_document,
    format_txt_document,
    resolve_engine,
    resolve_format,
    resolve_vlm_options,
)
from mlx_ocr.output import OCRTiming
from mlx_ocr.pipeline import PipelineResult
from mlx_ocr.types import BoundingBox, OCRResult, OCRTextBlock

runner = CliRunner()


def make_result(text: str) -> OCRResult:
    """Build a minimal OCR result for CLI formatting tests."""
    return OCRResult(
        blocks=(
            OCRTextBlock(
                text=text,
                box=BoundingBox(points=((1.0, 2.0), (3.0, 2.0), (3.0, 4.0), (1.0, 4.0))),
                detection_score=0.8,
                recognition_score=0.7,
            ),
        ),
        text=text,
        engine="ppocrv6",
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


class FakeOCR:
    """Small fake pipeline for CLI contract tests."""

    def __init__(self, text: str = "Hello") -> None:
        self.text = text
        self.closed = False

    def predict(self, image: np.ndarray) -> PipelineResult:
        """Return a deterministic OCR result for any image."""
        return PipelineResult(
            result=make_result(self.text),
            timing=OCRTiming(det_s=0.0, rec_s=0.0, total_s=0.0),
        )

    def close(self) -> None:
        """Record that the fake pipeline was closed."""
        self.closed = True


class FakeVLMOCR:
    """Small fake VLM backend for CLI routing tests."""

    instances: ClassVar[list[FakeVLMOCR]] = []

    def __init__(self, model_id: str, engine: str, task: str, max_tokens: int) -> None:
        self.model_id = model_id
        self.engine = engine
        self.task = task
        self.max_tokens = max_tokens
        self.paths: list[Path] = []
        self.closed = False
        FakeVLMOCR.instances.append(self)

    @classmethod
    def from_hub(
        cls,
        model_id: str,
        *,
        engine: str = "glm-ocr",
        task: str = "text",
        max_tokens: int = 512,
    ) -> FakeVLMOCR:
        """Return a deterministic fake VLM OCR backend."""
        return cls(model_id, engine, task, max_tokens)

    def predict_path(
        self,
        path: Path | str,
        *,
        task: str | None = None,
        prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> OCRResult:
        """Record path routing and return a VLM-style OCR result."""
        image_path = Path(path)
        self.paths.append(image_path)
        assert image_path.is_file()
        assert task == self.task
        assert max_tokens == self.max_tokens
        text = f"VLM text from {image_path.name}"
        return OCRResult(
            blocks=(OCRTextBlock(text=text),),
            text=text,
            engine=self.engine,
            model=self.model_id,
            prompt=prompt or ("OCR:" if self.engine == "paddleocr-vl" else "Text Recognition:"),
        )

    def close(self) -> None:
        """Record that the fake VLM backend was closed."""
        self.closed = True


def fake_from_hub(
    cls: type[object],
    variant: str,
    *,
    drop_score: float = 0.5,
    rec_weight_source: str = "auto",
    compile_models: bool = True,
) -> FakeOCR:
    """Return a fake OCR pipeline while preserving the public call signature."""
    assert variant in {"tiny", "small", "medium"}
    assert drop_score >= 0.0
    assert rec_weight_source in {"auto", "hub", "paddle_pretrained"}
    assert isinstance(compile_models, bool)
    return FakeOCR()


def fake_read_bgr_image(path: Path) -> np.ndarray:
    """Return a deterministic image for CLI tests."""
    return np.zeros((2, 2, 3), dtype=np.uint8)


def test_resolve_format_accepts_supported_values() -> None:
    assert resolve_format("markdown") == "markdown"
    assert resolve_format("txt") == "txt"
    assert resolve_format("json") == "json"


def test_resolve_format_rejects_unsupported_value() -> None:
    with pytest.raises(typer.BadParameter, match="format must be one of"):
        resolve_format("system")


def test_resolve_engine_rejects_unsupported_value() -> None:
    with pytest.raises(typer.BadParameter, match="engine must be one of"):
        resolve_engine("bad-engine")


def test_resolve_engine_accepts_paddleocr_vl() -> None:
    assert resolve_engine("paddleocr-vl") == "paddleocr-vl"


def test_resolve_vlm_options_validates_task_schema_prompt_and_max_tokens() -> None:
    with pytest.raises(typer.BadParameter, match="vlm_task must be one of"):
        resolve_vlm_options("glm-ocr", "model", "layout", None, 512)
    with pytest.raises(typer.BadParameter, match="schema VLM OCR requires --prompt"):
        resolve_vlm_options("glm-ocr", "model", "schema", None, 512)
    with pytest.raises(typer.BadParameter, match="max_tokens must be at least 1"):
        resolve_vlm_options("glm-ocr", "model", "text", None, 0)

    options = resolve_vlm_options("glm-ocr", " model ", "schema", "extract fields", 8)
    assert options.model == "model"
    assert options.task == "schema"
    assert options.prompt == "extract fields"
    assert options.max_tokens == 8


def test_resolve_vlm_options_validates_engine_specific_tasks() -> None:
    with pytest.raises(typer.BadParameter, match="vlm_task must be one of"):
        resolve_vlm_options("glm-ocr", None, "chart", None, 512)

    options = resolve_vlm_options("paddleocr-vl", None, "chart", None, 512)

    assert options.engine == "paddleocr-vl"
    assert options.model == "PaddlePaddle/PaddleOCR-VL"
    assert options.task == "chart"


def test_format_markdown_document_adds_pdf_page_headings() -> None:
    markdown = format_markdown_document((make_page("A", 0), make_page("B", 1)))

    assert markdown == "## Page 1\n\nA\n\n## Page 2\n\nB\n"


def test_format_txt_document_adds_pdf_page_headings() -> None:
    text = format_txt_document((make_page("A", 0), make_page("B", 1)))

    assert text == "# Page 1\nA\n\n# Page 2\nB\n"


def test_format_json_document_keeps_paddlex_res_and_page_metadata(tmp_path: Path) -> None:
    document_path = tmp_path / "doc.pdf"
    document_path.write_bytes(b"%PDF")
    document = InputDocument(path=document_path, stem="doc", is_pdf=True)

    data = json.loads(format_json_document(document, (make_page("A", 0), make_page("B", 1))))

    assert data["input_path"] == str(document_path.resolve())
    assert [page["page_index"] for page in data["pages"]] == [0, 1]
    assert [page["res"]["rec_texts"] for page in data["pages"]] == [["A"], ["B"]]
    assert data["pages"][0]["res"]["dt_scores"] == [0.8]
    assert data["pages"][0]["res"]["rec_scores"] == [0.7]
    assert "blocks" not in data["pages"][0]["res"]


def test_format_json_document_uses_generalized_vlm_result(tmp_path: Path) -> None:
    document_path = tmp_path / "img.png"
    document_path.write_bytes(b"png")
    document = InputDocument(path=document_path, stem="img", is_pdf=False)
    page = PageOCRResult(
        rendered=RenderedPage(
            image=np.zeros((1, 1, 3), dtype=np.uint8),
            input_path=str(document_path),
            output_name="img.png",
            page_index=None,
        ),
        result=OCRResult(
            blocks=(OCRTextBlock(text="Generated"),),
            text="Generated",
            engine="glm-ocr",
            model="mlx-community/GLM-OCR-bf16",
            prompt="Text Recognition:",
        ),
    )

    data = json.loads(format_json_document(document, (page,)))

    page_data = data["pages"][0]
    assert "res" not in page_data
    assert page_data["result"]["engine"] == "glm-ocr"
    assert page_data["result"]["text"] == "Generated"
    assert page_data["result"]["blocks"] == [
        {
            "text": "Generated",
            "box": None,
            "detection_score": None,
            "recognition_score": None,
        }
    ]


def test_format_json_document_uses_generalized_paddleocr_vl_result(tmp_path: Path) -> None:
    document_path = tmp_path / "img.png"
    document_path.write_bytes(b"png")
    document = InputDocument(path=document_path, stem="img", is_pdf=False)
    page = PageOCRResult(
        rendered=RenderedPage(
            image=np.zeros((1, 1, 3), dtype=np.uint8),
            input_path=str(document_path),
            output_name="img.png",
            page_index=None,
        ),
        result=OCRResult(
            blocks=(OCRTextBlock(text="Generated"),),
            text="Generated",
            engine="paddleocr-vl",
            model="PaddlePaddle/PaddleOCR-VL",
            prompt="OCR:",
        ),
    )

    data = json.loads(format_json_document(document, (page,)))

    page_data = data["pages"][0]
    assert "res" not in page_data
    assert page_data["result"]["engine"] == "paddleocr-vl"
    assert page_data["result"]["prompt"] == "OCR:"


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


def test_cli_help_exposes_documented_public_options() -> None:
    result = runner.invoke(cli.app, ["--help"])

    assert result.exit_code == 0
    for option in (
        "--path",
        "--output",
        "--format",
        "--engine",
        "--variant",
        "--start",
        "--end",
        "--rec-weight-source",
        "--no-compile",
        "--vlm-model",
        "--vlm-task",
        "--prompt",
        "--max-tokens",
        "--quiet",
    ):
        assert option in result.output


def test_cli_writes_documented_image_output_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli.PP_OCRv6, "from_hub", classmethod(fake_from_hub))
    monkeypatch.setattr(cli, "read_bgr_image", fake_read_bgr_image)
    image_path = tmp_path / "input.png"
    output_dir = tmp_path / "ocr-output"
    image_path.write_bytes(b"not decoded in this test")

    result = runner.invoke(
        cli.app,
        [
            "--path",
            str(image_path),
            "--format",
            "json",
            "--output",
            str(output_dir),
            "--quiet",
        ],
    )

    assert result.exit_code == 0
    output_path = output_dir / "input" / "ocr" / "input.json"
    assert output_path.is_file()
    assert not (output_dir / "input" / "ocr" / "input_origin.pdf").exists()
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["pages"][0]["res"]["rec_texts"] == ["Hello"]


def test_cli_routes_image_input_through_vlm_predict_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_read_bgr_image(path: Path) -> np.ndarray:
        raise AssertionError(f"GLM-OCR should not decode image inputs: {path}")

    FakeVLMOCR.instances.clear()
    monkeypatch.setattr(cli, "VLMOCR", FakeVLMOCR)
    monkeypatch.setattr(cli, "read_bgr_image", fail_read_bgr_image)
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"not decoded in this test")

    result = runner.invoke(
        cli.app,
        ["--path", str(image_path), "--engine", "glm-ocr", "--format", "markdown"],
    )

    assert result.exit_code == 0
    assert result.output == "VLM text from input.png\n"
    assert len(FakeVLMOCR.instances) == 1
    assert FakeVLMOCR.instances[0].paths == [image_path]
    assert FakeVLMOCR.instances[0].closed


def test_cli_routes_image_input_through_paddleocr_vl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeVLMOCR.instances.clear()
    monkeypatch.setattr(cli, "VLMOCR", FakeVLMOCR)
    image_path = tmp_path / "chart.png"
    image_path.write_bytes(b"not decoded in this test")

    result = runner.invoke(
        cli.app,
        ["--path", str(image_path), "--engine", "paddleocr-vl", "--vlm-task", "chart"],
    )

    assert result.exit_code == 0
    assert result.output == "VLM text from chart.png\n"
    assert len(FakeVLMOCR.instances) == 1
    assert FakeVLMOCR.instances[0].engine == "paddleocr-vl"
    assert FakeVLMOCR.instances[0].model_id == "PaddlePaddle/PaddleOCR-VL"
    assert FakeVLMOCR.instances[0].task == "chart"


def test_cli_ppocrv6_ignores_invalid_vlm_only_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli.PP_OCRv6, "from_hub", classmethod(fake_from_hub))
    monkeypatch.setattr(cli, "read_bgr_image", fake_read_bgr_image)
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"not decoded in this test")

    result = runner.invoke(
        cli.app,
        ["--path", str(image_path), "--engine", "ppocrv6", "--vlm-task", "schema"],
    )

    assert result.exit_code == 0
    assert result.output == "Hello\n"


def test_cli_glm_ocr_requires_prompt_for_schema_task(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"not decoded in this test")

    result = runner.invoke(
        cli.app,
        ["--path", str(image_path), "--engine", "glm-ocr", "--vlm-task", "schema"],
    )

    assert result.exit_code != 0
    assert "schema VLM OCR requires --prompt" in result.output


def test_cli_glm_ocr_rejects_chart_task(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"not decoded in this test")

    result = runner.invoke(
        cli.app,
        ["--path", str(image_path), "--engine", "glm-ocr", "--vlm-task", "chart"],
    )

    assert result.exit_code != 0
    assert "vlm_task must be one of" in result.output


def test_cli_writes_vlm_json_without_paddlex_res(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeVLMOCR.instances.clear()
    monkeypatch.setattr(cli, "VLMOCR", FakeVLMOCR)
    monkeypatch.setattr(cli, "read_bgr_image", fake_read_bgr_image)
    image_path = tmp_path / "input.png"
    output_dir = tmp_path / "ocr-output"
    image_path.write_bytes(b"not decoded in this test")

    result = runner.invoke(
        cli.app,
        [
            "--path",
            str(image_path),
            "--engine",
            "glm-ocr",
            "--format",
            "json",
            "--output",
            str(output_dir),
            "--quiet",
        ],
    )

    assert result.exit_code == 0
    data = json.loads((output_dir / "input" / "ocr" / "input.json").read_text(encoding="utf-8"))
    page = data["pages"][0]
    assert "res" not in page
    assert page["result"]["engine"] == "glm-ocr"
    assert page["result"]["text"] == "VLM text from input.png"
    assert page["result"]["blocks"][0]["text"] == "VLM text from input.png"


def test_cli_routes_pdf_pages_to_vlm_as_temp_png(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_render_pdf_pages(
        path: Path,
        *,
        start: int | None,
        end: int | None,
    ) -> tuple[RenderedPage, ...]:
        assert path == pdf_path
        assert start is None
        assert end is None
        return (
            RenderedPage(
                image=np.zeros((2, 2, 3), dtype=np.uint8),
                input_path=f"{path.resolve()}#page=0",
                output_name="doc_page_0001",
                page_index=0,
            ),
        )

    FakeVLMOCR.instances.clear()
    monkeypatch.setattr(cli, "VLMOCR", FakeVLMOCR)
    monkeypatch.setattr(cli, "render_pdf_pages", fake_render_pdf_pages)
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF")

    result = runner.invoke(
        cli.app, ["--path", str(pdf_path), "--engine", "glm-ocr", "--format", "txt"]
    )

    assert result.exit_code == 0
    assert result.output == "# Page 1\nVLM text from doc_page_0001.png\n"
    temp_png = FakeVLMOCR.instances[0].paths[0]
    assert temp_png.name == "doc_page_0001.png"
    assert not temp_png.exists()


def test_cli_writes_documented_pdf_output_layout_and_page_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_render_pdf_pages(
        path: Path,
        *,
        start: int | None,
        end: int | None,
    ) -> tuple[RenderedPage, ...]:
        assert path == pdf_path
        assert start == 0
        assert end == 2
        return (
            RenderedPage(
                image=np.zeros((2, 2, 3), dtype=np.uint8),
                input_path=f"{path.resolve()}#page=0",
                output_name="doc_page_0001",
                page_index=0,
            ),
        )

    monkeypatch.setattr(cli.PP_OCRv6, "from_hub", classmethod(fake_from_hub))
    monkeypatch.setattr(cli, "render_pdf_pages", fake_render_pdf_pages)
    pdf_path = tmp_path / "doc.pdf"
    output_dir = tmp_path / "ocr-output"
    pdf_path.write_bytes(b"%PDF")

    result = runner.invoke(
        cli.app,
        [
            "--path",
            str(pdf_path),
            "--format",
            "txt",
            "--start",
            "0",
            "--end",
            "2",
            "--output",
            str(output_dir),
            "--quiet",
        ],
    )

    assert result.exit_code == 0
    output_path = output_dir / "doc" / "ocr" / "doc.txt"
    origin_path = output_dir / "doc" / "ocr" / "doc_origin.pdf"
    assert output_path.read_text(encoding="utf-8") == "# Page 1\nHello\n"
    assert origin_path.read_bytes() == b"%PDF"


def test_cli_rejects_pdf_page_range_for_image_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli.PP_OCRv6, "from_hub", classmethod(fake_from_hub))
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"not decoded in this test")

    result = runner.invoke(cli.app, ["--path", str(image_path), "--start", "0"])

    assert result.exit_code != 0
    assert "--start/--end can only be used with PDF inputs" in result.output


def test_pyproject_declares_documented_console_scripts() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    assert data["project"]["scripts"] == {
        "mlx-ocr": "mlx_ocr.cli:main",
        "mlx-ocr-mcp": "mlx_ocr.mcp:main",
    }
