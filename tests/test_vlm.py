"""Tests for the optional VLM OCR backend."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

import pytest

import mlx_ocr.vlm as vlm
from mlx_ocr import VLMOCR
from mlx_ocr.vlm import GLMOCRTask, VLMOCRTask, _resolve_prompt


class FakeMLXVLM(ModuleType):
    """Stub mlx-vlm root module for deterministic unit tests."""

    def __init__(self) -> None:
        super().__init__("mlx_vlm")
        self.model = SimpleNamespace(config=SimpleNamespace(name="fake-config"))
        self.processor = SimpleNamespace(name="fake-processor")
        self.loaded_model_id: str | None = None
        self.generate_calls: list[tuple[object, object, str, list[str], int]] = []

    def load(self, model_id: str) -> tuple[object, object]:
        """Return fake model objects and record the requested model id."""
        self.loaded_model_id = model_id
        return self.model, self.processor

    def generate(
        self,
        model: object,
        processor: object,
        prompt: str,
        *,
        image: list[str],
        max_tokens: int,
    ) -> FakeGenerateResult:
        """Record generation inputs and return fake generated text."""
        self.generate_calls.append((model, processor, prompt, image, max_tokens))
        return FakeGenerateResult(text="generated text")


class FakePromptUtils(ModuleType):
    """Stub mlx-vlm prompt_utils module for deterministic unit tests."""

    def __init__(self) -> None:
        super().__init__("mlx_vlm.prompt_utils")
        self.template_calls: list[tuple[object, object, str, int]] = []

    def apply_chat_template(
        self,
        processor: object,
        config: object,
        prompt: str,
        *,
        num_images: int,
    ) -> str:
        """Record template inputs and return a formatted prompt."""
        self.template_calls.append((processor, config, prompt, num_images))
        return f"formatted:{prompt}"


@dataclass(frozen=True)
class FakeGenerateResult:
    """Fake mlx-vlm generation result."""

    text: str


@pytest.fixture
def fake_mlx_vlm(monkeypatch: pytest.MonkeyPatch) -> tuple[FakeMLXVLM, FakePromptUtils]:
    """Install a fake mlx_vlm module for one test."""
    module = FakeMLXVLM()
    prompt_utils = FakePromptUtils()
    monkeypatch.setitem(sys.modules, "mlx_vlm", module)
    monkeypatch.setitem(sys.modules, "mlx_vlm.prompt_utils", prompt_utils)
    return module, prompt_utils


def test_from_hub_raises_clear_error_when_mlx_vlm_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_import_error(name: str) -> ModuleType:
        if name == "mlx_vlm":
            raise ImportError(name)
        return __import__(name)

    monkeypatch.setattr(vlm, "import_module", raise_import_error)

    with pytest.raises(RuntimeError, match="uv sync --extra vlm"):
        VLMOCR.from_hub()


def test_from_hub_raises_clear_error_when_prompt_utils_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_module = FakeMLXVLM()

    def import_fake_modules(name: str) -> ModuleType:
        if name == "mlx_vlm":
            return root_module
        if name == "mlx_vlm.prompt_utils":
            raise ImportError(name)
        return __import__(name)

    monkeypatch.setattr(vlm, "import_module", import_fake_modules)

    with pytest.raises(RuntimeError, match="uv sync --extra vlm"):
        VLMOCR.from_hub()


def test_from_hub_rejects_invalid_task(fake_mlx_vlm: tuple[FakeMLXVLM, FakePromptUtils]) -> None:
    with pytest.raises(ValueError, match="supported tasks"):
        VLMOCR.from_hub(task=cast(VLMOCRTask, "layout"))


def test_from_hub_uses_paddleocr_vl_default_model(
    fake_mlx_vlm: tuple[FakeMLXVLM, FakePromptUtils],
) -> None:
    root_module, _ = fake_mlx_vlm

    ocr = VLMOCR.from_hub(engine="paddleocr-vl")

    assert root_module.loaded_model_id == "PaddlePaddle/PaddleOCR-VL"
    assert ocr.engine == "paddleocr-vl"
    assert ocr.model_id == "PaddlePaddle/PaddleOCR-VL"


@pytest.mark.parametrize("max_tokens", [0, -1])
def test_from_hub_rejects_invalid_max_tokens(
    fake_mlx_vlm: tuple[FakeMLXVLM, FakePromptUtils],
    max_tokens: int,
) -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        VLMOCR.from_hub(max_tokens=max_tokens)


def test_from_hub_rejects_blank_model_id(
    fake_mlx_vlm: tuple[FakeMLXVLM, FakePromptUtils],
) -> None:
    with pytest.raises(ValueError, match="model_id"):
        VLMOCR.from_hub("  ")


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("text", "Text Recognition:"),
        ("formula", "Formula Recognition:"),
        ("table", "Table Recognition:"),
    ],
)
def test_resolve_prompt_uses_default_task_prompts(task: GLMOCRTask, expected: str) -> None:
    assert _resolve_prompt(task, None) == expected


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("text", "OCR:"),
        ("formula", "Formula Recognition:"),
        ("table", "Table Recognition:"),
        ("chart", "Chart Recognition:"),
    ],
)
def test_resolve_prompt_uses_paddleocr_vl_default_task_prompts(
    task: VLMOCRTask,
    expected: str,
) -> None:
    assert _resolve_prompt(task, None, "paddleocr-vl") == expected


def test_resolve_prompt_rejects_chart_for_glm_ocr() -> None:
    with pytest.raises(ValueError, match="unsupported glm-ocr OCR task"):
        _resolve_prompt("chart", None, "glm-ocr")


def test_resolve_prompt_uses_custom_prompt() -> None:
    assert _resolve_prompt("text", "Read all headings") == "Read all headings"


def test_resolve_prompt_requires_schema_prompt() -> None:
    with pytest.raises(ValueError, match="schema OCR requires"):
        _resolve_prompt("schema", None)


def test_resolve_prompt_uses_schema_prompt() -> None:
    assert _resolve_prompt("schema", "Return JSON with total") == "Return JSON with total"


def test_predict_path_returns_block_ocr_result(
    tmp_path: Path,
    fake_mlx_vlm: tuple[FakeMLXVLM, FakePromptUtils],
) -> None:
    root_module, prompt_utils = fake_mlx_vlm
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"not a real image; fake backend does not decode")
    ocr = VLMOCR.from_hub("test/model", task="formula", max_tokens=32)

    result = ocr.predict_path(image_path, max_tokens=12)

    assert root_module.loaded_model_id == "test/model"
    assert prompt_utils.template_calls == [
        (root_module.processor, root_module.model.config, "Formula Recognition:", 1)
    ]
    assert root_module.generate_calls == [
        (
            root_module.model,
            root_module.processor,
            "formatted:Formula Recognition:",
            [str(image_path)],
            12,
        )
    ]
    assert result.engine == "glm-ocr"
    assert result.model == "test/model"
    assert result.prompt == "Formula Recognition:"
    assert result.text == "generated text"
    assert len(result.blocks) == 1
    assert result.blocks[0].text == "generated text"
    assert result.blocks[0].box is None
    assert result.blocks[0].detection_score is None
    assert result.blocks[0].recognition_score is None


def test_predict_path_rejects_invalid_task(
    tmp_path: Path,
    fake_mlx_vlm: tuple[FakeMLXVLM, FakePromptUtils],
) -> None:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"not a real image; fake backend does not decode")
    ocr = VLMOCR.from_hub()

    with pytest.raises(ValueError, match="supported tasks"):
        ocr.predict_path(image_path, task=cast(VLMOCRTask, "layout"))


def test_predict_path_allows_chart_for_paddleocr_vl(
    tmp_path: Path,
    fake_mlx_vlm: tuple[FakeMLXVLM, FakePromptUtils],
) -> None:
    root_module, prompt_utils = fake_mlx_vlm
    image_path = tmp_path / "chart.png"
    image_path.write_bytes(b"not a real image; fake backend does not decode")
    ocr = VLMOCR.from_hub(engine="paddleocr-vl", task="chart")

    result = ocr.predict_path(image_path)

    assert prompt_utils.template_calls == [
        (root_module.processor, root_module.model.config, "Chart Recognition:", 1)
    ]
    assert result.engine == "paddleocr-vl"
    assert result.prompt == "Chart Recognition:"


@pytest.mark.parametrize("max_tokens", [0, -1])
def test_predict_path_rejects_invalid_max_tokens(
    tmp_path: Path,
    fake_mlx_vlm: tuple[FakeMLXVLM, FakePromptUtils],
    max_tokens: int,
) -> None:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"not a real image; fake backend does not decode")
    ocr = VLMOCR.from_hub()

    with pytest.raises(ValueError, match="max_tokens"):
        ocr.predict_path(image_path, max_tokens=max_tokens)


def test_predict_path_rejects_missing_file(
    fake_mlx_vlm: tuple[FakeMLXVLM, FakePromptUtils],
) -> None:
    ocr = VLMOCR.from_hub()

    with pytest.raises(FileNotFoundError, match="image path"):
        ocr.predict_path("missing.png")
