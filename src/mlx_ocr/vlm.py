"""Optional mlx-vlm OCR backends for VLM OCR models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Literal, Protocol, cast

from mlx_ocr.types import OCRResult, OCRTextBlock

logger = logging.getLogger(__name__)

VLMEngineName = Literal["glm-ocr", "paddleocr-vl"]
VLMOCRTask = Literal["text", "formula", "table", "schema", "chart"]
GLMOCRTask = Literal["text", "formula", "table", "schema"]

_DEFAULT_MODEL_IDS: dict[VLMEngineName, str] = {
    "glm-ocr": "mlx-community/GLM-OCR-bf16",
    "paddleocr-vl": "PaddlePaddle/PaddleOCR-VL",
}
_DEFAULT_PROMPTS: dict[VLMEngineName, dict[VLMOCRTask, str]] = {
    "glm-ocr": {
        "text": "Text Recognition:",
        "formula": "Formula Recognition:",
        "table": "Table Recognition:",
        "schema": "",
    },
    "paddleocr-vl": {
        "text": "OCR:",
        "formula": "Formula Recognition:",
        "table": "Table Recognition:",
        "schema": "",
        "chart": "Chart Recognition:",
    },
}
_MISSING_VLM_MESSAGE = "Install VLM OCR support with `uv sync --extra vlm`."


class _ModelConfig(Protocol):
    """Minimal mlx-vlm model config protocol."""


class _Model(Protocol):
    """Minimal mlx-vlm model protocol."""

    config: _ModelConfig


class _GenerateResult(Protocol):
    """Minimal mlx-vlm generation result protocol."""

    text: str


class _MLXVLMRootModule(Protocol):
    """Minimal mlx-vlm root module protocol used by the VLM backend."""

    def load(self, model_id: str) -> tuple[object, object]:
        """Load an mlx-vlm model and processor."""

    def generate(
        self,
        model: object,
        processor: object,
        prompt: str,
        *,
        image: list[str],
        max_tokens: int,
    ) -> _GenerateResult:
        """Generate text for one image."""


class _MLXVLMPromptUtilsModule(Protocol):
    """Minimal mlx-vlm prompt utility module protocol."""

    def apply_chat_template(
        self,
        processor: object,
        config: _ModelConfig,
        prompt: str,
        *,
        num_images: int,
    ) -> str:
        """Format a prompt for an mlx-vlm model."""


@dataclass(frozen=True)
class _MLXVLMDependencies:
    """Lazy optional mlx-vlm dependency functions."""

    root: _MLXVLMRootModule
    prompt_utils: _MLXVLMPromptUtilsModule


def _load_mlx_vlm() -> _MLXVLMDependencies:
    try:
        root_module = import_module("mlx_vlm")
        prompt_utils_module = import_module("mlx_vlm.prompt_utils")
    except ImportError as exc:
        raise RuntimeError(_MISSING_VLM_MESSAGE) from exc
    return _MLXVLMDependencies(
        root=cast(_MLXVLMRootModule, root_module),
        prompt_utils=cast(_MLXVLMPromptUtilsModule, prompt_utils_module),
    )


def _validate_engine(engine: str) -> VLMEngineName:
    if engine not in _DEFAULT_MODEL_IDS:
        supported = ", ".join(sorted(_DEFAULT_MODEL_IDS))
        msg = f"unsupported VLM OCR engine: {engine!r}; supported engines: {supported}"
        raise ValueError(msg)
    return cast(VLMEngineName, engine)


def _validate_task(engine: VLMEngineName, task: str) -> VLMOCRTask:
    supported_tasks = _DEFAULT_PROMPTS[engine]
    if task not in supported_tasks:
        supported = ", ".join(sorted(supported_tasks))
        msg = f"unsupported {engine} OCR task: {task!r}; supported tasks: {supported}"
        raise ValueError(msg)
    return cast(VLMOCRTask, task)


def _validate_max_tokens(max_tokens: int) -> int:
    if max_tokens < 1:
        msg = "max_tokens must be at least 1"
        raise ValueError(msg)
    return max_tokens


def _resolve_prompt(task: VLMOCRTask, prompt: str | None, engine: VLMEngineName = "glm-ocr") -> str:
    engine = _validate_engine(engine)
    task = _validate_task(engine, task)
    if task == "schema":
        if prompt is None or not prompt.strip():
            msg = "schema OCR requires a prompt or schema text"
            raise ValueError(msg)
        return prompt
    if prompt is not None:
        return prompt
    return _DEFAULT_PROMPTS[engine][task]


@dataclass(frozen=True)
class VLMOCR:
    """VLM OCR backend powered by optional mlx-vlm inference."""

    model_id: str
    model: object
    processor: object
    engine: VLMEngineName = "glm-ocr"
    default_task: VLMOCRTask = "text"
    max_tokens: int = 512

    @classmethod
    def from_hub(
        cls,
        model_id: str | None = None,
        *,
        engine: VLMEngineName = "glm-ocr",
        task: VLMOCRTask = "text",
        max_tokens: int = 512,
    ) -> VLMOCR:
        """Load a VLM OCR model from Hugging Face through mlx-vlm.

        Args:
            model_id: Hugging Face model identifier for an mlx-vlm compatible
                OCR checkpoint. Defaults to the selected engine preset.
            engine: VLM OCR engine preset.
            task: Default OCR task used when ``predict_path`` does not receive
                an explicit task.
            max_tokens: Default generation token limit.

        Returns:
            Initialized VLM OCR backend.

        Raises:
            RuntimeError: If the optional ``mlx-vlm`` dependency is missing.
            ValueError: If ``model_id``, ``task``, or ``max_tokens`` is invalid.
        """
        engine = _validate_engine(engine)
        resolved_model_id = _DEFAULT_MODEL_IDS[engine] if model_id is None else model_id.strip()
        if not resolved_model_id:
            msg = "model_id must be a non-empty string"
            raise ValueError(msg)
        task = _validate_task(engine, task)
        max_tokens = _validate_max_tokens(max_tokens)
        mlx_vlm = _load_mlx_vlm()
        model, processor = mlx_vlm.root.load(resolved_model_id)
        return cls(
            model_id=resolved_model_id,
            model=model,
            processor=processor,
            engine=engine,
            default_task=task,
            max_tokens=max_tokens,
        )

    def predict_path(
        self,
        path: Path | str,
        *,
        task: VLMOCRTask | None = None,
        prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> OCRResult:
        """Run OCR on a local image path.

        Args:
            path: Local image file path.
            task: VLM OCR task. Defaults to the instance task.
            prompt: Optional custom prompt. Required for ``schema``.
            max_tokens: Optional generation token limit for this call.

        Returns:
            OCR result containing generated text as a single block.

        Raises:
            FileNotFoundError: If ``path`` is missing or is not a file.
            RuntimeError: If the optional ``mlx-vlm`` dependency is missing.
            ValueError: If ``task`` or ``max_tokens`` is invalid, or if
                ``schema`` is selected without a prompt.
        """
        image_path = Path(path)
        if not image_path.is_file():
            msg = f"image path does not exist or is not a file: {image_path}"
            raise FileNotFoundError(msg)

        selected_task = _validate_task(self.engine, self.default_task if task is None else task)
        resolved_prompt = _resolve_prompt(selected_task, prompt, self.engine)
        token_limit = _validate_max_tokens(self.max_tokens if max_tokens is None else max_tokens)
        mlx_vlm = _load_mlx_vlm()
        model = cast(_Model, self.model)
        formatted_prompt = mlx_vlm.prompt_utils.apply_chat_template(
            self.processor,
            model.config,
            resolved_prompt,
            num_images=1,
        )
        result = mlx_vlm.root.generate(
            self.model,
            self.processor,
            formatted_prompt,
            image=[str(image_path)],
            max_tokens=token_limit,
        )
        text = result.text
        return OCRResult(
            blocks=(OCRTextBlock(text=text),),
            text=text,
            engine=self.engine,
            model=self.model_id,
            prompt=resolved_prompt,
        )

    def close(self) -> None:
        """Release backend resources.

        The current mlx-vlm backend does not require explicit cleanup.
        """
