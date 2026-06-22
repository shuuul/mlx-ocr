"""Hugging Face Hub model identifiers for PP-OCRv6 safetensors checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ModelVariant = Literal["tiny", "small", "medium"]
ModelTask = Literal["det", "rec"]

_DET_REPOS: dict[ModelVariant, str] = {
    "tiny": "PaddlePaddle/PP-OCRv6_tiny_det_safetensors",
    "small": "PaddlePaddle/PP-OCRv6_small_det_safetensors",
    "medium": "PaddlePaddle/PP-OCRv6_medium_det_safetensors",
}

_REC_REPOS: dict[ModelVariant, str] = {
    "tiny": "PaddlePaddle/PP-OCRv6_tiny_rec_safetensors",
    "small": "PaddlePaddle/PP-OCRv6_small_rec_safetensors",
    "medium": "PaddlePaddle/PP-OCRv6_medium_rec_safetensors",
}

PP_OCRV6_COLLECTION_URL = "https://huggingface.co/collections/PaddlePaddle/pp-ocrv6"


@dataclass(frozen=True)
class HubModelRef:
    """Reference to a PP-OCRv6 safetensors repo on the Hugging Face Hub."""

    repo_id: str
    variant: ModelVariant
    task: ModelTask


def hub_model_ref(variant: ModelVariant, task: ModelTask) -> HubModelRef:
    """Resolve the Hub repo id for a PP-OCRv6 variant and task.

    Args:
        variant: Model size tier (`tiny`, `small`, or `medium`).
        task: Pipeline stage (`det` for detection, `rec` for recognition).

    Returns:
        Structured reference containing the repo id and metadata.

    Raises:
        KeyError: If variant or task is not supported.
    """
    if task == "det":
        repo_id = _DET_REPOS[variant]
    else:
        repo_id = _REC_REPOS[variant]
    return HubModelRef(repo_id=repo_id, variant=variant, task=task)


def list_hub_models() -> tuple[HubModelRef, ...]:
    """Return all supported PP-OCRv6 safetensors Hub references."""
    refs: list[HubModelRef] = []
    for variant in ("tiny", "small", "medium"):
        for task in ("det", "rec"):
            refs.append(hub_model_ref(variant, task))
    return tuple(refs)
