"""Pipeline configuration parsed from Hub inference artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from mlx4ocr.hub.download import HubArtifacts
from mlx4ocr.postprocess.db import postprocess_params_from_inference


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime settings for PP-OCRv6 end-to-end inference."""

    det_postprocess_params: dict[str, float | int | str]
    rec_image_shape: tuple[int, int, int]
    characters: tuple[str, ...]
    det_limit_side_len: int
    det_limit_type: str
    drop_score: float
    rec_batch_num: int
    det_box_type: str


def _require_mapping(value: object, key: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"expected mapping for {key}, got {type(value).__name__}")
    return value


def _characters_from_postprocess(postprocess: Mapping[str, object]) -> tuple[str, ...]:
    raw = postprocess.get("character_dict")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("PostProcess.character_dict must be a sequence of characters")
    characters = tuple(str(char) for char in raw)
    use_space_char = bool(postprocess.get("use_space_char", True))
    if use_space_char:
        return ("blank", *characters, " ")
    return ("blank", *characters)


def rec_image_shape_from_inference(inference: Mapping[str, object]) -> tuple[int, int, int]:
    """Extract recognition ``(C, H, W)`` shape from ``inference.yml``."""
    preprocess = _require_mapping(inference["PreProcess"], "PreProcess")
    transform_ops = preprocess.get("transform_ops")
    if not isinstance(transform_ops, Sequence):
        raise ValueError("PreProcess.transform_ops must be a sequence")

    for op in transform_ops:
        if not isinstance(op, Mapping):
            continue
        resize = op.get("RecResizeImg")
        if isinstance(resize, Mapping):
            image_shape = resize.get("image_shape")
            if not isinstance(image_shape, Sequence) or len(image_shape) != 3:
                raise ValueError("RecResizeImg.image_shape must contain three integers")
            return (int(image_shape[0]), int(image_shape[1]), int(image_shape[2]))
    return (3, 48, 320)


def det_resize_params_from_inference(
    inference: Mapping[str, object],
) -> tuple[int, str]:
    """Extract detection resize settings from ``inference.yml``."""
    preprocess = _require_mapping(inference["PreProcess"], "PreProcess")
    transform_ops = preprocess.get("transform_ops")
    if not isinstance(transform_ops, Sequence):
        return 960, "min"

    for op in transform_ops:
        if not isinstance(op, Mapping):
            continue
        resize = op.get("DetResizeForTest")
        if resize is None:
            continue
        if not isinstance(resize, Mapping):
            return 960, "min"
        limit_side_len = int(resize.get("limit_side_len", 960))
        limit_type = str(resize.get("limit_type", "min"))
        return limit_side_len, limit_type
    return 960, "min"


def pipeline_config_from_artifacts(
    det_artifacts: HubArtifacts,
    rec_artifacts: HubArtifacts,
    *,
    drop_score: float = 0.5,
    rec_batch_num: int = 6,
    det_box_type: str = "quad",
) -> PipelineConfig:
    """Build pipeline settings from downloaded detection and recognition artifacts.

    Args:
        det_artifacts: Local detection Hub files.
        rec_artifacts: Local recognition Hub files.
        drop_score: Minimum recognition score to keep a detection.
        rec_batch_num: Maximum recognition batch size.
        det_box_type: ``quad`` or ``poly`` crop mode.

    Returns:
        Parsed pipeline configuration.
    """
    if det_box_type not in {"quad", "poly"}:
        raise ValueError(f"det_box_type must be 'quad' or 'poly', got {det_box_type!r}")

    det_inference = det_artifacts.inference_data
    rec_inference = rec_artifacts.inference_data
    rec_postprocess = _require_mapping(rec_inference["PostProcess"], "PostProcess")
    limit_side_len, limit_type = det_resize_params_from_inference(det_inference)

    return PipelineConfig(
        det_postprocess_params=postprocess_params_from_inference(det_inference),
        rec_image_shape=rec_image_shape_from_inference(rec_inference),
        characters=_characters_from_postprocess(rec_postprocess),
        det_limit_side_len=limit_side_len,
        det_limit_type=limit_type,
        drop_score=drop_score,
        rec_batch_num=rec_batch_num,
        det_box_type=det_box_type,
    )
