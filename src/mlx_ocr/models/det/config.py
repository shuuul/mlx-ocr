"""Detection model configuration parsed from Hugging Face artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from mlx_ocr.hub.download import HubArtifacts

BlockSpec = tuple[int, int, int, int | tuple[int, int], bool]
NeckKind = Literal["replkfpn", "replkpan"]


@dataclass(frozen=True)
class DetModelConfig:
    """Structured PP-OCRv6 detection architecture settings."""

    variant: str
    stem_mid: int
    stem_out: int
    stage_channels: tuple[int, int, int, int]
    block_configs: tuple[tuple[BlockSpec, ...], ...]
    neck_kind: NeckKind
    neck_out_channels: int
    dilated_kernel_size: int
    intracl: bool
    head_kernel_list: tuple[int, int, int]


def _require_mapping(value: object, key: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"expected mapping for {key}, got {type(value).__name__}")
    return value


def _parse_block_configs(raw: object) -> tuple[tuple[BlockSpec, ...], ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("block_configs must be a nested sequence")
    stages: list[tuple[BlockSpec, ...]] = []
    for stage in raw:
        if not isinstance(stage, Sequence) or isinstance(stage, (str, bytes)):
            raise ValueError("each block stage must be a sequence")
        specs: list[BlockSpec] = []
        for block in stage:
            if not isinstance(block, Sequence) or len(block) != 5:
                raise ValueError(f"invalid block spec: {block!r}")
            kernel, in_ch, out_ch, stride, use_se = block
            parsed_stride: int | tuple[int, int]
            if isinstance(stride, Sequence) and not isinstance(stride, (str, bytes)):
                if len(stride) != 2:
                    raise ValueError(f"invalid stride tuple: {stride!r}")
                parsed_stride = (int(stride[0]), int(stride[1]))
            else:
                parsed_stride = int(stride)
            specs.append((int(kernel), int(in_ch), int(out_ch), parsed_stride, bool(use_se)))
        stages.append(tuple(specs))
    if len(stages) != 4:
        raise ValueError(f"expected 4 detection stages, got {len(stages)}")
    return tuple(stages)


def det_config_from_artifacts(artifacts: HubArtifacts) -> DetModelConfig:
    """Build a detection config from Hub ``config.json``.

    Args:
        artifacts: Downloaded model artifacts for a detection checkpoint.

    Returns:
        Parsed detection architecture config.

    Raises:
        ValueError: If required config fields are missing or malformed.
    """
    config = artifacts.config_data
    neck_raw = config.get("neck_config", config)
    neck = _require_mapping(neck_raw, "neck_config")
    backbone = _require_mapping(neck["backbone_config"], "backbone_config")
    stem_channels = backbone.get("stem_channels", [3, 24, 48])
    if not isinstance(stem_channels, Sequence) or len(stem_channels) != 3:
        raise ValueError("stem_channels must contain three integers")
    stem_mid = int(stem_channels[1])
    stem_out = int(stem_channels[2])

    layer_list = neck.get("layer_list_out_channels")
    if layer_list is None:
        block_configs = _parse_block_configs(backbone["block_configs"])
        layer_list = [spec[-2][2] for spec in block_configs]
    if not isinstance(layer_list, Sequence) or len(layer_list) != 4:
        raise ValueError("layer_list_out_channels must contain four stage widths")

    neck_out = int(neck["neck_out_channels"])
    intracl = int(neck.get("intraclass_block_number", 0)) > 0
    neck_kind: NeckKind = "replkpan" if intracl else "replkfpn"
    dilated_kernel_size = int(neck.get("dilated_kernel_size", 9 if intracl else 5))

    kernel_list_raw = neck.get("kernel_list", [3, 2, 2])
    if not isinstance(kernel_list_raw, Sequence) or len(kernel_list_raw) != 3:
        raise ValueError("kernel_list must contain three integers")
    kernel_list = (int(kernel_list_raw[0]), int(kernel_list_raw[1]), int(kernel_list_raw[2]))

    variant = str(config.get("model_type", "unknown")).removeprefix("pp_ocrv6_")
    variant = variant.removesuffix("_det")
    if variant.endswith("_det"):
        variant = variant[:-4]

    return DetModelConfig(
        variant=variant,
        stem_mid=stem_mid,
        stem_out=stem_out,
        stage_channels=tuple(int(ch) for ch in layer_list),
        block_configs=_parse_block_configs(backbone["block_configs"]),
        neck_kind=neck_kind,
        neck_out_channels=neck_out,
        dilated_kernel_size=dilated_kernel_size,
        intracl=intracl,
        head_kernel_list=kernel_list,
    )
