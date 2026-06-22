"""Recognition model configuration parsed from Hugging Face artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from mlx4ocr.hub.download import HubArtifacts

BlockSpec = tuple[int, int, int, int | tuple[int, int], bool]
EncoderKind = Literal["reshape", "lightsvtr"]
StemKind = Literal["simple", "branch"]


@dataclass(frozen=True)
class RecModelConfig:
    """Structured PP-OCRv6 recognition architecture settings."""

    variant: str
    stem_kind: StemKind
    stem_mid: int
    stem_out: int
    block_configs: tuple[tuple[BlockSpec, ...], ...]
    out_channels: int
    encoder_kind: EncoderKind
    encoder_dims: int
    encoder_depth: int
    encoder_mlp_ratio: float
    local_kernel: int
    use_guide: bool
    mid_channels: int | None
    num_classes: int


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
    return tuple(stages)


def _stem_kind_from_config(stem_type: str) -> StemKind:
    if stem_type in {"small", "simple"}:
        return "simple"
    if stem_type in {"large", "branch"}:
        return "branch"
    raise ValueError(f"unsupported stem_type: {stem_type!r}")


def rec_config_from_artifacts(artifacts: HubArtifacts) -> RecModelConfig:
    """Build a recognition config from Hub ``config.json``.

    Args:
        artifacts: Downloaded model artifacts for a recognition checkpoint.

    Returns:
        Parsed recognition architecture config.

    Raises:
        ValueError: If required config fields are missing or malformed.
    """
    config = artifacts.config_data
    backbone = _require_mapping(config["backbone_config"], "backbone_config")
    stem_channels = backbone.get("stem_channels", [3, 24, 48])
    if not isinstance(stem_channels, Sequence) or len(stem_channels) != 3:
        raise ValueError("stem_channels must contain three integers")
    stem_mid = int(stem_channels[1])
    stem_out = int(stem_channels[2])

    stem_type = str(backbone.get("stem_type", "small"))
    block_configs = _parse_block_configs(backbone["block_configs"])

    out_channels = 0
    for stage in reversed(block_configs):
        if stage:
            out_channels = stage[-1][2]
            break
    if out_channels == 0:
        raise ValueError("block_configs must contain at least one block")

    variant = str(config.get("model_type", "unknown")).removeprefix("pp_ocrv6_")
    variant = variant.removesuffix("_rec")

    num_classes = int(config["head_out_channels"])
    hidden_size = int(config.get("hidden_size", out_channels))

    if "depth" in config:
        encoder_kind: EncoderKind = "lightsvtr"
        encoder_dims = hidden_size
        encoder_depth = int(config["depth"])
        encoder_mlp_ratio = float(config.get("mlp_ratio", 4.0))
        kernel_raw = config.get("conv_kernel_size", [1, 7])
        if not isinstance(kernel_raw, Sequence) or len(kernel_raw) != 2:
            raise ValueError("conv_kernel_size must contain two integers")
        local_kernel = int(kernel_raw[1])
        use_guide = False
        mid_channels = None
    else:
        encoder_kind = "reshape"
        encoder_dims = out_channels
        encoder_depth = 0
        encoder_mlp_ratio = 4.0
        local_kernel = 7
        use_guide = True
        mid_channels = hidden_size

    return RecModelConfig(
        variant=variant,
        stem_kind=_stem_kind_from_config(stem_type),
        stem_mid=stem_mid,
        stem_out=stem_out,
        block_configs=block_configs,
        out_channels=out_channels,
        encoder_kind=encoder_kind,
        encoder_dims=encoder_dims,
        encoder_depth=encoder_depth,
        encoder_mlp_ratio=encoder_mlp_ratio,
        local_kernel=local_kernel,
        use_guide=use_guide,
        mid_channels=mid_channels,
        num_classes=num_classes,
    )
