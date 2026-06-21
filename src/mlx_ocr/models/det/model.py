"""Detection model assembly and weight loading."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping

import mlx.core as mx
import mlx.nn as nn

from mlx_ocr.hub.download import HubArtifacts
from mlx_ocr.hub.weights import (
    WeightMapper,
    align_tensor_to_parameter,
    flatten_module_parameters,
    load_safetensors,
    paddle_conv_weight_to_mlx,
    rewrite_hub_key,
)
from mlx_ocr.models.common.fuse import fuse_for_inference
from mlx_ocr.models.det.backbone import PPLCNetV4Det
from mlx_ocr.models.det.config import DetModelConfig, det_config_from_artifacts
from mlx_ocr.models.det.head import DBHead
from mlx_ocr.models.det.neck import RepLKFPN, RepLKPAN

logger = logging.getLogger(__name__)

_INDEXED_PREFIXES = (
    "encoder.blocks",
    "encoder.blocks_",
    "insert_conv",
    "input_conv",
    "input_channel_adjustment_convolution",
    "input_feature_projection_convolution",
    "path_aggregation_head_convolution",
    "path_aggregation_lateral_convolution",
    "intraclass_blocks",
)


def rewrite_detection_hub_key(source_key: str) -> str:
    """Rewrite a Hub detection key into an MLX module parameter path."""
    key = rewrite_hub_key(source_key, strip_prefixes=("model.",))
    token_replacements = (
        ("token_squeeze_excitation.convolutions.0", "token_squeeze_excitation.conv1"),
        ("token_squeeze_excitation.convolutions.2", "token_squeeze_excitation.conv2"),
    )
    for old, new in token_replacements:
        key = key.replace(old, new)

    key = re.sub(r"\.blocks\.(\d+)", r".blocks_\1", key)
    for prefix in _INDEXED_PREFIXES:
        key = re.sub(rf"{re.escape(prefix)}\.(\d+)", rf"{prefix}_\1", key)
    return key


def paddle_conv_transpose_weight_to_mlx(weight: mx.array) -> mx.array:
    """Convert Paddle IOHW transposed-conv weights to MLX OHWI layout."""
    if weight.ndim != 4:
        raise ValueError(f"expected rank-4 transposed-conv weight, got {weight.shape}")
    return mx.transpose(weight, (1, 2, 3, 0))


def _is_conv_transpose_parameter(target_key: str) -> bool:
    """Return whether a flattened parameter belongs to a transposed convolution."""
    return "conv_up.conv.weight" in target_key or target_key.endswith("conv_final.weight")


def align_detection_tensor(
    value: mx.array,
    expected_shape: tuple[int, ...],
    *,
    target_key: str = "",
) -> mx.array:
    """Align Hub tensors with MLX detection module parameters."""
    if len(value.shape) == 4 and len(expected_shape) == 4:
        dim0, dim1, kernel_h, kernel_w = value.shape
        if _is_conv_transpose_parameter(target_key):
            if expected_shape == (dim1, kernel_h, kernel_w, dim0):
                return paddle_conv_transpose_weight_to_mlx(value)
        elif expected_shape == (dim0, kernel_h, kernel_w, dim1):
            return paddle_conv_weight_to_mlx(value)

        if expected_shape == (dim1, kernel_h, kernel_w, dim0):
            return paddle_conv_transpose_weight_to_mlx(value)
        if expected_shape == (dim0, kernel_h, kernel_w, dim1):
            return paddle_conv_weight_to_mlx(value)

    if value.shape == expected_shape:
        return value

    return align_tensor_to_parameter(value, expected_shape)


def build_detection_mapper(
    source_keys: Mapping[str, mx.array],
    module: nn.Module,
) -> WeightMapper:
    """Build a mapper from Hub keys to flattened MLX parameter paths."""
    targets = tuple(flatten_module_parameters(module))
    mapping: dict[str, str] = {}
    for source_key in source_keys:
        rewritten = rewrite_detection_hub_key(source_key)
        if rewritten in targets:
            mapping[source_key] = rewritten
            continue

        suffix_matches = [target for target in targets if rewritten.endswith(target)]
        if len(suffix_matches) == 1:
            mapping[source_key] = suffix_matches[0]
            continue
        if len(suffix_matches) > 1:
            mapping[source_key] = max(suffix_matches, key=len)
            continue

        prefix_matches = [target for target in targets if target.endswith(rewritten)]
        if len(prefix_matches) == 1:
            mapping[source_key] = prefix_matches[0]
            continue
        if len(prefix_matches) > 1:
            mapping[source_key] = max(prefix_matches, key=len)
            continue

        raise ValueError(
            f"expected exactly one target for {source_key!r} -> {rewritten!r}, found 0"
        )
    return WeightMapper.from_pairs(mapping)


def load_detection_weights(module: nn.Module, tensors: Mapping[str, mx.array]) -> None:
    """Load Hub safetensors into a detection module."""
    mapper = build_detection_mapper(tensors, module)
    expected = flatten_module_parameters(module)
    mapped = mapper.map_tensors(tensors)

    aligned: dict[str, mx.array] = {}
    shape_errors: list[str] = []
    for key, value in mapped.items():
        if key not in expected:
            continue
        try:
            aligned[key] = align_detection_tensor(value, expected[key].shape, target_key=key)
        except ValueError:
            shape_errors.append(f"{key}: expected {expected[key].shape}, got {value.shape}")

    if shape_errors:
        detail = "\n".join(shape_errors)
        raise ValueError(f"shape mismatch while loading detection weights:\n{detail}")

    missing = tuple(sorted(set(expected) - set(aligned)))
    consumed = {source for source, target in mapper.mapping.items() if target in aligned}
    unexpected = tuple(sorted(set(tensors) - consumed))
    if missing or unexpected:
        raise ValueError(
            "strict detection weight load failed: "
            f"missing={missing or ()}, unexpected={unexpected or ()}"
        )

    module.load_weights(list(aligned.items()), strict=True)
    logger.info("Loaded %d tensors into %s", len(aligned), module.__class__.__name__)


class DetectionModel(nn.Module):
    """PP-OCRv6 DB detection network."""

    def __init__(self, config: DetModelConfig) -> None:
        self.backbone = PPLCNetV4Det(
            config.stem_mid,
            config.stem_out,
            config.block_configs,
        )
        if config.neck_kind == "replkpan":
            self.neck = RepLKPAN(
                config.stage_channels,
                config.neck_out_channels,
                intracl=config.intracl,
            )
        else:
            self.neck = RepLKFPN(
                config.stage_channels,
                config.neck_out_channels,
                dilated_kernel_size=config.dilated_kernel_size,
            )
        self.head = DBHead(config.neck_out_channels, kernel_list=config.head_kernel_list)

    def __call__(self, x: mx.array) -> mx.array:
        """Run detection forward pass.

        Args:
            x: Input tensor in NHWC layout ``[B, H, W, 3]``.

        Returns:
            Probability map ``[B, H, W, 1]``.
        """
        features = self.backbone(x)
        fused = self.neck(features)
        return self.head(fused)

    @classmethod
    def from_artifacts(cls, artifacts: HubArtifacts) -> DetectionModel:
        """Construct a detection model from Hub artifacts."""
        config = det_config_from_artifacts(artifacts)
        model = cls(config)
        tensors = load_safetensors(artifacts.weights)
        load_detection_weights(model, tensors)
        model.eval()
        fuse_for_inference(model)
        return model
