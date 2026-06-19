"""Recognition model assembly and weight loading."""

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
from mlx_ocr.models.rec.backbone import PPLCNetV4Rec
from mlx_ocr.models.rec.config import RecModelConfig, rec_config_from_artifacts
from mlx_ocr.models.rec.encoder import EncoderWithLightSVTR, im2seq
from mlx_ocr.models.rec.head import CTCHead

logger = logging.getLogger(__name__)

_INDEXED_PREFIXES = (
    "backbone.encoder.blocks",
    "backbone.encoder.blocks_",
    "encoder.svtr_block",
    "encoder.svtr_block_",
)


def rewrite_recognition_hub_key(source_key: str) -> str:
    """Rewrite a Hub recognition key into an MLX module parameter path."""
    key = source_key
    if key.startswith("model.backbone."):
        key = "backbone." + key[len("model.backbone.") :]
    elif key.startswith("model."):
        key = key[len("model.") :]
    if key.startswith("head.encoder."):
        key = "encoder." + key[len("head.encoder.") :]

    key = rewrite_hub_key(key, strip_prefixes=())
    token_replacements = (
        ("token_squeeze_excitation.convolutions.0", "token_squeeze_excitation.conv1"),
        ("token_squeeze_excitation.convolutions.2", "token_squeeze_excitation.conv2"),
        ("encoder.conv_block.0", "encoder.conv_reduce"),
        ("encoder.conv_block.1", "encoder.skip_conv"),
        ("encoder.conv_block.2", "encoder.local_conv"),
        ("layer_norm1", "norm1"),
        ("layer_norm2", "norm2"),
    )
    for old, new in token_replacements:
        key = key.replace(old, new)

    key = re.sub(r"\.blocks\.(\d+)", r".blocks_\1", key)
    key = re.sub(r"encoder\.svtr_block\.(\d+)", r"encoder.svtr_block_\1", key)
    for prefix in _INDEXED_PREFIXES:
        key = re.sub(rf"{re.escape(prefix)}\.(\d+)", rf"{prefix}_\1", key)
    return key


def paddle_conv1d_weight_to_mlx(weight: mx.array, expected_shape: tuple[int, ...]) -> mx.array:
    """Convert Paddle Conv1D weights to MLX depthwise or pointwise Conv2d layout."""
    if weight.ndim != 3:
        raise ValueError(f"expected rank-3 Conv1D weight, got shape {weight.shape}")
    out_channels, in_channels, kernel = weight.shape
    pointwise = mx.reshape(
        mx.transpose(weight, (0, 2, 1)),
        (out_channels, 1, kernel, in_channels),
    )
    if pointwise.shape == expected_shape:
        return pointwise
    depthwise = mx.reshape(weight, (out_channels, 1, kernel, in_channels))
    if depthwise.shape == expected_shape:
        return depthwise
    raise ValueError(
        f"cannot align Conv1D weight {weight.shape} to expected shape {expected_shape}"
    )


def align_recognition_tensor(
    value: mx.array,
    expected_shape: tuple[int, ...],
    *,
    target_key: str = "",
) -> mx.array:
    """Align Hub tensors with MLX recognition module parameters."""
    if len(value.shape) == 4 and len(expected_shape) == 4:
        out_channels, in_channels, kernel_h, kernel_w = value.shape
        if expected_shape == (out_channels, kernel_h, kernel_w, in_channels):
            return paddle_conv_weight_to_mlx(value)

    if value.shape == expected_shape:
        return value

    if len(value.shape) == 3 and len(expected_shape) == 4 and ".conv" in target_key:
        return paddle_conv1d_weight_to_mlx(value, expected_shape)

    return align_tensor_to_parameter(value, expected_shape)


def build_recognition_mapper(
    source_keys: Mapping[str, mx.array],
    module: nn.Module,
) -> WeightMapper:
    """Build a mapper from Hub keys to flattened MLX parameter paths."""
    targets = tuple(flatten_module_parameters(module))
    mapping: dict[str, str] = {}
    for source_key in source_keys:
        rewritten = rewrite_recognition_hub_key(source_key)
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


def load_recognition_weights(module: nn.Module, tensors: Mapping[str, mx.array]) -> None:
    """Load Hub safetensors into a recognition module."""
    mapper = build_recognition_mapper(tensors, module)
    expected = flatten_module_parameters(module)
    mapped = mapper.map_tensors(tensors)

    aligned: dict[str, mx.array] = {}
    shape_errors: list[str] = []
    for key, value in mapped.items():
        if key not in expected:
            continue
        try:
            aligned[key] = align_recognition_tensor(
                value,
                expected[key].shape,
                target_key=key,
            )
        except ValueError:
            shape_errors.append(f"{key}: expected {expected[key].shape}, got {value.shape}")

    if shape_errors:
        detail = "\n".join(shape_errors)
        raise ValueError(f"shape mismatch while loading recognition weights:\n{detail}")

    missing = tuple(sorted(set(expected) - set(aligned)))
    consumed = {source for source, target in mapper.mapping.items() if target in aligned}
    unexpected = tuple(sorted(set(tensors) - consumed))
    if missing or unexpected:
        raise ValueError(
            "strict recognition weight load failed: "
            f"missing={missing or ()}, unexpected={unexpected or ()}"
        )

    module.load_weights(list(aligned.items()), strict=True)
    logger.info("Loaded %d tensors into %s", len(aligned), module.__class__.__name__)


class RecognitionModel(nn.Module):
    """PP-OCRv6 CTC recognition network."""

    def __init__(self, config: RecModelConfig) -> None:
        self.backbone = PPLCNetV4Rec(
            config.stem_kind,
            config.stem_mid,
            config.stem_out,
            config.block_configs,
        )
        self.encoder: EncoderWithLightSVTR | None
        if config.encoder_kind == "lightsvtr":
            self.encoder = EncoderWithLightSVTR(
                config.out_channels,
                config.encoder_dims,
                depth=config.encoder_depth,
                mlp_ratio=config.encoder_mlp_ratio,
                local_kernel=config.local_kernel,
            )
        else:
            self.encoder = None
        self.head = CTCHead(
            config.encoder_dims,
            config.num_classes,
            use_guide=config.use_guide,
            mid_channels=config.mid_channels,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Run recognition forward pass.

        Args:
            x: Input tensor in NHWC layout ``[B, H, W, 3]``.

        Returns:
            Softmax probabilities ``[B, T, num_classes]``.
        """
        features = self.backbone(x)
        if self.encoder is not None:
            features = self.encoder(features)
        sequence = im2seq(features)
        return self.head(sequence)

    @classmethod
    def from_artifacts(cls, artifacts: HubArtifacts) -> RecognitionModel:
        """Construct a recognition model from Hub artifacts."""
        config = rec_config_from_artifacts(artifacts)
        model = cls(config)
        tensors = load_safetensors(artifacts.weights)
        load_recognition_weights(model, tensors)
        model.eval()
        return model


def load_recognition_model(artifacts: HubArtifacts) -> RecognitionModel:
    """Load a recognition model with weights from Hub artifacts."""
    return RecognitionModel.from_artifacts(artifacts)
