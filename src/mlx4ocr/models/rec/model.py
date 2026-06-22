"""Recognition model assembly and weight loading."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping

import mlx.core as mx
import mlx.nn as nn

from mlx4ocr.hub.download import HubArtifacts
from mlx4ocr.hub.rec_weight_patch import (
    RecognitionWeightSource,
    load_patched_recognition_tensors,
    resolve_recognition_weight_source,
)
from mlx4ocr.hub.weights import (
    WeightMapper,
    align_tensor_to_parameter,
    flatten_module_parameters,
    load_safetensors,
    paddle_conv_weight_to_mlx,
    rewrite_hub_key,
)
from mlx4ocr.models.common.fuse import fuse_for_inference
from mlx4ocr.models.rec.backbone import PPLCNetV4Rec
from mlx4ocr.models.rec.config import RecModelConfig, rec_config_from_artifacts
from mlx4ocr.models.rec.encoder import EncoderWithLightSVTR, im2seq
from mlx4ocr.models.rec.head import CTCHead

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


def split_recognition_attention_tensors(
    tensors: Mapping[str, mx.array],
) -> tuple[dict[str, mx.array], dict[str, mx.array]]:
    """Split fused QKV and output-projection Hub tensors for ``MultiHeadAttention``.

    Hub checkpoints store a single fused ``qkv`` linear and a ``projection`` output
    layer. ``nn.MultiHeadAttention`` expects separate ``query_proj``, ``key_proj``,
    ``value_proj``, and ``out_proj`` parameters.

    Args:
        tensors: Raw safetensors from a recognition Hub repo.

    Returns:
        A pair of ``(direct_targets, remaining_sources)`` where ``direct_targets``
        maps MLX parameter paths to aligned tensors and ``remaining_sources`` keeps
        all other Hub keys for the generic mapper.
    """
    direct: dict[str, mx.array] = {}
    remaining: dict[str, mx.array] = {}
    for source_key, value in tensors.items():
        rewritten = rewrite_recognition_hub_key(source_key)
        if rewritten.endswith((".self_attn.qkv.weight", ".self_attn.qkv.bias")):
            prefix = rewritten[: rewritten.rindex(".qkv.")]
            part = "weight" if rewritten.endswith(".weight") else "bias"
            chunk = value.shape[0] // 3
            for proj, index in (("query_proj", 0), ("key_proj", 1), ("value_proj", 2)):
                direct[f"{prefix}.{proj}.{part}"] = value[index * chunk : (index + 1) * chunk]
            continue
        if rewritten.endswith((".self_attn.projection.weight", ".self_attn.projection.bias")):
            part = "weight" if rewritten.endswith(".weight") else "bias"
            prefix = rewritten[: rewritten.rindex(".projection.")]
            direct[f"{prefix}.out_proj.{part}"] = value
            continue
        remaining[source_key] = value
    return direct, remaining


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
    direct, remaining = split_recognition_attention_tensors(tensors)
    mapper = build_recognition_mapper(remaining, module)
    expected = flatten_module_parameters(module)
    mapped = mapper.map_tensors(remaining)
    mapped.update(direct)

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
    unexpected = tuple(sorted(set(remaining) - consumed))
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
    def from_artifacts(
        cls,
        artifacts: HubArtifacts,
        *,
        weight_source: RecognitionWeightSource = "auto",
    ) -> RecognitionModel:
        """Construct a recognition model from Hub artifacts.

        Args:
            artifacts: Downloaded recognition Hub files.
            weight_source: ``hub`` loads raw safetensors; ``paddle_pretrained``
                patches known-bad head encoder weights for small/medium from
                official Paddle pretrained checkpoints; ``auto`` picks the
                default per variant.

        Returns:
            Loaded recognition model in eval mode.
        """
        config = rec_config_from_artifacts(artifacts)
        model = cls(config)
        tensors = load_safetensors(artifacts.weights)
        resolved = resolve_recognition_weight_source(artifacts.variant, weight_source)
        if resolved == "paddle_pretrained":
            tensors = load_patched_recognition_tensors(artifacts.variant, tensors)
        load_recognition_weights(model, tensors)
        model.eval()
        fuse_for_inference(model)
        return model
