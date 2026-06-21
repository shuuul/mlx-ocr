"""Patch Hugging Face recognition safetensors with official Paddle pretrained values."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import mlx.core as mx
import numpy as np

from mlx_ocr.hub.paddle_pretrained import load_pretrained_rec
from mlx_ocr.hub.registry import ModelVariant
from mlx_ocr.hub.weights import paddle_conv_weight_to_mlx

logger = logging.getLogger(__name__)

RecognitionWeightSource = Literal["hub", "paddle_pretrained", "auto"]

_VARIANTS_NEEDING_PATCH: frozenset[ModelVariant] = frozenset({"small", "medium"})

_HEAD_CONV_PAIRS: tuple[tuple[str, str], ...] = (
    ("head.ctc_encoder.encoder.conv_reduce", "head.encoder.conv_block.0"),
    ("head.ctc_encoder.encoder.skip_conv", "head.encoder.conv_block.1"),
)

_BN_SUFFIX_MAP: tuple[tuple[str, str], ...] = (
    ("weight", "weight"),
    ("bias", "bias"),
    ("_mean", "running_mean"),
    ("_variance", "running_var"),
)


def resolve_recognition_weight_source(
    variant: ModelVariant,
    weight_source: RecognitionWeightSource,
) -> Literal["hub", "paddle_pretrained"]:
    """Resolve the effective recognition weight source for a variant.

    Args:
        variant: Model size tier parsed from Hub ``config.json``.
        weight_source: Requested source, or ``auto`` to pick a default.

    Returns:
        Either ``hub`` (raw safetensors) or ``paddle_pretrained`` (patched).
    """
    if weight_source == "auto":
        if variant in _VARIANTS_NEEDING_PATCH:
            return "paddle_pretrained"
        return "hub"
    return weight_source


def patch_recognition_hub_tensors(
    variant: ModelVariant,
    tensors: Mapping[str, mx.array],
    pretrained: Mapping[str, np.ndarray],
) -> dict[str, mx.array]:
    """Replace corrupted LightSVTR head conv weights in Hub safetensors.

    Hugging Face ``small`` and ``medium`` recognition checkpoints swap
    ``conv_reduce`` and ``skip_conv`` tensors in ``head.encoder.conv_block``
    and ship incorrect conv weights. Backbone weights match Paddle inference;
    only the two 1×1 head conv blocks need correction from official pretrained
    checkpoints.

    Args:
        variant: Model size tier.
        tensors: Safetensors loaded from the Hub repo.
        pretrained: Official Paddle pretrained arrays keyed by checkpoint names.

    Returns:
        A new tensor mapping with patched head encoder weights.

    Raises:
        KeyError: If a required pretrained or Hub tensor is missing.
        ValueError: If ``variant`` does not require patching.
    """
    if variant not in _VARIANTS_NEEDING_PATCH:
        raise ValueError(f"recognition weight patch does not apply to variant {variant!r}")

    patched = dict(tensors)
    for paddle_prefix, hub_prefix in _HEAD_CONV_PAIRS:
        conv_key = f"{paddle_prefix}.conv.weight"
        conv_weight = pretrained[conv_key]
        patched[f"{hub_prefix}.convolution.weight"] = paddle_conv_weight_to_mlx(
            mx.array(conv_weight)
        )
        for paddle_suffix, hub_suffix in _BN_SUFFIX_MAP:
            paddle_key = f"{paddle_prefix}.norm.{paddle_suffix}"
            patched[f"{hub_prefix}.normalization.{hub_suffix}"] = mx.array(pretrained[paddle_key])

    logger.info(
        "Patched %d head encoder tensors for %s recognition weights from Paddle pretrained",
        len(_HEAD_CONV_PAIRS) * (1 + len(_BN_SUFFIX_MAP)),
        variant,
    )
    return patched


def load_patched_recognition_tensors(
    variant: ModelVariant,
    tensors: Mapping[str, mx.array],
    *,
    cache_dir: Path | None = None,
) -> dict[str, mx.array]:
    """Apply the Paddle pretrained head patch to Hub recognition tensors.

    Args:
        variant: Model size tier.
        tensors: Safetensors loaded from the Hub repo.
        cache_dir: Optional pretrained checkpoint cache directory.

    Returns:
        Patched tensor mapping ready for MLX weight loading.
    """
    pretrained = load_pretrained_rec(variant, cache_dir=cache_dir)
    return patch_recognition_hub_tensors(variant, tensors, pretrained)
