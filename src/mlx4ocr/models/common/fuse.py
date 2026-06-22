"""Inference-time operator fusion for PP-OCRv6 models."""

from __future__ import annotations

import logging

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

_ConvModule = nn.Conv2d | nn.ConvTranspose2d


def fuse_conv_batch_norm(
    conv: _ConvModule,
    bn: nn.BatchNorm,
) -> tuple[mx.array, mx.array]:
    """Fold batch-norm statistics into convolution weights for inference.

    Args:
        conv: A ``Conv2d`` or ``ConvTranspose2d`` module.
        bn: Batch normalization applied immediately after ``conv``.

    Returns:
        Fused ``(weight, bias)`` tensors for ``conv``.
    """
    weight = conv.weight
    conv_bias = conv.bias if "bias" in conv else mx.zeros((weight.shape[0],))
    if "weight" not in bn:
        raise ValueError("BatchNorm must be affine to fuse with convolution")
    scale = bn.weight * mx.rsqrt(bn.running_var + bn.eps)
    scale_4d = mx.reshape(scale, (-1, 1, 1, 1))
    fused_weight = weight * scale_4d
    fused_bias = bn.bias + scale * (conv_bias - bn.running_mean)
    return fused_weight, fused_bias


def apply_conv_batch_norm_fusion(conv: _ConvModule, bn: nn.BatchNorm) -> None:
    """Replace ``conv`` parameters with fused weights and drop ``bn`` usage.

    Args:
        conv: Convolution module updated in place.
        bn: Batch normalization module whose statistics are folded into ``conv``.
    """
    fused_weight, fused_bias = fuse_conv_batch_norm(conv, bn)
    conv.weight = fused_weight
    conv.bias = fused_bias


def fuse_for_inference(module: nn.Module) -> int:
    """Fuse supported conv + batch-norm blocks throughout a module tree.

    Args:
        module: Root module, typically a loaded detection or recognition model.

    Returns:
        Number of blocks fused.
    """
    fused = 0
    for child in module.modules():
        fuse = getattr(child, "fuse_for_inference", None)
        if callable(fuse) and fuse():
            fused += 1
    if fused:
        logger.info("Fused %d conv+batch-norm blocks for inference", fused)
    return fused
