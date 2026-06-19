"""Normalization layers backed by MLX fast kernels."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class LayerNorm(nn.Module):
    """Layer normalization using ``mx.fast.layer_norm``.

    Matches Paddle ``LayerNorm`` semantics while routing through the optimized
    MLX kernel used internally by ``nn.LayerNorm``.
    """

    def __init__(
        self,
        dims: int,
        *,
        eps: float = 1e-5,
        affine: bool = True,
        bias: bool = True,
    ) -> None:
        if affine:
            self.weight = mx.ones((dims,))
            if bias:
                self.bias = mx.zeros((dims,))
        self.eps = eps
        self.dims = dims

    def __call__(self, x: mx.array) -> mx.array:
        """Apply layer normalization over the last dimension.

        Args:
            x: Input tensor with feature dimension ``dims`` on the last axis.

        Returns:
            Normalized tensor with the same shape as ``x``.
        """
        weight = self.weight if "weight" in self else None
        bias = self.bias if "bias" in self else None
        return mx.fast.layer_norm(x, weight, bias, self.eps)
