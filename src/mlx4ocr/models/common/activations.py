"""Activation functions used by PP-OCRv6 LCNet blocks."""

from __future__ import annotations

from typing import Literal

import mlx.core as mx
import mlx.nn as nn

ActivationName = Literal["relu", "gelu", "hswish", "none"]


class HardSigmoid(nn.Module):
    """Paddle ``nn.Hardsigmoid`` equivalent: ``relu6(x + 3) / 6``."""

    def __call__(self, x: mx.array) -> mx.array:
        """Apply hard sigmoid element-wise.

        Args:
            x: Input tensor.

        Returns:
            Values in ``[0, 1]``.
        """
        return mx.clip(mx.maximum(x + 3.0, 0.0), 0.0, 6.0) / 6.0


class HardSigmoidClip(nn.Module):
    """Paddle ``F.hardsigmoid`` with ``slope=0.2`` and ``offset=0.5``."""

    def __call__(self, x: mx.array) -> mx.array:
        """Apply clipped hard sigmoid element-wise.

        Args:
            x: Input tensor.

        Returns:
            Values clipped to ``[0, 1]``.
        """
        return mx.clip(x * 0.2 + 0.5, 0.0, 1.0)


def build_activation(name: ActivationName) -> nn.Module | None:
    """Construct an activation module from a PP-OCRv6 config name.

    Args:
        name: Activation identifier from Paddle configs.

    Returns:
        An MLX activation module, or ``None`` when ``name`` is ``none``.
    """
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "hswish":
        return nn.Hardswish()
    if name == "none":
        return None
    raise ValueError(f"unsupported activation: {name}")
