"""Squeeze-and-excitation blocks for PP-OCRv6."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_ocr.models.common.activations import HardSigmoid


class SELayer(nn.Module):
    """Channel attention block used in LCNetV4 token mixers."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        """Initialize squeeze-and-excitation layers.

        Args:
            channels: Input and output channel count.
            reduction: Channel reduction ratio for the bottleneck.
        """
        hidden_channels = channels // reduction
        self.conv1 = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.hardsigmoid = HardSigmoid()

    def __call__(self, x: mx.array) -> mx.array:
        """Apply channel attention.

        Args:
            x: Input tensor in NHWC layout.

        Returns:
            Recalibrated tensor with the same shape as ``x``.
        """
        pooled = mx.mean(x, axis=(1, 2), keepdims=True)
        gate = self.hardsigmoid(self.conv2(self.relu(self.conv1(pooled))))
        return x * gate
