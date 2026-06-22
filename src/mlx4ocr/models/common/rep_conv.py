"""Inference-time reparameterized depthwise convolutions."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class RepDWConv(nn.Module):
    """Fused reparameterized depthwise convolution."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        """Initialize a fused depthwise convolution.

        Args:
            channels: Input and output channel count.
            kernel_size: Spatial kernel size.
        """
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            groups=channels,
            bias=True,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Apply the fused depthwise convolution.

        Args:
            x: Input tensor in NHWC layout.

        Returns:
            Depthwise-convolved output tensor.
        """
        return self.conv(x)


class DilatedReparamBlock(nn.Module):
    """Fused dilated large-kernel depthwise convolution."""

    def __init__(self, channels: int, kernel_size: int = 9) -> None:
        """Initialize a fused large-kernel depthwise convolution.

        Args:
            channels: Input and output channel count.
            kernel_size: Effective large-kernel size after branch fusion.
        """
        if kernel_size not in {5, 7, 9, 11, 13}:
            raise ValueError(
                "DilatedReparamBlock requires kernel_size in [5, 7, 9, 11, 13], "
                f"but got {kernel_size}"
            )
        padding = kernel_size // 2
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            groups=channels,
            bias=True,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Apply the fused large-kernel depthwise convolution.

        Args:
            x: Input tensor in NHWC layout.

        Returns:
            Depthwise-convolved output tensor.
        """
        return self.conv(x)
