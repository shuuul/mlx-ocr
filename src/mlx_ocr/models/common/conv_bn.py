"""Convolution + batch-norm building blocks for PP-OCRv6."""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
import mlx.nn as nn

from mlx_ocr.models.common.activations import ActivationName, build_activation

KernelSize = int | Sequence[int]
Padding = int | Sequence[int] | str


def _as_pair(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if len(value) != 2:
        raise ValueError(f"expected a scalar or length-2 sequence, got {value!r}")
    return int(value[0]), int(value[1])


def _resolve_padding(
    kernel_size: KernelSize,
    padding: Padding,
) -> int | tuple[int, int]:
    if isinstance(padding, str):
        if padding.upper() != "SAME":
            raise ValueError(f"unsupported padding string: {padding}")
        kernel_h, kernel_w = _as_pair(kernel_size)
        return kernel_h // 2, kernel_w // 2
    if isinstance(padding, int):
        return padding
    return _as_pair(padding)


class FusedConv2d(nn.Module):
    """Inference-time fused ``Conv2d`` with bias."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: KernelSize = 3,
        *,
        stride: int | tuple[int, int] = 1,
        padding: Padding = 0,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        """Initialize a fused convolution layer.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            kernel_size: Spatial kernel size.
            stride: Convolution stride.
            padding: Explicit padding or ``SAME`` for symmetric padding.
            groups: Convolution groups.
            bias: Whether the layer includes a bias vector.
        """
        resolved_padding = _resolve_padding(kernel_size, padding)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=resolved_padding,
            groups=groups,
            bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Run the fused convolution.

        Args:
            x: Input tensor in NHWC layout.

        Returns:
            Convolved output tensor.
        """
        return self.conv(x)


class Conv2DBN(nn.Module):
    """Convolution followed by batch normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: KernelSize = 1,
        *,
        stride: int | tuple[int, int] = 1,
        padding: Padding = 0,
        groups: int = 1,
    ) -> None:
        """Initialize an unfused conv + batch-norm block.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            kernel_size: Spatial kernel size.
            stride: Convolution stride.
            padding: Explicit padding or ``SAME`` for symmetric padding.
            groups: Convolution groups.
        """
        resolved_padding = _resolve_padding(kernel_size, padding)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=resolved_padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm(out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply convolution and batch normalization.

        Args:
            x: Input tensor in NHWC layout.

        Returns:
            Normalized output tensor.
        """
        return self.bn(self.conv(x))


class ConvBNAct(nn.Module):
    """Convolution, optional batch norm, and optional activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: KernelSize = 3,
        *,
        stride: int | tuple[int, int] = 1,
        padding: Padding = 1,
        groups: int = 1,
        use_act: bool = True,
        act_type: ActivationName = "relu",
        fused: bool = False,
    ) -> None:
        """Initialize a conv block used by LCNet stems and mixers.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            kernel_size: Spatial kernel size.
            stride: Convolution stride.
            padding: Explicit padding or ``SAME`` for symmetric padding.
            groups: Convolution groups.
            use_act: Whether to apply an activation function.
            act_type: Activation name from PP-OCR configs.
            fused: When ``True``, use a single fused conv with bias and skip BN.
        """
        self.use_act = use_act
        self.fused = fused
        resolved_padding = _resolve_padding(kernel_size, padding)

        if fused:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=resolved_padding,
                groups=groups,
                bias=True,
            )
            self.bn = None
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=resolved_padding,
                groups=groups,
                bias=False,
            )
            self.bn = nn.BatchNorm(out_channels)

        self.act = build_activation(act_type) if use_act else None

    def __call__(self, x: mx.array) -> mx.array:
        """Apply the conv block.

        Args:
            x: Input tensor in NHWC layout.

        Returns:
            Block output tensor.
        """
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x
