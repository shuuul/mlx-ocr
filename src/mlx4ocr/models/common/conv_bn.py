"""Convolution + batch-norm building blocks for PP-OCRv6."""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
import mlx.nn as nn

from mlx4ocr.models.common.activations import ActivationName, build_activation
from mlx4ocr.models.common.fuse import apply_conv_batch_norm_fusion

KernelSize = int | Sequence[int]
Padding = int | Sequence[int] | str


def _as_pair(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if len(value) != 2:
        raise ValueError(f"expected a scalar or length-2 sequence, got {value!r}")
    return int(value[0]), int(value[1])


def same_padding_pair(input_size: int, kernel_size: int, stride: int = 1) -> tuple[int, int]:
    """Return Paddle-compatible ``SAME`` padding for one spatial axis."""
    output_size = (input_size + stride - 1) // stride
    pad_total = max(0, (output_size - 1) * stride + kernel_size - input_size)
    pad_before = pad_total // 2
    return pad_before, pad_total - pad_before


def apply_same_padding_nhwc(
    x: mx.array,
    kernel_size: KernelSize,
    *,
    stride: int | tuple[int, int] = 1,
) -> mx.array:
    """Pad an NHWC tensor with Paddle ``SAME`` semantics."""
    kernel_h, kernel_w = _as_pair(kernel_size)
    stride_h, stride_w = _as_pair(stride)
    height, width = x.shape[1], x.shape[2]
    pad_h = same_padding_pair(height, kernel_h, stride_h)
    pad_w = same_padding_pair(width, kernel_w, stride_w)
    if pad_h == (0, 0) and pad_w == (0, 0):
        return x
    return mx.pad(x, ((0, 0), pad_h, pad_w, (0, 0)))


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
        bn_weight_init: float = 1.0,
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
        conv_padding: int | tuple[int, int]
        if isinstance(padding, str):
            conv_padding = 0
        elif isinstance(padding, int):
            conv_padding = padding
        else:
            conv_padding = _as_pair(padding)

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=conv_padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm(out_channels)
        if bn_weight_init != 1.0:
            self.bn.weight = (
                mx.zeros_like(self.bn.weight) if bn_weight_init == 0.0 else self.bn.weight
            )

    def fuse_for_inference(self) -> bool:
        """Fold batch norm into the convolution for inference."""
        if self.bn is None:
            return False
        apply_conv_batch_norm_fusion(self.conv, self.bn)
        self.bn = None
        return True

    def __call__(self, x: mx.array) -> mx.array:
        """Apply convolution and batch normalization.

        Args:
            x: Input tensor in NHWC layout.

        Returns:
            Normalized output tensor.
        """
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


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
        self.dynamic_same_padding = isinstance(padding, str) and padding.upper() == "SAME"
        self.kernel_size = _as_pair(kernel_size)
        self.stride = _as_pair(stride)
        resolved_padding = 0 if self.dynamic_same_padding else padding
        if not isinstance(resolved_padding, int):
            resolved_padding = _as_pair(resolved_padding)

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

    def fuse_for_inference(self) -> bool:
        """Fold batch norm into the convolution for inference."""
        if self.bn is None:
            return False
        apply_conv_batch_norm_fusion(self.conv, self.bn)
        self.bn = None
        return True

    def __call__(self, x: mx.array) -> mx.array:
        """Apply the conv block.

        Args:
            x: Input tensor in NHWC layout.

        Returns:
            Block output tensor.
        """
        if self.dynamic_same_padding:
            x = apply_same_padding_nhwc(
                x,
                self.kernel_size,
                stride=self.stride,
            )
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x
