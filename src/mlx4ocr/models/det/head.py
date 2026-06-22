"""DB detection head."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx4ocr.models.common.fuse import apply_conv_batch_norm_fusion


class DBHead(nn.Module):
    """Differentiable binarization head (inference binarize path only)."""

    def __init__(self, in_channels: int, kernel_list: tuple[int, int, int] = (3, 2, 2)) -> None:
        quarter = in_channels // 4
        k1, k2, k3 = kernel_list
        self.conv_down = ConvNormAct(
            in_channels,
            quarter,
            kernel_size=k1,
            padding=k1 // 2,
            bias=False,
        )
        self.conv_up = ConvTransposeNormAct(
            quarter,
            quarter,
            kernel_size=k2,
            stride=2,
            bias=True,
        )
        self.conv_final = nn.ConvTranspose2d(
            quarter,
            1,
            kernel_size=k3,
            stride=2,
            bias=True,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_down(x)
        x = self.conv_up(x)
        x = self.conv_final(x)
        return mx.sigmoid(x)


class ConvNorm(nn.Module):
    """Conv2d + batch norm without activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.norm: nn.BatchNorm | None = nn.BatchNorm(out_channels)

    def fuse_for_inference(self) -> bool:
        """Fold batch norm into the convolution for inference."""
        if self.norm is None:
            return False
        apply_conv_batch_norm_fusion(self.conv, self.norm)
        self.norm = None
        return True

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class ConvNormAct(nn.Module):
    """Conv2d + batch norm + ReLU with Hub ``norm`` parameter names."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ) -> None:
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.norm: nn.BatchNorm | None = nn.BatchNorm(out_channels)
        self.act = nn.ReLU()

    def fuse_for_inference(self) -> bool:
        """Fold batch norm into the convolution for inference."""
        if self.norm is None:
            return False
        apply_conv_batch_norm_fusion(self.conv, self.norm)
        self.norm = None
        return True

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        return self.act(x)


class ConvTransposeNormAct(nn.Module):
    """ConvTranspose2d + batch norm + ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int = 2,
        bias: bool = True,
    ) -> None:
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            bias=bias,
        )
        self.norm: nn.BatchNorm | None = nn.BatchNorm(out_channels)
        self.act = nn.ReLU()

    def fuse_for_inference(self) -> bool:
        """Fold batch norm into the transposed convolution for inference."""
        if self.norm is None:
            return False
        apply_conv_batch_norm_fusion(self.conv, self.norm)
        self.norm = None
        return True

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        return self.act(x)
