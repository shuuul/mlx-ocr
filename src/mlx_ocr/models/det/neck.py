"""Detection necks: RepLKFPN and RepLKPAN."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_ocr.models.common.activations import HardSigmoidClip
from mlx_ocr.models.det.intracl import IntraCLBlock


def upsample_nearest(x: mx.array, scale: int) -> mx.array:
    """Nearest-neighbor upsampling by an integer scale factor."""
    if scale == 1:
        return x
    return mx.repeat(mx.repeat(x, scale, axis=1), scale, axis=2)


class SEModule(nn.Module):
    """Squeeze-excitation module used in detection necks."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        hidden = channels // reduction
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.conv2 = nn.Conv2d(hidden, channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.hardsigmoid = HardSigmoidClip()

    def __call__(self, x: mx.array) -> mx.array:
        pooled = mx.mean(x, axis=(1, 2), keepdims=True)
        gate = self.hardsigmoid(self.conv2(self.relu(self.conv1(pooled))))
        return x * gate


class RSELayer(nn.Module):
    """1x1 conv followed by squeeze-excitation with optional shortcut."""

    def __init__(self, in_channels: int, out_channels: int, *, shortcut: bool = True) -> None:
        self.in_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.squeeze_excitation_block = SEModule(out_channels)
        self.shortcut = shortcut

    def __call__(self, x: mx.array) -> mx.array:
        convolved = self.in_conv(x)
        attended = self.squeeze_excitation_block(convolved)
        if self.shortcut:
            return convolved + attended
        return attended


class RepLKInputConv(nn.Module):
    """Dilated large-kernel depthwise + pointwise + SE used by RepLKFPN."""

    def __init__(self, channels: int, out_quarter: int, kernel_size: int) -> None:
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            groups=channels,
            bias=True,
        )
        self.pw = nn.Conv2d(
            channels,
            out_quarter,
            kernel_size=1,
            bias=False,
        )
        self.squeeze_excitation_module = SEModule(out_quarter)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)
        x = self.pw(x)
        return x + self.squeeze_excitation_module(x)


class RepLKFPN(nn.Module):
    """RepLK feature pyramid neck for tiny and small detection models."""

    def __init__(
        self,
        in_channels: tuple[int, int, int, int],
        out_channels: int,
        *,
        dilated_kernel_size: int = 5,
        shortcut: bool = True,
    ) -> None:
        quarter = out_channels // 4
        for index, channels in enumerate(in_channels):
            setattr(
                self,
                f"insert_conv_{index}",
                RSELayer(channels, out_channels, shortcut=shortcut),
            )
            setattr(
                self,
                f"input_conv_{index}",
                RepLKInputConv(out_channels, quarter, dilated_kernel_size),
            )

    def _insert(self, index: int, x: mx.array) -> mx.array:
        return getattr(self, f"insert_conv_{index}")(x)

    def _input(self, index: int, x: mx.array) -> mx.array:
        return getattr(self, f"input_conv_{index}")(x)

    def __call__(self, features: list[mx.array]) -> mx.array:
        c2, c3, c4, c5 = features
        in2 = self._insert(0, c2)
        in3 = self._insert(1, c3)
        in4 = self._insert(2, c4)
        in5 = self._insert(3, c5)

        out4 = in4 + upsample_nearest(in5, 2)
        out3 = in3 + upsample_nearest(out4, 2)
        out2 = in2 + upsample_nearest(out3, 2)

        p5 = self._input(3, in5)
        p4 = self._input(2, out4)
        p3 = self._input(1, out3)
        p2 = self._input(0, out2)

        p5 = upsample_nearest(p5, 8)
        p4 = upsample_nearest(p4, 4)
        p3 = upsample_nearest(p3, 2)
        return mx.concatenate([p5, p4, p3, p2], axis=-1)


class RepLKPAN(nn.Module):
    """RepLK path aggregation neck for medium detection models."""

    def __init__(
        self,
        in_channels: tuple[int, int, int, int],
        out_channels: int,
        *,
        intracl: bool = True,
    ) -> None:
        quarter = out_channels // 4
        for index, channels in enumerate(in_channels):
            setattr(
                self,
                f"input_channel_adjustment_convolution_{index}",
                nn.Conv2d(channels, out_channels, kernel_size=1, bias=False),
            )
            setattr(
                self,
                f"input_feature_projection_convolution_{index}",
                nn.Conv2d(
                    out_channels,
                    quarter,
                    kernel_size=9,
                    padding=4,
                    bias=True,
                ),
            )
            setattr(
                self,
                f"path_aggregation_lateral_convolution_{index}",
                nn.Conv2d(quarter, quarter, kernel_size=9, padding=4, bias=True),
            )
            if intracl:
                setattr(self, f"intraclass_blocks_{index}", IntraCLBlock(quarter, reduce_factor=2))

        for index in range(3):
            setattr(
                self,
                f"path_aggregation_head_convolution_{index}",
                nn.Conv2d(quarter, quarter, kernel_size=3, stride=2, padding=1, bias=False),
            )

    def _adjust(self, index: int, x: mx.array) -> mx.array:
        return getattr(self, f"input_channel_adjustment_convolution_{index}")(x)

    def _project(self, index: int, x: mx.array) -> mx.array:
        return getattr(self, f"input_feature_projection_convolution_{index}")(x)

    def _lateral(self, index: int, x: mx.array) -> mx.array:
        return getattr(self, f"path_aggregation_lateral_convolution_{index}")(x)

    def _intracl(self, index: int, x: mx.array) -> mx.array:
        block = getattr(self, f"intraclass_blocks_{index}", None)
        if block is None:
            return x
        return block(x)

    def _head(self, index: int, x: mx.array) -> mx.array:
        return getattr(self, f"path_aggregation_head_convolution_{index}")(x)

    def __call__(self, features: list[mx.array]) -> mx.array:
        c2, c3, c4, c5 = features
        in2 = self._adjust(0, c2)
        in3 = self._adjust(1, c3)
        in4 = self._adjust(2, c4)
        in5 = self._adjust(3, c5)

        out4 = in4 + upsample_nearest(in5, 2)
        out3 = in3 + upsample_nearest(out4, 2)
        out2 = in2 + upsample_nearest(out3, 2)

        f2 = self._project(0, out2)
        f3 = self._project(1, out3)
        f4 = self._project(2, out4)
        f5 = self._project(3, in5)

        pan3 = f3 + self._head(0, f2)
        pan4 = f4 + self._head(1, pan3)
        pan5 = f5 + self._head(2, pan4)

        p2 = self._lateral(0, f2)
        p3 = self._lateral(1, pan3)
        p4 = self._lateral(2, pan4)
        p5 = self._lateral(3, pan5)

        p2 = self._intracl(0, p2)
        p3 = self._intracl(1, p3)
        p4 = self._intracl(2, p4)
        p5 = self._intracl(3, p5)

        p5 = upsample_nearest(p5, 8)
        p4 = upsample_nearest(p4, 4)
        p3 = upsample_nearest(p3, 2)
        return mx.concatenate([p5, p4, p3, p2], axis=-1)
