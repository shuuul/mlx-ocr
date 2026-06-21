"""PPLCNetV4 recognition backbone."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_ocr.models.common import Conv2DBN, SELayer, build_activation
from mlx_ocr.models.det.backbone import StemBlock, TokenSqueezeExcitation
from mlx_ocr.models.rec.config import BlockSpec, StemKind


class LCNetV4RecBlock(nn.Module):
    """LCNetV4 inverted residual block for recognition."""

    def __init__(self, spec: BlockSpec) -> None:
        kernel, in_channels, out_channels, stride, use_se = spec
        self.has_residual = in_channels == out_channels and stride == 1
        self.use_rep_dw = in_channels == out_channels and stride == 1

        if self.use_rep_dw:
            padding = (kernel - 1) // 2
            self.token_conv = nn.Conv2d(
                in_channels,
                in_channels,
                kernel,
                padding=padding,
                groups=in_channels,
                bias=True,
            )
        else:
            padding = (kernel - 1) // 2
            self.token_conv = Conv2DBN(
                in_channels,
                in_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                groups=in_channels,
            )

        self.token_squeeze_excitation: TokenSqueezeExcitation | SELayer | None
        if use_se and self.use_rep_dw:
            self.token_squeeze_excitation = TokenSqueezeExcitation(in_channels)
        elif use_se:
            self.token_squeeze_excitation = SELayer(in_channels)
        else:
            self.token_squeeze_excitation = None

        hidden = in_channels * 2
        compress_init = 0.0 if self.has_residual else 1.0
        self.channel_conv1 = Conv2DBN(in_channels, hidden, kernel_size=1)
        self.channel_act = build_activation("gelu")
        self.channel_conv2 = Conv2DBN(
            hidden,
            out_channels,
            kernel_size=1,
            bn_weight_init=compress_init,
        )

    def _token_mixer(self, x: mx.array) -> mx.array:
        x = self.token_conv(x)
        if self.token_squeeze_excitation is not None:
            x = self.token_squeeze_excitation(x)
        return x

    def _channel_mixer(self, x: mx.array) -> mx.array:
        x = self.channel_conv1(x)
        if self.channel_act is not None:
            x = self.channel_act(x)
        return self.channel_conv2(x)

    def __call__(self, x: mx.array) -> mx.array:
        x = self._token_mixer(x)
        mixed = self._channel_mixer(x)
        if self.has_residual:
            return x + mixed
        return mixed


class SimpleStem(nn.Module):
    """Two-stage conv stem used by tiny recognition models."""

    def __init__(self, mid_channels: int, out_channels: int) -> None:
        self.conv1 = Conv2DBN(3, mid_channels, kernel_size=3, stride=2, padding=1)
        self.act = nn.GELU()
        self.conv2 = Conv2DBN(mid_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv2(self.act(self.conv1(x)))


class RecStage(nn.Module):
    """Sequential LCNetV4 blocks for one recognition stage."""

    def __init__(self, specs: tuple[BlockSpec, ...]) -> None:
        for index, spec in enumerate(specs):
            setattr(self, f"blocks_{index}", LCNetV4RecBlock(spec))
        self._depth = len(specs)

    def __call__(self, x: mx.array) -> mx.array:
        for index in range(self._depth):
            block = getattr(self, f"blocks_{index}")
            x = block(x)
        return x


class RecEncoder(nn.Module):
    """Hub-aligned recognition encoder wrapper."""

    def __init__(
        self,
        stem_kind: StemKind,
        stem_mid: int,
        stem_out: int,
        block_configs: tuple[tuple[BlockSpec, ...], ...],
    ) -> None:
        if stem_kind == "branch":
            self.convolution = StemBlock(3, stem_mid, stem_out)
        else:
            self.convolution = SimpleStem(stem_mid, stem_out)
        for index, blocks in enumerate(block_configs):
            setattr(self, f"blocks_{index}", RecStage(blocks))
        self._depth = len(block_configs)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.convolution(x)
        for index in range(self._depth):
            stage = getattr(self, f"blocks_{index}")
            x = stage(x)
        return x


class PPLCNetV4Rec(nn.Module):
    """PPLCNetV4 backbone for text recognition."""

    def __init__(
        self,
        stem_kind: StemKind,
        stem_mid: int,
        stem_out: int,
        block_configs: tuple[tuple[BlockSpec, ...], ...],
    ) -> None:
        self.encoder = RecEncoder(stem_kind, stem_mid, stem_out, block_configs)
        self.pool = nn.AvgPool2d(kernel_size=(3, 2), stride=(3, 2))

    def __call__(self, x: mx.array) -> mx.array:
        """Run recognition backbone forward pass.

        Args:
            x: Input tensor in NHWC layout ``[B, H, W, 3]``.

        Returns:
            Pooled feature map ``[B, 1, W', C]``.
        """
        x = self.encoder(x)
        if x.shape[1] < 3:
            raise ValueError(f"feature height {x.shape[1]} is smaller than pool kernel 3")
        return self.pool(x)
