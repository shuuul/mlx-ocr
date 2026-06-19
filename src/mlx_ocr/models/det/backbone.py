"""PPLCNetV4 detection backbone."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_ocr.models.common import Conv2DBN, ConvBNAct, HardSigmoid, SELayer, build_activation
from mlx_ocr.models.common.conv_bn import apply_same_padding_nhwc
from mlx_ocr.models.det.config import BlockSpec


class TokenSqueezeExcitation(nn.Module):
    """SE block aligned with Hub ``token_squeeze_excitation`` keys."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        hidden = channels // reduction
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.conv2 = nn.Conv2d(hidden, channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.hardsigmoid = HardSigmoid()

    def __call__(self, x: mx.array) -> mx.array:
        pooled = mx.mean(x, axis=(1, 2), keepdims=True)
        gate = self.hardsigmoid(self.conv2(self.relu(self.conv1(pooled))))
        return x * gate


class LCNetV4Block(nn.Module):
    """LCNetV4 inverted residual block for detection."""

    def __init__(self, spec: BlockSpec) -> None:
        _kernel, in_channels, out_channels, stride, use_se = spec
        self.has_residual = in_channels == out_channels and stride == 1
        self.use_rep_dw = in_channels == out_channels and stride == 1

        if self.use_rep_dw:
            padding = (_kernel - 1) // 2
            self.token_conv = nn.Conv2d(
                in_channels,
                in_channels,
                _kernel,
                padding=padding,
                groups=in_channels,
                bias=True,
            )
        else:
            padding = (_kernel - 1) // 2
            stride_value = stride if isinstance(stride, int) else stride[0]
            self.token_conv = Conv2DBN(
                in_channels,
                in_channels,
                kernel_size=_kernel,
                stride=stride_value,
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
        if isinstance(self.token_conv, Conv2DBN):
            x = self.token_conv(x)
        else:
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


class StemBlock(nn.Module):
    """Multi-branch stem with total stride 4."""

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int) -> None:
        self.stem1 = ConvBNAct(in_channels, mid_channels, kernel_size=3, stride=2, padding=1)
        self.stem2a = ConvBNAct(
            mid_channels,
            mid_channels // 2,
            kernel_size=2,
            stride=1,
            padding="SAME",
        )
        self.stem2b = ConvBNAct(
            mid_channels // 2,
            mid_channels,
            kernel_size=2,
            stride=1,
            padding="SAME",
        )
        self.stem3 = ConvBNAct(mid_channels * 2, mid_channels, kernel_size=3, stride=2, padding=1)
        self.stem4 = ConvBNAct(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.stem1(x)
        branch = self.stem2b(self.stem2a(x))
        pooled = self.pool(apply_same_padding_nhwc(x, kernel_size=2, stride=1))
        merged = mx.concatenate([pooled, branch], axis=-1)
        return self.stem4(self.stem3(merged))


class Stage(nn.Module):
    """Sequential LCNetV4 blocks for one detection scale."""

    def __init__(self, specs: tuple[BlockSpec, ...]) -> None:
        for index, spec in enumerate(specs):
            setattr(self, f"blocks_{index}", LCNetV4Block(spec))
        self._depth = len(specs)

    def __call__(self, x: mx.array) -> mx.array:
        for index in range(self._depth):
            block = getattr(self, f"blocks_{index}")
            x = block(x)
        return x


class Encoder(nn.Module):
    """Hub-aligned encoder wrapper."""

    def __init__(
        self,
        stem_mid: int,
        stem_out: int,
        block_configs: tuple[tuple[BlockSpec, ...], ...],
    ) -> None:
        self.convolution = StemBlock(3, stem_mid, stem_out)
        for index, blocks in enumerate(block_configs):
            setattr(self, f"blocks_{index}", Stage(blocks))
        self._depth = len(block_configs)

    def __call__(self, x: mx.array) -> list[mx.array]:
        x = self.convolution(x)
        outputs: list[mx.array] = []
        for index in range(self._depth):
            stage = getattr(self, f"blocks_{index}")
            x = stage(x)
            outputs.append(x)
        return outputs


class PPLCNetV4Det(nn.Module):
    """Four-scale PPLCNetV4 backbone for text detection."""

    def __init__(
        self,
        stem_mid: int,
        stem_out: int,
        block_configs: tuple[tuple[BlockSpec, ...], ...],
    ) -> None:
        self.encoder = Encoder(stem_mid, stem_out, block_configs)

    def __call__(self, x: mx.array) -> list[mx.array]:
        return self.encoder(x)
