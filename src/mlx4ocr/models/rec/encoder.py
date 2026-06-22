"""Recognition sequence encoders (Im2Seq and LightSVTR)."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx4ocr.models.common import Conv2DBN, LayerNorm


def im2seq(x: mx.array) -> mx.array:
    """Flatten a pooled feature map into a width-major sequence.

    Args:
        x: Feature map ``[B, 1, W, C]``.

    Returns:
        Sequence tensor ``[B, W, C]``.
    """
    if x.ndim != 4:
        raise ValueError(f"expected rank-4 feature map, got shape {x.shape}")
    if x.shape[1] != 1:
        raise ValueError(f"expected height 1 before Im2Seq, got {x.shape[1]}")
    return mx.squeeze(x, axis=1)


class SVTRMlp(nn.Module):
    """Feed-forward block used by LightSVTR."""

    def __init__(self, dim: int, mlp_ratio: float) -> None:
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.act(self.fc1(x)))


class SVTRBlock(nn.Module):
    """Transformer block with post-norm residual connections."""

    def __init__(self, dim: int, mlp_ratio: float, num_heads: int = 8) -> None:
        self.norm1 = LayerNorm(dim, eps=1e-5)
        self.self_attn = nn.MultiHeadAttention(dim, num_heads, bias=True)
        self.norm2 = LayerNorm(dim, eps=1e-5)
        self.mlp = SVTRMlp(dim, mlp_ratio)

    def __call__(self, x: mx.array) -> mx.array:
        normed = self.norm1(x)
        x = x + self.self_attn(normed, normed, normed)
        return x + self.mlp(self.norm2(x))


class EncoderWithLightSVTR(nn.Module):
    """Lightweight SVTR encoder neck for small/medium recognition."""

    def __init__(
        self,
        in_channels: int,
        dims: int,
        *,
        depth: int = 2,
        mlp_ratio: float = 4.0,
        local_kernel: int = 7,
    ) -> None:
        pad_w = local_kernel // 2
        self.conv_reduce = Conv2DBN(in_channels, dims, kernel_size=1)
        self.skip_conv = Conv2DBN(in_channels, dims, kernel_size=1)
        self.local_conv = Conv2DBN(
            dims,
            dims,
            kernel_size=(1, local_kernel),
            padding=(0, pad_w),
            groups=dims,
        )
        for index in range(depth):
            setattr(self, f"svtr_block_{index}", SVTRBlock(dims, mlp_ratio))
        self._depth = depth
        self.norm = LayerNorm(dims, eps=1e-6)
        self.act = nn.SiLU()

    def __call__(self, x: mx.array) -> mx.array:
        skip = self.act(self.skip_conv(x))
        z = self.act(self.conv_reduce(x))
        z = z + self.act(self.local_conv(z))
        batch, height, width, channels = z.shape
        z = mx.reshape(z, (batch, height * width, channels))
        for index in range(self._depth):
            block = getattr(self, f"svtr_block_{index}")
            z = block(z)
        z = self.norm(z)
        z = mx.reshape(z, (batch, height, width, channels))
        return z + skip
