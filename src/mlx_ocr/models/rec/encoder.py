"""Recognition sequence encoders (Im2Seq and LightSVTR)."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_ocr.models.common import Conv2DBN


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


class SVTRAttention(nn.Module):
    """Global self-attention block used by LightSVTR."""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True) -> None:
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.projection = nn.Linear(dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        batch, seq_len, _channels = x.shape
        qkv = self.qkv(x)
        qkv = mx.reshape(qkv, (batch, seq_len, 3, self.num_heads, self.head_dim))
        qkv = mx.transpose(qkv, (2, 0, 3, 1, 4))
        q = qkv[0] * self.scale
        k = qkv[1]
        v = qkv[2]
        attn = mx.softmax(q @ mx.transpose(k, (0, 1, 3, 2)), axis=-1)
        out = attn @ v
        out = mx.reshape(mx.transpose(out, (0, 2, 1, 3)), (batch, seq_len, -1))
        return self.projection(out)


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
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.self_attn = SVTRAttention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.mlp = SVTRMlp(dim, mlp_ratio)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.self_attn(self.norm1(x))
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
        self.norm = nn.LayerNorm(dims, eps=1e-6)
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
