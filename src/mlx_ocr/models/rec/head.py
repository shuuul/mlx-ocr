"""CTC recognition head."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class CTCHead(nn.Module):
    """CTC classification head with optional guide layer and hidden projection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        use_guide: bool = False,
        mid_channels: int | None = None,
    ) -> None:
        self.use_guide = use_guide
        if use_guide:
            self.conv1 = nn.Conv2d(
                in_channels,
                in_channels,
                (1, 5),
                padding=(0, 2),
                groups=in_channels,
                bias=False,
            )
            self.norm1 = nn.BatchNorm(in_channels)
            self.act1 = nn.Hardswish()
            self.conv2 = nn.Conv2d(in_channels, in_channels, 1, bias=False)
            self.norm2 = nn.BatchNorm(in_channels)
            self.act2 = nn.Hardswish()
        if mid_channels is None:
            self.head = nn.Linear(in_channels, out_channels)
        else:
            self.fc1 = nn.Linear(in_channels, mid_channels)
            self.fc2 = nn.Linear(mid_channels, out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        """Run CTC head inference forward pass.

        Args:
            x: Sequence tensor ``[B, T, C]``.

        Returns:
            Softmax probabilities ``[B, T, num_classes]``.
        """
        if self.use_guide:
            batch, width, channels = x.shape
            spatial = mx.reshape(x, (batch, 1, width, channels))
            spatial = self.act1(self.norm1(self.conv1(spatial)))
            spatial = self.act2(self.norm2(self.conv2(spatial)))
            x = mx.reshape(spatial, (batch, width, channels))

        if hasattr(self, "head"):
            logits = self.head(x)
        else:
            logits = self.fc2(self.fc1(x))
        return mx.softmax(logits, axis=-1)
