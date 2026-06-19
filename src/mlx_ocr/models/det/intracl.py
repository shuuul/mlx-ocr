"""Intra-class contrastive block for medium detection neck."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_ocr.models.det.head import ConvNorm


class IntraCLBlock(nn.Module):
    """IntraCL relation block used by RepLKPAN."""

    def __init__(self, channels: int, reduce_factor: int = 4) -> None:
        reduced = channels // reduce_factor
        self.conv_reduce_channel = nn.Conv2d(channels, reduced, kernel_size=1)
        self.conv_final = ConvNorm(reduced, channels, kernel_size=1, padding=0, bias=True)

        self.vertical_long_to_small_conv_longratio = nn.Conv2d(
            reduced, reduced, kernel_size=(7, 1), padding=(3, 0)
        )
        self.vertical_long_to_small_conv_midratio = nn.Conv2d(
            reduced, reduced, kernel_size=(5, 1), padding=(2, 0)
        )
        self.vertical_long_to_small_conv_shortratio = nn.Conv2d(
            reduced, reduced, kernel_size=(3, 1), padding=(1, 0)
        )

        self.horizontal_small_to_long_conv_longratio = nn.Conv2d(
            reduced, reduced, kernel_size=(1, 7), padding=(0, 3)
        )
        self.horizontal_small_to_long_conv_midratio = nn.Conv2d(
            reduced, reduced, kernel_size=(1, 5), padding=(0, 2)
        )
        self.horizontal_small_to_long_conv_shortratio = nn.Conv2d(
            reduced, reduced, kernel_size=(1, 3), padding=(0, 1)
        )

        self.symmetric_conv_long_longratio = nn.Conv2d(
            reduced, reduced, kernel_size=(7, 7), padding=(3, 3)
        )
        self.symmetric_conv_long_midratio = nn.Conv2d(
            reduced, reduced, kernel_size=(5, 5), padding=(2, 2)
        )
        self.symmetric_conv_long_shortratio = nn.Conv2d(
            reduced, reduced, kernel_size=(3, 3), padding=(1, 1)
        )

        self.relu = nn.ReLU()

    def __call__(self, x: mx.array) -> mx.array:
        reduced = self.conv_reduce_channel(x)

        stage7 = (
            self.symmetric_conv_long_longratio(reduced)
            + self.vertical_long_to_small_conv_longratio(reduced)
            + self.horizontal_small_to_long_conv_longratio(reduced)
        )
        stage5 = (
            self.symmetric_conv_long_midratio(stage7)
            + self.vertical_long_to_small_conv_midratio(stage7)
            + self.horizontal_small_to_long_conv_midratio(stage7)
        )
        stage3 = (
            self.symmetric_conv_long_shortratio(stage5)
            + self.vertical_long_to_small_conv_shortratio(stage5)
            + self.horizontal_small_to_long_conv_shortratio(stage5)
        )

        relation = self.relu(self.conv_final(stage3))
        return x + relation
