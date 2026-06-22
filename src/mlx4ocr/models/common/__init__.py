"""Shared MLX primitives for PP-OCRv6 model code."""

from mlx4ocr.models.common.activations import ActivationName, HardSigmoid, build_activation
from mlx4ocr.models.common.conv_bn import Conv2DBN, ConvBNAct
from mlx4ocr.models.common.fuse import fuse_for_inference
from mlx4ocr.models.common.norm import LayerNorm
from mlx4ocr.models.common.rep_conv import DilatedReparamBlock, RepDWConv
from mlx4ocr.models.common.se import SELayer

__all__ = [
    "ActivationName",
    "Conv2DBN",
    "ConvBNAct",
    "DilatedReparamBlock",
    "HardSigmoid",
    "LayerNorm",
    "RepDWConv",
    "SELayer",
    "build_activation",
    "fuse_for_inference",
]
