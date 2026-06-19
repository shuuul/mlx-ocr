"""Tests for shared MLX primitives and safetensors loading."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from huggingface_hub import hf_hub_download
from safetensors.numpy import save_file

from mlx_ocr.hub.weights import (
    WeightMapper,
    align_tensor_to_parameter,
    flatten_module_parameters,
    load_into_module,
    load_safetensors,
    paddle_conv_weight_to_mlx,
    rewrite_hub_key,
)
from mlx_ocr.models.common import (
    Conv2DBN,
    ConvBNAct,
    DilatedReparamBlock,
    RepDWConv,
    SELayer,
    build_activation,
)
from mlx_ocr.models.common.activations import HardSigmoid, HardSigmoidClip


def test_hard_sigmoid_clip_matches_paddle_functional() -> None:
    act = HardSigmoidClip()
    x = mx.array([-3.0, 0.0, 3.0])
    y = np.asarray(act(x))
    assert y[0] == 0.0
    assert y[1] == pytest.approx(0.5)
    assert y[2] == 1.0


def test_hard_sigmoid_matches_paddle_formula() -> None:
    act = HardSigmoid()
    x = mx.array([-3.0, -2.9, 0.0, 3.0])
    y = np.asarray(act(x))
    assert y[0] == pytest.approx(0.0)
    assert y[1] == pytest.approx(0.016666666, rel=1e-5)
    assert y[2] == pytest.approx(0.5)
    assert y[3] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("channels", "kernel_size"),
    [(16, 3), (32, 9)],
)
def test_rep_dw_conv_preserves_shape(channels: int, kernel_size: int) -> None:
    layer = RepDWConv(channels, kernel_size=kernel_size)
    x = mx.random.normal((1, 17, 23, channels))
    y = layer(x)
    assert y.shape == x.shape


def test_dilated_reparam_block_rejects_invalid_kernel() -> None:
    with pytest.raises(ValueError, match="kernel_size"):
        DilatedReparamBlock(8, kernel_size=8)


def test_se_layer_recalibrates_channels() -> None:
    layer = SELayer(24, reduction=4)
    x = mx.random.normal((2, 8, 8, 24))
    y = layer(x)
    assert y.shape == x.shape


def test_conv_bn_act_fused_and_unfused_paths() -> None:
    x = mx.random.normal((1, 7, 7, 8))
    unfused = ConvBNAct(8, 16, kernel_size=3, fused=False)
    fused = ConvBNAct(8, 16, kernel_size=3, fused=True)
    assert unfused(x).shape == (1, 7, 7, 16)
    assert fused(x).shape == (1, 7, 7, 16)
    assert "bn.weight" in flatten_module_parameters(unfused)
    assert "bn.weight" not in flatten_module_parameters(fused)


def test_build_activation_none_returns_none() -> None:
    assert build_activation("none") is None


def test_paddle_conv_weight_to_mlx_transposes_oihw() -> None:
    weight = mx.arange(24, dtype=mx.float32).reshape(2, 3, 2, 2)
    converted = paddle_conv_weight_to_mlx(weight)
    assert converted.shape == (2, 2, 2, 3)
    assert int(converted[0, 0, 0, 0]) == 0
    assert int(converted[0, 0, 0, 1]) == 4


def test_align_tensor_to_parameter_accepts_paddle_conv_layout() -> None:
    conv = Conv2DBN(3, 4, kernel_size=3, padding=1)
    expected_shape = flatten_module_parameters(conv)["conv.weight"].shape
    paddle_weight = mx.random.normal((4, 3, 3, 3))
    aligned = align_tensor_to_parameter(paddle_weight, expected_shape)
    assert aligned.shape == expected_shape


def test_load_safetensors_round_trip(tmp_path: Path) -> None:
    arrays = {
        "layer.weight": np.ones((4, 3, 3, 3), dtype=np.float32),
        "layer.bias": np.zeros((4,), dtype=np.float32),
    }
    path = tmp_path / "weights.safetensors"
    save_file(arrays, path)

    loaded = load_safetensors(path)
    assert set(loaded) == set(arrays)
    assert np.allclose(np.asarray(loaded["layer.weight"]), arrays["layer.weight"])


def test_rewrite_hub_key_suffixes() -> None:
    key = "encoder.blocks.0.channel_conv1.convolution.weight"
    rewritten = rewrite_hub_key(key)
    assert rewritten == "encoder.blocks.0.channel_conv1.conv.weight"


def test_load_into_module_strict_success(tmp_path: Path) -> None:
    conv = Conv2DBN(3, 4, kernel_size=3, padding=1)
    params = flatten_module_parameters(conv)

    source_arrays = {
        "src.conv.weight": np.asarray(params["conv.weight"]),
        "src.conv.bias": np.zeros((4,), dtype=np.float32),
        "src.bn.weight": np.asarray(params["bn.weight"]),
        "src.bn.bias": np.asarray(params["bn.bias"]),
        "src.bn.running_mean": np.asarray(params["bn.running_mean"]),
        "src.bn.running_var": np.asarray(params["bn.running_var"]),
    }
    path = tmp_path / "block.safetensors"
    save_file(source_arrays, path)

    mapper = WeightMapper.from_pairs(
        {
            "src.conv.weight": "conv.weight",
            "src.bn.weight": "bn.weight",
            "src.bn.bias": "bn.bias",
            "src.bn.running_mean": "bn.running_mean",
            "src.bn.running_var": "bn.running_var",
        }
    )
    result = load_into_module(conv, load_safetensors(path), mapper, strict=False)
    assert result.missing == ()
    assert result.unexpected == ("src.conv.bias",)


def test_load_into_module_strict_failure_reports_missing(tmp_path: Path) -> None:
    conv = Conv2DBN(3, 4, kernel_size=3, padding=1)
    arrays = {"only.weight": np.zeros((4, 3, 3, 3), dtype=np.float32)}
    path = tmp_path / "partial.safetensors"
    save_file(arrays, path)

    mapper = WeightMapper.from_pairs({"only.weight": "conv.weight"})
    with pytest.raises(ValueError, match="strict weight load failed"):
        load_into_module(conv, load_safetensors(path), mapper, strict=True)


def test_weight_mapper_from_transform_matches_suffixes() -> None:
    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.channel_conv1 = Conv2DBN(8, 16, kernel_size=1)

    block = Block()
    target_keys = tuple(flatten_module_parameters(block))
    source_keys = (
        "model.backbone.encoder.blocks.0.channel_conv1.convolution.weight",
        "model.backbone.encoder.blocks.0.channel_conv1.normalization.weight",
        "model.backbone.encoder.blocks.0.channel_conv1.normalization.bias",
        "model.backbone.encoder.blocks.0.channel_conv1.normalization.running_mean",
        "model.backbone.encoder.blocks.0.channel_conv1.normalization.running_var",
    )
    mapper = WeightMapper.from_transform(
        source_keys,
        target_keys,
        strip_prefixes=("model.backbone.encoder.",),
    )
    assert mapper.map_key(source_keys[0]) == "channel_conv1.conv.weight"
    assert mapper.map_key(source_keys[1]) == "channel_conv1.bn.weight"


def test_load_hub_subset_into_rep_dw_conv() -> None:
    path = Path(
        hf_hub_download("PaddlePaddle/PP-OCRv6_medium_det_safetensors", "model.safetensors")
    )
    tensors = load_safetensors(path)
    source_weight = tensors[
        "model.backbone.encoder.blocks.0.blocks.0.token_conv.weight"
    ]
    channels = int(source_weight.shape[0])
    layer = RepDWConv(channels, kernel_size=3)
    mapper = WeightMapper.from_pairs(
        {
            "model.backbone.encoder.blocks.0.blocks.0.token_conv.weight": "conv.weight",
            "model.backbone.encoder.blocks.0.blocks.0.token_conv.bias": "conv.bias",
        }
    )
    subset = {key: tensors[key] for key in mapper.mapping}
    result = load_into_module(layer, subset, mapper, strict=True)
    assert result.loaded == ("conv.bias", "conv.weight")
    assert result.missing == ()
    assert result.unexpected == ()
