"""Tests for Paddle-pretrained recognition weight patching."""

from __future__ import annotations

import pickle
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx4ocr.hub.paddle_pretrained import load_pretrained_rec_state
from mlx4ocr.hub.rec_weight_patch import (
    patch_recognition_hub_tensors,
    resolve_recognition_weight_source,
)
from mlx4ocr.hub.weights import paddle_conv_weight_to_mlx
from mlx4ocr.models.rec.model import split_recognition_attention_tensors


def test_split_recognition_attention_tensors_expands_fused_qkv() -> None:
    """Fused QKV and projection Hub keys map to MultiHeadAttention parameters."""
    dim = 120
    tensors = {
        "head.encoder.svtr_block.0.self_attn.qkv.weight": mx.arange(
            dim * 3 * dim, dtype=mx.float32
        ).reshape(dim * 3, dim),
        "head.encoder.svtr_block.0.self_attn.qkv.bias": mx.arange(dim * 3, dtype=mx.float32),
        "head.encoder.svtr_block.0.self_attn.projection.weight": mx.ones((dim, dim)),
        "head.encoder.svtr_block.0.self_attn.projection.bias": mx.full((dim,), 2.0),
        "head.encoder.svtr_block.0.mlp.fc1.weight": mx.zeros((1, 1)),
    }
    direct, remaining = split_recognition_attention_tensors(tensors)
    assert set(remaining) == {"head.encoder.svtr_block.0.mlp.fc1.weight"}
    assert np.asarray(direct["encoder.svtr_block_0.self_attn.query_proj.weight"]).shape == (
        dim,
        dim,
    )
    assert np.asarray(direct["encoder.svtr_block_0.self_attn.key_proj.weight"]).shape == (dim, dim)
    assert np.asarray(direct["encoder.svtr_block_0.self_attn.value_proj.weight"]).shape == (
        dim,
        dim,
    )
    assert int(np.asarray(direct["encoder.svtr_block_0.self_attn.query_proj.weight"])[0, 0]) == 0
    assert (
        int(np.asarray(direct["encoder.svtr_block_0.self_attn.key_proj.weight"])[0, 0]) == dim * dim
    )
    assert int(np.asarray(direct["encoder.svtr_block_0.self_attn.value_proj.bias"])[0]) == dim * 2
    assert float(np.asarray(direct["encoder.svtr_block_0.self_attn.out_proj.bias"])[0]) == 2.0


def test_resolve_recognition_weight_source_auto() -> None:
    assert resolve_recognition_weight_source("tiny", "auto") == "hub"
    assert resolve_recognition_weight_source("small", "auto") == "paddle_pretrained"
    assert resolve_recognition_weight_source("medium", "auto") == "paddle_pretrained"


def test_patch_swaps_head_encoder_conv_reduce_and_skip_conv(
    tmp_path: Path,
) -> None:
    """Patch restores conv_reduce on block 0 and skip_conv on block 1."""
    pretrained_path = tmp_path / "PP-OCRv6_small_rec_pretrained.pdparams"
    reduce_weight = np.arange(120 * 384, dtype=np.float32).reshape(120, 384, 1, 1)
    skip_weight = np.arange(120 * 384, dtype=np.float32).reshape(120, 384, 1, 1) + 1000.0
    pretrained = {
        "head.ctc_encoder.encoder.conv_reduce.conv.weight": reduce_weight,
        "head.ctc_encoder.encoder.conv_reduce.norm.weight": np.ones(120, dtype=np.float32),
        "head.ctc_encoder.encoder.conv_reduce.norm.bias": np.zeros(120, dtype=np.float32),
        "head.ctc_encoder.encoder.conv_reduce.norm._mean": np.zeros(120, dtype=np.float32),
        "head.ctc_encoder.encoder.conv_reduce.norm._variance": np.ones(120, dtype=np.float32),
        "head.ctc_encoder.encoder.skip_conv.conv.weight": skip_weight,
        "head.ctc_encoder.encoder.skip_conv.norm.weight": np.full(120, 2.0, dtype=np.float32),
        "head.ctc_encoder.encoder.skip_conv.norm.bias": np.full(120, 3.0, dtype=np.float32),
        "head.ctc_encoder.encoder.skip_conv.norm._mean": np.full(120, 4.0, dtype=np.float32),
        "head.ctc_encoder.encoder.skip_conv.norm._variance": np.full(120, 5.0, dtype=np.float32),
    }
    pretrained_path.write_bytes(pickle.dumps(pretrained))

    hub_tensors = {
        "head.encoder.conv_block.0.convolution.weight": mx.zeros((120, 384, 1, 1)),
        "head.encoder.conv_block.0.normalization.weight": mx.zeros((120,)),
        "head.encoder.conv_block.0.normalization.bias": mx.zeros((120,)),
        "head.encoder.conv_block.0.normalization.running_mean": mx.zeros((120,)),
        "head.encoder.conv_block.0.normalization.running_var": mx.zeros((120,)),
        "head.encoder.conv_block.1.convolution.weight": mx.zeros((120, 384, 1, 1)),
        "head.encoder.conv_block.1.normalization.weight": mx.zeros((120,)),
        "head.encoder.conv_block.1.normalization.bias": mx.zeros((120,)),
        "head.encoder.conv_block.1.normalization.running_mean": mx.zeros((120,)),
        "head.encoder.conv_block.1.normalization.running_var": mx.zeros((120,)),
    }

    state = load_pretrained_rec_state(pretrained_path)
    patched = patch_recognition_hub_tensors("small", hub_tensors, state)

    expected_reduce = np.asarray(
        paddle_conv_weight_to_mlx(mx.array(reduce_weight)),
        dtype=np.float32,
    )
    expected_skip = np.asarray(
        paddle_conv_weight_to_mlx(mx.array(skip_weight)),
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        np.asarray(patched["head.encoder.conv_block.0.convolution.weight"]),
        expected_reduce,
    )
    np.testing.assert_allclose(
        np.asarray(patched["head.encoder.conv_block.1.convolution.weight"]),
        expected_skip,
    )
    np.testing.assert_allclose(
        np.asarray(patched["head.encoder.conv_block.0.normalization.weight"]),
        np.ones(120, dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(patched["head.encoder.conv_block.1.normalization.weight"]),
        np.full(120, 2.0, dtype=np.float32),
    )


def test_patch_rejects_tiny() -> None:
    with pytest.raises(ValueError, match="does not apply"):
        patch_recognition_hub_tensors("tiny", {}, {})
