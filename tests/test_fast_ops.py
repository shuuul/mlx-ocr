"""Tests for MLX fast-operator helpers and inference fusion."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_ocr.hub.download import download_model
from mlx_ocr.models.common.conv_bn import Conv2DBN
from mlx_ocr.models.common.fuse import fuse_conv_batch_norm, fuse_for_inference
from mlx_ocr.models.common.norm import LayerNorm
from mlx_ocr.models.det.model import load_detection_model
from mlx_ocr.models.rec.model import load_recognition_model
from mlx_ocr.preprocess.rec import rec_preprocess_crop_from_image
from tests.conftest import GOLDEN_ROOT, load_golden_npy
from tests.reference.compare import assert_allclose


def test_layer_norm_matches_nn_layer_norm() -> None:
    """Project LayerNorm delegates to the same fast kernel as nn.LayerNorm."""
    dims = 32
    custom = LayerNorm(dims, eps=1e-5)
    reference = nn.LayerNorm(dims, eps=1e-5)
    custom.update(reference.parameters())
    x = mx.random.normal((2, 17, dims))
    mx.eval(x)
    actual = custom(x)
    expected = reference(x)
    mx.eval(actual, expected)
    assert float(mx.max(mx.abs(actual - expected))) == 0.0


def test_fuse_conv_batch_norm_matches_unfused_output() -> None:
    """Conv+BN fusion preserves eval-mode outputs."""
    block = Conv2DBN(8, 16, kernel_size=3, padding=1)
    block.eval()
    x = mx.random.normal((1, 24, 24, 8))
    expected = block(x)
    fused_weight, fused_bias = fuse_conv_batch_norm(block.conv, block.bn)
    block.conv.weight = fused_weight
    block.conv.bias = fused_bias
    block.bn = None
    actual = block(x)
    mx.eval(actual, expected)
    assert_allclose(
        np.asarray(actual, dtype=np.float32),
        np.asarray(expected, dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_fuse_for_inference_preserves_recognition_golden(
    sample_bgr_image: np.ndarray,
) -> None:
    """Fused recognition weights still match committed softmax goldens."""
    model = load_recognition_model(download_model("small", "rec"))
    preprocessed = rec_preprocess_crop_from_image(sample_bgr_image)
    actual = np.asarray(model(preprocessed.image), dtype=np.float32)
    expected = load_golden_npy(GOLDEN_ROOT / "small" / "rec" / "softmax.npy")
    assert_allclose(actual, expected, rtol=1e-4, atol=2e-3)


def test_fuse_for_inference_runs_on_detection_model() -> None:
    """Detection models fuse conv+bn blocks after loading."""
    model = load_detection_model(download_model("tiny", "det"))
    x = mx.zeros((1, 64, 64, 3))
    output = model(x)
    mx.eval(output)
    assert output.shape == (1, 64, 64, 1)
    assert fuse_for_inference(model) == 0
