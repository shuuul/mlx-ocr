"""Detection preprocess parity tests."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from mlx4ocr.preprocess.det import det_preprocess, normalize_det_image_mlx, resize_det_image
from tests.conftest import GOLDEN_ROOT, load_golden_npy
from tests.reference.compare import assert_allclose


def test_det_preprocess_matches_reference(
    sample_bgr_image: np.ndarray,
    det_preprocessed: tuple[np.ndarray, np.ndarray],
) -> None:
    """MLX detection preprocess matches vendored reference operators."""
    reference_tensor, reference_shape = det_preprocessed
    result = det_preprocess(sample_bgr_image)
    actual = np.asarray(result.nchw)
    assert_allclose(actual, reference_tensor, err_msg="det preprocessed tensor")
    assert result.shape == (
        float(reference_shape[0]),
        float(reference_shape[1]),
        float(reference_shape[2]),
        float(reference_shape[3]),
    )


def test_det_preprocess_golden_per_variant(
    sample_bgr_image: np.ndarray,
    variant: str,
) -> None:
    """Preprocess output matches committed golden tensors for each variant."""
    result = det_preprocess(sample_bgr_image)
    expected_tensor = load_golden_npy(GOLDEN_ROOT / variant / "det" / "preprocessed.npy")
    expected_shape = load_golden_npy(GOLDEN_ROOT / variant / "det" / "shape.npy")
    assert_allclose(np.asarray(result.nchw), expected_tensor, err_msg=f"{variant} preprocess")
    assert_allclose(
        np.asarray(result.shape, dtype=np.float32),
        expected_shape,
        err_msg=f"{variant} shape",
    )


def test_mlx_det_normalize_matches_cpu_preprocess(sample_bgr_image: np.ndarray) -> None:
    """MLX detection normalization matches the CPU preprocessing path."""
    expected = det_preprocess(sample_bgr_image)
    resized, shape = resize_det_image(sample_bgr_image)
    actual = normalize_det_image_mlx(mx.array(resized))
    assert_allclose(np.asarray(actual), np.asarray(expected.image), err_msg="mlx det normalize")
    assert_allclose(
        np.asarray(shape, dtype=np.float32),
        np.asarray(expected.shape, dtype=np.float32),
        err_msg="mlx det shape",
    )
