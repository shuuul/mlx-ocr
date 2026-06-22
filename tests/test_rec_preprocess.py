"""Recognition preprocess parity tests."""

from __future__ import annotations

import numpy as np

from mlx4ocr.preprocess.rec import rec_preprocess
from tests.conftest import GOLDEN_ROOT, load_golden_npy
from tests.reference.compare import assert_allclose
from tests.reference.preprocess.rec_resize_norm import resize_norm_img


def test_rec_preprocess_matches_reference(sample_bgr_image: np.ndarray) -> None:
    """MLX recognition preprocess matches vendored resize_norm_img."""
    crop = sample_bgr_image[130:190, 40:280].copy()
    height, width = crop.shape[:2]
    max_wh_ratio = width / float(height)
    reference = np.expand_dims(
        resize_norm_img(crop, max_wh_ratio=max_wh_ratio),
        axis=0,
    )
    result = rec_preprocess(crop, max_wh_ratio=max_wh_ratio)
    assert_allclose(np.asarray(result.nchw), reference, err_msg="rec preprocessed tensor")


def test_rec_preprocess_golden_per_variant(sample_bgr_image: np.ndarray, variant: str) -> None:
    """Preprocess output matches committed golden tensors for each variant."""
    crop = sample_bgr_image[130:190, 40:280].copy()
    height, width = crop.shape[:2]
    result = rec_preprocess(crop, max_wh_ratio=width / float(height))
    expected = load_golden_npy(GOLDEN_ROOT / variant / "rec" / "preprocessed.npy")
    assert_allclose(np.asarray(result.nchw), expected, err_msg=f"{variant} rec preprocess")
