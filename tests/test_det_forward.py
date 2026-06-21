"""Detection forward parity tests against golden probability maps."""

from __future__ import annotations

import numpy as np
import pytest

from mlx_ocr.hub.download import download_model
from mlx_ocr.hub.registry import ModelVariant
from mlx_ocr.models.det import DetectionModel
from mlx_ocr.preprocess.det import det_preprocess, nhwc_prob_to_nchw
from tests.conftest import GOLDEN_ROOT, load_golden_npy
from tests.reference.compare import assert_allclose


@pytest.mark.parametrize("variant", ("tiny", "small", "medium"))
def test_det_forward_matches_golden(
    sample_bgr_image: np.ndarray,
    variant: str,
) -> None:
    """Loaded detection models reproduce committed golden probability maps."""
    model = DetectionModel.from_artifacts(download_model(variant, "det"))
    preprocessed = det_preprocess(sample_bgr_image)
    prob_map = model(preprocessed.image)
    actual = nhwc_prob_to_nchw(prob_map)
    expected = load_golden_npy(GOLDEN_ROOT / variant / "det" / "prob_map.npy")
    assert_allclose(actual, expected, rtol=1e-4, atol=2e-3, err_msg=f"{variant} prob map")


def test_detection_model_from_artifacts(sample_bgr_image: np.ndarray) -> None:
    """``DetectionModel.from_artifacts`` loads tiny weights from the Hub."""
    artifacts = download_model("tiny", "det")
    model = DetectionModel.from_artifacts(artifacts)
    preprocessed = det_preprocess(sample_bgr_image)
    output = model(preprocessed.image)
    assert output.shape[0] == 1
    assert output.shape[-1] == 1


@pytest.mark.parametrize("variant", ("tiny", "small", "medium"))
def test_det_weight_load_strict(variant: ModelVariant) -> None:
    """Strict weight loading succeeds for every detection variant."""
    model = DetectionModel.from_artifacts(download_model(variant, "det"))
    assert isinstance(model, DetectionModel)
