"""Recognition forward parity tests against golden softmax outputs."""

from __future__ import annotations

import numpy as np
import pytest
from huggingface_hub import hf_hub_download

from mlx_ocr.hub.download import download_model
from mlx_ocr.hub.registry import ModelVariant
from mlx_ocr.models.rec import RecognitionModel, load_recognition_model
from mlx_ocr.preprocess.rec import rec_preprocess_crop_from_image
from tests.conftest import GOLDEN_ROOT, load_golden_npy
from tests.reference.compare import assert_allclose


@pytest.mark.parametrize("variant", ("tiny", "small", "medium"))
def test_rec_forward_matches_golden(
    sample_bgr_image: np.ndarray,
    variant: str,
) -> None:
    """Loaded recognition models reproduce committed golden softmax outputs."""
    model = load_recognition_model(download_model(variant, "rec"))
    preprocessed = rec_preprocess_crop_from_image(sample_bgr_image)
    softmax = model(preprocessed.image)
    actual = np.asarray(softmax, dtype=np.float32)
    expected = load_golden_npy(GOLDEN_ROOT / variant / "rec" / "softmax.npy")
    assert_allclose(actual, expected, rtol=1e-4, atol=2e-3, err_msg=f"{variant} rec softmax")


def test_recognition_model_from_artifacts(sample_bgr_image: np.ndarray) -> None:
    """``RecognitionModel.from_artifacts`` loads tiny weights from the Hub."""
    artifacts = download_model("tiny", "rec")
    model = RecognitionModel.from_artifacts(artifacts)
    preprocessed = rec_preprocess_crop_from_image(sample_bgr_image)
    output = model(preprocessed.image)
    assert output.shape[0] == 1
    assert output.shape[-1] == 6906


@pytest.mark.parametrize("variant", ("tiny", "small", "medium"))
def test_rec_weight_load_strict(variant: ModelVariant) -> None:
    """Strict weight loading succeeds for every recognition variant."""
    hf_hub_download(
        f"PaddlePaddle/PP-OCRv6_{variant}_rec_safetensors",
        "model.safetensors",
    )
    model = load_recognition_model(download_model(variant, "rec"))
    assert isinstance(model, RecognitionModel)
