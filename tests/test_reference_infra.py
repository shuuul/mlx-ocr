"""Smoke tests for vendored reference infrastructure."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import GOLDEN_ROOT, load_golden_npy
from tests.reference.compare import assert_allclose
from tests.reference.postprocess.ctc_decode import CTCLabelDecode
from tests.reference.postprocess.db_postprocess import DBPostProcess
from tests.reference.preprocess.rec_resize_norm import resize_norm_img


def test_det_preprocess_golden_matches_reference(
    det_preprocessed: tuple[np.ndarray, np.ndarray],
) -> None:
    """Vendored det preprocess reproduces committed golden tensors."""
    tensor, shape = det_preprocessed
    expected_tensor = load_golden_npy(GOLDEN_ROOT / "medium" / "det" / "preprocessed.npy")
    expected_shape = load_golden_npy(GOLDEN_ROOT / "medium" / "det" / "shape.npy")
    assert_allclose(tensor, expected_tensor, err_msg="det preprocessed tensor")
    assert_allclose(shape, expected_shape, err_msg="det shape metadata")


def test_rec_resize_norm_output_shape(sample_bgr_image: np.ndarray) -> None:
    """Recognition resize produces CHW tensor padded to dynamic width."""
    crop = sample_bgr_image[130:190, 40:280]
    h, w = crop.shape[:2]
    tensor = resize_norm_img(crop, max_wh_ratio=w / float(h))
    assert tensor.shape[0] == 3
    assert tensor.shape[1] == 48
    assert tensor.shape[2] == int(48 * (w / float(h)))


def test_db_postprocess_runs_on_prob_map() -> None:
    """DB postprocess accepts numpy probability maps."""
    prob = np.zeros((1, 1, 32, 32), dtype=np.float32)
    prob[0, 0, 8:24, 8:24] = 0.9
    post = DBPostProcess(thresh=0.2, box_thresh=0.1, unclip_ratio=1.4)
    result = post({"maps": prob}, [(32.0, 32.0, 1.0, 1.0)])
    assert isinstance(result, list)
    assert "points" in result[0]


def test_ctc_decode_blank_produces_empty_string() -> None:
    """CTC decode returns empty text when all predictions are blank."""
    dict_path = Path(__file__).resolve().parent / "data" / "dict" / "ppocrv6_dict.txt"
    decoder = CTCLabelDecode(character_dict_path=dict_path)
    preds = np.zeros((1, 5, 100), dtype=np.float32)
    preds[:, :, 0] = 1.0
    text = decoder(preds)
    assert text[0][0] == ""


@pytest.mark.parametrize("variant", ("tiny", "small", "medium"))
def test_golden_files_exist(variant: str) -> None:
    """Committed golden tensors exist for every supported variant."""
    det_dir = GOLDEN_ROOT / variant / "det"
    rec_dir = GOLDEN_ROOT / variant / "rec"
    for name in ("preprocessed.npy", "prob_map.npy", "shape.npy"):
        assert (det_dir / name).is_file(), f"missing {det_dir / name}"
    for name in ("preprocessed.npy", "softmax.npy"):
        assert (rec_dir / name).is_file(), f"missing {rec_dir / name}"
    e2e = GOLDEN_ROOT / "e2e" / f"{variant}.json"
    assert e2e.is_file()
    payload = json.loads(e2e.read_text(encoding="utf-8"))
    assert payload["variant"] == variant
    assert "det_boxes" in payload
    assert "rec_text" in payload
