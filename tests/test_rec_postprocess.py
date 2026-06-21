"""Recognition CTC decode parity tests."""

from __future__ import annotations

import numpy as np

from mlx_ocr.postprocess.ctc import ctc_decode, load_character_dict
from tests.conftest import DICT_ROOT, GOLDEN_ROOT, load_golden_npy
from tests.reference.postprocess.ctc_decode import CTCLabelDecode


def test_ctc_decode_matches_reference_on_golden_softmax(variant: str) -> None:
    """MLX CTC decode matches vendored reference on golden softmax outputs."""
    softmax = load_golden_npy(GOLDEN_ROOT / variant / "rec" / "softmax.npy")
    dict_path = DICT_ROOT / ("ppocrv6_tiny_dict.txt" if variant == "tiny" else "ppocrv6_dict.txt")
    reference = CTCLabelDecode(character_dict_path=dict_path, use_space_char=True)(softmax)
    decoded = ctc_decode(softmax, load_character_dict(dict_path, use_space_char=True))
    assert len(decoded) == len(reference)
    for actual, (text, score) in zip(decoded, reference, strict=True):
        assert actual.text == text
        np.testing.assert_allclose(actual.score, score, rtol=1e-5, atol=1e-5)


def test_ctc_decode_blank_sequence() -> None:
    """An all-blank prediction decodes to empty text with zero confidence."""
    preds = np.zeros((1, 4, 8), dtype=np.float32)
    preds[:, :, 0] = 1.0
    dict_path = DICT_ROOT / "ppocrv6_dict.txt"
    decoded = ctc_decode(preds, load_character_dict(dict_path, use_space_char=True))
    assert decoded[0].text == ""
    assert decoded[0].score == 0.0
