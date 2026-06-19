"""Run mlx-ocr pipeline on committed PaddleOCR example images."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from mlx_ocr import PP_OCRv6

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = REPO_ROOT / "examples"
EXAMPLE_IMAGES = EXAMPLES_ROOT / "images"

IMG_10 = EXAMPLE_IMAGES / "img_10.jpg"
SAMPLE_DOC = EXAMPLE_IMAGES / "sample_doc.jpg"
GENERAL_OCR = EXAMPLE_IMAGES / "general_ocr_002.png"


@pytest.fixture(scope="session")
def img_10_bgr() -> np.ndarray:
    """Load the Paddle det config demo image."""
    image = cv2.imread(str(IMG_10), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"missing example image: {IMG_10}")
    return image


@pytest.mark.parametrize("variant", ("tiny", "small", "medium"))
def test_img_10_recognizes_english_lines(img_10_bgr: np.ndarray, variant: str) -> None:
    """PP-OCRv6 det demo image yields the three English notice lines."""
    result = PP_OCRv6.from_hub(variant).predict(img_10_bgr).result
    texts = [recognition.text for recognition in result.recognitions]
    assert len(texts) == 3
    assert "Please lower your volume" in texts[0]
    assert "when you" in texts[1] and "pass by" in texts[1]
    assert "residential areas" in texts[2]


@pytest.mark.parametrize("variant", ("tiny", "small", "medium"))
def test_sample_doc_example_matches_golden_e2e(variant: str) -> None:
    """Synthetic doc example keeps four detected lines for every variant."""
    image = cv2.imread(str(SAMPLE_DOC), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"missing example image: {SAMPLE_DOC}")
    result = PP_OCRv6.from_hub(variant, drop_score=0.0).predict(image).result
    assert len(result.detections) == 4
    assert len(result.recognitions) == 4


def test_general_ocr_example_runs() -> None:
    """Official PaddleX boarding-pass demo image runs without error."""
    image = cv2.imread(str(GENERAL_OCR), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"missing example image: {GENERAL_OCR}")
    pipeline_result = PP_OCRv6.from_hub("medium", drop_score=0.0).predict(image)
    assert len(pipeline_result.result.detections) > 0
    assert pipeline_result.timing.total_s > 0.0
