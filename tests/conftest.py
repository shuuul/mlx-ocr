"""Pytest fixtures for vendored reference code and golden tensors."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.reference.preprocess.det_normalize import NormalizeImage, ToCHWImage
from tests.reference.preprocess.det_resize import DetResizeForTest

TESTS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TESTS_ROOT.parent
EXAMPLES_ROOT = REPO_ROOT / "examples"
DATA_ROOT = TESTS_ROOT / "data"
REFERENCE_ROOT = TESTS_ROOT / "reference"
GOLDEN_ROOT = DATA_ROOT / "golden"
IMAGES_ROOT = EXAMPLES_ROOT / "images"
DICT_ROOT = DATA_ROOT / "dict"

VARIANTS = ("tiny", "small", "medium")


@pytest.fixture(scope="session")
def sample_bgr_image() -> np.ndarray:
    """Load the canonical OCR test image as BGR uint8."""
    image_path = IMAGES_ROOT / "sample_doc.jpg"
    if not image_path.is_file():
        raise FileNotFoundError(f"missing test image: {image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to decode test image: {image_path}")
    return image


@pytest.fixture
def det_preprocessed(
    sample_bgr_image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run vendored detection preprocess on the sample image."""
    resize = DetResizeForTest(limit_side_len=960, limit_type="min")
    normalize = NormalizeImage(
        scale="1./255.",
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        order="hwc",
    )
    to_chw = ToCHWImage()
    data: dict[str, object] = {"image": sample_bgr_image.copy()}
    data = resize(data)
    shape = np.asarray(data["shape"], dtype=np.float32)
    data = normalize(data)
    data = to_chw(data)
    image = np.asarray(data["image"], dtype=np.float32)
    tensor = np.expand_dims(image, axis=0)
    return tensor, shape


def load_golden_npy(path: Path) -> np.ndarray:
    """Load a golden ``.npy`` tensor from disk.

    Args:
        path: Absolute or relative path to the array file.

    Returns:
        Loaded float32 array.

    Raises:
        FileNotFoundError: If the golden file does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"missing golden file: {path}")
    return np.load(path)


@pytest.fixture(params=VARIANTS)
def variant(request: pytest.FixtureRequest) -> Iterator[str]:
    """Parametrize tests over PP-OCRv6 model variants."""
    yield request.param
