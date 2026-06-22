"""Recognition preprocessing aligned with PaddleOCR predict_rec.py."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import mlx.core as mx
import numpy as np


@dataclass(frozen=True)
class RecPreprocessResult:
    """Recognition preprocess output."""

    image: mx.array

    @property
    def nchw(self) -> mx.array:
        """Return the image tensor in NCHW layout."""
        return mx.transpose(self.image, (0, 3, 1, 2))


def resize_norm_img(
    image: np.ndarray,
    max_wh_ratio: float,
    *,
    rec_image_shape: tuple[int, int, int] = (3, 48, 320),
) -> np.ndarray:
    """Resize, normalize, and pad a recognition crop.

    Args:
        image: BGR crop with shape ``[H, W, 3]``.
        max_wh_ratio: Maximum width-to-height ratio in the current batch.
        rec_image_shape: ``(C, H, W)`` target shape before dynamic width scaling.

    Returns:
        Normalized CHW tensor padded to the computed target width.
    """
    img_c, img_h, img_w = rec_image_shape
    img_w = int(img_h * max_wh_ratio)
    height, width = image.shape[:2]
    ratio = width / float(height)
    resized_w = min(math.ceil(img_h * ratio), img_w)

    resized = cv2.resize(image, (resized_w, img_h))
    normalized = resized.astype(np.float32).transpose((2, 0, 1)) / 255.0
    normalized = (normalized - 0.5) / 0.5
    padding = np.zeros((img_c, img_h, img_w), dtype=np.float32)
    padding[:, :, :resized_w] = normalized
    return padding


def rec_preprocess(
    image: np.ndarray,
    *,
    max_wh_ratio: float | None = None,
    rec_image_shape: tuple[int, int, int] = (3, 48, 320),
) -> RecPreprocessResult:
    """Preprocess a BGR crop for PP-OCRv6 recognition.

    Args:
        image: Source crop in BGR uint8 layout ``[H, W, 3]``.
        max_wh_ratio: Optional batch max width/height ratio. Defaults to the crop ratio.
        rec_image_shape: ``(C, H, W)`` base shape from inference configs.

    Returns:
        Normalized NHWC batch tensor ready for MLX inference.

    Raises:
        ValueError: If ``image`` is not a 3-channel BGR array.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected BGR image [H, W, 3], got shape {image.shape}")

    height, width = image.shape[:2]
    ratio = max_wh_ratio if max_wh_ratio is not None else width / float(height)
    chw = resize_norm_img(image, ratio, rec_image_shape=rec_image_shape)
    batch = np.expand_dims(chw, axis=0)
    nhwc = np.transpose(batch, (0, 2, 3, 1))
    return RecPreprocessResult(image=mx.array(nhwc))
