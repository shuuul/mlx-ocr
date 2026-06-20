"""Detection preprocessing aligned with PaddleOCR inference.yml."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import mlx.core as mx
import numpy as np

_NORMALIZE_SCALE = np.array(
    [1.0 / (255.0 * 0.229), 1.0 / (255.0 * 0.224), 1.0 / (255.0 * 0.225)],
    dtype=np.float32,
)
_NORMALIZE_BIAS = np.array(
    [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    dtype=np.float32,
)
_NORMALIZE_SCALE_MX = mx.array(_NORMALIZE_SCALE)
_NORMALIZE_BIAS_MX = mx.array(_NORMALIZE_BIAS)


@dataclass(frozen=True)
class DetPreprocessResult:
    """Detection preprocess output."""

    image: mx.array
    shape: tuple[float, float, float, float]

    @property
    def nchw(self) -> mx.array:
        """Return the image tensor in NCHW layout."""
        return mx.transpose(self.image, (0, 3, 1, 2))


def _resize_for_test(
    image: np.ndarray,
    *,
    limit_side_len: int,
    limit_type: str,
    max_side_limit: int = 4000,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    src_h, src_w = image.shape[:2]
    if limit_type == "max":
        ratio = (
            float(limit_side_len) / src_h
            if max(src_h, src_w) > limit_side_len and src_h >= src_w
            else float(limit_side_len) / src_w
            if max(src_h, src_w) > limit_side_len
            else 1.0
        )
    elif limit_type == "min":
        ratio = (
            float(limit_side_len) / src_h
            if min(src_h, src_w) < limit_side_len and src_h <= src_w
            else float(limit_side_len) / src_w
            if min(src_h, src_w) < limit_side_len
            else 1.0
        )
    else:
        raise ValueError(f"unsupported limit_type: {limit_type!r}")

    resize_h = int(src_h * ratio)
    resize_w = int(src_w * ratio)
    if max(resize_h, resize_w) > max_side_limit:
        scale = float(max_side_limit) / max(resize_h, resize_w)
        resize_h = int(resize_h * scale)
        resize_w = int(resize_w * scale)

    resize_h = max(int(round(resize_h / 32) * 32), 32)
    resize_w = max(int(round(resize_w / 32) * 32), 32)
    resized = cv2.resize(image, (resize_w, resize_h))
    ratio_h = resize_h / float(src_h)
    ratio_w = resize_w / float(src_w)
    return resized, (float(src_h), float(src_w), ratio_h, ratio_w)


def _normalize_image(image: np.ndarray) -> np.ndarray:
    normalized = image.astype(np.float32)
    cv2.multiply(normalized, _NORMALIZE_SCALE, dst=normalized)
    cv2.add(normalized, _NORMALIZE_BIAS, dst=normalized)
    return normalized


def normalize_det_image_mlx(image: mx.array) -> mx.array:
    """Normalize a resized BGR image with MLX operations.

    Args:
        image: Resized BGR image in HWC layout.

    Returns:
        Normalized NHWC batch tensor ready for detection inference.

    Raises:
        ValueError: If ``image`` is not a 3-channel HWC tensor.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected resized BGR image [H, W, 3], got shape {image.shape}")
    normalized = image.astype(mx.float32) * _NORMALIZE_SCALE_MX + _NORMALIZE_BIAS_MX
    return mx.expand_dims(normalized, axis=0)


def resize_det_image(
    image: np.ndarray,
    *,
    limit_side_len: int = 960,
    limit_type: str = "min",
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Resize a BGR image for PP-OCRv6 detection.

    Args:
        image: Source image in BGR uint8 layout ``[H, W, 3]``.
        limit_side_len: Maximum/minimum side length before 32-pixel padding.
        limit_type: ``min`` or ``max`` limit behavior from Paddle configs.

    Returns:
        Resized image and source-shape metadata.

    Raises:
        ValueError: If ``image`` is not a 3-channel BGR array.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected BGR image [H, W, 3], got shape {image.shape}")

    if sum(image.shape[:2]) < 64:
        pad_h = max(32, image.shape[0])
        pad_w = max(32, image.shape[1])
        padded = np.zeros((pad_h, pad_w, 3), dtype=image.dtype)
        padded[: image.shape[0], : image.shape[1]] = image
        image = padded

    return _resize_for_test(
        image,
        limit_side_len=limit_side_len,
        limit_type=limit_type,
    )


def det_preprocess(
    image: np.ndarray,
    *,
    limit_side_len: int = 960,
    limit_type: str = "min",
) -> DetPreprocessResult:
    """Preprocess a BGR image for PP-OCRv6 detection.

    Args:
        image: Source image in BGR uint8 layout ``[H, W, 3]``.
        limit_side_len: Maximum/minimum side length before 32-pixel padding.
        limit_type: ``min`` or ``max`` limit behavior from Paddle configs.

    Returns:
        Normalized NHWC batch tensor and source-shape metadata.

    Raises:
        ValueError: If ``image`` is not a 3-channel BGR array.
    """
    resized, (src_h, src_w, ratio_h, ratio_w) = resize_det_image(
        image,
        limit_side_len=limit_side_len,
        limit_type=limit_type,
    )
    normalized = _normalize_image(resized)
    chw = np.transpose(normalized, (2, 0, 1))
    batch = np.expand_dims(chw, axis=0)
    nhwc = np.transpose(batch, (0, 2, 3, 1))
    shape_arr = np.array([src_h, src_w, ratio_h, ratio_w], dtype=np.float32)
    return DetPreprocessResult(
        image=mx.array(nhwc),
        shape=(
            float(shape_arr[0]),
            float(shape_arr[1]),
            float(shape_arr[2]),
            float(shape_arr[3]),
        ),
    )


def nchw_to_nhwc(tensor: mx.array | np.ndarray) -> mx.array:
    """Convert a batch tensor from NCHW to NHWC."""
    array = mx.array(tensor) if isinstance(tensor, np.ndarray) else tensor
    if array.ndim != 4:
        raise ValueError(f"expected rank-4 tensor, got shape {array.shape}")
    if array.shape[1] == 3:
        return mx.transpose(array, (0, 2, 3, 1))
    return array


def nhwc_prob_to_nchw(prob_map: mx.array) -> np.ndarray:
    """Convert an NHWC probability map batch to Paddle NCHW numpy layout."""
    if prob_map.ndim != 4:
        raise ValueError(f"expected rank-4 probability map, got {prob_map.shape}")
    nchw = mx.transpose(prob_map, (0, 3, 1, 2))
    return np.asarray(nchw, dtype=np.float32)
