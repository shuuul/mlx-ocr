"""Image preprocessing aligned with PaddleOCR inference.yml."""

from mlx_ocr.preprocess.det import (
    DetPreprocessResult,
    det_preprocess,
    nchw_to_nhwc,
    nhwc_prob_to_nchw,
)
from mlx_ocr.preprocess.rec import (
    RecPreprocessResult,
    rec_preprocess,
    rec_preprocess_crop_from_image,
    resize_norm_img,
)

__all__ = [
    "DetPreprocessResult",
    "RecPreprocessResult",
    "det_preprocess",
    "nchw_to_nhwc",
    "nhwc_prob_to_nchw",
    "rec_preprocess",
    "rec_preprocess_crop_from_image",
    "resize_norm_img",
]
