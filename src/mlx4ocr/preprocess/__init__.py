"""Image preprocessing aligned with PaddleOCR inference.yml."""

from mlx4ocr.preprocess.det import (
    DetPreprocessResult,
    det_preprocess,
    nchw_to_nhwc,
    nhwc_prob_to_nchw,
)
from mlx4ocr.preprocess.rec import (
    RecPreprocessResult,
    rec_preprocess,
    resize_norm_img,
)

__all__ = [
    "DetPreprocessResult",
    "RecPreprocessResult",
    "det_preprocess",
    "nchw_to_nhwc",
    "nhwc_prob_to_nchw",
    "rec_preprocess",
    "resize_norm_img",
]
