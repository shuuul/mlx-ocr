# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Vendored from tools/infer/predict_rec.py — default resize_norm_img path."""

from __future__ import annotations

import math

import cv2
import numpy as np


def resize_norm_img(
    img: np.ndarray,
    max_wh_ratio: float,
    *,
    rec_image_shape: tuple[int, int, int] = (3, 48, 320),
) -> np.ndarray:
    """Resize, normalize, and pad a recognition crop for PP-OCRv6 CTC models.

    Args:
        img: BGR crop with shape ``[H, W, 3]``.
        max_wh_ratio: Maximum width-to-height ratio in the current batch.
        rec_image_shape: ``(C, H, W)`` target shape before dynamic width scaling.

    Returns:
        Normalized CHW tensor padded to ``rec_image_shape`` width.
    """
    imgC, imgH, imgW = rec_image_shape
    if imgC != img.shape[2]:
        raise ValueError(f"expected {imgC} channels, got {img.shape[2]}")

    imgW = int(imgH * max_wh_ratio)
    h, w = img.shape[:2]
    ratio = w / float(h)
    resized_w = min(math.ceil(imgH * ratio), imgW)

    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype("float32")
    resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im
