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
"""Vendored from ppocr/data/imaug/operators.py — DetResizeForTest (numpy only)."""

from __future__ import annotations

import math
import sys
from typing import Any

import cv2
import numpy as np


class DetResizeForTest:
    """Resize detection input to a 32-pixel grid for PP-OCR DB models."""

    def __init__(self, **kwargs: Any) -> None:
        self.resize_type = 0
        self.keep_ratio = False
        self.max_side_limit = kwargs.get("max_side_limit", 4000)
        if "image_shape" in kwargs:
            self.image_shape = kwargs["image_shape"]
            self.resize_type = 1
            if "keep_ratio" in kwargs:
                self.keep_ratio = kwargs["keep_ratio"]
        elif "limit_side_len" in kwargs:
            self.limit_side_len = kwargs["limit_side_len"]
            self.limit_type = kwargs.get("limit_type", "min")
        elif "resize_long" in kwargs:
            self.resize_type = 2
            self.resize_long = kwargs.get("resize_long", 960)
        else:
            self.limit_side_len = 736
            self.limit_type = "min"

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """Resize ``data["image"]`` and attach ``data["shape"]`` metadata."""
        img = data["image"]
        src_h, src_w, _ = img.shape
        if sum([src_h, src_w]) < 64:
            img = self.image_padding(img)

        if self.resize_type == 0:
            img, [ratio_h, ratio_w] = self.resize_image_type0(img)
        elif self.resize_type == 2:
            img, [ratio_h, ratio_w] = self.resize_image_type2(img)
        else:
            img, [ratio_h, ratio_w] = self.resize_image_type1(img)
        data["image"] = img
        data["shape"] = np.array([src_h, src_w, ratio_h, ratio_w])
        return data

    def image_padding(self, im: np.ndarray, value: int = 0) -> np.ndarray:
        h, w, c = im.shape
        im_pad = np.zeros((max(32, h), max(32, w), c), np.uint8) + value
        im_pad[:h, :w, :] = im
        return im_pad

    def resize_image_type1(self, img: np.ndarray) -> tuple[np.ndarray, list[float]]:
        resize_h, resize_w = self.image_shape
        ori_h, ori_w = img.shape[:2]
        if self.keep_ratio is True:
            resize_w = ori_w * resize_h / ori_h
            n = math.ceil(resize_w / 32)
            resize_w = n * 32
        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        return img, [ratio_h, ratio_w]

    def resize_image_type0(self, img: np.ndarray) -> tuple[np.ndarray, list[float]]:
        limit_side_len = self.limit_side_len
        h, w, _c = img.shape

        if self.limit_type == "max":
            if max(h, w) > limit_side_len:
                ratio = float(limit_side_len) / h if h > w else float(limit_side_len) / w
            else:
                ratio = 1.0
        elif self.limit_type == "min":
            if min(h, w) < limit_side_len:
                ratio = float(limit_side_len) / h if h < w else float(limit_side_len) / w
            else:
                ratio = 1.0
        elif self.limit_type == "resize_long":
            ratio = float(limit_side_len) / max(h, w)
        else:
            raise ValueError(f"unsupported limit type: {self.limit_type!r}")

        resize_h = int(h * ratio)
        resize_w = int(w * ratio)
        if max(resize_h, resize_w) > self.max_side_limit:
            ratio = float(self.max_side_limit) / max(resize_h, resize_w)
            resize_h, resize_w = int(resize_h * ratio), int(resize_w * ratio)

        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)

        if int(resize_w) <= 0 or int(resize_h) <= 0:
            raise ValueError(f"invalid resize dimensions: {(resize_h, resize_w)}")

        try:
            img = cv2.resize(img, (int(resize_w), int(resize_h)))
        except Exception:
            print(img.shape, resize_w, resize_h)
            sys.exit(0)
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return img, [ratio_h, ratio_w]

    def resize_image_type2(self, img: np.ndarray) -> tuple[np.ndarray, list[float]]:
        h, w, _ = img.shape
        resize_w = w
        resize_h = h

        if resize_h > resize_w:
            ratio = float(self.resize_long) / resize_h
        else:
            ratio = float(self.resize_long) / resize_w

        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        max_stride = 128
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return img, [ratio_h, ratio_w]
