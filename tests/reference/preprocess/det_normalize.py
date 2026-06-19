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
"""Vendored from ppocr/data/imaug/operators.py — NormalizeImage and ToCHWImage."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


class NormalizeImage:
    """Normalize image by subtracting mean and dividing by std."""

    def __init__(
        self,
        scale: str | float | None = None,
        mean: list[float] | None = None,
        std: list[float] | None = None,
        order: str = "chw",
        **kwargs: Any,
    ) -> None:
        del kwargs
        if isinstance(scale, str):
            scale = 1.0 / 255.0 if scale == "1./255." else float(scale)
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if order == "chw" else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype("float32")
        self.std = np.array(std).reshape(shape).astype("float32")

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        img = data["image"]
        if isinstance(img, Image.Image):
            img = np.array(img)
        if not isinstance(img, np.ndarray):
            raise TypeError("invalid input 'img' in NormalizeImage")
        data["image"] = (img.astype("float32") * self.scale - self.mean) / self.std
        return data


class ToCHWImage:
    """Convert HWC image layout to CHW."""

    def __init__(self, **kwargs: Any) -> None:
        del kwargs

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        img = data["image"]
        if isinstance(img, Image.Image):
            img = np.array(img)
        data["image"] = img.transpose((2, 0, 1))
        return data
