"""Frozen domain records for mlx-ocr inference outputs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned quadrilateral in image coordinates.

    Points are ordered clockwise starting from the top-left corner.
    """

    points: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]

    @property
    def x_min(self) -> float:
        """Minimum x coordinate across all corners."""
        return min(point[0] for point in self.points)

    @property
    def y_min(self) -> float:
        """Minimum y coordinate across all corners."""
        return min(point[1] for point in self.points)

    @property
    def x_max(self) -> float:
        """Maximum x coordinate across all corners."""
        return max(point[0] for point in self.points)

    @property
    def y_max(self) -> float:
        """Maximum y coordinate across all corners."""
        return max(point[1] for point in self.points)


@dataclass(frozen=True)
class TextDetection:
    """Single detected text region."""

    box: BoundingBox
    score: float


@dataclass(frozen=True)
class TextRecognition:
    """Recognized text for one cropped region."""

    text: str
    score: float


@dataclass(frozen=True)
class OCRResult:
    """End-to-end OCR output for one image."""

    detections: tuple[TextDetection, ...]
    recognitions: tuple[TextRecognition, ...]
