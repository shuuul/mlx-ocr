"""Frozen domain records for mlx4ocr inference outputs."""

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
class OCRTextBlock:
    """Recognized text block in an OCR result."""

    text: str
    box: BoundingBox | None = None
    detection_score: float | None = None
    recognition_score: float | None = None


@dataclass(frozen=True)
class OCRResult:
    """End-to-end OCR output for one image."""

    blocks: tuple[OCRTextBlock, ...]
    text: str
    engine: str
    model: str | None = None
    prompt: str | None = None

    @classmethod
    def from_ppocrv6(
        cls,
        detections: tuple[TextDetection, ...],
        recognitions: tuple[TextRecognition, ...],
        *,
        model: str | None = None,
    ) -> OCRResult:
        """Build a PP-OCRv6 result from aligned detection and recognition tuples.

        Args:
            detections: Filtered text detections in reading order.
            recognitions: Filtered text recognitions aligned with ``detections``.
            model: Optional model variant or identifier.

        Returns:
            Generalized block-based OCR result.
        """
        blocks = tuple(
            OCRTextBlock(
                text=recognition.text,
                box=detection.box,
                detection_score=detection.score,
                recognition_score=recognition.score,
            )
            for detection, recognition in zip(detections, recognitions, strict=True)
        )
        return cls(
            blocks=blocks,
            text="\n".join(block.text for block in blocks),
            engine="ppocrv6",
            model=model,
        )
