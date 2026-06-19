"""Frozen domain records for mlx-ocr inference outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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

    def to_system_results_entries(self) -> list[dict[str, object]]:
        """Return Paddle ``predict_system`` JSON entries."""
        from mlx_ocr.output import to_system_results_entries

        return to_system_results_entries(self)

    def to_system_results_line(self, basename: str) -> str:
        """Format one ``system_results.txt`` line."""
        from mlx_ocr.output import to_system_results_line

        return to_system_results_line(self, basename)

    def save_system_results(self, output_dir: Path, basename: str) -> Path:
        """Write ``system_results.txt`` under ``output_dir``."""
        from mlx_ocr.output import save_system_results

        return save_system_results(self, output_dir, basename)

    def to_paddlex_res(
        self,
        *,
        input_path: str | None = None,
        page_index: int | None = None,
        text_rec_score_thresh: float = 0.0,
    ) -> dict[str, object]:
        """Build a PaddleOCR 3.x ``res`` mapping."""
        from mlx_ocr.output import to_paddlex_res

        return to_paddlex_res(
            self,
            input_path=input_path,
            page_index=page_index,
            text_rec_score_thresh=text_rec_score_thresh,
        )

    def save_to_json(
        self,
        save_path: Path,
        *,
        input_path: str | None = None,
        page_index: int | None = None,
        text_rec_score_thresh: float = 0.0,
    ) -> Path:
        """Save PaddleX-style ``{stem}_res.json``."""
        from mlx_ocr.output import save_to_json

        return save_to_json(
            self,
            save_path,
            input_path=input_path,
            page_index=page_index,
            text_rec_score_thresh=text_rec_score_thresh,
        )

    def print(self) -> None:
        """Print text and score lines to stdout."""
        from mlx_ocr.output import print_result

        print_result(self)
