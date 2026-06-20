"""Upstream-aligned OCR result serialization and display."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from mlx_ocr.types import BoundingBox, OCRResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OCRTiming:
    """Elapsed seconds for pipeline stages."""

    det_s: float
    rec_s: float
    total_s: float

    def as_dict(self) -> dict[str, float]:
        """Return timing fields as a plain mapping."""
        return {"det": self.det_s, "rec": self.rec_s, "all": self.total_s}


def box_points_int32(box: BoundingBox) -> list[list[int]]:
    """Convert a bounding box to Paddle ``predict_system`` int32 points."""
    return [[int(round(x)), int(round(y))] for x, y in box.points]


def to_system_results_entries(result: OCRResult) -> list[dict[str, object]]:
    """Build Paddle ``predict_system`` JSON entries.

    Args:
        result: Filtered OCR output with aligned detections and recognitions.

    Returns:
        List of ``{"transcription": str, "points": [[x,y], ...]}`` dicts.
    """
    entries: list[dict[str, object]] = []
    for detection, recognition in zip(result.detections, result.recognitions, strict=True):
        entries.append(
            {
                "transcription": recognition.text,
                "points": box_points_int32(detection.box),
            }
        )
    return entries


def to_system_results_line(result: OCRResult, basename: str) -> str:
    """Format one ``system_results.txt`` line for Paddle parity.

    Args:
        result: OCR output for a single image.
        basename: Image file name without directory.

    Returns:
        TSV line ``{basename}\\t{json}\\n``.
    """
    payload = to_system_results_entries(result)
    return f"{basename}\t{json.dumps(payload, ensure_ascii=False)}\n"


def save_system_results(
    result: OCRResult,
    output_dir: Path,
    basename: str,
) -> Path:
    """Write ``system_results.txt`` in Paddle ``predict_system`` format.

    Args:
        result: OCR output for a single image.
        output_dir: Directory to create or reuse.
        basename: Image file name for the TSV key.

    Returns:
        Path to the written ``system_results.txt`` file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "system_results.txt"
    path.write_text(to_system_results_line(result, basename), encoding="utf-8")
    logger.info("Wrote %s", path)
    return path


def to_markdown(
    result: OCRResult,
    *,
    title: str | None = None,
    input_path: str | None = None,
) -> str:
    """Format OCR output as a Markdown document.

    Args:
        result: OCR output for a single image.
        title: Optional document title. Defaults to the input file name or
            ``OCR Result``.
        input_path: Optional source image path recorded below the heading.

    Returns:
        Markdown text containing recognized text and region metadata.
    """
    heading = title or (Path(input_path).name if input_path is not None else "OCR Result")
    lines = [f"# {heading}", ""]
    if input_path is not None:
        lines.extend((f"Source: `{input_path}`", ""))

    lines.extend(("## Text", ""))
    if not result.recognitions:
        lines.extend(("No text detected.", ""))
    else:
        for recognition in result.recognitions:
            lines.append(recognition.text)
        lines.append("")

    lines.extend(
        (
            "## Regions",
            "",
            "| # | Text | Score | Points |",
            "|---:|---|---:|---|",
        )
    )
    for index, (detection, recognition) in enumerate(
        zip(result.detections, result.recognitions, strict=True),
        start=1,
    ):
        text = recognition.text.replace("|", "\\|").replace("\n", "<br>")
        points = json.dumps(box_points_int32(detection.box), ensure_ascii=False)
        lines.append(f"| {index} | {text} | {recognition.score:.6f} | `{points}` |")
    lines.append("")
    return "\n".join(lines)


def save_to_markdown(
    result: OCRResult,
    save_path: Path,
    *,
    input_path: str | None = None,
) -> Path:
    """Save OCR output as Markdown.

    Args:
        result: OCR output for a single image.
        save_path: Target file or directory. Directories receive ``{stem}.md``.
        input_path: Optional source image path recorded in Markdown.

    Returns:
        Path to the written Markdown file.
    """
    if save_path.suffix != ".md":
        stem = Path(input_path or "result").stem
        save_path.mkdir(parents=True, exist_ok=True)
        markdown_path = save_path / f"{stem}.md"
    else:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path = save_path

    markdown_path.write_text(
        to_markdown(result, title=markdown_path.stem, input_path=input_path),
        encoding="utf-8",
    )
    logger.info("Wrote %s", markdown_path)
    return markdown_path


def to_paddlex_res(
    result: OCRResult,
    *,
    input_path: str | None = None,
    page_index: int | None = None,
    text_rec_score_thresh: float = 0.0,
) -> dict[str, object]:
    """Build a PaddleOCR 3.x ``res`` dict from filtered OCR output.

    Args:
        result: Filtered OCR output with aligned detections and recognitions.
        input_path: Optional source image path.
        page_index: Optional PDF page index.
        text_rec_score_thresh: Score threshold recorded in metadata.

    Returns:
        Mapping compatible with PaddleX ``save_to_json`` core fields.
    """
    rec_polys = [box_points_int32(detection.box) for detection in result.detections]
    rec_texts = [recognition.text for recognition in result.recognitions]
    rec_scores = [recognition.score for recognition in result.recognitions]
    rec_boxes = [
        [detection.box.x_min, detection.box.y_min, detection.box.x_max, detection.box.y_max]
        for detection in result.detections
    ]
    return {
        "input_path": input_path,
        "page_index": page_index,
        "dt_polys": rec_polys,
        "dt_scores": [detection.score for detection in result.detections],
        "rec_texts": rec_texts,
        "rec_scores": rec_scores,
        "rec_polys": rec_polys,
        "rec_boxes": rec_boxes,
        "text_rec_score_thresh": text_rec_score_thresh,
    }


def save_to_json(
    result: OCRResult,
    save_path: Path,
    *,
    input_path: str | None = None,
    page_index: int | None = None,
    text_rec_score_thresh: float = 0.0,
) -> Path:
    """Save OCR output as PaddleX-style JSON.

    Args:
        result: OCR output for a single image.
        save_path: Target file or directory. Directories receive
            ``{stem}_res.json``.
        input_path: Optional source image path stored in JSON.
        page_index: Optional PDF page index stored in JSON.
        text_rec_score_thresh: Score threshold recorded in metadata.

    Returns:
        Path to the written JSON file.
    """
    if save_path.suffix != ".json":
        stem = Path(input_path or "result").stem
        save_path.mkdir(parents=True, exist_ok=True)
        json_path = save_path / f"{stem}_res.json"
    else:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        json_path = save_path

    payload = {"res": to_paddlex_res(
        result,
        input_path=input_path,
        page_index=page_index,
        text_rec_score_thresh=text_rec_score_thresh,
    )}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %s", json_path)
    return json_path


def print_result(result: OCRResult) -> None:
    """Print OCR lines in Paddle ``predict_system`` debug style."""
    for _, recognition in zip(result.detections, result.recognitions, strict=True):
        print(f"{recognition.text}\t{recognition.score:.3f}")
