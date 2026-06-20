"""Typer command line interface for PP-OCRv6 inference."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated, Literal, cast

import numpy as np
import typer
from PIL import Image

from mlx_ocr.hub.rec_weight_patch import RecognitionWeightSource
from mlx_ocr.hub.registry import ModelVariant
from mlx_ocr.output import to_system_results_line
from mlx_ocr.pipeline import PP_OCRv6

logger = logging.getLogger(__name__)

OutputFormat = Literal["text", "system", "json", "markdown"]
IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Run PP-OCRv6 OCR with MLX.",
)


def read_bgr_image(path: Path) -> np.ndarray:
    """Read an image file as a BGR uint8 array.

    Args:
        path: Existing image file path.

    Returns:
        Image data in OpenCV-compatible BGR channel order.

    Raises:
        FileNotFoundError: If ``path`` is missing.
        ValueError: If Pillow cannot decode the image.
    """
    if not path.is_file():
        raise FileNotFoundError(f"missing image: {path}")

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.uint8)
    return array[:, :, ::-1].copy()


def expand_image_paths(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    """Expand input files and directories to sorted image files.

    Args:
        paths: User-provided ``--path`` values.

    Returns:
        Tuple of image file paths.

    Raises:
        typer.BadParameter: If an input path is missing, unsupported, or no
            image files are found.
    """
    if not paths:
        raise typer.BadParameter("provide at least one --path/-p value")

    image_paths: list[Path] = []
    for path in paths:
        if not path.exists():
            raise typer.BadParameter(f"missing input path: {path}")
        if path.is_file():
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                raise typer.BadParameter(f"unsupported image file: {path}")
            image_paths.append(path)
            continue
        if path.is_dir():
            image_paths.extend(
                file_path
                for file_path in sorted(path.iterdir())
                if file_path.is_file() and file_path.suffix.lower() in IMAGE_SUFFIXES
            )
            continue
        raise typer.BadParameter(f"unsupported input path: {path}")

    if not image_paths:
        raise typer.BadParameter("no supported image files found")
    return tuple(image_paths)


def resolve_formats(values: list[str] | None, *, quiet: bool) -> tuple[OutputFormat, ...]:
    """Resolve requested output formats.

    Args:
        values: Raw repeated ``--format`` values.
        quiet: Whether default console text should be suppressed.

    Returns:
        Ordered unique output formats.
    """
    default: tuple[OutputFormat, ...] = ("system", "json", "markdown") if quiet else (
        "text",
        "system",
        "json",
        "markdown",
    )
    raw_values = list(default if values is None else cast(list[OutputFormat], values))
    formats: list[OutputFormat] = []
    for value in raw_values:
        if value not in formats:
            formats.append(value)
    return tuple(formats)


def run_ocr(
    *,
    paths: tuple[Path, ...],
    output_dir: Path,
    formats: tuple[OutputFormat, ...],
    variant: ModelVariant,
    drop_score: float,
    rec_weight_source: RecognitionWeightSource,
    compile_models: bool,
    quiet: bool,
) -> None:
    """Run OCR and write selected outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ocr = PP_OCRv6.from_hub(
        variant,
        drop_score=drop_score,
        rec_weight_source=rec_weight_source,
        compile_models=compile_models,
    )

    system_lines: list[str] = []
    try:
        for image_path in paths:
            pipeline_result = ocr.predict(read_bgr_image(image_path))
            result = pipeline_result.result
            timing = pipeline_result.timing
            input_path = str(image_path.resolve())

            if "text" in formats:
                typer.echo(f"# {image_path.name}")
                result.print()
                typer.echo(
                    "timing: "
                    f"det={timing.det_s:.3f}s rec={timing.rec_s:.3f}s total={timing.total_s:.3f}s"
                )

            if "system" in formats:
                system_lines.append(to_system_results_line(result, image_path.name))
            if "json" in formats:
                result.save_to_json(output_dir, input_path=input_path)
            if "markdown" in formats:
                result.save_to_markdown(output_dir, input_path=input_path)
    finally:
        ocr.close()

    if system_lines:
        system_path = output_dir / "system_results.txt"
        system_path.write_text("".join(system_lines), encoding="utf-8")
        logger.info("Wrote %s", system_path)

    if not quiet:
        typer.echo(f"Wrote outputs under {output_dir}")


@app.command()
def ocr(
    path: Annotated[
        list[Path] | None,
        typer.Option(
            "--path",
            "-p",
            help="Input image file or directory. Repeat to process multiple inputs.",
        ),
    ] = None,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory for selected file formats."),
    ] = Path("ocr-output"),
    variant: Annotated[
        str,
        typer.Option(
            "--variant",
            help="PP-OCRv6 model size tier. Defaults to PaddleOCR medium.",
        ),
    ] = "medium",
    output_format: Annotated[
        list[str] | None,
        typer.Option(
            "--format",
            "-f",
            help="Output format: text, system, json, or markdown. Repeat to select multiple.",
        ),
    ] = None,
    drop_score: Annotated[
        float,
        typer.Option(
            "--drop-score",
            help="Minimum recognition score to keep a detected text line.",
        ),
    ] = 0.5,
    rec_weight_source: Annotated[
        str,
        typer.Option(
            "--rec-weight-source",
            help="Recognition weights: auto, hub, or paddle_pretrained.",
        ),
    ] = "auto",
    no_compile: Annotated[
        bool,
        typer.Option("--no-compile", help="Disable MLX compiled model functions."),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", help="Suppress default console text output."),
    ] = False,
) -> None:
    """Run OCR on image files and write PaddleOCR-style outputs."""
    logging.basicConfig(level=logging.WARNING if quiet else logging.INFO, format="%(message)s")

    valid_variants = {"tiny", "small", "medium"}
    if variant not in valid_variants:
        raise typer.BadParameter(f"variant must be one of: {', '.join(sorted(valid_variants))}")

    valid_weight_sources = {"auto", "hub", "paddle_pretrained"}
    if rec_weight_source not in valid_weight_sources:
        raise typer.BadParameter(
            f"rec_weight_source must be one of: {', '.join(sorted(valid_weight_sources))}"
        )

    valid_formats = {"text", "system", "json", "markdown"}
    if output_format is not None:
        for value in output_format:
            if value not in valid_formats:
                message = f"format must be one of: {', '.join(sorted(valid_formats))}"
                raise typer.BadParameter(message)

    run_ocr(
        paths=expand_image_paths(tuple(path or ())),
        output_dir=output,
        formats=resolve_formats(output_format, quiet=quiet),
        variant=cast(ModelVariant, variant),
        drop_score=drop_score,
        rec_weight_source=cast(RecognitionWeightSource, rec_weight_source),
        compile_models=not no_compile,
        quiet=quiet,
    )


def main() -> None:
    """Run the Typer application."""
    app()


if __name__ == "__main__":
    sys.exit(main())
