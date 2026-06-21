"""Typer command line interface for PP-OCRv6 inference."""

from __future__ import annotations

import logging
import shutil
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, cast

import numpy as np
import typer
from PIL import Image

from mlx_ocr.hub.rec_weight_patch import RecognitionWeightSource
from mlx_ocr.hub.registry import ModelVariant
from mlx_ocr.output import print_result, save_to_json, to_markdown, to_system_results_line
from mlx_ocr.pipeline import PP_OCRv6

logger = logging.getLogger(__name__)

OutputFormat = Literal["text", "system", "json", "markdown"]
IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
PDF_SUFFIX = ".pdf"
SUPPORTED_SUFFIXES = IMAGE_SUFFIXES | frozenset({PDF_SUFFIX})

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Run PP-OCRv6 OCR with MLX.",
)


@dataclass(frozen=True)
class InputDocument:
    """User-provided input document."""

    path: Path
    stem: str
    is_pdf: bool


@dataclass(frozen=True)
class RenderedPage:
    """Rendered page or image ready for OCR."""

    image: np.ndarray
    input_path: str
    output_name: str
    page_index: int | None


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


def collect_input_documents(paths: tuple[Path, ...]) -> tuple[InputDocument, ...]:
    """Expand input files and directories to supported documents.

    Args:
        paths: User-provided ``--path`` values.

    Returns:
        Tuple of image and PDF input documents.

    Raises:
        typer.BadParameter: If an input path is missing, unsupported, or no
            supported documents are found.
    """
    if not paths:
        raise typer.BadParameter("provide at least one --path/-p value")

    document_paths: list[Path] = []
    for path in paths:
        if not path.exists():
            raise typer.BadParameter(f"missing input path: {path}")
        if path.is_file():
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                raise typer.BadParameter(f"unsupported input file: {path}")
            document_paths.append(path)
            continue
        if path.is_dir():
            document_paths.extend(
                file_path
                for file_path in sorted(path.iterdir())
                if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_SUFFIXES
            )
            continue
        raise typer.BadParameter(f"unsupported input path: {path}")

    if not document_paths:
        raise typer.BadParameter("no supported image or PDF files found")
    return tuple(
        InputDocument(path=path, stem=path.stem, is_pdf=path.suffix.lower() == PDF_SUFFIX)
        for path in document_paths
    )


def iter_rendered_pages(
    document: InputDocument,
    *,
    start: int | None,
    end: int | None,
) -> Iterator[RenderedPage]:
    """Yield rendered OCR inputs for an image or PDF document.

    Args:
        document: Input image or PDF document.
        start: Optional 0-based first PDF page to process.
        end: Optional 0-based last PDF page to process, inclusive.

    Yields:
        Rendered OCR inputs in document order.
    """
    if document.is_pdf:
        yield from render_pdf_pages(document.path, start=start, end=end)
        return
    if start is not None or end is not None:
        raise typer.BadParameter("--start/--end can only be used with PDF inputs")
    yield RenderedPage(
        image=read_bgr_image(document.path),
        input_path=str(document.path.resolve()),
        output_name=document.path.name,
        page_index=None,
    )


def render_pdf_pages(path: Path, *, start: int | None, end: int | None) -> Iterator[RenderedPage]:
    """Render PDF pages to BGR arrays.

    Args:
        path: PDF file path.
        start: Optional 0-based first page.
        end: Optional 0-based last page, inclusive.

    Yields:
        Rendered PDF pages in page order.

    Raises:
        typer.BadParameter: If the page range is invalid.
        RuntimeError: If the PDF rendering dependency is unavailable.
    """
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("Install PDF support with `uv sync` or `pip install pymupdf`.") from exc

    with fitz.open(path) as document:
        page_count = document.page_count
        if page_count == 0:
            raise typer.BadParameter(f"PDF has no pages: {path}")
        first_page = 0 if start is None else start
        last_page = page_count - 1 if end is None else end
        if first_page < 0 or last_page < first_page or last_page >= page_count:
            raise typer.BadParameter(
                f"invalid page range {first_page}-{last_page} for {path} with {page_count} pages"
            )

        matrix = fitz.Matrix(2.0, 2.0)
        for page_index in range(first_page, last_page + 1):
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                pixmap.height,
                pixmap.width,
                pixmap.n,
            )
            yield RenderedPage(
                image=rgb[:, :, :3][:, :, ::-1].copy(),
                input_path=f"{path.resolve()}#page={page_index}",
                output_name=f"{path.stem}_page_{page_index + 1:04d}",
                page_index=page_index,
            )


def resolve_formats(values: list[str] | None, *, quiet: bool) -> tuple[OutputFormat, ...]:
    """Resolve requested output formats.

    Args:
        values: Raw repeated ``--format`` values.
        quiet: Whether default console text should be suppressed.

    Returns:
        Ordered unique output formats.
    """
    default: tuple[OutputFormat, ...] = (
        ("system", "json", "markdown")
        if quiet
        else (
            "text",
            "system",
            "json",
            "markdown",
        )
    )
    raw_values = list(default if values is None else cast(list[OutputFormat], values))
    formats: list[OutputFormat] = []
    for value in raw_values:
        if value not in formats:
            formats.append(value)
    return tuple(formats)


def run_ocr(
    *,
    documents: tuple[InputDocument, ...],
    output_dir: Path,
    formats: tuple[OutputFormat, ...],
    variant: ModelVariant,
    drop_score: float,
    rec_weight_source: RecognitionWeightSource,
    compile_models: bool,
    quiet: bool,
    start: int | None,
    end: int | None,
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
        for document in documents:
            parse_dir = output_dir / document.stem / "ocr"
            parse_dir.mkdir(parents=True, exist_ok=True)
            if document.is_pdf:
                shutil.copyfile(document.path, parse_dir / f"{document.stem}_origin.pdf")

            markdown_pages: list[str] = []
            for rendered in iter_rendered_pages(document, start=start, end=end):
                pipeline_result = ocr.predict(rendered.image)
                result = pipeline_result.result
                timing = pipeline_result.timing

                if "text" in formats:
                    typer.echo(f"# {rendered.output_name}")
                    print_result(result)
                    typer.echo(
                        "timing: "
                        f"det={timing.det_s:.3f}s rec={timing.rec_s:.3f}s "
                        f"total={timing.total_s:.3f}s"
                    )

                if "system" in formats:
                    system_lines.append(to_system_results_line(result, rendered.output_name))
                if "json" in formats:
                    json_path = parse_dir / f"{rendered.output_name}_res.json"
                    save_to_json(
                        result,
                        json_path,
                        input_path=rendered.input_path,
                        page_index=rendered.page_index,
                    )
                if "markdown" in formats:
                    markdown_pages.append(to_markdown(result).strip())

            if "markdown" in formats:
                markdown_path = parse_dir / f"{document.stem}.md"
                markdown_text = "\n\n".join(page for page in markdown_pages if page).strip()
                markdown_path.write_text(
                    markdown_text + ("\n" if markdown_text else ""),
                    encoding="utf-8",
                )
                logger.info("Wrote %s", markdown_path)
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
    start: Annotated[
        int | None,
        typer.Option("--start", "-s", help="0-based first PDF page to process."),
    ] = None,
    end: Annotated[
        int | None,
        typer.Option("--end", "-e", help="0-based last PDF page to process, inclusive."),
    ] = None,
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
        documents=collect_input_documents(tuple(path or ())),
        output_dir=output,
        formats=resolve_formats(output_format, quiet=quiet),
        variant=cast(ModelVariant, variant),
        drop_score=drop_score,
        rec_weight_source=cast(RecognitionWeightSource, rec_weight_source),
        compile_models=not no_compile,
        quiet=quiet,
        start=start,
        end=end,
    )


def main() -> None:
    """Run the Typer application."""
    app()


if __name__ == "__main__":
    sys.exit(main())
