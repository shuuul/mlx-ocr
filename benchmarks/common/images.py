"""Resolve benchmark images from the repository."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PADDLE_OCR_ROOT = REPO_ROOT.parent / "PaddleOCR"
EXAMPLE_CACHE = REPO_ROOT / ".cache" / "paddle_examples"

DEFAULT_IMAGE_NAMES: tuple[str, ...] = (
    "img_10.jpg",
    "word_1.jpg",
    "sample_doc.jpg",
)


def resolve_example_images(
    image_paths: tuple[Path, ...] | None = None,
) -> list[Path]:
    """Resolve benchmark images from explicit paths or repository defaults.

    Args:
        image_paths: Optional explicit image paths. When omitted, known example
            images under ``examples/images`` are used.

    Returns:
        Existing image paths in caller order.

    Raises:
        FileNotFoundError: When no images could be resolved.
        ValueError: When an explicit path does not exist.
    """
    if image_paths:
        resolved: list[Path] = []
        for path in image_paths:
            if not path.is_file():
                raise ValueError(f"image not found: {path}")
            resolved.append(path)
        return resolved

    examples_dir = REPO_ROOT / "examples" / "images"
    candidates = [
        *(examples_dir / name for name in DEFAULT_IMAGE_NAMES),
        EXAMPLE_CACHE / "img_10.jpg",
        PADDLE_OCR_ROOT / "doc" / "imgs_en" / "img_10.jpg",
    ]
    images: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved_path = path.resolve()
        if resolved_path in seen or not path.is_file():
            continue
        seen.add(resolved_path)
        images.append(path)
    if not images:
        raise FileNotFoundError("no benchmark images found")
    return images
