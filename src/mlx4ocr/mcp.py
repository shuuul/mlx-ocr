"""MCP server entry point for mlx4ocr."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import ParamSpec, Protocol, TypeVar, cast

from mlx4ocr.cli import read_bgr_image
from mlx4ocr.hub.rec_weight_patch import RecognitionWeightSource
from mlx4ocr.hub.registry import ModelVariant
from mlx4ocr.output import to_markdown
from mlx4ocr.pipeline import PP_OCRv6

P = ParamSpec("P")
R = TypeVar("R")


class MCPServer(Protocol):
    """Minimal FastMCP protocol used by this module."""

    def tool(self) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Return a decorator registering an MCP tool."""

    def run(self) -> None:
        """Run the server transport."""


def create_server() -> MCPServer:
    """Create the optional MCP server.

    Returns:
        A FastMCP server exposing OCR tools.

    Raises:
        RuntimeError: If the optional MCP dependency is not installed.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        message = "Install MCP support with `uv sync --extra mcp` or `pip install mlx4ocr[mcp]`."
        raise RuntimeError(message) from exc

    server = cast(MCPServer, FastMCP("mlx4ocr"))

    @server.tool()
    def ocr_markdown(
        image_path: str,
        variant: str = "medium",
        drop_score: float = 0.5,
        rec_weight_source: str = "auto",
    ) -> str:
        """Run OCR on a local image and return Markdown.

        Args:
            image_path: Local image file path visible to the MCP server process.
            variant: Model size tier: ``tiny``, ``small``, or ``medium``.
            drop_score: Minimum recognition score to keep a detected text line.
            rec_weight_source: Recognition weight source: ``auto``, ``hub``, or
                ``paddle_pretrained``.

        Returns:
            Markdown OCR output.
        """
        if variant not in {"tiny", "small", "medium"}:
            raise ValueError(f"unsupported variant: {variant}")
        if rec_weight_source not in {"auto", "hub", "paddle_pretrained"}:
            raise ValueError(f"unsupported rec_weight_source: {rec_weight_source}")

        path = Path(image_path)
        ocr = PP_OCRv6.from_hub(
            cast(ModelVariant, variant),
            drop_score=drop_score,
            rec_weight_source=cast(RecognitionWeightSource, rec_weight_source),
        )
        try:
            result = ocr.predict(read_bgr_image(path)).result
            return to_markdown(result)
        finally:
            ocr.close()

    return server


def main() -> int:
    """Run the MCP server over stdio."""
    try:
        server = create_server()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    server.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
