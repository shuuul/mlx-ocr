# mlx4ocr Source Guide

Root `AGENTS.md` rules apply here. These additional rules protect the public
runtime surface.

## Public API and Packaging

- Treat `mlx4ocr.__init__` exports, `PP_OCRv6`, `download_model`, CLI commands,
  MCP entry points, output schemas, and `py.typed` as public surface.
- Any public API, CLI option, output schema, dependency, or console script change
  must update README examples and targeted tests in the same task.
- Do not add runtime dependencies without updating `pyproject.toml` and the
  install/requirements documentation.

## Runtime Boundaries

- Validate filesystem paths, Hub inputs, model artifact structure, and user CLI
  values at the boundary.
- Keep OCR pipeline internals direct after configs/artifacts have been validated;
  avoid repeated defensive checks and silent fallbacks in controlled data paths.
- Preserve explicit errors for missing weights, malformed configs, failed image
  decoding, and unsupported public options.

## MLX and Model Code

- Use DeepWiki MCP for MLX API behavior when changing model layers, tensor
  layouts, compilation, or weight loading.
- Keep PaddleOCR parity comments and conversion code factual and minimal.
- Prefer small, typed dataclasses and tuples for structured values crossing
  module boundaries.
