# mlx-ocr

[![Lint](https://github.com/shuuul/mlx-ocr/actions/workflows/lint.yml/badge.svg)](https://github.com/shuuul/mlx-ocr/actions/workflows/lint.yml)

Apple Silicon OCR powered by [MLX](https://github.com/ml-explore/mlx) and
[PP-OCRv6](https://huggingface.co/collections/PaddlePaddle/pp-ocrv6).

`mlx-ocr` reimplements PP-OCRv6 detection and recognition for local macOS
inference. It downloads official Hugging Face `safetensors` weights on demand
and runs the OCR pipeline without a PaddlePaddle runtime.

> [!NOTE]
> This project is pre-alpha. APIs and output details may change while the MLX
> port is being completed and validated.

## Features

- PP-OCRv6 text detection and recognition on Apple Silicon with MLX.
- Official `tiny`, `small`, and `medium` Hugging Face model variants.
- Image, PDF, and non-recursive directory inputs from the CLI.
- Plain text, Markdown, and PaddleOCR-style JSON output.
- Optional saved output layout compatible with document OCR workflows.
- Optional MCP server and installable agent skill for compatible coding agents.

## Requirements

- macOS on Apple Silicon.
- Python 3.12 or newer.
- [`uv`](https://docs.astral.sh/uv/) for local development and CLI execution.
- Internet access on first run to download model weights from Hugging Face.

## Installation

Install directly from GitHub with `uv tool`:

```bash
uv tool install git+https://github.com/shuuul/mlx-ocr.git
```

Or run the CLI without installing it permanently:

```bash
uvx --from git+https://github.com/shuuul/mlx-ocr.git mlx-ocr --help
```

For development from a checkout:

```bash
git clone https://github.com/shuuul/mlx-ocr.git
cd mlx-ocr
uv sync --group dev
```

## Quick start

Run OCR on an image and print Markdown to stdout:

```bash
mlx-ocr --path input.png --format markdown
```

From a development checkout, you can run the bundled examples with `uv run`:

```bash
uv run mlx-ocr --path examples/images/img_10.jpg --format markdown
```

Use `uvx` when running directly from GitHub without installation:

```bash
uvx --from git+https://github.com/shuuul/mlx-ocr.git \
  mlx-ocr --path input.png --format markdown
```

Python API:

```python
import cv2

from mlx_ocr import PP_OCRv6

image = cv2.imread("examples/images/img_10.jpg")
ocr = PP_OCRv6.from_hub("medium")

try:
    result = ocr.predict(image)
    result.result.print()
    print(result.timing.as_dict())
finally:
    ocr.close()
```

## CLI usage

The CLI accepts image files, PDF files, or a non-recursive directory of supported
inputs:

```bash
mlx-ocr --path examples/images --format json --output ocr-output
```

Supported output formats:

- `txt` — recognized text only.
- `markdown` — recognized text as Markdown, preserving PDF page headings.
- `json` — PaddleOCR-style `res` fields with PDF `page_index` metadata.

PDF page ranges use 0-based inclusive page indexes:

```bash
mlx-ocr --path docs/report.pdf --format markdown --start 0 --end 2
```

When `--output` is omitted, results are printed to stdout. When `--output` is
provided, files are written with this layout:

- `<output>/<stem>/ocr/<stem>.txt` — plain text output.
- `<output>/<stem>/ocr/<stem>.md` — Markdown output.
- `<output>/<stem>/ocr/<stem>.json` — JSON output.
- `<output>/<stem>/ocr/<stem>_origin.pdf` — original PDF copy for PDF inputs.

Useful options:

```bash
mlx-ocr --help
mlx-ocr --path input.png --variant tiny --format txt
mlx-ocr --path input.pdf --rec-weight-source auto --no-compile
```

## MCP server

Optional MCP support is available as an extra:

```bash
uv sync --extra mcp
uv run mlx-ocr-mcp
```

The MCP server exposes an `ocr_markdown` tool that reads a local image path and
returns Markdown OCR output.

## Agent skill

This repository includes an agent skill for compatible coding agents. Install it
with `npx skills`:

```bash
npx skills add shuuul/mlx-ocr
```

After installation, compatible agents can use the skill to run `mlx-ocr`
directly from GitHub with `uvx` or `uv tool` on macOS.

## Model variants

| Variant | Detection Hub repo | Recognition Hub repo |
|---------|-------------------|----------------------|
| `tiny` | `PaddlePaddle/PP-OCRv6_tiny_det_safetensors` | `PaddlePaddle/PP-OCRv6_tiny_rec_safetensors` |
| `small` | `PaddlePaddle/PP-OCRv6_small_det_safetensors` | `PaddlePaddle/PP-OCRv6_small_rec_safetensors` |
| `medium` | `PaddlePaddle/PP-OCRv6_medium_det_safetensors` | `PaddlePaddle/PP-OCRv6_medium_rec_safetensors` |

Use `det_variant` or `rec_variant` when detection and recognition should use
different tiers. For example, MinerU `ch_server` is closest to small detection
with medium recognition:

```python
from mlx_ocr import PP_OCRv6

ocr = PP_OCRv6.from_hub("medium", det_variant="small")
```

To download model artifacts without constructing the OCR pipeline:

```python
from mlx_ocr import download_model

artifacts = download_model("medium", "det")
print(artifacts.config_data["model_type"])
```

### Recognition weights for `small` and `medium`

The Hugging Face `small` and `medium` recognition safetensors currently ship
swapped `head.encoder.conv_block` tensors. By default,
`PP_OCRv6.from_hub(..., rec_weight_source="auto")` and
`load_recognition_model()` patch the affected tensors from official Paddle
pretrained checkpoints. The checkpoints are downloaded once to
`.cache/paddle_pretrained/`; PaddlePaddle is not required.

Use `rec_weight_source="hub"` only when you explicitly want the raw Hugging Face
safetensors.

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv run prek run --all-files
```

See [AGENTS.md](AGENTS.md) for architecture notes and coding conventions.

## License

MIT License. See [LICENSE](LICENSE).
