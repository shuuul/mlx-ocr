# mlx-ocr

MLX-based [PP-OCRv6](https://huggingface.co/collections/PaddlePaddle/pp-ocrv6) inference on Apple Silicon.

This project reimplements PaddleOCRv6 with [MLX](https://github.com/ml-explore/mlx) and loads official Hugging Face `safetensors` weights for local detection and recognition on macOS.

Reference implementation: [`../PaddleOCR`](../PaddleOCR).

## Setup

```bash
uv sync
```

## Quick Start

```python
from mlx_ocr import download_model

artifacts = download_model("medium", "det")
print(artifacts.config_data["model_type"])
```

## Development

```bash
uv run pytest
uv run ruff check .
```

See [AGENTS.md](AGENTS.md) for architecture notes and coding conventions.

## Model Variants

| Variant | Detection Hub repo | Recognition Hub repo |
|---------|-------------------|----------------------|
| tiny    | `PaddlePaddle/PP-OCRv6_tiny_det_safetensors`   | `PaddlePaddle/PP-OCRv6_tiny_rec_safetensors`   |
| small   | `PaddlePaddle/PP-OCRv6_small_det_safetensors`  | `PaddlePaddle/PP-OCRv6_small_rec_safetensors`  |
| medium  | `PaddlePaddle/PP-OCRv6_medium_det_safetensors` | `PaddlePaddle/PP-OCRv6_medium_rec_safetensors` |

## License

Apache-2.0
