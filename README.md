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
import cv2
from mlx_ocr import PP_OCRv6

image = cv2.imread("examples/images/img_10.jpg")
ocr = PP_OCRv6.from_hub("medium")
result = ocr.predict(image)
result.result.print()
print(result.timing.as_dict())
```

Or use the example script:

```bash
uv run python examples/run_ocr.py --variant medium examples/images/img_10.jpg
```

Output formats align with PaddleOCR:

- `system_results.txt` — `tools/infer/predict_system.py` TSV (`transcription` + `points`)
- `{basename}_res.json` — PaddleOCR 3.x `save_to_json` layout

**Note:** Hugging Face `small`/`medium` recognition safetensors ship swapped head encoder weights. `PP_OCRv6.from_hub(..., rec_weight_source="auto")` (default) patches them from official Paddle pretrained checkpoints.

Download artifacts only:

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

### Recognition weights (small / medium)

Hugging Face `small` and `medium` recognition safetensors ship corrupted
`head.encoder.conv_block` tensors (`conv_reduce` and `skip_conv` are swapped and
use wrong conv weights). By default, `PP_OCRv6.from_hub()` and
`load_recognition_model()` patch those ten tensors from official Paddle
pretrained checkpoints (downloaded once to `.cache/paddle_pretrained/`; no
Paddle runtime required). Use `rec_weight_source="hub"` to load raw safetensors.

## License

Apache-2.0
