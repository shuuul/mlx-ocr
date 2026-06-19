# mlx-ocr examples

Sample images and scripts aligned with [PaddleOCR](../PaddleOCR) PP-OCRv6 configs and docs.

## Images

| File | Upstream reference | Use |
|------|-------------------|-----|
| `images/img_10.jpg` | `configs/det/PP-OCRv6/*_det.yml` → `doc/imgs_en/img_10.jpg` | English scene detection demo |
| `images/word_1.jpg` | `configs/rec/PP-OCRv6/*_rec.yml` → `doc/imgs_words/ch/word_1.jpg` | Single-line recognition sample |
| `images/sample_doc.jpg` | mlx-ocr golden / synthetic doc | Regression parity image |
| `images/general_ocr_002.png` | [PaddleOCR 3.x OCR docs](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/OCR.md) | Boarding-pass e2e demo |

Sources:

- `img_10.jpg`, `word_1.jpg`: PaddleOCR `release/2.7` branch on GitHub
- `general_ocr_002.png`: `paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/`

## Run

```bash
uv run python examples/run_ocr.py --variant medium images/img_10.jpg
uv run python examples/run_ocr.py --variant tiny --output output/ images/*.jpg
```

Output formats match upstream:

- `system_results.txt` — Paddle `tools/infer/predict_system.py` TSV
- `{basename}_res.json` — PaddleOCR 3.x `save_to_json` layout
