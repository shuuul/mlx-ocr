# mlx-ocr examples

Sample images and documents aligned with [PaddleOCR](../PaddleOCR) PP-OCRv6 configs and docs.

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

Use the project CLI for all examples:

```bash
uv run mlx-ocr --path examples/images/img_10.jpg --format markdown
uv run mlx-ocr --path examples/images --format json --output ocr-output
uv run mlx-ocr --path /path/to/document.pdf --format markdown --start 0 --end 2
```

`--format` selects the result format:

- `txt` — plain recognized text
- `markdown` — Markdown body text, with PDF page headings for multi-page input
- `json` — JSON output with PaddleOCR-style `res` fields and PDF `page_index` metadata

`--output` is optional. If omitted, the selected format is printed to stdout. If
provided, files are written under `<output>/<stem>/ocr/`.
