# PaddleOCR reference vendoring for mlx-ocr parity tests

Vendored copies of PaddleOCR preprocess/postprocess code and configs used as the
numerical reference for mlx-ocr. No Paddle runtime is required to import or run
these modules.

## Source commit

PaddleOCR commit: `ef346e0b402934477489001a4d253a20dbeb72a5`

## Provenance

| Vendored file | PaddleOCR source |
|---------------|------------------|
| `preprocess/det_resize.py` | `ppocr/data/imaug/operators.py` (`DetResizeForTest`) |
| `preprocess/det_normalize.py` | `ppocr/data/imaug/operators.py` (`NormalizeImage`, `ToCHWImage`) |
| `preprocess/rec_resize_norm.py` | `tools/infer/predict_rec.py` (`resize_norm_img`, default CTC path) |
| `postprocess/db_postprocess.py` | `ppocr/postprocess/db_postprocess.py` (`DBPostProcess`) |
| `postprocess/ctc_decode.py` | `ppocr/postprocess/rec_postprocess.py` (`BaseRecLabelDecode`, `CTCLabelDecode`) |
| `configs/det/*.yml` | `configs/det/PP-OCRv6/` |
| `configs/rec/*.yml` | `configs/rec/PP-OCRv6/` |

## Golden tensors

Golden arrays live under `tests/data/golden/{variant}/{det,rec}/`:

- `det/prob_map.npy` — DBHead probability map for fixed preprocessed input
- `det/preprocessed.npy` — detection preprocess output (input to the model)
- `det/shape.npy` — `[src_h, src_w, ratio_h, ratio_w]`
- `rec/softmax.npy` — CTC softmax logits for fixed preprocessed input
- `rec/preprocessed.npy` — recognition preprocess output (input to the model)

E2E JSON references live under `tests/data/golden/e2e/{variant}.json`.

## Regenerating goldens

Golden model outputs require Paddle inference models and are **not** part of the
default dev dependency set. Regenerate locally with:

```bash
uv run python tests/scripts/regen_golden.py --variant medium
uv run python tests/scripts/regen_golden.py --all
```

The script downloads official PaddleX inference tarballs when needed and writes
arrays into `tests/data/golden/`. CI only reads committed goldens.

## Test image

`tests/data/images/sample_doc.jpg` is a synthetic document image used for
preprocess and forward parity. Replace with `doc/imgs_en/img_10.jpg` from
PaddleOCR when available.
