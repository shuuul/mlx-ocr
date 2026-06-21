---
name: mlx-ocr
description: Runs OCR with the mlx-ocr command line tool from GitHub using uvx or uv tool. Use when extracting text from images or PDFs on macOS with Apple Silicon/MLX.
license: MIT
compatibility: Requires macOS, Python 3.12+, uv, and local file access. Intended for Apple Silicon MLX inference; do not suggest Linux or Windows usage.
metadata:
  source: https://github.com/shuuul/mlx-ocr
---

# mlx-ocr

Use this skill to guide users through local OCR with `mlx-ocr` from
`https://github.com/shuuul/mlx-ocr`.

## Constraints

- Only support macOS. The project targets Apple Silicon inference through MLX.
- Use local files only; validate that input paths exist before running OCR.
- Prefer `uvx` for one-off runs and `uv tool install` for repeated use.
- Use the packaged CLI instead of repository example scripts or ad hoc Python scripts.
- Do not promise Linux, Windows, GPU CUDA, or cloud OCR support.

## One-off OCR with uvx

Run the command directly from the GitHub repository:

```bash
uvx --from git+https://github.com/shuuul/mlx-ocr mlx-ocr \
  --path /path/to/input.pdf \
  --format markdown
```

Supported formats:

- `--format txt` for plain recognized text
- `--format markdown` for Markdown text output
- `--format json` for structured JSON with OCR fields and PDF page metadata

By default, output is printed to stdout. Save files by adding `--output`:

```bash
uvx --from git+https://github.com/shuuul/mlx-ocr mlx-ocr \
  --path /path/to/input.pdf \
  --format json \
  --output ocr-output
```

Saved files are written under `<output>/<stem>/ocr/`.

## Repeated use with uv tool

Install once:

```bash
uv tool install git+https://github.com/shuuul/mlx-ocr
```

Then run:

```bash
mlx-ocr --path /path/to/image.jpg --format markdown
```

Upgrade later:

```bash
uv tool upgrade mlx-ocr
```

Remove when no longer needed:

```bash
uv tool uninstall mlx-ocr
```

## PDF page ranges

Use zero-based inclusive page indexes:

```bash
mlx-ocr --path /path/to/document.pdf --format markdown --start 0 --end 2
```

For multi-page PDFs, text and Markdown outputs preserve page boundaries with page
headings. JSON output includes `page_index` for each page.

## Recommended checks before running

1. Confirm the user is on macOS.
2. Confirm `uv` is installed:

   ```bash
   uv --version
   ```

3. Confirm the input file exists:

   ```bash
   test -f /path/to/input.pdf
   ```

4. Pick the narrowest output format that matches the request.

## Common examples

Image to Markdown on stdout:

```bash
uvx --from git+https://github.com/shuuul/mlx-ocr mlx-ocr \
  --path receipt.jpg \
  --format markdown
```

PDF pages 1-3 to a saved Markdown file tree:

```bash
uvx --from git+https://github.com/shuuul/mlx-ocr mlx-ocr \
  --path report.pdf \
  --format markdown \
  --start 0 \
  --end 2 \
  --output ocr-output
```

Directory of images to JSON:

```bash
mlx-ocr --path ./images --format json --output ocr-output
```
