# mlx-ocr Benchmarks Guide

Root `AGENTS.md` rules apply here. These additional rules keep benchmarks useful
without making routine development depend on heavyweight external runtimes.

## Dependency Isolation

- Keep mlx-ocr, PaddleOCR, and MinerU benchmark environments isolated. Do not add
  PaddleOCR or MinerU dependencies to the main project environment unless the
  user explicitly requests it.
- Use `PADDLE_BENCHMARK_PYTHON` and `MINERU_BENCHMARK_PYTHON` for external
  backend subprocesses.
- Do not require network access, model downloads, or GPU availability in normal
  unit tests.

## Results and Reproducibility

- Do not commit generated `benchmarks/results/*.json`, local virtualenvs,
  downloaded model caches, or `__pycache__` directories.
- Benchmark README commands must match actual CLI defaults and be honest about
  required setup for each backend.
- When adding benchmark output fields, prefer explicit, serializable metadata
  that helps reproduce the run: backend, variant, image, timings, memory, device,
  and dependency context where available.

## Tests

- Keep benchmark tests fast and deterministic. Mock subprocesses, runner
  pipelines, `PP_OCRv6.from_hub()`, PaddleOCR, and MinerU rather than running
  full OCR benchmarks.
- Test public benchmark contracts: argument parsing, backend selection, JSON
  serialization, comparison table formatting, and output path behavior.
- Preserve explicit failures for invalid benchmark inputs instead of silent
  fallback behavior.
