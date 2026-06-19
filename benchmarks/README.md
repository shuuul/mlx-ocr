# PP-OCRv6 Backend Benchmarks

Compare **mlx-ocr** (MLX on Apple Silicon), **PaddleOCR CPU** (`engine=paddle`), and **PaddleOCR ONNX** (`engine=onnxruntime` with `CPUExecutionProvider` on macOS) on the same PP-OCRv6 images and model variants.

## Setup

### mlx-ocr backend

```bash
uv sync --group dev
```

### PaddleOCR backends

PaddleOCR 3.7 pins `numpy<2.4`, which conflicts with mlx-ocr's `numpy>=2.4`. Install Paddle benchmarks in a **separate virtualenv** and point the orchestrator at it:

```bash
python3.12 -m venv .venv-paddle
source .venv-paddle/bin/activate
pip install -r benchmarks/requirements-paddle.txt
```

Then run the full suite with:

```bash
PADDLE_BENCHMARK_PYTHON=.venv-paddle/bin/python uv run python -m benchmarks.run --variant medium
```

Requirements:

- Python 3.12+
- macOS with Apple Silicon for the `mlx` backend
- PaddleOCR 3.7+ and PaddlePaddle CPU wheels in `.venv-paddle`
- `onnxruntime` in the Paddle venv for the ONNX backend (CPU EP on macOS)

Paddle and ONNX runners execute in separate subprocesses so RSS measurements are not polluted by other backends.

## Run

Benchmark all backends on the default example images (`examples/images/`):

```bash
uv run python -m benchmarks.run --variant medium
```

Full comparison across variants:

```bash
uv run python -m benchmarks.run \
  --backends mlx,paddle_cpu,paddle_onnx \
  --variants tiny small medium \
  --warmup 2 \
  --runs 5 \
  --output benchmarks/results/run_latest.json
```

Run a single backend:

```bash
uv run python -m benchmarks.runners.mlx_ocr \
  --variant medium \
  --output benchmarks/results/mlx_medium.json
```

Compare saved results:

```bash
uv run python -m benchmarks.compare benchmarks/results/*.json
uv run python -m benchmarks.compare benchmarks/results/run_latest.json --format csv
```

## Backends

| ID | Engine | Device | Notes |
|----|--------|--------|-------|
| `mlx` | MLX | Apple Silicon GPU | `PP_OCRv6.from_hub()` |
| `paddle_cpu` | `paddle` | `cpu` | MKL-DNN enabled by default |
| `paddle_onnx` | `onnxruntime` | `cpu` | `CPUExecutionProvider` on macOS |

Paddle backends disable document preprocessing and text-line orientation so the benchmark matches mlx-ocr det+rec scope:

- `use_doc_orientation_classify=False`
- `use_doc_unwarping=False`
- `use_textline_orientation=False`

## Metrics

Each runner records per `(backend, variant, image)`:

- **Latency**: `load_s`, `warmup_s`, `infer_mean_s`, `infer_std_s` via `time.perf_counter()`
- **Memory**: process peak RSS (`rss_mb`) after model load and after inference
- **MLX only**: `mlx_peak_mb`, `mlx_active_mb`, `mlx_cache_mb`
- **Sanity**: `detections` count and recognized `texts`

RSS is process resident set size, not GPU VRAM.

## ONNX GPU on Linux

macOS cannot use CUDA. On Linux with an NVIDIA GPU, edit `benchmarks/runners/paddle_onnx.py` to use GPU execution:

```python
records = benchmark_variant(
    images,
    backend="paddle_onnx",
    variant=args.variant,
    engine="onnxruntime",
    device="gpu:0",
    ...
)
```

Install `onnxruntime-gpu` and ensure CUDA is available. Optionally pass `engine_config` with `providers=["CUDAExecutionProvider"]` through `build_paddle_pipeline()` if needed for your environment.

## Legacy entry point

`tests/scripts/benchmark_pipeline.py` is a thin wrapper around `benchmarks/runners/mlx_ocr.py` for backward compatibility.
