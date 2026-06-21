# Roadmap: optional VLM OCR backend with GLM-OCR

This document tracks the VLM OCR roadmap for `mlx-ocr`: keep PP-OCRv6 as the
default local OCR pipeline, and support GLM-OCR through an optional `vlm` extra
powered by [`mlx-vlm`](https://github.com/Blaizzy/mlx-vlm).

## Current status

Implemented:

- `OCRResult` is now block-based and shared by PP-OCRv6 and VLM OCR.
- `PP_OCRv6` still performs detection + recognition, but returns
  `OCRResult.blocks` instead of parallel `detections` / `recognitions`.
- `VLMOCR` is available as a lazy optional backend using
  `mlx-community/GLM-OCR-bf16` through `mlx-vlm`.
- The `vlm` extra is available with `mlx-vlm>=0.6.3`.
- CLI engine selection is available:
  `--engine ppocrv6` and `--engine glm-ocr`.
- PP-OCRv6 CLI JSON remains PaddleOCR/PaddleX-compatible `res` output.
- GLM-OCR CLI JSON uses a generalized `result` object and does not fake
  geometry or confidence scores.
- A smoke test on `examples/ppocrv6.pdf` succeeded with GLM-OCR after setting
  `HF_HUB_DISABLE_XET=1` for the initial model download.

Not implemented yet:

- Real quality/performance benchmarks for GLM-OCR versus PP-OCRv6.
- Schema/table/formula examples with committed fixtures and expected outputs.
- User-facing download guidance for slow or stalled Hugging Face/Xet downloads.
- VLM generation statistics in `OCRResult`.

## Goal

Support GLM-OCR as an optional OCR engine without making the existing PP-OCRv6
runtime heavier or less direct.

Initial target, now mostly implemented:

- Dependency: `mlx-vlm>=0.6.3`, installed only through the `vlm` extra.
- Model: `mlx-community/GLM-OCR-bf16`.
- Primary task: full-image text OCR.
- Secondary prompt presets: formula recognition, table recognition, and
  schema-guided structured extraction.
- Public UX: preserve the current `mlx-ocr` CLI and Python API style while
  making the selected OCR engine explicit.

## Why this is not a small model swap

The current `src/mlx_ocr` pipeline is a classic OCR stack:

```diagram
╭────────────╮   ╭────────────╮   ╭─────────────╮   ╭──────────────╮
│ BGR image  │──▶│ DB detector │──▶│ crop regions │──▶│ CTC recognizer│
╰────────────╯   ╰─────┬──────╯   ╰──────┬──────╯   ╰──────┬───────╯
                       │                 │                 │
                       ▼                 ▼                 ▼
                TextDetection      region images     TextRecognition
```

That shape produces aligned detection boxes and recognition strings. It used to
be exposed as `OCRResult(detections, recognitions)`; it is now normalized into
`OCRResult.blocks`.

GLM-OCR through `mlx-vlm` is a vision-language generation path:

```diagram
╭────────────╮   ╭──────────────╮   ╭──────────────╮   ╭──────────────╮
│ image path │──▶│ chat template│──▶│ VLM generate │──▶│ generated text│
╰────────────╯   ╰──────────────╯   ╰──────────────╯   ╰──────────────╯
```

The upstream API returns generated text and generation statistics, not
PP-OCR-style text boxes. Treating GLM-OCR as a detector/recognizer replacement
would blur contracts and create fake geometry. The first integration should make
that difference explicit.

## Research finding: what GLM-OCR exposes today

Before adding a new contract, verify whether GLM-OCR can populate the existing
types without losing meaning. The current evidence says raw GLM-OCR inference
through `mlx-vlm` is text generation, not structured detector output:

- `mlx-vlm` exposes GLM-OCR through `load()`, `apply_chat_template()`, and
  `generate()`.
- The documented result is `GenerationResult.text`.
- The documented tasks are prompt-defined text outputs:
  `Text Recognition:`, `Formula Recognition:`, `Table Recognition:`, and
  schema-guided JSON extraction.
- No documented `mlx-vlm` GLM-OCR API returns bounding boxes, polygons,
  word/line coordinates, or OCR confidence scores.
- Upstream GLM-OCR SDK documentation mentions Markdown plus JSON layout details,
  but that pipeline adds a separate layout detector and formatter around the
  model service. Those layout details are not the raw `mlx-vlm` GLM-OCR output.

Implication: reusing the current `OCRResult` unchanged would require inventing a
`TextDetection.box` and `TextRecognition.score` for generated full-image text.
That would be worse than a breaking API change. We will generalize the existing
types so both PP-OCRv6 and VLM OCR can use one honest result model.

## Proposed package surface

### Optional dependency

The `vlm` extra is present in `pyproject.toml`:

```toml
[project.optional-dependencies]
vlm = [
    "mlx-vlm>=0.6.3",
]
```

The lower bound is pinned to the version resolved and tested during the first
integration because older `mlx-vlm` versions may not expose the same
`load()` / `generate()` / `prompt_utils.apply_chat_template()` API shape.

README install examples should document:

```bash
uv sync --extra vlm
# or
uv tool install 'git+https://github.com/shuuul/mlx-ocr.git[vlm]'
```

### Python API

Keep `PP_OCRv6` as the default local OCR engine, but allow its result types to
change. The VLM-facing API uses the same generalized output model:

```python
from mlx_ocr import VLMOCR

ocr = VLMOCR.from_hub()
result = ocr.predict_path("document.png")
print(result.text)
```

The public class is currently:

- `VLMOCR`: generic optional VLM OCR engine.

Do not add a `GLMOCR` alias unless it adds real model-specific behavior beyond
the existing `task`, `prompt`, `model_id`, and `max_tokens` parameters.

### CLI

The CLI engine selector preserves PP-OCRv6 as the default behavior:

```bash
mlx-ocr --path input.png --engine ppocrv6 --format markdown
mlx-ocr --path input.png --engine glm-ocr --format markdown
```

VLM-specific options:

- `--vlm-model mlx-community/GLM-OCR-bf16`
- `--vlm-task text|formula|table|schema`
- `--prompt "Text Recognition:"`
- `--max-tokens 512`

VLM-only options are validated only when `--engine glm-ocr` is selected, so
PP-OCRv6 invocations are not blocked by irrelevant VLM options. Schema mode
requires `--prompt`.

## Proposed source refactor

The current `src/mlx_ocr` package mixes three concerns in public-facing modules:
input rendering, engine execution, and output formatting. To add GLM-OCR cleanly,
introduce a small engine boundary before adding `mlx-vlm` code.

### Implemented layout

```text
src/mlx_ocr/
  vlm.py             # optional mlx-vlm backend, imports mlx_vlm lazily
  pipeline/
    ocr.py           # keep PP_OCRv6 detector/recognizer implementation
  cli.py             # input collection/rendering + engine selection
  output.py          # formatting for structured and generated OCR results
  types.py           # shared public result dataclasses
```

We intentionally did not add an `engines/` package yet. `VLMOCR` is a small
module-level backend, and `PP_OCRv6` remains in `pipeline/ocr.py`. Add an engine
package only if a third backend or shared lifecycle abstraction makes it useful.

### Result model

Use one generalized public OCR result model. It is acceptable to break the
current PP-OCRv6 types to avoid adding a parallel VLM-only contract.

Previous PP-OCRv6-only type invariants:

- `TextDetection` always has a `BoundingBox` and detection `score`.
- `TextRecognition` always has recognized `text` and recognition `score`.
- `OCRResult` assumed aligned `detections` and `recognitions`; output helpers
  used PaddleOCR-style JSON geometry fields.

GLM-OCR can honestly provide:

- generated text;
- the model id;
- the prompt/task;
- generation statistics if we choose to expose them;
- prompt-defined JSON text for structured extraction.

GLM-OCR cannot honestly provide through `mlx-vlm` today:

- text-region boxes;
- detection scores;
- recognition confidence scores comparable to CTC confidence.

The implemented result model replaces the strict detector/recognizer pair with
page-level OCR blocks where geometry and confidence are optional metadata:

```python
@dataclass(frozen=True)
class OCRTextBlock:
    text: str
    box: BoundingBox | None = None
    detection_score: float | None = None
    recognition_score: float | None = None


@dataclass(frozen=True)
class OCRResult:
    blocks: tuple[OCRTextBlock, ...]
    text: str
    engine: str
    model: str | None = None
    prompt: str | None = None
```

PP-OCRv6 maps each detected region to one `OCRTextBlock` with `box`,
`detection_score`, and `recognition_score`. GLM-OCR maps the generated page text
to one block with no box or confidence scores. This keeps the public result
honest: missing geometry means the engine did not produce geometry.

Compatibility accessors were not kept. They should only be added later if they
are useful and unambiguous:

```python
@property
def detections(self) -> tuple[TextDetection, ...]: ...

@property
def recognitions(self) -> tuple[TextRecognition, ...]: ...
```

Those accessors should either return only blocks that contain complete geometric
metadata or be removed entirely. Do not keep them as fake compatibility shims if
they obscure VLM output semantics.

Formatting rules:

- `txt` and `markdown`: render `OCRResult.text`.
- CLI `json`: keep emitting PaddleOCR/PaddleX-compatible PP-OCRv6 `res` fields
  and require all emitted blocks to have boxes and scores.
- GLM-OCR CLI `json` uses a generalized `result` shape with `engine`, `model`,
  `prompt`, `text`, and `blocks` rather than claiming PaddleOCR geometry
  compatibility.

### Engine boundary

Use a compact protocol with no speculative methods:

```python
class OCREngine(Protocol):
    def predict_path(self, path: Path) -> OCRResult: ...
    def close(self) -> None: ...
```

Both PP-OCRv6 and GLM-OCR should return this same generalized `OCRResult`.

For PP-OCRv6, the CLI renders images/PDF pages to BGR arrays and calls
`PP_OCRv6.predict(image)`. For GLM-OCR image files, the CLI passes the original
path directly to `VLMOCR.predict_path()` without decoding the image. For GLM-OCR
PDF pages, the CLI renders pages to temporary PNG files, calls
`VLMOCR.predict_path()` on each temporary PNG, and cleans up the temporary files.

### Lazy optional imports

`mlx-vlm` must be imported only inside the VLM engine module or constructor:

```python
try:
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
except ImportError as exc:
    raise RuntimeError("Install VLM OCR support with `uv sync --extra vlm`.") from exc
```

This keeps the default PP-OCRv6 install path unchanged.

## GLM-OCR implementation notes

The upstream `mlx-vlm` GLM-OCR path uses:

```python
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("mlx-community/GLM-OCR-bf16")
prompt = apply_chat_template(processor, model.config, "Text Recognition:", num_images=1)
result = generate(model, processor, prompt, image=["document.png"], max_tokens=512)
text = result.text
```

Prompt presets:

| Task | Prompt |
| --- | --- |
| Text OCR | `Text Recognition:` |
| Formula OCR | `Formula Recognition:` |
| Table OCR | `Table Recognition:` |
| Structured extraction | `请按下列JSON格式输出图中信息:` plus caller schema |

The current integration supports one image/page per generation call. Add batching
later only if profiling shows prompt/image processing overhead is a material
bottleneck.

## Phased plan

### Phase 1: block-based result model — done

- Add `docs/roadmap.md`.
- Generalize existing result types so geometry and confidence are optional block
  metadata, not required top-level OCR invariants.
- Update output helpers and tests around the new `OCRResult.blocks` shape.
- Add tests for formatter behavior before wiring `mlx-vlm`.
- Keep `PP_OCRv6` inference behavior unchanged.

### Phase 2: optional dependency and VLM backend — done

- Add `vlm` optional dependency.
- Add lazy-imported VLM engine module.
- Load `mlx-community/GLM-OCR-bf16` once and reuse `(model, processor)`.
- Implement text OCR prompt preset.
- Add targeted unit tests with `mlx_vlm.load()` and `mlx_vlm.generate()` mocked;
  do not require network or full inference in normal tests.

### Phase 3: CLI and output integration — done

- Add `--engine ppocrv6|glm-ocr` with `ppocrv6` as default.
- Add VLM prompt/model/max-token options.
- Update README examples and public API exports.
- Add CLI tests for argument validation and output shapes.

### Phase 4: quality, examples, and structured tasks — next

- Add committed smoke fixtures for GLM-OCR text/table/formula/schema prompts.
- Add examples under `examples/` only for commands backed by tests.
- Decide how schema extraction should be represented:
  - raw generated text only;
  - raw generated text plus parsed JSON when valid;
  - or a separate helper that parses user-provided schema output outside
    `OCRResult`.
- Compare GLM-OCR and PP-OCRv6 outputs on a small fixed set of images/PDF pages.
- Document recommended `--max-tokens` ranges for single images versus PDF pages.

### Phase 5: download, performance, and memory tuning — next

- Document or automate `HF_HUB_DISABLE_XET=1` fallback for stalled GLM-OCR model
  downloads.
- Measure first-token latency, total generation latency, and MLX peak memory.
- Decide whether a shared memory policy should cover both PP-OCRv6 and VLM
  engines.
- Consider model/session reuse for multi-page PDFs and directories.
- Consider exposing generation stats from `mlx-vlm` in `OCRResult` metadata if
  they are useful for debugging and benchmarking.

### Phase 6: public polish and release readiness — next

- Add README examples for all supported `--vlm-task` values.
- Add a concise troubleshooting section for missing `mlx-vlm`, slow downloads,
  and large-memory models.
- Add a package metadata test that protects the `vlm` extra and README command
  examples.
- Decide whether `mlx-ocr[vlm]` install examples should be shown for pip/uv tool
  users in addition to `uv sync --extra vlm` for development checkouts.

## Open decisions

1. Should schema extraction parse valid generated JSON into an additional field,
   or should `OCRResult.text` remain the only authoritative output?
2. Should `--engine glm-ocr` keep the current default `--format markdown`, or
   should future commands make generated-text JSON easier to request?
3. Should GLM-OCR expose generation statistics in `OCRResult`, and if so should
   that be a typed metadata dataclass or a backend-specific field?
4. Should PDF pages continue to use temporary PNG files, or should we adopt an
   upstream-supported in-memory image path if `mlx-vlm` adds one?
5. Should we add a dedicated `GLMOCR` convenience class, or keep only `VLMOCR`
   until another VLM OCR model is supported?

## Definition of done for first GLM-OCR release

- `uv sync --extra vlm` installs the optional backend. Done.
- `mlx-ocr --engine glm-ocr --path image.png --format markdown` returns generated
  text from GLM-OCR. Done.
- `mlx-ocr --engine glm-ocr --path examples/ppocrv6.pdf --format markdown`
  processes PDF pages through temporary PNGs. Smoke-tested.
- Default `mlx-ocr --path image.png` still uses PP-OCRv6. Done.
- PP-OCRv6 inference behavior remains unchanged, but its public result shape
  uses `OCRResult.blocks` instead of parallel `detections` and `recognitions`.
  Done.
- Normal test suite does not require network, Hugging Face downloads, or full VLM
  inference. Done.
