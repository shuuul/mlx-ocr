# mlx-ocr Agent Guide

## Project Goal

Rewrite [PP-OCRv6](https://huggingface.co/collections/PaddlePaddle/pp-ocrv6) for Apple Silicon inference using [MLX](https://github.com/ml-explore/mlx). Load official Hugging Face weights and run text detection + recognition locally on macOS.

## MLX Documentation

Use the **DeepWiki MCP** to look up [MLX](https://github.com/ml-explore/mlx) APIs and behavior instead of guessing from source or stale training data. Target repository: `ml-explore/mlx`.

- `read_wiki_structure` — list available MLX documentation topics
- `read_wiki_contents` — read full MLX wiki/docs for a topic area
- `ask_question` — ask targeted questions (e.g. `nn.Module` usage, weight loading, ops, array layout)

Prefer this when porting layers, mapping safetensors weights, or debugging MLX-specific inference issues.

## Hugging Face Weights

Primary weight source: [PaddlePaddle/pp-ocrv6 collection](https://huggingface.co/collections/PaddlePaddle/pp-ocrv6).

| Variant | Detection | Recognition |
|---------|-----------|-------------|
| tiny    | `PaddlePaddle/PP-OCRv6_tiny_det_safetensors`   | `PaddlePaddle/PP-OCRv6_tiny_rec_safetensors`   |
| small   | `PaddlePaddle/PP-OCRv6_small_det_safetensors`  | `PaddlePaddle/PP-OCRv6_small_rec_safetensors`  |
| medium  | `PaddlePaddle/PP-OCRv6_medium_det_safetensors` | `PaddlePaddle/PP-OCRv6_medium_rec_safetensors` |

Each repo ships `config.json`, `inference.yml`, `model.safetensors`, and `preprocessor_config.json`. Prefer safetensors checkpoints for MLX weight loading.

## Repository Layout

```
src/mlx_ocr/
  hub/          # HF download + weight loading
  models/       # MLX modules (backbone, neck, head)
  preprocess/   # image transforms matching inference.yml
  postprocess/  # DB decode, CTC decode
  pipeline/     # det + rec orchestration
  types.py      # frozen dataclasses for structured outputs
tests/
```

Use scoped `AGENTS.md` files for directory-specific rules when a subtree has
stable concerns that should not burden the whole repository. Keep nested guides
short and actionable; root rules still apply unless a narrower guide adds more
specific constraints.

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv run prek run --all-files
```

Python 3.12+. Package manager: [uv](https://docs.astral.sh/uv/).

## Agent Workflow

- Use subagents for bounded review or verification work when public-facing
  changes cross multiple areas, such as README/packaging, CLI behavior, source
  API, and tests.
- Keep subagent tasks read-only unless the edit target is isolated and clearly
  assigned. Integrate and validate the final changes yourself.
- Ask review subagents for evidence: file paths, concrete gaps, and whether a
  change is necessary or only nice-to-have.
- Prefer parallel subagents for independent checks, for example one reviewing
  `src/` public API and packaging, and another reviewing `tests/` coverage.
- Do not use subagents as a substitute for reading the files you will edit or
  for running the final verification command.

## Public Surface Rules

- Treat README examples, `pyproject.toml` metadata and scripts,
  `mlx_ocr.__init__` exports, CLI options/output formats, MCP entry points, and
  structured output dataclasses as public-facing surfaces.
- Update README and tests when a public CLI option, output layout, Python API,
  model variant behavior, optional dependency, or install command changes.
- Keep public documentation honest about project maturity and required platform
  assumptions. This project targets macOS on Apple Silicon.
- Do not document a command as generally available unless it is backed by the
  package metadata, console script, or committed repository workflow.

## Coding Rules

### General Coding Rules

#### Core Defaults

- Default to forward development.
- Do not preserve backward compatibility unless explicitly requested.
- Prefer the simplest implementation that satisfies the requirement.
- Remove dead code, obsolete branches, compatibility layers, unused parameters, and stale helper functions during edits.
- Keep public APIs small and direct.

#### Abstraction and Code Shape

- Do not add thin wrapper functions that only rename a function, forward arguments, or mirror an existing API.
- Add a wrapper only when it contributes at least one of:
  - domain meaning,
  - input validation,
  - output normalization,
  - composition of multiple operations,
  - a materially clearer call boundary.
- Do not duplicate logic across files or functions.
- Extract shared code only when the abstraction is clearer than the repeated code.
- Prefer one obvious implementation for each behavior.
- Merge overlapping helpers and remove single-use indirection.

#### Interfaces and Data Flow

- Keep interfaces compact and explicit.
- Add parameters only when the current task requires them.
- Avoid speculative extensibility.
- Prefer direct data flow over layered indirection.
- Prefer structured outputs over ad hoc dictionaries and loosely shaped blobs.

#### Validation and Failure Semantics

- Validate external inputs at boundaries.
- Treat the OCR pipeline internals as controlled once boundary inputs,
  downloaded artifacts, and parsed configs have been validated.
- Do not add repeated defensive checks inside internal helpers for shapes,
  enum-like strings, or impossible states that are already guaranteed by the
  caller or by committed model configs.
- Fail loudly when required data, required outputs, or required intermediate artifacts are missing.
- Preserve useful failure signals.
- Raise explicit, domain-appropriate errors when validation fails.
- Do not swallow exceptions.
- Do not add silent fallback paths for required behavior.

#### Readability and Maintainability

- Prefer readability over cleverness.
- Prefer explicit control flow over dense one-liners.
- Keep nesting shallow.
- Split files when they contain multiple unrelated responsibilities.
- Write comments only for non-obvious logic, invariants, edge cases, or algorithmic intent.
- Keep comments factual and concise.

#### Anti-Patterns

- Thin wrappers with no behavioral value.
- Public methods that only import and forward to module-level functions.
- Test-only convenience helpers exported from production packages.
- Internal validation repeated across multiple layers of the same controlled
  data path.
- Redundant helpers that duplicate existing code paths.
- Silent fallback logic for required data.
- Broad exception handling with vague recovery.
- Compatibility shims added by default.
- Dead branches kept "just in case".
- Comment noise that restates the code.

### Python Rules

- Write fully typed Python. No `Any`. No `# type: ignore` unless explicitly requested.
- Prefer `@dataclass(frozen=True)` for domain records; use `tuple`/`frozenset` inside frozen records.
- Google-style docstrings on all public functions; module-level docstrings on every package module.
- Use module-level `logger = logging.getLogger(__name__)`.
- Validate filesystem and Hub inputs at boundaries; use `Path` for file paths.
- Add or update tests for every behavioral change.
- Match existing naming and import style in the file you edit.

## Implementation Notes

- **Detection**: DBHead + PPLCNetV4 backbone + RepLKPAN neck (see `PP-OCRv6_*_det.yml`).
- **Recognition**: SVTR_LCNet with MultiHead (CTC + NRTR); start with CTC decode for inference parity.
- **Weight porting**: Map safetensors keys from HF `config.json` architecture to MLX `nn.Module` parameters. Verify shapes against PaddleOCR reference when adding layers.
- **Preprocessing**: Follow `inference.yml` `PreProcess` blocks (BGR decode, normalize, CHW layout).
