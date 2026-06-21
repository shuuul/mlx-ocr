# mlx-ocr Tests Guide

Root `AGENTS.md` rules apply here. These additional rules keep tests fast,
deterministic, and useful for public behavior.

## Test Scope

- Prefer fast unit tests for public CLI contracts, output formats, package
  metadata, and boundary validation.
- Do not require network access, Hugging Face downloads, or full MLX model
  inference unless an existing parity/integration test already does so.
- Mock `PP_OCRv6.from_hub()` and image/PDF rendering when testing CLI argument
  parsing, output layout, and documented command behavior.

## Golden and Reference Data

- Keep committed golden/parity tests focused on numerical/model correctness.
- Do not regenerate or update golden files unless OCR numerical behavior
  intentionally changed.
- Use existing fixtures from `tests/conftest.py` and committed `examples/`
  assets where practical.

## Documentation Coverage

- When README or `examples/README.md` documents a command, option, output shape,
  or saved file layout, add or update a targeted test that protects the contract.
- Keep documentation-command tests focused on argument parsing, output shape, and
  file layout rather than full model inference.
