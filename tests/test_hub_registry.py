"""Tests for Hugging Face Hub registry."""

from mlx4ocr.hub import hub_model_ref, list_hub_models


def test_hub_model_ref_medium_det() -> None:
    ref = hub_model_ref("medium", "det")
    assert ref.repo_id == "PaddlePaddle/PP-OCRv6_medium_det_safetensors"
    assert ref.variant == "medium"
    assert ref.task == "det"


def test_list_hub_models_count() -> None:
    refs = list_hub_models()
    assert len(refs) == 6
    repo_ids = {ref.repo_id for ref in refs}
    assert "PaddlePaddle/PP-OCRv6_tiny_rec_safetensors" in repo_ids
