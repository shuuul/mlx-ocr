"""End-to-end PP-OCRv6 pipeline tests against golden JSON baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from mlx_ocr.hub.download import HubArtifacts
from mlx_ocr.hub.rec_weight_patch import RecognitionWeightSource
from mlx_ocr.hub.registry import ModelTask, ModelVariant
from mlx_ocr.pipeline import PP_OCRv6, sorted_detections
from mlx_ocr.pipeline.crop import sorted_box_indices
from mlx_ocr.pipeline.memory import MemoryPolicy
from mlx_ocr.postprocess.ctc import ctc_decode
from mlx_ocr.preprocess.rec import rec_preprocess_crop_from_image
from mlx_ocr.types import TextDetection
from tests.conftest import GOLDEN_ROOT, load_golden_npy
from tests.reference.compare import assert_allclose

BOX_ATOL = 6.0


def _load_e2e_golden(variant: str) -> dict[str, object]:
    path = GOLDEN_ROOT / "e2e" / f"{variant}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing e2e golden: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sort_golden_boxes(boxes: list[list[list[float]]]) -> list[list[list[float]]]:
    array = np.asarray(boxes, dtype=np.float32)
    order = sorted_box_indices(array)
    return [boxes[int(index)] for index in order]


def _detection_points(detection: TextDetection) -> np.ndarray:
    return np.asarray(detection.box.points, dtype=np.float32)


def _assert_boxes_match(
    actual: tuple[TextDetection, ...],
    expected_boxes: list[list[list[float]]],
    *,
    variant: str,
) -> None:
    assert len(actual) == len(expected_boxes), (
        f"{variant}: expected {len(expected_boxes)} detections, got {len(actual)}"
    )
    actual_points = np.stack([_detection_points(detection) for detection in actual], axis=0)
    expected_points = np.asarray(expected_boxes, dtype=np.float32)
    np.testing.assert_allclose(
        actual_points,
        expected_points,
        atol=BOX_ATOL,
        err_msg=f"{variant}: detection boxes differ",
    )


@pytest.mark.parametrize("variant", ("tiny", "small", "medium"))
def test_pipeline_e2e_matches_golden(
    sample_bgr_image: np.ndarray,
    variant: str,
) -> None:
    """Pipeline detections and recognizer output match committed golden baselines."""
    golden = _load_e2e_golden(variant)
    pipeline = PP_OCRv6.from_hub(variant, drop_score=0.0)
    result = pipeline(sample_bgr_image)

    det_boxes = golden["det_boxes"]
    assert isinstance(det_boxes, list)
    expected_boxes = _sort_golden_boxes(det_boxes)
    ordered = sorted_detections(result.detections)
    _assert_boxes_match(ordered, expected_boxes, variant=variant)

    preprocessed = rec_preprocess_crop_from_image(sample_bgr_image)
    softmax = np.asarray(pipeline.recognizer(preprocessed.image), dtype=np.float32)
    expected_softmax = load_golden_npy(GOLDEN_ROOT / variant / "rec" / "softmax.npy")
    assert_allclose(softmax, expected_softmax, rtol=1e-4, atol=2e-3, err_msg=f"{variant} rec")

    recognition = ctc_decode(softmax, pipeline.config.characters)[0]
    expected_recognition = ctc_decode(expected_softmax, pipeline.config.characters)[0]
    assert recognition.text == expected_recognition.text
    assert abs(recognition.score - expected_recognition.score) <= 2e-3


def test_pp_ocrv6_from_hub_returns_pipeline(sample_bgr_image: np.ndarray) -> None:
    """``PP_OCRv6.from_hub`` constructs a callable pipeline."""
    pipeline = PP_OCRv6.from_hub("tiny")
    result = pipeline(sample_bgr_image)
    assert len(result.detections) == len(result.recognitions)
    assert len(result.detections) > 0


def test_pp_ocrv6_from_hub_accepts_stage_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``from_hub`` can mix detection and recognition tiers."""
    downloads: list[tuple[ModelVariant, ModelTask]] = []
    constructed: list[tuple[ModelVariant, ModelVariant, ModelVariant]] = []

    def fake_download_model(
        variant: ModelVariant,
        task: ModelTask,
        *,
        cache_dir: Path | None = None,
        local_dir: Path | None = None,
    ) -> HubArtifacts:
        del cache_dir, local_dir
        downloads.append((variant, task))
        return HubArtifacts(
            root=Path("/tmp") / f"{variant}_{task}",
            config=Path("/tmp/config.json"),
            inference=Path("/tmp/inference.yml"),
            weights=Path("/tmp/model.safetensors"),
            preprocessor=Path("/tmp/preprocessor_config.json"),
            variant=variant,
            task=task,
        )

    def fake_from_artifacts(
        cls: type[PP_OCRv6],
        variant: ModelVariant,
        det_artifacts: HubArtifacts,
        rec_artifacts: HubArtifacts,
        *,
        drop_score: float = 0.5,
        rec_batch_num: int = 6,
        det_box_type: str = "quad",
        rec_weight_source: RecognitionWeightSource = "auto",
        memory_policy: MemoryPolicy | None = None,
        compile_models: bool = True,
    ) -> PP_OCRv6:
        del cls, drop_score, rec_batch_num, det_box_type, rec_weight_source, memory_policy
        del compile_models
        constructed.append((variant, det_artifacts.variant, rec_artifacts.variant))
        return cast(PP_OCRv6, object())

    monkeypatch.setattr("mlx_ocr.pipeline.ocr.download_model", fake_download_model)
    monkeypatch.setattr(PP_OCRv6, "from_artifacts", classmethod(fake_from_artifacts))

    pipeline = PP_OCRv6.from_hub("medium", det_variant="small")

    assert pipeline is not None
    assert downloads == [("small", "det"), ("medium", "rec")]
    assert constructed == [("medium", "small", "medium")]
