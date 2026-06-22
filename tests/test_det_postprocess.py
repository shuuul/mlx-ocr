"""Detection postprocess parity tests."""

from __future__ import annotations

import numpy as np

from mlx4ocr.postprocess.db import db_postprocess
from tests.reference.postprocess.db_postprocess import DBPostProcess


def _reference_points(
    prob_map: np.ndarray,
    shape: tuple[float, float, float, float],
    *,
    box_thresh: float,
) -> list[list[list[float]]]:
    post = DBPostProcess(
        thresh=0.2,
        box_thresh=box_thresh,
        max_candidates=3000,
        unclip_ratio=1.4,
    )
    result = post({"maps": prob_map}, [shape])
    points = result[0]["points"]
    if isinstance(points, np.ndarray):
        return points.tolist()
    return points


def test_db_postprocess_matches_reference_prob_map() -> None:
    """MLX DB postprocess matches vendored reference on a synthetic map."""
    prob = np.zeros((1, 1, 64, 64), dtype=np.float32)
    prob[0, 0, 16:48, 16:48] = 0.95
    shape = (64.0, 64.0, 1.0, 1.0)

    reference = _reference_points(prob, shape, box_thresh=0.1)
    detections = db_postprocess(prob, shape, box_thresh=0.1)
    assert len(detections) == len(reference)
    for det, ref_box in zip(detections, reference, strict=True):
        actual = np.asarray(det.box.points, dtype=np.float32)
        expected = np.asarray(ref_box, dtype=np.float32)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0)


def test_db_postprocess_empty_map() -> None:
    """An empty probability map yields no detections."""
    prob = np.zeros((1, 1, 32, 32), dtype=np.float32)
    detections = db_postprocess(prob, (32.0, 32.0, 1.0, 1.0))
    assert detections == ()
