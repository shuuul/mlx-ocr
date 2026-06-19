"""DB text detection post-processing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon

from mlx_ocr.types import BoundingBox, TextDetection


def db_postprocess(
    prob_map: np.ndarray,
    shape: tuple[float, float, float, float],
    *,
    thresh: float = 0.2,
    box_thresh: float = 0.45,
    max_candidates: int = 3000,
    unclip_ratio: float = 1.4,
    score_mode: str = "fast",
) -> tuple[TextDetection, ...]:
    """Decode a DB probability map into quadrilateral detections.

    Args:
        prob_map: Probability map in NCHW layout ``[1, 1, H, W]``.
        shape: ``(src_h, src_w, ratio_h, ratio_w)`` from preprocess.
        thresh: Binarization threshold.
        box_thresh: Minimum mean score inside a candidate box.
        max_candidates: Maximum number of contours to inspect.
        unclip_ratio: Polygon unclip ratio for box expansion.
        score_mode: ``fast`` or ``slow`` box scoring mode.

    Returns:
        Detected text regions in source-image coordinates.
    """
    if prob_map.ndim != 4 or prob_map.shape[1] != 1:
        raise ValueError(f"expected prob map [B, 1, H, W], got {prob_map.shape}")
    if score_mode not in {"fast", "slow"}:
        raise ValueError(f"score_mode must be 'fast' or 'slow', got {score_mode!r}")

    src_h, src_w, _ratio_h, _ratio_w = shape
    pred = prob_map[0, 0]
    mask = pred > thresh
    boxes, scores = _boxes_from_bitmap(
        pred,
        mask,
        dest_width=src_w,
        dest_height=src_h,
        box_thresh=box_thresh,
        max_candidates=max_candidates,
        unclip_ratio=unclip_ratio,
        score_mode=score_mode,
    )
    detections: list[TextDetection] = []
    for box, score in zip(boxes, scores, strict=True):
        points = tuple((float(x), float(y)) for x, y in box)
        if len(points) != 4:
            continue
        detections.append(
            TextDetection(
                box=BoundingBox(
                    points=(
                        points[0],
                        points[1],
                        points[2],
                        points[3],
                    )
                ),
                score=float(score),
            )
        )
    return tuple(detections)


def db_postprocess_batch(
    prob_maps: np.ndarray,
    shapes: Sequence[tuple[float, float, float, float]],
    params: Mapping[str, float | int | str],
) -> tuple[tuple[TextDetection, ...], ...]:
    """Decode a batch of DB probability maps.

    Args:
        prob_maps: Batch of maps in NCHW layout ``[B, 1, H, W]``.
        shapes: Preprocess shape metadata for each batch item.
        params: Post-process parameters from ``inference.yml``.

    Returns:
        Detections for each batch item.
    """
    results: list[tuple[TextDetection, ...]] = []
    for index in range(prob_maps.shape[0]):
        results.append(
            db_postprocess(
                prob_maps[index : index + 1],
                shapes[index],
                thresh=float(params["thresh"]),
                box_thresh=float(params["box_thresh"]),
                max_candidates=int(params["max_candidates"]),
                unclip_ratio=float(params["unclip_ratio"]),
                score_mode=str(params.get("score_mode", "fast")),
            )
        )
    return tuple(results)


def postprocess_params_from_inference(
    inference: Mapping[str, object],
) -> dict[str, float | int | str]:
    """Extract DB post-process parameters from ``inference.yml`` data."""
    post = inference.get("PostProcess")
    if not isinstance(post, Mapping):
        raise ValueError("inference.yml missing PostProcess mapping")
    return {
        "thresh": float(post["thresh"]),
        "box_thresh": float(post["box_thresh"]),
        "max_candidates": int(post["max_candidates"]),
        "unclip_ratio": float(post["unclip_ratio"]),
        "score_mode": str(post.get("score_mode", "fast")),
    }


def _boxes_from_bitmap(
    pred: np.ndarray,
    bitmap: np.ndarray,
    *,
    dest_width: float,
    dest_height: float,
    box_thresh: float,
    max_candidates: int,
    unclip_ratio: float,
    score_mode: str,
) -> tuple[list[list[tuple[float, float]]], list[float]]:
    height, width = bitmap.shape
    outs = cv2.findContours(
        (bitmap * 255).astype(np.uint8),
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contours = outs[0] if len(outs) == 2 else outs[1]

    boxes: list[list[tuple[float, float]]] = []
    scores: list[float] = []
    for contour in contours[:max_candidates]:
        points, short_side = _mini_boxes(contour)
        if short_side < 3:
            continue
        points_arr = np.array(points, dtype=np.float32)
        score = (
            _box_score_fast(pred, points_arr)
            if score_mode == "fast"
            else _box_score_slow(pred, contour)
        )
        if box_thresh > score:
            continue

        expanded = _unclip(points_arr, unclip_ratio)
        if len(expanded) != 1:
            continue
        expanded_arr = np.array(expanded[0], dtype=np.float32).reshape(-1, 1, 2)
        expanded_points, short_side = _mini_boxes(expanded_arr)
        if short_side < 5:
            continue

        box_arr = np.array(expanded_points, dtype=np.float32)
        box_arr[:, 0] = np.clip(np.round(box_arr[:, 0] / width * dest_width), 0, dest_width)
        box_arr[:, 1] = np.clip(np.round(box_arr[:, 1] / height * dest_height), 0, dest_height)
        boxes.append([(float(x), float(y)) for x, y in box_arr])
        scores.append(score)
    return boxes, scores


def _mini_boxes(contour: np.ndarray) -> tuple[list[list[float]], float]:
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda item: item[0])
    if points[1][1] > points[0][1]:
        index_1, index_4 = 0, 1
    else:
        index_1, index_4 = 1, 0
    if points[3][1] > points[2][1]:
        index_2, index_3 = 2, 3
    else:
        index_2, index_3 = 3, 2
    box = [points[index_1], points[index_2], points[index_3], points[index_4]]
    return box, float(min(bounding_box[1]))


def _box_score_fast(bitmap: np.ndarray, box: np.ndarray) -> float:
    h, w = bitmap.shape[:2]
    work = box.copy()
    xmin = int(np.clip(np.floor(work[:, 0].min()), 0, w - 1))
    xmax = int(np.clip(np.ceil(work[:, 0].max()), 0, w - 1))
    ymin = int(np.clip(np.floor(work[:, 1].min()), 0, h - 1))
    ymax = int(np.clip(np.ceil(work[:, 1].max()), 0, h - 1))
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    work[:, 0] -= xmin
    work[:, 1] -= ymin
    cv2.fillPoly(mask, work.reshape(1, -1, 2).astype(np.int32), 1)
    return float(cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0])


def _box_score_slow(bitmap: np.ndarray, contour: np.ndarray) -> float:
    h, w = bitmap.shape[:2]
    work = contour.reshape(-1, 2).copy()
    xmin = int(np.clip(work[:, 0].min(), 0, w - 1))
    xmax = int(np.clip(work[:, 0].max(), 0, w - 1))
    ymin = int(np.clip(work[:, 1].min(), 0, h - 1))
    ymax = int(np.clip(work[:, 1].max(), 0, h - 1))
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    work[:, 0] -= xmin
    work[:, 1] -= ymin
    cv2.fillPoly(mask, work.reshape(1, -1, 2).astype(np.int32), 1)
    return float(cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0])


def _unclip(box: np.ndarray, unclip_ratio: float) -> list[list[float]]:
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    return offset.Execute(distance)
