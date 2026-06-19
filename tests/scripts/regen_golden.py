#!/usr/bin/env python3
"""Regenerate golden tensors from official Paddle inference models.

Requires ``paddlepaddle`` installed in the active environment (not a dev dependency).
Run manually when updating parity baselines:

    uv pip install paddlepaddle
    uv run python tests/scripts/regen_golden.py --all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

logger = logging.getLogger(__name__)

Variant = Literal["tiny", "small", "medium"]

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = REPO_ROOT / "tests"
GOLDEN_ROOT = TESTS_ROOT / "data" / "golden"
IMAGES_ROOT = TESTS_ROOT / "data" / "images"
REFERENCE_ROOT = TESTS_ROOT / "reference"

MODEL_BASE_URL = (
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0"
)
VARIANTS: tuple[Variant, ...] = ("tiny", "small", "medium")


def _import_reference() -> None:
    sys.path.insert(0, str(TESTS_ROOT))


def _require_paddle() -> object:
    try:
        import paddle.inference as paddle_infer
    except ImportError as exc:
        raise SystemExit(
            "paddlepaddle is required to regenerate model goldens. "
            "Install with: uv pip install paddlepaddle"
        ) from exc
    return paddle_infer


def ensure_sample_image(path: Path) -> np.ndarray:
    """Create or load the canonical BGR test image."""
    if path.is_file():
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            return image

    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.ones((800, 600, 3), dtype=np.uint8) * 255
    cv2.rectangle(image, (20, 20), (580, 780), (200, 200, 200), 2)
    cv2.putText(
        image,
        "Hello OCR",
        (50, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "PP-OCRv6",
        (50, 320),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "Golden Test",
        (50, 460),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "2026",
        (50, 600),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write test image: {path}")
    logger.info("wrote synthetic test image to %s", path)
    return image


def download_model(variant: Variant, task: Literal["det", "rec"], cache_dir: Path) -> Path:
    """Download and extract an official Paddle inference model tarball."""
    model_name = f"PP-OCRv6_{variant}_{task}_infer"
    model_dir = cache_dir / model_name
    if (model_dir / "inference.json").is_file() and (model_dir / "inference.pdiparams").is_file():
        return model_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    tar_path = cache_dir / f"{model_name}.tar"
    url = f"{MODEL_BASE_URL}/{model_name}.tar"
    logger.info("downloading %s", url)
    urllib.request.urlretrieve(url, tar_path)
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=cache_dir, filter="data")
    tar_path.unlink(missing_ok=True)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"expected extracted model dir: {model_dir}")
    return model_dir


def run_paddle_inference(model_dir: Path, tensor: np.ndarray) -> np.ndarray:
    """Run a Paddle inference model on a single batch input."""
    paddle_infer = _require_paddle()
    config = paddle_infer.Config(
        str(model_dir / "inference.json"),
        str(model_dir / "inference.pdiparams"),
    )
    config.disable_gpu()
    config.disable_glog_info()
    predictor = paddle_infer.create_predictor(config)
    input_name = predictor.get_input_names()[0]
    input_handle = predictor.get_input_handle(input_name)
    input_handle.reshape(tensor.shape)
    input_handle.copy_from_cpu(tensor.astype(np.float32))
    predictor.run()
    output_name = predictor.get_output_names()[0]
    return predictor.get_output_handle(output_name).copy_to_cpu()


def det_preprocess(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run vendored detection preprocess."""
    from reference.preprocess.det_normalize import NormalizeImage, ToCHWImage
    from reference.preprocess.det_resize import DetResizeForTest

    resize = DetResizeForTest(limit_side_len=960, limit_type="min")
    normalize = NormalizeImage(
        scale="1./255.",
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        order="hwc",
    )
    to_chw = ToCHWImage()
    data: dict[str, object] = {"image": image.copy()}
    data = resize(data)
    shape = np.asarray(data["shape"], dtype=np.float32)
    data = normalize(data)
    data = to_chw(data)
    tensor = np.expand_dims(np.asarray(data["image"], dtype=np.float32), axis=0)
    return tensor, shape


def rec_preprocess(image: np.ndarray) -> np.ndarray:
    """Run vendored recognition preprocess on a fixed crop."""
    from reference.preprocess.rec_resize_norm import resize_norm_img

    crop = image[130:190, 40:280].copy()
    if crop.size == 0:
        raise ValueError("recognition crop is empty; check sample image layout")
    h, w = crop.shape[:2]
    max_wh_ratio = w / float(h)
    tensor = resize_norm_img(crop, max_wh_ratio=max_wh_ratio)
    return np.expand_dims(tensor, axis=0)


def save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array.astype(np.float32))
    logger.info("wrote %s shape=%s", path.relative_to(REPO_ROOT), array.shape)


def regen_variant(variant: Variant, *, cache_dir: Path, image: np.ndarray) -> None:
    """Regenerate golden arrays for one PP-OCRv6 variant."""
    _import_reference()
    from reference.postprocess.ctc_decode import CTCLabelDecode
    from reference.postprocess.db_postprocess import DBPostProcess

    det_dir = download_model(variant, "det", cache_dir)
    rec_dir = download_model(variant, "rec", cache_dir)

    det_input, det_shape = det_preprocess(image)
    det_out = run_paddle_inference(det_dir, det_input)
    prob_map = det_out[:, 0:1, :, :]

    det_golden = GOLDEN_ROOT / variant / "det"
    save_npy(det_golden / "preprocessed.npy", det_input)
    save_npy(det_golden / "shape.npy", det_shape)
    save_npy(det_golden / "prob_map.npy", prob_map)

    rec_input = rec_preprocess(image)
    rec_out = run_paddle_inference(rec_dir, rec_input)

    rec_golden = GOLDEN_ROOT / variant / "rec"
    save_npy(rec_golden / "preprocessed.npy", rec_input)
    save_npy(rec_golden / "softmax.npy", rec_out)

    dict_path = TESTS_ROOT / "data" / "dict" / (
        "ppocrv6_tiny_dict.txt" if variant == "tiny" else "ppocrv6_dict.txt"
    )
    decoder = CTCLabelDecode(character_dict_path=dict_path, use_space_char=True)
    rec_text = decoder(rec_out)

    db = DBPostProcess(thresh=0.2, box_thresh=0.45, max_candidates=3000, unclip_ratio=1.4)
    det_boxes = db({"maps": prob_map}, [tuple(det_shape.tolist())])

    points = det_boxes[0]["points"]
    if isinstance(points, np.ndarray):
        serializable_boxes = points.tolist()
    else:
        serializable_boxes = points

    e2e_path = GOLDEN_ROOT / "e2e" / f"{variant}.json"
    e2e_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "variant": variant,
        "det_boxes": serializable_boxes,
        "rec_text": rec_text[0][0],
        "rec_confidence": rec_text[0][1],
    }
    e2e_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("wrote %s", e2e_path.relative_to(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=VARIANTS,
        help="Regenerate goldens for a single variant.",
    )
    parser.add_argument("--all", action="store_true", help="Regenerate all variants.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / ".cache" / "paddle_infer",
        help="Directory for downloaded inference models.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    if not args.all and args.variant is None:
        raise SystemExit("pass --variant NAME or --all")

    image_path = IMAGES_ROOT / "sample_doc.jpg"
    image = ensure_sample_image(image_path)
    variants: tuple[Variant, ...] = VARIANTS if args.all else (args.variant,)  # type: ignore[assignment]

    for variant in variants:
        logger.info("regenerating goldens for %s", variant)
        regen_variant(variant, cache_dir=args.cache_dir, image=image)


if __name__ == "__main__":
    main()
