#!/usr/bin/env python3
"""Run PP-OCRv6 on PaddleOCR example images."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from mlx_ocr import PP_OCRv6
from mlx_ocr.hub.rec_weight_patch import RecognitionWeightSource
from mlx_ocr.hub.registry import ModelVariant
from mlx_ocr.output import print_result, save_to_json, save_to_markdown, to_system_results_line

EXAMPLES_ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGES = (
    EXAMPLES_ROOT / "images" / "img_10.jpg",
    EXAMPLES_ROOT / "images" / "sample_doc.jpg",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "images",
        nargs="*",
        type=Path,
        help="BGR image paths (default: img_10.jpg and sample_doc.jpg)",
    )
    parser.add_argument(
        "--variant",
        choices=("tiny", "small", "medium"),
        default="medium",
        help="PP-OCRv6 model size tier",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXAMPLES_ROOT / "output",
        help="Directory for system_results.txt and JSON output",
    )
    parser.add_argument(
        "--drop-score",
        type=float,
        default=0.5,
        help="Minimum recognition score (predict_system default: 0.5)",
    )
    parser.add_argument(
        "--rec-weight-source",
        choices=("auto", "hub", "paddle_pretrained"),
        default="auto",
        help="Recognition weights: auto patches small/medium HF head weights",
    )
    return parser.parse_args()


def read_bgr_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to read image: {path}")
    return image


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    image_paths = list(args.images) if args.images else list(DEFAULT_IMAGES)
    for path in image_paths:
        if not path.is_file():
            raise FileNotFoundError(f"missing image: {path}")

    variant: ModelVariant = args.variant
    rec_weight_source: RecognitionWeightSource = args.rec_weight_source
    ocr = PP_OCRv6.from_hub(
        variant,
        drop_score=args.drop_score,
        rec_weight_source=rec_weight_source,
    )
    args.output.mkdir(parents=True, exist_ok=True)

    system_lines: list[str] = []
    for image_path in image_paths:
        image = read_bgr_image(image_path)
        pipeline_result = ocr.predict(image)
        result = pipeline_result.result
        timing = pipeline_result.timing
        basename = image_path.name

        print(f"=== {basename} ({variant}) ===")
        print_result(result)
        print(
            f"timing: det={timing.det_s:.3f}s rec={timing.rec_s:.3f}s total={timing.total_s:.3f}s"
        )

        system_lines.append(to_system_results_line(result, basename))
        save_to_json(result, args.output, input_path=str(image_path.resolve()))
        save_to_markdown(result, args.output, input_path=str(image_path.resolve()))

    (args.output / "system_results.txt").write_text("".join(system_lines), encoding="utf-8")
    print(f"Wrote results under {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
