"""End-to-end PP-OCRv6 detection and recognition pipeline."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_ocr.hub.download import HubArtifacts, download_model
from mlx_ocr.hub.rec_weight_patch import RecognitionWeightSource
from mlx_ocr.hub.registry import ModelVariant
from mlx_ocr.models.det.model import DetectionModel, load_detection_model
from mlx_ocr.models.rec.model import RecognitionModel, load_recognition_model
from mlx_ocr.output import OCRTiming
from mlx_ocr.pipeline.config import PipelineConfig, pipeline_config_from_artifacts
from mlx_ocr.pipeline.crop import crop_text_regions, sorted_detections
from mlx_ocr.postprocess.ctc import ctc_decode
from mlx_ocr.postprocess.db import db_postprocess
from mlx_ocr.preprocess.det import det_preprocess, nhwc_prob_to_nchw
from mlx_ocr.preprocess.rec import rec_preprocess
from mlx_ocr.types import OCRResult, TextDetection, TextRecognition

logger = logging.getLogger(__name__)


def recognize_crops(
    recognizer: RecognitionModel,
    crops: tuple[np.ndarray, ...],
    *,
    rec_image_shape: tuple[int, int, int],
    characters: tuple[str, ...],
    rec_batch_num: int,
) -> tuple[TextRecognition, ...]:
    """Run batched recognition on cropped text regions.

    Args:
        recognizer: Loaded recognition model.
        crops: BGR crops in detection order.
        rec_image_shape: Base ``(C, H, W)`` shape from inference configs.
        characters: CTC character table with blank at index 0.
        rec_batch_num: Maximum crops per forward pass.

    Returns:
        Recognition results in the same order as ``crops``.
    """
    if not crops:
        return ()

    width_ratios = np.asarray([crop.shape[1] / float(crop.shape[0]) for crop in crops])
    sort_indices = np.argsort(width_ratios)
    results: list[TextRecognition | None] = [None] * len(crops)

    for batch_start in range(0, len(crops), rec_batch_num):
        batch_end = min(len(crops), batch_start + rec_batch_num)
        batch_indices = sort_indices[batch_start:batch_end]
        max_wh_ratio = rec_image_shape[2] / float(rec_image_shape[1])
        for index in batch_indices:
            crop = crops[int(index)]
            max_wh_ratio = max(max_wh_ratio, crop.shape[1] / float(crop.shape[0]))

        batch_tensors: list[mx.array] = []
        for index in batch_indices:
            preprocessed = rec_preprocess(
                crops[int(index)],
                max_wh_ratio=max_wh_ratio,
                rec_image_shape=rec_image_shape,
            )
            batch_tensors.append(preprocessed.image)

        batch_input = mx.concatenate(batch_tensors, axis=0)
        softmax = np.asarray(recognizer(batch_input), dtype=np.float32)
        decoded = ctc_decode(softmax, characters)
        for offset, recognition in enumerate(decoded):
            original_index = int(batch_indices[offset])
            results[original_index] = recognition

    if any(result is None for result in results):
        raise RuntimeError("recognition produced incomplete batch results")
    return tuple(result for result in results if result is not None)


@dataclass(frozen=True)
class PipelineResult:
    """OCR output with stage timings."""

    result: OCRResult
    timing: OCRTiming


@dataclass(frozen=True)
class PP_OCRv6:
    """PP-OCRv6 detection and recognition pipeline."""

    variant: ModelVariant
    detector: DetectionModel
    recognizer: RecognitionModel
    config: PipelineConfig

    @classmethod
    def from_artifacts(
        cls,
        variant: ModelVariant,
        det_artifacts: HubArtifacts,
        rec_artifacts: HubArtifacts,
        *,
        drop_score: float = 0.5,
        rec_batch_num: int = 6,
        det_box_type: str = "quad",
        rec_weight_source: RecognitionWeightSource = "auto",
    ) -> PP_OCRv6:
        """Construct a pipeline from downloaded Hub artifacts.

        Args:
            variant: Model size tier.
            det_artifacts: Local detection Hub files.
            rec_artifacts: Local recognition Hub files.
            drop_score: Minimum recognition score to keep a detection.
            rec_batch_num: Maximum recognition batch size.
            det_box_type: ``quad`` or ``poly`` crop mode.
            rec_weight_source: Recognition weight loading mode.

        Returns:
            Initialized pipeline with loaded weights.
        """
        config = pipeline_config_from_artifacts(
            det_artifacts,
            rec_artifacts,
            drop_score=drop_score,
            rec_batch_num=rec_batch_num,
            det_box_type=det_box_type,
        )
        return cls(
            variant=variant,
            detector=load_detection_model(det_artifacts),
            recognizer=load_recognition_model(
                rec_artifacts,
                weight_source=rec_weight_source,
            ),
            config=config,
        )

    @classmethod
    def from_hub(
        cls,
        variant: ModelVariant,
        *,
        cache_dir: Path | None = None,
        drop_score: float = 0.5,
        rec_batch_num: int = 6,
        det_box_type: str = "quad",
        rec_weight_source: RecognitionWeightSource = "auto",
    ) -> PP_OCRv6:
        """Download Hub weights and construct a PP-OCRv6 pipeline.

        Args:
            variant: Model size tier.
            cache_dir: Optional Hugging Face cache directory.
            drop_score: Minimum recognition score to keep a detection.
            rec_batch_num: Maximum recognition batch size.
            det_box_type: ``quad`` or ``poly`` crop mode.
            rec_weight_source: Recognition weight loading mode. ``auto`` uses
                raw Hub safetensors for ``tiny`` and Paddle-pretrained head
                patches for ``small``/``medium``.

        Returns:
            Initialized pipeline with loaded weights.
        """
        det_artifacts = download_model(variant, "det", cache_dir=cache_dir)
        rec_artifacts = download_model(variant, "rec", cache_dir=cache_dir)
        return cls.from_artifacts(
            variant,
            det_artifacts,
            rec_artifacts,
            drop_score=drop_score,
            rec_batch_num=rec_batch_num,
            det_box_type=det_box_type,
            rec_weight_source=rec_weight_source,
        )

    def __call__(self, image: np.ndarray) -> OCRResult:
        """Run detection and recognition on a BGR image."""
        return self.predict(image).result

    def predict(self, image: np.ndarray) -> PipelineResult:
        """Run detection and recognition and return timings.

        Args:
            image: Source image in BGR uint8 layout ``[H, W, 3]``.

        Returns:
            OCR output plus per-stage elapsed seconds.

        Raises:
            ValueError: If ``image`` is not a 3-channel BGR array.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected BGR image [H, W, 3], got shape {image.shape}")

        total_start = time.perf_counter()

        preprocessed = det_preprocess(
            image,
            limit_side_len=self.config.det_limit_side_len,
            limit_type=self.config.det_limit_type,
        )
        det_start = time.perf_counter()
        prob_tensor = self.detector(preprocessed.image)
        mx.eval(prob_tensor)
        prob_map = nhwc_prob_to_nchw(prob_tensor)
        det_s = time.perf_counter() - det_start

        detections = db_postprocess(
            prob_map,
            preprocessed.shape,
            thresh=float(self.config.det_postprocess_params["thresh"]),
            box_thresh=float(self.config.det_postprocess_params["box_thresh"]),
            max_candidates=int(self.config.det_postprocess_params["max_candidates"]),
            unclip_ratio=float(self.config.det_postprocess_params["unclip_ratio"]),
            score_mode=str(self.config.det_postprocess_params.get("score_mode", "fast")),
        )
        ordered = sorted_detections(detections)
        if not ordered:
            logger.info("No text regions detected")
            total_s = time.perf_counter() - total_start
            return PipelineResult(
                result=OCRResult(detections=(), recognitions=()),
                timing=OCRTiming(det_s=det_s, rec_s=0.0, total_s=total_s),
            )

        crops = crop_text_regions(
            image,
            ordered,
            box_type=self.config.det_box_type,
        )
        rec_start = time.perf_counter()
        recognitions = recognize_crops(
            self.recognizer,
            crops,
            rec_image_shape=self.config.rec_image_shape,
            characters=self.config.characters,
            rec_batch_num=self.config.rec_batch_num,
        )
        rec_s = time.perf_counter() - rec_start

        filtered_detections: list[TextDetection] = []
        filtered_recognitions: list[TextRecognition] = []
        for detection, recognition in zip(ordered, recognitions, strict=True):
            if recognition.score < self.config.drop_score:
                continue
            filtered_detections.append(detection)
            filtered_recognitions.append(recognition)

        logger.info(
            "Detected %d regions, kept %d after drop_score=%.2f",
            len(ordered),
            len(filtered_detections),
            self.config.drop_score,
        )
        total_s = time.perf_counter() - total_start
        return PipelineResult(
            result=OCRResult(
                detections=tuple(filtered_detections),
                recognitions=tuple(filtered_recognitions),
            ),
            timing=OCRTiming(det_s=det_s, rec_s=rec_s, total_s=total_s),
        )
