"""End-to-end OCR pipeline orchestration."""

from mlx4ocr.pipeline.config import PipelineConfig, pipeline_config_from_artifacts
from mlx4ocr.pipeline.crop import crop_text_regions, sorted_detections
from mlx4ocr.pipeline.memory import MemoryPolicy
from mlx4ocr.pipeline.ocr import PipelineResult, PP_OCRv6, recognize_crops

__all__ = [
    "MemoryPolicy",
    "PP_OCRv6",
    "PipelineConfig",
    "PipelineResult",
    "crop_text_regions",
    "pipeline_config_from_artifacts",
    "recognize_crops",
    "sorted_detections",
]
