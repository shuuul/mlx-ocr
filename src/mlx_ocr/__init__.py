"""MLX-based PP-OCRv6 inference on Apple Silicon."""

from mlx_ocr.hub import (
    PP_OCRV6_COLLECTION_URL,
    HubArtifacts,
    HubModelRef,
    ModelTask,
    ModelVariant,
    download_model,
    hub_model_ref,
    list_hub_models,
)
from mlx_ocr.output import OCRTiming
from mlx_ocr.pipeline import MemoryPolicy, PipelineResult, PP_OCRv6
from mlx_ocr.types import BoundingBox, OCRResult, OCRTextBlock
from mlx_ocr.vlm import VLMOCR

__all__ = [
    "PP_OCRV6_COLLECTION_URL",
    "VLMOCR",
    "BoundingBox",
    "HubArtifacts",
    "HubModelRef",
    "MemoryPolicy",
    "ModelTask",
    "ModelVariant",
    "OCRResult",
    "OCRTextBlock",
    "OCRTiming",
    "PP_OCRv6",
    "PipelineResult",
    "download_model",
    "hub_model_ref",
    "list_hub_models",
]

__version__ = "0.1.0"
