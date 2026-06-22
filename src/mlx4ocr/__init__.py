"""MLX-based PP-OCRv6 inference on Apple Silicon."""

from mlx4ocr.hub import (
    PP_OCRV6_COLLECTION_URL,
    HubArtifacts,
    HubModelRef,
    ModelTask,
    ModelVariant,
    download_model,
    hub_model_ref,
    list_hub_models,
)
from mlx4ocr.output import OCRTiming
from mlx4ocr.pipeline import MemoryPolicy, PipelineResult, PP_OCRv6
from mlx4ocr.types import BoundingBox, OCRResult, OCRTextBlock
from mlx4ocr.vlm import VLMOCR

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

__version__ = "0.1.2"
