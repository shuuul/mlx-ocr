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
from mlx_ocr.types import BoundingBox, OCRResult, TextDetection, TextRecognition

__all__ = [
    "PP_OCRV6_COLLECTION_URL",
    "BoundingBox",
    "HubArtifacts",
    "HubModelRef",
    "ModelTask",
    "ModelVariant",
    "OCRResult",
    "TextDetection",
    "TextRecognition",
    "download_model",
    "hub_model_ref",
    "list_hub_models",
]

__version__ = "0.1.0"
