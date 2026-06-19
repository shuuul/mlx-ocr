"""Hugging Face Hub integration for PP-OCRv6 weights."""

from mlx_ocr.hub.download import HubArtifacts, download_model
from mlx_ocr.hub.registry import (
    PP_OCRV6_COLLECTION_URL,
    HubModelRef,
    ModelTask,
    ModelVariant,
    hub_model_ref,
    list_hub_models,
)

__all__ = [
    "PP_OCRV6_COLLECTION_URL",
    "HubArtifacts",
    "HubModelRef",
    "ModelTask",
    "ModelVariant",
    "download_model",
    "hub_model_ref",
    "list_hub_models",
]
