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
from mlx_ocr.hub.weights import (
    WeightLoadResult,
    WeightMapper,
    align_tensor_to_parameter,
    flatten_module_parameters,
    load_into_module,
    load_safetensors,
    paddle_conv_weight_to_mlx,
    rewrite_hub_key,
)

__all__ = [
    "PP_OCRV6_COLLECTION_URL",
    "HubArtifacts",
    "HubModelRef",
    "ModelTask",
    "ModelVariant",
    "WeightLoadResult",
    "WeightMapper",
    "align_tensor_to_parameter",
    "download_model",
    "flatten_module_parameters",
    "hub_model_ref",
    "list_hub_models",
    "load_into_module",
    "load_safetensors",
    "paddle_conv_weight_to_mlx",
    "rewrite_hub_key",
]
