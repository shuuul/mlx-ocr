"""PP-OCRv6 recognition model exports."""

from mlx_ocr.models.rec.config import RecModelConfig, rec_config_from_artifacts
from mlx_ocr.models.rec.model import RecognitionModel, load_recognition_model

__all__ = [
    "RecModelConfig",
    "RecognitionModel",
    "load_recognition_model",
    "rec_config_from_artifacts",
]
