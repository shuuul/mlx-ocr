"""PP-OCRv6 recognition model exports."""

from mlx4ocr.models.rec.config import RecModelConfig, rec_config_from_artifacts
from mlx4ocr.models.rec.model import RecognitionModel

__all__ = [
    "RecModelConfig",
    "RecognitionModel",
    "rec_config_from_artifacts",
]
