"""PP-OCRv6 detection model exports."""

from mlx_ocr.models.det.config import DetModelConfig, det_config_from_artifacts
from mlx_ocr.models.det.model import DetectionModel

__all__ = [
    "DetModelConfig",
    "DetectionModel",
    "det_config_from_artifacts",
]
