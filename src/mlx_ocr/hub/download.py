"""Download PP-OCRv6 artifacts from the Hugging Face Hub."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

from mlx_ocr.hub.registry import ModelTask, ModelVariant, hub_model_ref

logger = logging.getLogger(__name__)

_REQUIRED_FILES: tuple[str, ...] = (
    "config.json",
    "inference.yml",
    "model.safetensors",
    "preprocessor_config.json",
)


@dataclass(frozen=True)
class HubArtifacts:
    """Local paths to files downloaded from a PP-OCRv6 Hub repo."""

    root: Path
    config: Path
    inference: Path
    weights: Path
    preprocessor: Path

    @property
    def config_data(self) -> Mapping[str, object]:
        """Parsed `config.json` contents."""
        return json.loads(self.config.read_text(encoding="utf-8"))

    @property
    def inference_data(self) -> Mapping[str, object]:
        """Parsed `inference.yml` contents."""
        return yaml.safe_load(self.inference.read_text(encoding="utf-8"))


def download_model(
    variant: ModelVariant,
    task: ModelTask,
    *,
    cache_dir: Path | None = None,
    local_dir: Path | None = None,
) -> HubArtifacts:
    """Download a PP-OCRv6 safetensors repo and return local artifact paths.

    Args:
        variant: Model size tier.
        task: `det` or `rec`.
        cache_dir: Optional Hugging Face cache directory.
        local_dir: Optional explicit destination directory.

    Returns:
        Paths to required model artifacts.

    Raises:
        FileNotFoundError: If a required artifact is missing after download.
    """
    ref = hub_model_ref(variant, task)
    logger.info("Downloading %s from Hugging Face Hub", ref.repo_id)

    download_kwargs: dict[str, object] = {
        "repo_id": ref.repo_id,
        "repo_type": "model",
        "allow_patterns": list(_REQUIRED_FILES),
    }
    if cache_dir is not None:
        download_kwargs["cache_dir"] = str(cache_dir)
    if local_dir is not None:
        download_kwargs["local_dir"] = str(local_dir)

    root = Path(snapshot_download(**download_kwargs))

    paths = {name: root / name for name in _REQUIRED_FILES}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing required artifacts for {ref.repo_id}: {', '.join(missing)}"
        )

    return HubArtifacts(
        root=root,
        config=paths["config.json"],
        inference=paths["inference.yml"],
        weights=paths["model.safetensors"],
        preprocessor=paths["preprocessor_config.json"],
    )
