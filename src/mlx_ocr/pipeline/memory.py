"""MLX allocator policy for PP-OCRv6 inference."""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass

import mlx.core as mx

logger = logging.getLogger(__name__)

_BYTES_PER_MB = 1024 * 1024


@dataclass(frozen=True)
class MemoryPolicy:
    """MLX memory policy aligned with mlx-lm / mlx-vlm inference patterns.

    Hot-path inference does not clear the allocator cache on every image.
    Clearing is deferred to stage boundaries (high cache pressure), periodic
    batch intervals, or explicit ``PP_OCRv6.close()`` calls.
    """

    cache_limit_mb: float | None = None
    clear_cache_every_n: int = 0
    clear_cache_after_det_cache_mb: float | None = 2048.0


@dataclass
class PipelineMemoryRuntime:
    """Mutable runtime state for applying a :class:`MemoryPolicy`."""

    policy: MemoryPolicy
    prediction_count: int = 0

    def apply_init_limits(self) -> None:
        """Apply one-time allocator limits from the policy."""
        if self.policy.cache_limit_mb is None:
            return
        limit_bytes = int(self.policy.cache_limit_mb * _BYTES_PER_MB)
        mx.set_cache_limit(limit_bytes)
        logger.debug("Set MLX cache limit to %.1f MB", self.policy.cache_limit_mb)

    def maybe_clear_after_det(self) -> None:
        """Clear allocator cache after detection when cache pressure is high."""
        threshold_mb = self.policy.clear_cache_after_det_cache_mb
        if threshold_mb is None:
            return
        cache_mb = mx.get_cache_memory() / _BYTES_PER_MB
        if cache_mb < threshold_mb:
            return
        mx.clear_cache()
        logger.debug(
            "Cleared MLX cache after detection (cache was %.1f MB, threshold %.1f MB)",
            cache_mb,
            threshold_mb,
        )

    def on_predict_end(self) -> None:
        """Apply periodic cache clearing after a completed prediction."""
        self.prediction_count += 1
        every_n = self.policy.clear_cache_every_n
        if every_n <= 0 or self.prediction_count % every_n != 0:
            return
        mx.clear_cache()
        logger.debug(
            "Cleared MLX cache after prediction %d (every_n=%d)",
            self.prediction_count,
            every_n,
        )

    def release(self) -> None:
        """Release cached allocator memory at pipeline teardown."""
        gc.collect()
        mx.clear_cache()
        logger.debug("Released MLX pipeline memory cache")
