"""Process and MLX memory measurement helpers."""

from __future__ import annotations

import resource
import sys

from benchmarks.common.types import MemorySnapshot


def rss_mb() -> float:
    """Return peak process RSS in megabytes."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    return usage / 1024


def mlx_memory_snapshot() -> tuple[float, float, float]:
    """Return MLX active, peak, and cache memory in megabytes."""
    import mlx.core as mx

    return (
        mx.get_active_memory() / (1024 * 1024),
        mx.get_peak_memory() / (1024 * 1024),
        mx.get_cache_memory() / (1024 * 1024),
    )


def memory_snapshot(*, include_mlx: bool = False) -> MemorySnapshot:
    """Capture process RSS and optional MLX allocator memory."""
    if not include_mlx:
        return MemorySnapshot(rss_mb=rss_mb())

    active, peak, cache = mlx_memory_snapshot()
    return MemorySnapshot(
        rss_mb=rss_mb(),
        mlx_active_mb=active,
        mlx_peak_mb=peak,
        mlx_cache_mb=cache,
    )
