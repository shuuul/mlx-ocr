"""Tests for pipeline MLX memory policy."""

from __future__ import annotations

from unittest.mock import patch

from mlx4ocr.pipeline.memory import MemoryPolicy, PipelineMemoryRuntime


def test_memory_policy_defaults_disable_periodic_clear() -> None:
    policy = MemoryPolicy()
    assert policy.clear_cache_every_n == 0
    assert policy.clear_cache_after_det_cache_mb == 2048.0
    assert policy.cache_limit_mb is None


def test_on_predict_end_clears_every_n_predictions() -> None:
    runtime = PipelineMemoryRuntime(MemoryPolicy(clear_cache_every_n=3))
    with patch("mlx4ocr.pipeline.memory.mx.clear_cache") as clear_cache:
        runtime.on_predict_end()
        runtime.on_predict_end()
        clear_cache.assert_not_called()
        runtime.on_predict_end()
        clear_cache.assert_called_once()


def test_maybe_clear_after_det_respects_threshold() -> None:
    runtime = PipelineMemoryRuntime(MemoryPolicy(clear_cache_after_det_cache_mb=100.0))
    with (
        patch("mlx4ocr.pipeline.memory.mx.get_cache_memory", return_value=50 * 1024 * 1024),
        patch("mlx4ocr.pipeline.memory.mx.clear_cache") as clear_cache,
    ):
        runtime.maybe_clear_after_det()
        clear_cache.assert_not_called()

    with (
        patch("mlx4ocr.pipeline.memory.mx.get_cache_memory", return_value=150 * 1024 * 1024),
        patch("mlx4ocr.pipeline.memory.mx.clear_cache") as clear_cache,
    ):
        runtime.maybe_clear_after_det()
        clear_cache.assert_called_once()


def test_maybe_clear_after_det_disabled_when_threshold_none() -> None:
    runtime = PipelineMemoryRuntime(MemoryPolicy(clear_cache_after_det_cache_mb=None))
    with patch("mlx4ocr.pipeline.memory.mx.clear_cache") as clear_cache:
        runtime.maybe_clear_after_det()
        clear_cache.assert_not_called()


def test_apply_init_limits_sets_cache_limit() -> None:
    runtime = PipelineMemoryRuntime(MemoryPolicy(cache_limit_mb=512.0))
    with patch("mlx4ocr.pipeline.memory.mx.set_cache_limit") as set_cache_limit:
        runtime.apply_init_limits()
        set_cache_limit.assert_called_once_with(512 * 1024 * 1024)


def test_release_clears_cache() -> None:
    runtime = PipelineMemoryRuntime(MemoryPolicy())
    with (
        patch("mlx4ocr.pipeline.memory.gc.collect") as collect,
        patch("mlx4ocr.pipeline.memory.mx.clear_cache") as clear_cache,
    ):
        runtime.release()
        collect.assert_called_once()
        clear_cache.assert_called_once()
