"""Safetensors loading and strict weight mapping for MLX modules."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from safetensors import safe_open

logger = logging.getLogger(__name__)

KeyMapper = Callable[[str], str | None]

_SUFFIX_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (".convolution.weight", ".conv.weight"),
    (".convolution.bias", ".conv.bias"),
    (".normalization.weight", ".bn.weight"),
    (".normalization.bias", ".bn.bias"),
    (".normalization.running_mean", ".bn.running_mean"),
    (".normalization.running_var", ".bn.running_var"),
    (".depthwise_convolution.weight", ".conv.weight"),
    (".depthwise_convolution.bias", ".conv.bias"),
    (".pointwise_convolution.weight", ".pw.weight"),
)


@dataclass(frozen=True)
class WeightLoadResult:
    """Outcome of loading external tensors into an MLX module."""

    loaded: tuple[str, ...]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]


@dataclass(frozen=True)
class WeightMapper:
    """Map Hugging Face safetensor keys to MLX parameter paths."""

    mapping: Mapping[str, str]

    @classmethod
    def from_pairs(cls, pairs: Mapping[str, str]) -> WeightMapper:
        """Build a mapper from explicit source-to-target pairs."""
        return cls(mapping=dict(pairs))

    @classmethod
    def from_transform(
        cls,
        source_keys: Iterable[str],
        target_keys: Iterable[str],
        *,
        strip_prefixes: Sequence[str] = (),
        suffix_replacements: Sequence[tuple[str, str]] = _SUFFIX_REPLACEMENTS,
    ) -> WeightMapper:
        """Build a mapper by rewriting source keys and matching targets.

        Args:
            source_keys: Safetensor keys from a Hub checkpoint.
            target_keys: Flattened MLX parameter paths from the destination module.
            strip_prefixes: Optional prefixes removed from each source key.
            suffix_replacements: Substring replacements applied after prefix stripping.

        Returns:
            A mapper containing one entry per source key.

        Raises:
            ValueError: If a rewritten source key matches zero or multiple targets.
        """
        targets = tuple(target_keys)
        mapping: dict[str, str] = {}

        for source_key in source_keys:
            rewritten = rewrite_hub_key(
                source_key,
                strip_prefixes=strip_prefixes,
                suffix_replacements=suffix_replacements,
            )
            match = _match_transformed_key(rewritten, targets)
            if match is None:
                raise ValueError(
                    f"expected exactly one target for {source_key!r} -> {rewritten!r}, "
                    "found 0"
                )
            mapping[source_key] = match

        return cls(mapping=mapping)

    def map_key(self, source_key: str) -> str | None:
        """Resolve a source key to a target parameter path."""
        return self.mapping.get(source_key)

    def map_tensors(self, tensors: Mapping[str, mx.array]) -> dict[str, mx.array]:
        """Translate a safetensors dictionary into MLX parameter paths."""
        mapped: dict[str, mx.array] = {}
        for source_key, value in tensors.items():
            target_key = self.map_key(source_key)
            if target_key is None:
                continue
            if target_key in mapped:
                raise ValueError(f"duplicate target parameter {target_key!r}")
            mapped[target_key] = value
        return mapped


def paddle_conv_weight_to_mlx(weight: mx.array) -> mx.array:
    """Convert Paddle or PyTorch OIHW conv weights to MLX OHWI layout.

    Args:
        weight: Convolution kernel with shape ``(out_channels, in_channels, kH, kW)``.

    Returns:
        Kernel transposed to MLX layout ``(out_channels, kH, kW, in_channels)``.
    """
    if weight.ndim != 4:
        raise ValueError(f"expected rank-4 conv weight, got shape {weight.shape}")
    return mx.transpose(weight, (0, 2, 3, 1))


def align_tensor_to_parameter(
    value: mx.array,
    expected_shape: tuple[int, ...],
) -> mx.array:
    """Align an external tensor shape with an MLX parameter, including conv layouts.

    Args:
        value: Source checkpoint tensor.
        expected_shape: Destination MLX parameter shape.

    Returns:
        Tensor ready to assign to the destination parameter.

    Raises:
        ValueError: If the tensor cannot be aligned to ``expected_shape``.
    """
    if value.shape == expected_shape:
        return value

    if len(value.shape) == 4 and len(expected_shape) == 4:
        out_channels, in_channels, kernel_h, kernel_w = value.shape
        if expected_shape == (out_channels, kernel_h, kernel_w, in_channels):
            return paddle_conv_weight_to_mlx(value)

    if len(value.shape) == 4 and len(expected_shape) == 4:
        out_channels, in_channels, kernel_h, kernel_w = value.shape
        paddle_shape = (out_channels, in_channels, kernel_h, kernel_w)
        if paddle_shape == (
            expected_shape[0],
            expected_shape[3],
            expected_shape[1],
            expected_shape[2],
        ):
            return paddle_conv_weight_to_mlx(value)

    raise ValueError(
        f"cannot align tensor with shape {value.shape} to parameter shape {expected_shape}"
    )


def rewrite_hub_key(
    source_key: str,
    *,
    strip_prefixes: Sequence[str] = (),
    suffix_replacements: Sequence[tuple[str, str]] = _SUFFIX_REPLACEMENTS,
) -> str:
    """Rewrite a Hub tensor key using prefix stripping and suffix substitutions.

    Args:
        source_key: Original safetensor key.
        strip_prefixes: Prefixes removed from ``source_key`` when present.
        suffix_replacements: Ordered substring replacements applied to the key.

    Returns:
        Rewritten key used for target matching.
    """
    rewritten = source_key
    for prefix in strip_prefixes:
        if rewritten.startswith(prefix):
            rewritten = rewritten[len(prefix) :]
            break

    for old, new in suffix_replacements:
        if rewritten.endswith(old) or old in rewritten:
            rewritten = rewritten.replace(old, new)
    return rewritten


def load_safetensors(path: Path) -> dict[str, mx.array]:
    """Load a safetensors checkpoint into MLX arrays.

    Args:
        path: Path to a ``.safetensors`` file.

    Returns:
        Mapping from tensor names to MLX arrays.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"missing safetensors file: {path}")

    tensors: dict[str, mx.array] = {}
    with safe_open(path, framework="numpy") as handle:
        for key in handle.keys():
            tensors[key] = mx.array(handle.get_tensor(key))
    return tensors


def flatten_module_parameters(module: nn.Module) -> dict[str, mx.array]:
    """Flatten an MLX module tree into dotted parameter paths."""
    return dict(tree_flatten(module.parameters()))


def load_into_module(
    module: nn.Module,
    tensors: Mapping[str, mx.array],
    mapper: WeightMapper | KeyMapper | Mapping[str, str],
    *,
    strict: bool = True,
) -> WeightLoadResult:
    """Load external tensors into an MLX module with optional strict validation.

    Args:
        module: Destination MLX module.
        tensors: Source checkpoint tensors keyed by safetensor names.
        mapper: Explicit mapping, ``WeightMapper``, or per-key transform callable.
        strict: When ``True``, require every module parameter to be present in
            ``tensors`` and reject unmapped source keys. Pass only the checkpoint
            tensors that belong to ``module`` when loading a submodule.

    Returns:
        Structured load report.

    Raises:
        ValueError: If strict loading cannot be satisfied.
    """
    expected = flatten_module_parameters(module)
    mapped = _map_tensors(tensors, mapper)

    missing = tuple(sorted(set(expected) - set(mapped)))
    consumed_sources = _source_keys_for_targets(tensors, mapper, mapped)
    unexpected = tuple(sorted(set(tensors) - consumed_sources))

    aligned: dict[str, mx.array] = {}
    shape_errors: list[str] = []
    for key, value in mapped.items():
        if key not in expected:
            continue
        target_shape = expected[key].shape
        try:
            aligned[key] = align_tensor_to_parameter(value, target_shape)
        except ValueError:
            shape_errors.append(
                f"{key}: expected shape {target_shape}, got {value.shape}"
            )

    if shape_errors:
        detail = "\n".join(shape_errors)
        raise ValueError(f"shape mismatch while loading weights:\n{detail}")

    mapped = aligned

    if strict and (missing or unexpected):
        raise ValueError(
            "strict weight load failed: "
            f"missing={missing or ()}, unexpected={unexpected or ()}"
        )

    module.load_weights(list(mapped.items()), strict=strict)
    loaded = tuple(sorted(mapped))
    if loaded:
        logger.info("Loaded %d tensors into %s", len(loaded), module.__class__.__name__)
    return WeightLoadResult(loaded=loaded, missing=missing, unexpected=unexpected)


def _match_transformed_key(
    rewritten: str,
    targets: Sequence[str],
) -> str | None:
    """Resolve a rewritten Hub key to a unique MLX parameter path."""
    if rewritten in targets:
        return rewritten

    suffix_matches = [target for target in targets if rewritten.endswith(target)]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    if len(suffix_matches) > 1:
        return max(suffix_matches, key=len)

    prefix_matches = [target for target in targets if target.endswith(rewritten)]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    if len(prefix_matches) > 1:
        return max(prefix_matches, key=len)

    return None


def _map_tensors(
    tensors: Mapping[str, mx.array],
    mapper: WeightMapper | KeyMapper | Mapping[str, str],
) -> dict[str, mx.array]:
    if isinstance(mapper, WeightMapper):
        return mapper.map_tensors(tensors)

    mapped: dict[str, mx.array] = {}
    for source_key, value in tensors.items():
        target_key: str | None
        if isinstance(mapper, Mapping):
            target_key = mapper.get(source_key)
        else:
            target_key = mapper(source_key)
        if target_key is None:
            continue
        if target_key in mapped:
            raise ValueError(f"duplicate target parameter {target_key!r}")
        mapped[target_key] = value
    return mapped


def _source_keys_for_targets(
    tensors: Mapping[str, mx.array],
    mapper: WeightMapper | KeyMapper | Mapping[str, str],
    mapped: Mapping[str, mx.array],
) -> set[str]:
    consumed: set[str] = set()
    target_to_source = {target: source for source, target in _iter_source_targets(tensors, mapper)}
    for target in mapped:
        source = target_to_source.get(target)
        if source is not None:
            consumed.add(source)
    return consumed


def _iter_source_targets(
    tensors: Mapping[str, mx.array],
    mapper: WeightMapper | KeyMapper | Mapping[str, str],
) -> Iterable[tuple[str, str]]:
    for source_key in tensors:
        if isinstance(mapper, WeightMapper):
            target_key = mapper.map_key(source_key)
        elif isinstance(mapper, Mapping):
            target_key = mapper.get(source_key)
        else:
            target_key = mapper(source_key)
        if target_key is not None:
            yield source_key, target_key
