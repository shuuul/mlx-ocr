"""Numerical comparison helpers for parity tests."""

from __future__ import annotations

import numpy as np

DEFAULT_RTOL = 1e-4
DEFAULT_ATOL = 1e-5


def assert_allclose(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    err_msg: str = "",
) -> None:
    """Assert two arrays are close within tolerance, with a readable diff on failure.

    Args:
        actual: Array produced by mlx-ocr.
        expected: Reference or golden array.
        rtol: Relative tolerance passed to ``numpy.testing.assert_allclose``.
        atol: Absolute tolerance passed to ``numpy.testing.assert_allclose``.
        err_msg: Optional prefix for assertion errors.

    Raises:
        AssertionError: If arrays differ beyond tolerance.
    """
    actual_arr = np.asarray(actual)
    expected_arr = np.asarray(expected)
    prefix = f"{err_msg}: " if err_msg else ""
    if actual_arr.shape != expected_arr.shape:
        raise AssertionError(
            f"{prefix}shape mismatch: actual {actual_arr.shape} vs expected {expected_arr.shape}"
        )
    max_diff = float(np.max(np.abs(actual_arr - expected_arr)))
    np.testing.assert_allclose(
        actual_arr,
        expected_arr,
        rtol=rtol,
        atol=atol,
        err_msg=f"{prefix}max abs diff {max_diff:.6g}",
    )
