"""
Rust-vs-numpy parity tests for the ``terra_texture_rs.stretch_std`` kernel.

Checks the compiled kernel (``rust/src/lib.rs``) against an independent
numpy reference, and checks that :func:`TerraTexture.stretch.stretch_std`
actually dispatches to it.

Why a separate file:
    The whole module is skipped when the extension isn't built, so any
    numpy-only test placed here would be silently skipped too. Those
    live in ``test_stretch_fallback.py`` (fallback and diagnostics, using
    a fake extension) and the general stretch tests instead. Same pattern
    as ``test_blend_rust.py`` / ``test_derivatives_rust.py``.

Skip vs. fail:
    - Extension *not installed* -> the module is skipped (the common,
      expected case).
    - Extension installed but *fails to import* (ABI or Python-version
      mismatch, stale build, missing symbol) -> collection **errors**.
      ``pytest.importorskip`` would skip this case too, hiding exactly
      the breakage these tests exist to catch.

Dependencies:
    The built extension: ``uv sync --extra rust`` (or ``maturin develop``
    from ``rust/``).

Examples:
    Run just these tests::

        pytest tests/test_stretch_rust.py -v
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest

import TerraTexture.stretch as stretch_module
from TerraTexture.stretch import stretch_std

try:
    import terra_texture_rs
except ModuleNotFoundError as exc:
    if exc.name != "terra_texture_rs":
        raise  # the extension exists but a dependency of it is missing
    pytest.skip(
        "terra_texture_rs not built (uv sync --extra rust)",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

# Rust accumulates in float64, numpy in float32: agree statistically,
# not bit-for-bit.
_ATOL = 1e-4

# Elements at which the kernel switches to rayon-parallel branches.
_PARALLEL_THRESHOLD = 65_536


def _numpy_reference(arr: npt.NDArray[np.floating], n_std: float = 4) -> np.ndarray:
    """
    Compute the stretch with the plain, unoptimised numpy formula.

    Deliberately independent of ``stretch.py``'s own numpy fallback (which
    derives the variance from a known mean), so the tests compare against
    a separate implementation rather than the code under test.

    Args:
        arr (npt.NDArray[np.floating]): Input array; NaNs ignored.
        n_std (float): Half-width of the stretch window, in std units.

    Returns:
        np.ndarray: Stretched array in ``[0, 1]``, NaN where ``arr`` is NaN.
    """
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    lo, hi = mean - n_std * std, mean + n_std * std
    return np.clip((arr - lo) / (hi - lo + 1e-12), 0, 1)


def _random_array(
    seed: int,
    shape: tuple[int, int],
    scale: float,
) -> npt.NDArray[np.float32]:
    """
    Build a reproducible float32 array centred on zero.

    Args:
        seed (int): RNG seed.
        shape (tuple[int, int]): Array shape.
        scale (float): Values span roughly ``[-scale/2, scale/2]``.

    Returns:
        npt.NDArray[np.float32]: The array.
    """
    rng = np.random.default_rng(seed)
    return ((rng.random(shape) - 0.5) * scale).astype(np.float32)


# ---------------------------------------------------------------------------
# Kernel parity (calling terra_texture_rs directly)
# ---------------------------------------------------------------------------

def test_rust_stretch_std_matches_numpy() -> None:
    """The kernel matches numpy on a plain float32 array, dtype preserved."""
    arr = _random_array(11, (64, 64), 200)

    rust_out = terra_texture_rs.stretch_std(np.ascontiguousarray(arr), 4.0)

    assert rust_out.dtype == np.float32
    assert rust_out.shape == arr.shape
    np.testing.assert_allclose(rust_out, _numpy_reference(arr, 4), atol=_ATOL)


def test_rust_stretch_std_matches_numpy_with_nans() -> None:
    """NaNs are excluded from the stats and pass through as NaN."""
    arr = _random_array(12, (64, 64), 200)
    arr[10:20, 10:20] = np.nan

    rust_out = terra_texture_rs.stretch_std(np.ascontiguousarray(arr), 4.0)

    np.testing.assert_allclose(
        rust_out, _numpy_reference(arr, 4), atol=_ATOL, equal_nan=True
    )
    assert np.all(np.isnan(rust_out[10:20, 10:20]))
    assert not np.any(np.isnan(rust_out[:10, :10]))


def test_rust_stretch_std_matches_numpy_at_parallel_threshold() -> None:
    """300 x 300 = 90,000 elements exercises the rayon-parallel branches."""
    arr = _random_array(13, (300, 300), 500)
    arr[50:60, 50:60] = np.nan
    assert arr.size >= _PARALLEL_THRESHOLD

    rust_out = terra_texture_rs.stretch_std(np.ascontiguousarray(arr), 3.5)

    np.testing.assert_allclose(
        rust_out, _numpy_reference(arr, 3.5), atol=1e-3, equal_nan=True
    )


def test_rust_stretch_std_just_below_parallel_threshold() -> None:
    """The largest serial-branch size agrees too (off-by-one guard)."""
    arr = _random_array(18, (1, _PARALLEL_THRESHOLD - 1), 100)

    rust_out = terra_texture_rs.stretch_std(np.ascontiguousarray(arr), 2.0)

    np.testing.assert_allclose(rust_out, _numpy_reference(arr, 2.0), atol=_ATOL)


@pytest.mark.parametrize("n_std", [0.5, 1.0, 2.0, 4.0, 10.0])
def test_rust_stretch_std_different_n_std_values(n_std: float) -> None:
    """Parity holds across a range of stretch widths."""
    arr = _random_array(14, (48, 48), 100)

    rust_out = terra_texture_rs.stretch_std(np.ascontiguousarray(arr), n_std)

    np.testing.assert_allclose(rust_out, _numpy_reference(arr, n_std), atol=_ATOL)


def test_rust_stretch_std_constant_array_zero_variance() -> None:
    """A constant array (std = 0) maps to finite zeros, not 0/0 NaNs.

    The denominator collapses to the 1e-12 epsilon, matching numpy's
    handling of this degenerate case.
    """
    arr = np.full((16, 16), 42.0, dtype=np.float32)

    rust_out = terra_texture_rs.stretch_std(arr, 4.0)

    np.testing.assert_allclose(rust_out, _numpy_reference(arr, 4.0), atol=_ATOL)
    assert np.all(np.isfinite(rust_out))


def test_rust_stretch_std_does_not_modify_input() -> None:
    """The kernel writes a new array; the input is left untouched."""
    arr = _random_array(19, (32, 32), 100)
    arr[0, 0] = np.nan
    original = arr.copy()

    terra_texture_rs.stretch_std(arr, 4.0)

    np.testing.assert_array_equal(arr, original)


# ---------------------------------------------------------------------------
# Public API dispatch (TerraTexture.stretch.stretch_std)
# ---------------------------------------------------------------------------

def test_stretch_module_loaded_the_extension() -> None:
    """``stretch._rust`` is the real extension, with no import error.

    If this fails while the direct import above worked, the package is
    importing a different module name or path than this test file.
    """
    assert stretch_module._rust is terra_texture_rs
    assert stretch_module._RUST_IMPORT_ERROR is None
    assert hasattr(terra_texture_rs, "stretch_std")


def test_public_stretch_std_dispatches_to_rust_for_float32() -> None:
    """2-D float32 input goes to the kernel: results are bit-identical."""
    dem = np.random.default_rng(15).random((16, 16)).astype(np.float32)

    dispatched = stretch_std(dem, 4)
    direct = terra_texture_rs.stretch_std(dem, 4.0)

    np.testing.assert_array_equal(dispatched, direct)


@pytest.mark.parametrize(
    "make_view",
    [
        lambda a: a.T,
        lambda a: a[::2, ::3],
        lambda a: np.asfortranarray(a),
    ],
    ids=["transposed", "strided", "fortran-order"],
)
def test_public_stretch_std_handles_non_contiguous_input(
    make_view: Callable[[np.ndarray], np.ndarray],
) -> None:
    """Non-C-contiguous float32 views still dispatch correctly.

    ``stretch_std`` copies them to a contiguous array before calling the
    kernel; this guards against the kernel ever being handed raw strides.
    """
    base = _random_array(20, (40, 60), 100)
    view = make_view(base)

    out = stretch_std(view, 3.0)

    assert out.shape == view.shape
    np.testing.assert_allclose(out, _numpy_reference(view, 3.0), atol=_ATOL)


def test_public_stretch_std_all_nan_returns_all_nan() -> None:
    """An all-NaN input returns all-NaN through the Rust path too.

    Pins the kernel's zero-valid-count behaviour: it must propagate NaN,
    not return zeros or garbage from dividing by a count of zero.
    """
    arr = np.full((8, 8), np.nan, dtype=np.float32)

    out = stretch_std(arr, 4)

    assert out.shape == arr.shape
    assert np.all(np.isnan(out))


def test_public_stretch_std_validates_before_dispatch() -> None:
    """Bad arguments raise ValueError before reaching the kernel."""
    arr = _random_array(21, (8, 8), 10)

    with pytest.raises(ValueError, match="n_std"):
        stretch_std(arr, 0)
    with pytest.raises(ValueError, match="empty"):
        stretch_std(np.zeros((0, 4), dtype=np.float32), 4)


def test_public_stretch_std_falls_back_for_float64() -> None:
    """float64 never takes the float32-only fast path, yet stays correct.

    Also checks that the numpy fallback's variance-from-known-mean
    optimisation still matches the plain nanmean + nanstd formula.
    """
    arr = (np.random.default_rng(16).random((16, 16)) - 0.5) * 100

    out = stretch_std(arr, 4)

    assert out.dtype == np.float64
    np.testing.assert_allclose(out, _numpy_reference(arr, 4))


def test_numpy_fallback_matches_when_rust_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``_rust = None`` the numpy branch alone gives the right answer.

    Run here as well as in the fallback tests so both paths are checked
    against the same reference on a machine where Rust *is* installed.
    """
    monkeypatch.setattr(stretch_module, "_rust", None)
    arr = _random_array(17, (32, 32), 300)
    arr[5:10, 5:10] = np.nan

    out = stretch_std(arr, 4)

    np.testing.assert_allclose(
        out, _numpy_reference(arr, 4), atol=_ATOL, equal_nan=True
    )


def test_rust_and_numpy_paths_agree(monkeypatch: pytest.MonkeyPatch) -> None:
    """The public function gives the same answer with and without Rust."""
    arr = _random_array(22, (128, 128), 1000)
    arr[::7, ::11] = np.nan

    with_rust = stretch_std(arr, 2.5)
    monkeypatch.setattr(stretch_module, "_rust", None)
    without_rust = stretch_std(arr, 2.5)

    np.testing.assert_allclose(with_rust, without_rust, atol=_ATOL, equal_nan=True)
