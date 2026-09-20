"""
Rust-vs-numpy parity tests for terra_texture_rs's stretch_std kernel
(see rust/src/lib.rs).

Kept in a SEPARATE file from a general (non-Rust) stretch test file for
the same reason as test_blend_rust.py/test_derivatives_rust.py:
pytest's module-level `pytest.importorskip()` skips the whole file's
collection when the import fails, so mixing always-run numpy-only tests
in here would silently skip them too whenever terra_texture_rs isn't
built (the common case).

Skips entirely (not failing) when the extension hasn't been built --
`maturin develop` from `rust/` is required first.
"""

import numpy as np
import pytest

from TerraTexture.stretch import stretch_std

terra_texture_rs = pytest.importorskip("terra_texture_rs")


def _numpy_reference(arr, n_std=4):
    """The straightforward (pre-optimization) numpy formula, kept
    separate from stretch.py's own numpy fallback so this file can
    verify against an independent implementation, not just against
    whatever stretch.py itself currently does."""
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    lo, hi = mean - n_std * std, mean + n_std * std
    return np.clip((arr - lo) / (hi - lo + 1e-12), 0, 1)


def test_rust_stretch_std_matches_numpy():
    rng = np.random.default_rng(11)
    arr = (rng.random((64, 64)).astype(np.float32) - 0.5) * 200

    rust_out = terra_texture_rs.stretch_std(np.ascontiguousarray(arr), 4.0)
    numpy_out = _numpy_reference(arr, 4)

    assert rust_out.dtype == np.float32
    np.testing.assert_allclose(rust_out, numpy_out, atol=1e-4)


def test_rust_stretch_std_matches_numpy_with_nans():
    rng = np.random.default_rng(12)
    arr = (rng.random((64, 64)).astype(np.float32) - 0.5) * 200
    arr[10:20, 10:20] = np.nan

    rust_out = terra_texture_rs.stretch_std(np.ascontiguousarray(arr), 4.0)
    numpy_out = _numpy_reference(arr, 4)

    np.testing.assert_allclose(rust_out, numpy_out, atol=1e-4, equal_nan=True)
    assert np.all(np.isnan(rust_out[10:20, 10:20]))


def test_rust_stretch_std_matches_numpy_at_parallel_threshold():
    """300x300 = 90_000 >= 65_536 -- exercises the rayon-parallel
    reduction AND elementwise branches, not just the serial ones."""
    rng = np.random.default_rng(13)
    arr = (rng.random((300, 300)).astype(np.float32) - 0.5) * 500
    arr[50:60, 50:60] = np.nan

    rust_out = terra_texture_rs.stretch_std(np.ascontiguousarray(arr), 3.5)
    numpy_out = _numpy_reference(arr, 3.5)

    np.testing.assert_allclose(rust_out, numpy_out, atol=1e-3, equal_nan=True)


def test_rust_stretch_std_different_n_std_values():
    rng = np.random.default_rng(14)
    arr = (rng.random((48, 48)).astype(np.float32) - 0.5) * 100

    for n_std in [0.5, 1.0, 2.0, 4.0, 10.0]:
        rust_out = terra_texture_rs.stretch_std(np.ascontiguousarray(arr), n_std)
        numpy_out = _numpy_reference(arr, n_std)
        np.testing.assert_allclose(rust_out, numpy_out, atol=1e-4)


def test_rust_stretch_std_constant_array_zero_variance():
    """A constant array has std=0 -- denom collapses to the 1e-12
    epsilon, matching numpy's own handling of this degenerate case
    rather than dividing by exactly zero."""
    arr = np.full((16, 16), 42.0, dtype=np.float32)
    rust_out = terra_texture_rs.stretch_std(arr, 4.0)
    numpy_out = _numpy_reference(arr, 4.0)
    np.testing.assert_allclose(rust_out, numpy_out, atol=1e-4)
    assert np.all(np.isfinite(rust_out))


def test_public_stretch_std_dispatches_to_rust_for_float32():
    dem = np.random.default_rng(15).random((16, 16)).astype(np.float32)
    dispatched = stretch_std(dem, 4)
    direct = terra_texture_rs.stretch_std(dem, 4.0)
    np.testing.assert_array_equal(dispatched, direct)


def test_public_stretch_std_falls_back_for_float64():
    """float64 inputs never hit the Rust fast path (float32-only) --
    stays correct via the numpy fallback rather than erroring, even
    with the extension installed. The numpy fallback itself uses the
    variance-from-known-mean optimization (see stretch.py), verified
    here to still match the straightforward nanmean+nanstd formula."""
    arr = (np.random.default_rng(16).random((16, 16)).astype(np.float64) - 0.5) * 100
    out = stretch_std(arr, 4)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, _numpy_reference(arr, 4))


def test_stretch_std_numpy_fallback_matches_when_rust_unavailable(monkeypatch):
    """The dispatch guard's actual fallback condition is `_rust is
    None` -- simulate the extension being unavailable and confirm the
    (optimized) numpy branch alone still produces the right answer."""
    import TerraTexture.stretch as stretch_module

    monkeypatch.setattr(stretch_module, "_rust", None)
    rng = np.random.default_rng(17)
    arr = (rng.random((32, 32)).astype(np.float32) - 0.5) * 300
    arr[5:10, 5:10] = np.nan

    out = stretch_std(arr, 4)
    np.testing.assert_allclose(out, _numpy_reference(arr, 4), atol=1e-4, equal_nan=True)
