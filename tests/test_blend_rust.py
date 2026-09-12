"""
Rust-vs-numpy parity tests for terra_texture_rs (see rust/src/lib.rs).

Kept in a SEPARATE file from test_blend.py, not merged in: pytest's
module-level `pytest.importorskip()` skips the entire file's collection
when the import fails, not just the tests physically after that line --
mixing always-run numpy-only tests into the same file as these would
silently skip them too whenever terra_texture_rs isn't built (which is
the common case; see blend.py's module docstring on why that fallback
is load-bearing, not a degraded path). test_io.py's rasterio-only tests
follow this same one-dependency-per-file convention.

Skips entirely (not failing) when the extension hasn't been built --
`maturin develop` from `rust/` is required first.
"""

import numpy as np
import pytest

from terra_texture.blend import soft_light, luminosity_blend, _soft_light_numpy, _luminosity_blend_numpy

terra_texture_rs = pytest.importorskip("terra_texture_rs")


def test_rust_soft_light_matches_numpy():
    rng = np.random.default_rng(2)
    a = rng.random((64, 64)).astype(np.float32)
    b = rng.random((64, 64)).astype(np.float32)

    rust_out = terra_texture_rs.soft_light(np.ascontiguousarray(a), np.ascontiguousarray(b))
    numpy_out = _soft_light_numpy(a, b)

    assert rust_out.dtype == np.float32
    np.testing.assert_allclose(rust_out, numpy_out, atol=1e-5)


def test_rust_luminosity_blend_matches_numpy():
    rng = np.random.default_rng(3)
    backdrop = rng.random((32, 32, 3)).astype(np.float32)
    lum = rng.random((32, 32)).astype(np.float32)

    rust_out = terra_texture_rs.luminosity_blend(
        np.ascontiguousarray(backdrop), np.ascontiguousarray(lum),
    )
    numpy_out = _luminosity_blend_numpy(backdrop, lum)

    assert rust_out.dtype == np.float32
    np.testing.assert_allclose(rust_out, numpy_out, atol=1e-5)


def test_rust_luminosity_blend_matches_numpy_with_out_of_gamut_values():
    """Specifically exercise the low-clip AND high-clip branches inside
    _clip_color (values that go negative or above 1 after the luminosity
    shift) -- the Rust kernel's fused low/high-clip order has to match
    blend.py's sequential np.where reassignment exactly, and that only
    gets tested if some pixels actually hit both branches."""
    backdrop = np.array([[[0.95, 0.05, 0.5], [0.05, 0.95, 0.5]]], dtype=np.float32)
    lum = np.array([[0.99, 0.01]], dtype=np.float32)  # pushes both directions hard

    rust_out = terra_texture_rs.luminosity_blend(
        np.ascontiguousarray(backdrop), np.ascontiguousarray(lum),
    )
    numpy_out = _luminosity_blend_numpy(backdrop, lum)
    np.testing.assert_allclose(rust_out, numpy_out, atol=1e-5)


def test_public_soft_light_dispatches_to_rust_for_float32():
    """The public soft_light() should actually take the fast path (not
    just the direct terra_texture_rs call) when given float32 2D inputs
    of matching shape -- verifies the dispatch guard in blend.py, not
    just the kernel itself."""
    a = np.full((16, 16), 0.4, dtype=np.float32)
    b = np.full((16, 16), 0.6, dtype=np.float32)
    dispatched = soft_light(a, b)
    direct = terra_texture_rs.soft_light(a, b)
    np.testing.assert_array_equal(dispatched, direct)


def test_public_luminosity_blend_dispatches_to_rust_for_float32():
    backdrop = np.full((16, 16, 3), 0.5, dtype=np.float32)
    lum = np.full((16, 16), 0.7, dtype=np.float32)
    dispatched = luminosity_blend(backdrop, lum)
    direct = terra_texture_rs.luminosity_blend(backdrop, lum)
    np.testing.assert_array_equal(dispatched, direct)


def test_public_soft_light_falls_back_for_float64():
    """float64 inputs should never hit the Rust fast path (it's built
    strictly for float32) -- confirms this stays correct rather than
    erroring or silently truncating precision, even with the extension
    installed."""
    a = np.full((16, 16), 0.4, dtype=np.float64)
    b = np.full((16, 16), 0.6, dtype=np.float64)
    out = soft_light(a, b)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, _soft_light_numpy(a, b))
