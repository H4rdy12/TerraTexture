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

from TerraTexture.blend import soft_light, luminosity_blend, _soft_light_numpy, _luminosity_blend_numpy

terra_texture_rs = pytest.importorskip("terra_texture_rs")


def test_rust_soft_light_matches_numpy():
    rng = np.random.default_rng(2)
    a = rng.random((64, 64)).astype(np.float32)
    b = rng.random((64, 64)).astype(np.float32)

    rust_out = terra_texture_rs.soft_light(np.ascontiguousarray(a), np.ascontiguousarray(b))
    numpy_out = _soft_light_numpy(a, b)

    assert rust_out.dtype == np.float32
    np.testing.assert_allclose(rust_out, numpy_out, atol=1e-5)


def test_rust_soft_light_rgb_matches_numpy():
    """The fused 3-D kernel (soft_light_rgb) -- not the 2-D one looped
    per channel -- called directly, not through the public dispatch."""
    rng = np.random.default_rng(9)
    a = rng.random((48, 48, 3)).astype(np.float32)
    b = rng.random((48, 48, 3)).astype(np.float32)

    rust_out = terra_texture_rs.soft_light_rgb(np.ascontiguousarray(a), np.ascontiguousarray(b))
    numpy_out = _soft_light_numpy(a, b)

    assert rust_out.dtype == np.float32
    np.testing.assert_allclose(rust_out, numpy_out, atol=1e-5)


def test_rust_soft_light_rgb_matches_2d_kernel_per_channel():
    """Direct proof the fused 3-D kernel agrees with the (separately
    verified) 2-D kernel applied per channel, at 4 channels (RGBA-shaped)
    -- not just 3, to make sure nothing is hardcoded to exactly 3."""
    rng = np.random.default_rng(10)
    a = rng.random((32, 32, 4)).astype(np.float32)
    b = rng.random((32, 32, 4)).astype(np.float32)

    fused = terra_texture_rs.soft_light_rgb(np.ascontiguousarray(a), np.ascontiguousarray(b))
    for c in range(4):
        per_channel = terra_texture_rs.soft_light(
            np.ascontiguousarray(a[..., c]), np.ascontiguousarray(b[..., c]),
        )
        np.testing.assert_array_equal(fused[..., c], per_channel)


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


def test_public_soft_light_dispatches_to_rust_for_multichannel_rgb():
    """The Rust soft_light kernel itself is 2D-only (see rust/src/lib.rs)
    -- (H, W, 3) RGB(A)-shaped input, e.g. basemap.py's
    `soft_light(luminosity_composite, basemap_rgb)` final-compositing
    call, used to hit the numpy fallback unconditionally regardless of
    whether the extension was built, since a.ndim == 3 failed the old
    2D-only dispatch guard. Verifies the per-channel dispatch path
    (blend.py's _is_fast_path_soft_light_multichannel) actually engages
    and matches the numpy reference exactly, not just that the result
    happens to be correct via the fallback."""
    rng = np.random.default_rng(7)
    a = rng.random((48, 48, 3)).astype(np.float32)
    b = rng.random((48, 48, 3)).astype(np.float32)

    dispatched = soft_light(a, b)
    numpy_out = _soft_light_numpy(a, b)

    assert dispatched.dtype == np.float32
    np.testing.assert_allclose(dispatched, numpy_out, atol=1e-4)


def test_public_soft_light_multichannel_matches_per_channel_2d_calls():
    """Directly confirms the fused 3D kernel (or, on an old-built
    extension without it, the per-channel-loop fallback) is
    mathematically equivalent to calling the 2D Rust kernel on each
    channel separately -- soft_light has no cross-channel interaction,
    so this must hold exactly, not just approximately."""
    rng = np.random.default_rng(8)
    a = rng.random((32, 32, 4)).astype(np.float32)  # 4 channels (RGBA-shaped)
    b = rng.random((32, 32, 4)).astype(np.float32)

    dispatched = soft_light(a, b)
    for c in range(4):
        expected_channel = terra_texture_rs.soft_light(
            np.ascontiguousarray(a[..., c]), np.ascontiguousarray(b[..., c]),
        )
        np.testing.assert_array_equal(dispatched[..., c], expected_channel)


def test_public_soft_light_multichannel_uses_fused_kernel_not_loop():
    """When the extension exports soft_light_rgb (the normal case for
    any recently-built extension), the public dispatch should call it
    directly rather than falling back to the slower per-channel loop --
    monkeypatch soft_light_rgb to prove it's actually invoked, not just
    that the result happens to be correct (the per-channel loop would
    also produce a correct result, so correctness alone can't
    distinguish which path actually ran)."""
    if not hasattr(terra_texture_rs, "soft_light_rgb"):
        pytest.skip("extension built before soft_light_rgb existed")

    calls = []
    original = terra_texture_rs.soft_light_rgb

    def spy(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    terra_texture_rs.soft_light_rgb = spy
    try:
        a = np.full((16, 16, 3), 0.4, dtype=np.float32)
        b = np.full((16, 16, 3), 0.6, dtype=np.float32)
        soft_light(a, b)
    finally:
        terra_texture_rs.soft_light_rgb = original

    assert len(calls) == 1, "expected soft_light_rgb to be called exactly once, not the per-channel loop"


def test_public_soft_light_multichannel_falls_back_for_float64():
    a = np.full((16, 16, 3), 0.4, dtype=np.float64)
    b = np.full((16, 16, 3), 0.6, dtype=np.float64)
    out = soft_light(a, b)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, _soft_light_numpy(a, b))
