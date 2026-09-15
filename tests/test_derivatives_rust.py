"""
Rust-vs-numpy parity tests for terra_texture_rs's curvature/hillshade
kernels (see rust/src/lib.rs).

Kept in a SEPARATE file from test_derivatives.py for the same reason as
test_blend_rust.py: pytest's module-level `pytest.importorskip()` skips
the whole file's collection when the import fails, so mixing always-run
numpy-only tests in here would silently skip them too whenever
terra_texture_rs isn't built (the common case).

Skips entirely (not failing) when the extension hasn't been built --
`maturin develop` from `rust/` is required first.
"""

import numpy as np
import pytest

from TerraTexture.derivatives import curvatures, hillshade, _derivatives

terra_texture_rs = pytest.importorskip("terra_texture_rs")


def _numpy_curvatures(dem, cellsize):
    """Reference implementation lifted straight out of derivatives.py's
    numpy branch, so this file can compare the Rust path against it
    without relying on the dispatch logic under test also being correct."""
    p, q, r, t, s = _derivatives(dem, cellsize)
    p2q2 = p ** 2 + q ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        profile = -(r * p ** 2 + 2 * s * p * q + t * q ** 2) / (p2q2 * (1 + p2q2) ** 1.5)
        planform = -(r * q ** 2 - 2 * s * p * q + t * p ** 2) / (p2q2 ** 1.5)
    flat = p2q2 < 1e-9
    profile = np.where(flat, 0.0, np.nan_to_num(profile, nan=0.0, posinf=0.0, neginf=0.0))
    planform = np.where(flat, 0.0, np.nan_to_num(planform, nan=0.0, posinf=0.0, neginf=0.0))
    return profile, planform


def test_rust_curvatures_matches_numpy():
    rng = np.random.default_rng(4)
    dem = rng.random((48, 48)).astype(np.float32) * 100.0

    rust_profile, rust_planform = terra_texture_rs.curvatures(np.ascontiguousarray(dem), 2.0)
    numpy_profile, numpy_planform = _numpy_curvatures(dem, 2.0)

    assert rust_profile.dtype == np.float32
    np.testing.assert_allclose(rust_profile, numpy_profile, atol=1e-4)
    np.testing.assert_allclose(rust_planform, numpy_planform, atol=1e-4)


def test_rust_curvatures_matches_numpy_at_parallel_threshold():
    """Same as above but big enough (300x300 = 90_000 >= 65_536) to
    actually exercise the rayon-parallel branch inside curvatures_core,
    not just the serial one -- small arrays alone wouldn't catch a
    parallel-path-specific bug."""
    rng = np.random.default_rng(5)
    dem = rng.random((300, 300)).astype(np.float32) * 200.0

    rust_profile, rust_planform = terra_texture_rs.curvatures(np.ascontiguousarray(dem), 1.7)
    numpy_profile, numpy_planform = _numpy_curvatures(dem, 1.7)

    np.testing.assert_allclose(rust_profile, numpy_profile, atol=1e-4)
    np.testing.assert_allclose(rust_planform, numpy_planform, atol=1e-4)


def test_rust_curvatures_flat_dem_is_exactly_zero():
    dem = np.full((16, 16), 50.0, dtype=np.float32)
    profile, planform = terra_texture_rs.curvatures(dem, 1.0)
    assert np.all(profile == 0.0)
    assert np.all(planform == 0.0)


def test_rust_hillshade_matches_numpy():
    rng = np.random.default_rng(6)
    dem = rng.random((48, 48)).astype(np.float32) * 100.0

    rust_out = terra_texture_rs.hillshade(np.ascontiguousarray(dem), 2.0, 315.0, 45.0)

    az = np.float32(np.radians(360.0 - 315.0 + 90))
    alt = np.float32(np.radians(45.0))
    zy, zx = np.gradient(dem, 2.0)
    slope = np.pi / 2 - np.arctan(np.hypot(zx, zy))
    aspect = np.arctan2(-zx, zy)
    numpy_out = np.clip(
        np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect), 0, 1,
    )

    assert rust_out.dtype == np.float32
    np.testing.assert_allclose(rust_out, numpy_out, atol=1e-4)


def test_public_curvatures_dispatches_to_rust_for_float32():
    """The public curvatures() should actually take the fast path (not
    just the direct terra_texture_rs call) for a float32 2D DEM --
    verifies the dispatch guard in derivatives.py, not just the kernel."""
    dem = np.full((16, 16), 10.0, dtype=np.float32)
    dem[8, 8] = 15.0  # a bump, so curvature isn't trivially all-zero
    dispatched_profile, dispatched_planform = curvatures(dem, 1.0)
    direct_profile, direct_planform = terra_texture_rs.curvatures(dem, 1.0)
    np.testing.assert_array_equal(dispatched_profile, direct_profile)
    np.testing.assert_array_equal(dispatched_planform, direct_planform)


def test_public_hillshade_dispatches_to_rust_for_float32():
    dem = np.full((16, 16), 10.0, dtype=np.float32)
    dem[8, 8] = 15.0
    dispatched = hillshade(dem, 1.0, azimuth=200, altitude=60)
    direct = terra_texture_rs.hillshade(dem, 1.0, 200.0, 60.0)
    np.testing.assert_array_equal(dispatched, direct)


def test_public_curvatures_coerces_float64_input_and_still_dispatches_to_rust():
    """Unlike blend.py's soft_light()/luminosity_blend() (which check the
    caller's *actual* dtype before dispatching), curvatures() calls
    `np.asarray(dem, dtype=np.float32)` unconditionally at the top,
    before the fast-path guard ever runs -- so a float64 DEM is already
    float32 by the time `_is_fast_path_2d_f32()` looks at it, and still
    takes the Rust path. There's no dtype-based fallback to observe here
    the way there is in blend.py; only the *return* dtype is always
    float32 regardless of what went in. This documents that existing
    behaviour rather than asserting a fallback that can't actually
    happen this way."""
    dem64 = np.full((16, 16), 10.0, dtype=np.float64)
    dem64[8, 8] = 15.0
    profile, planform = curvatures(dem64, 1.0)
    assert profile.dtype == np.float32
    assert planform.dtype == np.float32
    numpy_profile, numpy_planform = _numpy_curvatures(dem64.astype(np.float32), 1.0)
    np.testing.assert_allclose(profile, numpy_profile)
    np.testing.assert_allclose(planform, numpy_planform)


def test_curvatures_numpy_path_still_correct_when_rust_unavailable(monkeypatch):
    """The dispatch guard's *actual* fallback condition is `_rust is
    None` (extension not built/importable) -- dtype/ndim checks are
    structurally almost always true given curvatures() always hands the
    guard an already-float32-coerced 2D array (see the test above).
    Simulate the extension being unavailable and confirm the pure-numpy
    branch alone still produces the right answer."""
    import TerraTexture.derivatives as derivatives_module

    monkeypatch.setattr(derivatives_module, "_rust", None)
    dem = np.full((16, 16), 10.0, dtype=np.float32)
    dem[8, 8] = 15.0
    profile, planform = curvatures(dem, 1.0)
    numpy_profile, numpy_planform = _numpy_curvatures(dem, 1.0)
    np.testing.assert_allclose(profile, numpy_profile)
    np.testing.assert_allclose(planform, numpy_planform)


def test_hillshade_numpy_path_still_correct_when_rust_unavailable(monkeypatch):
    import TerraTexture.derivatives as derivatives_module

    monkeypatch.setattr(derivatives_module, "_rust", None)
    dem = np.full((16, 16), 10.0, dtype=np.float32)
    dem[8, 8] = 15.0
    out = hillshade(dem, 1.0, azimuth=200, altitude=60)

    az = np.float32(np.radians(360.0 - 200 + 90))
    alt = np.float32(np.radians(60.0))
    zy, zx = np.gradient(dem, 1.0)
    slope = np.pi / 2 - np.arctan(np.hypot(zx, zy))
    aspect = np.arctan2(-zx, zy)
    numpy_out = np.clip(
        np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect), 0, 1,
    )
    np.testing.assert_allclose(out, numpy_out)


def test_rust_and_numpy_paths_preserve_nan_mask_identically():
    """Both curvatures() dispatch paths share the same NaN nearest-fill
    + re-mask logic in derivatives.py itself (only the clean-array math
    differs), so this should hold regardless of which kernel ran."""
    dem = np.full((30, 30), 50.0, dtype=np.float32)
    dem[10:15, 10:15] = np.nan
    profile, planform = curvatures(dem, 1.0)
    assert np.all(np.isnan(profile[10:15, 10:15]))
    assert np.all(np.isnan(planform[10:15, 10:15]))
    assert not np.any(np.isnan(profile[:5, :5]))
