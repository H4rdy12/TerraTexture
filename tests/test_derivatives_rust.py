"""
Rust-vs-numpy parity tests for the ``terra_texture_rs`` derivative kernels.

Checks the compiled ``curvatures`` and ``hillshade`` kernels
(``rust/src/lib.rs``) against independent numpy references, and checks
that :mod:`TerraTexture.derivatives` actually dispatches to them.

Hillshade is also checked against a first-principles ground truth
(unit surface normal dotted with the light vector). The numpy path used
to compute aspect with swapped ``atan2`` arguments, mirroring the light
across the NE-SW axis, and a kernel ported from it inherits the same
bug. Parity with numpy alone can't catch that, because both would be
wrong in the same way. If ``test_rust_hillshade_matches_ground_truth``
fails, change the kernel's aspect from ``atan2(-dz/dx, dz/dy)`` to
``atan2(dz/dy, -dz/dx)`` (in Rust: ``zy.atan2(-zx)``).

Why a separate file:
    The whole module is skipped when the extension isn't built, so any
    numpy-only test placed here would be silently skipped too. Those live
    in ``test_derivatives.py``, including dispatch and fallback tests
    that use fake kernels.

Skip vs. fail:
    - Extension *not installed* -> the module is skipped.
    - Extension installed but *fails to import* (ABI or Python-version
      mismatch, stale build, missing symbol) -> collection **errors**.
      ``pytest.importorskip`` would skip this case too, hiding exactly
      the breakage these tests exist to catch.

Dependencies:
    The built extension: ``uv sync --extra rust`` (or ``maturin develop``
    from ``rust/``).

Examples:
    Run just these tests::

        pytest tests/test_derivatives_rust.py -v
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import TerraTexture.derivatives as derivatives_module
from TerraTexture.derivatives import _derivatives, curvatures, hillshade

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
# Constants and references
# ---------------------------------------------------------------------------

# Rust and numpy use the same maths in float32, not bit-identical code.
_ATOL = 1e-4

# Elements at which the kernels switch to rayon-parallel branches.
_PARALLEL_THRESHOLD = 65_536


def _numpy_curvatures(
    dem: np.ndarray,
    cellsize: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reference curvatures, copied from derivatives.py's numpy branch.

    Kept as a separate copy so these tests don't depend on the dispatch
    code under test also being correct.

    Args:
        dem (np.ndarray): Void-free float32 DEM.
        cellsize (float): Pixel size.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(profile, planform)``.
    """
    p, q, r, t, s = _derivatives(dem, cellsize)
    p2q2 = p ** 2 + q ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        profile = -(r * p ** 2 + 2 * s * p * q + t * q ** 2) / (
            p2q2 * (1 + p2q2) ** 1.5
        )
        planform = -(r * q ** 2 - 2 * s * p * q + t * p ** 2) / (p2q2 ** 1.5)
    flat = p2q2 < 1e-9
    clean = {"nan": 0.0, "posinf": 0.0, "neginf": 0.0}
    profile = np.where(flat, 0.0, np.nan_to_num(profile, **clean))
    planform = np.where(flat, 0.0, np.nan_to_num(planform, **clean))
    return profile, planform


def _numpy_hillshade(
    dem: np.ndarray,
    cellsize: float,
    azimuth: float,
    altitude: float,
) -> np.ndarray:
    """
    Reference hillshade with the ArcGIS aspect, ``atan2(dz/dy, -dz/dx)``.

    Args:
        dem (np.ndarray): Void-free float32 DEM, north-up.
        cellsize (float): Pixel size.
        azimuth (float): Light direction, degrees clockwise from north.
        altitude (float): Light elevation, degrees.

    Returns:
        np.ndarray: Illumination in ``[0, 1]``.
    """
    az = np.float32(np.radians(360.0 - azimuth + 90))
    alt = np.float32(np.radians(altitude))
    zy, zx = np.gradient(dem, cellsize)
    slope = np.pi / 2 - np.arctan(np.hypot(zx, zy))
    aspect = np.arctan2(zy, -zx)
    shaded = (
        np.sin(alt) * np.sin(slope)
        + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    )
    return np.clip(shaded, 0, 1)


def _hillshade_truth(
    dem: np.ndarray,
    cellsize: float,
    azimuth: float,
    altitude: float,
) -> np.ndarray:
    """
    Hillshade from first principles: ``max(0, unit_normal . light)``.

    Uses (east, north, up) axes; shares no formula with either
    implementation apart from the finite-difference gradient.

    Args:
        dem (np.ndarray): North-up DEM.
        cellsize (float): Pixel size.
        azimuth (float): Light direction, degrees clockwise from north.
        altitude (float): Light elevation, degrees.

    Returns:
        np.ndarray: Illumination in ``[0, 1]``.
    """
    dz_dsouth, dz_deast = np.gradient(dem.astype(np.float64), cellsize)
    normal = np.stack([-dz_deast, dz_dsouth, np.ones_like(dz_deast)], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    az, alt = np.radians(azimuth), np.radians(altitude)
    light = np.array([np.sin(az) * np.cos(alt), np.cos(az) * np.cos(alt), np.sin(alt)])
    return np.clip(normal @ light, 0, 1)


def _random_dem(seed: int, shape: tuple[int, int], relief: float) -> np.ndarray:
    """
    Build a reproducible float32 DEM of uniform noise.

    Args:
        seed (int): RNG seed.
        shape (tuple[int, int]): DEM shape.
        relief (float): Elevations span ``[0, relief)``.

    Returns:
        np.ndarray: C-contiguous float32 DEM.
    """
    return (np.random.default_rng(seed).random(shape) * relief).astype(np.float32)


def _bump_dem() -> np.ndarray:
    """
    A flat 16 x 16 DEM with one raised cell, so curvature isn't all zero.

    Returns:
        np.ndarray: float32 DEM.
    """
    dem = np.full((16, 16), 10.0, dtype=np.float32)
    dem[8, 8] = 15.0
    return dem


# ---------------------------------------------------------------------------
# Extension sanity
# ---------------------------------------------------------------------------

def test_derivatives_module_loaded_the_extension() -> None:
    """``derivatives._rust`` is the real extension, with no import error."""
    assert derivatives_module._rust is terra_texture_rs
    assert derivatives_module._RUST_IMPORT_ERROR is None


def test_extension_exports_derivative_kernels() -> None:
    """Both kernels exist; a missing one means a stale build."""
    missing = [
        name for name in ("curvatures", "hillshade")
        if not hasattr(terra_texture_rs, name)
    ]

    assert not missing, (
        f"terra_texture_rs lacks {missing}; rebuild with "
        "`uv sync --extra rust --reinstall-package terra-texture-rs`"
    )


# ---------------------------------------------------------------------------
# Curvature kernel parity
# ---------------------------------------------------------------------------

def test_rust_curvatures_matches_numpy() -> None:
    """The kernel matches numpy, returning float32 arrays of the same shape."""
    dem = _random_dem(4, (48, 48), 100.0)

    rust_profile, rust_planform = terra_texture_rs.curvatures(dem, 2.0)
    numpy_profile, numpy_planform = _numpy_curvatures(dem, 2.0)

    assert rust_profile.dtype == rust_planform.dtype == np.float32
    assert rust_profile.shape == rust_planform.shape == dem.shape
    np.testing.assert_allclose(rust_profile, numpy_profile, atol=_ATOL)
    np.testing.assert_allclose(rust_planform, numpy_planform, atol=_ATOL)


def test_rust_curvatures_matches_numpy_at_parallel_threshold() -> None:
    """300 x 300 = 90,000 elements exercises the rayon-parallel branch."""
    dem = _random_dem(5, (300, 300), 200.0)
    assert dem.size >= _PARALLEL_THRESHOLD

    rust_profile, rust_planform = terra_texture_rs.curvatures(dem, 1.7)
    numpy_profile, numpy_planform = _numpy_curvatures(dem, 1.7)

    np.testing.assert_allclose(rust_profile, numpy_profile, atol=_ATOL)
    np.testing.assert_allclose(rust_planform, numpy_planform, atol=_ATOL)


@pytest.mark.parametrize("shape", [(3, 3), (3, 40), (40, 3), (7, 11)])
def test_rust_curvatures_small_and_non_square(shape: tuple[int, int]) -> None:
    """Minimum-size and non-square DEMs, where edge handling dominates."""
    dem = _random_dem(6, shape, 50.0)

    rust_profile, rust_planform = terra_texture_rs.curvatures(dem, 1.0)
    numpy_profile, numpy_planform = _numpy_curvatures(dem, 1.0)

    np.testing.assert_allclose(rust_profile, numpy_profile, atol=_ATOL)
    np.testing.assert_allclose(rust_planform, numpy_planform, atol=_ATOL)


def test_rust_curvatures_dome_analytic_values() -> None:
    """Exact ZT values on a dome: planform 1/x, profile 2/(1+4x^2)^1.5."""
    rows, cols = np.mgrid[-20:21, -20:21].astype(np.float32)
    dem = -(cols ** 2 + rows ** 2)

    profile, planform = terra_texture_rs.curvatures(dem, 1.0)

    assert planform[20, 26] == pytest.approx(1 / 6, rel=1e-4)
    assert profile[20, 26] == pytest.approx(2 / (1 + 4 * 36) ** 1.5, rel=1e-3)


def test_rust_curvatures_flat_dem_is_exactly_zero() -> None:
    """Flat cells are set to exactly 0, not tiny noise."""
    dem = np.full((16, 16), 50.0, dtype=np.float32)

    profile, planform = terra_texture_rs.curvatures(dem, 1.0)

    assert np.all(profile == 0.0)
    assert np.all(planform == 0.0)


# ---------------------------------------------------------------------------
# Hillshade kernel parity and ground truth
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("azimuth", [0, 90, 200, 315])
@pytest.mark.parametrize("altitude", [20, 45, 80])
def test_rust_hillshade_matches_numpy(azimuth: float, altitude: float) -> None:
    """The kernel matches the (corrected) numpy reference."""
    dem = _random_dem(7, (48, 48), 100.0)

    rust_out = terra_texture_rs.hillshade(dem, 2.0, float(azimuth), float(altitude))

    assert rust_out.dtype == np.float32
    np.testing.assert_allclose(
        rust_out, _numpy_hillshade(dem, 2.0, azimuth, altitude), atol=_ATOL
    )


@pytest.mark.parametrize("azimuth", [0, 45, 90, 135, 180, 225, 270, 315])
def test_rust_hillshade_matches_ground_truth(azimuth: float) -> None:
    """The kernel lights terrain from the requested direction.

    Fails if the kernel still uses the old swapped aspect, which mirrors
    the light across the NE-SW axis (see the module docstring for the
    one-line fix).
    """
    dem = np.random.default_rng(azimuth).random((25, 30)).cumsum(0).cumsum(1)
    dem = dem.astype(np.float32)

    rust_out = terra_texture_rs.hillshade(dem, 2.0, float(azimuth), 45.0)

    np.testing.assert_allclose(
        rust_out, _hillshade_truth(dem, 2.0, azimuth, 45.0), atol=_ATOL
    )


def test_rust_hillshade_matches_numpy_at_parallel_threshold() -> None:
    """300 x 300 exercises any parallel branch in the hillshade kernel."""
    dem = _random_dem(8, (300, 300), 300.0)

    rust_out = terra_texture_rs.hillshade(dem, 1.5, 315.0, 45.0)

    np.testing.assert_allclose(
        rust_out, _numpy_hillshade(dem, 1.5, 315, 45), atol=_ATOL
    )


def test_rust_hillshade_flat_dem_is_sin_altitude() -> None:
    """Flat ground receives sin(altitude)."""
    dem = np.full((8, 8), 100.0, dtype=np.float32)

    rust_out = terra_texture_rs.hillshade(dem, 1.0, 123.0, 30.0)

    np.testing.assert_allclose(rust_out, 0.5, atol=1e-6)


def test_kernels_do_not_modify_input() -> None:
    """Both kernels write new arrays and leave the DEM untouched."""
    dem = _random_dem(9, (32, 32), 100.0)
    original = dem.copy()

    terra_texture_rs.curvatures(dem, 1.0)
    terra_texture_rs.hillshade(dem, 1.0, 315.0, 45.0)

    np.testing.assert_array_equal(dem, original)


# ---------------------------------------------------------------------------
# Public API dispatch
# ---------------------------------------------------------------------------

def test_public_curvatures_dispatches_to_rust_for_float32() -> None:
    """Public curvatures() takes the fast path: bit-identical to the kernel."""
    dem = _bump_dem()

    dispatched = curvatures(dem, 1.0)
    direct = terra_texture_rs.curvatures(dem, 1.0)

    np.testing.assert_array_equal(dispatched[0], direct[0])
    np.testing.assert_array_equal(dispatched[1], direct[1])


def test_public_hillshade_dispatches_to_rust_for_float32() -> None:
    """Public hillshade() takes the fast path with the given light."""
    dem = _bump_dem()

    dispatched = hillshade(dem, 1.0, azimuth=200, altitude=60)
    direct = terra_texture_rs.hillshade(dem, 1.0, 200.0, 60.0)

    np.testing.assert_array_equal(dispatched, direct)


@pytest.mark.parametrize("dtype", [np.float64, np.int32])
def test_public_functions_convert_any_dtype_and_still_use_rust(dtype: type) -> None:
    """Every 2-D DEM is converted to float32 first, so Rust always runs.

    Unlike blend.py (which checks the caller's dtype), derivatives.py
    converts to float32 before dispatch, so there's no dtype-based
    fallback here; outputs are always float32.
    """
    dem = _bump_dem().astype(dtype)

    profile, planform = curvatures(dem, 1.0)
    shade = hillshade(dem, 1.0)

    assert profile.dtype == planform.dtype == shade.dtype == np.float32
    direct = terra_texture_rs.curvatures(dem.astype(np.float32), 1.0)
    np.testing.assert_array_equal(profile, direct[0])


@pytest.mark.parametrize(
    "make_view",
    [lambda d: d.T, lambda d: d[::2, ::3], lambda d: np.asfortranarray(d)],
    ids=["transposed", "strided", "fortran-order"],
)
def test_public_functions_handle_non_contiguous_input(
    make_view: Callable[[np.ndarray], np.ndarray],
) -> None:
    """Non-contiguous views are copied, then match the numpy reference."""
    dem = make_view(_random_dem(10, (40, 60), 100.0))

    profile, planform = curvatures(dem, 1.0)
    shade = hillshade(dem, 1.0)

    numpy_profile, numpy_planform = _numpy_curvatures(np.ascontiguousarray(dem), 1.0)
    np.testing.assert_allclose(profile, numpy_profile, atol=_ATOL)
    np.testing.assert_allclose(planform, numpy_planform, atol=_ATOL)
    np.testing.assert_allclose(
        shade, _numpy_hillshade(np.ascontiguousarray(dem), 1.0, 315, 45), atol=_ATOL
    )


def test_rust_path_preserves_nan_mask() -> None:
    """Voids are nearest-filled and re-masked around the kernel."""
    dem = np.full((30, 30), 50.0, dtype=np.float32)
    dem[10:15, 10:15] = np.nan

    profile, planform = curvatures(dem, 1.0)
    shade = hillshade(dem, 1.0)

    for out in (profile, planform, shade):
        assert np.all(np.isnan(out[10:15, 10:15]))
        assert np.isnan(out).sum() == 25


def test_rust_and_numpy_paths_agree(monkeypatch: pytest.MonkeyPatch) -> None:
    """Public results are the same with and without the extension."""
    dem = _random_dem(11, (64, 64), 150.0)
    dem[20:24, 30:35] = np.nan

    with_rust = (*curvatures(dem, 2.0), hillshade(dem, 2.0, 250, 35))
    monkeypatch.setattr(derivatives_module, "_rust", None)
    without = (*curvatures(dem, 2.0), hillshade(dem, 2.0, 250, 35))

    for fast, slow in zip(with_rust, without):
        np.testing.assert_allclose(fast, slow, atol=_ATOL, equal_nan=True)
