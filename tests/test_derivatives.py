"""
Tests for :mod:`TerraTexture.derivatives` that never need the extension.

Always runs, including on CI and in installs without a Rust toolchain.
Everything that needs the real ``terra_texture_rs`` lives in
``test_derivatives_rust.py``, which skips when it isn't built.

Every test here forces the pure-numpy path (the autouse fixture sets
``_rust = None``), so the maths tests check the reference implementation
even on machines where Rust is built. Dispatch tests then install fake
kernels on top.

Covers:

- Curvature: exact analytic values on quadratic surfaces (central
  differences are exact there), the sign convention, zero curvature on
  planes, and nodata handling.
- Hillshade: an independent vector-based ground truth (unit surface
  normal dotted with the light vector) across many light directions.
  This is the regression test for the swapped-``atan2`` bug that lit
  terrain from the mirror-image direction.
- Input handling, validation errors and the degrees-cellsize warning.
- Rust dispatch and fallback with fake kernels, and import diagnostics.

Fixtures ``flat_dem``, ``paraboloid_dem`` and ``dome_dem`` come from
``conftest.py``.

Examples:
    Run just these tests::

        pytest tests/test_derivatives.py -v
"""

from __future__ import annotations

import importlib
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

import TerraTexture.derivatives as derivatives_module
from TerraTexture.derivatives import curvatures, hillshade


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def numpy_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Force the pure-numpy path for every test in this file.

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest's monkeypatch fixture.

    Returns:
        None
    """
    monkeypatch.setattr(derivatives_module, "_rust", None)


def _quadratic(sign: float, half: int = 20, cellsize: float = 1.0) -> np.ndarray:
    """
    Build ``z = sign * (x^2 + y^2)`` on a square grid centred on the origin.

    Args:
        sign (float): ``+1`` for a bowl, ``-1`` for a dome.
        half (int): Half-width in cells; the grid is ``2 * half + 1``.
        cellsize (float): Spacing in metres; ``x`` and ``y`` are in metres.

    Returns:
        np.ndarray: float64 DEM, row 0 = north.
    """
    rows, cols = np.mgrid[-half:half + 1, -half:half + 1] * cellsize
    return sign * (cols ** 2 + rows ** 2)


def _hillshade_truth(
    dem: np.ndarray,
    cellsize: float,
    azimuth: float,
    altitude: float,
) -> np.ndarray:
    """
    Hillshade from first principles: ``max(0, unit_normal . light)``.

    Uses (east, north, up) axes. ``np.gradient`` along rows is dz/d(south),
    so dz/d(north) is its negative. Shares no formula with the code under
    test apart from the finite-difference gradient.

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


# ---------------------------------------------------------------------------
# Curvature: original behaviour tests (conftest fixtures)
# ---------------------------------------------------------------------------

def test_flat_dem_has_zero_curvature(flat_dem: tuple[np.ndarray, float]) -> None:
    """A flat DEM has zero curvature everywhere."""
    dem, cellsize = flat_dem

    profile, planform = curvatures(dem, cellsize)

    assert np.allclose(profile, 0.0)
    assert np.allclose(planform, 0.0)


def test_paraboloid_profile_curvature_sign(
    paraboloid_dem: tuple[np.ndarray, float],
) -> None:
    """A bowl is concave in profile (flow accelerates): negative off-centre."""
    dem, cellsize = paraboloid_dem

    profile, _ = curvatures(dem, cellsize)

    centre = dem.shape[0] // 2
    assert profile[centre, centre + 20] < 0


def test_dome_planform_curvature_sign(dome_dem: tuple[np.ndarray, float]) -> None:
    """A dome's contours diverge flow: positive planform off-centre."""
    dem, cellsize = dome_dem

    _, planform = curvatures(dem, cellsize)

    centre = dem.shape[0] // 2
    assert planform[centre, centre + 20] > 0


def test_hillshade_range(paraboloid_dem: tuple[np.ndarray, float]) -> None:
    """Hillshade stays within [0, 1]."""
    dem, cellsize = paraboloid_dem

    shade = hillshade(dem, cellsize)

    assert np.nanmin(shade) >= 0.0
    assert np.nanmax(shade) <= 1.0


# ---------------------------------------------------------------------------
# Curvature: exact analytic values and signs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cellsize", [1.0, 2.5])
def test_dome_curvature_matches_analytic_values(cellsize: float) -> None:
    """Exact ZT values on a dome, where central differences are exact.

    For ``z = -(x^2 + y^2)`` at ``(x, 0)``: ``p = -2x``, ``q = 0``,
    ``r = t = -2``, ``s = 0``, so planform = ``1 / x`` and
    profile = ``2 / (1 + 4 x^2)^1.5``.
    """
    dem = _quadratic(-1.0, cellsize=cellsize)
    centre, offset = 20, 6
    x = offset * cellsize

    profile, planform = curvatures(dem, cellsize)

    assert planform[centre, centre + offset] == pytest.approx(1 / x, rel=1e-4)
    assert profile[centre, centre + offset] == pytest.approx(
        2 / (1 + 4 * x ** 2) ** 1.5, rel=1e-3
    )


@pytest.mark.parametrize(
    ("sign", "expected"),
    [(-1.0, 1), (1.0, -1)],
    ids=["dome-convex", "bowl-concave"],
)
def test_curvature_signs_on_quadratic_surfaces(sign: float, expected: int) -> None:
    """Dome: both curvatures positive (convex). Bowl: both negative."""
    profile, planform = curvatures(_quadratic(sign), 1.0)

    ring = profile[20, 26], planform[20, 26], profile[14, 20], planform[14, 20]
    assert all(np.sign(value) == expected for value in ring)


def test_tilted_plane_has_zero_curvature() -> None:
    """A plane has zero curvature everywhere, edges included."""
    rows, cols = np.mgrid[0:15, 0:20].astype(float)

    profile, planform = curvatures(3.0 * cols - 1.5 * rows + 100, 2.0)

    np.testing.assert_allclose(profile, 0.0, atol=1e-6)
    np.testing.assert_allclose(planform, 0.0, atol=1e-6)


def test_curvature_is_rotation_invariant() -> None:
    """Rotating a symmetric dome 90 degrees rotates its curvature with it."""
    dem = _quadratic(-1.0) + 0.3 * np.mgrid[-20:21, -20:21][1]  # break symmetry

    profile, planform = curvatures(dem, 1.0)
    profile_rot, planform_rot = curvatures(np.rot90(dem), 1.0)

    np.testing.assert_allclose(np.rot90(profile), profile_rot, rtol=1e-4, atol=1e-7)
    np.testing.assert_allclose(np.rot90(planform), planform_rot, rtol=1e-4, atol=1e-7)


# ---------------------------------------------------------------------------
# Hillshade: ground truth and light direction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("azimuth", [0, 45, 90, 135, 180, 225, 270, 315])
@pytest.mark.parametrize("altitude", [20, 45, 70])
def test_hillshade_matches_vector_ground_truth(azimuth: float, altitude: float) -> None:
    """Hillshade equals max(0, normal . light) for every light direction.

    Regression test: swapped ``atan2`` arguments used to mirror the light
    across the NE-SW axis, giving errors up to 1.0.
    """
    rng = np.random.default_rng(azimuth + altitude)
    dem = rng.random((25, 30)).cumsum(0).cumsum(1)

    shade = hillshade(dem, 2.0, azimuth=azimuth, altitude=altitude)

    np.testing.assert_allclose(
        shade, _hillshade_truth(dem, 2.0, azimuth, altitude), atol=1e-5
    )


def test_default_light_brightens_north_west_flank() -> None:
    """Light from the NW (315 deg) lights the NW flank, shades the SE one."""
    shade = hillshade(_quadratic(-1.0) / 20, 1.0)

    north_west, south_east = shade[12, 12], shade[28, 28]
    assert north_west > 0.9
    assert south_east < 0.1


@pytest.mark.parametrize(
    ("azimuth", "lit_row", "lit_col"),
    [(0, 12, 20), (90, 20, 28), (180, 28, 20), (270, 20, 12)],
    ids=["north", "east", "south", "west"],
)
def test_lit_flank_faces_the_light(azimuth: float, lit_row: int, lit_col: int) -> None:
    """The dome flank facing each compass direction is its brightest."""
    shade = hillshade(_quadratic(-1.0) / 20, 1.0, azimuth=azimuth)

    flanks = [shade[12, 20], shade[20, 28], shade[28, 20], shade[20, 12]]
    assert shade[lit_row, lit_col] == max(flanks)


@pytest.mark.parametrize("altitude", [30, 60, 90])
def test_flat_dem_hillshade_is_sin_altitude(altitude: float) -> None:
    """Flat ground receives sin(altitude), whatever the azimuth."""
    shade = hillshade(np.zeros((6, 6)), 1.0, azimuth=123, altitude=altitude)

    np.testing.assert_allclose(shade, np.sin(np.radians(altitude)), atol=1e-6)


def test_azimuth_is_periodic() -> None:
    """Azimuths 360 degrees apart give the same shading."""
    dem = _quadratic(-1.0) / 20

    np.testing.assert_allclose(
        hillshade(dem, 1.0, azimuth=45), hillshade(dem, 1.0, azimuth=405), atol=1e-6
    )


# ---------------------------------------------------------------------------
# Nodata, dtypes and input handling
# ---------------------------------------------------------------------------

def test_curvature_preserves_nan_mask() -> None:
    """Voids stay NaN; cells away from them are not contaminated."""
    dem = np.full((30, 30), 50.0)
    dem[10:15, 10:15] = np.nan

    profile, planform = curvatures(dem, 1.0)

    assert np.all(np.isnan(profile[10:15, 10:15]))
    assert np.all(np.isnan(planform[10:15, 10:15]))
    assert not np.any(np.isnan(profile[:5, :5]))
    assert np.isnan(profile).sum() == 25


def test_hillshade_preserves_nan_mask() -> None:
    """Voids stay NaN in the hillshade, and only there."""
    dem = np.full((30, 30), 50.0)
    dem[5:8, 5:8] = np.nan

    shade = hillshade(dem, 1.0)

    assert np.all(np.isnan(shade[5:8, 5:8]))
    assert np.isnan(shade).sum() == 9


def test_all_nan_dem_gives_all_nan() -> None:
    """An all-NaN DEM returns all-NaN outputs rather than raising."""
    dem = np.full((5, 5), np.nan)

    profile, planform = curvatures(dem, 1.0)

    assert np.isnan(profile).all() and np.isnan(planform).all()
    assert np.isnan(hillshade(dem, 1.0)).all()


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16, np.int32])
def test_outputs_are_float32(dtype: Any) -> None:
    """Any numeric DEM dtype gives float32 outputs."""
    dem = (_quadratic(-1.0) + 1000).astype(dtype)

    profile, planform = curvatures(dem, 1.0)

    assert profile.dtype == planform.dtype == np.float32
    assert hillshade(dem, 1.0).dtype == np.float32


def test_input_is_not_modified() -> None:
    """Voids in the caller's array are not filled in place."""
    dem = _quadratic(-1.0).astype(np.float32)
    dem[3, 3] = np.nan
    original = dem.copy()

    curvatures(dem, 1.0)
    hillshade(dem, 1.0)

    np.testing.assert_array_equal(dem, original)


# ---------------------------------------------------------------------------
# Validation and warnings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("dem", "error", "match"),
    [
        (np.zeros(9), ValueError, "2-D"),
        (np.zeros((2, 5)), ValueError, "at least 3 x 3"),
        (np.zeros((3, 3, 3)), ValueError, "2-D"),
        (np.array([["a"] * 3] * 3), TypeError, "numeric"),
    ],
    ids=["1-d", "too-small", "3-d", "strings"],
)
def test_bad_dem_raises(dem: np.ndarray, error: type[Exception], match: str) -> None:
    """Unusable DEMs raise before any computation, in both functions."""
    with pytest.raises(error, match=match):
        curvatures(dem, 1.0)
    with pytest.raises(error, match=match):
        hillshade(dem, 1.0)


@pytest.mark.parametrize(
    ("cellsize", "match"),
    [
        (0, "positive and finite"),
        (float("nan"), "positive and finite"),
        (float("inf"), "positive and finite"),
        (-2.0, "abs\\(transform.e\\)"),
        ("ten", "must be a number"),
    ],
    ids=["zero", "nan", "inf", "negative", "string"],
)
def test_bad_cellsize_raises(cellsize: Any, match: str) -> None:
    """Invalid cellsizes raise; a negative one hints at transform.e."""
    with pytest.raises(ValueError, match=match):
        curvatures(np.zeros((4, 4)), cellsize)
    with pytest.raises(ValueError, match=match):
        hillshade(np.zeros((4, 4)), cellsize)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"altitude": -5}, "between 0 and 90"),
        ({"altitude": 91}, "between 0 and 90"),
        ({"azimuth": float("nan")}, "azimuth must be finite"),
        ({"azimuth": None}, "must be numbers"),
    ],
    ids=["altitude-negative", "altitude-over-90", "azimuth-nan", "azimuth-none"],
)
def test_bad_light_position_raises(kwargs: dict[str, Any], match: str) -> None:
    """Out-of-range or non-numeric light positions raise ValueError."""
    with pytest.raises(ValueError, match=match):
        hillshade(np.zeros((4, 4)), 1.0, **kwargs)


def test_degree_cellsize_warns(caplog: pytest.LogCaptureFixture) -> None:
    """A cellsize that looks like degrees logs a warning."""
    with caplog.at_level(logging.WARNING, logger=derivatives_module.__name__):
        hillshade(np.zeros((4, 4)), 0.0001)

    assert "looks like degrees" in caplog.text


def test_metric_cellsize_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    """Normal metric cellsizes are silent."""
    with caplog.at_level(logging.WARNING, logger=derivatives_module.__name__):
        curvatures(np.zeros((4, 4)), 0.5)
        hillshade(np.zeros((4, 4)), 30.0)

    assert caplog.records == []


# ---------------------------------------------------------------------------
# Rust dispatch and fallback (fake kernels)
# ---------------------------------------------------------------------------

def test_kernels_receive_contiguous_float32_and_python_floats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any DEM reaches the kernels as C-contiguous float32, scalars as floats."""
    seen: list[tuple[Any, ...]] = []

    def _curvatures(dem: np.ndarray, cellsize: float) -> Any:
        seen.append((dem.dtype, dem.flags.c_contiguous, type(cellsize)))
        return derivatives_module._curvatures_numpy(dem, cellsize)

    def _hillshade(dem: np.ndarray, cellsize: float, az: float, alt: float) -> Any:
        seen.append((dem.dtype, dem.flags.c_contiguous, type(az), type(alt)))
        return derivatives_module._hillshade_numpy(dem, cellsize, az, alt)

    monkeypatch.setattr(
        derivatives_module, "_rust",
        SimpleNamespace(curvatures=_curvatures, hillshade=_hillshade),
    )
    dem = _quadratic(-1.0).T  # float64, non-contiguous

    curvatures(dem, 1)
    hillshade(dem, 1, azimuth=300, altitude=40)

    assert seen == [
        (np.float32, True, float),
        (np.float32, True, float, float),
    ]


def _boom(*args: Any) -> Any:
    """
    Simulate a kernel failure.

    Args:
        *args (Any): Ignored.

    Raises:
        RuntimeError: Always.
    """
    raise RuntimeError("simulated kernel failure")


def test_failing_kernels_fall_back_to_numpy(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """Raising kernels log a warning and the numpy result is returned."""
    dem = _quadratic(-1.0)
    dem[2, 2] = np.nan
    expected_curv = curvatures(dem, 1.0)
    expected_shade = hillshade(dem, 1.0)
    monkeypatch.setattr(
        derivatives_module, "_rust",
        SimpleNamespace(curvatures=_boom, hillshade=_boom),
    )

    with caplog.at_level(logging.WARNING, logger=derivatives_module.__name__):
        got_curv = curvatures(dem, 1.0)
        got_shade = hillshade(dem, 1.0)

    for got, expected in zip((*got_curv, got_shade), (*expected_curv, expected_shade)):
        np.testing.assert_array_equal(got, expected)
    assert caplog.text.count("falling back to numpy") == 2


def test_stale_build_without_kernels_uses_numpy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An extension lacking the kernels is skipped, not called."""
    dem = _quadratic(-1.0)
    expected = hillshade(dem, 1.0)
    monkeypatch.setattr(derivatives_module, "_rust", SimpleNamespace())

    np.testing.assert_array_equal(hillshade(dem, 1.0), expected)


def test_invalid_input_never_reaches_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validation runs before dispatch."""
    calls: list[str] = []
    monkeypatch.setattr(
        derivatives_module, "_rust",
        SimpleNamespace(hillshade=lambda *a: calls.append("hillshade")),
    )

    with pytest.raises(ValueError):
        hillshade(np.zeros((4, 4)), -1.0)

    assert calls == []


# ---------------------------------------------------------------------------
# Import diagnostics (reloading derivatives against fake extensions)
# ---------------------------------------------------------------------------

@pytest.fixture
def reload_derivatives(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """
    Provide a function that reloads ``derivatives`` with a fake extension.

    Restores the real state afterwards. The original ``terra_texture_rs``
    module object (if any) is put back into ``sys.modules`` before the
    final reload: a PyO3 extension can't be initialised twice in one
    process, so it must be reused, never re-imported.

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest's monkeypatch fixture.

    Yields:
        Callable[[str, Path], ModuleType]: Writes the given source as a
            fake ``terra_texture_rs`` in the given directory and reloads
            ``derivatives`` against it.
    """
    original = sys.modules.get("terra_texture_rs")

    def _reload(source: str, directory: Path) -> ModuleType:
        (directory / "terra_texture_rs.py").write_text(source)
        sys.modules.pop("terra_texture_rs", None)
        monkeypatch.syspath_prepend(str(directory))
        return importlib.reload(derivatives_module)

    yield _reload

    monkeypatch.undo()
    sys.modules.pop("terra_texture_rs", None)
    if original is not None:
        sys.modules["terra_texture_rs"] = original
    importlib.reload(derivatives_module)


def test_broken_extension_logs_warning(
    tmp_path: Path, reload_derivatives: Any, caplog: pytest.LogCaptureFixture,
) -> None:
    """An installed-but-broken extension warns and records the reason."""
    with caplog.at_level(logging.DEBUG, logger=derivatives_module.__name__):
        module = reload_derivatives('raise ImportError("symbol not found")\n', tmp_path)

    assert module._rust is None
    assert isinstance(module._RUST_IMPORT_ERROR, ImportError)
    assert "installed but failed to import" in caplog.text


def test_stale_extension_reports_missing_kernel(
    tmp_path: Path, reload_derivatives: Any, caplog: pytest.LogCaptureFixture,
) -> None:
    """A build with only hillshade warns that curvatures is missing."""
    with caplog.at_level(logging.WARNING, logger=derivatives_module.__name__):
        module = reload_derivatives(
            "def hillshade(*args):\n    return None\n", tmp_path
        )

    assert module._rust is not None
    assert "no curvatures kernel" in caplog.text
    assert "no hillshade kernel" not in caplog.text


def test_missing_extension_is_quiet(
    monkeypatch: pytest.MonkeyPatch,
    reload_derivatives: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A simply-not-installed extension logs at DEBUG, never WARNING."""
    monkeypatch.setitem(sys.modules, "terra_texture_rs", None)

    with caplog.at_level(logging.DEBUG, logger=derivatives_module.__name__):
        module = importlib.reload(derivatives_module)

    assert module._rust is None
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert "not installed" in caplog.text
