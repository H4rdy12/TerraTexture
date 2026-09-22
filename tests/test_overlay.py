"""
Tests for :mod:`TerraTexture.overlay` (burning data onto relief).

Builds a synthetic relief ``layers`` dict directly (no basemap, no
network), so every expected value can be computed by hand.

Covers:

- The Luminosity burn: the composite's lightness equals the (compressed)
  relief lightness while the data's hue is kept.
- ``nan_fill``: relief fill vs. transparent alpha.
- ``luminosity_source`` / ``luminosity_range`` choices.
- Colour scaling: ``vmin`` / ``vmax``, ``norm``, and limits inferred
  from data containing NaN (a regression test: matplotlib's autoscale
  turned one NaN into NaN limits and a single flat colour).
- Reprojection: another grid and CRS, partial and zero overlap,
  ``nearest`` for categorical data.
- Inputs: numpy, ``(1, H, W)`` bands, xarray + rioxarray auto-georef.
- Every validation error, raised before any reprojection.

Dependencies:
    rasterio and matplotlib. xarray / rioxarray tests skip without them.

Examples:
    Run just these tests::

        pytest tests/test_overlay.py -v
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("rasterio")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize, TwoSlopeNorm  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

import TerraTexture.overlay as overlay_module  # noqa: E402
from TerraTexture.blend import _lum  # noqa: E402
from TerraTexture.overlay import burn_data_onto_relief  # noqa: E402


# ---------------------------------------------------------------------------
# Constants, helpers and fixtures
# ---------------------------------------------------------------------------

_H, _W = 30, 40
_CRS = "EPSG:32633"
_TRANSFORM = from_origin(500_000, 6_000_300, 10, 10)


@pytest.fixture
def layers() -> dict[str, Any]:
    """
    A synthetic relief ``layers`` dict on a 30 x 40, 10 m UTM grid.

    ``texture_luminosity`` and ``relief_luminosity`` are distinct
    constants so tests can tell which one was used.

    Returns:
        dict[str, Any]: The layers dict.
    """
    rng = np.random.default_rng(0)
    return {
        "final": rng.random((_H, _W, 3)).astype(np.float32),
        "texture_luminosity": np.full((_H, _W), 0.4, np.float32),
        "relief_luminosity": np.full((_H, _W), 0.7, np.float32),
        "transform": _TRANSFORM,
        "crs": _CRS,
        "shape": (_H, _W),
        "extent": (500_000, 500_400, 6_000_000, 6_000_300),
    }


def _data(seed: int = 1, void: bool = True) -> np.ndarray:
    """
    Random data on the relief grid, optionally with a NaN void.

    Args:
        seed (int): RNG seed.
        void (bool): Put NaN in rows 2-5, columns 3-8.

    Returns:
        np.ndarray: ``(30, 40)`` float32 data spanning roughly -6 to 6.
    """
    data = (np.random.default_rng(seed).normal(size=(_H, _W)) * 2).astype(np.float32)
    if void:
        data[2:6, 3:9] = np.nan
    return data


def _burn(layers: dict[str, Any], data: Any = None, **kwargs: Any) -> Any:
    """
    Burn with the relief grid's own georeferencing and limits -6..6.

    Args:
        layers (dict[str, Any]): Relief layers.
        data (Any): Data; defaults to :func:`_data`.
        **kwargs (Any): Overrides for :func:`burn_data_onto_relief`.

    Returns:
        tuple: ``(composite, mappable)``.
    """
    options: dict[str, Any] = {
        "data_transform": _TRANSFORM, "data_crs": _CRS, "vmin": -6, "vmax": 6,
    }
    options.update(kwargs)
    return burn_data_onto_relief(layers, _data() if data is None else data, **options)


# ---------------------------------------------------------------------------
# The Luminosity burn
# ---------------------------------------------------------------------------

def test_composite_lightness_is_compressed_relief(layers: dict[str, Any]) -> None:
    """Where data exists, lightness = lo + texture * (hi - lo)."""
    data = _data()

    composite, _sm = _burn(layers, data)

    valid = np.isfinite(data)
    expected = 0.15 + 0.4 * (0.9 - 0.15)
    np.testing.assert_allclose(_lum(composite)[valid], expected, atol=1e-5)


def test_composite_keeps_the_datas_hue(layers: dict[str, Any]) -> None:
    """Channel order (hue) of each pixel matches the colour-mapped data."""
    data = _data(void=False)
    cmap, norm = plt.get_cmap("RdYlBu_r"), Normalize(-6, 6)

    composite, _sm = _burn(layers, data)

    coloured = cmap(norm(data))[..., :3]
    distinct = np.ptp(coloured, axis=-1) > 0.1  # skip near-grey colours
    order_in = np.argsort(coloured[distinct], axis=-1)
    order_out = np.argsort(composite[distinct], axis=-1)
    np.testing.assert_array_equal(order_in[:, 2], order_out[:, 2])  # dominant channel


def test_output_shape_dtype_and_range(layers: dict[str, Any]) -> None:
    """RGB float32 on the relief grid, within [0, 1]."""
    composite, _sm = _burn(layers)

    assert composite.shape == (_H, _W, 3)
    assert composite.dtype == np.float32
    assert 0.0 <= composite.min() and composite.max() <= 1.0


def test_mappable_carries_cmap_and_norm(layers: dict[str, Any]) -> None:
    """The returned ScalarMappable is ready for a colour bar."""
    _composite, sm = _burn(layers, cmap="viridis", vmin=-2, vmax=3)

    assert sm.get_cmap().name == "viridis"
    assert (sm.norm.vmin, sm.norm.vmax) == (-2, 3)
    plt.colorbar(sm, ax=plt.subplots()[1])  # usable without error
    plt.close("all")


def test_accepts_colormap_object(layers: dict[str, Any]) -> None:
    """A Colormap instance works as well as a name."""
    composite, sm = _burn(layers, cmap=plt.get_cmap("magma"))

    assert sm.get_cmap().name == "magma"
    assert composite.shape == (_H, _W, 3)


# ---------------------------------------------------------------------------
# nan_fill
# ---------------------------------------------------------------------------

def test_nan_fill_relief_shows_plain_relief(layers: dict[str, Any]) -> None:
    """NaN data cells show relief_layers['final'] exactly."""
    data = _data()

    composite, _sm = _burn(layers, data)

    void = np.isnan(data)
    np.testing.assert_array_equal(composite[void], layers["final"][void])
    assert not np.array_equal(composite[~void], layers["final"][~void])


def test_nan_fill_transparent_returns_float32_rgba(layers: dict[str, Any]) -> None:
    """Transparent mode adds alpha: 0 in voids, 1 elsewhere, float32.

    The alpha channel used to be float64, doubling the composite's size.
    """
    data = _data()

    composite, _sm = _burn(layers, data, nan_fill="transparent")

    assert composite.shape == (_H, _W, 4)
    assert composite.dtype == np.float32
    np.testing.assert_array_equal(composite[..., 3], np.where(np.isnan(data), 0, 1))


# ---------------------------------------------------------------------------
# Luminosity source and range
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("source", "value"), [("texture", 0.4), ("full", 0.7)], ids=["texture", "full"],
)
def test_luminosity_source_selects_layer(
    layers: dict[str, Any], source: str, value: float,
) -> None:
    """'texture' reads texture_luminosity, 'full' reads relief_luminosity."""
    data = _data(void=False)

    composite, _sm = _burn(
        layers, data, luminosity_source=source, luminosity_range=None
    )

    np.testing.assert_allclose(_lum(composite), value, atol=1e-5)


def test_custom_luminosity_array(layers: dict[str, Any]) -> None:
    """An (H, W) array supplies the lightness directly."""
    ramp = np.tile(np.linspace(0.2, 0.8, _W, dtype=np.float32), (_H, 1))

    composite, _sm = _burn(
        layers, _data(void=False), luminosity_source=ramp, luminosity_range=None
    )

    np.testing.assert_allclose(_lum(composite), ramp, atol=1e-5)


def test_custom_luminosity_range(layers: dict[str, Any]) -> None:
    """(lo, hi) maps texture 0.4 to lo + 0.4 * (hi - lo)."""
    composite, _sm = _burn(layers, _data(void=False), luminosity_range=(0.3, 0.5))

    np.testing.assert_allclose(_lum(composite), 0.3 + 0.4 * 0.2, atol=1e-5)


def test_texture_missing_from_old_layers_gives_hint(layers: dict[str, Any]) -> None:
    """Layers without texture_luminosity explain how to proceed."""
    del layers["texture_luminosity"]

    with pytest.raises(ValueError, match="texture_luminosity.*older"):
        _burn(layers)

    composite, _sm = _burn(layers, luminosity_source="full")  # still works
    assert composite.shape == (_H, _W, 3)


# ---------------------------------------------------------------------------
# Colour scaling
# ---------------------------------------------------------------------------

def test_norm_overrides_vmin_vmax(layers: dict[str, Any]) -> None:
    """An explicit norm wins over vmin / vmax."""
    norm = TwoSlopeNorm(vcenter=0, vmin=-1, vmax=10)

    _composite, sm = _burn(layers, norm=norm, vmin=-6, vmax=6)

    assert sm.norm is norm


def test_missing_limits_inferred_from_finite_data(
    layers: dict[str, Any], caplog: pytest.LogCaptureFixture,
) -> None:
    """Without vmin/vmax, limits come from the finite data and are logged.

    Regression test: matplotlib's autoscale gave NaN limits when the data
    held any NaN, so the whole burn came out as one flat colour.
    """
    data = _data()

    with caplog.at_level(logging.INFO, logger=overlay_module.__name__):
        composite, sm = _burn(layers, data, vmin=None, vmax=None)

    assert sm.norm.vmin == pytest.approx(np.nanmin(data))
    assert sm.norm.vmax == pytest.approx(np.nanmax(data))
    assert "inferred from the data" in caplog.text
    valid = np.isfinite(data)
    assert len(np.unique(composite[valid].reshape(-1, 3), axis=0)) > 50


def test_one_missing_limit_is_inferred(layers: dict[str, Any]) -> None:
    """Only the missing limit is inferred; the given one is kept."""
    data = _data()

    _composite, sm = _burn(layers, data, vmin=-1.0, vmax=None)

    assert sm.norm.vmin == -1.0
    assert sm.norm.vmax == pytest.approx(np.nanmax(data))


@pytest.mark.parametrize(
    ("vmin", "vmax"), [(5, 5), (6, -6), (float("nan"), 1)],
    ids=["equal", "inverted", "nan"],
)
def test_bad_colour_limits_raise(
    layers: dict[str, Any], vmin: float, vmax: float,
) -> None:
    """vmin must be finite and below vmax."""
    with pytest.raises(ValueError, match="vmin < vmax"):
        _burn(layers, vmin=vmin, vmax=vmax)


# ---------------------------------------------------------------------------
# Reprojection
# ---------------------------------------------------------------------------

def test_data_on_another_grid_and_crs_is_reprojected(layers: dict[str, Any]) -> None:
    """Coarser lon/lat data lands on the relief's UTM grid."""
    data = np.full((10, 10), 3.0, np.float32)
    lonlat = from_origin(14.9, 54.2, 0.02, 0.02)  # generously covers the grid

    composite, _sm = burn_data_onto_relief(
        layers, data, data_transform=lonlat, data_crs="EPSG:4326", vmin=-6, vmax=6
    )

    assert composite.shape == (_H, _W, 3)
    assert not np.array_equal(composite, layers["final"])


def test_partial_overlap_fills_the_rest_with_relief(layers: dict[str, Any]) -> None:
    """Relief cells outside the data's extent show the plain relief."""
    west_half = np.ones((_H, _W // 2), np.float32)

    composite, _sm = _burn(layers, west_half)

    np.testing.assert_array_equal(composite[:, -5:], layers["final"][:, -5:])
    assert not np.array_equal(composite[:, :5], layers["final"][:, :5])


def test_no_overlap_warns_and_returns_relief(
    layers: dict[str, Any], caplog: pytest.LogCaptureFixture,
) -> None:
    """Data entirely off the relief grid warns and changes nothing."""
    far_away = from_origin(900_000, 7_000_000, 10, 10)

    with caplog.at_level(logging.WARNING, logger=overlay_module.__name__):
        composite, _sm = _burn(layers, _data(), data_transform=far_away)

    assert "No data overlaps" in caplog.text
    np.testing.assert_array_equal(composite, layers["final"])


def test_nearest_resampling_keeps_categories(layers: dict[str, Any]) -> None:
    """'nearest' keeps categorical values: only their colours appear."""
    classes = np.random.default_rng(3).integers(0, 3, (15, 20)).astype(np.float32)
    coarse = from_origin(500_000, 6_000_300, 20, 20)

    composite, _sm = burn_data_onto_relief(
        layers, classes, data_transform=coarse, data_crs=_CRS,
        vmin=0, vmax=2, resampling="nearest", luminosity_source="texture",
    )

    hues = {tuple(np.round(px / px.max(), 3)) for px in composite.reshape(-1, 3)}
    assert len(hues) <= 3


def test_resampling_enum_accepted(layers: dict[str, Any]) -> None:
    """A rasterio Resampling member works as well as its name."""
    by_enum, _ = _burn(layers, resampling=Resampling.nearest)
    by_name, _ = _burn(layers, resampling="nearest")

    np.testing.assert_array_equal(by_enum, by_name)


# ---------------------------------------------------------------------------
# Data inputs
# ---------------------------------------------------------------------------

def test_single_band_3d_data_is_squeezed(layers: dict[str, Any]) -> None:
    """(1, H, W) data, as rioxarray.open_rasterio returns, is accepted."""
    data = _data()

    three_d, _ = _burn(layers, data[None])
    two_d, _ = _burn(layers, data)

    np.testing.assert_array_equal(three_d, two_d)


def test_integer_data_is_accepted(layers: dict[str, Any]) -> None:
    """Integer rasters are converted to float for NaN-aware handling."""
    integers = np.arange(_H * _W).reshape(_H, _W)

    composite, _sm = _burn(layers, integers, vmin=0, vmax=1199)

    assert composite.shape == (_H, _W, 3)


def test_xarray_with_rioxarray_georeferencing(layers: dict[str, Any]) -> None:
    """A rioxarray DataArray supplies its own transform and CRS."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("rioxarray")
    data = _data()
    ys = 6_000_300 - 5 - 10 * np.arange(_H)
    xs = 500_000 + 5 + 10 * np.arange(_W)
    da = xr.DataArray(data, coords={"y": ys, "x": xs}, dims=("y", "x"))
    da = da.rio.write_crs(_CRS)

    from_xarray, _ = burn_data_onto_relief(layers, da, vmin=-6, vmax=6)
    from_numpy, _ = _burn(layers, data)

    np.testing.assert_array_equal(from_xarray, from_numpy)


def test_xarray_without_crs_explains_fix(layers: dict[str, Any]) -> None:
    """A DataArray lacking a CRS names what's missing and how to set it."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("rioxarray")
    da = xr.DataArray(_data(), dims=("y", "x"))

    with pytest.raises(ValueError, match="data_crs required.*write_crs"):
        burn_data_onto_relief(layers, da, vmin=-6, vmax=6)


# ---------------------------------------------------------------------------
# Validation (all before any reprojection)
# ---------------------------------------------------------------------------

@pytest.fixture
def reproject_spy() -> Any:
    """
    Patch rasterio's reproject to record whether it was called.

    Yields:
        MagicMock: The patched ``rasterio.warp.reproject``.
    """
    with patch("rasterio.warp.reproject") as spy:
        yield spy


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"nan_fill": "white"}, ValueError, "nan_fill"),
        ({"resampling": "bicubicc"}, ValueError, "Unknown resampling"),
        ({"luminosity_source": "elevation"}, ValueError, "luminosity_source"),
        ({"luminosity_source": np.zeros((5, 5))}, ValueError, r"\(5, 5\)"),
        ({"luminosity_range": (0.9, 0.1)}, ValueError, "lo < hi"),
        ({"luminosity_range": (-0.1, 0.5)}, ValueError, "lo < hi"),
        ({"luminosity_range": (0.2, 1.5)}, ValueError, "lo < hi"),
        ({"luminosity_range": (0.5,)}, ValueError, "two numbers"),
        ({"data_transform": None}, ValueError, "data_transform required"),
        ({"data_crs": None}, ValueError, "data_crs required"),
    ],
    ids=[
        "nan-fill", "resampling", "source-name", "source-shape",
        "range-inverted", "range-below-0", "range-above-1", "range-length",
        "no-transform", "no-crs",
    ],
)
def test_invalid_options_raise_before_reprojecting(
    layers: dict[str, Any], reproject_spy: Any,
    kwargs: dict[str, Any], error: type[Exception], match: str,
) -> None:
    """Bad options fail fast, before the (slow) reprojection."""
    with pytest.raises(error, match=match):
        _burn(layers, **kwargs)

    reproject_spy.assert_not_called()


@pytest.mark.parametrize(
    ("data", "error", "match"),
    [
        (np.zeros(10), ValueError, "2-D"),
        (np.zeros((3, 4, 5)), ValueError, "2-D"),
        (np.array([["a", "b"]]), TypeError, "numeric"),
    ],
    ids=["1-d", "multi-band", "strings"],
)
def test_bad_data_raises(
    layers: dict[str, Any], data: Any, error: type[Exception], match: str,
) -> None:
    """Data must be a numeric 2-D grid (or one band)."""
    with pytest.raises(error, match=match):
        _burn(layers, data)


def test_missing_layer_keys_are_all_listed(layers: dict[str, Any]) -> None:
    """Every missing required key is named, with where to get the dict."""
    del layers["transform"], layers["crs"]

    with pytest.raises(
        ValueError, match=r"\['transform', 'crs'\].*add_relief_basemap"
    ):
        _burn(layers)


def test_layers_must_be_a_dict() -> None:
    """Passing something that isn't the layers dict raises TypeError."""
    with pytest.raises(TypeError, match="layers dict"):
        burn_data_onto_relief(np.zeros((3, 3)), _data())


def test_final_shape_mismatch_raises(layers: dict[str, Any]) -> None:
    """An inconsistent layers dict (final vs shape) is caught."""
    layers["final"] = np.zeros((5, 5, 3))

    with pytest.raises(ValueError, match="doesn't match"):
        _burn(layers)


def test_invalid_crs_raises_value_error(layers: dict[str, Any]) -> None:
    """An unparseable data CRS raises ValueError naming it."""
    with pytest.raises(ValueError, match="EPSG:999999"):
        _burn(layers, data_crs="EPSG:999999")
