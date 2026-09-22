"""
Tests for :mod:`TerraTexture.basemap` (relief draped over imagery).

No network access. The STAC fetch is replaced through ``_AOI_PRODUCTS``
and ``contextily.bounds2img`` by a fake tile server that behaves like
the real one: it returns RGBA ``uint8`` tiles in EPSG:3857 covering the
requested lon/lat bounds. The imagery therefore really is warped onto
the DEM grid. ``contextily.set_cache_dir`` is patched so tests never
change global state.

Real geospatial correctness of the pieces is covered elsewhere
(``test_sources.py``, ``test_derivatives.py``, ``test_blend.py``); these
tests check how :mod:`basemap` wires them together.

Covers:

- CRS handling: ``aoi_bounds_crs`` vs ``target_crs``, per-product
  defaults, and ``dem_path`` DEMs used in their own CRS or reprojected.
- DEM sources: AOI query, single file, tile list, missing CRS.
- Imagery: RGBA handling, coverage warnings and black fill, fetch
  errors wrapped in :class:`BasemapFetchError`.
- Compositing: the returned layers and ``relief_strength``.
- Validation, logging warnings, saving, caching, profiling, and
  :func:`add_relief_basemap`.

Dependencies:
    contextily and rasterio (the ``basemap`` extra). Skipped entirely
    without contextily.

Examples:
    Run just these tests::

        pytest tests/test_basemap.py -v
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("contextily")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import rasterio  # noqa: E402
from rasterio.transform import from_bounds, from_origin  # noqa: E402
from rasterio.warp import transform_bounds  # noqa: E402

import TerraTexture.basemap as basemap_module  # noqa: E402
from TerraTexture.basemap import (  # noqa: E402
    BasemapFetchError,
    _StageTimer,
    add_relief_basemap,
    plot_dem_basemap_luminosity_relief,
)
from TerraTexture.blend import _lum  # noqa: E402
from TerraTexture.io import load_dem  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes and fixtures
# ---------------------------------------------------------------------------

class _FakeProvider:
    """Tile provider stand-in with the attributes basemap.py reads."""

    name = "Fake.Imagery"
    attribution = "(c) Fake Imagery"


def _synthetic_dem(shape: tuple[int, int] = (40, 50)) -> np.ndarray:
    """
    The package's synthetic demo DEM, with a small nodata void.

    Args:
        shape (tuple[int, int]): DEM shape.

    Returns:
        np.ndarray: float32 DEM.
    """
    dem, _ = load_dem(shape=shape)
    dem[3:6, 4:8] = np.nan
    return dem


@pytest.fixture(autouse=True)
def _close_figures() -> Any:
    """
    Close every figure after each test.

    Yields:
        None
    """
    yield
    plt.close("all")


@pytest.fixture
def services(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """
    Replace the STAC fetch, tile server and tile cache with fakes.

    The fake AOI fetch records its arguments and returns the synthetic
    DEM in ``target_crs``. The fake tile server returns a random RGBA
    image in EPSG:3857 covering the requested bounds (5 % margin), like
    ``contextily.bounds2img``. Set ``services.coverage`` below 1 to make
    the imagery cover only the western part of the request.

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest's monkeypatch fixture.

    Returns:
        SimpleNamespace: ``fetch_calls``, ``tile_calls``, ``cache_dirs``
            (lists of recorded calls) and ``coverage`` (float).
    """
    state = SimpleNamespace(
        fetch_calls=[], tile_calls=[], cache_dirs=[], coverage=1.0
    )

    def _fetch(
        bounds: Any, resolution: Any = None, bbox_crs: Any = None,
        target_crs: Any = None,
    ) -> tuple[np.ndarray, float, Any, Any]:
        state.fetch_calls.append({
            "bounds": bounds, "resolution": resolution,
            "bbox_crs": bbox_crs, "target_crs": target_crs,
        })
        transform = from_bounds(0, 0, 500, 400, 50, 40)
        return _synthetic_dem(), 10.0, transform, target_crs

    def _bounds2img(
        west: float, south: float, east: float, north: float, **kwargs: Any,
    ) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        state.tile_calls.append({"bounds": (west, south, east, north), **kwargs})
        x0, y0, x1, y1 = transform_bounds(
            "EPSG:4326", "EPSG:3857", west, south, east, north
        )
        pad = 0.05 * (x1 - x0)
        x0, x1, y0, y1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad
        if state.coverage < 1:
            x1 = x0 + (x1 - x0) * state.coverage
        tile = (np.random.default_rng(1).random((64, 80, 4)) * 255).astype(np.uint8)
        return tile, (x0, x1, y0, y1)

    monkeypatch.setitem(basemap_module._AOI_PRODUCTS, "arcticdem", _fetch)
    monkeypatch.setitem(basemap_module._AOI_PRODUCTS, "rema", _fetch)
    monkeypatch.setattr("contextily.bounds2img", _bounds2img)
    monkeypatch.setattr("contextily.set_cache_dir", state.cache_dirs.append)
    return state


def _run(**kwargs: Any) -> tuple[Any, Any, dict[str, Any]]:
    """
    Call the plot function with test defaults (fake provider, no show).

    Args:
        **kwargs (Any): Arguments overriding the defaults.

    Returns:
        tuple: ``(fig, ax, layers)``.
    """
    options: dict[str, Any] = {
        "aoi_bounds": (-10, 50, -5, 55), "source": _FakeProvider(), "show": False,
    }
    options.update(kwargs)
    if "dem_path" in kwargs:
        options.pop("aoi_bounds")
    return plot_dem_basemap_luminosity_relief(**options)


def _write_dem(
    path: Path,
    crs: str | None = "EPSG:32633",
    transform: Any = None,
    shape: tuple[int, int] = (40, 50),
) -> Path:
    """
    Write the synthetic DEM to a GeoTIFF (nodata -9999).

    Args:
        path (Path): Output path.
        crs (str | None): CRS to record, or ``None`` for none.
        transform (Any): Affine transform; defaults to 10 m UTM pixels.
        shape (tuple[int, int]): DEM shape.

    Returns:
        Path: ``path``.
    """
    dem = np.nan_to_num(_synthetic_dem(shape), nan=-9999.0)
    if transform is None:
        transform = from_origin(500_000, 6_000_400, 10, 10)
    with rasterio.open(
        path, "w", driver="GTiff", height=shape[0], width=shape[1], count=1,
        dtype="float32", crs=crs, transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(dem, 1)
    return path


# ---------------------------------------------------------------------------
# CRS handling (original regression tests)
# ---------------------------------------------------------------------------

def test_aoi_bounds_crs_independent_of_target_crs(services: SimpleNamespace) -> None:
    """lon/lat bounds combine with any target_crs.

    Regression test: aoi_bounds_crs used to be forced to equal target_crs.
    """
    _run(aoi_bounds_crs="EPSG:4326", target_crs="EPSG:3031")

    call = services.fetch_calls[0]
    assert call["bbox_crs"] == "EPSG:4326"
    assert call["target_crs"] == "EPSG:3031"
    assert call["bounds"] == (-10, 50, -5, 55)


def test_aoi_bounds_crs_defaults_to_4326(services: SimpleNamespace) -> None:
    """Bounds are treated as lon/lat unless told otherwise."""
    _run()

    assert services.fetch_calls[0]["bbox_crs"] == "EPSG:4326"


def test_tile_reproject_handles_rgba_tiles_correctly(services: SimpleNamespace) -> None:
    """RGBA tiles are reduced to RGB before warping.

    Regression test: a 4-band source against a 3-band destination made
    reproject() raise "Invalid destination shape".
    """
    _fig, _ax, layers = _run()

    assert layers["basemap"].shape[-1] == 3
    assert layers["final"].shape[-1] == 3


@pytest.mark.parametrize(
    ("product", "expected_crs"),
    [("arcticdem", "EPSG:3413"), ("rema", "EPSG:3031")],
)
def test_default_target_crs_is_product_native(
    services: SimpleNamespace, product: str, expected_crs: str,
) -> None:
    """Without target_crs, each product composites in its native CRS."""
    _fig, _ax, layers = _run(dem_product=product)

    assert services.fetch_calls[0]["target_crs"] == expected_crs
    assert layers["crs"] == expected_crs


def test_resolution_is_passed_to_fetch(services: SimpleNamespace) -> None:
    """arcticdem_resolution reaches the mosaic fetch (for any product)."""
    _run(dem_product="rema", arcticdem_resolution=10)

    assert services.fetch_calls[0]["resolution"] == 10


def test_default_source_is_esri_world_imagery(services: SimpleNamespace) -> None:
    """With no source, Esri World Imagery tiles are requested."""
    _run(source=None)

    assert "WorldImagery" in services.tile_calls[0]["source"].name


def test_zoom_and_connections_reach_contextily(services: SimpleNamespace) -> None:
    """zoom and tile_connections are passed through to bounds2img."""
    _run(zoom=12, tile_connections=4)

    call = services.tile_calls[0]
    assert (call["zoom"], call["n_connections"], call["ll"]) == (12, 4, True)


# ---------------------------------------------------------------------------
# DEM from dem_path
# ---------------------------------------------------------------------------

def test_dem_path_used_unmodified_in_its_own_crs(
    services: SimpleNamespace, tmp_path: Path,
) -> None:
    """With no target_crs, a file DEM keeps its CRS, grid and shape."""
    path = _write_dem(tmp_path / "dem.tif")

    _fig, _ax, layers = _run(dem_path=str(path))

    with rasterio.open(path) as src:
        assert layers["transform"] == src.transform
    assert layers["crs"] == "EPSG:32633"
    assert layers["shape"] == (40, 50)
    assert services.fetch_calls == []


def test_dem_path_reprojected_to_explicit_target(
    services: SimpleNamespace, tmp_path: Path,
) -> None:
    """A different target_crs reprojects the DEM onto that CRS."""
    path = _write_dem(tmp_path / "dem.tif")

    _fig, _ax, layers = _run(dem_path=path, target_crs="EPSG:3857")

    assert layers["crs"] == "EPSG:3857"
    assert layers["final"].shape[:2] == layers["shape"]


def test_dem_path_matching_target_is_not_reprojected(
    services: SimpleNamespace, tmp_path: Path,
) -> None:
    """A target_crs equal to the DEM's (any spelling) keeps the grid."""
    path = _write_dem(tmp_path / "dem.tif")

    _fig, _ax, layers = _run(dem_path=path, target_crs="epsg:32633")

    assert layers["shape"] == (40, 50)


def test_dem_path_tile_list_is_merged(
    services: SimpleNamespace, tmp_path: Path,
) -> None:
    """A list of adjacent tiles is mosaicked into one DEM."""
    left = _write_dem(tmp_path / "left.tif")
    right = _write_dem(
        tmp_path / "right.tif", transform=from_origin(500_500, 6_000_400, 10, 10)
    )

    _fig, _ax, layers = _run(dem_path=[left, right])

    assert layers["shape"] == (40, 100)


def test_dem_without_crs_raises(services: SimpleNamespace, tmp_path: Path) -> None:
    """Imagery can't be aligned to an un-georeferenced DEM."""
    path = _write_dem(tmp_path / "nocrs.tif", crs=None)

    with pytest.raises(ValueError, match="no CRS"):
        _run(dem_path=path)

    assert services.tile_calls == []


def test_invalid_target_crs_raises(services: SimpleNamespace, tmp_path: Path) -> None:
    """An unparseable target_crs raises ValueError naming it."""
    path = _write_dem(tmp_path / "dem.tif")

    with pytest.raises(ValueError, match="EPSG:999999"):
        _run(dem_path=path, target_crs="EPSG:999999")


# ---------------------------------------------------------------------------
# Imagery: coverage and fetch errors
# ---------------------------------------------------------------------------

def test_full_coverage_does_not_warn(
    services: SimpleNamespace, caplog: pytest.LogCaptureFixture,
) -> None:
    """Imagery covering the whole grid logs no coverage warning."""
    with caplog.at_level(logging.WARNING, logger=basemap_module.__name__):
        _run()

    assert "covers only" not in caplog.text


def test_partial_coverage_warns_and_fills_black(
    services: SimpleNamespace, caplog: pytest.LogCaptureFixture,
) -> None:
    """Uncovered cells are black (not garbage), with a warning.

    Regression test: the destination was np.empty, so uncovered cells
    held whatever memory happened to contain.
    """
    services.coverage = 0.5
    # EPSG:3857 keeps the DEM grid aligned with the tiles' x axis, so the
    # "western half" of the imagery maps onto the DEM's western columns.
    with caplog.at_level(logging.WARNING, logger=basemap_module.__name__):
        _f1, _a1, first = _run(target_crs="EPSG:3857")
        _f2, _a2, second = _run(target_crs="EPSG:3857")

    assert "covers only" in caplog.text
    basemap = first["basemap"]
    assert np.all(basemap[:, -5:] == 0.0)  # eastern edge: no imagery
    assert basemap[:, :5].max() > 0.0      # western edge: imagery
    np.testing.assert_array_equal(basemap, second["basemap"])


def test_tile_fetch_failure_raises_basemap_fetch_error(
    services: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Network errors become BasemapFetchError naming provider and zoom."""
    original = ConnectionError("tile server unreachable")

    def _fail(*args: Any, **kwargs: Any) -> None:
        raise original

    monkeypatch.setattr("contextily.bounds2img", _fail)

    with pytest.raises(BasemapFetchError, match="Fake.Imagery.*zoom=7") as exc_info:
        _run(zoom=7)

    assert exc_info.value.__cause__ is original
    assert "tile server unreachable" in str(exc_info.value)


def test_non_rgb_tile_image_raises(
    services: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A greyscale (2-D) tile image is rejected clearly."""
    monkeypatch.setattr(
        "contextily.bounds2img",
        lambda *a, **k: (np.zeros((8, 8), np.uint8), (0, 1, 0, 1)),
    )

    with pytest.raises(BasemapFetchError, match=r"shape \(8, 8\)"):
        _run()


# ---------------------------------------------------------------------------
# Layers and compositing
# ---------------------------------------------------------------------------

def test_layers_have_expected_keys_shapes_and_ranges(services: SimpleNamespace) -> None:
    """Every documented layer is present, on the DEM grid, in [0, 1]."""
    _fig, _ax, layers = _run()

    height, width = layers["shape"]
    for key in ("dem_grey", "curvature", "hillshade", "relief_luminosity",
                "texture_luminosity"):
        assert layers[key].shape == (height, width), key
    for key in ("basemap", "luminosity_composite", "final"):
        assert layers[key].shape == (height, width, 3), key
        assert 0.0 <= layers[key].min() and layers[key].max() <= 1.0, key
    assert not np.isnan(layers["relief_luminosity"]).any()
    assert not np.isnan(layers["texture_luminosity"]).any()
    assert len(layers["extent"]) == 4
    assert isinstance(layers["crs"], str)


def test_layers_include_stage_timings(services: SimpleNamespace) -> None:
    """Per-stage timings are always returned, whether or not profiling."""
    _fig, _ax, layers = _run()

    assert set(layers["timings"]) == {
        "dem_fetch", "curvature_compute", "hillshade_compute",
        "stretch_normalize", "relief_blend", "tile_fetch", "tile_reproject",
        "final_blend", "plotting",
    }
    assert all(seconds >= 0 for seconds in layers["timings"].values())


@pytest.mark.parametrize("strength", [0.0, 0.4, 1.0])
def test_relief_strength_sets_composite_luminosity(
    services: SimpleNamespace, strength: float,
) -> None:
    """The composite's lightness is the relief/imagery mix requested.

    Luminosity blending preserves the target lightness exactly, so
    Lum(composite) = strength * relief + (1 - strength) * Lum(basemap).
    """
    _fig, _ax, layers = _run(relief_strength=strength)

    expected = (
        strength * layers["relief_luminosity"]
        + (1 - strength) * _lum(layers["basemap"])
    )
    np.testing.assert_allclose(
        _lum(layers["luminosity_composite"]), expected, atol=1e-5
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"aoi_bounds": None}, "Provide either"),
        ({"dem_path": "x.tif", "aoi_bounds": (0, 0, 1, 1)}, "not both"),
        ({"dem_product": "greenland"}, "dem_product"),
        ({"dem_path": [], "aoi_bounds": None}, "empty list"),
        ({"zoom": -1}, "zoom"),
        ({"zoom": 30}, "zoom"),
        ({"zoom": 1.5}, "zoom"),
        ({"zoom": True}, "zoom"),
        ({"zoom": "high"}, "zoom"),
        ({"tile_connections": 0}, "tile_connections"),
        ({"tile_connections": 2.5}, "tile_connections"),
        ({"relief_strength": -0.1}, "relief_strength"),
        ({"relief_strength": 1.5}, "relief_strength"),
        ({"relief_strength": "0.5"}, "relief_strength"),
    ],
    ids=[
        "neither-source", "both-sources", "bad-product", "empty-tile-list",
        "zoom-negative", "zoom-too-high", "zoom-float", "zoom-bool",
        "zoom-string", "connections-zero", "connections-float",
        "strength-negative", "strength-over-1", "strength-string",
    ],
)
def test_invalid_options_raise_before_any_fetch(
    services: SimpleNamespace, kwargs: dict[str, Any], match: str,
) -> None:
    """Bad arguments raise ValueError before any DEM or tile request."""
    options: dict[str, Any] = {
        "aoi_bounds": (-10, 50, -5, 55), "source": _FakeProvider(), "show": False,
    }
    options.update(kwargs)

    with pytest.raises(ValueError, match=match):
        plot_dem_basemap_luminosity_relief(**options)

    assert services.fetch_calls == [] and services.tile_calls == []


# ---------------------------------------------------------------------------
# Grid warnings
# ---------------------------------------------------------------------------

def test_geographic_crs_warns(
    services: SimpleNamespace, tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """Compositing in degrees logs a warning about wrong derivatives."""
    path = _write_dem(
        tmp_path / "lonlat.tif", crs="EPSG:4326",
        transform=from_origin(14.0, 55.0, 0.001, 0.001),
    )

    with caplog.at_level(logging.WARNING):
        _run(dem_path=path)

    assert "is geographic" in caplog.text


def test_non_square_pixels_warn(
    services: SimpleNamespace, tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """Rectangular pixels log a warning (derivatives assume square)."""
    path = _write_dem(
        tmp_path / "rect.tif", transform=from_origin(500_000, 6_000_800, 10, 20)
    )

    with caplog.at_level(logging.WARNING, logger=basemap_module.__name__):
        _run(dem_path=path)

    assert "Non-square pixels" in caplog.text


# ---------------------------------------------------------------------------
# Output: plotting, saving, caching, profiling
# ---------------------------------------------------------------------------

def test_out_fig_is_saved_and_logged(
    services: SimpleNamespace, tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """out_fig writes the file and logs at INFO (instead of printing)."""
    out = tmp_path / "relief.png"

    with caplog.at_level(logging.INFO, logger=basemap_module.__name__):
        _run(out_fig=out, figsize=(2, 2))

    assert out.stat().st_size > 0
    assert f"Saved figure to {out}" in caplog.text


def test_unwritable_out_fig_raises_and_closes_figure(services: SimpleNamespace) -> None:
    """A failed save raises OSError without leaking the figure."""
    with pytest.raises(OSError):
        _run(out_fig="/nonexistent/dir/relief.png")

    assert plt.get_fignums() == []


def test_show_calls_plt_show(services: SimpleNamespace) -> None:
    """show=True displays the figure; show=False doesn't."""
    with patch.object(plt, "show") as mock_show:
        _run(show=False)
        mock_show.assert_not_called()
        _run(show=True)
        mock_show.assert_called_once()


def test_attribution_is_drawn(services: SimpleNamespace) -> None:
    """The provider's attribution appears on the axes when non-empty."""
    _fig, ax, _layers = _run()

    assert [t.get_text() for t in ax.texts] == ["(c) Fake Imagery"]


def test_empty_attribution_is_not_drawn(services: SimpleNamespace) -> None:
    """No text box is added for a provider without attribution."""
    provider = SimpleNamespace(name="Bare", attribution="")

    _fig, ax, _layers = _run(source=provider)

    assert len(ax.texts) == 0


def test_tile_cache_dir_is_created_and_set(
    services: SimpleNamespace, tmp_path: Path,
) -> None:
    """tile_cache_dir is created and handed to contextily."""
    cache = tmp_path / "tiles" / "nested"

    _run(tile_cache_dir=str(cache))

    assert cache.is_dir()
    assert services.cache_dirs == [str(cache)]


def test_uncreatable_tile_cache_dir_raises(
    services: SimpleNamespace, tmp_path: Path,
) -> None:
    """A cache path blocked by a file raises OSError naming it."""
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")

    with pytest.raises(OSError, match="tile cache directory"):
        _run(tile_cache_dir=str(blocker / "cache"))


def test_profile_prints_stage_table(
    services: SimpleNamespace, capsys: pytest.CaptureFixture[str],
) -> None:
    """profile=True prints the timing table; profile=False prints nothing."""
    _run(profile=False)
    assert capsys.readouterr().out == ""

    _run(profile=True)
    out = capsys.readouterr().out
    assert "stage" in out and "TOTAL" in out and "tile_reproject" in out


# ---------------------------------------------------------------------------
# add_relief_basemap
# ---------------------------------------------------------------------------

def test_add_relief_basemap_draws_on_every_axis(services: SimpleNamespace) -> None:
    """The same image lands on each axis at the requested zorder."""
    fig, axes = plt.subplots(1, 3)

    layers = add_relief_basemap(
        list(axes[:2]), aoi_bounds=(-10, 50, -5, 55), source=_FakeProvider(),
        zorder=-1,
    )

    for ax in axes[:2]:
        (image,) = ax.images
        np.testing.assert_array_equal(image.get_array(), layers["final"])
        assert image.get_zorder() == -1
        assert tuple(image.get_extent()) == pytest.approx(layers["extent"])
    assert len(axes[2].images) == 0
    assert len(services.tile_calls) == 1  # built once, not per axis
    assert plt.get_fignums() == [fig.number]  # throwaway figure closed


def test_add_relief_basemap_accepts_single_axes(services: SimpleNamespace) -> None:
    """A lone Axes (not in a list) works too."""
    _fig, ax = plt.subplots()

    add_relief_basemap(ax, aoi_bounds=(-10, 50, -5, 55), source=_FakeProvider())

    assert len(ax.images) == 1


def test_add_relief_basemap_passes_relief_strength(services: SimpleNamespace) -> None:
    """relief_strength reaches the composite (it was previously dropped)."""
    _fig, ax = plt.subplots()

    layers = add_relief_basemap(
        ax, aoi_bounds=(-10, 50, -5, 55), source=_FakeProvider(),
        relief_strength=0.3,
    )

    _f, _a, direct = _run(relief_strength=0.3)
    np.testing.assert_allclose(layers["final"], direct["final"])


@pytest.mark.parametrize(
    ("axes", "error", "match"),
    [([], ValueError, "empty"), (["not an axis"], TypeError, "matplotlib Axes")],
    ids=["empty", "wrong-type"],
)
def test_add_relief_basemap_rejects_bad_axes(
    services: SimpleNamespace, axes: Any, error: type[Exception], match: str,
) -> None:
    """Bad axes raise before anything is fetched."""
    with pytest.raises(error, match=match):
        add_relief_basemap(axes, aoi_bounds=(-10, 50, -5, 55))

    assert services.fetch_calls == []


# ---------------------------------------------------------------------------
# _StageTimer
# ---------------------------------------------------------------------------

def test_stage_timer_accumulates_repeated_stages() -> None:
    """Two blocks with the same name add up; order is first use."""
    timer = _StageTimer()

    with timer("a"):
        pass
    with timer("b"):
        pass
    with timer("a"):
        pass

    assert list(timer.stages) == ["a", "b"]


def test_stage_timer_supports_nesting() -> None:
    """Nested stages each get their own timing.

    Regression test: the old timer kept one shared start time, so a
    nested block overwrote the outer one's.
    """
    timer = _StageTimer()

    with timer("outer"):
        with timer("inner"):
            sum(range(10_000))

    assert timer.stages["outer"] >= timer.stages["inner"] > 0


def test_stage_timer_records_time_when_block_raises() -> None:
    """A failing stage is still timed, and the exception propagates."""
    timer = _StageTimer()

    with pytest.raises(RuntimeError):
        with timer("boom"):
            raise RuntimeError("stage failed")

    assert "boom" in timer.stages


def test_stage_timer_report_sorted_slowest_first() -> None:
    """The report lists stages by descending time, with a total."""
    timer = _StageTimer()
    timer.stages.update({"fast": 0.1, "slow": 0.9})

    lines = timer.format_report().splitlines()

    assert lines[2].startswith("slow") and lines[3].startswith("fast")
    assert lines[-1].startswith("TOTAL")
    assert "100.0%" in lines[-1]


def test_stage_timer_empty_report_is_blank(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Nothing timed means no table and no output."""
    timer = _StageTimer()

    assert timer.format_report() == ""
    timer.report()
    assert capsys.readouterr().out == ""
