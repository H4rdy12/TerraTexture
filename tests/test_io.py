"""
Tests for :mod:`TerraTexture.io` (DEM loading, archives and mosaics).

Covers three layers:

- Happy paths: the synthetic demo DEM, single GeoTIFFs, ``.tar.gz``
  archives, and two-tile mosaics (full extent and clipped to bounds).
- Pure helpers: the S3 URL rewrite, archive member selection and NaN
  filling. These need no network or real rasters.
- Error handling: every documented ``Raises`` path, plus the guarantee
  that tiles already opened are closed when another tile fails.

Nothing here touches the network. The ``prefer_s3`` tests monkeypatch
``rasterio.open`` to prove which path is passed to it; they do not prove
that GDAL can actually read a ``/vsis3/`` URL.

Dependencies:
    pytest, numpy and rasterio. The whole module is skipped if rasterio
    is not installed.

Examples:
    Run just this file::

        pytest tests/test_io.py -v
"""

from __future__ import annotations

import io as stdlib_io
import logging
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

# Skip before importing the module under test, so a core-only install
# reports a clean skip instead of a collection error.
rasterio = pytest.importorskip("rasterio")

from rasterio.errors import RasterioIOError  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

import TerraTexture.io as io_module  # noqa: E402
from TerraTexture.io import (  # noqa: E402
    DEMReadError,
    _fill_nan_nearest,
    _https_s3_url_to_vsis3,
    _open_raster,
    _open_raster_sync,
    _pick_dem_member,
    load_dem,
    load_dem_mosaic,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Projected CRS with metre units, so bounds and cellsizes are easy to read.
_TEST_CRS = "EPSG:32633"

# Real PGC URL shape, as returned by PGC's STAC API.
_PGC_REMA_URL = (
    "https://pgc-opendata-dems.s3.us-west-2.amazonaws.com/"
    "rema/mosaics/v2.0/10m/40_10/40_10_10m_v2.0_dem.tif"
)
_PGC_REMA_VSIS3 = (
    "/vsis3/pgc-opendata-dems/rema/mosaics/v2.0/10m/40_10/"
    "40_10_10m_v2.0_dem.tif"
)


# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------

def _write_geotiff(
    path: str | Path,
    dem: npt.NDArray[Any],
    nodata: float | None = None,
    origin_x: float = 0.0,
) -> None:
    """
    Write a single-band GeoTIFF with 1 m pixels in :data:`_TEST_CRS`.

    Args:
        path (str | Path): Output file path.
        dem (np.ndarray): 2-D array to write as band 1.
        nodata (float | None): Nodata value to record, if any.
        origin_x (float): X coordinate of the upper-left corner. Use this
            to place neighbouring tiles side by side for mosaic tests.

    Returns:
        None
    """
    transform = from_origin(origin_x, dem.shape[0], 1, 1)
    with rasterio.open(
        path, "w", driver="GTiff", height=dem.shape[0], width=dem.shape[1],
        count=1, dtype=dem.dtype, crs=_TEST_CRS, transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(dem, 1)


def _write_tar_gz(archive_path: Path, members: dict[str, Path]) -> Path:
    """
    Pack files into a ``.tar.gz`` archive, in the given order.

    Args:
        archive_path (Path): Output archive path.
        members (dict[str, Path]): Mapping of archive member name to the
            file on disk to store under that name.

    Returns:
        Path: ``archive_path``, for convenient chaining.
    """
    with tarfile.open(archive_path, "w:gz") as tar:
        for arcname, source in members.items():
            tar.add(source, arcname=arcname)
    return archive_path


@pytest.fixture
def adjacent_tiles(tmp_path: Path) -> tuple[Path, Path]:
    """
    Two 10 x 10 tiles side by side: left = 1.0 (x 0-10), right = 2.0 (x 10-20).

    The left tile has a 2 x 2 nodata hole at its top-left corner.

    Args:
        tmp_path (Path): pytest's per-test temporary directory.

    Returns:
        tuple[Path, Path]: ``(left_tile, right_tile)`` GeoTIFF paths.
    """
    left = np.ones((10, 10), dtype=np.float32)
    left[:2, :2] = -9999.0
    right = np.full((10, 10), 2.0, dtype=np.float32)

    left_path = tmp_path / "left_dem.tif"
    right_path = tmp_path / "right_dem.tif"
    _write_geotiff(left_path, left, nodata=-9999.0, origin_x=0.0)
    _write_geotiff(right_path, right, nodata=-9999.0, origin_x=10.0)
    return left_path, right_path


@pytest.fixture
def fake_rasterio_open(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """
    Replace ``rasterio.open`` with a stub that records its argument.

    ``_open_raster_sync`` imports rasterio locally, but that reuses the
    cached module in ``sys.modules``, so patching the module attribute
    here takes effect inside it too.

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest's monkeypatch fixture.

    Returns:
        dict[str, Any]: Filled with ``"path"`` once the stub is called.
    """
    captured: dict[str, Any] = {}

    def _fake_open(path: Any) -> str:
        captured["path"] = path
        return "FAKE_DATASET"

    monkeypatch.setattr(rasterio, "open", _fake_open)
    return captured


# ---------------------------------------------------------------------------
# load_dem: happy paths
# ---------------------------------------------------------------------------

def test_load_dem_synthetic_demo() -> None:
    """The synthetic DEM honours ``shape``, uses 10 m cells, has no NaNs."""
    dem, cellsize = load_dem(None, shape=(50, 60))

    assert dem.shape == (50, 60)
    assert dem.dtype == np.float32
    assert cellsize == 10.0
    assert np.isfinite(dem).all()


def test_load_dem_synthetic_is_deterministic() -> None:
    """The fixed RNG seed makes the demo DEM identical across calls."""
    first, _ = load_dem(shape=(30, 30))
    second, _ = load_dem(shape=(30, 30))

    np.testing.assert_array_equal(first, second)


def test_load_dem_from_geotiff(tmp_path: Path) -> None:
    """A plain GeoTIFF round-trips exactly, with the right cellsize."""
    dem = np.arange(400, dtype=np.float32).reshape(20, 20)
    path = tmp_path / "test.tif"
    _write_geotiff(path, dem)

    loaded, cellsize = load_dem(str(path))

    assert loaded.shape == (20, 20)
    assert cellsize == 1.0
    np.testing.assert_allclose(loaded, dem)


def test_load_dem_accepts_pathlib_path(tmp_path: Path) -> None:
    """``pathlib.Path`` works as well as ``str``."""
    dem = np.full((5, 5), 3.0, dtype=np.float32)
    path = tmp_path / "path.tif"
    _write_geotiff(path, dem)

    loaded, _ = load_dem(path)

    np.testing.assert_allclose(loaded, 3.0)


def test_load_dem_nodata_becomes_nan(tmp_path: Path) -> None:
    """Nodata sentinels become NaN; valid cells are untouched."""
    dem = np.full((10, 10), 5.0, dtype=np.float32)
    dem[3:5, 3:5] = -9999.0
    path = tmp_path / "nodata.tif"
    _write_geotiff(path, dem, nodata=-9999.0)

    loaded, _ = load_dem(str(path))

    assert np.all(np.isnan(loaded[3:5, 3:5]))
    assert not np.any(np.isnan(loaded[:2, :2]))


def test_load_dem_list_dispatches_to_mosaic(
    adjacent_tiles: tuple[Path, Path],
) -> None:
    """A list of paths is merged, returning only ``(dem, cellsize)``."""
    result = load_dem(list(adjacent_tiles))

    assert len(result) == 2
    dem, cellsize = result
    assert dem.shape == (10, 20)
    assert cellsize == 1.0


# ---------------------------------------------------------------------------
# load_dem: error handling
# ---------------------------------------------------------------------------

def test_load_dem_missing_file_raises_dem_read_error(tmp_path: Path) -> None:
    """A missing raster raises DEMReadError naming the file."""
    missing = tmp_path / "missing.tif"

    with pytest.raises(DEMReadError, match="missing.tif"):
        load_dem(str(missing))


def test_dem_read_error_is_an_os_error(tmp_path: Path) -> None:
    """Callers that already catch OSError keep working."""
    with pytest.raises(OSError):
        load_dem(str(tmp_path / "missing.tif"))


def test_load_dem_non_raster_file_raises_dem_read_error(
    tmp_path: Path,
) -> None:
    """A file GDAL cannot parse raises DEMReadError, chaining the cause."""
    bogus = tmp_path / "not_a_raster.tif"
    bogus.write_text("definitely not a GeoTIFF")

    with pytest.raises(DEMReadError) as exc_info:
        load_dem(str(bogus))

    assert exc_info.value.__cause__ is not None


@pytest.mark.parametrize(
    "shape",
    [(0, 10), (10, -1), (10,), ("a", "b"), None],
    ids=["zero-rows", "negative-cols", "one-dim", "non-numeric", "none"],
)
def test_load_dem_synthetic_rejects_bad_shape(shape: Any) -> None:
    """Invalid synthetic shapes raise ValueError instead of numpy errors."""
    with pytest.raises(ValueError, match="shape"):
        load_dem(None, shape=shape)


# ---------------------------------------------------------------------------
# Archives (.tar.gz / .tgz)
# ---------------------------------------------------------------------------

def test_open_raster_extracts_from_tar_gz(tmp_path: Path) -> None:
    """The DEM inside a .tar.gz is read from memory without extraction."""
    tif_path = tmp_path / "tile_dem.tif"
    _write_geotiff(tif_path, np.full((10, 10), 42.0, dtype=np.float32))
    archive = _write_tar_gz(
        tmp_path / "tile.tar.gz", {"tile_dem.tif": tif_path}
    )

    with _open_raster(str(archive)) as src:
        data = src.read(1)

    np.testing.assert_allclose(data, 42.0)


def test_open_raster_prefers_dem_tif_over_sibling_rasters(
    tmp_path: Path,
) -> None:
    """``*dem.tif`` wins over siblings regardless of member order."""
    dem_path = tmp_path / "tile_dem.tif"
    matchtag_path = tmp_path / "tile_matchtag.tif"
    _write_geotiff(dem_path, np.full((5, 5), 7.0, dtype=np.float32))
    _write_geotiff(matchtag_path, np.zeros((5, 5), dtype=np.float32))

    # matchtag added first, so preference logic (not ordering) must pick
    # the DEM.
    archive = _write_tar_gz(
        tmp_path / "tile.tar.gz",
        {"tile_matchtag.tif": matchtag_path, "tile_dem.tif": dem_path},
    )

    with _open_raster(str(archive)) as src:
        data = src.read(1)

    np.testing.assert_allclose(data, 7.0)


def test_open_raster_falls_back_to_any_tiff(tmp_path: Path) -> None:
    """With no ``*dem.tif`` member, the first .tif/.tiff is used."""
    tif_path = tmp_path / "elevation.tiff"
    _write_geotiff(tif_path, np.full((4, 4), 9.0, dtype=np.float32))
    archive = _write_tar_gz(
        tmp_path / "tile.tgz", {"elevation.tiff": tif_path}
    )

    with _open_raster(str(archive)) as src:
        data = src.read(1)

    np.testing.assert_allclose(data, 9.0)


def test_pick_dem_member_skips_directories() -> None:
    """A directory whose name ends in dem.tif is never chosen."""
    directory = tarfile.TarInfo("weird_dem.tif")
    directory.type = tarfile.DIRTYPE
    regular = tarfile.TarInfo("real_dem.tif")

    assert _pick_dem_member([directory, regular]) is regular


def test_pick_dem_member_returns_none_without_tiffs() -> None:
    """Archives with no TIFF members yield ``None``."""
    assert _pick_dem_member([tarfile.TarInfo("readme.txt")]) is None


def test_open_raster_archive_without_tiff_raises_value_error(
    tmp_path: Path,
) -> None:
    """An archive with no TIFF inside raises ValueError naming it."""
    readme = tmp_path / "README.txt"
    readme.write_text("no rasters here")
    archive = _write_tar_gz(tmp_path / "empty.tar.gz", {"README.txt": readme})

    with pytest.raises(ValueError, match="No TIFF file found"):
        with _open_raster(str(archive)):
            pass


def test_open_raster_corrupt_archive_raises_dem_read_error(
    tmp_path: Path,
) -> None:
    """A file named .tar.gz that isn't gzip raises DEMReadError."""
    archive = tmp_path / "corrupt.tar.gz"
    archive.write_bytes(b"this is not gzip data")

    with pytest.raises(DEMReadError, match="corrupt.tar.gz"):
        with _open_raster(str(archive)):
            pass


def test_open_raster_truncated_archive_raises_dem_read_error(
    tmp_path: Path,
) -> None:
    """A download cut off mid-archive raises DEMReadError."""
    tif_path = tmp_path / "tile_dem.tif"
    _write_geotiff(tif_path, np.ones((50, 50), dtype=np.float32))
    buffer = stdlib_io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        tar.add(tif_path, arcname="tile_dem.tif")
    archive = tmp_path / "truncated.tar.gz"
    archive.write_bytes(buffer.getvalue()[: len(buffer.getvalue()) // 2])

    with pytest.raises(DEMReadError):
        with _open_raster(str(archive)):
            pass


# ---------------------------------------------------------------------------
# load_dem_mosaic: happy paths
# ---------------------------------------------------------------------------

def test_mosaic_full_extent(adjacent_tiles: tuple[Path, Path]) -> None:
    """Two adjacent tiles merge into one array covering both."""
    dem, cellsize, transform, crs = load_dem_mosaic(list(adjacent_tiles))

    assert dem.shape == (10, 20)
    assert cellsize == 1.0
    assert str(crs).upper() == _TEST_CRS
    assert transform.c == 0.0
    np.testing.assert_allclose(dem[5, :10], 1.0)
    np.testing.assert_allclose(dem[5, 10:], 2.0)


def test_mosaic_nodata_becomes_nan(adjacent_tiles: tuple[Path, Path]) -> None:
    """Nodata holes in a tile are NaN in the mosaic."""
    dem, *_ = load_dem_mosaic(list(adjacent_tiles))

    assert np.all(np.isnan(dem[:2, :2]))
    assert not np.any(np.isnan(dem[5:, :]))


def test_mosaic_clipped_to_bounds(adjacent_tiles: tuple[Path, Path]) -> None:
    """Bounds straddling the seam clip the merge to exactly that box."""
    dem, _, transform, _ = load_dem_mosaic(
        list(adjacent_tiles),
        bounds=(8.0, 2.0, 12.0, 8.0),
        bounds_crs=_TEST_CRS,
    )

    assert dem.shape == (6, 4)
    assert transform.c == pytest.approx(8.0)
    np.testing.assert_allclose(dem[:, :2], 1.0)
    np.testing.assert_allclose(dem[:, 2:], 2.0)


def test_mosaic_accepts_archives_and_paths(
    tmp_path: Path, adjacent_tiles: tuple[Path, Path],
) -> None:
    """A mix of .tar.gz and pathlib.Path tiles merges correctly.

    Also a regression test: with the default ``prefer_s3=True``, a
    ``Path`` used to crash the S3 regex with a TypeError.
    """
    left, right = adjacent_tiles
    archive = _write_tar_gz(tmp_path / "left.tar.gz", {"left_dem.tif": left})

    dem, *_ = load_dem_mosaic([archive, right])

    assert dem.shape == (10, 20)


def test_mosaic_bounds_outside_tiles_warns(
    adjacent_tiles: tuple[Path, Path], caplog: pytest.LogCaptureFixture,
) -> None:
    """Bounds missing every tile return all-NaN and log a warning."""
    with caplog.at_level(logging.WARNING, logger=io_module.__name__):
        dem, *_ = load_dem_mosaic(
            list(adjacent_tiles),
            bounds=(100.0, 100.0, 110.0, 110.0),
            bounds_crs=_TEST_CRS,
        )

    assert np.isnan(dem).all()
    assert "no valid elevations" in caplog.text


# ---------------------------------------------------------------------------
# load_dem_mosaic: error handling
# ---------------------------------------------------------------------------

def test_mosaic_rejects_single_string_path() -> None:
    """A bare string raises TypeError instead of iterating its characters."""
    with pytest.raises(TypeError, match="single path"):
        load_dem_mosaic("tile_dem.tif")


def test_mosaic_rejects_single_pathlib_path(tmp_path: Path) -> None:
    """A bare Path raises TypeError too."""
    with pytest.raises(TypeError, match="single path"):
        load_dem_mosaic(tmp_path / "tile_dem.tif")


def test_mosaic_needs_two_paths(adjacent_tiles: tuple[Path, Path]) -> None:
    """A one-element list raises ValueError."""
    with pytest.raises(ValueError, match="at least two"):
        load_dem_mosaic([adjacent_tiles[0]])


@pytest.mark.parametrize(
    "bounds",
    [
        (1.0, 2.0, 3.0),
        (5.0, 0.0, 1.0, 10.0),
        (0.0, 5.0, 10.0, 1.0),
        (0.0, 0.0, float("nan"), 10.0),
        (0.0, 0.0, "ten", 10.0),
    ],
    ids=["three-values", "min-x>max-x", "min-y>max-y", "nan", "non-numeric"],
)
def test_mosaic_rejects_bad_bounds(
    adjacent_tiles: tuple[Path, Path], bounds: tuple[Any, ...],
) -> None:
    """Malformed bounds raise ValueError before any tile is opened."""
    with pytest.raises(ValueError, match="bounds"):
        load_dem_mosaic(list(adjacent_tiles), bounds=bounds)


def test_mosaic_reports_every_failed_tile(
    tmp_path: Path, adjacent_tiles: tuple[Path, Path],
) -> None:
    """All failing tiles are listed, not just the first."""
    paths = [
        *adjacent_tiles,
        tmp_path / "missing_a.tif",
        tmp_path / "missing_b.tar.gz",
    ]

    with pytest.raises(DEMReadError) as exc_info:
        load_dem_mosaic(paths)

    message = str(exc_info.value)
    assert "2 of 4 tile(s) failed" in message
    assert "missing_a.tif" in message
    assert "missing_b.tar.gz" in message
    assert exc_info.value.__cause__ is not None


def test_mosaic_closes_opened_tiles_when_another_fails(
    tmp_path: Path,
    adjacent_tiles: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tiles that opened successfully are closed if a sibling fails.

    Regression test: with ``executor.map`` a single failure used to leak
    every dataset that had already been opened.
    """
    closed: list[str] = []
    real_close = io_module._close_dataset

    def _spy_close(dataset: Any) -> None:
        closed.append(dataset.name)
        real_close(dataset)

    monkeypatch.setattr(io_module, "_close_dataset", _spy_close)

    with pytest.raises(DEMReadError):
        load_dem_mosaic([*adjacent_tiles, tmp_path / "missing.tif"])

    assert len(closed) == 2
    assert all(name.endswith("_dem.tif") for name in closed)


# ---------------------------------------------------------------------------
# _https_s3_url_to_vsis3 (pure string logic)
# ---------------------------------------------------------------------------

class TestHttpsS3UrlToVsis3:
    """
    URL rewrite from plain HTTPS S3 URLs to GDAL ``/vsis3/`` paths.

    Pure string logic, so no network is needed. These tests only cover
    the transform itself: correct on every S3 URL shape GDAL/boto3 use in
    practice, and a safe no-op passthrough for anything else. Whether
    ``/vsis3/`` is actually faster is not tested here (see the function's
    docstring).
    """

    def test_real_pgc_urls(self) -> None:
        """The exact URL shape PGC's STAC API returns is rewritten."""
        assert _https_s3_url_to_vsis3(_PGC_REMA_URL) == _PGC_REMA_VSIS3

    def test_virtual_hosted_style_with_region(self) -> None:
        """``<bucket>.s3.<region>.amazonaws.com`` is rewritten."""
        assert _https_s3_url_to_vsis3(
            "https://my-bucket.s3.eu-west-1.amazonaws.com/path/to/key.tif"
        ) == "/vsis3/my-bucket/path/to/key.tif"

    def test_virtual_hosted_style_dash_region(self) -> None:
        """Legacy ``<bucket>.s3-<region>.amazonaws.com`` is rewritten."""
        assert _https_s3_url_to_vsis3(
            "https://my-bucket.s3-eu-west-1.amazonaws.com/key.tif"
        ) == "/vsis3/my-bucket/key.tif"

    def test_virtual_hosted_style_region_less_legacy(self) -> None:
        """Region-less ``<bucket>.s3.amazonaws.com`` is rewritten."""
        assert _https_s3_url_to_vsis3(
            "https://my-bucket.s3.amazonaws.com/path/to/key.tif"
        ) == "/vsis3/my-bucket/path/to/key.tif"

    def test_path_style(self) -> None:
        """``s3.<region>.amazonaws.com/<bucket>/<key>`` is rewritten."""
        assert _https_s3_url_to_vsis3(
            "https://s3.us-west-2.amazonaws.com/my-bucket/path/to/key.tif"
        ) == "/vsis3/my-bucket/path/to/key.tif"

    def test_bucket_name_with_hyphens_and_dots_in_key(self) -> None:
        """Hyphens and dots (e.g. ``_v4.1_``) don't confuse the regex."""
        assert _https_s3_url_to_vsis3(
            "https://pgc-opendata-dems.s3.us-west-2.amazonaws.com/"
            "arcticdem/mosaics/v4.1/32m/57_20/57_20_32m_v4.1_dem.tif"
        ) == (
            "/vsis3/pgc-opendata-dems/arcticdem/mosaics/v4.1/32m/57_20/"
            "57_20_32m_v4.1_dem.tif"
        )

    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/not-s3/file.tif",
            "http://my-bucket.s3.amazonaws.com/key.tif",
            "https://my-bucket.s3.amazonaws.com.evil.example/key.tif",
            "/vsis3/my-bucket/key.tif",
            "/local/path/to/file.tar.gz",
            "relative/file.tif",
            "file.tif",
            "",
        ],
        ids=[
            "non-s3-https", "plain-http", "lookalike-host", "already-vsis3",
            "absolute-local", "relative-local", "bare-filename", "empty",
        ],
    )
    def test_non_s3_input_passes_through_unchanged(self, url: str) -> None:
        """Anything that isn't recognisably an HTTPS S3 URL is untouched."""
        assert _https_s3_url_to_vsis3(url) == url


# ---------------------------------------------------------------------------
# _open_raster_sync: prefer_s3 wiring
# ---------------------------------------------------------------------------

def test_open_raster_sync_applies_rewrite_when_prefer_s3_true(
    fake_rasterio_open: dict[str, Any],
) -> None:
    """``prefer_s3=True`` passes the rewritten /vsis3/ path to rasterio.

    Proves the wiring only; ``rasterio.open`` is stubbed because there
    is no network path to a real bucket in the test environment.
    """
    result = _open_raster_sync(_PGC_REMA_URL, prefer_s3=True)

    assert result == "FAKE_DATASET"
    assert fake_rasterio_open["path"] == _PGC_REMA_VSIS3


def test_open_raster_sync_leaves_url_unchanged_when_prefer_s3_false(
    fake_rasterio_open: dict[str, Any],
) -> None:
    """``prefer_s3=False`` passes the original HTTPS URL through."""
    _open_raster_sync(_PGC_REMA_URL, prefer_s3=False)

    assert fake_rasterio_open["path"] == _PGC_REMA_URL


def test_open_raster_sync_does_not_rewrite_pathlib_paths(
    fake_rasterio_open: dict[str, Any],
) -> None:
    """Non-string paths bypass the rewrite (regression: used to crash)."""
    local = Path("/data/tile_dem.tif")

    _open_raster_sync(local, prefer_s3=True)

    assert fake_rasterio_open["path"] is local


def test_open_raster_sync_failure_after_rewrite_suggests_opt_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed /vsis3/ open names both paths and suggests prefer_s3=False."""
    original = RasterioIOError("simulated S3 access denied")

    def _failing_open(path: Any) -> None:
        raise original

    monkeypatch.setattr(rasterio, "open", _failing_open)

    with pytest.raises(DEMReadError) as exc_info:
        _open_raster_sync(_PGC_REMA_URL, prefer_s3=True)

    message = str(exc_info.value)
    assert _PGC_REMA_VSIS3 in message
    assert _PGC_REMA_URL in message
    assert "prefer_s3=False" in message
    assert exc_info.value.__cause__ is original


def test_open_raster_sync_failure_without_rewrite_has_no_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The prefer_s3 hint only appears when a rewrite actually happened."""
    def _failing_open(path: Any) -> None:
        raise RasterioIOError("simulated failure")

    monkeypatch.setattr(rasterio, "open", _failing_open)

    with pytest.raises(DEMReadError) as exc_info:
        _open_raster_sync("https://example.com/tile.tif", prefer_s3=True)

    assert "prefer_s3" not in str(exc_info.value)


# ---------------------------------------------------------------------------
# _fill_nan_nearest
# ---------------------------------------------------------------------------

def test_fill_nan_nearest() -> None:
    """NaNs are filled from neighbours; the mask marks the original NaNs."""
    arr = np.array([[1.0, np.nan, 3.0], [np.nan, 5.0, np.nan]])

    filled, mask = _fill_nan_nearest(arr)

    assert not np.any(np.isnan(filled))
    assert mask[0, 1] and mask[1, 0] and mask[1, 2]
    assert not mask[0, 0]
    assert filled[0, 0] == 1.0


def test_fill_nan_nearest_without_nans_returns_input() -> None:
    """With nothing to fill, the input array is returned (no copy)."""
    arr = np.arange(6, dtype=float).reshape(2, 3)

    filled, mask = _fill_nan_nearest(arr)

    assert filled is arr
    assert not mask.any()


def test_fill_nan_nearest_all_nan_warns_and_returns_input(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An all-NaN array is returned unchanged with a logged warning."""
    arr = np.full((3, 3), np.nan)

    with caplog.at_level(logging.WARNING, logger=io_module.__name__):
        filled, mask = _fill_nan_nearest(arr)

    assert filled is arr
    assert mask.all()
    assert "entirely NaN" in caplog.text
