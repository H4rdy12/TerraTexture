import tarfile

import numpy as np
import pytest

from TerraTexture.io import _open_raster, load_dem, _fill_nan_nearest, _https_s3_url_to_vsis3, _open_raster_sync

rasterio = pytest.importorskip("rasterio")


def _write_geotiff(path, dem, nodata=None):
    from rasterio.transform import from_origin

    transform = from_origin(0, dem.shape[0], 1, 1)
    with rasterio.open(
        path, "w", driver="GTiff", height=dem.shape[0], width=dem.shape[1],
        count=1, dtype=dem.dtype, crs="EPSG:32633", transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(dem, 1)


def test_load_dem_synthetic_demo():
    dem, cellsize = load_dem(None, shape=(50, 60))
    assert dem.shape == (50, 60)
    assert cellsize == 10.0
    assert np.isfinite(dem).all()


def test_load_dem_from_geotiff(tmp_path):
    dem = (np.arange(400, dtype=np.float32).reshape(20, 20))
    path = tmp_path / "test.tif"
    _write_geotiff(str(path), dem)

    loaded, cellsize = load_dem(str(path))
    assert loaded.shape == (20, 20)
    assert cellsize == 1.0
    assert np.allclose(loaded, dem)


def test_load_dem_nodata_becomes_nan(tmp_path):
    dem = np.ones((10, 10), dtype=np.float32) * 5.0
    dem[3:5, 3:5] = -9999.0
    path = tmp_path / "nodata.tif"
    _write_geotiff(str(path), dem, nodata=-9999.0)

    loaded, _ = load_dem(str(path))
    assert np.all(np.isnan(loaded[3:5, 3:5]))
    assert not np.any(np.isnan(loaded[:2, :2]))


def test_open_raster_extracts_from_tar_gz(tmp_path):
    dem = np.ones((10, 10), dtype=np.float32) * 42.0
    tif_path = tmp_path / "tile_dem.tif"
    _write_geotiff(str(tif_path), dem)

    archive_path = tmp_path / "tile.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(tif_path, arcname="tile_dem.tif")

    with _open_raster(str(archive_path)) as src:
        data = src.read(1)
    assert np.allclose(data, 42.0)


def test_open_raster_prefers_dem_tif_over_sibling_rasters(tmp_path):
    dem = np.ones((5, 5), dtype=np.float32) * 7.0
    matchtag = np.zeros((5, 5), dtype=np.float32)

    dem_path = tmp_path / "tile_dem.tif"
    matchtag_path = tmp_path / "tile_matchtag.tif"
    _write_geotiff(str(dem_path), dem)
    _write_geotiff(str(matchtag_path), matchtag)

    archive_path = tmp_path / "tile.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        # matchtag added first to make sure preference logic (not
        # ordering) picks the DEM
        tar.add(matchtag_path, arcname="tile_matchtag.tif")
        tar.add(dem_path, arcname="tile_dem.tif")

    with _open_raster(str(archive_path)) as src:
        data = src.read(1)
    assert np.allclose(data, 7.0)


def test_fill_nan_nearest():
    arr = np.array([[1.0, np.nan, 3.0], [np.nan, 5.0, np.nan]])
    filled, mask = _fill_nan_nearest(arr)
    assert not np.any(np.isnan(filled))
    assert mask[0, 1] and mask[1, 0] and mask[1, 2]
    assert not mask[0, 0]


class TestHttpsS3UrlToVsis3:
    """_https_s3_url_to_vsis3() is pure string logic (no network needed
    to test it) -- see load_dem_mosaic()'s prefer_s3 docstring for what
    the rewrite is actually for and its (unverified-against-a-real-
    bucket-here) rationale. These tests only cover the URL transform
    itself: correct on every S3 URL shape GDAL/boto3 use in practice,
    and a safe no-op passthrough for anything else."""

    def test_real_pgc_urls(self):
        # the exact URL shape PGC's STAC API actually returns
        assert _https_s3_url_to_vsis3(
            "https://pgc-opendata-dems.s3.us-west-2.amazonaws.com/"
            "rema/mosaics/v2.0/10m/40_10/40_10_10m_v2.0_dem.tif"
        ) == "/vsis3/pgc-opendata-dems/rema/mosaics/v2.0/10m/40_10/40_10_10m_v2.0_dem.tif"

    def test_virtual_hosted_style_with_region(self):
        assert _https_s3_url_to_vsis3(
            "https://my-bucket.s3.eu-west-1.amazonaws.com/path/to/key.tif"
        ) == "/vsis3/my-bucket/path/to/key.tif"

    def test_virtual_hosted_style_region_less_legacy(self):
        assert _https_s3_url_to_vsis3(
            "https://my-bucket.s3.amazonaws.com/path/to/key.tif"
        ) == "/vsis3/my-bucket/path/to/key.tif"

    def test_path_style(self):
        assert _https_s3_url_to_vsis3(
            "https://s3.us-west-2.amazonaws.com/my-bucket/path/to/key.tif"
        ) == "/vsis3/my-bucket/path/to/key.tif"

    def test_non_s3_https_url_passes_through_unchanged(self):
        url = "https://example.com/not-s3/file.tif"
        assert _https_s3_url_to_vsis3(url) == url

    def test_local_path_passes_through_unchanged(self):
        for path in ["/local/path/to/file.tar.gz", "relative/file.tif", "file.tif"]:
            assert _https_s3_url_to_vsis3(path) == path

    def test_bucket_name_with_hyphens_and_dots_in_key(self):
        """Bucket names and object keys both commonly contain hyphens and
        dots (e.g. version numbers in filenames, like PGC's own
        `..._v2.0_dem.tif`) -- make sure the regex's `.` isn't
        accidentally anchoring on the wrong part of the URL."""
        assert _https_s3_url_to_vsis3(
            "https://pgc-opendata-dems.s3.us-west-2.amazonaws.com/"
            "arcticdem/mosaics/v4.1/32m/57_20/57_20_32m_v4.1_dem.tif"
        ) == "/vsis3/pgc-opendata-dems/arcticdem/mosaics/v4.1/32m/57_20/57_20_32m_v4.1_dem.tif"


def test_open_raster_sync_applies_rewrite_when_prefer_s3_true(monkeypatch):
    """Confirms the WIRING, not just the pure URL-rewrite function in
    isolation: _open_raster_sync(path, prefer_s3=True) must actually
    call rasterio.open() with the rewritten /vsis3/ path, not the
    original https:// one. Mocks rasterio.open itself since there's no
    real network path to a /vsis3/ URL to open in this test
    environment -- this only proves what path gets passed, not that
    GDAL successfully reads it."""
    import TerraTexture.io as io_module  # noqa: F401 (imported for clarity/context only)

    captured = {}

    def fake_rasterio_open(path):
        captured["path"] = path
        return "FAKE_DATASET"

    # _open_raster_sync does `import rasterio` locally inside the function
    # body -- that local import reuses the same cached module object in
    # sys.modules, so patching the real `rasterio` module's `open` here
    # takes effect there too, regardless of where the import statement is.
    import rasterio
    monkeypatch.setattr(rasterio, "open", fake_rasterio_open)

    https_url = (
        "https://pgc-opendata-dems.s3.us-west-2.amazonaws.com/"
        "rema/mosaics/v2.0/10m/40_10/40_10_10m_v2.0_dem.tif"
    )
    result = _open_raster_sync(https_url, prefer_s3=True)

    assert result == "FAKE_DATASET"
    assert captured["path"] == (
        "/vsis3/pgc-opendata-dems/rema/mosaics/v2.0/10m/40_10/40_10_10m_v2.0_dem.tif"
    )


def test_open_raster_sync_leaves_url_unchanged_when_prefer_s3_false(monkeypatch):
    import rasterio

    captured = {}

    def fake_rasterio_open(path):
        captured["path"] = path
        return "FAKE_DATASET"

    monkeypatch.setattr(rasterio, "open", fake_rasterio_open)

    https_url = "https://pgc-opendata-dems.s3.us-west-2.amazonaws.com/rema/mosaics/v2.0/10m/40_10/40_10_10m_v2.0_dem.tif"
    _open_raster_sync(https_url, prefer_s3=False)

    assert captured["path"] == https_url  # unchanged -- opted out
