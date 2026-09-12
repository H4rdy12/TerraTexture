import tarfile

import numpy as np
import pytest

from TerraTexture.io import _open_raster, load_dem, _fill_nan_nearest

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
