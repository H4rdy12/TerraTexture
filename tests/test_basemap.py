"""
Tests for basemap.py's aoi_bounds/target_crs CRS handling. Mocks out
the actual STAC fetch and imagery fetch entirely -- these tests check
which CRS values get passed to which internal calls, not real geospatial
correctness (that's covered by test_sources.py for the STAC-querying
side and test_derivatives.py/test_blend.py for the math).

Skips entirely if `contextily` isn't installed (it's only in the
`basemap` extra; CI's default test job only syncs `raster`).
"""

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

pytest.importorskip("contextily")

from TerraTexture.basemap import plot_dem_basemap_luminosity_relief  # noqa: E402


def _fake_mosaic_fetch(bounds, resolution=None, bbox_crs=None, target_crs=None):
    """Stand-in for arcticdem_mosaic()/rema_mosaic(): records what it was
    called with and returns a small synthetic DEM."""
    _fake_mosaic_fetch.last_call = {"bounds": bounds, "bbox_crs": bbox_crs, "target_crs": target_crs}
    dem = np.random.default_rng(0).random((20, 20)).astype(np.float32)
    from rasterio.transform import from_bounds
    transform = from_bounds(0, 0, 100, 100, 20, 20)
    resolved_crs = target_crs or "EPSG:3413"
    return dem, 5.0, transform, resolved_crs


def _fake_bounds2img(*args, **kwargs):
    """Stand-in for contextily.bounds2img(): a tiny fake RGB tile plus a
    plausible extent tuple, avoiding any real network call."""
    tile = (np.random.default_rng(1).random((20, 20, 3)) * 255).astype(np.uint8)
    return tile, (0, 100, 0, 100)


def test_aoi_bounds_crs_independent_of_target_crs():
    """The bug this guards against: aoi_bounds_crs used to be silently
    forced to equal target_crs, so lon/lat bounds (EPSG:4326) couldn't be
    combined with a different output target_crs at all."""
    with patch.dict("TerraTexture.basemap._AOI_PRODUCTS", {"arcticdem": _fake_mosaic_fetch}), \
         patch("contextily.bounds2img", side_effect=_fake_bounds2img), \
         patch("contextily.providers") as mock_providers:
        mock_providers.Esri.WorldImagery = MagicMock(attribution="")

        plot_dem_basemap_luminosity_relief(
            aoi_bounds=(-10, 50, -5, 55),   # plain lon/lat bounds
            aoi_bounds_crs="EPSG:4326",
            target_crs="EPSG:3031",          # deliberately NOT EPSG:4326 or the arcticdem default (3413)
            show=False,
        )

    call = _fake_mosaic_fetch.last_call
    assert call["bbox_crs"] == "EPSG:4326"
    assert call["target_crs"] == "EPSG:3031"
    assert call["bounds"] == (-10, 50, -5, 55)


def test_aoi_bounds_crs_defaults_to_4326():
    """Default should be EPSG:4326 -- the natural way most bounding boxes
    already come in (GPS, a web map, etc.) -- without the caller having
    to specify it explicitly."""
    with patch.dict("TerraTexture.basemap._AOI_PRODUCTS", {"arcticdem": _fake_mosaic_fetch}), \
         patch("contextily.bounds2img", side_effect=_fake_bounds2img), \
         patch("contextily.providers") as mock_providers:
        mock_providers.Esri.WorldImagery = MagicMock(attribution="")

        plot_dem_basemap_luminosity_relief(
            aoi_bounds=(-10, 50, -5, 55),
            show=False,
        )

    assert _fake_mosaic_fetch.last_call["bbox_crs"] == "EPSG:4326"
