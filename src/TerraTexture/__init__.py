"""
TerraTexture
==========

DEM curvature analysis + soft-light / luminosity-blended shaded relief
visualization.

Core pieces (no rasterio/contextily required):
    - derivatives.curvatures / hillshade  -- Zevenbergen & Thorne (1987)
    - blend.soft_light / luminosity_blend -- Photoshop/SVG blend modes
    - stretch.normalize / stretch_std     -- percentile / std-dev stretches
    - plotting.plot_dem_curvature_softlight -- 6-panel summary figure

Raster I/O (requires rasterio):
    - io.load_dem / load_dem_mosaic / _open_raster

Remote/open-data DEM sources (requires rasterio + requests; basemap
draping additionally requires contextily):
    - sources.stac_search / arcticdem_mosaic / rema_mosaic  -- dynamic
      STAC APIs (PGC)
    - sources.list_stac_collections / stac_collection_items /
      opentopography_mosaic  -- static/search-less STAC catalogs
      (OpenTopography, and any other STAC catalog by URL)
    - sources.make_demo_geotiff
    - basemap.plot_dem_basemap_luminosity_relief / add_relief_basemap
    - overlay.burn_data_onto_relief

Each of these groups is importable independently -- e.g. you can use
`TerraTexture.derivatives` and `TerraTexture.blend` with only numpy/scipy
installed, with no rasterio or contextily on the system at all.
"""

from .derivatives import curvatures, hillshade
from .blend import soft_light, luminosity_blend
from .stretch import normalize, stretch_std, bilinear_resample

__all__ = [
    "curvatures",
    "hillshade",
    "soft_light",
    "luminosity_blend",
    "normalize",
    "stretch_std",
    "bilinear_resample",
]

__version__ = "0.1.0"
