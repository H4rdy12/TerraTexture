"""
Remote/alternative DEM sources: querying the local PGC ArcticDEM STAC
catalog, and writing a synthetic-but-georeferenced demo GeoTIFF.

ArcticDEM_stac() has an optional dependency on DEMSquad_STAC, which isn't
on PyPI -- the import happens lazily inside the function, so importing
dem_relief (or even dem_relief.sources) elsewhere works fine without it
installed, as long as you don't actually call ArcticDEM_stac().
"""

import sys

import numpy as np

from .io import load_dem


def ArcticDEM_stac(
    bounds,
    resolution=32,
    aoi_crs="EPSG:3413",
    demsquad_path="/media/luna/hardydj/Scripts/DEMSquad_stable",
):
    """Query the local PGC ArcticDEM STAC catalog for the mosaic tiles
    intersecting an AOI, merge them, and return a DEM ready for the
    curvature/hillshade/luminosity pipeline -- an ADDITIONAL way to source
    a DEM (pass `aoi_bounds=` to plot_dem_basemap_luminosity_relief instead
    of `dem_path=`), not a replacement for the file-based / .tar.gz /
    multi-tile-mosaic options.

    This calls the exact same helper as
    DEMTimeSeriesPlotter.add_arcticdem_mosaic()
    (DEMSquad_STAC.stac_tools.Query_PGC_catalog(...).get_merged_mosaic()),
    so results are consistent with what that class already plots -- but
    this module has no hard dependency on DEMTimeSeriesPlotter or its
    Config setup.

    Parameters
    ----------
    bounds : tuple
        (x_min, y_min, x_max, y_max) in aoi_crs.
    resolution : int
        ArcticDEM mosaic resolution in meters (e.g. 2, 10, 32).
    aoi_crs : str
        CRS of `bounds`. The merged mosaic comes back in this same CRS.
    demsquad_path : str
        Added to sys.path (if not already present) so DEMSquad_STAC can
        be imported. Adjust if it lives somewhere else in your environment.

    Returns
    -------
    dem : 2D float array (common nodata sentinels, e.g. <= -9998, -> NaN)
    cellsize : float
    transform : affine.Affine
    crs : str (== aoi_crs)

    Example
    -------
    >>> from dem_relief.basemap import plot_dem_basemap_luminosity_relief
    >>> fig, ax, layers = plot_dem_basemap_luminosity_relief(
    ...     aoi_bounds=(x_min, y_min, x_max, y_max),
    ...     arcticdem_resolution=32,
    ...     out_png="relief_from_stac.png",
    ... )
    """
    if demsquad_path and demsquad_path not in sys.path:
        sys.path.insert(0, demsquad_path)
    try:
        from DEMSquad_STAC.stac_tools import Query_PGC_catalog
    except ImportError as exc:
        raise ImportError(
            "ArcticDEM_stac() requires DEMSquad_STAC to be importable. "
            f"Tried adding '{demsquad_path}' to sys.path -- pass the "
            "correct location via demsquad_path= if it lives elsewhere."
        ) from exc

    ref_array, ref_transform = Query_PGC_catalog(
        product="ArcticDEM",
        resolution=resolution,
        aoi=bounds,
        aoi_crs=aoi_crs,
    ).get_merged_mosaic()

    dem = np.asarray(ref_array, dtype=np.float32)
    # guard common nodata sentinels (e.g. -9999), same check used in
    # DEMTimeSeriesPlotter.add_arcticdem_mosaic
    dem = np.where(np.isfinite(dem) & (dem > -9998), dem, np.nan)

    cellsize = ref_transform.a
    return dem, cellsize, ref_transform, aoi_crs


def make_demo_geotiff(
    out_path="/tmp/demo_dem.tif",
    bounds_4326=(-3.25, 54.42, -3.10, 54.53),  # west, south, east, north (Lake District, UK)
):
    """Write the synthetic DEM from load_dem() to a real GeoTIFF covering
    `bounds_4326`, purely so plot_dem_basemap_luminosity_relief() has
    something real-world-georeferenced to fetch imagery for. Elevation
    VALUES are still fake -- only the footprint is real. Use your own DEM
    file for real work."""
    import rasterio
    from rasterio.transform import from_bounds

    dem, _ = load_dem(None)
    west, south, east, north = bounds_4326
    transform = from_bounds(west, south, east, north, dem.shape[1], dem.shape[0])

    with rasterio.open(
        out_path, "w", driver="GTiff", height=dem.shape[0], width=dem.shape[1],
        count=1, dtype=dem.dtype, crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(dem, 1)

    return out_path
