"""
Raster I/O: opening DEM files (including .tar.gz/.tgz mosaic archives),
merging multi-tile mosaics, and building the synthetic demo DEM.

Only depends on rasterio + numpy + scipy -- no contextily, no
DEMSquad_STAC. Import this module on its own if all you need is to load
a DEM.
"""

import tarfile
from contextlib import contextmanager

import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt


@contextmanager
def _open_raster(path):
    """Yield an open rasterio dataset for `path`. Transparently handles
    .tar.gz/.tgz archives (e.g. ArcticDEM mosaic tiles) by extracting the
    DEM GeoTIFF member into memory (preferring "*_dem.tif" over sibling
    matchtag/count/browse rasters in the same archive) and opening it via
    rasterio's MemoryFile -- no extraction to disk. Plain raster paths
    (.tif, .vrt, etc.) are opened directly."""
    import rasterio

    if str(path).endswith((".tar.gz", ".tgz")):
        from rasterio.io import MemoryFile

        with tarfile.open(path, "r:gz") as tar:
            members = tar.getmembers()
            tif_member = next(
                (m for m in members if m.name.lower().endswith("dem.tif")), None
            )
            if tif_member is None:
                tif_member = next(
                    (m for m in members if m.name.lower().endswith((".tif", ".tiff"))),
                    None,
                )
            if tif_member is None:
                raise ValueError(f"No TIFF file found inside archive {path}")
            with tar.extractfile(tif_member) as f:
                data = f.read()

        with MemoryFile(data) as memfile:
            with memfile.open() as src:
                yield src
    else:
        with rasterio.open(path) as src:
            yield src


def load_dem_mosaic(paths, target_crs=None):
    """Merge two or more DEM tiles into a single seamless array (e.g.
    neighbouring ArcticDEM mosaic tiles). Any mix of plain rasters and
    .tar.gz/.tgz archives is fine -- each is opened via _open_raster().

    Tiles are reprojected on the fly (via a WarpedVRT) into `target_crs`
    if their own CRS doesn't already match it, so this also works for
    tiles that come from slightly different source CRSs. Overlapping
    regions are mosaicked with rasterio's default "first valid pixel
    wins" strategy.

    Parameters
    ----------
    paths : sequence of str
        Two or more DEM file paths (rasters or .tar.gz/.tgz archives).
    target_crs : str or None
        CRS to merge into. Defaults to the first tile's own CRS.

    Returns
    -------
    dem : 2D float array (nodata -> NaN)
    cellsize : float
    transform : affine.Affine
    crs : the CRS the mosaic was merged into
    """
    from contextlib import ExitStack
    from rasterio.merge import merge as rio_merge
    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling as ResamplingEnum

    paths = list(paths)
    if len(paths) < 2:
        raise ValueError("load_dem_mosaic needs at least two tile paths")

    with ExitStack() as stack:
        srcs = [stack.enter_context(_open_raster(p)) for p in paths]
        if target_crs is None:
            target_crs = srcs[0].crs

        aligned = []
        for s in srcs:
            if s.crs is not None and str(s.crs).upper() != str(target_crs).upper():
                aligned.append(
                    stack.enter_context(
                        WarpedVRT(s, crs=target_crs, resampling=ResamplingEnum.bilinear)
                    )
                )
            else:
                aligned.append(s)

        nodata = aligned[0].nodata
        mosaic, transform = rio_merge(aligned, nodata=nodata)

    dem = mosaic[0].astype(np.float32)
    if nodata is not None:
        dem = np.where(dem == nodata, np.nan, dem)
    cellsize = transform.a
    return dem, cellsize, transform, target_crs


def load_dem(path=None, shape=(400, 400)):
    """Return (dem_array, cellsize). Reads a raster if `path` is given
    (transparently handling .tar.gz/.tgz archives via _open_raster()),
    otherwise builds a synthetic DEM (a few hills + a valley + noise).
    If `path` is a list/tuple of two or more paths, they're merged into
    one seamless mosaic via load_dem_mosaic() (transform/crs dropped from
    the return here for signature compatibility -- call load_dem_mosaic()
    directly if you need those).

    nodata cells are converted to NaN rather than left as sentinel values
    (e.g. -9999) so they don't corrupt derivatives/curvature downstream --
    curvatures() and hillshade() nan-fill internally for the calculation
    and remask the result afterwards.
    """
    if isinstance(path, (list, tuple)):
        dem, cellsize, _transform, _crs = load_dem_mosaic(path)
        return dem, cellsize

    if path is not None:
        with _open_raster(path) as src:
            dem = src.read(1).astype(np.float32)
            cellsize = src.transform[0]
            if src.nodata is not None:
                dem = np.where(dem == src.nodata, np.nan, dem)
        return dem, cellsize

    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:shape[0], 0:shape[1]]
    dem = np.zeros(shape, dtype=np.float32)

    hills = [(110, 110, 70, 70), (300, 250, 90, 60), (180, 330, 55, 45)]
    for cx, cy, amp, sigma in hills:
        dem += amp * np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2)))

    # a sinuous valley cutting across
    valley_centre = 200 + 60 * np.sin(y / 40.0)
    dem -= 35 * np.exp(-((x - valley_centre) ** 2) / (2 * 25 ** 2))

    dem += gaussian_filter(rng.standard_normal(shape), sigma=6) * 4
    return dem, 10.0  # 10 m cells


def _fill_nan_nearest(arr):
    """Fill NaNs with the value of the nearest valid pixel (for computing
    derivatives across small voids/edges without them poisoning the whole
    array). Returns (filled_array, nan_mask) so callers can remask results."""
    mask = np.isnan(arr)
    if not mask.any():
        return arr, mask
    idx = distance_transform_edt(mask, return_distances=False, return_indices=True)
    filled = arr[tuple(idx)]
    return filled, mask
