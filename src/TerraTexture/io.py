"""
Raster I/O: opening DEM files (including `.tar.gz`/`.tgz` mosaic archives),
merging multi-tile mosaics, and building the synthetic demo DEM.

Only depends on rasterio + numpy + scipy -- no contextily, no `requests`.
Import this module on its own if all you need is to load a DEM.
"""

import os
import re
import tarfile
from contextlib import contextmanager

import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt


# Matches both S3 URL styles GDAL/boto3 encounter in practice:
#   virtual-hosted: https://<bucket>.s3.<region>.amazonaws.com/<key>
#                   https://<bucket>.s3.amazonaws.com/<key>            (region-less, legacy/us-east-1)
#   path-style:     https://s3.<region>.amazonaws.com/<bucket>/<key>
_S3_VIRTUAL_HOSTED_RE = re.compile(
    r"^https://(?P<bucket>[^./]+)\.s3(?:[.-](?P<region>[a-z0-9-]+))?\.amazonaws\.com/(?P<key>.+)$"
)
_S3_PATH_STYLE_RE = re.compile(
    r"^https://s3(?:[.-][a-z0-9-]+)?\.amazonaws\.com/(?P<bucket>[^/]+)/(?P<key>.+)$"
)


def _https_s3_url_to_vsis3(url):
    """Rewrite a plain HTTPS S3 URL (either virtual-hosted or path-style)
    to GDAL's `/vsis3/<bucket>/<key>` virtual filesystem path, or return
    `url` unchanged if it doesn't match either S3 URL shape (e.g. a
    non-S3 HTTPS host, or an already-local path) -- always a safe no-op
    passthrough for anything that isn't recognizably S3.

    `/vsis3/` is GDAL's S3-aware I/O path, distinct from the generic
    `/vsicurl/` driver plain `rasterio.open("https://...")` uses --
    intended to reduce request overhead specifically for S3-hosted COGs
    (PGC's ArcticDEM/REMA mosaics, among others, are hosted this way).
    Requires `AWS_NO_SIGN_REQUEST=YES` in the GDAL environment for public
    buckets with no credentials configured -- see `load_dem_mosaic()`,
    which sets this via `rasterio.Env(...)` around every call that might
    use a rewritten path.

    NOTE: the actual speed benefit of `/vsis3/` over `/vsicurl/` for a
    given network path is a hypothesis, not something verified end-to-end
    against PGC's real bucket in this codebase's test environment (no
    network egress to AWS from there) -- only this URL-rewriting logic
    itself is unit-tested (tests/test_io.py). Benchmark it against your
    own network before relying on it; `prefer_s3=False` on
    `load_dem_mosaic()` opts back out to the previous `/vsicurl/`
    behaviour if it doesn't help (or actively hurts) on your setup."""
    m = _S3_VIRTUAL_HOSTED_RE.match(url)
    if m:
        return f"/vsis3/{m.group('bucket')}/{m.group('key')}"
    m = _S3_PATH_STYLE_RE.match(url)
    if m:
        return f"/vsis3/{m.group('bucket')}/{m.group('key')}"
    return url


def _fix_proj_env():
    """Defensively point PROJ at rasterio's own bundled database, rather
    than whatever a conda installation's PROJ_LIB may have set globally.

    Common failure mode this works around: conda's base environment
    auto-activates in every new shell and exports something like
    PROJ_LIB=/opt/anaconda3/share/proj into the process environment --
    regardless of which venv is actually active -- causing rasterio's
    CRS lookups to fail with a cryptic "lacks DATABASE.LAYOUT.VERSION.
    MAJOR / MINOR metadata. It comes from another PROJ installation."
    error, even though nothing about this package or its dependencies is
    actually broken. Running this once at import time means the fix
    applies automatically for every user, rather than requiring each
    person to debug their own machine's conda configuration or manually
    set environment variables in every notebook/kernel.

    Ordering here is load-bearing: rasterio's bundled GDAL/PROJ appears
    to read PROJ_LIB/PROJ_DATA once, at C-extension load time, and
    caches that decision -- setting the env var *after* `import rasterio`
    has already run does NOT fix a bad inherited PROJ_LIB (verified
    experimentally; only setting it before rasterio's compiled extension
    is actually loaded works). So this locates rasterio's install path
    via `importlib.util.find_spec()`, which does NOT execute/import the
    module, fixes the environment, and only *then* lets rasterio import
    normally wherever it's needed next.

    No-ops silently if rasterio isn't installed (core-only install) or
    if rasterio's own proj_data directory can't be found for any reason
    -- this is a best-effort fix, not a hard requirement.
    """
    import importlib.util

    spec = importlib.util.find_spec("rasterio")
    if spec is None or spec.origin is None:
        return  # rasterio not installed -- nothing to fix

    proj_data = os.path.join(os.path.dirname(spec.origin), "proj_data")
    if os.path.isdir(proj_data):
        os.environ["PROJ_DATA"] = proj_data
        os.environ.pop("PROJ_LIB", None)


_fix_proj_env()


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


def _open_raster_sync(path, prefer_s3=False):
    """Same logic as `_open_raster()`, but returns the opened dataset
    directly instead of as a context manager -- lets `load_dem_mosaic()`
    open multiple tiles concurrently via a thread pool (each open is
    I/O-bound: an HTTPS COG's header fetch, or a local .tar.gz archive's
    extraction), with the caller responsible for closing the returned
    dataset itself (e.g. by registering it on an ExitStack).

    For the .tar.gz case, the underlying MemoryFile is attached to the
    returned dataset as a private attribute so it isn't garbage
    collected out from under the still-open dataset.

    `prefer_s3=True` rewrites `path` via `_https_s3_url_to_vsis3()` first
    (a safe no-op for anything that isn't a plain HTTPS S3 URL) -- see
    that function's docstring for what this does and doesn't verify."""
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

        memfile = MemoryFile(data)
        dataset = memfile.open()
        dataset._terra_texture_memfile = memfile  # keep alive alongside dataset
        return dataset
    else:
        if prefer_s3:
            path = _https_s3_url_to_vsis3(path)
        return rasterio.open(path)


def _read_window_to_memory(src, window, nodata):
    """Read `window` from `src` (an already-open dataset or WarpedVRT)
    fully into RAM as a new, tiny in-memory dataset covering just that
    window. Used by load_dem_mosaic() to turn the slow, network-bound
    part of a merge (each source's windowed read) into something that
    can run concurrently across tiles via a thread pool, while leaving
    rasterio.merge.merge() itself untouched -- it still does its own
    per-source `.read()` calls internally, but against these local,
    already-resident-in-RAM datasets instead of the original remote
    ones, so those reads become effectively free.

    `boundless=True` matters here: `window` comes from the AOI's shared
    merge_bounds, which -- for any tile that doesn't cover the whole
    AOI by itself (the normal case when merging neighbouring tiles) --
    extends beyond that individual tile's own extent. `boundless=True`
    fills the out-of-range portion with `nodata` (or 0) instead of
    raising, matching what a windowed read against the full mosaic
    would have produced anyway."""
    from rasterio.io import MemoryFile

    fill_value = nodata if nodata is not None else 0
    data = src.read(window=window, boundless=True, fill_value=fill_value)
    win_transform = src.window_transform(window)

    profile = src.profile.copy()
    profile.update({
        "driver": "GTiff",
        "height": data.shape[1],
        "width": data.shape[2],
        "transform": win_transform,
        "count": data.shape[0],
    })
    if nodata is not None:
        profile["nodata"] = nodata

    memfile = MemoryFile()
    with memfile.open(**profile) as dst:
        dst.write(data)
    dataset = memfile.open()
    dataset._terra_texture_memfile = memfile  # keep alive alongside dataset, same as _open_raster_sync
    return dataset


def load_dem_mosaic(paths, target_crs=None, bounds=None, bounds_crs="EPSG:4326", prefer_s3=True):
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
    bounds : tuple or None
        (min_x, min_y, max_x, max_y) to clip the merge to, in
        `bounds_crs`. When given, only this region is read from each
        tile instead of each tile's full extent -- for HTTPS COG tiles
        this means GDAL's windowed /vsicurl reads fetch only the
        intersecting portion, which can be dramatically less data than
        the tile's full footprint when the AOI is much smaller than the
        tiles that happen to intersect it. Those per-tile windowed reads
        are ALSO done concurrently (one thread per tile) rather than one
        at a time -- see `_read_window_to_memory()` -- since each is
        independently network-bound. None (the default) merges each
        tile's full extent directly through `rasterio.merge.merge()`
        with no pre-fetch/parallelization, matching the previous
        behaviour (parallelizing a handful of truly enormous full-tile
        reads is a different tradeoff -- more peak memory for less
        certain benefit -- so it's deliberately left alone here).
    bounds_crs : str
        CRS of `bounds`. Reprojected internally to whatever `target_crs`
        resolves to before being passed to the merge.
    prefer_s3 : bool
        Rewrite plain-HTTPS S3 tile URLs to GDAL's `/vsis3/` virtual
        filesystem path before opening -- see `_https_s3_url_to_vsis3()`
        for exactly what this does and its verification status (the URL
        rewrite itself is unit-tested; the actual speed benefit over
        `/vsicurl/` on a given network path is NOT verified here and
        should be benchmarked on your own connection). Always a no-op
        for local files and any non-S3 HTTPS host, so this is safe to
        leave on by default; set False to force the previous, plain
        HTTPS-only behaviour if `/vsis3/` causes problems on your setup
        (older GDAL builds without S3 VSI support, a network that allows
        HTTPS but blocks direct AWS access, etc).

    Returns
    -------
    dem : 2D float array (nodata -> NaN)
    cellsize : float
    transform : affine.Affine
    crs : the CRS the mosaic was merged into
    """
    import functools
    from contextlib import ExitStack
    from concurrent.futures import ThreadPoolExecutor
    import rasterio
    from rasterio.merge import merge as rio_merge
    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling as ResamplingEnum
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds as window_from_bounds

    paths = list(paths)
    if len(paths) < 2:
        raise ValueError("load_dem_mosaic needs at least two tile paths")

    # AWS_NO_SIGN_REQUEST=YES is required for /vsis3/ reads against a
    # public, unsigned bucket (PGC's ArcticDEM/REMA mosaics are public)
    # -- harmless when prefer_s3=False or no path actually gets rewritten
    # (local files, non-S3 HTTPS hosts), since it only affects GDAL's S3
    # VSI driver behaviour.
    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES"), ExitStack() as stack:
        # Opening each tile is I/O-bound (an HTTPS COG's header fetch, or
        # a .tar.gz archive's local extraction) -- parallelize across
        # tiles rather than opening one at a time. ThreadPoolExecutor.map
        # preserves input order, which matters for rio_merge's "first
        # valid pixel wins" semantics.
        with ThreadPoolExecutor(max_workers=min(8, len(paths))) as executor:
            srcs = list(executor.map(functools.partial(_open_raster_sync, prefer_s3=prefer_s3), paths))
        for s in srcs:
            stack.callback(s.close)

        if target_crs is None:
            target_crs = srcs[0].crs

        aligned = []
        for s in srcs:
            if s.crs is not None and str(s.crs).upper() != str(target_crs).upper():
                aligned.append(
                    stack.enter_context(
                        WarpedVRT(
                            s, crs=target_crs, resampling=ResamplingEnum.bilinear,
                            warp_mem_limit=256, warp_extras={"NUM_THREADS": "ALL_CPUS"},
                        )
                    )
                )
            else:
                aligned.append(s)

        nodata = aligned[0].nodata

        merge_bounds = None
        if bounds is not None:
            merge_bounds = transform_bounds(bounds_crs, target_crs, *bounds)

        if merge_bounds is not None:
            # The actual windowed pixel read is the network-bound part;
            # rasterio.merge.merge() itself reads its sources ONE AT A
            # TIME internally, so with the AOI clip in play (the case
            # this branch handles), pre-fetch every tile's window
            # concurrently first, then hand merge() fully in-memory
            # sources so its own internal reads are effectively free.
            windows = [
                window_from_bounds(*merge_bounds, transform=s.transform) for s in aligned
            ]
            with ThreadPoolExecutor(max_workers=min(8, len(aligned))) as executor:
                prefetched = list(executor.map(
                    lambda pair: _read_window_to_memory(pair[0], pair[1], nodata),
                    zip(aligned, windows),
                ))
            for p in prefetched:
                stack.callback(p.close)
            merge_sources = prefetched
        else:
            merge_sources = aligned

        mosaic, transform = rio_merge(merge_sources, bounds=merge_bounds, nodata=nodata)

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
    