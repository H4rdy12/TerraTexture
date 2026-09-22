"""
Raster I/O for digital elevation models (DEMs).

This module is the loading layer of the terrain pipeline. It opens single
DEM rasters, merges multi-tile mosaics into one seamless array, and builds
a synthetic demo DEM so the rest of the pipeline can run with no data on
disk.

Supported inputs:

- Plain rasters that any GDAL driver can read (``.tif``, ``.vrt``, ...).
- ``.tar.gz`` / ``.tgz`` archives such as ArcticDEM / REMA mosaic tiles.
  The DEM GeoTIFF member is extracted straight into memory; nothing is
  written to disk.
- Remote Cloud Optimized GeoTIFFs over HTTPS. Plain S3 URLs can
  optionally be rewritten to GDAL's ``/vsis3/`` virtual filesystem.

Public API:

- :func:`load_dem` -- load one DEM, a mosaic, or the synthetic demo DEM.
- :func:`load_dem_mosaic` -- merge two or more tiles, optionally clipped
  to an area of interest, with concurrent tile opens and window reads.
- :class:`DEMReadError` -- raised when a raster cannot be opened, read,
  reprojected or merged.

Every loader returns ``float32`` elevations with nodata converted to
``NaN``, so sentinel values such as ``-9999`` never leak into slope,
curvature or hillshade calculations downstream.

Import side effect:
    On import, :func:`_fix_proj_env` points ``PROJ_DATA`` at rasterio's
    bundled PROJ database *before* rasterio's C extension loads. This
    works around a stale ``PROJ_LIB`` exported by an auto-activated conda
    base environment. rasterio itself is only imported lazily, inside the
    functions that need it, so that this fix always runs first.

Dependencies:
    numpy, scipy and rasterio. No contextily or ``requests``, so this
    module can be imported on its own when all you need is to load a DEM.

Examples:
    Load a single ArcticDEM tile straight from its archive::

        dem, cellsize = load_dem("41_17_2_2_2m_v4.1.tar.gz")

    Merge neighbouring remote tiles, clipped to a lon/lat box::

        dem, cellsize, transform, crs = load_dem_mosaic(
            [
                "https://pgc-opendata-dems.s3.us-west-2.amazonaws.com/a.tif",
                "https://pgc-opendata-dems.s3.us-west-2.amazonaws.com/b.tif",
            ],
            bounds=(-50.2, 69.1, -49.8, 69.3),
        )

    Build the synthetic demo DEM (400 x 400 cells, 10 m spacing)::

        dem, cellsize = load_dem()
"""

from __future__ import annotations

import functools
import importlib.util
import logging
import math
import os
import re
import tarfile
import zlib
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import numpy.typing as npt
from scipy.ndimage import distance_transform_edt, gaussian_filter

if TYPE_CHECKING:
    from affine import Affine
    from rasterio.crs import CRS
    from rasterio.io import DatasetReader
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import Window

    # Anything rasterio.open() accepts as a location.
    RasterPath = str | os.PathLike[str]
    # Anything that can be read like a dataset (plain or reprojected).
    ReadableDataset = DatasetReader | WarpedVRT
    # CRS as accepted by rasterio: an object, or "EPSG:xxxx" / WKT / PROJ.
    CRSLike = str | CRS


logger = logging.getLogger(__name__)

_T = TypeVar("_T")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Upper bound on concurrent tile opens / window reads (all I/O-bound).
_MAX_IO_WORKERS = 8

# File suffixes recognised as compressed DEM archives.
_ARCHIVE_SUFFIXES = (".tar.gz", ".tgz")

# Preferred archive member (the DEM, not matchtag/count/browse rasters).
_DEM_MEMBER_SUFFIX = "dem.tif"

# Fallback archive member suffixes if no "*dem.tif" exists.
_TIFF_SUFFIXES = (".tif", ".tiff")

# Attribute used to keep a MemoryFile alive alongside its open dataset.
_MEMFILE_ATTR = "_terra_texture_memfile"

# GDAL config for unsigned reads from public S3 buckets (PGC is public).
_GDAL_S3_ENV = {"AWS_NO_SIGN_REQUEST": "YES"}

# Matches both S3 URL styles GDAL/boto3 encounter in practice:
#   virtual-hosted: https://<bucket>.s3.<region>.amazonaws.com/<key>
#                   https://<bucket>.s3.amazonaws.com/<key>  (legacy)
#   path-style:     https://s3.<region>.amazonaws.com/<bucket>/<key>
_S3_VIRTUAL_HOSTED_RE = re.compile(
    r"^https://(?P<bucket>[^./]+)\.s3(?:[.-](?P<region>[a-z0-9-]+))?"
    r"\.amazonaws\.com/(?P<key>.+)$"
)
_S3_PATH_STYLE_RE = re.compile(
    r"^https://s3(?:[.-][a-z0-9-]+)?\.amazonaws\.com/"
    r"(?P<bucket>[^/]+)/(?P<key>.+)$"
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DEMReadError(OSError):
    """
    Raised when a DEM cannot be opened, read, reprojected or merged.

    Subclasses :class:`OSError` so existing callers that already catch
    ``OSError`` (or rasterio's ``RasterioIOError``, itself an ``OSError``)
    keep working. The underlying rasterio / tarfile exception is always
    chained as ``__cause__`` for debugging.
    """


# ---------------------------------------------------------------------------
# Environment setup (runs at import time)
# ---------------------------------------------------------------------------

def _fix_proj_env() -> None:
    """
    Point PROJ at rasterio's bundled database instead of an inherited one.

    Works around a common failure mode: conda's base environment
    auto-activates in every new shell and exports something like
    ``PROJ_LIB=/opt/anaconda3/share/proj`` regardless of which venv is
    active. rasterio's CRS lookups then fail with a cryptic
    ``lacks DATABASE.LAYOUT.VERSION.MAJOR / MINOR metadata. It comes from
    another PROJ installation.`` error, even though nothing is actually
    broken. Running this at import time fixes it for every user without
    them having to debug their own conda setup.

    Returns:
        None: The process environment is modified in place.

    Warning:
        Ordering is load-bearing. rasterio's bundled GDAL/PROJ reads
        ``PROJ_LIB`` / ``PROJ_DATA`` once, when its C extension loads, and
        caches the result -- setting the variable *after*
        ``import rasterio`` does not help (verified experimentally). This
        function therefore locates rasterio via
        :func:`importlib.util.find_spec`, which does not execute the
        module, and must run before anything imports rasterio.

    Note:
        Best-effort only. It no-ops (logging at DEBUG level) if rasterio is
        not installed, cannot be located, or ships no ``proj_data``
        directory. It never raises.
    """
    try:
        spec = importlib.util.find_spec("rasterio")
    except (ImportError, ValueError) as exc:
        logger.debug("Could not locate rasterio (%s); PROJ fix skipped.", exc)
        return

    if spec is None or spec.origin is None:
        logger.debug("rasterio not installed; PROJ fix skipped.")
        return

    proj_data = os.path.join(os.path.dirname(spec.origin), "proj_data")
    if not os.path.isdir(proj_data):
        logger.debug("No bundled proj_data at %s; PROJ fix skipped.", proj_data)
        return

    os.environ["PROJ_DATA"] = proj_data
    stale_proj_lib = os.environ.pop("PROJ_LIB", None)
    if stale_proj_lib is not None and stale_proj_lib != proj_data:
        logger.debug("Removed inherited PROJ_LIB=%s.", stale_proj_lib)


_fix_proj_env()


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def _https_s3_url_to_vsis3(url: str) -> str:
    """
    Rewrite a plain HTTPS S3 URL to GDAL's ``/vsis3/`` virtual path.

    Handles both virtual-hosted and path-style S3 URLs. Anything that is
    not recognisably S3 (a non-S3 HTTPS host, a local path, an existing
    ``/vsi*`` path) is returned unchanged, so this is always a safe
    passthrough.

    ``/vsis3/`` is GDAL's S3-aware I/O path, distinct from the generic
    ``/vsicurl/`` driver that plain ``rasterio.open("https://...")`` uses.
    It is intended to reduce request overhead for S3-hosted COGs, such as
    PGC's ArcticDEM / REMA mosaics.

    Args:
        url (str): URL or path to (possibly) rewrite.

    Returns:
        str: ``/vsis3/<bucket>/<key>`` for an S3 URL, otherwise ``url``.

    Note:
        Public buckets read without credentials need
        ``AWS_NO_SIGN_REQUEST=YES`` in the GDAL environment;
        :func:`load_dem_mosaic` sets this around every open and read.

    Warning:
        The speed benefit of ``/vsis3/`` over ``/vsicurl/`` is a
        hypothesis, not verified end-to-end against PGC's bucket (the test
        environment has no egress to AWS). Only this URL rewrite is
        unit-tested (``tests/test_io.py``). Benchmark it on your own
        network; ``prefer_s3=False`` on :func:`load_dem_mosaic` restores
        the previous ``/vsicurl/`` behaviour.

    Examples:
        >>> _https_s3_url_to_vsis3(
        ...     "https://my-bucket.s3.us-west-2.amazonaws.com/a/b.tif")
        '/vsis3/my-bucket/a/b.tif'
        >>> _https_s3_url_to_vsis3(
        ...     "https://s3.us-west-2.amazonaws.com/my-bucket/a/b.tif")
        '/vsis3/my-bucket/a/b.tif'
        >>> _https_s3_url_to_vsis3("https://example.com/b.tif")
        'https://example.com/b.tif'
    """
    for pattern in (_S3_VIRTUAL_HOSTED_RE, _S3_PATH_STYLE_RE):
        match = pattern.match(url)
        if match:
            return f"/vsis3/{match.group('bucket')}/{match.group('key')}"
    return url


def _in_gdal_s3_env(func: Callable[..., _T]) -> Callable[..., _T]:
    """
    Wrap ``func`` so it runs inside a ``rasterio.Env`` with S3 settings.

    Worker threads in a :class:`~concurrent.futures.ThreadPoolExecutor`
    may not see a ``rasterio.Env`` entered on the calling thread, because
    rasterio tracks its environment per thread. Re-entering the same
    settings inside each worker guarantees ``AWS_NO_SIGN_REQUEST`` is in
    effect for ``/vsis3/`` opens and reads regardless of that detail.

    Args:
        func (Callable[..., _T]): Function to run inside the environment.

    Returns:
        Callable[..., _T]: Wrapped function with the same signature.
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> _T:
        import rasterio

        with rasterio.Env(**_GDAL_S3_ENV):
            return func(*args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Opening rasters and archives
# ---------------------------------------------------------------------------

def _pick_dem_member(
    members: Sequence[tarfile.TarInfo],
) -> tarfile.TarInfo | None:
    """
    Choose the DEM GeoTIFF among the regular files in a tar archive.

    Prefers a member ending in ``dem.tif`` over sibling rasters such as
    matchtag, count or browse images; falls back to the first
    ``.tif`` / ``.tiff`` member.

    Args:
        members (Sequence[tarfile.TarInfo]): Members of an open archive.

    Returns:
        tarfile.TarInfo | None: The chosen member, or ``None`` if the
            archive holds no TIFF files at all.
    """
    files = [m for m in members if m.isfile()]
    for member in files:
        if member.name.lower().endswith(_DEM_MEMBER_SUFFIX):
            return member
    for member in files:
        if member.name.lower().endswith(_TIFF_SUFFIXES):
            return member
    return None


def _read_dem_bytes_from_archive(path: RasterPath) -> bytes:
    """
    Extract the DEM GeoTIFF from a ``.tar.gz`` / ``.tgz`` into memory.

    Args:
        path (RasterPath): Path to the archive.

    Returns:
        bytes: Raw bytes of the chosen GeoTIFF member.

    Raises:
        ValueError: If the archive contains no TIFF file.
        DEMReadError: If the archive is missing, unreadable, truncated or
            not a valid gzip-compressed tar file.
    """
    try:
        with tarfile.open(path, "r:gz") as tar:
            member = _pick_dem_member(tar.getmembers())
            if member is None:
                raise ValueError(f"No TIFF file found inside archive {path}")
            extracted = tar.extractfile(member)
            if extracted is None:
                raise ValueError(
                    f"Archive member {member.name!r} in {path} is not a "
                    "regular file"
                )
            with extracted as fileobj:
                return fileobj.read()
    except (tarfile.TarError, EOFError, zlib.error, OSError) as exc:
        raise DEMReadError(
            f"Could not read DEM archive {path}: {exc}"
        ) from exc


def _open_archive_dataset(path: RasterPath) -> DatasetReader:
    """
    Open the DEM inside a ``.tar.gz`` / ``.tgz`` archive as a dataset.

    The GeoTIFF is extracted into a rasterio ``MemoryFile``, which is
    attached to the returned dataset so it is not garbage collected while
    the dataset is still open. :func:`_close_dataset` closes both.

    Args:
        path (RasterPath): Path to the archive.

    Returns:
        DatasetReader: Open, in-memory dataset. The caller must close it,
            preferably via :func:`_close_dataset`.

    Raises:
        ValueError: If the archive contains no TIFF file.
        DEMReadError: If the archive cannot be read, or its TIFF member is
            not a raster GDAL can open.
    """
    from rasterio.errors import RasterioError
    from rasterio.io import MemoryFile

    data = _read_dem_bytes_from_archive(path)
    memfile = MemoryFile(data)
    try:
        dataset = memfile.open()
    except RasterioError as exc:
        memfile.close()
        raise DEMReadError(
            f"Archive {path} contains a TIFF that GDAL cannot open: {exc}"
        ) from exc
    setattr(dataset, _MEMFILE_ATTR, memfile)
    return dataset


def _close_dataset(dataset: ReadableDataset) -> None:
    """
    Close a dataset and any ``MemoryFile`` kept alive alongside it.

    Safe to call on plain datasets, archive-backed datasets and the
    in-memory window copies made by :func:`_read_window_to_memory`.
    Errors while closing are logged, never raised, so one bad close
    cannot mask the exception that triggered cleanup.

    Args:
        dataset (ReadableDataset): Dataset to close.

    Returns:
        None
    """
    try:
        dataset.close()
    except Exception as exc:  # cleanup must never raise
        logger.warning("Error closing dataset %s: %s", dataset.name, exc)

    memfile = getattr(dataset, _MEMFILE_ATTR, None)
    if memfile is not None:
        try:
            memfile.close()
        except Exception as exc:  # cleanup must never raise
            logger.warning("Error closing in-memory file: %s", exc)


def _open_raster_sync(
    path: RasterPath,
    prefer_s3: bool = False,
) -> DatasetReader:
    """
    Open a raster or DEM archive and return the dataset directly.

    Unlike :func:`_open_raster`, this is not a context manager, which lets
    :func:`load_dem_mosaic` open several tiles concurrently in a thread
    pool (each open is I/O-bound: a remote COG's header fetch, or a local
    archive's extraction). The caller is responsible for closing the
    result, e.g. by registering :func:`_close_dataset` on an
    :class:`~contextlib.ExitStack`.

    Args:
        path (RasterPath): Raster path, URL, or ``.tar.gz`` / ``.tgz``
            archive.
        prefer_s3 (bool): If ``True`` and ``path`` is a plain HTTPS S3 URL
            string, rewrite it via :func:`_https_s3_url_to_vsis3` first.
            Ignored for archives and non-string paths.

    Returns:
        DatasetReader: Open dataset. The caller must close it.

    Raises:
        ValueError: If an archive contains no TIFF file.
        DEMReadError: If the raster or archive cannot be opened. When the
            path was rewritten to ``/vsis3/``, the message says so and
            suggests retrying with ``prefer_s3=False``.
    """
    import rasterio
    from rasterio.errors import RasterioError

    if str(path).endswith(_ARCHIVE_SUFFIXES):
        return _open_archive_dataset(path)

    target = path
    if prefer_s3 and isinstance(path, str):
        target = _https_s3_url_to_vsis3(path)

    try:
        return rasterio.open(target)
    except RasterioError as exc:
        hint = ""
        if target is not path:
            hint = (
                f" (rewritten from {path}; retry with prefer_s3=False to "
                "use plain HTTPS)"
            )
        raise DEMReadError(
            f"Could not open raster {target}{hint}: {exc}"
        ) from exc


@contextmanager
def _open_raster(path: RasterPath) -> Iterator[DatasetReader]:
    """
    Open a raster or DEM archive as a context manager.

    ``.tar.gz`` / ``.tgz`` archives (e.g. ArcticDEM mosaic tiles) are
    handled transparently: the DEM GeoTIFF member is extracted into memory
    (preferring ``*dem.tif`` over matchtag/count/browse siblings) and
    opened from there, with no extraction to disk. Plain raster paths are
    opened directly.

    Args:
        path (RasterPath): Raster path, URL, or ``.tar.gz`` / ``.tgz``
            archive.

    Yields:
        DatasetReader: Open dataset, closed automatically on exit.

    Raises:
        ValueError: If an archive contains no TIFF file.
        DEMReadError: If the raster or archive cannot be opened.
    """
    dataset = _open_raster_sync(path, prefer_s3=False)
    try:
        yield dataset
    finally:
        _close_dataset(dataset)


# ---------------------------------------------------------------------------
# Mosaic helpers
# ---------------------------------------------------------------------------

def _read_window_to_memory(
    src: ReadableDataset,
    window: Window,
    nodata: float | None,
) -> DatasetReader:
    """
    Copy one window of a dataset into a small, new in-memory dataset.

    Used by :func:`load_dem_mosaic` to move the slow, network-bound part
    of a merge (each source's windowed read) into a thread pool, while
    leaving :func:`rasterio.merge.merge` untouched. ``merge()`` still reads
    its sources one at a time internally, but against these RAM-resident
    copies, so those reads become effectively free.

    ``boundless=True`` matters: ``window`` comes from the AOI's shared
    merge bounds, which for any tile that does not cover the whole AOI on
    its own (the normal case when merging neighbours) extends beyond that
    tile's extent. The out-of-range part is filled with ``nodata`` (or 0)
    instead of raising, matching what a windowed read of the full mosaic
    would have produced.

    Args:
        src (ReadableDataset): Open dataset or ``WarpedVRT`` to read from.
        window (Window): Pixel window to read, in ``src``'s pixel grid.
        nodata (float | None): Fill value for out-of-range pixels, also
            written as the copy's nodata. ``None`` fills with 0.

    Returns:
        DatasetReader: Open in-memory dataset covering exactly ``window``.
            The caller must close it via :func:`_close_dataset`.

    Raises:
        DEMReadError: If the windowed read or the in-memory write fails.
    """
    from rasterio.errors import RasterioError
    from rasterio.io import MemoryFile

    fill_value = nodata if nodata is not None else 0
    try:
        data = src.read(window=window, boundless=True, fill_value=fill_value)
    except RasterioError as exc:
        raise DEMReadError(
            f"Windowed read of {src.name} failed: {exc}"
        ) from exc

    # Block/tiling options inherited from the source are meaningless for
    # a small in-memory copy and make GDAL emit CPLE_IllegalArg warnings.
    profile = src.profile.copy()
    for key in ("blockxsize", "blockysize", "tiled"):
        profile.pop(key, None)
    profile.update({
        "driver": "GTiff",
        "height": data.shape[1],
        "width": data.shape[2],
        "transform": src.window_transform(window),
        "count": data.shape[0],
    })
    if nodata is not None:
        profile["nodata"] = nodata

    memfile = MemoryFile()
    try:
        with memfile.open(**profile) as dst:
            dst.write(data)
        dataset = memfile.open()
    except RasterioError as exc:
        memfile.close()
        raise DEMReadError(
            f"Could not buffer window of {src.name} in memory: {exc}"
        ) from exc
    setattr(dataset, _MEMFILE_ATTR, memfile)
    return dataset


def _map_datasets_concurrently(
    func: Callable[[_T], DatasetReader],
    items: Sequence[_T],
    labels: Sequence[str],
    stack: ExitStack,
) -> list[DatasetReader]:
    """
    Run a dataset-producing function over ``items`` in a thread pool.

    Every dataset that is successfully produced is registered on
    ``stack`` for cleanup *before* any error is raised, so a single failed
    tile never leaks the handles (or in-memory buffers) of the tiles that
    did open. Results keep the input order, which matters for
    :func:`rasterio.merge.merge`'s "first valid pixel wins" rule.

    Args:
        func (Callable[[_T], DatasetReader]): Function returning an open
            dataset for one item.
        items (Sequence[_T]): Inputs, e.g. tile paths.
        labels (Sequence[str]): Human-readable label per item, used in
            error messages. Must be the same length as ``items``.
        stack (ExitStack): Stack on which to register dataset cleanup.

    Returns:
        list[DatasetReader]: Open datasets, in the same order as
            ``items``.

    Raises:
        DEMReadError: If any item fails. The message lists every failing
            item; the first failure is chained as ``__cause__``.
    """
    workers = max(1, min(_MAX_IO_WORKERS, len(items)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(func, item) for item in items]

    results: list[DatasetReader] = []
    failures: list[tuple[str, BaseException]] = []
    for future, label in zip(futures, labels):
        try:
            dataset = future.result()
        except Exception as exc:  # collected and re-raised below
            failures.append((label, exc))
            continue
        stack.callback(_close_dataset, dataset)
        results.append(dataset)

    if failures:
        details = "; ".join(f"{label}: {exc}" for label, exc in failures)
        raise DEMReadError(
            f"{len(failures)} of {len(items)} tile(s) failed: {details}"
        ) from failures[0][1]
    return results


def _validate_bounds(
    bounds: Sequence[float],
) -> tuple[float, float, float, float]:
    """
    Check that ``bounds`` is a finite, non-empty ``(minx, miny, maxx, maxy)``.

    Args:
        bounds (Sequence[float]): Candidate bounding box.

    Returns:
        tuple[float, float, float, float]: The bounds as floats.

    Raises:
        ValueError: If ``bounds`` does not have four finite numbers, or
            ``min >= max`` on either axis.
    """
    try:
        min_x, min_y, max_x, max_y = (float(v) for v in bounds)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"bounds must be four numbers (min_x, min_y, max_x, max_y); "
            f"got {bounds!r}"
        ) from exc

    if not all(math.isfinite(v) for v in (min_x, min_y, max_x, max_y)):
        raise ValueError(f"bounds must be finite; got {bounds!r}")
    if min_x >= max_x or min_y >= max_y:
        raise ValueError(
            f"bounds must satisfy min < max on both axes; got {bounds!r}"
        )
    return min_x, min_y, max_x, max_y


def _align_to_crs(
    src: DatasetReader,
    target_crs: CRSLike,
    label: str,
    stack: ExitStack,
) -> ReadableDataset:
    """
    Return ``src`` unchanged, or wrapped in a ``WarpedVRT`` to ``target_crs``.

    Tiles with no CRS, or already in ``target_crs``, are passed through.
    Otherwise they are reprojected on the fly with bilinear resampling;
    the VRT is registered on ``stack`` for cleanup.

    Args:
        src (DatasetReader): Open source tile.
        target_crs (CRSLike): CRS the mosaic is merged into.
        label (str): Tile label for error messages.
        stack (ExitStack): Stack on which to register the VRT.

    Returns:
        ReadableDataset: ``src`` itself, or a ``WarpedVRT`` over it.

    Raises:
        DEMReadError: If the reprojection cannot be set up.
    """
    from rasterio.enums import Resampling
    from rasterio.errors import CRSError, RasterioError
    from rasterio.vrt import WarpedVRT

    if src.crs is None or str(src.crs).upper() == str(target_crs).upper():
        return src

    try:
        vrt = WarpedVRT(
            src,
            crs=target_crs,
            resampling=Resampling.bilinear,
            warp_mem_limit=256,
            warp_extras={"NUM_THREADS": "ALL_CPUS"},
        )
    except (CRSError, RasterioError) as exc:
        raise DEMReadError(
            f"Could not reproject {label} from {src.crs} to {target_crs}: "
            f"{exc}"
        ) from exc
    return stack.enter_context(vrt)


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_dem_mosaic(
    paths: Sequence[RasterPath],
    target_crs: CRSLike | None = None,
    bounds: Sequence[float] | None = None,
    bounds_crs: CRSLike = "EPSG:4326",
    prefer_s3: bool = True,
) -> tuple[npt.NDArray[np.float32], float, Affine, CRSLike | None]:
    """
    Merge two or more DEM tiles into a single seamless array.

    Typical use is stitching neighbouring ArcticDEM / REMA mosaic tiles.
    Any mix of plain rasters, remote COGs and ``.tar.gz`` / ``.tgz``
    archives is accepted.

    Processing steps::

        open tiles (concurrently)
            -> reproject each to target_crs if needed (WarpedVRT)
            -> [bounds given] read each tile's AOI window (concurrently)
            -> rasterio.merge.merge  ("first valid pixel wins")
            -> nodata -> NaN

    When ``bounds`` is given, only the intersecting window is read from
    each tile rather than its full footprint. For remote COGs this means
    GDAL fetches only the needed blocks -- often dramatically less data
    when the AOI is small relative to the tiles. Without ``bounds``, each
    tile's full extent goes straight through ``merge()`` with no
    pre-fetch: parallelising a handful of very large full-tile reads
    trades much higher peak memory for an uncertain gain, so that path is
    deliberately left sequential.

    Args:
        paths (Sequence[RasterPath]): Two or more tile paths, URLs or
            archives. Order matters: where tiles overlap, the earliest
            valid pixel wins.
        target_crs (CRSLike | None): CRS to merge into. Defaults to the
            first tile's own CRS.
        bounds (Sequence[float] | None): ``(min_x, min_y, max_x, max_y)``
            to clip the merge to, in ``bounds_crs``. ``None`` merges every
            tile's full extent.
        bounds_crs (CRSLike): CRS of ``bounds``; reprojected internally to
            ``target_crs``. Defaults to WGS84 lon/lat.
        prefer_s3 (bool): Rewrite plain-HTTPS S3 tile URLs to GDAL's
            ``/vsis3/`` path before opening (see
            :func:`_https_s3_url_to_vsis3`). A no-op for local files and
            non-S3 hosts, so safe to leave on. Set ``False`` if ``/vsis3/``
            causes problems, e.g. an older GDAL without S3 support, or a
            network that allows HTTPS but blocks direct AWS access.

    Returns:
        tuple: A 4-tuple ``(dem, cellsize, transform, crs)``:

            - ``dem`` (np.ndarray): 2-D ``float32`` elevations, nodata as
              ``NaN``.
            - ``cellsize`` (float): Pixel width in ``crs`` units.
            - ``transform`` (Affine): Affine transform of ``dem``.
            - ``crs`` (CRSLike | None): The CRS the mosaic was merged
              into, or ``None`` if no tile carried a CRS.

    Raises:
        TypeError: If ``paths`` is a single string / path rather than a
            sequence of them.
        ValueError: If fewer than two paths are given, ``bounds`` is
            malformed or cannot be reprojected, ``bounds`` is used with
            tiles that have no CRS, or an archive contains no TIFF.
        DEMReadError: If any tile cannot be opened, reprojected or read,
            or the merge itself fails. Every tile opened before the
            failure is closed.

    Warning:
        Tiles with a CRS but no nodata value are merged using the first
        tile's nodata; mixing nodata conventions across tiles can leave
        sentinel values in the result.

    Examples:
        >>> dem, cellsize, transform, crs = load_dem_mosaic(
        ...     ["tile_a.tar.gz", "tile_b.tar.gz"],
        ...     bounds=(-50.2, 69.1, -49.8, 69.3),
        ... )
    """
    import rasterio
    from rasterio.errors import CRSError, RasterioError
    from rasterio.merge import merge as rio_merge
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds as window_from_bounds

    # A bare string is itself a Sequence, and list("a.tif") silently
    # produces single-character "paths"; reject it with a clear message.
    if isinstance(paths, (str, bytes, os.PathLike)):
        raise TypeError(
            "load_dem_mosaic expects a list of tile paths, got a single "
            f"path {paths!r}; use load_dem() for one tile"
        )
    paths = list(paths)
    if len(paths) < 2:
        raise ValueError("load_dem_mosaic needs at least two tile paths")
    if bounds is not None:
        bounds = _validate_bounds(bounds)

    labels = [str(p) for p in paths]
    open_tile = _in_gdal_s3_env(
        functools.partial(_open_raster_sync, prefer_s3=prefer_s3)
    )

    # AWS_NO_SIGN_REQUEST is harmless when nothing is rewritten to
    # /vsis3/: it only affects GDAL's S3 driver.
    with rasterio.Env(**_GDAL_S3_ENV), ExitStack() as stack:
        srcs = _map_datasets_concurrently(open_tile, paths, labels, stack)

        if target_crs is None:
            target_crs = srcs[0].crs
        if target_crs is None:
            logger.warning(
                "First tile %s has no CRS; merging tiles as-is without "
                "reprojection.", labels[0],
            )
            aligned: list[ReadableDataset] = list(srcs)
        else:
            aligned = [
                _align_to_crs(src, target_crs, label, stack)
                for src, label in zip(srcs, labels)
            ]

        nodata = aligned[0].nodata

        merge_bounds = None
        if bounds is not None:
            if target_crs is None:
                raise ValueError(
                    "bounds cannot be applied: tiles have no CRS and no "
                    "target_crs was given"
                )
            try:
                merge_bounds = transform_bounds(
                    bounds_crs, target_crs, *bounds
                )
            except (CRSError, RasterioError) as exc:
                raise ValueError(
                    f"Could not reproject bounds {bounds} from {bounds_crs} "
                    f"to {target_crs}: {exc}"
                ) from exc

        merge_sources: list[ReadableDataset] = aligned
        if merge_bounds is not None:
            # merge() reads its sources one at a time, and those windowed
            # reads are the network-bound part -- so pre-fetch every
            # tile's window concurrently and hand merge() RAM-resident
            # copies instead.
            prefetch = _in_gdal_s3_env(
                lambda pair: _read_window_to_memory(pair[0], pair[1], nodata)
            )
            windows = [
                window_from_bounds(*merge_bounds, transform=src.transform)
                for src in aligned
            ]
            merge_sources = _map_datasets_concurrently(
                prefetch, list(zip(aligned, windows)), labels, stack
            )

        try:
            mosaic, transform = rio_merge(
                merge_sources, bounds=merge_bounds, nodata=nodata
            )
        except (RasterioError, ValueError) as exc:
            raise DEMReadError(
                f"Merging {len(paths)} tiles failed: {exc}"
            ) from exc

    dem = mosaic[0].astype(np.float32)
    if nodata is not None:
        dem = np.where(dem == nodata, np.nan, dem)
    if np.isnan(dem).all():
        logger.warning(
            "Mosaic contains no valid elevations; check that bounds "
            "overlap the tiles."
        )
    return dem, float(transform.a), transform, target_crs


def _synthetic_dem(
    shape: tuple[int, int],
) -> tuple[npt.NDArray[np.float32], float]:
    """
    Build a deterministic synthetic DEM: three hills, a valley and noise.

    Args:
        shape (tuple[int, int]): Grid size as ``(rows, cols)``.

    Returns:
        tuple[np.ndarray, float]: ``(dem, cellsize)`` with a ``float32``
            DEM and a fixed 10 m cellsize.

    Raises:
        ValueError: If ``shape`` is not two positive integers.
    """
    try:
        rows, cols = (int(n) for n in shape)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"shape must be two integers (rows, cols); got {shape!r}"
        ) from exc
    if rows <= 0 or cols <= 0:
        raise ValueError(f"shape must be positive; got {shape!r}")

    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:rows, 0:cols]
    dem = np.zeros((rows, cols), dtype=np.float32)

    # (centre_x, centre_y, amplitude_m, sigma_cells)
    hills = [(110, 110, 70, 70), (300, 250, 90, 60), (180, 330, 55, 45)]
    for cx, cy, amp, sigma in hills:
        dem += amp * np.exp(
            -(((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
        )

    # A sinuous valley cutting across the grid.
    valley_centre = 200 + 60 * np.sin(y / 40.0)
    dem -= 35 * np.exp(-((x - valley_centre) ** 2) / (2 * 25 ** 2))

    dem += gaussian_filter(rng.standard_normal((rows, cols)), sigma=6) * 4
    return dem, 10.0


def load_dem(
    path: RasterPath | Sequence[RasterPath] | None = None,
    shape: tuple[int, int] = (400, 400),
) -> tuple[npt.NDArray[np.float32], float]:
    """
    Load a DEM from disk, merge a mosaic, or build the synthetic demo DEM.

    Dispatch depends on ``path``:

    - A single path, URL or ``.tar.gz`` / ``.tgz`` archive: read band 1.
    - A list or tuple of two or more paths: merged via
      :func:`load_dem_mosaic`. Its transform and CRS are dropped here for
      signature compatibility; call :func:`load_dem_mosaic` directly if
      you need them.
    - ``None``: build a deterministic synthetic DEM of ``shape`` cells.

    Nodata cells are converted to ``NaN`` rather than left as sentinel
    values, so they don't corrupt derivatives downstream.
    ``curvatures()`` and ``hillshade()`` nan-fill internally for the
    calculation and remask the result afterwards.

    Args:
        path (RasterPath | Sequence[RasterPath] | None): What to load; see
            above. Defaults to ``None`` (synthetic DEM).
        shape (tuple[int, int]): ``(rows, cols)`` of the synthetic DEM.
            Ignored when ``path`` is given.

    Returns:
        tuple[np.ndarray, float]: ``(dem, cellsize)``, where ``dem`` is a
            2-D ``float32`` array and ``cellsize`` is the pixel width in
            CRS units (10.0 for the synthetic DEM).

    Raises:
        ValueError: If ``shape`` is invalid, a mosaic has fewer than two
            paths, or an archive contains no TIFF.
        DEMReadError: If the raster, archive or mosaic cannot be read.

    Examples:
        >>> dem, cellsize = load_dem()
        >>> dem.shape, cellsize
        ((400, 400), 10.0)
    """
    from rasterio.errors import RasterioError

    if isinstance(path, (list, tuple)):
        dem, cellsize, _transform, _crs = load_dem_mosaic(path)
        return dem, cellsize

    if path is None:
        return _synthetic_dem(shape)

    with _open_raster(path) as src:
        try:
            dem = src.read(1).astype(np.float32)
        except RasterioError as exc:
            raise DEMReadError(f"Could not read {path}: {exc}") from exc
        cellsize = float(src.transform.a)
        if src.nodata is not None:
            dem = np.where(dem == src.nodata, np.nan, dem)
    return dem, cellsize


# ---------------------------------------------------------------------------
# Array utilities
# ---------------------------------------------------------------------------

def _fill_nan_nearest(
    arr: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.bool_]]:
    """
    Fill NaNs with the value of the nearest valid pixel.

    Lets derivatives be computed across small voids and edges without the
    NaNs poisoning the whole array. Callers use the returned mask to
    remask their results afterwards.

    Args:
        arr (np.ndarray): 2-D array that may contain NaNs.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(filled, nan_mask)``. ``filled``
            is ``arr`` itself when there are no NaNs (no copy), and also
            when every cell is NaN (nothing to fill from; a warning is
            logged).
    """
    mask = np.isnan(arr)
    if not mask.any():
        return arr, mask
    if mask.all():
        logger.warning("Array is entirely NaN; nothing to fill from.")
        return arr, mask

    idx = distance_transform_edt(
        mask, return_distances=False, return_indices=True
    )
    return arr[tuple(idx)], mask
