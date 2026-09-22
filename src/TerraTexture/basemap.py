"""
Drape DEM relief over basemap imagery with the "luminosity blend" recipe.

Reproduces the ArcGIS Pro / Photoshop relief-over-imagery technique:
terrain derivatives are blended into one greyscale relief layer, which
then replaces the *lightness* of satellite imagery while keeping its hue
and saturation.

Public API:

- :func:`plot_dem_basemap_luminosity_relief` -- build the relief basemap
  and plot it in its own figure; returns every intermediate layer.
- :func:`add_relief_basemap` -- build it once and draw it as the
  background of one or more existing Axes (e.g. panels of a
  ``subplot_mosaic`` figure).
- :class:`BasemapFetchError` -- raised when imagery tiles can't be
  fetched or are unusable.

Pipeline::

    DEM (file, tile list, or public STAC AOI query)
      |-- curvatures -> profile x planform  --+  soft-light group,
      |-- hillshade                         --+  stretched +/-N std
      '-- normalised elevation              --+
                                               v
                                   relief luminosity (grey)
    basemap tiles (EPSG:3857) --warp onto DEM grid--> basemap RGB
                                               v
          luminosity_blend(basemap, relief) -> soft_light(.., basemap)

Everything is composited on **one grid**: the DEM's own, in
``target_crs``. A DEM already in that CRS is used unmodified at full
resolution; only the imagery is warped, so every layer is
pixel-aligned before blending.

Error handling:
    Bad arguments raise ``ValueError`` before any download starts.
    DEM loading failures raise :class:`TerraTexture.io.DEMReadError`,
    STAC failures :class:`TerraTexture.sources.STACError`, and imagery
    failures :class:`BasemapFetchError` -- each naming the source,
    bounds and zoom involved.

Logging:
    Progress (DEM loaded, tiles fetched, figure saved) is logged at INFO
    under ``TerraTexture.basemap``; details at DEBUG. Imagery that
    covers only part of the DEM, non-square pixels and geographic target
    CRSs are logged as WARNINGs. Enable with
    ``logging.basicConfig(level=logging.INFO)``. The ``profile=True``
    timing table is printed, since it's explicitly requested output.

Dependencies:
    rasterio and contextily (the ``basemap`` extra). The ``aoi_bounds``
    path also needs ``requests`` (see :mod:`TerraTexture.sources`) and
    queries PGC's public STAC API: no signup, API key or local software.

Examples:
    Greenland AOI from lon/lat bounds, saved to disk::

        fig, ax, layers = plot_dem_basemap_luminosity_relief(
            aoi_bounds=(-51.3, 69.1, -50.9, 69.3),
            out_fig="ilulissat.png",
            show=False,
        )

    Relief background under two data panels::

        fig, (ax1, ax2) = plt.subplots(1, 2)
        layers = add_relief_basemap([ax1, ax2], dem_path="tile_dem.tif")
"""

from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from .blend import _lum, luminosity_blend, soft_light
from .derivatives import curvatures, hillshade
from .io import DEMReadError, _open_raster, load_dem_mosaic
from .sources import arcticdem_mosaic, rema_mosaic
from .stretch import normalize, stretch_std

if TYPE_CHECKING:
    from affine import Affine
    from matplotlib.figure import Figure

    import numpy.typing as npt


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# AOI products and the functions that fetch them from PGC's STAC API.
_AOI_PRODUCTS: dict[str, Callable[..., Any]] = {
    "arcticdem": arcticdem_mosaic,
    "rema": rema_mosaic,
}

# Native CRS of each AOI product (the default target_crs).
_AOI_NATIVE_CRS = {
    "arcticdem": "EPSG:3413",  # NSIDC Sea Ice Polar Stereographic North
    "rema": "EPSG:3031",       # Antarctic Polar Stereographic
}

# Web-map tile zoom levels contextily / XYZ providers accept.
_MAX_ZOOM = 23

# Warn when imagery covers less than this fraction of the DEM grid.
_MIN_BASEMAP_COVERAGE = 0.99

# Warn when |pixel height| and pixel width differ by more than this.
_SQUARE_PIXEL_TOLERANCE = 0.01


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class BasemapFetchError(RuntimeError):
    """
    Raised when basemap imagery tiles can't be fetched or are unusable.

    Wraps whatever contextily / the network raised (chained as
    ``__cause__``) with the provider, zoom and bounds involved.
    """


# ---------------------------------------------------------------------------
# Stage timer
# ---------------------------------------------------------------------------

class _StageTimer:
    """
    Minimal per-stage wall-clock timer for ``profile=True``.

    Answers "which *stage* of this pipeline is slow" (DEM fetch, compute,
    tile fetch, reprojection, blending, plotting) with no extra installs.
    Not a general profiler: for line-by-line detail, including inside
    rasterio/contextily, use ``%prun`` or ``%lprun``.

    Repeated or nested ``with timer("name")`` blocks are supported;
    repeated names accumulate. Time is recorded even if the block raises.

    Attributes:
        stages (OrderedDict[str, float]): Seconds per stage, in first-use
            order.

    Examples:
        >>> timer = _StageTimer()
        >>> with timer("compute"):
        ...     pass
        >>> list(timer.stages)
        ['compute']
    """

    def __init__(self) -> None:
        """Create a timer with no recorded stages."""
        self.stages: OrderedDict[str, float] = OrderedDict()

    @contextmanager
    def __call__(self, name: str) -> Iterator[None]:
        """
        Time the enclosed block under ``name``.

        Args:
            name (str): Stage name.

        Yields:
            None
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.stages[name] = self.stages.get(name, 0.0) + elapsed
            logger.debug("Stage %s took %.3f s", name, elapsed)

    def format_report(self) -> str:
        """
        Format the stages as a table, slowest first.

        Returns:
            str: The table, or ``""`` if nothing was timed.
        """
        total = sum(self.stages.values())
        if total <= 0:
            return ""
        width = max(len(name) for name in self.stages) + 1
        header = f"{'stage':<{width}} {'seconds':>10} {'% of total':>12}"
        rule = "-" * len(header)
        lines = [header, rule]
        for name, seconds in sorted(
            self.stages.items(), key=lambda kv: kv[1], reverse=True
        ):
            lines.append(
                f"{name:<{width}} {seconds:>10.3f} {100 * seconds / total:>11.1f}%"
            )
        lines += [rule, f"{'TOTAL':<{width}} {total:>10.3f} {100.0:>11.1f}%"]
        return "\n".join(lines)

    def report(self) -> None:
        """
        Print the stage table (the output ``profile=True`` asks for).

        Returns:
            None
        """
        table = self.format_report()
        if table:
            print(f"\n{table}\n")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _same_crs(a: Any, b: Any) -> bool:
    """
    Return whether two CRS specs describe the same CRS.

    Compares parsed CRS objects, so ``"EPSG:3413"``, ``"epsg:3413"`` and a
    ``CRS`` object all match; falls back to case-insensitive strings if
    either can't be parsed.

    Args:
        a (Any): CRS object or string.
        b (Any): CRS object or string.

    Returns:
        bool: ``True`` if they are the same CRS.
    """
    from rasterio.crs import CRS
    from rasterio.errors import CRSError

    try:
        return CRS.from_user_input(a) == CRS.from_user_input(b)
    except CRSError:
        return str(a).upper() == str(b).upper()


def _validate_options(
    dem_path: Any,
    aoi_bounds: Any,
    dem_product: str,
    zoom: int | str,
    tile_connections: int,
    relief_strength: float,
) -> None:
    """
    Check the arguments that can be checked before any I/O.

    Args:
        dem_path (Any): See :func:`plot_dem_basemap_luminosity_relief`.
        aoi_bounds (Any): See :func:`plot_dem_basemap_luminosity_relief`.
        dem_product (str): See :func:`plot_dem_basemap_luminosity_relief`.
        zoom (int | str): See :func:`plot_dem_basemap_luminosity_relief`.
        tile_connections (int): See
            :func:`plot_dem_basemap_luminosity_relief`.
        relief_strength (float): See
            :func:`plot_dem_basemap_luminosity_relief`.

    Returns:
        None

    Raises:
        ValueError: If both or neither of ``dem_path`` / ``aoi_bounds``
            are given, or any option is out of range.
    """
    if dem_path is None and aoi_bounds is None:
        raise ValueError(
            "Provide either dem_path (file, .tar.gz or list of tiles) or "
            "aoi_bounds (to query PGC's public STAC API for that AOI)."
        )
    if dem_path is not None and aoi_bounds is not None:
        raise ValueError(
            "Provide dem_path OR aoi_bounds, not both (aoi_bounds would "
            "silently win)."
        )
    if aoi_bounds is not None and dem_product not in _AOI_PRODUCTS:
        raise ValueError(
            f"dem_product must be one of {sorted(_AOI_PRODUCTS)}, "
            f"got {dem_product!r}"
        )
    if isinstance(dem_path, (list, tuple)) and not dem_path:
        raise ValueError("dem_path is an empty list; give at least one tile")

    is_int_zoom = isinstance(zoom, int) and not isinstance(zoom, bool)
    if zoom != "auto" and not (is_int_zoom and 0 <= zoom <= _MAX_ZOOM):
        raise ValueError(
            f"zoom must be 'auto' or an integer 0-{_MAX_ZOOM}; got {zoom!r}"
        )
    if (
        not isinstance(tile_connections, int)
        or isinstance(tile_connections, bool)
        or tile_connections < 1
    ):
        raise ValueError(
            f"tile_connections must be a positive integer; got {tile_connections!r}"
        )
    if not (isinstance(relief_strength, (int, float)) and 0 <= relief_strength <= 1):
        raise ValueError(
            f"relief_strength must be between 0 and 1; got {relief_strength!r}"
        )


def _set_tile_cache(tile_cache_dir: str) -> None:
    """
    Point contextily's tile cache at a persistent directory.

    contextily's default cache is a temp dir deleted at interpreter exit,
    so every new process starts cold unless this is set.

    Args:
        tile_cache_dir (str): Directory to create (if needed) and use.

    Returns:
        None

    Raises:
        OSError: If the directory can't be created, naming the path.
    """
    import contextily as ctx

    path = os.path.expanduser(tile_cache_dir)
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        raise OSError(f"Cannot create tile cache directory {path}: {exc}") from exc
    ctx.set_cache_dir(path)
    logger.debug("Tile cache directory: %s", path)


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def _load_dem(
    dem_path: Any,
    aoi_bounds: Sequence[float] | None,
    aoi_bounds_crs: str,
    dem_product: str,
    resolution: int,
    target_crs: str | None,
) -> tuple[npt.NDArray[np.float32], Affine, str]:
    """
    Load the DEM onto the compositing grid.

    Resolution of ``target_crs=None``:

    - ``aoi_bounds``: the product's native CRS (EPSG:3413 / EPSG:3031).
    - ``dem_path`` (file or tiles): the DEM's own CRS, so it is used
      unmodified with no resampling.

    Args:
        dem_path (Any): File path, archive, or list of tile paths.
        aoi_bounds (Sequence[float] | None): AOI to query instead.
        aoi_bounds_crs (str): CRS of ``aoi_bounds``.
        dem_product (str): ``"arcticdem"`` or ``"rema"``.
        resolution (int): Mosaic resolution in metres (AOI path only).
        target_crs (str | None): Compositing CRS, or ``None`` (above).

    Returns:
        tuple: ``(dem, transform, crs)`` -- float32 DEM with NaN nodata,
            its affine transform, and the CRS as a string.

    Raises:
        ValueError: If the DEM has no CRS, or ``target_crs`` is invalid.
        TerraTexture.io.DEMReadError: If the DEM can't be read.
        TerraTexture.sources.STACError: If the AOI query fails.
    """
    import rasterio
    from rasterio.errors import CRSError, RasterioError
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    if aoi_bounds is not None:
        if target_crs is None:
            target_crs = _AOI_NATIVE_CRS[dem_product]
        logger.info(
            "Querying %s mosaics for %s (%s)", dem_product, tuple(aoi_bounds),
            aoi_bounds_crs,
        )
        dem, _cellsize, transform, crs = _AOI_PRODUCTS[dem_product](
            aoi_bounds,
            resolution=resolution,
            bbox_crs=aoi_bounds_crs,
            target_crs=target_crs,
        )
        return dem, transform, str(crs)

    if isinstance(dem_path, (list, tuple)):
        dem, _cellsize, transform, crs = load_dem_mosaic(
            dem_path, target_crs=target_crs
        )
        if crs is None:
            raise ValueError(
                "DEM tiles have no CRS, so basemap imagery can't be aligned "
                "to them; pass target_crs or use georeferenced tiles."
            )
        return dem, transform, str(crs)

    with _open_raster(dem_path) as src:
        if src.crs is None:
            raise ValueError(
                f"DEM {dem_path} has no CRS, so basemap imagery can't be "
                "aligned to it."
            )
        if target_crs is None or _same_crs(src.crs, target_crs):
            try:
                dem = src.read(1).astype(np.float32)
            except RasterioError as exc:
                raise DEMReadError(f"Could not read {dem_path}: {exc}") from exc
            if src.nodata is not None:
                dem = np.where(dem == src.nodata, np.nan, dem)
            crs = str(src.crs) if target_crs is None else str(target_crs)
            return dem, src.transform, crs

        logger.info("Reprojecting DEM from %s to %s", src.crs, target_crs)
        try:
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )
            dem = np.full((height, width), np.nan, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=dem,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                src_nodata=src.nodata,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
        except CRSError as exc:
            raise ValueError(f"Invalid target_crs {target_crs!r}: {exc}") from exc
        except RasterioError as exc:
            raise DEMReadError(
                f"Could not reproject {dem_path} to {target_crs}: {exc}"
            ) from exc
        return dem, transform, str(target_crs)


def _check_grid(dem: np.ndarray, transform: Affine, crs: str) -> None:
    """
    Log warnings for grids that will give misleading relief.

    Args:
        dem (np.ndarray): The loaded DEM.
        transform (Affine): Its affine transform.
        crs (str): Its CRS.

    Returns:
        None
    """
    from rasterio.crs import CRS
    from rasterio.errors import CRSError

    width, height = abs(transform.a), abs(transform.e)
    logger.info(
        "DEM grid: %d x %d cells, %.4g x %.4g units, %s",
        dem.shape[0], dem.shape[1], width, height, crs,
    )
    if width and abs(width - height) / width > _SQUARE_PIXEL_TOLERANCE:
        logger.warning(
            "Non-square pixels (%.4g x %.4g); derivatives assume square "
            "cells and use the width.", width, height,
        )
    try:
        if CRS.from_user_input(crs).is_geographic:
            logger.warning(
                "Compositing CRS %s is geographic (degrees); slopes and "
                "curvature will be wrong. Pass a projected target_crs.", crs,
            )
    except CRSError:
        logger.debug("Could not parse CRS %r for the geographic check", crs)


def _compute_relief(
    dem: np.ndarray,
    cellsize: float,
    azimuth: float,
    altitude: float,
    curvature_std: float,
    hillshade_std: float,
    timer: _StageTimer,
) -> dict[str, np.ndarray]:
    """
    Build the greyscale relief group from the DEM.

    Layer stack, bottom to top, each soft-lit onto the composite below::

        hillshade (+/-N std)
        profile x planform curvature (+/-N std, soft-lit together)
        elevation, white -> black (percentile-normalised)

    Args:
        dem (np.ndarray): float32 DEM, NaN for nodata.
        cellsize (float): Pixel size in metres.
        azimuth (float): Sun azimuth, degrees clockwise from north.
        altitude (float): Sun altitude, degrees.
        curvature_std (float): N for the curvature stretch.
        hillshade_std (float): N for the hillshade stretch.
        timer (_StageTimer): Records stage timings.

    Returns:
        dict[str, np.ndarray]: ``dem_grey``, ``curvature``, ``hillshade``,
            ``relief_luminosity`` and ``texture_luminosity``, each (H, W).
            The two luminosity layers have NaN replaced by neutral 0.5.
    """
    with timer("curvature_compute"):
        profile_curv, planform_curv = curvatures(dem, cellsize)
    with timer("hillshade_compute"):
        shade = hillshade(dem, cellsize, azimuth=azimuth, altitude=altitude)

    with timer("stretch_normalize"):
        profile_grey = stretch_std(profile_curv, curvature_std)
        planform_grey = stretch_std(planform_curv, curvature_std)
        hs_grey = stretch_std(shade, hillshade_std)
        dem_grey = normalize(dem)

    with timer("relief_blend"):
        curvature_combo = soft_light(profile_grey, planform_grey)
        group_composite = soft_light(hs_grey, curvature_combo)
        relief_luminosity = soft_light(group_composite, dem_grey)

    # Texture-only luminosity (hillshade + curvature, no elevation):
    # locally varying without dem_grey's broad darkening of low ground.
    # Exposed so a burned-in data layer can show topographic texture
    # without elevation crushing its colour.
    # NaN (voids) -> neutral mid-grey, so voids don't distort blending.
    return {
        "dem_grey": dem_grey,
        "curvature": curvature_combo,
        "hillshade": hs_grey,
        "relief_luminosity": np.nan_to_num(relief_luminosity, nan=0.5),
        "texture_luminosity": np.nan_to_num(group_composite, nan=0.5),
    }


def _provider_name(source: Any) -> str:
    """
    Return a short human-readable name for a tile provider.

    Args:
        source (Any): contextily / xyzservices provider, or a URL.

    Returns:
        str: The provider's ``name`` if it has one, else its string form.
    """
    name = getattr(source, "name", None)
    return name if isinstance(name, str) else str(source)[:80]


def _fetch_basemap(
    bounds: tuple[float, float, float, float],
    crs: str,
    transform: Affine,
    shape: tuple[int, int],
    source: Any,
    zoom: int | str,
    tile_connections: int,
    timer: _StageTimer,
) -> np.ndarray:
    """
    Fetch web-map tiles and warp them onto the DEM grid.

    Tiles arrive in EPSG:3857 as RGBA ``uint8``; the alpha band is
    dropped and the RGB bands are warped in one multi-band ``reproject``
    call onto exactly the DEM's transform, shape and CRS. Cells the
    imagery doesn't reach are filled with black, and a warning reports
    the coverage when it is incomplete.

    Args:
        bounds (tuple[float, float, float, float]): DEM ``(west, south,
            east, north)`` in ``crs``.
        crs (str): DEM CRS.
        transform (Affine): DEM transform.
        shape (tuple[int, int]): DEM ``(height, width)``.
        source (Any): contextily tile provider.
        zoom (int | str): Tile zoom level, or ``"auto"``.
        tile_connections (int): Parallel tile downloads.
        timer (_StageTimer): Records stage timings.

    Returns:
        np.ndarray: ``(H, W, 3)`` float32 RGB in ``[0, 1]``.

    Raises:
        ValueError: If the DEM bounds can't be converted to lon/lat.
        BasemapFetchError: If the tiles can't be fetched, or aren't an
            RGB(A) image.
    """
    import contextily as ctx
    from rasterio.errors import CRSError
    from rasterio.transform import from_bounds
    from rasterio.warp import Resampling, reproject, transform_bounds

    height, width = shape
    provider = _provider_name(source)

    with timer("tile_fetch"):
        try:
            lon_lat = transform_bounds(crs, "EPSG:4326", *bounds)
        except CRSError as exc:
            raise ValueError(f"Cannot convert DEM bounds from {crs}: {exc}") from exc
        logger.info(
            "Fetching %s tiles (zoom=%s) for lon/lat %s",
            provider, zoom, tuple(round(v, 5) for v in lon_lat),
        )
        try:
            image, extent = ctx.bounds2img(
                *lon_lat, zoom=zoom, source=source, ll=True,
                n_connections=tile_connections,
            )
        except Exception as exc:  # network, HTTP, PIL and provider errors
            raise BasemapFetchError(
                f"Could not fetch basemap tiles from {provider} (zoom={zoom}) "
                f"for lon/lat bounds {lon_lat}: {type(exc).__name__}: {exc}"
            ) from exc

        image = np.asarray(image)
        if image.ndim != 3 or image.shape[-1] < 3:
            raise BasemapFetchError(
                f"{provider} returned an image of shape {image.shape}; "
                "expected (rows, cols, 3 or 4)"
            )
        logger.debug("Fetched tile mosaic %s, extent %s", image.shape, extent)
        bm_west, bm_east, bm_south, bm_north = extent
        bm_transform = from_bounds(
            bm_west, bm_south, bm_east, bm_north, image.shape[1], image.shape[0]
        )

    with timer("tile_reproject"):
        # One multi-band warp instead of three single-band calls. contextily
        # always returns RGBA (Image.convert("RGBA") in contextily/tile.py),
        # so [..., :3] drops alpha; a 4-band source against a 3-band
        # destination makes reproject() raise "Invalid destination shape".
        src_bands = np.ascontiguousarray(
            np.moveaxis(image[:, :, :3].astype(np.float32), 2, 0)
        )
        # NaN-initialised so uncovered cells are detectable (np.empty
        # previously left them as uninitialised memory).
        dst_bands = np.full((3, height, width), np.nan, dtype=np.float32)
        reproject(
            source=src_bands,
            destination=dst_bands,
            src_transform=bm_transform,
            src_crs="EPSG:3857",
            dst_transform=transform,
            dst_crs=crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
            num_threads=os.cpu_count() or 1,
        )

    coverage = float(np.isfinite(dst_bands[0]).mean())
    if coverage < _MIN_BASEMAP_COVERAGE:
        logger.warning(
            "Basemap imagery covers only %.1f%% of the DEM grid; the rest "
            "is filled with black.", 100 * coverage,
        )
    rgb = np.moveaxis(np.nan_to_num(dst_bands, nan=0.0), 0, 2)
    return np.clip(rgb / 255.0, 0, 1)


def _composite(
    basemap_rgb: np.ndarray,
    relief_luminosity: np.ndarray,
    relief_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Blend the relief into the imagery.

    ``luminosity_blend`` replaces the imagery's lightness with the relief
    (optionally mixed with the imagery's own lightness by
    ``relief_strength``), then the imagery is soft-lit on top again to
    restore the colour contrast luminosity blending flattens.

    Args:
        basemap_rgb (np.ndarray): ``(H, W, 3)`` imagery in ``[0, 1]``.
        relief_luminosity (np.ndarray): ``(H, W)`` relief in ``[0, 1]``.
        relief_strength (float): 1 = relief replaces the imagery's
            lightness; 0 = imagery lightness unchanged.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(luminosity_composite, final)``,
            both ``(H, W, 3)``.
    """
    if relief_strength < 1.0:
        target = (
            relief_strength * relief_luminosity
            + (1 - relief_strength) * _lum(basemap_rgb)
        )
    else:
        target = relief_luminosity
    luminosity_composite = luminosity_blend(basemap_rgb, target)
    return luminosity_composite, soft_light(luminosity_composite, basemap_rgb)


def _plot(
    final: np.ndarray,
    extent: tuple[float, float, float, float],
    source: Any,
    figsize: tuple[float, float],
    out_fig: str | os.PathLike[str] | None,
    show: bool,
) -> tuple[Figure, Axes]:
    """
    Show the composite in a new figure, with attribution; save/show it.

    Args:
        final (np.ndarray): ``(H, W, 3)`` image.
        extent (tuple[float, float, float, float]): ``(west, east, south,
            north)`` for ``imshow``.
        source (Any): Tile provider, for its attribution text.
        figsize (tuple[float, float]): Figure size in inches.
        out_fig (str | os.PathLike[str] | None): Path to save to (600 dpi).
        show (bool): Whether to call ``plt.show()``.

    Returns:
        tuple[Figure, Axes]: The new figure and axes.

    Raises:
        OSError: If ``out_fig`` can't be written (the figure is closed).
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(final, extent=extent)
    ax.set_xticks([])
    ax.set_yticks([])
    attribution = getattr(source, "attribution", "")
    if isinstance(attribution, str) and attribution:
        ax.text(
            0.01, 0.01, attribution, transform=ax.transAxes,
            fontsize=6, color="white", ha="left", va="bottom",
            bbox={"facecolor": "black", "alpha": 0.5, "pad": 1, "linewidth": 0},
        )
    fig.tight_layout()

    if out_fig:
        try:
            fig.savefig(out_fig, dpi=600)
        except OSError:
            plt.close(fig)
            raise
        logger.info("Saved figure to %s", out_fig)
    if show:
        plt.show()
    return fig, ax


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_dem_basemap_luminosity_relief(
    dem_path: str | os.PathLike[str] | Sequence[str | os.PathLike[str]] | None = None,
    aoi_bounds: Sequence[float] | None = None,
    aoi_bounds_crs: str = "EPSG:4326",
    dem_product: str = "arcticdem",
    arcticdem_resolution: int = 32,
    source: Any = None,
    zoom: int | str = "auto",
    target_crs: str | None = None,
    azimuth: float = 315,
    altitude: float = 45,
    curvature_std: float = 4,
    hillshade_std: float = 4,
    relief_strength: float = 1.0,
    figsize: tuple[float, float] = (10, 10),
    out_fig: str | os.PathLike[str] | None = None,
    show: bool = True,
    tile_cache_dir: str | None = None,
    tile_connections: int = 16,
    profile: bool = False,
) -> tuple[Figure, Axes, dict[str, Any]]:
    """
    Drape a DEM's relief over basemap imagery and plot it.

    Layer stack, top to bottom (ArcGIS Pro / Photoshop recipe)::

        1. basemap imagery                                [Soft Light]
        2. relief group                                   [Luminosity]
           - DEM elevation, white -> black                (top of group)
           - profile x planform curvature, +/-N std       (middle)
           - hillshade, +/-N std                          (bottom)
        3. basemap imagery                                [Normal, base]

    Within the group each layer soft-lights onto everything below it. The
    resulting greyscale relief replaces only the *lightness* of the
    imagery (Luminosity mode), so its hue and saturation are preserved,
    unlike a per-channel soft-light burn. The imagery is then soft-lit on
    top once more to restore the colour contrast luminosity blending
    flattens.

    All layers share the DEM's grid in ``target_crs``; only the imagery
    is warped (from EPSG:3857), so everything is pixel-aligned.

    Args:
        dem_path (str | PathLike | Sequence | None): A georeferenced DEM
            (GeoTIFF or ``.tar.gz`` / ``.tgz`` archive such as an
            ArcticDEM tile), or a list of tiles to merge first. Give this
            **or** ``aoi_bounds``, not both.
        aoi_bounds (Sequence[float] | None): ``(x_min, y_min, x_max,
            y_max)`` in ``aoi_bounds_crs``. Queries PGC's public STAC API
            for intersecting mosaic tiles and merges them into the DEM.
        aoi_bounds_crs (str): CRS of ``aoi_bounds``. Defaults to
            EPSG:4326 (lon/lat, e.g. from GPS or a web map); independent
            of ``target_crs``.
        dem_product (str): ``"arcticdem"`` (Arctic incl. Greenland) or
            ``"rema"`` (Antarctica). Used only with ``aoi_bounds``.
        arcticdem_resolution (int): Mosaic resolution in metres (2, 10 or
            32) for whichever ``dem_product`` is chosen. The name predates
            REMA support; it applies to both.
        source (Any): contextily tile provider. Defaults to
            ``contextily.providers.Esri.WorldImagery``.
        zoom (int | str): Tile zoom level (0-23) for the imagery fetch,
            or ``"auto"``.
        target_crs (str | None): Compositing CRS. ``None`` uses the DEM's
            native CRS: EPSG:3413 (ArcticDEM) or EPSG:3031 (REMA) for
            ``aoi_bounds``, or the file's own CRS for ``dem_path`` (so it
            is used unmodified). Must be projected (metres).
        azimuth (float): Sun azimuth for the hillshade, degrees clockwise
            from north.
        altitude (float): Sun altitude for the hillshade, degrees 0-90.
        curvature_std (float): N for the curvature's mean +/- N std
            stretch (ArcGIS Pro "Standard Deviation" symbology).
        hillshade_std (float): N for the hillshade's stretch.
        relief_strength (float): 0-1. How much the relief replaces the
            imagery's own lightness. 1.0 (default) replaces it fully;
            try 0.5-0.7 if the result looks too dark or grey. 0.0 leaves
            the imagery's lightness untouched (almost no relief).
        figsize (tuple[float, float]): Figure size in inches.
        out_fig (str | PathLike | None): Save the figure here at 600 dpi.
        show (bool): Call ``plt.show()`` at the end.
        tile_cache_dir (str | None): Persistent directory for downloaded
            tiles. contextily's default cache is a temp dir deleted at
            exit, so every new process starts cold unless this is set.
        tile_connections (int): Parallel tile downloads (contextily's
            ``n_connections``; its own default is 1). Check the
            provider's usage policy before going above 16; some, such as
            OSM, allow only 2.
        profile (bool): Print a per-stage timing table (DEM fetch,
            curvature, hillshade, stretch, relief blend, tile fetch, tile
            reproject, final blend, plotting) at the end. Timings are
            always available in ``layers["timings"]``.

    Returns:
        tuple: ``(fig, ax, layers)`` where ``layers`` is a dict of:

            - ``basemap`` (H, W, 3): imagery warped onto the DEM grid.
            - ``dem_grey``, ``curvature``, ``hillshade`` (H, W): the
              stretched relief-group layers.
            - ``relief_luminosity`` (H, W): the full relief group.
            - ``texture_luminosity`` (H, W): hillshade + curvature only,
              without elevation (for burning data layers onto relief).
            - ``luminosity_composite``, ``final`` (H, W, 3): after the
              Luminosity blend, and after the final Soft Light.
            - ``extent``, ``transform``, ``crs``, ``shape``: the grid.
            - ``timings`` (dict[str, float]): seconds per stage.

    Raises:
        ValueError: If both or neither of ``dem_path`` / ``aoi_bounds``
            are given, an option is out of range, the DEM has no CRS, or
            ``target_crs`` is invalid.
        TerraTexture.io.DEMReadError: If the DEM can't be read.
        TerraTexture.sources.STACError: If the AOI query fails.
        BasemapFetchError: If the imagery can't be fetched.
        OSError: If ``tile_cache_dir`` or ``out_fig`` can't be written.
    """
    import contextily as ctx
    from rasterio.transform import array_bounds

    _validate_options(
        dem_path, aoi_bounds, dem_product, zoom, tile_connections, relief_strength
    )
    timer = _StageTimer()

    if source is None:
        source = ctx.providers.Esri.WorldImagery
    if tile_cache_dir is not None:
        _set_tile_cache(tile_cache_dir)

    with timer("dem_fetch"):
        dem, transform, crs = _load_dem(
            dem_path, aoi_bounds, aoi_bounds_crs, dem_product,
            arcticdem_resolution, target_crs,
        )
    height, width = dem.shape
    _check_grid(dem, transform, crs)
    west, south, east, north = array_bounds(height, width, transform)

    relief = _compute_relief(
        dem, transform.a, azimuth, altitude, curvature_std, hillshade_std, timer
    )
    basemap_rgb = _fetch_basemap(
        (west, south, east, north), crs, transform, (height, width),
        source, zoom, tile_connections, timer,
    )
    with timer("final_blend"):
        luminosity_composite, final = _composite(
            basemap_rgb, relief["relief_luminosity"], relief_strength
        )

    extent = (west, east, south, north)
    with timer("plotting"):
        fig, ax = _plot(final, extent, source, figsize, out_fig, show)

    if profile:
        timer.report()

    layers: dict[str, Any] = {
        "basemap": basemap_rgb,
        **relief,
        "luminosity_composite": luminosity_composite,
        "final": final,
        "extent": extent,
        "transform": transform,
        "crs": crs,
        "shape": (height, width),
        "timings": dict(timer.stages),
    }
    return fig, ax, layers


def add_relief_basemap(
    axes: Axes | Sequence[Axes],
    dem_path: str | os.PathLike[str] | Sequence[str | os.PathLike[str]] | None = None,
    aoi_bounds: Sequence[float] | None = None,
    aoi_bounds_crs: str = "EPSG:4326",
    dem_product: str = "arcticdem",
    arcticdem_resolution: int = 32,
    source: Any = None,
    zoom: int | str = "auto",
    target_crs: str | None = None,
    azimuth: float = 315,
    altitude: float = 45,
    curvature_std: float = 4,
    hillshade_std: float = 4,
    relief_strength: float = 1.0,
    zorder: float = 0,
    tile_cache_dir: str | None = None,
    tile_connections: int = 16,
    profile: bool = False,
) -> dict[str, Any]:
    """
    Build the relief basemap once and draw it under existing Axes.

    For putting the relief behind just the map panels of a larger
    multi-axes figure (e.g. ``plt.subplot_mosaic``), without re-fetching
    the DEM and imagery per panel and without a standalone figure.

    Call this *before* your own plotting on those axes: the image is drawn
    at ``zorder`` (default 0, the bottom), so later artists (zorder >= 1)
    sit on top. Opaque data layers hide the relief, so give them some
    transparency, e.g.::

        for coll in ax.collections:
            coll.set_alpha(0.75)

    Args:
        axes (Axes | Sequence[Axes]): Axes to draw the relief on.
        dem_path (str | PathLike | Sequence | None): See
            :func:`plot_dem_basemap_luminosity_relief`.
        aoi_bounds (Sequence[float] | None): See
            :func:`plot_dem_basemap_luminosity_relief`. Give ``dem_path``
            **or** ``aoi_bounds``.
        aoi_bounds_crs (str): See :func:`plot_dem_basemap_luminosity_relief`.
        dem_product (str): See :func:`plot_dem_basemap_luminosity_relief`.
        arcticdem_resolution (int): See
            :func:`plot_dem_basemap_luminosity_relief`.
        source (Any): See :func:`plot_dem_basemap_luminosity_relief`.
        zoom (int | str): See :func:`plot_dem_basemap_luminosity_relief`.
        target_crs (str | None): See
            :func:`plot_dem_basemap_luminosity_relief`.
        azimuth (float): See :func:`plot_dem_basemap_luminosity_relief`.
        altitude (float): See :func:`plot_dem_basemap_luminosity_relief`.
        curvature_std (float): See
            :func:`plot_dem_basemap_luminosity_relief`.
        hillshade_std (float): See
            :func:`plot_dem_basemap_luminosity_relief`.
        relief_strength (float): See
            :func:`plot_dem_basemap_luminosity_relief`.
        zorder (float): Drawing order of the relief image (0 = bottom).
        tile_cache_dir (str | None): See
            :func:`plot_dem_basemap_luminosity_relief`.
        tile_connections (int): See
            :func:`plot_dem_basemap_luminosity_relief`.
        profile (bool): See :func:`plot_dem_basemap_luminosity_relief`.

    Returns:
        dict[str, Any]: The same ``layers`` dict as
            :func:`plot_dem_basemap_luminosity_relief` (including
            ``extent``), e.g. for aligning a data overlay's limits.

    Raises:
        TypeError: If ``axes`` contains anything that isn't an ``Axes``.
        ValueError: If ``axes`` is empty, plus everything
            :func:`plot_dem_basemap_luminosity_relief` raises.

    Examples:
        >>> fig, (ax1, ax2) = plt.subplots(1, 2)  # doctest: +SKIP
        >>> layers = add_relief_basemap([ax1, ax2], aoi_bounds=bounds)  # doctest: +SKIP
    """
    axes_list = [axes] if isinstance(axes, Axes) else list(axes)
    if not axes_list:
        raise ValueError("axes is empty; pass at least one Axes")
    bad = [type(ax).__name__ for ax in axes_list if not isinstance(ax, Axes)]
    if bad:
        raise TypeError(f"axes must be matplotlib Axes; got {bad}")

    fig, _ax, layers = plot_dem_basemap_luminosity_relief(
        dem_path=dem_path,
        aoi_bounds=aoi_bounds,
        aoi_bounds_crs=aoi_bounds_crs,
        dem_product=dem_product,
        arcticdem_resolution=arcticdem_resolution,
        source=source,
        zoom=zoom,
        target_crs=target_crs,
        azimuth=azimuth,
        altitude=altitude,
        curvature_std=curvature_std,
        hillshade_std=hillshade_std,
        relief_strength=relief_strength,
        out_fig=None,
        show=False,
        tile_cache_dir=tile_cache_dir,
        tile_connections=tile_connections,
        profile=profile,
    )
    plt.close(fig)  # throwaway standalone figure; only the arrays matter

    for ax in axes_list:
        ax.imshow(layers["final"], extent=layers["extent"], zorder=zorder)
    logger.debug("Drew relief basemap on %d axes", len(axes_list))
    return layers
