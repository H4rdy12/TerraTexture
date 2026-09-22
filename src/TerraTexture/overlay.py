"""
Burn a colour-mapped data raster onto relief with the Luminosity blend.

Combines a scientific raster (elevation change dh/dt, ice velocity,
...) with the relief from :mod:`TerraTexture.basemap` so both signals
share every pixel. The data supplies hue and saturation (its colour ramp
is preserved exactly); the relief supplies only lightness, so the
terrain texture is embossed *into* the data's colours.

This is the same Luminosity technique basemap.py uses to drape imagery
over terrain. It differs from alpha-blending a translucent overlay,
which only lets the relief show through in the gaps rather than
combining the two signals.

Public API:

- :func:`burn_data_onto_relief` -- reproject, colour-map and burn a data
  raster onto a relief ``layers`` dict.

Pipeline::

    data (any grid/CRS) --reproject--> relief grid --cmap--> RGB
    relief layers --choose + compress lightness--> luminosity
                                          v
                     luminosity_blend(data RGB, luminosity)
                                          v
             NaN data -> relief imagery ('relief') or alpha 0 ('transparent')

Error handling:
    All options are validated before any reprojection: missing
    ``relief_layers`` keys (listed by name), non-2-D data, missing
    georeferencing, unknown resampling methods, bad luminosity ranges
    and bad colour limits raise ``ValueError`` with the fix in the
    message. An invalid CRS raises ``ValueError``; a failed reprojection
    raises ``RuntimeError`` with the rasterio error attached.

Logging:
    Logged under ``TerraTexture.overlay``. Colour limits inferred from
    the data are reported at INFO; data that doesn't overlap the relief
    at all is a WARNING; the fraction of relief cells covered is DEBUG.

Dependencies:
    rasterio (for reprojection) and matplotlib. xarray / rioxarray are
    optional: with them, ``data`` can be a ``DataArray`` whose CRS and
    transform are picked up automatically.

Examples:
    Burn a dh/dt grid onto relief, with a colour bar::

        layers = add_relief_basemap(ax, aoi_bounds=bounds)
        composite, sm = burn_data_onto_relief(
            layers, dhdt,  # rioxarray DataArray with CRS/transform set
            cmap="RdYlBu_r", vmin=-6, vmax=6,
        )
        ax.imshow(composite, extent=layers["extent"])
        plt.colorbar(sm, ax=ax, label="dh/dt (m/yr)")
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from .blend import luminosity_blend

if TYPE_CHECKING:
    import numpy.typing as npt
    from affine import Affine
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Colormap, Normalize
    from rasterio.enums import Resampling


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# relief_layers keys every call needs.
_REQUIRED_KEYS = ("final", "transform", "crs", "shape")

# luminosity_source names and the relief_layers key each one reads.
_LUMINOSITY_KEYS = {
    "texture": "texture_luminosity",
    "full": "relief_luminosity",
}

# Accepted nan_fill modes.
_NAN_FILL_MODES = ("relief", "transparent")


# ---------------------------------------------------------------------------
# Input resolution and validation
# ---------------------------------------------------------------------------

def _check_relief_layers(relief_layers: Any) -> tuple[int, int]:
    """
    Check that ``relief_layers`` has what every call needs.

    Args:
        relief_layers (Any): Candidate ``layers`` dict.

    Returns:
        tuple[int, int]: The relief grid's ``(height, width)``.

    Raises:
        TypeError: If ``relief_layers`` is not a dict-like mapping.
        ValueError: If required keys are missing (all are listed), or
            ``final`` doesn't match ``shape``.
    """
    if not hasattr(relief_layers, "keys"):
        raise TypeError(
            "relief_layers must be the layers dict returned by "
            "plot_dem_basemap_luminosity_relief() or add_relief_basemap(); "
            f"got {type(relief_layers).__name__}"
        )
    missing = [key for key in _REQUIRED_KEYS if key not in relief_layers]
    if missing:
        raise ValueError(
            f"relief_layers is missing {missing}; pass the layers dict "
            "returned by plot_dem_basemap_luminosity_relief() or "
            "add_relief_basemap()"
        )
    height, width = (int(n) for n in relief_layers["shape"])
    final_shape = np.shape(relief_layers["final"])
    if final_shape[:2] != (height, width):
        raise ValueError(
            f"relief_layers['final'] has shape {final_shape}, which doesn't "
            f"match relief_layers['shape'] {(height, width)}"
        )
    return height, width


def _resolve_data(
    data: Any,
    data_transform: Affine | None,
    data_crs: Any,
) -> tuple[np.ndarray, Affine, Any]:
    """
    Turn ``data`` into a float32 2-D array with its transform and CRS.

    ``xarray.DataArray`` inputs contribute ``.values``; with rioxarray,
    their CRS and transform are used when not given explicitly. A single
    leading band axis (``(1, H, W)``, as ``rioxarray.open_rasterio``
    returns) is squeezed away.

    Args:
        data (Any): numpy array, nested list, or ``xarray.DataArray``.
        data_transform (Affine | None): Explicit transform, or ``None``.
        data_crs (Any): Explicit CRS, or ``None``.

    Returns:
        tuple: ``(array, transform, crs)``.

    Raises:
        TypeError: If ``data`` is not numeric.
        ValueError: If ``data`` is not 2-D (after squeezing one band), or
            the transform / CRS is missing.
    """
    if hasattr(data, "values") and hasattr(data, "dims"):  # xarray.DataArray
        rio = getattr(data, "rio", None)
        if rio is not None:
            # rioxarray raises its own exception types when spatial
            # metadata is absent; treat any failure as "not available".
            if data_transform is None:
                try:
                    data_transform = rio.transform()
                except Exception as exc:  # noqa: BLE001 - optional metadata
                    logger.debug("No transform from data.rio: %s", exc)
            if data_crs is None:
                data_crs = rio.crs
        values = data.values
    else:
        values = data

    array = np.asarray(values)
    if array.dtype.kind not in "biuf":
        raise TypeError(f"data must be numeric; got dtype {array.dtype}")
    array = array.astype(np.float32, copy=False)
    if array.ndim == 3 and array.shape[0] == 1:
        logger.debug("Squeezing single-band data %s to 2-D", array.shape)
        array = array[0]
    if array.ndim != 2:
        raise ValueError(
            f"data must be 2-D (or a single band shaped (1, H, W)); got "
            f"shape {array.shape}"
        )
    if data_transform is None or data_crs is None:
        missing = [
            name for name, value in
            (("data_transform", data_transform), ("data_crs", data_crs))
            if value is None
        ]
        raise ValueError(
            f"{' and '.join(missing)} required: pass explicitly, or use an "
            "xarray.DataArray with rioxarray CRS/transform set "
            "(da.rio.write_crs(...))"
        )
    return array, data_transform, data_crs


def _resolve_resampling(resampling: str | Resampling) -> Resampling:
    """
    Convert a resampling name (or enum member) to ``Resampling``.

    Args:
        resampling (str | Resampling): e.g. ``"bilinear"``, ``"nearest"``.

    Returns:
        Resampling: The rasterio enum member.

    Raises:
        ValueError: If the name is unknown (valid names are listed).
    """
    from rasterio.enums import Resampling

    if isinstance(resampling, Resampling):
        return resampling
    try:
        return Resampling[str(resampling)]
    except KeyError:
        valid = sorted(member.name for member in Resampling)
        raise ValueError(
            f"Unknown resampling {resampling!r}; expected one of {valid}"
        ) from None


def _resolve_luminosity(
    relief_layers: Any,
    luminosity_source: str | npt.ArrayLike,
    luminosity_range: Sequence[float] | None,
    shape: tuple[int, int],
) -> np.ndarray:
    """
    Pick the lightness signal and compress it into ``luminosity_range``.

    Args:
        relief_layers (Any): The relief ``layers`` dict.
        luminosity_source (str | npt.ArrayLike): ``"texture"``, ``"full"``
            or an ``(H, W)`` array.
        luminosity_range (Sequence[float] | None): ``(lo, hi)`` band, or
            ``None`` for the raw values.
        shape (tuple[int, int]): Relief ``(height, width)``.

    Returns:
        np.ndarray: ``(H, W)`` lightness in ``[lo, hi]``.

    Raises:
        ValueError: If the source name is unknown, its layer is missing
            from ``relief_layers``, an array has the wrong shape, or the
            range is invalid.
    """
    if isinstance(luminosity_source, str):
        if luminosity_source not in _LUMINOSITY_KEYS:
            raise ValueError(
                f"luminosity_source must be one of {sorted(_LUMINOSITY_KEYS)} "
                f"or an (H, W) array; got {luminosity_source!r}"
            )
        key = _LUMINOSITY_KEYS[luminosity_source]
        if key not in relief_layers:
            raise ValueError(
                f"luminosity_source={luminosity_source!r} needs "
                f"relief_layers[{key!r}], which is missing (layers from an "
                "older TerraTexture?). Rebuild the relief or use the other "
                "source."
            )
        luminosity = np.asarray(relief_layers[key])
    else:
        luminosity = np.asarray(luminosity_source, dtype=np.float32)

    if luminosity.shape != shape:
        raise ValueError(
            f"luminosity has shape {luminosity.shape}; it must match the "
            f"relief grid {shape}"
        )

    if luminosity_range is None:
        return luminosity
    try:
        lo, hi = (float(v) for v in luminosity_range)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"luminosity_range must be two numbers (lo, hi) or None; got "
            f"{luminosity_range!r}"
        ) from exc
    if not (0 <= lo < hi <= 1):
        raise ValueError(
            f"luminosity_range needs 0 <= lo < hi <= 1; got ({lo}, {hi})"
        )
    return lo + luminosity * (hi - lo)


def _resolve_norm(
    norm: Normalize | None,
    vmin: float | None,
    vmax: float | None,
    data_on_grid: np.ndarray,
) -> Normalize:
    """
    Build the colour normalisation, inferring missing limits from data.

    matplotlib's own autoscaling uses plain min/max, so a single NaN makes
    both limits NaN and every pixel the "bad" colour. Missing limits are
    instead taken from the finite data (and logged).

    Args:
        norm (Normalize | None): Explicit norm; returned unchanged.
        vmin (float | None): Lower colour limit.
        vmax (float | None): Upper colour limit.
        data_on_grid (np.ndarray): Reprojected data (NaN = nodata).

    Returns:
        Normalize: The norm to apply.

    Raises:
        ValueError: If ``vmin >= vmax``.
    """
    from matplotlib.colors import Normalize

    if norm is not None:
        return norm
    if vmin is None or vmax is None:
        finite = data_on_grid[np.isfinite(data_on_grid)]
        if finite.size:
            vmin = float(finite.min()) if vmin is None else vmin
            vmax = float(finite.max()) if vmax is None else vmax
            logger.info(
                "Colour limits inferred from the data: vmin=%g, vmax=%g "
                "(pass vmin/vmax to fix them)", vmin, vmax,
            )
    if vmin is not None and vmax is not None:
        if not (math.isfinite(vmin) and math.isfinite(vmax)) or vmin >= vmax:
            raise ValueError(
                f"Colour limits need finite vmin < vmax; got vmin={vmin}, "
                f"vmax={vmax}"
            )
    return Normalize(vmin=vmin, vmax=vmax)


def _reproject_to_relief(
    array: np.ndarray,
    transform: Affine,
    crs: Any,
    relief_layers: Any,
    shape: tuple[int, int],
    resampling: Resampling,
) -> np.ndarray:
    """
    Reproject the data onto the relief's exact grid (NaN = nodata).

    Args:
        array (np.ndarray): float32 2-D data.
        transform (Affine): Data transform.
        crs (Any): Data CRS.
        relief_layers (Any): Relief ``layers`` dict (target grid).
        shape (tuple[int, int]): Relief ``(height, width)``.
        resampling (Resampling): Resampling method.

    Returns:
        np.ndarray: ``(H, W)`` float32 data on the relief grid.

    Raises:
        ValueError: If either CRS is invalid.
        RuntimeError: If rasterio fails to reproject.
    """
    from rasterio.errors import CRSError, RasterioError
    from rasterio.warp import reproject

    on_grid = np.full(shape, np.nan, dtype=np.float32)
    try:
        reproject(
            source=array,
            destination=on_grid,
            src_transform=transform,
            src_crs=crs,
            dst_transform=relief_layers["transform"],
            dst_crs=relief_layers["crs"],
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=resampling,
        )
    except CRSError as exc:
        raise ValueError(
            f"Invalid CRS (data_crs={crs!r}, relief crs="
            f"{relief_layers['crs']!r}): {exc}"
        ) from exc
    except RasterioError as exc:
        raise RuntimeError(f"Reprojecting data onto the relief failed: {exc}") from exc

    coverage = float(np.isfinite(on_grid).mean())
    logger.debug(
        "Data %s reprojected onto relief %s (%s); %.1f%% of cells have data",
        array.shape, shape, resampling.name, 100 * coverage,
    )
    if coverage == 0:
        logger.warning(
            "No data overlaps the relief grid after reprojection; check "
            "data_transform/data_crs. The result is the plain relief."
        )
    return on_grid


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def burn_data_onto_relief(
    relief_layers: dict[str, Any],
    data: Any,
    data_transform: Affine | None = None,
    data_crs: Any = None,
    cmap: str | Colormap = "RdYlBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Normalize | None = None,
    nan_fill: str = "relief",
    resampling: str | Resampling = "bilinear",
    luminosity_range: Sequence[float] | None = (0.15, 0.9),
    luminosity_source: str | npt.ArrayLike = "texture",
) -> tuple[np.ndarray, ScalarMappable]:
    """
    Colour-map a data raster and burn it onto relief (Luminosity blend).

    Steps:

    1. Reproject ``data`` onto the relief's exact grid.
    2. Colour-map it with ``cmap`` / ``norm``.
    3. Replace the colours' lightness with the relief's
       (:func:`TerraTexture.blend.luminosity_blend`), keeping their hue
       and saturation.
    4. Fill NaN (no-data) cells with the plain relief, or make them
       transparent.

    Args:
        relief_layers (dict[str, Any]): The ``layers`` dict from
            :func:`~TerraTexture.basemap.plot_dem_basemap_luminosity_relief`
            or :func:`~TerraTexture.basemap.add_relief_basemap`. Needs
            ``final``, ``transform``, ``crs``, ``shape``, plus
            ``texture_luminosity`` or ``relief_luminosity`` for the
            chosen ``luminosity_source``.
        data (Any): 2-D data (numpy array or ``xarray.DataArray``; a
            ``(1, H, W)`` single band is accepted). NaN = nodata.
        data_transform (Affine | None): Affine transform of ``data``'s
            grid. Required for numpy input; read from ``data.rio`` for a
            rioxarray ``DataArray`` if not given.
        data_crs (Any): CRS of ``data``'s grid. Same rule as
            ``data_transform``.
        cmap (str | Colormap): Matplotlib colormap (default
            ``"RdYlBu_r"``, matching DEMTimeSeriesPlotter's
            ``CMAP_SLOPE``).
        vmin (float | None): Lower colour limit. If omitted (and no
            ``norm``), taken from the finite data's minimum.
        vmax (float | None): Upper colour limit. Same rule as ``vmin``.
        norm (Normalize | None): Explicit norm (e.g. a ``TwoSlopeNorm``
            centred on zero); overrides ``vmin`` / ``vmax``.
        nan_fill (str): What shows where ``data`` is NaN after
            reprojection. ``"relief"`` (default): the plain imagery-draped
            relief (``relief_layers["final"]``), e.g. for bedrock or ocean
            masked out of a dh/dt product. ``"transparent"``: return RGBA
            with alpha 0 there.
        resampling (str | Resampling): rasterio resampling method for the
            reprojection (default ``"bilinear"``; use ``"nearest"`` for
            categorical data).
        luminosity_range (Sequence[float] | None): Compress the lightness
            into ``[lo, hi]`` before burning (default ``(0.15, 0.9)``).
            Luminosity blending forces the exact lightness, so values near
            0 or 1 crush any hue to black or white. ``None`` uses the raw
            ``[0, 1]`` values. Requires ``0 <= lo < hi <= 1``.
        luminosity_source (str | npt.ArrayLike): The terrain lightness:

            - ``"texture"`` (default): ``texture_luminosity`` --
              hillshade + curvature *without* elevation. Centred and
              locally varying, so terrain texture shows everywhere without
              low ground crushing the data's colour.
            - ``"full"``: ``relief_luminosity`` -- hillshade + curvature
              + elevation, the imagery relief's own signal. More dramatic,
              with more colour crushing at the extremes.
            - an ``(H, W)`` array on the relief grid.

    Returns:
        tuple: ``(composite, mappable)``:

            - ``composite`` (np.ndarray): ``(H, W, 3)`` RGB (or
              ``(H, W, 4)`` RGBA for ``nan_fill="transparent"``) float32
              in ``[0, 1]`` on the relief grid. Show with
              ``ax.imshow(composite, extent=relief_layers["extent"])``.
            - ``mappable`` (ScalarMappable): for a colour bar,
              ``plt.colorbar(mappable, ax=ax, label=...)``.

    Raises:
        TypeError: If ``relief_layers`` isn't a dict or ``data`` isn't
            numeric.
        ValueError: If ``relief_layers`` lacks required keys, ``data``
            isn't 2-D or lacks georeferencing, any option is invalid, the
            colour limits aren't ``vmin < vmax``, or a CRS is invalid.
        RuntimeError: If the reprojection fails.

    Examples:
        >>> layers = add_relief_basemap(ax, aoi_bounds=bounds)  # doctest: +SKIP
        >>> composite, sm = burn_data_onto_relief(  # doctest: +SKIP
        ...     layers, dhdt, cmap="RdYlBu_r", vmin=-6, vmax=6,
        ... )
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable

    # Validate everything cheap before the (potentially slow) reprojection.
    shape = _check_relief_layers(relief_layers)
    if nan_fill not in _NAN_FILL_MODES:
        raise ValueError(
            f"nan_fill must be one of {list(_NAN_FILL_MODES)}; got {nan_fill!r}"
        )
    resampling_enum = _resolve_resampling(resampling)
    luminosity = _resolve_luminosity(
        relief_layers, luminosity_source, luminosity_range, shape
    )
    array, transform, crs = _resolve_data(data, data_transform, data_crs)

    data_on_grid = _reproject_to_relief(
        array, transform, crs, relief_layers, shape, resampling_enum
    )

    # cmap(...) always returns float64 RGBA; cast down so it doesn't
    # upcast the whole composite (often the largest array here).
    norm = _resolve_norm(norm, vmin, vmax, data_on_grid)
    cmap_obj = plt.get_cmap(cmap)
    mappable = ScalarMappable(norm=norm, cmap=cmap_obj)
    data_rgba = cmap_obj(norm(data_on_grid)).astype(np.float32)
    nan_mask = np.isnan(data_on_grid)

    # Data supplies hue/saturation, relief supplies lightness.
    burned_rgb = luminosity_blend(data_rgba[:, :, :3], luminosity)

    if nan_fill == "relief":
        return (
            np.where(nan_mask[:, :, None], relief_layers["final"], burned_rgb),
            mappable,
        )
    alpha = np.where(nan_mask, np.float32(0.0), np.float32(1.0))
    return np.dstack([burned_rgb, alpha]), mappable
