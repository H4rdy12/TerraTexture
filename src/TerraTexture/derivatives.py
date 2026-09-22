"""
Terrain derivatives: profile / planform curvature and hillshade.

The analysis layer of the relief pipeline::

    DEM -> derivatives (this module) -> stretch -> blend -> plot / export

Public API:

- :func:`curvatures` -- profile and planform curvature.
- :func:`hillshade` -- analytical hillshading from a single light source.

Curvature:
    **Profile curvature** is the rate of change of slope along the
    direction of steepest descent; it controls whether flow accelerates
    or decelerates. **Planform curvature** is the curvature of contour
    lines, perpendicular to the slope; it controls whether flow converges
    or diverges.

    Formulas follow Zevenbergen & Thorne (1987), the basis of the 3x3
    curvature tools in ArcGIS, QGIS and GRASS::

        p = dz/dx,  q = dz/dy
        r = d2z/dx2,  t = d2z/dy2,  s = d2z/dxdy

        profile  = -(r p^2 + 2 s p q + t q^2) / ((p^2 + q^2) (1 + p^2 + q^2)^1.5)
        planform = -(r q^2 - 2 s p q + t p^2) / (p^2 + q^2)^1.5

    Derivatives come from applying ``np.gradient`` twice (central
    differences), not from the closed-form ZT 3x3 stencil; see
    ``docs/formulas.md`` for the derivation and how the two compare.

    Sign convention:

    ============  ==========================  ============================
    Curvature     Positive (convex)           Negative (concave)
    ============  ==========================  ============================
    Profile       flow decelerates            flow accelerates
    Planform      flow diverges (ridges)      flow converges (valleys)
    ============  ==========================  ============================

    Flat cells (``p^2 + q^2 < 1e-9``) have undefined curvature and are
    set to 0.

Units:
    ``cellsize`` must be in the same horizontal units as the elevations,
    normally metres. A DEM in geographic coordinates (EPSG:4326) has a
    cellsize in *degrees*, which makes every slope look near-vertical and
    the results meaningless. Reproject it to a metric CRS first; a
    cellsize below 0.001 logs a warning because it usually means degrees.
    Curvature is then in 1/metres.

Nodata:
    NaN cells (voids) are nearest-filled for the calculation, so they
    don't poison their neighbours' derivatives, then set back to NaN in
    the output. An all-NaN DEM gives all-NaN output.

Optional Rust acceleration:
    Same contract as :mod:`TerraTexture.blend` and
    :mod:`TerraTexture.stretch`. The DEM is converted to ``float32``
    first, so *every* 2-D DEM (whatever its original dtype) uses the
    compiled ``terra_texture_rs`` kernels when the extension is
    importable, and numpy otherwise. Both paths return ``float32`` and
    agree within float tolerance.

    If a kernel raises at runtime, the call logs a warning and returns the
    numpy result, so a broken extension costs speed, never results.

Diagnostics:
    ``_rust is not None`` shows whether the extension loaded; if it is
    ``None``, ``_RUST_IMPORT_ERROR`` holds the reason. "Not installed" is
    logged at DEBUG level. "Installed but failed to import" (an ABI or
    Python-version mismatch, a stale build, a missing symbol) is logged
    as a WARNING, and a build missing a kernel is reported at import.

Dependencies:
    numpy and scipy only (via :func:`TerraTexture.io._fill_nan_nearest`);
    no rasterio.

Examples:
    Curvature and hillshade for a 2 m DEM::

        dem, cellsize = load_dem("tile_dem.tif")
        profile, planform = curvatures(dem, cellsize)
        shade = hillshade(dem, cellsize, azimuth=315, altitude=45)
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from .io import _fill_nan_nearest

if TYPE_CHECKING:
    import numpy.typing as npt

    FloatArray = npt.NDArray[np.float32]


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional Rust extension
# ---------------------------------------------------------------------------

# Name of the compiled extension module (see rust/pyproject.toml).
_RUST_MODULE = "terra_texture_rs"

# Kernels this module can use.
_RUST_KERNELS = ("curvatures", "hillshade")

# Why the Rust extension is unavailable, or None if it loaded.
_RUST_IMPORT_ERROR: ImportError | None = None

try:
    import terra_texture_rs as _rust
except ImportError as _exc:
    _rust = None
    _RUST_IMPORT_ERROR = _exc
    if isinstance(_exc, ModuleNotFoundError) and _exc.name == _RUST_MODULE:
        logger.debug("%s not installed; using numpy.", _RUST_MODULE)
    else:
        # Installed but broken: worth surfacing, since it silently
        # disables acceleration.
        logger.warning(
            "%s is installed but failed to import (%s: %s); falling back "
            "to numpy. Rebuild it, e.g. `uv sync --extra rust "
            "--reinstall-package terra-texture-rs`.",
            _RUST_MODULE, type(_exc).__name__, _exc,
        )
    del _exc

# An extension built from an older crate may lack some kernels.
if _rust is not None:
    for _name in _RUST_KERNELS:
        if not hasattr(_rust, _name):
            logger.warning(
                "%s has no %s kernel (stale build?); %s() will use numpy.",
                _RUST_MODULE, _name, _name,
            )
    del _name


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Squared-gradient threshold below which a cell counts as flat.
_FLAT_THRESHOLD = 1e-9

# Smallest DEM the 3x3 method makes sense for (rows and columns).
_MIN_DEM_SIZE = 3

# Cellsizes below this are almost certainly degrees, not metres.
_DEGREES_SUSPECT_CELLSIZE = 1e-3


# ---------------------------------------------------------------------------
# Validation and dispatch helpers
# ---------------------------------------------------------------------------

def _prepare_dem(dem: npt.ArrayLike) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
    """
    Validate a DEM, convert it to float32 and nearest-fill its voids.

    Args:
        dem (npt.ArrayLike): 2-D elevation array; NaN marks nodata.

    Returns:
        tuple[FloatArray, np.ndarray]: ``(filled, nan_mask)``: the
            float32 DEM with voids filled, and a boolean mask of the
            original voids.

    Raises:
        TypeError: If ``dem`` is not numeric.
        ValueError: If ``dem`` is not 2-D, or is smaller than 3 x 3.
    """
    try:
        array = np.asarray(dem)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"dem must be array-like; got {type(dem).__name__}") from exc
    if array.dtype.kind not in "biuf":
        raise TypeError(f"dem must be numeric; got dtype {array.dtype}")
    if array.ndim != 2:
        raise ValueError(f"dem must be 2-D; got shape {array.shape}")
    if min(array.shape) < _MIN_DEM_SIZE:
        raise ValueError(
            f"dem must be at least {_MIN_DEM_SIZE} x {_MIN_DEM_SIZE} for a 3x3 "
            f"method; got shape {array.shape}"
        )
    return _fill_nan_nearest(array.astype(np.float32, copy=False))


def _check_cellsize(cellsize: float) -> float:
    """
    Validate the cell size and warn if it looks like degrees.

    Args:
        cellsize (float): Pixel size in the elevations' horizontal units.

    Returns:
        float: ``cellsize`` as a Python float.

    Raises:
        ValueError: If ``cellsize`` is not a positive, finite number. A
            negative value gets a hint, since it usually comes from a
            north-up transform's ``e`` term.
    """
    try:
        value = float(cellsize)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"cellsize must be a number; got {cellsize!r}") from exc
    if not math.isfinite(value) or value == 0:
        raise ValueError(f"cellsize must be positive and finite; got {cellsize!r}")
    if value < 0:
        raise ValueError(
            f"cellsize must be positive; got {value}. If it came from a "
            "raster transform's y term, pass abs(transform.e) or transform.a."
        )
    if value < _DEGREES_SUSPECT_CELLSIZE:
        logger.warning(
            "cellsize %g looks like degrees (geographic CRS); derivatives "
            "need metres. Reproject the DEM to a metric CRS first.", value,
        )
    return value


def _run_kernel(name: str, dem: FloatArray, *args: float) -> Any:
    """
    Call a Rust kernel if available, returning ``None`` to mean "use numpy".

    Args:
        name (str): Kernel function name on ``_rust``.
        dem (FloatArray): float32 2-D DEM (made C-contiguous here).
        *args (float): Extra scalar arguments for the kernel.

    Returns:
        Any: The kernel's result, or ``None`` if the extension or kernel
            is unavailable or the kernel raised (a warning is logged).

    Note:
        A Rust *panic* surfaces as pyo3's ``PanicException``, which
        derives from ``BaseException``. It is deliberately not caught,
        since that would also swallow ``KeyboardInterrupt``.
    """
    if _rust is None or not hasattr(_rust, name):
        return None
    try:
        return getattr(_rust, name)(np.ascontiguousarray(dem), *args)
    except Exception as exc:  # any kernel failure -> numpy fallback
        logger.warning(
            "Rust %s failed on %s DEM (%s: %s); falling back to numpy.",
            name, dem.shape, type(exc).__name__, exc,
        )
        return None


# ---------------------------------------------------------------------------
# numpy implementations
# ---------------------------------------------------------------------------

def _derivatives(
    dem: FloatArray,
    cellsize: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """
    First and second partial derivatives by repeated central differences.

    Args:
        dem (FloatArray): Void-free 2-D DEM.
        cellsize (float): Pixel size, in the elevations' units.

    Returns:
        tuple: ``(p, q, r, t, s)`` = ``(dz/dx, dz/dy, d2z/dx2, d2z/dy2,
            d2z/dxdy)``, each shaped like ``dem``.
    """
    zy, zx = np.gradient(dem, cellsize)    # first partials (q, p)
    zxy, zxx = np.gradient(zx, cellsize)   # d(zx)/dy, d(zx)/dx
    zyy, _ = np.gradient(zy, cellsize)     # d(zy)/dy
    return zx, zy, zxx, zyy, zxy


def _curvatures_numpy(
    dem: FloatArray,
    cellsize: float,
) -> tuple[FloatArray, FloatArray]:
    """
    Profile and planform curvature in pure numpy.

    Args:
        dem (FloatArray): Void-free float32 2-D DEM.
        cellsize (float): Pixel size, in the elevations' units.

    Returns:
        tuple[FloatArray, FloatArray]: ``(profile, planform)``; 0 on flat
            cells and wherever the formula is undefined.
    """
    p, q, r, t, s = _derivatives(dem, cellsize)
    p2q2 = p ** 2 + q ** 2

    # Flat cells divide by ~0; those results are replaced just below.
    with np.errstate(divide="ignore", invalid="ignore"):
        profile = -(r * p ** 2 + 2 * s * p * q + t * q ** 2) / (
            p2q2 * (1 + p2q2) ** 1.5
        )
        planform = -(r * q ** 2 - 2 * s * p * q + t * p ** 2) / (p2q2 ** 1.5)

    flat = p2q2 < _FLAT_THRESHOLD
    profile = np.where(
        flat, 0.0, np.nan_to_num(profile, nan=0.0, posinf=0.0, neginf=0.0)
    )
    planform = np.where(
        flat, 0.0, np.nan_to_num(planform, nan=0.0, posinf=0.0, neginf=0.0)
    )
    return profile, planform


def _hillshade_numpy(
    dem: FloatArray,
    cellsize: float,
    azimuth: float,
    altitude: float,
) -> FloatArray:
    """
    Hillshade in pure numpy.

    Args:
        dem (FloatArray): Void-free float32 2-D DEM.
        cellsize (float): Pixel size, in the elevations' units.
        azimuth (float): Light direction, degrees clockwise from north.
        altitude (float): Light elevation above the horizon, degrees.

    Returns:
        FloatArray: Illumination in ``[0, 1]``.
    """
    # np.radians() on a Python scalar returns a *strong* float64 numpy
    # scalar; left uncast it would upcast every float32 array it touches.
    az = np.float32(np.radians(360.0 - azimuth + 90))
    alt = np.float32(np.radians(altitude))
    zy, zx = np.gradient(dem, cellsize)
    slope = np.pi / 2 - np.arctan(np.hypot(zx, zy))
    # ArcGIS convention: aspect = atan2(dz/dy, -dz/dx), with dz/dy taken
    # down the rows (north -> south) as np.gradient does. The arguments
    # were previously swapped, which mirrored the light across the NE-SW
    # axis: the default NW light shaded terrain as if lit from the SE.
    aspect = np.arctan2(zy, -zx)
    shaded = (
        np.sin(alt) * np.sin(slope)
        + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    )
    return np.clip(shaded, 0, 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def curvatures(
    dem: npt.ArrayLike,
    cellsize: float,
) -> tuple[FloatArray, FloatArray]:
    """
    Compute profile and planform curvature of a DEM.

    See the module docstring for the Zevenbergen & Thorne formulas and
    the sign convention (positive = convex, negative = concave).

    Dispatch::

        Rust extension with a curvatures kernel -> Rust
        otherwise, or the kernel raised         -> numpy

    Args:
        dem (npt.ArrayLike): 2-D elevation array, at least 3 x 3. NaN
            marks nodata. Converted to ``float32``.
        cellsize (float): Pixel size in the same horizontal units as the
            elevations (normally metres). Must be positive.

    Returns:
        tuple[FloatArray, FloatArray]: ``(profile, planform)`` float32
            arrays shaped like ``dem``, in 1/(elevation units). NaN where
            ``dem`` is NaN; 0 on flat cells.

    Raises:
        TypeError: If ``dem`` is not numeric.
        ValueError: If ``dem`` is not 2-D or is smaller than 3 x 3, or
            ``cellsize`` is not positive and finite.

    Examples:
        >>> y, x = np.mgrid[-10:11, -10:11].astype(float)
        >>> profile, planform = curvatures(-(x**2 + y**2), 1.0)  # a dome
        >>> bool(planform[10, 15] > 0)  # convex -> positive
        True
    """
    cellsize = _check_cellsize(cellsize)
    dem_filled, nan_mask = _prepare_dem(dem)

    result = _run_kernel("curvatures", dem_filled, cellsize)
    if result is not None:
        profile, planform = result
    else:
        profile, planform = _curvatures_numpy(dem_filled, cellsize)

    return (
        np.where(nan_mask, np.nan, profile),
        np.where(nan_mask, np.nan, planform),
    )


def hillshade(
    dem: npt.ArrayLike,
    cellsize: float,
    azimuth: float = 315,
    altitude: float = 45,
) -> FloatArray:
    """
    Compute an analytical hillshade of a DEM, in ``[0, 1]``.

    Illumination is the cosine of the angle between each cell's surface
    normal and a light source at ``azimuth`` / ``altitude``; cells facing
    away from the light are clipped to 0. The defaults (315 deg, 45 deg,
    light from the north-west) are the cartographic standard.

    Dispatch::

        Rust extension with a hillshade kernel -> Rust
        otherwise, or the kernel raised        -> numpy

    Args:
        dem (npt.ArrayLike): 2-D elevation array, at least 3 x 3, north-up
            (row 0 = north, as rasterio reads a standard GeoTIFF). NaN
            marks nodata. Converted to ``float32``.
        cellsize (float): Pixel size in the same horizontal units as the
            elevations (normally metres). Must be positive.
        azimuth (float): Direction the light comes *from*, in degrees
            clockwise from north (0 = N, 90 = E). Any finite value; it is
            used modulo 360.
        altitude (float): Light elevation above the horizon, in degrees,
            from 0 (horizon) to 90 (overhead).

    Returns:
        FloatArray: float32 illumination shaped like ``dem``, in
            ``[0, 1]``. NaN where ``dem`` is NaN.

    Warning:
        Before this version the numpy path computed aspect with swapped
        ``atan2`` arguments, mirroring the light across the NE-SW axis
        (the default NW light shaded as if from the SE). The Rust
        ``hillshade`` kernel must use the same corrected formula,
        ``atan2(dz/dy, -dz/dx)``; ``tests/test_derivatives_rust.py``
        checks it against an independent ground truth.

    Raises:
        TypeError: If ``dem`` is not numeric.
        ValueError: If ``dem`` is not 2-D or is smaller than 3 x 3,
            ``cellsize`` is not positive and finite, ``azimuth`` is not
            finite, or ``altitude`` is outside ``[0, 90]``.

    Examples:
        >>> flat = np.zeros((5, 5))
        >>> float(hillshade(flat, 1.0, altitude=90)[2, 2])  # sun overhead
        1.0
    """
    cellsize = _check_cellsize(cellsize)
    try:
        azimuth, altitude = float(azimuth), float(altitude)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"azimuth and altitude must be numbers; got {azimuth!r}, {altitude!r}"
        ) from exc
    if not math.isfinite(azimuth):
        raise ValueError(f"azimuth must be finite; got {azimuth}")
    if not (math.isfinite(altitude) and 0 <= altitude <= 90):
        raise ValueError(f"altitude must be between 0 and 90 degrees; got {altitude}")

    dem_filled, nan_mask = _prepare_dem(dem)

    shaded = _run_kernel("hillshade", dem_filled, cellsize, azimuth, altitude)
    if shaded is None:
        shaded = _hillshade_numpy(dem_filled, cellsize, azimuth, altitude)
    return np.where(nan_mask, np.nan, shaded)
