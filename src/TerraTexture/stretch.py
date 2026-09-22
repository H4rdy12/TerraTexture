"""
Array stretching, normalisation and NaN-safe resampling.

Helpers that turn raw DEM derivatives (curvature, hillshade, slope, ...)
into display-ready ``[0, 1]`` arrays, and resample arrays onto a new grid
without nodata voids bleeding or vanishing. Used by the blending and
plotting layers.

Public API:

- :func:`normalize` -- percentile stretch (robust to outliers).
- :func:`stretch_std` -- mean +/- N standard deviations stretch, the
  "Standard Deviation" symbology ArcGIS Pro uses for curvature and
  hillshade.
- :func:`bilinear_resample` -- bilinear resampling that preserves NaN
  voids.

NaN convention:
    NaN means nodata everywhere in this package. Every function ignores
    NaNs when computing statistics and passes them through unchanged.
    ``+/-inf`` is *not* treated as nodata: it will dominate the stats,
    so convert it to NaN first if your data can contain it. An all-NaN
    input returns an all-NaN result with a logged warning rather than a
    numpy ``RuntimeWarning``.

Optional Rust acceleration:
    Same contract as :mod:`TerraTexture.blend` and
    :mod:`TerraTexture.derivatives`. If the compiled ``terra_texture_rs``
    extension is importable, :func:`stretch_std` dispatches plain 2-D
    ``float32`` arrays to its fused kernel: one reduction pass computing
    sum, sum of squares and count together, then one fused elementwise
    pass for the clip-stretch. numpy's ``nanmean`` + ``nanstd`` would
    instead make ``nanstd`` recompute the mean internally.

    Everything else uses the pure-numpy path, which avoids the same
    redundant pass by computing the variance from the already-known
    mean. Both paths agree within float tolerance (verified in
    ``tests/test_stretch_rust.py``).

    If the Rust kernel raises for any reason, the call logs a warning and
    falls back to numpy, so a broken extension degrades performance, never
    results. :func:`normalize` is not accelerated: percentiles need
    sorting/selection rather than a single reduction, and numpy's
    selection routine is already optimised C.

Diagnostics:
    Check ``_rust is not None`` to see whether the extension loaded. If it
    is ``None``, ``_RUST_IMPORT_ERROR`` holds the reason. "Not installed"
    is logged at DEBUG level; "installed but failed to import" (an ABI or
    Python-version mismatch, a stale build, a missing symbol) is logged
    as a WARNING, because it silently costs speed.

Dependencies:
    numpy and scipy only; no rasterio.

Examples:
    Stretch a curvature grid for display::

        curv_display = stretch_std(curvature, n_std=2.5)

    Clip the extreme 2 % of a hillshade::

        hs_display = normalize(hillshade, low=2, high=98)

    Resample a DEM to a basemap's pixel grid::

        dem_on_basemap = bilinear_resample(dem, basemap.shape[:2])
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.ndimage import zoom

from .io import _fill_nan_nearest

if TYPE_CHECKING:
    import numpy.typing as npt


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional Rust extension
# ---------------------------------------------------------------------------

# Name of the compiled extension module (see rust/pyproject.toml).
_RUST_MODULE = "terra_texture_rs"

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

# An extension built from an older crate may lack this kernel.
if _rust is not None and not hasattr(_rust, "stretch_std"):
    logger.warning(
        "%s has no stretch_std kernel (stale build?); stretch_std() will "
        "use numpy.", _RUST_MODULE,
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Added to stretch denominators so constant inputs map to 0, not 0/0.
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _as_float_array(arr: npt.ArrayLike, name: str = "arr") -> np.ndarray:
    """
    Convert input to a non-empty floating-point numpy array.

    Floating dtypes are kept as-is (so ``float32`` stays ``float32``).
    Integer and boolean inputs are converted to ``float32``, so that
    percentile values aren't truncated to integers and NaN can be
    represented in the output.

    Args:
        arr (npt.ArrayLike): Input data.
        name (str): Argument name, used in error messages.

    Returns:
        np.ndarray: A floating-point array (no copy if already floating).

    Raises:
        TypeError: If ``arr`` is not numeric (e.g. strings or objects).
        ValueError: If ``arr`` is empty.
    """
    try:
        array = np.asarray(arr)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be array-like; got {type(arr).__name__}") from exc

    if array.dtype.kind in "biu":
        array = array.astype(np.float32)
    elif array.dtype.kind != "f":
        raise TypeError(
            f"{name} must be numeric; got dtype {array.dtype}"
        )
    if array.size == 0:
        raise ValueError(f"{name} is empty")
    return array


def _all_nan_like(arr: np.ndarray, func: str) -> np.ndarray:
    """
    Return an all-NaN array shaped like ``arr`` and log why.

    Args:
        arr (np.ndarray): Floating-point input array.
        func (str): Name of the calling function, for the log message.

    Returns:
        np.ndarray: Array of NaNs with ``arr``'s shape and dtype.
    """
    logger.warning(
        "%s: input %s contains no valid (non-NaN) values; returning all-NaN.",
        func, arr.shape,
    )
    return np.full_like(arr, np.nan)


def _clip_stretch(arr: np.ndarray, lo: Any, hi: Any) -> np.ndarray:
    """
    Linearly map ``[lo, hi]`` onto ``[0, 1]`` and clip; NaNs pass through.

    Args:
        arr (np.ndarray): Floating-point input.
        lo (Any): Value mapped to 0 (a scalar of ``arr``'s dtype).
        hi (Any): Value mapped to 1 (a scalar of ``arr``'s dtype).

    Returns:
        np.ndarray: Stretched array, same shape and dtype as ``arr``. If
            ``hi == lo`` (constant input), every valid cell maps to 0.
    """
    return np.clip((arr - lo) / (hi - lo + _EPS), 0, 1)


def _is_fast_path_stretch_std(arr: Any) -> bool:
    """
    Return whether :func:`stretch_std` can use the Rust kernel for ``arr``.

    The kernel only accepts 2-D ``float32`` numpy arrays; anything else
    (other dtypes, 1-D/3-D arrays, lists) goes through numpy.

    Args:
        arr (Any): Candidate input.

    Returns:
        bool: ``True`` if the extension is loaded, has a ``stretch_std``
            kernel, and ``arr`` is a 2-D ``float32`` ndarray.
    """
    return (
        _rust is not None
        and hasattr(_rust, "stretch_std")
        and isinstance(arr, np.ndarray)
        and arr.dtype == np.float32
        and arr.ndim == 2
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize(
    arr: npt.ArrayLike,
    low: float = 1,
    high: float = 99,
) -> np.ndarray:
    """
    Percentile-stretch an array to ``[0, 1]``, robust to outliers.

    Values at or below the ``low`` percentile map to 0, values at or
    above the ``high`` percentile map to 1, with a linear ramp between.
    NaNs are ignored when computing the percentiles and pass through as
    NaN.

    The input's floating dtype is preserved. ``np.nanpercentile`` always
    returns ``float64`` scalars, and under NEP 50 a numpy ``float64``
    scalar (unlike a Python float) *does* promote a ``float32`` array,
    so the percentiles are cast back to the input dtype first. Integer
    inputs are converted to ``float32``.

    Args:
        arr (npt.ArrayLike): Input values, any shape.
        low (float): Lower percentile, in ``[0, 100)``.
        high (float): Upper percentile, in ``(low, 100]``.

    Returns:
        np.ndarray: Stretched array with ``arr``'s shape, values in
            ``[0, 1]`` or NaN. A constant input maps to all zeros; an
            all-NaN input returns all-NaN (with a logged warning).

    Raises:
        TypeError: If ``arr`` is not numeric.
        ValueError: If ``arr`` is empty, or the percentiles are not
            finite with ``0 <= low < high <= 100``.

    Examples:
        >>> normalize(np.array([0.0, 50.0, 100.0]), low=0, high=100)
        array([0. , 0.5, 1. ])
    """
    if not (math.isfinite(low) and math.isfinite(high) and 0 <= low < high <= 100):
        raise ValueError(
            f"Percentiles need 0 <= low < high <= 100; got low={low}, high={high}"
        )
    array = _as_float_array(arr)

    # All-NaN input makes nanpercentile emit a RuntimeWarning; detect the
    # NaN result instead and report it through logging.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = np.nanpercentile(array, [low, high]).astype(array.dtype)
    if np.isnan(lo):
        return _all_nan_like(array, "normalize")
    if lo == hi:
        logger.debug("normalize: percentiles equal (%s); output is all 0.", lo)
    return _clip_stretch(array, lo, hi)


def stretch_std(arr: npt.ArrayLike, n_std: float = 4) -> np.ndarray:
    """
    Stretch ``mean +/- n_std * std`` onto ``[0, 1]``.

    This is the "Standard Deviation" stretch symbology ArcGIS Pro uses
    for curvature and hillshade. Values below ``mean - n_std * std`` map
    to 0, values above ``mean + n_std * std`` map to 1. NaNs are ignored
    for the statistics and pass through as NaN. The standard deviation is
    the population std (``ddof=0``), matching ``np.nanstd``.

    Dispatch::

        2-D float32 ndarray + Rust extension loaded -> Rust fused kernel
        anything else, or the Rust kernel raised  -> numpy

    Both paths give the same result within float tolerance. The Rust
    kernel accumulates in ``float64`` for robustness, so it is not
    expected to bit-match numpy's ``float32`` accumulation, only to agree
    with it statistically.

    Args:
        arr (npt.ArrayLike): Input values, any shape (Rust handles 2-D
            ``float32`` only; everything else uses numpy).
        n_std (float): Half-width of the stretch window, in standard
            deviations. Must be positive and finite.

    Returns:
        np.ndarray: Stretched array with ``arr``'s shape, values in
            ``[0, 1]`` or NaN. A constant input maps to all zeros; an
            all-NaN input returns all-NaN (with a logged warning, numpy
            path).

    Raises:
        TypeError: If ``arr`` is not numeric.
        ValueError: If ``arr`` is empty, or ``n_std`` is not positive and
            finite.

    Examples:
        >>> stretch_std(np.array([[-1.0, 0.0, 1.0]], dtype=np.float32), n_std=1)
        array([[0. , 0.5, 1. ]], dtype=float32)
    """
    if not (isinstance(n_std, (int, float, np.number))
            and math.isfinite(n_std) and n_std > 0):
        raise ValueError(f"n_std must be a positive finite number; got {n_std!r}")

    if _is_fast_path_stretch_std(arr):
        if arr.size == 0:
            raise ValueError("arr is empty")
        try:
            return _rust.stretch_std(np.ascontiguousarray(arr), float(n_std))
        except Exception as exc:  # any kernel failure -> numpy fallback
            logger.warning(
                "Rust stretch_std failed on %s array (%s: %s); falling back "
                "to numpy.", arr.shape, type(exc).__name__, exc,
            )

    array = _as_float_array(arr)

    # Variance from the already-computed mean, instead of np.nanstd()
    # (which recomputes its own mean) -- saves one full-array pass.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(array)
        if np.isnan(mean):
            return _all_nan_like(array, "stretch_std")
        std = np.sqrt(np.nanmean((array - mean) ** 2))

    if std == 0:
        logger.debug("stretch_std: input is constant; output is all 0.")
    lo = mean - n_std * std
    hi = mean + n_std * std
    return _clip_stretch(array, lo, hi)


def bilinear_resample(
    arr: npt.ArrayLike,
    target_shape: Sequence[int],
) -> npt.NDArray[np.float32]:
    """
    Bilinearly resample a 2-D array to ``target_shape``, preserving voids.

    Naive bilinear resampling of an array with NaNs spreads each NaN into
    its neighbours. Instead, voids are nearest-filled first (via
    :func:`TerraTexture.io._fill_nan_nearest`), the filled array is
    resampled bilinearly, and the NaN mask -- resampled with
    nearest-neighbour -- is re-applied. Voids keep roughly their original
    footprint: they neither bleed into valid data nor vanish.

    Args:
        arr (npt.ArrayLike): 2-D input array. Converted to ``float32``.
        target_shape (Sequence[int]): Output ``(rows, cols)``; both must
            be positive integers.

    Returns:
        npt.NDArray[np.float32]: Resampled array of exactly
            ``target_shape``. All-NaN input gives all-NaN output.

    Raises:
        TypeError: If ``arr`` is not numeric.
        ValueError: If ``arr`` is not 2-D or is empty, or
            ``target_shape`` is not two positive integers.

    Examples:
        >>> bilinear_resample(np.zeros((10, 10)), (20, 30)).shape
        (20, 30)
    """
    array = _as_float_array(arr).astype(np.float32, copy=False)
    if array.ndim != 2:
        raise ValueError(f"arr must be 2-D; got shape {array.shape}")

    try:
        rows, cols = (int(n) for n in target_shape)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"target_shape must be two integers (rows, cols); got {target_shape!r}"
        ) from exc
    if rows <= 0 or cols <= 0:
        raise ValueError(f"target_shape must be positive; got {target_shape!r}")

    if (rows, cols) == array.shape:
        return array.copy()

    filled, nan_mask = _fill_nan_nearest(array)
    zoom_factors = (rows / array.shape[0], cols / array.shape[1])
    resampled = zoom(filled, zoom_factors, order=1)

    if nan_mask.any():
        mask_resampled = zoom(nan_mask.astype(np.float32), zoom_factors, order=0) > 0.5
        resampled = np.where(mask_resampled, np.nan, resampled).astype(
            np.float32, copy=False
        )

    # scipy rounds the output size from the zoom factors; this should
    # always land on target_shape, but fail loudly rather than return a
    # misaligned grid if it ever doesn't.
    if resampled.shape != (rows, cols):
        raise RuntimeError(
            f"bilinear_resample produced {resampled.shape}, expected {(rows, cols)}"
        )
    logger.debug("Resampled %s -> %s", array.shape, resampled.shape)
    return resampled
