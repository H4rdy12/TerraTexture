"""
Photoshop / SVG-spec blend modes: soft light and luminosity.

The compositing layer of the relief pipeline. Both functions take arrays
of values in ``[0, 1]`` and return a blended array in ``[0, 1]``:

- :func:`soft_light` -- Photoshop "Soft Light", applied per element.
  Used to layer greyscale relief (hillshade, curvature) and for final
  compositing onto imagery.
- :func:`luminosity_blend` -- SVG / Photoshop "Luminosity":
  ``SetLum(backdrop, Lum(source))``. Replaces the lightness of an RGB
  backdrop with a greyscale layer while keeping the backdrop's hue *and*
  saturation. This is the correct way to drape relief over colour
  imagery; per-channel soft light can desaturate or shift hue.

Pipeline position::

    DEM -> derivatives -> stretch (to [0, 1]) -> blend -> plot / export

This module is pure numpy with no geospatial dependencies (no rasterio,
no matplotlib), so it can be unit-tested in isolation. See
``docs/formulas.md`` for the formulas.

Input conventions:
    - Values are expected in ``[0, 1]``. Out-of-range values are not an
      error, but results are only meaningful inside that range.
    - Inputs must be floating point. Integer input (typically ``uint8``
      imagery in ``0..255``) raises ``TypeError``: blend math on raw
      ``0..255`` values gives silently wrong colours, so divide by 255
      first.
    - NaN (nodata) propagates: a NaN in either input gives NaN in the
      output at that pixel.
    - ``float32`` in gives ``float32`` out. Mixing ``float32`` and
      ``float64`` promotes to ``float64`` and uses numpy.

Optional Rust acceleration:
    If the compiled ``terra_texture_rs`` extension (see ``rust/``) is
    importable, fused kernels compute each pixel in one pass instead of
    numpy's several temporary-array passes, for the same result:

    ===================================  ====================================
    Input (all ``float32``)              Kernel
    ===================================  ====================================
    ``soft_light``: 2-D, same shape      ``soft_light``
    ``soft_light``: 3-D, same shape      ``soft_light_rgb`` (or a per-channel
                                         loop of ``soft_light`` on older
                                         builds)
    ``luminosity_blend``: (H, W, 3) and  ``luminosity_blend``
    (H, W)
    ===================================  ====================================

    Anything else -- other dtypes, broadcasting shapes, lists -- uses
    numpy. The fallback is **load-bearing, not incidental**: this
    package's design goal is that :mod:`TerraTexture.derivatives` and
    :mod:`TerraTexture.blend` need nothing beyond numpy/scipy, so nobody
    has to install a Rust toolchain to run curvature analysis.

    If a kernel raises at runtime, the call logs a warning and returns
    the numpy result, so a broken extension costs speed, never results.

Diagnostics:
    ``_rust is not None`` shows whether the extension loaded; if it is
    ``None``, ``_RUST_IMPORT_ERROR`` holds the reason. "Not installed" is
    logged at DEBUG level. "Installed but failed to import" (an ABI or
    Python-version mismatch, a stale build, a missing symbol) is logged
    as a WARNING, and a build missing some kernels is reported at import.

    Build the extension with ``uv sync --extra rust`` (or
    ``maturin develop`` from ``rust/``). Rust-vs-numpy parity tests live
    in ``tests/test_blend_rust.py`` and run only when it is built.

Dependencies:
    numpy only.

Examples:
    Soft-light a curvature layer onto a hillshade::

        relief = soft_light(stretch_std(hillshade), stretch_std(curvature))

    Drape that relief over satellite imagery (``uint8`` -> ``[0, 1]``)::

        rgb = basemap_uint8.astype(np.float32) / 255
        draped = luminosity_blend(rgb, relief)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional Rust extension
# ---------------------------------------------------------------------------

# Name of the compiled extension module (see rust/pyproject.toml).
_RUST_MODULE = "terra_texture_rs"

# Kernels this module can use, and what happens without each one.
_RUST_KERNELS: dict[str, str] = {
    "soft_light": "soft_light() will use numpy",
    "soft_light_rgb": (
        "3-D soft_light() will loop the 2-D kernel per channel (slower)"
    ),
    "luminosity_blend": "luminosity_blend() will use numpy",
}

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
    for _name, _effect in _RUST_KERNELS.items():
        if not hasattr(_rust, _name):
            logger.warning(
                "%s has no %s kernel (stale build?); %s.",
                _RUST_MODULE, _name, _effect,
            )
    del _name, _effect


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Rec. 601 luma weights used by the SVG / Photoshop Lum() function.
_LUM_WEIGHTS = (0.3, 0.59, 0.11)

# Guards ClipColor() denominators against division by zero.
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _has_kernel(name: str) -> bool:
    """
    Return whether the Rust extension is loaded and exports ``name``.

    Args:
        name (str): Kernel function name, e.g. ``"soft_light"``.

    Returns:
        bool: ``True`` if ``_rust.<name>`` exists.
    """
    return _rust is not None and hasattr(_rust, name)


def _is_python_scalar(value: object) -> bool:
    """
    Return whether ``value`` is a plain Python ``int`` or ``float``.

    Args:
        value (object): Candidate input.

    Returns:
        bool: ``True`` for Python numbers (``bool`` excluded).
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_float_array(value: npt.ArrayLike, name: str) -> np.ndarray:
    """
    Convert an input to a floating-point numpy array, without copying.

    Plain Python ``int`` / ``float`` scalars (e.g. ``soft_light(a, 0.7)``)
    are accepted and become 0-d ``float64`` arrays for validation. Callers
    pass the original scalar on to numpy, so it still follows numpy's
    weak-scalar promotion (a ``float32`` partner stays ``float32``).

    Args:
        value (npt.ArrayLike): Array or array-like of values in ``[0, 1]``.
        name (str): Argument name, used in error messages.

    Returns:
        np.ndarray: ``value`` as a floating ndarray (dtype unchanged).

    Raises:
        TypeError: If ``value`` isn't numeric, or is integer/boolean.
            Integer inputs are rejected rather than converted because
            ``uint8`` imagery in ``0..255`` would blend silently wrong.
    """
    if _is_python_scalar(value):
        return np.asarray(value, dtype=np.float64)
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be array-like; got {type(value).__name__}"
        ) from exc

    if array.dtype.kind in "biu":
        raise TypeError(
            f"{name} has integer dtype {array.dtype}; blend inputs must be "
            "floats in [0, 1]. For 8-bit imagery use "
            "`arr.astype(np.float32) / 255`."
        )
    if array.dtype.kind != "f":
        raise TypeError(f"{name} must be numeric; got dtype {array.dtype}")
    return array


def _run_kernel(name: str, *args: np.ndarray) -> np.ndarray | None:
    """
    Call a Rust kernel, returning ``None`` (and logging) if it raises.

    All arguments are made C-contiguous first, as the kernels require.

    Args:
        name (str): Kernel function name on ``_rust``.
        *args (np.ndarray): Arrays to pass to the kernel.

    Returns:
        np.ndarray | None: The kernel's output, or ``None`` if the kernel
            raised, in which case the caller should fall back to numpy.

    Note:
        A Rust *panic* surfaces as pyo3's ``PanicException``, which
        derives from ``BaseException``. It is deliberately not caught,
        since that would also swallow ``KeyboardInterrupt``.
    """
    try:
        return getattr(_rust, name)(*(np.ascontiguousarray(a) for a in args))
    except Exception as exc:  # any kernel failure -> numpy fallback
        shapes = ", ".join(str(a.shape) for a in args)
        logger.warning(
            "Rust %s failed on arrays %s (%s: %s); falling back to numpy.",
            name, shapes, type(exc).__name__, exc,
        )
        return None


def _is_fast_path_soft_light_2d(a: np.ndarray, b: np.ndarray) -> bool:
    """
    Return whether :func:`soft_light` can use the 2-D Rust kernel.

    Args:
        a (np.ndarray): Base array.
        b (np.ndarray): Blend array.

    Returns:
        bool: ``True`` for two 2-D ``float32`` arrays of equal shape with
            the ``soft_light`` kernel available.
    """
    return (
        _has_kernel("soft_light")
        and a.dtype == np.float32 and b.dtype == np.float32
        and a.ndim == 2 and a.shape == b.shape
    )


def _is_fast_path_soft_light_multichannel(a: np.ndarray, b: np.ndarray) -> bool:
    """
    Return whether :func:`soft_light` can use Rust for ``(H, W, C)`` input.

    Typical caller: ``basemap.py``'s final
    ``soft_light(luminosity_composite, basemap_rgb)`` compositing step.
    Prefers the dedicated 3-D kernel ``soft_light_rgb``: one call over
    the whole contiguous buffer, instead of one call per channel (each
    needing its own contiguous copy, since a channel slice ``a[..., c]``
    of an ``(H, W, C)`` array is non-contiguous). Soft light has no
    cross-channel interaction, so both routes compute the same thing.

    Args:
        a (np.ndarray): Base array.
        b (np.ndarray): Blend array.

    Returns:
        bool: ``True`` for two 3-D ``float32`` arrays of equal shape with
            either ``soft_light_rgb`` or the 2-D ``soft_light`` kernel
            available.
    """
    return (
        (_has_kernel("soft_light_rgb") or _has_kernel("soft_light"))
        and a.dtype == np.float32 and b.dtype == np.float32
        and a.ndim == 3 and a.shape == b.shape
    )


def _is_fast_path_luminosity_blend(
    backdrop_rgb: np.ndarray,
    luminosity: np.ndarray,
) -> bool:
    """
    Return whether :func:`luminosity_blend` can use the Rust kernel.

    Args:
        backdrop_rgb (np.ndarray): Backdrop array.
        luminosity (np.ndarray): Luminosity array.

    Returns:
        bool: ``True`` for a ``float32`` ``(H, W, 3)`` backdrop and a
            ``float32`` ``(H, W)`` luminosity with the kernel available.
    """
    return (
        _has_kernel("luminosity_blend")
        and backdrop_rgb.dtype == np.float32 and luminosity.dtype == np.float32
        and backdrop_rgb.ndim == 3 and backdrop_rgb.shape[-1] == 3
        and luminosity.shape == backdrop_rgb.shape[:2]
    )


def _soft_light_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Soft light in pure numpy (supports broadcasting).

    Formula, per element::

        b <= 0.5:  2ab + a^2 (1 - 2b)
        b >  0.5:  2a (1 - b) + sqrt(a) (2b - 1)

    ``a`` is clipped to ``[0, 1]`` inside the square root so out-of-range
    input can't produce NaN there.

    Args:
        a (np.ndarray): Base array in ``[0, 1]``.
        b (np.ndarray): Blend array in ``[0, 1]``.

    Returns:
        np.ndarray: Blended array, clipped to ``[0, 1]``.
    """
    return np.clip(
        np.where(
            b <= 0.5,
            2 * a * b + a ** 2 * (1 - 2 * b),
            2 * a * (1 - b) + np.sqrt(np.clip(a, 0, 1)) * (2 * b - 1),
        ),
        0, 1,
    )


def _lum(rgb: np.ndarray) -> np.ndarray:
    """
    Return the SVG / Photoshop luminosity of an ``(..., 3)`` RGB array.

    Args:
        rgb (np.ndarray): RGB values in ``[0, 1]``, channels last.

    Returns:
        np.ndarray: ``0.3 R + 0.59 G + 0.11 B``, shape ``rgb.shape[:-1]``.
    """
    red, green, blue = _LUM_WEIGHTS
    return red * rgb[..., 0] + green * rgb[..., 1] + blue * rgb[..., 2]


def _clip_color(rgb: np.ndarray) -> np.ndarray:
    """
    Pull RGB triples back into gamut around their luminosity (ClipColor).

    Implements the SVG compositing spec's ``ClipColor()``. Out-of-gamut
    channels are scaled towards the pixel's own luminosity instead of
    being clipped independently, which would shift hue and saturation.

    Args:
        rgb (np.ndarray): ``(..., 3)`` RGB array, possibly out of gamut.

    Returns:
        np.ndarray: RGB array in ``[0, 1]``, same shape.
    """
    lum = _lum(rgb)[..., None]
    channel_min = rgb.min(axis=-1, keepdims=True)
    channel_max = rgb.max(axis=-1, keepdims=True)
    rgb = np.where(
        channel_min < 0,
        lum + (rgb - lum) * lum / (lum - channel_min + _EPS),
        rgb,
    )
    rgb = np.where(
        channel_max > 1,
        lum + (rgb - lum) * (1 - lum) / (channel_max - lum + _EPS),
        rgb,
    )
    return np.clip(rgb, 0, 1)


def _luminosity_blend_numpy(
    backdrop_rgb: np.ndarray,
    luminosity: np.ndarray,
) -> np.ndarray:
    """
    Luminosity blend in pure numpy: ``SetLum(backdrop, luminosity)``.

    Args:
        backdrop_rgb (np.ndarray): ``(H, W, 3)`` RGB backdrop.
        luminosity (np.ndarray): ``(H, W)`` target luminosity.

    Returns:
        np.ndarray: ``(H, W, 3)`` blended RGB in ``[0, 1]``.
    """
    delta = luminosity[..., None] - _lum(backdrop_rgb)[..., None]
    return _clip_color(backdrop_rgb + delta)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def soft_light(base: npt.ArrayLike, blend: npt.ArrayLike) -> np.ndarray:
    """
    Blend two layers with Photoshop-style soft light.

    Darkens where ``blend < 0.5``, lightens where ``blend > 0.5``, and
    leaves ``base`` unchanged where ``blend == 0.5``. Applied per element,
    so it works on greyscale ``(H, W)`` layers and on ``(H, W, C)``
    imagery alike.

    Dispatch::

        both float32, 2-D, same shape  -> Rust soft_light
        both float32, 3-D, same shape  -> Rust soft_light_rgb
                                          (per-channel soft_light on
                                          older builds)
        anything else, or kernel error -> numpy (supports broadcasting,
                                          e.g. a scalar or (H, W, 1)
                                          blend)

    Args:
        base (npt.ArrayLike): Base layer, floats in ``[0, 1]``.
        blend (npt.ArrayLike): Blend layer, floats in ``[0, 1]``. Must be
            broadcast-compatible with ``base``. Python scalars (e.g.
            ``0.7``) are fine and take the other input's dtype.

    Returns:
        np.ndarray: Blended layer in ``[0, 1]`` with the broadcast shape
            of the inputs. NaN wherever either input is NaN.

    Raises:
        TypeError: If either input is non-numeric, or an integer-typed
            *array* (e.g. ``uint8`` imagery not yet divided by 255).
        ValueError: If the shapes can't be broadcast together.

    Examples:
        >>> soft_light(np.array([0.5]), np.array([0.5]))
        array([0.5])
    """
    a = _as_float_array(base, "base")
    b = _as_float_array(blend, "blend")
    try:
        np.broadcast_shapes(a.shape, b.shape)
    except ValueError as exc:
        raise ValueError(
            f"base {a.shape} and blend {b.shape} shapes are not compatible"
        ) from exc

    if _is_fast_path_soft_light_2d(a, b):
        result = _run_kernel("soft_light", a, b)
        if result is not None:
            return result

    elif _is_fast_path_soft_light_multichannel(a, b):
        if _has_kernel("soft_light_rgb"):
            result = _run_kernel("soft_light_rgb", a, b)
            if result is not None:
                return result
        else:
            # Extension built before soft_light_rgb existed: loop the 2-D
            # kernel per channel -- slower, but still faster than numpy.
            out = np.empty_like(a)
            for channel in range(a.shape[-1]):
                layer = _run_kernel("soft_light", a[..., channel], b[..., channel])
                if layer is None:
                    break
                out[..., channel] = layer
            else:
                return out

    # Python scalars go to numpy unconverted, so numpy's weak-scalar rules
    # apply exactly as before: float32 stays float32, bit for bit.
    return _soft_light_numpy(
        base if _is_python_scalar(base) else a,
        blend if _is_python_scalar(blend) else b,
    )


def luminosity_blend(
    backdrop_rgb: npt.ArrayLike,
    luminosity: npt.ArrayLike,
) -> np.ndarray:
    """
    Replace an RGB image's lightness with a greyscale layer (Luminosity).

    SVG / Photoshop "Luminosity" blend mode: ``SetLum(backdrop,
    Lum(source))``. The backdrop keeps its hue *and* saturation; only its
    lightness becomes ``luminosity``. This is the correct operation for
    draping greyscale relief over colour imagery -- unlike per-channel
    soft light, it cannot desaturate or shift hue. Out-of-gamut results
    are pulled back with the spec's ``ClipColor()`` rather than clipped
    per channel.

    Dispatch::

        float32 (H, W, 3) + float32 (H, W)  -> Rust luminosity_blend
        anything else, or kernel error      -> numpy

    Args:
        backdrop_rgb (npt.ArrayLike): ``(H, W, 3)`` RGB, floats in
            ``[0, 1]``. RGBA must be sliced to RGB first
            (``rgba[..., :3]``).
        luminosity (npt.ArrayLike): ``(H, W)`` greyscale, floats in
            ``[0, 1]``.

    Returns:
        np.ndarray: ``(H, W, 3)`` RGB in ``[0, 1]``. NaN wherever either
            input is NaN.

    Raises:
        TypeError: If either input is non-numeric or integer-typed.
        ValueError: If ``backdrop_rgb`` is not ``(..., 3)``, or
            ``luminosity``'s shape doesn't match ``backdrop_rgb.shape[:-1]``.

    Examples:
        >>> grey = np.full((1, 1, 3), 0.2)
        >>> luminosity_blend(grey, np.full((1, 1), 0.8))
        array([[[0.8, 0.8, 0.8]]])
    """
    backdrop = _as_float_array(backdrop_rgb, "backdrop_rgb")
    lum = _as_float_array(luminosity, "luminosity")

    if backdrop.ndim < 1 or backdrop.shape[-1] != 3:
        hint = (
            " For RGBA imagery pass rgba[..., :3]."
            if backdrop.ndim >= 1 and backdrop.shape[-1] == 4 else ""
        )
        raise ValueError(
            f"backdrop_rgb must have 3 channels in its last axis; got shape "
            f"{backdrop.shape}.{hint}"
        )
    if lum.shape != backdrop.shape[:-1]:
        raise ValueError(
            f"luminosity shape {lum.shape} must match backdrop_rgb's "
            f"{backdrop.shape[:-1]} (backdrop_rgb is {backdrop.shape})"
        )

    if _is_fast_path_luminosity_blend(backdrop, lum):
        result = _run_kernel("luminosity_blend", backdrop, lum)
        if result is not None:
            return result

    return _luminosity_blend_numpy(backdrop, lum)
