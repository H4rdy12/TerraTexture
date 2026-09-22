"""
Offline relief figures and an interactive before/after comparison slider.

Visual summaries that need only numpy, scipy and matplotlib -- no
rasterio or contextily, no network. Use these for curvature analysis on
an in-memory DEM; see :mod:`TerraTexture.basemap` for relief draped over
real-world imagery.

Public API:

- :func:`plot_dem_curvature_softlight` -- a 6-panel figure: elevation,
  profile and planform curvature, plain hillshade, curvature-enhanced
  ("soft-lit") hillshade, and elevation colours lit by that relief.
- :func:`compare_slider` -- a swipe-to-compare HTML widget for two
  images in a Jupyter notebook (e.g. ``layers["basemap"]`` against
  ``layers["final"]``).

Soft-lit relief::

    profile, planform --normalise, average, smooth--> curvature signal
    hillshade  --soft_light(hillshade, curvature signal)--> soft-lit relief
    elevation colours --soft_light(colours, soft-lit relief)--> composite

NaN (nodata):
    Voids stay NaN in every returned array and show as blank in the
    figure. Curvature smoothing is NaN-aware, so voids don't spread
    across the smoothing kernel. In :func:`compare_slider`, NaN pixels
    are drawn black.

Logging:
    Logged under ``TerraTexture.plotting``: a saved figure at INFO;
    image downsampling and NaN pixels in slider images at DEBUG.

Dependencies:
    numpy, scipy and matplotlib. :func:`compare_slider` also needs
    IPython (a Jupyter kernel) and Pillow (installed with matplotlib).

Examples:
    Summary figure for a DEM already in memory::

        dem, cellsize = load_dem("tile_dem.tif")
        fig, axes, results = plot_dem_curvature_softlight(dem, cellsize)

    Swipe between imagery and relief in a notebook::

        compare_slider(layers["basemap"], layers["final"],
                       labels=("Imagery", "Relief"))
"""

from __future__ import annotations

import base64
import html
import io
import logging
import math
import os
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from .blend import soft_light
from .derivatives import curvatures, hillshade
from .stretch import normalize

if TYPE_CHECKING:
    import numpy.typing as npt
    from IPython.display import HTML
    from matplotlib.figure import Figure


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Resolution for saved summary figures.
_SAVE_DPI = 150

# Neutral soft-light value: blending with 0.5 leaves the base unchanged,
# so filling voids with it before smoothing adds no false relief.
_NEUTRAL = 0.5

# Channel counts compare_slider can display (grey, RGB, RGBA).
_DISPLAYABLE_CHANNELS = (1, 3, 4)


# ---------------------------------------------------------------------------
# 6-panel summary figure
# ---------------------------------------------------------------------------

def _check_positive(name: str, value: float, allow_zero: bool = False) -> float:
    """
    Check that a numeric option is finite and positive.

    Args:
        name (str): Argument name, for the error message.
        value (float): Value to check.
        allow_zero (bool): Accept 0 as well.

    Returns:
        float: ``value`` as a float.

    Raises:
        ValueError: If ``value`` isn't a finite number > 0 (or >= 0).
    """
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number; got {value!r}") from exc
    if not math.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
        bound = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be finite and {bound}; got {value!r}")
    return number


def _curvature_signal(
    profile: np.ndarray,
    planform: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    Combine both curvatures into one smoothed ``[0, 1]`` "form" signal.

    Each curvature is percentile-normalised, the two are averaged, and
    the result is Gaussian-smoothed (raw curvature is very noisy).
    Smoothing is NaN-aware: voids are temporarily filled with neutral
    0.5, then restored, so they don't grow by the kernel's radius.

    Args:
        profile (np.ndarray): Profile curvature (NaN = nodata).
        planform (np.ndarray): Planform curvature (NaN = nodata).
        sigma (float): Gaussian sigma in cells; 0 disables smoothing.

    Returns:
        np.ndarray: Curvature signal, NaN where either input is NaN.
    """
    signal = 0.5 * normalize(profile) + 0.5 * normalize(planform)
    if sigma == 0:
        return signal
    voids = np.isnan(signal)
    if not voids.any():
        return gaussian_filter(signal, sigma=sigma)
    smoothed = gaussian_filter(np.where(voids, _NEUTRAL, signal), sigma=sigma)
    return np.where(voids, np.nan, smoothed)


def plot_dem_curvature_softlight(
    dem: npt.ArrayLike,
    cellsize: float = 1.0,
    azimuth: float = 315,
    altitude: float = 45,
    curvature_smooth_sigma: float = 1.0,
    elev_cmap: str = "terrain",
    curv_cmap: str = "RdBu_r",
    curv_vlim: float = 0.05,
    figsize: tuple[float, float] = (16, 10),
    out_png: str | os.PathLike[str] | None = None,
    show: bool = True,
) -> tuple[Figure, np.ndarray, dict[str, np.ndarray]]:
    """
    Draw the 6-panel curvature / soft-light relief summary for a DEM.

    Panels::

        DEM (elevation)     | profile curvature   | planform curvature
        standard hillshade  | soft-lit relief     | elevation x relief

    "Soft-lit relief" is the hillshade soft-light-blended with a smoothed
    curvature signal, which punches up ridges and channels. The final
    panel colours elevation with ``elev_cmap`` and lights it with that
    relief.

    Args:
        dem (npt.ArrayLike): 2-D elevation array (NaN = nodata), at least
            3 x 3.
        cellsize (float): Grid spacing in the elevations' units (normally
            metres); scales the derivatives.
        azimuth (float): Sun azimuth for the hillshade, degrees clockwise
            from north.
        altitude (float): Sun altitude, degrees 0-90.
        curvature_smooth_sigma (float): Gaussian sigma (in cells) for the
            combined curvature signal; 0 disables smoothing.
        elev_cmap (str): Colormap for elevation.
        curv_cmap (str): Diverging colormap for the curvature panels.
        curv_vlim (float): Symmetric colour limit (+/-) for the curvature
            panels, in 1/(elevation units).
        figsize (tuple[float, float]): Figure size in inches.
        out_png (str | PathLike | None): Save the figure here (150 dpi).
        show (bool): Call ``plt.show()`` at the end.

    Returns:
        tuple: ``(fig, axes, results)``:

            - ``fig`` (Figure) and ``axes`` (2 x 3 ndarray of Axes).
            - ``results`` (dict[str, np.ndarray]): ``profile``,
              ``planform``, ``hillshade``, ``soft_lit`` (H, W) and
              ``composite`` (H, W, 3), all float32.

    Raises:
        TypeError: If ``dem`` is not numeric.
        ValueError: If ``dem`` isn't 2-D / is smaller than 3 x 3, or
            ``cellsize``, ``azimuth``, ``altitude``,
            ``curvature_smooth_sigma`` or ``curv_vlim`` is invalid, or a
            colormap name is unknown.
        OSError: If ``out_png`` can't be written (the figure is closed).

    Examples:
        >>> dem, cellsize = load_dem()  # doctest: +SKIP
        >>> fig, axes, results = plot_dem_curvature_softlight(  # doctest: +SKIP
        ...     dem, cellsize, show=False)
    """
    sigma = _check_positive(
        "curvature_smooth_sigma", curvature_smooth_sigma, allow_zero=True
    )
    curv_vlim = _check_positive("curv_vlim", curv_vlim)
    elev_colormap = plt.get_cmap(elev_cmap)  # ValueError names valid maps
    plt.get_cmap(curv_cmap)

    dem = np.asarray(dem, dtype=np.float32)
    profile, planform = curvatures(dem, cellsize)
    shade = hillshade(dem, cellsize, azimuth=azimuth, altitude=altitude)

    curv_signal = _curvature_signal(profile, planform, sigma)
    lit = soft_light(shade, curv_signal)

    # cmap(...) always returns float64 RGBA; cast down so soft_light()
    # doesn't upcast the whole composite.
    elev_rgb = elev_colormap(normalize(dem))[:, :, :3].astype(np.float32)
    lit_rgb = np.repeat(lit[:, :, None], 3, axis=2)
    composite = soft_light(elev_rgb, lit_rgb)

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    panels: list[tuple[Any, np.ndarray, dict[str, Any], str, bool]] = [
        (axes[0, 0], dem, {"cmap": elev_cmap}, "DEM (elevation)", True),
        (axes[0, 1], profile,
         {"cmap": curv_cmap, "vmin": -curv_vlim, "vmax": curv_vlim},
         "Profile curvature\n(+convex/decel, -concave/accel)", True),
        (axes[0, 2], planform,
         {"cmap": curv_cmap, "vmin": -curv_vlim, "vmax": curv_vlim},
         "Planform curvature\n(+ridges/divergent, -channels/convergent)", True),
        (axes[1, 0], shade, {"cmap": "gray"}, "Standard hillshade", False),
        (axes[1, 1], lit, {"cmap": "gray"},
         "Soft-lit relief\n(hillshade \u2295 curvature)", False),
        (axes[1, 2], composite, {}, "Elevation \u2295 soft-lit relief", False),
    ]
    for ax, image, style, title, colorbar in panels:
        mappable = ax.imshow(image, **style)
        ax.set_title(title)
        if colorbar:
            fig.colorbar(mappable, ax=ax, shrink=0.7)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()

    if out_png:
        try:
            fig.savefig(out_png, dpi=_SAVE_DPI)
        except OSError:
            plt.close(fig)
            raise
        logger.info("Saved figure to %s", out_png)
    if show:
        plt.show()

    results = {
        "profile": profile,
        "planform": planform,
        "hillshade": shade,
        "soft_lit": lit,
        "composite": composite,
    }
    return fig, axes, results


# ---------------------------------------------------------------------------
# Before/after comparison slider
# ---------------------------------------------------------------------------

# Both images fill the container (width:100%), so they stay aligned at any
# display size; the "before" image is revealed with clip-path. Pointer
# events cover mouse, pen and touch.
_COMPARE_SLIDER_TEMPLATE = """
__TITLE_HTML__
<div id="__WIDGET_ID__" style="position:relative; width:__WIDTH__px; max-width:100%;
    user-select:none; touch-action:none;">
  <img src="__AFTER_URI__" alt="__LABEL_RIGHT__"
       style="display:block; width:100%; height:auto;">
  <img class="before" src="__BEFORE_URI__" alt="__LABEL_LEFT__"
       style="position:absolute; top:0; left:0; width:100%; height:100%;
              clip-path:inset(0 50% 0 0);">
  <div class="handle" style="position:absolute; top:0; left:50%; width:2px; height:100%;
    background:white; box-shadow:0 0 4px rgba(0,0,0,0.6); cursor:ew-resize;">
    <div style="position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
        width:32px; height:32px; border-radius:50%; background:white;
        box-shadow:0 0 6px rgba(0,0,0,0.5); display:flex; align-items:center;
        justify-content:center; font-family:sans-serif; font-size:14px;">&#8596;</div>
  </div>
  <div style="position:absolute; top:8px; left:8px; background:rgba(0,0,0,0.55);
    color:white; padding:2px 8px; border-radius:4px; font-family:sans-serif;
    font-size:12px; pointer-events:none;">__LABEL_LEFT__</div>
  <div style="position:absolute; top:8px; right:8px; background:rgba(0,0,0,0.55);
    color:white; padding:2px 8px; border-radius:4px; font-family:sans-serif;
    font-size:12px; pointer-events:none;">__LABEL_RIGHT__</div>
</div>
<input type="range" min="0" max="100" value="50" step="0.1" id="__WIDGET_ID___slider"
    aria-label="Comparison position"
    style="width:__WIDTH__px; max-width:100%; margin-top:8px;">
<script>
(function() {
  const container = document.getElementById("__WIDGET_ID__");
  const before = container.querySelector(".before");
  const handle = container.querySelector(".handle");
  const slider = document.getElementById("__WIDGET_ID___slider");

  function setPct(pct) {
    pct = Math.max(0, Math.min(100, pct));
    before.style.clipPath = "inset(0 " + (100 - pct) + "% 0 0)";
    handle.style.left = pct + "%";
    slider.value = pct;
  }

  function fromPointer(e) {
    const rect = container.getBoundingClientRect();
    setPct(((e.clientX - rect.left) / rect.width) * 100);
  }

  slider.addEventListener("input", () => setPct(Number(slider.value)));

  let dragging = false;
  container.addEventListener("pointerdown", (e) => {
    dragging = true;
    container.setPointerCapture(e.pointerId);
    fromPointer(e);
    e.preventDefault();
  });
  container.addEventListener("pointermove", (e) => { if (dragging) fromPointer(e); });
  container.addEventListener("pointerup", () => { dragging = false; });
  container.addEventListener("pointercancel", () => { dragging = false; });
})();
</script>
"""


def _to_display_uint8(arr: np.ndarray, name: str) -> np.ndarray:
    """
    Convert an image array to ``uint8`` grey/RGB/RGBA for the browser.

    Args:
        arr (np.ndarray): 2-D, ``(H, W, 1)``, RGB or RGBA image. Floats
            are taken as ``[0, 1]`` (clipped; NaN becomes black);
            ``uint8`` is used as-is; booleans map to black/white.
        name (str): Argument name, for error messages.

    Returns:
        np.ndarray: ``uint8`` array, 2-D (grey), RGB or RGBA.

    Raises:
        TypeError: If ``arr`` has a non-``uint8`` integer or non-numeric
            dtype (its value range is ambiguous).
        ValueError: If ``arr`` isn't 2-D or 3-D with 1, 3 or 4 channels.
    """
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim not in (2, 3) or (
        arr.ndim == 3 and arr.shape[-1] not in _DISPLAYABLE_CHANNELS
    ):
        raise ValueError(
            f"{name} must be (H, W), (H, W, 3) or (H, W, 4); got shape "
            f"{arr.shape}"
        )
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.bool_:
        return arr.astype(np.uint8) * 255
    if arr.dtype.kind != "f":
        raise TypeError(
            f"{name} has dtype {arr.dtype}; pass floats in [0, 1] or uint8 "
            "(e.g. scale 16-bit imagery with arr / arr.max())"
        )

    nan_count = int(np.isnan(arr).sum())
    if nan_count:
        logger.debug("%s: %d NaN pixel(s) drawn as black", name, nan_count)
        arr = np.nan_to_num(arr, nan=0.0)
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def _downsample(arr: np.ndarray, max_dim: int) -> np.ndarray:
    """
    Stride-decimate so the longest side is at most ``max_dim`` pixels.

    Uses ceiling division for the step: ``int(4000 / 1200) = 3`` would
    leave 1334 px and break the "at most ``max_dim``" guarantee.

    Args:
        arr (np.ndarray): Image, rows and columns first.
        max_dim (int): Maximum longest side in pixels.

    Returns:
        np.ndarray: A strided view of ``arr``.
    """
    step = max(1, -(-max(arr.shape[:2]) // max_dim))
    return arr[::step, ::step]


def _png_data_uri(arr: np.ndarray) -> str:
    """
    Encode a ``uint8`` image as a base64 PNG ``data:`` URI.

    Args:
        arr (np.ndarray): 2-D, RGB or RGBA ``uint8`` image.

    Returns:
        str: ``data:image/png;base64,...``.
    """
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(arr)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def compare_slider(
    before: npt.ArrayLike,
    after: npt.ArrayLike,
    labels: Sequence[str] = ("Before", "After"),
    max_dim: int = 1200,
    title: str | None = None,
) -> HTML:
    """
    Build a swipe-to-compare HTML widget for two same-sized images.

    A draggable divider (or the range slider beneath it) reveals more of
    one image or the other -- the usual before/after pattern of mapping
    and photo tools. The two images are encoded to PNG once; after that
    the browser does all the work, so dragging stays smooth even for
    large arrays (unlike redrawing a matplotlib figure on every move).

    Both images are stride-downsampled to at most ``max_dim`` pixels on
    their longest side, since there's no point embedding more pixels
    than a notebook can show. This is for viewing, not keeping: save the
    full-resolution arrays directly (``plt.imsave("out.png", arr)``).

    Works with mouse, pen and touch, and stays aligned when the notebook
    is narrower than the widget.

    Args:
        before (npt.ArrayLike): Left-hand image: ``(H, W)`` greyscale,
            ``(H, W, 3)`` RGB or ``(H, W, 4)`` RGBA. Floats are taken as
            ``[0, 1]`` (clipped; NaN drawn black); ``uint8`` is used
            as-is. Matches the arrays in basemap's ``layers`` dict and
            :func:`plot_dem_curvature_softlight`'s ``results``.
        after (npt.ArrayLike): Right-hand image; same height and width as
            ``before`` (channel counts may differ).
        labels (Sequence[str]): ``(before, after)`` captions shown at the
            top-left / top-right. HTML is escaped, so any text is safe.
        max_dim (int): Longest side, in pixels, of the embedded images.
        title (str | None): Optional heading above the widget (escaped).

    Returns:
        IPython.display.HTML: Rendered automatically as a notebook cell's
            output; wrap in ``display(...)`` if it isn't the cell's last
            expression.

    Raises:
        ImportError: If IPython or Pillow is not installed.
        TypeError: If an image has an ambiguous dtype (e.g. ``uint16``).
        ValueError: If the images differ in height/width or have an
            unsupported shape, ``labels`` isn't two items, or
            ``max_dim`` isn't a positive integer.

    Examples:
        >>> compare_slider(layers["basemap"], layers["final"],  # doctest: +SKIP
        ...                labels=("Imagery", "Relief"))
    """
    try:
        from IPython.display import HTML
    except ImportError as exc:
        raise ImportError(
            "compare_slider needs IPython; run it in Jupyter or install "
            "ipykernel (`uv sync --group dev`)."
        ) from exc

    if isinstance(labels, str) or len(labels) != 2:
        raise ValueError(f"labels must be two strings (before, after); got {labels!r}")
    if isinstance(max_dim, bool) or not isinstance(max_dim, int) or max_dim < 1:
        raise ValueError(f"max_dim must be a positive integer; got {max_dim!r}")

    before_arr, after_arr = np.asarray(before), np.asarray(after)
    if before_arr.shape[:2] != after_arr.shape[:2]:
        raise ValueError(
            "before and after must have the same height and width; got "
            f"{before_arr.shape} vs {after_arr.shape}"
        )

    before_img = _downsample(_to_display_uint8(before_arr, "before"), max_dim)
    after_img = _downsample(_to_display_uint8(after_arr, "after"), max_dim)
    if before_img.shape[:2] != before_arr.shape[:2]:
        logger.debug(
            "Downsampled %s -> %s for display", before_arr.shape[:2],
            before_img.shape[:2],
        )

    width = before_img.shape[1]
    title_html = (
        '<h4 style="margin:0 0 8px 0; font-family:sans-serif;">'
        f"{html.escape(str(title))}</h4>"
        if title else ""
    )
    widget = (
        _COMPARE_SLIDER_TEMPLATE
        .replace("__TITLE_HTML__", title_html)
        .replace("__WIDGET_ID__", f"compare_{uuid.uuid4().hex[:8]}")
        .replace("__WIDTH__", str(width))
        .replace("__BEFORE_URI__", _png_data_uri(before_img))
        .replace("__AFTER_URI__", _png_data_uri(after_img))
        .replace("__LABEL_LEFT__", html.escape(str(labels[0])))
        .replace("__LABEL_RIGHT__", html.escape(str(labels[1])))
    )
    return HTML(widget)
