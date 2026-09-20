"""
6-panel summary figure: elevation, both curvatures, plain hillshade,
soft-lit relief, and a final elevation+relief composite.

Depends only on numpy/scipy/matplotlib -- no rasterio or contextily.
Use this for offline curvature analysis on an in-memory DEM array; see
`TerraTexture.basemap` for the version that drapes relief over real-world
basemap imagery.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from .derivatives import curvatures, hillshade
from .blend import soft_light
from .stretch import normalize


def plot_dem_curvature_softlight(
    dem,
    cellsize=1.0,
    azimuth=315,
    altitude=45,
    curvature_smooth_sigma=1.0,
    elev_cmap="terrain",
    curv_cmap="RdBu_r",
    curv_vlim=0.05,
    figsize=(16, 10),
    out_png=None,
    show=True,
):
    """Compute profile/planform curvature + soft-light shaded relief for a
    DEM and draw the 6-panel summary figure (elevation, both curvatures,
    plain hillshade, soft-lit relief, and a final elevation+relief composite).

    Parameters
    ----------
    dem : 2D array
        Elevation values.
    cellsize : float
        Grid spacing (map units per pixel); used to scale derivatives.
    azimuth, altitude : float
        Sun position (degrees) for the hillshade.
    curvature_smooth_sigma : float
        Gaussian smoothing applied to the blended curvature signal before
        it's used for shading (raw curvature is very noisy). Set to 0 to
        disable.
    elev_cmap, curv_cmap : str
        Matplotlib colormap names for the elevation and curvature panels.
    curv_vlim : float
        Symmetric colour limit (+/-) for the curvature panels.
    figsize : tuple
        Figure size in inches.
    out_png : str or None
        If given, save the figure to this path.
    show : bool
        If True, call plt.show().

    Returns
    -------
    fig, axes : the matplotlib Figure and Axes array
    results : dict with keys 'profile', 'planform', 'hillshade',
        'soft_lit', 'composite' holding the intermediate arrays
    """
    dem = np.asarray(dem, dtype=np.float32)

    profile, planform = curvatures(dem, cellsize)
    hs = hillshade(dem, cellsize, azimuth=azimuth, altitude=altitude)

    # curvature "form" signal: blend of both curvature types, smoothed slightly
    curv_signal = 0.5 * normalize(profile) + 0.5 * normalize(planform)
    if curvature_smooth_sigma > 0:
        curv_signal = gaussian_filter(curv_signal, sigma=curvature_smooth_sigma)

    # soft-light the hillshade with curvature -> ridges/channels get punched up
    lit = soft_light(hs, curv_signal)

    # final composite: elevation colour, lit by the soft-light-enhanced shading
    # (plt.get_cmap()(...) always returns float64 RGBA regardless of input
    # dtype -- cast down since 8-bit-display colour values don't need it,
    # and leaving it would upcast the whole composite via soft_light() below)
    elev_rgb = plt.get_cmap(elev_cmap)(normalize(dem))[:, :, :3].astype(np.float32)
    lit_rgb = np.repeat(lit[:, :, None], 3, axis=2)
    composite = soft_light(elev_rgb, lit_rgb)

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    im0 = axes[0, 0].imshow(dem, cmap=elev_cmap)
    axes[0, 0].set_title("DEM (elevation)")
    plt.colorbar(im0, ax=axes[0, 0], shrink=0.7)

    im1 = axes[0, 1].imshow(profile, cmap=curv_cmap, vmin=-curv_vlim, vmax=curv_vlim)
    axes[0, 1].set_title("Profile curvature\n(+convex/decel, -concave/accel)")
    plt.colorbar(im1, ax=axes[0, 1], shrink=0.7)

    im2 = axes[0, 2].imshow(planform, cmap=curv_cmap, vmin=-curv_vlim, vmax=curv_vlim)
    axes[0, 2].set_title("Planform curvature\n(+ridges/divergent, -channels/convergent)")
    plt.colorbar(im2, ax=axes[0, 2], shrink=0.7)

    axes[1, 0].imshow(hs, cmap="gray")
    axes[1, 0].set_title("Standard hillshade")

    axes[1, 1].imshow(lit, cmap="gray")
    axes[1, 1].set_title("Soft-lit relief\n(hillshade \u2295 curvature)")

    axes[1, 2].imshow(composite)
    axes[1, 2].set_title("Elevation \u2295 soft-lit relief")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    if out_png:
        plt.savefig(out_png, dpi=150)
        print(f"Saved figure to {out_png}")
    if show:
        plt.show()

    results = {
        "profile": profile,
        "planform": planform,
        "hillshade": hs,
        "soft_lit": lit,
        "composite": composite,
    }
    return fig, axes, results

_COMPARE_SLIDER_TEMPLATE = """
__TITLE_HTML__
<div id="__WIDGET_ID__" style="position:relative; width:__WIDTH__px; max-width:100%; user-select:none;">
  <img src="__AFTER_URI__" style="display:block; width:100%; height:auto;">
  <div class="clip-wrap" style="position:absolute; top:0; left:0; width:50%; height:100%; overflow:hidden;">
    <img src="__BEFORE_URI__" style="display:block; width:__WIDTH__px; max-width:none; height:auto;">
  </div>
  <div class="handle" style="position:absolute; top:0; left:50%; width:2px; height:100%; background:white; box-shadow:0 0 4px rgba(0,0,0,0.6); cursor:ew-resize;">
    <div style="position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); width:32px; height:32px; border-radius:50%; background:white; box-shadow:0 0 6px rgba(0,0,0,0.5); display:flex; align-items:center; justify-content:center; font-family:sans-serif; font-size:14px;">&#8596;</div>
  </div>
  <div style="position:absolute; top:8px; left:8px; background:rgba(0,0,0,0.55); color:white; padding:2px 8px; border-radius:4px; font-family:sans-serif; font-size:12px; pointer-events:none;">__LABEL_LEFT__</div>
  <div style="position:absolute; top:8px; right:8px; background:rgba(0,0,0,0.55); color:white; padding:2px 8px; border-radius:4px; font-family:sans-serif; font-size:12px; pointer-events:none;">__LABEL_RIGHT__</div>
</div>
<input type="range" min="0" max="100" value="50" id="__WIDGET_ID___slider" style="width:__WIDTH__px; max-width:100%; margin-top:8px;">
<script>
(function() {
  const container = document.getElementById("__WIDGET_ID__");
  const clipWrap = container.querySelector(".clip-wrap");
  const handle = container.querySelector(".handle");
  const slider = document.getElementById("__WIDGET_ID___slider");
 
  function setPct(pct) {
    pct = Math.max(0, Math.min(100, pct));
    clipWrap.style.width = pct + "%";
    handle.style.left = pct + "%";
  }
 
  slider.addEventListener("input", () => setPct(slider.value));
 
  let dragging = false;
  handle.addEventListener("mousedown", (e) => { dragging = true; e.preventDefault(); });
  window.addEventListener("mouseup", () => dragging = false);
  container.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const rect = container.getBoundingClientRect();
    const pct = ((e.clientX - rect.left) / rect.width) * 100;
    setPct(pct);
    slider.value = pct;
  });
})();
</script>
"""
 
 
def compare_slider(before, after, labels=("Before", "After"), max_dim=1200, title=None):
    """Interactive before/after swipe-comparison slider for two
    same-shaped image arrays (e.g. two entries from
    TerraTexture.basemap.plot_dem_basemap_luminosity_relief's returned
    `layers` dict, like `layers['basemap']` vs `layers['final']`),
    rendered as a self-contained HTML widget via IPython.display.
 
    A draggable divider (or the range slider beneath it) reveals more of
    one image or the other -- the standard "before/after" pattern used
    by many mapping/photo-comparison tools. Rendered via HTML/CSS (a
    clipped, absolutely-positioned image pair) rather than redrawing a
    matplotlib figure on every slider move, which would be noticeably
    laggy for a large array -- each image is encoded to PNG exactly
    once, up front, then the browser handles the rest with no further
    Python involvement.
 
    Both arrays are downsampled for display first (same strided
    decimation as `TerraTexture.basemap`'s own interactive preview) --
    there's no reason to base64-embed more pixels than a browser can
    actually show in the notebook. This is for looking at, not for
    keeping; use the full-resolution arrays directly (e.g.
    `plt.imsave('out.png', layers['final'])`) for anything you want to
    save.
 
    Parameters
    ----------
    before, after : 2D or 3D arrays
        Same height/width. Float arrays are assumed to be in [0, 1]
        (clipped defensively) and converted to uint8; already-uint8
        arrays are used as-is. 2D (single-channel) arrays are shown as
        greyscale. Matches what TerraTexture.basemap's `layers` dict
        and TerraTexture.plotting's own `results` dict return.
    labels : (str, str)
        Labels shown at the top-left/top-right of the widget --
        corresponds to (before, after) regardless of which side the
        slider currently favours.
    max_dim : int
        Longest side, in pixels, of the (downsampled) images actually
        embedded in the widget.
    title : str or None
        Optional heading shown above the slider.
 
    Returns
    -------
    IPython.display.HTML -- Jupyter renders this automatically as the
    cell's output; wrap in `display(...)` if it's not the last
    expression in the cell.
    """
    import base64
    import io as _io
    import uuid
 
    from IPython.display import HTML
    from PIL import Image
 
    before = np.asarray(before)
    after = np.asarray(after)
    if before.shape[:2] != after.shape[:2]:
        raise ValueError(
            f"before/after must have matching height/width, got {before.shape} vs {after.shape}"
        )
 
    def _to_uint8_rgb(arr):
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        return arr
 
    def _downsample(arr, max_dim):
        h, w = arr.shape[:2]
        # ceiling division, not int() (which floors) -- int(4000/1200)=3
        # leaves a 1334px result, which VIOLATES the "at most max_dim"
        # contract this function promises; ceiling guarantees the result
        # never exceeds max_dim (verified in plotting tests).
        step = max(1, -(-max(h, w) // max_dim))
        return arr[::step, ::step]
 
    def _to_data_uri(arr):
        arr = _downsample(_to_uint8_rgb(arr), max_dim)
        buf = _io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}", arr.shape
 
    before_uri, shape = _to_data_uri(before)
    after_uri, _ = _to_data_uri(after)
    height, width = shape[:2]
 
    widget_id = f"compare_{uuid.uuid4().hex[:8]}"
    title_html = f'<h4 style="margin:0 0 8px 0; font-family:sans-serif;">{title}</h4>' if title else ""
 
    html = (
        _COMPARE_SLIDER_TEMPLATE
        .replace("__TITLE_HTML__", title_html)
        .replace("__WIDGET_ID__", widget_id)
        .replace("__WIDTH__", str(width))
        .replace("__BEFORE_URI__", before_uri)
        .replace("__AFTER_URI__", after_uri)
        .replace("__LABEL_LEFT__", str(labels[0]))
        .replace("__LABEL_RIGHT__", str(labels[1]))
    )
    return HTML(html)
