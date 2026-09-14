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
