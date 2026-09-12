"""
Curvature and hillshade computation.

Computes profile curvature (rate of change of slope, along the direction
of steepest descent -- controls acceleration/deceleration of flow) and
planform curvature (curvature of contour lines, perpendicular to slope
direction -- controls flow convergence/divergence).

Formulas follow Zevenbergen & Thorne (1987), the standard used by
ArcGIS/QGIS/GRASS 3x3-window curvature tools:

    p = dz/dx, q = dz/dy
    r = d2z/dx2, t = d2z/dy2, s = d2z/dxdy

    profile curvature = -(r*p^2 + 2*s*p*q + t*q^2) / ((p^2+q^2)*(1+p^2+q^2)^1.5)
    planform curvature = -(r*q^2 - 2*s*p*q + t*p^2) / (p^2+q^2)^1.5

Sign convention: positive profile curvature = convex (flow decelerates),
negative = concave (flow accelerates). Positive planform = convex
(flow diverges, e.g. ridges), negative = concave (flow converges, e.g.
channels/valleys). Flat areas (p^2+q^2 -> 0) are set to zero.

Only depends on numpy/scipy -- no rasterio required. See docs/formulas.md
for the full derivation and a discussion of the gradient-of-gradient
approximation used here vs. the closed-form ZT 3x3 stencil.
"""

import numpy as np

from .io import _fill_nan_nearest


def _derivatives(dem, cellsize):
    zy, zx = np.gradient(dem, cellsize)       # first partials (p, q)
    zxy, zxx = np.gradient(zx, cellsize)       # d(zx)/dy, d(zx)/dx
    zyy, _ = np.gradient(zy, cellsize)         # d(zy)/dy
    return zx, zy, zxx, zyy, zxy               # p, q, r, t, s


def curvatures(dem, cellsize):
    """Return (profile_curvature, planform_curvature) arrays, same shape as
    dem. NaN cells in `dem` (nodata/voids) are nearest-filled for the
    calculation and set back to NaN in the output."""
    dem_filled, nan_mask = _fill_nan_nearest(np.asarray(dem, dtype=np.float32))

    p, q, r, t, s = _derivatives(dem_filled, cellsize)
    p2q2 = p ** 2 + q ** 2

    with np.errstate(divide="ignore", invalid="ignore"):
        profile = -(r * p ** 2 + 2 * s * p * q + t * q ** 2) / (p2q2 * (1 + p2q2) ** 1.5)
        planform = -(r * q ** 2 - 2 * s * p * q + t * p ** 2) / (p2q2 ** 1.5)

    # flat cells (p2q2 ~ 0) -> undefined -> set to 0
    flat = p2q2 < 1e-9
    profile = np.where(flat, 0.0, np.nan_to_num(profile, nan=0.0, posinf=0.0, neginf=0.0))
    planform = np.where(flat, 0.0, np.nan_to_num(planform, nan=0.0, posinf=0.0, neginf=0.0))

    profile = np.where(nan_mask, np.nan, profile)
    planform = np.where(nan_mask, np.nan, planform)
    return profile, planform


def hillshade(dem, cellsize, azimuth=315, altitude=45):
    dem_filled, nan_mask = _fill_nan_nearest(np.asarray(dem, dtype=np.float32))
    # np.radians() on a Python scalar always returns a *strong* float64
    # numpy scalar (not a weak Python float) -- left uncast, it would
    # silently upcast every float32 array it's later multiplied against.
    az = np.float32(np.radians(360.0 - azimuth + 90))
    alt = np.float32(np.radians(altitude))
    zy, zx = np.gradient(dem_filled, cellsize)
    slope = np.pi / 2 - np.arctan(np.hypot(zx, zy))
    aspect = np.arctan2(-zx, zy)
    shaded = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    shaded = np.clip(shaded, 0, 1)
    return np.where(nan_mask, np.nan, shaded)
