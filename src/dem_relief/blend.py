"""
Photoshop/SVG-spec blend modes: soft light and luminosity.

This module is pure numpy array math with zero geospatial dependencies
(no rasterio, no matplotlib) -- it's deliberately kept dependency-free so
it can be unit-tested in isolation, and so it's the natural place to drop
in a Rust/Numba-accelerated implementation later without touching
anything else in the package (see docs/formulas.md for background on the
soft-light and luminosity blend formulas).
"""

import numpy as np


def soft_light(base, blend):
    """Photoshop-style soft light blend, base & blend arrays in [0, 1]."""
    a, b = base, blend
    return np.clip(
        np.where(
            b <= 0.5,
            2 * a * b + a ** 2 * (1 - 2 * b),
            2 * a * (1 - b) + np.sqrt(np.clip(a, 0, 1)) * (2 * b - 1),
        ),
        0, 1,
    )


def _lum(rgb):
    """Photoshop/SVG-spec luminosity of an (..., 3) RGB array in [0, 1]."""
    return 0.3 * rgb[..., 0] + 0.59 * rgb[..., 1] + 0.11 * rgb[..., 2]


def _clip_color(rgb):
    """SVG compositing spec ClipColor(): pull an RGB triple back into
    [0, 1] gamut around its own luminosity, rather than a naive per-channel
    clip (which would shift hue/saturation)."""
    lum = _lum(rgb)[..., None]
    n = rgb.min(axis=-1, keepdims=True)
    x = rgb.max(axis=-1, keepdims=True)
    rgb = np.where(n < 0, lum + (rgb - lum) * lum / (lum - n + 1e-12), rgb)
    rgb = np.where(x > 1, lum + (rgb - lum) * (1 - lum) / (x - lum + 1e-12), rgb)
    return np.clip(rgb, 0, 1)


def luminosity_blend(backdrop_rgb, luminosity):
    """SVG/Photoshop 'Luminosity' blend mode: SetLum(backdrop, Lum(source)).
    Keeps backdrop_rgb's hue AND saturation intact, replacing only its
    lightness with `luminosity` (H, W) array in [0, 1]. This is the
    correct operation for draping a greyscale relief over colour imagery
    -- unlike per-channel soft-light, it can't desaturate or shift hue.
    """
    d = luminosity[..., None] - _lum(backdrop_rgb)[..., None]
    return _clip_color(backdrop_rgb + d)