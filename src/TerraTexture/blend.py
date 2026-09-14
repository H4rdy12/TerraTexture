"""
Photoshop/SVG-spec blend modes: soft light and luminosity.

This module is pure numpy array math with zero geospatial dependencies
(no rasterio, no matplotlib) -- it's deliberately kept dependency-free so
it can be unit-tested in isolation. See `docs/formulas.md` for background
on the soft-light and luminosity blend formulas.

## Optional Rust acceleration

If the compiled `terra_texture_rs` extension (see `rust/`) is importable,
`soft_light()` and `luminosity_blend()` dispatch to its fused kernels for
plain contiguous float32 arrays of the expected shape -- same numerical
result, computed in one pass per pixel instead of numpy's several
temporary-array-allocating passes. Falls back to the pure-numpy
implementation whenever the extension isn't built, the platform has no
prebuilt wheel, or the inputs don't match the fast path's requirements
(wrong dtype, non-contiguous, mismatched shape). That fallback is
**load-bearing, not incidental**: this package's stated design goal is
that `TerraTexture.derivatives`/`TerraTexture.blend` have zero hard
dependencies beyond numpy/scipy, and nobody should have to install a
Rust toolchain just to run curvature analysis.

Build the extension with `maturin develop` from `rust/` (see that
directory's README/Cargo.toml comments) to get the accelerated path;
`tests/test_blend.py` has Rust-vs-numpy parity tests that only run (via
`pytest.importorskip`) when it's actually built.
"""

import numpy as np

try:
    import terra_texture_rs as _rust
except ImportError:
    _rust = None


def _is_fast_path_soft_light(a, b):
    return (
        _rust is not None
        and isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
        and a.dtype == np.float32 and b.dtype == np.float32
        and a.ndim == 2 and a.shape == b.shape
    )


def _soft_light_numpy(a, b):
    return np.clip(
        np.where(
            b <= 0.5,
            2 * a * b + a ** 2 * (1 - 2 * b),
            2 * a * (1 - b) + np.sqrt(np.clip(a, 0, 1)) * (2 * b - 1),
        ),
        0, 1,
    )


def soft_light(base, blend):
    """Photoshop-style soft light blend, base & blend arrays in [0, 1].

    Dispatches to the Rust kernel (see module docstring) when available
    and the inputs are plain float32 2D arrays of matching shape;
    otherwise uses the pure-numpy implementation below. Same result
    either way."""
    a, b = base, blend
    if _is_fast_path_soft_light(a, b):
        return _rust.soft_light(np.ascontiguousarray(a), np.ascontiguousarray(b))
    return _soft_light_numpy(a, b)


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


def _is_fast_path_luminosity_blend(backdrop_rgb, luminosity):
    return (
        _rust is not None
        and isinstance(backdrop_rgb, np.ndarray) and isinstance(luminosity, np.ndarray)
        and backdrop_rgb.dtype == np.float32 and luminosity.dtype == np.float32
        and backdrop_rgb.ndim == 3 and backdrop_rgb.shape[-1] == 3
        and luminosity.shape == backdrop_rgb.shape[:2]
    )


def _luminosity_blend_numpy(backdrop_rgb, luminosity):
    d = luminosity[..., None] - _lum(backdrop_rgb)[..., None]
    return _clip_color(backdrop_rgb + d)


def luminosity_blend(backdrop_rgb, luminosity):
    """SVG/Photoshop 'Luminosity' blend mode: SetLum(backdrop, Lum(source)).
    Keeps backdrop_rgb's hue AND saturation intact, replacing only its
    lightness with `luminosity` (H, W) array in [0, 1]. This is the
    correct operation for draping a greyscale relief over colour imagery
    -- unlike per-channel soft-light, it can't desaturate or shift hue.

    Dispatches to the Rust kernel (see module docstring) when available
    and the inputs are plain float32 arrays of the expected shape --
    (H, W, 3) for backdrop_rgb, (H, W) for luminosity; otherwise uses
    the pure-numpy implementation below. Same result either way.
    """
    if _is_fast_path_luminosity_blend(backdrop_rgb, luminosity):
        return _rust.luminosity_blend(
            np.ascontiguousarray(backdrop_rgb), np.ascontiguousarray(luminosity),
        )
    return _luminosity_blend_numpy(backdrop_rgb, luminosity)
