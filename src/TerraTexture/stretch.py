"""
Array stretching / normalization and NaN-safe resampling helpers.

Only depends on numpy/scipy -- no rasterio required.

## Optional Rust acceleration

Same contract as `TerraTexture.blend`/`TerraTexture.derivatives`: if the
compiled `terra_texture_rs` extension is importable, `stretch_std()`
dispatches to its fused kernel for plain contiguous float32 2D arrays --
one reduction pass computing sum/sum-of-squares/count together (rather
than numpy's separate nanmean + nanstd calls, where nanstd recomputes
its own mean internally), then one fused elementwise pass for the
clip-stretch. Falls back to the pure-numpy implementation below
otherwise, which itself avoids the same redundant-mean call (computing
variance from an already-known mean rather than calling `np.nanstd()`
separately) -- a real, free win either way, verified in
`tests/test_stretch_rust.py`.

`normalize()` (percentile-based) is NOT Rust-accelerated -- computing a
percentile fundamentally needs sorting/selection, not just a reduction,
and numpy's underlying sort is already a well-optimized C routine; the
same fused-single-pass trick that helps `stretch_std()` doesn't apply
there in the same way.
"""

import numpy as np
from scipy.ndimage import zoom

from .io import _fill_nan_nearest

try:
    import terra_texture_rs as _rust
except ImportError:
    _rust = None


def _is_fast_path_stretch_std(arr):
    return (
        _rust is not None
        and isinstance(arr, np.ndarray)
        and arr.dtype == np.float32
        and arr.ndim == 2
    )


def normalize(arr, low=1, high=99):
    """Percentile-stretch an array to [0, 1] (robust to outliers). NaNs
    (nodata) are ignored for the percentile calc and pass through as NaN.

    Preserves `arr`'s own dtype: `np.nanpercentile` always computes
    internally in float64 and returns float64 scalars regardless of
    input dtype, which would otherwise silently upcast a float32 input
    to float64 in the subtraction below (unlike a plain Python float,
    a numpy float64 scalar is NOT exempt from promotion under NEP 50)."""
    arr = np.asarray(arr)
    lo, hi = np.nanpercentile(arr, [low, high]).astype(arr.dtype)
    return np.clip((arr - lo) / (hi - lo + 1e-12), 0, 1)


def stretch_std(arr, n_std=4):
    """Mean +/- n_std*std stretch to [0, 1] -- the "stretched N standard
    deviations" symbology used for curvature/hillshade in ArcGIS Pro.
    NaNs are ignored for the stats and pass through as NaN.

    Dispatches to the Rust kernel (see module docstring) when available
    and `arr` is a plain float32 2D array; otherwise uses the pure-numpy
    implementation below. Same result either way (within float
    tolerance -- the Rust kernel accumulates its reduction in f64 for
    numerical robustness, which is not expected to bit-match numpy's
    own float32 accumulation, only to agree with it statistically)."""
    if _is_fast_path_stretch_std(arr):
        return _rust.stretch_std(np.ascontiguousarray(arr), float(n_std))

    mean = np.nanmean(arr)
    # variance from the already-computed mean, instead of calling
    # np.nanstd() separately (which recomputes its own mean internally)
    # -- avoids one redundant full-array reduction pass
    variance = np.nanmean((arr - mean) ** 2)
    std = np.sqrt(variance)
    lo, hi = mean - n_std * std, mean + n_std * std
    return np.clip((arr - lo) / (hi - lo + 1e-12), 0, 1)


def bilinear_resample(arr, target_shape):
    """Bilinear-resample a 2D array to `target_shape` (rows, cols), NaN-safe:
    nodata is nearest-filled before resampling and the (also-resampled)
    nodata mask is re-applied afterwards so voids don't bleed or vanish."""
    arr = np.asarray(arr, dtype=np.float32)
    filled, nan_mask = _fill_nan_nearest(arr)
    zy, zx = target_shape[0] / arr.shape[0], target_shape[1] / arr.shape[1]
    resampled = zoom(filled, (zy, zx), order=1)
    if nan_mask.any():
        mask_resampled = zoom(nan_mask.astype(np.float32), (zy, zx), order=0) > 0.5
        resampled = np.where(mask_resampled, np.nan, resampled)
    return resampled
