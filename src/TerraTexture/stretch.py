"""
Array stretching / normalization and NaN-safe resampling helpers.

Only depends on numpy/scipy -- no rasterio required.
"""

import numpy as np
from scipy.ndimage import zoom

from .io import _fill_nan_nearest


def normalize(arr, low=1, high=99):
    """Percentile-stretch an array to [0, 1] (robust to outliers). NaNs
    (nodata) are ignored for the percentile calc and pass through as NaN."""
    lo, hi = np.nanpercentile(arr, [low, high])
    return np.clip((arr - lo) / (hi - lo + 1e-12), 0, 1)


def stretch_std(arr, n_std=4):
    """Mean +/- n_std*std stretch to [0, 1] -- the "stretched N standard
    deviations" symbology used for curvature/hillshade in ArcGIS Pro.
    NaNs are ignored for the stats and pass through as NaN."""
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    lo, hi = mean - n_std * std, mean + n_std * std
    return np.clip((arr - lo) / (hi - lo + 1e-12), 0, 1)


def bilinear_resample(arr, target_shape):
    """Bilinear-resample a 2D array to `target_shape` (rows, cols), NaN-safe:
    nodata is nearest-filled before resampling and the (also-resampled)
    nodata mask is re-applied afterwards so voids don't bleed or vanish."""
    arr = np.asarray(arr, dtype=float)
    filled, nan_mask = _fill_nan_nearest(arr)
    zy, zx = target_shape[0] / arr.shape[0], target_shape[1] / arr.shape[1]
    resampled = zoom(filled, (zy, zx), order=1)
    if nan_mask.any():
        mask_resampled = zoom(nan_mask.astype(float), (zy, zx), order=0) > 0.5
        resampled = np.where(mask_resampled, np.nan, resampled)
    return resampled
