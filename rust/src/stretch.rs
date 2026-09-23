//! Standard-deviation contrast stretch, as a fused single-pass kernel.
//!
//! Mirrors `TerraTexture.stretch`'s `stretch_std()`: values are mapped
//! linearly so that `mean - n_std·std` becomes 0 and `mean + n_std·std`
//! becomes 1, then clipped to \[0, 1\]. NaN in means NaN out, and NaNs
//! are excluded from the mean and standard deviation.
//!
//! # Why this is faster than numpy
//!
//! The numpy version calls `np.nanmean()` and then `np.nanstd()`, which
//! recomputes the mean internally, so the array is reduced several times
//! before the elementwise subtract/divide/clip. Here one reduction pass
//! collects the sum, sum of squares and count together. (Deduplicating
//! only the mean call in numpy gives about 1.1×; numpy's C reductions are
//! already efficient, so the win is fewer passes, not language speed.)
//!
//! # Precision
//!
//! Statistics are accumulated in `f64` for robustness on large arrays.
//! This deliberately does **not** bit-match numpy's `float32` pairwise
//! summation; it is verified against it within a statistical tolerance
//! in `tests/test_stretch_rust.py`, which is the right bar for a
//! mean/std computation.

use ndarray::{Array2, ArrayView2, Zip};
use rayon::prelude::*;

use crate::common::PARALLEL_THRESHOLD;

/// Stretch one value: `v` (`f32`, may be NaN), the lower bound `lo` and
/// the range `denom = hi - lo + 1e-12`. Returns `(v - lo) / denom`
/// clamped to \[0, 1\], or NaN if `v` is NaN (matching `np.clip`, which
/// propagates NaN).
#[inline]
fn stretch_pixel(v: f32, lo: f32, denom: f32) -> f32 {
    if v.is_nan() {
        f32::NAN // matches np.clip((arr - lo) / denom, 0, 1): NaN propagates, never clamped away
    } else {
        ((v - lo) / denom).clamp(0.0, 1.0)
    }
}

/// Single-threaded NaN-skipping reduction.
///
/// Returns `(sum, sum_of_squares, count)` as `(f64, f64, u64)`, over the
/// non-NaN elements of `arr` only.
fn sum_stats_serial(arr: ArrayView2<f32>) -> (f64, f64, u64) {
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    let mut count = 0u64;
    for &v in arr.iter() {
        if !v.is_nan() {
            let vd = v as f64;
            sum += vd;
            sumsq += vd * vd;
            count += 1;
        }
    }
    (sum, sumsq, count)
}

/// Parallel version of [`sum_stats_serial`], same return type.
///
/// Uses rayon's fold + reduce over the array's flat slice, which needs a
/// standard-layout (C-contiguous) array. If `arr` isn't one, falls back
/// to the serial path rather than panicking. In practice it always is,
/// because `stretch.py`'s dispatch passes an `ascontiguousarray()`'d
/// array.
fn sum_stats_parallel(arr: ArrayView2<f32>) -> (f64, f64, u64) {
    // rayon's fold+reduce needs a flat parallel iterator; arrays here are
    // standard/C-contiguous (stretch.py's dispatch passes a
    // `.ascontiguousarray()`'d array), so `.as_slice()` is reliably
    // `Some`. Falls back to the serial path in the (should-be-unreachable
    // in practice) case it isn't, rather than panicking.
    match arr.as_slice() {
        Some(slice) => slice
            .par_iter()
            .fold(
                || (0.0f64, 0.0f64, 0u64),
                |(s, sq, c), &v| {
                    if v.is_nan() {
                        (s, sq, c)
                    } else {
                        let vd = v as f64;
                        (s + vd, sq + vd * vd, c + 1)
                    }
                },
            )
            .reduce(
                || (0.0f64, 0.0f64, 0u64),
                |(s1, sq1, c1), (s2, sq2, c2)| (s1 + s2, sq1 + sq2, c1 + c2),
            ),
        None => sum_stats_serial(arr),
    }
}

/// Contrast-stretch an array to \[0, 1\] using mean ± `n_std` standard
/// deviations, ignoring NaNs.
///
/// # Arguments
///
/// * `arr` - `ArrayView2<f32>`, shape (H, W): input values, any range.
///   NaN is allowed and marks missing data. Any memory layout, but the
///   parallel reduction is only used on C-contiguous input.
/// * `n_std` - `f32`: half-width of the stretch window in standard
///   deviations (e.g. `2.0` maps mean − 2σ → 0 and mean + 2σ → 1).
/// * `out` - `&mut Array2<f32>`, shape (H, W): overwritten with the
///   stretched values in \[0, 1\], and NaN wherever `arr` is NaN.
///
/// Uses the population standard deviation (numpy's default `ddof=0`).
/// If every element is NaN, or the array is empty, mean and std are
/// taken as 0.
///
/// A constant array has std 0, so the window collapses to a point and
/// only the `1e-12` guard keeps the division finite. Every value then
/// equals the mean, so every output is 0, as in the numpy version.
///
/// Runs serially below [`PARALLEL_THRESHOLD`]
/// elements and in parallel at or above it.
///
/// # Panics
///
/// If `out` does not have the same shape as `arr`.
///
/// # Example
///
/// ```
/// use ndarray::{array, Array2};
/// use terra_texture_rs::stretch_std_core;
///
/// let arr = array![[1.0_f32, 2.0], [3.0, f32::NAN]];
/// let mut out = Array2::<f32>::zeros(arr.raw_dim());
/// stretch_std_core(arr.view(), 1.0, &mut out);
///
/// assert!((out[[0, 1]] - 0.5).abs() < 1e-6); // the mean maps to 0.5
/// assert!(out[[1, 1]].is_nan());             // NaN passes through
/// ```
pub fn stretch_std_core(arr: ArrayView2<f32>, n_std: f32, out: &mut Array2<f32>) {
    let n = arr.len();
    let (sum, sumsq, count) = if n >= PARALLEL_THRESHOLD {
        sum_stats_parallel(arr)
    } else {
        sum_stats_serial(arr)
    };

    let mean_f64 = if count > 0 { sum / count as f64 } else { 0.0 };
    let variance_f64 = if count > 0 {
        (sumsq / count as f64 - mean_f64 * mean_f64).max(0.0) // guard tiny negative from float error
    } else {
        0.0
    };
    let mean = mean_f64 as f32;
    let std = variance_f64.sqrt() as f32;
    let lo = mean - n_std * std;
    let hi = mean + n_std * std;
    let denom = hi - lo + 1e-12;

    let combine = |o: &mut f32, &v: &f32| *o = stretch_pixel(v, lo, denom);
    let z = Zip::from(out).and(&arr);
    if n >= PARALLEL_THRESHOLD {
        z.par_for_each(combine);
    } else {
        z.for_each(combine);
    }
}
