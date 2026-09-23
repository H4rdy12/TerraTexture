//! `stretch_std` -- fused single-pass mean/std reduction + clip-stretch.
//!
//! Mirrors `TerraTexture.stretch`'s `stretch_std()` exactly: NaN in ->
//! NaN out; mean +/- n_std*std clipped to [0,1]. The numpy version calls
//! `np.nanmean()` then `np.nanstd()` separately -- nanstd recomputes its
//! own mean internally, so the array's mean ends up computed twice, plus
//! a separate variance pass, plus the elementwise subtract/divide/clip
//! (verified: deduplicating just the mean call gives only ~1.1x, numpy's
//! C reductions are already efficient -- the real win here is doing ONE
//! reduction pass (sum, sum-of-squares, count together) instead of
//! several, not language speed per se).
//!
//! Accumulates in f64 for numerical robustness on large arrays -- this
//! is NOT intended to bit-match numpy's own (float32, pairwise
//! summation) internal accumulation, and isn't expected to; verified
//! against it within a statistical tolerance instead (see
//! `tests/test_stretch_rust.py`), which is the correct bar for a
//! mean/std computation, not exact equality.

use ndarray::{Array2, ArrayView2, Zip};
use rayon::prelude::*;

use crate::common::PARALLEL_THRESHOLD;

#[inline]
fn stretch_pixel(v: f32, lo: f32, denom: f32) -> f32 {
    if v.is_nan() {
        f32::NAN // matches np.clip((arr - lo) / denom, 0, 1): NaN propagates, never clamped away
    } else {
        ((v - lo) / denom).clamp(0.0, 1.0)
    }
}

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

/// Pure computation: mean +/- n_std*std stretch to [0,1], NaN-safe.
/// Matches `stretch.py`'s `stretch_std()` within float tolerance (see
/// module docs).
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