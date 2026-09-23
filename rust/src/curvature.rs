//! Profile + planform curvature -- fused DEM-derivative kernel.
//!
//! Mirrors `TerraTexture.derivatives` exactly: same "gradient of
//! gradient" approximation (`np.gradient` applied twice), same
//! `edge_order=1` boundary handling (see `common::gradient2d`), same
//! Zevenbergen & Thorne curvature formulas. Verified bit-parity against
//! `derivatives.py` in `tests/test_derivatives_rust.py`.
//!
//! Callers are responsible for nan-filling the DEM first
//! (`derivatives.py`'s `_fill_nan_nearest`) and re-masking the NaN/void
//! cells afterwards -- Rust owns the plain numeric math on a clean
//! float32 array, Python owns the NaN bookkeeping.

use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

use crate::common::{gradient2d, PARALLEL_THRESHOLD};

#[inline]
fn curvature_pixel(p: f32, q: f32, r: f32, t: f32, s: f32) -> (f32, f32) {
    let p2q2 = p * p + q * q;
    if p2q2 < 1e-9 {
        return (0.0, 0.0); // flat cell -- matches derivatives.py's `flat` mask
    }
    let profile_raw = -(r * p * p + 2.0 * s * p * q + t * q * q) / (p2q2 * (1.0 + p2q2).powf(1.5));
    let planform_raw = -(r * q * q - 2.0 * s * p * q + t * p * p) / p2q2.powf(1.5);

    // matches np.nan_to_num(..., nan=0.0, posinf=0.0, neginf=0.0)
    let clean = |v: f32| if v.is_finite() { v } else { 0.0 };
    (clean(profile_raw), clean(planform_raw))
}

/// Pure computation: profile + planform curvature. `dem` must already be
/// NaN-free (caller nan-fills; see module docs). Matches
/// `derivatives.py`'s `curvatures()` (minus its NaN re-masking, which
/// stays the caller's job) exactly.
pub fn curvatures_core(
    dem: ArrayView2<f32>,
    cellsize: f32,
    profile_out: &mut Array2<f32>,
    planform_out: &mut Array2<f32>,
) {
    let (zy, zx) = gradient2d(dem, cellsize);
    let (zxy, zxx) = gradient2d(zx.view(), cellsize);
    let (zyy, _zyx) = gradient2d(zy.view(), cellsize); // zyx discarded, matches derivatives.py

    let n = dem.len();
    // 2 outputs + 5 inputs = 7 producers, one over ndarray::Zip's max
    // arity of 6 -- fall back to plain contiguous slices + rayon here
    // instead (all arrays are freshly `zeros()`-allocated, hence
    // standard/C-contiguous, so `.as_slice()` is always `Some`).
    let zx_s = zx.as_slice().expect("gradient2d output not contiguous");
    let zy_s = zy.as_slice().expect("gradient2d output not contiguous");
    let zxx_s = zxx.as_slice().expect("gradient2d output not contiguous");
    let zyy_s = zyy.as_slice().expect("gradient2d output not contiguous");
    let zxy_s = zxy.as_slice().expect("gradient2d output not contiguous");
    let profile_s = profile_out.as_slice_mut().expect("profile_out not contiguous");
    let planform_s = planform_out.as_slice_mut().expect("planform_out not contiguous");

    let compute = |i: usize, po: &mut f32, plo: &mut f32| {
        let (profile, planform) = curvature_pixel(zx_s[i], zy_s[i], zxx_s[i], zyy_s[i], zxy_s[i]);
        *po = profile;
        *plo = planform;
    };

    if n >= PARALLEL_THRESHOLD {
        profile_s
            .par_iter_mut()
            .zip(planform_s.par_iter_mut())
            .enumerate()
            .for_each(|(i, (po, plo))| compute(i, po, plo));
    } else {
        profile_s
            .iter_mut()
            .zip(planform_s.iter_mut())
            .enumerate()
            .for_each(|(i, (po, plo))| compute(i, po, plo));
    }
}
