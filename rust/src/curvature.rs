//! Profile and planform curvature of a DEM, as a fused kernel.
//!
//! Mirrors `TerraTexture.derivatives`' `curvatures()` exactly: the same
//! "gradient of gradient" approximation (`np.gradient` applied twice),
//! the same `edge_order=1` boundary handling, and the same Zevenbergen &
//! Thorne curvature formulas. Verified bit-parity against
//! `derivatives.py` in `tests/test_derivatives_rust.py`.
//!
//! # NaN handling
//!
//! The DEM passed in must be NaN-free. The Python caller nan-fills it
//! first (`derivatives.py`'s `_fill_nan_nearest`) and re-masks the
//! NaN/void cells in the outputs afterwards. Rust owns the plain numeric
//! math on a clean `f32` array; Python owns the NaN bookkeeping.

use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

use crate::common::{gradient2d, PARALLEL_THRESHOLD};

/// Profile and planform curvature for one cell.
///
/// Arguments are the cell's partial derivatives, all `f32`:
/// `p` = dz/dx, `q` = dz/dy, `r` = d²z/dx², `t` = d²z/dy²,
/// `s` = d²z/dxdy.
///
/// Returns `(profile, planform)` as `f32`. Flat cells
/// (`p² + q² < 1e-9`) return `(0.0, 0.0)`, and any non-finite result
/// becomes `0.0`, matching `np.nan_to_num(..., nan=0, posinf=0, neginf=0)`.
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

/// Compute profile and planform curvature of a DEM.
///
/// Profile curvature is curvature in the direction of steepest slope
/// (affects flow acceleration); planform curvature is curvature
/// perpendicular to it (affects flow convergence). Both use the
/// Zevenbergen & Thorne sign convention of `derivatives.py`.
///
/// # Arguments
///
/// * `dem` - `ArrayView2<f32>`, shape (H, W): elevations. **Must be
///   NaN-free** (see module docs). Any memory layout. H and W should
///   each be at least 2; see **Edge cases**.
/// * `cellsize` - `f32`: grid spacing, in the same units as the
///   elevations, applied to both axes.
/// * `profile_out` - `&mut Array2<f32>`, shape (H, W), C-contiguous:
///   overwritten with profile curvature.
/// * `planform_out` - `&mut Array2<f32>`, shape (H, W), C-contiguous:
///   overwritten with planform curvature.
///
/// Flat cells and any non-finite results are written as `0.0`.
///
/// Runs serially below [`PARALLEL_THRESHOLD`]
/// elements and in parallel at or above it. The gradient passes
/// themselves always run serially.
///
/// # Edge cases
///
/// Where numpy's `np.gradient` would raise `ValueError` because an axis
/// has length 1, this treats that axis's derivatives as zero instead.
///
/// # Panics
///
/// * If `profile_out` or `planform_out` is not C-contiguous. Arrays made
///   with `Array2::zeros(shape)` always are.
/// * If `profile_out` or `planform_out` has more elements than `dem`.
/// * If exactly one of H and W is 0.
///
/// Output shapes are not otherwise checked: outputs with fewer elements
/// than `dem` are partly filled, and a different shape with the same
/// element count is filled in the wrong layout. Always allocate both
/// outputs with `dem.raw_dim()`.
///
/// # Example
///
/// ```
/// use ndarray::Array2;
/// use terra_texture_rs::curvatures_core;
///
/// // a uniformly tilted plane has no curvature anywhere
/// let dem = Array2::from_shape_fn((5, 5), |(i, j)| (i + 2 * j) as f32);
/// let mut profile = Array2::<f32>::zeros(dem.raw_dim());
/// let mut planform = Array2::<f32>::zeros(dem.raw_dim());
/// curvatures_core(dem.view(), 1.0, &mut profile, &mut planform);
///
/// assert!(profile.iter().all(|&v| v.abs() < 1e-6));
/// assert!(planform.iter().all(|&v| v.abs() < 1e-6));
/// ```
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
