//! Hillshade of a DEM, as a fused kernel.
//!
//! Mirrors `TerraTexture.derivatives`' `hillshade()` exactly, including
//! its `az = 360 - azimuth + 90` convention and `np.gradient`'s
//! `edge_order=1` boundary handling. Verified against `derivatives.py`
//! in `tests/test_derivatives_rust.py`.
//!
//! # NaN handling
//!
//! Same as [`curvature`](crate::curvature): the DEM must be NaN-free,
//! and the Python caller nan-fills before and re-masks after.
//!
//! # Formula
//!
//! The textbook form is
//!
//! ```text
//! slope  = π/2 - atan(hypot(zx, zy))
//! aspect = atan2(-zx, zy)
//! shaded = sin(alt)·sin(slope) + cos(alt)·cos(slope)·cos(az - aspect)
//! ```
//!
//! Expanding it with `sin(atan g) = g/√(1+g²)`, `cos(atan g) = 1/√(1+g²)`
//! and the angle-difference identity, the `g = hypot(zx, zy)` factor
//! cancels, leaving
//!
//! ```text
//! shaded = (sin(alt) + cos(alt)·(cos(az)·zy - sin(az)·zx)) / √(1 + zx² + zy²)
//! ```
//!
//! That is one square root per pixel and no inverse trig. Flat cells
//! (zx = zy = 0) need no special case, since the denominator is 1 rather
//! than the 0/0 that `hypot(0, 0)` would produce in the aspect term.

use ndarray::{Array2, ArrayView2, Zip};

use crate::common::{gradient2d, PARALLEL_THRESHOLD};

/// Hillshade for one cell.
///
/// Takes the cell's gradients `zx`, `zy` and the precomputed sines and
/// cosines of the (converted) azimuth and altitude, all `f32`. Returns
/// the illumination as an `f32` clamped to \[0, 1\]. See the module docs
/// for the formula.
#[inline]
fn hillshade_pixel(zx: f32, zy: f32, sin_az: f32, cos_az: f32, sin_alt: f32, cos_alt: f32) -> f32 {
    let denom = (1.0 + zx * zx + zy * zy).sqrt();
    let shaded = (sin_alt + cos_alt * (cos_az * zy - sin_az * zx)) / denom;
    shaded.clamp(0.0, 1.0)
}

/// Compute a hillshade (simulated illumination) of a DEM.
///
/// # Arguments
///
/// * `dem` - `ArrayView2<f32>`, shape (H, W): elevations. **Must be
///   NaN-free** (see module docs). Any memory layout. H and W should
///   each be at least 2; an axis of length 1 is treated as flat, where
///   numpy would raise.
/// * `cellsize` - `f32`: grid spacing, in the same units as the
///   elevations.
/// * `azimuth` - `f32`, degrees: direction the light comes from,
///   clockwise from north (e.g. `315.0` for north-west). Converted
///   internally with `derivatives.py`'s `az = 360 - azimuth + 90`.
/// * `altitude` - `f32`, degrees: height of the light above the horizon
///   (`0.0` = horizon, `90.0` = directly overhead).
/// * `out` - `&mut Array2<f32>`, shape (H, W): overwritten with the
///   hillshade, every element in \[0, 1\] (0 = fully shaded, 1 = fully lit).
///
/// The four trig values are computed once per call, not per pixel.
/// Runs serially below [`PARALLEL_THRESHOLD`]
/// elements and in parallel at or above it.
///
/// # Panics
///
/// * If `out` does not have the same shape as `dem`.
/// * If exactly one of H and W is 0.
///
/// # Example
///
/// ```
/// use ndarray::Array2;
/// use terra_texture_rs::hillshade_core;
///
/// // flat ground lit from directly overhead is fully lit
/// let dem = Array2::<f32>::zeros((4, 4));
/// let mut out = Array2::<f32>::zeros(dem.raw_dim());
/// hillshade_core(dem.view(), 1.0, 315.0, 90.0, &mut out);
///
/// assert!(out.iter().all(|&v| (v - 1.0).abs() < 1e-6));
/// ```
pub fn hillshade_core(dem: ArrayView2<f32>, cellsize: f32, azimuth: f32, altitude: f32, out: &mut Array2<f32>) {
    let (zy, zx) = gradient2d(dem, cellsize);
    let az = (360.0 - azimuth + 90.0).to_radians();
    let alt = altitude.to_radians();
    // Computed ONCE for the whole DEM, not per pixel: 4 trig calls total
    // instead of up to 2*H*W.
    let (sin_az, cos_az) = az.sin_cos();
    let (sin_alt, cos_alt) = alt.sin_cos();

    let n = dem.len();
    let combine = |o: &mut f32, &zx: &f32, &zy: &f32| {
        *o = hillshade_pixel(zx, zy, sin_az, cos_az, sin_alt, cos_alt);
    };
    let z = Zip::from(out).and(&zx).and(&zy);
    if n >= PARALLEL_THRESHOLD {
        z.par_for_each(combine);
    } else {
        z.for_each(combine);
    }
}
