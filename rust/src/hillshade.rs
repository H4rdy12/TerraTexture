//! Hillshade -- fused DEM-derivative kernel.
//!
//! Mirrors `TerraTexture.derivatives`' `hillshade()` exactly, including
//! its `az = 360 - azimuth + 90` convention and `np.gradient`'s
//! `edge_order=1` boundary handling (see `common::gradient2d`). Verified
//! against `derivatives.py` in `tests/test_derivatives_rust.py`.
//!
//! Callers are responsible for nan-filling the DEM first and re-masking
//! NaN/void cells afterwards -- same division of labour as `curvature`.

use ndarray::{Array2, ArrayView2, Zip};

use crate::common::{gradient2d, PARALLEL_THRESHOLD};

#[inline]
fn hillshade_pixel(zx: f32, zy: f32, sin_az: f32, cos_az: f32, sin_alt: f32, cos_alt: f32) -> f32 {
    // Algebraic expansion of the original
    //   slope = pi/2 - atan(hypot(zx, zy))
    //   aspect = atan2(-zx, zy)
    //   shaded = sin(alt)*sin(slope) + cos(alt)*cos(slope)*cos(az - aspect)
    // using sin(atan(g)) = g/sqrt(1+g^2), cos(atan(g)) = 1/sqrt(1+g^2),
    // and cos(az-aspect) = cos(az)cos(aspect) + sin(az)sin(aspect) with
    // sin(aspect) = -zx/g, cos(aspect) = zy/g (g = hypot(zx, zy)). The
    // factor of `g` cancels completely, leaving one sqrt and no
    // atan/atan2/sin(slope)/cos(slope) per pixel at all -- verified
    // bit-for-bit (float32 tolerance) against derivatives.py's original
    // formula in tests/test_derivatives_rust.py, including the flat
    // (zx=zy=0) case, which needs no special-casing here since
    // sqrt(1+0+0)=1 rather than a 0/0 from hypot(0,0).
    let denom = (1.0 + zx * zx + zy * zy).sqrt();
    let shaded = (sin_alt + cos_alt * (cos_az * zy - sin_az * zx)) / denom;
    shaded.clamp(0.0, 1.0)
}

/// Pure computation: hillshade. `dem` must already be NaN-free (see
/// module docs). `azimuth`/`altitude` in degrees, matching
/// `derivatives.py`'s `hillshade()` signature exactly.
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