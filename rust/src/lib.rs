//! Fused elementwise kernels for TerraTexture, accelerating the
//! pure-numpy implementations in `terra_texture.blend`,
//! `terra_texture.derivatives` and `terra_texture.stretch` without
//! changing their public behaviour.
//!
//! The Python package tries to import this compiled extension and falls
//! back to its numpy implementation if the import fails (unbuilt,
//! unsupported platform, or a plain `pip install` without the compiled
//! wheel). That fallback is load-bearing, not incidental: the package's
//! stated design goal is that `terra_texture.derivatives` and
//! `terra_texture.blend` have zero hard dependencies beyond numpy/scipy.
//!
//! # Kernels
//!
//! | Module | Core function(s) | Input | Output |
//! |---|---|---|---|
//! | [`soft_light`] | [`soft_light_core`], [`soft_light_rgb_core`] | two `f32` arrays, same shape, (H, W) or (H, W, C) | same shape |
//! | [`luminosity_blend`] | [`luminosity_blend_core`] | `f32` (H, W, 3) + `f32` (H, W) | `f32` (H, W, 3) |
//! | [`curvature`] | [`curvatures_core`] | `f32` DEM (H, W), NaN-free | two `f32` (H, W) |
//! | [`hillshade`] | [`hillshade_core`] | `f32` DEM (H, W), NaN-free | `f32` (H, W) in \[0, 1\] |
//! | [`stretch`] | [`stretch_std_core`] | `f32` (H, W), NaN allowed | `f32` (H, W) in \[0, 1\] or NaN |
//!
//! All arrays are `f32` throughout, matching the `float32` arrays the
//! Python side passes in.
//!
//! # Structure
//!
//! Each kernel is split into two layers:
//! - a `*_core` function in its kernel module: pure [`ndarray`] in, pure
//!   `ndarray` out, no PyO3 types anywhere. This is what
//!   `benches/blend_bench.rs` and any `#[test]`s call directly, with no
//!   Python interpreter needed.
//! - a `#[pyfunction]` wrapper in the private `python` module: unwraps
//!   the numpy arrays into `ndarray` views, calls the core function, and
//!   wraps the result back up. `python` is the only module that imports
//!   `pyo3` or `numpy`.
//!
//! This split is why `[lib] crate-type` includes `"rlib"` alongside the
//! `"cdylib"` Python needs: an rlib is what `cargo bench`/`cargo test`
//! link against to call the core functions in-process.
//!
//! The public kernel functions are re-exported at the crate root, so
//! `use terra_texture_rs::soft_light_serial;`-style imports (e.g. in
//! `benches/blend_bench.rs`) work alongside the full module paths.
//!
//! # Output buffers
//!
//! Every core function writes into a caller-allocated `out` array rather
//! than returning a new one, so the Python wrappers can allocate once
//! and hand the buffer straight back to numpy. `out` must have the shape
//! stated in each function's docs; mismatched shapes panic (see each
//! function's **Panics** section).
//!
//! # The serial/parallel threshold
//!
//! Rayon's parallel dispatch has fixed per-call overhead (splitting work,
//! synchronizing threads) that only pays for itself once there's enough
//! work per thread to amortize it. Below [`PARALLEL_THRESHOLD`] elements,
//! kernels run a plain serial loop instead. The threshold is a
//! **provisional placeholder**, not a measured value; see its docs.

mod common;
pub mod curvature;
pub mod hillshade;
pub mod luminosity_blend;
mod python;
pub mod soft_light;
pub mod stretch;

pub use common::PARALLEL_THRESHOLD;
pub use curvature::curvatures_core;
pub use hillshade::hillshade_core;
pub use luminosity_blend::{luminosity_blend_core, luminosity_blend_parallel, luminosity_blend_serial};
pub use soft_light::{
    soft_light_core, soft_light_parallel, soft_light_rgb_core, soft_light_rgb_parallel, soft_light_rgb_serial,
    soft_light_serial,
};
pub use stretch::stretch_std_core;

// //! Fused elementwise kernels for `terra_texture.blend`, accelerating the
// //! pure-numpy implementations in `blend.py` without changing their
// //! public behaviour.
// //!
// //! `blend.py` tries to import this compiled extension and falls back to
// //! its numpy implementation if the import fails (unbuilt, unsupported
// //! platform, or a plain `pip install` without the compiled wheel) --
// //! that fallback is load-bearing, not incidental: the package's stated
// //! design goal is that `terra_texture.derivatives`/`terra_texture.blend`
// //! have zero hard dependencies beyond numpy/scipy.
// //!
// //! ## Structure
// //!
// //! Each kernel is split into two layers:
// //! - a `*_core` function: pure `ndarray` in, pure `ndarray` out, no PyO3
// //!   types anywhere. This is what `benches/blend_bench.rs` and any future
// //!   `#[test]`s call directly -- no Python interpreter needed to run them.
// //! - a `#[pyfunction]` wrapper: unwraps the numpy arrays into `ndarray`
// //!   views, calls the core function, wraps the result back up. This is
// //!   the only layer that knows about Python at all.
// //!
// //! This split is why `[lib] crate-type` includes `"rlib"` alongside the
// //! `"cdylib"` Python needs -- an rlib is what `cargo bench`/`cargo test`
// //! link against to call the core functions in-process.
// //!
// //! ## The serial/parallel threshold
// //!
// //! Rayon's parallel dispatch has fixed per-call overhead (splitting work,
// //! synchronizing threads) that only pays for itself once there's enough
// //! work per thread to amortize it. Below `PARALLEL_THRESHOLD` elements,
// //! kernels run a plain serial loop instead of `par_for_each`. The
// //! threshold below is a **provisional placeholder**, not a measured
// //! value -- run `cargo bench` (see `benches/blend_bench.rs`) to find the
// //! actual crossover point on real hardware and update it here.

// use ndarray::{Array2, Array3, ArrayView2, ArrayView3, Zip};
// use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
// use pyo3::prelude::*;
// use rayon::prelude::*;

// // TODO(benchmark): this is a guess, not a measurement. Run `cargo bench`
// // and replace it with whatever `blend_bench.rs` actually finds as the
// // point where `par_for_each` starts winning over a serial loop on the
// // target hardware. 256x256 = 65_536 is picked only because it "feels"
// // like a plausible order of magnitude for rayon's dispatch overhead to
// // have been amortized -- treat it as unverified until benchmarked.
// pub const PARALLEL_THRESHOLD: usize = 65_536;

// #[inline]
// fn soft_light_pixel(a: f32, b: f32) -> f32 {
//     let v = if b <= 0.5 {
//         2.0 * a * b + a * a * (1.0 - 2.0 * b)
//     } else {
//         // matches Python's np.sqrt(np.clip(a, 0, 1)) -- clamped to BOTH
//         // 0 and 1 before the sqrt, not just floored at 0.
//         2.0 * a * (1.0 - b) + a.clamp(0.0, 1.0).sqrt() * (2.0 * b - 1.0)
//     };
//     v.clamp(0.0, 1.0)
// }

// /// Serial (single-threaded) soft light -- exposed publicly alongside
// /// `soft_light_parallel` purely so `benches/blend_bench.rs` can compare
// /// them directly at every array size, independent of whatever
// /// `PARALLEL_THRESHOLD` currently guesses. `soft_light_core()` below is
// /// the actual dispatch entry point everything else should call.
// pub fn soft_light_serial(a: ArrayView2<f32>, b: ArrayView2<f32>, out: &mut Array2<f32>) {
//     Zip::from(out)
//         .and(&a)
//         .and(&b)
//         .for_each(|o, &a, &b| *o = soft_light_pixel(a, b));
// }

// /// Parallel (rayon) soft light -- see `soft_light_serial`'s doc comment.
// pub fn soft_light_parallel(a: ArrayView2<f32>, b: ArrayView2<f32>, out: &mut Array2<f32>) {
//     Zip::from(out)
//         .and(&a)
//         .and(&b)
//         .par_for_each(|o, &a, &b| *o = soft_light_pixel(a, b));
// }

// /// Pure computation: Photoshop-style soft light blend. `base`/`blend`
// /// arrays in [0, 1], same shape. Matches `blend.py`'s `soft_light()`
// /// exactly (verified against it in `tests/test_blend.py`'s Rust-parity
// /// tests, when the extension is built). Dispatches to the serial or
// /// parallel path based on `PARALLEL_THRESHOLD` -- this is the function
// /// the `#[pyfunction]` wrapper and any other caller should use.
// pub fn soft_light_core(a: ArrayView2<f32>, b: ArrayView2<f32>, out: &mut Array2<f32>) {
//     if a.len() >= PARALLEL_THRESHOLD {
//         soft_light_parallel(a, b, out);
//     } else {
//         soft_light_serial(a, b, out);
//     }
// }

// /// Same as `soft_light_core` but over 3-D (H, W, C) arrays -- e.g. RGB
// /// or RGBA imagery -- in ONE pass instead of the Python-side dispatch
// /// calling the 2-D kernel once per channel. `soft_light_pixel` is
// /// already channel-agnostic (a plain f32 -> f32 elementwise formula
// /// with no cross-channel interaction), so this is the exact same
// /// per-element math as the 2-D path; the only reason this exists as a
// /// separate kernel rather than just reshaping and reusing the 2-D one
// /// is to avoid the per-channel `np.ascontiguousarray()` copy the
// /// Python-side loop needed (each `a[..., c]` channel slice of a
// /// contiguous (H, W, C) array is itself non-contiguous). Operating on
// /// the whole (H, W, C) buffer directly, already contiguous, skips that
// /// entirely -- one input read, one output write, per element, with no
// /// intermediate per-channel arrays at all.
// pub fn soft_light_rgb_serial(a: ArrayView3<f32>, b: ArrayView3<f32>, out: &mut Array3<f32>) {
//     Zip::from(out)
//         .and(&a)
//         .and(&b)
//         .for_each(|o, &a, &b| *o = soft_light_pixel(a, b));
// }

// pub fn soft_light_rgb_parallel(a: ArrayView3<f32>, b: ArrayView3<f32>, out: &mut Array3<f32>) {
//     Zip::from(out)
//         .and(&a)
//         .and(&b)
//         .par_for_each(|o, &a, &b| *o = soft_light_pixel(a, b));
// }

// pub fn soft_light_rgb_core(a: ArrayView3<f32>, b: ArrayView3<f32>, out: &mut Array3<f32>) {
//     if a.len() >= PARALLEL_THRESHOLD {
//         soft_light_rgb_parallel(a, b, out);
//     } else {
//         soft_light_rgb_serial(a, b, out);
//     }
// }

// #[inline]
// fn luminosity_blend_pixel(r0: f32, g0: f32, b0: f32, target_lum: f32) -> (f32, f32, f32) {
//     const EPS: f32 = 1e-12;

//     let lum_backdrop = 0.3 * r0 + 0.59 * g0 + 0.11 * b0;
//     let d = target_lum - lum_backdrop;
//     let (r1, g1, b1) = (r0 + d, g0 + d, b0 + d);

//     // lum(r1, g1, b1) == target_lum exactly, since 0.3 + 0.59 + 0.11 ==
//     // 1.0 -- reuse target_lum as `l` instead of recomputing the weighted
//     // sum a second time. This is the fusion-enabled simplification that
//     // isn't available to the two-pass numpy version (soft_light() and
//     // _clip_color() run as separate, unrelated calls there).
//     let l = target_lum;

//     let n = r1.min(g1).min(b1);
//     let x = r1.max(g1).max(b1);

//     // low clip: uses the ORIGINAL n, applied to (r1, g1, b1)
//     let (r2, g2, b2) = if n < 0.0 {
//         let scale = l / (l - n + EPS);
//         (l + (r1 - l) * scale, l + (g1 - l) * scale, l + (b1 - l) * scale)
//     } else {
//         (r1, g1, b1)
//     };

//     // high clip: uses the ORIGINAL x, but applied to the (possibly
//     // already low-clipped) (r2, g2, b2) -- matches blend.py's sequential
//     // `rgb = np.where(...)` reassignment order exactly.
//     let (r3, g3, b3) = if x > 1.0 {
//         let scale = (1.0 - l) / (x - l + EPS);
//         (l + (r2 - l) * scale, l + (g2 - l) * scale, l + (b2 - l) * scale)
//     } else {
//         (r2, g2, b2)
//     };

//     (r3.clamp(0.0, 1.0), g3.clamp(0.0, 1.0), b3.clamp(0.0, 1.0))
// }

// /// Serial variant -- see `soft_light_serial`'s doc comment for why this
// /// is exposed publicly alongside `luminosity_blend_parallel`.
// pub fn luminosity_blend_serial(backdrop_rgb: ArrayView3<f32>, luminosity: ArrayView2<f32>, out: &mut Array3<f32>) {
//     Zip::from(out.outer_iter_mut())
//         .and(backdrop_rgb.outer_iter())
//         .and(luminosity.outer_iter())
//         .for_each(luminosity_blend_row);
// }

// /// Parallel (rayon) variant -- see `soft_light_serial`'s doc comment.
// pub fn luminosity_blend_parallel(backdrop_rgb: ArrayView3<f32>, luminosity: ArrayView2<f32>, out: &mut Array3<f32>) {
//     Zip::from(out.outer_iter_mut())
//         .and(backdrop_rgb.outer_iter())
//         .and(luminosity.outer_iter())
//         .par_for_each(luminosity_blend_row);
// }

// #[inline]
// fn luminosity_blend_row(
//     mut out_row: ndarray::ArrayViewMut2<f32>,
//     backdrop_row: ArrayView2<f32>,
//     lum_row: ndarray::ArrayView1<f32>,
// ) {
//     let w = out_row.shape()[0];
//     for x in 0..w {
//         let r0 = backdrop_row[[x, 0]];
//         let g0 = backdrop_row[[x, 1]];
//         let b0 = backdrop_row[[x, 2]];
//         let l = lum_row[x];
//         let (r, g, b) = luminosity_blend_pixel(r0, g0, b0, l);
//         out_row[[x, 0]] = r;
//         out_row[[x, 1]] = g;
//         out_row[[x, 2]] = b;
//     }
// }

// /// Pure computation: SVG/Photoshop 'Luminosity' blend mode. `backdrop_rgb`
// /// is (H, W, 3) in [0, 1]; `luminosity` is (H, W) in [0, 1]. Matches
// /// `blend.py`'s `luminosity_blend()` (== `_clip_color(backdrop_rgb + d)`)
// /// exactly, fused into a single per-pixel pass with no intermediate
// /// arrays -- see the module docs above for the algebraic shortcut this
// /// enables over the two-function numpy version. Dispatches to the
// /// serial or parallel path based on `PARALLEL_THRESHOLD`.
// pub fn luminosity_blend_core(backdrop_rgb: ArrayView3<f32>, luminosity: ArrayView2<f32>, out: &mut Array3<f32>) {
//     let (h, w, _) = backdrop_rgb.dim();
//     if h * w >= PARALLEL_THRESHOLD {
//         luminosity_blend_parallel(backdrop_rgb, luminosity, out);
//     } else {
//         luminosity_blend_serial(backdrop_rgb, luminosity, out);
//     }
// }

// // ============================================================================
// // Curvature (profile/planform) + hillshade -- fused DEM-derivative kernels.
// //
// // Mirrors TerraTexture.derivatives exactly: same "gradient of gradient"
// // approximation (np.gradient applied twice), same edge_order=1 boundary
// // handling (one-sided forward/backward differences at the array edges,
// // second-order central differences everywhere else -- numpy's default),
// // same Zevenbergen & Thorne curvature formulas. Verified bit-parity
// // against derivatives.py in tests/test_derivatives_rust.py.
// //
// // Callers are responsible for nan-filling the DEM first (derivatives.py's
// // _fill_nan_nearest) and re-masking the NaN/void cells afterwards -- same
// // division of labour as blend.py's Rust dispatch: Rust owns the plain
// // numeric math on a clean float32 array, Python owns the NaN bookkeeping.
// // ============================================================================

// /// numpy-compatible `np.gradient(arr, spacing)` along axis 0 (rows) and
// /// axis 1 (columns), default `edge_order=1`: one-sided forward/backward
// /// difference at the first/last index of each axis, second-order central
// /// difference everywhere in between. Returns (d/d_axis0, d/d_axis1) --
// /// same order as numpy's `zy, zx = np.gradient(dem, cellsize)`.
// fn gradient2d(arr: ArrayView2<f32>, spacing: f32) -> (Array2<f32>, Array2<f32>) {
//     let (h, w) = arr.dim();
//     let mut d_axis0 = Array2::<f32>::zeros((h, w));
//     let mut d_axis1 = Array2::<f32>::zeros((h, w));

//     // axis 0 (down rows), column by column
//     if h == 1 {
//         // np.gradient on a length-1 axis returns zeros
//         d_axis0.fill(0.0);
//     } else {
//         for j in 0..w {
//             d_axis0[[0, j]] = (arr[[1, j]] - arr[[0, j]]) / spacing;
//             d_axis0[[h - 1, j]] = (arr[[h - 1, j]] - arr[[h - 2, j]]) / spacing;
//         }
//         for i in 1..h - 1 {
//             for j in 0..w {
//                 d_axis0[[i, j]] = (arr[[i + 1, j]] - arr[[i - 1, j]]) / (2.0 * spacing);
//             }
//         }
//     }

//     // axis 1 (across columns), row by row
//     if w == 1 {
//         d_axis1.fill(0.0);
//     } else {
//         for i in 0..h {
//             d_axis1[[i, 0]] = (arr[[i, 1]] - arr[[i, 0]]) / spacing;
//             d_axis1[[i, w - 1]] = (arr[[i, w - 1]] - arr[[i, w - 2]]) / spacing;
//             for j in 1..w - 1 {
//                 d_axis1[[i, j]] = (arr[[i, j + 1]] - arr[[i, j - 1]]) / (2.0 * spacing);
//             }
//         }
//     }

//     (d_axis0, d_axis1)
// }

// #[inline]
// fn curvature_pixel(p: f32, q: f32, r: f32, t: f32, s: f32) -> (f32, f32) {
//     let p2q2 = p * p + q * q;
//     if p2q2 < 1e-9 {
//         return (0.0, 0.0); // flat cell -- matches derivatives.py's `flat` mask
//     }
//     let profile_raw = -(r * p * p + 2.0 * s * p * q + t * q * q) / (p2q2 * (1.0 + p2q2).powf(1.5));
//     let planform_raw = -(r * q * q - 2.0 * s * p * q + t * p * p) / p2q2.powf(1.5);

//     // matches np.nan_to_num(..., nan=0.0, posinf=0.0, neginf=0.0)
//     let clean = |v: f32| if v.is_finite() { v } else { 0.0 };
//     (clean(profile_raw), clean(planform_raw))
// }

// /// Pure computation: profile + planform curvature. `dem` must already be
// /// NaN-free (caller nan-fills; see module note above). Matches
// /// `derivatives.py`'s `curvatures()` (minus its NaN re-masking, which
// /// stays the caller's job) exactly.
// pub fn curvatures_core(
//     dem: ArrayView2<f32>,
//     cellsize: f32,
//     profile_out: &mut Array2<f32>,
//     planform_out: &mut Array2<f32>,
// ) {
//     let (zy, zx) = gradient2d(dem, cellsize);
//     let (zxy, zxx) = gradient2d(zx.view(), cellsize);
//     let (zyy, _zyx) = gradient2d(zy.view(), cellsize); // zyx discarded, matches derivatives.py

//     let n = dem.len();
//     // 2 outputs + 5 inputs = 7 producers, one over ndarray::Zip's max
//     // arity of 6 -- fall back to plain contiguous slices + rayon here
//     // instead (all arrays are freshly `zeros()`-allocated, hence
//     // standard/C-contiguous, so `.as_slice()` is always `Some`).
//     let zx_s = zx.as_slice().expect("gradient2d output not contiguous");
//     let zy_s = zy.as_slice().expect("gradient2d output not contiguous");
//     let zxx_s = zxx.as_slice().expect("gradient2d output not contiguous");
//     let zyy_s = zyy.as_slice().expect("gradient2d output not contiguous");
//     let zxy_s = zxy.as_slice().expect("gradient2d output not contiguous");
//     let profile_s = profile_out.as_slice_mut().expect("profile_out not contiguous");
//     let planform_s = planform_out.as_slice_mut().expect("planform_out not contiguous");

//     let compute = |i: usize, po: &mut f32, plo: &mut f32| {
//         let (profile, planform) = curvature_pixel(zx_s[i], zy_s[i], zxx_s[i], zyy_s[i], zxy_s[i]);
//         *po = profile;
//         *plo = planform;
//     };

//     if n >= PARALLEL_THRESHOLD {
//         profile_s
//             .par_iter_mut()
//             .zip(planform_s.par_iter_mut())
//             .enumerate()
//             .for_each(|(i, (po, plo))| compute(i, po, plo));
//     } else {
//         profile_s
//             .iter_mut()
//             .zip(planform_s.iter_mut())
//             .enumerate()
//             .for_each(|(i, (po, plo))| compute(i, po, plo));
//     }
// }

// #[inline]
// fn hillshade_pixel(zx: f32, zy: f32, sin_az: f32, cos_az: f32, sin_alt: f32, cos_alt: f32) -> f32 {
//     // Algebraic expansion of the original
//     //   slope = pi/2 - atan(hypot(zx, zy))
//     //   aspect = atan2(-zx, zy)
//     //   shaded = sin(alt)*sin(slope) + cos(alt)*cos(slope)*cos(az - aspect)
//     // using sin(atan(g)) = g/sqrt(1+g^2), cos(atan(g)) = 1/sqrt(1+g^2),
//     // and cos(az-aspect) = cos(az)cos(aspect) + sin(az)sin(aspect) with
//     // sin(aspect) = -zx/g, cos(aspect) = zy/g (g = hypot(zx, zy)). The
//     // factor of `g` cancels completely, leaving one sqrt and no
//     // atan/atan2/sin(slope)/cos(slope) per pixel at all -- verified
//     // bit-for-bit (float32 tolerance) against derivatives.py's original
//     // formula in tests/test_derivatives_rust.py, including the flat
//     // (zx=zy=0) case, which needs no special-casing here since
//     // sqrt(1+0+0)=1 rather than a 0/0 from hypot(0,0).
//     let denom = (1.0 + zx * zx + zy * zy).sqrt();
//     let shaded = (sin_alt + cos_alt * (cos_az * zy - sin_az * zx)) / denom;
//     shaded.clamp(0.0, 1.0)
// }

// /// Pure computation: hillshade. `dem` must already be NaN-free (see
// /// module note above). `azimuth`/`altitude` in degrees, matching
// /// `derivatives.py`'s `hillshade()` signature exactly (including its
// /// az = 360 - azimuth + 90 convention).
// pub fn hillshade_core(dem: ArrayView2<f32>, cellsize: f32, azimuth: f32, altitude: f32, out: &mut Array2<f32>) {
//     let (zy, zx) = gradient2d(dem, cellsize);
//     let az = (360.0 - azimuth + 90.0).to_radians();
//     let alt = altitude.to_radians();
//     // sin_cos() computes both in one call and, more importantly, these
//     // are computed ONCE for the whole DEM -- not per pixel like the
//     // original az.sin()/alt.cos()/etc. calls inside the old
//     // hillshade_pixel were (a much bigger win than the sin_cos()
//     // fusion itself: 4 trig calls total instead of up to 2*H*W).
//     let (sin_az, cos_az) = az.sin_cos();
//     let (sin_alt, cos_alt) = alt.sin_cos();

//     let (h, w) = dem.dim();
//     let n = h * w;
//     let combine = |o: &mut f32, &zx: &f32, &zy: &f32| {
//         *o = hillshade_pixel(zx, zy, sin_az, cos_az, sin_alt, cos_alt);
//     };
//     let z = Zip::from(out).and(&zx).and(&zy);
//     if n >= PARALLEL_THRESHOLD {
//         z.par_for_each(combine);
//     } else {
//         z.for_each(combine);
//     }
// }

// // ============================================================================
// // stretch_std -- fused single-pass mean/std reduction + clip-stretch.
// //
// // Mirrors TerraTexture.stretch's stretch_std() exactly: NaN in -> NaN
// // out; mean +/- n_std*std clipped to [0,1]. The numpy version calls
// // np.nanmean() then np.nanstd() separately -- nanstd recomputes its own
// // mean internally, so the array's mean ends up computed twice, plus a
// // separate variance pass, plus the elementwise subtract/divide/clip
// // (verified: deduplicating just the mean call gives only ~1.1x, numpy's
// // C reductions are already efficient -- the real win here is doing ONE
// // reduction pass (sum, sum-of-squares, count together) instead of
// // several, not language speed per se).
// //
// // Accumulates in f64 for numerical robustness on large arrays -- this
// // is NOT intended to bit-match numpy's own (float32, pairwise
// // summation) internal accumulation, and isn't expected to; verified
// // against it within a statistical tolerance instead (see
// // tests/test_stretch_rust.py), which is the correct bar for a mean/std
// // computation, not exact equality.
// // ============================================================================

// #[inline]
// fn stretch_pixel(v: f32, lo: f32, denom: f32) -> f32 {
//     if v.is_nan() {
//         f32::NAN // matches np.clip((arr - lo) / denom, 0, 1): NaN propagates, never clamped away
//     } else {
//         ((v - lo) / denom).clamp(0.0, 1.0)
//     }
// }

// fn sum_stats_serial(arr: ArrayView2<f32>) -> (f64, f64, u64) {
//     let mut sum = 0.0f64;
//     let mut sumsq = 0.0f64;
//     let mut count = 0u64;
//     for &v in arr.iter() {
//         if !v.is_nan() {
//             let vd = v as f64;
//             sum += vd;
//             sumsq += vd * vd;
//             count += 1;
//         }
//     }
//     (sum, sumsq, count)
// }

// fn sum_stats_parallel(arr: ArrayView2<f32>) -> (f64, f64, u64) {
//     // rayon's fold+reduce needs a flat parallel iterator; ndarray's own
//     // arrays are standard/C-contiguous here (always freshly allocated
//     // or a `.ascontiguousarray()`'d caller-provided one -- see
//     // stretch.py's dispatch), so `.as_slice()` is reliably `Some`. Falls
//     // back to the serial path in the (should-be-unreachable in
//     // practice) case it isn't, rather than panicking.
//     match arr.as_slice() {
//         Some(slice) => slice
//             .par_iter()
//             .fold(
//                 || (0.0f64, 0.0f64, 0u64),
//                 |(s, sq, c), &v| {
//                     if v.is_nan() {
//                         (s, sq, c)
//                     } else {
//                         let vd = v as f64;
//                         (s + vd, sq + vd * vd, c + 1)
//                     }
//                 },
//             )
//             .reduce(
//                 || (0.0f64, 0.0f64, 0u64),
//                 |(s1, sq1, c1), (s2, sq2, c2)| (s1 + s2, sq1 + sq2, c1 + c2),
//             ),
//         None => sum_stats_serial(arr),
//     }
// }

// /// Pure computation: mean +/- n_std*std stretch to [0,1], NaN-safe.
// /// Matches `stretch.py`'s `stretch_std()` exactly (within float
// /// tolerance -- see module note above).
// pub fn stretch_std_core(arr: ArrayView2<f32>, n_std: f32, out: &mut Array2<f32>) {
//     let n = arr.len();
//     let (sum, sumsq, count) = if n >= PARALLEL_THRESHOLD {
//         sum_stats_parallel(arr)
//     } else {
//         sum_stats_serial(arr)
//     };

//     let mean_f64 = if count > 0 { sum / count as f64 } else { 0.0 };
//     let variance_f64 = if count > 0 {
//         (sumsq / count as f64 - mean_f64 * mean_f64).max(0.0) // guard tiny negative from float error
//     } else {
//         0.0
//     };
//     let mean = mean_f64 as f32;
//     let std = variance_f64.sqrt() as f32;
//     let lo = mean - n_std * std;
//     let hi = mean + n_std * std;
//     let denom = hi - lo + 1e-12;

//     let combine = |o: &mut f32, &v: &f32| *o = stretch_pixel(v, lo, denom);
//     let z = Zip::from(out).and(&arr);
//     if n >= PARALLEL_THRESHOLD {
//         z.par_for_each(combine);
//     } else {
//         z.for_each(combine);
//     }
// }

// // to REMOVE GIL since we're in Rust only... we can use PyO3 idiom Python::allow_threads
// // Release the GIL for the actual compute: soft_light_core may fan
// // out across rayon's thread pool, and none of that work touches
// // any Python object (a/b/out are plain ndarray views/buffers, not
// // PyAny) -- so there's no reason another Python thread (e.g. a
// // contextily tile-fetch thread) should be blocked from running
// // while this executes. GIL is re-acquired automatically before
// // this closure returns and `out` gets wrapped back into a PyArray.

// #[pyfunction]
// fn soft_light<'py>(
//     py: Python<'py>,
//     base: PyReadonlyArray2<'py, f32>,
//     blend: PyReadonlyArray2<'py, f32>,
// ) -> Bound<'py, PyArray2<f32>> {
//     let a = base.as_array();
//     let b = blend.as_array();
//     let mut out = Array2::<f32>::zeros(a.raw_dim());
//     py.detach(|| {
//         soft_light_core(a, b, &mut out);
//     });
//     out.into_pyarray(py)
// }

// #[pyfunction]
// fn soft_light_rgb<'py>(
//     py: Python<'py>,
//     base: PyReadonlyArray3<'py, f32>,
//     blend: PyReadonlyArray3<'py, f32>,
// ) -> Bound<'py, PyArray3<f32>> {
//     let a = base.as_array();
//     let b = blend.as_array();
//     let mut out = Array3::<f32>::zeros(a.raw_dim());
//     py.detach(|| {
//         soft_light_rgb_core(a, b, &mut out);
//     });
//     out.into_pyarray(py)
// }

// #[pyfunction]
// fn luminosity_blend<'py>(
//     py: Python<'py>,
//     backdrop_rgb: PyReadonlyArray3<'py, f32>,
//     luminosity: PyReadonlyArray2<'py, f32>,
// ) -> Bound<'py, PyArray3<f32>> {
//     let backdrop = backdrop_rgb.as_array();
//     let lum = luminosity.as_array();
//     let mut out = Array3::<f32>::zeros(backdrop.raw_dim());
//     py.detach(|| {
//         luminosity_blend_core(backdrop, lum, &mut out);
//     });
//     out.into_pyarray(py)
// }

// #[pyfunction]
// fn curvatures<'py>(
//     py: Python<'py>,
//     dem: PyReadonlyArray2<'py, f32>,
//     cellsize: f32,
// ) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>) {
//     let d = dem.as_array();
//     let mut profile = Array2::<f32>::zeros(d.raw_dim());
//     let mut planform = Array2::<f32>::zeros(d.raw_dim());
//     py.detach(|| {
//         curvatures_core(d, cellsize, &mut profile, &mut planform);
//     });
//     (profile.into_pyarray(py), planform.into_pyarray(py))
// }

// #[pyfunction]
// fn hillshade<'py>(
//     py: Python<'py>,
//     dem: PyReadonlyArray2<'py, f32>,
//     cellsize: f32,
//     azimuth: f32,
//     altitude: f32,
// ) -> Bound<'py, PyArray2<f32>> {
//     let d = dem.as_array();
//     let mut out = Array2::<f32>::zeros(d.raw_dim());
//     py.detach(|| {
//         hillshade_core(d, cellsize, azimuth, altitude, &mut out);
//     });
//     out.into_pyarray(py)
// }

// #[pyfunction]
// fn stretch_std<'py>(py: Python<'py>, arr: PyReadonlyArray2<'py, f32>, n_std: f32) -> Bound<'py, PyArray2<f32>> {
//     let a = arr.as_array();
//     let mut out = Array2::<f32>::zeros(a.raw_dim());
//     py.detach(|| {
//         stretch_std_core(a, n_std, &mut out);
//     });
//     out.into_pyarray(py)
// }

// #[pymodule]
// fn terra_texture_rs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
//     m.add_function(wrap_pyfunction!(soft_light, m)?)?;
//     m.add_function(wrap_pyfunction!(soft_light_rgb, m)?)?;
//     m.add_function(wrap_pyfunction!(luminosity_blend, m)?)?;
//     m.add_function(wrap_pyfunction!(curvatures, m)?)?;
//     m.add_function(wrap_pyfunction!(hillshade, m)?)?;
//     m.add_function(wrap_pyfunction!(stretch_std, m)?)?;
//     Ok(())
// }
