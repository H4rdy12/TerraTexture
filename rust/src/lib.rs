//! Fused elementwise kernels for `terra_texture.blend`, accelerating the
//! pure-numpy implementations in `blend.py` without changing their
//! public behaviour.
//!
//! `blend.py` tries to import this compiled extension and falls back to
//! its numpy implementation if the import fails (unbuilt, unsupported
//! platform, or a plain `pip install` without the compiled wheel) --
//! that fallback is load-bearing, not incidental: the package's stated
//! design goal is that `terra_texture.derivatives`/`terra_texture.blend`
//! have zero hard dependencies beyond numpy/scipy.
//!
//! ## Structure
//!
//! Each kernel is split into two layers:
//! - a `*_core` function: pure `ndarray` in, pure `ndarray` out, no PyO3
//!   types anywhere. This is what `benches/blend_bench.rs` and any future
//!   `#[test]`s call directly -- no Python interpreter needed to run them.
//! - a `#[pyfunction]` wrapper: unwraps the numpy arrays into `ndarray`
//!   views, calls the core function, wraps the result back up. This is
//!   the only layer that knows about Python at all.
//!
//! This split is why `[lib] crate-type` includes `"rlib"` alongside the
//! `"cdylib"` Python needs -- an rlib is what `cargo bench`/`cargo test`
//! link against to call the core functions in-process.
//!
//! ## The serial/parallel threshold
//!
//! Rayon's parallel dispatch has fixed per-call overhead (splitting work,
//! synchronizing threads) that only pays for itself once there's enough
//! work per thread to amortize it. Below `PARALLEL_THRESHOLD` elements,
//! kernels run a plain serial loop instead of `par_for_each`. The
//! threshold below is a **provisional placeholder**, not a measured
//! value -- run `cargo bench` (see `benches/blend_bench.rs`) to find the
//! actual crossover point on real hardware and update it here.

use ndarray::{Array2, Array3, ArrayView2, ArrayView3, Zip};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;

// TODO(benchmark): this is a guess, not a measurement. Run `cargo bench`
// and replace it with whatever `blend_bench.rs` actually finds as the
// point where `par_for_each` starts winning over a serial loop on the
// target hardware. 256x256 = 65_536 is picked only because it "feels"
// like a plausible order of magnitude for rayon's dispatch overhead to
// have been amortized -- treat it as unverified until benchmarked.
pub const PARALLEL_THRESHOLD: usize = 65_536;

#[inline]
fn soft_light_pixel(a: f32, b: f32) -> f32 {
    let v = if b <= 0.5 {
        2.0 * a * b + a * a * (1.0 - 2.0 * b)
    } else {
        // matches Python's np.sqrt(np.clip(a, 0, 1)) -- clamped to BOTH
        // 0 and 1 before the sqrt, not just floored at 0.
        2.0 * a * (1.0 - b) + a.clamp(0.0, 1.0).sqrt() * (2.0 * b - 1.0)
    };
    v.clamp(0.0, 1.0)
}

/// Serial (single-threaded) soft light -- exposed publicly alongside
/// `soft_light_parallel` purely so `benches/blend_bench.rs` can compare
/// them directly at every array size, independent of whatever
/// `PARALLEL_THRESHOLD` currently guesses. `soft_light_core()` below is
/// the actual dispatch entry point everything else should call.
pub fn soft_light_serial(a: ArrayView2<f32>, b: ArrayView2<f32>, out: &mut Array2<f32>) {
    Zip::from(out)
        .and(&a)
        .and(&b)
        .for_each(|o, &a, &b| *o = soft_light_pixel(a, b));
}

/// Parallel (rayon) soft light -- see `soft_light_serial`'s doc comment.
pub fn soft_light_parallel(a: ArrayView2<f32>, b: ArrayView2<f32>, out: &mut Array2<f32>) {
    Zip::from(out)
        .and(&a)
        .and(&b)
        .par_for_each(|o, &a, &b| *o = soft_light_pixel(a, b));
}

/// Pure computation: Photoshop-style soft light blend. `base`/`blend`
/// arrays in [0, 1], same shape. Matches `blend.py`'s `soft_light()`
/// exactly (verified against it in `tests/test_blend.py`'s Rust-parity
/// tests, when the extension is built). Dispatches to the serial or
/// parallel path based on `PARALLEL_THRESHOLD` -- this is the function
/// the `#[pyfunction]` wrapper and any other caller should use.
pub fn soft_light_core(a: ArrayView2<f32>, b: ArrayView2<f32>, out: &mut Array2<f32>) {
    if a.len() >= PARALLEL_THRESHOLD {
        soft_light_parallel(a, b, out);
    } else {
        soft_light_serial(a, b, out);
    }
}

#[inline]
fn luminosity_blend_pixel(r0: f32, g0: f32, b0: f32, target_lum: f32) -> (f32, f32, f32) {
    const EPS: f32 = 1e-12;

    let lum_backdrop = 0.3 * r0 + 0.59 * g0 + 0.11 * b0;
    let d = target_lum - lum_backdrop;
    let (r1, g1, b1) = (r0 + d, g0 + d, b0 + d);

    // lum(r1, g1, b1) == target_lum exactly, since 0.3 + 0.59 + 0.11 ==
    // 1.0 -- reuse target_lum as `l` instead of recomputing the weighted
    // sum a second time. This is the fusion-enabled simplification that
    // isn't available to the two-pass numpy version (soft_light() and
    // _clip_color() run as separate, unrelated calls there).
    let l = target_lum;

    let n = r1.min(g1).min(b1);
    let x = r1.max(g1).max(b1);

    // low clip: uses the ORIGINAL n, applied to (r1, g1, b1)
    let (r2, g2, b2) = if n < 0.0 {
        let scale = l / (l - n + EPS);
        (l + (r1 - l) * scale, l + (g1 - l) * scale, l + (b1 - l) * scale)
    } else {
        (r1, g1, b1)
    };

    // high clip: uses the ORIGINAL x, but applied to the (possibly
    // already low-clipped) (r2, g2, b2) -- matches blend.py's sequential
    // `rgb = np.where(...)` reassignment order exactly.
    let (r3, g3, b3) = if x > 1.0 {
        let scale = (1.0 - l) / (x - l + EPS);
        (l + (r2 - l) * scale, l + (g2 - l) * scale, l + (b2 - l) * scale)
    } else {
        (r2, g2, b2)
    };

    (r3.clamp(0.0, 1.0), g3.clamp(0.0, 1.0), b3.clamp(0.0, 1.0))
}

/// Serial variant -- see `soft_light_serial`'s doc comment for why this
/// is exposed publicly alongside `luminosity_blend_parallel`.
pub fn luminosity_blend_serial(backdrop_rgb: ArrayView3<f32>, luminosity: ArrayView2<f32>, out: &mut Array3<f32>) {
    Zip::from(out.outer_iter_mut())
        .and(backdrop_rgb.outer_iter())
        .and(luminosity.outer_iter())
        .for_each(luminosity_blend_row);
}

/// Parallel (rayon) variant -- see `soft_light_serial`'s doc comment.
pub fn luminosity_blend_parallel(backdrop_rgb: ArrayView3<f32>, luminosity: ArrayView2<f32>, out: &mut Array3<f32>) {
    Zip::from(out.outer_iter_mut())
        .and(backdrop_rgb.outer_iter())
        .and(luminosity.outer_iter())
        .par_for_each(luminosity_blend_row);
}

#[inline]
fn luminosity_blend_row(
    mut out_row: ndarray::ArrayViewMut2<f32>,
    backdrop_row: ArrayView2<f32>,
    lum_row: ndarray::ArrayView1<f32>,
) {
    let w = out_row.shape()[0];
    for x in 0..w {
        let r0 = backdrop_row[[x, 0]];
        let g0 = backdrop_row[[x, 1]];
        let b0 = backdrop_row[[x, 2]];
        let l = lum_row[x];
        let (r, g, b) = luminosity_blend_pixel(r0, g0, b0, l);
        out_row[[x, 0]] = r;
        out_row[[x, 1]] = g;
        out_row[[x, 2]] = b;
    }
}

/// Pure computation: SVG/Photoshop 'Luminosity' blend mode. `backdrop_rgb`
/// is (H, W, 3) in [0, 1]; `luminosity` is (H, W) in [0, 1]. Matches
/// `blend.py`'s `luminosity_blend()` (== `_clip_color(backdrop_rgb + d)`)
/// exactly, fused into a single per-pixel pass with no intermediate
/// arrays -- see the module docs above for the algebraic shortcut this
/// enables over the two-function numpy version. Dispatches to the
/// serial or parallel path based on `PARALLEL_THRESHOLD`.
pub fn luminosity_blend_core(backdrop_rgb: ArrayView3<f32>, luminosity: ArrayView2<f32>, out: &mut Array3<f32>) {
    let (h, w, _) = backdrop_rgb.dim();
    if h * w >= PARALLEL_THRESHOLD {
        luminosity_blend_parallel(backdrop_rgb, luminosity, out);
    } else {
        luminosity_blend_serial(backdrop_rgb, luminosity, out);
    }
}

// ============================================================================
// Curvature (profile/planform) + hillshade -- fused DEM-derivative kernels.
//
// Mirrors TerraTexture.derivatives exactly: same "gradient of gradient"
// approximation (np.gradient applied twice), same edge_order=1 boundary
// handling (one-sided forward/backward differences at the array edges,
// second-order central differences everywhere else -- numpy's default),
// same Zevenbergen & Thorne curvature formulas. Verified bit-parity
// against derivatives.py in tests/test_derivatives_rust.py.
//
// Callers are responsible for nan-filling the DEM first (derivatives.py's
// _fill_nan_nearest) and re-masking the NaN/void cells afterwards -- same
// division of labour as blend.py's Rust dispatch: Rust owns the plain
// numeric math on a clean float32 array, Python owns the NaN bookkeeping.
// ============================================================================

/// numpy-compatible `np.gradient(arr, spacing)` along axis 0 (rows) and
/// axis 1 (columns), default `edge_order=1`: one-sided forward/backward
/// difference at the first/last index of each axis, second-order central
/// difference everywhere in between. Returns (d/d_axis0, d/d_axis1) --
/// same order as numpy's `zy, zx = np.gradient(dem, cellsize)`.
fn gradient2d(arr: ArrayView2<f32>, spacing: f32) -> (Array2<f32>, Array2<f32>) {
    let (h, w) = arr.dim();
    let mut d_axis0 = Array2::<f32>::zeros((h, w));
    let mut d_axis1 = Array2::<f32>::zeros((h, w));

    // axis 0 (down rows), column by column
    if h == 1 {
        // np.gradient on a length-1 axis returns zeros
        d_axis0.fill(0.0);
    } else {
        for j in 0..w {
            d_axis0[[0, j]] = (arr[[1, j]] - arr[[0, j]]) / spacing;
            d_axis0[[h - 1, j]] = (arr[[h - 1, j]] - arr[[h - 2, j]]) / spacing;
        }
        for i in 1..h - 1 {
            for j in 0..w {
                d_axis0[[i, j]] = (arr[[i + 1, j]] - arr[[i - 1, j]]) / (2.0 * spacing);
            }
        }
    }

    // axis 1 (across columns), row by row
    if w == 1 {
        d_axis1.fill(0.0);
    } else {
        for i in 0..h {
            d_axis1[[i, 0]] = (arr[[i, 1]] - arr[[i, 0]]) / spacing;
            d_axis1[[i, w - 1]] = (arr[[i, w - 1]] - arr[[i, w - 2]]) / spacing;
            for j in 1..w - 1 {
                d_axis1[[i, j]] = (arr[[i, j + 1]] - arr[[i, j - 1]]) / (2.0 * spacing);
            }
        }
    }

    (d_axis0, d_axis1)
}

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
/// NaN-free (caller nan-fills; see module note above). Matches
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

#[inline]
fn hillshade_pixel(zx: f32, zy: f32, az: f32, alt: f32) -> f32 {
    let slope = std::f32::consts::FRAC_PI_2 - (zx.hypot(zy)).atan();
    let aspect = (-zx).atan2(zy);
    let shaded = alt.sin() * slope.sin() + alt.cos() * slope.cos() * (az - aspect).cos();
    shaded.clamp(0.0, 1.0)
}

/// Pure computation: hillshade. `dem` must already be NaN-free (see
/// module note above). `azimuth`/`altitude` in degrees, matching
/// `derivatives.py`'s `hillshade()` signature exactly (including its
/// az = 360 - azimuth + 90 convention).
pub fn hillshade_core(dem: ArrayView2<f32>, cellsize: f32, azimuth: f32, altitude: f32, out: &mut Array2<f32>) {
    let (zy, zx) = gradient2d(dem, cellsize);
    let az = (360.0 - azimuth + 90.0).to_radians();
    let alt = altitude.to_radians();

    let (h, w) = dem.dim();
    let n = h * w;
    let combine = |o: &mut f32, &zx: &f32, &zy: &f32| {
        *o = hillshade_pixel(zx, zy, az, alt);
    };
    let z = Zip::from(out).and(&zx).and(&zy);
    if n >= PARALLEL_THRESHOLD {
        z.par_for_each(combine);
    } else {
        z.for_each(combine);
    }
}

// to REMOVE GIL since we're in Rust only... we can use PyO3 idiom Python::allow_threads
// Release the GIL for the actual compute: soft_light_core may fan
// out across rayon's thread pool, and none of that work touches
// any Python object (a/b/out are plain ndarray views/buffers, not
// PyAny) -- so there's no reason another Python thread (e.g. a
// contextily tile-fetch thread) should be blocked from running
// while this executes. GIL is re-acquired automatically before
// this closure returns and `out` gets wrapped back into a PyArray.

#[pyfunction]
fn soft_light<'py>(
    py: Python<'py>,
    base: PyReadonlyArray2<'py, f32>,
    blend: PyReadonlyArray2<'py, f32>,
) -> Bound<'py, PyArray2<f32>> {
    let a = base.as_array();
    let b = blend.as_array();
    let mut out = Array2::<f32>::zeros(a.raw_dim());
    py.allow_threads(|| {
        soft_light_core(a, b, &mut out);
    });
    out.into_pyarray_bound(py)
}

#[pyfunction]
fn luminosity_blend<'py>(
    py: Python<'py>,
    backdrop_rgb: PyReadonlyArray3<'py, f32>,
    luminosity: PyReadonlyArray2<'py, f32>,
) -> Bound<'py, PyArray3<f32>> {
    let backdrop = backdrop_rgb.as_array();
    let lum = luminosity.as_array();
    let mut out = Array3::<f32>::zeros(backdrop.raw_dim());
    py.allow_threads(|| {
        luminosity_blend_core(backdrop, lum, &mut out);
    });
    out.into_pyarray_bound(py)
}

#[pyfunction]
fn curvatures<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f32>,
    cellsize: f32,
) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>) {
    let d = dem.as_array();
    let mut profile = Array2::<f32>::zeros(d.raw_dim());
    let mut planform = Array2::<f32>::zeros(d.raw_dim());
    py.allow_threads(|| {
        curvatures_core(d, cellsize, &mut profile, &mut planform);
    });
    (profile.into_pyarray_bound(py), planform.into_pyarray_bound(py))
}

#[pyfunction]
fn hillshade<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f32>,
    cellsize: f32,
    azimuth: f32,
    altitude: f32,
) -> Bound<'py, PyArray2<f32>> {
    let d = dem.as_array();
    let mut out = Array2::<f32>::zeros(d.raw_dim());
    py.allow_threads(|| {
        hillshade_core(d, cellsize, azimuth, altitude, &mut out);
    });
    out.into_pyarray_bound(py)
}

#[pymodule]
fn terra_texture_rs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(soft_light, m)?)?;
    m.add_function(wrap_pyfunction!(luminosity_blend, m)?)?;
    m.add_function(wrap_pyfunction!(curvatures, m)?)?;
    m.add_function(wrap_pyfunction!(hillshade, m)?)?;
    Ok(())
}
