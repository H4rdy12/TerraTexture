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

#[pyfunction]
fn soft_light<'py>(
    py: Python<'py>,
    base: PyReadonlyArray2<'py, f32>,
    blend: PyReadonlyArray2<'py, f32>,
) -> Bound<'py, PyArray2<f32>> {
    let a = base.as_array();
    let b = blend.as_array();
    let mut out = Array2::<f32>::zeros(a.raw_dim());
    soft_light_core(a, b, &mut out);
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
    luminosity_blend_core(backdrop, lum, &mut out);
    out.into_pyarray_bound(py)
}

#[pymodule]
fn terra_texture_rs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(soft_light, m)?)?;
    m.add_function(wrap_pyfunction!(luminosity_blend, m)?)?;
    Ok(())
}
