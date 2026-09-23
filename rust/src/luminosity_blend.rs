//! SVG/Photoshop 'Luminosity' blend mode, fused into a single per-pixel
//! pass.
//!
//! Matches `blend.py`'s `luminosity_blend()`, i.e.
//! `_clip_color(backdrop_rgb + d)`, exactly. The numpy version runs the
//! luminosity shift and `_clip_color()` as two separate, unrelated
//! passes with intermediate arrays; fusing them here enables one
//! algebraic shortcut on top of skipping the intermediates.
//!
//! # Algorithm
//!
//! Per pixel, with luminance `lum(r, g, b) = 0.3r + 0.59g + 0.11b`:
//!
//! 1. Shift every channel by `d = target_lum - lum(backdrop)`.
//! 2. Clip the shifted colour back into \[0, 1\] while preserving its
//!    luminance: first pull the minimum channel up to 0 if it went
//!    negative, then pull the maximum channel down to 1 if it went over.
//! 3. Clamp each channel to \[0, 1\] to absorb float error.
//!
//! The fusion shortcut: after step 1, `lum(shifted) == target_lum`
//! exactly, because the weights sum to 1.0. So step 2 reuses
//! `target_lum` instead of recomputing the weighted sum.
//!
//! # Which function to call
//!
//! Call [`luminosity_blend_core`]. The `*_serial`/`*_parallel` variants
//! are public only so `benches/blend_bench.rs` can time both paths to
//! tune [`PARALLEL_THRESHOLD`].

use ndarray::{Array3, ArrayView1, ArrayView2, ArrayView3, ArrayViewMut2, Zip};

use crate::common::PARALLEL_THRESHOLD;

/// Luminosity blend for one pixel.
///
/// Takes the backdrop colour `(r0, g0, b0)` and the `target_lum` to
/// impose, all `f32` nominally in \[0, 1\]. Returns the blended
/// `(r, g, b)` as `f32`, each clamped to \[0, 1\]. See the module docs for
/// the algorithm.
#[inline]
fn luminosity_blend_pixel(r0: f32, g0: f32, b0: f32, target_lum: f32) -> (f32, f32, f32) {
    const EPS: f32 = 1e-12;

    let lum_backdrop = 0.3 * r0 + 0.59 * g0 + 0.11 * b0;
    let d = target_lum - lum_backdrop;
    let (r1, g1, b1) = (r0 + d, g0 + d, b0 + d);

    // Fusion shortcut (see module docs): lum(r1, g1, b1) == target_lum.
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

/// Luminosity blend for one image row.
///
/// * `out_row` - `ArrayViewMut2<f32>`, shape (W, 3): written.
/// * `backdrop_row` - `ArrayView2<f32>`, shape (W, 3): read.
/// * `lum_row` - `ArrayView1<f32>`, shape (W,): read.
#[inline]
fn luminosity_blend_row(mut out_row: ArrayViewMut2<f32>, backdrop_row: ArrayView2<f32>, lum_row: ArrayView1<f32>) {
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

/// Luminosity blend, single-threaded.
///
/// Same arguments, output and panics as [`luminosity_blend_core`], but
/// always runs serially. Public for benchmarking only (see module docs).

pub fn luminosity_blend_serial(backdrop_rgb: ArrayView3<f32>, luminosity: ArrayView2<f32>, out: &mut Array3<f32>) {
    Zip::from(out.outer_iter_mut())
        .and(backdrop_rgb.outer_iter())
        .and(luminosity.outer_iter())
        .for_each(luminosity_blend_row);
}

/// Luminosity blend, parallelised across rows with rayon.
///
/// Same arguments, output and panics as [`luminosity_blend_core`], but
/// always runs in parallel. Public for benchmarking only (see module
/// docs).
pub fn luminosity_blend_parallel(backdrop_rgb: ArrayView3<f32>, luminosity: ArrayView2<f32>, out: &mut Array3<f32>) {
    Zip::from(out.outer_iter_mut())
        .and(backdrop_rgb.outer_iter())
        .and(luminosity.outer_iter())
        .par_for_each(luminosity_blend_row);
}

/// Replace the luminance of an RGB image with a target luminance,
/// keeping its hue and saturation. The entry point for this blend.
///
/// # Arguments
///
/// * `backdrop_rgb` - `ArrayView3<f32>`, shape (H, W, 3): the colour
///   image, channels in R, G, B order, values nominally in \[0, 1\].
/// * `luminosity` - `ArrayView2<f32>`, shape (H, W): the target
///   luminance per pixel (e.g. a hillshade), values nominally in \[0, 1\].
/// * `out` - `&mut Array3<f32>`, shape (H, W, 3): overwritten with the
///   blended RGB image, every element in \[0, 1\].
///
/// Runs serially below [`PARALLEL_THRESHOLD`]
/// **pixels** (H × W, not H × W × 3) and in parallel at or above it.
///
/// # Panics
///
/// * If `backdrop_rgb`, `luminosity` and `out` disagree on H.
/// * If the channel axis of `backdrop_rgb` or `out` has fewer than 3
///   entries (out-of-bounds index).
///
/// Shapes are only checked by indexing, so some mismatches pass
/// silently instead of panicking: a W mismatch where `backdrop_rgb` or
/// `luminosity` is wider than `out` ignores the extra columns, and a
/// channel count above 3 reads and writes only channels 0 to 2. Always
/// pass matching (H, W) and exactly 3 channels.
///
/// # Example
///
/// ```
/// use ndarray::{Array2, Array3};
/// use terra_texture_rs::luminosity_blend_core;
///
/// // a flat mid-grey image, relit to luminance 0.8
/// let backdrop = Array3::<f32>::from_elem((2, 2, 3), 0.5);
/// let lum = Array2::<f32>::from_elem((2, 2), 0.8);
/// let mut out = Array3::<f32>::zeros(backdrop.raw_dim());
/// luminosity_blend_core(backdrop.view(), lum.view(), &mut out);
///
/// // grey stays grey, now at the target luminance
/// assert!(out.iter().all(|&v| (v - 0.8).abs() < 1e-5));
/// ```
pub fn luminosity_blend_core(backdrop_rgb: ArrayView3<f32>, luminosity: ArrayView2<f32>, out: &mut Array3<f32>) {
    let (h, w, _) = backdrop_rgb.dim();
    if h * w >= PARALLEL_THRESHOLD {
        luminosity_blend_parallel(backdrop_rgb, luminosity, out);
    } else {
        luminosity_blend_serial(backdrop_rgb, luminosity, out);
    }
}
