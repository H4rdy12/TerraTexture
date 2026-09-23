//! SVG/Photoshop 'Luminosity' blend mode, fused into a single per-pixel
//! pass.
//!
//! Matches `blend.py`'s `luminosity_blend()` (== `_clip_color(backdrop_rgb
//! + d)`) exactly. The numpy version runs the luminosity shift and
//! `_clip_color()` as two separate, unrelated passes with intermediate
//! arrays; fusing them here enables one algebraic shortcut on top of
//! skipping the intermediates:
//!
//! After shifting every channel by `d = target_lum - lum(backdrop)`,
//! `lum(r1, g1, b1) == target_lum` exactly, since the luminance weights
//! sum to 1.0 (0.3 + 0.59 + 0.11). So `_clip_color`'s own luminance
//! recomputation can be replaced by reusing `target_lum` directly -- one
//! fewer weighted sum per pixel.
//!
//! The `*_serial`/`*_parallel` variants are public purely so
//! `benches/blend_bench.rs` can compare them directly; call
//! `luminosity_blend_core` everywhere else.

use ndarray::{Array3, ArrayView1, ArrayView2, ArrayView3, ArrayViewMut2, Zip};

use crate::common::PARALLEL_THRESHOLD;

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

/// Serial variant -- see module docs for why this is public.
pub fn luminosity_blend_serial(backdrop_rgb: ArrayView3<f32>, luminosity: ArrayView2<f32>, out: &mut Array3<f32>) {
    Zip::from(out.outer_iter_mut())
        .and(backdrop_rgb.outer_iter())
        .and(luminosity.outer_iter())
        .for_each(luminosity_blend_row);
}

/// Parallel (rayon) variant -- see module docs for why this is public.
pub fn luminosity_blend_parallel(backdrop_rgb: ArrayView3<f32>, luminosity: ArrayView2<f32>, out: &mut Array3<f32>) {
    Zip::from(out.outer_iter_mut())
        .and(backdrop_rgb.outer_iter())
        .and(luminosity.outer_iter())
        .par_for_each(luminosity_blend_row);
}

/// Pure computation: 'Luminosity' blend. `backdrop_rgb` is (H, W, 3) in
/// [0, 1]; `luminosity` is (H, W) in [0, 1]. Dispatches to the serial or
/// parallel path based on `PARALLEL_THRESHOLD` (counted in pixels, H*W,
/// not elements).
pub fn luminosity_blend_core(backdrop_rgb: ArrayView3<f32>, luminosity: ArrayView2<f32>, out: &mut Array3<f32>) {
    let (h, w, _) = backdrop_rgb.dim();
    if h * w >= PARALLEL_THRESHOLD {
        luminosity_blend_parallel(backdrop_rgb, luminosity, out);
    } else {
        luminosity_blend_serial(backdrop_rgb, luminosity, out);
    }
}
