//! Photoshop-style soft light blend, 2-D (H, W) and 3-D (H, W, C).
//!
//! Matches `blend.py`'s `soft_light()` exactly (verified against it in
//! `tests/test_blend.py`'s Rust-parity tests, when the extension is
//! built).
//!
//! The `*_serial`/`*_parallel` variants are public purely so
//! `benches/blend_bench.rs` can compare them directly at every array
//! size, independent of whatever `PARALLEL_THRESHOLD` currently guesses.
//! The `*_core` functions are the actual dispatch entry points
//! everything else should call.

use ndarray::{Array2, Array3, ArrayView2, ArrayView3, Zip};

use crate::common::PARALLEL_THRESHOLD;

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

// ---------------------------------------------------------------------------
// 2-D (H, W)
// ---------------------------------------------------------------------------

/// Serial (single-threaded) soft light -- see module docs for why this
/// is public.
pub fn soft_light_serial(a: ArrayView2<f32>, b: ArrayView2<f32>, out: &mut Array2<f32>) {
    Zip::from(out)
        .and(&a)
        .and(&b)
        .for_each(|o, &a, &b| *o = soft_light_pixel(a, b));
}

/// Parallel (rayon) soft light -- see module docs for why this is public.
pub fn soft_light_parallel(a: ArrayView2<f32>, b: ArrayView2<f32>, out: &mut Array2<f32>) {
    Zip::from(out)
        .and(&a)
        .and(&b)
        .par_for_each(|o, &a, &b| *o = soft_light_pixel(a, b));
}

/// Pure computation: soft light blend. `base`/`blend` arrays in [0, 1],
/// same shape. Dispatches to the serial or parallel path based on
/// `PARALLEL_THRESHOLD`.
pub fn soft_light_core(a: ArrayView2<f32>, b: ArrayView2<f32>, out: &mut Array2<f32>) {
    if a.len() >= PARALLEL_THRESHOLD {
        soft_light_parallel(a, b, out);
    } else {
        soft_light_serial(a, b, out);
    }
}

// ---------------------------------------------------------------------------
// 3-D (H, W, C)
// ---------------------------------------------------------------------------
//
// Same as the 2-D path but over 3-D (H, W, C) arrays -- e.g. RGB or RGBA
// imagery -- in ONE pass instead of the Python-side dispatch calling the
// 2-D kernel once per channel. `soft_light_pixel` is already
// channel-agnostic (a plain f32 -> f32 elementwise formula with no
// cross-channel interaction), so this is the exact same per-element math
// as the 2-D path; the only reason this exists as a separate kernel
// rather than just reshaping and reusing the 2-D one is to avoid the
// per-channel `np.ascontiguousarray()` copy the Python-side loop needed
// (each `a[..., c]` channel slice of a contiguous (H, W, C) array is
// itself non-contiguous). Operating on the whole (H, W, C) buffer
// directly, already contiguous, skips that entirely -- one input read,
// one output write, per element, with no intermediate per-channel arrays
// at all.

/// Serial 3-D soft light -- see module docs for why this is public.
pub fn soft_light_rgb_serial(a: ArrayView3<f32>, b: ArrayView3<f32>, out: &mut Array3<f32>) {
    Zip::from(out)
        .and(&a)
        .and(&b)
        .for_each(|o, &a, &b| *o = soft_light_pixel(a, b));
}

/// Parallel 3-D soft light -- see module docs for why this is public.
pub fn soft_light_rgb_parallel(a: ArrayView3<f32>, b: ArrayView3<f32>, out: &mut Array3<f32>) {
    Zip::from(out)
        .and(&a)
        .and(&b)
        .par_for_each(|o, &a, &b| *o = soft_light_pixel(a, b));
}

/// Pure computation: soft light over (H, W, C). Dispatches to the serial
/// or parallel path based on `PARALLEL_THRESHOLD`.
pub fn soft_light_rgb_core(a: ArrayView3<f32>, b: ArrayView3<f32>, out: &mut Array3<f32>) {
    if a.len() >= PARALLEL_THRESHOLD {
        soft_light_rgb_parallel(a, b, out);
    } else {
        soft_light_rgb_serial(a, b, out);
    }
}