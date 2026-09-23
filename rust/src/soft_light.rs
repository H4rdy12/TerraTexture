//! Photoshop-style soft light blend, over 2-D (H, W) and 3-D (H, W, C)
//! arrays.
//!
//! Matches `blend.py`'s `soft_light()` exactly (verified in
//! `tests/test_blend.py`'s Rust-parity tests, when the extension is
//! built). Per element, with base `a` and blend `b`:
//!
//! ```text
//! b <= 0.5:  2ab + a²(1 - 2b)
//! b >  0.5:  2a(1 - b) + sqrt(clamp(a, 0, 1)) · (2b - 1)
//! ```
//!
//! and the result is clamped to \[0, 1\].
//!
//! # Which function to call
//!
//! Call [`soft_light_core`] (2-D) or [`soft_light_rgb_core`] (3-D). They
//! pick the serial or parallel path using
//! [`PARALLEL_THRESHOLD`].
//!
//! The `*_serial`/`*_parallel` variants are public only so
//! `benches/blend_bench.rs` can time both paths at every array size to
//! tune that threshold. Benches are separate crates and can only call
//! `pub` items.

use ndarray::{Array2, Array3, ArrayView2, ArrayView3, Zip};

use crate::common::PARALLEL_THRESHOLD;

/// Soft light for one element: base `a`, blend `b`, both `f32` nominally
/// in \[0, 1\]. Returns an `f32` clamped to \[0, 1\]. See the module docs
/// for the formula.
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

/// Soft light blend of two 2-D arrays, single-threaded.
///
/// Same arguments, output and panics as [`soft_light_core`], but always
/// runs serially regardless of size. Public for benchmarking only (see
/// module docs).
pub fn soft_light_serial(a: ArrayView2<f32>, b: ArrayView2<f32>, out: &mut Array2<f32>) {
    Zip::from(out)
        .and(&a)
        .and(&b)
        .for_each(|o, &a, &b| *o = soft_light_pixel(a, b));
}

/// Soft light blend of two 2-D arrays, parallelised with rayon.
///
/// Same arguments, output and panics as [`soft_light_core`], but always
/// runs in parallel regardless of size. Public for benchmarking only
/// (see module docs).
pub fn soft_light_parallel(a: ArrayView2<f32>, b: ArrayView2<f32>, out: &mut Array2<f32>) {
    Zip::from(out)
        .and(&a)
        .and(&b)
        .par_for_each(|o, &a, &b| *o = soft_light_pixel(a, b));
}

/// Soft light blend of two 2-D arrays. The entry point for 2-D inputs.
///
/// # Arguments
///
/// * `a` - `ArrayView2<f32>`, shape (H, W): the base layer, values
///   nominally in \[0, 1\]. Any memory layout.
/// * `b` - `ArrayView2<f32>`, shape (H, W): the blend layer, values
///   nominally in \[0, 1\]. Any memory layout.
/// * `out` - `&mut Array2<f32>`, shape (H, W): overwritten with the
///   result, every element in \[0, 1\]. Its previous contents are ignored.
///
/// Runs serially below [`PARALLEL_THRESHOLD`]
/// elements and in parallel at or above it.
///
/// # Panics
///
/// If `a`, `b` and `out` do not all have the same shape.
///
/// # Example
///
/// ```
/// use ndarray::{array, Array2};
/// use terra_texture_rs::soft_light_core;
///
/// let base = array![[0.2_f32, 0.8], [0.5, 1.0]];
/// let blend = array![[0.5_f32, 0.5], [0.0, 1.0]];
/// let mut out = Array2::<f32>::zeros(base.raw_dim());
/// soft_light_core(base.view(), blend.view(), &mut out);
///
/// // blend = 0.5 leaves the base unchanged
/// assert!((out[[0, 0]] - 0.2).abs() < 1e-6);
/// assert!(out.iter().all(|&v| (0.0..=1.0).contains(&v)));
/// ```
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
// Same per-element math as the 2-D path (soft_light_pixel has no
// cross-channel interaction). This exists as a separate kernel only to
// avoid the per-channel np.ascontiguousarray() copy the Python-side loop
// needed: each a[..., c] slice of a contiguous (H, W, C) array is itself
// non-contiguous. Working on the whole buffer directly means one input
// read and one output write per element, with no per-channel arrays.

/// Soft light blend of two 3-D arrays, single-threaded.
///
/// Same arguments, output and panics as [`soft_light_rgb_core`], but
/// always runs serially. Public for benchmarking only (see module docs).
pub fn soft_light_rgb_serial(a: ArrayView3<f32>, b: ArrayView3<f32>, out: &mut Array3<f32>) {
    Zip::from(out)
        .and(&a)
        .and(&b)
        .for_each(|o, &a, &b| *o = soft_light_pixel(a, b));
}

/// Soft light blend of two 3-D arrays, parallelised with rayon.
///
/// Same arguments, output and panics as [`soft_light_rgb_core`], but
/// always runs in parallel. Public for benchmarking only (see module
/// docs).
pub fn soft_light_rgb_parallel(a: ArrayView3<f32>, b: ArrayView3<f32>, out: &mut Array3<f32>) {
    Zip::from(out)
        .and(&a)
        .and(&b)
        .par_for_each(|o, &a, &b| *o = soft_light_pixel(a, b));
}

/// Soft light blend of two 3-D (H, W, C) arrays, e.g. RGB or RGBA
/// imagery, in a single pass. The entry point for multi-channel inputs.
///
/// Every channel is blended independently with the same formula as
/// [`soft_light_core`]; any channel count C works.
///
/// # Arguments
///
/// * `a` - `ArrayView3<f32>`, shape (H, W, C): the base layer, values
///   nominally in \[0, 1\]. Any memory layout.
/// * `b` - `ArrayView3<f32>`, shape (H, W, C): the blend layer, values
///   nominally in \[0, 1\]. Any memory layout.
/// * `out` - `&mut Array3<f32>`, shape (H, W, C): overwritten with the
///   result, every element in \[0, 1\].
///
/// Runs serially below [`PARALLEL_THRESHOLD`]
/// elements (H × W × C) and in parallel at or above it.
///
/// # Panics
///
/// If `a`, `b` and `out` do not all have the same shape.
pub fn soft_light_rgb_core(a: ArrayView3<f32>, b: ArrayView3<f32>, out: &mut Array3<f32>) {
    if a.len() >= PARALLEL_THRESHOLD {
        soft_light_rgb_parallel(a, b, out);
    } else {
        soft_light_rgb_serial(a, b, out);
    }
}
