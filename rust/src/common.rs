//! Pieces shared across kernel modules: the serial/parallel dispatch
//! threshold, and the numpy-compatible 2-D gradient used by both
//! `curvature` and `hillshade`.

use ndarray::{Array2, ArrayView2};

/// Element count at or above which kernels switch from a serial loop to
/// rayon's parallel iteration.
///
/// Type: `usize`, counted in array elements. For
/// [`luminosity_blend_core`](crate::luminosity_blend_core) it is counted
/// in pixels (H × W) rather than elements (H × W × 3).
///
/// **Provisional placeholder, not a measured value.** 256 × 256 = 65,536
/// was picked only as a plausible order of magnitude for rayon's
/// dispatch overhead to be amortized. Run `cargo bench` and replace it
/// with the crossover point `benches/blend_bench.rs` actually measures
/// on the target hardware.
// TODO(benchmark): replace with a measured value.
pub const PARALLEL_THRESHOLD: usize = 65_536;

/// numpy-compatible `np.gradient(arr, spacing)` along both axes.
///
/// Uses numpy's default `edge_order=1`: a one-sided forward/backward
/// difference at the first/last index of each axis, and a second-order
/// central difference everywhere in between.
///
/// # Arguments
///
/// * `arr` - `ArrayView2<f32>`, shape (H, W). Any memory layout.
/// * `spacing` - `f32`, grid spacing (the DEM cell size), applied to
///   both axes.
///
/// # Returns
///
/// `(d_axis0, d_axis1)`: two newly allocated, C-contiguous
/// `Array2<f32>` of shape (H, W), the derivative down the rows and across
/// the columns respectively. Same order as numpy's
/// `zy, zx = np.gradient(dem, cellsize)`.
///
/// # Differences from numpy
///
/// If an axis has length 1, this returns zeros for that axis. numpy
/// instead raises `ValueError` (it needs at least `edge_order + 1 = 2`
/// elements per axis).
///
/// # Panics
///
/// If either axis has length 0 while the other does not (out-of-bounds
/// index on the empty axis).
pub(crate) fn gradient2d(arr: ArrayView2<f32>, spacing: f32) -> (Array2<f32>, Array2<f32>) {
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
