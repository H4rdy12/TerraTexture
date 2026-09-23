//! Pieces shared across kernel modules: the serial/parallel dispatch
//! threshold, and the numpy-compatible 2-D gradient used by both
//! `curvature` and `hillshade`.

use ndarray::{Array2, ArrayView2};

// TODO(benchmark): this is a guess, not a measurement. Run `cargo bench`
// and replace it with whatever `blend_bench.rs` actually finds as the
// point where `par_for_each` starts winning over a serial loop on the
// target hardware. 256x256 = 65_536 is picked only because it "feels"
// like a plausible order of magnitude for rayon's dispatch overhead to
// have been amortized -- treat it as unverified until benchmarked.
pub const PARALLEL_THRESHOLD: usize = 65_536;

/// numpy-compatible `np.gradient(arr, spacing)` along axis 0 (rows) and
/// axis 1 (columns), default `edge_order=1`: one-sided forward/backward
/// difference at the first/last index of each axis, second-order central
/// difference everywhere in between. Returns (d/d_axis0, d/d_axis1) --
/// same order as numpy's `zy, zx = np.gradient(dem, cellsize)`.
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