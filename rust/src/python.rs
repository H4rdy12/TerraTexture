//! The PyO3 layer: every `#[pyfunction]` wrapper plus the `#[pymodule]`.
//! This is the only module in the crate that imports `pyo3` or `numpy`.
//!
//! Each wrapper unwraps the numpy arrays into `ndarray` views, calls the
//! matching `*_core` function, and wraps the result into a new numpy
//! array.
//!
//! # Python docstrings
//!
//! PyO3 turns the `///` comments on each `#[pyfunction]` into that
//! function's Python `__doc__`, so `help(terra_texture_rs.hillshade)`
//! shows them. They are therefore written for Python callers, in numpy
//! docstring style, with numpy dtypes rather than Rust types. PyO3 also
//! generates the call signature, so the docstrings don't repeat it.
//!
//! # Types at the boundary
//!
//! Every array argument must be a `numpy.ndarray` of dtype `float32`.
//! Any other dtype (including `float64`) raises `TypeError`; the Python
//! side is responsible for `.astype(np.float32)`. Non-contiguous arrays
//! are accepted. Scalar arguments accept any Python `float` or `int`.
//!
//! A shape mismatch in the core function panics, which PyO3 surfaces in
//! Python as `pyo3_runtime.PanicException`. That class derives from
//! `BaseException`, not `Exception`, so `except Exception:` will not
//! catch it: validate shapes on the Python side before calling.
//!
//! # Releasing the GIL
//!
//! Each core call runs inside `py.detach(...)`, which releases the GIL
//! for the duration. The core functions may fan out across rayon's
//! thread pool and touch no Python objects, so there's no reason another
//! Python thread (e.g. a contextily tile-fetch thread) should be blocked
//! meanwhile. The GIL is re-acquired before `detach` returns.
//!
//! # Naming
//!
//! Core functions are imported by name rather than via their modules
//! (`use crate::soft_light;`) because the wrappers share those modules'
//! names, and `#[pyfunction]` generates helper items under the
//! function's name.

use ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;

use crate::curvature::curvatures_core;
use crate::hillshade::hillshade_core;
use crate::luminosity_blend::luminosity_blend_core;
use crate::soft_light::{soft_light_core, soft_light_rgb_core};
use crate::stretch::stretch_std_core;

/// Photoshop-style soft light blend of two 2-D arrays.
///
/// Parameters
/// ----------
/// base : numpy.ndarray, float32, shape (H, W)
///     Base layer, values in [0, 1].
/// blend : numpy.ndarray, float32, shape (H, W)
///     Blend layer, values in [0, 1]. Same shape as `base`.
///
/// Returns
/// -------
/// numpy.ndarray, float32, shape (H, W)
///     Blended result, values in [0, 1].
#[pyfunction]
fn soft_light<'py>(
    py: Python<'py>,
    base: PyReadonlyArray2<'py, f32>,
    blend: PyReadonlyArray2<'py, f32>,
) -> Bound<'py, PyArray2<f32>> {
    let a = base.as_array();
    let b = blend.as_array();
    let mut out = Array2::<f32>::zeros(a.raw_dim());
    py.detach(|| soft_light_core(a, b, &mut out));
    out.into_pyarray(py)
}

/// Photoshop-style soft light blend of two multi-channel arrays, in one
/// pass over all channels.
///
/// Parameters
/// ----------
/// base : numpy.ndarray, float32, shape (H, W, C)
///     Base layer (e.g. RGB or RGBA), values in [0, 1].
/// blend : numpy.ndarray, float32, shape (H, W, C)
///     Blend layer, values in [0, 1]. Same shape as `base`.
///
/// Returns
/// -------
/// numpy.ndarray, float32, shape (H, W, C)
///     Blended result, values in [0, 1]. Each channel is blended
///     independently.
#[pyfunction]
fn soft_light_rgb<'py>(
    py: Python<'py>,
    base: PyReadonlyArray3<'py, f32>,
    blend: PyReadonlyArray3<'py, f32>,
) -> Bound<'py, PyArray3<f32>> {
    let a = base.as_array();
    let b = blend.as_array();
    let mut out = Array3::<f32>::zeros(a.raw_dim());
    py.detach(|| soft_light_rgb_core(a, b, &mut out));
    out.into_pyarray(py)
}

/// 'Luminosity' blend: give an RGB image a new luminance while keeping
/// its hue and saturation.
///
/// Parameters
/// ----------
/// backdrop_rgb : numpy.ndarray, float32, shape (H, W, 3)
///     Colour image in R, G, B order, values in [0, 1].
/// luminosity : numpy.ndarray, float32, shape (H, W)
///     Target luminance per pixel (e.g. a hillshade), values in [0, 1].
///
/// Returns
/// -------
/// numpy.ndarray, float32, shape (H, W, 3)
///     Blended RGB image, values in [0, 1].
#[pyfunction]
fn luminosity_blend<'py>(
    py: Python<'py>,
    backdrop_rgb: PyReadonlyArray3<'py, f32>,
    luminosity: PyReadonlyArray2<'py, f32>,
) -> Bound<'py, PyArray3<f32>> {
    let backdrop = backdrop_rgb.as_array();
    let lum = luminosity.as_array();
    let mut out = Array3::<f32>::zeros(backdrop.raw_dim());
    py.detach(|| luminosity_blend_core(backdrop, lum, &mut out));
    out.into_pyarray(py)
}

/// Profile and planform curvature of a DEM.
///
/// Parameters
/// ----------
/// dem : numpy.ndarray, float32, shape (H, W)
///     Elevations. Must contain no NaN: nan-fill first and re-mask the
///     results afterwards.
/// cellsize : float
///     Grid spacing, in the same units as the elevations.
///
/// Returns
/// -------
/// profile : numpy.ndarray, float32, shape (H, W)
///     Curvature in the direction of steepest slope.
/// planform : numpy.ndarray, float32, shape (H, W)
///     Curvature perpendicular to the slope.
///
/// Flat cells and non-finite results are 0.
#[pyfunction]
fn curvatures<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f32>,
    cellsize: f32,
) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>) {
    let d = dem.as_array();
    let mut profile = Array2::<f32>::zeros(d.raw_dim());
    let mut planform = Array2::<f32>::zeros(d.raw_dim());
    py.detach(|| curvatures_core(d, cellsize, &mut profile, &mut planform));
    (profile.into_pyarray(py), planform.into_pyarray(py))
}

/// Hillshade (simulated illumination) of a DEM.
///
/// Parameters
/// ----------
/// dem : numpy.ndarray, float32, shape (H, W)
///     Elevations. Must contain no NaN: nan-fill first and re-mask the
///     result afterwards.
/// cellsize : float
///     Grid spacing, in the same units as the elevations.
/// azimuth : float
///     Direction the light comes from, in degrees clockwise from north
///     (e.g. 315 for north-west).
/// altitude : float
///     Height of the light above the horizon, in degrees (0 to 90).
///
/// Returns
/// -------
/// numpy.ndarray, float32, shape (H, W)
///     Illumination in [0, 1]: 0 is fully shaded, 1 fully lit.
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
    py.detach(|| hillshade_core(d, cellsize, azimuth, altitude, &mut out));
    out.into_pyarray(py)
}

/// Contrast-stretch to [0, 1] using mean +/- n_std standard deviations.
///
/// Parameters
/// ----------
/// arr : numpy.ndarray, float32, shape (H, W)
///     Input values. NaN marks missing data and is ignored when
///     computing the mean and standard deviation.
/// n_std : float
///     Half-width of the stretch window, in standard deviations.
///
/// Returns
/// -------
/// numpy.ndarray, float32, shape (H, W)
///     Stretched values in [0, 1], with NaN wherever `arr` is NaN.
#[pyfunction]
fn stretch_std<'py>(py: Python<'py>, arr: PyReadonlyArray2<'py, f32>, n_std: f32) -> Bound<'py, PyArray2<f32>> {
    let a = arr.as_array();
    let mut out = Array2::<f32>::zeros(a.raw_dim());
    py.detach(|| stretch_std_core(a, n_std, &mut out));
    out.into_pyarray(py)
}

/// Compiled Rust kernels for TerraTexture: soft light and luminosity
/// blending, DEM curvature and hillshade, and standard-deviation
/// stretching. Optional accelerated backend: `terra_texture` falls back
/// to pure numpy when this module isn't built.
#[pymodule]
fn terra_texture_rs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(soft_light, m)?)?;
    m.add_function(wrap_pyfunction!(soft_light_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(luminosity_blend, m)?)?;
    m.add_function(wrap_pyfunction!(curvatures, m)?)?;
    m.add_function(wrap_pyfunction!(hillshade, m)?)?;
    m.add_function(wrap_pyfunction!(stretch_std, m)?)?;
    Ok(())
}
