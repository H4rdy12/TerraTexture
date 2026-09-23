//! The PyO3 layer: every `#[pyfunction]` wrapper plus the `#[pymodule]`.
//! This is the only module in the crate that imports `pyo3` or `numpy`.
//!
//! Each wrapper does the same three things: unwrap the numpy arrays into
//! `ndarray` views, call the matching `*_core` function, wrap the result
//! back into a numpy array.
//!
//! ## Releasing the GIL
//!
//! The core call in every wrapper runs inside `py.detach(...)`, which
//! releases the GIL for the duration. The core functions may fan out
//! across rayon's thread pool, and none of that work touches any Python
//! object (inputs/outputs are plain ndarray views/buffers, not `PyAny`),
//! so there's no reason another Python thread (e.g. a contextily
//! tile-fetch thread) should be blocked while it runs. The GIL is
//! re-acquired automatically before `detach` returns, before the output
//! gets wrapped back into a `PyArray`.
//!
//! Core functions are imported by name rather than via their modules
//! (`use crate::soft_light;`) because the wrappers here share those
//! modules' names, and `#[pyfunction]` generates helper items under the
//! function's name.

use ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;

use crate::curvature::curvatures_core;
use crate::hillshade::hillshade_core;
use crate::luminosity_blend::luminosity_blend_core;
use crate::soft_light::{soft_light_core, soft_light_rgb_core};
use crate::stretch::stretch_std_core;

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

#[pyfunction]
fn stretch_std<'py>(py: Python<'py>, arr: PyReadonlyArray2<'py, f32>, n_std: f32) -> Bound<'py, PyArray2<f32>> {
    let a = arr.as_array();
    let mut out = Array2::<f32>::zeros(a.raw_dim());
    py.detach(|| stretch_std_core(a, n_std, &mut out));
    out.into_pyarray(py)
}

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