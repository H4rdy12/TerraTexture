//! Starting point for a PyO3-accelerated version of `terra_texture.blend`.
//!
//! Not yet imported by the Python package -- `terra_texture/blend.py` still
//! uses the pure-numpy implementation. Build this with `maturin develop`
//! (from this `rust/` directory) to get an importable `terra_texture_rs`
//! module, then wire it in behind a try/except ImportError fallback in
//! `blend.py` once it's validated against `tests/test_blend.py`'s cases.
//!
//! This single `soft_light` kernel is meant as a template for also
//! porting `luminosity_blend`/`_clip_color` and the curvature formula in
//! `derivatives.py`, which are the other pure-elementwise hot paths.

use ndarray::Zip;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyfunction]
fn soft_light<'py>(
    py: Python<'py>,
    base: PyReadonlyArray2<'py, f32>,
    blend: PyReadonlyArray2<'py, f32>,
) -> Bound<'py, PyArray2<f32>> {
    let a = base.as_array();
    let b = blend.as_array();
    let mut out = ndarray::Array2::<f32>::zeros(a.raw_dim());

    Zip::from(&mut out)
        .and(&a)
        .and(&b)
        .par_for_each(|o, &a, &b| {
            *o = if b <= 0.5 {
                2.0 * a * b + a * a * (1.0 - 2.0 * b)
            } else {
                2.0 * a * (1.0 - b) + a.max(0.0).sqrt() * (2.0 * b - 1.0)
            };
            *o = o.clamp(0.0, 1.0);
        });

    out.into_pyarray_bound(py)
}

#[pymodule]
fn terra_texture_rs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(soft_light, m)?)?;
    Ok(())
}
