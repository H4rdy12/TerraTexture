# terra_texture_rs

Compiled Rust extension for [TerraTexture](https://github.com/H4rdy12/TerraTexture): `pyo3`/`numpy`/`ndarray`-backed elementwise blend and derivatives math (soft-light / luminosity blending, curvature derivatives).

This package isn't meant to be installed on its own — it's an optional accelerated backend for `TerraTexture`, installed via:

```bash
pip install terratexture[rust]
```

`terra_texture.blend` and `terra_texture.derivatives` fall back to a pure-numpy implementation automatically if this extension isn't installed, so `terratexture` works fully without it. Installing `terra_texture_rs` swaps in the compiled version for the elementwise-math hot path.

See the main [TerraTexture repository](https://github.com/H4rdy12/TerraTexture) for documentation, the CLI, and the raster/basemap/STAC workflow this extension supports.

## Development

Built with [maturin](https://www.maturin.rs/). From the `rust/` directory of the TerraTexture repo, with a venv active:

```bash
maturin develop --release
```

Run `cargo bench` to tune `PARALLEL_THRESHOLD` in `src/lib.rs` against real measurements.

## License

MIT
