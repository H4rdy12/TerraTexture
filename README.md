# TerraTexture

DEM curvature analysis and soft-light / luminosity-blended shaded relief
visualization.

Computes **profile curvature** (rate of change of slope along the
direction of steepest descent -- controls flow acceleration/deceleration)
and **planform curvature** (curvature of contour lines, perpendicular to
slope direction -- controls flow convergence/divergence) following
Zevenbergen & Thorne (1987), then blends them with a standard analytical
hillshade using Photoshop-style **soft light** and **luminosity** blend
modes to make ridges and channels pop out of the relief -- optionally
draped over real basemap imagery.

## Install

This project is managed with [uv](https://docs.astral.sh/uv/). Install
uv itself first if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # or: pipx install uv / brew install uv
```

Then, from the repo root:

```bash
uv sync                                    # core deps only (numpy/scipy/matplotlib)
uv sync --extra raster                     # + loading real DEM files (rasterio)
uv sync --extra basemap                    # + draping relief over basemap imagery (rasterio + contextily)
uv sync --group dev                        # + pytest, flake8 (dev tooling)
uv sync --group dev --extra raster         # dev tooling + raster extras together (typical local setup)
```

`uv sync` creates/updates a `.venv/` in the repo root and a `uv.lock`
lockfile pinning exact versions -- commit `uv.lock` so CI and everyone
on the project resolve identical dependency versions. Run anything
inside that environment with `uv run`, e.g. `uv run pytest` or
`uv run dem-relief curvature`, without manually activating the venv.

The core install (numpy/scipy/matplotlib only) is enough for
`TerraTexture.derivatives`, `TerraTexture.blend`, `TerraTexture.stretch`, and
`TerraTexture.plotting` on an in-memory array or the built-in synthetic
demo DEM. Real raster I/O and basemap imagery need the optional extras
above.

ArcticDEM STAC querying (`TerraTexture.sources.ArcticDEM_stac`)
additionally needs `DEMSquad_STAC`, which isn't on PyPI -- install it
manually and either add it to your `PYTHONPATH` or pass
`demsquad_path=` pointing at its location.

## Quickstart

```bash
uv run python examples/quickstart.py
```

```python
from TerraTexture.io import load_dem
from TerraTexture.plotting import plot_dem_curvature_softlight

dem, cellsize = load_dem(None)  # synthetic demo DEM
fig, axes, results = plot_dem_curvature_softlight(dem, cellsize=cellsize)
```

Or via the CLI:

```bash
uv run dem-relief curvature                       # synthetic demo DEM
uv run dem-relief curvature path/to/dem.tif --out relief.png
uv run dem-relief basemap path/to/arcticdem_tile.tar.gz --out relief.png
```

## Draping relief over basemap imagery

```python
from TerraTexture.basemap import plot_dem_basemap_luminosity_relief

fig, ax, layers = plot_dem_basemap_luminosity_relief(
    dem_path="path/to/15_44_32m_v4.1.tar.gz",   # ArcticDEM mosaic tile, or any GeoTIFF
    out_png="relief.png",
)
```

See `examples/arcticdem_basemap.py` for a runnable version (pass `--dem`
or set `TerraTexture_DEMO_TILE`).

## Burning scientific data onto relief

```python
from TerraTexture.basemap import add_relief_basemap
from TerraTexture.overlay import burn_data_onto_relief

layers = add_relief_basemap(ax, aoi_bounds=my_bounds)
composite, mappable = burn_data_onto_relief(
    layers, dhdt_data, cmap="RdYlBu_r", vmin=-6, vmax=6,
)
ax.imshow(composite, extent=layers["extent"])
```

## Package layout

| Module              | Purpose                                             | Extra dependencies      |
|---------------------|------------------------------------------------------|--------------------------|
| `TerraTexture.io`         | Load DEM rasters, merge multi-tile mosaics       | `rasterio`               |
| `TerraTexture.sources`    | ArcticDEM STAC queries, synthetic demo GeoTIFF   | `rasterio`, `DEMSquad_STAC` (STAC only) |
| `TerraTexture.derivatives`| Profile/planform curvature, hillshade            | none (numpy/scipy)       |
| `TerraTexture.blend`      | Soft light & luminosity blend modes              | none (numpy)             |
| `TerraTexture.stretch`    | Percentile / std-dev stretches, resampling       | none (numpy/scipy)       |
| `TerraTexture.plotting`   | 6-panel curvature + relief summary figure        | `matplotlib`             |
| `TerraTexture.basemap`    | Relief draped over basemap imagery               | `rasterio`, `contextily` |
| `TerraTexture.overlay`    | Burn a data raster onto relief via luminosity    | `rasterio`               |
| `TerraTexture.cli`        | Command-line entry point                         | -                         |

`TerraTexture.derivatives` and `TerraTexture.blend` have zero geospatial
dependencies by design -- they're the modules to target first for a
Numba or Rust-accelerated implementation, since they're pure elementwise
array math with no I/O.

## Testing

```bash
uv run pytest
```

`tests/test_derivatives.py` and `tests/test_blend.py` run without
rasterio installed. `tests/test_io.py` is skipped automatically (via
`pytest.importorskip`) if rasterio isn't in the synced environment --
run `uv sync --group dev --extra raster` first to include it.

## Linting

Python (flake8, config in `pyproject.toml`'s `[tool.flake8]`):

```bash
uv run flake8 src tests examples
```

Rust (once `rust/TerraTexture_rs` has real code beyond the stub):

```bash
cd rust
cargo fmt --check
cargo clippy --all-targets -- -D warnings
```

Both run in CI on every push/PR (see `.github/workflows/test.yml`).

## Background

See [`docs/formulas.md`](docs/formulas.md) for the full curvature
derivation, sign conventions, and blend-mode math.

## License

MIT
