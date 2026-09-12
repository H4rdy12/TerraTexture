# TerraTexture

Tool to generate textured basemaps using open source DEMs and basemap
imagery from Contextily.

Texture is derived from DEM curvature and visualised using soft-light /
luminosity-blended techniques.

![Code Quality](https://github.com/H4rdy12/TerraTexture/actions/workflows/ci.yml/badge.svg)

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
`uv run terratexture curvature`, without manually activating the venv.

The core install (numpy/scipy/matplotlib only) is enough for
`terra_texture.derivatives`, `terra_texture.blend`, `terra_texture.stretch`, and
`terra_texture.plotting` on an in-memory array or the built-in synthetic
demo DEM. Real raster I/O and basemap imagery need the optional extras
above.

Fetching open-source ArcticDEM/REMA mosaic tiles for an AOI
(`terra_texture.sources`) additionally needs `requests` -- included in the
`raster`/`basemap` extras above. No signup, API key, or local software
required: it queries PGC's public STAC API directly.

```python
from terra_texture.sources import arcticdem_mosaic, rema_mosaic

# bounds in EPSG:3413 (ArcticDEM's native CRS); covers the Arctic,
# including Greenland
dem, cellsize, transform, crs = arcticdem_mosaic(
    bounds=(-200000, -2300000, 0, -2100000), resolution=32,
)

# bounds in EPSG:4326 for REMA (Antarctica)
dem, cellsize, transform, crs = rema_mosaic(
    bounds=(-70, -75, -65, -73), resolution=32, bbox_crs="EPSG:4326",
)
```

`plot_dem_basemap_luminosity_relief(aoi_bounds=..., dem_product="arcticdem" | "rema")`
uses the same functions under the hood -- see "Draping relief over
basemap imagery" below.

## Open-data DEM sources
 
`terra_texture.sources` (needs `requests`, included in the
`raster`/`basemap` extras above) fetches real elevation data for an AOI
from public, unauthenticated STAC catalogs -- no signup, no API key, no
local software to install. It supports two catalogs today, and each
needs a genuinely different query approach, because STAC catalogs come
in two practically-different flavours:
 
- **Dynamic STAC APIs** implement the STAC Item Search extension: a
  `POST /search` endpoint that accepts a bbox and does the spatial
  filtering server-side. Fast, and scales to catalogs with millions of
  items.
- **Static (or search-less) STAC catalogs** are just a crawlable tree of
  Catalog → Collection → Item JSON documents with no `/search`
  endpoint at all. Querying means picking a Collection first, then
  fetching its items and filtering by bbox yourself.
`sources.py` has one engine for each, and the product-specific functions
below are thin wrappers around whichever engine fits their catalog.
 
### PGC (ArcticDEM / REMA): dynamic STAC search
 
The [Polar Geospatial Center](https://www.pgc.umn.edu) (PGC, University
of Minnesota) publishes ArcticDEM (covers the Arctic, including
Greenland) and REMA (Antarctica) as a fully dynamic, public STAC API at
`https://stac.pgc.umn.edu/api/v1` -- see PGC's own
[STAC access guide](https://www.pgc.umn.edu/guides/stereo-derived-elevation-models/stac-access-static-and-dynamic-api/).
`stac_search()` POSTs a bbox + collection ID and follows pagination;
`arcticdem_mosaic()`/`rema_mosaic()` wrap it with the right collection
naming for each product's mosaic resolutions (2m/10m/32m):
 
```python
from terra_texture.sources import arcticdem_mosaic, rema_mosaic
 
# bounds in EPSG:3413 (ArcticDEM's native CRS); covers the Arctic,
# including Greenland
dem, cellsize, transform, crs = arcticdem_mosaic(
    bounds=(-200000, -2300000, 0, -2100000), resolution=32,
)
 
# bounds in EPSG:4326 for REMA (Antarctica)
dem, cellsize, transform, crs = rema_mosaic(
    bounds=(-70, -75, -65, -73), resolution=32, bbox_crs="EPSG:4326",
)
```
 
`plot_dem_basemap_luminosity_relief(aoi_bounds=..., dem_product="arcticdem" | "rema")`
uses the same functions under the hood -- see "Draping relief over
basemap imagery" below.
 
### OpenTopography: static catalog, choosing from a large collection
 
OpenTopography publishes a much larger, more heterogeneous set of raster
DEM datasets (global products like SRTM/COP30/NASADEM alongside many
regional lidar-derived DEMs) as a **static** STAC catalog with no search
endpoint at all (`https://portal.opentopography.org/stac/raster_catalog.json`).
`list_stac_collections()` crawls its root Catalog for the available
Collections (datasets) so you can pick one; `stac_collection_items()`
then fetches that Collection's items and filters them by bbox
intersection client-side, since the server can't do it for you. The
workflow is "list what's available, then fetch from whichever one you
pick":
 
```python
from terra_texture.sources import list_stac_collections, opentopography_mosaic, OT_STAC_ROOT
 
for collection in list_stac_collections(OT_STAC_ROOT):
    print(collection["id"])
 
dem, cellsize, transform, crs = opentopography_mosaic(
    "SRTM GL1",                      # collection id from the list above
    bounds=(-121.8, 36.5, -121.6, 36.7),  # (min_lon, min_lat, max_lon, max_lat)
)
```
 
Or via the CLI:
 
```bash
terratexture opentopography list
terratexture opentopography fetch "SRTM GL1" -121.8 36.5 -121.6 36.7 --out srtm.tif
```
 
`asset_key` (default `"data"`) controls which item asset is treated as
the DEM -- if it's wrong for a given collection, you'll get a `KeyError`
listing that collection's actual asset keys, so it's self-correcting.
 
## Quickstart
 
```bash
uv run python examples/quickstart.py
```
 
```python
from terra_texture.io import load_dem
from terra_texture.plotting import plot_dem_curvature_softlight
 
dem, cellsize = load_dem(None)  # synthetic demo DEM
fig, axes, results = plot_dem_curvature_softlight(dem, cellsize=cellsize)
```
 
Or via the CLI:
 
```bash
uv run terratexture curvature                       # synthetic demo DEM
uv run terratexture curvature path/to/dem.tif --out relief.png
uv run terratexture basemap path/to/arcticdem_tile.tar.gz --out relief.png
uv run terratexture opentopography list             # browse available datasets
uv run terratexture opentopography fetch "SRTM GL1" -121.8 36.5 -121.6 36.7 --out srtm.tif
```
 
## Draping relief over basemap imagery
 
```python
from terra_texture.basemap import plot_dem_basemap_luminosity_relief
 
fig, ax, layers = plot_dem_basemap_luminosity_relief(
    dem_path="path/to/15_44_32m_v4.1.tar.gz",   # ArcticDEM mosaic tile, or any GeoTIFF
    out_png="relief.png",
)
 
# or, without any local file, straight from PGC's public STAC API:
fig, ax, layers = plot_dem_basemap_luminosity_relief(
    aoi_bounds=(-200000, -2300000, 0, -2100000),
    dem_product="arcticdem",   # or "rema" for Antarctica
    arcticdem_resolution=32,
    out_png="relief.png",
)
```
 
See `examples/arcticdem_basemap.py` for a runnable version (pass `--dem`
or set `TERRA_TEXTURE_DEMO_TILE`).
 
## Burning scientific data onto relief
 
```python
from terra_texture.basemap import add_relief_basemap
from terra_texture.overlay import burn_data_onto_relief
 
layers = add_relief_basemap(ax, aoi_bounds=my_bounds)
composite, mappable = burn_data_onto_relief(
    layers, dhdt_data, cmap="RdYlBu_r", vmin=-6, vmax=6,
)
ax.imshow(composite, extent=layers["extent"])
```
 
## Package layout
 
| Module              | Purpose                                             | Extra dependencies      |
|---------------------|------------------------------------------------------|--------------------------|
| `terra_texture.io`         | Load DEM rasters, merge multi-tile mosaics       | `rasterio`               |
| `terra_texture.sources`    | Generic STAC engines (dynamic search + static-catalog crawl) + ArcticDEM/REMA/OpenTopography product methods, synthetic demo GeoTIFF | `rasterio`, `requests` |
| `terra_texture.derivatives`| Profile/planform curvature, hillshade            | none (numpy/scipy)       |
| `terra_texture.blend`      | Soft light & luminosity blend modes              | none (numpy)             |
| `terra_texture.stretch`    | Percentile / std-dev stretches, resampling       | none (numpy/scipy)       |
| `terra_texture.plotting`   | 6-panel curvature + relief summary figure        | `matplotlib`             |
| `terra_texture.basemap`    | Relief draped over basemap imagery               | `rasterio`, `contextily` |
| `terra_texture.overlay`    | Burn a data raster onto relief via luminosity    | `rasterio`               |
| `terra_texture.cli`        | Command-line entry point                         | -                         |
 
`terra_texture.derivatives` and `terra_texture.blend` have zero geospatial
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
 
Rust (once `rust/terra_texture_rs` has real code beyond the stub):
 
```bash
cd rust
cargo fmt --check
cargo clippy --all-targets -- -D warnings
```
 
Both run in CI on every push/PR (see `.github/workflows/ci.yml`).
 
## CI/CD
 
`.github/workflows/ci.yml` runs on every push and PR:
 
| Job            | What it checks                                              |
|----------------|--------------------------------------------------------------|
| `build-python` | `uv build` produces a valid sdist + wheel                    |
| `build-rust`   | `cargo build --release` compiles `rust/terra_texture_rs`         |
| `lint-python`  | `flake8` on `src`, `tests`, `examples`                        |
| `lint-rust`    | `cargo fmt --check` + `cargo clippy -- -D warnings`            |
| `test-python`  | `pytest` across Python 3.10/3.11/3.12                          |
| `all-checks`   | Aggregator: fails if any job above failed/was cancelled       |
 
`all-checks` exists so branch protection only needs to require **one**
named check, instead of every job being re-listed (and re-edited) in
GitHub's settings every time a job is added, renamed, or removed here.
 
`.github/workflows/build-wheels.yml` is a separate, currently-disabled
workflow for building/publishing Rust extension wheels on version tags,
once `rust/terra_texture_rs` has real kernels in it.
 
## Branch protection
 
GitHub repo settings, not something a workflow file can enforce, so
this isn't automatic just by adding `ci.yml` -- it has to be configured
once on the repo itself:
 
```bash
./scripts/setup-branch-protection.sh H4rdy12/TerraTexture
```
 
This requires the [GitHub CLI](https://cli.github.com/) (`gh auth
login` first) and admin rights on the repo. It configures `main` so
that:
 
- direct pushes are blocked -- all changes go through a PR
- the `all-checks` CI job must pass before merging
- at least 1 approving review is required, dismissed on new commits
- [`CODEOWNERS`](.github/CODEOWNERS) review is required for matched paths
- the branch must be up to date with `main` before merging
- force-pushes and branch deletion are blocked
`all-checks` won't be selectable as a required check until CI has run
at least once against `main` -- push the repo, open one PR to trigger
the workflow, then run the script (or re-run it if GitHub didn't pick
up the context the first time).
 
Prefer the UI instead? Settings → Branches → Add branch protection rule
→ `main`, and tick the equivalent boxes by hand; the script above is
just a faster, repeatable way to do the same thing.
 
[`.github/CODEOWNERS`](.github/CODEOWNERS) is already set to `@H4rdy12`
— update it if that changes (e.g. adding more maintainers).
 
## Background
 
See [`docs/formulas.md`](docs/formulas.md) for the full curvature
derivation, sign conventions, and blend-mode math.
 
## License
 
MIT
 