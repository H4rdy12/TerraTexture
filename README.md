# TerraTexture

<div class="light-mode">
  <img align="center" width="400" height="250" src="https://raw.githubusercontent.com/H4rdy12/TerraTexture/de7f3f2bf93542a2d2e78ead0ac1d14290e6df9b/examples/resources/TerraTexture_logo_1.svg#gh-light-mode-only" />
</div>
<div class="dark-mode" style="display:none;">
  <img align="center" width="400" height="225" src="https://raw.githubusercontent.com/H4rdy12/TerraTexture/de7f3f2bf93542a2d2e78ead0ac1d14290e6df9b/examples/resources/TerraTexture_logo_1.svg#gh-dark-mode-only"-->
</div>

Tool to generate textured basemaps using open source DEMs and basemap
imagery from Contextily.

Texture is derived from DEM curvature and visualised using soft-light /
luminosity-blended techniques.

![Code Quality](https://github.com/H4rdy12/TerraTexture/actions/workflows/checks.yml/badge.svg)

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
uv add --dev ipykernel          # or: uv pip install ipykernel
uv run python -m ipykernel install --user --name terratexture --display-name "TerraTexture (uv)"
```

`uv sync` creates/updates a `.venv/` in the repo root and a `uv.lock`
lockfile pinning exact versions -- commit `uv.lock` so CI and everyone
on the project resolve identical dependency versions. Run anything
inside that environment with `uv run`, e.g. `uv run pytest` or
`uv run terratexture curvature`, without manually activating the venv.

The core install (numpy/scipy/matplotlib only) is enough for
`TerraTexture.derivatives`, `TerraTexture.blend`, `TerraTexture.stretch`, and
`TerraTexture.plotting` on an in-memory array or the built-in synthetic
demo DEM. Real raster I/O and basemap imagery need the optional extras
above.

Fetching open-source ArcticDEM/REMA mosaic tiles for an AOI
(`TerraTexture.sources`) additionally needs `requests` -- included in the
`raster`/`basemap` extras above. No signup, API key, or local software
required: it queries PGC's public STAC API directly.

```python
from TerraTexture.sources import arcticdem_mosaic, rema_mosaic

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

`TerraTexture.sources` (needs `requests`, included in the
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

#### Data attribution

ArcticDEM and REMA are published by PGC under **CC BY 4.0** with a
required [acknowledgement policy](https://www.pgc.umn.edu/guides/user-services/acknowledgement-policy/)
— if you use real data fetched via `arcticdem_mosaic()`/`rema_mosaic()`
in a publication, report, or map, you must cite PGC per their policy,
not just this tool. OpenTopography datasets carry their own per-collection
licensing and citation requirements (visible in each collection's STAC
metadata via `describe_stac_collection()`) — check before publishing
results derived from `opentopography_mosaic()`.
```python
from TerraTexture.sources import arcticdem_mosaic, rema_mosaic

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
from TerraTexture.sources import list_stac_collections, opentopography_mosaic, OT_STAC_ROOT

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
from TerraTexture.io import load_dem
from TerraTexture.plotting import plot_dem_curvature_softlight

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
from TerraTexture.basemap import plot_dem_basemap_luminosity_relief

fig, ax, layers = plot_dem_basemap_luminosity_relief(
    dem_path="path/to/15_44_32m_v4.1.tar.gz",   # ArcticDEM mosaic tile, or any GeoTIFF
    out_png="relief.png",
)

# or, without any local file, straight from PGC's public STAC API --
# aoi_bounds defaults to plain lon/lat (EPSG:4326):
fig, ax, layers = plot_dem_basemap_luminosity_relief(
    aoi_bounds=(-45, 68, -43, 69),   # lon/lat, covers part of Greenland
    dem_product="arcticdem",   # or "rema" for Antarctica
    arcticdem_resolution=32,
    out_png="relief.png",
)

# aoi_bounds_crs and target_crs are independent -- pass bounds in
# whatever CRS you already have them in, get the DEM back in whatever
# CRS you want, regardless of whether those two match:
fig, ax, layers = plot_dem_basemap_luminosity_relief(
    aoi_bounds=(-200000, -2300000, 0, -2100000),  # already in EPSG:3413
    aoi_bounds_crs="EPSG:3413",
    target_crs="EPSG:3413",
    dem_product="arcticdem",
    arcticdem_resolution=32,
    out_png="relief.png",
)
```

See `examples/arcticdem_basemap.py` for a runnable version (pass `--dem`
or set `TERRA_TEXTURE_DEMO_TILE`).

### Customizing the basemap and hillshade

`source`, `zoom`, `azimuth`, and `altitude` are ordinary keyword
arguments passed straight through to `contextily.bounds2img()` and the
hillshade calculation respectively -- nothing is hardcoded:

```python
import contextily as ctx
 
fig, ax, layers = plot_dem_basemap_luminosity_relief(
    dem_path="path/to/dem.tif",
    source=ctx.providers.Esri.WorldShadedRelief,  # any contextily/xyzservices provider
    zoom=12,                                       # int, or "auto" (default)
    azimuth=225,                                   # sun direction, degrees (default 315 = NW)
    altitude=30,                                   # sun elevation, degrees (default 45)
    out_png="relief.png",
)
```

`contextily.providers` has dozens of options nested by family
(`ctx.providers.<Family>.<Variant>`), and some require a personal API
key you'd have to supply yourself (their placeholder value is literally
`"<insert your API key here>"` until you do — including, as of writing,
all of CartoDB's variants). These work with no key or signup at all:

| Provider | Style |
|---|---|
| `Esri.WorldImagery` (default) | Satellite/aerial |
| `Esri.WorldTopoMap` | Topographic |
| `Esri.WorldShadedRelief` | Plain relief shading, no imagery |
| `Esri.WorldTerrain` | Terrain with labels |
| `Esri.NatGeoWorldMap` | National Geographic style |
| `Esri.OceanBasemap` | Bathymetry-focused |
| `OpenStreetMap.Mapnik` | Standard OSM |
| `OpenTopoMap` | Contour-line topographic |

Stadia, Thunderforest, MapBox, MapTiler, and Jawg all require your own
API key (set via the provider object, e.g.
`ctx.providers.Stadia.AlidadeSmooth(api_key="...")`) before they'll
return real tiles.

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

## License

MIT
