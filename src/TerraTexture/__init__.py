"""
# TerraTexture

**Curvature-driven shaded relief that brings out the texture of terrain.**

## What is it?

TerraTexture turns a digital elevation model (DEM) into relief imagery
that shows *shape*, not just height. It computes terrain derivatives
(slope, aspect, profile and plan curvature, hillshade) and combines them
with Photoshop/SVG-style blend modes such as soft light and luminosity,
so ridges, gullies, moraines and breaks of slope stand out far more
clearly than in a plain hillshade. The result can be plotted on its own,
draped over a web basemap, or used as a backdrop for your own data.

<!-- terratexture:example-html -->

## Why?

TerraTexture grew out of a plotting tool I built during my PhD. 
Working with DEMs in polar regions it seemed a shame not to use 
those DEMs to show the texture of the surfaces they describe.
Default basemaps hide much of the real topographic complexity,
especially over ice, where the imagery is often a near-uniform white.

The classic GIS fix is to burn basemap imagery onto a hillshade.
TerraTexture instead blends the basemap with a composite of three
layers: curvature, hillshade and the raw DEM. Sharp, high-relief
features come out bright, while smooth, low-relief surfaces fall
darker, so the texture of the terrain carries through the imagery
rather than being flattened by it.

The project also became a chance to explore Rust for performance.
Polar DEM mosaics at around 10 m resolution span millions of square
kilometres, which means hundreds of billions of pixels, so both
memory use and speed matter. The heavy computations run in a Rust
core that releases Python's Global Interpreter Lock (GIL), so the
work can be split across threads and use every CPU core rather
than one.

The result is a small, dependency-light toolkit that does this
reproducibly. It pulls DEMs from open data, namely the
OpenTopography STAC collection and the Polar Geospatial Center (PGC)
mosaics, and uses the open-source contextily library for basemap
imagery.

## Example

```python
import TerraTexture as tt

path = tt.sources.make_demo_geotiff("demo.tif")
dem = tt.io.load_dem(path)
fig = tt.plotting.plot_dem_curvature_softlight(dem)
```

## Core breakdown

Each group is importable on its own. Submodules are loaded lazily, so
``import TerraTexture`` never pulls in rasterio or contextily until you
touch a module that needs them.

### Analysis and blending (numpy + scipy only)

- `derivatives` -- `curvatures`, `hillshade`
  (Zevenbergen & Thorne, 1987)
- `blend` -- `soft_light`, `luminosity_blend`
  (Photoshop / SVG blend modes)
- `stretch` -- `normalize`, `stretch_std`
  (percentile and standard-deviation stretches)
- `plotting` -- `plot_dem_curvature_softlight` (6-panel summary figure)

### Raster I/O (+ rasterio)

- `io` -- `load_dem`, `load_dem_mosaic`, `_open_raster`

### Open-data DEM sources (+ rasterio, requests)

- `sources` -- dynamic STAC APIs (PGC): `stac_search`,
  `arcticdem_mosaic`, `rema_mosaic`
- `sources` -- static / search-less STAC catalogs (OpenTopography or any
  catalog by URL): `list_stac_collections`, `stac_collection_items`,
  `opentopography_mosaic`
- `sources` -- `make_demo_geotiff` for quick synthetic test data

### Basemaps and overlays (+ contextily)

- `basemap` -- `plot_dem_basemap_luminosity_relief`, `add_relief_basemap`
- `overlay` -- `burn_data_onto_relief`

### Command line

- `cli` -- command-line entry point
"""

from __future__ import annotations

import importlib
from importlib import resources
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # Real imports for IDEs / type checkers only.
    from . import (  # noqa: F401
        basemap,
        blend,
        cli,
        derivatives,
        io,
        overlay,
        plotting,
        sources,
        stretch,
    )


# ---------------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------------

__version__ = "0.1.0"

__all__ = [
    "basemap",
    "blend",
    "cli",
    "derivatives",
    "io",
    "overlay",
    "plotting",
    "sources",
    "stretch",
]


# ---------------------------------------------------------------------------
# Docstring example: splice the before/after HTML widget in from a file
# ---------------------------------------------------------------------------

# Marker in the module docstring that is replaced by the HTML file contents.
_EXAMPLE_MARKER = "<!-- terratexture:example-html -->"

# Package-relative path of the interactive before/after comparison widget.
_EXAMPLE_HTML = "assets/before_after.html"


def _load_example_html() -> str:
    """
    Read the before/after comparison widget shipped with the package.

    Returns:
        str: The raw HTML, or an empty string if the asset is missing
            (e.g. an install that did not include package data), so a
            broken asset never breaks ``import TerraTexture``.
    """
    try:
        asset = resources.files(__name__).joinpath(_EXAMPLE_HTML)
        return asset.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return ""


if __doc__:  # ``python -OO`` strips docstrings, leaving ``None``.
    __doc__ = __doc__.replace(_EXAMPLE_MARKER, _load_example_html())


# ---------------------------------------------------------------------------
# Lazy submodule loading (PEP 562)
# ---------------------------------------------------------------------------

def __getattr__(name: str) -> ModuleType:
    """
    Import a public submodule on first attribute access.

    This keeps optional heavy dependencies (rasterio, contextily,
    requests) out of ``import TerraTexture`` until they are actually used.

    Args:
        name (str): Attribute being looked up on the package.

    Returns:
        ModuleType: The imported submodule.

    Raises:
        AttributeError: If ``name`` is not a public submodule.
    """
    if name in __all__:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module  # Cache so __getattr__ isn't hit again.
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """
    List package attributes, including not-yet-imported submodules.

    Returns:
        list[str]: Sorted attribute names for tab completion.
    """
    return sorted(set(globals()) | set(__all__))

