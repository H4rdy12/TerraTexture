"""
Drape an ArcticDEM mosaic tile's relief over Esri World Imagery using the
luminosity-blend recipe.

Requires rasterio + contextily, and a real ArcticDEM mosaic tile
(.tar.gz) -- these are large (several GB uncompressed) and aren't
included in this repo. Download one from the PGC ArcticDEM mosaic
index (https://www.pgc.umn.edu/data/arcticdem/) and pass its path via
--dem, or set the TERRA_TEXTURE_DEMO_TILE environment variable.

    python examples/arcticdem_basemap.py --dem /path/to/15_44_32m_v4.1.tar.gz
"""

import argparse
import os

from TerraTexture.basemap import plot_dem_basemap_luminosity_relief


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dem",
        default=os.environ.get("TERRA_TEXTURE_DEMO_TILE"),
        help="Path to an ArcticDEM mosaic tile (.tar.gz) or any georeferenced DEM raster.",
    )
    parser.add_argument("--out", default="dem_basemap_luminosity_relief.png")
    args = parser.parse_args()

    if not args.dem:
        raise SystemExit(
            "No DEM tile given. Pass --dem /path/to/tile.tar.gz or set "
            "the TERRA_TEXTURE_DEMO_TILE environment variable."
        )

    fig, ax, layers = plot_dem_basemap_luminosity_relief(
        args.dem,
        out_png=args.out,
    )


if __name__ == "__main__":
    main()
