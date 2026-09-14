"""
Command-line entry point: `terratexture` / `python -m TerraTexture`.

### Subcommands
| Command | Does |
|---|---|
| `terratexture curvature DEM_PATH` | `plot_dem_curvature_softlight()` |
| `terratexture basemap DEM_PATH` | `plot_dem_basemap_luminosity_relief()` |
| `terratexture opentopography list` | list available OpenTopography collections |
| `terratexture opentopography fetch ...` | fetch + merge tiles from a chosen collection |

Run `curvature` with no `DEM_PATH` to use the synthetic demo DEM (basemap
needs a real georeferenced file since it fetches imagery for an actual
location).
"""

import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="terratexture",
        description="DEM curvature analysis + soft-light / luminosity-blended shaded relief.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_curv = sub.add_parser("curvature", help="6-panel curvature + soft-lit relief figure")
    p_curv.add_argument(
        "dem_path", nargs="?", default=None,
        help="Path to a DEM raster (.tif, .tar.gz, ...). Omit for a synthetic demo DEM.",
    )
    p_curv.add_argument(
        "--cellsize", type=float, default=None,
        help="Override cell size (map units/pixel). Auto-detected from raster if omitted.",
    )
    p_curv.add_argument("--out", dest="out_png", default=None, help="Save figure to this PNG path.")
    p_curv.add_argument("--no-show", action="store_true", help="Don't open an interactive window.")

    p_base = sub.add_parser("basemap", help="Luminosity-blended relief draped over basemap imagery")
    p_base.add_argument("dem_path", help="Path to a georeferenced DEM raster (.tif, .tar.gz, ...).")
    p_base.add_argument("--azimuth", type=float, default=315)
    p_base.add_argument("--altitude", type=float, default=45)
    p_base.add_argument("--target-crs", default="EPSG:3413")
    p_base.add_argument("--out", dest="out_png", default=None, help="Save figure to this PNG path.")
    p_base.add_argument("--no-show", action="store_true", help="Don't open an interactive window.")

    p_ot = sub.add_parser(
        "opentopography",
        help="Browse/fetch OpenTopography's public raster DEM STAC catalog",
    )
    ot_sub = p_ot.add_subparsers(dest="ot_command", required=True)

    ot_list = ot_sub.add_parser("list", help="List available OpenTopography DEM collections")
    ot_list.add_argument(
        "--catalog-url", default=None,
        help="Override the STAC catalog root URL (defaults to OpenTopography's).",
    )

    ot_fetch = ot_sub.add_parser(
        "fetch", help="Fetch + merge DEM tiles from a chosen collection intersecting a bbox",
    )
    ot_fetch.add_argument("collection", help="Collection id/title (see 'opentopography list') or a direct URL.")
    ot_fetch.add_argument("min_lon", type=float)
    ot_fetch.add_argument("min_lat", type=float)
    ot_fetch.add_argument("max_lon", type=float)
    ot_fetch.add_argument("max_lat", type=float)
    ot_fetch.add_argument(
        "--bbox-crs", default="EPSG:4326",
        help="CRS of the four bounds above (default: EPSG:4326, i.e. plain lon/lat as the "
             "argument names suggest). Override if you're passing bounds in some other CRS "
             "-- the min_lon/min_lat/etc. names stop being literally accurate at that point, "
             "they just mean 'first pair of coordinates, second pair of coordinates'.",
    )
    ot_fetch.add_argument("--asset-key", default="data", help="Item asset holding the DEM (default: 'data').")
    ot_fetch.add_argument(
        "--out", dest="out_tif", default="opentopography_dem.tif",
        help="Output GeoTIFF path for the merged DEM.",
    )
    ot_fetch.add_argument("--max-items", type=int, default=None)

    p_pgc = sub.add_parser(
        "pgc",
        help="Fetch ArcticDEM/REMA mosaic tiles for an AOI from PGC's public STAC API",
    )
    pgc_sub = p_pgc.add_subparsers(dest="pgc_command", required=True)

    pgc_fetch = pgc_sub.add_parser(
        "fetch", help="Fetch + merge ArcticDEM or REMA mosaic tiles intersecting a bbox",
    )
    pgc_fetch.add_argument("product", choices=["arcticdem", "rema"])
    pgc_fetch.add_argument("min_x", type=float)
    pgc_fetch.add_argument("min_y", type=float)
    pgc_fetch.add_argument("max_x", type=float)
    pgc_fetch.add_argument("max_y", type=float)
    pgc_fetch.add_argument("--bbox-crs", default="EPSG:4326", help="CRS of the four bounds above.")
    pgc_fetch.add_argument("--resolution", type=int, default=32, choices=[2, 10, 32])
    pgc_fetch.add_argument("--target-crs", default=None, help="Output CRS (default: product's native CRS).")
    pgc_fetch.add_argument("--out", dest="out_tif", default="pgc_dem.tif")
    pgc_fetch.add_argument("--max-items", type=int, default=None)

    args = parser.parse_args(argv)

    if args.command == "curvature":
        from .io import load_dem
        from .plotting import plot_dem_curvature_softlight

        dem, cellsize = load_dem(args.dem_path)
        if args.cellsize is not None:
            cellsize = args.cellsize
        plot_dem_curvature_softlight(
            dem, cellsize=cellsize, out_png=args.out_png, show=not args.no_show,
        )

    elif args.command == "basemap":
        from .basemap import plot_dem_basemap_luminosity_relief

        plot_dem_basemap_luminosity_relief(
            dem_path=args.dem_path,
            azimuth=args.azimuth,
            altitude=args.altitude,
            target_crs=args.target_crs,
            out_png=args.out_png,
            show=not args.no_show,
        )

    elif args.command == "opentopography":
        from .sources import list_stac_collections, opentopography_mosaic, OT_STAC_ROOT

        if args.ot_command == "list":
            catalog_url = args.catalog_url or OT_STAC_ROOT
            for collection in list_stac_collections(catalog_url):
                print(f"{collection['id']}\t{collection['href']}")

        elif args.ot_command == "fetch":
            import rasterio

            bounds = (args.min_lon, args.min_lat, args.max_lon, args.max_lat)
            dem, cellsize, transform, crs = opentopography_mosaic(
                args.collection, bounds, bbox_crs=args.bbox_crs,
                asset_key=args.asset_key, max_items=args.max_items,
            )
            with rasterio.open(
                args.out_tif, "w", driver="GTiff", height=dem.shape[0], width=dem.shape[1],
                count=1, dtype=dem.dtype, crs=crs, transform=transform,
            ) as dst:
                dst.write(dem, 1)
            print(f"Saved merged DEM to {args.out_tif} (cellsize={cellsize}, crs={crs})")

    elif args.command == "pgc":
        from .sources import arcticdem_mosaic, rema_mosaic

        if args.pgc_command == "fetch":
            import rasterio

            fetch = arcticdem_mosaic if args.product == "arcticdem" else rema_mosaic
            bounds = (args.min_x, args.min_y, args.max_x, args.max_y)
            dem, cellsize, transform, crs = fetch(
                bounds, resolution=args.resolution, bbox_crs=args.bbox_crs,
                target_crs=args.target_crs, max_items=args.max_items,
            )
            with rasterio.open(
                args.out_tif, "w", driver="GTiff", height=dem.shape[0], width=dem.shape[1],
                count=1, dtype=dem.dtype, crs=crs, transform=transform,
            ) as dst:
                dst.write(dem, 1)
            print(f"Saved merged DEM to {args.out_tif} (cellsize={cellsize}, crs={crs})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
