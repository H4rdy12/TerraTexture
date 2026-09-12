"""
Command-line entry point: `dem-relief` / `python -m dem_relief`.

Two subcommands:
    dem-relief curvature  DEM_PATH   -> plot_dem_curvature_softlight()
    dem-relief basemap    DEM_PATH   -> plot_dem_basemap_luminosity_relief()

Run with no DEM_PATH to use the synthetic demo DEM (curvature only --
basemap needs a real georeferenced file since it fetches imagery for an
actual location).
"""

import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="dem-relief",
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

    return 0


if __name__ == "__main__":
    sys.exit(main())
