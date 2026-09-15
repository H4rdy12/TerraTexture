"""
Drape DEM relief over real-world basemap imagery using the ArcGIS Pro /
Photoshop "luminosity blend" recipe.

Requires rasterio + contextily. The `aoi_bounds=` code path additionally
uses `requests` (via TerraTexture.sources) to query PGC's public STAC
API -- no signup, no API key, no local software required.
"""

import numpy as np
import matplotlib.pyplot as plt

from .io import _open_raster, load_dem_mosaic
from .sources import arcticdem_mosaic, rema_mosaic
from .derivatives import curvatures, hillshade
from .blend import soft_light, luminosity_blend
from .stretch import normalize, stretch_std

_AOI_PRODUCTS = {
    "arcticdem": arcticdem_mosaic,
    "rema": rema_mosaic,
}


def plot_dem_basemap_luminosity_relief(
    dem_path=None,
    aoi_bounds=None,
    aoi_bounds_crs="EPSG:4326",
    dem_product="arcticdem",
    arcticdem_resolution=32,
    source=None,
    zoom="auto",
    target_crs=None,
    azimuth=315,
    altitude=45,
    curvature_std=4,
    hillshade_std=4,
    relief_strength=1.0,
    figsize=(10, 10),
    out_fig=None,
    show=True,
    tile_cache_dir=None,
    tile_connections=16,
):
    """Drape a DEM's relief over basemap imagery using the ArcGIS Pro /
    Photoshop "luminosity blend" recipe, layer stack top -> bottom:

        1. basemap imagery                          [Soft Light]
        2. relief group                              [Luminosity]
           - DEM elevation, white->black              (top of group)
           - profile curvature (X) planform curvature,
             +/-N-std stretch, soft-light blended       (middle)
           - hillshade, +/-N-std stretch               (bottom of group)
        3. basemap imagery                           [Normal, base]

    Everything is computed and composited on ONE grid: the DEM's own,
    in `target_crs` (default EPSG:3413, NSIDC Sea Ice Polar Stereographic
    North -- ArcticDEM's native CRS). If the DEM is already in target_crs
    (the normal case for ArcticDEM tiles) it is used completely unmodified,
    at full native resolution -- no resampling loss. Only the basemap
    imagery gets warped, from the tile source's native EPSG:3857 onto the
    DEM's exact transform/shape/CRS, so every layer is pixel-for-pixel
    aligned before any blending happens.

    Within the group, each layer soft-lights onto the composite of
    everything below it (hillshade is the base). The resulting single
    greyscale "relief luminosity" raster then replaces only the *lightness*
    of the basemap imagery via the SVG/Photoshop Luminosity blend mode
    (luminosity_blend()) -- imagery hue & saturation are preserved exactly,
    unlike a naive per-channel soft-light burn. A second copy of the
    imagery is then soft-lit on top of that result to restore some of the
    colour punch/contrast luminosity blending tends to flatten.

    Parameters
    ----------
    dem_path : str, sequence of str, or None
        Path to a georeferenced raster (GeoTIFF, or .tar.gz/.tgz archive
        e.g. an ArcticDEM mosaic tile). Pass a list/tuple of two or more
        paths to merge neighbouring tiles into one seamless mosaic first
        (via load_dem_mosaic()) before computing relief. Provide EITHER
        this OR aoi_bounds, not both.
    aoi_bounds : tuple or None
        (x_min, y_min, x_max, y_max) in `aoi_bounds_crs`. ADDITIONAL
        alternative to dem_path: instead of a local file, query PGC's
        public STAC API (https://stac.pgc.umn.edu/api/v1) for mosaic
        tiles intersecting this AOI, merge them, and use that as the
        DEM. No signup, API key, or local software required -- see
        TerraTexture.sources.
    aoi_bounds_crs : str
        CRS of `aoi_bounds`. Defaults to EPSG:4326 (plain lon/lat) --
        the natural way most people already have a bounding box, e.g.
        from a GPS device or a web map. Independent of `target_crs`:
        you can hand this lon/lat bounds while still getting the merged
        DEM back in ArcticDEM/REMA's native polar-stereographic CRS (or
        any other `target_crs` you choose) -- the two used to be forced
        to match, which meant lon/lat bounds couldn't be used at all
        unless you reprojected them yourself first.
    dem_product : {"arcticdem", "rema"}
        Which PGC dataset to query when aoi_bounds is given. "arcticdem"
        covers the Arctic (including Greenland); "rema" covers
        Antarctica. Ignored when dem_path is given instead.
    arcticdem_resolution : int
        Mosaic resolution in meters (2, 10, or 32) for whichever
        dem_product is selected -- the name is a holdover from when this
        only supported ArcticDEM; it applies to REMA too. Only used when
        aoi_bounds is given.
    source : contextily provider or None
        Tile source, default contextily.providers.Esri.WorldImagery.
    zoom : int or "auto"
        Tile zoom level for the *source* imagery fetch (before it gets
        warped onto the DEM grid).
    target_crs : str or None
        CRS everything is composited in. If None (the default), resolves
        to the DEM's own native CRS: EPSG:3413 for ArcticDEM, EPSG:3031
        for REMA when using aoi_bounds, or EPSG:3413 when using dem_path
        (matching prior behaviour). Set explicitly if you need a
        different common CRS.
    azimuth, altitude : float
        Sun position (degrees) for the hillshade.
    curvature_std, hillshade_std : float
        N for the mean +/- N*std stretch applied to curvature and
        hillshade respectively (ArcGIS Pro's "stretched N standard
        deviations" symbology).
    relief_strength : float
        How much the DEM/curvature/hillshade relief signal overrides the
        basemap imagery's own natural brightness, from 0 to 1. Default
        1.0 replaces the basemap's lightness entirely with the relief
        signal (the original behaviour). Lower values blend toward the
        basemap's own natural luminosity instead -- try 0.5-0.7 if the
        result looks too dark/grey relative to the actual basemap
        colours. 0.0 shows essentially no relief texture at all (the
        basemap's own lightness is left untouched), so it's rarely
        useful on its own -- this is a blend control, not a fade-out.
    figsize : tuple
    out_fig : str or None
        Path to save the figure to (via plt.savefig), or None to skip
        saving.
    show : bool
    tile_cache_dir : str or None
        Directory to persist downloaded basemap tiles across process
        runs. contextily's default cache lives in a fresh tempdir that
        is deleted at interpreter exit (see contextily.tile), so every
        fresh script/CLI invocation is a cold cache unless this is set.
        Pass a real path to make the *first* run of a *later* process
        fast too, not just repeat calls within the same session. None
        (default) keeps contextily's own ephemeral-cache behaviour.
    tile_connections : int
        Number of parallel connections contextily uses to fetch basemap
        tiles (passed through as `n_connections`). contextily itself
        defaults to 1, i.e. one tile fetched at a time -- the dominant
        cost at high zoom, since tile count grows ~4x per zoom level.
        16 is a reasonable default; check your tile provider's usage
        policy before going higher (some, e.g. OSM, cap this at 2).

    Returns
    -------
    fig, ax : matplotlib Figure/Axes
    layers : dict with 'basemap', 'dem_grey', 'curvature', 'hillshade',
        'relief_luminosity', 'luminosity_composite', 'final' arrays, for
        inspecting or re-blending any individual stage.
    """
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
    from rasterio.transform import array_bounds, from_bounds
    import contextily as ctx

    if source is None:
        source = ctx.providers.Esri.WorldImagery

    if tile_cache_dir is not None:
        # contextily's default cache is a tempdir wiped at process exit
        # (contextily.tile._clear_cache via atexit) -- every fresh
        # process is a cold cache unless we point it somewhere durable.
        import os
        os.makedirs(os.path.expanduser(tile_cache_dir), exist_ok=True)
        ctx.set_cache_dir(os.path.expanduser(tile_cache_dir))

    if dem_path is None and aoi_bounds is None:
        raise ValueError(
            "Provide either dem_path (file/.tar.gz/list of tiles) or "
            "aoi_bounds (to query PGC's public STAC API for that AOI)."
        )
    if aoi_bounds is not None and dem_product not in _AOI_PRODUCTS:
        raise ValueError(f"dem_product must be one of {sorted(_AOI_PRODUCTS)}, got {dem_product!r}")

    if target_crs is None:
        if aoi_bounds is not None:
            target_crs = "EPSG:3413" if dem_product == "arcticdem" else "EPSG:3031"
        else:
            target_crs = "EPSG:3413"

    # -- open the DEM (path, mosaic list, or public STAC AOI query);
    #    only reproject if it isn't already in target_crs --
    if aoi_bounds is not None:
        fetch_mosaic = _AOI_PRODUCTS[dem_product]
        dem, cellsize, transform, mosaic_crs = fetch_mosaic(
            aoi_bounds,
            resolution=arcticdem_resolution,
            bbox_crs=aoi_bounds_crs,
            target_crs=target_crs,
        )
        height, width = dem.shape
        target_crs = str(mosaic_crs)
    elif isinstance(dem_path, (list, tuple)):
        dem, cellsize, transform, mosaic_crs = load_dem_mosaic(dem_path, target_crs=target_crs)
        height, width = dem.shape
        if str(mosaic_crs).upper() != target_crs.upper():
            # load_dem_mosaic warped everything into mosaic_crs already;
            # honour whatever CRS it actually merged into
            target_crs = str(mosaic_crs)
    else:
        with _open_raster(dem_path) as src:
            if src.crs is not None and str(src.crs).upper() == target_crs.upper():
                dem = src.read(1).astype(np.float32)
                transform, width, height = src.transform, src.width, src.height
                if src.nodata is not None:
                    dem = np.where(dem == src.nodata, np.nan, dem)
            else:
                transform, width, height = calculate_default_transform(
                    src.crs, target_crs, src.width, src.height, *src.bounds
                )
                dem = np.full((height, width), np.nan, dtype=np.float32)
                reproject(
                    source=rasterio.band(src, 1),
                    destination=dem,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    src_nodata=src.nodata,
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                )
    cellsize = transform.a
    west, south, east, north = array_bounds(height, width, transform)

    # -- relief components, computed at the DEM's own native resolution --
    profile, planform = curvatures(dem, cellsize)
    hs = hillshade(dem, cellsize, azimuth=azimuth, altitude=altitude)

    profile_grey = stretch_std(profile, curvature_std)
    planform_grey = stretch_std(planform, curvature_std)
    hs_grey = stretch_std(hs, hillshade_std)
    dem_grey = normalize(dem)  # white->black elevation

    # -- group stack, bottom to top, each soft-lighting onto the composite below --
    curvature_combo = soft_light(profile_grey, planform_grey)      # profile (X) planform
    group_composite = soft_light(hs_grey, curvature_combo)          # curvature over hillshade
    relief_luminosity = soft_light(group_composite, dem_grey)       # DEM over that -> final grey

    # texture-only luminosity (hillshade + curvature, no elevation): centred and
    # locally-varying, without the broad monotonic darkening dem_grey imposes
    # across low-elevation terrain -- exposed separately for burn_data_onto_relief
    # so a burned data layer can show topographic texture without elevation
    # crushing its colour at low elevation (see 'texture_luminosity' in layers)
    texture_luminosity = np.nan_to_num(group_composite, nan=0.5)

    # NaNs (nodata/voids) shouldn't distort the blend maths; treat as neutral mid-grey
    relief_luminosity = np.nan_to_num(relief_luminosity, nan=0.5)

    # -- fetch basemap imagery (tile servers always serve EPSG:3857) then
    #    warp it onto the DEM's exact grid: same transform/shape/target_crs
    #    as everything else, so nothing needs resampling downstream --
    lon_west, lat_south, lon_east, lat_north = transform_bounds(
        target_crs, "EPSG:4326", west, south, east, north
    )
    basemap_3857, extent_3857 = ctx.bounds2img(
        lon_west, lat_south, lon_east, lat_north, zoom=zoom, source=source, ll=True,
        n_connections=tile_connections,
    )
    bm_west, bm_east, bm_south, bm_north = extent_3857
    bm_transform = from_bounds(
        bm_west, bm_south, bm_east, bm_north, basemap_3857.shape[1], basemap_3857.shape[0]
    )

    basemap_rgb = np.empty((height, width, 3), dtype=np.float32)
    for b in range(3):
        band_dst = np.empty((height, width), dtype=np.float32)
        reproject(
            source=basemap_3857[:, :, b].astype(np.float32),
            destination=band_dst,
            src_transform=bm_transform,
            src_crs="EPSG:3857",
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear,
        )
        basemap_rgb[:, :, b] = band_dst
    basemap_rgb = np.clip(basemap_rgb / 255.0, 0, 1)

    if relief_strength < 1.0:
        basemap_luminosity = (
            0.3 * basemap_rgb[..., 0] + 0.59 * basemap_rgb[..., 1] + 0.11 * basemap_rgb[..., 2]
        )
        target_luminosity = (
            relief_strength * relief_luminosity + (1 - relief_strength) * basemap_luminosity
        )
    else:
        target_luminosity = relief_luminosity

    # -- group's own blend mode against the basemap below it: Luminosity --
    luminosity_composite = luminosity_blend(basemap_rgb, target_luminosity)

    # -- topmost layer: imagery again, Soft Light, to restore colour punch --
    final = soft_light(luminosity_composite, basemap_rgb)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(final, extent=(west, east, south, north))
    ax.set_xticks([])
    ax.set_yticks([])
    attribution = getattr(source, "attribution", "")
    if attribution:
        ax.text(
            0.01, 0.01, attribution, transform=ax.transAxes,
            fontsize=6, color="white", ha="left", va="bottom",
            bbox=dict(facecolor="black", alpha=0.5, pad=1, linewidth=0),
        )
    plt.tight_layout()

    if out_fig:
        plt.savefig(out_fig, dpi=600)
        print(f"Saved figure to {out_fig}")
    if show:
        plt.show()

    layers = {
        "basemap": basemap_rgb,
        "dem_grey": dem_grey,
        "curvature": curvature_combo,
        "hillshade": hs_grey,
        "relief_luminosity": relief_luminosity,
        "texture_luminosity": texture_luminosity,
        "luminosity_composite": luminosity_composite,
        "final": final,
        "extent": (west, east, south, north),
        "transform": transform,
        "crs": target_crs,
        "shape": (height, width),
    }
    return fig, ax, layers


def add_relief_basemap(
    axes,
    dem_path=None,
    aoi_bounds=None,
    aoi_bounds_crs="EPSG:4326",
    dem_product="arcticdem",
    arcticdem_resolution=32,
    source=None,
    zoom="auto",
    target_crs=None,
    azimuth=315,
    altitude=45,
    curvature_std=4,
    hillshade_std=4,
    zorder=0,
):
    """Build the luminosity-blended relief basemap ONCE, then stamp the
    same image onto one or more existing matplotlib Axes as a background
    layer -- for dropping the relief into just the spatial panels of a
    larger multi-axes figure (e.g. a plt.subplot_mosaic layout) without
    re-querying the DEM/imagery per panel and without creating its own
    standalone figure.

    Typical use: call this BEFORE your other per-axis plotting calls, so
    the relief sits underneath. The image is drawn at `zorder` (default
    0, i.e. bottom); anything you plot afterwards on the same axes with
    the default zorder (~1+) will appear on top of it. Since data layers
    are typically opaque pcolormeshes, you'll usually want to set some
    transparency on them afterwards so the relief shows through -- e.g.:

        for coll in ax.collections:
            coll.set_alpha(0.75)

    Parameters
    ----------
    axes : matplotlib.axes.Axes or sequence of Axes
        Axis (or axes) to draw the relief background on.
    dem_path, aoi_bounds, aoi_bounds_crs, dem_product, arcticdem_resolution,
    source, zoom, target_crs, azimuth, altitude, curvature_std, hillshade_std :
        Same as plot_dem_basemap_luminosity_relief(); provide dem_path OR
        aoi_bounds, not both.
    zorder : float
        Drawing order for the relief image. Default 0 (background).

    Returns
    -------
    layers : dict
        Same dict plot_dem_basemap_luminosity_relief() returns (including
        'extent'), in case you want to reuse the relief array or bounds
        elsewhere (e.g. to align a data overlay's axis limits).

    Example
    -------
    >>> spatial_axes = [ax1, ax2]
    >>> add_relief_basemap(spatial_axes, aoi_bounds=my_bounds)
    """
    if isinstance(axes, plt.Axes):
        axes = [axes]

    _fig, _ax, layers = plot_dem_basemap_luminosity_relief(
        dem_path=dem_path,
        aoi_bounds=aoi_bounds,
        aoi_bounds_crs=aoi_bounds_crs,
        dem_product=dem_product,
        arcticdem_resolution=arcticdem_resolution,
        source=source,
        zoom=zoom,
        target_crs=target_crs,
        azimuth=azimuth,
        altitude=altitude,
        curvature_std=curvature_std,
        hillshade_std=hillshade_std,
        out_fig=None,
        show=False,
    )
    plt.close(_fig)  # throwaway standalone figure -- only the arrays matter here

    for ax in axes:
        ax.imshow(layers["final"], extent=layers["extent"], zorder=zorder)

    return layers
