"""
Burn a scientific data raster (e.g. dh/dt, velocity) onto an existing
relief via the SVG/Photoshop Luminosity blend mode.

Requires rasterio (for reprojecting the data onto the relief's grid).
"""

import numpy as np

from .blend import luminosity_blend


def burn_data_onto_relief(
    relief_layers,
    data,
    data_transform=None,
    data_crs=None,
    cmap="RdYlBu_r",
    vmin=None,
    vmax=None,
    norm=None,
    nan_fill="relief",
    resampling="bilinear",
    luminosity_range=(0.15, 0.9),
    luminosity_source="texture",
):
    """Colour-map a 2D data array (e.g. dh/dt, velocity) and burn it onto
    an existing relief via the SVG/Photoshop Luminosity blend mode -- the
    same technique plot_dem_basemap_luminosity_relief() uses to drape
    imagery over terrain, applied here to a scientific raster instead.
    The data supplies hue/saturation (its own colour ramp is preserved
    exactly); the relief supplies only the lightness, so the terrain's
    hillshade+curvature texture is embossed directly INTO the data's
    colours -- unlike plotting the data as a separate translucent overlay
    (alpha blending), which just partially reveals the relief in the gaps
    rather than actually combining the two signals.

    Parameters
    ----------
    relief_layers : dict
        The `layers` dict returned by plot_dem_basemap_luminosity_relief()
        or add_relief_basemap() (needs 'relief_luminosity', 'final',
        'transform', 'crs', 'shape').
    data : 2D array or xarray.DataArray
        The data to burn on. NaN = nodata/masked.
    data_transform : affine.Affine or None
        Affine transform of `data`'s own grid. Required if `data` is a
        plain numpy array. If `data` is an xarray.DataArray with a `.rio`
        accessor (rioxarray) carrying CRS/transform, both are picked up
        automatically when not given explicitly.
    data_crs : str or None
        CRS of `data`'s own grid. Same auto-detection rule as
        data_transform.
    cmap : str
        Matplotlib colormap for `data` (default 'RdYlBu_r', matching
        DEMTimeSeriesPlotter's CMAP_SLOPE).
    vmin, vmax : float or None
        Colour scale limits. Required unless `norm` is given.
    norm : matplotlib.colors.Normalize or None
        Explicit norm (e.g. a diverging TwoSlopeNorm centred on zero),
        overriding vmin/vmax.
    nan_fill : {'relief', 'transparent'}
        What shows through where `data` is NaN after reprojection onto the
        relief grid. 'relief' (default) shows the plain imagery-draped
        relief (relief_layers['final']) with no data colour there -- e.g.
        for bedrock/ocean pixels masked out of a dh/dt product. 'transparent'
        returns an RGBA array with alpha=0 there instead.
    resampling : str
        rasterio.enums.Resampling name for reprojecting `data` onto the
        relief's grid (default 'bilinear'; use 'nearest' for categorical
        data).
    luminosity_range : (float, float) or None
        Compress the chosen luminosity source into this [lo, hi] band
        before burning (default (0.15, 0.9)). The Luminosity blend mode
        forces the composite's exact lightness to match the luminosity
        value, so values near 0 or 1 crush the data's colour toward
        black/white regardless of hue. Pass None to use the raw [0, 1]
        luminosity unmodified.
    luminosity_source : {'texture', 'full'} or 2D array
        Which signal supplies the terrain lightness for the burn:
        'texture' (default) uses relief_layers['texture_luminosity'] --
        hillshade + curvature only, WITHOUT elevation. This stays centred
        and locally-varying, so terrain texture (ridges, valleys, slope
        detail) is visible everywhere without elevation's broad monotonic
        darkening at low elevation crushing the data's colour -- the
        ArcGIS Pro "stretched N std" hillshade/curvature look, without the
        white-to-black DEM greyscale layered on top of it. 'full' uses
        relief_layers['relief_luminosity'] (hillshade+curvature+elevation,
        the same signal the plain imagery-draped relief uses) for a more
        dramatic, elevation-shaded look, at the cost of more colour
        crushing at extremes even with luminosity_range compression. You
        can also pass your own precomputed (H, W) array directly.

    Returns
    -------
    composite : (H, W, 3) or (H, W, 4) float array in [0, 1], at
        relief_layers['shape'] resolution. Pass straight to
        ax.imshow(composite, extent=relief_layers['extent']).
    mappable : matplotlib.cm.ScalarMappable
        For a colorbar: plt.colorbar(mappable, ax=ax, label=...).

    Example
    -------
    >>> layers = add_relief_basemap(ax, aoi_bounds=plotter.bounds)
    >>> composite, sm = burn_data_onto_relief(
    ...     layers, slope_data,  # an xarray.DataArray with rio CRS/transform
    ...     cmap="RdYlBu_r", vmin=-6, vmax=6,
    ... )
    >>> ax.imshow(composite, extent=layers['extent'])
    >>> plt.colorbar(sm, ax=ax, label="dh/dt (m/yr)")
    """
    import matplotlib.pyplot as plt
    from rasterio.warp import reproject, Resampling as ResamplingEnum
    import matplotlib.cm as _cm
    from matplotlib.colors import Normalize

    # -- resolve data as a plain float ndarray + its own transform/crs --
    if hasattr(data, "values"):  # xarray.DataArray
        if data_transform is None and hasattr(data, "rio"):
            data_transform = data.rio.transform()
        if data_crs is None and hasattr(data, "rio"):
            data_crs = data.rio.crs
        data_arr = np.asarray(data.values, dtype=np.float32)
    else:
        data_arr = np.asarray(data, dtype=np.float32)

    if data_transform is None or data_crs is None:
        raise ValueError(
            "data_transform and data_crs are required (either passed "
            "explicitly, or auto-detectable via data.rio on an "
            "xarray.DataArray with rioxarray CRS/transform set)."
        )

    height, width = relief_layers["shape"]
    dst_transform = relief_layers["transform"]
    dst_crs = relief_layers["crs"]

    # -- reproject data onto the relief's exact grid --
    data_on_grid = np.full((height, width), np.nan, dtype=np.float32)
    reproject(
        source=data_arr,
        destination=data_on_grid,
        src_transform=data_transform,
        src_crs=data_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=getattr(ResamplingEnum, resampling),
    )

    # -- colour-map the data --
    if norm is None:
        norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)
    mappable = _cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    data_rgba = cmap_obj(norm(data_on_grid))  # (H, W, 4)
    nan_mask = np.isnan(data_on_grid)

    # -- burn: data supplies hue/saturation, relief supplies luminance --
    if isinstance(luminosity_source, str):
        if luminosity_source not in ("texture", "full"):
            raise ValueError("luminosity_source must be 'texture', 'full', or an array")
        relief_luminosity = relief_layers[
            "texture_luminosity" if luminosity_source == "texture" else "relief_luminosity"
        ]
    else:
        relief_luminosity = np.asarray(luminosity_source, dtype=np.float32)

    if luminosity_range is not None:
        lo, hi = luminosity_range
        relief_luminosity = lo + relief_luminosity * (hi - lo)
    burned_rgb = luminosity_blend(data_rgba[:, :, :3], relief_luminosity)

    if nan_fill == "relief":
        composite = np.where(nan_mask[:, :, None], relief_layers["final"], burned_rgb)
    elif nan_fill == "transparent":
        alpha = np.where(nan_mask, 0.0, 1.0)
        composite = np.dstack([burned_rgb, alpha])
    else:
        raise ValueError("nan_fill must be 'relief' or 'transparent'")

    return composite, mappable
