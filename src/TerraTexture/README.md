# Package layout
 
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
 
See [`docs/formulas.md`](docs/formulas.md) for the full curvature
derivation, sign conventions, and blend-mode math.
