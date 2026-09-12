"""
Quickstart: run the curvature + soft-lit relief pipeline on the built-in
synthetic demo DEM. No external data or extra dependencies (rasterio,
contextily) required beyond numpy/scipy/matplotlib.

    python examples/quickstart.py
"""

from TerraTexture.io import load_dem
from TerraTexture.plotting import plot_dem_curvature_softlight

if __name__ == "__main__":
    dem, cellsize = load_dem(None)  # synthetic demo DEM: hills + valley + noise

    fig, axes, results = plot_dem_curvature_softlight(
        dem,
        cellsize=cellsize,
        out_png="quickstart_relief.png",
        show=False,
    )

    print("Panels saved to quickstart_relief.png")
    print("Profile curvature range:", results["profile"].min(), results["profile"].max())
    print("Planform curvature range:", results["planform"].min(), results["planform"].max())
