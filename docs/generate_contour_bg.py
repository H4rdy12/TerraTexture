#!/usr/bin/env python3
"""
Regenerate the docs' contour-line background SVG (from TerraTexture's own
synthetic demo DEM) and re-embed it as a base64 data URI into both
docs_template/layout.css (module docs pages) and docs_template/landing.html
(the cover page) -- these are the only two places the image is used.

The SVG is deliberately hand-built from simplified contour paths (via
skimage.measure.find_contours + a cheap point-decimation pass) rather than
exported from matplotlib -- matplotlib's own SVG export of the same plot
is over 30x larger (verbose per-point paths, clip-path boilerplate) and
not worth the extra weight for a repeating background texture.

Usage:
    uv run python docs/generate_contour_bg.py
    uv run python docs/generate_contour_bg.py --stroke-width 1.0 --opacity 0.3
    uv run python docs/generate_contour_bg.py --levels 15 --color "#5a5a5a"

Requires scikit-image, which is NOT a project dependency (only needed to
regenerate this one static asset, not to build/use the docs themselves):
    uv pip install scikit-image
"""

import argparse
import base64
import re
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = REPO_ROOT / "docs" / "docs_template"
FILES_TO_UPDATE = [TEMPLATE_DIR / "layout.css", TEMPLATE_DIR / "landing.html"]


def simplify(points, tolerance):
    """Cheap distance-based point decimation (not true Douglas-Peucker,
    but good enough here and much simpler) -- keeps the SVG small."""
    if len(points) < 3:
        return points
    out = [points[0]]
    for p in points[1:]:
        if np.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > tolerance:
            out.append(p)
    out.append(points[-1])
    return out


def generate_svg(stroke_width, opacity, color, num_levels, tolerance, downsample, smooth_sigma):
    from skimage import measure
    from TerraTexture.io import load_dem

    dem, _cellsize = load_dem(None)  # our own synthetic demo DEM
    small = zoom(gaussian_filter(dem, sigma=smooth_sigma), downsample)
    h, w = small.shape

    levels = np.linspace(small.min(), small.max(), num_levels)[1:-1]

    paths = []
    for lvl in levels:
        for c in measure.find_contours(small, lvl):
            if len(c) < 6:
                continue
            pts = simplify(c[:, ::-1], tolerance=tolerance)
            if len(pts) >= 3:
                paths.append(pts)

    polylines = "\n".join(
        '<polyline points="' + " ".join(f"{x:.0f},{y:.0f}" for x, y in pts) + '"/>'
        for pts in paths
    )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">\n'
        f'<g fill="none" stroke="{color}" stroke-width="{stroke_width}" '
        f'stroke-opacity="{opacity}" stroke-linejoin="round" stroke-linecap="round">\n'
        f"{polylines}\n"
        f"</g>\n</svg>"
    )
    print(f"Generated {len(paths)} contour paths, {sum(len(p) for p in paths)} points, "
          f"{len(svg)} bytes (SVG)")
    return svg


def embed(svg_text):
    b64 = base64.b64encode(svg_text.encode()).decode("ascii")
    print(f"Base64: {len(b64)} bytes")

    pattern = re.compile(r"(data:image/svg\+xml;base64,)[A-Za-z0-9+/=]+")
    for path in FILES_TO_UPDATE:
        content = path.read_text()
        new_content, count = pattern.subn(r"\1" + b64, content)
        if count == 0:
            print(f"  !! WARNING: no data URI found in {path}, nothing replaced")
            continue
        path.write_text(new_content)
        print(f"  updated {path} ({count} occurrence(s))")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stroke-width", type=float, default=0.65, help="Line thickness (default: 0.65)")
    parser.add_argument("--opacity", type=float, default=0.22, help="Line opacity, 0-1 (default: 0.22)")
    parser.add_argument("--color", default="#696969", help="Line color (default: dimgrey, #696969)")
    parser.add_argument("--levels", type=int, default=11, help="Number of contour levels (default: 11)")
    parser.add_argument("--tolerance", type=float, default=2.5, help="Point-decimation tolerance, larger = fewer points/smaller file (default: 2.5)")
    parser.add_argument("--downsample", type=float, default=0.5, help="DEM downsample factor before contouring, smaller = coarser/smaller file (default: 0.5)")
    parser.add_argument("--smooth-sigma", type=float, default=4.0, help="Gaussian smoothing applied to the DEM before contouring (default: 4.0)")
    args = parser.parse_args()

    svg = generate_svg(
        stroke_width=args.stroke_width, opacity=args.opacity, color=args.color,
        num_levels=args.levels, tolerance=args.tolerance,
        downsample=args.downsample, smooth_sigma=args.smooth_sigma,
    )
    embed(svg)
    print("\nDone. Run `uv run python docs/build_docs.py` to see the change.")


if __name__ == "__main__":
    main()
