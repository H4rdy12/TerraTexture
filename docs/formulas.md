# Formulas and conventions

## Curvature (Zevenbergen & Thorne, 1987)

Given first partials `p = dz/dx`, `q = dz/dy` and second partials
`r = d2z/dx2`, `t = d2z/dy2`, `s = d2z/dxdy`:

```
profile curvature  = -(r*p^2 + 2*s*p*q + t*q^2) / ((p^2+q^2) * (1+p^2+q^2)^1.5)
planform curvature = -(r*q^2 - 2*s*p*q + t*p^2) /  (p^2+q^2)^1.5
```

This is the formulation used by ArcGIS's `Curvature` tool, QGIS's terrain
analysis plugins, and GRASS's `r.slope.aspect`.

**Profile curvature** is measured along the direction of steepest
descent -- it describes how slope itself changes as you move downhill,
and therefore controls how flow *accelerates or decelerates*.

**Planform curvature** is measured across the slope, i.e. along a
contour line -- it describes whether flow lines are converging or
diverging, and therefore controls channel/ridge formation.

### Sign convention

- **Profile**: positive = convex (flow decelerates), negative = concave
  (flow accelerates).
- **Planform**: positive = convex (flow diverges -- ridges), negative =
  concave (flow converges -- channels/valleys).
- **Flat cells** (`p^2 + q^2 -> 0`, i.e. no local gradient) have
  undefined curvature by this formula and are set to `0`.

This sign convention matches ArcGIS's default; some other tools (GRASS
in particular) use the opposite sign for one or both curvatures, so
always check convention when comparing outputs across software.

## Implementation note: gradient-of-gradient vs. the ZT stencil

This implementation computes `p, q, r, t, s` via two chained calls to
`numpy.gradient` (`np.gradient` on the DEM, then `np.gradient` again on
the resulting `p`/`q` arrays). This is a reasonable finite-difference
approximation and is fast, but it is **not** algebraically identical to
Zevenbergen & Thorne's original closed-form 3x3-window polynomial fit,
which uses specific finite-difference stencils derived from fitting a
quadratic surface to the 9 cells in a moving window. The two approaches
will agree closely in smooth interior regions but can diverge slightly:

- Near NaN/nodata boundaries, where `numpy.gradient`'s one-sided
  differencing at fill-boundary edges interacts differently with the
  chained second derivative than a single-pass 3x3 stencil would.
- On very noisy DEMs, where the double-differencing in the
  gradient-of-gradient approach can amplify high-frequency noise
  slightly more than the ZT stencil's direct second-derivative formula.

If you need bit-for-bit parity with ArcGIS/GRASS/QGIS curvature outputs,
replace `_derivatives()` in `TerraTexture/derivatives.py` with a direct
`scipy.ndimage.convolve` implementation of the ZT 3x3 kernels for
`r`, `t`, and `s`. This is also computationally cheaper (one convolution
pass per second derivative, vs. two sequential `np.gradient` calls).

## Blend modes

### Soft light (Photoshop formula)

For base `a` and blend `b`, both in `[0, 1]`:

```
if b <= 0.5:
    result = 2*a*b + a^2 * (1 - 2*b)
else:
    result = 2*a*(1-b) + sqrt(a) * (2*b - 1)
```

Used here to punch curvature signal into a hillshade, and to composite
the hillshade/curvature/elevation relief stack layer-by-layer.

A blend value of exactly `0.5` is the identity -- it leaves the base
layer completely unchanged, which is why a neutral-grey blend layer is
a no-op in Photoshop.

### Luminosity (SVG compositing spec / Photoshop)

```
SetLum(C, l) = ClipColor(C + (l - Lum(C)))
Lum(C) = 0.3*R + 0.59*G + 0.11*B
```

`ClipColor` pulls an out-of-gamut RGB triple back into `[0, 1]` by
scaling around its own luminosity, which is what keeps hue and
saturation intact rather than doing a naive per-channel clip (which
would shift both).

This is the correct blend mode for draping a greyscale relief signal
over colour imagery: it replaces only the *lightness* of the backdrop,
leaving its hue and saturation completely untouched -- unlike soft-light
or per-channel multiply, which can desaturate or shift the imagery's
colour.
