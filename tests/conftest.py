import numpy as np
import pytest


@pytest.fixture
def paraboloid_dem():
    """A bowl-shaped DEM: z = x^2 + y^2. Analytically, this surface is
    concave everywhere (accumulating flow) with well-defined constant
    curvature signs, which makes it useful for sanity-checking the
    curvature formulas without needing a full closed-form comparison."""
    n, cellsize = 101, 1.0
    half = n // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1] * cellsize
    dem = (x.astype(float) ** 2 + y.astype(float) ** 2) * 0.01
    return dem, cellsize


@pytest.fixture
def dome_dem():
    """A dome/hill: z = -(x^2 + y^2). Its circular contour lines curve
    away from the peak, so flow genuinely diverges as it moves downhill
    -- unlike a straight ridge (z = -x^2), whose contours are straight
    lines with zero planform curvature everywhere."""
    n, cellsize = 101, 1.0
    half = n // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1] * cellsize
    dem = -(x.astype(float) ** 2 + y.astype(float) ** 2) * 0.01
    return dem, cellsize


@pytest.fixture
def flat_dem():
    return np.full((50, 50), 100.0), 1.0
