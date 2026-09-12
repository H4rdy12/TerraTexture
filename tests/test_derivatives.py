import numpy as np

from TerraTexture.derivatives import curvatures, hillshade


def test_flat_dem_has_zero_curvature(flat_dem):
    dem, cellsize = flat_dem
    profile, planform = curvatures(dem, cellsize)
    assert np.allclose(profile, 0.0)
    assert np.allclose(planform, 0.0)


def test_paraboloid_profile_curvature_sign(paraboloid_dem):
    """A bowl (z = x^2 + y^2) should show concave profile curvature
    (flow accelerates downhill) away from the centre, i.e. negative,
    given this module's sign convention (+convex/decel, -concave/accel)."""
    dem, cellsize = paraboloid_dem
    profile, _ = curvatures(dem, cellsize)
    centre = dem.shape[0] // 2
    # sample a ring away from the centre/edges where the gradient is well-defined
    sample = profile[centre, centre + 20]
    assert sample < 0


def test_dome_planform_curvature_sign(dome_dem):
    """A dome (z = -(x^2+y^2)) has circular contours that diverge flow as
    it moves downhill from the peak, which should register as positive
    planform curvature (+ridges/divergent) off-centre."""
    dem, cellsize = dome_dem
    _, planform = curvatures(dem, cellsize)
    centre = dem.shape[0] // 2
    sample = planform[centre, centre + 20]
    assert sample > 0


def test_curvature_preserves_nan_mask():
    dem = np.ones((30, 30)) * 50.0
    dem[10:15, 10:15] = np.nan
    profile, planform = curvatures(dem, 1.0)
    assert np.all(np.isnan(profile[10:15, 10:15]))
    assert np.all(np.isnan(planform[10:15, 10:15]))
    assert not np.any(np.isnan(profile[:5, :5]))


def test_hillshade_range(paraboloid_dem):
    dem, cellsize = paraboloid_dem
    hs = hillshade(dem, cellsize)
    assert np.nanmin(hs) >= 0.0
    assert np.nanmax(hs) <= 1.0


def test_hillshade_preserves_nan_mask():
    dem = np.ones((30, 30)) * 50.0
    dem[5:8, 5:8] = np.nan
    hs = hillshade(dem, 1.0)
    assert np.all(np.isnan(hs[5:8, 5:8]))
