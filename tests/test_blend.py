import numpy as np

from terra_texture.blend import soft_light, luminosity_blend, _lum


def test_soft_light_output_in_unit_range():
    rng = np.random.default_rng(0)
    a = rng.random((20, 20))
    b = rng.random((20, 20))
    out = soft_light(a, b)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_soft_light_neutral_grey_blend_is_identity():
    """Soft-lighting with a blend layer of exactly 0.5 should leave the
    base layer unchanged (that's the point of a 0.5-grey blend layer in
    Photoshop's soft light mode)."""
    a = np.linspace(0, 1, 50)
    b = np.full_like(a, 0.5)
    out = soft_light(a, b)
    assert np.allclose(out, a, atol=1e-6)


def test_luminosity_blend_preserves_hue_and_saturation():
    """luminosity_blend should change only the lightness of backdrop_rgb,
    not its hue/saturation -- i.e. the ratios between channels stay the
    same modulo the shared additive shift."""
    backdrop = np.array([[[0.8, 0.2, 0.2]]])  # a reddish pixel
    target_lum = np.array([[0.9]])  # much brighter light
    out = luminosity_blend(backdrop, target_lum)
    assert np.isclose(_lum(out)[0, 0], 0.9, atol=1e-5)
    # red channel should still dominate green/blue after brightening
    assert out[0, 0, 0] > out[0, 0, 1]
    assert out[0, 0, 0] > out[0, 0, 2]


def test_luminosity_blend_output_in_unit_range():
    rng = np.random.default_rng(1)
    backdrop = rng.random((10, 10, 3))
    lum = rng.random((10, 10))
    out = luminosity_blend(backdrop, lum)
    assert out.min() >= 0.0
    assert out.max() <= 1.0
