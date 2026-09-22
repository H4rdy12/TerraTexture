"""
Rust-vs-numpy parity tests for the ``terra_texture_rs`` blend kernels.

Checks the compiled ``soft_light``, ``soft_light_rgb`` and
``luminosity_blend`` kernels (``rust/src/lib.rs``) against blend.py's
numpy reference, and checks that the public functions actually dispatch
to them.

Why a separate file:
    The whole module is skipped when the extension isn't built -- the
    common case, and a supported one (see blend.py's docstring on why the
    numpy fallback is load-bearing). Any numpy-only test placed here would
    be silently skipped too, so those live in ``test_blend.py``, including
    dispatch and fallback tests that use fake kernels.

Skip vs. fail:
    - Extension *not installed* -> the module is skipped.
    - Extension installed but *fails to import* (ABI or Python-version
      mismatch, stale build, missing symbol) -> collection **errors**.
      ``pytest.importorskip`` would skip this case too, hiding exactly
      the breakage these tests exist to catch.

Dependencies:
    The built extension: ``uv sync --extra rust`` (or ``maturin develop``
    from ``rust/``).

Examples:
    Run just these tests::

        pytest tests/test_blend_rust.py -v
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import TerraTexture.blend as blend_module
from TerraTexture.blend import (
    _luminosity_blend_numpy,
    _soft_light_numpy,
    luminosity_blend,
    soft_light,
)

try:
    import terra_texture_rs
except ModuleNotFoundError as exc:
    if exc.name != "terra_texture_rs":
        raise  # the extension exists but a dependency of it is missing
    pytest.skip(
        "terra_texture_rs not built (uv sync --extra rust)",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

# Kernels vs numpy: same formula, but not guaranteed bit-identical.
_ATOL = 1e-5

requires_rgb_kernel = pytest.mark.skipif(
    not hasattr(terra_texture_rs, "soft_light_rgb"),
    reason="extension built before soft_light_rgb existed; rebuild it",
)


def _rand(seed: int, shape: tuple[int, ...]) -> np.ndarray:
    """
    Build reproducible float32 values in ``[0, 1)``.

    Args:
        seed (int): RNG seed.
        shape (tuple[int, ...]): Array shape.

    Returns:
        np.ndarray: C-contiguous float32 array.
    """
    return np.random.default_rng(seed).random(shape).astype(np.float32)


# ---------------------------------------------------------------------------
# Extension sanity
# ---------------------------------------------------------------------------

def test_blend_module_loaded_the_extension() -> None:
    """``blend._rust`` is the real extension, with no import error.

    If this fails while the direct import above worked, the package is
    importing a different module name or path than this test file.
    """
    assert blend_module._rust is terra_texture_rs
    assert blend_module._RUST_IMPORT_ERROR is None


def test_extension_exports_all_blend_kernels() -> None:
    """All three kernels exist; a missing one means a stale build.

    blend.py copes with a missing kernel (slower path), so this is a
    reminder to rebuild rather than a correctness failure.
    """
    missing = [
        name for name in ("soft_light", "soft_light_rgb", "luminosity_blend")
        if not hasattr(terra_texture_rs, name)
    ]

    assert not missing, (
        f"terra_texture_rs lacks {missing}; rebuild with "
        "`uv sync --extra rust --reinstall-package terra-texture-rs`"
    )


# ---------------------------------------------------------------------------
# Kernel parity (calling terra_texture_rs directly)
# ---------------------------------------------------------------------------

def test_rust_soft_light_matches_numpy() -> None:
    """The 2-D kernel matches numpy, returning float32."""
    a, b = _rand(2, (64, 64)), _rand(20, (64, 64))

    rust_out = terra_texture_rs.soft_light(a, b)

    assert rust_out.dtype == np.float32
    assert rust_out.shape == a.shape
    np.testing.assert_allclose(rust_out, _soft_light_numpy(a, b), atol=_ATOL)


def test_rust_soft_light_matches_numpy_on_large_array() -> None:
    """A 300 x 300 array covers any size-dependent (parallel) branch."""
    a, b = _rand(40, (300, 300)), _rand(41, (300, 300))

    rust_out = terra_texture_rs.soft_light(a, b)

    np.testing.assert_allclose(rust_out, _soft_light_numpy(a, b), atol=_ATOL)


def test_rust_soft_light_at_branch_boundary_and_extremes() -> None:
    """Exact values where the formula switches branch or saturates."""
    blend_values = np.array([0.0, 0.5, np.nextafter(0.5, 1), 1.0], np.float32)
    a = np.tile(np.array([0.0, 0.3, 1.0], np.float32)[:, None], (1, 4))
    b = np.tile(blend_values, (3, 1))

    rust_out = terra_texture_rs.soft_light(a, b)

    np.testing.assert_allclose(rust_out, _soft_light_numpy(a, b), atol=_ATOL)


def test_rust_soft_light_out_of_range_matches_numpy() -> None:
    """Out-of-range input is handled like numpy: clipped, never NaN.

    numpy clips the base before its square root; the kernel must do the
    same, or negative bases would produce NaN.
    """
    a = np.array([[-0.5, 1.5, -0.1, 1.1]], np.float32)
    b = np.array([[0.9, 0.9, 0.2, 0.2]], np.float32)

    rust_out = terra_texture_rs.soft_light(a, b)

    assert np.all(np.isfinite(rust_out))
    np.testing.assert_allclose(rust_out, _soft_light_numpy(a, b), atol=_ATOL)


def test_rust_soft_light_propagates_nan() -> None:
    """A NaN in either input gives NaN at that pixel, like numpy."""
    a, b = _rand(42, (8, 8)), _rand(43, (8, 8))
    a[1, 1] = np.nan
    b[2, 2] = np.nan

    rust_out = terra_texture_rs.soft_light(a, b)

    np.testing.assert_allclose(
        rust_out, _soft_light_numpy(a, b), atol=_ATOL, equal_nan=True
    )
    assert np.isnan(rust_out[1, 1]) and np.isnan(rust_out[2, 2])


@requires_rgb_kernel
def test_rust_soft_light_rgb_matches_numpy() -> None:
    """The fused 3-D kernel, called directly, matches numpy."""
    a, b = _rand(9, (48, 48, 3)), _rand(90, (48, 48, 3))

    rust_out = terra_texture_rs.soft_light_rgb(a, b)

    assert rust_out.dtype == np.float32
    np.testing.assert_allclose(rust_out, _soft_light_numpy(a, b), atol=_ATOL)


@requires_rgb_kernel
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_rust_soft_light_rgb_matches_2d_kernel_per_channel(channels: int) -> None:
    """The fused kernel equals the 2-D kernel per channel, exactly.

    Soft light has no cross-channel interaction, so this must be exact,
    not approximate. Several channel counts check nothing assumes 3.
    """
    a, b = _rand(10, (32, 32, channels)), _rand(100, (32, 32, channels))

    fused = terra_texture_rs.soft_light_rgb(a, b)

    for channel in range(channels):
        per_channel = terra_texture_rs.soft_light(
            np.ascontiguousarray(a[..., channel]),
            np.ascontiguousarray(b[..., channel]),
        )
        np.testing.assert_array_equal(fused[..., channel], per_channel)


def test_rust_luminosity_blend_matches_numpy() -> None:
    """The luminosity kernel matches numpy, returning float32."""
    backdrop, lum = _rand(3, (32, 32, 3)), _rand(30, (32, 32))

    rust_out = terra_texture_rs.luminosity_blend(backdrop, lum)

    assert rust_out.dtype == np.float32
    assert rust_out.shape == backdrop.shape
    np.testing.assert_allclose(
        rust_out, _luminosity_blend_numpy(backdrop, lum), atol=_ATOL
    )


def test_rust_luminosity_blend_matches_numpy_with_out_of_gamut_values() -> None:
    """Both ClipColor branches match numpy.

    Pixels pushed below 0 and above 1 by the luminosity shift exercise the
    low-clip and high-clip branches. The kernel's fused low/high clip must
    match blend.py's sequential ``np.where`` reassignment exactly, which
    is only tested if some pixels actually hit both branches.
    """
    backdrop = np.array([[[0.95, 0.05, 0.5], [0.05, 0.95, 0.5]]], np.float32)
    lum = np.array([[0.99, 0.01]], np.float32)

    rust_out = terra_texture_rs.luminosity_blend(backdrop, lum)

    np.testing.assert_allclose(
        rust_out, _luminosity_blend_numpy(backdrop, lum), atol=_ATOL
    )


def test_rust_luminosity_blend_matches_numpy_on_large_array() -> None:
    """A 300 x 300 image covers any size-dependent (parallel) branch."""
    backdrop, lum = _rand(44, (300, 300, 3)), _rand(45, (300, 300))

    rust_out = terra_texture_rs.luminosity_blend(backdrop, lum)

    np.testing.assert_allclose(
        rust_out, _luminosity_blend_numpy(backdrop, lum), atol=_ATOL
    )


def test_rust_luminosity_blend_propagates_nan() -> None:
    """A NaN luminosity pixel gives NaN in all three output channels."""
    backdrop, lum = _rand(46, (6, 6, 3)), _rand(47, (6, 6))
    lum[2, 3] = np.nan

    rust_out = terra_texture_rs.luminosity_blend(backdrop, lum)

    assert np.all(np.isnan(rust_out[2, 3]))
    np.testing.assert_allclose(
        rust_out, _luminosity_blend_numpy(backdrop, lum), atol=_ATOL,
        equal_nan=True,
    )


def test_kernels_do_not_modify_inputs() -> None:
    """Every kernel writes a new array and leaves its inputs untouched."""
    a, b = _rand(48, (16, 16)), _rand(49, (16, 16))
    rgb, rgb2 = _rand(50, (16, 16, 3)), _rand(51, (16, 16, 3))
    originals = [x.copy() for x in (a, b, rgb, rgb2)]

    terra_texture_rs.soft_light(a, b)
    terra_texture_rs.luminosity_blend(rgb, a)
    if hasattr(terra_texture_rs, "soft_light_rgb"):
        terra_texture_rs.soft_light_rgb(rgb, rgb2)

    for current, original in zip((a, b, rgb, rgb2), originals):
        np.testing.assert_array_equal(current, original)


# ---------------------------------------------------------------------------
# Public API dispatch
# ---------------------------------------------------------------------------

def test_public_soft_light_dispatches_to_rust_for_float32() -> None:
    """2-D float32 input takes the fast path: bit-identical to the kernel."""
    a = np.full((16, 16), 0.4, dtype=np.float32)
    b = np.full((16, 16), 0.6, dtype=np.float32)

    np.testing.assert_array_equal(soft_light(a, b), terra_texture_rs.soft_light(a, b))


def test_public_luminosity_blend_dispatches_to_rust_for_float32() -> None:
    """float32 (H, W, 3) + (H, W) takes the fast path."""
    backdrop = np.full((16, 16, 3), 0.5, dtype=np.float32)
    lum = np.full((16, 16), 0.7, dtype=np.float32)

    np.testing.assert_array_equal(
        luminosity_blend(backdrop, lum),
        terra_texture_rs.luminosity_blend(backdrop, lum),
    )


def test_public_soft_light_multichannel_matches_numpy() -> None:
    """(H, W, 3) input, e.g. basemap.py's final compositing, is correct.

    Before the multichannel dispatch existed, 3-D input always fell back
    to numpy even with the extension built.
    """
    a, b = _rand(7, (48, 48, 3)), _rand(70, (48, 48, 3))

    dispatched = soft_light(a, b)

    assert dispatched.dtype == np.float32
    np.testing.assert_allclose(dispatched, _soft_light_numpy(a, b), atol=1e-4)


def test_public_soft_light_multichannel_matches_per_channel_2d_calls() -> None:
    """Public 3-D dispatch equals the 2-D kernel per channel, exactly.

    Holds whichever route runs: the fused kernel, or the per-channel loop
    on an older build.
    """
    a, b = _rand(8, (32, 32, 4)), _rand(80, (32, 32, 4))

    dispatched = soft_light(a, b)

    for channel in range(4):
        expected = terra_texture_rs.soft_light(
            np.ascontiguousarray(a[..., channel]),
            np.ascontiguousarray(b[..., channel]),
        )
        np.testing.assert_array_equal(dispatched[..., channel], expected)


@requires_rgb_kernel
def test_public_soft_light_multichannel_uses_fused_kernel_not_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With soft_light_rgb available, it's called once -- not the loop.

    Correctness alone can't tell the routes apart (both are right), so a
    spy proves which one actually ran.
    """
    calls: list[tuple[int, ...]] = []
    original: Callable[..., np.ndarray] = terra_texture_rs.soft_light_rgb

    def _spy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        calls.append(a.shape)
        return original(a, b)

    monkeypatch.setattr(terra_texture_rs, "soft_light_rgb", _spy)
    a = np.full((16, 16, 3), 0.4, dtype=np.float32)

    soft_light(a, np.full_like(a, 0.6))

    assert calls == [(16, 16, 3)]


@pytest.mark.parametrize(
    "make_view",
    [
        lambda a: a.T,
        lambda a: a[::2, ::3],
        lambda a: np.asfortranarray(a),
    ],
    ids=["transposed", "strided", "fortran-order"],
)
def test_public_soft_light_non_contiguous_matches_numpy(
    make_view: Callable[[np.ndarray], np.ndarray],
) -> None:
    """Non-contiguous float32 views are copied, then blended correctly."""
    a = make_view(_rand(52, (40, 60)))
    b = make_view(_rand(53, (40, 60)))

    out = soft_light(a, b)

    assert out.shape == a.shape
    np.testing.assert_allclose(out, _soft_light_numpy(a, b), atol=_ATOL)


@pytest.mark.parametrize("shape", [(16, 16), (16, 16, 3)], ids=["2d", "3d"])
def test_public_soft_light_falls_back_for_float64(shape: tuple[int, ...]) -> None:
    """float64 never takes the float32-only fast path, and stays float64."""
    a = np.full(shape, 0.4)
    b = np.full(shape, 0.6)

    out = soft_light(a, b)

    assert out.dtype == np.float64
    np.testing.assert_allclose(out, _soft_light_numpy(a, b))


def test_rust_and_numpy_paths_agree(monkeypatch: pytest.MonkeyPatch) -> None:
    """Public results are the same with and without the extension."""
    a, b = _rand(54, (64, 64)), _rand(55, (64, 64))
    rgb, rgb2 = _rand(56, (64, 64, 3)), _rand(57, (64, 64, 3))

    with_rust = [soft_light(a, b), soft_light(rgb, rgb2), luminosity_blend(rgb, a)]
    monkeypatch.setattr(blend_module, "_rust", None)
    without = [soft_light(a, b), soft_light(rgb, rgb2), luminosity_blend(rgb, a)]

    for fast, slow in zip(with_rust, without):
        np.testing.assert_allclose(fast, slow, atol=_ATOL)
