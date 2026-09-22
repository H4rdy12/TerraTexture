"""
Tests for :mod:`TerraTexture.blend` that never need the compiled extension.

Always runs, including on CI and in installs without a Rust toolchain.
Everything that needs the real ``terra_texture_rs`` lives in
``test_blend_rust.py``, which skips when it isn't built.

Covers:

- Blend-mode maths: hand-computed soft-light values, the fixed points
  and identity of the formula, darken/lighten behaviour, and the
  Luminosity mode's hue/saturation preservation and gamut clipping.
- Input handling: dtype preservation, Python scalars, broadcasting and
  NaN propagation.
- Error handling: integer imagery, non-numeric input, RGBA backdrops and
  mismatched shapes.
- Rust dispatch and fallback, using fake kernels: which inputs reach
  which kernel, the per-channel path for stale builds, kernels that
  raise, and import diagnostics.

The numpy path is forced (``_rust = None``) for the maths tests, so they
test the reference implementation even on machines where Rust is built.

Examples:
    Run just these tests::

        pytest tests/test_blend.py -v
"""

from __future__ import annotations

import importlib
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

import TerraTexture.blend as blend_module
from TerraTexture.blend import _lum, luminosity_blend, soft_light


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def numpy_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Force the pure-numpy path by hiding any Rust extension.

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest's monkeypatch fixture.

    Returns:
        None
    """
    monkeypatch.setattr(blend_module, "_rust", None)


class _RecordingKernels:
    """
    Fake extension whose kernels record calls and delegate to numpy.

    Delegating (rather than returning a sentinel) keeps results correct,
    so tests can check both *which* kernel ran and that output is right.

    Attributes:
        calls (list[str]): Kernel names, in call order.
    """

    def __init__(self, with_rgb: bool = True) -> None:
        """
        Create the fake.

        Args:
            with_rgb (bool): Whether to export ``soft_light_rgb``. ``False``
                simulates an extension built before that kernel existed.
        """
        self.calls: list[str] = []
        if with_rgb:
            self.soft_light_rgb = self._soft_light_rgb

    def soft_light(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Record and compute 2-D soft light.

        Args:
            a (np.ndarray): Base array.
            b (np.ndarray): Blend array.

        Returns:
            np.ndarray: numpy soft light of ``a`` and ``b``.
        """
        self.calls.append("soft_light")
        return blend_module._soft_light_numpy(a, b)

    def _soft_light_rgb(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Record and compute 3-D soft light.

        Args:
            a (np.ndarray): Base array.
            b (np.ndarray): Blend array.

        Returns:
            np.ndarray: numpy soft light of ``a`` and ``b``.
        """
        self.calls.append("soft_light_rgb")
        return blend_module._soft_light_numpy(a, b)

    def luminosity_blend(self, rgb: np.ndarray, lum: np.ndarray) -> np.ndarray:
        """
        Record and compute the luminosity blend.

        Args:
            rgb (np.ndarray): Backdrop.
            lum (np.ndarray): Luminosity.

        Returns:
            np.ndarray: numpy luminosity blend.
        """
        self.calls.append("luminosity_blend")
        return blend_module._luminosity_blend_numpy(rgb, lum)


@pytest.fixture
def fake_rust(monkeypatch: pytest.MonkeyPatch) -> _RecordingKernels:
    """
    Install a complete :class:`_RecordingKernels` as ``blend._rust``.

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest's monkeypatch fixture.

    Returns:
        _RecordingKernels: The installed fake.
    """
    kernels = _RecordingKernels()
    monkeypatch.setattr(blend_module, "_rust", kernels)
    return kernels


def _rand(seed: int, shape: tuple[int, ...], dtype: Any = np.float64) -> np.ndarray:
    """
    Build reproducible random values in ``[0, 1)``.

    Args:
        seed (int): RNG seed.
        shape (tuple[int, ...]): Array shape.
        dtype (Any): Output dtype.

    Returns:
        np.ndarray: The array.
    """
    return np.random.default_rng(seed).random(shape).astype(dtype)


# ---------------------------------------------------------------------------
# soft_light: maths
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("numpy_only")
class TestSoftLightMaths:
    """Properties of the Photoshop soft-light formula (numpy path)."""

    def test_output_in_unit_range(self) -> None:
        """Random inputs in [0, 1] give output in [0, 1]."""
        out = soft_light(_rand(0, (20, 20)), _rand(1, (20, 20)))

        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_neutral_grey_blend_is_identity(self) -> None:
        """A 0.5-grey blend layer leaves the base unchanged."""
        base = np.linspace(0, 1, 50)

        out = soft_light(base, np.full_like(base, 0.5))

        np.testing.assert_allclose(out, base, atol=1e-6)

    @pytest.mark.parametrize(
        ("base", "blend", "expected"),
        [
            (0.25, 0.25, 0.15625),  # 2ab + a^2 (1 - 2b)
            (0.25, 0.75, 0.375),    # 2a (1 - b) + sqrt(a) (2b - 1)
            (0.25, 0.0, 0.0625),    # blend 0 -> a^2
            (0.25, 1.0, 0.5),       # blend 1 -> sqrt(a)
        ],
        ids=["dark-branch", "light-branch", "blend-black", "blend-white"],
    )
    def test_hand_computed_values(
        self, base: float, blend: float, expected: float,
    ) -> None:
        """Both formula branches and their extremes match by hand."""
        out = soft_light(np.array([base]), np.array([blend]))

        np.testing.assert_allclose(out, [expected])

    @pytest.mark.parametrize("fixed", [0.0, 1.0])
    def test_black_and_white_base_are_fixed_points(self, fixed: float) -> None:
        """A pure black or white base is unchanged by any blend."""
        blend = np.linspace(0, 1, 21)

        out = soft_light(np.full_like(blend, fixed), blend)

        np.testing.assert_allclose(out, fixed, atol=1e-12)

    def test_dark_blend_darkens_light_blend_lightens(self) -> None:
        """blend < 0.5 darkens mid-tones; blend > 0.5 lightens them."""
        base = np.full(3, 0.4)

        out = soft_light(base, np.array([0.2, 0.5, 0.8]))

        assert out[0] < 0.4 < out[2]
        assert out[1] == pytest.approx(0.4)

    def test_continuous_across_branch_boundary(self) -> None:
        """No jump where the formula switches branch at blend = 0.5."""
        base = np.full(2, 0.3)

        out = soft_light(base, np.array([0.5, 0.5 + 1e-9]))

        assert out[1] == pytest.approx(out[0], abs=1e-8)

    def test_out_of_range_base_does_not_produce_nan(self) -> None:
        """Negative base values can't create NaN in the sqrt branch."""
        out = soft_light(np.array([-0.5, 1.5]), np.array([0.9, 0.9]))

        assert np.all(np.isfinite(out))
        assert out.min() >= 0.0 and out.max() <= 1.0


# ---------------------------------------------------------------------------
# soft_light: input handling
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("numpy_only")
class TestSoftLightInputs:
    """Dtypes, scalars, broadcasting and NaN (numpy path)."""

    def test_float32_stays_float32(self) -> None:
        """float32 in gives float32 out."""
        a = _rand(2, (4, 4), np.float32)

        assert soft_light(a, a).dtype == np.float32

    def test_mixed_precision_promotes_to_float64(self) -> None:
        """float32 with float64 gives float64."""
        out = soft_light(_rand(3, (4, 4), np.float32), _rand(4, (4, 4)))

        assert out.dtype == np.float64

    @pytest.mark.parametrize("scalar", [0.7, 1, 0])
    def test_python_scalar_blend_keeps_float32(self, scalar: float) -> None:
        """A Python int/float blend is accepted and doesn't promote float32."""
        a = _rand(5, (4, 4), np.float32)

        out = soft_light(a, scalar)

        assert out.dtype == np.float32
        expected = soft_light(a, np.full_like(a, scalar))
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_broadcasting_greyscale_over_rgb(self) -> None:
        """An (H, W, 1) blend broadcasts across (H, W, 3) channels."""
        rgb = _rand(6, (5, 5, 3))
        grey = _rand(7, (5, 5, 1))

        out = soft_light(rgb, grey)

        assert out.shape == (5, 5, 3)
        np.testing.assert_allclose(
            out[..., 1], soft_light(rgb[..., 1], grey[..., 0])
        )

    def test_accepts_lists(self) -> None:
        """Plain nested lists are converted like arrays."""
        out = soft_light([[0.25, 0.25]], [[0.25, 0.75]])

        np.testing.assert_allclose(out, [[0.15625, 0.375]])

    @pytest.mark.parametrize("which", ["base", "blend"])
    def test_nan_propagates(self, which: str) -> None:
        """A NaN in either input gives NaN at that pixel only."""
        a, b = _rand(8, (4, 4)), _rand(9, (4, 4))
        (a if which == "base" else b)[1, 2] = np.nan

        out = soft_light(a, b)

        assert np.isnan(out[1, 2])
        assert np.isnan(out).sum() == 1


# ---------------------------------------------------------------------------
# soft_light: errors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [np.uint8, np.int32, np.bool_])
def test_soft_light_rejects_integer_arrays(dtype: Any) -> None:
    """Integer/bool arrays raise TypeError pointing at the /255 fix."""
    image = np.zeros((4, 4), dtype=dtype)

    with pytest.raises(TypeError, match="/ 255"):
        soft_light(image, np.zeros((4, 4)))


def test_soft_light_rejects_non_numeric() -> None:
    """String input raises TypeError."""
    with pytest.raises(TypeError, match="numeric"):
        soft_light(np.array(["a", "b"]), np.array([0.5, 0.5]))


def test_soft_light_rejects_incompatible_shapes() -> None:
    """Non-broadcastable shapes raise ValueError naming both."""
    with pytest.raises(ValueError, match=r"\(2, 3\).*\(4, 5\)"):
        soft_light(np.zeros((2, 3)), np.zeros((4, 5)))


# ---------------------------------------------------------------------------
# luminosity_blend: maths
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("numpy_only")
class TestLuminosityBlendMaths:
    """Properties of SetLum / ClipColor (numpy path)."""

    def test_preserves_hue_and_saturation(self) -> None:
        """Only lightness changes: the red pixel stays red, at the new lum."""
        backdrop = np.array([[[0.8, 0.2, 0.2]]])

        out = luminosity_blend(backdrop, np.array([[0.9]]))

        assert _lum(out)[0, 0] == pytest.approx(0.9, abs=1e-5)
        assert out[0, 0, 0] > out[0, 0, 1]
        assert out[0, 0, 0] > out[0, 0, 2]

    def test_in_gamut_shift_is_uniform_across_channels(self) -> None:
        """Without clipping, every channel moves by the same amount."""
        backdrop = np.array([[[0.5, 0.4, 0.3]]])

        out = luminosity_blend(backdrop, np.array([[0.45]]))

        shift = out - backdrop
        np.testing.assert_allclose(shift, shift[..., :1].repeat(3, axis=-1))
        assert _lum(out)[0, 0] == pytest.approx(0.45)

    def test_grey_backdrop_becomes_target_grey(self) -> None:
        """A neutral grey stays neutral, at exactly the target lightness."""
        out = luminosity_blend(np.full((1, 1, 3), 0.2), np.array([[0.8]]))

        np.testing.assert_allclose(out, 0.8)

    def test_out_of_gamut_keeps_luminosity_and_channel_order(self) -> None:
        """ClipColor pulls saturated pixels into gamut without reordering.

        Pushes one pixel far above 1 and one below 0 to hit both clip
        branches.
        """
        backdrop = np.array([[[0.95, 0.05, 0.5], [0.05, 0.95, 0.5]]])
        target = np.array([[0.99, 0.01]])

        out = luminosity_blend(backdrop, target)

        assert out.min() >= 0.0 and out.max() <= 1.0
        np.testing.assert_allclose(_lum(out), target, atol=1e-6)
        for pixel in range(2):
            assert np.array_equal(
                np.argsort(out[0, pixel]), np.argsort(backdrop[0, pixel])
            )

    def test_random_output_in_unit_range_with_target_luminosity(self) -> None:
        """Random inputs stay in [0, 1] and hit the requested luminosity."""
        backdrop = _rand(10, (10, 10, 3))
        target = _rand(11, (10, 10))

        out = luminosity_blend(backdrop, target)

        assert out.min() >= 0.0 and out.max() <= 1.0
        np.testing.assert_allclose(_lum(out), target, atol=1e-6)

    def test_float32_stays_float32(self) -> None:
        """float32 in gives float32 out."""
        out = luminosity_blend(
            _rand(12, (4, 4, 3), np.float32), _rand(13, (4, 4), np.float32)
        )

        assert out.dtype == np.float32

    def test_nan_luminosity_propagates(self) -> None:
        """A NaN luminosity pixel gives NaN in all three channels."""
        target = _rand(14, (3, 3))
        target[1, 1] = np.nan

        out = luminosity_blend(_rand(15, (3, 3, 3)), target)

        assert np.all(np.isnan(out[1, 1]))
        assert np.isnan(out).sum() == 3


# ---------------------------------------------------------------------------
# luminosity_blend: errors
# ---------------------------------------------------------------------------

def test_luminosity_blend_rgba_suggests_slicing() -> None:
    """An RGBA backdrop is rejected with a hint to pass rgba[..., :3].

    Previously alpha was silently included in gamut clipping.
    """
    with pytest.raises(ValueError, match=r"rgba\[\.\.\., :3\]"):
        luminosity_blend(np.zeros((2, 2, 4)), np.zeros((2, 2)))


def test_luminosity_blend_rejects_2d_backdrop() -> None:
    """A greyscale backdrop has no channel axis and is rejected."""
    with pytest.raises(ValueError, match="3 channels"):
        luminosity_blend(np.zeros((4, 5)), np.zeros(4))


def test_luminosity_blend_rejects_mismatched_luminosity() -> None:
    """The luminosity grid must match the backdrop's (H, W)."""
    with pytest.raises(ValueError, match=r"\(3, 3\).*\(2, 2\)"):
        luminosity_blend(np.zeros((2, 2, 3)), np.zeros((3, 3)))


def test_luminosity_blend_rejects_uint8_imagery() -> None:
    """uint8 imagery raises TypeError pointing at the /255 fix."""
    with pytest.raises(TypeError, match="/ 255"):
        luminosity_blend(np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2)))


# ---------------------------------------------------------------------------
# Rust dispatch (fake kernels)
# ---------------------------------------------------------------------------

def test_2d_float32_uses_soft_light_kernel(fake_rust: _RecordingKernels) -> None:
    """Two same-shape 2-D float32 arrays go to the 2-D kernel."""
    a = _rand(20, (8, 8), np.float32)

    soft_light(a, a)

    assert fake_rust.calls == ["soft_light"]


def test_3d_float32_uses_fused_rgb_kernel(fake_rust: _RecordingKernels) -> None:
    """3-D float32 input calls soft_light_rgb once, not a per-channel loop."""
    rgb = _rand(21, (8, 8, 3), np.float32)

    soft_light(rgb, rgb)

    assert fake_rust.calls == ["soft_light_rgb"]


def test_3d_on_stale_build_loops_2d_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without soft_light_rgb, the 2-D kernel runs once per channel."""
    kernels = _RecordingKernels(with_rgb=False)
    monkeypatch.setattr(blend_module, "_rust", kernels)
    rgb = _rand(22, (8, 8, 4), np.float32)

    out = soft_light(rgb, rgb[::-1].copy())

    assert kernels.calls == ["soft_light"] * 4
    np.testing.assert_allclose(
        out, blend_module._soft_light_numpy(rgb, rgb[::-1]), atol=1e-6
    )


def test_luminosity_uses_kernel(fake_rust: _RecordingKernels) -> None:
    """float32 (H, W, 3) + (H, W) goes to the luminosity kernel."""
    luminosity_blend(_rand(23, (8, 8, 3), np.float32), _rand(24, (8, 8), np.float32))

    assert fake_rust.calls == ["luminosity_blend"]


@pytest.mark.parametrize(
    ("base", "blend"),
    [
        (np.zeros((4, 4)), np.zeros((4, 4))),
        (np.zeros((4, 4), np.float32), np.zeros((4, 4))),
        (np.zeros((4, 4, 3), np.float32), np.zeros((4, 4, 1), np.float32)),
        (np.zeros((4, 4), np.float32), 0.5),
        (np.zeros(4, np.float32), np.zeros(4, np.float32)),
    ],
    ids=["float64", "mixed-precision", "broadcast", "scalar", "1-d"],
)
def test_soft_light_non_fast_path_inputs_use_numpy(
    fake_rust: _RecordingKernels, base: Any, blend: Any,
) -> None:
    """Anything but same-shape 2-D/3-D float32 bypasses the kernels."""
    soft_light(base, blend)

    assert fake_rust.calls == []


def test_non_contiguous_inputs_reach_kernel_contiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strided views are copied to C order before the kernel sees them."""
    seen: list[bool] = []

    def _kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        seen.extend([a.flags.c_contiguous, b.flags.c_contiguous])
        return blend_module._soft_light_numpy(a, b)

    monkeypatch.setattr(blend_module, "_rust", SimpleNamespace(soft_light=_kernel))
    a = _rand(25, (16, 16), np.float32)

    soft_light(a[::2, ::2], a.T[::2, ::2])

    assert seen == [True, True]


def test_invalid_input_never_reaches_kernel(fake_rust: _RecordingKernels) -> None:
    """Validation runs before dispatch."""
    with pytest.raises(ValueError):
        luminosity_blend(np.zeros((2, 2, 4), np.float32), np.zeros((2, 2), np.float32))

    assert fake_rust.calls == []


# ---------------------------------------------------------------------------
# Rust fallback (kernels that raise)
# ---------------------------------------------------------------------------

def _boom(*args: Any) -> Any:
    """
    Simulate a kernel failure.

    Args:
        *args (Any): Ignored.

    Raises:
        RuntimeError: Always.
    """
    raise RuntimeError("simulated kernel failure")


_GREY = _rand(30, (6, 6), np.float32)
_RGB = _rand(31, (6, 6, 3), np.float32)
_RGB2 = _rand(32, (6, 6, 3), np.float32)


@pytest.mark.parametrize(
    ("kernels", "call"),
    [
        ({"soft_light": _boom}, lambda: soft_light(_GREY, _GREY[::-1].copy())),
        ({"soft_light": _boom, "soft_light_rgb": _boom},
         lambda: soft_light(_RGB, _RGB2)),
        ({"soft_light": _boom}, lambda: soft_light(_RGB, _RGB2)),
        ({"luminosity_blend": _boom}, lambda: luminosity_blend(_RGB, _GREY)),
    ],
    ids=["soft_light-2d", "soft_light_rgb", "per-channel-loop", "luminosity_blend"],
)
def test_failing_kernel_falls_back_to_numpy(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    kernels: dict[str, Any],
    call: Any,
) -> None:
    """A raising kernel logs a warning and the numpy result is returned."""
    monkeypatch.setattr(blend_module, "_rust", SimpleNamespace(**kernels))
    with caplog.at_level(logging.WARNING, logger=blend_module.__name__):
        with_failure = call()

    monkeypatch.setattr(blend_module, "_rust", None)
    numpy_result = call()

    np.testing.assert_array_equal(with_failure, numpy_result)
    assert "falling back to numpy" in caplog.text
    assert "simulated kernel failure" in caplog.text


# ---------------------------------------------------------------------------
# Import diagnostics (reloading blend against fake extensions)
# ---------------------------------------------------------------------------

@pytest.fixture
def reload_blend(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """
    Provide a function that reloads ``blend`` with a fake extension.

    Restores the real state afterwards. The original ``terra_texture_rs``
    module object (if any) is put back into ``sys.modules`` before the
    final reload: a PyO3 extension can't be initialised twice in one
    process, so it must be reused, never re-imported.

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest's monkeypatch fixture.

    Yields:
        Callable[[str, Path], ModuleType]: Writes the given source as a
            fake ``terra_texture_rs`` in the given directory and reloads
            ``blend`` against it.
    """
    original = sys.modules.get("terra_texture_rs")

    def _reload(source: str, directory: Path) -> ModuleType:
        (directory / "terra_texture_rs.py").write_text(source)
        sys.modules.pop("terra_texture_rs", None)
        monkeypatch.syspath_prepend(str(directory))
        return importlib.reload(blend_module)

    yield _reload

    monkeypatch.undo()
    sys.modules.pop("terra_texture_rs", None)
    if original is not None:
        sys.modules["terra_texture_rs"] = original
    importlib.reload(blend_module)


def test_broken_extension_logs_warning(
    tmp_path: Path, reload_blend: Any, caplog: pytest.LogCaptureFixture,
) -> None:
    """An installed-but-broken extension warns and records the reason."""
    with caplog.at_level(logging.DEBUG, logger=blend_module.__name__):
        module = reload_blend('raise ImportError("symbol not found")\n', tmp_path)

    assert module._rust is None
    assert isinstance(module._RUST_IMPORT_ERROR, ImportError)
    assert "installed but failed to import" in caplog.text


def test_stale_extension_reports_each_missing_kernel(
    tmp_path: Path, reload_blend: Any, caplog: pytest.LogCaptureFixture,
) -> None:
    """A build with only soft_light warns about the two missing kernels."""
    with caplog.at_level(logging.WARNING, logger=blend_module.__name__):
        module = reload_blend("def soft_light(a, b):\n    return a\n", tmp_path)

    assert module._rust is not None
    assert "no soft_light_rgb kernel" in caplog.text
    assert "no luminosity_blend kernel" in caplog.text
    assert "no soft_light kernel" not in caplog.text


def test_missing_extension_is_quiet(
    monkeypatch: pytest.MonkeyPatch,
    reload_blend: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A simply-not-installed extension logs at DEBUG, never WARNING."""
    monkeypatch.setitem(sys.modules, "terra_texture_rs", None)

    with caplog.at_level(logging.DEBUG, logger=blend_module.__name__):
        module = importlib.reload(blend_module)

    assert module._rust is None
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert "not installed" in caplog.text
