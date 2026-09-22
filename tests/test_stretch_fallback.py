"""
Fallback and diagnostics tests for :mod:`TerraTexture.stretch`'s Rust hook.

These tests never need the real compiled extension. They swap
``stretch._rust`` for small fake modules, or reload ``stretch`` against a
fake ``terra_texture_rs`` that fails on import, so they always run -- on
CI, on machines without a Rust toolchain, and in core-only installs.

Covers:

- Dispatch rules: which inputs go to the kernel and which to numpy.
- Runtime fallback: a kernel that raises logs a warning and returns the
  numpy result instead of propagating the error.
- Stale builds: an extension without a ``stretch_std`` function is
  skipped.
- Import diagnostics: "not installed" is logged quietly at DEBUG,
  "installed but broken" loudly as a WARNING, with the reason kept in
  ``_RUST_IMPORT_ERROR``.

Kernel-vs-numpy *numerical* parity needs the real extension and lives in
``test_stretch_rust.py``.

Examples:
    Run just these tests::

        pytest tests/test_stretch_fallback.py -v
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

import TerraTexture.stretch as stretch_module
from TerraTexture.stretch import stretch_std


# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------

def _numpy_reference(arr: np.ndarray, n_std: float = 4) -> np.ndarray:
    """
    Compute the stretch with the plain nanmean + nanstd formula.

    Args:
        arr (np.ndarray): Input array; NaNs ignored.
        n_std (float): Half-width of the stretch window, in std units.

    Returns:
        np.ndarray: Stretched array in ``[0, 1]``, NaN where ``arr`` is NaN.
    """
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    lo, hi = mean - n_std * std, mean + n_std * std
    return np.clip((arr - lo) / (hi - lo + 1e-12), 0, 1)


class _RecordingKernel:
    """
    Fake extension whose ``stretch_std`` records calls and returns a marker.

    Attributes:
        calls (list[tuple[np.ndarray, float]]): ``(arr, n_std)`` per call.
    """

    def __init__(self) -> None:
        """Create a kernel with no recorded calls."""
        self.calls: list[tuple[np.ndarray, float]] = []

    def stretch_std(self, arr: np.ndarray, n_std: float) -> str:
        """
        Record the call and return a sentinel instead of real output.

        Args:
            arr (np.ndarray): Array passed by the dispatcher.
            n_std (float): Stretch width passed by the dispatcher.

        Returns:
            str: ``"RUST"``, so tests can tell the kernel was used.
        """
        self.calls.append((arr, n_std))
        return "RUST"


class _FailingKernel:
    """Fake extension whose ``stretch_std`` always raises."""

    def stretch_std(self, arr: np.ndarray, n_std: float) -> Any:
        """
        Simulate a kernel failure.

        Args:
            arr (np.ndarray): Ignored.
            n_std (float): Ignored.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("simulated kernel failure")


@pytest.fixture
def recording_kernel(monkeypatch: pytest.MonkeyPatch) -> _RecordingKernel:
    """
    Install a :class:`_RecordingKernel` as ``stretch._rust``.

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest's monkeypatch fixture.

    Returns:
        _RecordingKernel: The installed fake, for inspecting calls.
    """
    kernel = _RecordingKernel()
    monkeypatch.setattr(stretch_module, "_rust", kernel)
    return kernel


@pytest.fixture
def float32_grid() -> np.ndarray:
    """
    A reproducible 2-D float32 array with a NaN void.

    Returns:
        np.ndarray: 32 x 32 float32 array.
    """
    rng = np.random.default_rng(30)
    arr = ((rng.random((32, 32)) - 0.5) * 100).astype(np.float32)
    arr[4:8, 4:8] = np.nan
    return arr


# ---------------------------------------------------------------------------
# Dispatch rules
# ---------------------------------------------------------------------------

def test_2d_float32_dispatches_to_kernel(
    recording_kernel: _RecordingKernel, float32_grid: np.ndarray,
) -> None:
    """A 2-D float32 ndarray goes to the kernel, with n_std as a float."""
    assert stretch_std(float32_grid, 3) == "RUST"

    (passed_arr, passed_n_std), = recording_kernel.calls
    assert passed_arr.flags.c_contiguous
    assert isinstance(passed_n_std, float) and passed_n_std == 3.0


def test_non_contiguous_input_is_made_contiguous(
    recording_kernel: _RecordingKernel, float32_grid: np.ndarray,
) -> None:
    """Strided views are copied to C order before reaching the kernel."""
    stretch_std(float32_grid[::2, ::3], 4)

    passed_arr, _ = recording_kernel.calls[0]
    assert passed_arr.flags.c_contiguous


@pytest.mark.parametrize(
    "arr",
    [
        np.zeros((4, 4), dtype=np.float64),
        np.zeros(16, dtype=np.float32),
        np.zeros((2, 4, 4), dtype=np.float32),
        np.zeros((4, 4), dtype=np.int32),
        [[0.0, 1.0], [2.0, 3.0]],
    ],
    ids=["float64", "1-d", "3-d", "int32", "list"],
)
def test_other_inputs_use_numpy(
    recording_kernel: _RecordingKernel, arr: Any,
) -> None:
    """Anything but a 2-D float32 ndarray bypasses the kernel."""
    result = stretch_std(arr, 4)

    assert recording_kernel.calls == []
    assert isinstance(result, np.ndarray)


def test_invalid_arguments_never_reach_kernel(
    recording_kernel: _RecordingKernel, float32_grid: np.ndarray,
) -> None:
    """Validation runs before dispatch."""
    with pytest.raises(ValueError, match="n_std"):
        stretch_std(float32_grid, -1)
    with pytest.raises(ValueError, match="empty"):
        stretch_std(np.zeros((0, 3), dtype=np.float32), 4)

    assert recording_kernel.calls == []


# ---------------------------------------------------------------------------
# Runtime fallback
# ---------------------------------------------------------------------------

def test_failing_kernel_falls_back_to_numpy(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    float32_grid: np.ndarray,
) -> None:
    """A kernel exception is logged and the numpy result is returned."""
    monkeypatch.setattr(stretch_module, "_rust", _FailingKernel())

    with caplog.at_level(logging.WARNING, logger=stretch_module.__name__):
        out = stretch_std(float32_grid, 4)

    np.testing.assert_allclose(
        out, _numpy_reference(float32_grid, 4), atol=1e-5, equal_nan=True
    )
    assert "Rust stretch_std failed" in caplog.text
    assert "simulated kernel failure" in caplog.text


def test_stale_build_without_kernel_uses_numpy(
    monkeypatch: pytest.MonkeyPatch, float32_grid: np.ndarray,
) -> None:
    """An extension lacking ``stretch_std`` is skipped, not called."""
    monkeypatch.setattr(stretch_module, "_rust", SimpleNamespace())

    out = stretch_std(float32_grid, 4)

    np.testing.assert_allclose(
        out, _numpy_reference(float32_grid, 4), atol=1e-5, equal_nan=True
    )


# ---------------------------------------------------------------------------
# Import diagnostics (reloading stretch against fake extensions)
# ---------------------------------------------------------------------------

@pytest.fixture
def reload_stretch(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Any]:
    """
    Provide a function that reloads ``stretch`` with a given fake extension.

    Restores the real state afterwards. The original ``terra_texture_rs``
    module object (if any) is put back into ``sys.modules`` *before* the
    final reload: a PyO3 extension cannot be initialised twice in one
    process, so it must be reused, never re-imported.

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest's monkeypatch fixture.

    Yields:
        Callable[[Path | None], ModuleType]: Reloads ``stretch`` with the
            given directory prepended to ``sys.path`` (or none), after
            evicting any cached ``terra_texture_rs``; returns the module.
    """
    original = sys.modules.get("terra_texture_rs")

    def _reload(fake_dir: Path | None) -> ModuleType:
        sys.modules.pop("terra_texture_rs", None)
        if fake_dir is not None:
            monkeypatch.syspath_prepend(str(fake_dir))
        return importlib.reload(stretch_module)

    yield _reload

    monkeypatch.undo()  # drop the fake sys.path entry first
    sys.modules.pop("terra_texture_rs", None)
    if original is not None:
        sys.modules["terra_texture_rs"] = original
    importlib.reload(stretch_module)


def _write_fake_extension(directory: Path, source: str) -> Path:
    """
    Write a pure-Python ``terra_texture_rs.py`` into ``directory``.

    Args:
        directory (Path): Directory to write into.
        source (str): Module source code.

    Returns:
        Path: ``directory``, for passing to the reload fixture.
    """
    (directory / "terra_texture_rs.py").write_text(source)
    return directory


def test_broken_extension_logs_warning(
    tmp_path: Path, reload_stretch: Any, caplog: pytest.LogCaptureFixture,
) -> None:
    """An installed-but-broken extension warns and records the reason."""
    fake = _write_fake_extension(
        tmp_path, 'raise ImportError("symbol not found: _PyFake")\n'
    )

    with caplog.at_level(logging.DEBUG, logger=stretch_module.__name__):
        module = reload_stretch(fake)

    assert module._rust is None
    assert isinstance(module._RUST_IMPORT_ERROR, ImportError)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "installed but failed to import" in warnings[0].getMessage()
    assert "symbol not found" in warnings[0].getMessage()


def test_extension_missing_its_own_dependency_counts_as_broken(
    tmp_path: Path, reload_stretch: Any, caplog: pytest.LogCaptureFixture,
) -> None:
    """A ModuleNotFoundError for a *different* module is still "broken".

    Only a missing ``terra_texture_rs`` itself means "not installed".
    """
    fake = _write_fake_extension(tmp_path, "import some_missing_dependency\n")

    with caplog.at_level(logging.WARNING, logger=stretch_module.__name__):
        module = reload_stretch(fake)

    assert module._rust is None
    assert "installed but failed to import" in caplog.text
    assert "some_missing_dependency" in caplog.text


def test_stale_extension_warns_at_import(
    tmp_path: Path, reload_stretch: Any, caplog: pytest.LogCaptureFixture,
) -> None:
    """An extension that loads but lacks the kernel warns once at import."""
    fake = _write_fake_extension(tmp_path, "OTHER_KERNEL = True\n")

    with caplog.at_level(logging.WARNING, logger=stretch_module.__name__):
        module = reload_stretch(fake)

    assert module._rust is not None
    assert module._RUST_IMPORT_ERROR is None
    assert "no stretch_std kernel" in caplog.text


def test_missing_extension_logs_debug_only(
    reload_stretch: Any,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A simply-not-installed extension is quiet: DEBUG, no WARNING."""
    # A None entry in sys.modules makes the import raise
    # ModuleNotFoundError, shadowing any real installation.
    monkeypatch.setitem(sys.modules, "terra_texture_rs", None)

    with caplog.at_level(logging.DEBUG, logger=stretch_module.__name__):
        module = importlib.reload(stretch_module)

    assert module._rust is None
    assert isinstance(module._RUST_IMPORT_ERROR, ModuleNotFoundError)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert "not installed" in caplog.text
