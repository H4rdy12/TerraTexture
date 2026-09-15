#!/usr/bin/env python3
"""
Benchmark numpy vs Rust (terra_texture_rs) for TerraTexture.derivatives's
curvatures() and hillshade() -- both wall-clock time and peak memory,
across a range of DEM sizes.

Companion to scripts/benchmark_blend.py -- same structure, same
subprocess-isolation approach, adapted for the derivatives module's two
functions instead of blend's two functions. See that file's docstring
for the full rationale (repeated only briefly below).

Requires the Rust extension to actually be built to get Rust numbers at
all (see rust/README / Cargo.toml comments for `maturin develop
--release`) -- without it, this still runs and reports numpy-only
numbers, it just can't show a comparison.

Usage:
    uv run python benchmark_derivatives.py
    uv run python benchmark_derivatives.py --sizes 64,256,1024,4096
    uv run python benchmark_derivatives.py --iterations 20

Memory is measured via peak RSS (resource.getrusage().ru_maxrss), a
monotonic high-water-mark for the whole process -- to get an accurate
PER-SIZE reading rather than a running maximum contaminated by earlier,
larger runs in the same process, each (size, backend, func) combination
is measured in its own fresh subprocess.

Note on what each backend actually calls, since -- unlike blend.py --
derivatives.py doesn't factor its numpy math out into a standalone
`_curvatures_numpy()`/`_hillshade_numpy()` function:
  - "numpy": a reference implementation kept local to this script (see
    `_numpy_curvatures`/`_numpy_hillshade` below), replicating
    derivatives.py's numpy branch exactly -- same formulas, same
    `_derivatives()` helper import, no NaN-fill preprocessing (the
    synthetic DEM here is dense/NaN-free, so that overhead would only
    blur the kernel-vs-kernel comparison).
  - "rust": calls `terra_texture_rs.curvatures()`/`.hillshade()`
    directly, same NaN-free-input assumption, for a fair apples-to-apples
    comparison against "numpy" above.
  - "dispatch": calls the real public `curvatures()`/`hillshade()`,
    whichever path it internally picks -- this DOES include the
    NaN-fill/remask overhead real callers pay, so it will read a little
    slower than "rust" alone even when Rust is doing the actual math.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

DEFAULT_SIZES = [16, 64, 128, 256, 512, 1024, 2048, 4096]

AZIMUTH, ALTITUDE = 315.0, 45.0


def _numpy_curvatures(dem, cellsize):
    """Reference numpy implementation, lifted from derivatives.py's
    curvatures() (minus NaN-fill/remask -- see module docstring)."""
    import numpy as np
    from TerraTexture.derivatives import _derivatives

    p, q, r, t, s = _derivatives(dem, cellsize)
    p2q2 = p ** 2 + q ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        profile = -(r * p ** 2 + 2 * s * p * q + t * q ** 2) / (p2q2 * (1 + p2q2) ** 1.5)
        planform = -(r * q ** 2 - 2 * s * p * q + t * p ** 2) / (p2q2 ** 1.5)
    flat = p2q2 < 1e-9
    profile = np.where(flat, 0.0, np.nan_to_num(profile, nan=0.0, posinf=0.0, neginf=0.0))
    planform = np.where(flat, 0.0, np.nan_to_num(planform, nan=0.0, posinf=0.0, neginf=0.0))
    return profile, planform


def _numpy_hillshade(dem, cellsize, azimuth=AZIMUTH, altitude=ALTITUDE):
    """Reference numpy implementation, lifted from derivatives.py's
    hillshade() (minus NaN-fill/remask -- see module docstring)."""
    import numpy as np

    az = np.float32(np.radians(360.0 - azimuth + 90))
    alt = np.float32(np.radians(altitude))
    zy, zx = np.gradient(dem, cellsize)
    slope = np.pi / 2 - np.arctan(np.hypot(zx, zy))
    aspect = np.arctan2(-zx, zy)
    shaded = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    return np.clip(shaded, 0, 1)


def _synthetic_dem(size):
    """A dense, NaN-free, non-degenerate float32 DEM -- bumpy enough that
    neither curvature formula collapses to the flat-cell (p2q2 < 1e-9)
    fast-out for most pixels, so the benchmark actually exercises the
    full formula rather than an early return."""
    import numpy as np

    rng = np.random.default_rng(0)
    i, j = np.mgrid[0:size, 0:size].astype(np.float32)
    dem = (
        np.sin(i * 0.05) * 20.0 + np.cos(j * 0.07) * 15.0
        + rng.standard_normal((size, size)).astype(np.float32) * 2.0
    )
    return dem.astype(np.float32)


def _run_one(size, backend, func, iterations):
    """Run inside an isolated subprocess: time+memory for one
    (size, backend, func) combination. Prints a JSON result to stdout."""
    import resource

    from TerraTexture.derivatives import curvatures, hillshade

    dem = _synthetic_dem(size)
    cellsize = 1.7

    if func == "curvatures":
        if backend == "numpy":
            call = lambda: _numpy_curvatures(dem, cellsize)  # noqa: E731
        elif backend == "rust":
            import terra_texture_rs
            call = lambda: terra_texture_rs.curvatures(dem, cellsize)  # noqa: E731
        else:  # "dispatch" -- the public function, whichever path it picks
            call = lambda: curvatures(dem, cellsize)  # noqa: E731
    else:  # hillshade
        if backend == "numpy":
            call = lambda: _numpy_hillshade(dem, cellsize, AZIMUTH, ALTITUDE)  # noqa: E731
        elif backend == "rust":
            import terra_texture_rs
            call = lambda: terra_texture_rs.hillshade(dem, cellsize, AZIMUTH, ALTITUDE)  # noqa: E731
        else:
            call = lambda: hillshade(dem, cellsize, AZIMUTH, ALTITUDE)  # noqa: E731

    # warm-up (first call pays one-off costs: page faults, cache misses,
    # and for "rust" a one-time import of the extension module)
    call()

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        call()
        times.append(time.perf_counter() - t0)

    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps({
        "median_seconds": statistics.median(times),
        "min_seconds": min(times),
        "peak_rss_kb": peak_rss_kb,
    }))


def measure(size, backend, func, iterations):
    """Spawn a fresh subprocess for one (size, backend, func) combination
    so peak-RSS measurements don't leak across runs."""
    result = subprocess.run(
        [sys.executable, __file__, "--run-one", "--size", str(size),
         "--backend", backend, "--func", func, "--iterations", str(iterations)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return None


def rust_available():
    result = subprocess.run(
        [sys.executable, "-c", "import terra_texture_rs"],
        capture_output=True, cwd=REPO_ROOT,
    )
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--sizes", type=str, default=",".join(str(s) for s in DEFAULT_SIZES),
        help="Comma-separated DEM side lengths (DEMs are size x size)",
    )
    parser.add_argument("--iterations", type=int, default=10, help="Timed iterations per measurement (after 1 warm-up)")
    parser.add_argument("--run-one", action="store_true", help=argparse.SUPPRESS)  # internal, for subprocess mode
    parser.add_argument("--size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--backend", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--func", type=str, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.run_one:
        _run_one(args.size, args.backend, args.func, args.iterations)
        return

    sizes = [int(s) for s in args.sizes.split(",")]
    have_rust = rust_available()
    if not have_rust:
        print("terra_texture_rs is not built -- showing numpy-only numbers.")
        print("Run `cd rust && maturin develop --release` first for a real comparison.\n")

    for func in ["curvatures", "hillshade"]:
        print(f"=== {func} ===")
        header = (
            f"{'size':>6} | {'elements':>10} | {'numpy (ms)':>12} | {'rust (ms)':>12} | "
            f"{'speedup':>8} | {'numpy RSS (MB)':>15} | {'rust RSS (MB)':>14}"
        )
        print(header)
        print("-" * len(header))

        prev_elements = 0
        for size in sizes:
            elements = size * size
            np_result = measure(size, "numpy", func, args.iterations)
            rust_result = measure(size, "rust", func, args.iterations) if have_rust else None

            np_ms = np_result["median_seconds"] * 1000 if np_result else float("nan")
            np_mb = np_result["peak_rss_kb"] / 1024 if np_result else float("nan")

            if rust_result:
                rust_ms = rust_result["median_seconds"] * 1000
                rust_mb = rust_result["peak_rss_kb"] / 1024
                speedup = f"{np_ms / rust_ms:.1f}x" if rust_ms > 0 else "n/a"
                rust_ms_s, rust_mb_s = f"{rust_ms:.3f}", f"{rust_mb:.1f}"
            else:
                rust_ms_s, rust_mb_s, speedup = "--", "--", "--"

            # mark the first size at/above PARALLEL_THRESHOLD -- this is
            # the row where the rust kernel switches from serial to
            # rayon-parallel
            marker = " <- rust switches to parallel here" if prev_elements < 65_536 <= elements else ""
            prev_elements = elements
            row = (
                f"{size:>6} | {elements:>10} | {np_ms:>12.3f} | {rust_ms_s:>12} | "
                f"{speedup:>8} | {np_mb:>15.1f} | {rust_mb_s:>14}{marker}"
            )
            print(row)
        print()

    print("PARALLEL_THRESHOLD in rust/src/lib.rs is 65,536 elements -- sizes")
    print("that straddle that (e.g. 256x256=65,536) are where the rust kernel")
    print("switches from serial to rayon-parallel; useful to see if the")
    print("crossover is actually where the guessed threshold assumes it is.")
    print()
    print("Note: curvatures() computes 5 intermediate arrays internally")
    print("(zx, zy, zxx, zyy, zxy) via three chained gradient passes on")
    print("both the numpy AND rust side (same 'gradient of gradient'")
    print("approximation, see derivatives.py's module docstring) -- expect")
    print("its memory/time numbers to run noticeably higher than")
    print("hillshade()'s single gradient pass at the same size.")


if __name__ == "__main__":
    main()
