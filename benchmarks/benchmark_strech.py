#!/usr/bin/env python3
"""
Benchmark numpy vs Rust (terra_texture_rs) for TerraTexture.stretch's
stretch_std() -- both wall-clock time and peak memory, across a range
of array sizes.

Companion to scripts/benchmark_blend.py and scripts/benchmark_derivatives.py
-- same structure, same subprocess-isolation approach, adapted for
stretch.py's one accelerated function. See benchmark_blend.py's
docstring for the full rationale (repeated only briefly below).

Requires the Rust extension to actually be built to get Rust numbers at
all (see rust/README / Cargo.toml comments for `maturin develop
--release`) -- without it, this still runs and reports numpy-only
numbers, it just can't show a comparison.

Usage:
    uv run python benchmark_stretch.py
    uv run python benchmark_stretch.py --sizes 64,256,1024,4096
    uv run python benchmark_stretch.py --iterations 20

Memory is measured via peak RSS (resource.getrusage().ru_maxrss), a
monotonic high-water-mark for the whole process -- to get an accurate
PER-SIZE reading rather than a running maximum contaminated by earlier,
larger runs in the same process, each (size, backend) combination is
measured in its own fresh subprocess.

Note on what each backend actually calls -- stretch.py's numpy fallback
already includes one optimization (variance computed from an
already-known mean, instead of calling np.nanstd() separately, which
recomputes its own mean internally -- see stretch.py's module
docstring). This script benchmarks that ALREADY-OPTIMIZED numpy path as
"numpy", not the older, more naive nanmean+nanstd formula, so the
reported speedup reflects what a real caller actually gains by having
the Rust extension built, not an inflated comparison against a
straw-man baseline.
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

N_STD = 4.0


def _numpy_stretch_std(arr, n_std=N_STD):
    """stretch.py's own (already-optimized) numpy fallback formula,
    reproduced here so this script can call it directly without going
    through the public dispatch (matching benchmark_blend.py's pattern
    of calling each backend's implementation directly, bypassing
    dispatch-guard overhead on both sides for a fair comparison)."""
    import numpy as np

    mean = np.nanmean(arr)
    variance = np.nanmean((arr - mean) ** 2)
    std = np.sqrt(variance)
    lo, hi = mean - n_std * std, mean + n_std * std
    return np.clip((arr - lo) / (hi - lo + 1e-12), 0, 1)


def _run_one(size, backend, iterations):
    """Run inside an isolated subprocess: time+memory for one
    (size, backend) combination. Prints a JSON result to stdout."""
    import numpy as np
    import resource

    from TerraTexture.stretch import stretch_std

    rng = np.random.default_rng(0)
    arr = (rng.random((size, size)).astype(np.float32) - 0.5) * 200
    # a modest NaN patch, matching real curvature/hillshade data with
    # nodata/voids -- large enough to exercise NaN-handling in both
    # backends, small enough not to dominate the array
    if size >= 4:
        n = max(1, size // 10)
        arr[:n, :n] = np.nan

    if backend == "numpy":
        call = lambda: _numpy_stretch_std(arr, N_STD)  # noqa: E731
    elif backend == "rust":
        import terra_texture_rs
        arr_c = np.ascontiguousarray(arr)
        call = lambda: terra_texture_rs.stretch_std(arr_c, N_STD)  # noqa: E731
    else:  # "dispatch" -- the public function, whichever path it picks
        call = lambda: stretch_std(arr, N_STD)  # noqa: E731

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


def measure(size, backend, iterations):
    """Spawn a fresh subprocess for one (size, backend) combination so
    peak-RSS measurements don't leak across runs."""
    result = subprocess.run(
        [sys.executable, __file__, "--run-one", "--size", str(size),
         "--backend", backend, "--iterations", str(iterations)],
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
        help="Comma-separated array side lengths (arrays are size x size)",
    )
    parser.add_argument("--iterations", type=int, default=10, help="Timed iterations per measurement (after 1 warm-up)")
    parser.add_argument("--run-one", action="store_true", help=argparse.SUPPRESS)  # internal, for subprocess mode
    parser.add_argument("--size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--backend", type=str, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.run_one:
        _run_one(args.size, args.backend, args.iterations)
        return

    sizes = [int(s) for s in args.sizes.split(",")]
    have_rust = rust_available()
    if not have_rust:
        print("terra_texture_rs is not built -- showing numpy-only numbers.")
        print("Run `cd rust && maturin develop --release` first for a real comparison.\n")

    print("=== stretch_std ===")
    header = (
        f"{'size':>6} | {'elements':>10} | {'numpy (ms)':>12} | {'rust (ms)':>12} | "
        f"{'speedup':>8} | {'numpy RSS (MB)':>15} | {'rust RSS (MB)':>14}"
    )
    print(header)
    print("-" * len(header))

    prev_elements = 0
    for size in sizes:
        elements = size * size
        np_result = measure(size, "numpy", args.iterations)
        rust_result = measure(size, "rust", args.iterations) if have_rust else None

        np_ms = np_result["median_seconds"] * 1000 if np_result else float("nan")
        np_mb = np_result["peak_rss_kb"] / 1024 if np_result else float("nan")

        if rust_result:
            rust_ms = rust_result["median_seconds"] * 1000
            rust_mb = rust_result["peak_rss_kb"] / 1024
            speedup = f"{np_ms / rust_ms:.1f}x" if rust_ms > 0 else "n/a"
            rust_ms_s, rust_mb_s = f"{rust_ms:.3f}", f"{rust_mb:.1f}"
        else:
            rust_ms_s, rust_mb_s, speedup = "--", "--", "--"

        # mark the first size at/above PARALLEL_THRESHOLD -- this is the
        # row where the rust kernel switches from a serial reduction to
        # a rayon fold+reduce (and the elementwise pass to par_for_each)
        marker = " <- rust switches to parallel here" if prev_elements < 65_536 <= elements else ""
        prev_elements = elements
        row = (
            f"{size:>6} | {elements:>10} | {np_ms:>12.3f} | {rust_ms_s:>12} | "
            f"{speedup:>8} | {np_mb:>15.1f} | {rust_mb_s:>14}{marker}"
        )
        print(row)
    print()

    print("PARALLEL_THRESHOLD in rust/src/lib.rs is 65,536 elements -- sizes")
    print("that straddle that are where the rust kernel switches from a")
    print("serial fold to a rayon fold+reduce for the mean/std reduction,")
    print("and from a serial to a par_for_each for the elementwise pass.")
    print()
    print("stretch_std's own win comes from doing ONE reduction pass (sum,")
    print("sum-of-squares, and NaN-count together) instead of numpy's")
    print("separate nanmean + nanstd calls (nanstd recomputes its own mean")
    print("internally) -- expect a more modest speedup here than")
    print("soft_light/luminosity_blend/curvatures/hillshade, since a")
    print("reduction is inherently less parallelism-friendly than a pure")
    print("elementwise map (every thread's partial result has to be")
    print("combined before the second, elementwise pass can even start).")


if __name__ == "__main__":
    main()
