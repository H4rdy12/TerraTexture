#!/usr/bin/env python3
"""
Benchmark numpy vs Rust (terra_texture_rs) for TerraTexture.blend's
soft_light() and luminosity_blend() -- both wall-clock time and peak
memory, across a range of array sizes.

Requires the Rust extension to actually be built to get Rust numbers at
all (see rust/README / Cargo.toml comments for `maturin develop
--release`) -- without it, this still runs and reports numpy-only
numbers, it just can't show a comparison.

Usage:
    uv run python scripts/benchmark_blend.py
    uv run python scripts/benchmark_blend.py --sizes 64,256,1024,4096
    uv run python scripts/benchmark_blend.py --iterations 20

Memory is measured via peak RSS (resource.getrusage().ru_maxrss), which
is a monotonic high-water-mark for the whole process -- to get an
accurate PER-SIZE reading rather than a running maximum contaminated by
earlier, larger runs in the same process, each (size, backend)
combination is measured in its own fresh subprocess.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SIZES = [16, 64, 128, 256, 512, 1024, 2048, 4096]


def _run_one(size, backend, func, iterations):
    """Run inside an isolated subprocess: time+memory for one
    (size, backend, func) combination. Prints a JSON result to stdout."""
    import numpy as np
    import resource

    from TerraTexture.blend import (
        _soft_light_numpy, _luminosity_blend_numpy, soft_light, luminosity_blend,
    )

    rng = np.random.default_rng(0)

    if func == "soft_light":
        a = rng.random((size, size)).astype(np.float32)
        b = rng.random((size, size)).astype(np.float32)
        if backend == "numpy":
            call = lambda: _soft_light_numpy(a, b)  # noqa: E731
        elif backend == "rust":
            import terra_texture_rs
            call = lambda: terra_texture_rs.soft_light(a, b)  # noqa: E731
        else:  # "dispatch" -- the public function, whichever path it picks
            call = lambda: soft_light(a, b)  # noqa: E731
    else:  # luminosity_blend
        backdrop = rng.random((size, size, 3)).astype(np.float32)
        lum = rng.random((size, size)).astype(np.float32)
        if backend == "numpy":
            call = lambda: _luminosity_blend_numpy(backdrop, lum)  # noqa: E731
        elif backend == "rust":
            import terra_texture_rs
            call = lambda: terra_texture_rs.luminosity_blend(backdrop, lum)  # noqa: E731
        else:
            call = lambda: luminosity_blend(backdrop, lum)  # noqa: E731

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
        help="Comma-separated array side lengths (arrays are size x size)",
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

    for func in ["soft_light", "luminosity_blend"]:
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


if __name__ == "__main__":
    main()
