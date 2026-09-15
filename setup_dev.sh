#!/usr/bin/env bash
#
# One-command dev setup: syncs the Python environment, then attempts to
# build the optional Rust extension (terra_texture_rs) on top of it.
#
# Deliberately does NOT make the Rust build part of `uv sync` itself --
# that would mean everyone needs a Rust toolchain just to install this
# package, breaking the core design goal that TerraTexture.derivatives/
# TerraTexture.blend work with zero Rust toolchain required (see
# blend.py's module docstring). Instead: `uv sync` is the hard
# requirement, the Rust build is best-effort on top of it -- if it
# fails (no cargo installed, wrong toolchain version, etc.), this script
# warns and continues rather than aborting, so you still end up with a
# fully working pure-Python environment either way.
#
# Usage:
#   ./scripts/setup_dev.sh
#   ./scripts/setup_dev.sh --no-rust     # skip the Rust build entirely

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SKIP_RUST=false
for arg in "$@"; do
    if [[ "$arg" == "--no-rust" ]]; then
        SKIP_RUST=true
    fi
done

echo "==> Syncing Python environment (uv sync --group dev --group docs --extra raster --extra basemap)"
uv sync --group dev --group docs --extra raster --extra basemap

echo
uv run pytest -q
echo "Python environment ready."

if [[ "$SKIP_RUST" == "true" ]]; then
    echo
    echo "==> Skipping Rust build (--no-rust passed)."
    exit 0
fi

echo
echo "==> Attempting optional Rust extension build (terra_texture_rs)"
echo "    This accelerates TerraTexture.blend's soft_light()/luminosity_blend()"
echo "    -- entirely optional, blend.py falls back to pure numpy if this"
echo "    isn't built or fails."

if ! uv sync --group rust -q; then
    echo "!! Could not install maturin (uv sync --group rust failed) -- skipping Rust build."
    echo "   Python setup above is complete and fully usable without it."
    exit 0
fi

if ! command -v cargo &> /dev/null; then
    echo "!! No 'cargo' found on PATH -- Rust toolchain isn't installed."
    echo "   Install it from https://rustup.rs, then re-run this script,"
    echo "   or just: cd rust && uv run maturin develop --release"
    echo "   Python setup above is complete and fully usable without it."
    exit 0
fi

if (cd rust && uv run maturin develop --release); then
    echo
    echo "==> Rust extension built successfully."
    uv run python3 -c "import terra_texture_rs; print('terra_texture_rs is now active:', terra_texture_rs.__file__)"
else
    echo
    echo "!! Rust build failed (see error above -- common cause: an outdated"
    echo "   Rust toolchain; try 'rustup update stable' and re-run this script)."
    echo "   Python setup above is complete and fully usable without it --"
    echo "   TerraTexture.blend falls back to pure numpy automatically."
fi
