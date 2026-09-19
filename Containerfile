# Containerfile for H4rdy12/TerraTexture
#
# Build:   podman build -t terratexture-dev -f Containerfile .
# Run:     podman run --rm -it -v $(pwd):/app:Z terratexture-dev bash
# Test:    podman run --rm -v $(pwd):/app:Z terratexture-dev uv run pytest
#
# Notes:
# - The `:Z` on -v is for SELinux hosts (Fedora/RHEL). Drop it if you're on
#   Debian/Ubuntu and it complains about an unknown flag.
# - This image is for local dev / CI (build + test + optional Rust extension).
#   It is NOT a manylinux wheel-builder — see the bottom of this file for that.

FROM python:3.11-slim-bookworm AS base

# System deps:
# - build-essential + curl: needed to install uv and the Rust toolchain
# - gdal/proj/geos libs: rasterio (raster/basemap extras) links against these
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    pkg-config \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
 && rm -rf /var/lib/apt/lists/*

# uv (fast Python package/dependency manager this repo is built around)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Rust toolchain (only needed for the optional rust/ extension via maturin)
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \
 && rustup --version && cargo --version

WORKDIR /app

# Copy dependency manifests first so `uv sync` layers cache independently of
# source-code changes.
COPY pyproject.toml uv.lock ./

# Pull in everything: core + raster + basemap extras, dev tooling, and the
# rust group (maturin) — mirrors `uv sync --group dev --extra raster` plus
# basemap + rust, since this image is meant to cover the whole project.
RUN uv sync --extra raster --extra basemap --group dev --group rust --no-install-project

# Now copy the actual source and install the project itself
COPY . .
RUN uv sync --extra raster --extra basemap --group dev --group rust

# If/when you want the Rust extension built in, uncomment:
# RUN cd rust && uv run maturin develop --release

ENV PATH="/app/.venv/bin:$PATH"

CMD ["bash"]

# ---------------------------------------------------------------------------
# Later, for producing PyPI-ready Linux wheels with the compiled Rust
# extension, you don't build this image — you run maturin's own manylinux
# image against your source instead, e.g.:
#
#   podman run --rm -v $(pwd):/io:Z ghcr.io/pyo3/maturin build \
#       --release --manylinux 2014 --out /io/dist -m rust/Cargo.toml
#
# or drive it via cibuildwheel with:
#   CIBW_CONTAINER_ENGINE=podman
# ---------------------------------------------------------------------------