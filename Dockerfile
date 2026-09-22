# Dockerfile for H4rdy12/TerraTexture
#
# Build:   DOCKER_BUILDKIT=1 docker build -t terratexture-dev .
# Run:     docker run --rm -it -v $(pwd):/app terratexture-dev bash
# Test:    docker run --rm -v $(pwd):/app terratexture-dev uv run pytest
#
# Notes:
# - Requires BuildKit (the RUN --mount=type=cache lines below need it).
#   DOCKER_BUILDKIT=1 is the default on recent Docker/Docker Desktop, but
#   set it explicitly if your `docker build` errors about --mount.
# - This image is for local dev / CI (build + test + optional Rust extension).
#   It is NOT a manylinux wheel-builder — see the bottom of this file for that.
#
# Size fix: uv's download cache used to get baked into the image on top of
# the already-installed .venv, roughly doubling the footprint of the
# geospatial extras. `RUN --mount=type=cache` keeps that cache OUTSIDE the
# image entirely, and persists it across builds.

FROM python:3.11-slim-bookworm AS base

# System deps:
# - build-essential + curl: needed to install uv and the Rust toolchain
# - gdal/proj/geos libs: rasterio (raster/basemap extras) links against these
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
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
#
# The uv cache mount is keyed so repeated builds (even after source changes
# below invalidate this layer) reuse already-downloaded wheels instead of
# re-fetching them, without those wheels ending up in the image.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --extra raster --extra basemap --group dev --group rust --no-install-project

# Now copy the actual source and install the project itself
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --extra raster --extra basemap --group dev --group rust

# If/when you want the Rust extension built in, uncomment:
# RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
#     cd rust && uv run maturin develop --release

ENV PATH="/app/.venv/bin:$PATH"

CMD ["bash"]

# ---------------------------------------------------------------------------
# Later, for producing PyPI-ready Linux wheels with the compiled Rust
# extension, you don't build this image — you run maturin's own manylinux
# image against your source instead, e.g.:
#
#   docker run --rm -v $(pwd):/io ghcr.io/pyo3/maturin build \
#       --release --manylinux 2014 --out /io/dist -m rust/Cargo.toml
#
# or drive it via cibuildwheel with:
#   CIBW_CONTAINER_ENGINE=docker
# ---------------------------------------------------------------------------
