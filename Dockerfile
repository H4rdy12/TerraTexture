# Dockerfile for H4rdy12/TerraTexture
#
# Two things you can build out of this file:
#
#   dev      (default target — what you get from a plain `docker build`)
#            Full environment: gcc, Rust toolchain, uv, dev/rust dep groups,
#            the compiled Rust extension. For local development and CI.
#
#   release  Lean runtime image: just the venv + source needed to run
#            `TerraTexture`, no compilers, no Rust toolchain, no dev tools.
#
# Build (dev):      DOCKER_BUILDKIT=1 docker build -t terratexture-dev .
# Run (dev):        docker run --rm -it -v $(pwd):/app terratexture-dev bash
# Test (dev):       docker run --rm -v $(pwd):/app terratexture-dev uv run pytest
#
# Build (release):  DOCKER_BUILDKIT=1 docker build --target release -t terratexture .
# Run (release):    docker run --rm terratexture
#
# Notes:
# - Requires BuildKit (the RUN --mount=type=cache lines below need it).
#   DOCKER_BUILDKIT=1 is the default on recent Docker/Docker Desktop, but
#   set it explicitly if your `docker build` errors about --mount.
# - It is NOT a manylinux wheel-builder — see the bottom of this file for that.
#
# Size fix: uv's download cache used to get baked into the image on top of
# the already-installed .venv, roughly doubling the footprint of the
# geospatial extras. `RUN --mount=type=cache` keeps that cache OUTSIDE the
# image entirely, and persists it across builds.

FROM python:3.11-slim-bookworm AS build

# System deps:
# - gcc + curl: needed to install uv, the Rust toolchain, and to link the
#   Rust extension (`cc` is rustc's default linker on Linux)
# - libc6-dev: the C runtime objects (Scrt1.o, crti.o) and libc/libm/etc.
#   link libraries. gcc only *recommends* it, so --no-install-recommends
#   leaves it out, and every Rust build script then fails to link.
# - NOTE: libgdal-dev / libproj-dev / libgeos-dev / pkg-config removed —
#   rasterio ships manylinux wheels with GDAL bundled statically, so the
#   Python side doesn't need system GDAL headers, and the rust/ crate
#   (pyo3 + numpy + ndarray + rayon) has no C dependency on GDAL/PROJ/GEOS
#   either — confirmed against rust/Cargo.toml.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc6-dev \
    curl \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/* /usr/share/doc/* /usr/share/man/* /usr/share/locale/*

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
#
# The rust group includes the local editable path dependency
# `terra-texture-rs` (./rust), but only pyproject.toml + uv.lock have been
# copied at this point, so ./rust doesn't exist yet:
# - --frozen installs straight from uv.lock without re-resolving against
#   local sources (which would need ./rust to be present)
# - --no-install-package terra-texture-rs skips the Rust extension here;
#   the second `uv sync` below installs it once the source is copied in.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --frozen --extra raster --extra basemap --group dev --group rust \
        --no-install-project --no-install-package terra-texture-rs

# Now copy the actual source and install the project itself
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --extra raster --extra basemap --group dev --group rust

# Run from /app (not `cd rust`): rust/ has its own pyproject.toml, so
# `uv run` inside it would create a separate rust/.venv and install the
# extension there. From /app, uv run uses /app/.venv, which is where the
# runtime-venv stage copies terra_texture_rs from.
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/app/rust/target,sharing=locked \
    uv run maturin develop --release --manifest-path rust/Cargo.toml

ENV PATH="/app/.venv/bin:$PATH"


# ---------------------------------------------------------------------------
# runtime-venv: builds the LEAN runtime venv (core + raster/basemap extras
# only — no dev/rust groups) directly at /app/.venv, the same absolute path
# it will live at in the final `release` image.
#
# This has to be its own fresh stage rather than a second venv created
# inside `build` at a different path (e.g. /opt/runtime-venv) and then
# COPYed over: venv console-script shebangs embed an absolute interpreter
# path at creation time. A venv built at one path and then relocated to
# another breaks every entry-point script (`exec .../TerraTexture: no such
# file or directory` — the console-script's #! line still points at the
# old, now-nonexistent path). Building it here, already at /app/.venv,
# means the shebang is correct from the moment it's created, and copying
# the finished venv to the *same* absolute path in `release` doesn't
# invalidate it.
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime-venv

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
WORKDIR /app
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv venv /app/.venv \
 && uv pip install --python /app/.venv/bin/python ".[raster,basemap]"

# Compiled Rust extension comes from `build`, which has the toolchain to
# produce it — copied in as a plain .so-bearing package dir, no recompiling
# needed against this venv.
COPY --from=build /app/.venv/lib/python3.11/site-packages/terra_texture_rs \
     /app/.venv/lib/python3.11/site-packages/terra_texture_rs


# ---------------------------------------------------------------------------
# release: lean runtime image. Carries over the venv built above (already
# at the right path, so entry-point scripts work) and the source it points
# to, plus pyproject metadata for entry-point resolution. No gcc, no Rust
# toolchain, no dev/docs/rust dependency groups.
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS release

# ca-certificates: contextily fetches basemap tiles over HTTPS at runtime.
# libexpat1: rasterio's wheel bundles GDAL statically, but GDAL itself
# dynamically links against libexpat (XML parsing for some format drivers)
# at runtime — this was missing and caused an ImportError.
# gdal-bin is NOT needed here — rasterio's wheel bundles GDAL statically.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libexpat1 \
 && rm -rf /var/lib/apt/lists/* /usr/share/doc/* /usr/share/man/* /usr/share/locale/*

WORKDIR /app
COPY --from=runtime-venv /app/.venv /app/.venv
COPY --from=runtime-venv /app/src /app/src
COPY --from=runtime-venv /app/pyproject.toml /app/pyproject.toml

ENV PATH="/app/.venv/bin:$PATH"

# `TerraTexture` is the console-script entry point from [project.scripts].
# ENTRYPOINT (not CMD) so `docker run terratexture curvature --help` appends
# args to TerraTexture instead of replacing the whole command.
ENTRYPOINT ["TerraTexture"]


# ---------------------------------------------------------------------------
# dev: default target (last stage = what a plain `docker build` produces).
# Identical to `build` above — the full toolchain stays available so you
# can `docker exec` in and run tests, re-run maturin develop, use cargo
# bench, etc. Split out as its own stage only so `--target release` can
# stop at `build` without also being told "this is the thing you run".
# ---------------------------------------------------------------------------
FROM build AS dev

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
