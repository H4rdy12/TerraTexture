#!/usr/bin/env python3
"""
Build TerraTexture's API docs with pdoc, using the branded template in
docs_template/ (light theme, brand blue/green sampled from the logo,
dimgrey topographic-contour background generated from the package's own
synthetic demo DEM -- see generate_contour_bg.py).

Usage:
    uv run python docs/build_docs.py
    # or, from the repo root with the venv active:
    python docs/build_docs.py

Output goes to docs/site/ :
    docs/site/index.html        -- branded landing page (the actual entry point)
    docs/site/logo.jpg          -- project logo
    docs/site/api/*.html        -- pdoc-generated module docs

Open docs/site/index.html directly in a browser, or serve it:
    python -m http.server --directory docs/site 8000
"""

import argparse
import datetime
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = REPO_ROOT / "docs" / "docs_template"
OUTPUT_DIR = REPO_ROOT / "docs" / "site"
API_OUTPUT_DIR = OUTPUT_DIR / "api"

# Rust crate: rustdoc names its output folder after the crate, so the
# landing page's Rust card links to rust/terra_texture_rs/index.html.
RUST_DIR = REPO_ROOT / "rust"
RUST_CRATE = "terra_texture_rs"
RUST_OUTPUT_DIR = OUTPUT_DIR / "rust"
 
# `version = "x.y.z"` inside Cargo.toml's [package] table. Regex rather
# than tomllib because tomllib only exists from Python 3.11 and this
# package supports 3.10.
_CARGO_PACKAGE_RE = re.compile(r"^\[package\]\s*$(.*?)(?=^\[|\Z)", re.M | re.S)
_CARGO_VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"', re.M)


def get_version():
    try:
        import importlib.metadata

        return importlib.metadata.version("terratexture")
    except Exception:
        return "0.0.0"


def get_rust_version(cargo_toml: Path = RUST_DIR / "Cargo.toml") -> str:
    """
    Read the Rust crate's version from its Cargo.toml.
 
    Only the ``[package]`` table is searched, so dependency versions like
    ``pyo3 = { version = "0.29" }`` can't be picked up by mistake.
 
    Args:
        cargo_toml (Path): Path to the crate's ``Cargo.toml``.
 
    Returns:
        str: The package version, e.g. ``"0.1.0"``, or ``"0.0.0"`` if the
            file is missing or has no ``[package]`` version.
    """
    try:
        text = cargo_toml.read_text(encoding="utf-8")
    except OSError:
        return "0.0.0"
 
    package = _CARGO_PACKAGE_RE.search(text)
    if package is None:
        return "0.0.0"
 
    version = _CARGO_VERSION_RE.search(package.group(1))
    return version.group(1) if version else "0.0.0"
 
 
def build_rust_docs() -> None:
    """
    Build the Rust API docs with rustdoc and copy them into the site.
 
    Runs ``cargo doc --no-deps`` in ``rust/``, which applies the branded
    theme via ``rust/.cargo/config.toml``, then copies ``target/doc/``
    (the crate folder plus rustdoc's shared static files and search
    index, which the pages need) to ``docs/site/rust/``.
 
    Raises:
        subprocess.CalledProcessError: If ``cargo doc`` fails.
        FileNotFoundError: If ``cargo`` isn't installed.
    """
    subprocess.run(["cargo", "doc", "--no-deps"], check=True, cwd=RUST_DIR)
    shutil.copytree(RUST_DIR / "target" / "doc", RUST_OUTPUT_DIR)


def main():
    parser = argparse.ArgumentParser(description="Build TerraTexture's docs site.")
    parser.add_argument(
        "--rust",
        action="store_true",
        help="also build the Rust API docs into docs/site/rust/ (needs cargo)",
    )
    args = parser.parse_args()

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    # -- pdoc-generated module docs, into docs/site/api/ --
    subprocess.run(
        [
            sys.executable, "-m", "pdoc",
            "TerraTexture",
            "--template-directory", str(TEMPLATE_DIR),
            "--docformat", "numpy",  # this codebase uses NumPy-style "Parameters\n----------" docstrings, not Google-style
            "--no-include-undocumented",
            "--math",
            "--mermaid",
            # logo is NOT passed via --logo/--logo-link here -- those apply
            # one fixed path to every page regardless of nesting depth,
            # which breaks for submodule pages. See module.html.jinja2's
            # nav_title block override instead, which computes the
            # correct relative path per page.
            "-o", str(API_OUTPUT_DIR),
        ],
        check=True,
        cwd=REPO_ROOT,
    )

    # -- logo, alongside the landing page --
    shutil.copy(TEMPLATE_DIR / "logo.jpg", OUTPUT_DIR / "logo.jpg")

    # -- landing page, with {{VERSION}}/{{BUILD_DATE}} filled in --
    landing_html = (TEMPLATE_DIR / "landing.html").read_text()
    landing_html = landing_html.replace("{{VERSION}}", get_version())
    landing_html = landing_html.replace(
        "{{BUILD_DATE}}", datetime.date.today().isoformat()
    )
    (OUTPUT_DIR / "index.html").write_text(landing_html)

    # -- optional: Rust API docs, into docs/site/rust/ --
    if args.rust:
        build_rust_docs()
        print(f"Rust docs built at {RUST_OUTPUT_DIR / RUST_CRATE}")

    print(f"Docs built at {OUTPUT_DIR}")
    print(f"Open: file://{(OUTPUT_DIR / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
