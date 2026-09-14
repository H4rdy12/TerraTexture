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

import datetime
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = REPO_ROOT / "docs" / "docs_template"
OUTPUT_DIR = REPO_ROOT / "docs" / "site"
API_OUTPUT_DIR = OUTPUT_DIR / "api"


def get_version():
    try:
        import importlib.metadata

        return importlib.metadata.version("terratexture")
    except Exception:
        return "0.0.0"


def main():
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
            "--logo", "../logo.jpg",  # relative to docs/site/api/*.html -> resolves to docs/site/logo.jpg
            "--logo-link", "../index.html",  # click the sidebar logo -> back to the landing page
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

    print(f"Docs built at {OUTPUT_DIR}")
    print(f"Open: file://{(OUTPUT_DIR / 'index.html').resolve()}")


if __name__ == "__main__":
    main()