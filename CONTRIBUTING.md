# Contributing to TerraTexture

Thanks for considering a contribution. This project is early-stage, so
expect some rough edges — issues and PRs that improve on that are welcome.

## Getting set up

This project uses [uv](https://docs.astral.sh/uv/) for environment
management.

```bash
git clone https://github.com/H4rdy12/TerraTexture.git
cd TerraTexture
uv sync --group dev --extra raster   # core dev setup; add --extra basemap if touching that code
```

Verify your setup:

```bash
uv run python -c "import TerraTexture; print(TerraTexture.__file__)"
uv run pytest
uv run flake8 src tests examples
```

## Making a change

1. **Open an issue first** for anything non-trivial (new features, API
   changes, dependency additions) so we can align on approach before
   you put in the work.
2. **Branch off `main`**, one logical change per branch/PR.
3. **Add tests** for new behaviour. Look at the existing test files for
   the pattern to follow:
   - Pure numpy/scipy logic (`derivatives.py`, `blend.py`, `stretch.py`)
     → tests run unconditionally, no optional dependency needed.
   - Code needing `rasterio` → put tests in their own file using
     `pytest.importorskip("rasterio")` at module level (see
     `tests/test_io.py`). **Don't mix these into a file with
     unconditional tests** — a module-level `importorskip` skips the
     *entire file's* collection when the import fails, not just the
     tests after it.
   - Rust kernel changes → add parity tests in `tests/test_blend_rust.py`
     comparing Rust output against the numpy reference implementation.
4. **Run the full check locally** before opening a PR:
   ```bash
   uv run pytest
   uv run flake8 src tests examples
   ```
   If you touched `rust/`, also run (requires a Rust toolchain):
   ```bash
   cd rust
   cargo fmt --check
   cargo clippy --all-targets -- -D warnings
   cargo build --release --all-targets
   ```

## Code style

- Python: `flake8`, config in `pyproject.toml`'s `[tool.flake8]`
  (`max-line-length = 120`). No separate formatter is enforced — match
  the surrounding code's style.
- Rust: `rustfmt` (config in `rust/rustfmt.toml`, also `max_width = 120`)
  and `clippy` with warnings denied. Run `cargo fmt` before committing
  rather than hand-formatting — chain/signature wrapping rules are
  stricter than they look and easy to get wrong by eye.
- Keep new geospatial-heavy modules (`io.py`, `basemap.py`, `sources.py`)
  separate from dependency-free ones (`derivatives.py`, `blend.py`,
  `stretch.py`) — the zero-dependency core is a deliberate design
  constraint, not an accident. If your change would add a new hard
  dependency to one of the dependency-free modules, flag it in your PR
  description so we can discuss.

## The Rust extension is currently optional [Under dev]

`rust/terra_texture_rs` accelerates `blend.py`'s hot paths but isn't
required to use or contribute to this project — `blend.py` falls back to
pure numpy automatically if the extension isn't built. You don't need a
Rust toolchain to work on anything outside `rust/`.

## Pull requests

- CI (`.github/workflows/ci.yml`) runs build, lint, and test jobs across
  Python 3.10–3.12 plus the Rust checks. All must pass (`all-checks`) —
  see the job list in the README's CI/CD section if you want the
  breakdown.
- At least one approving review is required before merging, and
  `main` is protected against direct pushes — see the README's "Branch
  protection" section for details.
- Keep PRs focused. If a review surfaces a good idea that's out of
  scope for the current change, open a follow-up issue rather than
  expanding the PR.

## Data attribution

If your contribution touches the ArcticDEM/REMA/OpenTopography sources
in `sources.py`, keep in mind these are third-party datasets with their
own licensing and citation requirements (see the README's "Data
attribution" note) — this project's own MIT license doesn't extend to
the data it fetches.

## Questions

Open an issue — no dedicated chat/forum for this project yet.