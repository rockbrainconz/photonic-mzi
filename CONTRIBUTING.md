# Contributing

[English](CONTRIBUTING.md) | [简体中文](CONTRIBUTING.zh-CN.md)

## Environment

```bash
python -m pip install --upgrade pip && pip install -e ".[dev]"
```

Editable installs require **pip 21.3 or later**. The project uses `pyproject.toml`
without `setup.py`, so older pip versions cannot install it in editable mode.

Development also works without installation: `tests/conftest.py` adds `src/` to
`sys.path`, and examples prefer the checkout's `src/` tree.

## Development loop

```bash
pytest -m "not slow"  # 128 fast tests
pytest                 # 133 tests, including rendering
ruff check .
```

Run the complete suite before opening a pull request. It includes all-frame glyph
scanning, GIF export, and construction at multiple matrix sizes. CI runs `ruff check`
but does not apply automatic formatting because the aligned matrix literals and
teaching comments are intentional.

Regenerate the English-first bilingual animation and README screenshots after changing
visual text or layout:

```bash
python tools/generate_docs_assets.py
```

## Release process

1. Move completed changelog entries from `Unreleased` into the new version section.
2. Update the version in `pyproject.toml` and `photonic_mzi.__version__`, then run the
   full test, lint, build, and `twine check --strict` suite.
3. Merge the release commit into `main` and wait for CI to pass.
4. Publish a GitHub Release tagged `v<version>`. The `Publish to PyPI` workflow verifies
   that the tag matches the package metadata, builds fresh distributions, and uploads
   them through PyPI Trusted Publishing.

The `pypi` GitHub environment and the matching PyPI Trusted Publisher must stay scoped
to `.github/workflows/release.yml`. PyPI versions are immutable; never reuse a released
version number.

## Project-specific rules

**The animation must use real computed values.** Optical fields shown on screen must
come from `photonic_mzi.processor`; do not invent a second set of values for appearance.

**Scan every animation frame for missing glyphs.** Matplotlib warns instead of failing
when a font lacks a character. `test_every_frame_renders_without_missing_glyphs` renders
all 494 frames. Prefer mathtext such as `$V^T$` over uncommon Unicode symbols.

**Highlight MZIs by `MZI.index`, not `mode`.** Multiple devices on different layers can
share a waveguide. Matching by mode would incorrectly highlight a whole row.

**Document physical approximations in docstrings.** Every new approximation must say
where it diverges from a physical device and which conclusions it cannot support.

## Adding a noise source

Define its statistics and physical meaning before adding a `NoiseModel` field:

- **Fixed device offset:** sample once in `PhotonicMatrixProcessor.__init__`; allow
  `calibrate()` to cancel it only if it is an additive controllable phase offset.
- **Per-sample i.i.d. jitter:** sample independently for each batch column in `_run_mesh`.
- **Time- or space-correlated drift:** store explicit state or a covariance model; do not
  represent it as an i.i.d. scalar.
- **Detection noise:** add it after coherent or square-law detection, never to the
  pre-detection complex field.

Add a test that locks down the new source's repeatability semantics.

## Implementing a Clements mesh

See [Model and validation notes](docs/validation.md). If you implement it:

1. retain Reck as a selectable topology;
2. test that Clements depth is `N`, not `2N-3`;
3. validate loss balance with topology depth or transfer matrices rather than treating
   `mode_mzi_count()` as an end-to-end path count.
