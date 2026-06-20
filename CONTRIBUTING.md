# Contributing to lfpack

## Development setup

```bash
git clone https://github.com/int-brain-lab/lfpack
cd lfpack
uv sync --group dev
git config core.hooksPath .githooks
```

The last command installs the pre-commit hook from the tracked `.githooks/` directory.

## Code style

All code is formatted and linted with [ruff](https://docs.astral.sh/ruff/).
Configuration lives in `pyproject.toml` (`line-length = 120`, rules `E F W I`).

Before committing, run:

```bash
uv run ruff format src/ tests/   # format
uv run ruff check  src/ tests/   # lint
```

The pre-commit hook installed above runs both checks automatically and blocks commits that fail.

## Docstrings

All public functions, classes, and methods use [NumPy/SciPy docstring format](https://numpydoc.readthedocs.io/en/latest/format.html).
These docstrings are rendered into the API reference by [quartodoc](https://machow.github.io/quartodoc/).

## Documentation

The docs site lives in `docs/` and follows the [Diátaxis](https://diataxis.fr) framework:

| Directory | Type | Purpose |
|---|---|---|
| `docs/tutorials/` | Tutorial | Guided learning paths |
| `docs/how-to/` | How-To | Step-by-step task guides |
| `docs/reference/` | Reference | API docs (quartodoc) + HDF5 spec |
| `docs/explanation/` | Explanation | Concepts, design choices, benchmarks |

Build the docs:

```bash
# Generate API reference pages from docstrings
uv run quartodoc build --config docs/_quarto.yml

# Render HTML (output → docs/_site/)
quarto render docs/

# Live preview with hot-reload
quarto preview docs/
```

## Tests

```bash
uv run pytest                     # run all tests
uv run pytest --cov --cov-report=term-missing   # with coverage report
```

All existing tests must pass and coverage must not regress before opening a pull request.

## Versioning

This project uses [Semantic Versioning](https://semver.org):

- **PATCH** (`0.1.x`) — bug fixes, no API changes.
- **MINOR** (`0.x.0`) — new backwards-compatible features or API additions.
- **MAJOR** (`x.0.0`) — breaking changes (e.g. new HDF5 layout, removed parameters).

Changes are recorded in `CHANGELOG.md` following the
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) convention.
The `[Unreleased]` section accumulates work until a release, at which point it is
renamed to `[x.y.z] - YYYY-MM-DD` and a matching git tag `vx.y.z` is pushed.