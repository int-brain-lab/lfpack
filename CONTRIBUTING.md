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

## Tests

```bash
uv run pytest
```

All existing tests must pass before opening a pull request.

## Versioning

This project uses [Semantic Versioning](https://semver.org):

- **PATCH** (`0.1.x`) — bug fixes, no API changes.
- **MINOR** (`0.x.0`) — new backwards-compatible features or API additions.
- **MAJOR** (`x.0.0`) — breaking changes (e.g. new HDF5 layout, removed parameters).

Changes are recorded in `CHANGELOG.md` following the
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) convention.
The `[Unreleased]` section accumulates work until a release, at which point it is
renamed to `[x.y.z] - YYYY-MM-DD` and a matching git tag `vx.y.z` is pushed.