# Contributing

## Setup
```bash
uv sync --extra dev
uv run pre-commit install
```

## Checks (CI enforces these — incl. the eval ship-gate)
```bash
uv run ruff check .
uv run ruff format .
uv run mypy .
uv run pytest
uv run flagship eval     # the end-to-end gate also runs in CI
```

## Conventions
- Type hints everywhere; mypy clean. Tests fully offline (scripted policy + hashing embedder).
- Each layer is behind a small interface (Embedder, Policy, Tool); extend by implementing one and
  wiring it in `factory.py`.
- Secrets via `.env` (never committed). Conventional-commit messages.
