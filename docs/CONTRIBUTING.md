### Contributing to HRM-CodeGen

Thank you for your interest in contributing! This document outlines how to propose changes and keep quality high.

- Branching: feature branches off `main`; use concise names like `feat/cli`, `fix/eval`.
- Tests: add/maintain unit tests in `tests/`; disabled or exploratory tests go in `tests/disabled/`.
- Lint/format: run `black`, `flake8`, and `pytest -q` locally before PRs.
- Docs: update relevant files in `docs/` and the index `docs/README.md` when you add features.
- Commit style: imperative tense, short subject, optional body with context.
- PR checklist:
  - Tests added/updated and passing
  - Docs updated
  - No linter warnings
  - Links to issues and design docs

Local quickstart:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

