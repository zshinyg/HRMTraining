---
status: active
owner: engineering
summary: Repository guidelines and best practices for HRM-CodeGen
last_reviewed: 2025-08-08
---

# Repository Guidelines and Best Practices

## Project Structure & Module Organization
- `hrm/`: Core HRM model (`model.py`, `layers.py`, `config.py`).
- `hrm_codegen/`: Generation utilities and mock model for experiments.
- `scripts/`: Entry points (`train.py`, `evaluate.py`, `convert_mbpp.py`, monitoring tools).
- `configs/`: YAML configs for HRM models and training setups (located at `configs/hrm/`).
- `tokenization/`: Tokenizer utilities and caching helpers.
- `data/`, `checkpoints/`, `outputs/`: Local artifacts (not versioned).
- `tests/` and `test/`: Pytest suites (`tests/disabled/` is skipped).
- `docs/`: Architecture, SDLC, and ops documentation.

## Build, Test, and Development

- Prefer virtualenv; avoid system-wide installs.
- Keep changes small, reversible, and covered by tests.
- Run `pytest -q` before pushing; keep tests short and hermetic.

### Commands
- Create env & install: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Quick sanity: `pytest -q`
- Prepare MBPP data: `python scripts/convert_mbpp.py --split all --output-dir data/mbpp --tokenizer gpt2`
- Train: `python scripts/train.py --config configs/hrm/mbpp_base.yaml --data-path data/mbpp/train.bin --eval-data-path data/mbpp/validation.bin --output-dir checkpoints/hrm-mbpp`
- Evaluate: `python scripts/evaluate.py --ckpt checkpoints/hrm-mbpp/latest.pt --split test --k 1 5 10`
- Format/lint: `black . && isort . && flake8`
- Optional Docker: `docker build -t hrm-codegen:latest .`

## Coding Style & Naming

- Follow PEP 8; use `black` and `isort` to format.
- Avoid deep nesting; use guard clauses; add concise docstrings for non-trivial functions.
- Use descriptive names; avoid abbreviations.

### Naming Conventions
- Python 3.10+, 4‑space indentation, line length per `black` defaults.
- Names: `snake_case` for functions/vars, `PascalCase` for classes, `SCREAMING_SNAKE_CASE` for constants.
- Configs: YAML in `configs/hrm/` with descriptive, kebab‑case filenames (e.g., `mbpp_base.yaml`).
- Keep large artifacts in `data/`, `checkpoints/`, `outputs/` (not in Git).

## Testing

- Keep unit tests fast and deterministic; avoid network and large downloads.
- Use temp dirs for data; don’t rely on global state.
- Mark slow/external tests and keep CI under time limits.

### Guidelines
- Framework: Pytest. Place tests under `tests/` as `test_*.py`; mark slow/external as needed.
- Run locally: `pytest -q` (disabled tests live in `tests/disabled/`).
- Tokenizer cache warmup (for offline runs): `python -c "from tokenization import get_tokenizer; get_tokenizer(force_reload=True)"`.

## Commits & Pull Requests

- Conventional-style prefixes (`feat:`, `fix:`, `docs:`, `ci:`, etc.).
- Small, atomic commits; write clear, imperative messages.
- Open draft PRs early; keep PRs focused and under ~300 lines where possible.

### Checklist
- Commit style follows Conventional Commit‑like prefixes seen in history: `feat:`, `fix:`, `docs:`, `chore:`, `ci(...)`, `trainer:`, etc. Use imperative mood and reference issues (e.g., `Fixes #123`).
- Branch naming: `feat/<area>-<desc>`, `fix/<area>-<desc>`, `docs/<topic>`.
- PR checklist: clear description, linked issue, tests passing (`pytest -q`), formatted (`black`/`isort`), lint‑clean (`flake8`). Include logs/metrics screenshots for training changes when relevant.

## Security & Configuration

- Never commit secrets; use GitHub Actions secrets and local `.env` files ignored by Git.
- Prefer sandboxed code execution paths; use `scripts/security/safe_code_executor.py`.
- Pin dependencies and avoid unnecessary upgrades; document any changes.

### Tips
- Secrets: follow `GITHUB_SECRETS_SETUP.md`; never commit tokens or dataset keys.
- Offline/repro: pin `requirements.txt`; set `TRANSFORMERS_OFFLINE=1` after caching.
- Hardware: GPU optional; MPS/CPU supported for tests. Adjust batch sizes in config when on limited memory.

