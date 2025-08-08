# Repository Guidelines

## Project Structure & Module Organization
- `hrm/`: Core HRM model (`model.py`, `layers.py`, `config.py`).
- `hrm_codegen/`: Generation utilities and mock model for experiments.
- `scripts/`: Entry points (`train.py`, `evaluate.py`, `convert_mbpp.py`, monitoring tools).
- `configs/`: YAML configs (symlink from `hrm/configs` → `configs/hrm/`).
- `tokenization/`: Tokenizer utilities and caching helpers.
- `data/`, `checkpoints/`, `outputs/`: Local artifacts (not versioned).
- `tests/` and `test/`: Pytest suites (`tests/disabled/` is skipped).
- `docs/`: Architecture, SDLC, and ops documentation.

## Build, Test, and Development Commands
- Create env & install: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Quick sanity: `pytest -q`
- Prepare MBPP data: `python scripts/convert_mbpp.py --split all --output-dir data/mbpp --tokenizer gpt2`
- Train: `python scripts/train.py --config configs/hrm/mbpp_base.yaml --data-path data/mbpp/train.bin --eval-data-path data/mbpp/validation.bin --output-dir checkpoints/hrm-mbpp`
- Evaluate: `python scripts/evaluate.py --ckpt checkpoints/hrm-mbpp/latest.pt --split test --k 1 5 10`
- Format/lint: `black . && isort . && flake8`
- Optional Docker: `docker build -t hrm-codegen:latest .`

## Coding Style & Naming Conventions
- Python 3.10+, 4‑space indentation, line length per `black` defaults.
- Names: `snake_case` for functions/vars, `PascalCase` for classes, `SCREAMING_SNAKE_CASE` for constants.
- Configs: YAML in `configs/hrm/` with descriptive, kebab‑case filenames (e.g., `mbpp_base.yaml`).
- Keep large artifacts in `data/`, `checkpoints/`, `outputs/` (not in Git).

## Testing Guidelines
- Framework: Pytest. Place tests under `tests/` as `test_*.py`; mark slow/external as needed.
- Run locally: `pytest -q` (disabled tests live in `tests/disabled/`).
- Tokenizer cache warmup (for offline runs): `python -c "from tokenization import get_tokenizer; get_tokenizer(force_reload=True)"`.

## Commit & Pull Request Guidelines
- Commit style follows Conventional Commit‑like prefixes seen in history: `feat:`, `fix:`, `docs:`, `chore:`, `ci(...)`, `trainer:`, etc. Use imperative mood and reference issues (e.g., `Fixes #123`).
- Branch naming: `feat/<area>-<desc>`, `fix/<area>-<desc>`, `docs/<topic>`.
- PR checklist: clear description, linked issue, tests passing (`pytest -q`), formatted (`black`/`isort`), lint‑clean (`flake8`). Include logs/metrics screenshots for training changes when relevant.

## Security & Configuration Tips
- Secrets: follow `GITHUB_SECRETS_SETUP.md`; never commit tokens or dataset keys.
- Offline/repro: pin `requirements.txt`; set `TRANSFORMERS_OFFLINE=1` after caching.
- Hardware: GPU optional; MPS/CPU supported for tests. Adjust batch sizes in config when on limited memory.

