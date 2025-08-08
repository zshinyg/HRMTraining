### SDLC Tracker – HRM-CodeGen

This document tracks engineering progress and decisions for the HRM-CodeGen project, following lightweight, clear SDLC practices suitable for a research codebase.

## 1. Scope & Goals

- Systematically validate core functionality via unit tests and small, reversible edits
- Keep tests green locally (pytest -q) and maintain clean, readable code
- Incrementally integrate HRM codegen path and extend evaluation while avoiding heavy dependencies in unit tests

## 2. Current Baseline

- Passing targets (design intent):
  - Standalone config: `hrm_codegen/config_standalone.py`
  - Tokenizer: `tokenization/__init__.py`
  - MBPP loader: `datasets/mbpp_loader.py`
  - Training loop with mock model: `training/trainer.py`
- Deferred/disabled: Full HRM integration tests under `tests/disabled/`

## 3. How to Run

- Unit tests (baseline):
  - `pytest -q`
- Tokenizer pre-cache (optional/offline):
  - `python -c "from tokenization import get_tokenizer; get_tokenizer(force_reload=True)"`
- Convert MBPP (for manual training/eval workflows):
  - `python scripts/convert_mbpp.py --split all --output-dir data/mbpp`

## 4. SDLC Working Agreements

- Small, clear edits; avoid broad refactors unless required by tests
- Match existing style and module structure; keep imports stable
- Format with `black` and `isort`; keep touched files flake8-clean
- Add brief docstrings where non-obvious; avoid noisy inline comments
- Prefer explicit error handling and guard clauses

## 5. Branching & PR Policy

- Branch names: `docs/<topic>`, `fix/<area>-<desc>`, `feat/<area>-<desc>`
- PR checklist (must):
  - Tests pass locally: `pytest -q`
  - Only intended files changed; formatted with `black`/`isort`
  - Terse summary, clear impact; link to this tracker if process-related
- Reviewers: 1 reviewer minimum; self-merge allowed after approval & green tests

## 6. Test & Validation Strategy

- Unit tests focus on lightweight paths (no heavy HRM deps)
- Data-free tests use temp dirs; no network assumptions beyond first-time tokenizer download
- When logic changes, add/extend tests in `tests/` and keep runtime short

## 7. Risk Register (active)

- Tokenizer/network hiccups on first run → Pre-cache or set offline mode
- Sequence masking/alignment regressions → Covered by MBPP loader and tokenizer tests
- Serialization changes with newer torch → Trainer uses `weights_only=False` when loading

## 8. Decision Log (append-only)

- 2025-08-08: Keep full HRM integration tests disabled until upstream deps are stabilized; prioritize lightweight paths

## 9. Near-Term Plan (Backlog)

- Run full unit test suite locally and fix any failures
- Generate MBPP artifacts for manual runs (`scripts/convert_mbpp.py`)
- Add a small codegen smoke test covering `hrm_codegen/generation.py` utilities
- Evaluate enabling a minimal HRM forward smoke test once `external/sapient-hrm` is confirmed stable

## 10. Definition of Done

- Tests green locally (`pytest -q`), or failures documented as unrelated
- Lint/format clean on touched files
- Behavior change is covered by or reflected in tests

## 11. Release & Artifacts

- Keep large artifacts out of VCS; use `outputs/` and `checkpoints/` locally
- Checkpoints and tokenizer cache reside under `checkpoints/`

## 12. Monitoring & Security (lightweight)

- Prefer CPU/MPS-friendly configs in unit tests to avoid OOM
- Never execute untrusted code outside `scripts/security/safe_code_executor.py`

---

Owner: Engineering
Status: Active
