---
status: active
owner: zshinyg
summary: Overview and prioritized tasks for HRM-CodeGen; tracks work across model, data, CI, and docs
last_reviewed: 2025-08-08
---

# HRM-CodeGen · Overview and Prioritized TODOs

## High-level overview
- Core model stacks:
  - `hrm/` implements a native PyTorch HRM adapted for codegen (`HRMModel`, custom layers, YAML configs).
  - `hrm_codegen/` wraps the vendored Sapient HRM for codegen with generation utilities and standalone configs.
- Training & eval:
  - Lightweight trainer in `training/trainer.py` with datasets from `datasets/mbpp_loader.py`.
  - Heavy-duty orchestration under `scripts/training/` and separate CLI trainers in `scripts/train.py`.
  - Evaluation in `scripts/evaluate.py`; benchmarks in `scripts/benchmark_*`.
- Tokenization: GPT‑2 tokenizer wrapper in `tokenization/`, cached under `checkpoints/tokenizer/`.
- Security: Docker-based safe executor in `scripts/security/safe_code_executor.py` (not yet wired into main eval path).
- Tests: Good coverage for tokenization, MBPP loader, and trainer; HRM forward/codegen tests exist but a suite is disabled.

---

## P0 — Critical path (correctness, consistency, and safety)
- Unify model path (pick one primary stack)
  - Decide between `hrm/` (native) and `hrm_codegen/` (vendored wrap) as the training/eval default.
  - Create thin adapters so the non-primary stack can still be used for experiments without duplicating code.
  - Remove duplicated config shapes; a single source of truth for config → model args.
- Fix generation correctness/perf in `hrm/HRMModel`
  - Respect `attention_mask` throughout forward/generation and apply hierarchical masks (`get_hierarchical_mask`).
  - Implement proper incremental decoding/KV-state caching (avoid full re-forward each step).
  - Preallocate tensors instead of cloning/concat in loops; vectorize repetition penalty.
  - Add targeted unit tests for causal masking, cache reuse, and output shapes on small toy inputs.
- Secure evaluation by default
  - Integrate `scripts/security/safe_code_executor.py` into `scripts/evaluate.py` and trainer’s Pass@k path with a flag: `--sandbox {docker|subprocess|off}`.
  - Provide a CI-safe fallback when Docker is unavailable; document expected behavior and limits.

## P1 — Quality, data, and CI
- Dataset pipeline consistency
  - Align `datasets/mbpp_loader.py` (raw JSON) with `scripts/convert_mbpp.py` (binary) or deprecate one.
  - Define a single schema (prompt/completion/tests) and enforce it end‑to‑end.
  - Add a schema validator test and a tiny, checked‑in sample for CI.
- Testing gaps
  - Enable and modernize `tests/disabled/test_hrm_forward.py.disabled` once deps are present; split into unit/integration.
  - Add unit tests for `hrm/layers.py` (HierarchicalAttention, recurrent layers) and `hrm/model.generate` behaviors.
- CI & dev‑experience
  - Add workflows: lint (black/isort/flake8), unit tests (CPU), minimal eval smoke test (no Docker required), optional sandbox test, docs lint+link check (added).
  - Pre-commit hooks and a short CONTRIBUTING quickstart.
  - Point contributors to `docs/index.md` as the entry point.
- Dependencies & versions
  - Review/refresh `requirements.txt` (torch/transformers/datasets) for security and compatibility.
  - Add optional extras (e.g., `sandbox`, `monitoring`) to keep base install light.

## P2 — Performance, UX, and docs
- Benchmarks → reality
  - Wire `scripts/benchmark_*` to real models/datasets with optional synthetic fallback; log tokens/s, memory, accuracy buckets.
- Monitoring & orchestration simplification
  - Connect `scripts/training/resource_monitor.py` to the light trainer via optional callbacks; reduce duplication with `monitoring_utils.py`.
- Tokenization options
  - Expose a code-specialized tokenizer option (e.g., code-parrot/GPT‑NeoX vocab) via config with on-disk caching.
- Docs & examples
  - Reconcile README paths (e.g., `configs/hrm/mbpp_base.yaml` vs `hrm/configs/…`).
  - Add a minimal “Train → Evaluate → Report” walkthrough with exact commands and expected outputs.

---

## Quick wins (1–2 days)
- Respect `attention_mask` in `hrm/model.py` forward/generate and add one unit test.
- Sandbox flag in `scripts/evaluate.py` that uses subprocess today, Docker when available; document.
- CI: add lint + unit test workflow and tiny MBPP sample for deterministic tests.
- Fix README config paths and add a working 5‑minute “hello world” run.

## Near-term (1–2 weeks)
- Implement incremental decoding cache in `HRMModel.generate` with tests; profile vs. baseline.
- Unify dataset pipeline and delete one redundant format; add schema test.
- Enable and stabilize HRM forward/codegen tests; add coverage for `hrm/layers.py`.
- Migrate Pass@k evaluation to SafeCodeExecutor with docker/subprocess backends.

## Medium-term (3–6 weeks)
- Choose primary model stack and de-duplicate configs; introduce adapter layer for the secondary.
- Flesh out benchmarks with real runs; produce a W&B dashboard template; track tokens/s and pass@k by difficulty.
- Reduce orchestration surface area; keep a single, documented path for research vs. production runs.

---

## Notes
- Secrets: follow `GITHUB_SECRETS_SETUP.md` for W&B/Docker/HF. Gate sandboxed eval in CI to trusted branches.
- Owners: default to `zshinyg`; assign explicit owners as the team is formed.