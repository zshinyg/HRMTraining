# P0 Tracker — Unify/Modify Model Path

- Owner: @zshinyg
- Branch: `spike/p0-model-path`
- Purpose: Track decisions and context for modifying/unifying the model path between `hrm/` and `hrm_codegen/` stacks.

## Scope
- Decide primary model stack for training/eval default: Sapient-HRM CodeGen (`hrm_codegen/`).
- Keep Native HRM (`hrm/`) as an experimental path; adapters will allow parity where feasible.
- De-duplicate and unify config pathing (`configs/hrm/` as single source of truth).

## Constraints
- Keep edits small, reversible; avoid broad refactors.
- Keep tests green (`pytest -q`).
- Match existing module structure; don’t break CLIs `scripts/train.py` and `scripts/evaluate.py`.

## Plan (incremental)
1) Discovery: map model creation call sites (`scripts/train.py`, `scripts/evaluate.py`, `training/trainer.py`).
2) Introduce a selector in config (e.g., `model_stack: sapient|native`) defaulting to `sapient`; wire to factory.
3) Update `README.md` and docs to reflect Sapient-HRM as default; cross-link upstream.
4) Add unit test: model instantiation from `configs/hrm/mbpp_dev.yaml` using Sapient-HRM path.
5) Dataset plug-and-play: add simple registry in `datasets/` and config-driven selection (HF MBPP first).

## Immediate deliverables (for hypothesis demo in 4 days)
- Script: `scripts/quick_eval_codegen.py` to evaluate Sapient-HRM CodeGen on MBPP dev (≈25 tasks)
  - Inputs: `configs/codegen_base.yaml`
  - Outputs: console + JSON summary (Pass@1, per-task timing)
  - Execution: subprocess isolation (no Docker); CPU/MPS/CUDA friendly
- Keep edits minimal; no training refactors; reuse `hrm_codegen/generation.py` helpers

## Linked docs
- `docs/CODEGEN_EVAL_PLAN.md` — end-to-end evaluation plan and commands

## Open Questions
- Any blockers merging configs (field name mismatches) between sapient/native?
- How to harmonize generation APIs for incremental decoding later?

## Notes
- Follow repo rule: prefer absolute config paths under `configs/hrm/`.
- Default dataset: HF MBPP; add registry for plug-and-play future datasets (HumanEval, CodeContests).
- No dependency changes unless required by tests; document if needed.