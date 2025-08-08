# P0 Tracker — Unify/Modify Model Path

- Owner: @zshinyg
- Branch: `spike/p0-model-path`
- Purpose: Track decisions and context for modifying/unifying the model path between `hrm/` and `hrm_codegen/` stacks.

## Scope
- Decide primary model stack for training/eval default (candidate: `hrm/`).
- Add thin adapter(s) so the non-primary stack remains usable for experiments.
- De-duplicate and unify config pathing (`configs/hrm/` as single source of truth).

## Constraints
- Keep edits small, reversible; avoid broad refactors.
- Keep tests green (`pytest -q`).
- Match existing module structure; don’t break CLIs `scripts/train.py` and `scripts/evaluate.py`.

## Plan (incremental)
1) Discovery: map current call sites and config loaders for model creation.
2) Introduce a minimal adapter to select the primary stack via config flag/env.
3) Update `README.md`/paths for consistency; add redirect notes.
4) Add/adjust 1–2 unit tests covering model instantiation from unified config.

## Open Questions
- Which stack has the most stable/maintained generation path today?
- Any blockers merging configs (field name mismatches)?

## Notes
- Follow repo rule: prefer absolute config paths under `configs/hrm/`.
- No dependency changes unless required by tests; document if needed.