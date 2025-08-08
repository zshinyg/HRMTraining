Status as of now

- Tests: green locally (`pytest -q`).
- Branch: fix/base-hrm-bringup (open a PR to merge).
- Trainer: optimizer_step returns LR-before-step and increments global_step; added _should_continue/_check_should_continue; robust Pass@k (handles mocked evaluator and generation errors deterministically).
- Dataset/Tokenizer: MBPP samples include attention_mask; dataset keeps task_id as int by default; Trainer adds a transform to tensorize task_id for batching; create_training_batch returns attention_mask.

Next 1–3 actions

1) Decide on using a proper DataLoader collate_fn to handle task_id tensorization instead of a Trainer-level transform.
2) Update README with a short note on attention_mask propagation and Pass@k evaluation behavior (mock-friendly and error-tolerant).
3) Optional: add a small unit test for evaluate_pass_k when model.generate raises exceptions to lock in behavior.

How to resume

```
git fetch && git checkout fix/base-hrm-bringup && git pull
source .venv/bin/activate && pytest -q
```

## Urgent Tasks (Phase 3 Critical Path)

- P0 — Today
  - C3 Smoke Train (Owner: SE Code Droid)
    - Do: Run `scripts/train.py` with `configs/hrm/mbpp_dev.yaml` on MBPP dev; monitor for NaN/Inf; save and reload `dev-best.pt`; ensure tests green with `pytest -q`.
    - Done when: loss drops ≥ 5% from step 0; no NaN/Inf; checkpoint saved and reloadable; tests pass.
  - C2 CI/CD Green (Owner: Infrastructure Code Droid)
    - Do: Ensure `.github/workflows/{ci.yml,benchmark.yml,monitoring.yml,security.yml}` succeed on PRs; cache deps; total CI ≤ 10 min; upload artifacts; enforce coverage and security scan.
    - Done when: all workflows green; time ≤ 10 min; coverage ≥ 90%; 0 Critical/High findings; badge visible in `README.md`.
  - Evaluation safety hardening (Owner: ML Eng 2 + Infra)
    - Do: Route Pass@k execution through `scripts/security/safe_code_executor.py` with strict time/resource limits and import filtering; add unit test for malicious snippet.
    - Done when: malicious snippet blocked; benign examples execute; CI runs these tests and passes.

- P1 — Next 24–48 h
  - Finalize and commit active edits (Owner: SE Code Droid)
    - Do: Clean up and commit changes in `real_hypothesis_test.py`, `scripts/{benchmark_training.py,evaluate.py,reliability_monitor.py}`, `training/trainer.py`; run `pytest -q` pre/post.
    - Done when: clean git status; tests green; brief PR notes.
  - W&B integration and secrets (Owner: Infrastructure Code Droid)
    - Do: Validate `WANDB_API_KEY` in CI; ensure runs created; add graceful fallback when secret absent.
    - Done when: artifacts uploaded to W&B; CI green even without secret.
  - MBPP data integrity (Owner: Research Droid)
    - Do: Re-run `scripts/convert_mbpp.py` for all splits; verify counts/checksums; confirm tokenizer round-trip in tests.
    - Done when: expected counts; checksums recorded; round-trip test passes.
  - CPU throughput benchmark (Owner: Infra + SE)
    - Do: Run `scripts/benchmark_inference.py` and `scripts/benchmark_training.py` on CPU; publish artifact.
    - Done when: ≥ 550 tok/s or gap documented with issue; artifact uploaded.
  - Reliability monitor in train runs (Owner: Infra)
    - Do: Hook `scripts/reliability_monitor.py` into training and Docker entrypoint.
    - Done when: monitor logs present; anomalies recorded; smoke test imports/entrypoint in CI.

- P2 — Near-term polish
  - Quickstart ≤ 15 min (Owner: Product + SE)
    - Do: Dry-run `README.md` Quickstart from clean env; fix deviations.
    - Done when: ≤ 15 min reproducible; docs updated.
  - Phase 4 API surface prep (Owner: Product + Eng)
    - Do: Freeze `hrm_codegen` SDK API (load, train, eval, generate); reserve `<PLAN>` tokens per hybrid proposal.
    - Done when: API doc draft ready; tokens reserved without breaking tests.