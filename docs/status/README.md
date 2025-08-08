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

```bash
git fetch && git checkout fix/base-hrm-bringup && git pull
source .venv/bin/activate && pytest -q
```
