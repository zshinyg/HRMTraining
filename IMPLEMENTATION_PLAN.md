# IMPLEMENTATION_PLAN.md  
_Phase 3 — Incremental Adaptation of Sapient HRM to Code Generation_  
_Version: 2025-08-05_

---

## Overview
Phase 3 converts the puzzle-oriented HRM into a causal language model able to train and evaluate on the MBPP code-generation task.  
Duration: **7 working days** (D0 → D6).  
Participants:  
• ML Eng 1 – Model/architecture lead  
• ML Eng 2 – Data & evaluation lead  
• MLOps – Tooling, CI, infra  
• PM – Coordination & unblocking  

Success at the end of Phase 3 = first end-to-end training run on 10 k MBPP samples with decreasing loss and a working `evaluate.py` Pass@k invocation on generated checkpoints.

---

## Gantt-Style Calendar (high-level)

| Day | Primary Focus | Integration Check |
|-----|---------------|-------------------|
| D0 | Environment lock-in & spec sign-off | – |
| D1 | Data pipeline + tokenizer | Loader unit tests pass |
| D2 | Model code adaptation (input + causal attn) | HRM forward pass unit tests |
| D3 | Generation API + shifted-loss + train script | Smoke train (100 steps) |
| D4 | Evaluation harness wiring + CI | `evaluate.py` on dummy ckpt |
| D5 | Full mini-run (10 k samples) + metrics logging | Pass@k ≥ 2 % |
| D6 | Buffer / hardening / retro | Closeout review |

Detailed tasks below.

---

## Task Breakdown

### Day 0 – Kick-off & Environment Freeze
Deliverables  
1. `requirements.txt` pinned (torch, flash-attn, transformers 4.42).  
2. Dockerfile/conda-env verified on GPU node.  
Success Criteria  
• `pip install -r requirements.txt` completes on dev GPU; torch detects CUDA.  
Dependencies  
• None.  
Testing & Validation  
• `python -c "import torch; print(torch.cuda.is_available())"` → True.  
Rollback  
• Use previous week’s environment lock.  
Handoff  
• MLOps delivers working image to ML Engs.

---

### Day 1 – Data Loader & Tokeniser
Tasks  
1. Implement `tokenization/__init__.py` wrapping GPT-2 BPE (ML Eng 2).  
2. Create `datasets/mbpp_loader.py` returning `{input_ids, labels}` (ML Eng 2).  
3. Add unit tests `tests/test_tokenizer.py`, `tests/test_mbpp_loader.py`.  
Deliverables  
• Loader and tokenizer modules committed.  
Success Criteria  
• Unit tests: tokeniser round-trip equality; loader len ==# records; labels shifted correctly (ignore_index = -100).  
Dependencies  
• Day 0 environment.  
Blocking  
• None.  
Validation  
• `pytest tests/test_*` green.  
Rollback  
• Switch to HF `datasets.load_dataset("mbpp")` fallback.  
Handoff  
• Sample batch (tensor shapes) provided to ML Eng 1 for model debug.

---

### Day 2 – Model Code Adaptation
Tasks  
1. Strip puzzle embeddings, set `puzzle_emb_len = 0` (ML Eng 1).  
2. Modify `Attention` to respect `causal=True` flag (ML Eng 1).  
3. Update HRM config schema (`task: codegen`, `causal: true`) (ML Eng 2).  
4. Add unit tests `tests/test_hrm_forward.py` verifying output shape.  
Deliverables  
• Patched `external/sapient-hrm` diff.  
Success Criteria  
• Forward pass on dummy batch completes with no grad error; shapes `[B,S,V]`.  
Dependencies  
• Tokenizer & loader sample batch.  
Blocking  
• Day 1 loader.  
Validation  
• Run test script under `torch.autograd.gradcheck` for small fp64 case.  
Rollback  
• Toggle `task=puzzle` path; revert to bidirectional attention.  
Handoff  
• Working forward method handed to Day 3 training script.

---

### Day 3 – Generation & Training Loop
Tasks  
1. Implement `HierarchicalReasoningModel_ACTV1.generate` (greedy & temp) (ML Eng 1).  
2. Introduce `losses_codegen.py` with cross-entropy ignore_index (ML Eng 1).  
3. Create `scripts/train_codegen.py` – leverages Accelerate (ML Eng 1).  
4. Add CI smoke test `tests/test_train_smoke.py` (ML Eng 2).  
Deliverables  
• Train script runnable locally for 100 steps.  
Success Criteria  
• `python scripts/train_codegen.py --dry-run` finishes; loss computed; GPU mem < 12 GB.  
Dependencies  
• Day 2 model modifications.  
Blocking  
• None.  
Validation  
• Loss drops ≥ 5 % over 100 steps on 1 k samples.  
Rollback  
• Disable generate(); fall back to teacher-forcing only.  
Handoff  
• First checkpoint `checkpoints/dev_step_000100.pt` to Day 4 evaluation work.

---

### Day 4 – Evaluation Harness & Continuous Integration
Tasks  
1. Wire `scripts/evaluate.py` to HRM `generate()` (ML Eng 2).  
2. Extend GitHub Action workflow: lint, unit tests, 1-epoch CPU smoke train, eval (MLOps).  
3. Add sandbox resource limits for code exec (E-1 mitigation) (ML Eng 2).  
Deliverables  
• CI pipeline passing in PR.  
Success Criteria  
• `evaluate.py` on 2 tasks returns Pass@1 not NaN; CI completes < 15 min.  
Dependencies  
• Checkpoint from Day 3.  
Blocking  
• None.  
Validation  
• AST parse check on generated code passes.  
Rollback  
• Use reference transformer baseline in evaluation for CI.  
Handoff  
• CI badge green shared in Slack; dataset & log dirs path documented.

---

### Day 5 – Mini-Run & Metric Verification
Tasks  
1. Launch training on 10 k MBPP samples (seq_len 256, batch 8) for 1 epoch (ML Eng 1).  
2. Capture metrics in W&B; monitor memory, loss, grad norm (ML Eng 1).  
3. Run `evaluate.py --k 1` on resulting ckpt (ML Eng 2).  
Deliverables  
• `checkpoints/mbpp_mini/best.pt`; score report JSON.  
Success Criteria  
• Train loss ↓ ≥ 30 % vs start; Pass@1 ≥ 2 %.  
Dependencies  
• CI-verified code.  
Blocking  
• GPU availability.  
Validation  
• Manual review: sample generations compile, produce expected output on at least 1 task.  
Rollback  
• Reduce learning rate 10× and retry; if failure persists, revert to transformer baseline checkpoint.  
Handoff  
• Metrics & checkpoint pushed to shared bucket for D6 review.

---

### Day 6 – Hardening, Buffer, Retrospective
Tasks  
1. Address residual bugs / flaky tests (all).  
2. Clean code & add docstrings; finalise `README` code-gen section (ML Eng 2).  
3. PM runs Phase 3 completion checklist & retro session.  
Deliverables  
• Tag `phase3-complete` on main branch; CHANGELOG entry.  
Success Criteria  
• All unit tests pass; CI green; documentation ≥ 90 % code cov in docstrings.  
Dependencies  
• Day 5 results.  
Rollback  
• Extend phase by 1 day (buffer) with PM approval.  
Handoff  
• Phase 4 tuning tasks created in Jira, including hyper-param sweep list.

---

## Dependency Graph (critical path)

```
[D0 env] → [D1 loader] → [D2 model] → [D3 train] → [D4 CI+eval] → [D5 mini-run] → [D6 wrap-up]
```

If any node slips, all downstream nodes shift equivalently; buffer is built into D6.

---

## Integration Checkpoints

1. **Loader × Model (D2 EOD)** – sample batch through forward.  
2. **Model × Train Loop (D3 noon)** – smoke train 100 steps.  
3. **Train Loop × Evaluation (D4 EOD)** – ckpt evaluated in CI.  
4. **End-to-End Mini-Run (D5)** – metrics published.

---

## Testing Matrix

| Component | Unit Test | Integration Test | CI Target |
|-----------|-----------|------------------|-----------|
| Tokeniser | ✅ | Loader sample | CPU |
| Loader | ✅ | Forward pass | CPU |
| Attention (causal) | ✅ mask correctness | Forward vs teacher logits | GPU |
| Generate() | – | Evaluate.py dry run | CPU |
| Train script | Smoke 100 steps | Mini-run | GPU |
| Evaluate | AST safety | Pass@k calc | CPU |

---

## Rollback Strategy Summary

| Change | Detection | Immediate Rollback |
|--------|-----------|--------------------|
| Model causal mask causes NaN | Loss becomes NaN/Inf | Revert to bidirectional (`causal=False`) flag, continue training |
| Tokenizer mismatch | Unit tests fail / sudden loss spike | Switch to saved vocab JSON from D1 |
| Evaluate sandbox crash | CI red | Use OpenAI original evaluator script |
| FlashAttention build fail | ImportError | Set `USE_FLASH=False` env; use PyTorch attention |

All rollbacks scripted via `scripts/rollback.sh` which reverts config flags without code revert.

---

## Handoff Requirements

| From → To | Artefact | Due |
|-----------|----------|-----|
| MLOps → Engs | Docker image tag `hrm-codegen:v1` | D0 16:00 |
| ML Eng 2 → ML Eng 1 | `datasets/mbpp_loader.py` + sample tensors | D1 EOD |
| ML Eng 1 → ML Eng 2 | Patched HRM wheel/installable | D2 EOD |
| ML Eng 1 → ML Eng 2 | `generate()` API doc | D3 noon |
| ML Eng 2 → Team | CI dashboard link | D4 EOD |
| ML Eng 1 → PM | W&B run URL, Pass@k JSON | D5 18:00 |
| PM → All | Phase 4 task list | D6 12:00 |

All artefacts stored in `s3://hrm-artifacts/<phase>/<day>/`.

---

## Communication Protocol
• Daily 15-min stand-up 10:00 PT.  
• Slack channel **#hrm-codegen** for async updates (thread per day).  
• Any blocker > 2 h → escalate to PM immediately.

---

_End of Implementation Plan — Phase 3._
