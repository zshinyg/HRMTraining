# COORDINATION_CHECKLIST.md  
_HRM-CodeGen • Daily Operator Guide_  
_Last updated: 2025-08-05_

---

## 1 · Immediate Action Items (Today)

- [ ] **Ping each Droid on Slack** (`#hrm-daily`) using templates in §4.  
- [ ] Verify **Research Droid** has cloned repo & can open `train_raw.json`, `test_raw.json`.  
- [ ] Verify **Infrastructure Droid** pushed initial CI workflow PR (`.github/workflows/lint-test.yml`).  
- [ ] Verify **Data Droid** created branch `data/dup-scan` and committed first analysis script.  
- [ ] Pull latest **Engineering** branch `feat/data-layer` and confirm tests pass locally (`pytest -q`).  
- [ ] Update `INTEGRATION_TIMELINE.md` status column (colour codes ✅ ⏳ ❌).  
- [ ] Post end-of-day summary in `#hrm-daily` & label blockers in `#hrm-blockers`.

---

## 2 · Daily Status Check Questions

Ask **each workstream**:

1. **What did you complete since last check-in?**  
2. **What are you working on today?**  
3. **Any blockers, risks, or resource needs?**  
4. **Is your ETA for the next integration gate still accurate?**  
5. **Do you need input from another Droid/team?**

Record answers in the GitHub Project board comment thread for that task.

---

## 3 · Blockers to Monitor

| Category | Specific Symptoms | Immediate Mitigation |
|----------|-------------------|----------------------|
| Gradient errors | `RuntimeError: inplace operation` resurfaces | Enable `torch.autograd.set_detect_anomaly(True)`, fallback smaller batch |
| CI Timeouts | GitHub Action > 20 min | Split job, enable cache, switch to self-hosted runner |
| Dataset licence conflict | New dataset not Apache/MIT/BSD | Escalate to legal review, drop dataset |
| Flash-Attn compile fail | `nvcc` error | Use CPU path, open infra ticket |
| Duplicate PR merges | Conflicting file changes | Lock branch, require 2 reviews |
| Raw data missing | `.gitignore` excludes critical file | Force-add with `git add -f`, document removal plan |

---

## 4 · Communication Templates

### 4.1  Stand-up Ping
> **Daily Stand-up**  
> 1️⃣ Done: …  
> 2️⃣ Doing: …  
> 3️⃣ Blockers: …

### 4.2  Blocker Escalation
> **@team** ⚠️ Blocker: _<issue>_  
> Impact: _High/Med/Low_  
> ETA slip: _<hours/days>_  
> Need: _<decision/help>_

### 4.3  Handoff Ready
> **Handoff Notice**  
> Artifact: _<file/branch>_  
> Meets criteria in §6 ✔  
> Please review by _<date>_.

---

## 5 · Decision Points – Immediate Attention

| Decision | Deadline | Owner | Impact if Delayed |
|----------|----------|-------|-------------------|
| Freeze data schema (`data_schema_v1.json`) | **Aug-06** | Data Droid | Retrains required |
| Choose causal mask impl (custom vs PyTorch) | Aug-07 | Eng Lead | Model refactor risk |
| CI runner selection (GH hosted vs self) | Aug-08 | Infra Lead | Build times |
| Metric threshold acceptance (Pass@1/10) | Aug-12 | Product + Research | Phase 3 GA |

Escalate undecided items 24 h before deadline.

---

## 6 · Handoff Validation Criteria

| Deliverable | Must Include | Verification |
|-------------|--------------|--------------|
| **Code PR** | Passing CI, unit tests, docs updated | Reviewer runs `scripts/test_setup.py` |
| **Research Report** | Exec summary, CSV stats, plots PNG/SVG | Check figs render in GitHub |
| **Data Clean Batch** | Duplicate list, diff file, updated `.bin` | Run `python tests/test_mbpp_loader.py` |
| **CI Workflow** | Lint, test, coverage, cache | Trigger workflow on dummy PR |
| **Model Checkpoint** | `config.yaml`, `.pt`, eval log | Run `scripts/evaluate.py --ckpt …` |

---

## 7 · Escalation Procedures

1. **Identify Severity**  
   - Critical: Blocks critical path, risk to timeline  
   - Major: Delays non-critical tasks >24 h  
   - Minor: Can be fixed within same day

2. **Notify**  
   - Post in `#hrm-blockers` with template (§4.2)  
   - Tag responsible owner + `@p-mgr`

3. **Assign**  
   - Create GitHub issue labelled `blocker` + milestone  
   - Assign owner, set due date ≤ 24 h (Critical) or 48 h (Major)

4. **Track**  
   - Update status hourly (Critical) or twice daily (Major)  
   - Product Droid logs in risk register (§5 INTEGRATION_TIMELINE.md)

5. **Escalate**  
   - If unresolved by due date, escalate to team lead & PM; schedule war-room call

6. **Post-Mortem**  
   - Within 48 h of resolution; document root cause, fix, prevention steps (`docs/post_mortems/`)

---

_This checklist is owned by **Product Droid** and should be reviewed/updated every morning before stand-up._
