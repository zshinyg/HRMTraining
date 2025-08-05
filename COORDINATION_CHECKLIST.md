# COORDINATION_CHECKLIST.md  
_HRM-CodeGen • Daily Operator Guide_  
_Last updated: 2025-08-05_

---

## 1 · Immediate Action Items (Today)

- [ ] **Ping each active Droid via Factory interface** (Product → Research, Infrastructure, SE) using the templates in §4.  
- [ ] Verify **Research Droid** has cloned repo & can open `train_raw.json`, `test_raw.json`.  
- [ ] Verify **Infrastructure Droid** pushed initial CI workflow PR (`.github/workflows/lint-test.yml`).  
- [ ] Pull latest **Engineering** branch `feat/data-layer` and confirm tests pass locally (`pytest -q`).  
- [ ] Update `INTEGRATION_TIMELINE.md` status column (✅ ⏳ ❌) and commit the change.  
- [ ] Post end-of-day summary **as a GitHub comment** on the coordination issue and label any blockers.

---

## 2 · Daily Status Check Questions

Ask **each workstream** via Factory message or GitHub comment:

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
| Dataset licence conflict | New dataset not Apache/MIT/BSD | Escalate for licence review, drop dataset |
| Flash-Attn compile fail | `nvcc` error | Use CPU path, open infra ticket |
| Duplicate PR merges | Conflicting file changes | Lock branch, require 2 reviews |
| Raw data missing | `.gitignore` excludes critical file | Force-add with `git add -f`, document removal plan |

---

## 4 · Communication Templates

### 4.1  Stand-up Message (Factory or GitHub)

> **Daily Stand-up**  
> 1️⃣ Done: …  
> 2️⃣ Doing: …  
> 3️⃣ Blockers: …

### 4.2  Blocker Alert (Factory message + GitHub issue)

> **⚠️ Blocker Alert**  
> Issue: _<brief description>_  
> Impact: _High/Med/Low_  
> ETA slip: _<hours/days>_  
> Help needed: _<decision/support>_

### 4.3  Handoff Notice (GitHub comment)

> **Handoff Ready**  
> Artifact: _<file/branch>_  
> Meets criteria (§6) ✔  
> Please review by _<date>_.

---

## 5 · Decision Points – Immediate Attention

| Decision | Deadline | Owner | Impact if Delayed |
|----------|----------|-------|-------------------|
| Freeze data schema (`data_schema_v1.json`) | **Aug-06** | Research Droid | Retrains required |
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
   - Major: Delays non-critical tasks > 24 h  
   - Minor: Can be fixed within same day

2. **Notify**  
   - Send **Factory message** with template (§4.2) to responsible Droid(s) and Product lead  
   - Create or update **GitHub issue** labelled `blocker`

3. **Assign**  
   - Assign owner in GitHub issue, set due date ≤ 24 h (Critical) or 48 h (Major)

4. **Track**  
   - Update issue hourly (Critical) or twice daily (Major)  
   - Product Droid logs in risk register (`INTEGRATION_TIMELINE.md`)

5. **Escalate**  
   - If unresolved by due date, mention Engineering lead + Product in Factory thread and bump issue priority

6. **Post-Mortem**  
   - Within 48 h of resolution; document root cause, fix, prevention steps in `docs/post_mortems/`

---

_This checklist is owned by **Product Droid** and should be reviewed/updated every morning before the daily coordination message._  
