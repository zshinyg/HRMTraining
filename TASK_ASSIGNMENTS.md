# TASK_ASSIGNMENTS.md
_HRM-CodeGen Project – Active Task & Ownership Tracker_  
_Last updated: 2025-08-05_

---

## 1 · Active Task Matrix

| ID | Task Description | Owner Droid / Team | Priority | ETA | Status |
|----|------------------|--------------------|----------|-----|--------|
| T-01 | **Fix HRM `generate()` interface mismatch** (tokenised `input_ids` vs prompt string) in `real_hypothesis_test.py` & training pipeline | **SE Code Droid** | **P0 – Critical** | Aug-05 PM | ⏳ In Progress |
| T-02 | **Gate C2 – CI/CD Pipeline Validation**: ensure GitHub Actions green incl. W&B upload step | Infrastructure Code Droid | P1 – High | Aug-06 | 🔄 Pending |
| T-03 | **Gate C3 – Real HRM Forward & Train smoke run** (1-epoch, loss ↓, no NaN) | SE Code Droid | P1 – High | Aug-09 | ⏳ Blocked → T-01 |
| T-04 | **Research Dashboard Polish**: finalise W&B panels, public link, KPI badges | Research Droid | P2 – Medium | Aug-07 | 🔄 Pending |
| T-05 | **Product Documentation Refresh & Road-map freeze for Phase 4** | Product Droid | P3 – Low | Aug-07 | 🔄 Ongoing |
| T-06 | **Security Scan Action**: run `security.yml`, zero Critical findings | Infrastructure Code Droid | P2 – Medium | Aug-08 | 🔄 Pending |

_Status Legend_: ✅ Done · ⏳ In Progress · 🔄 Pending · ⚠ Blocked

---

## 2 · Current Blocking Issues

| Blocker ID | Description | Impacted Tasks | Assigned To | Resolution ETA |
|------------|-------------|----------------|-------------|----------------|
| B-01 | HRM `generate()` requires `input_ids`; current code passes string prompt | T-01, T-03 | **SE Code Droid** | Aug-05 PM |
| B-02 | W&B secrets available but CI step not yet green | T-02 | Infrastructure Code Droid | Aug-06 |
| B-03 | Security workflow untested; could block Gate C2 merge | T-02, T-06 | Infrastructure Code Droid | Aug-08 |

---

## 3 · Task Dependencies

```
T-01 ─┐                       (interface fix)
      ├─► T-03 (HRM smoke train) ─► Gate C3
T-02 ─┘
T-02 (CI green) ─► Gate C2 ─┐
                            └─► prerequisite for merge to main before Phase 4
T-04 – independent (dashboard polish)
T-06 depends on T-02 (CI framework operational)
```

---

## 4 · Acceptance Criteria

| Task ID | Acceptance Criteria |
|---------|---------------------|
| T-01 | a) HRM generation succeeds on 3 sample prompts without error<br>b) `real_hypothesis_test.py` completes 5-sample run, outputs metrics JSON |
| T-02 | a) `ci.yml`, `benchmark.yml`, `monitoring.yml`, `security.yml` all green on PR<br>b) Total pipeline ≤ 10 min |
| T-03 | a) 1-epoch train completes, loss decreases >5 % from step 0<br>b) No NaN/Inf in gradients<br>c) Checkpoint `dev-best.pt` saved |
| T-04 | a) W&B dashboard publicly accessible (read-only)<br>b) Panels show Pass@k & latency metrics for latest run |
| T-05 | a) INTEGRATION_TIMELINE.md & SUCCESS_METRICS_DASHBOARD.md updated<br>b) Phase 4 roadmap approved by all leads |
| T-06 | a) Security Action completes with 0 Critical/High findings<br>b) Report stored in `artifacts/` |

---

## 5 · Professional Status Update Template

Managers / droids should post updates in the daily stand-up thread using:

```
[YYYY-MM-DD] <Droid> – <Task ID(s)> – <Status Emoji>
• Progress:
• Blockers / Help needed:
• Next ETA:
```

_Example_  
`[2025-08-05] SE Code Droid – T-01 – ⏳`  
• Progress: tokenisation fix pushed, unit test green  
• Blockers: none  
• Next ETA: full 5-sample test by 18:00 PT

---

## 6 · Next Review

Task board will be reviewed **Aug-06 09:30 PT** in the stand-up. Owners should update status before that time.

_Document owner: **Product Droid** – updates welcome via PR._  
