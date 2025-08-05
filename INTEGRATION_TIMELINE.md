# INTEGRATION_TIMELINE.md
_HRM-CodeGen • Master Coordination & Convergence Plan_  
_Last updated: 2025-08-05_

---

## 1 · Workstream Status Snapshot

| Workstream | Lead Droid / Team | Phase | Current % Done | Key Deliverables | ETA |
|------------|------------------|-------|----------------|------------------|-----|
| **Phase 3 Engineering** | Software-Engineering Droid + Eng team | Implementation | 60 % | Data layer adaptation in PR review | **Aug-07** |
| **Research** | Research Droid | Analysis | 10 % | Raw-stats pipeline scaffolded | **Aug-11** |
| **Infrastructure** | Infrastructure Droid | Setup | 25 % | CI lint/test workflow green | **Aug-09** |
| **Data** | Data Droid | Analysis | 5 % | Duplicate/quality scan script | **Aug-10** |
| **Product** | Product Droid (this doc) | Coordination | 100 % | Success criteria, roadmap, UX, docs | — |

---

## 2 · Dependency & Handoff Matrix

| From → To | Artifact / Signal | Blocking? | Notes |
|-----------|-------------------|-----------|-------|
| Engineering → Product | Working generate()/train | **Yes** | Required to freeze UX API |
| Engineering → Research | Checkpoint `dev-best.pt` | **No** | For baseline comparisons |
| Engineering → Infrastructure | Dockerfile + start script | **Yes** | CI packaging & benchmark jobs |
| Data → Engineering | Cleaned dataset splits | **Yes** | Needed before Phase 3 final tuning |
| Research → Product | Baseline benchmark report | **No** | Informs Phase 4 targets |
| Infrastructure → All | CI gates & badges | **Yes** | Must be green before merge to `main` |

_Handoff points marked **Yes** form the critical path._

---

## 3 · Critical Path Analysis (Phase 3 → Phase 4)

```
Data cleansing ─┐
                ├─► Engineering final tune ─► “dev-best.pt”
Research stats ─┘                           │
                                             │
Docker/CI setup ──► Perf bench ► Validation ─┘
```

**Longest chain = 6 days**  
Slack in path: Research (can continue during Phase 4).

---

## 4 · Integration Checkpoints & Validation Gates

| # | Date | Gate | Owner | Validation | Pass Criteria |
|---|------|------|-------|------------|---------------|
| C1 | Aug-07 | Data-Layer Merge | Eng | Unit tests + `pytest -q` | 100 % pass |
| C2 | Aug-08 | CI/CD Pipeline | Infra | GitHub Actions green on PR | 0 failures |
| C3 | Aug-09 | Model Forward-&-Train | Eng | 1-epoch train, no NaN | loss ↓ |
| C4 | Aug-10 | Research Baseline Drop | Research | `RESEARCH_FINDINGS.md` PR | review OK |
| C5 | Aug-11 | Perf Benchmark | Infra | `scripts/benchmark.py` | ≥550 tok/s |
| C6 | Aug-12 | Data Clean v1 | Data | `DATA_ANALYSIS_REPORT.md` PR | signed-off |
| C7 | Aug-13 | **Phase 3 Final QA** | Product | All success criteria met | see §7 |

_All gates must pass to enter Phase 4._

---

## 5 · Risk Register & Contingencies

| ID | Risk | Likelihood | Impact | Mitigation | Fallback |
|----|------|-----------|--------|------------|----------|
| R1 | Gradient instabilities persist | Med | High | Enable grad-checkpointing, anomaly-detect | Switch to smaller batch |
| R2 | CI job timeouts (macOS runners) | High | Med | Split tests, cache deps | Self-hosted runner |
| R3 | Data license conflict in expansion | Low | Med | OSS-license review checklist | Drop dataset |
| R4 | Flash-Attn compile failure on GPU | Med | High | CPU fallback bench path | Postpone perf goal |
| R5 | Cross-team merge conflicts | Med | Med | Feature branches + fast-forward merges | Lock code-freeze 24 h |

---

## 6 · Coordination Protocols & Communication Plan

| Channel | Purpose | Frequency |
|---------|---------|-----------|
| Slack `#hrm-daily` | Stand-up updates (<3 lines) | Daily 09:30 PT |
| Slack `#hrm-blockers` | Urgent blockers, mention @owner | As needed |
| Weekly Zoom | Progress review, gate decisions | Fridays 10:00 PT |
| GitHub Projects board | Task tracking / status | Real-time |
| Docs PR reviews | Knowledge handover | 48 h SLA |

*Single Source of Truth*: this `INTEGRATION_TIMELINE.md` in repo root (branch `droid/feature-development`, later `main`).  

Document owner: **Product Droid** – updates daily.

---

## 7 · Success Criteria per Milestone

| Milestone | Metrics | Owner | Acceptance |
|-----------|---------|-------|------------|
| **Phase 3 GA** | All §4 gates pass, Pass@1 ≥20%, training stable | Eng + Product | Tag `v0.3.0` |
| **Infra MVP** | CI <10 min, coverage ≥90 %, security scan 0 Critical | Infra | Badge “CI-Green” |
| **Research Report** | Baseline table vs HRM, 2 plots | Research | PR merged |
| **Data Clean v1** | Duplicate rate <1 %, schema validated | Data | PR merged |
| **Phase 4 Kick-off** | Roadmap locked, UX API frozen | Product | Meeting notes |

---

## 8 · Timeline to Phase 4 Convergence

| Date | Workstream Events | Status |
|------|-------------------|--------|
| **Aug-05** | Integration timeline published | ✅ |
| Aug-06 | Eng PR: MBPP token layer | ⏳ |
| Aug-07 | Gate C1 (Data-Layer Merge) |  |
| Aug-08 | Gate C2 (CI/CD) |  |
| Aug-09 | Gate C3 (Train stability) & Research mid-report |  |
| Aug-10 | Gate C4 (Research) & Gate C5 (Perf benchmark) |  |
| Aug-11 | Gate C6 (Data Clean v1) |  |
| **Aug-12** | Gate C7 – **Phase 3 Final QA** |  |
| **Aug-13** | Phase 4 Kick-off meeting |  |

**Critical Path Buffer**: 1 day (Aug-12) for lag/bug fixing.

---

## 9 · Update Process

1. Owner edits file → PR → require 1 review.  
2. Merge triggers **Docs CI** → renders timeline to GitHub Pages.  
3. Slack bot posts daily diff summary to `#hrm-daily`.  

---

_This timeline governs multi-droid collaboration and ensures seamless convergence into Phase 4._  
