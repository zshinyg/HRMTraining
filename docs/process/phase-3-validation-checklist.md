# PHASE_3_VALIDATION_CHECKLIST.md
_HRM-CodeGen · Definitive Guide for Phase 3 Validation_  
_Last updated: 2025-08-05_

---

## 1 · Purpose  

This checklist guarantees that **Phase 3 — HRM Adaptation & Infrastructure Convergence** meets every requirement needed to:  
1. Preserve the original Hierarchical Reasoning Model (HRM) architecture.  
2. Demonstrate technical readiness to **prove the hypothesis** that a 27 M-parameter HRM can outperform a 117 M-parameter GPT-2 baseline on MBPP code-generation benchmarks.  
3. Hand off a fully validated system into **Phase 4 – Optimisation & UX Release**.

Each item below must be _checked ✓_, evidence linked, and owner signed before Phase 3 can be declared GA.

---

## 2 · Gate-by-Gate Validation Checklist

| Gate | Date (ETA) | Objective | Step-by-Step Validation | Evidence (link / log) | Owner | Status |
|------|-----------|-----------|-------------------------|-----------------------|-------|--------|
| **C1** | Aug-07 | **Data-Layer Merge** | 1. `pytest -q` passes **70/70** unit tests.<br>2. MBPP loader returns 374/500 samples without error.<br>3. Tokeniser reproduces round-trip (text→tokens→text).<br>4. Code committed, PR merged to `develop`. | CI run # / Test log | SE Code Droid | ☐ |
| **C2** | Aug-08 | **CI/CD Pipeline Green** | 1. `.github/workflows/ci.yml` succeeds on PR.<br>2. Lint + type-check ≤ 5 min.<br>3. Coverage ≥ 90 %.<br>4. Security scan 0 Critical CVEs. | GitHub Action run | Infra Code Droid | ☐ |
| **C3** | Aug-09 | **Model Forward & 1-Epoch Train** | 1. HRM forward pass on CPU uses ≤ 12 GB RAM.<br>2. 1-epoch dev training completes with no NaN.<br>3. Checkpoint `dev-best.pt` saved + loadable.<br>4. Loss decreases ≥ 10 %. | `train.log` | SE Code Droid | ☐ |
| **C4** | Aug-10 | **Research Baseline Report** | 1. `DATA_ANALYSIS_REPORT.md` merged.<br>2. GPT-2-117 M baseline Pass@1/10 reproduced (26 %/42 %).<br>3. HRM dev checkpoint evaluated (any score accepted). | PR # / W&B run | Research Droid | ☐ |
| **C5** | Aug-11 | **Performance Benchmark** | 1. `benchmark.yml` GH Action completes.<br>2. Throughput ≥ 550 tok/s CPU; latency ≤ 300 ms.<br>3. Memory ≤ 12 GB.<br>4. Results posted to W&B dashboard. | Benchmark report | Infra Code Droid | ☐ |
| **C6** | Aug-12 | **Data-Clean v1** | 1. Duplicate rate < 1 %.<br>2. Schema validation script passes.<br>3. Cleaned `train_v1.bin`, `test_v1.bin` committed.<br>4. Re-train script accepts new splits. | Data QA log | Research Droid | ☐ |
| **C7** | Aug-13 | **Phase 3 Final QA** | 1. All gates C1-C6 status **✓**.<br>2. Pass@1 ≥ 20 %, Pass@10 ≥ 35 % on MBPP.<br>3. Flash-Attn fallback or compile success verified.<br>4. CI green on `main`.<br>5. Documentation hosted; quick-start completes in ≤ 10 min.<br>6. Risk register reviewed; no Medium+ open risks. | QA sheet / CI badge | Product Droid | ☐ |

---

## 3 · Hypothesis Validation Requirements  

| Requirement | Metric / Test | Threshold | Tool | Owner | Status |
|-------------|---------------|-----------|------|-------|--------|
| HRM outperforms GPT-2-117 M | Pass@1 | **≥ 30 %** | `scripts/evaluate.py` | SE Code Droid | ☐ |
|  | Pass@10 | **≥ 45 %** | ↑ | SE Code Droid | ☐ |
| Statistical Significance | 95 % CI for Pass@k diff excludes 0 | True | `stats.py` (bootstrap) | Research Droid | ☐ |
| Efficiency Advantage | Params | 27 M vs 117 M | model summary | Product | ✓ |
|  | Peak RAM (train) | ≤ 75 % of GPT-2 RAM | benchmark | Infra | ☐ |
|  | Latency (infer) | ≤ 80 % of GPT-2 latency | benchmark | Infra | ☐ |

---

## 4 · Technical Acceptance Criteria & Testing Procedures  

1. **Unit Tests** – Coverage ≥ 90 %, NIL xfails.  
2. **Integration Tests** – End-to-end train→evaluate pipeline on sample dataset passes.  
3. **Performance Tests** – `benchmark.yml` measures throughput & latency; thresholds in §3.  
4. **Security Tests** – Static scan (Bandit), dependency scan (Dependabot) 0 Critical issues.  
5. **Code Quality** – `ruff` lint score 0 errors; `black` formatting.  
6. **Reproducibility** – `make reproduce` recreates experiment with identical metrics ±1 %.  

---

## 5 · SE ↔ Infra Integration Validation  

| Integration Point | Check | Evidence | Status |
|-------------------|-------|----------|--------|
| Dockerfile runs `train.py` without manual steps | Container exit code 0 | ☐ |
| CI pulls HRM weights ≥ 50 MB via cache | CI log | ☐ |
| Evaluation workflow consumes SE checkpoint | W&B artifact | ☐ |
| Flash-Attn compilation script integrated | GH Action step success | ☐ |

---

## 6 · Success Metrics Tracking  

| Metric | Target | Collection Method | Dashboard |
|--------|--------|-------------------|-----------|
| Pass@1, Pass@10 | ≥ 20 % / ≥ 35 % (Gate C7) | `evaluate.py` JSON | W&B HRM-vs-GPT2 |
| Throughput tok/s | ≥ 550 CPU | `benchmark.py` | W&B Perf |
| Latency ms | ≤ 300 ms | ↑ | W&B Perf |
| RAM GB | ≤ 12 | `psutil` | W&B Perf |
| CI Duration | ≤ 10 min | GH Action | CI badge |

---

## 7 · Quality Assurance Checklist (Phase 3 Exit)  

- [ ] All gates C1-C6 ✅ in timeline.  
- [ ] All hypothesis metrics in §3 met or exceeded.  
- [ ] No open P1 or P2 GitHub issues.  
- [ ] Docs: Quick-start, API reference, CONTRIBUTING merged.  
- [ ] Code: `main` branch green; version tag `v0.3.0`.  
- [ ] Post-mortem for any Critical blocker resolved within Phase 3.  
- [ ] Stakeholder sign-off (Research, Infra, Eng leads).  

---

## 8 · Phase 4 Readiness & Handoff  

| Artifact | Required In Phase 4? | Location | Verified By |
|----------|---------------------|----------|-------------|
| `checkpoints/dev-best.pt` | Yes | `checkpoints/` | SE | ☐ |
| `configs/mbpp_opt.yaml` | Yes | `configs/` | Product | ☐ |
| CI/CD pipeline (`ci.yml`) | Yes | `.github/workflows/` | Infra | ☐ |
| Research baseline reports | Reference | `docs/research/` | Research | ☐ |
| UX API spec (`USER_EXPERIENCE_DESIGN.md`) | Contract | repo root | Product | ☐ |
| Risk register (R1-R5) closures | Review | `INTEGRATION_TIMELINE.md` | Product | ☐ |

Phase 4 kick-off meeting scheduled **Aug-13 10:00 PT** only after all checkboxes above are ticked.

---

## 9 · Risk Mitigation Validation Steps  

| Risk ID | Mitigation Done? | Evidence | Owner | Status |
|---------|------------------|----------|-------|--------|
| **R1** Gradient instability | Grad-checkpointing enabled, anomaly logs 0 | `train.log` | SE | ☐ |
| **R2** CI timeouts | Job split, cache hits ≥ 90 % | GH Action metrics | Infra | ☐ |
| **R3** Data license conflicts | Licence audit report uploaded | `docs/legal/` | Research | ☐ |
| **R4** Flash-Attn failure | CPU fallback bench passes thresholds | Benchmark run | SE/Infra | ☐ |
| **R5** Merge conflicts | Branch protection, no force pushes | GH settings screenshot | Product | ☐ |

_All risk rows must be **green** before Phase 3 GA._

---

### Final Sign-Off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Engineering Lead |  |  |  |
| Infrastructure Lead |  |  |  |
| Research Lead |  |  |  |
| Product Lead |  |  |  |

When all signatures are present and every **Status** column shows ✅, Phase 3 is formally complete and the project advances to Phase 4.  
