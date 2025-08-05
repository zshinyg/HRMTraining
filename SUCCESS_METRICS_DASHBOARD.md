# Success Metrics Dashboard – HRM vs GPT-2-117 M  
_HRM-CodeGen Hypothesis Validation_  
_Last updated: 2025-08-05_

---

## 1 · Overview & Hypothesis Objectives  
We aim to **prove that a 27 M-parameter Hierarchical Reasoning Model (HRM)** outperforms a **117 M-parameter GPT-2 baseline** on the MBPP code-generation benchmark.

Primary objectives:  
1. Demonstrate functional superiority (Pass@k accuracy).  
2. Demonstrate efficiency superiority (latency, RAM).  
3. Provide statistically significant evidence (p < 0.05) reproducible via a public dashboard.  

Success = all green thresholds in §7.

---

## 2 · Key Performance Indicators (KPIs)

| Category | Metric | HRM Target | GPT-2-117 M Baseline | Δ Must Exceed | Source |
|----------|--------|-----------|----------------------|---------------|--------|
| Accuracy | Pass@1 | **≥ 30 %** | 26 % | ≥ 4 pp | `evaluate.py` |
|          | Pass@10 | **≥ 45 %** | 42 % | ≥ 3 pp | ↑ |
| Efficiency | Params | 27 M | 117 M | — | model summary |
|            | Peak RAM (train) | ≤ 12 GB | 24 GB | ≤ 50 % | `benchmark.py` |
|            | Avg. latency (512 tok) | ≤ 150 ms | 280 ms | ≤ 54 % | ↑ |
|            | Throughput (tok/s CPU) | ≥ 1 100 | 550 | ≥ 2× | ↑ |
| Reliability | Train run failures | 0 | N/A | 0 | `reliability_monitor.py` |
| Statistical | p-value (Pass@1 diff) | < 0.05 | N/A | — | `stats.py` |

---

## 3 · Data Collection Pipeline & Flow

1. **Training / Eval Run** → emits `metrics.jsonl` per step.  
2. **`collect_metrics.py`** watches run directory, aggregates to `run_summary.json`.  
3. **GitHub Action** uploads summary to **Weights & Biases (W&B)** as run artifact.  
4. **`stats.py`** bootstraps HRM vs GPT-2 result sets, outputs `stats.json` (ci95, p-values).  
5. CI job pushes metrics + stats to W&B dashboard & stores copy in `results/`.  
6. Real-time dashboard pulls via W&B public API.

All scripts live in `scripts/metrics/`.

---

## 4 · Real-Time W&B Dashboard Design

### Panels & Layout (left → right):

1. **Run Selector**  
   • Dropdown: HRM-27M, GPT-2-117M, CodeT5-small  
2. **Accuracy Panel (Line)**  
   • Pass@1, Pass@10 vs step  
3. **Stat Summary Panel (Table)**  
   • Final Pass@k, Δ, p-value, verdict (✓/✗)  
4. **Efficiency Panel (Bar)**  
   • Latency, Throughput, RAM vs baseline  
5. **Training Curves (Line)**  
   • Loss, LR, Grad-norm  
6. **System Health (Realtime)**  
   • CPU%, Mem%, Temp, GPU util (if present)  
7. **Alert Feed**  
   • Issues from `reliability_monitor.py`

Dashboard saved as _W&B project_: **hrm-vs-gpt2**  
URL published in project README.

---

## 5 · Statistical Significance Framework

*Implemented in `stats.py`*

Algorithm:  
1. Collect binary success arrays for HRM & GPT-2 on MBPP test (n = 500).  
2. Compute Pass@k proportion `p̂`.  
3. **Bootstrap 10 000 resamples** of difference `δ = p̂_HRM – p̂_GPT2`.  
4. Calculate 95 % CI; compute two-tailed p-value.  
5. Accept superiority if **CI lower-bound > 0** _and_ **p < 0.05**.  
6. Write `stats.json`:

```json
{
  "pass_at_1": { "hrm": 0.312, "gpt2": 0.260, "delta": 0.052,
                 "ci95": [0.018, 0.089], "p_value": 0.003 },
  "verdict": "HRM Superior"
}
```

---

## 6 · Automated Reporting & Alerting

| Event | Trigger | Action |
|-------|---------|--------|
| CI Run Complete | Successful workflow | Post W&B link + stats to PR thread |
| KPI Fail | Any KPI below target | Mark PR ❌, comment with failing metrics |
| p-value ≥ 0.05 | Post-eval stats | Label run “INCONCLUSIVE”, notify `@team` |
| System Health Alert | `reliability_monitor.py` detects fault | Send Factory Blocker Alert + CI annotation |
| Gate C7 Pass | All thresholds green | Tag release `v0.3.0`, auto-generate summary PDF |

Alerts routed through GitHub comments; optional Slack webhook via secret.

---

## 7 · Success / Failure Decision Framework

| Verdict | Required Conditions (all true) |
|---------|--------------------------------|
| **SUCCESS** | Pass@1 ≥ 30 %, Pass@10 ≥ 45 % **AND** Δ_pass@1 ≥ 4 pp **AND** CI lower-bound > 0 **AND** p < 0.05 **AND** latency/RAM targets met |
| **PARTIAL** | Accuracy targets met **OR** efficiency targets met, but p ≥ 0.05 **OR** CI includes 0 |
| **FAILURE** | Any primary KPI below target **OR** p ≥ 0.1 **OR** training unstable |

Gate C7 cannot close without **SUCCESS**.

---

## 8 · Integration Points with Infrastructure Monitoring

1. **GitHub Actions**  
   • `benchmark.yml` uploads metrics to W&B, posts summary comment.  
2. **Docker Images**  
   • Entrypoint runs `reliability_monitor.py` in side-car.  
3. **Monitoring Workflow (`monitoring.yml`)**  
   • Nightly scheduled job pulls latest `stats.json`; opens GitHub Issue if thresholds violated.  
4. **Grafana/Prometheus (optional)**  
   • Export system metrics via `scripts/metrics/prom_exporter.py`.  
5. **CI Badge**  
   • `README.md` displays Pass@1 benchmark badge (green ≥ 30 %).  

---

## 9 · Implementation Checklist & Timeline

| # | Task | Owner | ETA |
|---|------|-------|-----|
| 1 | Implement `collect_metrics.py`, `stats.py` | Infra + Research | Aug-06 |
| 2 | Configure W&B project & dashboard panels | Infra | Aug-06 |
| 3 | Add metrics upload to `benchmark.yml` | Infra | Aug-07 |
| 4 | Write p-value calc unit tests | Research | Aug-07 |
| 5 | Create alerting GitHub Action | Infra | Aug-08 |
| 6 | Validate full pipeline on dev checkpoint | SE | Aug-09 |
| 7 | Publish public dashboard link | Product | Aug-10 |
| 8 | Gate C7 final evaluation | SE + Infra | Aug-13 |

_All steps must complete before Phase 4 kick-off._

---

_End of Success Metrics Dashboard specification_  
