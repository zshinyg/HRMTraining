# Phase 4 Roadmap  
_Operational plan for “HRM-CodeGen v0.4 – Optimisation & UX Release”_

---

## 1 · Strategic Goals & Objectives  

| # | Goal | Rationale | KPI |
|---|------|-----------|-----|
| G1 | **Elevate Pass@k accuracy** beyond MVP to match/beat GPT-2-117 M on MBPP | Competitive parity | Pass@1 ≥ 30 %, Pass@10 ≥ 45 % |
| G2 | **Performance & efficiency** – 2× training speed, ≤ 12 GB RAM | Lower infra cost | Tokens / sec, peak RAM |
| G3 | **Developer UX** – simple API + CLI + docs | Adoption | SUS ≥ 80 / 100 |
| G4 | **Extensibility** – plug-in datasets & sampling strategies | Research velocity | “dataset add” in < 10 mins |
| G5 | **Production readiness** – CI/CD, monitoring, security | Reliability | CI pass rate 100 %, CVE=0 |

---

## 2 · Feature Prioritisation Matrix  

| Feature | Impact | Effort | Priority |
|---------|--------|--------|----------|
| Causal-RoPE attention optimisation | High | Med | P0 |
| Gradient Checkpointing | High | Med | P0 |
| Mixed-Precision (bf16/fp16) | High | Med | P0 |
| API: `generate_code()` SDK | High | Low | P0 |
| CLI: `hrm-codegen train/eval` | Med | Low | P1 |
| Config-driven dataset loader | Med | Med | P1 |
| Beam-search & temperature scheduling | Med | Med | P1 |
| Web demo (Streamlit) | Low | Med | P2 |
| Hugging Face model card & upload | Med | Low | P2 |
| Documentation site (MkDocs) | Med | Med | P2 |

_Impact = expected KPI lift; Effort = eng-days; Priority: P0 (blocker), P1 (next), P2 (nice-to-have)._

---

## 3 · Performance Optimisation Targets & Strategies  

| Target | Baseline (Phase 3) | Phase 4 Goal | Strategy |
|--------|--------------------|--------------|----------|
| Training throughput | 550 tok/s (CPU) | ≥ 1100 tok/s | Gradient checkpointing + fused ops |
| GPU throughput (A100) | 32 k tok/s | ≥ 60 k tok/s | Flash-Attn v3, bf16 |
| Peak RAM | 16 GB | ≤ 12 GB | Activation recompute, dataset streaming |
| Eval latency (512 tok) | 280 ms | ≤ 150 ms | KV-cache, pytorch.compile |

---

## 4 · User Experience & API Design  

* **Python SDK**
  ```python
  from hrm_codegen import HRMCodeGen
  model = HRMCodeGen.from_pretrained("hrm-codegen-base")
  code = model.generate_code(prompt, max_tokens=128, temperature=0.7)
  ```
* **CLI**
  ```
  hrm-codegen train  --config configs/mbpp_opt.yaml
  hrm-codegen eval   --ckpt checkpoints/best.pt --k 10
  hrm-codegen gen    --prompt "Write a function..." --max 128
  ```
* **Docs**: Quick-start, API reference, tutorials.  
* **Web Demo**: Streamlit UI with prompt, generated code, test-run panel.

---

## 5 · Timeline & Milestones  

| Week | Milestone | Dependencies |
|------|-----------|--------------|
| W0   | **Phase 3 GA** – MVP passes success criteria | Phase 3 |
| W1   | Causal-RoPE + checkpointing merged (P0) | none |
| W1   | Mixed-precision training stable (P0) | CUDA/Flash-Attn |
| W2   | API/CLI beta released (P0) | model interface |
| W2   | Performance benchmarks hit 80 % goals | optimisation tasks |
| W3   | Beam search & temp scheduling (P1) | API stable |
| W3   | Dataset loader plug-in system (P1) | CLI |
| W4   | Documentation site live (P2) | API finalised |
| W4   | Stretch goals + final QA | all tasks |
| W4   | **Phase 4 Release Candidate** | tests, docs |

Slack stand-ups daily; demo every Friday.

---

## 6 · Phase 4 Success Metrics  

1. **Accuracy**: Pass@1 ≥ 30 %, Pass@10 ≥ 45 % on MBPP test  
2. **Perf**: CPU throughput ≥ 1100 tok/s; GPU ≥ 60 k tok/s  
3. **UX**: SUS ≥ 80 from 5 external testers  
4. **Reliability**: CI/CD green 3 consecutive days, test coverage ≥ 90 %  
5. **Docs**: Quick-start lead-time ≤ 10 min (new user → first generation)

---

## 7 · Resource Allocation & Risk Assessment  

| Area | Lead | Dev-Days | Risks | Mitigation |
|------|------|---------|-------|------------|
| Optimisation P0 | Eng-Lead | 24 | Kernel mismatch | fallback to eager |
| API/CLI & Docs  | Product Eng | 12 | Scope creep | freeze spec W1 |
| Dataset plug-ins | Data Eng | 8 | Format variance | schema validation |
| Web demo         | Frontend | 6 | Security | sandbox exec |

Contingency: 15 % buffer (7 days) for unknowns.

---

## 8 · Integration Plan for New Features  

1. **Branching strategy**:  
   * `main` – stable  
   * `feat/perf-*`, `feat/api-cli`, `feat/dataset-plugin`, merge via PR + CI.  
2. **CI gates**: Formatting (`black`), lint (`ruff`), tests, benchmarks.  
3. **Backward compatibility**: Deprecated flags supported for one release.  
4. **Release process**:  
   * Tag `v0.4.0-rc` → internal QA → `v0.4.0` public.  
   * Publish wheel & HF checkpoint.  
5. **Monitoring**: Prometheus exporter for throughput/latency, Sentry for errors.

---

_Approved by Product on 2025-08-05_  
