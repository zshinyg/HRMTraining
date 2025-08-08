# HRM → CodeGen Adaptation Plan
_Last updated: 2025-08-05_

---

## 1. Strategic Decision

We are pivoting from our in-house HRM implementation (currently blocked by gradient errors) to **adapt the proven open-source HRM** released by Sapient Inc. ([github.com/sapientinc/HRM](https://github.com/sapientinc/HRM)).  
Reasons:

* **Working baseline** – 6.4 k⭐; publicly verified on ARC, Sudoku, Maze.  
* **Time-to-market** – days instead of weeks of debugging.  
* **Risk reduction** – leverage a battle-tested architecture; focus engineering on domain adaptation (code generation).  
* **Learning value** – study their fixes for issues we encountered (in-place ops, training stability).  
* **Strategic focus** – invest effort where we differentiate (code-specific tokens, Pass@k evaluation) rather than reinventing core HRM logic.

---

## 2. Adaptation Phases, Tasks & Success Criteria

| Phase | Goal | Key Tasks | Success Criteria |
|-------|------|-----------|------------------|
| **1. Foundation Analysis** (Day 1) | Prove original HRM works on puzzles | • Clone repo • Install deps • Run their ARC/Sudoku demo | All demo scripts run w/o errors on local GPU/CPU |
| **2. Strategic Mapping** (Day 1-2) | Define what to keep / change | • Document component map • Identify tokenization & dataset swaps • Write adaptation spec | Reviewed & approved adaptation spec |
| **3. Incremental Adaptation** (Day 2-4) | Swap data & outputs step-by-step | 3.1 Data layer: add MBPP loader & GPT-2 tokenizer<br>3.2 Model output: enable autoregressive code generation<br>3.3 Eval layer: integrate Pass@k executor | a) Model trains on small MBPP subset<br>b) Generates syntactically valid code |
| **4. Validation & Tuning** (Day 4-7) | Achieve baseline Pass@k | • Full MBPP train run (dev config)<br>• Hyper-param search (lr, ctx len)<br>• Compare vs transformer baseline | Pass@1 ≥ 5 % on MBPP dev |
| **5. Hardening & Docs** (Week 2) | Production readiness | • Refactor configs • CI tests • Write user docs | Reproducible run from README in <15 min |

---

## 3. Architecture Mapping

| Component | Sapient HRM (Puzzle) | CodeGen Adaptation |
|-----------|----------------------|--------------------|
| **Input Representation** | 2-D grid / image tensor | Token IDs (GPT-2/BPE) |
| **Positional Encoding** | Learned 2-D | 1-D learned/rotary |
| **High-Level Module** | Global abstract planning | High-level program sketch |
| **Low-Level Module** | Cell-wise reasoning | Token-level generation |
| **Loss** | Cross-entropy over actions | Cross-entropy over next-token |
| **Evaluation** | Exact puzzle accuracy | Pass@k (safe exec) |
| **Dataset loaders** | ARC/Sudoku builders | MBPP/HumanEval loaders |
| **Inference** | Puzzle solver loop | Autoregressive generate() |

---

## 4. Risk Mitigation Strategies

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| HRM fails to generalise to code | Med | High | Start with small code subset; add code-specific positional encodings if needed |
| Tokenisation mismatch (2-D vs 1-D) | High | Med | Replace 2-D pos enc with standard 1-D; run ablations |
| CUDA extension build issues | Med | Med | Provide CPU fallback; Dockerfile with correct CUDA/Flash-Attention versions |
| Performance regression | Med | Med | Maintain transformer baseline for A/B comparison |
| Licensing / compliance | Low | High | HRM is Apache-2.0 – OK; include NOTICE |

---

## 5. Success Metrics

1. **Technical**  
   • Pass@1 / Pass@k on MBPP dev & test  
   • Training stability (no NaNs, ≤1% grad overflow)  
2. **Product**  
   • Time from git clone → first valid model ≤ 2 h  
   • Reproducible pipeline documented  
3. **Team**  
   • <2 blocking bugs per sprint after adaptation  
   • 80 %+ engineer satisfaction in retro

---

## 6. Timeline & Resource Allocation

| Week | Activities | Lead | Support |
|------|------------|------|---------|
| **Wk 1** | Phases 1-3 | 2 ML engineers | 1 product (coord) |
| **Wk 2** | Phase 4 tuning & Phase 5 hardening | 2 ML eng • 1 MLOps | Product, QA |
| **Wk 3+** | Perf optimisation & new datasets | 1-2 ML eng | – |

Budget: reuse existing GPU quota (A100 × 1 for dev, 8× for full). No new SaaS costs; W&B free tier.

---

## 7. Technical Considerations

* **Tokeniser** – GPT-2 BPE (same as MBPP baseline)  
* **Sequence Length** – start 256, scale to 512/1024; adjust mem usage  
* **FlashAttention** – required; ensure CUDA ≥ 11.8 or build FA-2/3  
* **Mixed Precision** – FP16 with GradScaler; verify HRM supports AMP  
* **Checkpoints** – standard PyTorch `state_dict`; maintain conversion script  
* **Config System** – integrate our YAML configs into HRM’s Hydra schema  
* **Evaluation harness** – reuse `scripts/evaluate.py` Pass@k executor; wrap HRM `model.generate`  
* **CI** – GitHub Actions runner: lint, unit tests, 1-epoch smoke train

---

## 8. References

* **Source HRM Repository:** https://github.com/sapientinc/HRM  
* **Paper:** Wang et al., “Hierarchical Reasoning Model”, arXiv:2506.21734  
* **Our CodeGen Baseline Repo:** (current repo)  
* **Datasets:**  
  * MBPP – https://huggingface.co/datasets/mbpp  
  * HumanEval – https://huggingface.co/datasets/openai_humaneval  

---

_This document is the single source of truth for the HRM→CodeGen adaptation project. All engineering tasks must reference the phases, tasks, and success criteria defined here._  
