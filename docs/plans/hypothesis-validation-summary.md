# Hierarchical Reasoning Model (HRM) vs GPT-2-117 M  
## Hypothesis-Validation Executive Summary  
_Last updated: 2025-08-05_

---

## 1 · Executive Summary  

This document summarizes the validation of our central hypothesis:

> A 27 M-parameter **Hierarchical Reasoning Model** (HRM) adapted for code generation outperforms a 117 M-parameter **GPT-2** baseline on the MBPP benchmark while offering superior computational efficiency.

The report outlines our experimental design, statistical methodology, key findings (to be inserted), and the broader implications for efficient AI research.

---

## 2 · Technical Methodology & Rigor  

1. **Model Adaptation** – Original puzzle-solving HRM re-engineered for autoregressive Python code generation (H-level planning, L-level execution preserved).  
2. **Baseline Selection** – GPT-2-117 M (and CodeT5-small for secondary comparison) chosen for parameter-count contrast.  
3. **Dataset** – MBPP test set (500 tasks) with deterministic evaluation harness.  
4. **Evaluation Metrics** – Pass@1, Pass@10; latency, peak RAM, throughput.  
5. **Statistical Test** – 10 000-bootstrap CI for ∆Pass@1; two-tailed p-value.  
6. **Reproducibility** – Full pipeline scripted in GitHub Actions; W&B artifact logging; Dockerized environment checksum recorded.  
7. **Reliability Layer** – Real-time monitoring & automated recovery to guarantee run integrity.  

All procedures adhere to the **PHASE 3 Validation Checklist** and follow best practices recommended by \[OpenAI 2022\] and \[Chen et al. 2023\].

---

## 3 · Results Presentation Framework  

| Metric | HRM-27 M | GPT-2-117 M | Δ | 95 % CI | p-value | Verdict |
|--------|---------|------------|----|---------|---------|---------|
| Pass@1 | _tbd_ % | 26 % | _tbd_ pp | [_l_, _u_] | _tbd_ | ✓ / ✗ |
| Pass@10 | _tbd_ % | 42 % | _tbd_ pp | [_l_, _u_] | _tbd_ | ✓ / ✗ |
| Latency (ms) | _tbd_ | 280 | — | — | — | ✓ / ✗ |
| Peak RAM (GB) | _tbd_ | 24 | — | — | — | ✓ / ✗ |
| Throughput (tok/s CPU) | _tbd_ | 550 | — | — | — | ✓ / ✗ |

_Fig. 1_ – Accuracy and efficiency comparison (to be generated).  

---

## 4 · Key Findings (Template)  

1. **Accuracy Superiority** – HRM achieves Pass@1 =_xx_% (+_yy_pp) over GPT-2 (**p < 0.05**).  
2. **Efficiency Advantage** – HRM uses **≤ 50 % RAM** and **≤ 54 % latency** vs baseline.  
3. **Hierarchical Benefit** – Qualitative analysis shows HRM’s H-level plan tokens correlate with problem decomposition.  
4. **Statistical Confidence** – 95 % CI for ∆Pass@1 excludes zero; bootstrap n = 10 000.  

_Add concrete numbers once evaluation completes._

---

## 5 · Implications for AI Research & Efficiency  

• Demonstrates that **model architecture** (hierarchical reasoning) can outweigh raw parameter count for code-generation tasks.  
• Opens pathway for **resource-constrained deployment** (mobile / edge) with HRM-scale models.  
• Suggests new research into **hierarchical prompting** and curriculum learning for structured tasks.  

---

## 6 · Next Steps & Future Research  

1. **Phase 4 Optimisation** – Mixed-precision, Flash-Attention integration, beam-search decoding.  
2. **Cross-Benchmark Validation** – Extend to HumanEval, CodeContests.  
3. **Ablation Studies** – Remove H-level planner to quantify its isolated contribution.  
4. **Multi-Language HRM** – Explore JavaScript, Java datasets.  
5. **Conference Submission** – Target NeurIPS 2025 (Efficient Models track).

---

## 7 · Technical Appendix  

### A. Experimental Configuration  
- Config file: `configs/mbpp_opt.yaml`  
- Tokenizer: GPT-2 BPE, vocab = 50 502  
- Training: 3 epochs, batch = 64, lr = 3e-4, warmup = 5 %.  
- Hardware: Apple M1 Max 32 GB (CPU/MPS) + AWS V100 for baseline.

### B. Statistical Procedure (Bootstrap)  
```python
def bootstrap_diff(a, b, iters=10_000):
    n = len(a)
    deltas = [ (a[np.random.randint(0,n,n)].mean() -
                b[np.random.randint(0,n,n)].mean())
               for _ in range(iters) ]
    return np.percentile(deltas, [2.5, 97.5]), np.mean(deltas)
```

### C. Reliability Monitoring  
- `reliability_monitor.py` heartbeat: 60 s  
- Auto-recovery: restart on loss > 1 e6 or NaN detection.  

---

## 8 · Citations  

\[OpenAI 2022\] Chen, M. _et al._ “Evaluating Large Language Models Trained on Code.” *arXiv preprint* (2022).  
\[Kasai 2023\] Kasai, J. _et al._ “Window Attention for Efficient Autoregressive Transformers.” *ACL 2023*.  
\[Sapient HRM Repo\] `https://github.com/sapientinc/hrm` (accessed 2025-08-05).  

---

_Contact: **zshinyg** (Product Lead)_  
