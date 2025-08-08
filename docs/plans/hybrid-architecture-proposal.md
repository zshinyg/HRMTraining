# HYBRID_ARCHITECTURE_PROPOSAL.md  
_HRM-CodeGen Project – Hybrid “Planner + Executor” Model_  
_Last updated: 2025-08-05_

---

## 1 · Architecture Overview & Rationale  

### 1.1 Concept  
```
Problem ► HRM-Planner ► High-Level Plan ► GPT-2-Executor ► Python Code
                                              │
                       ⇠ optional feedback ⇠──┘
```

1. **HRM-Planner (27 M params)**  
   • Generates an explicit, token-level “plan” (step-by-step reasoning, pseudo-code, function signatures).  
2. **GPT-2-Executor (117 M params)**  
   • Consumes the plan + original prompt and produces fully-formed Python code.  
3. **Feedback Loop (v2)**  
   • HRM verifies / refines GPT-2 output, optionally regenerates sub-steps.

### 1.2 Rationale  
• **Leverage strengths**: HRM’s structured reasoning + GPT-2’s fluent code synthesis.  
• **Mitigate weaknesses**: HRM’s slow decoder, GPT-2’s shallow reasoning.  
• **Parameter-efficient**: Only ~145 M total (< BERT-large) with clear modularity.  
• **Incremental adoption**: Minimal changes to current codebase; drop-in modules.

---

## 2 · Technical Implementation Design  

| Layer | Component | Key Details |
|-------|-----------|-------------|
| Inference 1 | `planner.generate_plan(prompt)` | HRM fine-tuned to emit JSON plan tokens (e.g. `{ "steps": [...] }`). |
| Transform | `plan_encoder(plan)` | Serialises plan into natural-language + special tokens (`<PLAN> … </PLAN>`). |
| Inference 2 | `executor.generate_code(prompt+plan)` | GPT-2 conditioned on concatenated prompt+plan, max_len 512. |
| Verification* | `planner.verify(code, plan)` | HRM checks structural compliance; returns score / edit ops. |

Implementation notes  
1. **Tokenizer update**: Reserve `<PLAN>`, `</PLAN>` and `<STEP>` tokens.  
2. **Training**  
   • HRM: continue-train on MBPP “solution outlines” extracted via heuristic.  
   • GPT-2: LoRA fine-tune with planner-augmented inputs (freeze 90 %).  
3. **API**  
```python
plan = planner.generate_plan(problem)
aug_prompt = f"{problem}\\n<PLAN>\\n{plan}\\n</PLAN>"
code = executor.generate_code(aug_prompt)
```
4. **Fallback-only mode** (MVP): Skip verification loop; evaluate raw code.

---

## 3 · Expected Benefits & Performance Characteristics  

| Metric | Pure GPT-2 | Pure HRM | Hybrid (Target) |
|--------|-----------|----------|-----------------|
| Pass@1 | 26 % | ≤10 %* | **≥34 %** |
| Pass@10| 42 % | ≤20 %* | **≥48 %** |
| Latency (ms, 512) | 280 | 2 000 | **350** |
| RAM (train, GB) | 24 | 16 | **26** |
| Parameters | 117 M | 64 M | **145 M** |

*early HRM baseline. Hybrid aims for +8 pp Pass@1 over GPT-2 with <1.3× latency.

---

## 4 · Integration with Current Codebase  

Directory additions  
```
hrm_codegen/
├─ planner_interface.py   # HRM wrapper
├─ plan_encoder.py        # Serialiser utilities
└─ hybrid_pipeline.py     # end-to-end orchestration
training/
└─ lora_finetune_executor.py
```

Modifications  
• `tokenization/__init__.py` – add `<PLAN>` tokens.  
• `scripts/evaluate.py` – support two-stage generation flow.  
• CI workflow – new job `hybrid_benchmark.yml`.

Compatibility  
✓ Existing HRM & GPT-2 modules untouched.  
✓ Hybrid pipeline selectable via CLI flag `--mode hybrid`.

---

## 5 · Development Timeline & Milestones  

| Date | Milestone | Owner | Deliverable |
|------|-----------|-------|-------------|
| Aug-06 | M0: Planning spec freeze | Product | This document merged |
| Aug-07 | M1: Tokenizer + plan_encoder implemented | SE | MR #12 |
| Aug-08 | M2: HRM plan generation fine-tune (100 MBPP outlines) | Research | ckpt `planner-mbpp.pt` |
| Aug-09 | M3: GPT-2 LoRA fine-tune w/ plans | Infra | ckpt `executor-hybrid.pt` |
| Aug-10 | M4: Hybrid pipeline integration test (20 samples) | SE | Pass@1 ≥30 % |
| Aug-11 | Gate C4: Full MBPP run, dashboard publish | Research | W&B run `hybrid-v0` |
| Aug-13 | M5: Verification loop prototype | Reliability | score ≥0.2 improves Pass@1 |
| Aug-14 | Gate C5: Statistical significance report | Product | ΔPass@1 ≥4 pp, p<0.05 |

---

## 6 · Success Metrics for Validation  

1. **Primary**  
   • Pass@1 ≥ 30 %  
   • Δ Pass@1 vs GPT-2 ≥ 4 pp, 95 % CI lower-bound > 0  
2. **Secondary**  
   • Latency ≤ 1.5× GPT-2 (≤ 420 ms @512)  
   • Memory ≤ 1.2× GPT-2  
   • Plan-quality BLEU ≥ 0.25 vs reference outlines  
3. **Ablation**  
   • Remove plan → Pass@1 drops ≥ 5 pp.

---

## 7 · Comparison with Pure HRM Approach  

| Criterion | Pure HRM (64 M) | Hybrid (64 M + 117 M) |
|-----------|-----------------|------------------------|
| Pass@1 (est.) | **≤10 %** (current early run) | **≥34 %** |
| Inference speed | Very slow (≈15× GPT-2) | Moderate (≈1.3× GPT-2) |
| Implementation effort | Minimal (already) | Medium (plan encoding + finetune) |
| Interpretability | High (explicit reasoning) | High (plan tokens logged) |
| Risk | Performance unlikely to surpass GPT-2 | Manageable – leverages proven GPT-2 |

Conclusion: Hybrid retains HRM’s reasoning advantage while solving speed/performance deficits.

---

## 8 · Immediate Next Steps for Prototyping  

1. **Merge this proposal (M0).**  
2. **Implement tokenizer & plan_encoder (M1).**  
3. **Generate initial HRM plans** for 100 MBPP tasks (script `export_plans.py`).  
4. **LoRA fine-tune GPT-2** on `(prompt + <PLAN>) → code` pairs (8-hour job).  
5. **Integrate `hybrid_pipeline.py`** and update evaluation harness.  
6. **Run hybrid on 20-sample dev set**, iterate plan format if needed.  
7. **Full MBPP benchmark**, publish W&B dashboard.  
8. **Decision gate**: If ΔPass@1 ≥ 4 pp, expand training to full dataset, else iterate plan schema.

---

_End of Hybrid Architecture Proposal_  
