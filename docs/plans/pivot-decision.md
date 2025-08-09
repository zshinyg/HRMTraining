# Decision Record: Pivot to Adapting Open-Source HRM  
_Date: 2025-08-05_

## 1. Current State
* Internal repo scaffolding complete (configs, data pipeline, evaluation harness, training script).
* Unit test suite passes; dummy forward pass works.
* **Blocking issue:** in-place-operation gradient error stops training; no model can be trained/evaluated.
* Time already spent debugging with no clear fix.

## 2. Discovery
* Identified actively-maintained open-source **Hierarchical Reasoning Model (HRM)** by Sapient Inc.  
  * 6.4 k⭐, Apache-2.0, proven on ARC/Sudoku/Maze.
  * Runs end-to-end without the gradient issues we face.

## 3. Strategic Decision
**Pivot** from continuing to debug our in-house HRM implementation → **adapt** the proven Sapient HRM for code-generation tasks (MBPP, HumanEval).

## 4. Rationale
* **Risk reduction:** adopt battle-tested architecture instead of weeks of low-value debugging.  
* **Time-to-market:** working baseline in days, not weeks.  
* **Focus on differentiation:** invest effort in code-specific tokenisation, Pass@k evaluation, and dataset work.  
* **Learning leverage:** reuse their fixes for training stability (FlashAttention, AMP, halt cycles).

## 5. Assets We Preserve
* Project structure & CI config.
* YAML configuration framework.
* MBPP data-conversion scripts & Pass@k evaluation harness.
* Testing utilities (`test_setup.py`) for smoke checks.
* Documentation and product roadmap.

## 6. Next Steps
1. Clone Sapient HRM into `external/hrm_open_source/` and verify their demos run locally.
2. Draft adaptation spec (see `ADAPTATION_PLAN.md`) mapping puzzle inputs → code tokens.
3. Replace dataset layer with MBPP loader and GPT-2 tokenizer.
4. Integrate our evaluation harness; run small-scale training to confirm gradients & loss decrease.
5. Deprecate old broken model code once new pipeline reproduces baseline results.
