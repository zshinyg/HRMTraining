### HRM-CodeGen Architecture Overview

This project adapts Sapient HRM to causal code generation.

- High-level planner (slow timescale) forms program strategy
- Low-level executor (fast timescale) emits token-level code
- Causal attention enforced throughout for autoregressive decoding
- MBPP data pipeline with GPT-2 tokenizer

Key modules:
- `hrm_codegen/model.py`: wrapper around Sapient HRM with codegen inputs/outputs
- `hrm/model.py`: in-house HRM (kept for reference)
- `scripts/train.py`, `scripts/evaluate.py`: training/evaluation
- `configs/hrm/*.yaml`: model/training configs

Generation:
- Greedy or sampling with temperature/top-k/p
- No KV-cache yet (Phase 4 item); sequence lengths are moderate

