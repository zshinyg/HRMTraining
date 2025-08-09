# Sapient-HRM CodeGen · Minimal Evaluation Plan

- Objective: Demonstrate Sapient HRM adapted for code generation with a small, reproducible MBPP evaluation.
- Audience: Interview demo; optimized for clarity and quick iteration (CPU/MPS/CUDA).

## Scope
- Fixed 100-task MBPP dev slice for evaluation consistency.
- Two runs:
  - GPT-2 zero-shot baseline (greedy decode)
  - Sapient-HRM CodeGen (short finetune later; quick eval now)

## Commands
- Baseline (GPT-2 zero-shot):
  - TOKENIZERS_PARALLELISM=false \
    python scripts/quick_eval_gpt2.py --num-tasks 100 --max-new-tokens 128
- HRM CodeGen quick eval (uses vendored Sapient if present, else mock):
  - TOKENIZERS_PARALLELISM=false \
    python scripts/quick_eval_codegen.py --num-tasks 100 --temperature 0.0 --top-p 1.0 --top-k 0

Outputs are saved under `outputs/codegen/` as JSON summaries.

## Dataset slice
- Use `MBPPConfig(dev_mode=True, dev_samples=100, seed=42)`
- For strict reproducibility, persist sampled `task_id`s in a follow-up step.

## Future (GPU) finetune
- Short finetune of Sapient-HRM CodeGen on MBPP:
  - Spin up 1× T4/A10, install requirements, then run a small trainer (to be added) for ~1–3k steps.
  - Re-run HRM quick eval on the same dev slice; report Pass@1.

## Acceptance criteria
- Scripts run locally on CPU/MPS; reproduce the JSON summaries.
- Same fixed dev slice for both runs.
- Clear README section with commands.

## Notes
- Silence tokenizers fork warnings: set `TOKENIZERS_PARALLELISM=false`.
- To use the real Sapient HRM locally, vendor the repo under `external/sapient-hrm` (submodule) and ensure imports via `models.*`.