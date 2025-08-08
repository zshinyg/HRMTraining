---
status: active
owner: engineering
summary: Technical specification of HRM-CodeGen architecture and interfaces
last_reviewed: 2025-08-08
---

# TECHNICAL_SPECIFICATION.md  
_HRM → Code Generation – Phase 2 (Strategic Mapping)_  
_Last updated: 2025-08-05_

---

## 1. Executive Summary
This document defines **how** the Sapient “Hierarchical Reasoning Model (HRM)” will be adapted from 2-D puzzle solving to autoregressive code generation on the MBPP dataset.

Phase 2 outputs:
* Component–by–component mapping between puzzle and code domains  
* File-level change list and implementation checklist for Phase 3  
* Risk register with mitigations  
* Integration plan to merge the upstream HRM with our existing project tree  
* Timeline, dependencies, and concrete success criteria

---

## 2. Detailed Component Mapping

| Layer / Concern | Sapient HRM (Puzzle) | Target HRM (Code) | Required Changes |
|-----------------|----------------------|-------------------|------------------|
| **Input** | 2-D grid indexed (`inputs`, `puzzle_identifiers`) | 1-D GPT-2/BPE token ids (`input_ids`) | Replace embedding path, drop puzzle-emb id branch |
| **Positional Encoding** | Learned 2-D or Rotary 2-D | 1-D Rotary (RoPE) | keep RoPE class but feed 1-D cos/sin; remove `puzzle_emb_len` offset |
| **Attention** | Bidirectional (`causal=False`) | Causal (`causal=True`) | Flip flag in `models/Attention`, ensure KV masking |
| **High-/Low-Level Modules** | Same | Same | No structural edits |
| **Loss** | Cross-entropy over puzzle actions + Q-head RL | Cross-entropy over next token (LM) | Remove Q-head & ACT halt schedule during Phase 3, or gate with config |
| **Evaluation** | Puzzle accuracy | Pass@k (safe exec) | Wire `scripts/evaluate.py` to `model.generate()` |
| **Datasets** | ARC/Sudoku loaders | MBPP/HumanEval | Add `datasets/mbpp_loader.py` |
| **Inference** | Multi-step solver loop | Autoregressive sampling | Reuse `generate()`; implement greedy/temperature sampling |

---

## 3. File-by-File Adaptation Specification

### 3.1 Files to **modify** inside `external/sapient-hrm`

| File | Change Summary | Owner | Complexity |
|------|----------------|-------|------------|
| `models/hrm/hrm_act_v1.py` | a) Replace `_input_embeddings` to accept `input_ids` only  ↦ remove `puzzle_identifiers` branch  b) Make `Attention` calls with `causal=True`  c) Drop Q-head or guard with `if config.enable_q_head` | ML eng 1 | Med |
| `models/layers.py` (Attention class) | Expose `causal` arg; when `True`, apply triangular mask | ML eng 1 | Low |
| `dataset/puzzle_dataset.py` | Deprecate; leave stub; add warning | ML eng 2 | Low |
| `config/*.yaml` | New schema fields: `task: codegen`, `tokenizer_name: gpt2`, `seq_len` | ML eng 2 | Low |

### 3.2 **New** files to create in project root

| File | Purpose |
|------|---------|
| `datasets/mbpp_loader.py` | PyTorch `Dataset` that yields `{input_ids, labels}` with right-shifted targets |
| `tokenization/__init__.py` | Wrap HuggingFace tokenizer with caching & BOS/EOS handling |
| `configs/codegen_base.yaml` | Reference config merging Sapient defaults + our overrides |
| `scripts/train_codegen.py` | Thin launcher using our Hydra/accelerate harness |
| `tests/test_codegen_smoke.py` | 1-epoch CPU smoke test in CI |

### 3.3 Files to **interface** in our infrastructure

| File | Action |
|------|--------|
| `scripts/evaluate.py` | Accept `--model-type hrm` to route generate; rely on `model.generate` after causal change |
| `configs/` | Import Sapient HRM defaults and overlay code-specific params |

---

## 4. Technical Implementation Details

1. **Tokenizer**  
   ```
   from transformers import AutoTokenizer  
   tok = AutoTokenizer.from_pretrained("gpt2", add_bos_token=True)  
   ```
   Save vocab JSON to `checkpoints/tokenizer.json`; freeze during training.

2. **Embedding Path**  
   ```python
   # hrm_act_v1.py  (~line 70)
   self.embed_tokens = CastedEmbedding(config.vocab_size, config.hidden_size, ...)
   # remove puzzle_emb*
   ```

3. **Causal Mask**  
   In `Attention.forward`, when `causal` flag true:  
   ```python
   mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), 1).bool()
   scores = scores.masked_fill(mask, -inf)
   ```

4. **Shifted Targets**  
   MBPP loader should output:  
   * `input_ids`: `[BOS] <prompt> <code>`  
   * `labels`: same shifted left, ignore-index = −100 for prompt tokens.

5. **Training Loop**  
   Utilize existing `accelerate` harness; criterion = `nn.CrossEntropyLoss(ignore_index=-100)`.  
   Gradient accumulation: 1 → 4 for seq_len 512 on A100.

6. **Generation**  
   Implement in `hrm_act_v1.generate` (new method): iterative decode until `EOS` or max_len 256.

7. **Checkpointing**  
   Maintain Sapient `state_dict`; add tokenizer files alongside.

---

## 5. Risk Assessment & Mitigation

| Risk | Prob. | Impact | Mitigation |
|------|-------|--------|------------|
| Memory blow-up with causal mask | Med | Med | FlashAttention-2 + fp16; seq_len 512 cap |
| Q-head remnants cause shape errors | High | High | Feature-flag `enable_q_head=False` in config |
| Tokeniser mismatch (GPT-2 vs MBPP reference) | Med | Med | Freeze vocab; add smoke test comparing hashes |
| Licensing drift | Low | High | Retain Apache-2.0 headers in modified files |
| CI timeout | Med | Low | Limit smoke test to 10 steps, CPU |

---

## 6. Integration Plan

1. **Directory Layout**  
   ```
   HRMTraining-Clean/  
     external/sapient-hrm/  (git-subtree; pinned commit)  
     datasets/mbpp_loader.py  
     tokenization/  
     configs/codegen_base.yaml  
   ```

2. **Config Merge Strategy**  
   Sapient uses plain `json`; we standardise on YAML via Hydra.  
   • Write `configs/sapient_default.yaml` auto-generated from upstream JSON  
   • Overlay ours: `hydra.merge_with` order: default < codegen_base < CLI overrides.

3. **CI**  
   Update GitHub Actions:  
   * `python -m pytest tests/test_codegen_smoke.py`  
   * `scripts/lint.sh` to run `ruff`.

4. **Data Pipeline**  
   MBPP already in `data/mbpp/`; loader consumes `.jsonl` produced by Phase 1 scripts.

5. **Sub-module/Synchronisation**  
   Use `git subtree pull` for Sapient updates; diff review checklist ensures causal flag not lost.

---

## 7. Implementation Timeline & Dependencies

| Day | Task | Owner | Blockers |
|-----|------|-------|----------|
| D0 | Approve spec; freeze scope | PM | – |
| D1 | Add tokenizer + MBPP loader | ML eng 2 | HF transformers |
| D2 | Modify `hrm_act_v1.py`, Attention mask | ML eng 1 | |
| D3 | Add generation method & train script | ML eng 1 | previous |
| D4 | Smoke train on 1k samples; fix bugs | ML eng 1-2 | CUDA |
| D5 | Integrate evaluate.py Pass@k | ML eng 2 | |
| D6 | CI green; doc update | ML eng 2 | |
| D7 | Phase 3 hand-off review | Team | |

External deps:  
* PyTorch ≥ 2.1, CUDA 11.8  
* FlashAttention-2  
* transformers v4.42

---

## 8. Success Criteria & Validation Steps

1. **Build/Run**  
   • `python scripts/train_codegen.py --config configs/codegen_base.yaml --dry-run` finishes without error.

2. **Functionality**  
   • After 1 epoch on 10 k MBPP samples, loss decreases >10 %.

3. **Evaluation**  
   • `python scripts/evaluate.py --ckpt checkpoints/dev_step.pt --k 1` produces Pass@1 > 2 % (dummy target).

4. **CI**  
   • GitHub Actions passes in < 15 min.

5. **Quality**  
   • All modified files pass `ruff` & `pyright`.

---

## 9. References

* Sapient HRM commit `af3e8b1` (2025-07-31)  
* MBPP dataset v1.1 (`data/mbpp/mbpp.jsonl`)  
* Our evaluation harness `scripts/evaluate.py`

_This specification is binding for Phase 3 implementation. Any deviations require a change request in the project tracker._
