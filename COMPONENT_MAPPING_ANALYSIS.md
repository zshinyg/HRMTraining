# COMPONENT_MAPPING_ANALYSIS.md  
_HRM → Code Generation – Detailed Component Mapping_  
_Last updated: 2025-08-05_

---

## Table of Contents
1. Input Layer  
2. Attention Mechanism  
3. Output Layer  
4. Training Objective  
5. Evaluation Harness  

Each section contains:  
• Current implementation (Sapient HRM)  
• Target implementation (CodeGen)  
• Code changes (file & line refs)  
• Technical challenges & solutions  
• Validation steps  

---

## 1. Input Layer

### 1.1 Current Implementation (Puzzle Domain)
*File:* `external/sapient-hrm/models/hrm/hrm_act_v1.py`  
*Lines:* 64-93 (`_input_embeddings`)  

```python
def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
    embedding = self.embed_tokens(input.to(torch.int32))
    if self.config.puzzle_emb_ndim > 0:
        puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
        ...
    if self.config.pos_encodings == "learned":
        embedding = 0.7071 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))
    return self.embed_scale * embedding
```

• Accepts *two* tensors: token indices (`input`) **and** puzzle-ID vector.  
• Concatenates zero-initialised puzzle embedding rows in front of the token sequence (adds `puzzle_emb_len` tokens).  
• Output embedding length = `seq_len + puzzle_emb_len`.

### 1.2 Target Implementation (Code Generation)
• Accept *single* tensor: `input_ids` (BPE tokens).  
• No puzzle-ID branch.  
• Keep learned/rotary 1-D positions, **no length shift**.  
• Resulting embedding length = `seq_len`.

### 1.3 Required Code Changes
| File | Line | Change |
|------|------|--------|
| `hrm_act_v1.py` | 60-95 | Delete `puzzle_identifiers` argument & branch.<br>`self.puzzle_emb_len = 0` set unconditionally.<br>Rename method param `input_ids`. |
| `hrm_act_v1.py` | 230 (forward) | Remove every reference to `puzzle_emb_len` in slicing (`[:, self.puzzle_emb_len:]`). |
| Any caller | – | Batch dictionary keys `inputs` → `input_ids`; drop `puzzle_identifiers`. |

**Before**

```python
input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
```

**After**

```python
input_embeddings = self._input_embeddings(batch["input_ids"])
```

### 1.4 Challenges & Solutions
1. *Hidden offset removal* – many shape expressions include `self.puzzle_emb_len`; search & set to `0`.  
2. *Backward compatibility* – introduce new config flag `task == "codegen"` to switch code path; default remains puzzle.  
3. *Tokenizer-vocab size* – GPT-2 vocab (50257) larger than puzzle vocab (~400); embed table resize required.

### 1.5 Validation
1. Unit test: pass dummy `input_ids` of shape `[2,16]` → output tensor shape `[2,16,hidden]`.  
2. Assert no parameter named `puzzle_emb*` when `task=="codegen"`.  
3. Compare embedding variance before/after (should change only due to vocab size scaling).

---

## 2. Attention Mechanism

### 2.1 Current Implementation
*File:* `external/sapient-hrm/models/layers.py`  
*Lines:* 120-210 (`class Attention`)  
• Instantiated in HRM blocks with `causal=False`.  
• Computes full QK^T without mask → bidirectional context.

### 2.2 Target Implementation
• `causal=True` to enforce autoregressive constraint (upper-triangular mask).  
• Keep FlashAttention fast-path if available.

### 2.3 Code Changes
| File | Line | Change |
|------|------|--------|
| `layers.py` | 136 | Add `self.causal` flag from ctor. |
| `layers.py` | 170 | Insert mask logic:  

```python
if self.causal:
    attn_mask = torch.triu(torch.ones(seq, seq, device=q.device), 1).bool()
    scores.masked_fill_(attn_mask, float("-inf"))
``` |
| `hrm_act_v1.py` | 41 | Instantiate with `causal=config.causal` where `config.causal = task=="codegen"`. |

### 2.4 Challenges & Solutions
1. *Performance* – tri-mask O(n²). Mitigation: if `flash_attn_2` available, pass `is_causal=True` flag.  
2. *Gradient correctness* – verify no accidental leakage by sampling logits for position *i* should not depend on tokens `>i`.  

### 2.5 Validation
1. Unit test: feed random sequence, compare logits difference when future token flipped – should be zero.  
2. Generate trivial sequence “A B C” with greedy decode; ensure only past tokens attend.  
3. Benchmark forward time pre/post; expect ≤10 % slowdown with FlashAttention.

---

## 3. Output Layer

### 3.1 Current Implementation
*File:* `hrm_act_v1.py`  
*Lines:* 46-58 (I/O definitions)  

```python
self.lm_head = CastedLinear(hidden, vocab_size, bias=False)
self.q_head  = CastedLinear(hidden, 2, bias=True)
...
output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
q_logits = self.q_head(z_H[:, 0])
```

• Outputs per-token logits **plus** a per-sequence Q-head used for adaptive computation time (halting).

### 3.2 Target Implementation
• Keep `lm_head` unchanged.  
• **Disable** Q-head by default. Optionally retain behind `enable_q_head` flag for research.  
• Remove slice `[:, self.puzzle_emb_len:]`.

### 3.3 Code Changes
| File | Line | Change |
|------|------|--------|
| `hrm_act_v1.py` | 48-49 | Wrap Q-head init in `if config.enable_q_head`. |
| `hrm_act_v1.py` | 173 | Set `output = self.lm_head(z_H)` (no slice). |
| Calling code | – | Training loss no longer expects Q-learning values. |

### 3.4 Challenges & Solutions
1. *State dict compatibility* – historical checkpoints will contain `q_head.*`; mark them `strict=False` during load.  
2. *Parameter count shift* – ~0.5 M params reduction; reflect in logs.

### 3.5 Validation
1. Forward pass returns logits shape `[B,S,vocab]`.  
2. Try loading old puzzle checkpoint with `strict=False`; ensure warning but no crash.  

---

## 4. Training Objective

### 4.1 Current Implementation
*File:* `external/sapient-hrm/losses.py`  
`PuzzleCrossEntropyLoss` mixes action-loss with Q-learning loss.

### 4.2 Target Implementation
*Standard causal language model loss*:  

```python
criterion = nn.CrossEntropyLoss(ignore_index=-100)
loss = criterion(logits.view(-1, vocab), labels.view(-1))
```

### 4.3 Code Changes
1. New file `losses_codegen.py` with above logic.  
2. Training script (`scripts/train_codegen.py`) selects loss by `task`.  
3. Remove references to `halt_loss`, `baseline` etc.

### 4.4 Challenges & Solutions
* Mixed precision overflow – enable GradScaler.  
* Token ignore regions – prompt portion masked with `-100`.

### 4.5 Validation
• Overfit on 8 MBPP samples till train loss → 0.  
• Verify gradients finite (`torch.isfinite`) every 100 steps.

---

## 5. Evaluation Harness

### 5.1 Current Implementation
*File:* `scripts/evaluate.py` (ours)  
Supposes model implements `generate()` already for baseline transformers.

### 5.2 Target Implementation
• Add HRM adapter:

```python
if args.model_type == "hrm":
    from external.sapient_hrm.models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
    model.generate = functools.partial(greedy_generate_hrm, model)  # inject
```

### 5.3 Code Changes
1. Implement `greedy_generate_hrm(model, input_ids, max_len, temperature)` in new util.  
2. Re-use safe-exec & Pass@k computation unchanged.

### 5.4 Challenges & Solutions
• HRM lacks kv-cache; generation currently O(L²). Short-term acceptable (seq ≤256). Long-term todo: add cache.

### 5.5 Validation
1. Run `python scripts/evaluate.py --model-type hrm --ckpt ... --num-samples 5` end-to-end.  
2. Inspect one output; confirm syntactically valid Python (`ast.parse`).

---

## Parameter Impact Summary
| Component | Params Δ | Memory Δ |
|-----------|----------|----------|
| Remove puzzle embeddings | – (puzzle_emb_ndim × id_count) | – few KB |
| Disable Q-head | −0.5 M | −2 MB |
| Larger vocab | +~25 M (50257 vs 512) | +100 MB (fp16) |

A100 40 GB remains sufficient for seq_len 512, batch 16, fp16.

---

## Integration Points
* Training script → `HierarchicalReasoningModel_ACTV1(config)` (codegen flag).  
* Loader emits `{'input_ids': ids, 'labels': ids_shifted}`.  
* Evaluate script calls model.generate.  
* CI smoke test imports full path to ensure import namespace unchanged.

---

## Appendix A – Diff Sketches

**Attention Mask**

```diff
- scores = (q @ k.transpose(-2, -1)) * self.scale
+ scores = (q @ k.transpose(-2, -1)) * self.scale
+ if self.causal:
+     mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), 1).bool()
+     scores.masked_fill_(mask, float("-inf"))
```

**Embedding Path**

```diff
- def _input_embeddings(self, input, puzzle_identifiers):
+ def _input_embeddings(self, input_ids):
-     embedding = self.embed_tokens(input.to(torch.int32))
-     if self.config.puzzle_emb_ndim > 0:
-         ...
+     embedding = self.embed_tokens(input_ids.to(torch.int32))
```

---

_This document serves as the authoritative reference for engineers implementing Phase 3 modifications. All code changes must follow the listed validation steps before merge._
