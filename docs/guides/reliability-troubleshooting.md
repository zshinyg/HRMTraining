---
status: active
owner: engineering
summary: Troubleshooting steps and playbooks for reliability issues
last_reviewed: 2025-08-08
---

# RELIABILITY_TROUBLESHOOTING_PLAN.md  
_v1.0 – 2025-08-05_  

Owner: **@zshinyg**  
Scope: Critical reliability steps for validating the HRM 27 M model vs GPT-2-117 M on the MBPP code-generation benchmark.

---

## 1. Immediate Flash-Attn Resolution  

Key finding: **flash-attn is **not** imported anywhere in `hrm/` or `scripts/`**. The “blocker” reported by Code Droids originates from the Phase-3 spec, not from runtime code.

### 1.1 Verification Checklist
| Step | Command | Expected |
|------|---------|----------|
| Search imports | `grep -R "flash_attn" -n hrm scripts | wc -l` | **0** hits |
| Run unit tests | `pytest -q` | All green |
| Import path smoke | `python -c "import hrm, torch; print('OK')"` | No ImportError |

If all pass ➜ flash-attn can be **dropped** from the dependency list.  
Add to `README` Known-Issues: _Flash-Attn optional only for large GPU nodes._

### 1.2 Optional GPU Optimisation Path
On CUDA boxes we can still leverage flash-attn:

```bash
# Example for CUDA 12.1 + A100
pip install flash-attn --no-build-isolation --no-index \
     --find-links https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/ \
     --extra-index-url https://download.pytorch.org/whl/cu121
export USE_FLASH_ATTN=1  # picked up by hrm.layers if env var exists
```

Patch point (if desired) inside `hrm/layers.py`:

```python
if os.getenv("USE_FLASH_ATTN") == "1":
    from flash_attn.ops.fused_dense import FusedDense  # example
    # swap in flash attention projection here
```

---

## 2. M1 Mac Optimisation Guide  

Apple Silicon can train the 27 M HRM in ~1.3 × real-time with correct settings.

1. **PyTorch ≥ 2.1** compiled with Metal backend.  
2. Environment flags:  
   ```bash
   export PYTORCH_ENABLE_MPS_FALLBACK=1   # avoid abort on unsupported ops
   export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.9
   ```
3. Code snippet for device pick-up:

```python
device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
model.to(device, dtype=torch.bfloat16)
```

4. Memory tips  
   • Keep batch‐size small (≤ 4) and accumulate grads (`--grad-accum 8`).  
   • Prefer `torch.compile(model, backend="aot_eager")` to fuse ops.  
   • Disable gradients on unused tensors: `with torch.no_grad(): embed_cache = ...`.

5. Profiling  
   ```python
   with torch.autograd.profiler.profile(with_flops=True, use_mps=True) as prof:
       loss = model(**batch)["loss"]; loss.backward()
   prof.table(sort_by="self_cpu_time_total")
   ```

---

## 3. Training Stability for 27 M HRM  

| Technique | Setting | Rationale |
|-----------|---------|-----------|
| Gradient clipping | `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` | Prevent exploding grads |
| Mixed precision | `dtype=torch.bfloat16` (MPS) / `torch.float16` (CUDA) | Memory & speed |
| LR schedule | Cosine w/ warm-up 500 steps | Smooth start |
| Checkpoints | Every 1 000 steps; keep last 3 | Fast recovery |
| Loss watchdog | `if not torch.isfinite(loss): raise RuntimeError` | Early failure detect |

Sample training loop hook:

```python
for step, batch in enumerate(loader, start=1):
    outputs = model(**batch)
    loss = outputs["loss"] / grad_accum
    loss.backward()
    if step % grad_accum == 0:
        clip_grad()
        optimizer.step(); optimizer.zero_grad()
        if step % 100 == 0:
            wandb.log({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]})
        if step % 1000 == 0:
            save_ckpt(step)
```

---

## 4. System Performance Optimisation  

1. **Batch-size autotune**

```python
def autotune_batch(model, seq_len=256, safety=0.8):
    max_mem = torch.mps.current_allocated_memory() if MPS else torch.cuda.mem_get_info()[0]
    for bs in (8,4,2,1):
        est = bs*seq_len*model.config.hidden_dim*4
        if est < max_mem*safety:
            return bs
```

2. **Efficient DataLoader**

```python
loader = DataLoader(ds,
        batch_size=BATCH,
        num_workers=os.cpu_count(),
        pin_memory=not MPS,
        persistent_workers=True,
        prefetch_factor=4)
```

3. **Re-use tensors** – allocate buffers once; call `tensor.copy_()` inside loop.

4. **Torch compile** – `torch.compile(model, mode="reduce-overhead")`.

---

## 5. Incident Prevention & Recovery  

| Monitor | Tool | Threshold | Action |
|---------|------|-----------|--------|
| GPU/MPS util | `nvidia-smi dmon` / `powermetrics` | util < 10 % for 5 m | Auto-restart job |
| Loss NAN | Custom callback | detect NaN / Inf | Roll back to last good ckpt |
| Disk free | Prometheus `node_filesystem_free_bytes` | < 5 GB | Pause checkpoints |
| Heartbeat | `scripts/heartbeat.py` → Redis | lapse > 120 s | PagerDuty critical |

**Auto-restart script (`scripts/recover.sh`)**

```bash
#!/usr/bin/env bash
CKPT=$(ls checkpoints/*.pt | tail -1)
python scripts/train_codegen.py --resume $CKPT --max_retries 3
```

Cron entry (M1):

```
*/10 * * * * pkill -f heartbeat || /bin/bash recover.sh
```

---

## 6. Validation Reliability (HRM vs GPT-2-117 M)  

1. **Deterministic set-up**

```yaml
seed: 42
torch_deterministic: true
numpy_seed: 42
cuda_deterministic: true
```

2. **Consistent tokenizer** – both models use `gpt2` BPE vocab (shared).

3. **Dataset hash**

```bash
md5sum data/mbpp/train.jsonl > data/.mbpp.md5
```
Validation script verifies hash before running.

4. **Metric parity** – use `evaluate.load("mbpp", "pass@k")`; pass the same `k=1,10,100` to both models.

5. **Report template**

```json
{
  "commit": "<sha>",
  "seed": 42,
  "hrm_pass@1": 0.031,
  "gpt2_pass@1": 0.024,
  "delta": 0.007
}
```

CI step `scripts/compare_results.py` fails PR if `delta < 0`.

---

## Appendix A – Quick Commands

```bash
# Full local validation on M1
make env-m1          # installs torch-mps + deps
python scripts/train_codegen.py --config configs/m1_mbpp.yaml
python scripts/evaluate.py --ckpt checkpoints/last.pt --k 1
```

```bash
# GPU speed test with optional flash-attn
USE_FLASH_ATTN=1 python scripts/benchmark.py --seq-len 256 --batch 16
```

---

End of Plan – owned by **@zshinyg**. Pull requests modifying this file require reviewer sign-off from Reliability DRI.
