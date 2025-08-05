# HRM-CodeGen – User Experience Design

_Last updated: 2025-08-05_

---

## 1 · User Personas & Primary Use-Cases

| Persona | Goals | Pain-points | Key Use-Cases |
|---------|-------|-------------|---------------|
| **P1 – Research Scientist** (ML PhD/Post-doc) | Prototype novel reasoning models; benchmark quickly | Long setup times; boiler-plate evaluation code | • Train HRM on custom dataset<br>• Run Pass@k evaluation<br>• Compare against GPT-2 |
| **P2 – Applied Engineer** (ML Engineer at start-up) | Integrate compact reasoning model into product | Heavy models too big; latency | • Fine-tune 27 M HRM<br>• Package as REST/gRPC service |
| **P3 – Educator** (University lecturer) | Teach advanced model architectures | Complex installs; unclear docs | • Run ready-made notebook demo<br>• Visualise hierarchical reasoning |
| **P4 – OSS Contributor** | Extend HRM to new domains | Hard to navigate codebase | • Add dataset plug-in<br>• Contribute optimisation PR |

---

## 2 · User Journey Mapping

```
Problem → Install → Prepare Data → Configure Model → Train → Evaluate → Iterate → Deploy
```

1. **Discover** toolkit on GitHub / HF.  
2. **Install** via `pip install hrm-codegen` (≈3 min).  
3. **Data Prep**: `hrm-codegen data mbpp` downloads/ converts MBPP.  
4. **Configure** with YAML (copy example, tweak).  
5. **Train**: `hrm-codegen train -c mbpp_dev.yaml` (see live metrics in W&B).  
6. **Evaluate**: `hrm-codegen eval --ckpt best.pt --k 1 10`.  
7. **Iterate** hyperparams via CLI or SDK.  
8. **Deploy** model (`hrm-codegen export onnx` → upload to HF).

Key satisfaction moments: install under 5 min, first Pass@k score within 30 min on laptop.

---

## 3 · Python SDK API Specification

```python
import hrm_codegen as hc

# 3.1  Load model
model = hc.HRMCodeGen.from_pretrained("sapient/hrm-codegen-base",
                                      device="auto")          # cpu/cuda/mps

# 3.2  Train
trainer = hc.Trainer(
    model=model,
    train_dataset="data/mbpp/train.bin",
    eval_dataset="data/mbpp/test.bin",
    config="configs/mbpp_dev.yaml",
)
trainer.fit()                        # returns metrics dict

# 3.3  Generate
code = model.generate_code(
    prompt="Write a python function to compute gcd",
    max_tokens=128,
    temperature=0.7,
    top_p=0.95,
)

# 3.4  Evaluate
metrics = hc.evaluator.pass_at_k(
    model=model,
    dataset="data/mbpp/test_raw.json",
    k=(1, 10),
    timeout=2.0,
)

# 3.5  Utilities
model.save_checkpoint("checkpoints/best.pt")
hc.utils.benchmark(model, seq_len=256, batch_size=8)
```

### Public Classes / Methods

| Object | Signature | Notes |
|--------|-----------|-------|
| `HRMCodeGen` | `.from_pretrained(path_or_hf_id, device="auto")` | Loads weights + config |
|  | `.generate_code(prompt, max_tokens, **sampling)` | Returns str |
|  | `.beam_search(prompt, max_tokens, num_beams)` | Optional |
| `Trainer` | `Trainer(model, train_dataset, eval_dataset, config)` | |
|  | `.fit(resume=False)` | Trains + logs |
| `evaluator` | `pass_at_k(model, dataset, k=(1,10), timeout=2.0)` | Returns dict |

Return types: `dict[str, float]` for metrics.

Exceptions: `hc.errors.TrainingError`, `hc.errors.EvaluationTimeout`.

---

## 4 · CLI Interface Design

```bash
# Top-level command
hrm-codegen [GLOBAL-OPTS] <subcommand> [ARGS]

GLOBAL-OPTS:
  -v, --verbose        Increase log level
  --device cpu|cuda|mps

SUBCOMMANDS:

  train        Train a model
  eval         Evaluate checkpoint(s)
  gen          Generate code from prompt
  data         Dataset utilities
  benchmark    Measure throughput
  export       Export model formats
```

### Examples & Workflows

```bash
# Install & prepare data
pip install hrm-codegen
hrm-codegen data mbpp --split train test   # downloads & tokenises

# Quick dev training run
hrm-codegen train --config configs/mbpp_dev.yaml \
                  --data data/mbpp/train.bin \
                  --eval-data data/mbpp/test.bin

# Evaluate Pass@k
hrm-codegen eval --ckpt checkpoints/best.pt --k 1 10

# Generate with checkpoint
hrm-codegen gen --ckpt checkpoints/best.pt \
                --prompt "Write a python bubble sort" \
                --max 128 --temperature 0.8
```

Flags auto-complete via `argcomplete`.

---

## 5 · Integration Patterns

| Tool / Workflow | Integration Method | Notes |
|-----------------|--------------------|-------|
| **Jupyter / Colab** | `import hrm_codegen` SDK; inline visualisations | IPython-friendly progress bars |
| **Weights & Biases** | Default logger inside `Trainer` | `--no-wandb` flag to disable |
| **Hugging Face Hub** | `.push_to_hub()` on model | Token via `HF_TOKEN` env var |
| **Docker/K8s** | `docker build -f docker/Dockerfile .` | Exposes REST `/generate` |
| **VS Code Notebook** | Example `.ipynb` templates | |

---

## 6 · Error Handling & Feedback

| Scenario | UX Behaviour | Message Pattern |
|----------|--------------|-----------------|
| Invalid YAML config | Abort + clear path & line pointer | `ConfigError: missing field "model.hidden_dim" (configs/mbpp.yaml:12)` |
| CUDA OOM | Automatic gradient-accum fallback suggestion | `OutOfMemoryError: reduce batch_size or enable gradient_checkpointing` |
| Eval timeout | Skip case, count as fail, warn once | `⏱️  Timeout (2s) on example #123` |
| Bad prompt | Validate empty/long inputs | `ValueError: prompt must be non-empty and <2048 chars` |
| Non-passing tests | Show failing test output diff | Rendered diff snippet |

Structured JSON logs for programmatic parsing.

---

## 7 · Performance Expectations & User Requirements

| Tier | Environment | Expectation |
|------|-------------|-------------|
| Laptop Dev | Apple M-series, 16 GB | Training (dev config) ≤ 30 min / epoch; inference 250 tok/s |
| Single GPU | RTX 4090 24 GB | Full config throughput ≥ 30 k tok/s |
| 8× GPU (A100) | 160 GB | Global batch 2304, 24 h to convergence |
| Memory Limit | ≤ 16 GB RAM (CPU) | Dataset streaming enabled |

User must install PyTorch ≥ 2.6; CUDA 12.6 for GPU.

---

## 8 · Example Workflows & Code Samples

### 8.1  Fine-tuning on Custom Dataset

```python
import hrm_codegen as hc
from datasets import load_dataset

# convert dataset to raw JSON {"text":..., "test_list":[...]}
ds = load_dataset("openai_humaneval", split="train")
ds.to_json("data/custom_raw.json")

hc.data.convert_raw_to_bin("data/custom_raw.json", "data/custom_train.bin")

model = hc.HRMCodeGen.from_pretrained("sapient/hrm-codegen-base")
trainer = hc.Trainer(model=model,
                     train_dataset="data/custom_train.bin",
                     eval_dataset=None,
                     config="configs/custom.yaml")
trainer.fit()
```

### 8.2  Batch Generation with CLI

```bash
cat prompts.txt | hrm-codegen gen \
       --ckpt checkpoints/best.pt \
       --batch 8 --max 128 --temperature 0.8 \
       > generated.txt
```

### 8.3  Integrating Into a Flask API

```python
from flask import Flask, request, jsonify
import hrm_codegen as hc

app = Flask(__name__)
model = hc.HRMCodeGen.from_pretrained("checkpoints/best.pt", device="cuda")

@app.post("/generate")
def generate():
    prompt = request.json["prompt"]
    code = model.generate_code(prompt, max_tokens=128)
    return jsonify({"code": code})

app.run(port=8000)
```

---

### Appendix A · Command Reference (cheat-sheet)

| Action | CLI | SDK |
|--------|-----|-----|
| Prepare MBPP | `hrm-codegen data mbpp` | `hc.data.prepare_mbpp()` |
| Train | `hrm-codegen train -c cfg.yaml` | `hc.Trainer(...).fit()` |
| Eval | `hrm-codegen eval --ckpt x.pt --k 1 10` | `hc.evaluator.pass_at_k(...)` |
| Generate | `hrm-codegen gen --prompt "..."` | `model.generate_code()` |
| Benchmark | `hrm-codegen benchmark` | `hc.utils.benchmark()` |
| Export | `hrm-codegen export onnx` | `model.export_onnx()` |

---

_This document defines the end-to-end user experience for HRM-CodeGen and serves as implementation guidance for Phase 4._  
