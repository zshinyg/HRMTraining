# HRM-CodeGen
[![CI](https://github.com/zshinyg/HRMTraining/actions/workflows/ci.yml/badge.svg)](https://github.com/zshinyg/HRMTraining/actions) · [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A research prototype that adapts the **Hierarchical Reasoning Model (HRM)**—originally introduced for abstract reasoning problems—to the domain of code generation.  
The project trains HRM on the **MBPP (Mostly Basic Python Problems)** dataset and provides an extensible framework for benchmarking HRM on any program-synthesis corpus.

---

## 1 Project Purpose
Large Language Models excel at natural language, yet specialised models are still competitive on program synthesis when trained efficiently.  
This repository investigates:

* How the lightweight **27 M-parameter HRM** architecture performs on entry-level programming tasks.
* Whether hierarchical temporal abstraction (high-level planning + low-level execution) is beneficial for code generation.
* A modular training/evaluation pipeline for quickly swapping new datasets into HRM.

---

## 2 HRM Architecture Overview

| Level | Module | Time-scale | Role in Code Generation |
|-------|--------|-----------|--------------------------|
| High  | Planner RNN | Slow | Forms coarse-grained program strategy (e.g. algorithm outline). |
| Low   | Executor RNN | Fast | Fills in syntactic details and token-level code. |

Key properties that make HRM appealing for coding tasks:

1. **Deep computation without long chains of logits** – reasoning is compressed into hidden states, mitigating exposure bias.  
2. **Single forward pass** – no expensive step-by-step generation required at train time.  
3. **Parameter efficiency** – comparable accuracy to larger transformers on symbolic tasks; attractive for domain-specific fine-tuning.

For a full technical description see [Sapient Inc., 2025] in the References section.

---

## 3 Repository Structure

```
.
├── data/                  # Downloaded & pre-processed datasets
│   └── mbpp/
├── hrm/                   # Core model implementation
│   ├── layers.py
│   ├── model.py
│   └── config.py
├── scripts/
│   ├── train.py           # Training entry-point
│   ├── evaluate.py        # Evaluation on MBPP
│   └── convert_mbpp.py    # JSONL → tokenised binary
└── README.md
```

---

## 4 Installation

Prerequisites:

* Python ≥ 3.10  
* PyTorch ≥ 2.2 with GPU/CPU as desired  
* CUDA 11.x (if training on GPU)

```bash
git clone https://github.com/zshinyg/HRMTraining.git
cd HRMTraining
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download the MBPP corpus (≈ 1 K tasks):

```bash
python scripts/convert_mbpp.py --split train
```

---

## 5 Usage

### 5.1 Training on MBPP

```bash
python scripts/train.py \
  --data-path data/mbpp/train.bin \
  --config hrm/configs/mbpp_base.yaml \
  --out-dir checkpoints/hrm_mbpp
```

The default config trains for 20 K steps on an A100 (≈ 2 h).  
Adjust `global_batch_size`, `lr`, and `epochs` as needed.

### 5.2 Evaluating Pass@k

```bash
python scripts/evaluate.py \
  --ckpt checkpoints/hrm_mbpp/step_20000.pt \
  --split test --k 1 5 10
```

Outputs aggregated pass@k and per-task success rates.

---

## 6 Extending to New Datasets

1. Implement a `DatasetBuilder` in `scripts/convert_<dataset>.py` that:  
   * reads raw JSON/CSV,  
   * yields `(prompt, solution, tests)` triples,  
   * serialises to the unified binary format used by HRM.  
2. Add a config file under `hrm/configs/` specifying:  
   * vocabulary path,  
   * context length,  
   * evaluation metric.  
3. Launch training/evaluation as shown above, pointing to the new data path.  

The model code is dataset-agnostic; only tokenisation and test-runner adapters change.

---

## 7 License

This project is licensed under the terms of the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 8 References

* Sapient Inc. (2025). **Hierarchical Reasoning Model**. GitHub: <https://github.com/sapientinc/HRM>.  
* Austin, J. et al. (2021). **Program Synthesis with Large Language Models**. MBPP dataset. Hugging Face: <https://huggingface.co/datasets/google-research-datasets/mbpp>.  
* EvalPlus (2023). **MBPP+ rigorous evaluation**. GitHub: <https://github.com/evalplus/evalplus>.

---

## 9 Roadmap

| Phase | Milestone | Status |
|-------|-----------|--------|
| 0.1   | Minimum HRM port to PyTorch & passing unit tests | ✅ |
| 0.2   | MBPP training script & baseline results | ✅ |
| 0.3   | Pass@k leaderboard vs. transformer baselines | 🚧 In&nbsp;Progress |
| 0.4   | Add EvalPlus MBPP+ robustness tests | ⏳ |
| 0.5   | Plug-and-play dataset registry (HumanEval, LeetCode) | ⏳ |
| 1.0   | Publish paper & dockerised release | ⏳ |

---

## 10 Contributing
Contributions, bug reports, and feature requests are welcome! Please open an issue to discuss major changes before submitting a pull request. All contributors are expected to follow the project’s Code of Conduct (to be added).
