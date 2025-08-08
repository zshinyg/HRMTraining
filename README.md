# HRM-CodeGen

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
├── notebooks/             # Exploratory analyses & visualisations
└── README.md
```

---

## 4 Installation

Prerequisites:

* Python ≥ 3.10
* PyTorch ≥ 2.2 with GPU/CPU as desired
* CUDA 11.x (if training on GPU)

```bash
git clone https://github.com/your-org/hrm-codegen.git
cd hrm-codegen
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Optional: prepare MBPP raw/binary artifacts (for manual training/eval workflows):

```bash
python scripts/convert_mbpp.py --split all --output-dir data/mbpp
```

---

## 5 Quickstart

1) Create a virtual environment and install dependencies (see Installation)

2) Run tests to verify your setup:

```bash
pytest -q
```

3) (Optional) Convert MBPP data for manual training/evaluation:

```bash
python scripts/convert_mbpp.py --split all --output-dir data/mbpp
```

4) Train/Evaluate using the examples below or your own configs.

---

## 6 Usage

### 6.1 Training on MBPP

```bash
python scripts/train.py \
  --data-path data/mbpp/train.bin \
  --config configs/hrm/mbpp_base.yaml \
  --out-dir checkpoints/hrm_mbpp
```

The default config trains for 20 K steps on an A100 (≈ 2 h).  
Adjust `global_batch_size`, `lr`, and `epochs` as needed.

### 6.2 Evaluating Pass@k

```bash
python scripts/evaluate.py \
  --ckpt checkpoints/hrm_mbpp/step_20000.pt \
  --split test --k 1 5 10
```

Outputs aggregated pass@k and per-task success rates.

---

## 7 Testing

- Run the full unit test suite:

```bash
pytest -q
```

- First-time tokenizer download: the GPT-2 tokenizer will be fetched and cached to `checkpoints/tokenizer/`. If you need to pre-cache (or run offline later):

```bash
python -c "from tokenization import get_tokenizer; get_tokenizer(force_reload=True)"
```

---

## 8 Extending to New Datasets

1. Implement a `DatasetBuilder` in `scripts/convert_<dataset>.py` that:
   * reads raw JSON/CSV,
   * yields `(prompt, solution, tests)` triples,
   * serialises to the unified binary format used by HRM.
2. Add a config file under `configs/hrm/` specifying:
   * vocabulary path,
   * context length,
   * evaluation metric.
3. Launch training/evaluation as shown above, pointing to the new data path.

The model code is dataset-agnostic; only tokenisation and test-runner adapters change.

---

## 9 Contributing & SDLC

- Development workflow emphasizes small, clear, reversible edits with tests. See the SDLC tracker for active decisions, risks, and PR policy:
  - `docs/SDLC_TRACKING.md`

- Branch naming:
  - `docs/<topic>`, `fix/<area>-<desc>`, `feat/<area>-<desc>`

- PR checklist (summary):
  - Tests pass locally (`pytest -q`)
  - Touched files formatted (`black`/`isort`); flake8-clean where applicable
  - Short summary, clear impact; link SDLC tracker when process-related

---

## 9.1 Daily Workflow & Status Checks

Use this lightweight loop when (re)starting work or handing off:

1) Sync and sanity-check

```bash
git fetch --all --prune
git checkout fix/base-hrm-bringup  # or your working branch
git pull
source .venv/bin/activate && pytest -q
```

2) Read current status/next steps

```bash
cat docs/STATUS.md
```

3) Track work as GitHub Issues and link them in PRs

- Use one Issue per task (e.g., "Introduce collate_fn for task_id batching")
- Reference Issues in commits/PRs (e.g., "Fixes #123")
- Issues: https://github.com/zshinyg/HRMTraining/issues

4) Before pushing

```bash
black path/to/changes && isort path/to/changes
pytest -q
git commit -m "<concise change summary>" && git push
```

Hand-off tip: update `docs/STATUS.md` with a one-paragraph summary and the next 1–3 actionable items.

---

## 10 Troubleshooting

- Tokenizer/network issues on first run: pre-cache as above, or set `TRANSFORMERS_OFFLINE=1` after caching.
- macOS without CUDA: training utilities fall back to MPS if available; otherwise CPU is supported for tests.
- Large artifacts: keep out of VCS. Use `outputs/`, `checkpoints/`, and `data/` locally.

---

## 11 Useful Links

- Documentation Index: `docs/README.md`
- SDLC Tracker: `docs/SDLC_TRACKING.md`
- Technical Specification: `TECHNICAL_SPECIFICATION.md`

---

## 12 Legal and License Information

This repository is distributed under the **MIT License**; see `LICENSE` for the
full text.  Unless otherwise noted, **all original source files are  
© 2025 zshinyg** and released under MIT.

### 7.1 Third-Party Dependencies & Attributions
The project depends on several open-source libraries released under permissive
licences compatible with MIT:

| Component | License | Attribution |
|-----------|---------|-------------|
| Sapient HRM (submodule) | Apache-2.0 | © Sapient Inc. (2022-2025) |
| HuggingFace Transformers | Apache-2.0 | © HuggingFace Inc. & contributors |
| HuggingFace Datasets | Apache-2.0 | © HuggingFace Inc. & contributors |
| PyTorch | BSD-3-Clause | © PyTorch Contributors |
| NumPy / SciPy | BSD-3-Clause | © NumPy & SciPy Developers |
| tqdm | MPL-2.0 | © tqdm Contributors |

A consolidated list of licences and notices is provided in the **`NOTICE`
file** at the repository root.

### 7.2 Open-Source Compliance
All inbound licences are permissive; no GPL/AGPL code is included.  Continuous
integration runs an automated licence scan to ensure ongoing compliance.

### 7.3 Legal Contact
Questions or concerns about licensing or intellectual-property matters may be
directed to **legal@zshinyg.dev**.

---

## 13 References

* Sapient Inc. (2025). **Hierarchical Reasoning Model**. GitHub: <https://github.com/sapientinc/HRM>.
* Austin, J. et al. (2021). **Program Synthesis with Large Language Models**. MBPP dataset. Hugging Face: <https://huggingface.co/datasets/google-research-datasets/mbpp>.
* EvalPlus (2023). **MBPP+ rigorous evaluation**. GitHub: <https://github.com/evalplus/evalplus>.

---

## 9 Roadmap

| Phase | Milestone | Status |
|-------|-----------|--------|
| 0.1   | Minimum HRM port to PyTorch & passing unit tests | ✅ |
| 0.2   | MBPP training script & baseline results | ✅ |
| 0.3   | Pass@k leaderboard vs. transformer baselines | ⏳ |
| 0.4   | Add EvalPlus MBPP+ robustness tests | ⏳ |
| 0.5   | Plug-and-play dataset registry (HumanEval, LeetCode) | ⏳ |
| 1.0   | Publish paper & dockerised release | ⏳ |

Contributions are welcome—please open issues or pull requests for feature requests, bug fixes, or new datasets.

---
