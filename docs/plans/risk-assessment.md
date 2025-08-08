# RISK_ASSESSMENT.md  
_HRM → Code Generation Adaptation – Comprehensive Risk Register_  
_Last updated: 2025-08-05_

| # | Category | Risk | Prob. | Impact | Technical Description | Triggers / Early Warnings | Mitigation Strategy (specific actions) | Fallback Plan (if mitigation fails) | Owner | Mitigation Deadline |
|---|----------|------|-------|--------|-----------------------|---------------------------|----------------------------------------|-------------------------------------|-------|--------------------|
| T-1 | Technical – Memory | Causal mask causes OOM on A100 during seq_len = 512 training | Med | Med | Triangular mask doubles attention memory; FlashAttention fallback may not engage | GPU memory > 90 % or CUDA OOM | • Enable FlashAttention-2 with `is_causal=True`<br>• Reduce batch size adaptively via gradient accumulation<br>• Monitor memory in `wandb` | Cut seq_len to 256 and resume training; run gradient checkpointing | ML Eng 1 | D+2 |
| T-2 | Technical – Performance | Generation lacks KV-cache, making inference O(L²) | Med | Low | Each decode step recomputes full attention | Decode latency > 1 s for 256 tokens | • Implement simple KV-cache in Attention.<br>• Limit max_gen_len to 128 in Phase 3 | Switch to teacher-forcing evaluation only; postpone live demo | ML Eng 1 | D+7 |
| T-3 | Technical – Compatibility | FlashAttention build fails on some dev machines | Med | Med | CUDA driver or GCC mismatch | `pip install` error, `undefined symbol` at import | • Provide Dockerfile with pinned CUDA 11.8 + FA-2 commit<br>• Include CPU fallback flag | Disable FA; use PyTorch native attention and halve batch | MLOps | D0 |
| T-4 | Integration – Code Conflicts | Upstream Sapient HRM updates break our fork | Med | High | `git subtree pull` introduces API changes (e.g., param rename) | Merge conflict or unit tests fail CI | • Freeze to commit `af3e8b1` for Phase 3<br>• Use patch files rather than inline edits | Hard-fork repository and drop subtree sync | ML Eng 2 | Continuous |
| T-5 | Integration – Dependency Drift | Transformers version mismatch between tokenizer and runtime | Low | Med | HF 4.42 introduces breaking changes in `AutoTokenizer` | ImportError during CI, tokeniser loads wrong vocab | • Pin `transformers==4.42.0` in `requirements.txt`<br>• Add dependency check in CI workflow | Vendor the tokenizer JSON; remove HF runtime dependency | MLOps | D0 |
| D-1 | Data Pipeline | GPT-2 tokenizer inserts unexpected spaces causing off-by-one labels | Med | Med | BPE merges differ from baseline; prompts mis-aligned | Sudden loss spikes, low Pass@k despite declining train loss | • Freeze vocab; run SHA-256 hash check of vocab.json in loader<br>• Add unit test asserting first 10 tokens of sample prompt | Regenerate dataset with simple character tokenizer; re-train baseline | ML Eng 2 | D+1 |
| D-2 | Data Pipeline | Corrupted MBPP records (non-UTF-8) crash loader | Low | Low | Some tasks include emojis or odd quotes | Loader raises UnicodeDecodeError | • Use `errors='replace'` in open(); log corrupted ids | Skip bad samples; continue training with warning | ML Eng 2 | D+1 |
| E-1 | Evaluation | Generated code executes unsafe ops (fork-bomb, file IO) | Med | High | Model outputs malicious Python | Timeouts, disk writes during evaluation | • Run code in isolated subprocess with 1 s wall timeout & resource limits (ulimit) in `evaluate.py`<br>• Disallow `import os, sys, subprocess` via AST filter | If sandbox breached, switch to Docker-based seccomp sandbox | ML Eng 1 | D+4 |
| E-2 | Evaluation | Pass@k metric mis-computed due to duplicate generations | Low | Med | `unique=True` flag omitted, inflating score | Pass@k > 1 on trivial baseline | • Ensure set-based uniqueness in implementation<br>• Add unit test with synthetic examples | Fall back to original OpenAI evaluation script | ML Eng 2 | D+3 |
| P-1 | Project Delivery | Phase 3 slips beyond 7 days due to bug backlog | Med | High | Under-estimated integration effort | > 3 critical bugs open on D+4 | • Daily stand-ups & burn-down chart<br>• Scope guard – non-essential features postponed | Re-baseline schedule; add extra ML engineer | PM | Daily |
| P-2 | Project Delivery | GPU quota pre-empted by another team | Low | High | Shared cluster allocation changed | Job eviction, queue wait > 12 h | • Reserve A100 for 1-week window; tag jobs high prio | Switch to on-demand cloud GPU (AWS g5.12x) | PM | D0 |
| P-3 | Project Delivery | Key engineer sick leave mid-sprint | Low | Med | Single-point knowledge holder | Unresponsive on Slack, tasks blocked | • Knowledge share sessions; document setup in README | PM reallocates tasks; slip non-critical tasks | PM | Continuous |

Legend: `D+N` = N days after Phase 3 kick-off (Day 0).

Owners:  
• **ML Eng 1** – Lead architecture & model changes  
• **ML Eng 2** – Data & evaluation layers  
• **MLOps** – Environment, CI, infra  
• **PM** – Schedule & resource management  

_This register will be reviewed in daily stand-ups; new risks appended with incremental numbering._  
