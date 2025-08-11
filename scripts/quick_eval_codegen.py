#!/usr/bin/env python
"""
Quick evaluation for Sapient-HRM CodeGen on MBPP dev subset.

This script loads the Sapient-HRM CodeGen model using `configs/codegen_base.yaml`,
samples a small number of MBPP test tasks (dev mode), generates one solution per
task, executes tests in a subprocess, and reports Pass@1 and timing.

Usage:
  python scripts/quick_eval_codegen.py \
    --config configs/codegen_base.yaml \
    --num-tasks 25 \
    --temperature 0.8 \
    --timeout 5 \
    --output outputs/codegen/quick_eval_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Prepend project root to avoid conflicts with installed packages (e.g., HF "datasets")
sys.path.insert(0, str(PROJECT_ROOT))

# Import dataset and codegen utilities
from datasets.mbpp_loader import MBPPConfig, MBPPDataset
from hrm_codegen.generation import generate_code, _execute_tests
from hrm_codegen.mock_model import MockHRMModel, MockHRMConfig


@dataclass
class TaskResult:
    task_id: int
    passed: bool
    gen_seconds: float
    error: Optional[str]


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick Sapient-HRM CodeGen evaluation on MBPP dev subset")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "codegen_base.yaml"))
    parser.add_argument("--num-tasks", type=int, default=25, help="Number of MBPP test tasks to sample")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p for sampling")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k for sampling")
    parser.add_argument("--timeout", type=float, default=5.0, help="Execution timeout (seconds) per task")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "outputs" / "codegen" / "quick_eval_results.json"))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_mbpp_dev(num_tasks: int, seed: int) -> MBPPDataset:
    cfg = MBPPConfig(dev_mode=True, dev_samples=num_tasks)
    # Ensure deterministic sampling
    cfg.seed = seed
    ds = MBPPDataset(split="test", config=cfg)
    return ds


def run_quick_eval(args: argparse.Namespace) -> Dict[str, Any]:
    set_seed(args.seed)
    device = select_device()

    # Try to load Sapient-HRM CodeGen; fall back to mock model if unavailable
    codegen_cfg = None
    model = None
    model_source = "unknown"
    try:
        # Defer heavy imports to runtime to catch missing vendor cleanly
        from hrm_codegen.config import load_config as load_codegen_config  # type: ignore
        from hrm_codegen.model import HRMCodeGenerator  # type: ignore

        codegen_cfg = load_codegen_config(args.config)
        model = HRMCodeGenerator.from_config(codegen_cfg).to(device)
        model.eval()
        model_source = "sapient_hrm"
    except Exception as e:
        # Fallback to mock model for a smoke test when Sapient HRM isn't present
        model_source = "mock"
        print(
            "WARNING: Using MockHRMModel fallback instead of Sapient HRM. "
            "To use the real HRM, clone the upstream into external/sapient-hrm or install the package 'sapient_hrm' in your environment.",
            file=sys.stderr,
        )
        mock_cfg = MockHRMConfig(
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            max_position_embeddings=512,
        )
        model = MockHRMModel(mock_cfg).to(device)
        model.eval()

    # Load MBPP dev subset
    dataset = load_mbpp_dev(args.num_tasks, args.seed)

    results: List[TaskResult] = []
    start_total = time.time()

    for i in range(len(dataset)):
        sample = dataset.samples[i]
        task_id = int(sample.get("task_id", i))
        prompt = sample["prompt"]
        test_cases: List[str] = sample.get("test_list", [])

        # Generate one solution and measure time
        gen_start = time.time()
        try:
            max_len = 256
            if codegen_cfg is not None:
                try:
                    max_len = int(codegen_cfg.evaluation.get("max_generate_tokens", 256))
                except Exception:
                    pass

            gen_text = generate_code(
                model,
                prompt,
                max_length=max_len,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                num_return_sequences=1,
                do_sample=True,
                clean_output=True,
            )
            if isinstance(gen_text, list):
                gen_text = gen_text[0]
            gen_seconds = time.time() - gen_start
        except Exception as e:
            # Failure during generation
            results.append(TaskResult(task_id=task_id, passed=False, gen_seconds=0.0, error=f"generation_error: {e}"))
            continue

        # Execute tests (subprocess runner inside helper)
        try:
            passed = _execute_tests(gen_text, test_cases, timeout=args.timeout)
            results.append(TaskResult(task_id=task_id, passed=bool(passed), gen_seconds=gen_seconds, error=None))
        except Exception as e:
            results.append(TaskResult(task_id=task_id, passed=False, gen_seconds=gen_seconds, error=f"execution_error: {e}"))

    total_seconds = time.time() - start_total

    passed_count = sum(1 for r in results if r.passed)
    pass_at_1 = passed_count / max(1, len(results))
    avg_gen_sec = sum(r.gen_seconds for r in results if r.gen_seconds is not None) / max(1, len(results))

    summary: Dict[str, Any] = {
        "num_tasks": len(results),
        "passed": passed_count,
        "pass_at_1": pass_at_1,
        "avg_gen_seconds": avg_gen_sec,
        "total_seconds": total_seconds,
        "device": str(device),
        "config_path": args.config,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "model_source": model_source,
    }

    # Print concise summary (always include model source)
    print(json.dumps(summary, indent=2))

    # Save detailed results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        payload = {
            "summary": summary,
            "results": [asdict(r) for r in results],
        }
        json.dump(payload, f, indent=2)
    print(f"Saved results to: {out_path}")

    return summary


def main() -> None:
    args = parse_args()
    run_quick_eval(args)


if __name__ == "__main__":
    main()

