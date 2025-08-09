#!/usr/bin/env python
"""
Quick GPT-2 zero-shot evaluation on MBPP dev subset.

Generates greedy completions (temperature=0.0) for a fixed number of tasks and
evaluates Pass@1 using the same subprocess harness as our HRM script.
"""

import argparse
import sys
from pathlib import Path
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Local imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.mbpp_loader import MBPPConfig, MBPPDataset
from hrm_codegen.generation import _execute_tests


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPT-2 zero-shot MBPP quick eval")
    p.add_argument("--model", type=str, default="gpt2")
    p.add_argument("--num-tasks", type=int, default=100)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default=str(Path("outputs/codegen/gpt2_quick_eval.json")))
    return p.parse_args()


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = select_device()

    # Load dataset slice
    ds = MBPPDataset(split="test", config=MBPPConfig(dev_mode=True, dev_samples=args.num_tasks, seed=args.seed))

    # Load model/tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    results = []
    passed = 0
    for i in range(len(ds)):
        sample = ds.samples[i]
        prompt = sample["prompt"]
        test_cases: List[str] = sample.get("test_list", [])

        # Greedy decode
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the completion beyond the prompt
        if gen_text.startswith(prompt):
            completion = gen_text[len(prompt) :]
        else:
            completion = gen_text

        ok = _execute_tests(completion, test_cases, timeout=5.0)
        passed += int(bool(ok))
        results.append({"task_id": int(sample.get("task_id", i)), "passed": bool(ok)})

    summary = {
        "num_tasks": len(ds),
        "passed": passed,
        "pass_at_1": passed / max(1, len(ds)),
        "model": args.model,
        "device": str(device),
    }

    print(json.dumps(summary, indent=2))

    out = {"summary": summary, "results": results}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved results to: {out_path}")
    return summary


if __name__ == "__main__":
    # Silence HF tokenizers fork warning if desired
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    run(parse_args())

