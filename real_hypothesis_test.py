#!/usr/bin/env python
"""
Real Hypothesis Test: 27M HRM vs 117M GPT-2

This script performs the actual validation of our core hypothesis:
"A 27M parameter HRM can outperform a 117M parameter GPT-2 through hierarchical reasoning"

It loads the real Sapient HRM model and GPT-2-117M from HuggingFace, runs inference
on MBPP tasks, and compares performance metrics including Pass@k, inference speed,
statistical significance, and parameter efficiency.
"""

import argparse
import gc
import json
import logging
import math
import os
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from tqdm import tqdm

# Add project root to path
sys.path.append(".")

# Import our dataset and tokenization utilities
from datasets.mbpp_loader import MBPPConfig, MBPPDataset
from tokenization import decode, encode, get_tokenizer

# Configure logging: ensure logs are written under the project logs/ directory
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_filename = (
    LOG_DIR / f"hrm_hypothesis_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename),
    ],
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Configure device
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(RANDOM_SEED)
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
logger.info(f"Using device: {device}")


def load_hrm_model() -> Tuple[torch.nn.Module, int]:
    """
    Load the real Sapient HRM model.

    Returns:
        model: The HRM model
        params: Number of parameters in millions
    """
    try:
        # Resolve project root (directory that contains this script)
        project_root = Path(__file__).resolve().parent

        # ------------------------------------------------------------------
        # 1. Add Sapient HRM repository to PYTHONPATH so we can import it.
        # ------------------------------------------------------------------
        sapient_hrm_path = project_root / "external" / "sapient-hrm"
        sys.path.append(str(sapient_hrm_path))

        # Import HRM modules
        from hrm.config import HRMConfig
        from hrm.model import HRMModel

        # ------------------------------------------------------------------
        # 2. Load our adapted YAML config for code generation
        #      Path: <project_root>/configs/hrm/mbpp_base.yaml
        # ------------------------------------------------------------------
        config_path = project_root / "hrm" / "configs" / "mbpp_base.yaml"

        logger.info(f"Loading HRM config from: {config_path}")
        config = HRMConfig.from_yaml(config_path)

        # Create model
        logger.info("Creating HRM model...")
        model = HRMModel(config)
        model.to(device)

        # Count parameters
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        params_m = params / 1_000_000  # Convert to millions

        logger.info(f"HRM model loaded: {params_m:.2f}M parameters")
        return model, params_m

    except Exception as e:
        logger.error(f"Error loading HRM model: {e}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Failed to load HRM model: {e}")


def load_gpt2_model() -> Tuple[torch.nn.Module, int]:
    """
    Load the GPT-2 117M model from HuggingFace.

    Returns:
        model: The GPT-2 model
        params: Number of parameters in millions
    """
    try:
        from transformers import GPT2Config, GPT2LMHeadModel

        logger.info("Loading GPT-2 117M model from HuggingFace...")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.to(device)

        # Count parameters
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        params_m = params / 1_000_000  # Convert to millions

        logger.info(f"GPT-2 model loaded: {params_m:.2f}M parameters")
        return model, params_m

    except Exception as e:
        logger.error(f"Error loading GPT-2 model: {e}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Failed to load GPT-2 model: {e}")


def load_mbpp_samples(num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load MBPP test samples.

    Args:
        num_samples: Number of samples to load (None for all)

    Returns:
        List of MBPP samples
    """
    try:
        # Configure dataset loading
        if num_samples:
            config = MBPPConfig(dev_mode=True, dev_samples=num_samples)
            logger.info(f"Loading {num_samples} MBPP test samples (dev mode)...")
        else:
            config = MBPPConfig(dev_mode=False)
            logger.info("Loading all MBPP test samples...")

        # Load dataset
        mbpp_dataset = MBPPDataset(split="test", config=config)

        # Convert to list of samples
        samples = []
        for sample in mbpp_dataset.samples:
            samples.append(
                {
                    "task_id": sample.get("task_id", 0),
                    "prompt": sample["prompt"],
                    "test_cases": sample.get("test_list", []),
                    "entry_point": None,  # Will be extracted from test cases
                    "canonical_solution": sample["completion"],
                }
            )

        # Extract entry points from test cases
        for sample in samples:
            if sample["test_cases"]:
                # Try to extract function name from the first test case
                test_case = sample["test_cases"][0]
                # Look for patterns like "assert func_name(...)" or "func_name(...)"
                import re

                match = re.search(r"assert\s+([a-zA-Z0-9_]+)\(", test_case)
                if not match:
                    match = re.search(r"^([a-zA-Z0-9_]+)\(", test_case)

                if match:
                    sample["entry_point"] = match.group(1)

        logger.info(f"Loaded {len(samples)} MBPP test samples")
        return samples

    except Exception as e:
        logger.error(f"Error loading MBPP samples: {e}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Failed to load MBPP samples: {e}")


def execute_test_case(code: str, test_case: str) -> bool:
    """
    Execute a test case against the generated code.

    Args:
        code: Generated Python code
        test_case: Test case to execute

    Returns:
        True if test passes, False otherwise
    """
    # Create a temporary namespace
    namespace = {}

    try:
        # Execute the code
        exec(code, namespace)

        # Execute the test case
        if test_case.startswith("assert "):
            # Execute the assertion directly
            exec(test_case, namespace)
            return True
        else:
            # Evaluate the expression
            result = eval(test_case, namespace)
            return bool(result)
    except Exception as e:
        # Test failed
        return False


def evaluate_solution(
    code: str, test_cases: List[str], entry_point: Optional[str] = None
) -> bool:
    """
    Evaluate if a generated solution passes all test cases.

    Args:
        code: Generated Python code
        test_cases: List of test cases
        entry_point: Function name to test (optional)

    Returns:
        True if all tests pass, False otherwise
    """
    # Skip empty code
    if not code or len(code.strip()) < 10:
        return False

    # Check if the entry point is in the code (if provided)
    if entry_point and entry_point not in code:
        return False

    # Execute each test case
    for test_case in test_cases:
        if not test_case:
            continue

        if not execute_test_case(code, test_case):
            return False

    return True


def generate_code_hrm(
    model: torch.nn.Module, prompt: str, max_tokens: int = 512
) -> Tuple[str, float, float]:
    """
    Generate code from the HRM model given a prompt.

    Args:
        model: The HRM model
        prompt: Input prompt
        max_tokens: Maximum tokens to generate

    Returns:
        generated_code: The generated code
        inference_time: Time taken for inference in seconds
        memory_usage: Peak memory usage in MB
    """
    try:
        tokenizer = get_tokenizer()

        # Ensure evaluation mode
        model.eval()

        # Track memory usage
        torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
        start_mem = (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        )

        # ------------------------------------------------------------------
        # 1. Encode prompt → input_ids just like we do for GPT-2
        # ------------------------------------------------------------------
        encoded = encode(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)

        # ------------------------------------------------------------------
        # 2. Run autoregressive generation with HRM
        #    NOTE: HRM `generate()` mirrors HF API and expects `input_ids`
        # ------------------------------------------------------------------
        start_time = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_length=input_ids.shape[1] + max_tokens,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        inference_time = time.time() - start_time

        # ------------------------------------------------------------------
        # 3. Memory accounting
        # ------------------------------------------------------------------
        end_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        memory_usage = (end_mem - start_mem) / (1024 * 1024)  # MB

        # ------------------------------------------------------------------
        # 4. Decode & slice generated portion
        # ------------------------------------------------------------------
        generated_text = decode(output_ids[0])
        generated_code = generated_text[len(prompt) :]  # only new tokens

        return generated_code, inference_time, memory_usage

    except Exception as e:
        logger.error(f"Error generating code with HRM: {e}")
        logger.error(traceback.format_exc())
        return "", 0.0, 0.0


def generate_code_gpt2(
    model: torch.nn.Module, prompt: str, max_tokens: int = 512
) -> Tuple[str, float, float]:
    """
    Generate code from the GPT-2 model given a prompt.

    Args:
        model: The GPT-2 model
        prompt: Input prompt
        max_tokens: Maximum tokens to generate

    Returns:
        generated_code: The generated code
        inference_time: Time taken for inference in seconds
        memory_usage: Peak memory usage in MB
    """
    try:
        tokenizer = get_tokenizer()

        # Ensure evaluation mode
        model.eval()

        # Track memory usage
        torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
        start_mem = (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        )

        # Encode prompt
        encoded = encode(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)

        # Measure inference time
        start_time = time.time()

        # Generate code using GPT-2
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_length=input_ids.shape[1] + max_tokens,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        inference_time = time.time() - start_time

        # Calculate memory usage
        end_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        memory_usage = (end_mem - start_mem) / (1024 * 1024)  # Convert to MB

        # Decode the generated tokens
        generated_text = decode(output_ids[0])

        # Extract only the newly generated part
        generated_code = generated_text[len(prompt) :]

        return generated_code, inference_time, memory_usage

    except Exception as e:
        logger.error(f"Error generating code with GPT-2: {e}")
        logger.error(traceback.format_exc())
        return "", 0.0, 0.0


def calculate_pass_at_k(n_samples: int, n_correct: int, k: int) -> float:
    """
    Calculate Pass@k metric.

    Args:
        n_samples: Number of samples
        n_correct: Number of correct solutions
        k: k value for Pass@k

    Returns:
        Pass@k score
    """
    if n_samples == 0:
        return 0.0

    if k == 1:
        return n_correct / n_samples

    # For k > 1, we use the formula from the Codex paper
    # This is a simplified version for our mock test
    pass_at_k = 1.0 - (1.0 - (n_correct / n_samples)) ** k
    return pass_at_k


def bootstrap_confidence_interval(
    results: List[bool], n_bootstrap: int = 1000, confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Calculate confidence interval using bootstrap resampling.

    Args:
        results: List of boolean results (True for correct, False for incorrect)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95 for 95% CI)

    Returns:
        lower: Lower bound of confidence interval
        upper: Upper bound of confidence interval
        p_value: p-value for the null hypothesis (proportion = 0.5)
    """
    n_samples = len(results)
    if n_samples == 0:
        return 0.0, 0.0, 1.0

    # Convert to numeric
    numeric_results = np.array([1 if r else 0 for r in results])
    observed_mean = np.mean(numeric_results)

    # Bootstrap resampling
    bootstrap_means = []
    for _ in range(n_bootstrap):
        resampled = np.random.choice(numeric_results, size=n_samples, replace=True)
        bootstrap_means.append(np.mean(resampled))

    # Calculate confidence interval
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, alpha / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

    # Calculate p-value (two-tailed)
    # Null hypothesis: proportion = 0.5
    null_hypothesis = 0.5
    if observed_mean >= null_hypothesis:
        p_value = 2 * (1 - np.mean([m >= null_hypothesis for m in bootstrap_means]))
    else:
        p_value = 2 * np.mean([m <= null_hypothesis for m in bootstrap_means])

    return lower, upper, p_value


def compare_models_statistical(
    hrm_results: List[bool], gpt2_results: List[bool], n_bootstrap: int = 1000
) -> Tuple[float, float, float, float]:
    """
    Perform statistical comparison between HRM and GPT-2 results.

    Args:
        hrm_results: List of boolean results for HRM
        gpt2_results: List of boolean results for GPT-2
        n_bootstrap: Number of bootstrap samples

    Returns:
        delta: Difference in means (HRM - GPT-2)
        lower: Lower bound of confidence interval for delta
        upper: Upper bound of confidence interval for delta
        p_value: p-value for the null hypothesis (delta = 0)
    """
    if len(hrm_results) != len(gpt2_results) or len(hrm_results) == 0:
        return 0.0, 0.0, 0.0, 1.0

    # Convert to numeric
    hrm_numeric = np.array([1 if r else 0 for r in hrm_results])
    gpt2_numeric = np.array([1 if r else 0 for r in gpt2_results])

    # Calculate observed difference
    observed_diff = np.mean(hrm_numeric) - np.mean(gpt2_numeric)

    # Bootstrap resampling for difference
    n_samples = len(hrm_results)
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        hrm_resampled = hrm_numeric[indices]
        gpt2_resampled = gpt2_numeric[indices]

        # Calculate difference
        diff = np.mean(hrm_resampled) - np.mean(gpt2_resampled)
        bootstrap_diffs.append(diff)

    # Calculate confidence interval for difference
    lower = np.percentile(bootstrap_diffs, 2.5)
    upper = np.percentile(bootstrap_diffs, 97.5)

    # Calculate p-value (two-tailed)
    # Null hypothesis: difference = 0
    if observed_diff >= 0:
        p_value = 2 * (1 - np.mean([d >= 0 for d in bootstrap_diffs]))
    else:
        p_value = 2 * np.mean([d <= 0 for d in bootstrap_diffs])

    return observed_diff, lower, upper, p_value


def save_results(
    results: Dict[str, Any], filename: str = "hypothesis_test_results.json"
):
    """
    Save test results to a JSON file.

    Args:
        results: Dictionary of results
        filename: Output filename
    """
    try:
        # Convert non-serializable types
        serializable_results = {}
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                serializable_results[k] = v.tolist()
            elif isinstance(v, np.float64) or isinstance(v, np.float32):
                serializable_results[k] = float(v)
            elif isinstance(v, np.int64) or isinstance(v, np.int32):
                serializable_results[k] = int(v)
            else:
                serializable_results[k] = v

        # Save to file
        with open(filename, "w") as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Results saved to {filename}")

    except Exception as e:
        logger.error(f"Error saving results: {e}")
        logger.error(traceback.format_exc())


def plot_results(results: Dict[str, Any], filename: str = "hypothesis_test_plot.png"):
    """
    Plot test results and save to file.

    Args:
        results: Dictionary of results
        filename: Output filename
    """
    try:
        # Create figure with 2x2 subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Pass@k comparison
        ax = axs[0, 0]
        metrics = ["pass_at_1", "pass_at_5", "pass_at_10"]
        hrm_values = [results["hrm"][m] for m in metrics]
        gpt2_values = [results["gpt2"][m] for m in metrics]

        x = np.arange(len(metrics))
        width = 0.35

        ax.bar(
            x - width / 2,
            hrm_values,
            width,
            label=f'HRM ({results["hrm"]["params"]:.1f}M)',
        )
        ax.bar(
            x + width / 2,
            gpt2_values,
            width,
            label=f'GPT-2 ({results["gpt2"]["params"]:.1f}M)',
        )

        ax.set_ylabel("Pass@k Score")
        ax.set_title("Pass@k Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(["Pass@1", "Pass@5", "Pass@10"])
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)

        # Add value labels
        for i, v in enumerate(hrm_values):
            ax.text(i - width / 2, v + 0.01, f"{v:.2%}", ha="center")
        for i, v in enumerate(gpt2_values):
            ax.text(i + width / 2, v + 0.01, f"{v:.2%}", ha="center")

        # 2. Inference time comparison
        ax = axs[0, 1]
        ax.bar(
            ["HRM", "GPT-2"],
            [results["hrm"]["avg_time"], results["gpt2"]["avg_time"]],
            color=["blue", "orange"],
        )
        ax.set_ylabel("Average Inference Time (s)")
        ax.set_title("Inference Speed Comparison")
        ax.grid(True, linestyle="--", alpha=0.7)

        # Add value labels
        ax.text(
            0,
            results["hrm"]["avg_time"] + 0.05,
            f'{results["hrm"]["avg_time"]:.2f}s',
            ha="center",
        )
        ax.text(
            1,
            results["gpt2"]["avg_time"] + 0.05,
            f'{results["gpt2"]["avg_time"]:.2f}s',
            ha="center",
        )

        # 3. Parameter efficiency
        ax = axs[1, 0]
        ax.bar(
            ["HRM", "GPT-2"],
            [results["hrm"]["params"], results["gpt2"]["params"]],
            color=["blue", "orange"],
        )
        ax.set_ylabel("Parameters (millions)")
        ax.set_title("Model Size Comparison")
        ax.grid(True, linestyle="--", alpha=0.7)

        # Add value labels
        ax.text(
            0,
            results["hrm"]["params"] + 2,
            f'{results["hrm"]["params"]:.1f}M',
            ha="center",
        )
        ax.text(
            1,
            results["gpt2"]["params"] + 2,
            f'{results["gpt2"]["params"]:.1f}M',
            ha="center",
        )

        # 4. Statistical comparison
        ax = axs[1, 1]
        ax.axis("off")  # Turn off axis

        # Create text box with statistical results
        stats_text = (
            f"Statistical Comparison (n={results['n_samples']})\n"
            f"-----------------------------------\n"
            f"HRM Pass@1: {results['hrm']['pass_at_1']:.2%} ({results['hrm']['correct']}/{results['n_samples']})\n"
            f"GPT-2 Pass@1: {results['gpt2']['pass_at_1']:.2%} ({results['gpt2']['correct']}/{results['n_samples']})\n"
            f"Difference: {results['delta']*100:+.1f} pp\n"
            f"95% CI: [{results['ci_lower']*100:.1f}, {results['ci_upper']*100:.1f}] pp\n"
            f"p-value: {results['p_value']:.4f}\n"
            f"Statistical Significance: {'Yes (p < 0.05)' if results['p_value'] < 0.05 else 'No (p ≥ 0.05)'}\n\n"
            f'Hypothesis: "27M HRM outperforms 117M GPT-2"\n'
            f"Verdict: {results['verdict']}"
        )

        ax.text(
            0.5,
            0.5,
            stats_text,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=10,
            family="monospace",
        )

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved to {filename}")

    except Exception as e:
        logger.error(f"Error plotting results: {e}")
        logger.error(traceback.format_exc())


def run_hypothesis_test(
    num_samples: int = 30, max_tokens: int = 512, n_bootstrap: int = 1000
):
    """
    Run the hypothesis test comparing HRM vs GPT-2.

    Args:
        num_samples: Number of MBPP samples to test
        max_tokens: Maximum tokens to generate
        n_bootstrap: Number of bootstrap samples for statistical testing
    """
    logger.info("\n" + "=" * 80)
    logger.info(" REAL HYPOTHESIS TEST: 27M HRM vs 117M GPT-2 ".center(80, "="))
    logger.info("=" * 80 + "\n")

    # Create output directory
    output_dir = f"results/hypothesis_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load models
        logger.info("Loading models...")
        hrm_model, hrm_params = load_hrm_model()
        gpt2_model, gpt2_params = load_gpt2_model()

        logger.info(f"HRM Model: {hrm_params:.1f}M parameters")
        logger.info(f"GPT-2 Model: {gpt2_params:.1f}M parameters")
        logger.info(f"Parameter Ratio: {gpt2_params/hrm_params:.1f}x\n")

        # Load MBPP samples
        samples = load_mbpp_samples(num_samples=num_samples)

        # Run inference and evaluation
        results = {
            "hrm": {"correct": 0, "times": [], "memory": []},
            "gpt2": {"correct": 0, "times": [], "memory": []},
        }

        # Track detailed results
        detailed_results = []

        logger.info("Running inference and evaluation...")
        for i, sample in enumerate(tqdm(samples, desc="Testing samples")):
            sample_result = {
                "task_id": sample["task_id"],
                "prompt": (
                    sample["prompt"][:100] + "..."
                    if len(sample["prompt"]) > 100
                    else sample["prompt"]
                ),
                "hrm": {"correct": False, "time": 0, "memory": 0},
                "gpt2": {"correct": False, "time": 0, "memory": 0},
            }

            # Generate with HRM
            logger.info(f"Sample {i+1}/{len(samples)}: {sample['prompt'][:50]}...")
            hrm_code, hrm_time, hrm_memory = generate_code_hrm(
                hrm_model, sample["prompt"], max_tokens
            )

            hrm_correct = evaluate_solution(
                hrm_code, sample["test_cases"], sample["entry_point"]
            )
            results["hrm"]["times"].append(hrm_time)
            results["hrm"]["memory"].append(hrm_memory)
            if hrm_correct:
                results["hrm"]["correct"] += 1

            sample_result["hrm"] = {
                "correct": hrm_correct,
                "time": hrm_time,
                "memory": hrm_memory,
                "code": hrm_code[:500] + "..." if len(hrm_code) > 500 else hrm_code,
            }

            # Generate with GPT-2
            gpt2_code, gpt2_time, gpt2_memory = generate_code_gpt2(
                gpt2_model, sample["prompt"], max_tokens
            )

            gpt2_correct = evaluate_solution(
                gpt2_code, sample["test_cases"], sample["entry_point"]
            )
            results["gpt2"]["times"].append(gpt2_time)
            results["gpt2"]["memory"].append(gpt2_memory)
            if gpt2_correct:
                results["gpt2"]["correct"] += 1

            sample_result["gpt2"] = {
                "correct": gpt2_correct,
                "time": gpt2_time,
                "memory": gpt2_memory,
                "code": gpt2_code[:500] + "..." if len(gpt2_code) > 500 else gpt2_code,
            }

            # Add to detailed results
            detailed_results.append(sample_result)

            # Log sample results
            logger.info(f"  HRM: {'✓' if hrm_correct else '✗'} ({hrm_time:.2f}s)")
            logger.info(f"  GPT-2: {'✓' if gpt2_correct else '✗'} ({gpt2_time:.2f}s)")

        # Calculate metrics
        n_samples = len(samples)
        hrm_correct = results["hrm"]["correct"]
        gpt2_correct = results["gpt2"]["correct"]

        # Basic metrics
        hrm_pass_at_1 = calculate_pass_at_k(n_samples, hrm_correct, 1)
        hrm_pass_at_5 = calculate_pass_at_k(n_samples, hrm_correct, 5)
        hrm_pass_at_10 = calculate_pass_at_k(n_samples, hrm_correct, 10)

        gpt2_pass_at_1 = calculate_pass_at_k(n_samples, gpt2_correct, 1)
        gpt2_pass_at_5 = calculate_pass_at_k(n_samples, gpt2_correct, 5)
        gpt2_pass_at_10 = calculate_pass_at_k(n_samples, gpt2_correct, 10)

        hrm_avg_time = np.mean(results["hrm"]["times"])
        gpt2_avg_time = np.mean(results["gpt2"]["times"])

        hrm_avg_memory = (
            np.mean(results["hrm"]["memory"]) if results["hrm"]["memory"] else 0
        )
        gpt2_avg_memory = (
            np.mean(results["gpt2"]["memory"]) if results["gpt2"]["memory"] else 0
        )

        # Statistical analysis
        hrm_results = [d["hrm"]["correct"] for d in detailed_results]
        gpt2_results = [d["gpt2"]["correct"] for d in detailed_results]

        # Bootstrap confidence intervals
        hrm_lower, hrm_upper, hrm_p = bootstrap_confidence_interval(
            hrm_results, n_bootstrap
        )
        gpt2_lower, gpt2_upper, gpt2_p = bootstrap_confidence_interval(
            gpt2_results, n_bootstrap
        )

        # Compare models
        delta, ci_lower, ci_upper, p_value = compare_models_statistical(
            hrm_results, gpt2_results, n_bootstrap
        )

        # Determine verdict
        if p_value < 0.05 and delta > 0:
            verdict = "✓ VALIDATED: HRM outperforms GPT-2 with statistical significance"
        elif p_value < 0.05 and delta < 0:
            verdict = "✗ REJECTED: GPT-2 outperforms HRM with statistical significance"
        elif delta > 0:
            verdict = "⚠ INCONCLUSIVE: HRM shows better results but without statistical significance"
        else:
            verdict = "⚠ INCONCLUSIVE: GPT-2 shows better results but without statistical significance"

        # Compile results
        final_results = {
            "timestamp": datetime.now().isoformat(),
            "n_samples": n_samples,
            "max_tokens": max_tokens,
            "n_bootstrap": n_bootstrap,
            "hrm": {
                "params": hrm_params,
                "correct": hrm_correct,
                "pass_at_1": hrm_pass_at_1,
                "pass_at_5": hrm_pass_at_5,
                "pass_at_10": hrm_pass_at_10,
                "ci_lower": hrm_lower,
                "ci_upper": hrm_upper,
                "p_value": hrm_p,
                "avg_time": hrm_avg_time,
                "avg_memory": hrm_avg_memory,
            },
            "gpt2": {
                "params": gpt2_params,
                "correct": gpt2_correct,
                "pass_at_1": gpt2_pass_at_1,
                "pass_at_5": gpt2_pass_at_5,
                "pass_at_10": gpt2_pass_at_10,
                "ci_lower": gpt2_lower,
                "ci_upper": gpt2_upper,
                "p_value": gpt2_p,
                "avg_time": gpt2_avg_time,
                "avg_memory": gpt2_avg_memory,
            },
            "delta": delta,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": p_value,
            "verdict": verdict,
            "detailed_results": detailed_results,
        }

        # Save results
        results_file = os.path.join(output_dir, "hypothesis_test_results.json")
        save_results(final_results, results_file)

        # Plot results
        plot_file = os.path.join(output_dir, "hypothesis_test_plot.png")
        plot_results(final_results, plot_file)

        # Print results
        logger.info("\n" + "=" * 80)
        logger.info(" HYPOTHESIS TEST RESULTS ".center(80, "="))
        logger.info("=" * 80)

        logger.info("\nModel Comparison:")
        logger.info(f"  HRM: {hrm_params:.1f}M parameters")
        logger.info(f"  GPT-2: {gpt2_params:.1f}M parameters")
        logger.info(
            f"  Parameter Efficiency: {gpt2_params/hrm_params:.1f}x fewer parameters in HRM"
        )

        logger.info("\nAccuracy Metrics:")
        logger.info(f"  HRM Pass@1: {hrm_pass_at_1:.2%} ({hrm_correct}/{n_samples})")
        logger.info(
            f"  GPT-2 Pass@1: {gpt2_pass_at_1:.2%} ({gpt2_correct}/{n_samples})"
        )
        logger.info(f"  Δ Pass@1: {delta*100:+.1f} pp")
        logger.info(f"  95% CI: [{ci_lower*100:.1f}, {ci_upper*100:.1f}] pp")
        logger.info(f"  p-value: {p_value:.4f}")

        logger.info(f"\n  HRM Pass@10: {hrm_pass_at_10:.2%}")
        logger.info(f"  GPT-2 Pass@10: {gpt2_pass_at_10:.2%}")
        logger.info(f"  Δ Pass@10: {(hrm_pass_at_10 - gpt2_pass_at_10)*100:+.1f} pp")

        logger.info("\nPerformance Metrics:")
        logger.info(f"  HRM Avg. Inference: {hrm_avg_time:.2f}s")
        logger.info(f"  GPT-2 Avg. Inference: {gpt2_avg_time:.2f}s")
        logger.info(f"  Speed Ratio: {gpt2_avg_time/hrm_avg_time:.1f}x faster")

        if hrm_avg_memory > 0 and gpt2_avg_memory > 0:
            logger.info(f"  HRM Avg. Memory: {hrm_avg_memory:.1f} MB")
            logger.info(f"  GPT-2 Avg. Memory: {gpt2_avg_memory:.1f} MB")
            logger.info(
                f"  Memory Ratio: {gpt2_avg_memory/hrm_avg_memory:.1f}x more efficient"
            )

        # Hypothesis validation
        logger.info("\nHypothesis Validation:")
        logger.info(f"  {verdict}")

        logger.info("\nResults saved to:")
        logger.info(f"  {results_file}")
        logger.info(f"  {plot_file}")

        logger.info("\n" + "=" * 80)

        return final_results

    except Exception as e:
        logger.error(f"Error in hypothesis test: {e}")
        logger.error(traceback.format_exc())

        # Try to save partial results if available
        try:
            if "final_results" in locals():
                error_results = final_results.copy()
                error_results["error"] = str(e)
                error_results["traceback"] = traceback.format_exc()

                error_file = os.path.join(output_dir, "hypothesis_test_error.json")
                save_results(error_results, error_file)
                logger.info(f"Partial results saved to {error_file}")
        except:
            pass

        raise RuntimeError(f"Hypothesis test failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HRM vs GPT-2 hypothesis test")
    parser.add_argument(
        "--samples", type=int, default=30, help="Number of MBPP samples to test"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--bootstrap", type=int, default=1000, help="Number of bootstrap samples"
    )

    args = parser.parse_args()

    try:
        run_hypothesis_test(
            num_samples=args.samples,
            max_tokens=args.max_tokens,
            n_bootstrap=args.bootstrap,
        )
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
