#!/usr/bin/env python
"""
Evaluation script for the Hierarchical Reasoning Model (HRM) for code generation.

This script handles the complete evaluation pipeline for the HRM model, including:
- Loading a trained model from checkpoint
- Loading MBPP test data
- Generating code solutions for test problems
- Executing generated code against test cases safely
- Computing Pass@k metrics (Pass@1, Pass@5, Pass@10)
- Providing detailed evaluation results and statistics
- Saving evaluation results to JSON files

Usage:
    python evaluate.py --ckpt checkpoints/hrm-mbpp/best_model.pt --split test --k 1 5 10
    python evaluate.py --ckpt checkpoints/hrm-mbpp/step_10000.pt --split val --num-samples 10
    python evaluate.py --ckpt checkpoints/hrm-mbpp/best_model.pt --split test --output-file results.json
"""

import argparse
import concurrent.futures
import contextlib
import io
import json
import logging
import multiprocessing
import os
import pickle
import random
import resource
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Add parent directory to path to import HRM modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hrm.config import HRMConfig
from hrm.model import HRMModel, create_hrm_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Dataclass for storing evaluation results."""
    
    task_id: str
    prompt: str
    reference: str
    generations: List[str]
    test_results: List[bool]
    execution_times: List[float]
    execution_errors: List[Optional[str]]
    pass_at_k: Dict[int, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class MBPPEvalDataset(Dataset):
    """Dataset for MBPP evaluation data in binary format."""
    
    def __init__(
        self,
        data_path: str,
        max_length: int = 1024,
        pad_token_id: int = 0,
    ):
        """
        Initialize the MBPP evaluation dataset.
        
        Args:
            data_path: Path to the binary data file.
            max_length: Maximum sequence length.
            pad_token_id: ID of the padding token.
        """
        self.data_path = data_path
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        
        # Load data
        logger.info(f"Loading data from {data_path}")
        try:
            with open(data_path, "rb") as f:
                self.examples = pickle.load(f)
            logger.info(f"Loaded {len(self.examples)} examples")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get an example from the dataset.
        
        Args:
            idx: Index of the example.
            
        Returns:
            Dictionary containing the example data.
        """
        example = self.examples[idx]
        
        # Ensure all sequences have the same length
        input_ids = example["input_ids"][:self.max_length]
        attention_mask = example["attention_mask"][:self.max_length]
        
        # Pad sequences if necessary
        if len(input_ids) < self.max_length:
            padding_length = self.max_length - len(input_ids)
            input_ids = input_ids + [self.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        
        return {
            "task_id": example["task_id"],
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "prompt": example["prompt"],
            "completion": example["completion"],
            "test_cases": example.get("test_cases", []),
        }


class CodeExecutor:
    """
    Safe executor for running generated code against test cases.
    
    This class provides a sandboxed environment for executing Python code,
    with proper resource limits, timeouts, and error handling.
    """
    
    def __init__(
        self,
        timeout: int = 5,
        max_memory_mb: int = 1024,
        use_subprocess: bool = True,
    ):
        """
        Initialize the code executor.
        
        Args:
            timeout: Maximum execution time in seconds.
            max_memory_mb: Maximum memory usage in MB.
            use_subprocess: Whether to use subprocess for isolation.
        """
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.use_subprocess = use_subprocess
    
    def _set_resource_limits(self):
        """Set resource limits for the current process."""
        # Set memory limit
        memory_bytes = self.max_memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        
        # Set CPU time limit
        resource.setrlimit(resource.RLIMIT_CPU, (self.timeout, self.timeout))
    
    def _execute_in_subprocess(self, code: str, test_case: str) -> Tuple[bool, float, Optional[str]]:
        """
        Execute code in a subprocess for isolation.
        
        Args:
            code: Code to execute.
            test_case: Test case to run.
            
        Returns:
            Tuple of (success, execution_time, error_message).
        """
        # Create a temporary file with the code and test case
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(code + "\n\n" + test_case)
            temp_file = f.name
        
        try:
            # Run the code in a subprocess with timeout
            start_time = time.time()
            
            # Use subprocess.run with timeout
            process = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            
            execution_time = time.time() - start_time
            
            # Check if the code executed successfully
            if process.returncode == 0:
                return True, execution_time, None
            else:
                return False, execution_time, process.stderr
        
        except subprocess.TimeoutExpired:
            return False, self.timeout, "Execution timed out"
        except Exception as e:
            return False, 0.0, str(e)
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_file)
            except:
                pass
    
    def _execute_in_process(self, code: str, test_case: str) -> Tuple[bool, float, Optional[str]]:
        """
        Execute code in the current process with restricted resources.
        
        Args:
            code: Code to execute.
            test_case: Test case to run.
            
        Returns:
            Tuple of (success, execution_time, error_message).
        """
        # Create a string IO for capturing stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        # Prepare the namespace
        namespace = {}
        
        # Combine code and test case
        full_code = code + "\n\n" + test_case
        
        try:
            # Set resource limits
            self._set_resource_limits()
            
            # Redirect stdout and stderr
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                # Set alarm for timeout
                signal.signal(signal.SIGALRM, lambda signum, frame: TimeoutError("Execution timed out"))
                signal.alarm(self.timeout)
                
                # Execute the code
                start_time = time.time()
                exec(full_code, namespace)
                execution_time = time.time() - start_time
                
                # Disable the alarm
                signal.alarm(0)
            
            return True, execution_time, None
        
        except Exception as e:
            execution_time = time.time() - start_time
            return False, execution_time, f"{type(e).__name__}: {str(e)}\n{stderr_capture.getvalue()}"
        
        finally:
            # Disable the alarm
            signal.alarm(0)
    
    def execute(self, code: str, test_case: str) -> Tuple[bool, float, Optional[str]]:
        """
        Execute code with the given test case.
        
        Args:
            code: Code to execute.
            test_case: Test case to run.
            
        Returns:
            Tuple of (success, execution_time, error_message).
        """
        if self.use_subprocess:
            return self._execute_in_subprocess(code, test_case)
        else:
            return self._execute_in_process(code, test_case)


def compute_pass_at_k(
    n_samples: int,
    n_correct: int,
    k: int,
) -> float:
    """
    Compute pass@k metric.
    
    Args:
        n_samples: Number of samples.
        n_correct: Number of correct samples.
        k: k value for pass@k.
        
    Returns:
        pass@k value.
    """
    if n_samples == 0:
        return 0.0
    
    if n_correct == 0:
        return 0.0
    
    if k > n_samples:
        k = n_samples
    
    # Compute pass@k using the formula from the MBPP paper
    # The probability of getting at least one correct solution in k attempts
    # when there are n_correct correct solutions out of n_samples
    n = n_samples
    c = n_correct
    
    # Edge cases
    if c == 0:
        return 0.0
    if c >= k:
        return 1.0
    
    # Use the exact formula for pass@k
    # 1 - Product_{i=0}^{k-1} (n-c-i)/(n-i)
    result = 1.0
    for i in range(k):
        result *= (n - c - i) / (n - i)
    
    return 1.0 - result


def evaluate_model(
    model: HRMModel,
    dataset: MBPPEvalDataset,
    tokenizer,
    config: HRMConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> List[EvaluationResult]:
    """
    Evaluate the model on the dataset.
    
    Args:
        model: Model to evaluate.
        dataset: Dataset to evaluate on.
        tokenizer: Tokenizer for decoding.
        config: Model configuration.
        args: Command line arguments.
        device: Device to evaluate on.
        
    Returns:
        List of evaluation results.
    """
    logger.info(f"Evaluating model on {len(dataset)} examples")
    
    # Set model to evaluation mode
    model.eval()
    
    # Create code executor
    executor = CodeExecutor(
        timeout=args.timeout,
        max_memory_mb=args.max_memory,
        use_subprocess=args.use_subprocess,
    )
    
    # Initialize results
    results = []
    
    # Process each example
    for idx in tqdm(range(len(dataset)), desc="Evaluating"):
        example = dataset[idx]
        
        # Get input data
        task_id = example["task_id"]
        input_ids = example["input_ids"].unsqueeze(0).to(device)
        attention_mask = example["attention_mask"].unsqueeze(0).to(device)
        prompt = example["prompt"]
        reference = example["completion"]
        test_cases = example["test_cases"]
        
        # Skip if no test cases
        if not test_cases:
            logger.warning(f"Skipping example {task_id} with no test cases")
            continue
        
        # Generate multiple samples
        generations = []
        with torch.no_grad():
            for i in range(args.num_samples):
                # Generate code
                if args.use_beam_search:
                    output_ids = model.beam_search(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=input_ids.size(1) + args.max_new_tokens,
                        num_beams=args.num_beams,
                        temperature=args.temperature,
                        repetition_penalty=args.repetition_penalty,
                        early_stopping=True,
                    )
                else:
                    output_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=input_ids.size(1) + args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        do_sample=True,
                        repetition_penalty=args.repetition_penalty,
                    )
                
                # Decode the generated code
                generated_text = tokenizer.decode(
                    output_ids[0, input_ids.size(1):],
                    skip_special_tokens=True,
                )
                
                # Extract the code part (everything after the prompt)
                prompt_text = tokenizer.decode(
                    input_ids[0],
                    skip_special_tokens=True,
                )
                
                # Add to generations
                generations.append(generated_text)
        
        # Execute test cases for each generation
        test_results = []
        execution_times = []
        execution_errors = []
        
        for generation in generations:
            # Test against all test cases
            generation_passed = True
            generation_time = 0.0
            generation_error = None
            
            for test_case in test_cases:
                success, exec_time, error = executor.execute(generation, test_case)
                
                if not success:
                    generation_passed = False
                    generation_time = exec_time
                    generation_error = error
                    break
                
                generation_time = max(generation_time, exec_time)
            
            # Add results
            test_results.append(generation_passed)
            execution_times.append(generation_time)
            execution_errors.append(generation_error)
        
        # Compute pass@k for different k values
        pass_at_k_values = {}
        for k in args.k:
            if k <= args.num_samples:
                pass_at_k_values[k] = compute_pass_at_k(
                    n_samples=args.num_samples,
                    n_correct=sum(test_results),
                    k=k,
                )
        
        # Create result
        result = EvaluationResult(
            task_id=task_id,
            prompt=prompt,
            reference=reference,
            generations=generations,
            test_results=test_results,
            execution_times=execution_times,
            execution_errors=execution_errors,
            pass_at_k=pass_at_k_values,
        )
        
        # Add to results
        results.append(result)
        
        # Log progress
        if (idx + 1) % 10 == 0:
            logger.info(f"Processed {idx + 1}/{len(dataset)} examples")
            
            # Calculate current pass@k
            current_pass_at_k = {}
            for k in args.k:
                if k <= args.num_samples:
                    current_pass_at_k[k] = np.mean([
                        r.pass_at_k.get(k, 0.0) for r in results
                    ])
            
            logger.info(f"Current pass@k: {current_pass_at_k}")
    
    return results


def save_results(
    results: List[EvaluationResult],
    output_file: str,
    args: argparse.Namespace,
    config: HRMConfig,
) -> None:
    """
    Save evaluation results to a file.
    
    Args:
        results: Evaluation results.
        output_file: Output file path.
        args: Command line arguments.
        config: Model configuration.
    """
    # Convert results to dictionaries
    results_dict = [result.to_dict() for result in results]
    
    # Calculate overall metrics
    overall_metrics = {}
    
    # Pass@k for different k values
    for k in args.k:
        if k <= args.num_samples:
            overall_metrics[f"pass@{k}"] = np.mean([
                result.pass_at_k.get(k, 0.0) for result in results
            ])
    
    # Success rate
    overall_metrics["success_rate"] = np.mean([
        sum(result.test_results) / len(result.test_results)
        for result in results
    ])
    
    # Average execution time
    overall_metrics["avg_execution_time"] = np.mean([
        np.mean([t for t in result.execution_times if t > 0])
        for result in results
        if any(t > 0 for t in result.execution_times)
    ])
    
    # Create output data
    output_data = {
        "results": results_dict,
        "metrics": overall_metrics,
        "args": vars(args),
        "config": config.to_dict() if hasattr(config, "to_dict") else {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # Save to file
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")
    
    # Log overall metrics
    logger.info("Overall metrics:")
    for name, value in overall_metrics.items():
        logger.info(f"  {name}: {value:.4f}")


def load_model_and_tokenizer(
    checkpoint_path: str,
    device: torch.device,
    tokenizer_path: Optional[str] = None,
) -> Tuple[HRMModel, Any, HRMConfig]:
    """
    Load model and tokenizer from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint.
        device: Device to load the model on.
        tokenizer_path: Path to the tokenizer.
        
    Returns:
        Tuple of (model, tokenizer, config).
    """
    logger.info(f"Loading model from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Load config
    if "config" in checkpoint:
        config_dict = checkpoint["config"]
        config = HRMConfig.from_dict(config_dict)
    else:
        raise ValueError(f"Config not found in checkpoint: {checkpoint_path}")
    
    # Create model
    model = create_hrm_model(config)
    
    # Load model weights
    model.load_state_dict(checkpoint["model"])
    
    # Move model to device
    model.to(device)
    
    # Load tokenizer
    tokenizer = None
    if tokenizer_path:
        if TRANSFORMERS_AVAILABLE:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            logger.info(f"Loaded tokenizer from {tokenizer_path}")
        else:
            logger.warning("Transformers not available, skipping tokenizer loading")
    
    return model, tokenizer, config


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a Hierarchical Reasoning Model (HRM) for code generation"
    )
    
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to the checkpoint file",
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Data split to evaluate on",
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to the data file (overrides split)",
    )
    
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to the tokenizer",
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to the output file",
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to generate per example",
    )
    
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="k values for pass@k metric",
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Temperature for sampling",
    )
    
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p for sampling",
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k for sampling",
    )
    
    parser.add_argument(
        "--use-beam-search",
        action="store_true",
        help="Use beam search instead of sampling",
    )
    
    parser.add_argument(
        "--num-beams",
        type=int,
        default=5,
        help="Number of beams for beam search",
    )
    
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty",
    )
    
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate",
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=5,
        help="Timeout for code execution in seconds",
    )
    
    parser.add_argument(
        "--max-memory",
        type=int,
        default=1024,
        help="Maximum memory for code execution in MB",
    )
    
    parser.add_argument(
        "--use-subprocess",
        action="store_true",
        help="Use subprocess for code execution",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model and tokenizer
    model, tokenizer, config = load_model_and_tokenizer(
        checkpoint_path=args.ckpt,
        device=device,
        tokenizer_path=args.tokenizer_path,
    )
    
    # Determine data path
    if args.data_path:
        data_path = args.data_path
    else:
        if args.split == "train":
            data_path = config.data.train_data_path
        elif args.split == "val":
            data_path = config.data.val_data_path
        else:  # test
            data_path = config.data.test_data_path
    
    # Load dataset
    dataset = MBPPEvalDataset(
        data_path=data_path,
        max_length=config.data.max_seq_length,
    )
    
    # Evaluate model
    results = evaluate_model(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        config=config,
        args=args,
        device=device,
    )
    
    # Determine output file
    if args.output_file:
        output_file = args.output_file
    else:
        checkpoint_name = os.path.basename(args.ckpt).split(".")[0]
        output_dir = os.path.join(config.logging.output_dir, "eval_results")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(
            output_dir,
            f"{checkpoint_name}_{args.split}_{time.strftime('%Y%m%d-%H%M%S')}.json",
        )
    
    # Save results
    save_results(
        results=results,
        output_file=output_file,
        args=args,
        config=config,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
    except Exception as e:
        logger.exception(f"Error during evaluation: {e}")
        sys.exit(1)
