#!/usr/bin/env python
"""
Training benchmark script for the Hierarchical Reasoning Model (HRM).

This script measures the training performance of the HRM model across different
configurations and batch sizes. It tracks resource usage (CPU, GPU, memory),
throughput (samples/second, tokens/second), and provides detailed timing
breakdowns for different training phases.

Usage:
    python benchmark_training.py --config configs/mbpp_base.yaml --dataset mbpp --steps 100 --batch-sizes 8,16,32,64
    python benchmark_training.py --config configs/mbpp_base.yaml --track-resources --wandb-run-id benchmark_20250805

Results are saved in a structured JSON format for further analysis and comparison.
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import psutil
import torch
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Import project-specific modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hrm.config import HRMConfig
from hrm.model import HRMModel

# Configure logging (write under logs/)
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "benchmark_training.log"),
    ],
)
logger = logging.getLogger("benchmark_training")

# Try importing optional dependencies
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available, skipping wandb integration")

try:
    import GPUtil

    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    logger.warning("GPUtil not available, GPU monitoring will be limited")

try:
    import pynvml

    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except (ImportError, pynvml.NVMLError):
    PYNVML_AVAILABLE = False
    logger.warning("pynvml not available, detailed GPU monitoring will be limited")


class ResourceMonitor:
    """Monitors CPU, GPU, and memory usage during training."""

    def __init__(self, track_gpu: bool = True, sampling_interval: float = 0.5):
        """
        Initialize the resource monitor.

        Args:
            track_gpu: Whether to track GPU usage
            sampling_interval: Time between resource usage samples in seconds
        """
        self.track_gpu = track_gpu and torch.cuda.is_available()
        self.sampling_interval = sampling_interval
        self.process = psutil.Process(os.getpid())
        self.running = False
        self.stats = {
            "cpu_percent": [],
            "memory_percent": [],
            "memory_gb": [],
            "timestamps": [],
        }

        if self.track_gpu:
            self.stats.update(
                {
                    "gpu_utilization": [],
                    "gpu_memory_used_gb": [],
                    "gpu_memory_total_gb": [],
                    "gpu_temperature": [],
                }
            )

            # Get GPU device information
            if GPUTIL_AVAILABLE:
                self.gpus = GPUtil.getGPUs()
                if len(self.gpus) > 0:
                    self.gpu_id = torch.cuda.current_device()
                    self.gpu = self.gpus[self.gpu_id]
                else:
                    self.track_gpu = False
            elif PYNVML_AVAILABLE:
                self.gpu_id = torch.cuda.current_device()
                try:
                    self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
                except pynvml.NVMLError:
                    self.track_gpu = False
            else:
                self.track_gpu = False

    def start(self):
        """Start monitoring resources in a separate thread."""
        import threading

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop(self):
        """Stop monitoring resources."""
        self.running = False
        if hasattr(self, "monitor_thread"):
            self.monitor_thread.join(timeout=2.0)

    def _monitor_resources(self):
        """Monitor resources at regular intervals."""
        while self.running:
            timestamp = time.time()

            # CPU and memory usage
            self.stats["cpu_percent"].append(self.process.cpu_percent())
            memory_info = self.process.memory_info()
            memory_gb = memory_info.rss / (1024**3)
            self.stats["memory_gb"].append(memory_gb)
            self.stats["memory_percent"].append(self.process.memory_percent())
            self.stats["timestamps"].append(timestamp)

            # GPU usage
            if self.track_gpu:
                if GPUTIL_AVAILABLE:
                    # Refresh GPU info
                    self.gpus = GPUtil.getGPUs()
                    self.gpu = self.gpus[self.gpu_id]

                    self.stats["gpu_utilization"].append(self.gpu.load * 100)
                    self.stats["gpu_memory_used_gb"].append(self.gpu.memoryUsed / 1024)
                    self.stats["gpu_memory_total_gb"].append(
                        self.gpu.memoryTotal / 1024
                    )
                    self.stats["gpu_temperature"].append(self.gpu.temperature)
                elif PYNVML_AVAILABLE:
                    try:
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(
                            self.gpu_handle
                        )
                        memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                        temperature = pynvml.nvmlDeviceGetTemperature(
                            self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU
                        )

                        self.stats["gpu_utilization"].append(utilization.gpu)
                        self.stats["gpu_memory_used_gb"].append(
                            memory_info.used / (1024**3)
                        )
                        self.stats["gpu_memory_total_gb"].append(
                            memory_info.total / (1024**3)
                        )
                        self.stats["gpu_temperature"].append(temperature)
                    except pynvml.NVMLError as e:
                        logger.warning(f"Error getting GPU stats: {e}")

            time.sleep(self.sampling_interval)

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Calculate summary statistics from the collected data.

        Returns:
            Dictionary of summary statistics
        """
        summary = {}

        # Helper function to calculate stats safely
        def safe_stats(values):
            if not values:
                return {"mean": 0, "max": 0, "min": 0, "std": 0}
            return {
                "mean": float(np.mean(values)),
                "max": float(np.max(values)),
                "min": float(np.min(values)),
                "std": float(np.std(values)),
            }

        # Calculate stats for each metric
        for key in self.stats:
            if key != "timestamps" and self.stats[key]:
                summary[key] = safe_stats(self.stats[key])

        return summary


class TimingContext:
    """Context manager for timing code execution."""

    def __init__(self, name: str, timings: Dict[str, List[float]]):
        """
        Initialize the timing context.

        Args:
            name: Name of the timing context
            timings: Dictionary to store timing results
        """
        self.name = name
        self.timings = timings
        if name not in self.timings:
            self.timings[name] = []

    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and record the elapsed time."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - self.start_time
        self.timings[self.name].append(elapsed)


def load_dataset(config: HRMConfig, dataset_name: str) -> Tuple[Dataset, int]:
    """
    Load the specified dataset.

    Args:
        config: HRM configuration
        dataset_name: Name of the dataset to load

    Returns:
        Tuple of (dataset, vocab_size)
    """
    # This is a placeholder for actual dataset loading
    # In a real implementation, you would load the dataset based on the name
    from torch.utils.data import TensorDataset

    # For benchmarking purposes, we'll create a synthetic dataset
    logger.info(f"Creating synthetic dataset for {dataset_name}")

    vocab_size = config.model.vocab_size
    seq_length = config.data.max_seq_length
    dataset_size = 10000  # Large enough for benchmarking

    # Create random input_ids and attention_mask
    input_ids = torch.randint(0, vocab_size, (dataset_size, seq_length))
    attention_mask = torch.ones_like(input_ids)
    labels = torch.randint(0, vocab_size, (dataset_size, seq_length))

    # Set random positions in attention_mask to 0 to simulate padding
    for i in range(dataset_size):
        pad_length = torch.randint(0, seq_length // 4, (1,)).item()
        attention_mask[i, -pad_length:] = 0
        labels[i, -pad_length:] = -100  # Masked positions

    dataset = TensorDataset(input_ids, attention_mask, labels)

    return dataset, vocab_size


def create_dataloader(
    dataset: Dataset, batch_size: int, config: HRMConfig
) -> DataLoader:
    """
    Create a DataLoader for the dataset.

    Args:
        dataset: The dataset to load
        batch_size: Batch size for the DataLoader
        config: HRM configuration

    Returns:
        DataLoader for the dataset
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )


def create_model(config: HRMConfig, vocab_size: int) -> HRMModel:
    """
    Create an HRM model instance.

    Args:
        config: HRM configuration
        vocab_size: Size of the vocabulary

    Returns:
        Initialized HRM model
    """
    # Update vocab size in config if needed
    if vocab_size != config.model.vocab_size:
        logger.info(
            f"Updating vocab_size from {config.model.vocab_size} to {vocab_size}"
        )
        config.model.vocab_size = vocab_size

    model = HRMModel(config.model)

    if torch.cuda.is_available():
        model = model.cuda()

    return model


def create_optimizer(model: HRMModel, config: HRMConfig) -> torch.optim.Optimizer:
    """
    Create an optimizer for the model.

    Args:
        model: HRM model
        config: HRM configuration

    Returns:
        Optimizer for the model
    """
    from torch.optim import AdamW

    # Get optimizer parameters with weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config.training.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=config.training.learning_rate,
        eps=1e-8,
    )

    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer, config: HRMConfig, total_steps: int
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create a learning rate scheduler.

    Args:
        optimizer: Optimizer for the model
        config: HRM configuration
        total_steps: Total number of training steps

    Returns:
        Learning rate scheduler
    """
    from torch.optim.lr_scheduler import LambdaLR

    # Determine warmup steps
    if config.training.warmup_ratio > 0:
        warmup_steps = int(total_steps * config.training.warmup_ratio)
    else:
        warmup_steps = config.training.warmup_steps

    # Define scheduler function based on config
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        if config.training.scheduler.value == "constant":
            return 1.0

        if config.training.scheduler.value == "linear":
            return max(
                0.0,
                float(total_steps - current_step)
                / float(max(1, total_steps - warmup_steps)),
            )

        if config.training.scheduler.value in ["cosine", "warmup_cosine"]:
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

        # Default: constant
        return 1.0

    scheduler = LambdaLR(optimizer, lr_lambda)

    return scheduler


def benchmark_training(
    config: HRMConfig,
    dataset_name: str,
    batch_sizes: List[int],
    steps: int,
    track_resources: bool = False,
    wandb_run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Benchmark training performance for different batch sizes.

    Args:
        config: HRM configuration
        dataset_name: Name of the dataset to benchmark
        batch_sizes: List of batch sizes to benchmark
        steps: Number of training steps per batch size
        track_resources: Whether to track resource usage
        wandb_run_id: W&B run ID for logging

    Returns:
        Dictionary of benchmark results
    """
    results = {
        "config_name": config.name,
        "dataset": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "steps_per_batch_size": steps,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "torch_version": torch.__version__,
        "batch_sizes": {},
    }

    # Initialize W&B if available
    if WANDB_AVAILABLE and wandb_run_id:
        if not wandb.run:
            wandb.init(id=wandb_run_id, resume="allow")

        # Log config
        wandb.config.update(
            {
                "config_name": config.name,
                "dataset": dataset_name,
                "steps": steps,
                "batch_sizes": batch_sizes,
            }
        )

    # Load dataset
    dataset, vocab_size = load_dataset(config, dataset_name)

    # Benchmark each batch size
    for batch_size in batch_sizes:
        logger.info(f"Benchmarking batch size: {batch_size}")

        # Create dataloader
        dataloader = create_dataloader(dataset, batch_size, config)

        # Create model
        model = create_model(config, vocab_size)
        model.train()

        # Create optimizer and scheduler
        optimizer = create_optimizer(model, config)
        scheduler = create_scheduler(optimizer, config, steps)

        # Initialize mixed precision if enabled
        scaler = GradScaler() if config.training.use_mixed_precision else None

        # Initialize resource monitor if tracking resources
        resource_monitor = None
        if track_resources:
            resource_monitor = ResourceMonitor(track_gpu=True)
            resource_monitor.start()

        # Initialize timing dictionary
        timings = {}

        # Training loop
        total_tokens = 0
        total_samples = 0
        total_loss = 0.0

        data_iter = iter(dataloader)

        # Warmup
        logger.info("Warming up...")
        for _ in range(min(5, steps // 10)):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            if torch.cuda.is_available():
                batch = [t.cuda() for t in batch]

            with torch.no_grad():
                model(batch[0], batch[1], labels=batch[2])

        # Synchronize before starting benchmark
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Main benchmark loop
        logger.info(f"Running {steps} steps...")
        start_time = time.time()

        for step in tqdm(range(steps), desc=f"Batch size {batch_size}"):
            # Get batch
            with TimingContext("data_loading", timings):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

                if torch.cuda.is_available():
                    batch = [t.cuda() for t in batch]

            # Forward pass
            with TimingContext("forward", timings):
                if config.training.use_mixed_precision:
                    with autocast():
                        outputs = model(batch[0], batch[1], labels=batch[2])
                        loss = outputs.loss
                else:
                    outputs = model(batch[0], batch[1], labels=batch[2])
                    loss = outputs.loss

            # Backward pass
            with TimingContext("backward", timings):
                if config.training.use_mixed_precision:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            # Optimizer step
            with TimingContext("optimizer", timings):
                if config.training.use_mixed_precision:
                    scaler.unscale_(optimizer)

                # Gradient clipping
                clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

                if config.training.use_mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

            # Update metrics
            total_loss += loss.item()
            total_samples += batch[0].size(0)
            total_tokens += (batch[1] > 0).sum().item()

            # Log to W&B
            if WANDB_AVAILABLE and wandb_run_id and step % 10 == 0:
                wandb.log(
                    {
                        f"train/loss_bs{batch_size}": loss.item(),
                        f"train/lr_bs{batch_size}": scheduler.get_last_lr()[0],
                        f"train/step_bs{batch_size}": step,
                    }
                )

        # Calculate metrics
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Stop resource monitor
        if resource_monitor:
            resource_monitor.stop()
            resource_stats = resource_monitor.get_summary_stats()
        else:
            resource_stats = {}

        # Calculate timing statistics
        timing_stats = {}
        for key, values in timings.items():
            timing_stats[key] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "std": float(np.std(values)),
                "total": float(np.sum(values)),
            }

        # Calculate throughput metrics
        samples_per_second = total_samples / elapsed_time
        tokens_per_second = total_tokens / elapsed_time
        ms_per_step = (elapsed_time * 1000) / steps
        avg_loss = total_loss / steps

        # Store results for this batch size
        batch_results = {
            "throughput": {
                "samples_per_second": samples_per_second,
                "tokens_per_second": tokens_per_second,
                "ms_per_step": ms_per_step,
            },
            "metrics": {
                "loss": avg_loss,
            },
            "timing": timing_stats,
            "resources": resource_stats,
            "memory": {
                "peak_allocated_gb": (
                    float(torch.cuda.max_memory_allocated() / (1024**3))
                    if torch.cuda.is_available()
                    else 0
                ),
                "peak_reserved_gb": (
                    float(torch.cuda.max_memory_reserved() / (1024**3))
                    if torch.cuda.is_available()
                    else 0
                ),
            },
        }

        results["batch_sizes"][str(batch_size)] = batch_results

        # Log batch size results to W&B
        if WANDB_AVAILABLE and wandb_run_id:
            flat_metrics = {
                f"benchmark/throughput/samples_per_second_bs{batch_size}": samples_per_second,
                f"benchmark/throughput/tokens_per_second_bs{batch_size}": tokens_per_second,
                f"benchmark/throughput/ms_per_step_bs{batch_size}": ms_per_step,
                f"benchmark/metrics/loss_bs{batch_size}": avg_loss,
            }

            # Add timing metrics
            for key, stats in timing_stats.items():
                for stat_name, value in stats.items():
                    if stat_name != "total":  # Skip total to reduce clutter
                        flat_metrics[
                            f"benchmark/timing/{key}_{stat_name}_bs{batch_size}"
                        ] = value

            # Add resource metrics if available
            if resource_stats:
                for key, stats in resource_stats.items():
                    for stat_name, value in stats.items():
                        flat_metrics[
                            f"benchmark/resources/{key}_{stat_name}_bs{batch_size}"
                        ] = value

            # Add memory metrics
            for key, value in batch_results["memory"].items():
                flat_metrics[f"benchmark/memory/{key}_bs{batch_size}"] = value

            wandb.log(flat_metrics)

        # Clean up to prevent CUDA OOM
        del model, optimizer, scheduler, dataloader
        if scaler:
            del scaler
        torch.cuda.empty_cache()

        logger.info(f"Batch size {batch_size} results:")
        logger.info(
            f"  Throughput: {samples_per_second:.2f} samples/s, {tokens_per_second:.2f} tokens/s"
        )
        logger.info(f"  Time per step: {ms_per_step:.2f} ms")
        logger.info(f"  Avg loss: {avg_loss:.4f}")

    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark HRM training performance")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration YAML file",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="mbpp",
        help="Dataset name to use for benchmarking",
    )

    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="8,16,32,64",
        help="Comma-separated list of batch sizes to benchmark",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of training steps per batch size",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Path to save the benchmark results",
    )

    parser.add_argument(
        "--track-resources",
        action="store_true",
        help="Track CPU, GPU, and memory usage during training",
    )

    parser.add_argument(
        "--wandb-run-id",
        type=str,
        default=None,
        help="W&B run ID for logging benchmark results",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    """Main function to run the benchmark."""
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Parse batch sizes
    batch_sizes = [int(bs) for bs in args.batch_sizes.split(",")]

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = HRMConfig.from_yaml(args.config)

    # Print benchmark parameters
    logger.info(f"Benchmarking dataset: {args.dataset}")
    logger.info(f"Batch sizes: {batch_sizes}")
    logger.info(f"Steps per batch size: {args.steps}")
    logger.info(f"Tracking resources: {args.track_resources}")
    logger.info(f"W&B run ID: {args.wandb_run_id}")

    # Set up signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Received interrupt signal, shutting down...")
        if WANDB_AVAILABLE and wandb.run:
            wandb.finish()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Run benchmark
        results = benchmark_training(
            config=config,
            dataset_name=args.dataset,
            batch_sizes=batch_sizes,
            steps=args.steps,
            track_resources=args.track_resources,
            wandb_run_id=args.wandb_run_id,
        )

        # Add metadata to results
        results["args"] = vars(args)
        results["config"] = config.to_dict()

        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Benchmark results saved to {output_path}")

        # Finish W&B run
        if WANDB_AVAILABLE and args.wandb_run_id and wandb.run:
            # Upload results file as artifact
            artifact = wandb.Artifact(
                name=f"benchmark-results-{args.dataset}",
                type="benchmark",
                description=f"Training benchmark results for {args.dataset}",
            )
            artifact.add_file(output_path)
            wandb.log_artifact(artifact)

            wandb.finish()

    except Exception as e:
        logger.error(f"Error during benchmarking: {e}")
        logger.error(traceback.format_exc())

        if WANDB_AVAILABLE and wandb.run:
            wandb.finish(exit_code=1)

        sys.exit(1)


if __name__ == "__main__":
    main()
