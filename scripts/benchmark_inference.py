#!/usr/bin/env python
"""
Inference benchmark script for the Hierarchical Reasoning Model (HRM).

This script measures the inference performance of the HRM model across different
configurations, batch sizes, and generation parameters. It tracks:
- Latency (first token, per-token, total generation time)
- Throughput (tokens/second)
- Resource usage (CPU, GPU, memory)
- Generation quality metrics
- Performance across different generation strategies (beam search, sampling, greedy)

Usage:
    python benchmark_inference.py --config configs/mbpp_base.yaml --dataset mbpp --batch-sizes 1,4,8
    python benchmark_inference.py --config configs/mbpp_base.yaml --generation-params temperature=0.8,top_p=0.95,top_k=50
    python benchmark_inference.py --config configs/mbpp_base.yaml --generation-mode beam --beam-sizes 1,4,10
    python benchmark_inference.py --config configs/mbpp_base.yaml --stress-test --max-new-tokens 1024

Results are saved in a structured JSON format for further analysis and comparison.
"""

import argparse
import json
import logging
import os
import random
import signal
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Import project-specific modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hrm.config import HRMConfig
from hrm.model import HRMModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("benchmark_inference")

# Try importing optional dependencies
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available, skipping wandb integration")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available, CPU monitoring will be limited")

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
    """Monitors CPU, GPU, and memory usage during inference."""

    def __init__(self, track_gpu: bool = True, sampling_interval: float = 0.1):
        """
        Initialize the resource monitor.

        Args:
            track_gpu: Whether to track GPU usage
            sampling_interval: Time between resource usage samples in seconds
        """
        self.track_gpu = track_gpu and torch.cuda.is_available()
        self.sampling_interval = sampling_interval
        
        if PSUTIL_AVAILABLE:
            self.process = psutil.Process(os.getpid())
        else:
            self.process = None
            
        self.running = False
        self.stats = {
            "cpu_percent": [],
            "memory_percent": [],
            "memory_gb": [],
            "timestamps": [],
        }
        
        if self.track_gpu:
            self.stats.update({
                "gpu_utilization": [],
                "gpu_memory_used_gb": [],
                "gpu_memory_total_gb": [],
                "gpu_temperature": [],
            })
            
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
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2.0)
            
    def _monitor_resources(self):
        """Monitor resources at regular intervals."""
        while self.running:
            timestamp = time.time()
            
            # CPU and memory usage
            if PSUTIL_AVAILABLE and self.process:
                self.stats["cpu_percent"].append(self.process.cpu_percent())
                memory_info = self.process.memory_info()
                memory_gb = memory_info.rss / (1024 ** 3)
                self.stats["memory_gb"].append(memory_gb)
                self.stats["memory_percent"].append(self.process.memory_percent())
            else:
                self.stats["cpu_percent"].append(0.0)
                self.stats["memory_gb"].append(0.0)
                self.stats["memory_percent"].append(0.0)
                
            self.stats["timestamps"].append(timestamp)
            
            # GPU usage
            if self.track_gpu:
                if GPUTIL_AVAILABLE:
                    # Refresh GPU info
                    self.gpus = GPUtil.getGPUs()
                    self.gpu = self.gpus[self.gpu_id]
                    
                    self.stats["gpu_utilization"].append(self.gpu.load * 100)
                    self.stats["gpu_memory_used_gb"].append(self.gpu.memoryUsed / 1024)
                    self.stats["gpu_memory_total_gb"].append(self.gpu.memoryTotal / 1024)
                    self.stats["gpu_temperature"].append(self.gpu.temperature)
                elif PYNVML_AVAILABLE:
                    try:
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                        memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                        temperature = pynvml.nvmlDeviceGetTemperature(
                            self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU
                        )
                        
                        self.stats["gpu_utilization"].append(utilization.gpu)
                        self.stats["gpu_memory_used_gb"].append(memory_info.used / (1024 ** 3))
                        self.stats["gpu_memory_total_gb"].append(memory_info.total / (1024 ** 3))
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
    Load the specified dataset for inference benchmarking.
    
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
    logger.info(f"Creating synthetic dataset for {dataset_name} inference benchmarking")
    
    vocab_size = config.model.vocab_size
    seq_length = config.data.context_length  # Use context length for prompts
    dataset_size = 100  # Smaller dataset for inference benchmarking
    
    # Create random input_ids and attention_mask for prompts
    input_ids = torch.randint(0, vocab_size, (dataset_size, seq_length))
    attention_mask = torch.ones_like(input_ids)
    
    # Set random positions in attention_mask to 0 to simulate padding
    for i in range(dataset_size):
        pad_length = torch.randint(0, seq_length // 4, (1,)).item()
        attention_mask[i, -pad_length:] = 0
    
    dataset = TensorDataset(input_ids, attention_mask)
    
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
        shuffle=False,  # No need to shuffle for inference benchmarking
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
        logger.info(f"Updating vocab_size from {config.model.vocab_size} to {vocab_size}")
        config.model.vocab_size = vocab_size
    
    model = HRMModel(config.model)
    
    if torch.cuda.is_available():
        model = model.cuda()
        
    return model


def benchmark_single_forward_pass(
    model: HRMModel,
    dataloader: DataLoader,
    batch_size: int,
    steps: int,
) -> Dict[str, Any]:
    """
    Benchmark a single forward pass (non-generative).
    
    Args:
        model: HRM model
        dataloader: DataLoader for the dataset
        batch_size: Batch size
        steps: Number of steps to benchmark
        
    Returns:
        Dictionary of benchmark results
    """
    model.eval()
    
    # Initialize timing dictionary
    timings = {}
    
    # Initialize metrics
    total_tokens = 0
    
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
            model(batch[0], batch[1])
    
    # Synchronize before starting benchmark
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Main benchmark loop
    logger.info(f"Running {steps} steps of single forward pass...")
    start_time = time.time()
    
    for step in tqdm(range(steps), desc=f"Single forward pass (BS={batch_size})"):
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
            with torch.no_grad():
                outputs = model(batch[0], batch[1])
        
        # Update metrics
        total_tokens += (batch[1] > 0).sum().item()
    
    # Calculate metrics
    end_time = time.time()
    elapsed_time = end_time - start_time
    
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
    tokens_per_second = total_tokens / elapsed_time
    ms_per_step = (elapsed_time * 1000) / steps
    
    results = {
        "throughput": {
            "tokens_per_second": tokens_per_second,
            "ms_per_step": ms_per_step,
        },
        "timing": timing_stats,
        "memory": {
            "peak_allocated_gb": float(torch.cuda.max_memory_allocated() / (1024 ** 3)) if torch.cuda.is_available() else 0,
            "peak_reserved_gb": float(torch.cuda.max_memory_reserved() / (1024 ** 3)) if torch.cuda.is_available() else 0,
        },
    }
    
    return results


def benchmark_generation(
    model: HRMModel,
    dataloader: DataLoader,
    batch_size: int,
    steps: int,
    max_new_tokens: int,
    generation_params: Dict[str, Any],
    generation_mode: str,
    num_beams: int = 1,
    track_token_latency: bool = True,
) -> Dict[str, Any]:
    """
    Benchmark text generation.
    
    Args:
        model: HRM model
        dataloader: DataLoader for the dataset
        batch_size: Batch size
        steps: Number of steps to benchmark
        max_new_tokens: Maximum number of tokens to generate
        generation_params: Parameters for generation (temperature, top_p, top_k)
        generation_mode: Generation mode (greedy, sampling, beam)
        num_beams: Number of beams for beam search
        track_token_latency: Whether to track per-token latency
        
    Returns:
        Dictionary of benchmark results
    """
    model.eval()
    
    # Initialize timing dictionary
    timings = {}
    
    # Initialize token latency tracking
    token_latencies = defaultdict(list)
    
    data_iter = iter(dataloader)
    
    # Warmup
    logger.info("Warming up...")
    for _ in range(min(2, steps // 5)):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        if torch.cuda.is_available():
            batch = [t.cuda() for t in batch]
        
        with torch.no_grad():
            if generation_mode == "beam":
                model.beam_search(
                    batch[0],
                    batch[1],
                    max_new_tokens=min(32, max_new_tokens),
                    num_beams=num_beams,
                    **generation_params
                )
            else:
                model.generate(
                    batch[0],
                    batch[1],
                    max_new_tokens=min(32, max_new_tokens),
                    **generation_params
                )
    
    # Synchronize before starting benchmark
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Main benchmark loop
    generation_desc = f"{'Beam' if generation_mode == 'beam' else 'Greedy/Sampling'} generation"
    if generation_mode == "beam":
        generation_desc += f" (beams={num_beams})"
    logger.info(f"Running {steps} steps of {generation_desc}...")
    
    total_tokens_generated = 0
    total_prompts = 0
    
    for step in tqdm(range(steps), desc=f"{generation_desc} (BS={batch_size})"):
        # Get batch
        with TimingContext("data_loading", timings):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            if torch.cuda.is_available():
                batch = [t.cuda() for t in batch]
        
        # Generation
        with TimingContext("total_generation", timings):
            with torch.no_grad():
                # Track first token latency
                with TimingContext("first_token", timings):
                    # Start generation
                    if generation_mode == "beam":
                        # For beam search, we need to track the first token separately
                        first_token = model.beam_search(
                            batch[0],
                            batch[1],
                            max_new_tokens=1,
                            num_beams=num_beams,
                            **generation_params
                        )
                        
                        # Continue generation
                        with TimingContext("remaining_tokens", timings):
                            generated = model.beam_search(
                                batch[0],
                                batch[1],
                                max_new_tokens=max_new_tokens,
                                num_beams=num_beams,
                                **generation_params
                            )
                    else:
                        # For greedy/sampling, track per-token latency if requested
                        if track_token_latency:
                            # Generate tokens one by one to measure per-token latency
                            input_ids = batch[0]
                            attention_mask = batch[1]
                            
                            for i in range(max_new_tokens):
                                token_start = time.time()
                                
                                # Generate next token
                                next_token = model.generate(
                                    input_ids,
                                    attention_mask,
                                    max_new_tokens=1,
                                    **generation_params
                                )
                                
                                # Calculate token latency
                                if torch.cuda.is_available():
                                    torch.cuda.synchronize()
                                token_end = time.time()
                                token_latency = token_end - token_start
                                token_latencies[i].append(token_latency)
                                
                                # Update input for next iteration
                                input_ids = next_token
                                attention_mask = torch.ones_like(input_ids)
                                
                                # Stop if we generated an EOS token (simplified check)
                                if (next_token[:, -1] == 50256).any():  # Assuming 50256 is EOS
                                    break
                            
                            generated = input_ids
                        else:
                            # Generate all tokens at once
                            generated = model.generate(
                                batch[0],
                                batch[1],
                                max_new_tokens=max_new_tokens,
                                **generation_params
                            )
        
        # Calculate tokens generated
        tokens_generated = generated.size(1) - batch[0].size(1)
        total_tokens_generated += tokens_generated * batch[0].size(0)
        total_prompts += batch[0].size(0)
    
    # Calculate metrics
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
    
    # Calculate token latency statistics
    token_latency_stats = {}
    if track_token_latency and token_latencies:
        for token_idx, latencies in token_latencies.items():
            token_latency_stats[f"token_{token_idx}"] = {
                "mean": float(np.mean(latencies)),
                "median": float(np.median(latencies)),
                "min": float(np.min(latencies)),
                "max": float(np.max(latencies)),
                "std": float(np.std(latencies)),
            }
    
    # Calculate throughput metrics
    total_generation_time = sum(timings["total_generation"])
    tokens_per_second = total_tokens_generated / total_generation_time
    avg_tokens_per_prompt = total_tokens_generated / total_prompts
    
    # First token latency
    first_token_latency = np.mean(timings["first_token"]) if "first_token" in timings else 0
    
    results = {
        "throughput": {
            "tokens_per_second": tokens_per_second,
            "avg_tokens_per_prompt": avg_tokens_per_prompt,
            "prompts_per_second": total_prompts / total_generation_time,
        },
        "latency": {
            "first_token_ms": first_token_latency * 1000,
            "total_generation_ms": (total_generation_time / total_prompts) * 1000,
        },
        "timing": timing_stats,
        "token_latency": token_latency_stats,
        "memory": {
            "peak_allocated_gb": float(torch.cuda.max_memory_allocated() / (1024 ** 3)) if torch.cuda.is_available() else 0,
            "peak_reserved_gb": float(torch.cuda.max_memory_reserved() / (1024 ** 3)) if torch.cuda.is_available() else 0,
        },
    }
    
    return results


def benchmark_inference(
    config: HRMConfig,
    dataset_name: str,
    batch_sizes: List[int],
    steps: int,
    max_new_tokens: int,
    generation_params: Dict[str, Any],
    generation_modes: List[str],
    beam_sizes: List[int],
    track_resources: bool = False,
    track_token_latency: bool = True,
    stress_test: bool = False,
    wandb_run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Benchmark inference performance for different batch sizes and generation parameters.
    
    Args:
        config: HRM configuration
        dataset_name: Name of the dataset to benchmark
        batch_sizes: List of batch sizes to benchmark
        steps: Number of inference steps per batch size
        max_new_tokens: Maximum number of tokens to generate
        generation_params: Parameters for generation (temperature, top_p, top_k)
        generation_modes: List of generation modes to benchmark
        beam_sizes: List of beam sizes to benchmark (for beam search)
        track_resources: Whether to track resource usage
        track_token_latency: Whether to track per-token latency
        stress_test: Whether to run stress tests with long sequences
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
        "generation_params": generation_params,
        "max_new_tokens": max_new_tokens,
        "batch_sizes": {},
        "stress_test": {} if stress_test else None,
    }
    
    # Initialize W&B if available
    if WANDB_AVAILABLE and wandb_run_id:
        if not wandb.run:
            wandb.init(id=wandb_run_id, resume="allow")
        
        # Log config
        wandb.config.update({
            "config_name": config.name,
            "dataset": dataset_name,
            "steps": steps,
            "batch_sizes": batch_sizes,
            "max_new_tokens": max_new_tokens,
            "generation_params": generation_params,
            "generation_modes": generation_modes,
            "beam_sizes": beam_sizes,
        })
    
    # Load dataset
    dataset, vocab_size = load_dataset(config, dataset_name)
    
    # Benchmark each batch size
    for batch_size in batch_sizes:
        logger.info(f"Benchmarking batch size: {batch_size}")
        
        # Create dataloader
        dataloader = create_dataloader(dataset, batch_size, config)
        
        # Create model
        model = create_model(config, vocab_size)
        model.eval()
        
        # Initialize resource monitor if tracking resources
        resource_monitor = None
        if track_resources:
            resource_monitor = ResourceMonitor(track_gpu=True)
            resource_monitor.start()
        
        # Benchmark single forward pass
        logger.info("Benchmarking single forward pass...")
        forward_results = benchmark_single_forward_pass(
            model=model,
            dataloader=dataloader,
            batch_size=batch_size,
            steps=steps,
        )
        
        # Benchmark generation for each mode
        generation_results = {}
        
        for mode in generation_modes:
            mode_results = {}
            
            if mode == "beam":
                # Benchmark beam search with different beam sizes
                for beam_size in beam_sizes:
                    logger.info(f"Benchmarking beam search with {beam_size} beams...")
                    beam_results = benchmark_generation(
                        model=model,
                        dataloader=dataloader,
                        batch_size=batch_size,
                        steps=steps,
                        max_new_tokens=max_new_tokens,
                        generation_params=generation_params,
                        generation_mode="beam",
                        num_beams=beam_size,
                        track_token_latency=False,  # Not applicable for beam search
                    )
                    mode_results[f"beam_{beam_size}"] = beam_results
            else:
                # Benchmark greedy or sampling generation
                logger.info(f"Benchmarking {mode} generation...")
                # Adjust generation parameters based on mode
                mode_params = generation_params.copy()
                if mode == "greedy":
                    mode_params["temperature"] = 0.0
                    mode_params["top_p"] = 1.0
                    mode_params["top_k"] = 0
                
                mode_results = benchmark_generation(
                    model=model,
                    dataloader=dataloader,
                    batch_size=batch_size,
                    steps=steps,
                    max_new_tokens=max_new_tokens,
                    generation_params=mode_params,
                    generation_mode=mode,
                    track_token_latency=track_token_latency,
                )
            
            generation_results[mode] = mode_results
        
        # Stop resource monitor
        if resource_monitor:
            resource_monitor.stop()
            resource_stats = resource_monitor.get_summary_stats()
        else:
            resource_stats = {}
        
        # Store results for this batch size
        batch_results = {
            "forward_pass": forward_results,
            "generation": generation_results,
            "resources": resource_stats,
        }
        
        results["batch_sizes"][str(batch_size)] = batch_results
        
        # Log batch size results to W&B
        if WANDB_AVAILABLE and wandb_run_id:
            flat_metrics = {}
            
            # Log forward pass metrics
            for key, value in forward_results["throughput"].items():
                flat_metrics[f"benchmark/forward/{key}_bs{batch_size}"] = value
            
            # Log generation metrics for each mode
            for mode, mode_results in generation_results.items():
                if mode == "beam":
                    for beam_key, beam_result in mode_results.items():
                        beam_size = beam_key.split("_")[1]
                        for metric_key, metric_value in beam_result["throughput"].items():
                            flat_metrics[f"benchmark/generation/{mode}/{beam_key}/{metric_key}_bs{batch_size}"] = metric_value
                        for metric_key, metric_value in beam_result["latency"].items():
                            flat_metrics[f"benchmark/generation/{mode}/{beam_key}/{metric_key}_bs{batch_size}"] = metric_value
                else:
                    for metric_key, metric_value in mode_results["throughput"].items():
                        flat_metrics[f"benchmark/generation/{mode}/{metric_key}_bs{batch_size}"] = metric_value
                    for metric_key, metric_value in mode_results["latency"].items():
                        flat_metrics[f"benchmark/generation/{mode}/{metric_key}_bs{batch_size}"] = metric_value
            
            # Log resource metrics if available
            if resource_stats:
                for key, stats in resource_stats.items():
                    for stat_name, value in stats.items():
                        flat_metrics[f"benchmark/resources/{key}_{stat_name}_bs{batch_size}"] = value
            
            wandb.log(flat_metrics)
        
        # Clean up to prevent CUDA OOM
        del model, dataloader
        torch.cuda.empty_cache()
        
        logger.info(f"Batch size {batch_size} results:")
        logger.info(f"  Forward pass: {forward_results['throughput']['tokens_per_second']:.2f} tokens/s")
        for mode, mode_results in generation_results.items():
            if mode == "beam":
                for beam_key, beam_result in mode_results.items():
                    logger.info(f"  {beam_key}: {beam_result['throughput']['tokens_per_second']:.2f} tokens/s")
            else:
                logger.info(f"  {mode}: {mode_results['throughput']['tokens_per_second']:.2f} tokens/s")
    
    # Run stress tests if requested
    if stress_test:
        logger.info("Running stress tests with long sequences...")
        
        # Create a special dataset with long sequences
        long_seq_lengths = [512, 1024, 2048, 4096]
        
        for seq_length in long_seq_lengths:
            if seq_length > config.model.max_position_embeddings:
                logger.warning(f"Skipping sequence length {seq_length} as it exceeds model's max position embeddings")
                continue
                
            logger.info(f"Testing sequence length: {seq_length}")
            
            # Create synthetic dataset with this sequence length
            from torch.utils.data import TensorDataset
            
            dataset_size = 10  # Small dataset for stress testing
            input_ids = torch.randint(0, vocab_size, (dataset_size, seq_length))
            attention_mask = torch.ones_like(input_ids)
            
            dataset = TensorDataset(input_ids, attention_mask)
            dataloader = create_dataloader(dataset, 1, config)  # Batch size 1 for stress test
            
            # Create model
            model = create_model(config, vocab_size)
            model.eval()
            
            # Initialize resource monitor
            if track_resources:
                resource_monitor = ResourceMonitor(track_gpu=True)
                resource_monitor.start()
            
            # Benchmark forward pass
            forward_results = benchmark_single_forward_pass(
                model=model,
                dataloader=dataloader,
                batch_size=1,
                steps=min(5, steps),  # Fewer steps for stress test
            )
            
            # Benchmark generation (greedy only for stress test)
            generation_results = benchmark_generation(
                model=model,
                dataloader=dataloader,
                batch_size=1,
                steps=min(5, steps),  # Fewer steps for stress test
                max_new_tokens=min(512, max_new_tokens),  # Limit generation length for stress test
                generation_params={"temperature": 0.0, "top_p": 1.0, "top_k": 0},  # Greedy
                generation_mode="greedy",
                track_token_latency=False,  # Skip token latency for stress test
            )
            
            # Stop resource monitor
            if resource_monitor:
                resource_monitor.stop()
                resource_stats = resource_monitor.get_summary_stats()
            else:
                resource_stats = {}
            
            # Store results for this sequence length
            results["stress_test"][str(seq_length)] = {
                "forward_pass": forward_results,
                "generation": generation_results,
                "resources": resource_stats,
            }
            
            # Log stress test results to W&B
            if WANDB_AVAILABLE and wandb_run_id:
                flat_metrics = {
                    f"stress_test/seq_length": seq_length,
                    f"stress_test/forward/tokens_per_second": forward_results["throughput"]["tokens_per_second"],
                    f"stress_test/generation/tokens_per_second": generation_results["throughput"]["tokens_per_second"],
                    f"stress_test/generation/first_token_ms": generation_results["latency"]["first_token_ms"],
                    f"stress_test/generation/total_generation_ms": generation_results["latency"]["total_generation_ms"],
                }
                
                if resource_stats:
                    for key, stats in resource_stats.items():
                        for stat_name, value in stats.items():
                            flat_metrics[f"stress_test/resources/{key}_{stat_name}"] = value
                
                wandb.log(flat_metrics)
            
            # Clean up
            del model, dataloader
            torch.cuda.empty_cache()
            
            logger.info(f"Sequence length {seq_length} results:")
            logger.info(f"  Forward pass: {forward_results['throughput']['tokens_per_second']:.2f} tokens/s")
            logger.info(f"  Generation: {generation_results['throughput']['tokens_per_second']:.2f} tokens/s")
    
    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark HRM inference performance")
    
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
        default="1,4,8,16",
        help="Comma-separated list of batch sizes to benchmark",
    )
    
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of inference steps per batch size",
    )
    
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate",
    )
    
    parser.add_argument(
        "--generation-params",
        type=str,
        default="temperature=0.8,top_p=0.95,top_k=50",
        help="Comma-separated list of generation parameters in key=value format",
    )
    
    parser.add_argument(
        "--generation-modes",
        type=str,
        default="greedy,sampling,beam",
        help="Comma-separated list of generation modes to benchmark",
    )
    
    parser.add_argument(
        "--beam-sizes",
        type=str,
        default="1,4,10",
        help="Comma-separated list of beam sizes to benchmark (for beam search)",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="inference_benchmark_results.json",
        help="Path to save the benchmark results",
    )
    
    parser.add_argument(
        "--track-resources",
        action="store_true",
        help="Track CPU, GPU, and memory usage during inference",
    )
    
    parser.add_argument(
        "--track-token-latency",
        action="store_true",
        help="Track per-token latency (slower but more detailed)",
    )
    
    parser.add_argument(
        "--stress-test",
        action="store_true",
        help="Run stress tests with long sequences",
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


def parse_generation_params(params_str: str) -> Dict[str, Any]:
    """Parse generation parameters from string."""
    params = {}
    
    for param in params_str.split(","):
        key, value = param.split("=")
        
        # Convert value to appropriate type
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        elif "." in value:
            try:
                value = float(value)
            except ValueError:
                pass
        else:
            try:
                value = int(value)
            except ValueError:
                pass
        
        params[key] = value
    
    return params


def main():
    """Main function to run the benchmark."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Parse batch sizes
    batch_sizes = [int(bs) for bs in args.batch_sizes.split(",")]
    
    # Parse generation modes
    generation_modes = args.generation_modes.split(",")
    
    # Parse beam sizes
    beam_sizes = [int(bs) for bs in args.beam_sizes.split(",")]
    
    # Parse generation parameters
    generation_params = parse_generation_params(args.generation_params)
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = HRMConfig.from_yaml(args.config)
    
    # Print benchmark parameters
    logger.info(f"Benchmarking dataset: {args.dataset}")
    logger.info(f"Batch sizes: {batch_sizes}")
    logger.info(f"Steps per batch size: {args.steps}")
    logger.info(f"Max new tokens: {args.max_new_tokens}")
    logger.info(f"Generation parameters: {generation_params}")
    logger.info(f"Generation modes: {generation_modes}")
    logger.info(f"Beam sizes: {beam_sizes}")
    logger.info(f"Tracking resources: {args.track_resources}")
    logger.info(f"Tracking token latency: {args.track_token_latency}")
    logger.info(f"Running stress tests: {args.stress_test}")
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
        results = benchmark_inference(
            config=config,
            dataset_name=args.dataset,
            batch_sizes=batch_sizes,
            steps=args.steps,
            max_new_tokens=args.max_new_tokens,
            generation_params=generation_params,
            generation_modes=generation_modes,
            beam_sizes=beam_sizes,
            track_resources=args.track_resources,
            track_token_latency=args.track_token_latency,
            stress_test=args.stress_test,
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
                name=f"inference-benchmark-results-{args.dataset}",
                type="benchmark",
                description=f"Inference benchmark results for {args.dataset}",
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
