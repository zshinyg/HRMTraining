#!/usr/bin/env python
"""
Training script for the Hierarchical Reasoning Model (HRM) for code generation.

This script handles the complete training pipeline for the HRM model, including:
- Data loading from preprocessed binary files
- Model initialization with configuration
- Training loop with proper loss computation and backpropagation
- Evaluation on validation set at regular intervals
- Checkpoint saving and loading
- Learning rate scheduling
- Mixed precision training
- Distributed training
- Logging with wandb and tensorboard
- Early stopping and best model tracking
- Memory optimization techniques

Usage:
    python train.py --config configs/hrm/mbpp_base.yaml --data-path data/mbpp/train.bin
    python train.py --config configs/hrm/mbpp_base.yaml --resume checkpoints/hrm-mbpp/step_10000.pt
    python train.py --config configs/hrm/mbpp_base.yaml --distributed --world-size 4
"""

import argparse
import json
import logging
import os
import pickle
import random
import re
import shutil
import signal
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    LinearLR,
    ReduceLROnPlateau,
)
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    import wandb

    WANDB_AVAILABLE = True
    # Monitoring utilities will use wandb if available
except ImportError:
    WANDB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Enable PyTorch anomaly detection to pinpoint in-place operation errors
# ---------------------------------------------------------------------------
# NOTE: This MUST be placed after torch import and before any model
# construction / forward passes so that stack-traces are captured early.
torch.autograd.set_detect_anomaly(True)

# ---------------------------------------------------------------------------
# Monitoring utilities (Priority-1 instrumentation)
# ---------------------------------------------------------------------------
# These lightweight helpers are placed in scripts/monitoring_utils.py by
# Research Droid.  Importing them is safe even if the module is absent in
# certain CI contexts because training will only run when the file exists.
# ---------------------------------------------------------------------------
try:
    from monitoring_utils import (
        init_monitoring,
        collect_system_metrics,
        check_anomalies,
        log_metrics as log_system_metrics,
    )

    MONITORING_AVAILABLE = True
except ImportError:  # Fallback so training still runs without monitoring
    MONITORING_AVAILABLE = False

# Add parent directory to path to import HRM modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hrm.config import HRMConfig, SchedulerType
from hrm.model import HRMModel, create_hrm_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MBPPDataset(Dataset):
    """Dataset for MBPP data in binary format."""

    def __init__(
        self,
        data_path: str,
        max_length: int = 1024,
        pad_token_id: int = 0,
    ):
        """
        Initialize the MBPP dataset.

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
        input_ids = example["input_ids"][: self.max_length]
        attention_mask = example["attention_mask"][: self.max_length]
        labels = example["labels"][: self.max_length]

        # Pad sequences if necessary
        if len(input_ids) < self.max_length:
            padding_length = self.max_length - len(input_ids)
            input_ids = input_ids + [self.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            labels = labels + [-100] * padding_length

        return {
            "task_id": example["task_id"],
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "prompt": example["prompt"],
            "completion": example["completion"],
        }


def create_optimizer(
    model: nn.Module,
    config: HRMConfig,
) -> Optimizer:
    """
    Create an optimizer for the model.

    Args:
        model: Model to optimize.
        config: Training configuration.

    Returns:
        Optimizer.
    """
    # Get parameters with custom learning rates
    optimizer_grouped_parameters = []

    # Word embeddings
    if config.training.embedding_lr is not None:
        optimizer_grouped_parameters.append(
            {
                "params": [
                    p for n, p in model.named_parameters() if "token_embeddings" in n
                ],
                "lr": config.training.embedding_lr,
                "weight_decay": config.training.weight_decay,
            }
        )

    # High-level module
    if config.training.high_level_lr is not None:
        optimizer_grouped_parameters.append(
            {
                "params": [p for n, p in model.named_parameters() if "high_level" in n],
                "lr": config.training.high_level_lr,
                "weight_decay": config.training.weight_decay,
            }
        )

    # Low-level module
    if config.training.low_level_lr is not None:
        optimizer_grouped_parameters.append(
            {
                "params": [p for n, p in model.named_parameters() if "low_level" in n],
                "lr": config.training.low_level_lr,
                "weight_decay": config.training.weight_decay,
            }
        )

    # All other parameters
    if not optimizer_grouped_parameters:
        optimizer_grouped_parameters.append(
            {
                "params": [p for n, p in model.named_parameters()],
                "lr": config.training.learning_rate,
                "weight_decay": config.training.weight_decay,
            }
        )
    else:
        # Add parameters not covered by custom learning rates
        param_names = set()
        for group in optimizer_grouped_parameters:
            for p in group["params"]:
                param_names.add(id(p))

        remaining_params = [
            p for n, p in model.named_parameters() if id(p) not in param_names
        ]

        if remaining_params:
            optimizer_grouped_parameters.append(
                {
                    "params": remaining_params,
                    "lr": config.training.learning_rate,
                    "weight_decay": config.training.weight_decay,
                }
            )

    # Create optimizer
    if config.training.optimizer == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
    elif config.training.optimizer == "adam":
        optimizer = torch.optim.Adam(
            optimizer_grouped_parameters,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
    elif config.training.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            optimizer_grouped_parameters,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            momentum=0.9,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.training.optimizer}")

    return optimizer


def create_scheduler(
    optimizer: Optimizer,
    config: HRMConfig,
    num_training_steps: int,
) -> Union[
    torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
]:
    """
    Create a learning rate scheduler.

    Args:
        optimizer: Optimizer to schedule.
        config: Training configuration.
        num_training_steps: Total number of training steps.

    Returns:
        Learning rate scheduler.
    """
    # Calculate warmup steps
    if config.training.warmup_ratio > 0:
        config.training.warmup_steps = int(
            num_training_steps * config.training.warmup_ratio
        )

    # Create scheduler
    if config.training.scheduler == SchedulerType.COSINE:
        return CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps,
            eta_min=0.0,
        )
    elif config.training.scheduler == SchedulerType.LINEAR:
        return LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=num_training_steps,
        )
    elif config.training.scheduler == SchedulerType.CONSTANT:
        return LambdaLR(optimizer, lambda _: 1.0)
    elif config.training.scheduler == SchedulerType.WARMUP_COSINE:
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.training.warmup_steps,
            num_training_steps=num_training_steps,
        )
    elif config.training.scheduler == SchedulerType.WARMUP_LINEAR:
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.training.warmup_steps,
            num_training_steps=num_training_steps,
        )
    elif config.training.scheduler == SchedulerType.WARMUP_CONSTANT:
        return get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.training.warmup_steps,
        )
    else:
        raise ValueError(f"Unsupported scheduler: {config.training.scheduler}")


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer: Optimizer to schedule.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.
        last_epoch: The index of the last epoch when resuming training.

    Returns:
        Learning rate scheduler.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer: Optimizer to schedule.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.
        num_cycles: Number of cycles for cosine decay.
        last_epoch: The index of the last epoch when resuming training.

    Returns:
        Learning rate scheduler.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + np.cos(np.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_constant_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create a schedule with a learning rate that increases linearly from 0 to the initial lr set in the optimizer during
    warmup, then remains constant.

    Args:
        optimizer: Optimizer to schedule.
        num_warmup_steps: Number of warmup steps.
        last_epoch: The index of the last epoch when resuming training.

    Returns:
        Learning rate scheduler.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch)


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


def setup_distributed(
    rank: int,
    world_size: int,
    backend: str = "nccl",
) -> None:
    """
    Initialize distributed training.

    Args:
        rank: Rank of the current process.
        world_size: Number of processes.
        backend: Backend to use for distributed training.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize process group
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup_distributed() -> None:
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_model_size(model: nn.Module) -> int:
    """
    Get the size of the model in bytes.

    Args:
        model: Model to measure.

    Returns:
        Size of the model in bytes.
    """
    return sum(p.numel() * p.element_size() for p in model.parameters())


def format_metrics(metrics: Dict[str, float]) -> str:
    """
    Format metrics for logging.

    Args:
        metrics: Dictionary of metrics.

    Returns:
        Formatted string.
    """
    return ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any,
    scaler: Optional[GradScaler],
    config: HRMConfig,
    step: int,
    epoch: int,
    metrics: Dict[str, float],
    is_best: bool,
    output_dir: str,
) -> None:
    """
    Save a checkpoint.

    Args:
        model: Model to save.
        optimizer: Optimizer to save.
        scheduler: Scheduler to save.
        scaler: Gradient scaler to save.
        config: Configuration to save.
        step: Current step.
        epoch: Current epoch.
        metrics: Current metrics.
        is_best: Whether this is the best model so far.
        output_dir: Directory to save the checkpoint.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Unwrap model if using DDP
    if isinstance(model, DDP):
        model_to_save = model.module
    else:
        model_to_save = model

    # Create checkpoint
    checkpoint = {
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "config": config.to_dict(),
        "step": step,
        "epoch": epoch,
        "metrics": metrics,
    }

    # Save checkpoint
    checkpoint_path = os.path.join(output_dir, f"step_{step}.pt")
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")

    # Save as best model if applicable
    if is_best:
        best_path = os.path.join(output_dir, "best_model.pt")
        shutil.copyfile(checkpoint_path, best_path)
        logger.info(f"Saved best model to {best_path}")

    # Save latest checkpoint
    latest_path = os.path.join(output_dir, "latest.pt")
    shutil.copyfile(checkpoint_path, latest_path)

    # Remove old checkpoints if needed
    if config.logging.save_total_limit > 0:
        checkpoints = sorted(
            [f for f in os.listdir(output_dir) if re.match(r"step_\d+\.pt", f)],
            key=lambda x: int(x.split("_")[1].split(".")[0]),
        )

        # Keep the latest save_total_limit checkpoints
        if len(checkpoints) > config.logging.save_total_limit:
            for checkpoint_to_remove in checkpoints[: -config.logging.save_total_limit]:
                os.remove(os.path.join(output_dir, checkpoint_to_remove))
                logger.info(f"Removed old checkpoint: {checkpoint_to_remove}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[GradScaler] = None,
    map_location: str = "cpu",
) -> Tuple[
    nn.Module,
    Optional[Optimizer],
    Optional[Any],
    Optional[GradScaler],
    int,
    int,
    Dict[str, float],
]:
    """
    Load a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint.
        model: Model to load.
        optimizer: Optimizer to load.
        scheduler: Scheduler to load.
        scaler: Gradient scaler to load.
        map_location: Device to load the checkpoint to.

    Returns:
        Tuple of (model, optimizer, scheduler, scaler, step, epoch, metrics).
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    # Load model
    if isinstance(model, DDP):
        model.module.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint["model"])

    # Load optimizer
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Load scheduler
    if (
        scheduler is not None
        and "scheduler" in checkpoint
        and checkpoint["scheduler"] is not None
    ):
        scheduler.load_state_dict(checkpoint["scheduler"])

    # Load scaler
    if (
        scaler is not None
        and "scaler" in checkpoint
        and checkpoint["scaler"] is not None
    ):
        scaler.load_state_dict(checkpoint["scaler"])

    # Get step and epoch
    step = checkpoint.get("step", 0)
    epoch = checkpoint.get("epoch", 0)

    # Get metrics
    metrics = checkpoint.get("metrics", {})

    logger.info(f"Loaded checkpoint from step {step}, epoch {epoch}")

    return model, optimizer, scheduler, scaler, step, epoch, metrics


def log_samples(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    tokenizer,
    step: int,
    writer: Optional[SummaryWriter] = None,
    prefix: str = "train",
    num_samples: int = 2,
) -> None:
    """
    Log sample predictions.

    Args:
        model: Model to use for predictions.
        batch: Batch of data.
        tokenizer: Tokenizer for decoding.
        step: Current step.
        writer: TensorBoard writer.
        prefix: Prefix for logging.
        num_samples: Number of samples to log.
    """
    # Ensure model is in evaluation mode
    model.eval()

    # Get inputs
    input_ids = batch["input_ids"][:num_samples].to(model.device)
    attention_mask = batch["attention_mask"][:num_samples].to(model.device)
    labels = batch["labels"][:num_samples].to(model.device)

    # Generate predictions
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

    # Get logits and predictions
    logits = outputs["logits"]
    predictions = torch.argmax(logits, dim=-1)

    # Decode inputs, labels, and predictions
    for i in range(min(num_samples, input_ids.size(0))):
        # Get input prompt
        input_prompt = tokenizer.decode(
            input_ids[i][attention_mask[i] == 1],
            skip_special_tokens=True,
        )

        # Get target
        target_ids = labels[i][labels[i] != -100]
        target = tokenizer.decode(target_ids, skip_special_tokens=True)

        # Get prediction
        pred_ids = predictions[i][labels[i] != -100]
        prediction = tokenizer.decode(pred_ids, skip_special_tokens=True)

        # Log to tensorboard
        if writer is not None:
            writer.add_text(
                f"{prefix}/sample_{i}/input",
                input_prompt,
                global_step=step,
            )
            writer.add_text(
                f"{prefix}/sample_{i}/target",
                target,
                global_step=step,
            )
            writer.add_text(
                f"{prefix}/sample_{i}/prediction",
                prediction,
                global_step=step,
            )

        # Log to console
        logger.info(f"{prefix} Sample {i}:")
        logger.info(f"Input: {input_prompt[:100]}...")
        logger.info(f"Target: {target[:100]}...")
        logger.info(f"Prediction: {prediction[:100]}...")
        logger.info("---")

    # Set model back to training mode
    model.train()


def evaluate(
    model: nn.Module,
    eval_dataloader: DataLoader,
    device: torch.device,
    config: HRMConfig,
    step: int,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
    tokenizer=None,
) -> Dict[str, float]:
    """
    Evaluate the model.

    Args:
        model: Model to evaluate.
        eval_dataloader: DataLoader for evaluation data.
        device: Device to evaluate on.
        config: Evaluation configuration.
        step: Current step.
        epoch: Current epoch.
        writer: TensorBoard writer.
        tokenizer: Tokenizer for decoding.

    Returns:
        Dictionary of evaluation metrics.
    """
    logger.info(f"Evaluating at step {step}, epoch {epoch}")

    # Set model to evaluation mode
    model.eval()

    # Initialize metrics
    total_loss = 0.0
    total_samples = 0

    # Evaluate
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            # Move batch to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                return_dict=True,
            )

            # Get loss
            loss = outputs["loss"]

            # Update metrics
            batch_size = batch["input_ids"].size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Stop evaluation if max steps reached
            if batch_idx >= config.evaluation.eval_steps:
                break

    # Calculate metrics
    metrics = {
        "eval/loss": total_loss / total_samples,
    }

    # Log metrics
    logger.info(f"Evaluation metrics: {format_metrics(metrics)}")

    # Log to tensorboard
    if writer is not None:
        for name, value in metrics.items():
            writer.add_scalar(name, value, global_step=step)

    # Log sample predictions
    if tokenizer is not None:
        log_samples(
            model=model,
            batch=next(iter(eval_dataloader)),
            tokenizer=tokenizer,
            step=step,
            writer=writer,
            prefix="eval",
            num_samples=config.logging.num_log_samples,
        )

    # Set model back to training mode
    model.train()

    return metrics


def train_epoch(
    model: nn.Module,
    train_dataloader: DataLoader,
    optimizer: Optimizer,
    scheduler: Any,
    scaler: Optional[GradScaler],
    device: torch.device,
    config: HRMConfig,
    epoch: int,
    global_step: int,
    writer: Optional[SummaryWriter] = None,
    tokenizer=None,
    eval_dataloader: Optional[DataLoader] = None,
) -> Tuple[int, Dict[str, float], bool]:
    """
    Train for one epoch.

    Args:
        model: Model to train.
        train_dataloader: DataLoader for training data.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        scaler: Gradient scaler for mixed precision training.
        device: Device to train on.
        config: Training configuration.
        epoch: Current epoch.
        global_step: Current global step.
        writer: TensorBoard writer.
        tokenizer: Tokenizer for decoding.
        eval_dataloader: DataLoader for evaluation data.

    Returns:
        Tuple of (global_step, best_metrics, early_stop).
    """
    # Set model to training mode
    model.train()

    # Initialize metrics
    epoch_loss = 0.0
    epoch_samples = 0

    # Initialize best metrics and early stopping
    best_metrics = {"eval/loss": float("inf")}
    early_stop = False

    # Get world size for distributed training
    world_size = 1
    if dist.is_initialized():
        world_size = dist.get_world_size()

    # Create progress bar
    progress_bar = tqdm(
        total=len(train_dataloader),
        desc=f"Epoch {epoch}",
        disable=not (
            dist.is_initialized() and dist.get_rank() == 0 or not dist.is_initialized()
        ),
    )

    # Initialize gradient accumulation
    optimizer.zero_grad()

    # Train
    for batch_idx, batch in enumerate(train_dataloader):
        # Move batch to device
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Forward pass with mixed precision if enabled
        if config.training.use_mixed_precision and scaler is not None:
            with autocast(
                dtype=(
                    torch.float16
                    if config.training.mixed_precision_dtype == "float16"
                    else torch.bfloat16
                )
            ):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    return_dict=True,
                )
                loss = outputs["loss"]

                # Scale loss for gradient accumulation
                loss = loss / config.training.gradient_accumulation_steps
        else:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                return_dict=True,
            )
            loss = outputs["loss"]

            # Scale loss for gradient accumulation
            loss = loss / config.training.gradient_accumulation_steps

        # Backward pass with mixed precision if enabled
        if config.training.use_mixed_precision and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update metrics
        batch_size = batch["input_ids"].size(0)
        epoch_loss += (
            loss.item() * config.training.gradient_accumulation_steps * batch_size
        )
        epoch_samples += batch_size

        # Optimizer step if gradient accumulation is complete
        if (batch_idx + 1) % config.training.gradient_accumulation_steps == 0:
            # Clip gradients
            if config.training.max_grad_norm > 0:
                if config.training.use_mixed_precision and scaler is not None:
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.training.max_grad_norm,
                )

            # Optimizer step with mixed precision if enabled
            if config.training.use_mixed_precision and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            # Scheduler step
            if scheduler is not None:
                if not isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step()

            # Zero gradients
            optimizer.zero_grad()

            # Increment global step
            global_step += 1

            # Log metrics
            if global_step % config.logging.log_interval == 0:
                # Calculate metrics
                metrics = {
                    "train/loss": epoch_loss / epoch_samples,
                    "train/lr": (
                        scheduler.get_last_lr()[0]
                        if scheduler is not None
                        and not isinstance(scheduler, ReduceLROnPlateau)
                        else optimizer.param_groups[0]["lr"]
                    ),
                }

                # Collect system metrics (Research Droid Priority-1)
                if MONITORING_AVAILABLE:
                    system_metrics = collect_system_metrics()
                    metrics.update(system_metrics)

                    # Check for anomalies
                    anomalies = check_anomalies(metrics)
                    if anomalies:
                        logger.warning(f"ANOMALIES DETECTED: {anomalies}")

                # Log to console
                logger.info(
                    f"Epoch {epoch}, Step {global_step}: {format_metrics(metrics)}"
                )

                # Log to tensorboard
                if writer is not None:
                    for name, value in metrics.items():
                        writer.add_scalar(name, value, global_step=global_step)

                # Log to wandb
                if config.logging.use_wandb and WANDB_AVAILABLE:
                    wandb.log(metrics, step=global_step)

                # Log system metrics to wandb
                if (
                    MONITORING_AVAILABLE
                    and config.logging.use_wandb
                    and WANDB_AVAILABLE
                ):
                    log_system_metrics(metrics, step=global_step)

                # Log gradient norm
                if config.logging.log_grad_norm:
                    grad_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            grad_norm += p.grad.data.norm(2).item() ** 2
                    grad_norm = grad_norm**0.5

                    if writer is not None:
                        writer.add_scalar(
                            "train/grad_norm", grad_norm, global_step=global_step
                        )

                    if config.logging.use_wandb and WANDB_AVAILABLE:
                        wandb.log({"train/grad_norm": grad_norm}, step=global_step)

                # Log memory usage
                if config.logging.log_memory:
                    if torch.cuda.is_available():
                        memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024
                        memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024

                        if writer is not None:
                            writer.add_scalar(
                                "memory/allocated_mb",
                                memory_allocated,
                                global_step=global_step,
                            )
                            writer.add_scalar(
                                "memory/reserved_mb",
                                memory_reserved,
                                global_step=global_step,
                            )

                        if config.logging.use_wandb and WANDB_AVAILABLE:
                            wandb.log(
                                {
                                    "memory/allocated_mb": memory_allocated,
                                    "memory/reserved_mb": memory_reserved,
                                },
                                step=global_step,
                            )

            # Evaluate
            if (
                global_step % config.evaluation.eval_interval == 0
                and eval_dataloader is not None
            ):
                eval_metrics = evaluate(
                    model=model,
                    eval_dataloader=eval_dataloader,
                    device=device,
                    config=config,
                    step=global_step,
                    epoch=epoch,
                    writer=writer,
                    tokenizer=tokenizer,
                )

                # Log to wandb
                if config.logging.use_wandb and WANDB_AVAILABLE:
                    wandb.log(eval_metrics, step=global_step)

                # Update best metrics and check for early stopping
                if eval_metrics["eval/loss"] < best_metrics["eval/loss"]:
                    best_metrics = eval_metrics

                    # Save best model
                    if config.logging.save_best:
                        save_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                            config=config,
                            step=global_step,
                            epoch=epoch,
                            metrics=eval_metrics,
                            is_best=True,
                            output_dir=config.logging.output_dir,
                        )

                # Scheduler step for ReduceLROnPlateau
                if scheduler is not None and isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(eval_metrics["eval/loss"])

            # Save checkpoint
            if global_step % config.logging.save_interval == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    config=config,
                    step=global_step,
                    epoch=epoch,
                    metrics={"train/loss": epoch_loss / epoch_samples},
                    is_best=False,
                    output_dir=config.logging.output_dir,
                )

            # Log samples
            if (
                config.logging.log_samples
                and global_step % (config.logging.log_interval * 10) == 0
                and tokenizer is not None
            ):
                log_samples(
                    model=model,
                    batch=batch,
                    tokenizer=tokenizer,
                    step=global_step,
                    writer=writer,
                    prefix="train",
                    num_samples=config.logging.num_log_samples,
                )

        # Update progress bar
        progress_bar.update(1)
        progress_bar.set_postfix(
            {"loss": loss.item() * config.training.gradient_accumulation_steps}
        )

    # Close progress bar
    progress_bar.close()

    # Calculate epoch metrics
    epoch_metrics = {
        "train/epoch_loss": epoch_loss / epoch_samples,
        "train/epoch": epoch,
    }

    # Log epoch metrics
    logger.info(f"Epoch {epoch} metrics: {format_metrics(epoch_metrics)}")

    # Log to tensorboard
    if writer is not None:
        for name, value in epoch_metrics.items():
            writer.add_scalar(name, value, global_step=global_step)

    # Log to wandb
    if config.logging.use_wandb and WANDB_AVAILABLE:
        wandb.log(epoch_metrics, step=global_step)

    return global_step, best_metrics, early_stop


def train(
    config: HRMConfig,
    train_dataloader: DataLoader,
    eval_dataloader: Optional[DataLoader],
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any,
    device: torch.device,
    tokenizer=None,
    resume_from: Optional[str] = None,
) -> None:
    """
    Train the model.

    Args:
        config: Training configuration.
        train_dataloader: DataLoader for training data.
        eval_dataloader: DataLoader for evaluation data.
        model: Model to train.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Device to train on.
        tokenizer: Tokenizer for decoding.
        resume_from: Path to checkpoint to resume from.
    """
    # Create output directory
    os.makedirs(config.logging.output_dir, exist_ok=True)

    # Create tensorboard writer
    writer = None
    if config.logging.use_tensorboard:
        log_dir = os.path.join(
            config.logging.log_dir,
            datetime.now().strftime("%Y%m%d-%H%M%S"),
        )
        writer = SummaryWriter(log_dir=log_dir)
        logger.info(f"TensorBoard logs will be saved to {log_dir}")

    # Initialize wandb
    if config.logging.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=config.logging.wandb_project,
            entity=config.logging.wandb_entity,
            name=config.logging.wandb_run_name
            or f"{config.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=config.to_dict(),
        )
        logger.info(f"Wandb initialized: {wandb.run.name}")

        # Initialize system monitoring (Research Droid Priority-1)
        if MONITORING_AVAILABLE:
            init_monitoring()
            logger.info("System monitoring initialized - real-time metrics enabled")

    # Initialize mixed precision
    scaler = None
    if config.training.use_mixed_precision:
        scaler = GradScaler()
        logger.info("Mixed precision training enabled")

    # Move model to device
    model.to(device)

    # Wrap model with DDP if distributed
    if dist.is_initialized():
        model = DDP(model, device_ids=[device], output_device=device)
        logger.info(
            f"Model wrapped with DDP: {dist.get_rank()}/{dist.get_world_size()}"
        )

    # Calculate total training steps
    total_steps = len(train_dataloader) * config.training.epochs
    logger.info(f"Total training steps: {total_steps}")

    # Resume from checkpoint if specified
    global_step = 0
    start_epoch = 0
    best_metrics = {"eval/loss": float("inf")}

    if resume_from is not None:
        model, optimizer, scheduler, scaler, global_step, start_epoch, best_metrics = (
            load_checkpoint(
                checkpoint_path=resume_from,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                map_location=device,
            )
        )
        logger.info(f"Resumed from checkpoint: {resume_from}")

    # Save config
    config_path = os.path.join(config.logging.output_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)
    logger.info(f"Saved config to {config_path}")

    # Train
    early_stop = False
    for epoch in range(start_epoch, config.training.epochs):
        # Set epoch for distributed sampler
        if dist.is_initialized() and isinstance(
            train_dataloader.sampler, DistributedSampler
        ):
            train_dataloader.sampler.set_epoch(epoch)

        # Train for one epoch
        global_step, best_metrics, early_stop = train_epoch(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            config=config,
            epoch=epoch,
            global_step=global_step,
            writer=writer,
            tokenizer=tokenizer,
            eval_dataloader=eval_dataloader,
        )

        # Stop training if early stopping triggered
        if early_stop:
            logger.info("Early stopping triggered")
            break

    # Final evaluation
    if eval_dataloader is not None:
        eval_metrics = evaluate(
            model=model,
            eval_dataloader=eval_dataloader,
            device=device,
            config=config,
            step=global_step,
            epoch=config.training.epochs,
            writer=writer,
            tokenizer=tokenizer,
        )

        # Log to wandb
        if config.logging.use_wandb and WANDB_AVAILABLE:
            wandb.log(eval_metrics, step=global_step)

    # Save final model
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        config=config,
        step=global_step,
        epoch=config.training.epochs,
        metrics=best_metrics,
        is_best=False,
        output_dir=config.logging.output_dir,
    )

    # Close tensorboard writer
    if writer is not None:
        writer.close()

    # Close wandb
    if config.logging.use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    logger.info("Training completed")


def train_worker(
    rank: int,
    world_size: int,
    config: HRMConfig,
    args: argparse.Namespace,
) -> None:
    """
    Worker function for distributed training.

    Args:
        rank: Rank of the current process.
        world_size: Number of processes.
        config: Training configuration.
        args: Command line arguments.
    """
    # Initialize distributed training
    if world_size > 1:
        setup_distributed(rank, world_size)
        logger.info(f"Initialized distributed training: rank {rank}/{world_size}")

    # Set device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Set random seed
    set_seed(config.seed + rank)

    # Create datasets
    train_dataset = MBPPDataset(
        data_path=args.data_path,
        max_length=config.data.max_seq_length,
    )

    eval_dataset = None
    if args.eval_data_path:
        eval_dataset = MBPPDataset(
            data_path=args.eval_data_path,
            max_length=config.data.max_seq_length,
        )

    # Create samplers
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )

        eval_sampler = None
        if eval_dataset is not None:
            eval_sampler = DistributedSampler(
                eval_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
            )
    else:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = None if eval_dataset is None else None

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.global_batch_size // world_size,
        sampler=train_sampler,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    eval_dataloader = None
    if eval_dataset is not None:
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=config.training.global_batch_size // world_size,
            sampler=eval_sampler,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
        )

    # Create model
    model = create_hrm_model(config)
    logger.info(
        f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Create optimizer
    optimizer = create_optimizer(model, config)

    # Calculate total training steps
    total_steps = len(train_dataloader) * config.training.epochs

    # Create scheduler
    scheduler = create_scheduler(optimizer, config, total_steps)

    # Load tokenizer
    tokenizer = None
    if args.tokenizer_path:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        logger.info(f"Loaded tokenizer from {args.tokenizer_path}")

    # Train
    try:
        train(
            config=config,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            tokenizer=tokenizer,
            resume_from=args.resume,
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted")
    finally:
        # Clean up distributed training
        if world_size > 1:
            cleanup_distributed()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a Hierarchical Reasoning Model (HRM) for code generation"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the training data",
    )

    parser.add_argument(
        "--eval-data-path",
        type=str,
        default=None,
        help="Path to the evaluation data",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save outputs (overrides config)",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to the tokenizer",
    )

    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed training",
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        help="Number of processes for distributed training",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )

    return parser.parse_args()


def main() -> None:
    """Main function."""
    # Parse arguments
    args = parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    config = HRMConfig.from_dict(config_dict)

    # Override config with command line arguments
    if args.output_dir:
        config.logging.output_dir = args.output_dir

    if args.seed is not None:
        config.seed = args.seed

    # Set random seed
    set_seed(config.seed)

    # Log configuration
    logger.info(f"Configuration: {config.name} (version {config.version})")
    logger.info(f"Description: {config.description}")
    logger.info(f"Model parameters: {config.model.total_params:,}")

    # Handle distributed training
    if args.distributed:
        logger.info(f"Launching distributed training with {args.world_size} processes")
        mp.spawn(
            train_worker,
            args=(args.world_size, config, args),
            nprocs=args.world_size,
            join=True,
        )
    else:
        # Train on a single process
        train_worker(0, 1, config, args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.exception(f"Error during training: {e}")
        sys.exit(1)
