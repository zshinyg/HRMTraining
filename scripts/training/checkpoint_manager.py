#!/usr/bin/env python
"""
Checkpoint Manager for HRM Training

This module provides a robust checkpointing system for training neural networks,
with a focus on preventing data loss, ensuring checkpoint integrity, and enabling
easy recovery from training interruptions.

Features:
- Atomic checkpoint saving (prevents corruption during saving)
- Checkpoint integrity validation with hash verification
- Automatic backup of critical checkpoints
- Incremental checkpointing with configurable frequency
- Checkpoint rotation and pruning strategies
- Disk space monitoring to prevent storage issues
- Distributed training support with rank-aware operations
- Compression options to reduce storage requirements
- Recovery mechanisms with automatic fallback
- Partial state loading (model only, optimizer only, etc.)

Usage:
    from scripts.training.checkpoint_manager import CheckpointManager

    # Initialize the manager
    ckpt_manager = CheckpointManager(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir="checkpoints/run_1",
        save_interval_steps=1000,
        max_checkpoints=5,
    )

    # During training
    for step in range(num_steps):
        # Training code...

        # Save checkpoint
        if step % save_interval == 0:
            ckpt_manager.save_checkpoint(
                step=step,
                epoch=epoch,
                metrics={"loss": loss.item()},
                is_best=is_best,
            )

    # Resume training
    model, optimizer, scheduler, step, epoch = ckpt_manager.load_latest_checkpoint()
"""

import hashlib
import json
import logging
import os
import re
import shutil
import signal
import tempfile
import time
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class CheckpointFormat(Enum):
    """Enum for checkpoint formats."""

    STANDARD = "standard"  # Regular PyTorch save format
    SAFETENSORS = "safetensors"  # Using safetensors library if available
    SHARDED = "sharded"  # Split into multiple files for large models


class CheckpointRotationStrategy(Enum):
    """Enum for checkpoint rotation strategies."""

    KEEP_LAST_N = "keep_last_n"  # Keep the last N checkpoints
    EXPONENTIAL_BACKOFF = (
        "exponential_backoff"  # Keep checkpoints with exponential spacing
    )
    KEEP_BEST_N = "keep_best_n"  # Keep the N best checkpoints based on a metric
    KEEP_MILESTONE = (
        "keep_milestone"  # Keep milestone checkpoints (e.g., every 10k steps)
    )


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""

    path: str
    step: int
    epoch: int
    timestamp: float
    metrics: Dict[str, float]
    is_best: bool = False
    is_milestone: bool = False
    is_backup: bool = False
    hash_value: Optional[str] = None
    format: CheckpointFormat = CheckpointFormat.STANDARD
    size_bytes: int = 0
    is_valid: bool = True
    validation_error: Optional[str] = None


class CheckpointManager:
    """
    Manager for checkpoint saving, loading, and validation.

    This class provides a robust checkpointing system with features to prevent
    data loss, validate checkpoint integrity, and enable easy recovery from
    training interruptions.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        output_dir: str = "checkpoints",
        save_interval_steps: int = 1000,
        save_interval_minutes: Optional[int] = None,
        max_checkpoints: int = 5,
        keep_best_metric: Optional[str] = None,
        rotation_strategy: Union[
            str, CheckpointRotationStrategy
        ] = CheckpointRotationStrategy.KEEP_LAST_N,
        use_atomic_save: bool = True,
        validate_on_save: bool = True,
        validate_on_load: bool = True,
        backup_best: bool = True,
        backup_latest: bool = True,
        compression: Optional[str] = None,
        checkpoint_format: Union[str, CheckpointFormat] = CheckpointFormat.STANDARD,
        disk_monitor_enabled: bool = True,
        min_free_space_gb: float = 5.0,
        distributed_aware: bool = False,
        master_rank: int = 0,
        on_save_start: Optional[Callable] = None,
        on_save_end: Optional[Callable] = None,
        on_load_start: Optional[Callable] = None,
        on_load_end: Optional[Callable] = None,
    ):
        """
        Initialize the checkpoint manager.

        Args:
            model: Model to checkpoint
            optimizer: Optimizer to checkpoint
            scheduler: Learning rate scheduler to checkpoint
            scaler: Gradient scaler for mixed precision training
            output_dir: Directory to save checkpoints
            save_interval_steps: Steps between checkpoints
            save_interval_minutes: Minutes between checkpoints (alternative to steps)
            max_checkpoints: Maximum number of checkpoints to keep
            keep_best_metric: Metric name to use for keeping best checkpoints
            rotation_strategy: Strategy for rotating checkpoints
            use_atomic_save: Whether to use atomic save (temp file + rename)
            validate_on_save: Whether to validate checkpoints after saving
            validate_on_load: Whether to validate checkpoints when loading
            backup_best: Whether to keep a backup of the best checkpoint
            backup_latest: Whether to keep a backup of the latest checkpoint
            compression: Compression format (None, 'gzip', 'zip')
            checkpoint_format: Format for saving checkpoints
            disk_monitor_enabled: Whether to monitor disk space
            min_free_space_gb: Minimum free space in GB to allow checkpointing
            distributed_aware: Whether to be aware of distributed training
            master_rank: Rank that should save checkpoints in distributed training
            on_save_start: Callback before saving checkpoint
            on_save_end: Callback after saving checkpoint
            on_load_start: Callback before loading checkpoint
            on_load_end: Callback after loading checkpoint
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler

        # Setup output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint settings
        self.save_interval_steps = save_interval_steps
        self.save_interval_minutes = save_interval_minutes
        self.max_checkpoints = max_checkpoints
        self.keep_best_metric = keep_best_metric

        # Convert string to enum if needed
        if isinstance(rotation_strategy, str):
            try:
                self.rotation_strategy = CheckpointRotationStrategy(rotation_strategy)
            except ValueError:
                logger.warning(
                    f"Invalid rotation strategy: {rotation_strategy}, using KEEP_LAST_N"
                )
                self.rotation_strategy = CheckpointRotationStrategy.KEEP_LAST_N
        else:
            self.rotation_strategy = rotation_strategy

        # Save and validation settings
        self.use_atomic_save = use_atomic_save
        self.validate_on_save = validate_on_save
        self.validate_on_load = validate_on_load
        self.backup_best = backup_best
        self.backup_latest = backup_latest

        # Compression and format settings
        self.compression = compression

        # Convert string to enum if needed
        if isinstance(checkpoint_format, str):
            try:
                self.checkpoint_format = CheckpointFormat(checkpoint_format)
            except ValueError:
                logger.warning(
                    f"Invalid checkpoint format: {checkpoint_format}, using STANDARD"
                )
                self.checkpoint_format = CheckpointFormat.STANDARD
        else:
            self.checkpoint_format = checkpoint_format

        # Disk monitoring
        self.disk_monitor_enabled = disk_monitor_enabled
        self.min_free_space_gb = min_free_space_gb

        # Distributed training settings
        self.distributed_aware = distributed_aware
        self.master_rank = master_rank
        self.is_master = True  # Default for non-distributed
        if self.distributed_aware and dist.is_initialized():
            self.is_master = dist.get_rank() == self.master_rank

        # Callbacks
        self.on_save_start = on_save_start
        self.on_save_end = on_save_end
        self.on_load_start = on_load_start
        self.on_load_end = on_load_end

        # State
        self.last_checkpoint_time = time.time()
        self.checkpoint_history: List[CheckpointMetadata] = []
        self.best_metric_value: Optional[float] = None
        self.best_checkpoint: Optional[CheckpointMetadata] = None
        self.latest_checkpoint: Optional[CheckpointMetadata] = None

        # Load checkpoint history
        self._load_checkpoint_history()

        # Register signal handlers for graceful shutdown
        self._register_signal_handlers()

        logger.info(
            f"Checkpoint manager initialized with output directory: {self.output_dir}"
        )
        logger.info(
            f"Rotation strategy: {self.rotation_strategy.value}, Max checkpoints: {self.max_checkpoints}"
        )

    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""

        # Save checkpoint on termination signals
        def signal_handler(sig, frame):
            logger.warning(f"Received signal {sig}, saving emergency checkpoint...")
            self.save_checkpoint(
                step=0,  # Will be overridden if we have a latest checkpoint
                epoch=0,  # Will be overridden if we have a latest checkpoint
                metrics={},
                is_emergency=True,
                tag="emergency",
            )
            # Re-raise the signal after saving
            signal.signal(sig, signal.SIG_DFL)
            os.kill(os.getpid(), sig)

        # Register for common termination signals
        signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination request

    def _load_checkpoint_history(self):
        """Load checkpoint history from metadata file."""
        metadata_path = self.output_dir / "checkpoint_metadata.json"
        if not metadata_path.exists():
            logger.info("No checkpoint history found")
            return

        try:
            with open(metadata_path, "r") as f:
                history_data = json.load(f)

            # Convert to CheckpointMetadata objects
            self.checkpoint_history = []
            for item in history_data:
                metadata = CheckpointMetadata(
                    path=item["path"],
                    step=item["step"],
                    epoch=item["epoch"],
                    timestamp=item["timestamp"],
                    metrics=item["metrics"],
                    is_best=item.get("is_best", False),
                    is_milestone=item.get("is_milestone", False),
                    is_backup=item.get("is_backup", False),
                    hash_value=item.get("hash_value"),
                    format=CheckpointFormat(item.get("format", "standard")),
                    size_bytes=item.get("size_bytes", 0),
                    is_valid=item.get("is_valid", True),
                    validation_error=item.get("validation_error"),
                )

                # Check if file exists
                if not os.path.exists(metadata.path):
                    logger.warning(f"Checkpoint file not found: {metadata.path}")
                    continue

                self.checkpoint_history.append(metadata)

                # Update best and latest checkpoints
                if metadata.is_best and (
                    self.best_checkpoint is None
                    or metadata.step > self.best_checkpoint.step
                ):
                    self.best_checkpoint = metadata

                if (
                    self.latest_checkpoint is None
                    or metadata.step > self.latest_checkpoint.step
                ):
                    self.latest_checkpoint = metadata

                # Update best metric value
                if self.keep_best_metric and self.keep_best_metric in metadata.metrics:
                    metric_value = metadata.metrics[self.keep_best_metric]
                    if (
                        self.best_metric_value is None
                        or metric_value < self.best_metric_value
                    ):
                        self.best_metric_value = metric_value

            logger.info(
                f"Loaded checkpoint history with {len(self.checkpoint_history)} checkpoints"
            )
            if self.latest_checkpoint:
                logger.info(
                    f"Latest checkpoint: step {self.latest_checkpoint.step}, epoch {self.latest_checkpoint.epoch}"
                )
            if self.best_checkpoint:
                logger.info(
                    f"Best checkpoint: step {self.best_checkpoint.step}, epoch {self.best_checkpoint.epoch}"
                )

        except Exception as e:
            logger.error(f"Error loading checkpoint history: {e}")
            self.checkpoint_history = []

    def _save_checkpoint_history(self):
        """Save checkpoint history to metadata file."""
        if not self.is_master:
            return

        metadata_path = self.output_dir / "checkpoint_metadata.json"

        # Convert to serializable format
        history_data = []
        for metadata in self.checkpoint_history:
            history_data.append(
                {
                    "path": metadata.path,
                    "step": metadata.step,
                    "epoch": metadata.epoch,
                    "timestamp": metadata.timestamp,
                    "metrics": metadata.metrics,
                    "is_best": metadata.is_best,
                    "is_milestone": metadata.is_milestone,
                    "is_backup": metadata.is_backup,
                    "hash_value": metadata.hash_value,
                    "format": metadata.format.value,
                    "size_bytes": metadata.size_bytes,
                    "is_valid": metadata.is_valid,
                    "validation_error": metadata.validation_error,
                }
            )

        # Save with atomic write
        if self.use_atomic_save:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=self.output_dir, delete=False
            ) as temp_file:
                json.dump(history_data, temp_file, indent=2)
                temp_path = temp_file.name

            # Rename to final path
            shutil.move(temp_path, metadata_path)
        else:
            with open(metadata_path, "w") as f:
                json.dump(history_data, f, indent=2)

    def _check_disk_space(self) -> bool:
        """
        Check if there's enough disk space for checkpointing.

        Returns:
            bool: Whether there's enough disk space
        """
        if not self.disk_monitor_enabled:
            return True

        try:
            # Get disk usage statistics
            disk_usage = shutil.disk_usage(self.output_dir)
            free_space_gb = disk_usage.free / (1024**3)  # Convert to GB

            if free_space_gb < self.min_free_space_gb:
                logger.warning(
                    f"Low disk space: {free_space_gb:.2f}GB free, "
                    f"minimum required: {self.min_free_space_gb}GB"
                )

                # Try to free up space by removing old checkpoints
                self._emergency_cleanup()

                # Check again
                disk_usage = shutil.disk_usage(self.output_dir)
                free_space_gb = disk_usage.free / (1024**3)

                if free_space_gb < self.min_free_space_gb:
                    logger.error(
                        f"Still not enough disk space after cleanup: {free_space_gb:.2f}GB free. "
                        f"Skipping checkpoint."
                    )
                    return False
                else:
                    logger.info(
                        f"Successfully freed up space: {free_space_gb:.2f}GB now available"
                    )

            return True

        except Exception as e:
            logger.error(f"Error checking disk space: {e}")
            return True  # Proceed with checkpointing on error

    def _emergency_cleanup(self):
        """
        Emergency cleanup to free disk space.

        This method removes old checkpoints to free up disk space in case of low storage.
        """
        if not self.checkpoint_history:
            return

        logger.warning("Performing emergency cleanup to free disk space")

        # Sort checkpoints by step
        sorted_checkpoints = sorted(
            [
                cp
                for cp in self.checkpoint_history
                if not cp.is_best and not cp.is_backup
            ],
            key=lambda x: x.step,
        )

        # Remove the oldest half of checkpoints
        checkpoints_to_remove = sorted_checkpoints[: len(sorted_checkpoints) // 2]

        for checkpoint in checkpoints_to_remove:
            try:
                if os.path.exists(checkpoint.path):
                    os.remove(checkpoint.path)
                    logger.info(f"Removed checkpoint: {checkpoint.path}")

                # Remove from history
                self.checkpoint_history.remove(checkpoint)
            except Exception as e:
                logger.error(f"Error removing checkpoint {checkpoint.path}: {e}")

        # Save updated history
        self._save_checkpoint_history()

    def _compute_checkpoint_hash(self, checkpoint_path: str) -> str:
        """
        Compute a hash of the checkpoint file for integrity verification.

        Args:
            checkpoint_path: Path to the checkpoint file

        Returns:
            str: Hash value
        """
        try:
            hasher = hashlib.md5()
            with open(checkpoint_path, "rb") as f:
                # Read in chunks to handle large files
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Error computing checkpoint hash: {e}")
            return ""

    def _validate_checkpoint(self, checkpoint_path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint file

        Returns:
            Tuple[bool, Optional[str]]: Whether the checkpoint is valid and error message if not
        """
        try:
            # Try to load the checkpoint
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            # Check required keys
            required_keys = ["model"]
            for key in required_keys:
                if key not in checkpoint:
                    return False, f"Missing required key: {key}"

            # Check model state dict
            model_state = checkpoint["model"]
            if not isinstance(model_state, dict):
                return False, "Model state is not a dictionary"

            # Additional validation could be performed here
            # For example, checking that all expected parameters are present

            return True, None

        except Exception as e:
            return False, f"Error validating checkpoint: {e}"

    def _should_save_checkpoint(self, step: int) -> bool:
        """
        Determine if a checkpoint should be saved based on step and time intervals.

        Args:
            step: Current training step

        Returns:
            bool: Whether a checkpoint should be saved
        """
        # Always save if this is the first step
        if not self.checkpoint_history:
            return True

        # Check step interval
        if step % self.save_interval_steps == 0:
            return True

        # Check time interval if specified
        if self.save_interval_minutes is not None:
            time_since_last = (
                time.time() - self.last_checkpoint_time
            ) / 60  # Convert to minutes
            if time_since_last >= self.save_interval_minutes:
                return True

        return False

    def _is_milestone(self, step: int) -> bool:
        """
        Determine if a step is a milestone for checkpointing.

        Args:
            step: Current training step

        Returns:
            bool: Whether the step is a milestone
        """
        # Define milestone steps (e.g., every 10k steps)
        milestone_intervals = [10000, 50000, 100000]
        return any(step % interval == 0 for interval in milestone_intervals)

    def _get_checkpoint_name(
        self,
        step: int,
        epoch: int,
        tag: Optional[str] = None,
        is_best: bool = False,
        is_backup: bool = False,
    ) -> str:
        """
        Generate a checkpoint filename.

        Args:
            step: Current training step
            epoch: Current epoch
            tag: Optional tag to add to the filename
            is_best: Whether this is the best checkpoint
            is_backup: Whether this is a backup checkpoint

        Returns:
            str: Checkpoint filename
        """
        # Start with base name
        if is_best:
            name = "best_model"
        elif tag:
            name = f"{tag}_step_{step}"
        else:
            name = f"step_{step}"

        # Add epoch if provided
        if epoch > 0:
            name += f"_epoch_{epoch}"

        # Add backup indicator
        if is_backup:
            name += "_backup"

        # Add extension based on format and compression
        if self.checkpoint_format == CheckpointFormat.SAFETENSORS:
            extension = ".safetensors"
        else:
            extension = ".pt"

        # Add compression extension if needed
        if self.compression == "gzip":
            extension += ".gz"
        elif self.compression == "zip":
            extension += ".zip"

        return name + extension

    def _rotate_checkpoints(self):
        """
        Apply the checkpoint rotation strategy to manage storage.

        This method implements different strategies for keeping a limited number
        of checkpoints while ensuring important ones are preserved.
        """
        if not self.is_master:
            return

        # Skip rotation if we don't have enough checkpoints yet
        if len(self.checkpoint_history) <= self.max_checkpoints:
            return

        # Filter out checkpoints that should always be kept
        protected_checkpoints = [
            cp
            for cp in self.checkpoint_history
            if cp.is_best or cp.is_backup or cp.is_milestone
        ]

        regular_checkpoints = [
            cp
            for cp in self.checkpoint_history
            if not (cp.is_best or cp.is_backup or cp.is_milestone)
        ]

        # Apply rotation strategy
        if self.rotation_strategy == CheckpointRotationStrategy.KEEP_LAST_N:
            # Sort by step (descending)
            sorted_checkpoints = sorted(
                regular_checkpoints, key=lambda x: x.step, reverse=True
            )

            # Keep the newest N checkpoints
            to_keep = sorted_checkpoints[: self.max_checkpoints]
            to_remove = sorted_checkpoints[self.max_checkpoints :]

        elif self.rotation_strategy == CheckpointRotationStrategy.EXPONENTIAL_BACKOFF:
            # Keep checkpoints with exponentially increasing spacing
            # e.g., keep the last checkpoint, then the one from 2 steps ago,
            # then 4 steps ago, then 8 steps ago, etc.
            sorted_checkpoints = sorted(
                regular_checkpoints, key=lambda x: x.step, reverse=True
            )

            to_keep = []
            if sorted_checkpoints:
                # Always keep the most recent checkpoint
                latest_step = sorted_checkpoints[0].step

                # Calculate which checkpoints to keep based on exponential spacing
                keep_steps = set()
                interval = 1
                current_step = latest_step

                # Generate exponentially spaced steps to keep
                while current_step > 0 and len(keep_steps) < self.max_checkpoints:
                    keep_steps.add(current_step)
                    interval *= 2
                    current_step = latest_step - interval

                # Select checkpoints to keep
                for checkpoint in sorted_checkpoints:
                    if checkpoint.step in keep_steps:
                        to_keep.append(checkpoint)
                    elif len(to_keep) < self.max_checkpoints:
                        # Fill up to max_checkpoints with the newest ones
                        to_keep.append(checkpoint)

            # Determine which to remove
            to_keep_paths = {cp.path for cp in to_keep}
            to_remove = [
                cp for cp in sorted_checkpoints if cp.path not in to_keep_paths
            ]

        elif self.rotation_strategy == CheckpointRotationStrategy.KEEP_BEST_N:
            # Only applicable if we have a metric to track
            if self.keep_best_metric:
                # Sort by the specified metric (ascending, assuming lower is better)
                valid_checkpoints = [
                    cp
                    for cp in regular_checkpoints
                    if self.keep_best_metric in cp.metrics
                ]

                sorted_checkpoints = sorted(
                    valid_checkpoints, key=lambda x: x.metrics[self.keep_best_metric]
                )

                # Keep the best N checkpoints
                to_keep = sorted_checkpoints[: self.max_checkpoints]
                to_keep_paths = {cp.path for cp in to_keep}
                to_remove = [
                    cp for cp in regular_checkpoints if cp.path not in to_keep_paths
                ]
            else:
                # Fall back to KEEP_LAST_N if no metric specified
                logger.warning(
                    "KEEP_BEST_N strategy selected but no keep_best_metric specified. "
                    "Falling back to KEEP_LAST_N."
                )
                sorted_checkpoints = sorted(
                    regular_checkpoints, key=lambda x: x.step, reverse=True
                )
                to_keep = sorted_checkpoints[: self.max_checkpoints]
                to_remove = sorted_checkpoints[self.max_checkpoints :]

        elif self.rotation_strategy == CheckpointRotationStrategy.KEEP_MILESTONE:
            # Sort by step (descending)
            sorted_checkpoints = sorted(
                regular_checkpoints, key=lambda x: x.step, reverse=True
            )

            # Always keep milestone checkpoints and the latest N regular checkpoints
            latest_to_keep = sorted_checkpoints[: self.max_checkpoints]
            remaining = sorted_checkpoints[self.max_checkpoints :]

            # Mark milestone checkpoints to keep
            milestone_to_keep = [cp for cp in remaining if self._is_milestone(cp.step)]

            # Combine and determine which to remove
            to_keep = latest_to_keep + milestone_to_keep
            to_keep_paths = {cp.path for cp in to_keep}
            to_remove = [
                cp for cp in regular_checkpoints if cp.path not in to_keep_paths
            ]

        else:
            logger.error(f"Unknown rotation strategy: {self.rotation_strategy}")
            return

        # Remove checkpoints marked for deletion
        for checkpoint in to_remove:
            try:
                if os.path.exists(checkpoint.path):
                    os.remove(checkpoint.path)
                    logger.info(f"Removed checkpoint: {checkpoint.path}")

                # Remove from history
                self.checkpoint_history.remove(checkpoint)
            except Exception as e:
                logger.error(f"Error removing checkpoint {checkpoint.path}: {e}")

        # Update checkpoint history
        self.checkpoint_history = protected_checkpoints + to_keep
        self._save_checkpoint_history()

    def save_checkpoint(
        self,
        step: int,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        is_emergency: bool = False,
        tag: Optional[str] = None,
    ) -> Optional[str]:
        """
        Save a checkpoint.

        Args:
            step: Current training step
            epoch: Current epoch
            metrics: Dictionary of metrics
            is_best: Whether this is the best checkpoint
            is_emergency: Whether this is an emergency checkpoint
            tag: Optional tag to add to the filename

        Returns:
            Optional[str]: Path to the saved checkpoint, or None if saving failed
        """
        # Skip if not master in distributed training
        if self.distributed_aware and not self.is_master:
            return None

        # Check if we should save a checkpoint
        if not is_best and not is_emergency and not self._should_save_checkpoint(step):
            return None

        # Check disk space
        if not self._check_disk_space():
            return None

        # Call save start callback if provided
        if self.on_save_start:
            self.on_save_start(step=step, epoch=epoch, metrics=metrics)

        # If this is an emergency checkpoint and we have a latest checkpoint,
        # use its step and epoch to avoid overwriting progress information
        if is_emergency and self.latest_checkpoint:
            step = self.latest_checkpoint.step
            epoch = self.latest_checkpoint.epoch
            # Merge metrics
            for k, v in self.latest_checkpoint.metrics.items():
                if k not in metrics:
                    metrics[k] = v

        # Update best metric if applicable
        if self.keep_best_metric and self.keep_best_metric in metrics:
            metric_value = metrics[self.keep_best_metric]
            if self.best_metric_value is None or metric_value < self.best_metric_value:
                self.best_metric_value = metric_value
                is_best = True

        # Generate checkpoint name
        checkpoint_name = self._get_checkpoint_name(
            step=step,
            epoch=epoch,
            tag=tag,
            is_best=is_best,
        )

        # Create checkpoint path
        checkpoint_path = str(self.output_dir / checkpoint_name)

        # Prepare model for saving
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model_to_save = self.model.module
        else:
            model_to_save = self.model

        # Create checkpoint
        checkpoint = {
            "model": model_to_save.state_dict(),
            "step": step,
            "epoch": epoch,
            "metrics": metrics,
            "timestamp": time.time(),
        }

        # Add optimizer state if available
        if self.optimizer is not None:
            checkpoint["optimizer"] = self.optimizer.state_dict()

        # Add scheduler state if available
        if self.scheduler is not None:
            checkpoint["scheduler"] = self.scheduler.state_dict()

        # Add scaler state if available
        if self.scaler is not None:
            checkpoint["scaler"] = self.scaler.state_dict()

        # Save checkpoint
        try:
            # Use atomic save if enabled
            if self.use_atomic_save:
                # Create temporary file
                with tempfile.NamedTemporaryFile(
                    dir=self.output_dir, delete=False
                ) as temp_file:
                    temp_path = temp_file.name

                # Save to temporary file
                torch.save(checkpoint, temp_path)

                # Rename to final path
                shutil.move(temp_path, checkpoint_path)
            else:
                # Direct save
                torch.save(checkpoint, checkpoint_path)

            # Get file size
            size_bytes = os.path.getsize(checkpoint_path)

            # Validate checkpoint if enabled
            is_valid = True
            validation_error = None

            if self.validate_on_save:
                hash_value = self._compute_checkpoint_hash(checkpoint_path)
                is_valid, validation_error = self._validate_checkpoint(checkpoint_path)

                if not is_valid:
                    logger.error(f"Checkpoint validation failed: {validation_error}")

                    # Try to recover if validation fails
                    if self.latest_checkpoint and os.path.exists(
                        self.latest_checkpoint.path
                    ):
                        logger.warning(
                            f"Falling back to latest valid checkpoint: {self.latest_checkpoint.path}"
                        )
                        # Remove invalid checkpoint
                        os.remove(checkpoint_path)
                        return None
            else:
                hash_value = None

            # Create metadata
            metadata = CheckpointMetadata(
                path=checkpoint_path,
                step=step,
                epoch=epoch,
                timestamp=time.time(),
                metrics=metrics,
                is_best=is_best,
                is_milestone=self._is_milestone(step),
                is_backup=False,
                hash_value=hash_value,
                format=self.checkpoint_format,
                size_bytes=size_bytes,
                is_valid=is_valid,
                validation_error=validation_error,
            )

            # Update checkpoint history
            self.checkpoint_history.append(metadata)

            # Update latest checkpoint
            self.latest_checkpoint = metadata

            # Update best checkpoint if applicable
            if is_best:
                self.best_checkpoint = metadata

                # Create best checkpoint link or copy
                best_path = str(self.output_dir / "best_model.pt")
                if os.path.exists(best_path):
                    os.remove(best_path)

                if self.backup_best:
                    # Create a copy
                    shutil.copy2(checkpoint_path, best_path)
                else:
                    # Create a symbolic link
                    os.symlink(os.path.basename(checkpoint_path), best_path)

            # Create latest checkpoint link or copy
            latest_path = str(self.output_dir / "latest.pt")
            if os.path.exists(latest_path):
                os.remove(latest_path)

            if self.backup_latest:
                # Create a copy
                shutil.copy2(checkpoint_path, latest_path)
            else:
                # Create a symbolic link
                os.symlink(os.path.basename(checkpoint_path), latest_path)

            # Save checkpoint history
            self._save_checkpoint_history()

            # Apply rotation strategy
            self._rotate_checkpoints()

            # Update last checkpoint time
            self.last_checkpoint_time = time.time()

            logger.info(
                f"Saved checkpoint to {checkpoint_path} (size: {size_bytes / 1024 / 1024:.2f} MB)"
            )

            # Call save end callback if provided
            if self.on_save_end:
                self.on_save_end(
                    step=step,
                    epoch=epoch,
                    metrics=metrics,
                    path=checkpoint_path,
                    is_valid=is_valid,
                )

            return checkpoint_path

        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")

            # Try to remove the incomplete checkpoint
            if os.path.exists(checkpoint_path):
                try:
                    os.remove(checkpoint_path)
                except Exception:
                    pass

            return None

    def create_backup(
        self,
        checkpoint_metadata: Optional[CheckpointMetadata] = None,
        tag: str = "backup",
    ) -> Optional[str]:
        """
        Create a backup of a checkpoint.

        Args:
            checkpoint_metadata: Metadata of the checkpoint to backup (uses latest if None)
            tag: Tag to add to the backup filename

        Returns:
            Optional[str]: Path to the backup checkpoint, or None if backup failed
        """
        # Skip if not master in distributed training
        if self.distributed_aware and not self.is_master:
            return None

        # Use latest checkpoint if none specified
        if checkpoint_metadata is None:
            if self.latest_checkpoint is None:
                logger.warning("No checkpoint to backup")
                return None
            checkpoint_metadata = self.latest_checkpoint

        # Check if source checkpoint exists
        if not os.path.exists(checkpoint_metadata.path):
            logger.error(f"Source checkpoint not found: {checkpoint_metadata.path}")
            return None

        # Generate backup name
        backup_name = self._get_checkpoint_name(
            step=checkpoint_metadata.step,
            epoch=checkpoint_metadata.epoch,
            tag=tag,
            is_best=checkpoint_metadata.is_best,
            is_backup=True,
        )

        # Create backup path
        backup_path = str(self.output_dir / backup_name)

        try:
            # Create backup
            shutil.copy2(checkpoint_metadata.path, backup_path)

            # Create backup metadata
            backup_metadata = CheckpointMetadata(
                path=backup_path,
                step=checkpoint_metadata.step,
                epoch=checkpoint_metadata.epoch,
                timestamp=time.time(),
                metrics=checkpoint_metadata.metrics,
                is_best=checkpoint_metadata.is_best,
                is_milestone=checkpoint_metadata.is_milestone,
                is_backup=True,
                hash_value=checkpoint_metadata.hash_value,
                format=checkpoint_metadata.format,
                size_bytes=os.path.getsize(backup_path),
                is_valid=checkpoint_metadata.is_valid,
                validation_error=checkpoint_metadata.validation_error,
            )

            # Update checkpoint history
            self.checkpoint_history.append(backup_metadata)

            # Save checkpoint history
            self._save_checkpoint_history()

            logger.info(f"Created backup checkpoint at {backup_path}")

            return backup_path

        except Exception as e:
            logger.error(f"Error creating backup: {e}")

            # Try to remove the incomplete backup
            if os.path.exists(backup_path):
                try:
                    os.remove(backup_path)
                except Exception:
                    pass

            return None

    def load_checkpoint(
        self,
        checkpoint_path: str,
        load_model: bool = True,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        load_scaler: bool = True,
        map_location: Optional[torch.device] = None,
        strict: bool = True,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Load a checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint
            load_model: Whether to load model weights
            load_optimizer: Whether to load optimizer state
            load_scheduler: Whether to load scheduler state
            load_scaler: Whether to load scaler state
            map_location: Device to load the checkpoint to
            strict: Whether to strictly enforce that the keys in state_dict match

        Returns:
            Tuple[Optional[Dict[str, Any]], Optional[str]]: Checkpoint data and error message if any
        """
        # Call load start callback if provided
        if self.on_load_start:
            self.on_load_start(path=checkpoint_path)

        try:
            # Validate checkpoint if enabled
            if self.validate_on_load:
                is_valid, validation_error = self._validate_checkpoint(checkpoint_path)
                if not is_valid:
                    return None, f"Checkpoint validation failed: {validation_error}"

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=map_location or "cpu")

            # Load model weights if requested
            if load_model and "model" in checkpoint and self.model is not None:
                if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                    self.model.module.load_state_dict(
                        checkpoint["model"], strict=strict
                    )
                else:
                    self.model.load_state_dict(checkpoint["model"], strict=strict)

            # Load optimizer state if requested
            if (
                load_optimizer
                and "optimizer" in checkpoint
                and self.optimizer is not None
            ):
                self.optimizer.load_state_dict(checkpoint["optimizer"])

            # Load scheduler state if requested
            if (
                load_scheduler
                and "scheduler" in checkpoint
                and self.scheduler is not None
            ):
                self.scheduler.load_state_dict(checkpoint["scheduler"])

            # Load scaler state if requested
            if load_scaler and "scaler" in checkpoint and self.scaler is not None:
                self.scaler.load_state_dict(checkpoint["scaler"])

            # Call load end callback if provided
            if self.on_load_end:
                self.on_load_end(
                    path=checkpoint_path,
                    step=checkpoint.get("step", 0),
                    epoch=checkpoint.get("epoch", 0),
                    metrics=checkpoint.get("metrics", {}),
                )

            logger.info(f"Successfully loaded checkpoint from {checkpoint_path}")

            return checkpoint, None

        except Exception as e:
            error_msg = f"Error loading checkpoint: {e}"
            logger.error(error_msg)
            return None, error_msg

    def load_latest_checkpoint(
        self,
        map_location: Optional[torch.device] = None,
        strict: bool = True,
    ) -> Tuple[nn.Module, Optional[Optimizer], Optional[Any], int, int]:
        """
        Load the latest checkpoint.

        Args:
            map_location: Device to load the checkpoint to
            strict: Whether to strictly enforce that the keys in state_dict match

        Returns:
            Tuple[nn.Module, Optional[Optimizer], Optional[Any], int, int]:
                Model, optimizer, scheduler, step, epoch
        """
        if self.latest_checkpoint is None:
            logger.warning("No checkpoint found to load")
            return self.model, self.optimizer, self.scheduler, 0, 0

        checkpoint_path = self.latest_checkpoint.path

        # Load checkpoint
        checkpoint, error = self.load_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location=map_location,
            strict=strict,
        )

        if checkpoint is None:
            logger.error(f"Failed to load latest checkpoint: {error}")
            return self.model, self.optimizer, self.scheduler, 0, 0

        # Get step and epoch
        step = checkpoint.get("step", 0)
        epoch = checkpoint.get("epoch", 0)

        logger.info(f"Resumed from step {step}, epoch {epoch}")

        return self.model, self.optimizer, self.scheduler, step, epoch

    def load_best_checkpoint(
        self,
        map_location: Optional[torch.device] = None,
        strict: bool = True,
    ) -> Tuple[nn.Module, Optional[Optimizer], Optional[Any], int, int]:
        """
        Load the best checkpoint.

        Args:
            map_location: Device to load the checkpoint to
            strict: Whether to strictly enforce that the keys in state_dict match

        Returns:
            Tuple[nn.Module, Optional[Optimizer], Optional[Any], int, int]:
                Model, optimizer, scheduler, step, epoch
        """
        if self.best_checkpoint is None:
            logger.warning("No best checkpoint found, trying to load latest")
            return self.load_latest_checkpoint(map_location=map_location, strict=strict)

        checkpoint_path = self.best_checkpoint.path

        # Load checkpoint
        checkpoint, error = self.load_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location=map_location,
            strict=strict,
        )

        if checkpoint is None:
            logger.error(f"Failed to load best checkpoint: {error}")
            return self.model, self.optimizer, self.scheduler, 0, 0

        # Get step and epoch
        step = checkpoint.get("step", 0)
        epoch = checkpoint.get("epoch", 0)

        logger.info(f"Loaded best checkpoint from step {step}, epoch {epoch}")

        return self.model, self.optimizer, self.scheduler, step, epoch

    def find_closest_checkpoint(
        self,
        target_step: int,
    ) -> Optional[CheckpointMetadata]:
        """
        Find the checkpoint closest to the target step.

        Args:
            target_step: Target step to find

        Returns:
            Optional[CheckpointMetadata]: Metadata of the closest checkpoint
        """
        if not self.checkpoint_history:
            return None

        # Find the checkpoint with the closest step
        closest_checkpoint = min(
            self.checkpoint_history, key=lambda x: abs(x.step - target_step)
        )

        return closest_checkpoint

    def get_checkpoint_info(self) -> Dict[str, Any]:
        """
        Get information about available checkpoints.

        Returns:
            Dict[str, Any]: Checkpoint information
        """
        info = {
            "total_checkpoints": len(self.checkpoint_history),
            "latest_checkpoint": None,
            "best_checkpoint": None,
            "checkpoint_dir": str(self.output_dir),
            "available_steps": sorted([cp.step for cp in self.checkpoint_history]),
        }

        if self.latest_checkpoint:
            info["latest_checkpoint"] = {
                "path": self.latest_checkpoint.path,
                "step": self.latest_checkpoint.step,
                "epoch": self.latest_checkpoint.epoch,
                "timestamp": self.latest_checkpoint.timestamp,
                "metrics": self.latest_checkpoint.metrics,
            }

        if self.best_checkpoint:
            info["best_checkpoint"] = {
                "path": self.best_checkpoint.path,
                "step": self.best_checkpoint.step,
                "epoch": self.best_checkpoint.epoch,
                "timestamp": self.best_checkpoint.timestamp,
                "metrics": self.best_checkpoint.metrics,
            }

        return info

    def verify_all_checkpoints(self) -> Dict[str, Any]:
        """
        Verify the integrity of all checkpoints.

        Returns:
            Dict[str, Any]: Verification results
        """
        results = {
            "total_checkpoints": len(self.checkpoint_history),
            "valid_checkpoints": 0,
            "invalid_checkpoints": 0,
            "errors": [],
        }

        for checkpoint in self.checkpoint_history:
            # Skip if file doesn't exist
            if not os.path.exists(checkpoint.path):
                results["invalid_checkpoints"] += 1
                results["errors"].append(
                    {
                        "path": checkpoint.path,
                        "error": "File not found",
                    }
                )
                continue

            # Verify checkpoint
            is_valid, error = self._validate_checkpoint(checkpoint.path)

            # Update checkpoint metadata
            checkpoint.is_valid = is_valid
            checkpoint.validation_error = error

            if is_valid:
                results["valid_checkpoints"] += 1
            else:
                results["invalid_checkpoints"] += 1
                results["errors"].append(
                    {
                        "path": checkpoint.path,
                        "error": error,
                    }
                )

        # Save updated checkpoint history
        self._save_checkpoint_history()

        return results

    def cleanup(self):
        """Clean up resources used by the checkpoint manager."""
        # Save checkpoint history
        self._save_checkpoint_history()


def create_checkpoint_manager(
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    output_dir: str = "checkpoints",
    config: Optional[Dict[str, Any]] = None,
) -> CheckpointManager:
    """
    Create a checkpoint manager with sensible defaults.

    Args:
        model: Model to checkpoint
        optimizer: Optimizer to checkpoint
        scheduler: Learning rate scheduler to checkpoint
        scaler: Gradient scaler for mixed precision training
        output_dir: Directory to save checkpoints
        config: Additional configuration options

    Returns:
        CheckpointManager: Configured checkpoint manager
    """
    # Default configuration
    default_config = {
        "save_interval_steps": 1000,
        "save_interval_minutes": 30,
        "max_checkpoints": 5,
        "rotation_strategy": "exponential_backoff",
        "use_atomic_save": True,
        "validate_on_save": True,
        "validate_on_load": True,
        "backup_best": True,
        "backup_latest": True,
        "disk_monitor_enabled": True,
        "min_free_space_gb": 5.0,
    }

    # Merge with provided config
    if config:
        for key, value in config.items():
            default_config[key] = value

    # Create manager
    manager = CheckpointManager(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        output_dir=output_dir,
        save_interval_steps=default_config["save_interval_steps"],
        save_interval_minutes=default_config["save_interval_minutes"],
        max_checkpoints=default_config["max_checkpoints"],
        rotation_strategy=default_config["rotation_strategy"],
        use_atomic_save=default_config["use_atomic_save"],
        validate_on_save=default_config["validate_on_save"],
        validate_on_load=default_config["validate_on_load"],
        backup_best=default_config["backup_best"],
        backup_latest=default_config["backup_latest"],
        disk_monitor_enabled=default_config["disk_monitor_enabled"],
        min_free_space_gb=default_config["min_free_space_gb"],
    )

    return manager


if __name__ == "__main__":
    """Run checkpoint manager diagnostics."""
    import argparse

    parser = argparse.ArgumentParser(description="Checkpoint Manager Diagnostics")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Checkpoint directory to analyze",
    )
    parser.add_argument("--verify", action="store_true", help="Verify all checkpoints")
    parser.add_argument(
        "--cleanup", action="store_true", help="Clean up invalid checkpoints"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backups of best and latest checkpoints",
    )
    args = parser.parse_args()

    print(f"\n=== Checkpoint Manager Diagnostics ===")
    print(f"Checkpoint directory: {args.checkpoint_dir}")

    # Create a dummy model for the checkpoint manager
    dummy_model = nn.Linear(10, 10)

    # Create checkpoint manager
    manager = CheckpointManager(
        model=dummy_model,
        output_dir=args.checkpoint_dir,
    )

    # Print checkpoint info
    info = manager.get_checkpoint_info()
    print(f"\nFound {info['total_checkpoints']} checkpoints")

    if info["latest_checkpoint"]:
        latest = info["latest_checkpoint"]
        print(f"\nLatest checkpoint:")
        print(f"  Step: {latest['step']}")
        print(f"  Epoch: {latest['epoch']}")
        print(f"  Path: {latest['path']}")
        print(f"  Metrics: {latest['metrics']}")

    if info["best_checkpoint"]:
        best = info["best_checkpoint"]
        print(f"\nBest checkpoint:")
        print(f"  Step: {best['step']}")
        print(f"  Epoch: {best['epoch']}")
        print(f"  Path: {best['path']}")
        print(f"  Metrics: {best['metrics']}")

    # Verify checkpoints if requested
    if args.verify:
        print(f"\nVerifying checkpoints...")
        results = manager.verify_all_checkpoints()
        print(f"  Valid checkpoints: {results['valid_checkpoints']}")
        print(f"  Invalid checkpoints: {results['invalid_checkpoints']}")

        if results["errors"]:
            print(f"\nErrors found:")
            for error in results["errors"]:
                print(f"  {error['path']}: {error['error']}")

    # Create backups if requested
    if args.backup:
        print(f"\nCreating backups...")
        if manager.best_checkpoint:
            backup_path = manager.create_backup(
                manager.best_checkpoint, tag="best_backup"
            )
            if backup_path:
                print(f"  Created backup of best checkpoint: {backup_path}")

        if manager.latest_checkpoint:
            backup_path = manager.create_backup(
                manager.latest_checkpoint, tag="latest_backup"
            )
            if backup_path:
                print(f"  Created backup of latest checkpoint: {backup_path}")

    # Clean up if requested
    if args.cleanup:
        print(f"\nCleaning up...")
        manager._emergency_cleanup()
        print(f"  Cleanup complete")

    print("\nDiagnostics complete!")
