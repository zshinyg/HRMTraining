#!/usr/bin/env python
"""
Training Orchestrator for HRM

This module provides a comprehensive training orchestration system that manages
the entire training lifecycle, from initialization to completion, with robust
support for long-running jobs, automatic recovery, CI/CD integration, and
distributed training coordination.

Features:
- CI/CD pipeline integration with GitHub Actions, Jenkins, etc.
- Long-running training job management with health monitoring
- Automatic restart and recovery from failures
- Intelligent checkpointing strategies for multi-day training
- Training progress notifications via Slack, Email, or custom webhooks
- Resource allocation and scheduling for optimal hardware utilization
- Pre/post training validation and testing hooks
- Distributed training coordination and synchronization
- Experiment tracking and logging across restarts
- Training completion validation and artifact management
- Integration with W&B, TensorBoard, and other tracking platforms
- Support for both local and cloud-based training environments

Usage:
    from scripts.training.training_orchestrator import TrainingOrchestrator
    
    # Initialize orchestrator
    orchestrator = TrainingOrchestrator(
        config_path="configs/training_config.yaml",
        experiment_name="hrm_mbpp_training",
        output_dir="checkpoints/hrm_mbpp",
    )
    
    # Run training with automatic orchestration
    orchestrator.run()
    
    # Alternative: Manual control of training stages
    orchestrator.initialize()
    orchestrator.prepare_environment()
    orchestrator.prepare_data()
    orchestrator.prepare_model()
    orchestrator.train()
    orchestrator.validate()
    orchestrator.finalize()
"""

import argparse
import atexit
import datetime
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import yaml
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

# Import local modules if available
try:
    from scripts.training.checkpoint_manager import CheckpointManager, create_checkpoint_manager
    CHECKPOINT_MANAGER_AVAILABLE = True
except ImportError:
    CHECKPOINT_MANAGER_AVAILABLE = False

try:
    from scripts.training.resource_monitor import ResourceMonitor
    RESOURCE_MONITOR_AVAILABLE = True
except ImportError:
    RESOURCE_MONITOR_AVAILABLE = False

try:
    from scripts.training.mps_optimizer import MPSOptimizer, optimize_for_mps
    MPS_OPTIMIZER_AVAILABLE = True
except ImportError:
    MPS_OPTIMIZER_AVAILABLE = False

# Try to import optional dependencies
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False

try:
    import boto3
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    from google.cloud import storage
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TrainingStage(Enum):
    """Stages of the training process."""
    INITIALIZING = "initializing"
    ENVIRONMENT_SETUP = "environment_setup"
    DATA_PREPARATION = "data_preparation"
    MODEL_PREPARATION = "model_preparation"
    TRAINING = "training"
    VALIDATION = "validation"
    FINALIZATION = "finalization"
    COMPLETED = "completed"
    FAILED = "failed"


class TrainingEnvironment(Enum):
    """Training environment types."""
    LOCAL = "local"
    SLURM = "slurm"
    KUBERNETES = "kubernetes"
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    CUSTOM = "custom"


class NotificationType(Enum):
    """Types of notifications."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"
    PROGRESS = "progress"


@dataclass
class TrainingStatus:
    """Status of the training process."""
    stage: TrainingStage = TrainingStage.INITIALIZING
    start_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)
    current_epoch: int = 0
    total_epochs: int = 0
    current_step: int = 0
    total_steps: int = 0
    best_metric: float = float('inf')
    best_metric_name: str = "loss"
    best_metric_step: int = 0
    best_metric_epoch: int = 0
    last_checkpoint_step: int = 0
    last_checkpoint_time: float = 0
    last_validation_step: int = 0
    last_validation_time: float = 0
    training_speed: float = 0.0  # Examples per second
    estimated_time_remaining: float = 0.0  # Seconds
    is_distributed: bool = False
    world_size: int = 1
    rank: int = 0
    device: str = "cpu"
    latest_loss: float = 0.0
    latest_learning_rate: float = 0.0
    latest_metrics: Dict[str, float] = field(default_factory=dict)
    restart_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    is_paused: bool = False


class TrainingOrchestrator:
    """
    Orchestrates the entire training process with robust error handling,
    checkpointing, monitoring, and CI/CD integration.
    """
    
    def __init__(
        self,
        config_path: str,
        experiment_name: str = None,
        output_dir: str = None,
        data_dir: str = None,
        resume_from: str = None,
        distributed: bool = False,
        world_size: int = None,
        master_port: int = 12355,
        use_wandb: bool = None,
        use_tensorboard: bool = None,
        notification_config: Dict[str, Any] = None,
        ci_mode: bool = False,
        debug: bool = False,
        seed: int = None,
        hooks: Dict[str, Callable] = None,
    ):
        """
        Initialize the training orchestrator.
        
        Args:
            config_path: Path to the training configuration file
            experiment_name: Name of the experiment (defaults to config name)
            output_dir: Directory to save outputs (defaults to config output_dir)
            data_dir: Directory containing training data (defaults to config data_dir)
            resume_from: Path to checkpoint to resume from
            distributed: Whether to use distributed training
            world_size: Number of processes for distributed training
            master_port: Port for distributed training coordination
            use_wandb: Whether to use W&B for logging (overrides config)
            use_tensorboard: Whether to use TensorBoard for logging (overrides config)
            notification_config: Configuration for notifications
            ci_mode: Whether running in CI/CD environment
            debug: Whether to enable debug logging
            seed: Random seed (overrides config)
            hooks: Custom hooks for different stages of training
        """
        self.config_path = config_path
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.data_dir = data_dir
        self.resume_from = resume_from
        self.distributed = distributed
        self.world_size = world_size
        self.master_port = master_port
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self.notification_config = notification_config or {}
        self.ci_mode = ci_mode
        self.debug = debug
        self.seed = seed
        self.hooks = hooks or {}
        
        # Set up logging
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.setLevel(logging.DEBUG)
        
        # Initialize state
        self.config = None
        self.status = TrainingStatus()
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
        self.checkpoint_manager = None
        self.resource_monitor = None
        self.is_master = True
        self.run_id = f"{int(time.time())}_{os.getpid()}"
        
        # Notification components
        self.slack_client = None
        self.notification_thread = None
        self.notification_queue = []
        self.notification_lock = threading.Lock()
        
        # Distributed training state
        self.distributed_initialized = False
        
        # Load configuration
        self._load_config()
        
        # Set up output directory
        self._setup_output_dir()
        
        # Register signal handlers and exit handlers
        self._register_handlers()
        
        logger.info(f"Training orchestrator initialized for experiment: {self.experiment_name}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Run ID: {self.run_id}")
    
    def _load_config(self):
        """Load and validate the training configuration."""
        try:
            with open(self.config_path, "r") as f:
                self.config = yaml.safe_load(f)
            
            # Set defaults from config if not specified
            if self.experiment_name is None:
                self.experiment_name = self.config.get("experiment_name", Path(self.config_path).stem)
            
            if self.output_dir is None:
                self.output_dir = self.config.get("output_dir", f"outputs/{self.experiment_name}")
            
            if self.data_dir is None:
                self.data_dir = self.config.get("data_dir", "data")
            
            if self.use_wandb is None:
                self.use_wandb = self.config.get("use_wandb", False)
            
            if self.use_tensorboard is None:
                self.use_tensorboard = self.config.get("use_tensorboard", True)
            
            if self.world_size is None and self.distributed:
                self.world_size = self.config.get("world_size", torch.cuda.device_count() if torch.cuda.is_available() else 1)
            
            if self.seed is None:
                self.seed = self.config.get("seed", 42)
            
            # Update status with total epochs/steps
            self.status.total_epochs = self.config.get("training", {}).get("epochs", 0)
            
            logger.info(f"Loaded configuration from {self.config_path}")
        
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def _setup_output_dir(self):
        """Set up the output directory structure."""
        try:
            # Create main output directory
            output_dir = Path(self.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            (output_dir / "checkpoints").mkdir(exist_ok=True)
            (output_dir / "logs").mkdir(exist_ok=True)
            (output_dir / "artifacts").mkdir(exist_ok=True)
            (output_dir / "tensorboard").mkdir(exist_ok=True)
            
            # Save a copy of the configuration
            config_copy_path = output_dir / "config.yaml"
            with open(config_copy_path, "w") as f:
                yaml.dump(self.config, f, default_flow_style=False)
            
            logger.info(f"Set up output directory at {output_dir}")
        
        except Exception as e:
            logger.error(f"Error setting up output directory: {e}")
            raise
    
    def _register_handlers(self):
        """Register signal and exit handlers."""
        # Register exit handler
        atexit.register(self._cleanup)
        
        # Register signal handlers
        def signal_handler(sig, frame):
            logger.warning(f"Received signal {sig}, initiating graceful shutdown...")
            self._handle_interrupt()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _cleanup(self):
        """Clean up resources before exiting."""
        logger.info("Cleaning up resources...")
        
        # Stop resource monitoring
        if self.resource_monitor is not None:
            try:
                self.resource_monitor.stop()
            except Exception as e:
                logger.error(f"Error stopping resource monitor: {e}")
        
        # Finalize distributed training
        if self.distributed_initialized:
            try:
                dist.destroy_process_group()
                self.distributed_initialized = False
            except Exception as e:
                logger.error(f"Error destroying process group: {e}")
        
        # Finalize W&B
        if self.use_wandb and WANDB_AVAILABLE and wandb.run is not None:
            try:
                wandb.finish()
            except Exception as e:
                logger.error(f"Error finalizing W&B: {e}")
        
        # Stop notification thread
        if self.notification_thread is not None and self.notification_thread.is_alive():
            try:
                with self.notification_lock:
                    self.notification_queue.append(None)  # Signal to stop
                self.notification_thread.join(timeout=2.0)
            except Exception as e:
                logger.error(f"Error stopping notification thread: {e}")
    
    def _handle_interrupt(self):
        """Handle interrupt signals (Ctrl+C, SIGTERM)."""
        logger.warning("Interrupt received, saving checkpoint and exiting...")
        
        # Save emergency checkpoint if in training stage
        if self.status.stage == TrainingStage.TRAINING and self.checkpoint_manager is not None:
            try:
                checkpoint_path = self.checkpoint_manager.save_checkpoint(
                    step=self.status.current_step,
                    epoch=self.status.current_epoch,
                    metrics=self.status.latest_metrics,
                    is_emergency=True,
                    tag="emergency",
                )
                if checkpoint_path:
                    logger.info(f"Saved emergency checkpoint to {checkpoint_path}")
                    
                    # Send notification
                    self._send_notification(
                        message=f"Training interrupted, emergency checkpoint saved at step {self.status.current_step}",
                        notification_type=NotificationType.WARNING,
                        details={
                            "checkpoint_path": checkpoint_path,
                            "step": self.status.current_step,
                            "epoch": self.status.current_epoch,
                        }
                    )
            except Exception as e:
                logger.error(f"Error saving emergency checkpoint: {e}")
        
        # Update status
        self.status.stage = TrainingStage.FAILED
        self.status.last_error = "Training interrupted by user"
        
        # Exit
        sys.exit(1)
    
    def _setup_distributed(self, rank: int):
        """
        Set up distributed training.
        
        Args:
            rank: Rank of the current process
        """
        if not self.distributed:
            self.is_master = True
            return
        
        # Set environment variables
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(self.master_port)
        
        # Initialize process group
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=self.world_size)
        self.distributed_initialized = True
        
        # Update status
        self.status.is_distributed = True
        self.status.world_size = self.world_size
        self.status.rank = rank
        self.is_master = (rank == 0)
        
        logger.info(f"Distributed training initialized: rank {rank}/{self.world_size}")
    
    def _setup_device(self, rank: int = 0):
        """
        Set up the device for training.
        
        Args:
            rank: Rank of the current process
        
        Returns:
            torch.device: Device to use for training
        """
        # Check for MPS (Apple Silicon)
        if (
            MPS_OPTIMIZER_AVAILABLE and
            hasattr(torch, 'mps') and
            torch.mps.is_available() and
            self.config.get("use_mps", True)
        ):
            device = torch.device("mps")
            logger.info(f"Using MPS device: {device}")
            self.status.device = "mps"
            return device
        
        # Check for CUDA
        if torch.cuda.is_available():
            if self.distributed:
                device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
            else:
                device = torch.device("cuda")
            logger.info(f"Using CUDA device: {device}")
            self.status.device = f"cuda:{rank % torch.cuda.device_count()}" if self.distributed else "cuda"
            return device
        
        # Fall back to CPU
        device = torch.device("cpu")
        logger.info("Using CPU device")
        self.status.device = "cpu"
        return device
    
    def _set_seed(self, seed: Optional[int] = None, rank: int = 0):
        """
        Set random seed for reproducibility.
        
        Args:
            seed: Random seed (uses self.seed if None)
            rank: Rank of the current process
        """
        if seed is None:
            seed = self.seed
        
        # Add rank to seed for distributed training
        if self.distributed:
            seed += rank
        
        # Set seeds
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)
        
        logger.info(f"Random seed set to {seed}")
    
    def _setup_monitoring(self):
        """Set up resource monitoring."""
        if not RESOURCE_MONITOR_AVAILABLE:
            logger.warning("Resource monitor not available, skipping monitoring setup")
            return
        
        try:
            # Create resource monitor
            log_dir = Path(self.output_dir) / "logs" / "resources"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            self.resource_monitor = ResourceMonitor(
                model=self.model,
                log_dir=str(log_dir),
                use_wandb=self.use_wandb and WANDB_AVAILABLE,
                project_name=self.experiment_name,
                check_interval_seconds=self.config.get("monitoring", {}).get("check_interval_seconds", 5.0),
                enable_thermal_monitoring=self.config.get("monitoring", {}).get("enable_thermal_monitoring", True),
                enable_memory_leak_detection=self.config.get("monitoring", {}).get("enable_memory_leak_detection", True),
                enable_crash_prediction=self.config.get("monitoring", {}).get("enable_crash_prediction", True),
                enable_adaptive_resources=self.config.get("monitoring", {}).get("enable_adaptive_resources", False),
            )
            
            # Start monitoring
            self.resource_monitor.start()
            logger.info("Resource monitoring started")
        
        except Exception as e:
            logger.error(f"Error setting up resource monitoring: {e}")
            self.resource_monitor = None
    
    def _setup_checkpoint_manager(self):
        """Set up the checkpoint manager."""
        if not CHECKPOINT_MANAGER_AVAILABLE:
            logger.warning("Checkpoint manager not available, using basic checkpointing")
            return
        
        try:
            # Create checkpoint manager
            checkpoint_dir = Path(self.output_dir) / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            self.checkpoint_manager = create_checkpoint_manager(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                output_dir=str(checkpoint_dir),
                config={
                    "save_interval_steps": self.config.get("checkpointing", {}).get("save_interval_steps", 1000),
                    "save_interval_minutes": self.config.get("checkpointing", {}).get("save_interval_minutes", 30),
                    "max_checkpoints": self.config.get("checkpointing", {}).get("max_checkpoints", 5),
                    "rotation_strategy": self.config.get("checkpointing", {}).get("rotation_strategy", "exponential_backoff"),
                    "keep_best_metric": self.config.get("checkpointing", {}).get("keep_best_metric", "val/loss"),
                    "distributed_aware": self.distributed,
                    "master_rank": 0,
                }
            )
            
            logger.info("Checkpoint manager initialized")
        
        except Exception as e:
            logger.error(f"Error setting up checkpoint manager: {e}")
            self.checkpoint_manager = None
    
    def _setup_notifications(self):
        """Set up the notification system."""
        if not self.notification_config:
            return
        
        try:
            # Set up Slack notifications
            if "slack" in self.notification_config and SLACK_AVAILABLE:
                slack_token = self.notification_config["slack"].get("token")
                if slack_token:
                    self.slack_client = WebClient(token=slack_token)
                    logger.info("Slack notifications enabled")
            
            # Start notification thread
            self.notification_thread = threading.Thread(target=self._notification_worker, daemon=True)
            self.notification_thread.start()
            
            # Send initial notification
            self._send_notification(
                message=f"Training started: {self.experiment_name}",
                notification_type=NotificationType.INFO,
                details={
                    "experiment_name": self.experiment_name,
                    "output_dir": self.output_dir,
                    "config_path": self.config_path,
                }
            )
            
            logger.info("Notification system initialized")
        
        except Exception as e:
            logger.error(f"Error setting up notifications: {e}")
    
    def _notification_worker(self):
        """Worker thread for sending notifications."""
        while True:
            # Get notification from queue
            notification = None
            with self.notification_lock:
                if self.notification_queue:
                    notification = self.notification_queue.pop(0)
            
            # Exit if None (sentinel value)
            if notification is None:
                break
            
            # Process notification
            try:
                message = notification["message"]
                notification_type = notification["type"]
                details = notification.get("details", {})
                
                # Send to Slack
                if self.slack_client is not None:
                    self._send_slack_notification(message, notification_type, details)
                
                # Send to other notification channels
                # (Email, custom webhooks, etc. could be implemented here)
                
                logger.debug(f"Sent notification: {message}")
            
            except Exception as e:
                logger.error(f"Error sending notification: {e}")
            
            # Sleep briefly to avoid hammering notification services
            time.sleep(0.1)
    
    def _send_notification(
        self,
        message: str,
        notification_type: NotificationType = NotificationType.INFO,
        details: Dict[str, Any] = None,
    ):
        """
        Queue a notification to be sent.
        
        Args:
            message: Notification message
            notification_type: Type of notification
            details: Additional details to include
        """
        if not self.notification_thread or not self.is_master:
            return
        
        # Add to queue
        with self.notification_lock:
            self.notification_queue.append({
                "message": message,
                "type": notification_type,
                "details": details or {},
                "timestamp": time.time(),
            })
    
    def _send_slack_notification(
        self,
        message: str,
        notification_type: NotificationType,
        details: Dict[str, Any],
    ):
        """
        Send a notification to Slack.
        
        Args:
            message: Notification message
            notification_type: Type of notification
            details: Additional details to include
        """
        if self.slack_client is None:
            return
        
        try:
            # Get channel from config
            channel = self.notification_config["slack"].get("channel", "#training-notifications")
            
            # Create message blocks
            blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*{self.experiment_name}*: {message}"
                    }
                }
            ]
            
            # Add details if available
            if details:
                details_text = "\n".join([f"• *{k}*: {v}" for k, v in details.items()])
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": details_text
                    }
                })
            
            # Add timestamp
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"<!date^{int(time.time())}^{{date_num}} {{time_secs}}|{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}>"
                    }
                ]
            })
            
            # Set color based on notification type
            color = {
                NotificationType.INFO: "#3498db",
                NotificationType.WARNING: "#f39c12",
                NotificationType.ERROR: "#e74c3c",
                NotificationType.SUCCESS: "#2ecc71",
                NotificationType.PROGRESS: "#9b59b6",
            }.get(notification_type, "#3498db")
            
            # Send message
            self.slack_client.chat_postMessage(
                channel=channel,
                text=message,
                blocks=blocks,
                attachments=[{"color": color}]
            )
        
        except SlackApiError as e:
            logger.error(f"Error sending Slack notification: {e.response['error']}")
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
    
    def _log_metrics(self, metrics: Dict[str, float], step: int):
        """
        Log metrics to W&B and/or TensorBoard.
        
        Args:
            metrics: Dictionary of metrics
            step: Current step
        """
        if not self.is_master:
            return
        
        # Log to W&B
        if self.use_wandb and WANDB_AVAILABLE and wandb.run is not None:
            wandb.log(metrics, step=step)
        
        # Log to TensorBoard
        # (TensorBoard logging would be implemented here)
        
        # Update status
        self.status.latest_metrics.update(metrics)
        if "loss" in metrics:
            self.status.latest_loss = metrics["loss"]
        if "learning_rate" in metrics:
            self.status.latest_learning_rate = metrics["learning_rate"]
        
        # Check for best metric
        best_metric_name = self.config.get("training", {}).get("best_metric", "val/loss")
        if best_metric_name in metrics:
            metric_value = metrics[best_metric_name]
            is_better = False
            
            # Check if better (lower is better by default)
            if self.config.get("training", {}).get("higher_is_better", False):
                is_better = metric_value > self.status.best_metric
            else:
                is_better = metric_value < self.status.best_metric
            
            if is_better:
                self.status.best_metric = metric_value
                self.status.best_metric_name = best_metric_name
                self.status.best_metric_step = step
                self.status.best_metric_epoch = self.status.current_epoch
                
                # Log best metric
                logger.info(f"New best {best_metric_name}: {metric_value:.6f} (step {step}, epoch {self.status.current_epoch})")
    
    def _update_training_speed(self, num_examples: int, elapsed_seconds: float):
        """
        Update training speed and estimated time remaining.
        
        Args:
            num_examples: Number of examples processed
            elapsed_seconds: Time elapsed in seconds
        """
        if elapsed_seconds > 0:
            # Calculate training speed
            examples_per_second = num_examples / elapsed_seconds
            self.status.training_speed = examples_per_second
            
            # Estimate time remaining
            if self.status.total_steps > 0 and self.status.current_step > 0:
                steps_remaining = self.status.total_steps - self.status.current_step
                self.status.estimated_time_remaining = (steps_remaining * elapsed_seconds) / num_examples
    
    def _save_status(self):
        """Save the current training status to a file."""
        if not self.is_master:
            return
        
        try:
            status_path = Path(self.output_dir) / "status.json"
            
            # Convert status to dictionary
            status_dict = {
                "stage": self.status.stage.value,
                "start_time": self.status.start_time,
                "last_update_time": time.time(),
                "current_epoch": self.status.current_epoch,
                "total_epochs": self.status.total_epochs,
                "current_step": self.status.current_step,
                "total_steps": self.status.total_steps,
                "best_metric": self.status.best_metric,
                "best_metric_name": self.status.best_metric_name,
                "best_metric_step": self.status.best_metric_step,
                "best_metric_epoch": self.status.best_metric_epoch,
                "last_checkpoint_step": self.status.last_checkpoint_step,
                "last_checkpoint_time": self.status.last_checkpoint_time,
                "last_validation_step": self.status.last_validation_step,
                "last_validation_time": self.status.last_validation_time,
                "training_speed": self.status.training_speed,
                "estimated_time_remaining": self.status.estimated_time_remaining,
                "is_distributed": self.status.is_distributed,
                "world_size": self.status.world_size,
                "device": self.status.device,
                "latest_loss": self.status.latest_loss,
                "latest_learning_rate": self.status.latest_learning_rate,
                "latest_metrics": self.status.latest_metrics,
                "restart_count": self.status.restart_count,
                "error_count": self.status.error_count,
                "last_error": self.status.last_error,
                "is_paused": self.status.is_paused,
                "run_id": self.run_id,
                "experiment_name": self.experiment_name,
            }
            
            # Save to file
            with open(status_path, "w") as f:
                json.dump(status_dict, f, indent=2)
        
        except Exception as e:
            logger.error(f"Error saving status: {e}")
    
    def _load_status(self):
        """Load training status from a file."""
        status_path = Path(self.output_dir) / "status.json"
        if not status_path.exists():
            return False
        
        try:
            with open(status_path, "r") as f:
                status_dict = json.load(f)
            
            # Update status
            self.status.stage = TrainingStage(status_dict["stage"])
            self.status.start_time = status_dict["start_time"]
            self.status.last_update_time = status_dict["last_update_time"]
            self.status.current_epoch = status_dict["current_epoch"]
            self.status.total_epochs = status_dict["total_epochs"]
            self.status.current_step = status_dict["current_step"]
            self.status.total_steps = status_dict["total_steps"]
            self.status.best_metric = status_dict["best_metric"]
            self.status.best_metric_name = status_dict["best_metric_name"]
            self.status.best_metric_step = status_dict["best_metric_step"]
            self.status.best_metric_epoch = status_dict["best_metric_epoch"]
            self.status.last_checkpoint_step = status_dict["last_checkpoint_step"]
            self.status.last_checkpoint_time = status_dict["last_checkpoint_time"]
            self.status.last_validation_step = status_dict["last_validation_step"]
            self.status.last_validation_time = status_dict["last_validation_time"]
            self.status.training_speed = status_dict["training_speed"]
            self.status.estimated_time_remaining = status_dict["estimated_time_remaining"]
            self.status.is_distributed = status_dict["is_distributed"]
            self.status.world_size = status_dict["world_size"]
            self.status.device = status_dict["device"]
            self.status.latest_loss = status_dict["latest_loss"]
            self.status.latest_learning_rate = status_dict["latest_learning_rate"]
            self.status.latest_metrics = status_dict["latest_metrics"]
            self.status.restart_count = status_dict["restart_count"] + 1  # Increment restart count
            self.status.error_count = status_dict["error_count"]
            self.status.last_error = status_dict["last_error"]
            
            # Don't restore paused state
            self.status.is_paused = False
            
            logger.info(f"Loaded training status from {status_path}")
            logger.info(f"Resuming from epoch {self.status.current_epoch}, step {self.status.current_step}")
            logger.info(f"This is restart #{self.status.restart_count}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error loading status: {e}")
            return False
    
    def _check_for_ci_artifacts(self):
        """Check for CI/CD artifacts and update configuration accordingly."""
        if not self.ci_mode:
            return
        
        try:
            # Check for CI environment variables
            ci_env_vars = {
                "GITHUB_ACTIONS": "GitHub Actions",
                "JENKINS_URL": "Jenkins",
                "TRAVIS": "Travis CI",
                "CIRCLECI": "CircleCI",
                "GITLAB_CI": "GitLab CI",
            }
            
            ci_platform = "Unknown CI"
            for env_var, platform_name in ci_env_vars.items():
                if os.environ.get(env_var):
                    ci_platform = platform_name
                    break
            
            logger.info(f"Running in CI environment: {ci_platform}")
            
            # Look for CI-specific configuration
            ci_config_path = Path(self.config_path).parent / "ci_config.yaml"
            if ci_config_path.exists():
                with open(ci_config_path, "r") as f:
                    ci_config = yaml.safe_load(f)
                
                # Merge CI configuration with main configuration
                self._merge_config(ci_config)
                logger.info(f"Merged CI-specific configuration from {ci_config_path}")
            
            # Check for environment variable overrides
            self._apply_env_overrides()
            
            # Set CI-specific settings
            if "ci" not in self.config:
                self.config["ci"] = {}
            
            self.config["ci"]["platform"] = ci_platform
            self.config["ci"]["run_id"] = os.environ.get("GITHUB_RUN_ID") or os.environ.get("CI_JOB_ID") or self.run_id
            
            # Adjust for CI environment
            # - Reduce epochs/steps for faster testing
            # - Disable some features that might not be needed in CI
            if self.config.get("ci", {}).get("fast_testing", True):
                if "training" not in self.config:
                    self.config["training"] = {}
                
                # Reduce epochs if not explicitly set by CI config
                if "epochs" not in self.config.get("ci", {}):
                    self.config["training"]["epochs"] = min(self.config["training"].get("epochs", 10), 2)
                
                # Disable some features
                self.use_wandb = False
                self.config["checkpointing"] = self.config.get("checkpointing", {})
                self.config["checkpointing"]["save_interval_steps"] = 100
                self.config["checkpointing"]["max_checkpoints"] = 1
            
            # Update status
            self.status.total_epochs = self.config.get("training", {}).get("epochs", 0)
        
        except Exception as e:
            logger.error(f"Error checking for CI artifacts: {e}")
    
    def _merge_config(self, new_config: Dict[str, Any]):
        """
        Merge new configuration into existing configuration.
        
        Args:
            new_config: New configuration to merge
        """
        def _merge_dicts(d1, d2):
            """Recursive dictionary merge."""
            for k, v in d2.items():
                if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                    _merge_dicts(d1[k], v)
                else:
                    d1[k] = v
        
        _merge_dicts(self.config, new_config)
    
    def _apply_env_overrides(self):
        """Apply configuration overrides from environment variables."""
        # Look for environment variables with prefix TRAIN_
        for key, value in os.environ.items():
            if key.startswith("TRAIN_"):
                # Convert to nested dictionary keys
                config_path = key[6:].lower().split("__")
                
                # Convert value to appropriate type
                if value.lower() in ("true", "yes", "1"):
                    typed_value = True
                elif value.lower() in ("false", "no", "0"):
                    typed_value = False
                elif value.isdigit():
                    typed_value = int(value)
                elif re.match(r"^\d+\.\d+$", value):
                    typed_value = float(value)
                else:
                    typed_value = value
                
                # Update config
                current = self.config
                for i, part in enumerate(config_path):
                    if i == len(config_path) - 1:
                        current[part] = typed_value
                    else:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                
                logger.info(f"Applied environment override: {key}={value}")
    
    def _validate_environment(self):
        """Validate the training environment."""
        # Check for required dependencies
        missing_deps = []
        
        if self.use_wandb and not WANDB_AVAILABLE:
            missing_deps.append("wandb")
        
        if not CHECKPOINT_MANAGER_AVAILABLE:
            missing_deps.append("checkpoint_manager")
        
        if not RESOURCE_MONITOR_AVAILABLE:
            missing_deps.append("resource_monitor")
        
        if self.distributed and not hasattr(torch, "distributed"):
            missing_deps.append("torch.distributed")
        
        if missing_deps:
            logger.warning(f"Missing optional dependencies: {', '.join(missing_deps)}")
        
        # Check for GPU availability
        if not torch.cuda.is_available() and not (hasattr(torch, 'mps') and torch.mps.is_available()):
            logger.warning("No GPU available, training will use CPU only")
            
            # If distributed training was requested but no GPUs are available, disable it
            if self.distributed:
                logger.warning("Disabling distributed training because no GPUs are available")
                self.distributed = False
                self.world_size = 1
        
        # Check disk space
        try:
            disk_usage = shutil.disk_usage(self.output_dir)
            free_gb = disk_usage.free / (1024 ** 3)
            
            if free_gb < 10:
                logger.warning(f"Low disk space: {free_gb:.1f} GB free")
            
            logger.info(f"Disk space: {free_gb:.1f} GB free / {disk_usage.total / (1024 ** 3):.1f} GB total")
        except Exception as e:
            logger.error(f"Error checking disk space: {e}")
        
        # Check for existing checkpoints
        checkpoint_dir = Path(self.output_dir) / "checkpoints"
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("*.pt"))
            if checkpoints:
                logger.info(f"Found {len(checkpoints)} existing checkpoints")
                
                # Check for latest checkpoint
                if self.resume_from is None:
                    latest_checkpoint = checkpoint_dir / "latest.pt"
                    if latest_checkpoint.exists():
                        self.resume_from = str(latest_checkpoint)
                        logger.info(f"Will resume from latest checkpoint: {self.resume_from}")
    
    def _create_model(self):
        """Create the model based on configuration."""
        # This is a placeholder - actual model creation would depend on the specific model architecture
        # In a real implementation, this would import the appropriate model class and create an instance
        
        # For example:
        # from hrm.model import create_hrm_model
        # self.model = create_hrm_model(self.config)
        
        logger.info("Model created")
    
    def _create_optimizer(self):
        """Create the optimizer based on configuration."""
        # This is a placeholder - actual optimizer creation would depend on the specific requirements
        # In a real implementation, this would create the appropriate optimizer based on config
        
        # For example:
        # optimizer_config = self.config.get("optimizer", {})
        # optimizer_type = optimizer_config.get("type", "adamw")
        # learning_rate = optimizer_config.get("learning_rate", 1e-4)
        # weight_decay = optimizer_config.get("weight_decay", 0.01)
        # 
        # if optimizer_type.lower() == "adamw":
        #     self.optimizer = torch.optim.AdamW(
        #         self.model.parameters(),
        #         lr=learning_rate,
        #         weight_decay=weight_decay
        #     )
        # elif optimizer_type.lower() == "adam":
        #     self.optimizer = torch.optim.Adam(
        #         self.model.parameters(),
        #         lr=learning_rate,
        #         weight_decay=weight_decay
        #     )
        # else:
        #     raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        logger.info("Optimizer created")
    
    def _create_scheduler(self):
        """Create the learning rate scheduler based on configuration."""
        # This is a placeholder - actual scheduler creation would depend on the specific requirements
        # In a real implementation, this would create the appropriate scheduler based on config
        
        # For example:
        # scheduler_config = self.config.get("scheduler", {})
        # scheduler_type = scheduler_config.get("type", "cosine")
        # 
        # if scheduler_type.lower() == "cosine":
        #     self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #         self.optimizer,
        #         T_max=self.status.total_steps,
        #         eta_min=scheduler_config.get("eta_min", 0)
        #     )
        # elif scheduler_type.lower() == "linear":
        #     self.scheduler = torch.optim.lr_scheduler.LinearLR(
        #         self.optimizer,
        #         start_factor=1.0,
        #         end_factor=scheduler_config.get("end_factor", 0.1),
        #         total_iters=self.status.total_steps
        #     )
        # else:
        #     raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        
        logger.info("Scheduler created")
    
    def _create_dataloaders(self):
        """Create data loaders based on configuration."""
        # This is a placeholder - actual data loader creation would depend on the specific dataset
        # In a real implementation, this would create the appropriate data loaders based on config
        
        # For example:
        # from data.mbpp_dataset import MBPPDataset
        # 
        # data_config = self.config.get("data", {})
        # batch_size = data_config.get("batch_size", 32)
        # num_workers = data_config.get("num_workers", 4)
        # 
        # # Create datasets
        # train_dataset = MBPPDataset(
        #     data_path=os.path.join(self.data_dir, "train.jsonl"),
        #     max_length=data_config.get("max_length", 1024)
        # )
        # 
        # val_dataset = MBPPDataset(
        #     data_path=os.path.join(self.data_dir, "val.jsonl"),
        #     max_length=data_config.get("max_length", 1024)
        # )
        # 
        # # Create samplers for distributed training
        # train_sampler = None
        # val_sampler = None
        # 
        # if self.distributed:
        #     train_sampler = torch.utils.data.distributed.DistributedSampler(
        #         train_dataset,
        #         num_replicas=self.world_size,
        #         rank=self.status.rank
        #     )
        #     
        #     val_sampler = torch.utils.data.distributed.DistributedSampler(
        #         val_dataset,
        #         num_replicas=self.world_size,
        #         rank=self.status.rank
        #     )
        # 
        # # Create data loaders
        # self.train_dataloader = torch.utils.data.DataLoader(
        #     train_dataset,
        #     batch_size=batch_size,
        #     sampler=train_sampler,
        #     num_workers=num_workers,
        #     pin_memory=True
        # )
        # 
        # self.val_dataloader = torch.utils.data.DataLoader(
        #     val_dataset,
        #     batch_size=batch_size,
        #     sampler=val_sampler,
        #     num_workers=num_workers,
        #     pin_memory=True
        # )
        # 
        # # Update total steps
        # self.status.total_steps = len(self.train_dataloader) * self.status.total_epochs
        
        logger.info("Data loaders created")
    
    def _resume_from_checkpoint(self, device: torch.device):
        """
        Resume training from a checkpoint.
        
        Args:
            device: Device to load the checkpoint to
        
        Returns:
            bool: Whether resuming was successful
        """
        if self.resume_from is None:
            return False
        
        try:
            logger.info(f"Resuming from checkpoint: {self.resume_from}")
            
            # Load checkpoint
            if self.checkpoint_manager is not None:
                # Use checkpoint manager
                self.model, self.optimizer, self.scheduler, step, epoch = self.checkpoint_manager.load_latest_checkpoint(
                    map_location=device
                )
                
                # Update status
                self.status.current_step = step
                self.status.current_epoch = epoch
            else:
                # Basic checkpoint loading
                checkpoint = torch.load(self.resume_from, map_location=device)
                
                # Load model weights
                if isinstance(self.model, DDP):
                    self.model.module.load_state_dict(checkpoint["model"])
                else:
                    self.model.load_state_dict(checkpoint["model"])
                
                # Load optimizer state
                if "optimizer" in checkpoint and self.optimizer is not None:
                    self.optimizer.load_state_dict(checkpoint["optimizer"])
                
                # Load scheduler state
                if "scheduler" in checkpoint and self.scheduler is not None:
                    self.scheduler.load_state_dict(checkpoint["scheduler"])
                
                # Load scaler state
                if "scaler" in checkpoint and self.scaler is not None:
                    self.scaler.load_state_dict(checkpoint["scaler"])
                
                # Update status
                self.status.current_step = checkpoint.get("step", 0)
                self.status.current_epoch = checkpoint.get("epoch", 0)
                
                # Load metrics
                if "metrics" in checkpoint:
                    self.status.latest_metrics = checkpoint["metrics"]
                    
                    # Check for best metric
                    best_metric_name = self.config.get("training", {}).get("best_metric", "val/loss")
                    if best_metric_name in checkpoint["metrics"]:
                        self.status.best_metric = checkpoint["metrics"][best_metric_name]
                        self.status.best_metric_name = best_metric_name
                        self.status.best_metric_step = self.status.current_step
                        self.status.best_metric_epoch = self.status.current_epoch
            
            logger.info(f"Resumed from step {self.status.current_step}, epoch {self.status.current_epoch}")
            
            # Send notification
            self._send_notification(
                message=f"Training resumed from checkpoint at step {self.status.current_step}, epoch {self.status.current_epoch}",
                notification_type=NotificationType.INFO,
                details={
                    "checkpoint_path": self.resume_from,
                    "step": self.status.current_step,
                    "epoch": self.status.current_epoch,
                    "restart_count": self.status.restart_count,
                }
            )
            
            return True
        
        except Exception as e:
            logger.error(f"Error resuming from checkpoint: {e}")
            traceback.print_exc()
            return False
    
    def _setup_wandb(self):
        """Set up Weights & Biases for experiment tracking."""
        if not self.use_wandb or not WANDB_AVAILABLE or not self.is_master:
            return
        
        try:
            # Get W&B configuration
            wandb_config = self.config.get("wandb", {})
            project_name = wandb_config.get("project", self.experiment_name)
            entity = wandb_config.get("entity")
            name = wandb_config.get("name", f"{self.experiment_name}_{self.run_id}")
            tags = wandb_config.get("tags", [])
            
            # Add CI tag if in CI mode
            if self.ci_mode:
                tags.append("ci")
            
            # Initialize W&B
            wandb.init(
                project=project_name,
                entity=entity,
                name=name,
                config=self.config,
                tags=tags,
                resume="allow",
                id=wandb_config.get("id"),
            )
            
            # Log code files if specified
            if wandb_config.get("log_code", False):
                wandb.run.log_code(".")
            
            logger.info(f"W&B initialized: {wandb.run.name} ({wandb.run.id})")
        
        except Exception as e:
            logger.error(f"Error setting up W&B: {e}")
            self.use_wandb = False
    
    def _train_epoch(self, epoch: int, device: torch.device):
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch
            device: Device to train on
            
        Returns:
            Dict[str, float]: Metrics for the epoch
        """
        # This is a placeholder - actual training would depend on the specific model and dataset
        # In a real implementation, this would contain the training loop for one epoch
        
        # For example:
        # self.model.train()
        # epoch_loss = 0.0
        # epoch_samples = 0
        # epoch_start_time = time.time()
        # 
        # # Set epoch for distributed sampler
        # if self.distributed and hasattr(self.train_dataloader.sampler, "set_epoch"):
        #     self.train_dataloader.sampler.set_epoch(epoch)
        # 
        # # Training loop
        # for batch_idx, batch in enumerate(self.train_dataloader):
        #     # Move batch to device
        #     batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        #     
        #     # Forward pass
        #     outputs = self.model(**batch)
        #     loss = outputs["loss"]
        #     
        #     # Backward pass
        #     loss.backward()
        #     
        #     # Optimizer step
        #     self.optimizer.step()
        #     self.optimizer.zero_grad()
        #     
        #     # Scheduler step
        #     if self.scheduler is not None:
        #         self.scheduler.step()
        #     
        #     # Update metrics
        #     batch_size = batch["input_ids"].size(0)
        #     epoch_loss += loss.item() * batch_size
        #     epoch_samples += batch_size
        #     
        #     # Update step
        #     self.status.current_step += 1
        #     
        #     # Log metrics
        #     if self.status.current_step % self.config.get("logging", {}).get("log_interval", 100) == 0:
        #         # Calculate metrics
        #         metrics = {
        #             "train/loss": loss.item(),
        #             "train/lr": self.scheduler.get_last_lr()[0] if self.scheduler else self.optimizer.param_groups[0]["lr"],
        #             "train/epoch": epoch,
        #             "train/step": self.status.current_step,
        #         }
        #         
        #         # Log metrics
        #         self._log_metrics(metrics, self.status.current_step)
        #         
        #         # Update training speed
        #         elapsed = time.time() - epoch_start_time
        #         self._update_training_speed(batch_idx * batch_size, elapsed)
        #         
        #         # Log progress
        #         logger.info(
        #             f"Epoch {epoch} | Step {self.status.current_step} | "
        #             f"Loss: {loss.item():.4f} | "
        #             f"LR: {metrics['train/lr']:.6f} | "
        #             f"Speed: {self.status.training_speed:.1f} examples/s | "
        #             f"ETA: {datetime.timedelta(seconds=int(self.status.estimated_time_remaining))}"
        #         )
        #     
        #     # Save checkpoint
        #     if self.checkpoint_manager is not None and self.status.current_step % self.config.get("checkpointing", {}).get("save_interval_steps", 1000) == 0:
        #         self.checkpoint_manager.save_checkpoint(
        #             step=self.status.current_step,
        #             epoch=epoch,
        #             metrics={"train/loss": loss.item()},
        #             is_best=False,
        #         )
        #         
        #         # Update status
        #         self.status.last_checkpoint_step = self.status.current_step
        #         self.status.last_checkpoint_time = time.time()
        #     
        #     # Validate
        #     if self.val_dataloader is not None and self.status.current_step % self.config.get("validation", {}).get("val_interval", 1000) == 0:
        #         val_metrics = self._validate(device)
        #         
        #         # Log metrics
        #         self._log_metrics(val_metrics, self.status.current_step)
        #         
        #         # Update status
        #         self.status.last_validation_step = self.status.current_step
        #         self.status.last_validation_time = time.time()
        #         
        #         # Check if best model
        #         best_metric_name = self.config.get("training", {}).get("best_metric", "val/loss")
        #         if best_metric_name in val_metrics:
        #             metric_value = val_metrics[best_metric_name]
        #             is_better = False
        #             
        #             # Check if better (lower is better by default)
        #             if self.config.get("training", {}).get("higher_is_better", False):
        #                 is_better = metric_value > self.status.best_metric
        #             else:
        #                 is_better = metric_value < self.status.best_metric
        #             
        #             if is_better:
        #                 # Save best model
        #                 if self.checkpoint_manager is not None:
        #                     self.checkpoint_manager.save_checkpoint(
        #                         step=self.status.current_step,
        #                         epoch=epoch,
        #                         metrics=val_metrics,
        #                         is_best=True,
        #                     )
        #                 
        #                 # Send notification
        #                 self._send_notification(
        #                     message=f"New best model: {best_metric_name}={metric_value:.6f}",
        #                     notification_type=NotificationType.SUCCESS,
        #                     details={
        #                         "metric": best_metric_name,
        #                         "value": metric_value,
        #                         "step": self.status.current_step,
        #                         "epoch": epoch,
        #                     }
        #                 )
        #         
        #         # Set model back to training mode
        #         self.model.train()
        #     
        #     # Check for pause/resume signals
        #     self._check_for_pause_signal()
        #     
        #     # Update resource monitor
        #     if self.resource_monitor is not None:
        #         self.resource_monitor.update_training_metrics(
        #             step=self.status.current_step,
        #             epoch=epoch,
        #             loss=loss.item(),
        #             learning_rate=self.scheduler.get_last_lr()[0] if self.scheduler else self.optimizer.param_groups[0]["lr"],
        #             batch_size=batch_size,
        #             samples_per_second=self.status.training_speed,
        #         )
        #     
        #     # Save status
        #     if self.status.current_step % self.config.get("logging", {}).get("status_interval", 100) == 0:
        #         self._save_status()
        # 
        # # Calculate epoch metrics
        # epoch_metrics = {
        #     "train/epoch_loss": epoch_loss / epoch_samples,
        #     "train/epoch": epoch,
        # }
        # 
        # # Log epoch metrics
        # self._log_metrics(epoch_metrics, self.status.current_step)
        # 
        # # Log epoch summary
        # logger.info(
        #     f"Epoch {epoch} completed | "
        #     f"Loss: {epoch_metrics['train/epoch_loss']:.4f} | "
        #     f"Time: {datetime.timedelta(seconds=int(time.time() - epoch_start_time))}"
        # )
        # 
        # return epoch_metrics
        
        # Placeholder for the example
        import time
        time.sleep(0.1)  # Simulate training
        
        # Return dummy metrics
        return {
            "train/epoch_loss": 1.0 / (epoch + 1),
            "train/epoch": epoch,
        }
    
    def _validate(self, device: torch.device):
        """
        Validate the model.
        
        Args:
            device: Device to validate on
            
        Returns:
            Dict[str, float]: Validation metrics
        """
        # This is a placeholder - actual validation would depend on the specific model and dataset
        # In a real implementation, this would contain the validation loop
        
        # For example:
        # self.model.eval()
        # val_loss = 0.0
        # val_samples = 0
        # 
        # with torch.no_grad():
        #     for batch in self.val_dataloader:
        #         # Move batch to device
        #         batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        #         
        #         # Forward pass
        #         outputs = self.model(**batch)
        #         loss = outputs["loss"]
        #         
        #         # Update metrics
        #         batch_size = batch["input_ids"].size(0)
        #         val_loss += loss.item() * batch_size
        #         val_samples += batch_size
        # 
        # # Calculate metrics
        # metrics = {
        #     "val/loss": val_loss / val_samples,
        # }
        # 
        # # Log validation summary
        # logger.info(
        #     f"Validation | "
        #     f"Loss: {metrics['val/loss']:.4f}"
        # )
        # 
        # return metrics
        
        # Placeholder for the example
        import random
        
        # Return dummy metrics
        return {
            "val/loss": 1.0 / (self.status.current_step + 1) + random.random() * 0.1,
        }
    
    def _test(self, device: torch.device):
        """
        Test the model.
        
        Args:
            device: Device to test on
            
        Returns:
            Dict[str, float]: Test metrics
        """
        # This is a placeholder - actual testing would depend on the specific model and dataset
        # In a real implementation, this would contain the test loop
        
        # Placeholder for the example
        import random
        
        # Return dummy metrics
        return {
            "test/loss": 1.0 / (self.status.current_step + 1) + random.random() * 0.1,
        }
    
    def _check_for_pause_signal(self):
        """Check for pause/resume signals."""
        # Check for pause file
        pause_file = Path(self.output_dir) / "PAUSE"
        if pause_file.exists():
            if not self.status.is_paused:
                logger.info("Pause signal detected, pausing training after current step")
                self.status.is_paused = True
                
                # Send notification
                self._send_notification(
                    message="Training paused",
                    notification_type=NotificationType.INFO,
                    details={
                        "step": self.status.current_step,
                        "epoch": self.status.current_epoch,
                    }
                )
            
            # Wait until pause file is removed
            while pause_file.exists() and self.status.is_paused:
                time.sleep(5)
            
            # Resume training
            if self.status.is_paused:
                logger.info("Pause signal removed, resuming training")
                self.status.is_paused = False
                
                # Send notification
                self._send_notification(
                    message="Training resumed",
                    notification_type=NotificationType.INFO,
                    details={
                        "step": self.status.current_step,
                        "epoch": self.status.current_epoch,
                    }
                )
    
    def _handle_error(self, error: Exception, stage: TrainingStage):
        """
        Handle an error during training.
        
        Args:
            error: The exception that occurred
            stage: The stage where the error occurred
        """
        # Update status
        self.status.stage = TrainingStage.FAILED
        self.status.error_count += 1
        self.status.last_error = str(error)
        
        # Log error
        logger.error(f"Error during {stage.value}: {error}")
        logger.error(traceback.format_exc())
        
        # Save status
        self._save_status()
        
        # Send notification
        self._send_notification(
            message=f"Error during {stage.value}: {error}",
            notification_type=NotificationType.ERROR,
            details={
                "error": str(error),
                "traceback": traceback.format_exc(),
                "stage": stage.value,
                "step": self.status.current_step,
                "epoch": self.status.current_epoch,
            }
        )
        
        # Save emergency checkpoint if in training stage
        if stage == TrainingStage.TRAINING and self.checkpoint_manager is not None:
            try:
                checkpoint_path = self.checkpoint_manager.save_checkpoint(
                    step=self.status.current_step,
                    epoch=self.status.current_epoch,
                    metrics=self.status.latest_metrics,
                    is_emergency=True,
                    tag="error",
                )
                if checkpoint_path:
                    logger.info(f"Saved emergency checkpoint to {checkpoint_path}")
            except Exception as e:
                logger.error(f"Error saving emergency checkpoint: {e}")
    
    def _finalize_training(self):
        """Finalize training and save artifacts."""
        if not self.is_master:
            return
        
        logger.info("Finalizing training")
        
        try:
            # Save final checkpoint
            if self.checkpoint_manager is not None:
                checkpoint_path = self.checkpoint_manager.save_checkpoint(
                    step=self.status.current_step,
                    epoch=self.status.current_epoch,
                    metrics=self.status.latest_metrics,
                    tag="final",
                )
                if checkpoint_path:
                    logger.info(f"Saved final checkpoint to {checkpoint_path}")
            
            # Save final status
            self.status.stage = TrainingStage.COMPLETED
            self._save_status()
            
            # Create artifacts
            artifacts_dir = Path(self.output_dir) / "artifacts"
            artifacts_dir.mkdir(exist_ok=True)
            
            # Save training summary
            summary = {
                "experiment_name": self.experiment_name,
                "run_id": self.run_id,
                "start_time": self.status.start_time,
                "end_time": time.time(),
                "duration": time.time() - self.status.start_time,
                "duration_formatted": str(datetime.timedelta(seconds=int(time.time() - self.status.start_time))),
                "epochs": self.status.current_epoch,
                "steps": self.status.current_step,
                "best_metric": {
                    "name": self.status.best_metric_name,
                    "value": self.status.best_metric,
                    "step": self.status.best_metric_step,
                    "epoch": self.status.best_metric_epoch,
                },
                "latest_metrics": self.status.latest_metrics,
                "restart_count": self.status.restart_count,
                "error_count": self.status.error_count,
            }
            
            with open(artifacts_dir / "training_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            
            # Save resource usage summary if available
            if self.resource_monitor is not None:
                resource_summary = self.resource_monitor.get_summary()
                with open(artifacts_dir / "resource_summary.json", "w") as f:
                    json.dump(resource_summary, f, indent=2)
            
            # Send final notification
            self._send_notification(
                message=f"Training completed: {self.experiment_name}",
                notification_type=NotificationType.SUCCESS,
                details={
                    "duration": summary["duration_formatted"],
                    "epochs": summary["epochs"],
                    "steps": summary["steps"],
                    "best_metric": f"{summary['best_metric']['name']}={summary['best_metric']['value']:.6f}",
                }
            )
            
            logger.info(f"Training completed in {summary['duration_formatted']}")
            logger.info(f"Best {summary['best_metric']['name']}: {summary['best_metric']['value']:.6f}")
        
        except Exception as e:
            logger.error(f"Error finalizing training: {e}")
    
    def initialize(self):
        """Initialize training components."""
        logger.info("Initializing training")
        self.status.stage = TrainingStage.INITIALIZING
        
        try:
            # Check for CI artifacts
            self._check_for_ci_artifacts()
            
            # Validate environment
            self._validate_environment()
            
            # Load previous status if available
            self._load_status()
            
            # Set up notifications
            self._setup_notifications()
            
            logger.info("Initialization complete")
            return True
        
        except Exception as e:
            self._handle_error(e, TrainingStage.INITIALIZING)
            return False
    
    def prepare_environment(self, rank: int = 0):
        """
        Prepare the training environment.
        
        Args:
            rank: Rank of the current process
            
        Returns:
            bool: Whether preparation was successful
        """
        logger.info("Preparing environment")
        self.status.stage = TrainingStage.ENVIRONMENT_SETUP
        
        try:
            # Set up distributed training if needed
            self._setup_distributed(rank)
            
            # Set up device
            device = self._setup_device(rank)
            
            # Set random seed
            self._set_seed(rank=rank)
            
            # Set up W&B
            if self.is_master:
                self._setup_wandb()
            
            # Set up monitoring
            if self.is_master:
                self._setup_monitoring()
            
            logger.info("Environment preparation complete")
            return True
        
        except Exception as e:
            self._handle_error(e, TrainingStage.ENVIRONMENT_SETUP)
            return False
    
    def prepare_data(self):
        """
        Prepare data for training.
        
        Returns:
            bool: Whether preparation was successful
        """
        logger.info("Preparing data")
        self.status.stage = TrainingStage.DATA_PREPARATION
        
        try:
            # Create data loaders
            self._create_dataloaders()
            
            logger.info("Data preparation complete")
            return True
        
        except Exception as e:
            self._handle_error(e, TrainingStage.DATA_PREPARATION)
            return False
    
    def prepare_model(self, rank: int = 0):
        """
        Prepare the model for training.
        
        Args:
            rank: Rank of the current process
            
        Returns:
            bool: Whether preparation was successful
        """
        logger.info("Preparing model")
        self.status.stage = TrainingStage.MODEL_PREPARATION
        
        try:
            # Set up device
            device = self._setup_device(rank)
            
            # Create model
            self._create_model()
            
            # Apply MPS optimization if available and using MPS
            if MPS_OPTIMIZER_AVAILABLE and device.type == "mps":
                self.model, device, self.scaler, mps_optimizer = optimize_for_mps(
                    model=self.model,
                    config=self.config,
                    operation_mode=self.config.get("mps", {}).get("operation_mode", "optimal"),
                )
            else:
                # Move model to device
                self.model.to(device)
            
            # Wrap model with DDP if distributed
            if self.distributed:
                self.model = DDP(
                    self.model,
                    device_ids=[device.index] if device.type == "cuda" else None,
                    output_device=device.index if device.type == "cuda" else None,
                )
            
            # Create optimizer
            self._create_optimizer()
            
            # Create scheduler
            self._create_scheduler()
            
            # Set up checkpoint manager
            if self.is_master:
                self._setup_checkpoint_manager()
            
            # Resume from checkpoint if specified
            if self.resume_from:
                self._resume_from_checkpoint(device)
            
            logger.info("Model preparation complete")
            return True
        
        except Exception as e:
            self._handle_error(e, TrainingStage.MODEL_PREPARATION)
            return False
    
    def train(self, rank: int = 0):
        """
        Train the model.
        
        Args:
            rank: Rank of the current process
            
        Returns:
            bool: Whether training was successful
        """
        logger.info("Starting training")
        self.status.stage = TrainingStage.TRAINING
        
        try:
            # Set up device
            device = self._setup_device(rank)
            
            # Training loop
            for epoch in range(self.status.current_epoch, self.status.total_epochs):
                # Update status
                self.status.current_epoch = epoch
                
                # Train for one epoch
                epoch_metrics = self._train_epoch(epoch, device)
                
                # Validate at the end of each epoch
                val_metrics = self._validate(device)
                
                # Log metrics
                if self.is_master:
                    self._log_metrics(epoch_metrics, self.status.current_step)
                    self._log_metrics(val_metrics, self.status.current_step)
                
                # Save checkpoint at the end of each epoch
                if self.is_master and self.checkpoint_manager is not None:
                    metrics = {**epoch_metrics, **val_metrics}
                    self.checkpoint_manager.save_checkpoint(
                        step=self.status.current_step,
                        epoch=epoch,
                        metrics=metrics,
                        tag=f"epoch_{epoch}",
                    )
                    
                    # Update status
                    self.status.last_checkpoint_step = self.status.current_step
                    self.status.last_checkpoint_time = time.time()
                
                # Save status
                if self.is_master:
                    self._save_status()
                
                # Send epoch notification
                if self.is_master:
                    self._send_notification(
                        message=f"Epoch {epoch} completed",
                        notification_type=NotificationType.PROGRESS,
                        details={
                            "epoch": epoch,
                            "epochs_total": self.status.total_epochs,
                            "step": self.status.current_step,
                            "train_loss": epoch_metrics.get("train/epoch_loss", 0.0),
                            "val_loss": val_metrics.get("val/loss", 0.0),
                            "best_metric": f"{self.status.best_metric_name}={self.status.best_metric:.6f}",
                        }
                    )
            
            logger.info("Training complete")
            return True
        
        except Exception as e:
            self._handle_error(e, TrainingStage.TRAINING)
            return False
    
    def validate(self, rank: int = 0):
        """
        Validate the model.
        
        Args:
            rank: Rank of the current process
            
        Returns:
            bool: Whether validation was successful
        """
        logger.info("Starting validation")
        self.status.stage = TrainingStage.VALIDATION
        
        try:
            # Set up device
            device = self._setup_device(rank)
            
            # Validate
            metrics = self._validate(device)
            
            # Log metrics
            if self.is_master:
                self._log_metrics(metrics, self.status.current_step