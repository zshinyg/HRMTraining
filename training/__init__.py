"""
Training package for HRM code generation.

This package contains training utilities, trainers, optimizers, and schedulers
for the Hierarchical Reasoning Model (HRM) adaptation to code generation tasks.
It provides a flexible training framework that supports both full and incremental
training, with configurable optimization strategies and evaluation metrics.

Key components:
- Trainer: Main training loop implementation with gradient accumulation
- Optimizers: AdamW with weight decay and customizable parameters
- Schedulers: Learning rate scheduling with warmup and decay
- Checkpointing: Model saving/loading with best model tracking
- Metrics: Training and evaluation metrics for code generation
- Logging: Integration with various logging backends (console, WandB, TensorBoard)
"""

__version__ = "0.1.0"

# Import main components (these will be implemented in separate files)
# from .trainer import Trainer, TrainingConfig, TrainingState
# from .optimization import create_optimizer, create_scheduler
# from .checkpointing import save_checkpoint, load_checkpoint, save_best_model
# from .metrics import compute_loss, compute_metrics, log_metrics
# from .callbacks import TrainingCallback, EarlyStoppingCallback, CheckpointCallback

# Public exports
__all__ = [
    # Core training components
    "Trainer",
    "TrainingConfig",
    "TrainingState",
    # Optimization
    "create_optimizer",
    "create_scheduler",
    # Checkpointing
    "save_checkpoint",
    "load_checkpoint",
    "save_best_model",
    # Metrics and logging
    "compute_loss",
    "compute_metrics",
    "log_metrics",
    # Callbacks
    "TrainingCallback",
    "EarlyStoppingCallback",
    "CheckpointCallback",
]

# Note: The imports are commented out until the corresponding modules are implemented.
# They will be uncommented as each module is completed during Day 3 implementation.
