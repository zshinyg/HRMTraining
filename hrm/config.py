"""
Configuration module for the Hierarchical Reasoning Model (HRM) adapted for code generation.

This module provides dataclasses for configuring all aspects of the HRM model:
- Model architecture (dimensions, layers, etc.)
- Training hyperparameters (learning rate, batch size, etc.)
- Data processing (vocabulary, sequence length, etc.)
- Evaluation metrics and settings
- Logging and checkpointing

The configuration can be loaded from YAML files and includes validation methods.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import yaml
from torch import nn


class ActivationType(str, Enum):
    """Supported activation functions."""

    RELU = "relu"
    GELU = "gelu"
    SWISH = "swish"
    SILU = "silu"
    TANH = "tanh"


class OptimType(str, Enum):
    """Supported optimizers."""

    ADAMW = "adamw"
    ADAM = "adam"
    SGD = "sgd"
    ADAFACTOR = "adafactor"


class SchedulerType(str, Enum):
    """Supported learning rate schedulers."""

    COSINE = "cosine"
    LINEAR = "linear"
    CONSTANT = "constant"
    WARMUP_COSINE = "warmup_cosine"
    WARMUP_LINEAR = "warmup_linear"
    WARMUP_CONSTANT = "warmup_constant"


@dataclass
class ModelConfig:
    """Configuration for the HRM model architecture."""

    # General model parameters
    hidden_dim: int = 768
    num_layers: int = 4

    # Hierarchical structure parameters
    high_level_dim: int = 512  # Planner hidden dimension
    low_level_dim: int = 768  # Executor hidden dimension
    high_level_layers: int = 2  # Number of planner layers
    low_level_layers: int = 4  # Number of executor layers

    # Attention and processing parameters
    num_heads: int = 8
    dropout: float = 0.1
    activation: ActivationType = ActivationType.GELU
    layer_norm_eps: float = 1e-5

    # Code generation specific parameters
    vocab_size: int = 32000  # Typically larger for code
    max_position_embeddings: int = 2048  # Longer for code sequences
    tie_word_embeddings: bool = True

    # Hierarchical timing parameters
    high_level_steps: int = 8  # How many steps the high-level module takes
    timing_ratio: int = 4  # Ratio of low-level to high-level steps

    # Initialization
    initializer_range: float = 0.02

    def __post_init__(self):
        """Validate the configuration."""
        if self.high_level_dim <= 0 or self.low_level_dim <= 0:
            raise ValueError("Hidden dimensions must be positive")

        if self.high_level_layers <= 0 or self.low_level_layers <= 0:
            raise ValueError("Number of layers must be positive")

        if self.timing_ratio <= 0:
            raise ValueError("Timing ratio must be positive")

        if self.vocab_size <= 0:
            raise ValueError("Vocabulary size must be positive")

    @property
    def low_level_steps(self) -> int:
        """Calculate the number of low-level steps based on the timing ratio."""
        return self.high_level_steps * self.timing_ratio

    @property
    def total_params(self) -> int:
        """Estimate the total number of parameters in the model."""
        # This is a rough estimate based on the dimensions
        high_level_params = (
            self.high_level_dim * self.high_level_dim * 4 * self.high_level_layers
        )
        low_level_params = (
            self.low_level_dim * self.low_level_dim * 4 * self.low_level_layers
        )
        embedding_params = self.vocab_size * self.hidden_dim
        output_params = self.hidden_dim * self.vocab_size

        return high_level_params + low_level_params + embedding_params + output_params


@dataclass
class DataConfig:
    """Configuration for data processing."""

    # Data paths
    train_data_path: str = "data/mbpp/train.bin"
    val_data_path: str = "data/mbpp/val.bin"
    test_data_path: str = "data/mbpp/test.bin"

    # Tokenization
    vocab_file: str = "data/mbpp/vocab.json"
    merges_file: Optional[str] = None
    tokenizer_type: str = "gpt2"  # Options: "gpt2", "bert", "custom"

    # Sequence parameters
    max_seq_length: int = 1024
    context_length: int = 512  # Length of context provided to the model
    target_length: int = 512  # Length of target code to generate

    # Data processing
    num_workers: int = 4
    pin_memory: bool = True

    # Augmentation and preprocessing
    use_augmentation: bool = False
    augmentation_factor: int = 1
    filter_by_length: bool = True
    min_length: int = 10
    max_length: int = 1024

    def __post_init__(self):
        """Validate the configuration."""
        if self.max_seq_length <= 0:
            raise ValueError("Max sequence length must be positive")

        if self.context_length <= 0 or self.target_length <= 0:
            raise ValueError("Context and target lengths must be positive")

        if self.context_length + self.target_length > self.max_seq_length:
            raise ValueError(
                f"Context ({self.context_length}) + target ({self.target_length}) "
                f"exceeds max sequence length ({self.max_seq_length})"
            )


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""

    # Basic training parameters
    epochs: int = 20
    global_batch_size: int = 64
    gradient_accumulation_steps: int = 1

    # Optimizer settings
    optimizer: OptimType = OptimType.ADAMW
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Scheduler settings
    scheduler: SchedulerType = SchedulerType.WARMUP_COSINE
    warmup_steps: int = 1000
    warmup_ratio: float = 0.1

    # Mixed precision training
    use_mixed_precision: bool = True
    mixed_precision_dtype: str = "float16"  # "float16" or "bfloat16"

    # Distributed training
    distributed: bool = False
    local_rank: int = -1
    world_size: int = 1

    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1

    # Learning rate for specific components
    embedding_lr: Optional[float] = None
    high_level_lr: Optional[float] = None
    low_level_lr: Optional[float] = None

    def __post_init__(self):
        """Validate the configuration."""
        if self.epochs <= 0:
            raise ValueError("Number of epochs must be positive")

        if self.global_batch_size <= 0:
            raise ValueError("Batch size must be positive")

        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

        if self.weight_decay < 0:
            raise ValueError("Weight decay must be non-negative")

        if self.warmup_ratio < 0 or self.warmup_ratio > 1:
            raise ValueError("Warmup ratio must be between 0 and 1")

    @property
    def effective_batch_size(self) -> int:
        """Calculate the effective batch size with gradient accumulation."""
        return self.global_batch_size * self.gradient_accumulation_steps


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""

    # Evaluation settings
    eval_interval: int = 1000
    eval_steps: int = 100

    # Metrics
    metrics: List[str] = field(default_factory=lambda: ["pass@1", "pass@5", "pass@10"])

    # Generation parameters
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    num_beams: int = 1
    num_return_sequences: int = 1
    max_new_tokens: int = 512

    # Code execution settings
    timeout: int = 5  # Seconds
    max_memory: int = 1024  # MB
    use_sandbox: bool = True

    # Test case validation
    test_cases_per_problem: int = 3
    additional_test_cases: bool = False

    def __post_init__(self):
        """Validate the configuration."""
        if self.eval_interval <= 0:
            raise ValueError("Evaluation interval must be positive")

        if self.temperature <= 0:
            raise ValueError("Temperature must be positive")

        if self.top_p <= 0 or self.top_p > 1:
            raise ValueError("Top-p must be between 0 and 1")

        if self.top_k <= 0:
            raise ValueError("Top-k must be positive")


@dataclass
class LoggingConfig:
    """Configuration for logging and checkpoints."""

    # Directories
    output_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Checkpoint settings
    save_interval: int = 5000
    save_total_limit: int = 5
    save_best: bool = True

    # Logging settings
    log_interval: int = 100
    log_grad_norm: bool = False
    log_memory: bool = False
    log_samples: bool = True
    num_log_samples: int = 3

    # Wandb integration
    use_wandb: bool = False
    wandb_project: str = "hrm-codegen"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # Tensorboard
    use_tensorboard: bool = True

    def __post_init__(self):
        """Validate the configuration and create directories."""
        if self.save_interval <= 0:
            raise ValueError("Save interval must be positive")

        if self.log_interval <= 0:
            raise ValueError("Log interval must be positive")

        # Create directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)


@dataclass
class HRMConfig:
    """Master configuration class for the HRM model."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Meta information
    name: str = "hrm-codegen"
    description: str = "HRM model adapted for code generation"
    version: str = "0.1.0"
    seed: int = 42

    @classmethod
    def from_yaml(cls, yaml_file: Union[str, Path]) -> HRMConfig:
        """Load configuration from a YAML file."""
        with open(yaml_file, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> HRMConfig:
        """Load configuration from a dictionary."""
        model_config = ModelConfig(**config_dict.get("model", {}))
        data_config = DataConfig(**config_dict.get("data", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        evaluation_config = EvaluationConfig(**config_dict.get("evaluation", {}))
        logging_config = LoggingConfig(**config_dict.get("logging", {}))

        # Extract top-level keys that aren't nested configs
        top_level_keys = {"name", "description", "version", "seed"}
        top_level_dict = {k: v for k, v in config_dict.items() if k in top_level_keys}

        return cls(
            model=model_config,
            data=data_config,
            training=training_config,
            evaluation=evaluation_config,
            logging=logging_config,
            **top_level_dict,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        config_dict = asdict(self)
        return config_dict

    def save(self, yaml_file: Union[str, Path]) -> None:
        """Save the configuration to a YAML file."""
        config_dict = self.to_dict()

        with open(yaml_file, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def validate(self) -> bool:
        """Validate the entire configuration."""
        try:
            self.model.__post_init__()
            self.data.__post_init__()
            self.training.__post_init__()
            self.evaluation.__post_init__()
            self.logging.__post_init__()
            return True
        except ValueError as e:
            print(f"Configuration validation error: {e}")
            return False


def get_default_mbpp_config() -> HRMConfig:
    """Return a default configuration for MBPP dataset."""
    config = HRMConfig(
        name="hrm-mbpp",
        description="HRM model trained on MBPP dataset",
        model=ModelConfig(
            hidden_dim=768,
            high_level_dim=512,
            low_level_dim=768,
            high_level_layers=2,
            low_level_layers=4,
            num_heads=8,
            dropout=0.1,
            vocab_size=50257,  # GPT-2 vocabulary size
            max_position_embeddings=1024,
            high_level_steps=8,
            timing_ratio=4,
        ),
        data=DataConfig(
            train_data_path="data/mbpp/train.bin",
            val_data_path="data/mbpp/val.bin",
            test_data_path="data/mbpp/test.bin",
            vocab_file="data/mbpp/vocab.json",
            max_seq_length=1024,
            context_length=256,
            target_length=768,
        ),
        training=TrainingConfig(
            epochs=20,
            global_batch_size=32,
            gradient_accumulation_steps=4,
            learning_rate=5e-5,
            weight_decay=0.01,
            scheduler=SchedulerType.WARMUP_COSINE,
            warmup_steps=1000,
        ),
        evaluation=EvaluationConfig(
            eval_interval=1000,
            metrics=["pass@1", "pass@5", "pass@10"],
            temperature=0.8,
            top_p=0.95,
            max_new_tokens=512,
        ),
        logging=LoggingConfig(
            output_dir="checkpoints/hrm-mbpp",
            log_dir="logs/hrm-mbpp",
            save_interval=5000,
        ),
    )

    return config


if __name__ == "__main__":
    # Example usage
    config = get_default_mbpp_config()
    print(f"Model has approximately {config.model.total_params:,} parameters")

    # Save to YAML
    config.save("configs/hrm/mbpp_base.yaml")

    # Load from YAML
    loaded_config = HRMConfig.from_yaml("configs/hrm/mbpp_base.yaml")
    assert config.model.hidden_dim == loaded_config.model.hidden_dim

    print("Configuration validation successful!")
