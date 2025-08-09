"""
Configuration module for HRM code generation adaptation.

This module provides configuration classes and utilities for adapting
the Hierarchical Reasoning Model (HRM) to code generation tasks.
"""

import os
import sys
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field

import yaml
from pydantic import BaseModel, validator, root_validator

# --------------------------------------------------------------------------- #
# Ensure the `external/` directory (which contains the vendored Sapient HRM
# code under the package name `sapient_hrm`) is on the Python path.  This
# mirrors the logic used in other hrm_codegen modules so that the package can
# be imported without requiring an editable install or site-wide install.
# --------------------------------------------------------------------------- #
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
# The project root is the parent directory of `hrm_codegen`
_ROOT_DIR = os.path.dirname(_CUR_DIR)
_EXTERNAL_DIR = os.path.join(_ROOT_DIR, "external")

if _EXTERNAL_DIR not in sys.path:
    sys.path.append(_EXTERNAL_DIR)
    sys.path.append(_ROOT_DIR)

# Add the *actual* vendored Sapient-HRM source directory so that the internal
# package layout starting at ``models`` is importable.  The folder on disk is
# named ``sapient-hrm`` (with a hyphen), which is *not* a valid Python package
# name; however, its sub-packages (e.g. ``models``) are.  By inserting the
# directory itself onto ``sys.path`` we can import those sub-packages
# directly.
_SAP_HRM_DIR = os.path.join(_EXTERNAL_DIR, "sapient-hrm")
if _SAP_HRM_DIR not in sys.path:
    sys.path.append(_SAP_HRM_DIR)

# Import original HRM config from vendored Sapient code via direct path (models.*)
# We add `external/sapient-hrm` to sys.path above, so importing `models.*` works.
from models.hrm.hrm_act_v1 import (  # type: ignore
    HierarchicalReasoningModel_ACTV1Config,
)


class CodeGenConfig(BaseModel):
    """
    Configuration for HRM code generation adaptation.

    This class extends the original HRM configuration with code generation
    specific parameters and provides methods for loading from YAML files.
    """

    # Task specification
    task: str = "codegen"
    tokenizer_name: str = "gpt2"

    # Model architecture
    model: Dict[str, Any] = {
        "name": "HierarchicalReasoningModel_ACTV1",
        "causal": True,
        "hidden_size": 768,
        "num_heads": 12,
        "expansion": 4.0,
        "pos_encodings": "rope",
        "H_layers": 6,
        "L_layers": 2,
        "H_cycles": 1,
        "L_cycles": 2,
        "puzzle_emb_ndim": 0,
        "num_puzzle_identifiers": 1,
        "vocab_size": 50257,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "forward_dtype": "bfloat16",
        "halt_max_steps": 1,
        "halt_exploration_prob": 0.0,
        "enable_q_head": False,
    }

    # Data configuration
    data: Dict[str, Any] = {
        "train_path": "data/mbpp/train_raw.json",
        "test_path": "data/mbpp/test_raw.json",
        "seq_len": 512,
        "batch_size": 8,
        "prompt_template": "# {text}\n\n",
        "include_tests_in_prompt": True,
        "test_template": "# Test cases:\n# {test}\n\n",
        "dev_mode": False,
        "dev_samples": 100,
        "validate_data": True,
    }

    # Training configuration
    training: Dict[str, Any] = {
        "optimizer": {
            "name": "AdamW",
            "lr": 5e-5,
            "weight_decay": 0.01,
            "beta1": 0.9,
            "beta2": 0.95,
        },
        "scheduler": {"name": "cosine", "warmup_steps": 100},
        "max_steps": 10000,
        "eval_every": 500,
        "save_every": 1000,
        "gradient_accumulation_steps": 4,
        "fp16": True,
        "bf16": False,
    }

    # Evaluation configuration
    evaluation: Dict[str, Any] = {
        "metric": "pass@k",
        "k_values": [1, 5, 10],
        "max_generate_tokens": 256,
        "temperature": 0.8,
        "num_samples": 100,
        "timeout_seconds": 5,
    }

    # Logging and checkpoints
    logging: Dict[str, Any] = {
        "log_level": "INFO",
        "use_wandb": True,
        "project_name": "hrm-codegen",
    }

    checkpoints: Dict[str, Any] = {"path": "checkpoints/codegen", "keep_last_n": 3}

    @validator("task")
    def validate_task(cls, v):
        """Validate that the task is supported."""
        if v not in ["codegen", "puzzle"]:
            raise ValueError(f"Task {v} not supported. Must be one of: codegen, puzzle")
        return v

    @root_validator
    def validate_codegen_config(cls, values):
        """Validate code generation specific parameters."""
        task = values.get("task")
        model = values.get("model", {})

        if task == "codegen":
            # Ensure causal is True for code generation
            if not model.get("causal", True):
                raise ValueError("Causal must be True for code generation task")

            # Ensure puzzle_emb_ndim is 0 for code generation
            if model.get("puzzle_emb_ndim", 0) != 0:
                raise ValueError("puzzle_emb_ndim must be 0 for code generation task")

        return values

    def to_hrm_config(
        self, batch_size: Optional[int] = None, seq_len: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Convert this config to the format expected by the original HRM model.

        Args:
            batch_size: Override batch size if provided
            seq_len: Override sequence length if provided

        Returns:
            Dictionary with HRM configuration parameters
        """
        # Start with model config
        hrm_config = dict(self.model)

        # Override batch size and sequence length if provided
        hrm_config["batch_size"] = batch_size or self.data["batch_size"]
        hrm_config["seq_len"] = seq_len or self.data["seq_len"]

        # Remove code generation specific parameters not expected by original HRM
        hrm_config.pop("causal", None)
        hrm_config.pop("enable_q_head", None)
        hrm_config.pop("name", None)

        return hrm_config

    def create_hrm_config(
        self, batch_size: Optional[int] = None, seq_len: Optional[int] = None
    ) -> HierarchicalReasoningModel_ACTV1Config:
        """
        Create an instance of the original HRM configuration.

        Args:
            batch_size: Override batch size if provided
            seq_len: Override sequence length if provided

        Returns:
            HierarchicalReasoningModel_ACTV1Config instance
        """
        config_dict = self.to_hrm_config(batch_size, seq_len)
        return HierarchicalReasoningModel_ACTV1Config(**config_dict)


@dataclass
class RuntimeConfig:
    """Runtime configuration with merged defaults and user overrides."""

    config: CodeGenConfig

    # Runtime parameters that may be overridden at execution time
    device: str = "cuda"
    precision: str = "bf16-mixed"
    seed: int = 42
    debug: bool = False

    # Paths
    config_path: Optional[str] = None
    output_dir: str = field(default_factory=lambda: os.path.join("outputs", "codegen"))

    def __post_init__(self):
        """Ensure output directory exists."""
        os.makedirs(self.output_dir, exist_ok=True)


def load_config(config_path: str) -> CodeGenConfig:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        CodeGenConfig instance
    """
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return CodeGenConfig(**config_dict)


def merge_configs(
    base_config: CodeGenConfig, override_config: Dict[str, Any]
) -> CodeGenConfig:
    """
    Merge a base configuration with override values.

    Args:
        base_config: Base configuration
        override_config: Dictionary with override values

    Returns:
        New CodeGenConfig with merged values
    """
    # Convert base config to dict
    base_dict = base_config.dict()

    # Deep merge the override values
    merged_dict = _deep_merge(base_dict, override_config)

    # Create new config with merged values
    return CodeGenConfig(**merged_dict)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries, with override values taking precedence.

    Args:
        base: Base dictionary
        override: Dictionary with override values

    Returns:
        Merged dictionary
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = _deep_merge(result[key], value)
        else:
            # Override or add the value
            result[key] = value

    return result


def get_default_config() -> CodeGenConfig:
    """
    Get the default configuration for code generation.

    Returns:
        Default CodeGenConfig instance
    """
    return CodeGenConfig()


def create_runtime_config(
    config_path: Optional[str] = None, **overrides
) -> RuntimeConfig:
    """
    Create a runtime configuration with optional overrides.

    Args:
        config_path: Path to the YAML configuration file (optional)
        **overrides: Override values for the configuration

    Returns:
        RuntimeConfig instance
    """
    # Load base config
    if config_path and os.path.exists(config_path):
        config = load_config(config_path)
    else:
        config = get_default_config()

    # Apply overrides to config if any
    config_overrides = overrides.pop("config", {})
    if config_overrides:
        config = merge_configs(config, config_overrides)

    # Create runtime config with remaining overrides
    runtime_config = RuntimeConfig(config=config, config_path=config_path, **overrides)

    return runtime_config
