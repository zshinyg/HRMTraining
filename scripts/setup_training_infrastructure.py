#!/usr/bin/env python
"""
Training Infrastructure Setup Script

This script provides a comprehensive setup for the HRM training infrastructure,
validating the environment, initializing components, setting up monitoring and
recovery systems, and ensuring all prerequisites are met before training begins.

Features:
- Environment validation and dependency checking
- System requirements verification (CPU, GPU, memory, disk space)
- Training infrastructure component initialization
- Monitoring, recovery, and checkpoint system setup
- Configuration validation and default creation
- Logging and experiment tracking setup
- Directory structure creation and permission validation
- GPU/MPS availability testing and optimization
- CI/CD integration and webhook validation
- Health checks and status reporting

Usage:
    python scripts/setup_training_infrastructure.py --config configs/training_config.yaml
    python scripts/setup_training_infrastructure.py --experiment-name hrm_training --output-dir outputs
    python scripts/setup_training_infrastructure.py --validate-only --check-gpu
    python scripts/setup_training_infrastructure.py --initialize-all --force
"""

import argparse
import json
import logging
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("setup_infrastructure")

# Optional imports with fallbacks
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import git
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# Default configuration values
DEFAULT_CONFIG = {
    "experiment_name": "hrm_training",
    "output_dir": "outputs",
    "data_dir": "data",
    "model": {
        "type": "hrm",
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "gradient_checkpointing": False,
    },
    "training": {
        "batch_size": 32,
        "learning_rate": 1e-4,
        "max_epochs": 10,
        "gradient_accumulation_steps": 1,
        "mixed_precision": False,
        "seed": 42,
    },
    "optimizer": {
        "type": "adamw",
        "weight_decay": 0.01,
        "beta1": 0.9,
        "beta2": 0.999,
    },
    "scheduler": {
        "type": "cosine",
        "warmup_steps": 1000,
    },
    "data": {
        "train_file": "mbpp_train.jsonl",
        "val_file": "mbpp_valid.jsonl",
        "test_file": "mbpp_test.jsonl",
        "max_seq_length": 1024,
    },
    "checkpointing": {
        "save_every_n_steps": 1000,
        "keep_last_n": 5,
        "save_best": True,
        "metric": "val/loss",
        "mode": "min",
    },
    "logging": {
        "log_every_n_steps": 100,
        "eval_every_n_steps": 1000,
        "use_wandb": True,
        "use_tensorboard": True,
    },
    "monitoring": {
        "enabled": True,
        "check_interval_seconds": 10.0,
        "resource_check_interval_seconds": 30.0,
        "log_interval_seconds": 60.0,
    },
    "recovery": {
        "enabled": True,
        "auto_recovery_enabled": True,
        "max_recovery_attempts": 5,
    },
    "distributed": {
        "enabled": False,
        "backend": "nccl",
        "world_size": 1,
        "master_addr": "localhost",
        "master_port": 12355,
    },
    "ci": {
        "enabled": False,
        "provider": "github",
        "status_update_enabled": True,
        "artifact_collection_enabled": True,
    },
}

# Required dependencies
REQUIRED_PACKAGES = [
    "torch>=1.10.0",
    "numpy>=1.20.0",
    "pyyaml>=5.1",
    "tqdm>=4.45.0",
]

OPTIONAL_PACKAGES = [
    "wandb>=0.12.0",
    "tensorboard>=2.5.0",
    "psutil>=5.8.0",
    "matplotlib>=3.4.0",
    "pandas>=1.3.0",
    "scikit-learn>=1.0.0",
    "gitpython>=3.1.0",
    "requests>=2.25.0",
]

# Minimum system requirements
MIN_SYSTEM_REQUIREMENTS = {
    "cpu_cores": 2,
    "ram_gb": 8,
    "disk_space_gb": 10,
    "python_version": (3, 7),
}


class InfrastructureSetup:
    """
    Training Infrastructure Setup
    
    This class manages the setup and validation of the complete training infrastructure,
    including environment validation, component initialization, and health checks.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        experiment_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        data_dir: Optional[str] = None,
        validate_only: bool = False,
        initialize_all: bool = False,
        force: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize the infrastructure setup.
        
        Args:
            config_path: Path to configuration file
            experiment_name: Name of the experiment (overrides config)
            output_dir: Directory to save outputs (overrides config)
            data_dir: Directory containing data (overrides config)
            validate_only: Only validate, don't initialize
            initialize_all: Initialize all components
            force: Force initialization even if validation fails
            verbose: Enable verbose logging
        """
        # Set logging level
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # Store initialization parameters
        self.config_path = config_path
        self.validate_only = validate_only
        self.initialize_all = initialize_all
        self.force = force
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Override config with constructor arguments
        if experiment_name:
            self.config["experiment_name"] = experiment_name
        
        if output_dir:
            self.config["output_dir"] = output_dir
        
        if data_dir:
            self.config["data_dir"] = data_dir
        
        # Initialize state
        self.experiment_name = self.config["experiment_name"]
        self.output_dir = Path(self.config["output_dir"])
        self.data_dir = Path(self.config["data_dir"])
        
        # Component initialization state
        self.components_initialized = set()
        self.components_failed = set()
        
        # Validation state
        self.validation_results = {}
        self.environment_valid = False
        self.config_valid = False
        self.system_valid = False
        
        # Component references
        self.orchestrator = None
        self.health_monitor = None
        self.recovery_manager = None
        self.failure_analyzer = None
        self.checkpoint_manager = None
        self.resource_monitor = None
        self.mps_optimizer = None
        
        logger.info(f"Infrastructure setup initialized for experiment: {self.experiment_name}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        # Start with default config
        config = DEFAULT_CONFIG.copy()
        
        # Load from file if provided
        if config_path:
            try:
                with open(config_path, "r") as f:
                    file_config = yaml.safe_load(f)
                
                # Update config with file values (recursive update)
                self._update_config_recursive(config, file_config)
                
                logger.info(f"Loaded configuration from {config_path}")
            
            except Exception as e:
                logger.error(f"Error loading configuration from {config_path}: {e}")
                logger.warning("Using default configuration")
        
        return config
    
    def _update_config_recursive(self, base_config: Dict[str, Any], update_config: Dict[str, Any]):
        """
        Update configuration recursively.
        
        Args:
            base_config: Base configuration to update
            update_config: Configuration with updates
        """
        for key, value in update_config.items():
            if (
                key in base_config and
                isinstance(base_config[key], dict) and
                isinstance(value, dict)
            ):
                self._update_config_recursive(base_config[key], value)
            else:
                base_config[key] = value
    
    def validate_environment(self) -> bool:
        """
        Validate the environment for training.
        
        Returns:
            bool: Whether the environment is valid
        """
        logger.info("Validating environment")
        
        # Track validation results
        results = {}
        
        # Check Python version
        python_version = sys.version_info
        min_python = MIN_SYSTEM_REQUIREMENTS["python_version"]
        python_valid = python_version >= min_python
        
        results["python_version"] = {
            "value": f"{python_version.major}.{python_version.minor}.{python_version.micro}",
            "required": f">={min_python[0]}.{min_python[1]}",
            "valid": python_valid,
        }
        
        if not python_valid:
            logger.error(
                f"Python version {python_version.major}.{python_version.minor} is not supported. "
                f"Please use Python {min_python[0]}.{min_python[1]} or higher."
            )
        
        # Check required packages
        missing_packages = []
        package_results = {}
        
        for package_spec in REQUIRED_PACKAGES:
            package_name = package_spec.split(">=")[0].split("==")[0].strip()
            try:
                module = __import__(package_name)
                version = getattr(module, "__version__", "unknown")
                package_results[package_name] = {
                    "installed": True,
                    "version": version,
                    "required": package_spec.split(package_name)[1] if ">" in package_spec or "=" in package_spec else "any",
                }
            except ImportError:
                missing_packages.append(package_name)
                package_results[package_name] = {
                    "installed": False,
                    "version": None,
                    "required": package_spec.split(package_name)[1] if ">" in package_spec or "=" in package_spec else "any",
                }
        
        results["required_packages"] = package_results
        
        if missing_packages:
            logger.error(f"Missing required packages: {', '.join(missing_packages)}")
            logger.error(f"Please install them using: pip install {' '.join(missing_packages)}")
        
        # Check optional packages
        optional_results = {}
        
        for package_spec in OPTIONAL_PACKAGES:
            package_name = package_spec.split(">=")[0].split("==")[0].strip()
            try:
                module = __import__(package_name)
                version = getattr(module, "__version__", "unknown")
                optional_results[package_name] = {
                    "installed": True,
                    "version": version,
                    "required": package_spec.split(package_name)[1] if ">" in package_spec or "=" in package_spec else "any",
                }
            except ImportError:
                optional_results[package_name] = {
                    "installed": False,
                    "version": None,
                    "required": package_spec.split(package_name)[1] if ">" in package_spec or "=" in package_spec else "any",
                }
        
        results["optional_packages"] = optional_results
        
        # Check system resources
        system_results = {}
        system_valid = True
        
        if PSUTIL_AVAILABLE:
            # Check CPU
            cpu_count = psutil.cpu_count(logical=True)
            cpu_valid = cpu_count >= MIN_SYSTEM_REQUIREMENTS["cpu_cores"]
            
            system_results["cpu_cores"] = {
                "value": cpu_count,
                "required": MIN_SYSTEM_REQUIREMENTS["cpu_cores"],
                "valid": cpu_valid,
            }
            
            system_valid = system_valid and cpu_valid
            
            # Check RAM
            ram_gb = psutil.virtual_memory().total / (1024 ** 3)
            ram_valid = ram_gb >= MIN_SYSTEM_REQUIREMENTS["ram_gb"]
            
            system_results["ram_gb"] = {
                "value": f"{ram_gb:.1f}",
                "required": MIN_SYSTEM_REQUIREMENTS["ram_gb"],
                "valid": ram_valid,
            }
            
            system_valid = system_valid and ram_valid
            
            # Check disk space
            disk_gb = psutil.disk_usage("/").free / (1024 ** 3)
            disk_valid = disk_gb >= MIN_SYSTEM_REQUIREMENTS["disk_space_gb"]
            
            system_results["disk_space_gb"] = {
                "value": f"{disk_gb:.1f}",
                "required": MIN_SYSTEM_REQUIREMENTS["disk_space_gb"],
                "valid": disk_valid,
            }
            
            system_valid = system_valid and disk_valid
        else:
            system_results["note"] = "psutil not available, skipping system resource checks"
        
        results["system_resources"] = system_results
        
        # Check GPU availability
        gpu_results = {}
        
        if TORCH_AVAILABLE:
            # Check CUDA
            cuda_available = torch.cuda.is_available()
            gpu_results["cuda_available"] = cuda_available
            
            if cuda_available:
                gpu_results["cuda_version"] = torch.version.cuda
                gpu_results["cuda_device_count"] = torch.cuda.device_count()
                
                # Get device properties for each GPU
                gpu_results["devices"] = []
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    gpu_results["devices"].append({
                        "name": props.name,
                        "total_memory_gb": props.total_memory / (1024 ** 3),
                        "compute_capability": f"{props.major}.{props.minor}",
                    })
            
            # Check MPS (Apple Silicon)
            mps_available = hasattr(torch, 'mps') and torch.backends.mps.is_available()
            gpu_results["mps_available"] = mps_available
        else:
            gpu_results["note"] = "PyTorch not available, skipping GPU checks"
        
        results["gpu"] = gpu_results
        
        # Check data directory
        data_results = {}
        
        if self.data_dir.exists():
            data_results["exists"] = True
            
            # Check for required data files
            required_files = [
                self.config["data"]["train_file"],
                self.config["data"]["val_file"],
                self.config["data"]["test_file"],
            ]
            
            missing_files = []
            for file in required_files:
                if not (self.data_dir / file).exists():
                    missing_files.append(file)
            
            data_results["missing_files"] = missing_files
            data_results["valid"] = len(missing_files) == 0
        else:
            data_results["exists"] = False
            data_results["valid"] = False
        
        results["data_directory"] = data_results
        
        # Check output directory
        output_results = {}
        
        if self.output_dir.exists():
            output_results["exists"] = True
            
            # Check if writable
            try:
                test_file = self.output_dir / ".write_test"
                with open(test_file, "w") as f:
                    f.write("test")
                test_file.unlink()
                output_results["writable"] = True
            except Exception:
                output_results["writable"] = False
        else:
            output_results["exists"] = False
            output_results["writable"] = True  # Assume we can create it
        
        results["output_directory"] = output_results
        
        # Check git repository
        git_results = {}
        
        if GIT_AVAILABLE:
            try:
                repo = git.Repo(search_parent_directories=True)
                git_results["exists"] = True
                git_results["branch"] = repo.active_branch.name
                git_results["commit"] = repo.head.commit.hexsha
                git_results["dirty"] = repo.is_dirty()
            except Exception:
                git_results["exists"] = False
        else:
            git_results["note"] = "GitPython not available, skipping git checks"
        
        results["git"] = git_results
        
        # Check CI/CD integration
        ci_results = {}
        
        if self.config["ci"]["enabled"]:
            ci_provider = self.config["ci"]["provider"]
            ci_results["provider"] = ci_provider
            
            # Check for CI environment variables
            if ci_provider == "github":
                ci_env_vars = ["GITHUB_ACTIONS", "GITHUB_REPOSITORY", "GITHUB_WORKFLOW"]
                ci_results["env_vars"] = {var: var in os.environ for var in ci_env_vars}
                ci_results["running_in_ci"] = "GITHUB_ACTIONS" in os.environ
            elif ci_provider == "gitlab":
                ci_env_vars = ["GITLAB_CI", "CI_PROJECT_NAME", "CI_PIPELINE_ID"]
                ci_results["env_vars"] = {var: var in os.environ for var in ci_env_vars}
                ci_results["running_in_ci"] = "GITLAB_CI" in os.environ
            elif ci_provider == "jenkins":
                ci_env_vars = ["JENKINS_URL", "BUILD_ID", "JOB_NAME"]
                ci_results["env_vars"] = {var: var in os.environ for var in ci_env_vars}
                ci_results["running_in_ci"] = "JENKINS_URL" in os.environ
            else:
                ci_results["note"] = f"Unknown CI provider: {ci_provider}"
        
        results["ci"] = ci_results
        
        # Check W&B integration
        wandb_results = {}
        
        if WANDB_AVAILABLE and self.config["logging"]["use_wandb"]:
            wandb_results["available"] = True
            wandb_results["api_key_set"] = "WANDB_API_KEY" in os.environ
            
            if not wandb_results["api_key_set"]:
                logger.warning("W&B API key not set. Set the WANDB_API_KEY environment variable.")
        else:
            wandb_results["available"] = False
            if self.config["logging"]["use_wandb"]:
                logger.warning("W&B is enabled in config but not installed. Install with: pip install wandb")
        
        results["wandb"] = wandb_results
        
        # Store validation results
        self.validation_results = results
        
        # Determine if environment is valid
        required_valid = (
            python_valid and
            not missing_packages and
            system_valid and
            results["data_directory"]["valid"] and
            results["output_directory"]["writable"]
        )
        
        self.environment_valid = required_valid
        
        if required_valid:
            logger.info("Environment validation successful")
        else:
            logger.error("Environment validation failed")
        
        return required_valid
    
    def validate_config(self) -> bool:
        """
        Validate the configuration.
        
        Returns:
            bool: Whether the configuration is valid
        """
        logger.info("Validating configuration")
        
        # Track validation results
        results = {}
        valid = True
        
        # Check required top-level keys
        required_keys = ["experiment_name", "output_dir", "data_dir", "model", "training", "data"]
        missing_keys = [key for key in required_keys if key not in self.config]
        
        results["missing_required_keys"] = missing_keys
        valid = valid and not missing_keys
        
        if missing_keys:
            logger.error(f"Missing required configuration keys: {', '.join(missing_keys)}")
        
        # Check model configuration
        if "model" in self.config:
            model_config = self.config["model"]
            model_results = {}
            
            # Check required model keys
            model_required_keys = ["type"]
            model_missing_keys = [key for key in model_required_keys if key not in model_config]
            
            model_results["missing_required_keys"] = model_missing_keys
            valid = valid and not model_missing_keys
            
            if model_missing_keys:
                logger.error(f"Missing required model configuration keys: {', '.join(model_missing_keys)}")
            
            results["model"] = model_results
        
        # Check training configuration
        if "training" in self.config:
            training_config = self.config["training"]
            training_results = {}
            
            # Check required training keys
            training_required_keys = ["batch_size", "learning_rate", "max_epochs"]
            training_missing_keys = [key for key in training_required_keys if key not in training_config]
            
            training_results["missing_required_keys"] = training_missing_keys
            valid = valid and not training_missing_keys
            
            if training_missing_keys:
                logger.error(f"Missing required training configuration keys: {', '.join(training_missing_keys)}")
            
            # Check batch size
            if "batch_size" in training_config:
                batch_size = training_config["batch_size"]
                if not isinstance(batch_size, int) or batch_size <= 0:
                    training_results["invalid_batch_size"] = batch_size
                    valid = False
                    logger.error(f"Invalid batch size: {batch_size}. Must be a positive integer.")
            
            # Check learning rate
            if "learning_rate" in training_config:
                lr = training_config["learning_rate"]
                if not isinstance(lr, (int, float)) or lr <= 0:
                    training_results["invalid_learning_rate"] = lr
                    valid = False
                    logger.error(f"Invalid learning rate: {lr}. Must be a positive number.")
            
            results["training"] = training_results
        
        # Check data configuration
        if "data" in self.config:
            data_config = self.config["data"]
            data_results = {}
            
            # Check required data keys
            data_required_keys = ["train_file", "val_file", "test_file"]
            data_missing_keys = [key for key in data_required_keys if key not in data_config]
            
            data_results["missing_required_keys"] = data_missing_keys
            valid = valid and not data_missing_keys
            
            if data_missing_keys:
                logger.error(f"Missing required data configuration keys: {', '.join(data_missing_keys)}")
            
            # Check data files exist
            data_dir = Path(self.config["data_dir"])
            missing_files = []
            
            for key in ["train_file", "val_file", "test_file"]:
                if key in data_config:
                    file_path = data_dir / data_config[key]
                    if not file_path.exists():
                        missing_files.append(data_config[key])
            
            data_results["missing_files"] = missing_files
            if missing_files:
                logger.warning(f"Data files not found: {', '.join(missing_files)}")
                logger.warning(f"Expected in directory: {data_dir}")
            
            results["data"] = data_results
        
        # Check distributed configuration if enabled
        if self.config.get("distributed", {}).get("enabled", False):
            dist_config = self.config["distributed"]
            dist_results = {}
            
            # Check required distributed keys
            dist_required_keys = ["backend", "world_size", "master_addr", "master_port"]
            dist_missing_keys = [key for key in dist_required_keys if key not in dist_config]
            
            dist_results["missing_required_keys"] = dist_missing_keys
            valid = valid and not dist_missing_keys
            
            if dist_missing_keys:
                logger.error(f"Missing required distributed configuration keys: {', '.join(dist_missing_keys)}")
            
            # Check world size
            if "world_size" in dist_config:
                world_size = dist_config["world_size"]
                if not isinstance(world_size, int) or world_size <= 0:
                    dist_results["invalid_world_size"] = world_size
                    valid = False
                    logger.error(f"Invalid world size: {world_size}. Must be a positive integer.")
            
            results["distributed"] = dist_results
        
        # Store validation results
        self.config_valid = valid
        
        if valid:
            logger.info("Configuration validation successful")
        else:
            logger.error("Configuration validation failed")
        
        return valid
    
    def create_directories(self) -> bool:
        """
        Create necessary directories for training.
        
        Returns:
            bool: Whether directory creation was successful
        """
        logger.info("Creating directories")
        
        try:
            # Create output directory
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created output directory: {self.output_dir}")
            
            # Create subdirectories
            dirs_to_create = [
                "checkpoints",
                "logs",
                "tensorboard",
                "wandb",
                "artifacts",
                "health",
                "recovery",
                "failure_analysis",
                "reports",
                "visualizations",
            ]
            
            for dir_name in dirs_to_create:
                dir_path = self.output_dir / dir_name
                dir_path.mkdir(exist_ok=True)
                logger.debug(f"Created directory: {dir_path}")
            
            # Create data directory if it doesn't exist
            if not self.data_dir.exists():
                self.data_dir.mkdir(parents=True, exist_ok=True)
                logger.warning(f"Created data directory: {self.data_dir}")
                logger.warning("Data directory was created but may be empty. Please add required data files.")
            
            return True
        
        except Exception as e:
            logger.error(f"Error creating directories: {e}")
            return False
    
    def initialize_logging(self) -> bool:
        """
        Initialize logging for training.
        
        Returns:
            bool: Whether logging initialization was successful
        """
        logger.info("Initializing logging")
        
        try:
            # Create log directory
            log_dir = self.output_dir / "logs"
            log_dir.mkdir(exist_ok=True)
            
            # Set up file handler for root logger
            log_file = log_dir / f"{self.experiment_name}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ))
            
            # Add handler to root logger
            root_logger = logging.getLogger()
            root_logger.addHandler(file_handler)
            
            logger.info(f"Logging to file: {log_file}")
            
            # Initialize W&B if enabled
            if WANDB_AVAILABLE and self.config["logging"]["use_wandb"]:
                if "WANDB_API_KEY" in os.environ:
                    # Initialize W&B
                    wandb.init(
                        project=self.experiment_name,
                        name=f"{self.experiment_name}_{int(time.time())}",
                        config=self.config,
                        dir=str(self.output_dir / "wandb"),
                    )
                    logger.info("Initialized W&B logging")
                else:
                    logger.warning("W&B API key not set. Set the WANDB_API_KEY environment variable.")
            
            # Initialize TensorBoard if enabled
            if self.config["logging"]["use_tensorboard"]:
                try:
                    from torch.utils.tensorboard import SummaryWriter
                    tensorboard_dir = self.output_dir / "tensorboard"
                    writer = SummaryWriter(log_dir=tensorboard_dir)
                    writer.add_text("config", str(self.config))
                    logger.info(f"Initialized TensorBoard logging to {tensorboard_dir}")
                except ImportError:
                    logger.warning("TensorBoard not available. Install with: pip install tensorboard")
            
            self.components_initialized.add("logging")
            return True
        
        except Exception as e:
            logger.error(f"Error initializing logging: {e}")
            self.components_failed.add("logging")
            return False
    
    def initialize_checkpoint_manager(self) -> bool:
        """
        Initialize the checkpoint manager.
        
        Returns:
            bool: Whether initialization was successful
        """
        logger.info("Initializing checkpoint manager")
        
        try:
            # Import checkpoint manager
            sys.path.insert(0, ".")
            from scripts.training.checkpoint_manager import CheckpointManager
            
            # Create checkpoint directory
            checkpoint_dir = self.output_dir / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            
            # Initialize checkpoint manager
            self.checkpoint_manager = CheckpointManager(
                experiment_name=self.experiment_name,
                checkpoint_dir=str(checkpoint_dir),
                config=self.config["checkpointing"],
            )
            
            logger.info("Checkpoint manager initialized")
            self.components_initialized.add("checkpoint_manager")
            return True
        
        except Exception as e:
            logger.error(f"Error initializing checkpoint manager: {e}")
            logger.debug(traceback.format_exc())
            self.components_failed.add("checkpoint_manager")
            return False
    
    def initialize_health_monitor(self) -> bool:
        """
        Initialize the health monitor.
        
        Returns:
            bool: Whether initialization was successful
        """
        logger.info("Initializing health monitor")
        
        try:
            # Import health monitor
            sys.path.insert(0, ".")
            from scripts.training.health_monitor import HealthMonitor
            
            # Initialize health monitor
            self.health_monitor = HealthMonitor(
                experiment_name=self.experiment_name,
                output_dir=str(self.output_dir),
                config=self.config.get("monitoring", {}),
            )
            
            logger.info("Health monitor initialized")
            self.components_initialized.add("health_monitor")
            return True
        
        except Exception as e:
            logger.error(f"Error initializing health monitor: {e}")
            logger.debug(traceback.format_exc())
            self.components_failed.add("health_monitor")
            return False
    
    def initialize_recovery_manager(self) -> bool:
        """
        Initialize the recovery manager.
        
        Returns:
            bool: Whether initialization was successful
        """
        logger.info("Initializing recovery manager")
        
        try:
            # Import recovery manager
            sys.path.insert(0, ".")
            from scripts.training.recovery_manager import RecoveryManager
            
            # Initialize recovery manager
            self.recovery_manager = RecoveryManager(
                experiment_name=self.experiment_name,
                output_dir=str(self.output_dir),
                config=self.config.get("recovery", {}),
            )
            
            logger.info("Recovery manager initialized")
            self.components_initialized.add("recovery_manager")
            return True
        
        except Exception as e:
            logger.error(f"Error initializing recovery manager: {e}")
            logger.debug(traceback.format_exc())
            self.components_failed.add("recovery_manager")
            return False
    
    def initialize_failure_analyzer(self) -> bool:
        """
        Initialize the failure analyzer.
        
        Returns:
            bool: Whether initialization was successful
        """
        logger.info("Initializing failure analyzer")
        
        try:
            # Import failure analyzer
            sys.path.insert(0, ".")
            from scripts.training.failure_analyzer import FailureAnalyzer
            
            # Initialize failure analyzer
            self.failure_analyzer = FailureAnalyzer(
                experiment_name=self.experiment_name,
                output_dir=str(self.output_dir),
            )
            
            logger.info("Failure analyzer initialized")
            self.components_initialized.add("failure_analyzer")
            return True
        
        except Exception as e:
            logger.error(f"Error initializing failure analyzer: {e}")
            logger.debug(traceback.format_exc())
            self.components_failed.add("failure_analyzer")
            return False
    
    def initialize_resource_monitor(self) -> bool:
        """
        Initialize the resource monitor.
        
        Returns:
            bool: Whether initialization was successful
        """
        logger.info("Initializing resource monitor")
        
        try:
            # Import resource monitor
            sys.path.insert(0, ".")
            from scripts.training.resource_monitor import ResourceMonitor
            
            # Initialize resource monitor
            self.resource_monitor = ResourceMonitor(
                experiment_name=self.experiment_name,
                output_dir=str(self.output_dir),
            )
            
            logger.info("Resource monitor initialized")
            self.components_initialized.add("resource_monitor")
            return True
        
        except Exception as e:
            logger.error(f"Error initializing resource monitor: {e}")
            logger.debug(traceback.format_exc())
            self.components_failed.add("resource_monitor")
            return False
    
    def initialize_mps_optimizer(self) -> bool:
        """
        Initialize the MPS optimizer for Apple Silicon.
        
        Returns:
            bool: Whether initialization was successful
        """
        logger.info("Initializing MPS optimizer")
        
        # Check if MPS is available
        if not (TORCH_AVAILABLE and hasattr(torch, 'mps') and torch.backends.mps.is_available()):
            logger.info("MPS not available, skipping MPS optimizer initialization")
            return False
        
        try:
            # Import MPS optimizer
            sys.path.insert(0, ".")
            from scripts.training.mps_optimizer import MPSOptimizer
            
            # Initialize MPS optimizer
            self.mps_optimizer = MPSOptimizer(
                experiment_name=self.experiment_name,
                output_dir=str(self.output_dir),
            )
            
            logger.info("MPS optimizer initialized")
            self.components_initialized.add("mps_optimizer")
            return True
        
        except Exception as e:
            logger.error(f"Error initializing MPS optimizer: {e}")
            logger.debug(traceback.format_exc())
            self.components_failed.add("mps_optimizer")
            return False
    
    def initialize_training_orchestrator(self) -> bool:
        """
        Initialize the training orchestrator.
        
        Returns:
            bool: Whether initialization was successful
        """
        logger.info("Initializing training orchestrator")
        
        try:
            # Import training orchestrator
            sys.path.insert(0, ".")
            from scripts.training.training_orchestrator import TrainingOrchestrator
            
            # Initialize training orchestrator
            self.orchestrator = TrainingOrchestrator(
                config_path=self.config_path,
                experiment_name=self.experiment_name,
                output_dir=str(self.output_dir),
            )
            
            # Register components with orchestrator
            if self.health_monitor:
                self.orchestrator.register_health_monitor(self.health_monitor)
            
            if self.recovery_manager:
                self.orchestrator.register_recovery_manager(self.recovery_manager)
                self.recovery_manager.register_orchestrator(self.orchestrator)
            
            if self.failure_analyzer:
                self.orchestrator.register_failure_analyzer(self.failure_analyzer)
            
            if self.checkpoint_manager:
                self.orchestrator.register_checkpoint_manager(self.checkpoint_manager)
            
            if self.resource_monitor:
                self.orchestrator.register_resource_monitor(self.resource_monitor)
            
            if self.mps_optimizer:
                self.orchestrator.register_mps_optimizer(self.mps_optimizer)
            
            logger.info("Training orchestrator initialized")
            self.components_initialized.add("training_orchestrator")
            return True
        
        except Exception as e:
            logger.error(f"Error initializing training orchestrator: {e}")
            logger.debug(traceback.format_exc())
            self.components_failed.add("training_orchestrator")
            return False
    
    def test_gpu_performance(self) -> Dict[str, Any]:
        """
        Test GPU performance.
        
        Returns:
            Dict[str, Any]: Performance test results
        """
        logger.info("Testing GPU performance")
        
        results = {
            "cuda": {"available": False},
            "mps": {"available": False},
        }
        
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, skipping GPU performance test")
            return results
        
        # Test CUDA performance
        if torch.cuda.is_available():
            results["cuda"]["available"] = True
            results["cuda"]["device_count"] = torch.cuda.device_count()
            
            try:
                # Create a test tensor and measure transfer speed
                start_time = time.time()
                x = torch.randn(1000, 1000, device="cuda")
                torch.cuda.synchronize()
                transfer_time = time.time() - start_time
                
                # Measure matmul performance
                start_time = time.time()
                for _ in range(10):
                    y = torch.matmul(x, x)
                    torch.cuda.synchronize()
                matmul_time = (time.time() - start_time) / 10
                
                # Measure memory bandwidth
                start_time = time.time()
                for _ in range(10):
                    y = x + x
                    torch.cuda.synchronize()
                memory_time = (time.time() - start_time) / 10
                
                results["cuda"]["transfer_time_ms"] = transfer_time * 1000
                results["cuda"]["matmul_time_ms"] = matmul_time * 1000
                results["cuda"]["memory_time_ms"] = memory_time * 1000
                
                logger.info(f"CUDA performance: {results['cuda']}")
            
            except Exception as e:
                logger.error(f"Error testing CUDA performance: {e}")
                results["cuda"]["error"] = str(e)
        
        # Test MPS performance (Apple Silicon)
        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
            results["mps"]["available"] = True
            
            try:
                # Create a test tensor and measure transfer speed
                start_time = time.time()
                x = torch.randn(1000, 1000, device="mps")
                transfer_time = time.time() - start_time
                
                # Measure matmul performance
                start_time = time.time()
                for _ in range(10):
                    y = torch.matmul(x, x)
                matmul_time = (time.time() - start_time) / 10
                
                # Measure memory bandwidth
                start_time = time.time()
                for _ in range(10):
                    y = x + x
                memory_time = (time.time() - start_time) / 10
                
                results["mps"]["transfer_time_ms"] = transfer_time * 1000
                results["mps"]["matmul_time_ms"] = matmul_time * 1000
                results["mps"]["memory_time_ms"] = memory_time * 1000
                
                logger.info(f"MPS performance: {results['mps']}")
            
            except Exception as e:
                logger.error(f"Error testing MPS performance: {e}")
                results["mps"]["error"] = str(e)
        
        return results
    
    def test_ci_integration(self) -> Dict[str, Any]:
        """
        Test CI/CD integration.
        
        Returns:
            Dict[str, Any]: Integration test results
        """
        logger.info("Testing CI/CD integration")
        
        results = {
            "enabled": self.config.get("ci", {}).get("enabled", False),
            "provider": self.config.get("ci", {}).get("provider", "unknown"),
            "running_in_ci": False,
            "webhook_test": None,
        }
        
        if not results["enabled"]:
            logger.info("CI integration not enabled, skipping test")
            return results
        
        # Check if running in CI
        ci_provider = results["provider"]
        
        if ci_provider == "github":
            results["running_in_ci"] = "GITHUB_ACTIONS" in os.environ
            results["env_vars"] = {
                "GITHUB_ACTIONS": "GITHUB_ACTIONS" in os.environ,
                "GITHUB_REPOSITORY": "GITHUB_REPOSITORY" in os.environ,
                "GITHUB_WORKFLOW": "GITHUB_WORKFLOW" in os.environ,
            }
        elif ci_provider == "gitlab":
            results["running_in_ci"] = "GITLAB_CI" in os.environ
            results["env_vars"] = {
                "GITLAB_CI": "GITLAB_CI" in os.environ,
                "CI_PROJECT_NAME": "CI_PROJECT_NAME" in os.environ,
                "CI_PIPELINE_ID": "CI_PIPELINE_ID" in os.environ,
            }
        elif ci_provider == "jenkins":
            results["running_in_ci"] = "JENKINS_URL" in os.environ
            results["env_vars"] = {
                "JENKINS_URL": "JENKINS_URL" in os.environ,
                "BUILD_ID": "BUILD_ID" in os.environ,
                "JOB_NAME": "JOB_NAME" in os.environ,
            }
        
        # Test webhook if available
        webhook_url = os.environ.get("CI_WEBHOOK_URL")
        if webhook_url and REQUESTS_AVAILABLE:
            try:
                response = requests.post(
                    webhook_url,
                    json={
                        "event": "test",
                        "experiment": self.experiment_name,
                        "timestamp": time.time(),
                        "message": "CI integration test",
                    },
                    timeout=5,
                )
                
                results["webhook_test"] = {
                    "success": response.status_code == 200,
                    "status_code": response.status_code,
                }
                
                logger.info(f"Webhook test result: {results['webhook_test']}")
            
            except Exception as e:
                logger.error(f"Error testing webhook: {e}")
                results["webhook_test"] = {"success": False, "error": str(e)}
        
        return results
    
    def run_health_check(self) -> Dict[str, Any]:
        """
        Run a comprehensive health check of the training infrastructure.
        
        Returns:
            Dict[str, Any]: Health check results
        """
        logger.info("Running health check")
        
        results = {
            "timestamp": time.time(),
            "environment_valid": self.environment_valid,
            "config_valid": self.config_valid,
            "components_initialized": list(self.components_initialized),
            "components_failed": list(self.components_failed),
            "directories_created": False,
            "gpu_test": None,
            "ci_test": None,
        }
        
        # Check directories
        output_dir_exists = self.output_dir.exists()
        data_dir_exists = self.data_dir.exists()
        
        results["directories"] = {
            "output_dir_exists": output_dir_exists,
            "data_dir_exists": data_dir_exists,
        }
        
        if output_dir_exists:
            checkpoint_dir = self.output_dir / "checkpoints"
            results["directories"]["checkpoint_dir_exists"] = checkpoint_dir.exists()
        
        results["directories_created"] = (
            output_dir_exists and
            data_dir_exists and
            results["directories"].get("checkpoint_dir_exists", False)
        )
        
        # Check GPU
        if TORCH_AVAILABLE:
            results["gpu"] = {
                "cuda_available": torch.cuda.is_available(),
                "mps_available": hasattr(torch, 'mps') and torch.backends.mps.is_available(),
            }
            
            if torch.cuda.is_available():
                results["gpu"]["cuda_device_count"] = torch.cuda.device_count()
                results["gpu"]["cuda_version"] = torch.version.cuda
        
        # Check components
        results["components"] = {
            "orchestrator": self.orchestrator is not None,
            "health_monitor": self.health_monitor is not None,
            "recovery_manager": self.recovery_manager is not None,
            "failure_analyzer": self.failure_analyzer is not None,
            "checkpoint_manager": self.checkpoint_manager is not None,
            "resource_monitor": self.resource_monitor is not None,
            "mps_optimizer": self.mps_optimizer is not None,
        }
        
        # Run GPU performance test if requested
        if "gpu_test" in self.validation_results:
            results["gpu_test"] = self.validation_results["gpu_test"]
        
        # Run CI integration test if requested
        if "ci_test" in self.validation_results:
            results["ci_test"] = self.validation_results["ci_test"]
        
        # Overall health status
        results["healthy"] = (
            self.environment_valid and
            self.config_valid and
            results["directories_created"] and
            len(self.components_failed) == 0
        )
        
        return results
    
    def save_config(self) -> bool:
        """
        Save the configuration to a file.
        
        Returns:
            bool: Whether the save was successful
        """
        logger.info("Saving configuration")
        
        try:
            # Create output directory if it doesn't exist
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save config to YAML file
            config_file = self.output_dir / "config.yaml"
            with open(config_file, "w") as f:
                yaml.dump(self.config, f, default_flow_style=False)
            
            logger.info(f"Configuration saved to {config_file}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def save_health_check(self, health_check: Dict[str, Any]) -> bool:
        """
        Save health check results to a file.
        
        Args:
            health_check: Health check results
            
        Returns:
            bool: Whether the save was successful
        """
        logger.info("Saving health check results")
        
        try:
            # Create health directory if it doesn't exist
            health_dir = self.output_dir / "health"
            health_dir.mkdir(parents=True, exist_ok=True)
            
            # Save health check to JSON file
            health_file = health_dir / "setup_health_check.json"
            with open(health_file, "w") as f:
                json.dump(health_check, f, indent=2)
            
            logger.info(f"Health check results saved to {health_file}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving health check results: {e}")
            return False
    
    def setup(self) -> Dict[str, Any]:
        """
        Set up the training infrastructure.
        
        Returns:
            Dict[str, Any]: Setup results
        """
        logger.info(f"Setting up training infrastructure for experiment: {self.experiment_name}")
        start_time = time.time()
        
        results = {
            "experiment_name": self.experiment_name,
            "output_dir": str(self.output_dir),
            "data_dir": str(self.data_dir),
            "start_time": start_time,
            "validate_only": self.validate_only,
            "environment_valid": False,
            "config_valid": False,
            "components_initialized": [],
            "components_failed": [],
            "success": False,
        }
        
        # Validate environment
        env_valid = self.validate_environment()
        results["environment_valid"] = env_valid
        
        # Validate configuration
        config_valid = self.validate_config()
        results["config_valid"] = config_valid
        
        # Create directories
        if (env_valid and config_valid) or self.force:
            dirs_created = self.create_directories()
            results["directories_created"] = dirs_created
            
            # Save configuration
            config_saved = self.save_config()
            results["config_saved"] = config_saved
            
            # If validate only, stop here
            if self.validate_only:
                logger.info("Validation complete, skipping initialization")
                results["success"] = env_valid and config_valid
                results["duration"] = time.time() - start_time
                return results
            
            # Initialize components
            if self.initialize_all:
                # Initialize all components
                self.initialize_logging()
                self.initialize_checkpoint_manager()
                self.initialize_health_monitor()
                self.initialize_recovery_manager()
                self.initialize_failure_analyzer()
                self.initialize_resource_monitor()
                self.initialize_mps_optimizer()
                self.initialize_training_orchestrator()
            else:
                # Initialize only required components
                self.initialize_logging()
                self.initialize_checkpoint_manager()
                
                # Initialize optional components based on config
                if self.config.get("monitoring", {}).get("enabled", True):
                    self.initialize_health_monitor()
                
                if self.config.get("recovery", {}).get("enabled", True):
                    self.initialize_recovery_manager()
                
                # Initialize MPS optimizer if on Apple Silicon
                if (
                    TORCH_AVAILABLE and
                    hasattr(torch, 'mps') and
                    torch.backends.mps.is_available()
                ):
                    self.initialize_mps_optimizer()
                
                # Always initialize orchestrator last
                self.initialize_training_orchestrator()
            
            # Run GPU performance test if requested
            if "check_gpu" in results and results["check_gpu"]:
                gpu_test = self.test_gpu_performance()
                results["gpu_test"] = gpu_test
                self.validation_results["gpu_test"] = gpu_test
            
            # Run CI integration test if requested
            if "check_ci" in results and results["check_ci"]:
                ci_test = self.test_ci_integration()
                results["ci_test"] = ci_test
                self.validation_results["ci_test"] = ci_test
            
            # Run health check
            health_check = self.run_health_check()
            results["health_check"] = health_check
            
            # Save health check
            self.save_health_check(health_check)
            
            # Update results
            results["components_initialized"] = list(self.components_initialized)
            results["components_failed"] = list(self.components_failed)
            results["success"] = (
                env_valid and
                config_valid and
                dirs_created and
                len(self.components_failed) == 0
            )
        
        results["duration"] = time.time() - start_time
        
        if results["success"]:
            logger.info(
                f"Training infrastructure setup completed successfully in {results['duration']:.2f} seconds"
            )
        else:
            logger.error(
                f"Training infrastructure setup failed after {results['duration']:.2f} seconds"
            )
        
        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Training Infrastructure Setup")
    
    # Configuration options
    parser.add_argument("--config", dest="config_path", help="Path to configuration file")
    parser.add_argument("--experiment-name", help="Name of the experiment")
    parser.add_argument("--output-dir", help="Directory to save outputs")
    parser.add_argument("--data-dir", help="Directory containing data")
    
    # Validation and initialization options
    parser.add_argument("--validate-only", action="store_true", help="Only validate, don't initialize")
    parser.add_argument("--initialize-all", action="store_true", help="Initialize all components")
    parser.add_argument("--force", action="store_true", help="Force initialization even if validation fails")
    
    # Testing options
    parser.add_argument("--check-gpu", action="store_true", help="Run GPU performance test")
    parser.add_argument("--check-ci", action="store_true", help="Test CI/CD integration")
    
    # Output options
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Create setup instance
    setup = InfrastructureSetup(
        config_path=args.config_path,
        experiment_name=args.experiment_name,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        validate_only=args.validate_only,
        initialize_all=args.initialize_all,
        force=args.force,
        verbose=args.verbose,
    )
    
    # Run setup
    results = setup.setup()
    
    # Add test flags to results
    results["check_gpu"] = args.check_gpu
    results["check_ci"] = args.check_ci
    
    # Output results
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print("\nTraining Infrastructure Setup Results:")
        print(f"Experiment: {results['experiment_name']}")
        print(f"Output Directory: {results['output_dir']}")
        print(f"Environment Valid: {results['environment_valid']}")
        print(f"Configuration Valid: {results['config_valid']}")
        print(f"Components Initialized: {', '.join(results['components_initialized']) or 'None'}")
        print(f"Components Failed: {', '.join(results['components_failed']) or 'None'}")
        print(f"Success: {results['success']}")
        print(f"Duration: {results['duration']:.2f} seconds")
    
    # Return exit code
    return 0 if results["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
