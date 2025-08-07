#!/usr/bin/env python
"""
Recovery Manager for Training Jobs

This module provides a comprehensive recovery management system for training jobs,
implementing automatic restart mechanisms, checkpoint validation, intelligent
scheduling, and failure pattern analysis to ensure robust training even in
unstable environments.

Features:
- Automatic restart mechanisms for various failure scenarios
- Checkpoint recovery with validation and corruption detection
- Intelligent job scheduling and resource management
- Exponential backoff and retry strategies for transient failures
- Distributed training failure handling and node coordination
- Rollback mechanisms for corrupted states
- Automatic parameter tuning for recovery scenarios
- Failure pattern analysis and prevention
- Integration with CI/CD pipeline recovery hooks
- Cost-aware recovery strategies for cloud environments

Usage:
    from scripts.training.recovery_manager import RecoveryManager
    
    # Initialize recovery manager
    recovery_manager = RecoveryManager(
        experiment_name="hrm_training",
        output_dir="outputs/hrm_training",
        config_path="configs/training_config.yaml",
        enable_auto_recovery=True,
    )
    
    # Register with training orchestrator
    recovery_manager.register_orchestrator(training_orchestrator)
    
    # Start recovery management
    recovery_manager.start()
    
    # Handle a specific failure
    recovery_manager.handle_failure(
        failure_type="cuda_out_of_memory",
        step=1000,
        traceback=traceback.format_exc(),
    )
    
    # Stop recovery management
    recovery_manager.stop()
"""

import atexit
import concurrent.futures
import copy
import datetime
import glob
import json
import logging
import math
import os
import platform
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import warnings
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import yaml

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
    import boto3
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    from google.cloud import storage as gcp_storage
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

try:
    from azure.storage.blob import BlobServiceClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import sklearn
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of training failures."""
    CUDA_OUT_OF_MEMORY = "cuda_out_of_memory"
    CPU_OUT_OF_MEMORY = "cpu_out_of_memory"
    DISK_OUT_OF_SPACE = "disk_out_of_space"
    CHECKPOINT_CORRUPTION = "checkpoint_corruption"
    GRADIENT_EXPLOSION = "gradient_explosion"
    GRADIENT_VANISHING = "gradient_vanishing"
    NAN_LOSS = "nan_loss"
    INF_LOSS = "inf_loss"
    DEADLOCK = "deadlock"
    TIMEOUT = "timeout"
    NODE_FAILURE = "node_failure"
    NETWORK_FAILURE = "network_failure"
    DATA_CORRUPTION = "data_corruption"
    PERMISSION_ERROR = "permission_error"
    ENVIRONMENT_ERROR = "environment_error"
    UNKNOWN = "unknown"
    
    @classmethod
    def from_exception(cls, exception: Exception, traceback_str: str = None) -> 'FailureType':
        """
        Determine failure type from exception and traceback.
        
        Args:
            exception: The exception that occurred
            traceback_str: String representation of the traceback
            
        Returns:
            FailureType: The determined failure type
        """
        if traceback_str is None:
            traceback_str = str(exception)
        
        # Check for CUDA OOM
        if "CUDA out of memory" in traceback_str or "cuda runtime error" in traceback_str.lower():
            return cls.CUDA_OUT_OF_MEMORY
        
        # Check for CPU OOM
        if "MemoryError" in traceback_str or "memory" in str(exception).lower() and "allocation" in str(exception).lower():
            return cls.CPU_OUT_OF_MEMORY
        
        # Check for disk space issues
        if "No space left on device" in traceback_str or "disk space" in str(exception).lower():
            return cls.DISK_OUT_OF_SPACE
        
        # Check for NaN/Inf loss
        if "nan" in str(exception).lower() or "inf" in str(exception).lower():
            if "loss" in str(exception).lower():
                if "nan" in str(exception).lower():
                    return cls.NAN_LOSS
                else:
                    return cls.INF_LOSS
        
        # Check for gradient issues
        if "gradient" in str(exception).lower():
            if "explod" in str(exception).lower():
                return cls.GRADIENT_EXPLOSION
            elif "vanish" in str(exception).lower():
                return cls.GRADIENT_VANISHING
        
        # Check for checkpoint corruption
        if "checkpoint" in str(exception).lower() and ("corrupt" in str(exception).lower() or "invalid" in str(exception).lower()):
            return cls.CHECKPOINT_CORRUPTION
        
        # Check for network issues
        if "network" in str(exception).lower() or "connection" in str(exception).lower() or "timeout" in str(exception).lower():
            if "timeout" in str(exception).lower():
                return cls.TIMEOUT
            else:
                return cls.NETWORK_FAILURE
        
        # Check for permission issues
        if "permission" in str(exception).lower() or "access" in str(exception).lower():
            return cls.PERMISSION_ERROR
        
        # Check for environment issues
        if "environment" in str(exception).lower() or "not found" in str(exception).lower():
            return cls.ENVIRONMENT_ERROR
        
        # Default to unknown
        return cls.UNKNOWN


class RecoveryStrategy(Enum):
    """Recovery strategies for training failures."""
    RESTART = "restart"
    REDUCE_BATCH_SIZE = "reduce_batch_size"
    REDUCE_LEARNING_RATE = "reduce_learning_rate"
    INCREASE_GRADIENT_ACCUMULATION = "increase_gradient_accumulation"
    ENABLE_GRADIENT_CHECKPOINTING = "enable_gradient_checkpointing"
    ENABLE_MIXED_PRECISION = "enable_mixed_precision"
    ROLLBACK_CHECKPOINT = "rollback_checkpoint"
    SKIP_BAD_BATCH = "skip_bad_batch"
    CLEAN_MEMORY = "clean_memory"
    CLEAN_DISK = "clean_disk"
    RELOCATE_NODE = "relocate_node"
    RECONFIGURE_ENVIRONMENT = "reconfigure_environment"
    TERMINATE = "terminate"
    CUSTOM = "custom"


class RecoveryPhase(Enum):
    """Phases of the recovery process."""
    DETECTION = "detection"
    ANALYSIS = "analysis"
    PLANNING = "planning"
    EXECUTION = "execution"
    VERIFICATION = "verification"
    COMPLETED = "completed"
    FAILED = "failed"


class RecoveryPriority(Enum):
    """Priority levels for recovery actions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CloudProvider(Enum):
    """Supported cloud providers."""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    CUSTOM = "custom"
    NONE = "none"


@dataclass
class FailureContext:
    """Context information for a training failure."""
    failure_type: FailureType
    timestamp: float = field(default_factory=time.time)
    exception: Optional[Exception] = None
    traceback_str: Optional[str] = None
    step: Optional[int] = None
    epoch: Optional[int] = None
    batch_idx: Optional[int] = None
    node_rank: Optional[int] = None
    device: Optional[str] = None
    memory_stats: Optional[Dict[str, Any]] = None
    disk_stats: Optional[Dict[str, Any]] = None
    network_stats: Optional[Dict[str, Any]] = None
    process_stats: Optional[Dict[str, Any]] = None
    training_metrics: Optional[Dict[str, Any]] = None
    model_stats: Optional[Dict[str, Any]] = None
    checkpoint_path: Optional[str] = None
    config_snapshot: Optional[Dict[str, Any]] = None
    custom_data: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class RecoveryAction:
    """Action to be taken for recovery."""
    strategy: RecoveryStrategy
    priority: RecoveryPriority = RecoveryPriority.MEDIUM
    params: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None
    estimated_cost: Optional[float] = None
    estimated_time: Optional[float] = None
    success_probability: Optional[float] = None
    dependencies: List[RecoveryStrategy] = field(default_factory=list)
    custom_handler: Optional[Callable] = None
    
    def __post_init__(self):
        """Set description if not provided."""
        if self.description is None:
            self.description = f"Apply {self.strategy.value} recovery strategy"


@dataclass
class RecoveryPlan:
    """Plan for recovering from a training failure."""
    failure_context: FailureContext
    actions: List[RecoveryAction] = field(default_factory=list)
    phase: RecoveryPhase = RecoveryPhase.PLANNING
    created_at: float = field(default_factory=time.time)
    executed_at: Optional[float] = None
    completed_at: Optional[float] = None
    success: Optional[bool] = None
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None
    result_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BackoffConfig:
    """Configuration for exponential backoff."""
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 3600.0  # 1 hour
    multiplier: float = 2.0
    jitter_factor: float = 0.1
    max_retries: int = 10


@dataclass
class CheckpointValidationConfig:
    """Configuration for checkpoint validation."""
    validate_model_load: bool = True
    validate_optimizer_state: bool = True
    validate_training_state: bool = True
    validate_metrics: bool = True
    checksum_validation: bool = True
    test_forward_pass: bool = False
    test_backward_pass: bool = False
    corruption_detection: bool = True
    auto_repair: bool = True
    max_repair_attempts: int = 3


@dataclass
class FailureAnalysisConfig:
    """Configuration for failure analysis."""
    pattern_recognition_enabled: bool = True
    root_cause_analysis_enabled: bool = True
    correlation_analysis_enabled: bool = True
    anomaly_detection_enabled: bool = True
    history_window_size: int = 100
    min_pattern_occurrences: int = 3
    anomaly_detection_method: str = "isolation_forest"
    anomaly_detection_contamination: float = 0.05


@dataclass
class DistributedRecoveryConfig:
    """Configuration for distributed training recovery."""
    node_coordination_enabled: bool = True
    node_health_check_interval_seconds: float = 30.0
    node_timeout_seconds: float = 300.0
    master_node_election_enabled: bool = True
    synchronization_timeout_seconds: float = 600.0
    partial_restart_enabled: bool = True
    all_reduce_timeout_seconds: float = 60.0


@dataclass
class CloudRecoveryConfig:
    """Configuration for cloud-based recovery."""
    provider: CloudProvider = CloudProvider.NONE
    instance_relocation_enabled: bool = False
    spot_instance_handling_enabled: bool = False
    cost_aware_recovery_enabled: bool = False
    auto_scaling_enabled: bool = False
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    credentials_path: Optional[str] = None
    backup_region: Optional[str] = None
    max_cost_increase_percent: float = 20.0


@dataclass
class CIIntegrationConfig:
    """Configuration for CI/CD integration."""
    enabled: bool = False
    provider: str = "github"
    workflow_file: Optional[str] = None
    notification_enabled: bool = True
    auto_retry_enabled: bool = True
    artifact_collection_enabled: bool = True
    status_update_enabled: bool = True
    webhook_url: Optional[str] = None
    api_token_env_var: str = "CI_API_TOKEN"


@dataclass
class AutoTuningConfig:
    """Configuration for automatic parameter tuning."""
    enabled: bool = True
    tunable_parameters: List[str] = field(default_factory=list)
    max_tuning_attempts: int = 5
    tuning_strategy: str = "bayesian"
    objective_metric: str = "loss"
    exploration_factor: float = 0.2
    parameter_constraints: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class RecoveryManagerConfig:
    """Configuration for the recovery manager."""
    # General settings
    enabled: bool = True
    auto_recovery_enabled: bool = True
    recovery_dir: str = "recovery"
    check_interval_seconds: float = 5.0
    recovery_timeout_seconds: float = 3600.0
    max_recovery_attempts: int = 10
    recovery_cooldown_seconds: float = 60.0
    
    # Component configurations
    backoff: BackoffConfig = field(default_factory=BackoffConfig)
    checkpoint_validation: CheckpointValidationConfig = field(default_factory=CheckpointValidationConfig)
    failure_analysis: FailureAnalysisConfig = field(default_factory=FailureAnalysisConfig)
    distributed_recovery: DistributedRecoveryConfig = field(default_factory=DistributedRecoveryConfig)
    cloud_recovery: CloudRecoveryConfig = field(default_factory=CloudRecoveryConfig)
    ci_integration: CIIntegrationConfig = field(default_factory=CIIntegrationConfig)
    auto_tuning: AutoTuningConfig = field(default_factory=AutoTuningConfig)
    
    # Strategy mappings
    failure_strategy_map: Dict[str, List[str]] = field(default_factory=dict)
    
    # Callbacks
    pre_recovery_callback: Optional[Callable] = None
    post_recovery_callback: Optional[Callable] = None
    custom_strategy_handlers: Dict[str, Callable] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default strategy mappings if not provided."""
        if not self.failure_strategy_map:
            self.failure_strategy_map = {
                FailureType.CUDA_OUT_OF_MEMORY.value: [
                    RecoveryStrategy.CLEAN_MEMORY.value,
                    RecoveryStrategy.REDUCE_BATCH_SIZE.value,
                    RecoveryStrategy.ENABLE_GRADIENT_CHECKPOINTING.value,
                    RecoveryStrategy.ENABLE_MIXED_PRECISION.value,
                ],
                FailureType.CPU_OUT_OF_MEMORY.value: [
                    RecoveryStrategy.CLEAN_MEMORY.value,
                    RecoveryStrategy.REDUCE_BATCH_SIZE.value,
                ],
                FailureType.DISK_OUT_OF_SPACE.value: [
                    RecoveryStrategy.CLEAN_DISK.value,
                ],
                FailureType.CHECKPOINT_CORRUPTION.value: [
                    RecoveryStrategy.ROLLBACK_CHECKPOINT.value,
                ],
                FailureType.GRADIENT_EXPLOSION.value: [
                    RecoveryStrategy.REDUCE_LEARNING_RATE.value,
                    RecoveryStrategy.ENABLE_GRADIENT_CHECKPOINTING.value,
                ],
                FailureType.GRADIENT_VANISHING.value: [
                    RecoveryStrategy.INCREASE_GRADIENT_ACCUMULATION.value,
                ],
                FailureType.NAN_LOSS.value: [
                    RecoveryStrategy.SKIP_BAD_BATCH.value,
                    RecoveryStrategy.ROLLBACK_CHECKPOINT.value,
                    RecoveryStrategy.REDUCE_LEARNING_RATE.value,
                ],
                FailureType.INF_LOSS.value: [
                    RecoveryStrategy.SKIP_BAD_BATCH.value,
                    RecoveryStrategy.ROLLBACK_CHECKPOINT.value,
                    RecoveryStrategy.REDUCE_LEARNING_RATE.value,
                ],
                FailureType.DEADLOCK.value: [
                    RecoveryStrategy.RESTART.value,
                ],
                FailureType.TIMEOUT.value: [
                    RecoveryStrategy.RESTART.value,
                ],
                FailureType.NODE_FAILURE.value: [
                    RecoveryStrategy.RELOCATE_NODE.value,
                    RecoveryStrategy.RESTART.value,
                ],
                FailureType.NETWORK_FAILURE.value: [
                    RecoveryStrategy.RESTART.value,
                ],
                FailureType.DATA_CORRUPTION.value: [
                    RecoveryStrategy.RESTART.value,
                ],
                FailureType.PERMISSION_ERROR.value: [
                    RecoveryStrategy.RECONFIGURE_ENVIRONMENT.value,
                ],
                FailureType.ENVIRONMENT_ERROR.value: [
                    RecoveryStrategy.RECONFIGURE_ENVIRONMENT.value,
                ],
                FailureType.UNKNOWN.value: [
                    RecoveryStrategy.RESTART.value,
                ],
            }
        
        # Initialize tunable parameters if not provided
        if not self.auto_tuning.tunable_parameters:
            self.auto_tuning.tunable_parameters = [
                "learning_rate",
                "batch_size",
                "gradient_accumulation_steps",
                "gradient_checkpointing",
                "mixed_precision",
            ]


class RecoveryManager:
    """
    Recovery Manager for Training Jobs
    
    This class provides comprehensive recovery management for training jobs,
    implementing automatic restart mechanisms, checkpoint validation, intelligent
    scheduling, and failure pattern analysis to ensure robust training even in
    unstable environments.
    """
    
    def __init__(
        self,
        experiment_name: str,
        output_dir: str,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        enable_auto_recovery: Optional[bool] = None,
    ):
        """
        Initialize the recovery manager.
        
        Args:
            experiment_name: Name of the experiment
            output_dir: Directory to save outputs
            config_path: Path to configuration file (optional)
            config: Configuration dictionary (optional, overrides config_path)
            enable_auto_recovery: Whether to enable automatic recovery (optional, overrides config)
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.config_path = config_path
        
        # Load configuration
        self.config = self._load_config(config_path, config)
        
        # Override config with constructor arguments
        if enable_auto_recovery is not None:
            self.config.auto_recovery_enabled = enable_auto_recovery
        
        # Initialize state
        self.running = False
        self.failure_history = []
        self.recovery_history = []
        self.active_recovery_plans = []
        self.completed_recovery_plans = []
        self.pattern_database = defaultdict(list)
        self.anomaly_detector = None
        self.parameter_tuner = None
        self.cloud_client = None
        self.ci_client = None
        
        # Training orchestrator reference
        self.orchestrator = None
        
        # Locks for thread safety
        self.recovery_lock = threading.Lock()
        self.history_lock = threading.Lock()
        self.plan_lock = threading.Lock()
        
        # Monitoring threads
        self.monitoring_thread = None
        self.recovery_thread = None
        
        # Counters and stats
        self.start_time = None
        self.last_check_time = 0
        self.recovery_attempts = 0
        self.last_recovery_time = 0
        self.successful_recoveries = 0
        self.failed_recoveries = 0
        
        # Backoff state
        self.current_backoff_delay = self.config.backoff.initial_delay_seconds
        
        # Initialize directories
        self._init_directories()
        
        # Register signal handlers and exit handler
        self._register_handlers()
        
        # Initialize components
        self._init_components()
        
        logger.info(f"Recovery manager initialized for experiment: {experiment_name}")
    
    def _load_config(
        self,
        config_path: Optional[str],
        config_dict: Optional[Dict[str, Any]],
    ) -> RecoveryManagerConfig:
        """
        Load configuration from file or dictionary.
        
        Args:
            config_path: Path to configuration file
            config_dict: Configuration dictionary
            
        Returns:
            RecoveryManagerConfig: Configuration object
        """
        # Start with default config
        config = RecoveryManagerConfig()
        
        # Load from file if provided
        if config_path:
            try:
                with open(config_path, "r") as f:
                    file_config = yaml.safe_load(f)
                
                # Extract recovery manager config
                if "recovery_manager" in file_config:
                    recovery_config = file_config["recovery_manager"]
                else:
                    recovery_config = file_config
                
                # Update config with file values
                self._update_config_from_dict(config, recovery_config)
                
                logger.info(f"Loaded recovery manager configuration from {config_path}")
            
            except Exception as e:
                logger.error(f"Error loading configuration from {config_path}: {e}")
        
        # Override with provided config dict
        if config_dict:
            self._update_config_from_dict(config, config_dict)
            logger.info("Applied custom configuration")
        
        return config
    
    def _update_config_from_dict(self, config: RecoveryManagerConfig, config_dict: Dict[str, Any]):
        """
        Update configuration from dictionary.
        
        Args:
            config: Configuration object to update
            config_dict: Dictionary with configuration values
        """
        # Update top-level settings
        for key, value in config_dict.items():
            if key == "backoff" and isinstance(value, dict):
                for bk_key, bk_value in value.items():
                    if hasattr(config.backoff, bk_key):
                        setattr(config.backoff, bk_key, bk_value)
            
            elif key == "checkpoint_validation" and isinstance(value, dict):
                for cv_key, cv_value in value.items():
                    if hasattr(config.checkpoint_validation, cv_key):
                        setattr(config.checkpoint_validation, cv_key, cv_value)
            
            elif key == "failure_analysis" and isinstance(value, dict):
                for fa_key, fa_value in value.items():
                    if hasattr(config.failure_analysis, fa_key):
                        setattr(config.failure_analysis, fa_key, fa_value)
            
            elif key == "distributed_recovery" and isinstance(value, dict):
                for dr_key, dr_value in value.items():
                    if hasattr(config.distributed_recovery, dr_key):
                        setattr(config.distributed_recovery, dr_key, dr_value)
            
            elif key == "cloud_recovery" and isinstance(value, dict):
                for cr_key, cr_value in value.items():
                    if cr_key == "provider" and isinstance(cr_value, str):
                        try:
                            config.cloud_recovery.provider = CloudProvider(cr_value)
                        except ValueError:
                            logger.warning(f"Invalid cloud provider: {cr_value}")
                    elif hasattr(config.cloud_recovery, cr_key):
                        setattr(config.cloud_recovery, cr_key, cr_value)
            
            elif key == "ci_integration" and isinstance(value, dict):
                for ci_key, ci_value in value.items():
                    if hasattr(config.ci_integration, ci_key):
                        setattr(config.ci_integration, ci_key, ci_value)
            
            elif key == "auto_tuning" and isinstance(value, dict):
                for at_key, at_value in value.items():
                    if hasattr(config.auto_tuning, at_key):
                        setattr(config.auto_tuning, at_key, at_value)
            
            elif key == "failure_strategy_map" and isinstance(value, dict):
                config.failure_strategy_map.update(value)
            
            elif hasattr(config, key):
                setattr(config, key, value)
    
    def _init_directories(self):
        """Initialize directories for recovery manager outputs."""
        try:
            # Create main directories
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create recovery manager subdirectory
            self.recovery_dir = self.output_dir / self.config.recovery_dir
            self.recovery_dir.mkdir(exist_ok=True)
            
            # Create subdirectories
            (self.recovery_dir / "logs").mkdir(exist_ok=True)
            (self.recovery_dir / "plans").mkdir(exist_ok=True)
            (self.recovery_dir / "checkpoints").mkdir(exist_ok=True)
            (self.recovery_dir / "analysis").mkdir(exist_ok=True)
            (self.recovery_dir / "tuning").mkdir(exist_ok=True)
            
            logger.debug(f"Initialized recovery manager directories in {self.recovery_dir}")
        
        except Exception as e:
            logger.error(f"Error initializing directories: {e}")
    
    def _register_handlers(self):
        """Register signal handlers and exit handler."""
        # Register exit handler
        atexit.register(self._cleanup)
        
        # Register signal handlers
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}, stopping recovery manager...")
            self.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _cleanup(self):
        """Clean up resources before exiting."""
        if self.running:
            self.stop()
    
    def _init_components(self):
        """Initialize recovery manager components."""
        # Initialize anomaly detector if enabled
        if self.config.failure_analysis.anomaly_detection_enabled and SKLEARN_AVAILABLE:
            try:
                self.anomaly_detector = IsolationForest(
                    contamination=self.config.failure_analysis.anomaly_detection_contamination,
                    random_state=42,
                )
                logger.debug("Initialized anomaly detector")
            except Exception as e:
                logger.error(f"Error initializing anomaly detector: {e}")
        
        # Initialize cloud client if enabled
        if self.config.cloud_recovery.provider != CloudProvider.NONE:
            self._init_cloud_client()
        
        # Initialize CI client if enabled
        if self.config.ci_integration.enabled:
            self._init_ci_client()
        
        # Initialize parameter tuner if enabled
        if self.config.auto_tuning.enabled:
            self._init_parameter_tuner()
    
    def _init_cloud_client(self):
        """Initialize cloud client based on provider."""
        provider = self.config.cloud_recovery.provider
        
        try:
            if provider == CloudProvider.AWS and AWS_AVAILABLE:
                # Initialize AWS client
                session = boto3.Session()
                self.cloud_client = {
                    "ec2": session.client("ec2"),
                    "s3": session.client("s3"),
                    "cloudwatch": session.client("cloudwatch"),
                }
                logger.info("Initialized AWS cloud client")
            
            elif provider == CloudProvider.GCP and GCP_AVAILABLE:
                # Initialize GCP client
                self.cloud_client = {
                    "storage": gcp_storage.Client(),
                }
                logger.info("Initialized GCP cloud client")
            
            elif provider == CloudProvider.AZURE and AZURE_AVAILABLE:
                # Initialize Azure client
                connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
                if connection_string:
                    self.cloud_client = {
                        "blob": BlobServiceClient.from_connection_string(connection_string),
                    }
                    logger.info("Initialized Azure cloud client")
                else:
                    logger.warning("Azure connection string not found in environment variables")
            
            elif provider == CloudProvider.CUSTOM:
                # Custom provider requires manual initialization
                logger.info("Custom cloud provider requires manual client initialization")
            
            else:
                logger.warning(f"Unsupported cloud provider: {provider}")
        
        except Exception as e:
            logger.error(f"Error initializing cloud client: {e}")
            self.cloud_client = None
    
    def _init_ci_client(self):
        """Initialize CI integration client."""
        provider = self.config.ci_integration.provider.lower()
        
        try:
            if provider == "github":
                # Check for GitHub token
                github_token = os.environ.get(self.config.ci_integration.api_token_env_var)
                if github_token:
                    # Simple dictionary-based client for GitHub
                    self.ci_client = {
                        "provider": "github",
                        "token": github_token,
                        "webhook_url": self.config.ci_integration.webhook_url,
                    }
                    logger.info("Initialized GitHub CI client")
                else:
                    logger.warning(f"GitHub token not found in environment variable {self.config.ci_integration.api_token_env_var}")
            
            elif provider == "gitlab":
                # Check for GitLab token
                gitlab_token = os.environ.get(self.config.ci_integration.api_token_env_var)
                if gitlab_token:
                    # Simple dictionary-based client for GitLab
                    self.ci_client = {
                        "provider": "gitlab",
                        "token": gitlab_token,
                        "webhook_url": self.config.ci_integration.webhook_url,
                    }
                    logger.info("Initialized GitLab CI client")
                else:
                    logger.warning(f"GitLab token not found in environment variable {self.config.ci_integration.api_token_env_var}")
            
            elif provider == "jenkins":
                # Check for Jenkins token
                jenkins_token = os.environ.get(self.config.ci_integration.api_token_env_var)
                if jenkins_token:
                    # Simple dictionary-based client for Jenkins
                    self.ci_client = {
                        "provider": "jenkins",
                        "token": jenkins_token,
                        "webhook_url": self.config.ci_integration.webhook_url,
                    }
                    logger.info("Initialized Jenkins CI client")
                else:
                    logger.warning(f"Jenkins token not found in environment variable {self.config.ci_integration.api_token_env_var}")
            
            else:
                logger.warning(f"Unsupported CI provider: {provider}")
        
        except Exception as e:
            logger.error(f"Error initializing CI client: {e}")
            self.ci_client = None
    
    def _init_parameter_tuner(self):
        """Initialize parameter tuner."""
        try:
            # Simple dictionary-based parameter tuner
            self.parameter_tuner = {
                "strategy": self.config.auto_tuning.tuning_strategy,
                "tunable_parameters": self.config.auto_tuning.tunable_parameters,
                "max_attempts": self.config.auto_tuning.max_tuning_attempts,
                "objective_metric": self.config.auto_tuning.objective_metric,
                "exploration_factor": self.config.auto_tuning.exploration_factor,
                "constraints": self.config.auto_tuning.parameter_constraints,
                "history": [],
            }
            logger.debug("Initialized parameter tuner")
        
        except Exception as e:
            logger.error(f"Error initializing parameter tuner: {e}")
            self.parameter_tuner = None
    
    def register_orchestrator(self, orchestrator):
        """
        Register a training orchestrator with the recovery manager.
        
        Args:
            orchestrator: Training orchestrator instance
        """
        self.orchestrator = orchestrator
        logger.info("Registered training orchestrator")
    
    def start(self):
        """Start recovery management."""
        if self.running:
            logger.warning("Recovery manager is already running")
            return
        
        if not self.config.enabled:
            logger.info("Recovery manager is disabled, not starting")
            return
        
        logger.info("Starting recovery manager")
        self.running = True
        self.start_time = time.time()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="recovery_monitor_thread",
        )
        self.monitoring_thread.start()
        
        # Start recovery thread
        self.recovery_thread = threading.Thread(
            target=self._recovery_loop,
            daemon=True,
            name="recovery_execution_thread",
        )
        self.recovery_thread.start()
        
        logger.info("Recovery manager started")
    
    def stop(self):
        """Stop recovery management."""
        if not self.running:
            return
        
        logger.info("Stopping recovery manager")
        self.running = False
        
        # Wait for threads to finish
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        if self.recovery_thread and self.recovery_thread.is_alive():
            self.recovery_thread.join(timeout=5.0)
        
        # Save final report
        self._save_recovery_report()
        
        logger.info("Recovery manager stopped")
    
    def handle_failure(
        self,
        failure_type: Union[str, FailureType, Exception],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        batch_idx: Optional[int] = None,
        traceback_str: Optional[str] = None,
        exception: Optional[Exception] = None,
        checkpoint_path: Optional[str] = None,
        custom_data: Optional[Dict[str, Any]] = None,
    ) -> RecoveryPlan:
        """
        Handle a training failure.
        
        Args:
            failure_type: Type of failure or exception
            step: Current training step
            epoch: Current training epoch
            batch_idx: Current batch index
            traceback_str: String representation of the traceback
            exception: The exception that occurred
            checkpoint_path: Path to the latest checkpoint
            custom_data: Additional custom data for the failure context
            
        Returns:
            RecoveryPlan: The created recovery plan
        """
        # Convert failure type to enum if needed
        if isinstance(failure_type, Exception):
            exception = failure_type
            failure_type = FailureType.from_exception(exception, traceback_str)
        elif isinstance(failure_type, str):
            try:
                failure_type = FailureType(failure_type)
            except ValueError:
                failure_type = FailureType.UNKNOWN
        
        # Create failure context
        context = FailureContext(
            failure_type=failure_type,
            step=step,
            epoch=epoch,
            batch_idx=batch_idx,
            traceback_str=traceback_str,
            exception=exception,
            checkpoint_path=checkpoint_path,
            custom_data=custom_data or {},
        )
        
        # Add system stats if available
        if PSUTIL_AVAILABLE:
            try:
                # Memory stats
                memory = psutil.virtual_memory()
                context.memory_stats = {
                    "total": memory.total / (1024 ** 3),  # GB
                    "available": memory.available / (1024 ** 3),  # GB
                    "used": memory.used / (1024 ** 3),  # GB
                    "percent": memory.percent,
                }
                
                # Disk stats
                disk = psutil.disk_usage(self.output_dir)
                context.disk_stats = {
                    "total": disk.total / (1024 ** 3),  # GB
                    "free": disk.free / (1024 ** 3),  # GB
                    "used": disk.used / (1024 ** 3),  # GB
                    "percent": disk.percent,
                }
                
                # Process stats
                process = psutil.Process()
                context.process_stats = {
                    "cpu_percent": process.cpu_percent(),
                    "memory_info": {
                        "rss": process.memory_info().rss / (1024 ** 3),  # GB
                        "vms": process.memory_info().vms / (1024 ** 3),  # GB
                    },
                    "num_threads": process.num_threads(),
                    "num_fds": process.num_fds() if hasattr(process, "num_fds") else None,
                }
            except Exception as e:
                logger.warning(f"Error collecting system stats: {e}")
        
        # Add GPU stats if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                context.device = f"cuda:{torch.cuda.current_device()}"
                context.custom_data["gpu_info"] = {
                    "name": torch.cuda.get_device_name(torch.cuda.current_device()),
                    "memory_allocated": torch.cuda.memory_allocated() / (1024 ** 3),  # GB
                    "memory_reserved": torch.cuda.memory_reserved() / (1024 ** 3),  # GB
                    "max_memory_allocated": torch.cuda.max_memory_allocated() / (1024 ** 3),  # GB
                }
            except Exception as e:
                logger.warning(f"Error collecting GPU stats: {e}")
        
        # Add to failure history
        with self.history_lock:
            self.failure_history.append(context)
        
        # Log failure
        logger.error(
            f"Training failure detected: {failure_type.value} at step={step}, epoch={epoch}, "
            f"batch_idx={batch_idx}"
        )
        if traceback_str:
            logger.debug(f"Traceback: {traceback_str}")
        
        # Create recovery plan
        plan = self._create_recovery_plan(context)
        
        # Add to active plans
        with self.plan_lock:
            self.active_recovery_plans.append(plan)
        
        # Execute plan if auto-recovery is enabled
        if self.config.auto_recovery_enabled:
            # Execute in recovery thread
            pass
        
        return plan
    
    def execute_plan(self, plan: RecoveryPlan) -> bool:
        """
        Execute a recovery plan.
        
        Args:
            plan: Recovery plan to execute
            
        Returns:
            bool: Whether execution was successful
        """
        if not self.running:
            logger.warning("Recovery manager is not running, cannot execute plan")
            return False
        
        # Check if plan is already completed
        if plan.phase == RecoveryPhase.COMPLETED:
            logger.info("Recovery plan is already completed")
            return True
        
        if plan.phase == RecoveryPhase.FAILED:
            logger.warning("Recovery plan has already failed")
            return False
        
        # Update plan phase
        plan.phase = RecoveryPhase.EXECUTION
        plan.executed_at = time.time()
        
        # Log execution
        logger.info(
            f"Executing recovery plan for {plan.failure_context.failure_type.value} "
            f"with {len(plan.actions)} actions"
        )
        
        # Call pre-recovery callback if defined
        if self.config.pre_recovery_callback:
            try:
                self.config.pre_recovery_callback(plan)
            except Exception as e:
                logger.error(f"Error in pre-recovery callback: {e}")
        
        # Execute actions in order
        success = True
        for i, action in enumerate(plan.actions):
            try:
                logger.info(
                    f"Executing recovery action {i+1}/{len(plan.actions)}: "
                    f"{action.strategy.value} ({action.priority.value})"
                )
                
                # Execute action
                action_success = self._execute_action(action, plan)
                
                if not action_success:
                    logger.warning(f"Recovery action {action.strategy.value} failed")
                    success = False
                    break
                
                logger.info(f"Recovery action {action.strategy.value} completed successfully")
            
            except Exception as e:
                logger.error(f"Error executing recovery action {action.strategy.value}: {e}")
                logger.debug(traceback.format_exc())
                success = False
                plan.error_message = str(e)
                break
        
        # Update plan status
        if success:
            plan.phase = RecoveryPhase.VERIFICATION
            
            # Verify recovery
            verification_success = self._verify_recovery(plan)
            
            if verification_success:
                plan.phase = RecoveryPhase.COMPLETED
                plan.completed_at = time.time()
                plan.success = True
                
                # Update recovery stats
                self.successful_recoveries += 1
                
                # Reset backoff delay on success
                self.current_backoff_delay = self.config.backoff.initial_delay_seconds
                
                logger.info(
                    f"Recovery plan for {plan.failure_context.failure_type.value} "
                    f"completed successfully in {plan.completed_at - plan.executed_at:.2f} seconds"
                )
            else:
                plan.phase = RecoveryPhase.FAILED
                plan.success = False
                plan.error_message = "Recovery verification failed"
                
                # Update recovery stats
                self.failed_recoveries += 1
                
                # Apply backoff for retry
                self._apply_backoff()
                
                logger.warning(
                    f"Recovery plan for {plan.failure_context.failure_type.value} "
                    f"failed verification"
                )
        else:
            plan.phase = RecoveryPhase.FAILED
            plan.success = False
            
            # Update recovery stats
            self.failed_recoveries += 1
            
            # Apply backoff for retry
            self._apply_backoff()
            
            logger.warning(
                f"Recovery plan for {plan.failure_context.failure_type.value} "
                f"failed during execution"
            )
        
        # Call post-recovery callback if defined
        if self.config.post_recovery_callback:
            try:
                self.config.post_recovery_callback(plan)
            except Exception as e:
                logger.error(f"Error in post-recovery callback: {e}")
        
        # Move plan to completed plans
        with self.plan_lock:
            if plan in self.active_recovery_plans:
                self.active_recovery_plans.remove(plan)
            self.completed_recovery_plans.append(plan)
        
        # Save plan to disk
        self._save_recovery_plan(plan)
        
        # Update CI status if enabled
        if self.config.ci_integration.enabled and self.config.ci_integration.status_update_enabled:
            self._update_ci_status(plan)
        
        return success
    
    def _execute_action(self, action: RecoveryAction, plan: RecoveryPlan) -> bool:
        """
        Execute a recovery action.
        
        Args:
            action: Recovery action to execute
            plan: Recovery plan context
            
        Returns:
            bool: Whether execution was successful
        """
        # Check for custom handler
        if action.custom_handler:
            try:
                return action.custom_handler(action, plan, self)
            except Exception as e:
                logger.error(f"Error in custom handler for {action.strategy.value}: {e}")
                return False
        
        # Check for strategy-specific handler in config
        if action.strategy.value in self.config.custom_strategy_handlers:
            try:
                return self.config.custom_strategy_handlers[action.strategy.value](action, plan, self)
            except Exception as e:
                logger.error(f"Error in custom strategy handler for {action.strategy.value}: {e}")
                return False
        
        # Execute built-in strategies
        if action.strategy == RecoveryStrategy.RESTART:
            return self._execute_restart(action, plan)
        
        elif action.strategy == RecoveryStrategy.REDUCE_BATCH_SIZE:
            return self._execute_reduce_batch_size(action, plan)
        
        elif action.strategy == RecoveryStrategy.REDUCE_LEARNING_RATE:
            return self._execute_reduce_learning_rate(action, plan)
        
        elif action.strategy == RecoveryStrategy.INCREASE_GRADIENT_ACCUMULATION:
            return self._execute_increase_gradient_accumulation(action, plan)
        
        elif action.strategy == RecoveryStrategy.ENABLE_GRADIENT_CHECKPOINTING:
            return self._execute_enable_gradient_checkpointing(action, plan)
        
        elif action.strategy == RecoveryStrategy.ENABLE_MIXED_PRECISION:
            return self._execute_enable_mixed_precision(action, plan)
        
        elif action.strategy == RecoveryStrategy.ROLLBACK_CHECKPOINT:
            return self._execute_rollback_checkpoint(action, plan)
        
        elif action.strategy == RecoveryStrategy.SKIP_BAD_BATCH:
            return self._execute_skip_bad_batch(action, plan)
        
        elif action.strategy == RecoveryStrategy.CLEAN_MEMORY:
            return self._execute_clean_memory(action, plan)
        
        elif action.strategy == RecoveryStrategy.CLEAN_DISK:
            return self._execute_clean_disk(action, plan)
        
        elif action.strategy == RecoveryStrategy.RELOCATE_NODE:
            return self._execute_relocate_node(action, plan)
        
        elif action.strategy == RecoveryStrategy.RECONFIGURE_ENVIRONMENT:
            return self._execute_reconfigure_environment(action, plan)
        
        elif action.strategy == RecoveryStrategy.TERMINATE:
            return self._execute_terminate(action, plan)
        
        else:
            logger.warning(f"Unsupported recovery strategy: {action.strategy.value}")
            return False
    
    def _execute_restart(self, action: RecoveryAction, plan: RecoveryPlan) -> bool:
        """
        Execute restart recovery strategy.
        
        Args:
            action: Recovery action
            plan: Recovery plan
            
        Returns:
            bool: Whether execution was successful
        """
        logger.info("Executing restart recovery strategy")
        
        # Check if orchestrator is available
        if self.orchestrator is None:
            logger.warning("No orchestrator registered, cannot restart training")
            
            # Create a signal file for external monitoring to detect
            signal_file = self.recovery_dir / "RESTART_TRAINING"
            with open(signal_file, "w") as f:
                f.write(f"timestamp: {time.time()}\n")
                f.write(f"failure_type: {plan.failure_context.failure_type.value}\n")
                f.write(f"step: {plan.failure_context.step}\n")
                f.write(f"epoch: {plan.failure_context.epoch}\n")
            
            logger.info(f"Created restart signal file: {signal_file}")
            return True
        
        # Get restart parameters
        clean_start = action.params.get("clean_start", False)
        preserve_checkpoint = action.params.get("preserve_checkpoint", True)
        wait_seconds = action.params.get("wait_seconds", 5)
        
        try:
            # Save current state if needed
            if preserve_checkpoint and not clean_start:
                if hasattr(self.orchestrator, "save_checkpoint"):
                    checkpoint_path = self.orchestrator.save_checkpoint(
                        is_recovery=True,
                        tag="pre_restart",
                    )
                    logger.info(f"Saved pre-restart checkpoint to {checkpoint_path}")
            
            # Wait if specified
            if wait_seconds > 0:
                logger.info(f"Waiting {wait_seconds} seconds before restart...")
                time.sleep(wait_seconds)
            
            # Restart training
            if hasattr(self.orchestrator, "restart"):
                self.orchestrator.restart(clean=clean_start)
                logger.info("Restarted training via orchestrator")
                return True
            
            # Alternative: create signal file
            signal_file = self.recovery_dir / "RESTART_TRAINING"
            with open(signal_file, "w") as f:
                f.write(f"timestamp: {time.time()}\n")
                f.write(f"failure_type: {plan.failure_context.failure_type.value}\n")
                f.write(f"step: {plan.failure_context.step}\n")
                f.write(f"epoch: {plan.failure_context.epoch}\n")
                f.write(f"clean_start: {clean_start}\n")
            
            logger.info(f"Created restart signal file: {signal_file}")
            return True
        
        except Exception as e:
            logger.error(f"Error executing restart: {e}")
            return False
    
    def _execute_reduce_batch_size(self, action: RecoveryAction, plan: RecoveryPlan) -> bool:
        """
        Execute reduce batch size recovery strategy.
        
        Args:
            action: Recovery action
            plan: Recovery plan
            
        Returns:
            bool: Whether execution was successful
        """
        logger.info("Executing reduce batch size recovery strategy")
        
        # Get parameters
        reduction_factor = action.params.get("reduction_factor", 0.5)
        min_batch_size = action.params.get("min_batch_size", 1)
        
        try:
            # Check if orchestrator is available
            if self.orchestrator is None:
                logger.warning("No orchestrator registered, creating signal file")
                
                # Create a signal file for external monitoring to detect
                signal_file = self.recovery_dir / "REDUCE_BATCH_SIZE"
                with open(signal_file, "w") as f:
                    f.write(f"timestamp: {time.time()}\n")
                    f.write(f"failure_type: {plan.failure_context.failure_type.value}\n")
                    f.write(f"reduction_factor: {reduction_factor}\n")
                    f.write(f"min_batch_size: {min_batch_size}\n")
                
                logger.info(f"Created batch size reduction signal file: {signal_file}")
                return True
            
            # Get current batch size from orchestrator
            current_batch_size = None
            if hasattr(self.orchestrator, "config") and "data" in self.orchestrator.config:
                current_batch_size = self.orchestrator.config["data"].get("batch_size")
            
            if current_batch_size is None:
                logger.warning("Could not determine current batch size")
                current_batch_size = 32  # Assume default
            
            # Calculate new batch size
            new_batch_size = max(int(current_batch_size * reduction_factor), min_batch_size)
            
            logger.info(f"Reducing batch size from {current_batch_size} to {new_batch_size}")
            
            # Update batch size in orchestrator
            if hasattr(self.orchestrator, "update_config"):
                self.orchestrator.update_config({"data": {"batch_size": new_batch_size}})
                logger.info(f"Updated batch size to {new_batch_size} via orchestrator")
                return True
            
            # Alternative: create signal file
            signal_file = self.recovery_dir / "REDUCE_BATCH_SIZE"
            with open(signal_file, "w") as f:
                f.write(f"timestamp: {time.time()}\n")
                f.write(f"failure_type: {plan.failure_context.failure_type.value}\n")
                f.write(f"current_batch_size: {current_batch_size}\n")
                f.write(f"new_batch_size: {new_batch_size}\n")
            
            logger.info(f"Created batch size reduction signal file: {signal_file}")
            return True
        
        except Exception as e:
            logger.error(f"Error executing batch size reduction: {e}")
            return False
    
    def _execute_reduce_learning_rate(self, action: RecoveryAction, plan: RecoveryPlan) -> bool:
        """
        Execute reduce learning rate recovery strategy.
        
        Args:
            action: Recovery action
            plan: Recovery plan
            
        Returns:
            bool: Whether execution was successful
        """
        logger.info("Executing reduce learning rate recovery strategy")
        
        # Get parameters
        reduction_factor = action.params.get("reduction_factor", 0.1)
        min_lr = action.params.get("min_lr", 1e-6)
        
        try:
            # Check if orchestrator is available
            if self.orchestrator is None:
                logger.warning("No orchestrator registered, creating signal file")
                
                # Create a signal file for external monitoring to detect
                signal_file = self.recovery_dir / "REDUCE_LEARNING_RATE"
                with open(signal_file, "w") as f:
                    f.write(f"timestamp: {time.time()}\n")
                    f.write(f"failure_type: {plan.failure_context.failure_type.value}\n")
                    f.write(f"reduction_factor: {reduction_factor}\n")
                    f.write(f"min_lr: {min_lr}\n")
                
                logger.info(f"Created learning rate reduction signal file: {signal_file}")
                return True
            
            # Get current learning rate from orchestrator
            current_lr = None
            if hasattr(self.orchestrator, "config") and "optimizer" in self.orchestrator.config:
                current_lr = self.orchestrator.config["optimizer"].get("learning_rate")
            
            if current_lr is None:
                logger.warning("Could not determine current learning rate")
                current_lr = 1e-3  # Assume default
            
            # Calculate new learning rate
            new_lr = max(current_lr * reduction_factor, min_lr)
            
            logger.info(f"Reducing learning rate from {current_lr} to {new_lr}")
            
            # Update learning rate in orchestrator
            if hasattr(self.orchestrator, "update_config"):
                self.orchestrator.update_config({"optimizer": {"learning_rate": new_lr}})
                logger.info(f"Updated learning rate to {new_lr} via orchestrator")
                return True
            
            # Alternative: create signal file
            signal_file = self.recovery_dir / "REDUCE_LEARNING_RATE"
            with open(signal_file, "w") as f:
                f.write(f"timestamp: {time.time()}\n")
                f.write(f"failure_type: {plan.failure_context.failure_type.value}\n")
                f.write(f"current_lr: {current_lr}\n")
                f.write(f"new_lr: {new_lr}\n")
            
            logger.info(f"Created learning rate reduction signal file: {signal_file}")
            return True
        
        except Exception as e:
            logger.error(f"Error executing learning rate reduction: {e}")
            return False
    
    def _execute_increase_gradient_accumulation(self, action: RecoveryAction, plan: RecoveryPlan) -> bool:
        """
        Execute increase gradient accumulation recovery strategy.
        
        Args:
            action: Recovery action
            plan: Recovery plan
            
        Returns:
            bool: Whether execution was successful
        """
        logger.info("Executing increase gradient accumulation recovery strategy")
        
        # Get parameters
        increase_factor = action.params.get("increase_factor", 2)
        max_steps = action.params.get("max_steps", 32)
        
        try:
            # Check if orchestrator is available
            if self.orchestrator is None:
                logger.warning("No orchestrator registered, creating signal file")
                
                # Create a signal file for external monitoring to detect
                signal_file = self.recovery_dir / "INCREASE_GRADIENT_ACCUMULATION"
                with open(signal_file, "w") as f:
                    f.write(f"timestamp: {time.time()}\n")
                    f.write(f"failure_type: {plan.failure_context.failure_type.value}\n")
                    f.write(f"increase_factor: {increase_factor}\n")
                    f.write(f"max_steps: {max_steps}\n")
                
                logger.info(f"Created gradient accumulation signal file: {signal_file}")
                return True
            
            # Get current gradient accumulation steps from orchestrator
            current_steps = None
            if hasattr(self.orchestrator, "config") and "training" in self.orchestrator.config:
                current_steps = self.orchestrator.config["training"].get("gradient_accumulation_steps", 1)
            
            if current_steps is None:
                logger.warning("Could not determine current gradient accumulation steps")
                current_steps = 1  # Assume default
            
            # Calculate new steps
            new_steps = min(current_steps * increase_factor, max_steps)
            
            logger.info(f"Increasing gradient accumulation steps from {current_steps} to {new_steps}")
            
            # Update steps in orchestrator
            if hasattr(self.orchestrator, "update_config"):
                self.orchestrator.update_config({"training": {"gradient_accumulation_steps": new_steps}})
                logger.info(f"Updated gradient accumulation steps to {new_steps} via orchestrator")
                return True
            
            # Alternative: create signal file
            signal_file = self.recovery_dir / "INCREASE_GRADIENT_ACCUMULATION"
            with open(signal_file, "w") as f:
                f.write(f"timestamp: {time.time()}\n")
                f.write(f"failure_type: {plan.failure_context.failure_type.value}\n")
                f.write(f"current_steps: {current_steps}\n")
                f.write(f"new_steps: {new_steps}\n")
            
            logger.info(f"Created gradient accumulation signal file: {signal_file}")
            return True
        
        except Exception as e:
            logger.error(f"Error executing gradient accumulation increase: {e}")
            return False
    
    def _execute_enable_gradient_checkpointing(self, action: RecoveryAction, plan: RecoveryPlan) -> bool:
        """
        Execute enable gradient checkpointing recovery strategy.
        
        Args:
            action: Recovery action
            plan: Recovery plan
            
        Returns:
            bool: Whether execution was successful
        """
        logger.info("Executing enable gradient checkpointing recovery strategy")
        
        try:
            # Check if orchestrator is available
            if self.orchestrator is None:
                logger.warning("No orchestrator registered, creating signal file")
                
                # Create a signal file for external monitoring to detect
                signal_file = self.recovery_dir / "ENABLE_GRADIENT_CHECKPOINTING"
                with open(signal_file, "w") as f:
                    f.write(f"timestamp: {time.time()}\n")
                    f.write(f"failure_type: {plan.failure_context.failure_type.value}\n")
                
                logger.info(f"Created gradient checkpointing signal file: {signal_file}")
                return True
            
            # Update config in orchestrator
            if hasattr(self.orchestrator, "update_config"):
                self.orchestrator.update_config({"model": {"gradient_checkpointing": True}})
                logger.info("Enabled gradient checkpointing via orchestrator")
                return True
            
            # Alternative: create signal file
            signal_file = self.recovery_dir / "ENABLE_GRADIENT_CHECKPOINTING"
            with open(signal_file, "w") as f:
                f.write(f"timestamp: {time.time()}\n")
                f.write(f"failure_type: {plan.failure_context.failure_type.value}\n")
            
            logger.info(f"Created gradient checkpointing signal file: {signal_file}")
            return True
        
        except Exception as e:
            logger.error(f"Error enabling gradient checkpointing: {e}")
            return False
    
    def _execute_enable_mixed_precision(self, action: RecoveryAction, plan: RecoveryPlan) -> bool:
        """
        Execute enable mixed precision recovery strategy.
        
        Args:
            action: Recovery action
            plan: Recovery plan
            
        Returns:
            bool: Whether execution was successful
        """
        logger.info("Executing enable mixed precision recovery strategy")
        
        try:
            # Check if orchestrator is available
            if self.orchestrator is None:
                logger.warning("No orchestrator registered, creating signal file")
                
                # Create a signal file for external monitoring to detect
                signal_file = self.recovery_dir / "ENABLE_MIXED_PRECISION"
                with open(signal_file, "w") as f:
                    f.write(f"timestamp: {time.time()}\n")
                    f.write(f"failure_type: {plan.failure_context.failure_type.value}\n")
                
                logger.info(f"Created mixed precision signal file: {signal_file}")
                return True
            
            # Update config in orchestrator
            if hasattr(self.orchestrator, "update_config"):
                self.orchestrator.update_config({"training": {"mixed_precision": True}})
                logger.info("Enabled mixed precision via orchestrator")
                return True
            
            # Alternative: create signal file
            signal_file = self.recovery_dir / "ENABLE_MIXED_PRECISION"
            with open(signal_file, "w") as f:
                f.write(f"timestamp: {time.time()}\n")
                f.write(f"failure_type: {plan.failure_context.failure_type.value}\n")
            
            logger.info(f"Created mixed precision signal file: {signal_file}")
            return True
        
        except Exception as e:
            logger.error(f"Error enabling mixed precision: {e}")
            return False
    
    def _execute_rollback_checkpoint(self, action: RecoveryAction, plan: RecoveryPlan) -> bool:
        """
        Execute rollback checkpoint recovery strategy.
        
        Args:
            action: Recovery action
            plan: Recovery plan
            
        Returns:
            bool: Whether execution was successful
        """
        logger.info("Executing rollback checkpoint recovery strategy")
        
        # Get parameters
        num_checkpoints_back = action.params.get("num_checkpoints_back", 1)
        specific_checkpoint = action.params.get("specific_checkpoint")
        
        try:
            # Check if orchestrator is available
            if self.orchestrator is None:
                logger.warning("No orchestrator registered, creating signal file")
                
                # Create a signal file for external monitoring to detect
                signal_file = self.recovery_dir / "ROLLBACK_CHECKPOINT"
                with open(signal_file, "w") as f:
                    f.write(f"timestamp: {time.time()}\n")
                    f.write(f"failure_type: {plan.failure_context.failure_type.value}\n")
                    f.write(f"num_checkpoints_back: {num_checkpoints_back}\n")
                    if specific_checkpoint:
                        f.write(f"specific_checkpoint: {specific_checkpoint}\n")
                
                logger.info(f"Created checkpoint rollback signal file: {signal_file}")
                return True
            
            # Find checkpoint to roll back to
            checkpoint_path = None
            
            if specific_checkpoint:
                checkpoint_path = specific_checkpoint
            elif hasattr(self.orchestrator, "get_checkpoint_history"):
                checkpoint_history = self.orchestrator.get_checkpoint_history()
                if len(checkpoint_history) > num_checkpoints_back:
                    checkpoint_path = checkpoint_history[-(num_checkpoints_back+1)]
            else:
                # Try to find checkpoints in the output directory
                checkpoint_dir = self.output_dir / "checkpoints"
                if checkpoint_dir.exists():
                    checkpoints = sorted(
                        [f for f in checkpoint_dir.glob("*.pt") if "emergency" not in f.name],
                        key=lambda f: f.stat().st_mtime,
                        reverse=True,
                    )
                    
                    if len(checkpoints) > num_checkpoints_back:
                        checkpoint_path = str(checkpoints[num_checkpoints_back])
            
            if not checkpoint_path:
                logger.warning("Could not find checkpoint to roll back to")
                return False
            
            logger.info(f"Rolling back to checkpoint: {checkpoint_path}")
            
            # Validate checkpoint
            if self.config.checkpoint_validation.corruption_detection:
                is_valid = self._validate_checkpoint(checkpoint_path)
                if not is_valid:
                    logger.warning(f"Checkpoint validation failed for {checkpoint_path}")
                    
                    if self.config.checkpoint_validation.auto_repair:
                        repaired = self._repair_checkpoint(checkpoint_path)
                        if not repaired:
                            logger.error(f"Failed to repair checkpoint {checkpoint_path}")
                            return False
                    else:
                        return False
            
            # Load checkpoint in orchestrator
            if hasattr(self.orchestrator, "load_checkpoint"):
                self.orchestrator.load_checkpoint(checkpoint_path)
                logger.info(f"Loaded checkpoint {checkpoint_path} via orchestrator")
                return True
            
            # Alternative: create signal file
            signal_file = self.recovery_dir / "ROLLBACK_CHECKPOINT"
            with open(signal_file, "w") as f:
                f.write(f"timestamp: {time.time()}\n")
                f.write(f"failure_type: {plan.failure_context.failure_type.value}\n")
                f.write(f"checkpoint_path: {checkpoint_path}\n")
            
            logger.info(f"Created checkpoint rollback signal file: {signal_file}")
            return True
        
        except Exception as e:
            logger.error(f"Error executing checkpoint rollback: {e}")
            return False
    
    def _execute_skip_bad_batch(self, action: RecoveryAction, plan: RecoveryPlan) -> bool:
        """
        Execute skip bad batch recovery strategy.
        
        Args:
            action: Recovery action
            plan: Recovery plan
            
        Returns:
            bool: Whether execution was successful
        """
        logger.info("Executing skip bad batch recovery strategy")
        
        try:
            # Get batch information
            batch_idx = plan.failure_context.batch_idx
            if batch_idx is None:
                logger.warning("No batch index available in failure context")
                batch_idx = -1
            
            # Check if orchestrator is available
            if self.orchestrator is None:
                logger.warning("No orchestrator registered, creating signal file")
                
                # Create a signal file for external monitoring to detect
                signal_file = self.recovery_dir / "SKIP_BAD_BATCH"
                with open(signal_file, "w") as f:
                    f.write(f"timestamp: {time.time()}\n")
                    f.write(f"failure_type: {plan.failure_context.failure_type.value}\n")
                    f.write(f"batch_idx: {batch_idx}\n")
                
                logger.info(f"Created skip batch signal file: {signal_file}")
                return True
            
            # Skip batch in orchestrator
            if hasattr(self.orchestrator, "skip_batch"):
                self.orchestrator.skip_batch(batch_idx)
                logger.info(f"Skipped batch {batch_idx} via orchestrator")
                return True
            
            # Alternative: create signal file
            signal_file = self.recovery_dir / "SKIP_BAD_BATCH"
            with open(signal_file, "w") as f:
                f.write(f"timestamp: {time.time()}\n")
                f.write(f"failure_type: {plan.failure_context.failure_type.value}\n")
                f.write(f"batch_idx: {batch_idx}\n")
            
            logger.info(f"Created skip batch signal file: {signal_file}")
            return True
        
        except Exception as e:
            logger.error(f"Error executing skip bad batch: {e}")
            return False
    
    def _execute_clean_memory(self, action: RecoveryAction, plan: RecoveryPlan) -> bool:
        """
        Execute clean memory recovery strategy.
        
        Args:
            action: Recovery action
            plan: Recovery plan
            
        Returns:
            bool: Whether execution was successful
        """
        logger.info("Executing clean memory recovery strategy")
        
        try:
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache if available
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA memory cache")
            
            # Clear MPS cache if available
            if TORCH_AVAILABLE and hasattr(torch, 'mps') and torch.mps.is_available():
                torch.mps.empty_cache()
                logger.info("Cleared MPS memory cache")
            
            # Check if orchestrator is available
            if self.orchestrator is not None and hasattr(self.orchestrator, "clean_memory"):
                self.orchestrator.clean_memory()
                logger.info("Cleaned memory via orchestrator")
            
            # Create signal file
            signal_file = self.recovery_dir / "CLEAN_MEMORY"
            with open(signal_file, "w") as f:
                f.write(f"timestamp: {time.time()}\n")
                f.write(f"failure_type: {plan.failure_context.failure_type.value}\n")
            
            logger.info("Memory cleaned successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error cleaning memory: {e}")
            return False
    
    def _execute_clean_disk(self, action: RecoveryAction, plan: RecoveryPlan) -> bool:
        """
        Execute clean disk recovery strategy.
        
        Args:
            action: Recovery action
            plan: Recovery plan
            
        Returns:
            bool: Whether execution was successful
        """
        logger.info("Executing clean disk recovery strategy")
        
        # Get parameters
        min_free_gb = action.params.get("min_free_gb", 10)
        keep_last_n_checkpoints = action.params.get("keep_last_n_checkpoints", 3)
        clean_tensorboard = action.params.get("clean_tensorboard", True)
        clean_logs = action.params.get("clean_logs", True)
        
        try:
            # Check current disk space
            if PSUTIL_AVAILABLE:
                disk = psutil.disk_usage(self.output_dir)
                free_gb = disk.free / (1024 ** 3)
                logger.info(f"Current free disk space: {free_gb:.2f} GB")
                
                if free_gb >= min_free_gb:
                    logger.info(f"Sufficient disk space available ({free_gb:.2f} GB), skipping cleanup")
                    return True
            
            # Clean old checkpoints
            checkpoint_dir = self.output_dir / "checkpoints"
            if checkpoint_dir.exists():
                checkpoints = sorted(
                    [f for f in checkpoint_dir.glob("*.pt") if "best" not in f.name and "latest" not in f.name],
                    key=lambda f: f.stat().st_mtime,
                    reverse=True,
                )
                
                # Keep the N most recent checkpoints
                checkpoints_to_remove = checkpoints[keep_last_n_checkpoints:]
                
                for checkpoint in checkpoints_to_remove:
                    logger.info(f"Removing old checkpoint: {checkpoint}")
                    checkpoint.unlink()
            
            # Clean TensorBoard logs if enabled
            if clean_tensorboard:
                tensorboard_dir = self.output_dir / "tensorboard"
                if tensorboard_dir.exists():
                    # Keep only the most recent TensorBoard files
                    tb_files = sorted(
                        tensorboard_dir.glob("events.out.tfevents.*"),
                        key=lambda f: f.stat().st_mtime,
                        reverse=True,
                    )
                    
                    # Keep the most recent file
                    tb_files_to_remove = tb_files[1:]
                    
                    for tb_file in tb_files_to_remove:
                        logger.info(f"Removing old TensorBoard file: {tb_file}")
                        tb_file.unlink()
            
            # Clean logs if enabled
            if clean_logs:
                logs_dir = self.output_dir / "logs"
                if logs_dir.exists():
                    # Remove log files older than 7 days
                    current_time = time.time()
                    for log_file in logs_dir.glob("*.log"):
                        file_age_days = (current_time - log_file.stat().st_mtime) / (24 * 3600)
                        if file_age_days > 7:
                            logger.info(f"Removing old log file: {log_file}")
                            log_file.unlink()
            
            # Check disk space after cleanup
            if PSUTIL_AVAILABLE:
                disk = psutil.disk_usage(self.output_dir)
                free_gb_after = disk.free / (1024 ** 3)
                logger.info(f"Free disk space after cleanup: {free_gb_after:.2f} GB")
                
                if free_gb_after < min_free_gb:
                    logger.warning(
                        f"Still insufficient disk space after cleanup: {free_gb_after:.2f} GB "
                        f"(required: {min_free_gb} GB)"
                    )
                    return False
            
            logger.info("Disk cleanup completed successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error cleaning disk: {e}")
            return False
    
    def _execute_relocate_node(self, action: RecoveryAction, plan: RecoveryPlan) -> bool:
        """
        Execute relocate node recovery strategy.
        
        Args:
            action: Recovery action
            plan: Recovery plan
            
        Returns:
            bool: Whether execution was successful
        """
        logger.info("Executing relocate node recovery strategy")
        
        # Check if cloud recovery is enabled
        if self.config.cloud_recovery.provider == CloudProvider.NONE:
            logger.warning("Cloud recovery is not enabled, cannot relocate node")
            return False
        
        # Check if cloud client is available
        if self.cloud_client is None:
            logger.warning("Cloud client is not available, cannot relocate node")
            return False
        
        # Get parameters
        node_rank = plan.failure_context.node_rank
        if node_rank is None:
            logger.warning("No node