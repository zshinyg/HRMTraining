#!/usr/bin/env python
"""
Health Monitor for Training Jobs

This module provides a comprehensive health monitoring system for training jobs,
tracking system resources, training metrics, and job status. It implements
automatic recovery mechanisms, anomaly detection, performance profiling,
and integration with external monitoring systems.

Features:
- System resource monitoring (CPU, GPU, memory, disk, network)
- Training metrics anomaly detection and early stopping
- Heartbeat monitoring for CI/CD job health checks
- Automatic recovery mechanisms for common failure scenarios
- Performance profiling and bottleneck detection
- Network connectivity monitoring for distributed training
- Thermal monitoring and performance throttling alerts
- Cost tracking for cloud training resources
- Automated cleanup for failed runs and orphaned resources
- Integration with external monitoring systems (Prometheus, Grafana, etc.)

Usage:
    from scripts.training.health_monitor import HealthMonitor
    
    # Initialize monitor
    monitor = HealthMonitor(
        experiment_name="hrm_training",
        output_dir="outputs/hrm_training",
        config_path="configs/training_config.yaml",
        enable_recovery=True,
    )
    
    # Start monitoring
    monitor.start()
    
    # Register model and optimizer for gradient/weight monitoring
    monitor.register_model(model, optimizer)
    
    # Update training metrics
    monitor.update_training_metrics(loss=2.3, accuracy=0.85, step=100)
    
    # Check health status
    status = monitor.check_health()
    if not status.healthy:
        print(f"Health check failed: {status.reason}")
    
    # Stop monitoring
    monitor.stop()
"""

import atexit
import concurrent.futures
import datetime
import gc
import glob
import json
import logging
import os
import platform
import re
import shutil
import signal
import socket
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
import psutil
import yaml

# Optional imports with fallbacks
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

try:
    import py3nvml.py3nvml as nvml
    NVML_AVAILABLE = True
except ImportError:
    try:
        import pynvml as nvml
        NVML_AVAILABLE = True
    except ImportError:
        NVML_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import boto3
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    from google.cloud import monitoring_v3
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

try:
    import azure.monitor.opentelemetry.exporter
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status of the training job."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"
    RECOVERING = "recovering"
    UNKNOWN = "unknown"


class RecoveryAction(Enum):
    """Recovery actions for health issues."""
    NONE = "none"
    RESTART_PROCESS = "restart_process"
    CLEAR_CACHE = "clear_cache"
    REDUCE_BATCH_SIZE = "reduce_batch_size"
    REDUCE_LEARNING_RATE = "reduce_learning_rate"
    CHECKPOINT_AND_RESUME = "checkpoint_and_resume"
    SKIP_BAD_BATCH = "skip_bad_batch"
    TERMINATE = "terminate"
    CUSTOM = "custom"


class MonitoringLevel(Enum):
    """Monitoring detail level."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    DETAILED = "detailed"
    DEBUG = "debug"


class AnomalyDetectionMethod(Enum):
    """Method used for anomaly detection."""
    THRESHOLD = "threshold"
    MOVING_AVERAGE = "moving_average"
    Z_SCORE = "z_score"
    IQR = "iqr"
    ISOLATION_FOREST = "isolation_forest"
    AUTOENCODER = "autoencoder"


@dataclass
class ResourceStats:
    """Statistics about system resources."""
    timestamp: float = field(default_factory=time.time)
    
    # CPU stats
    cpu_percent: float = 0.0
    cpu_count: int = 0
    cpu_freq_current: Optional[float] = None
    cpu_freq_max: Optional[float] = None
    cpu_temperature: Optional[float] = None
    
    # Memory stats
    memory_used: float = 0.0  # GB
    memory_available: float = 0.0  # GB
    memory_total: float = 0.0  # GB
    memory_percent: float = 0.0
    swap_used: float = 0.0  # GB
    swap_free: float = 0.0  # GB
    swap_percent: float = 0.0
    
    # Disk stats
    disk_used: float = 0.0  # GB
    disk_free: float = 0.0  # GB
    disk_total: float = 0.0  # GB
    disk_percent: float = 0.0
    disk_read_speed: float = 0.0  # MB/s
    disk_write_speed: float = 0.0  # MB/s
    
    # Network stats
    network_sent: float = 0.0  # MB/s
    network_received: float = 0.0  # MB/s
    network_connections: int = 0
    network_errors: int = 0
    
    # GPU stats
    gpu_count: int = 0
    gpu_stats: List[Dict[str, Any]] = field(default_factory=list)
    
    # Process stats
    process_count: int = 0
    process_threads: int = 0
    process_memory: float = 0.0  # GB
    process_cpu_percent: float = 0.0
    
    # System load
    load_1min: float = 0.0
    load_5min: float = 0.0
    load_15min: float = 0.0


@dataclass
class TrainingMetrics:
    """Training metrics for health monitoring."""
    timestamp: float = field(default_factory=time.time)
    step: int = 0
    epoch: int = 0
    
    # Basic metrics
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    
    # Performance metrics
    samples_per_second: Optional[float] = None
    seconds_per_step: Optional[float] = None
    
    # Gradient stats
    gradient_norm: Optional[float] = None
    gradient_max: Optional[float] = None
    gradient_min: Optional[float] = None
    gradient_mean: Optional[float] = None
    gradient_std: Optional[float] = None
    
    # Weight stats
    weight_norm: Optional[float] = None
    weight_max: Optional[float] = None
    weight_min: Optional[float] = None
    weight_mean: Optional[float] = None
    weight_std: Optional[float] = None
    
    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    timestamp: float = field(default_factory=time.time)
    status: HealthStatus = HealthStatus.UNKNOWN
    reason: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    recovery_action: RecoveryAction = RecoveryAction.NONE
    recovery_details: Optional[str] = None
    
    @property
    def healthy(self) -> bool:
        """Return True if the status is HEALTHY."""
        return self.status == HealthStatus.HEALTHY


@dataclass
class AnomalyDetectionConfig:
    """Configuration for anomaly detection."""
    method: AnomalyDetectionMethod = AnomalyDetectionMethod.MOVING_AVERAGE
    window_size: int = 50
    threshold: float = 3.0
    min_samples: int = 10
    exclude_metrics: List[str] = field(default_factory=list)
    custom_thresholds: Dict[str, float] = field(default_factory=dict)


@dataclass
class CostTrackingConfig:
    """Configuration for cost tracking."""
    enabled: bool = False
    provider: str = "aws"  # aws, gcp, azure
    instance_type: Optional[str] = None
    instance_cost_per_hour: Optional[float] = None
    gpu_cost_per_hour: Optional[float] = None
    storage_cost_per_gb_month: Optional[float] = None
    network_cost_per_gb: Optional[float] = None
    currency: str = "USD"


@dataclass
class ExternalMonitoringConfig:
    """Configuration for external monitoring systems."""
    prometheus_enabled: bool = False
    prometheus_port: int = 8000
    grafana_enabled: bool = False
    grafana_url: Optional[str] = None
    grafana_api_key: Optional[str] = None
    cloudwatch_enabled: bool = False
    stackdriver_enabled: bool = False
    azure_monitor_enabled: bool = False
    custom_webhook_url: Optional[str] = None


@dataclass
class HealthMonitorConfig:
    """Configuration for health monitoring."""
    # Monitoring settings
    monitoring_level: MonitoringLevel = MonitoringLevel.STANDARD
    check_interval_seconds: float = 5.0
    resource_check_interval_seconds: float = 10.0
    heartbeat_interval_seconds: float = 30.0
    log_interval_seconds: float = 60.0
    
    # Thresholds
    cpu_threshold_percent: float = 95.0
    memory_threshold_percent: float = 95.0
    gpu_memory_threshold_percent: float = 95.0
    gpu_temperature_threshold_celsius: float = 85.0
    disk_threshold_percent: float = 95.0
    
    # Recovery settings
    enable_recovery: bool = True
    max_recovery_attempts: int = 3
    recovery_cooldown_seconds: float = 300.0
    
    # Anomaly detection
    anomaly_detection: AnomalyDetectionConfig = field(default_factory=AnomalyDetectionConfig)
    
    # Early stopping
    enable_early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    early_stopping_metric: str = "val/loss"
    early_stopping_mode: str = "min"  # min or max
    
    # Cost tracking
    cost_tracking: CostTrackingConfig = field(default_factory=CostTrackingConfig)
    
    # External monitoring
    external_monitoring: ExternalMonitoringConfig = field(default_factory=ExternalMonitoringConfig)
    
    # Cleanup settings
    enable_auto_cleanup: bool = True
    cleanup_threshold_hours: float = 24.0
    keep_last_n_checkpoints: int = 3
    
    # Advanced settings
    enable_thermal_monitoring: bool = True
    enable_network_monitoring: bool = True
    enable_process_monitoring: bool = True
    enable_gradient_monitoring: bool = True
    track_memory_leaks: bool = True
    track_cuda_memory_fragmentation: bool = True


class HealthMonitor:
    """
    Health monitoring system for training jobs.
    
    This class provides comprehensive health monitoring for training jobs,
    including system resources, training metrics, and job status. It implements
    automatic recovery mechanisms, anomaly detection, and integration with
    external monitoring systems.
    """
    
    def __init__(
        self,
        experiment_name: str,
        output_dir: str,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        monitoring_level: Optional[MonitoringLevel] = None,
        enable_recovery: Optional[bool] = None,
    ):
        """
        Initialize the health monitor.
        
        Args:
            experiment_name: Name of the experiment
            output_dir: Directory to save outputs
            config_path: Path to configuration file (optional)
            config: Configuration dictionary (optional, overrides config_path)
            monitoring_level: Level of monitoring detail (optional, overrides config)
            enable_recovery: Whether to enable automatic recovery (optional, overrides config)
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.config_path = config_path
        
        # Load configuration
        self.config = self._load_config(config_path, config)
        
        # Override config with constructor arguments
        if monitoring_level is not None:
            self.config.monitoring_level = monitoring_level
        
        if enable_recovery is not None:
            self.config.enable_recovery = enable_recovery
        
        # Initialize state
        self.running = False
        self.resource_stats_history = []
        self.training_metrics_history = []
        self.health_check_history = []
        self.recovery_history = []
        self.cost_history = []
        
        # Monitoring threads
        self.monitoring_thread = None
        self.heartbeat_thread = None
        self.prometheus_server = None
        
        # Locks for thread safety
        self.stats_lock = threading.Lock()
        self.metrics_lock = threading.Lock()
        self.health_lock = threading.Lock()
        
        # Model and optimizer for gradient monitoring
        self.model = None
        self.optimizer = None
        
        # Counters and stats
        self.start_time = None
        self.last_resource_check_time = 0
        self.last_heartbeat_time = 0
        self.last_log_time = 0
        self.recovery_attempts = 0
        self.last_recovery_time = 0
        self.last_disk_io = (0, 0)  # (read_bytes, write_bytes)
        self.last_network_io = (0, 0)  # (bytes_sent, bytes_recv)
        
        # Early stopping state
        self.early_stopping_best_value = None
        self.early_stopping_counter = 0
        
        # Cost tracking state
        self.cost_start_time = None
        self.total_cost = 0.0
        
        # Prometheus metrics
        self.prometheus_metrics = {}
        
        # Initialize directories
        self._init_directories()
        
        # Register signal handlers and exit handler
        self._register_handlers()
        
        # Initialize external monitoring systems
        if self.config.external_monitoring.prometheus_enabled and PROMETHEUS_AVAILABLE:
            self._init_prometheus()
        
        logger.info(f"Health monitor initialized for experiment: {experiment_name}")
    
    def _load_config(
        self,
        config_path: Optional[str],
        config_dict: Optional[Dict[str, Any]],
    ) -> HealthMonitorConfig:
        """
        Load configuration from file or dictionary.
        
        Args:
            config_path: Path to configuration file
            config_dict: Configuration dictionary
            
        Returns:
            HealthMonitorConfig: Configuration object
        """
        # Start with default config
        config = HealthMonitorConfig()
        
        # Load from file if provided
        if config_path:
            try:
                with open(config_path, "r") as f:
                    file_config = yaml.safe_load(f)
                
                # Extract health monitor config
                if "health_monitor" in file_config:
                    monitor_config = file_config["health_monitor"]
                else:
                    monitor_config = file_config
                
                # Update config with file values
                self._update_config_from_dict(config, monitor_config)
                
                logger.info(f"Loaded health monitor configuration from {config_path}")
            
            except Exception as e:
                logger.error(f"Error loading configuration from {config_path}: {e}")
        
        # Override with provided config dict
        if config_dict:
            self._update_config_from_dict(config, config_dict)
            logger.info("Applied custom configuration")
        
        return config
    
    def _update_config_from_dict(self, config: HealthMonitorConfig, config_dict: Dict[str, Any]):
        """
        Update configuration from dictionary.
        
        Args:
            config: Configuration object to update
            config_dict: Dictionary with configuration values
        """
        # Update top-level settings
        for key, value in config_dict.items():
            if key == "monitoring_level" and isinstance(value, str):
                try:
                    setattr(config, key, MonitoringLevel(value))
                except ValueError:
                    logger.warning(f"Invalid monitoring level: {value}")
            elif key == "anomaly_detection" and isinstance(value, dict):
                for ad_key, ad_value in value.items():
                    if ad_key == "method" and isinstance(ad_value, str):
                        try:
                            config.anomaly_detection.method = AnomalyDetectionMethod(ad_value)
                        except ValueError:
                            logger.warning(f"Invalid anomaly detection method: {ad_value}")
                    elif hasattr(config.anomaly_detection, ad_key):
                        setattr(config.anomaly_detection, ad_key, ad_value)
            elif key == "cost_tracking" and isinstance(value, dict):
                for ct_key, ct_value in value.items():
                    if hasattr(config.cost_tracking, ct_key):
                        setattr(config.cost_tracking, ct_key, ct_value)
            elif key == "external_monitoring" and isinstance(value, dict):
                for em_key, em_value in value.items():
                    if hasattr(config.external_monitoring, em_key):
                        setattr(config.external_monitoring, em_key, em_value)
            elif hasattr(config, key):
                setattr(config, key, value)
    
    def _init_directories(self):
        """Initialize directories for health monitor outputs."""
        try:
            # Create main directories
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create health monitor subdirectory
            self.health_dir = self.output_dir / "health"
            self.health_dir.mkdir(exist_ok=True)
            
            # Create subdirectories
            (self.health_dir / "logs").mkdir(exist_ok=True)
            (self.health_dir / "stats").mkdir(exist_ok=True)
            (self.health_dir / "alerts").mkdir(exist_ok=True)
            (self.health_dir / "reports").mkdir(exist_ok=True)
            
            logger.debug(f"Initialized health monitor directories in {self.health_dir}")
        
        except Exception as e:
            logger.error(f"Error initializing directories: {e}")
    
    def _register_handlers(self):
        """Register signal handlers and exit handler."""
        # Register exit handler
        atexit.register(self._cleanup)
        
        # Register signal handlers
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}, stopping health monitor...")
            self.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _cleanup(self):
        """Clean up resources before exiting."""
        if self.running:
            self.stop()
    
    def _init_prometheus(self):
        """Initialize Prometheus metrics and server."""
        try:
            # Create metrics
            # System metrics
            self.prometheus_metrics["cpu_percent"] = Gauge(
                "training_cpu_percent", "CPU usage percentage"
            )
            self.prometheus_metrics["memory_percent"] = Gauge(
                "training_memory_percent", "Memory usage percentage"
            )
            self.prometheus_metrics["disk_percent"] = Gauge(
                "training_disk_percent", "Disk usage percentage"
            )
            
            # GPU metrics
            self.prometheus_metrics["gpu_memory_percent"] = Gauge(
                "training_gpu_memory_percent", "GPU memory usage percentage", ["gpu_id"]
            )
            self.prometheus_metrics["gpu_temperature"] = Gauge(
                "training_gpu_temperature", "GPU temperature in Celsius", ["gpu_id"]
            )
            self.prometheus_metrics["gpu_utilization"] = Gauge(
                "training_gpu_utilization", "GPU utilization percentage", ["gpu_id"]
            )
            
            # Training metrics
            self.prometheus_metrics["training_loss"] = Gauge(
                "training_loss", "Training loss"
            )
            self.prometheus_metrics["training_samples_per_second"] = Gauge(
                "training_samples_per_second", "Training throughput in samples per second"
            )
            self.prometheus_metrics["training_step"] = Gauge(
                "training_step", "Current training step"
            )
            
            # Health metrics
            self.prometheus_metrics["health_status"] = Gauge(
                "training_health_status", "Health status (0=unknown, 1=healthy, 2=warning, 3=critical, 4=failed, 5=recovering)"
            )
            self.prometheus_metrics["recovery_attempts"] = Counter(
                "training_recovery_attempts_total", "Total number of recovery attempts"
            )
            
            # Cost metrics
            self.prometheus_metrics["total_cost"] = Gauge(
                "training_total_cost", f"Total cost in {self.config.cost_tracking.currency}"
            )
            
            # Start server
            port = self.config.external_monitoring.prometheus_port
            start_http_server(port)
            logger.info(f"Started Prometheus metrics server on port {port}")
        
        except Exception as e:
            logger.error(f"Error initializing Prometheus: {e}")
            self.config.external_monitoring.prometheus_enabled = False
    
    def _update_prometheus_metrics(self, resource_stats: ResourceStats, training_metrics: Optional[TrainingMetrics] = None):
        """
        Update Prometheus metrics.
        
        Args:
            resource_stats: Current resource statistics
            training_metrics: Current training metrics (optional)
        """
        if not self.config.external_monitoring.prometheus_enabled or not PROMETHEUS_AVAILABLE:
            return
        
        try:
            # Update system metrics
            self.prometheus_metrics["cpu_percent"].set(resource_stats.cpu_percent)
            self.prometheus_metrics["memory_percent"].set(resource_stats.memory_percent)
            self.prometheus_metrics["disk_percent"].set(resource_stats.disk_percent)
            
            # Update GPU metrics
            for i, gpu_stat in enumerate(resource_stats.gpu_stats):
                self.prometheus_metrics["gpu_memory_percent"].labels(gpu_id=str(i)).set(
                    gpu_stat.get("memory_percent", 0)
                )
                self.prometheus_metrics["gpu_temperature"].labels(gpu_id=str(i)).set(
                    gpu_stat.get("temperature", 0)
                )
                self.prometheus_metrics["gpu_utilization"].labels(gpu_id=str(i)).set(
                    gpu_stat.get("utilization", 0)
                )
            
            # Update training metrics
            if training_metrics:
                if training_metrics.loss is not None:
                    self.prometheus_metrics["training_loss"].set(training_metrics.loss)
                
                if training_metrics.samples_per_second is not None:
                    self.prometheus_metrics["training_samples_per_second"].set(
                        training_metrics.samples_per_second
                    )
                
                self.prometheus_metrics["training_step"].set(training_metrics.step)
            
            # Update health metrics
            health_status_value = {
                HealthStatus.UNKNOWN: 0,
                HealthStatus.HEALTHY: 1,
                HealthStatus.WARNING: 2,
                HealthStatus.CRITICAL: 3,
                HealthStatus.FAILED: 4,
                HealthStatus.RECOVERING: 5,
            }.get(self._get_current_health_status(), 0)
            
            self.prometheus_metrics["health_status"].set(health_status_value)
            
            # Update cost metrics
            self.prometheus_metrics["total_cost"].set(self.total_cost)
        
        except Exception as e:
            logger.error(f"Error updating Prometheus metrics: {e}")
    
    def start(self):
        """Start health monitoring."""
        if self.running:
            logger.warning("Health monitor is already running")
            return
        
        logger.info("Starting health monitor")
        self.running = True
        self.start_time = time.time()
        self.cost_start_time = time.time()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="health_monitor_thread",
        )
        self.monitoring_thread.start()
        
        # Start heartbeat thread
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name="heartbeat_thread",
        )
        self.heartbeat_thread.start()
        
        # Create initial health check file
        self._update_heartbeat_file(HealthStatus.HEALTHY)
        
        logger.info("Health monitor started")
    
    def stop(self):
        """Stop health monitoring."""
        if not self.running:
            return
        
        logger.info("Stopping health monitor")
        self.running = False
        
        # Wait for threads to finish
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=5.0)
        
        # Save final report
        self._save_final_report()
        
        logger.info("Health monitor stopped")
    
    def register_model(self, model, optimizer=None):
        """
        Register model and optimizer for gradient monitoring.
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer (optional)
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, gradient monitoring disabled")
            return
        
        self.model = model
        self.optimizer = optimizer
        logger.debug("Registered model and optimizer for gradient monitoring")
    
    def update_training_metrics(self, step: int, **metrics):
        """
        Update training metrics.
        
        Args:
            step: Current training step
            **metrics: Training metrics (e.g., loss, accuracy, etc.)
        """
        # Create metrics object
        training_metrics = TrainingMetrics(
            timestamp=time.time(),
            step=step,
        )
        
        # Set standard metrics if provided
        if "loss" in metrics:
            training_metrics.loss = float(metrics.pop("loss"))
        
        if "learning_rate" in metrics:
            training_metrics.learning_rate = float(metrics.pop("learning_rate"))
        
        if "epoch" in metrics:
            training_metrics.epoch = int(metrics.pop("epoch"))
        
        if "samples_per_second" in metrics:
            training_metrics.samples_per_second = float(metrics.pop("samples_per_second"))
        
        if "seconds_per_step" in metrics:
            training_metrics.seconds_per_step = float(metrics.pop("seconds_per_step"))
        
        # Add remaining metrics as custom metrics
        for key, value in metrics.items():
            training_metrics.custom_metrics[key] = float(value)
        
        # Add gradient stats if enabled and model is available
        if (
            self.config.enable_gradient_monitoring and
            TORCH_AVAILABLE and
            self.model is not None and
            self.optimizer is not None
        ):
            self._update_gradient_stats(training_metrics)
        
        # Update metrics history
        with self.metrics_lock:
            self.training_metrics_history.append(training_metrics)
            
            # Trim history if too long
            max_history = 1000
            if len(self.training_metrics_history) > max_history:
                self.training_metrics_history = self.training_metrics_history[-max_history:]
        
        # Check for early stopping
        if self.config.enable_early_stopping:
            self._check_early_stopping(training_metrics)
        
        # Check for anomalies
        if len(self.training_metrics_history) >= self.config.anomaly_detection.min_samples:
            self._check_for_anomalies(training_metrics)
        
        # Log to W&B if available
        if WANDB_AVAILABLE and wandb.run is not None:
            self._log_metrics_to_wandb(training_metrics)
    
    def _update_gradient_stats(self, metrics: TrainingMetrics):
        """
        Update gradient statistics in metrics.
        
        Args:
            metrics: Training metrics to update
        """
        try:
            # Get parameter gradients
            all_grads = []
            for param in self.model.parameters():
                if param.grad is not None:
                    grad = param.grad.detach().cpu().flatten()
                    all_grads.append(grad)
            
            if not all_grads:
                return
            
            # Concatenate all gradients
            all_grads = torch.cat(all_grads)
            
            # Calculate gradient statistics
            metrics.gradient_norm = float(torch.norm(all_grads).item())
            metrics.gradient_max = float(torch.max(all_grads).item())
            metrics.gradient_min = float(torch.min(all_grads).item())
            metrics.gradient_mean = float(torch.mean(all_grads).item())
            metrics.gradient_std = float(torch.std(all_grads).item())
            
            # Calculate weight statistics
            all_weights = []
            for param in self.model.parameters():
                weights = param.detach().cpu().flatten()
                all_weights.append(weights)
            
            all_weights = torch.cat(all_weights)
            
            metrics.weight_norm = float(torch.norm(all_weights).item())
            metrics.weight_max = float(torch.max(all_weights).item())
            metrics.weight_min = float(torch.min(all_weights).item())
            metrics.weight_mean = float(torch.mean(all_weights).item())
            metrics.weight_std = float(torch.std(all_weights).item())
        
        except Exception as e:
            logger.error(f"Error updating gradient stats: {e}")
    
    def _check_early_stopping(self, metrics: TrainingMetrics):
        """
        Check if early stopping criteria are met.
        
        Args:
            metrics: Current training metrics
        """
        # Get the metric to monitor
        metric_name = self.config.early_stopping_metric
        metric_value = None
        
        # Check in custom metrics
        if metric_name in metrics.custom_metrics:
            metric_value = metrics.custom_metrics[metric_name]
        # Check in standard metrics
        elif metric_name == "loss" and metrics.loss is not None:
            metric_value = metrics.loss
        
        if metric_value is None:
            return
        
        # Initialize best value if not set
        if self.early_stopping_best_value is None:
            self.early_stopping_best_value = metric_value
            self.early_stopping_counter = 0
            return
        
        # Check if improvement
        improved = False
        if self.config.early_stopping_mode == "min":
            improved = metric_value < (self.early_stopping_best_value - self.config.early_stopping_min_delta)
        else:
            improved = metric_value > (self.early_stopping_best_value + self.config.early_stopping_min_delta)
        
        if improved:
            self.early_stopping_best_value = metric_value
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
            
            # Log warning if getting close to early stopping
            if self.early_stopping_counter == self.config.early_stopping_patience - 2:
                logger.warning(
                    f"Early stopping counter: {self.early_stopping_counter}/{self.config.early_stopping_patience}. "
                    f"No improvement in {metric_name} for {self.early_stopping_counter} evaluations."
                )
            
            # Check if patience exceeded
            if self.early_stopping_counter >= self.config.early_stopping_patience:
                logger.warning(
                    f"Early stopping triggered after {self.early_stopping_counter} evaluations "
                    f"without improvement in {metric_name}."
                )
                
                # Create health check result
                result = HealthCheckResult(
                    status=HealthStatus.WARNING,
                    reason=f"Early stopping triggered: no improvement in {metric_name} for {self.early_stopping_counter} evaluations",
                    details={
                        "metric_name": metric_name,
                        "best_value": self.early_stopping_best_value,
                        "current_value": metric_value,
                        "patience": self.config.early_stopping_patience,
                        "min_delta": self.config.early_stopping_min_delta,
                        "mode": self.config.early_stopping_mode,
                    },
                    recovery_action=RecoveryAction.NONE,
                )
                
                # Add to health check history
                with self.health_lock:
                    self.health_check_history.append(result)
                
                # Raise exception to trigger early stopping
                raise EarlyStoppingException(
                    f"Early stopping triggered: no improvement in {metric_name} "
                    f"for {self.early_stopping_counter} evaluations"
                )
    
    def _check_for_anomalies(self, metrics: TrainingMetrics):
        """
        Check for anomalies in training metrics.
        
        Args:
            metrics: Current training metrics
        """
        # Skip if not enough history
        if len(self.training_metrics_history) < self.config.anomaly_detection.min_samples:
            return
        
        # Get detection method
        method = self.config.anomaly_detection.method
        
        # Check for anomalies in loss
        if metrics.loss is not None and "loss" not in self.config.anomaly_detection.exclude_metrics:
            loss_history = [m.loss for m in self.training_metrics_history[-self.config.anomaly_detection.window_size:] if m.loss is not None]
            
            if len(loss_history) >= self.config.anomaly_detection.min_samples:
                is_anomaly, details = self._detect_anomaly(
                    "loss",
                    metrics.loss,
                    loss_history,
                    method,
                    self.config.anomaly_detection.threshold,
                )
                
                if is_anomaly:
                    logger.warning(f"Anomaly detected in loss: {metrics.loss:.4f}, {details}")
                    
                    # Create health check result
                    result = HealthCheckResult(
                        status=HealthStatus.WARNING,
                        reason=f"Anomaly detected in loss: {metrics.loss:.4f}",
                        details={
                            "metric": "loss",
                            "value": metrics.loss,
                            "detection_method": method.value,
                            "details": details,
                        },
                        recovery_action=RecoveryAction.NONE,
                    )
                    
                    # Add to health check history
                    with self.health_lock:
                        self.health_check_history.append(result)
        
        # Check for anomalies in gradient stats
        if (
            metrics.gradient_norm is not None and
            "gradient_norm" not in self.config.anomaly_detection.exclude_metrics
        ):
            grad_norm_history = [
                m.gradient_norm for m in self.training_metrics_history[-self.config.anomaly_detection.window_size:]
                if m.gradient_norm is not None
            ]
            
            if len(grad_norm_history) >= self.config.anomaly_detection.min_samples:
                is_anomaly, details = self._detect_anomaly(
                    "gradient_norm",
                    metrics.gradient_norm,
                    grad_norm_history,
                    method,
                    self.config.anomaly_detection.threshold,
                )
                
                if is_anomaly:
                    logger.warning(f"Anomaly detected in gradient norm: {metrics.gradient_norm:.4f}, {details}")
                    
                    # Create health check result
                    result = HealthCheckResult(
                        status=HealthStatus.WARNING,
                        reason=f"Anomaly detected in gradient norm: {metrics.gradient_norm:.4f}",
                        details={
                            "metric": "gradient_norm",
                            "value": metrics.gradient_norm,
                            "detection_method": method.value,
                            "details": details,
                        },
                        recovery_action=RecoveryAction.NONE,
                    )
                    
                    # Add to health check history
                    with self.health_lock:
                        self.health_check_history.append(result)
        
        # Check for anomalies in custom metrics
        for metric_name, metric_value in metrics.custom_metrics.items():
            if metric_name in self.config.anomaly_detection.exclude_metrics:
                continue
            
            metric_history = [
                m.custom_metrics.get(metric_name) for m in self.training_metrics_history[-self.config.anomaly_detection.window_size:]
                if metric_name in m.custom_metrics
            ]
            
            if len(metric_history) >= self.config.anomaly_detection.min_samples:
                # Use custom threshold if available
                threshold = self.config.anomaly_detection.custom_thresholds.get(
                    metric_name, self.config.anomaly_detection.threshold
                )
                
                is_anomaly, details = self._detect_anomaly(
                    metric_name,
                    metric_value,
                    metric_history,
                    method,
                    threshold,
                )
                
                if is_anomaly:
                    logger.warning(f"Anomaly detected in {metric_name}: {metric_value:.4f}, {details}")
                    
                    # Create health check result
                    result = HealthCheckResult(
                        status=HealthStatus.WARNING,
                        reason=f"Anomaly detected in {metric_name}: {metric_value:.4f}",
                        details={
                            "metric": metric_name,
                            "value": metric_value,
                            "detection_method": method.value,
                            "details": details,
                        },
                        recovery_action=RecoveryAction.NONE,
                    )
                    
                    # Add to health check history
                    with self.health_lock:
                        self.health_check_history.append(result)
    
    def _detect_anomaly(
        self,
        metric_name: str,
        current_value: float,
        history: List[float],
        method: AnomalyDetectionMethod,
        threshold: float,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect anomalies in a metric.
        
        Args:
            metric_name: Name of the metric
            current_value: Current value of the metric
            history: History of metric values
            method: Anomaly detection method
            threshold: Anomaly detection threshold
            
        Returns:
            Tuple[bool, Dict[str, Any]]: (is_anomaly, details)
        """
        details = {}
        
        if method == AnomalyDetectionMethod.THRESHOLD:
            # Simple threshold-based detection
            if metric_name.endswith("loss"):
                # For loss, check if value is much higher than min
                min_value = min(history)
                is_anomaly = current_value > min_value * threshold
                details = {
                    "min_value": min_value,
                    "threshold": threshold,
                    "ratio": current_value / min_value if min_value > 0 else float('inf'),
                }
            else:
                # For other metrics, check if outside range
                max_value = max(history)
                min_value = min(history)
                is_anomaly = (
                    current_value > max_value * threshold or
                    (current_value < min_value / threshold and min_value > 0)
                )
                details = {
                    "min_value": min_value,
                    "max_value": max_value,
                    "threshold": threshold,
                }
        
        elif method == AnomalyDetectionMethod.MOVING_AVERAGE:
            # Moving average-based detection
            mean_value = sum(history) / len(history)
            std_value = (sum((x - mean_value) ** 2 for x in history) / len(history)) ** 0.5
            
            z_score = (current_value - mean_value) / std_value if std_value > 0 else 0
            is_anomaly = abs(z_score) > threshold
            
            details = {
                "mean": mean_value,
                "std": std_value,
                "z_score": z_score,
                "threshold": threshold,
            }
        
        elif method == AnomalyDetectionMethod.Z_SCORE:
            # Z-score-based detection
            mean_value = sum(history) / len(history)
            std_value = (sum((x - mean_value) ** 2 for x in history) / len(history)) ** 0.5
            
            z_score = (current_value - mean_value) / std_value if std_value > 0 else 0
            is_anomaly = abs(z_score) > threshold
            
            details = {
                "mean": mean_value,
                "std": std_value,
                "z_score": z_score,
                "threshold": threshold,
            }
        
        elif method == AnomalyDetectionMethod.IQR:
            # IQR-based detection
            sorted_history = sorted(history)
            q1_idx = int(len(sorted_history) * 0.25)
            q3_idx = int(len(sorted_history) * 0.75)
            
            q1 = sorted_history[q1_idx]
            q3 = sorted_history[q3_idx]
            iqr = q3 - q1
            
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            is_anomaly = current_value < lower_bound or current_value > upper_bound
            
            details = {
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "threshold": threshold,
            }
        
        else:
            # Fallback to simple threshold
            mean_value = sum(history) / len(history)
            is_anomaly = current_value > mean_value * threshold
            
            details = {
                "mean": mean_value,
                "threshold": threshold,
                "ratio": current_value / mean_value if mean_value > 0 else float('inf'),
            }
        
        return is_anomaly, details
    
    def _log_metrics_to_wandb(self, metrics: TrainingMetrics):
        """
        Log metrics to Weights & Biases.
        
        Args:
            metrics: Training metrics to log
        """
        if not WANDB_AVAILABLE or wandb.run is None:
            return
        
        try:
            # Create metrics dict
            wandb_metrics = {}
            
            # Add standard metrics
            if metrics.loss is not None:
                wandb_metrics["health/loss"] = metrics.loss
            
            if metrics.learning_rate is not None:
                wandb_metrics["health/learning_rate"] = metrics.learning_rate
            
            if metrics.samples_per_second is not None:
                wandb_metrics["health/samples_per_second"] = metrics.samples_per_second
            
            if metrics.seconds_per_step is not None:
                wandb_metrics["health/seconds_per_step"] = metrics.seconds_per_step
            
            # Add gradient stats
            if metrics.gradient_norm is not None:
                wandb_metrics["health/gradient_norm"] = metrics.gradient_norm
            
            if metrics.gradient_max is not None:
                wandb_metrics["health/gradient_max"] = metrics.gradient_max
            
            if metrics.gradient_min is not None:
                wandb_metrics["health/gradient_min"] = metrics.gradient_min
            
            if metrics.gradient_mean is not None:
                wandb_metrics["health/gradient_mean"] = metrics.gradient_mean
            
            if metrics.gradient_std is not None:
                wandb_metrics["health/gradient_std"] = metrics.gradient_std
            
            # Add weight stats
            if metrics.weight_norm is not None:
                wandb_metrics["health/weight_norm"] = metrics.weight_norm
            
            if metrics.weight_max is not None:
                wandb_metrics["health/weight_max"] = metrics.weight_max
            
            if metrics.weight_min is not None:
                wandb_metrics["health/weight_min"] = metrics.weight_min
            
            if metrics.weight_mean is not None:
                wandb_metrics["health/weight_mean"] = metrics.weight_mean
            
            if metrics.weight_std is not None:
                wandb_metrics["health/weight_std"] = metrics.weight_std
            
            # Add custom metrics
            for key, value in metrics.custom_metrics.items():
                wandb_metrics[f"health/custom/{key}"] = value
            
            # Log to W&B
            wandb.log(wandb_metrics, step=metrics.step)
        
        except Exception as e:
            logger.error(f"Error logging metrics to W&B: {e}")
    
    def check_health(self) -> HealthCheckResult:
        """
        Check the health of the training job.
        
        Returns:
            HealthCheckResult: Health check result
        """
        # Get latest resource stats
        resource_stats = self._get_current_resource_stats()
        
        # Check for issues
        issues = []
        
        # Check CPU usage
        if resource_stats.cpu_percent > self.config.cpu_threshold_percent:
            issues.append(
                f"High CPU usage: {resource_stats.cpu_percent:.1f}% (threshold: {self.config.cpu_threshold_percent:.1f}%)"
            )
        
        # Check memory usage
        if resource_stats.memory_percent > self.config.memory_threshold_percent:
            issues.append(
                f"High memory usage: {resource_stats.memory_percent:.1f}% (threshold: {self.config.memory_threshold_percent:.1f}%)"
            )
        
        # Check disk usage
        if resource_stats.disk_percent > self.config.disk_threshold_percent:
            issues.append(
                f"High disk usage: {resource_stats.disk_percent:.1f}% (threshold: {self.config.disk_threshold_percent:.1f}%)"
            )
        
        # Check GPU usage
        for i, gpu_stat in enumerate(resource_stats.gpu_stats):
            if gpu_stat.get("memory_percent", 0) > self.config.gpu_memory_threshold_percent:
                issues.append(
                    f"High GPU {i} memory usage: {gpu_stat.get('memory_percent', 0):.1f}% "
                    f"(threshold: {self.config.gpu_memory_threshold_percent:.1f}%)"
                )
            
            if gpu_stat.get("temperature", 0) > self.config.gpu_temperature_threshold_celsius:
                issues.append(
                    f"High GPU {i} temperature: {gpu_stat.get('temperature', 0):.1f}°C "
                    f"(threshold: {self.config.gpu_temperature_threshold_celsius:.1f}°C)"
                )
        
        # Check training metrics
        if self.training_metrics_history:
            latest_metrics = self.training_metrics_history[-1]
            
            # Check for NaN loss
            if latest_metrics.loss is not None and (
                np.isnan(latest_metrics.loss) or np.isinf(latest_metrics.loss)
            ):
                issues.append(f"Invalid loss value: {latest_metrics.loss}")
            
            # Check for zero gradients
            if (
                latest_metrics.gradient_norm is not None and
                latest_metrics.gradient_norm < 1e-8 and
                len(self.training_metrics_history) > 10
            ):
                issues.append(f"Near-zero gradient norm: {latest_metrics.gradient_norm:.2e}")
            
            # Check for exploding gradients
            if (
                latest_metrics.gradient_norm is not None and
                latest_metrics.gradient_norm > 1e3 and
                len(self.training_metrics_history) > 10
            ):
                issues.append(f"Exploding gradient norm: {latest_metrics.gradient_norm:.2e}")
        
        # Determine status based on issues
        if not issues:
            status = HealthStatus.HEALTHY
            reason = "No issues detected"
            recovery_action = RecoveryAction.NONE
        elif any("Invalid loss" in issue for issue in issues):
            status = HealthStatus.CRITICAL
            reason = issues[0]
            recovery_action = RecoveryAction.REDUCE_BATCH_SIZE
        elif any("Exploding gradient" in issue for issue in issues):
            status = HealthStatus.CRITICAL
            reason = issues[0]
            recovery_action = RecoveryAction.REDUCE_LEARNING_RATE
        elif any("High GPU" in issue and "temperature" in issue for issue in issues):
            status = HealthStatus.CRITICAL
            reason = issues[0]
            recovery_action = RecoveryAction.REDUCE_BATCH_SIZE
        elif any("High memory usage" in issue for issue in issues):
            status = HealthStatus.WARNING
            reason = issues[0]
            recovery_action = RecoveryAction.CLEAR_CACHE
        elif any("High disk usage" in issue for issue in issues):
            status = HealthStatus.WARNING
            reason = issues[0]
            recovery_action = RecoveryAction.NONE
        else:
            status = HealthStatus.WARNING
            reason = issues[0]
            recovery_action = RecoveryAction.NONE
        
        # Create result
        result = HealthCheckResult(
            status=status,
            reason=reason,
            details={
                "resource_stats": {
                    "cpu_percent": resource_stats.cpu_percent,
                    "memory_percent": resource_stats.memory_percent,
                    "disk_percent": resource_stats.disk_percent,
                    "gpu_stats": resource_stats.gpu_stats,
                },
                "issues": issues,
            },
            recovery_action=recovery_action,
        )
        
        # Add to health check history
        with self.health_lock:
            self.health_check_history.append(result)
        
        # Execute recovery action if enabled
        if (
            self.config.enable_recovery and
            result.recovery_action != RecoveryAction.NONE and
            status != HealthStatus.HEALTHY
        ):
            self._execute_recovery_action(result)
        
        return result
    
    def _execute_recovery_action(self, health_check: HealthCheckResult):
        """
        Execute a recovery action.
        
        Args:
            health_check: Health check result with recovery action
        """
        # Check if recovery is allowed
        if not self.config.enable_recovery:
            return
        
        # Check recovery cooldown
        if (
            time.time() - self.last_recovery_time < self.config.recovery_cooldown_seconds and
            self.recovery_attempts > 0
        ):
            logger.info(
                f"Recovery action {health_check.recovery_action.value} skipped due to cooldown "
                f"({time.time() - self.last_recovery_time:.1f}s < {self.config.recovery_cooldown_seconds:.1f}s)"
            )
            return
        
        # Check max recovery attempts
        if self.recovery_attempts >= self.config.max_recovery_attempts:
            logger.warning(
                f"Maximum recovery attempts reached ({self.recovery_attempts}/{self.config.max_recovery_attempts}), "
                f"skipping recovery action {health_check.recovery_action.value}"
            )
            return
        
        # Execute action
        logger.info(f"Executing recovery action: {health_check.recovery_action.value}")
        
        # Update state
        self.recovery_attempts += 1
        self.last_recovery_time = time.time()
        
        # Add to recovery history
        self.recovery_history.append({
            "timestamp": time.time(),
            "action": health_check.recovery_action.value,
            "reason": health_check.reason,
            "attempt": self.recovery_attempts,
        })
        
        # Update Prometheus counter
        if (
            self.config.external_monitoring.prometheus_enabled and
            PROMETHEUS_AVAILABLE and
            "recovery_attempts" in self.prometheus_metrics
        ):
            self.prometheus_metrics["recovery_attempts"].inc()
        
        # Execute specific action
        if health_check.recovery_action == RecoveryAction.CLEAR_CACHE:
            self._execute_clear_cache()
        elif health_check.recovery_action == RecoveryAction.REDUCE_BATCH_SIZE:
            self._execute_reduce_batch_size()
        elif health_check.recovery_action == RecoveryAction.REDUCE_LEARNING_RATE:
            self._execute_reduce_learning_rate()
        elif health_check.recovery_action == RecoveryAction.CHECKPOINT_AND_RESUME:
            self._execute_checkpoint_and_resume()
        elif health_check.recovery_action == RecoveryAction.RESTART_PROCESS:
            self._execute_restart_process()
        elif health_check.recovery_action == RecoveryAction.SKIP_BAD_BATCH:
            self._execute_skip_bad_batch()
        elif health_check.recovery_action == RecoveryAction.TERMINATE:
            self._execute_terminate()
    
    def _execute_clear_cache(self):
        """Execute clear cache recovery action."""
        logger.info("Clearing memory cache")
        
        # Clear Python memory
        gc.collect()
        
        # Clear CUDA cache if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear MPS cache if available
        if TORCH_AVAILABLE and hasattr(torch, 'mps') and torch.mps.is_available():
            torch.mps.empty_cache()
        
        logger.info("Memory cache cleared")
    
    def _execute_reduce_batch_size(self):
        """Execute reduce batch size recovery action."""
        logger.info("Recommending batch size reduction")
        
        # This is a recommendation only, as we can't directly modify the batch size
        # Create a signal file for the training process to detect
        signal_file = self.health_dir / "REDUCE_BATCH_SIZE"
        with open(signal_file, "w") as f:
            f.write(f"timestamp: {time.time()}\n")
            f.write(f"reason: {self.health_check_history[-1].reason}\n")
            f.write(f"recovery_attempt: {self.recovery_attempts}\n")
        
        logger.info(f"Created batch size reduction signal file: {signal_file}")
    
    def _execute_reduce_learning_rate(self):
        """Execute reduce learning rate recovery action."""
        logger.info("Recommending learning rate reduction")
        
        # This is a recommendation only, as we can't directly modify the learning rate
        # Create a signal file for the training process to detect
        signal_file = self.health_dir / "REDUCE_LEARNING_RATE"
        with open(signal_file, "w") as f:
            f.write(f"timestamp: {time.time()}\n")
            f.write(f"reason: {self.health_check_history[-1].reason}\n")
            f.write(f"recovery_attempt: {self.recovery_attempts}\n")
        
        logger.info(f"Created learning rate reduction signal file: {signal_file}")
    
    def _execute_checkpoint_and_resume(self):
        """Execute checkpoint and resume recovery action."""
        logger.info("Recommending checkpoint and resume")
        
        # This is a recommendation only, as we can't directly trigger a checkpoint
        # Create a signal file for the training process to detect
        signal_file = self.health_dir / "CHECKPOINT_NOW"
        with open(signal_file, "w") as f:
            f.write(f"timestamp: {time.time()}\n")
            f.write(f"reason: {self.health_check_history[-1].reason}\n")
            f.write(f"recovery_attempt: {self.recovery_attempts}\n")
        
        logger.info(f"Created checkpoint signal file: {signal_file}")
    
    def _execute_restart_process(self):
        """Execute restart process recovery action."""
        logger.warning("Process restart requested, but can only be done externally")
        
        # Create a signal file for external monitoring to detect
        signal_file = self.health_dir / "RESTART_PROCESS"
        with open(signal_file, "w") as f:
            f.write(f"timestamp: {time.time()}\n")
            f.write(f"reason: {self.health_check_history[-1].reason}\n")
            f.write(f"recovery_attempt: {self.recovery_attempts}\n")
            f.write(f"pid: {os.getpid()}\n")
        
        logger.info(f"Created process restart signal file: {signal_file}")
    
    def _execute_skip_bad_batch(self):
        """Execute skip bad batch recovery action."""
        logger.info("Recommending skipping bad batch")
        
        # This is a recommendation only, as we can't directly skip a batch
        # Create a signal file for the training process to detect
        signal_file = self.health_dir / "SKIP_BATCH"
        with open(signal_file, "w") as f:
            f.write(f"timestamp: {time.time()}\n")
            f.write(f"reason: {self.health_check_history[-1].reason}\n")
            f.write(f"recovery_attempt: {self.recovery_attempts}\n")
        
        logger.info(f"Created skip batch signal file: {signal_file}")
    
    def _execute_terminate(self):
        """Execute terminate recovery action."""
        logger.critical("Training termination requested due to unrecoverable error")
        
        # Create a signal file for external monitoring to detect
        signal_file = self.health_dir / "TERMINATE"
        with open(signal_file, "w") as f:
            f.write(f"timestamp: {time.time()}\n")
            f.write(f"reason: {self.health_check_history[-1].reason}\n")
            f.write(f"recovery_attempt: {self.recovery_attempts}\n")
            f.write(f"pid: {os.getpid()}\n")
        
        logger.info(f"Created termination signal file: {signal_file}")
        
        # This will be detected by the training process or external monitoring
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Check if it's time to collect resource stats
                current_time = time.time()
                if current_time - self.last_resource_check_time >= self.config.resource_check_interval_seconds:
                    # Collect resource stats
                    resource_stats = self._collect_resource_stats()
                    
                    # Update history
                    with self.stats_lock:
                        self.resource_stats_history.append(resource_stats)
                        
                        # Trim history if too long
                        max_history = 1000
                        if len(self.resource_stats_history) > max_history:
                            self.resource_stats_history = self.resource_stats_history[-max_history:]
                    
                    # Update Prometheus metrics
                    if self.config.external_monitoring.prometheus_enabled and PROMETHEUS_AVAILABLE:
                        latest_training_metrics = None
                        if self.training_metrics_history:
                            latest_training_metrics = self.training_metrics_history[-1]
                        
                        self._update_prometheus_metrics(resource_stats, latest_training_metrics)
                    
                    # Update cost tracking
                    if self.config.cost_tracking.enabled:
                        self._update_cost_tracking(resource_stats)
                    
                    # Log stats periodically
                    if current_time - self.last_log_time >= self.config.log_interval_seconds:
                        self._log_resource_stats(resource_stats)
                        self.last_log_time = current_time
                    
                    # Update last check time
                    self.last_resource_check_time = current_time
                
                # Check health
                if current_time - self.last_resource_check_time >= self.config.check_interval_seconds:
                    self.check_health()
                
                # Check for cleanup
                if (
                    self.config.enable_auto_cleanup and
                    current_time - self.last_resource_check_time >= 3600  # Once per hour
                ):
                    self._perform_cleanup()
                
                # Sleep briefly
                time.sleep(0.1)
            
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                logger.debug(traceback.format_exc())
                time.sleep(1.0)
    
    def _heartbeat_loop(self):
        """Heartbeat monitoring loop."""
        while self.running:
            try:
                # Update heartbeat file
                current_time = time.time()
                if current_time - self.last_heartbeat_time >= self.config.heartbeat_interval_seconds:
                    # Get current health status
                    status = self._get_current_health_status()
                    
                    # Update heartbeat file
                    self._update_heartbeat_file(status)
                    
                    # Update last heartbeat time
                    self.last_heartbeat_time = current_time
                
                # Sleep briefly
                time.sleep(1.0)
            
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                logger.debug(traceback.format_exc())
                time.sleep(5.0)
    
    def _update_heartbeat_file(self, status: HealthStatus):
        """
        Update the heartbeat file.
        
        Args:
            status: Current health status
        """
        try:
            # Create heartbeat file
            heartbeat_file = self.health_dir / "HEARTBEAT"
            with open(heartbeat_file, "w") as f:
                f.write(f"timestamp: {time.time()}\n")
                f.write(f"status: {status.value}\n")
                f.write(f"pid: {os.getpid()}\n")
                f.write(f"uptime: {time.time() - self.start_time:.1f}s\n")
                
                # Add resource stats
                if self.resource_stats_history:
                    latest_stats = self.resource_stats_history[-1]
                    f.write(f"cpu_percent: {latest_stats.cpu_percent:.1f}%\n")
                    f.write(f"memory_percent: {latest_stats.memory_percent:.1f}%\n")
                    f.write(f"disk_percent: {latest_stats.disk_percent:.1f}%\n")
                
                # Add training metrics
                if self.training_metrics_history:
                    latest_metrics = self.training_metrics_history[-1]
                    f.write(f"step: {latest_metrics.step}\n")
                    if latest_metrics.loss is not None:
                        f.write(f"loss: {latest_metrics.loss:.6f}\n")
                    if latest_metrics.samples_per_second is not None:
                        f.write(f"samples_per_second: {latest_metrics.samples_per_second:.1f}\n")
                
                # Add recovery info
                f.write(f"recovery_attempts: {self.recovery_attempts}\n")
                if self.recovery_attempts > 0 and self.last_recovery_time > 0:
                    f.write(f"last_recovery: {time.time() - self.last_recovery_time:.1f}s ago\n")
            
            logger.debug(f"Updated heartbeat file: {heartbeat_file}")
        
        except Exception as e:
            logger.error(f"Error updating heartbeat file: {e}")
    
    def _get_current_health_status(self) -> HealthStatus:
        """
        Get the current health status.
        
        Returns:
            HealthStatus: Current health status
        """
        if not self.health_check_history:
            return HealthStatus.UNKNOWN
        
        # Get latest health check
        with self.health_lock:
            latest_check = self.health_check_history[-1]
        
        # Check if recent
        if time.time() - latest_check.timestamp > 60:
            return HealthStatus.UNKNOWN
        
        return latest_check.status
    
    def _get_current_resource_stats(self) -> ResourceStats:
        """
        Get the current resource statistics.
        
        Returns:
            ResourceStats: Current resource statistics
        """
        if not self.resource_stats_history:
            return self._collect_resource_stats()
        
        with self.stats_lock:
            return self.resource_stats_history[-1]
    
    def _collect_resource_stats(self) -> ResourceStats:
        """
        Collect system resource statistics.
        
        Returns:
            ResourceStats: Collected resource statistics
        """
        stats = ResourceStats(timestamp=time.time())
        
        try:
            # Get CPU stats
            stats.cpu_percent = psutil.cpu_percent(interval=0.1)
            stats.cpu_count = psutil.cpu_count(logical=True)
            
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                stats.cpu_freq_current = cpu_freq.current
                stats.cpu_freq_max = cpu_freq.max
            
            # Get memory stats
            memory = psutil.virtual_memory()
            stats.memory_used = memory.used / (1024 ** 3)  # GB
            stats.memory_available = memory.available / (1024 ** 3)  # GB
            stats.memory_total = memory.total / (1024 ** 3)  # GB
            stats.memory_percent = memory.percent
            
            swap = psutil.swap_memory()
            stats.swap_used = swap.used / (1024 ** 3)  # GB
            stats.swap_free = swap.free / (1024 ** 3)  # GB
            stats.swap_percent = swap.percent
            
            # Get disk stats
            disk = psutil.disk_usage(self.output_dir)
            stats.disk_used = disk.used / (1024 ** 3)  # GB
            stats.disk_free = disk.free / (1024 ** 3)  # GB
            stats.disk_total = disk.total / (1024 ** 3)  # GB
            stats.disk_percent = disk.percent
            
            # Get disk I/O stats
            disk_io = psutil.disk_io_counters()
            if disk_io and self.last_disk_io != (0, 0):
                read_bytes_delta = disk_io.read_bytes - self.last_disk_io[0]
                write_bytes_delta = disk_io.write_bytes - self.last_disk_io[1]
                time_delta = time.time() - self.last_resource_check_time
                
                if time_delta > 0:
                    stats.disk_read_speed = read_bytes_delta / time_delta / (1024 ** 2)  # MB/s
                    stats.disk_write_speed = write_bytes_delta / time_delta / (1024 ** 2)  # MB/s
            
            if disk_io:
                self.last_disk_io = (disk_io.read_bytes, disk_io.write_bytes)
            
            # Get network stats
            net_io = psutil.net_io_counters()
            if net_io and self.last_network_io != (0, 0):
                sent_bytes_delta = net_io.bytes_sent - self.last_network_io[0]
                recv_bytes_delta = net_io.bytes_recv - self.last_network_io[1]
                time_delta = time.time() - self.last_resource_check_time
                
                if time_delta > 0:
                    stats.network_sent = sent_bytes_delta / time_delta / (1024 ** 2)  # MB/s
                    stats.network_received = recv_bytes_delta / time_delta / (1024 ** 2)  # MB/s
            
            if net_io:
                self.last_network_io = (net_io.bytes_sent, net_io.bytes_recv)
                stats.network_connections = len(psutil.net_connections())
                stats.network_errors = net_io.errin + net_io.errout
            
            # Get GPU stats
            stats.gpu_stats = self._collect_gpu_stats()
            stats.gpu_count = len(stats.gpu_stats)
            
            # Get process stats
            process = psutil.Process()
            stats.process_count = len(psutil.pids())
            stats.process_threads = process.num_threads()
            stats.process_memory = process.memory_info().rss / (1024 ** 3)  # GB
            stats.process_cpu_percent = process.cpu_percent(interval=0.1)
            
            # Get system load
            load = psutil.getloadavg()
            stats.load_1min = load[0]
            stats.load_5min = load[1]
            stats.load_15min = load[2]
            
            # Get CPU temperature if available
            if self.config.enable_thermal_monitoring:
                try:
                    if hasattr(psutil, "sensors_temperatures"):
                        temps = psutil.sensors_temperatures()
                        if temps:
                            # Get the highest temperature from any sensor
                            max_temp = 0
                            for name, entries in temps.items():
                                for entry in entries:
                                    if entry.current > max_temp:
                                        max_temp = entry.current
                            
                            if max_temp > 0:
                                stats.cpu_temperature = max_temp
                except Exception as e:
                    logger.debug(f"Error getting CPU temperature: {e}")
        
        except Exception as e:
            logger.error(f"Error collecting resource stats: {e}")
            logger.debug(traceback.format_exc())
        
        return stats
    
    def _collect_gpu_stats(self) -> List[Dict[str, Any]]:
        """
        Collect GPU statistics.
        
        Returns:
            List[Dict[str, Any]]: List of GPU statistics
        """
        gpu_stats = []
        
        # Try NVML first
        if NVML_AVAILABLE:
            try:
                nvml.nvmlInit()
                device_count = nvml.nvmlDeviceGetCount()
                
                for i in range(device_count):
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # Get memory info
                    memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                    memory_total = memory_info.total / (1024 ** 3)  # GB
                    memory_used = memory_info.used / (1024 ** 3)  # GB
                    memory_free = memory_info.free / (1024 ** 3)  # GB
                    memory_percent = (memory_used / memory_total) * 100 if memory_total > 0 else 0
                    
                    # Get temperature
                    temperature = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                    
                    # Get utilization
                    utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = utilization.gpu
                    memory_util = utilization.memory
                    
                    # Get power usage
                    power_usage = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # W
                    power_limit = nvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0  # W
                    
                    # Get fan speed
                    try:
                        fan_speed = nvml.nvmlDeviceGetFanSpeed(handle)
                    except:
                        fan_speed = 0
                    
                    # Get throttling reasons
                    throttle_reasons = {}
                    try:
                        throttle_reasons_raw = nvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
                        throttle_reasons = {
                            "power": bool(throttle_reasons_raw & nvml.nvmlClocksThrottleReasonPowerConstraint),
                            "thermal": bool(throttle_reasons_raw & nvml.nvmlClocksThrottleReasonThermalConstraint),
                            "active": bool(throttle_reasons_raw & nvml.nvmlClocksThrottleReasonActiveCountLimit),
                        }
                    except:
                        pass
                    
                    # Get performance state
                    try:
                        perf_state = nvml.nvmlDeviceGetPerformanceState(handle)
                    except:
                        perf_state = 0
                    
                    # Get device name
                    device_name = nvml.nvmlDeviceGetName(handle).decode("utf-8")
                    
                    gpu_stats.append({
                        "index": i,
                        "name": device_name,
                        "memory_total": memory_total,
                        "memory_used": memory_used,
                        "memory_free": memory_free,
                        "memory_percent": memory_percent,
                        "temperature": temperature,
                        "utilization": gpu_util,
                        "memory_utilization": memory_util,
                        "power_usage": power_usage,
                        "power_limit": power_limit,
                        "fan_speed": fan_speed,
                        "throttle_reasons": throttle_reasons,
                        "performance_state": perf_state,
                    })
                
                nvml.nvmlShutdown()
                return gpu_stats
            
            except Exception as e:
                logger.debug(f"Error collecting GPU stats with NVML: {e}")
        
        # Try GPUtil as fallback
        if GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                
                for i, gpu in enumerate(gpus):
                    gpu_stats.append({
                        "index": i,
                        "name": gpu.name,
                        "memory_total": gpu.memoryTotal / 1024,  # GB
                        "memory_used": gpu.memoryUsed / 1024,  # GB
                        "memory_free": (gpu.memoryTotal - gpu.memoryUsed) / 1024,  # GB
                        "memory_percent": gpu.memoryUtil * 100,
                        "temperature": gpu.temperature,
                        "utilization": gpu.load * 100,
                    })
                
                return gpu_stats
            
            except Exception as e:
                logger.debug(f"